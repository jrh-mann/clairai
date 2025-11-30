import requests
import multiprocessing
import sys
import os
import time
import threading
import json
import ctypes
import numpy as np
import cupy as cp
from collections import defaultdict

# Add src directory to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# -------------------------------------------------------------------------
# 1. API Client Helper
# -------------------------------------------------------------------------
def send_request(prompt, server_url="http://localhost:8000/v1/chat/completions", stream=False):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": stream,
    }
    try:
        resp = requests.post(server_url, json=payload, stream=stream, timeout=60)
        resp.raise_for_status()
        if stream:
            return resp.iter_lines(decode_unicode=True)
        else:
            return resp.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None

# -------------------------------------------------------------------------
# 2. Sidecar Process (With Root Identity Tracking)
# -------------------------------------------------------------------------
def sidecar_process(probe_data_list):
    import time
    import json
    import os
    import cupy as cp
    import numpy as np
    import ctypes

    IPC_PATH = "/tmp/vllm_probe.ipc"
    META_PATH = "/tmp/vllm_probe_meta.json"
    
    while not os.path.exists(IPC_PATH):
        time.sleep(0.1)
    
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    
    RING_SIZE = meta["ring_size"]
    SLOT_SIZE_BYTES = meta["slot_size_bytes"]
    NUM_PROBES = meta["num_probes"]
    
    with open(IPC_PATH, "rb") as f:
        handle_bytes = f.read()
    
    gpu_mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes)
    data_base_ptr = gpu_mem_ptr + 128
    cudart = ctypes.CDLL('libcudart.so')
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    def read_head():
        cp.cuda.Device().synchronize()
        host_buf = (ctypes.c_uint64 * 1)()
        cudart.cudaMemcpy(ctypes.byref(host_buf), ctypes.c_void_p(gpu_mem_ptr), 8, 2)
        return int(host_buf[0])
    
    last_seq = read_head()
    active_streams = {} # { current_block_id: { 'root_id': int, 'total_tokens': int } }

    while True:
        curr_seq = read_head()
        if curr_seq > last_seq:
            start = last_seq if (curr_seq - last_seq) <= RING_SIZE else curr_seq - 1
            
            for seq in range(start, curr_seq):
                slot_idx = seq % RING_SIZE
                src_ptr = int(data_base_ptr + (slot_idx * SLOT_SIZE_BYTES))
                cp.cuda.Device().synchronize()
                
                # 1. Read Headers
                h_nums = (ctypes.c_uint64 * 2)() 
                cudart.cudaMemcpy(ctypes.byref(h_nums), ctypes.c_void_p(src_ptr), 12, 2)
                batch_id = h_nums[0]
                
                h_num_req = (ctypes.c_int32 * 1)()
                cudart.cudaMemcpy(ctypes.byref(h_num_req), ctypes.c_void_p(src_ptr + 8), 4, 2)
                num_requests = h_num_req[0]
                
                if num_requests == 0: continue

                # 2. Read Layout Arrays
                ids_count = num_requests * 2
                qsl_count = num_requests + 1
                
                h_ids = (ctypes.c_int32 * ids_count)()
                cudart.cudaMemcpy(ctypes.byref(h_ids), ctypes.c_void_p(src_ptr + 12), ids_count * 4, 2)
                
                h_qsl = (ctypes.c_int32 * qsl_count)()
                cudart.cudaMemcpy(ctypes.byref(h_qsl), ctypes.c_void_p(src_ptr + 12 + ids_count*4), qsl_count * 4, 2)
                qsl = np.array(h_qsl)
                
                num_tokens = int(qsl[-1])
                scores_floats = num_tokens * NUM_PROBES
                h_scores = (ctypes.c_float * scores_floats)()
                cudart.cudaMemcpy(ctypes.byref(h_scores), ctypes.c_void_p(src_ptr + 12 + ids_count*4 + qsl_count*4), scores_floats * 4, 2)
                all_scores = np.frombuffer(h_scores, dtype=np.float32).reshape(num_tokens, NUM_PROBES)

                # 3. Process Streams
                for i in range(num_requests):
                    curr_block = h_ids[2*i]
                    prev_block = h_ids[2*i+1]
                    start_t, end_t = qsl[i], qsl[i+1]
                    req_scores = all_scores[start_t:end_t]
                    
                    if len(req_scores) == 0: continue

                    if curr_block not in active_streams:
                        if prev_block in active_streams:
                            active_streams[curr_block] = active_streams.pop(prev_block)
                        else:
                            active_streams[curr_block] = {'root_id': curr_block, 'total_tokens': 0}
                    
                    stream_ctx = active_streams[curr_block]
                    base_token_idx = stream_ctx['total_tokens']
                    root_id = stream_ctx['root_id']

                    for t_offset in range(len(req_scores)):
                        probe_data_list.append({
                            'root_id': int(root_id),      
                            'block_id': int(curr_block),  
                            'seq': int(batch_id),
                            'token_idx': int(base_token_idx + t_offset),
                            'probes': req_scores[t_offset].tolist()
                        })
                    stream_ctx['total_tokens'] += len(req_scores)

            last_seq = curr_seq
        else:
            time.sleep(0.001)

# -------------------------------------------------------------------------
# 3. Request Maker
# -------------------------------------------------------------------------
def make_request(request_id, prompt, streamed_responses_dict, lock):
    gen = send_request(prompt, stream=True)
    tokens = []
    if gen:
        for line in gen:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("choices") and data["choices"][0].get("delta"):
                        content = data["choices"][0]["delta"].get("content")
                        if content: tokens.append(content)
                except: pass
    with lock:
        streamed_responses_dict[request_id] = {'prompt': prompt, 'tokens': tokens, 'count': len(tokens)}

def vllm_process(streamed_responses_dict, prompts, lock):
    threads = []
    for i, p in enumerate(prompts):
        t = threading.Thread(target=make_request, args=(i, p, streamed_responses_dict, lock))
        threads.append(t)
        t.start()
        time.sleep(0.2)
    for t in threads: t.join()

# -------------------------------------------------------------------------
# 4. Main Execution & Analysis
# -------------------------------------------------------------------------
if __name__ == "__main__":
    REQUESTS = [
        "The quick brown fox jumps over the lazy dog",      
        "Explain the theory of relativity in simple terms", 
        "Write a haiku about a GPU",                        
    ]
    
    manager = multiprocessing.Manager()
    streamed_responses = manager.dict()
    probe_data = manager.list()
    lock = manager.Lock()
    
    print("="*60)
    print("🚀 STARTING vLLM PROBE INTEGRATION TEST")
    print("="*60)

    p_sidecar = multiprocessing.Process(target=sidecar_process, args=(probe_data,))
    p_sidecar.start()
    print("[MAIN] Sidecar started. Warming up...")
    time.sleep(2) 
    
    print(f"[MAIN] Sending {len(REQUESTS)} concurrent requests...")
    p_vllm = multiprocessing.Process(target=vllm_process, args=(streamed_responses, REQUESTS, lock))
    p_vllm.start()
    p_vllm.join()
    print("[MAIN] Requests complete.")
    
    print("[MAIN] Waiting for probe data flush (2s)...")
    time.sleep(2)
    p_sidecar.terminate()
    
    # ---------------------------------------------------------------------
    # ANALYSIS START
    # ---------------------------------------------------------------------
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE SYSTEM ANALYSIS")
    print("="*80)

    # 1. Reconstruct Logical Streams
    probe_streams = defaultdict(list)
    for p in probe_data:
        probe_streams[p['root_id']].append(p)
    
    for rid in probe_streams:
        probe_streams[rid].sort(key=lambda x: x['token_idx'])

    print(f"\n1. RECONSTRUCTION METRICS")
    print(f"   - Raw Probe Points Captured: {len(probe_data)}")
    print(f"   - Reconstructed Logical Streams: {len(probe_streams)}")

    # 2. Smart Match Probe Streams to API Requests
    matches = [] # List of (req_id, root_id, prefill_len)
    used_roots = set()

    print(f"\n2. STREAM MATCHING (Smart Heuristic)")
    print(f"   {'ReqID':<6} | {'PromptEst':<10} | {'GenTokens':<10} | {'MatchRootID':<15} | {'TruePrefill':<12}")
    print("   " + "-"*65)

    for req_id in sorted(streamed_responses.keys()):
        api_data = streamed_responses[req_id]
        gen_count = api_data['count']
        # Rough estimate: 1 token ~ 4 chars. Used only for tie-breaking.
        est_prompt_len = len(api_data['prompt']) // 4
        
        best_match = None
        best_score = float('inf')
        
        for root_id, points in probe_streams.items():
            if root_id in used_roots: continue
            
            probe_total = len(points)
            calc_prefill = probe_total - gen_count
            
            # Constraints: Prefill positive and sane (<2000)
            if calc_prefill >= 0 and calc_prefill < 2000:
                # Score: Difference between calculated prefill and estimated prompt size
                score = abs(calc_prefill - est_prompt_len)
                if score < best_score:
                    best_score = score
                    best_match = (root_id, calc_prefill)
        
        if best_match:
            root_id, true_prefill = best_match
            used_roots.add(root_id)
            matches.append((req_id, root_id, true_prefill))
            print(f"   {req_id:<6} | {est_prompt_len:<10} | {gen_count:<10} | {root_id:<15} | {true_prefill:<12}")
        else:
            print(f"   {req_id:<6} | {est_prompt_len:<10} | {gen_count:<10} | {'NO MATCH':<15} | {'-':<12}")

    # 3. Stitching Verification
    print("\n3. DATA STITCHING & VERIFICATION")
    
    for req_id, root_id, prefill_len in matches:
        api_tokens = streamed_responses[req_id]['tokens']
        probe_points = probe_streams[root_id]
        
        print(f"\n   🔎 REQUEST {req_id} <-> ROOT BLOCK {root_id}")
        print(f"      - True Prefill Size: {prefill_len} tokens")
        print(f"      - Generated Tokens: {len(api_tokens)}")
        
        print(f"\n      --- TOKEN ALIGNMENT TABLE (First 3 + Last 3) ---")
        print(f"      {'Idx':<5} | {'Type':<8} | {'Token Text':<20} | {'Probe[0] Value':<15} | {'Status'}")
        print("      " + "-"*70)
        
        # Display Prefill
        for i in range(min(3, prefill_len)):
            val = probe_points[i]['probes'][0]
            print(f"      {i:<5} | {'PROMPT':<8} | {'<hidden>':<20} | {val:<15.6f} | {'-'}")
        
        if prefill_len > 3: print(f"      {'...':<5} | {'...':<8} | {'...':<20} | {'...':<15} |")

        # Display Generated
        display_indices = list(range(3)) + list(range(len(api_tokens)-3, len(api_tokens)))
        display_indices = sorted(list(set(i for i in display_indices if 0 <= i < len(api_tokens))))

        success = True
        for i in display_indices:
            token_text = api_tokens[i].replace('\n', '\\n')
            probe_idx = prefill_len + i
            
            if probe_idx < len(probe_points):
                val = probe_points[probe_idx]['probes'][0]
                status = "✅ OK"
            else:
                val = 0.0
                status = "❌ MISSING"
                success = False
            
            print(f"      {probe_idx:<5} | {'GEN':<8} | {token_text[:20]:<20} | {val:<15.6f} | {status}")

        if success: print("\n      ✅ INTEGRATION STATUS: SUCCESS")
        else: print("\n      ❌ INTEGRATION STATUS: DATA MISMATCH")

    print("\n" + "="*80)