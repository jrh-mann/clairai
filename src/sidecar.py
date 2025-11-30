import time
import json
import os
import cupy as cp
import numpy as np
import ctypes

IPC_PATH = "/tmp/vllm_probe.ipc"
META_PATH = "/tmp/vllm_probe_meta.json"

def main():
    print(">> Waiting for IPC handle...")
    while not os.path.exists(IPC_PATH):
        time.sleep(0.5)

    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    
    RING_SIZE = meta["ring_size"]
    SLOT_SIZE_BYTES = meta["slot_size_bytes"]
    NUM_PROBES = meta["num_probes"]
    
    with open(IPC_PATH, "rb") as f:
        handle_bytes = f.read()
    
    # Attach to VRAM
    gpu_mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes)
    data_base_ptr = gpu_mem_ptr + 128
    
    print(f">> Connected to IPC Ring Buffer.")
    
    # --- STABLE TRACKING STATE ---
    # Dictionary mapping Unique Block ID -> Data Storage
    # Structure: { current_block_id: { 'total_tokens': 0, 'data': [], 'last_seen_seq': 0 } }
    active_streams = {} 
    
    cudart = ctypes.CDLL('libcudart.so')

    def read_head():
        cp.cuda.Device().synchronize()
        host_buf = (ctypes.c_uint64 * 1)()
        cudart.cudaMemcpy(ctypes.byref(host_buf), ctypes.c_void_p(gpu_mem_ptr), 8, 2)
        return int(host_buf[0])
    
    last_seq = read_head()
    print(f">> Starting from sequence: {last_seq}")
    
    while True:
        curr_seq = read_head()
        
        if curr_seq > last_seq:
            # Process new sequences
            if curr_seq - last_seq > RING_SIZE:
                start = curr_seq - 1 # Jump to latest if we fell behind
            else:
                start = last_seq

            for seq in range(start, curr_seq):
                slot_idx = seq % RING_SIZE
                src_ptr = data_base_ptr + (slot_idx * SLOT_SIZE_BYTES)
                
                cp.cuda.Device().synchronize()
                
                # 1. Read Header
                # Batch ID (0-8)
                h_batch = (ctypes.c_uint64 * 1)()
                cudart.cudaMemcpy(ctypes.byref(h_batch), ctypes.c_void_p(src_ptr), 8, 2)
                batch_id = h_batch[0]
                
                # Num Requests (8-12)
                h_num_req = (ctypes.c_int32 * 1)()
                cudart.cudaMemcpy(ctypes.byref(h_num_req), ctypes.c_void_p(src_ptr + 8), 4, 2)
                num_requests = h_num_req[0]
                
                if num_requests == 0: continue

                # 2. Read STABLE IDs (Chain Links)
                # Offset 12. Size: num_requests * 2 * 4 bytes
                # Shape: [num_requests, 2] -> (Current_Block, Prev_Block)
                ids_size = num_requests * 2
                h_ids = (ctypes.c_int32 * ids_size)()
                cudart.cudaMemcpy(
                    ctypes.byref(h_ids), 
                    ctypes.c_void_p(src_ptr + 12), 
                    ids_size * 4, 
                    2
                )
                # Reshape into list of tuples: [(curr, prev), (curr, prev)...]
                chain_ids = []
                for i in range(num_requests):
                    chain_ids.append((h_ids[2*i], h_ids[2*i+1]))

                # 3. Read Query Start Loc
                # Offset: 12 + (ids_size * 4)
                qsl_offset = 12 + (ids_size * 4)
                qsl_size = num_requests + 1
                h_qsl = (ctypes.c_int32 * qsl_size)()
                cudart.cudaMemcpy(
                    ctypes.byref(h_qsl), 
                    ctypes.c_void_p(src_ptr + qsl_offset), 
                    qsl_size * 4, 
                    2
                )
                qsl = np.array([h_qsl[i] for i in range(qsl_size)])
                
                # 4. Read Scores
                scores_offset = qsl_offset + (qsl_size * 4)
                num_tokens = qsl[-1]
                scores_floats = num_tokens * NUM_PROBES
                h_scores = (ctypes.c_float * scores_floats)()
                cudart.cudaMemcpy(
                    ctypes.byref(h_scores), 
                    ctypes.c_void_p(src_ptr + scores_offset), 
                    scores_floats * 4, 
                    2
                )
                all_scores = np.array(h_scores).reshape(num_tokens, NUM_PROBES)

                # --- TRACKING LOGIC ---
                current_batch_ids = set()

                for i in range(num_requests):
                    # Data for this specific request
                    curr_block, prev_block = chain_ids[i]
                    start_t, end_t = qsl[i], qsl[i+1]
                    req_scores = all_scores[start_t:end_t]
                    
                    if len(req_scores) == 0: continue

                    # Identify the Stream
                    stream_id = None
                    
                    # Case A: Steady State (Same block)
                    if curr_block in active_streams:
                        stream_id = curr_block
                        
                    # Case B: Transition (Block Jump)
                    # We look for a stream that ended at 'prev_block'
                    elif prev_block in active_streams:
                        print(f"🔀 [Seq {batch_id}] Transition: {prev_block} -> {curr_block}")
                        # Migrate data to new ID
                        active_streams[curr_block] = active_streams.pop(prev_block)
                        stream_id = curr_block
                        
                    # Case C: New Request
                    else:
                        print(f"✨ [Seq {batch_id}] New Stream detected at Block {curr_block}")
                        active_streams[curr_block] = {
                            'total_tokens': 0, 
                            'data': [],
                            'start_seq': batch_id
                        }
                        stream_id = curr_block
                    
                    # Store Data
                    stream = active_streams[stream_id]
                    stream['last_seen_seq'] = batch_id
                    
                    # Calculate Global Token Index for display
                    current_global_idx = stream['total_tokens']
                    stream['total_tokens'] += (end_t - start_t)
                    
                    # (Optional) Store scores to disk here
                    # stream['data'].append(req_scores) 
                    
                    avg = req_scores.mean()
                    print(f"   Req (Block {stream_id}): Tokens {current_global_idx}-{stream['total_tokens']-1} | Avg: {avg:.4f}")
                    
                    current_batch_ids.add(stream_id)

                # --- GARBAGE COLLECTION ---
                # Remove streams that weren't seen in this batch (finished requests)
                # We allow a grace period of 2 sequences just in case
                for key in list(active_streams.keys()):
                    if active_streams[key]['last_seen_seq'] < (batch_id - 5):
                        print(f"🏁 [Seq {batch_id}] Stream {key} Finished. Total Tokens: {active_streams[key]['total_tokens']}")
                        del active_streams[key]

            last_seq = curr_seq
        else:
            time.sleep(0.001)

if __name__ == "__main__":
    main()