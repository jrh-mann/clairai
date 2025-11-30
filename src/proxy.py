import asyncio
import uvicorn
import httpx
import time
import json
import threading
import ctypes
import os
import sys
import numpy as np
import cupy as cp
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional

# --- CONFIGURATION ---
VLLM_URL = "http://localhost:8000/v1/chat/completions"
IPC_PATH = "/tmp/vllm_probe.ipc"
META_PATH = "/tmp/vllm_probe_meta.json"
MAX_BUFFER_AGE_SEC = 30  # Cleanup unmatched streams after 30s

# --- SHARED STATE ---
# Stores: root_id -> {'tokens': deque([arrays]), 'total_len': int, 'last_update': timestamp}
probe_buffer: Dict[int, Dict] = {}
buffer_lock = threading.Lock()

# -----------------------------------------------------------------------------
# 1. THE IPC WORKER (Background Thread)
#    Continuously drains the GPU Ring Buffer and updates 'probe_buffer'
# -----------------------------------------------------------------------------
class IPCMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.cudart = ctypes.CDLL('libcudart.so')
        self.cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    def run(self):
        print(f"[IPC] ⏳ Waiting for IPC file at {IPC_PATH}...")
        while not os.path.exists(IPC_PATH):
            time.sleep(1)
        
        with open(META_PATH, 'r') as f:
            meta = json.load(f)
            
        RING_SIZE = meta["ring_size"]
        SLOT_SIZE_BYTES = meta["slot_size_bytes"]
        NUM_PROBES = meta["num_probes"]
        
        with open(IPC_PATH, "rb") as f:
            handle_bytes = f.read()
            
        # Attach to GPU Memory
        gpu_mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes)
        data_base_ptr = gpu_mem_ptr + 128
        
        def read_head():
            cp.cuda.Device().synchronize()
            host_buf = (ctypes.c_uint64 * 1)()
            self.cudart.cudaMemcpy(ctypes.byref(host_buf), ctypes.c_void_p(gpu_mem_ptr), 8, 2)
            return int(host_buf[0])

        last_seq = read_head()
        print(f"[IPC] 🚀 Connected. Ring Size: {RING_SIZE}. Starting at Seq: {last_seq}")

        # Local state to handle block transitions (Chain Linking)
        # current_block -> {root_id: int}
        active_links = {}

        while self.running:
            curr_seq = read_head()
            
            if curr_seq == last_seq:
                time.sleep(0.0005) # Hyper-fast poll (0.5ms)
                continue
                
            # Handle ring buffer wrap
            start = last_seq if (curr_seq - last_seq) <= RING_SIZE else curr_seq - 1
            
            for seq in range(start, curr_seq):
                slot_idx = seq % RING_SIZE
                src_ptr = int(data_base_ptr + (slot_idx * SLOT_SIZE_BYTES))
                
                # --- READ METADATA (Batch Info + IDs) ---
                # Read BatchID (8) + NumReqs (4)
                h_meta = (ctypes.c_uint8 * 12)()
                self.cudart.cudaMemcpy(ctypes.byref(h_meta), ctypes.c_void_p(src_ptr), 12, 2)
                batch_id = int.from_bytes(h_meta[0:8], 'little')
                num_requests = int.from_bytes(h_meta[8:12], 'little')

                if num_requests == 0: continue

                # --- READ ARRAYS ---
                ids_count = num_requests * 2
                qsl_count = num_requests + 1
                
                # 1. Stable IDs (Int32)
                h_ids = (ctypes.c_int32 * ids_count)()
                self.cudart.cudaMemcpy(ctypes.byref(h_ids), ctypes.c_void_p(src_ptr + 12), ids_count * 4, 2)
                
                # 2. Query Start Loc (Int32)
                h_qsl = (ctypes.c_int32 * qsl_count)()
                self.cudart.cudaMemcpy(ctypes.byref(h_qsl), ctypes.c_void_p(src_ptr + 12 + ids_count*4), qsl_count * 4, 2)
                
                # 3. Scores (Float32)
                num_tokens = h_qsl[num_requests] # Last element
                scores_floats = num_tokens * NUM_PROBES
                h_scores = (ctypes.c_float * scores_floats)()
                self.cudart.cudaMemcpy(ctypes.byref(h_scores), ctypes.c_void_p(src_ptr + 12 + ids_count*4 + qsl_count*4), scores_floats * 4, 2)
                
                # Convert to numpy (Zero-copy from the C-types buffer)
                # Copy is essential to detach from ring buffer
                scores_np = np.ctypeslib.as_array(h_scores).reshape(num_tokens, NUM_PROBES).copy()
                
                # --- PROCESS STREAMS ---
                with buffer_lock:
                    now = time.time()
                    for i in range(num_requests):
                        curr_block = h_ids[2*i]
                        prev_block = h_ids[2*i+1]
                        
                        start_t = h_qsl[i]
                        end_t = h_qsl[i+1]
                        
                        if start_t >= end_t: continue

                        # Chain Link Logic
                        if curr_block not in active_links:
                            if prev_block in active_links:
                                active_links[curr_block] = active_links.pop(prev_block)
                            else:
                                active_links[curr_block] = curr_block # Root is itself
                        
                        root_id = active_links[curr_block]
                        
                        # Store Data
                        if root_id not in probe_buffer:
                            probe_buffer[root_id] = {'tokens': deque(), 'total_len': 0, 'last_update': now}
                        
                        req_slice = scores_np[start_t:end_t]
                        probe_buffer[root_id]['tokens'].append(req_slice)
                        probe_buffer[root_id]['total_len'] += (end_t - start_t)
                        probe_buffer[root_id]['last_update'] = now

            last_seq = curr_seq
            
            # --- GARBAGE COLLECTION ---
            if curr_seq % 500 == 0:
                self.cleanup()

    def cleanup(self):
        now = time.time()
        with buffer_lock:
            to_remove = [k for k, v in probe_buffer.items() 
                         if now - v['last_update'] > MAX_BUFFER_AGE_SEC]
            for k in to_remove:
                del probe_buffer[k]

# -----------------------------------------------------------------------------
# 2. FASTAPI PROXY SERVER
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    monitor = IPCMonitor()
    monitor.start()
    yield
    monitor.running = False

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient(timeout=120.0, limits=httpx.Limits(max_keepalive_connections=20, max_connections=200))

@app.get("/v1/models")
async def proxy_models():
    """Forward /v1/models requests to vLLM."""
    try:
        vllm_models_url = VLLM_URL.replace("/chat/completions", "/models")
        response = await client.get(vllm_models_url)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def get_stream_delta(root_id: Optional[int], prompt_len_est: int, last_sent_count: int) -> Tuple[Optional[int], Optional[List[List[float]]]]:
    """
    Efficiently fetches ONLY the new tokens since `last_sent_count`.
    Complexity: O(New_Tokens) instead of O(Total_Tokens).
    """
    with buffer_lock:
        # A. If we don't have a root_id yet, find the best match
        if root_id is None:
            best_score = float('inf')
            candidates = list(probe_buffer.keys())
            for rid in candidates:
                # Heuristic: Match based on length proximity to prompt length
                # We assume the stream started recently, so total_len should be close to current gen_count
                total_len = probe_buffer[rid]['total_len']
                
                # Check if stream is viable (has enough data to be the one)
                if total_len > 0:
                     # A match is likely if the buffer length roughly matches where we expect to be
                     # For the first token, total_len is small.
                    score = abs(total_len - prompt_len_est) 
                    # Refined match: Prefer larger buffers that are close to prompt+gen
                    if score < best_score:
                        best_score = score
                        root_id = rid
            
            # If no good match found or empty, return None
            if root_id is None or root_id not in probe_buffer:
                return None, None

        # B. We have a root_id, check for new data
        stream_info = probe_buffer[root_id]
        current_total_len = stream_info['total_len']
        
        # If no new data
        if current_total_len <= last_sent_count:
            return root_id, None
            
        # C. Extract Delta (Zero-Copy Logic)
        # We need to skip the first `last_sent_count` tokens and take the rest.
        # Since 'tokens' is a deque of chunks, we iterate to find the start point.
        
        delta_arrays = []
        tokens_scanned = 0
        needed_start = last_sent_count
        
        for chunk in stream_info['tokens']:
            chunk_len = len(chunk)
            chunk_end = tokens_scanned + chunk_len
            
            # If this chunk is entirely in the past, skip it
            if chunk_end <= needed_start:
                tokens_scanned += chunk_len
                continue
                
            # If this chunk contains new data
            start_in_chunk = max(0, needed_start - tokens_scanned)
            
            # Append slice (numpy view, cheap)
            delta_arrays.append(chunk[start_in_chunk:])
            tokens_scanned += chunk_len
            
        if not delta_arrays:
            return root_id, None

        # D. Concatenate ONLY the new data and convert to list
        # We do the heavy lifting (tolist) OUTSIDE this function if possible,
        # but here we are inside lock. To minimize lock time, we could copy.
        # However, for delta (small N), concatenation is fast.
        delta_matrix = np.concatenate(delta_arrays, axis=0)
        
        return root_id, delta_matrix.T.tolist()

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    body["stream"] = True 
    
    # Heuristic for matching: prompt length in tokens (approx)
    messages = body.get("messages", [])
    prompt_text = " ".join([m.get("content", "") for m in messages])
    prompt_len_est = len(prompt_text) // 4

    async def generate():
        gen_token_count = 0
        # Track how much probe data we have successfully sent to the client
        # This allows us to send only DELTAS (Efficiency Gain)
        last_sent_probe_count = 0
        matched_root_id = None
        
        try:
            async with client.stream("POST", VLLM_URL, json=body) as resp:
                if resp.status_code != 200:
                    error_msg = f"data: {json.dumps({'error': 'vLLM Error'})}\n\n"
                    yield error_msg
                    return

                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            # 1. Forward Text Token
                            yield_line = f"{line}\n\n"
                            yield yield_line
                            
                            gen_token_count += 1
                            
                            # 2. Opportunistic Probe Update (Non-blocking)
                            # Only try to fetch probes every few tokens to save CPU
                            if gen_token_count % 3 == 0:
                                matched_root_id, delta_scores = get_stream_delta(
                                    matched_root_id, 
                                    prompt_len_est + gen_token_count, 
                                    last_sent_probe_count
                                )
                                
                                if delta_scores:
                                    # Calculate how many tokens this delta represents
                                    # delta_scores is [n_probes, n_tokens]
                                    num_new_tokens = len(delta_scores[0])
                                    
                                    probe_chunk = {
                                        "type": "probe_update",
                                        "probe_scores": delta_scores, # DELTA ONLY
                                        "token_count": last_sent_probe_count + num_new_tokens
                                    }
                                    probe_line = f"data: {json.dumps(probe_chunk)}\n\n"
                                    yield probe_line
                                    
                                    last_sent_probe_count += num_new_tokens
                                    
                        except Exception as e: 
                            pass # Don't kill the stream if probe logic hiccups
                    
                    if line == "data: [DONE]":
                        break
                        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        # 3. Final Flush (Ensure we got everything)
        # Wait a tick for IPC to catch the very last token
        await asyncio.sleep(0.01)
        _, final_delta = get_stream_delta(matched_root_id, 0, last_sent_probe_count)
        
        if final_delta:
            final_chunk = {
                "type": "probe_update",
                "probe_scores": final_delta,
                "token_count": last_sent_probe_count + len(final_delta[0])
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", # Disable Nginx buffering if present
        }
    )

if __name__ == "__main__":
    print("🚀 Starting Hyper-Optimized vLLM Proxy...")
    uvicorn.run(app, host="0.0.0.0", port=6969, log_level="error")