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
# Stores: root_id -> {'tokens': deque([arrays]), 'last_update': timestamp}
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
                time.sleep(0.001) # 1ms poll interval
                continue
                
            # Handle ring buffer wrap
            start = last_seq if (curr_seq - last_seq) <= RING_SIZE else curr_seq - 1
            
            for seq in range(start, curr_seq):
                slot_idx = seq % RING_SIZE
                src_ptr = int(data_base_ptr + (slot_idx * SLOT_SIZE_BYTES))
                cp.cuda.Device().synchronize()
                
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
                # We copy here to detach from the ring buffer so we can store it
                scores_np = np.ctypeslib.as_array(h_scores).reshape(num_tokens, NUM_PROBES).copy()
                
                # --- PROCESS STREAMS ---
                with buffer_lock:
                    now = time.time()
                    for i in range(num_requests):
                        curr_block = h_ids[2*i]
                        prev_block = h_ids[2*i+1]
                        
                        start_t = h_qsl[i]
                        end_t = h_qsl[i+1]
                        
                        # Empty slot check
                        if start_t >= end_t: continue

                        # Chain Link Logic
                        if curr_block not in active_links:
                            if prev_block in active_links:
                                # Transition A -> B: Inherit Root ID
                                active_links[curr_block] = active_links.pop(prev_block)
                            else:
                                # New Stream
                                active_links[curr_block] = curr_block # Root is itself
                        
                        root_id = active_links[curr_block]
                        
                        # Store Data
                        if root_id not in probe_buffer:
                            probe_buffer[root_id] = {'tokens': [], 'last_update': now}
                        
                        # Extract this request's slice
                        req_slice = scores_np[start_t:end_t]
                        probe_buffer[root_id]['tokens'].append(req_slice)
                        probe_buffer[root_id]['last_update'] = now

            last_seq = curr_seq
            
            # --- GARBAGE COLLECTION (Every ~100 updates) ---
            if curr_seq % 100 == 0:
                self.cleanup()

    def cleanup(self):
        """Remove streams older than MAX_BUFFER_AGE_SEC"""
        now = time.time()
        with buffer_lock:
            # Create list to avoid modifying while iterating
            to_remove = [k for k, v in probe_buffer.items() 
                         if now - v['last_update'] > MAX_BUFFER_AGE_SEC]
            for k in to_remove:
                del probe_buffer[k]

# -----------------------------------------------------------------------------
# 2. FASTAPI PROXY SERVER
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    monitor = IPCMonitor()
    monitor.start()
    yield
    # Shutdown
    monitor.running = False

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to handle preflight OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient(timeout=120.0)

@app.get("/v1/models")
async def proxy_models():
    """Forward /v1/models requests to vLLM."""
    try:
        vllm_models_url = VLLM_URL.replace("/chat/completions", "/models")
        response = await client.get(vllm_models_url)
        if response.status_code == 200:
            return JSONResponse(content=response.json())
        else:
            return JSONResponse(status_code=response.status_code, content={"error": "vLLM Error"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Upstream connection failed: {str(e)}"})

def find_matching_stream(gen_count: int, prompt_len_est: int, pop: bool = False) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Finds the probe stream that best matches the request dimensions.
    Returns: (root_id, probe_data) where probe_data is shape (n_probes, n_output_tokens) OR (None, None)
    If pop=True, removes the stream from buffer. Otherwise, leaves it for incremental updates.
    """
    best_root = None
    best_score = float('inf')
    
    with buffer_lock:
        candidates = list(probe_buffer.keys())
        
        for root_id in candidates:
            stream_data = probe_buffer[root_id]['tokens']
            total_probe_tokens = sum(len(chunk) for chunk in stream_data)
            calc_prefill = total_probe_tokens - gen_count
            
            if calc_prefill >= 0 and calc_prefill < 2048:
                score = abs(calc_prefill - prompt_len_est)
                if score < best_score:
                    best_score = score
                    best_root = root_id
        
        if best_root is not None:
            data_chunks = probe_buffer[best_root]['tokens'] if not pop else probe_buffer.pop(best_root)['tokens']
            
            full_matrix = np.concatenate(data_chunks, axis=0)
            
            if gen_count > 0 and len(full_matrix) >= gen_count:
                output_matrix = full_matrix[-gen_count:]
            elif gen_count > 0:
                # Not enough tokens yet, return what we have
                output_matrix = full_matrix
            else:
                output_matrix = np.empty((0, full_matrix.shape[1]))

            return best_root, output_matrix.T.tolist()
            
    return None, None

def match_stream(gen_count: int, prompt_len_est: int) -> Optional[np.ndarray]:
    """Legacy wrapper that pops the stream."""
    _, result = find_matching_stream(gen_count, prompt_len_est, pop=True)
    return result

@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 1. Always stream to count tokens accurately
    body["stream"] = True 
    
    # 2. Estimate Prompt Length (for matching)
    messages = body.get("messages", [])
    prompt_text = " ".join([m.get("content", "") for m in messages])
    prompt_len_est = len(prompt_text) // 4

    async def generate():
        full_content = ""
        gen_token_count = 0
        token_strings = []
        last_probe_update = 0
        matched_root_id = None  # Track which stream we matched
        
        try:
            async with client.stream("POST", VLLM_URL, json=body) as resp:
                if resp.status_code != 200:
                    error_msg = json.dumps({"error": "vLLM Error"})
                    yield f"data: {error_msg}\n\n"
                    return

                # Stream tokens as they arrive
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk['choices'][0].get('delta') or chunk['choices'][0].get('message')
                            if delta and delta.get('content'):
                                content = delta['content']
                                full_content += content
                                gen_token_count += 1
                                token_strings.append(content)
                                
                                # Stream token to frontend immediately
                                token_chunk = {
                                    "type": "token",
                                    "token": content,
                                    "index": gen_token_count - 1
                                }
                                yield f"data: {json.dumps(token_chunk)}\n\n"
                                
                                # Try to get probe data incrementally (every 5 tokens or after initial delay)
                                if gen_token_count >= 5 and (gen_token_count - last_probe_update >= 5 or gen_token_count == 5):
                                    await asyncio.sleep(0.005)  # Brief wait for IPC
                                    
                                    # Try incremental matching (don't pop, just peek)
                                    root_id, probe_scores = find_matching_stream(gen_token_count, prompt_len_est, pop=False)
                                    
                                    if root_id is not None and probe_scores is not None:
                                        if matched_root_id is None:
                                            matched_root_id = root_id
                                        
                                        # Only send if we have data for current token count
                                        if probe_scores and len(probe_scores[0]) >= gen_token_count:
                                            # Trim to current token count
                                            trimmed_scores = [row[:gen_token_count] for row in probe_scores]
                                            
                                            probe_chunk = {
                                                "type": "probe_update",
                                                "probe_scores": trimmed_scores,
                                                "token_count": gen_token_count
                                            }
                                            yield f"data: {json.dumps(probe_chunk)}\n\n"
                                            last_probe_update = gen_token_count
                        except: pass
                    
                    # Forward [DONE] marker
                    if line == "data: [DONE]":
                        break
                        
        except Exception as e:
            error_msg = json.dumps({"error": f"Upstream connection failed: {str(e)}"})
            yield f"data: {error_msg}\n\n"
            return

        # Wait briefly for IPC to catch up
        await asyncio.sleep(0.01)

        # Find & Attach Final Probe Data (more accurate match)
        probe_data = match_stream(gen_token_count, prompt_len_est)
        
        # If we got better data from final match, send it
        if probe_data and matched_root_id:
            final_probe_chunk = {
                "type": "probe_final",
                "probe_scores": probe_data,
                "token_count": gen_token_count
            }
            yield f"data: {json.dumps(final_probe_chunk)}\n\n"

        # Send final message
        final_payload = {
            "type": "done",
            "content": full_content,
            "tokens": token_strings,
            "probe_scores": probe_data,
            "usage": {
                "prompt_tokens": prompt_len_est,
                "completion_tokens": gen_token_count,
                "total_tokens": prompt_len_est + gen_token_count
            }
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    print("🚀 Starting vLLM Probe Proxy on port 6969...")
    print(f"   Targeting vLLM at: {VLLM_URL}")
    uvicorn.run(app, host="0.0.0.0", port=6969)