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
    vllm_ptr = meta.get("vllm_ptr", "unknown")
    
    with open(IPC_PATH, "rb") as f:
        handle_bytes = f.read()
    
    # Attach to VRAM
    gpu_mem_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes)
    
    # View Data (Offset 128)
    data_base_ptr = gpu_mem_ptr + 128
    
    print(f">> Connected. vLLM ptr: {vllm_ptr}, Sidecar ptr: {hex(gpu_mem_ptr)}")
    print(f">> [DEBUG] Note: Virtual addresses differ, but should point to same physical memory")
    
    # [DEBUG] Read test pattern to verify IPC is working
    def read_test_pattern():
        """Read test pattern at offset 8"""
        cp.cuda.Device().synchronize()
        cudart = ctypes.CDLL('libcudart.so')
        host_buf = (ctypes.c_uint64 * 1)()
        cudart.cudaMemcpy(
            ctypes.byref(host_buf),
            ctypes.c_void_p(gpu_mem_ptr + 8),  # Offset 8
            8,
            ctypes.c_int(2)
        )
        return hex(host_buf[0])
    
    test_val = read_test_pattern()
    print(f">> [DEBUG] Test pattern at offset 8: {test_val} (expected: 0xdeadbeefcafebabe)")
    if test_val.lower() != "0xdeadbeefcafebabe":
        print(f">> [ERROR] IPC memory sharing is NOT working! Test pattern mismatch!")
    else:
        print(f">> [DEBUG] ✓ IPC memory sharing verified!")
    
    def read_head():
        """Read head value using raw CUDA memcpy to avoid any caching"""
        cp.cuda.Device().synchronize()
        # Use raw CUDA runtime to copy directly from GPU to CPU
        cudart = ctypes.CDLL('libcudart.so')
        # Allocate host buffer
        host_buf = (ctypes.c_uint64 * 1)()
        # Copy from device to host
        cudart.cudaMemcpy(
            ctypes.byref(host_buf),
            ctypes.c_void_p(gpu_mem_ptr),
            8,  # 8 bytes for uint64
            ctypes.c_int(2)  # cudaMemcpyDeviceToHost = 2
        )
        return int(host_buf[0])
    
    # [FIX] Initialize last_seq to CURRENT head value, not 0
    # This way we only process NEW data from this point forward
    last_seq = read_head()
    print(f">> Starting from sequence: {last_seq} (skipping warmup data)")
    
    # [DEBUG] Verify we can read the head multiple times
    for i in range(3):
        test_val = read_head()
        print(f">> [DEBUG] Head read #{i+1}: {test_val}")
        time.sleep(1)
    
    # Debug counter to track when head is stuck
    no_update_count = 0
    
    while True:
        # [FIX] Read head value with fresh view each time
        curr_seq = read_head()
        
        # [DEBUG] Print head value more frequently to see if it's changing
        if no_update_count % 100 == 0 and no_update_count > 0:  # Every 100ms
            print(f"🔍 Debug: Head value = {curr_seq}, last_seq = {last_seq}")
        
        if curr_seq > last_seq:
            delta = curr_seq - last_seq
            print(f">> Update! Head: {curr_seq} (Delta: {delta})")
            no_update_count = 0
            
            # Handle Wrap-around or Garbage Jump
            if delta > RING_SIZE:
                print(f"!! JUMP DETECTED ({delta} > {RING_SIZE})!! Skipping to latest.")
                start = curr_seq - 1
            else:
                start = last_seq

            for seq in range(start, curr_seq):
                slot_idx = seq % RING_SIZE
                src_ptr = data_base_ptr + (slot_idx * SLOT_SIZE_BYTES)
                
                cp.cuda.Device().synchronize()
                cudart = ctypes.CDLL('libcudart.so')
                
                # Read batch_id (8 bytes at offset 0)
                host_batch_id = (ctypes.c_uint64 * 1)()
                cudart.cudaMemcpy(
                    ctypes.byref(host_batch_id),
                    ctypes.c_void_p(src_ptr),
                    8,
                    ctypes.c_int(2)  # cudaMemcpyDeviceToHost
                )
                batch_id = int(host_batch_id[0])
                
                # Read num_requests (4 bytes at offset 8)
                host_num_requests = (ctypes.c_int32 * 1)()
                cudart.cudaMemcpy(
                    ctypes.byref(host_num_requests),
                    ctypes.c_void_p(src_ptr + 8),
                    4,
                    ctypes.c_int(2)
                )
                num_requests = int(host_num_requests[0])
                
                if num_requests == 0:
                    continue  # Skip empty slots
                
                # Read query_start_loc (num_requests+1) * 4 bytes at offset 12
                query_start_loc_size = num_requests + 1
                host_query_start_loc = (ctypes.c_int32 * query_start_loc_size)()
                cudart.cudaMemcpy(
                    ctypes.byref(host_query_start_loc),
                    ctypes.c_void_p(src_ptr + 12),
                    query_start_loc_size * 4,
                    ctypes.c_int(2)
                )
                query_start_loc = np.array([int(host_query_start_loc[i]) for i in range(query_start_loc_size)])
                
                # Calculate num_tokens from query_start_loc (last element)
                num_tokens = int(query_start_loc[-1])
                
                # Read probe_scores: num_tokens * num_probes * 4 bytes
                scores_offset = 12 + (query_start_loc_size * 4)
                scores_size = num_tokens * NUM_PROBES
                
                host_scores = (ctypes.c_float * scores_size)()
                cudart.cudaMemcpy(
                    ctypes.byref(host_scores),
                    ctypes.c_void_p(src_ptr + scores_offset),
                    scores_size * 4,
                    ctypes.c_int(2)
                )
                scores_array = np.array([float(host_scores[i]) for i in range(scores_size)])
                
                # Reshape to [num_tokens, num_probes]
                probe_scores = scores_array.reshape(num_tokens, NUM_PROBES)
                
                # Process each request in the batch
                for req_idx in range(num_requests):
                    start_token = int(query_start_loc[req_idx])
                    end_token = int(query_start_loc[req_idx + 1])
                    request_scores = probe_scores[start_token:end_token]
                    
                    if request_scores.size > 0:
                        avg_score = request_scores.mean()
                        print(f"[{seq}] Batch ID: {batch_id}, Request {req_idx}: tokens {start_token}-{end_token} | Avg Score: {avg_score:.4f}")
                    else:
                        print(f"[{seq}] Batch ID: {batch_id}, Request {req_idx}: empty")

            last_seq = curr_seq
        else:
            # [FIX] Better debug output when stuck
            no_update_count += 1
            if no_update_count % 1000 == 0:  # Every ~1 second (1000 * 0.001s)
                print(f"⏳ Waiting... Head stuck at {curr_seq} (no change for {no_update_count}ms)")
            time.sleep(0.001)

if __name__ == "__main__":
    main()