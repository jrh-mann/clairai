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
    MAX_BATCH = meta["max_batch"]
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
                
                # Read metadata (first 12 bytes: 3 uint32s)
                metadata_mem = cp.cuda.UnownedMemory(src_ptr, 12, None)
                metadata_ptr = cp.cuda.MemoryPointer(metadata_mem, 0)
                metadata_arr = cp.ndarray((3,), dtype=cp.uint32, memptr=metadata_ptr)
                metadata = cp.asnumpy(metadata_arr)
                num_tokens = int(metadata[0])
                slot_seq_low = int(metadata[1])
                slot_seq_high = int(metadata[2])
                slot_seq = slot_seq_low | (slot_seq_high << 32)
                
                # Read probe scores (after 12-byte metadata)
                data_ptr = src_ptr + 12
                raw_data = cp.ndarray(
                    shape=((SLOT_SIZE_BYTES - 12) // 4,), dtype=cp.float32,
                    memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(data_ptr, SLOT_SIZE_BYTES - 12, None), 0)
                )
                host_data = cp.asnumpy(raw_data)
                
                # Reshape: [MAX_BATCH, NUM_PROBES]
                reshaped = host_data[:MAX_BATCH*NUM_PROBES].reshape(MAX_BATCH, NUM_PROBES)
                
                # Only show rows with actual data (up to num_tokens)
                active_data = reshaped[:num_tokens] if num_tokens > 0 else reshaped
                active_mask = np.any(active_data != 0, axis=1) if num_tokens > 0 else np.any(reshaped != 0, axis=1)
                
                if np.any(active_mask):
                    print(f"[Slot {seq}] 🟢 slot_seq={slot_seq}, num_tokens={num_tokens}, active_rows={np.sum(active_mask)}")
                    # Show probe scores for first active token
                    first_active_idx = np.where(active_mask)[0][0] if np.any(active_mask) else 0
                    print(f"       Token {first_active_idx}: probe_scores = {active_data[first_active_idx, :]}")
                else:
                    print(f"[Slot {seq}] 🔴 slot_seq={slot_seq}, num_tokens={num_tokens}, but all zeros")

            last_seq = curr_seq
        else:
            # [FIX] Better debug output when stuck
            no_update_count += 1
            if no_update_count % 1000 == 0:  # Every ~1 second (1000 * 0.001s)
                print(f"⏳ Waiting... Head stuck at {curr_seq} (no change for {no_update_count}ms)")
            time.sleep(0.001)

if __name__ == "__main__":
    main()