Overview

This setup modifies your existing clairai probe plugin to use CUDA IPC (Inter-Process Communication).

The Writer (vLLM): A custom C++/CUDA kernel inside the model process writes activation scores directly to a fixed ring buffer in VRAM. This bypasses the Python interpreter, making it compatible with CUDA Graphs.

The Reader (Sidecar): A separate Python process reads from that same VRAM location via a shared handle, streaming data without slowing down the model.

1. Directory Structure

Ensure your project looks like this (new files marked with +):

Plaintext
clairai/
├── requirements.txt
├── sidecar.py                  <-- (+) The consumer script
├── src/
│   ├── setup.py                <-- (Modified) To build CUDA extension
│   └── vllm_probe/
│       ├── __init__.py
│       ├── model.py            <-- (Modified) To use IPC buffer
│       └── ipc_writer.cu       <-- (+) The C++ Kernel
2. Installation & Build

Step A: Create the Kernel

Create src/vllm_probe/ipc_writer.cu. This file contains the "Graph-Safe" writing logic.

<details> <summary><b>Click to see code for <code>ipc_writer.cu</code></b></summary>

C++
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void write_kernel(
    const float* __restrict__ src_data,
    void* ring_base_ptr,
    int feature_dim,
    int ring_size,
    int slot_size_bytes,
    uint64_t* write_head_ptr
) {
    uint64_t current_seq = *write_head_ptr;
    int slot_idx = current_seq % ring_size;
    
    char* base_char = (char*)ring_base_ptr;
    float* dst_slot = (float*)(base_char + 128 + (slot_idx * slot_size_bytes));

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < feature_dim; i += blockDim.x * gridDim.x) {
        dst_slot[i] = src_data[i];
    }

    __threadfence_system(); // CRITICAL: Ensures data is visible to Sidecar

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd((unsigned long long*)write_head_ptr, 1ULL);
    }
}

void launch_ipc_write(torch::Tensor src, int64_t ring_base_addr, int ring_size, int slot_size_bytes) {
    int feature_dim = src.numel();
    const float* src_ptr = src.data_ptr<float>();
    void* ring_base = (void*)ring_base_addr;
    uint64_t* head_ptr = (uint64_t*)ring_base;

    int threads = 256;
    int blocks = std::min((feature_dim + threads - 1) / threads, 1024);

    write_kernel<<<blocks, threads>>>(src_ptr, ring_base, feature_dim, ring_size, slot_size_bytes, head_ptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("write", &launch_ipc_write, "IPC Ring Buffer Write");
}
</details>

Step B: Update setup.py

Modify src/setup.py to compile the CUDA extension.

Python
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="vllm_probe",
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='vllm_ipc', 
            sources=['vllm_probe/ipc_writer.cu'],
            extra_compile_args={'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    entry_points={
        "vllm.general_plugins": ["register_probe = vllm_probe:register_plugin"]
    }
)
Step C: Update model.py

Update src/vllm_probe/model.py with the logic provided in the previous turn (initializing the buffer and calling vllm_ipc.write in the hook).

Step D: Compile

Run this from the project root:

Bash
pip install -e src/
You should see output about building 'vllm_ipc' extension using nvcc.

3. The Sidecar

Create sidecar.py in your root directory. This script will attach to the running vLLM process.

<details> <summary><b>Click to see code for <code>sidecar.py</code></b></summary>

Python
import time, json, os, cupy as cp, numpy as np

IPC_PATH = "/tmp/vllm_probe.ipc"
META_PATH = "/tmp/vllm_probe_meta.json"

def main():
    print(">> [Sidecar] Waiting for vLLM...")
    while not os.path.exists(IPC_PATH): time.sleep(0.5)

    with open(META_PATH, 'r') as f: meta = json.load(f)
    with open(IPC_PATH, "rb") as f: handle = f.read()
    
    gpu_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle)
    head = cp.ndarray((1,), dtype=cp.uint64, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(gpu_ptr, 128, None), 0))
    data_base = gpu_ptr + 128
    
    last_seq = 0
    print(f">> [Sidecar] Connected! Ring Size: {meta['ring_size']}")

    try:
        while True:
            curr_seq = int(head[0])
            if curr_seq > last_seq:
                # Catch up logic
                start = max(last_seq, curr_seq - meta['ring_size'])
                for seq in range(start, curr_seq):
                    offset = (seq % meta['ring_size']) * meta['slot_size_bytes']
                    
                    # Zero-Copy Read via PCIe
                    raw = cp.ndarray((meta['slot_size_bytes']//4,), dtype=cp.float32, 
                                     memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(data_base + offset, meta['slot_size_bytes'], None), 0))
                    
                    # Move to Host (CPU)
                    host_data = cp.asnumpy(raw).reshape(-1, meta['num_probes'])
                    
                    # Filter empty rows (if batch < max)
                    active = host_data[np.any(host_data != 0, axis=1)]
                    if len(active) > 0:
                        print(f"[{seq}] Batch: {len(active)} | Avg Score: {active.mean():.4f}")
                
                last_seq = curr_seq
            else:
                time.sleep(0.001) # Low latency polling
    except KeyboardInterrupt:
        cp.cuda.runtime.ipcCloseMemHandle(gpu_ptr)

if __name__ == "__main__":
    main()
</details>

Requirements for Sidecar:

Bash
pip install cupy-cuda12x  # Match your CUDA version (e.g., 11x or 12x)
4. Execution Guide

You need two terminal windows.

Terminal 1: Start the Sidecar

Start this first. It will wait patiently for vLLM to create the shared memory file.

Bash
python sidecar.py
Terminal 2: Start vLLM

Launch vLLM normally. The vllm.general_plugins mechanism will load your modified code automatically.

Bash
# 1. Set required env vars
export TARGET_LAYER=15
export PROBE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# 2. Run vLLM (ensure --enforce-eager is NOT needed anymore!)
vllm serve $PROBE_MODEL \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048
5. Verification

In Terminal 2 (vLLM):

Look for the log: >> [IPC] Ring Buffer Active: ... MB

Look for: >> [IPC] Handle exported to /tmp/vllm_probe.ipc

In Terminal 1 (Sidecar):

It should immediately print >> [Sidecar] Connected!

As soon as you send a request to vLLM (e.g., via curl), you should see streamed outputs: [1] Batch: 1 | Avg Score: 2.532

Troubleshooting

CUDA error: out of memory: The Ring Buffer allocates persistent VRAM. If vLLM takes 90% (default), you might not have room.

Fix: Reduce vLLM memory usage: vllm serve ... --gpu-memory-utilization 0.85

Sidecar reads all zeros: The __threadfence_system() in the CUDA kernel is missing or the L2 cache isn't flushing. Ensure you compiled the kernel with -O3.

ModuleNotFoundError: No module named 'vllm_ipc': You didn't run pip install -e src/ or you are running vllm from a different python environment.