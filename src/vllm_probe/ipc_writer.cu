// src/vllm_probe/ipc_writer.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// --- THE KERNEL ---
// This runs on the GPU. It calculates the write address based on the shared
// 'head' counter, copies the data, fences memory, and then increments the head.
__global__ void write_kernel(
    const float* __restrict__ src_data, // The flattened input tensor
    void* ring_base_ptr,                // Raw pointer to the IPC shared memory
    int feature_dim,                    // Number of floats to copy
    int ring_size,                      // Total number of slots in the ring
    int slot_size_bytes,                // Byte size of one slot
    uint64_t* write_head_ptr,
    uint64_t request_id
) {
    // 1. READ HEAD (Do not increment yet)
    // We assume a single writer stream (serialized forward pass), so reading 
    // the current head is safe without atomic reservation here.
    uint64_t current_seq = *write_head_ptr;
    int slot_idx = current_seq % ring_size;

    // 2. CALCULATE ADDRESS
    // Layout: [Head(128B)] [Slot 0] [Slot 1] ...
    char* base_char = (char*)ring_base_ptr;

    char* dst_slot = base_char + 128 + (slot_idx * slot_size_bytes);
    
    // Write request_id at the start of the slot (first 8 bytes)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *((uint64_t*)dst_slot) = request_id;
    }

    // 3. COPY DATA (Grid-Stride Loop)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < feature_dim; i += blockDim.x * gridDim.x) {
        dst_slot[i] = src_data[i];
    }

    // 4. MEMORY FENCE
    // This is the most critical line. We must ensure the data is visible 
    // over PCIe (System scope) before we update the head counter.
    __threadfence_system();

    // 5. UPDATE HEAD
    // Only one thread commits the transaction.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd((unsigned long long*)write_head_ptr, 1ULL);
        // [FIX] Additional fence after atomic to ensure visibility to sidecar process
        __threadfence_system();
    }
}

// --- THE LAUNCHER ---
// Called from Python. Sets up the kernel launch.
void launch_ipc_write(
    torch::Tensor src, 
    int64_t ring_base_addr, 
    int ring_size,
    int slot_size_bytes,
    uint64_t request_id
) {
    int feature_dim = src.numel();
    const float* src_ptr = src.data_ptr<float>();
    
    void* ring_base = (void*)ring_base_addr;
    uint64_t* head_ptr = (uint64_t*)ring_base; // Head is at offset 0

    int threads = 256;
    // Calculate blocks needed to cover the feature_dim
    int blocks = (feature_dim + threads - 1) / threads;
    // Cap blocks to avoid excessive launch overhead for massive vectors
    blocks = std::min(blocks, 1024);

    write_kernel<<<blocks, threads>>>(
        src_ptr, 
        ring_base, 
        feature_dim, 
        ring_size, 
        slot_size_bytes, 
        head_ptr,
        request_id
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("write", &launch_ipc_write, "IPC Ring Buffer Write (CUDA)");
}