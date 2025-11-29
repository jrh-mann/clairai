// src/vllm_probe/ipc_writer.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// --- THE KERNEL ---
// This runs on the GPU. It calculates the write address based on the shared
// 'head' counter, copies the data, fences memory, and then increments the head.
__global__ void write_kernel(
    const float* __restrict__ src_data, // The flattened input tensor
    const uint32_t* __restrict__ metadata, // Metadata: [num_tokens, slot_seq_low, slot_seq_high]
    void* ring_base_ptr,                // Raw pointer to the IPC shared memory
    int feature_dim,                    // Number of floats to copy
    int ring_size,                      // Total number of slots in the ring
    int slot_size_bytes,                // Byte size of one slot
    int metadata_size_bytes,            // Size of metadata header in bytes
    uint64_t* write_head_ptr            // Pointer to the monotonic counter
) {
    // 1. READ HEAD (Do not increment yet)
    // We assume a single writer stream (serialized forward pass), so reading 
    // the current head is safe without atomic reservation here.
    // Use volatile to ensure CUDA graphs read the current value, not a captured one
    volatile uint64_t* volatile_head_ptr = (volatile uint64_t*)write_head_ptr;
    uint64_t current_seq = *volatile_head_ptr;
    int slot_idx = current_seq % ring_size;

    // 2. CALCULATE ADDRESS
    // Layout: [Head(128B)] [Slot 0] [Slot 1] ...
    // Each slot: [metadata (12B)] [probe_scores...]
    char* base_char = (char*)ring_base_ptr;
    // 128 bytes offset for the header
    char* dst_slot = base_char + 128 + (slot_idx * slot_size_bytes);

    // 3. WRITE METADATA (only thread 0)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t* dst_metadata = (uint32_t*)dst_slot;
        dst_metadata[0] = metadata[0]; // num_tokens
        dst_metadata[1] = metadata[1]; // slot_seq_low
        dst_metadata[2] = metadata[2]; // slot_seq_high
    }
    __syncthreads();

    // 4. COPY DATA (Grid-Stride Loop)
    // Data starts after metadata
    float* dst_data = (float*)(dst_slot + metadata_size_bytes);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < feature_dim; i += blockDim.x * gridDim.x) {
        dst_data[i] = src_data[i];
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
    torch::Tensor metadata,
    int64_t ring_base_addr, 
    int ring_size,
    int slot_size_bytes,
    int metadata_size_bytes
) {
    int feature_dim = src.numel();
    const float* src_ptr = src.data_ptr<float>();
    const uint32_t* metadata_ptr = metadata.data_ptr<uint32_t>();
    
    void* ring_base = (void*)ring_base_addr;
    uint64_t* head_ptr = (uint64_t*)ring_base; // Head is at offset 0

    int threads = 256;
    // Calculate blocks needed to cover the feature_dim
    int blocks = (feature_dim + threads - 1) / threads;
    // Cap blocks to avoid excessive launch overhead for massive vectors
    blocks = std::min(blocks, 1024);

    write_kernel<<<blocks, threads>>>(
        src_ptr,
        metadata_ptr,
        ring_base, 
        feature_dim, 
        ring_size, 
        slot_size_bytes,
        metadata_size_bytes,
        head_ptr
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("write", &launch_ipc_write, "IPC Ring Buffer Write (CUDA)");
}