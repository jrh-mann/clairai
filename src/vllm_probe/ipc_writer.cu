// src/vllm_probe/ipc_writer.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Slot layout:
// [batch_id: 8 bytes]
// [num_requests: 4 bytes]
// [query_start_loc: (num_requests+1) * 4 bytes]
// [probe_scores: num_tokens * num_probes * 4 bytes]

__global__ void write_batch_kernel(
    uint64_t batch_id,
    const int32_t* __restrict__ query_start_loc,  // GPU tensor
    int num_requests,
    const float* __restrict__ probe_scores,       // [num_tokens, num_probes]
    int num_tokens,
    int num_probes,
    void* ring_base_ptr,
    int ring_size,
    int slot_size_bytes,
    uint64_t* write_head_ptr
) {
    // 1. READ HEAD
    uint64_t current_seq = *write_head_ptr;
    int slot_idx = current_seq % ring_size;

    // 2. CALCULATE SLOT ADDRESS
    char* base_char = (char*)ring_base_ptr;
    char* dst_slot = base_char + 128 + (slot_idx * slot_size_bytes);

    // 3.  WRITE HEADER (Single thread)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Write batch_id (8 bytes)
        *((uint64_t*)dst_slot) = batch_id;
        
        // Write num_requests (4 bytes)
        *((int32_t*)(dst_slot + 8)) = num_requests;
    }
    __syncthreads();

    // 4. WRITE QUERY_START_LOC (Grid-Stride Loop)
    int query_start_loc_size = num_requests + 1;
    char* query_start_loc_dst = dst_slot + 12;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < query_start_loc_size; 
         i += blockDim.x * gridDim.x) {
        ((int32_t*)query_start_loc_dst)[i] = query_start_loc[i];
    }

    // 5. WRITE PROBE SCORES (Grid-Stride Loop)
    int scores_offset = 12 + (query_start_loc_size * 4);
    char* scores_dst = dst_slot + scores_offset;
    int total_scores = num_tokens * num_probes;
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < total_scores; 
         i += blockDim.x * gridDim.x) {
        ((float*)scores_dst)[i] = probe_scores[i];
    }

    // 6.  MEMORY FENCE
    __threadfence_system();

    // 7. UPDATE HEAD
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd((unsigned long long*)write_head_ptr, 1ULL);
        __threadfence_system();
    }
}

void launch_ipc_write_batch(
    int64_t batch_id,
    torch::Tensor query_start_loc,  // [num_requests + 1] GPU tensor
    torch::Tensor probe_scores,     // [num_tokens, num_probes] GPU tensor
    int64_t ring_base_addr,
    int ring_size,
    int slot_size_bytes
) {
    // Extract dimensions
    int num_requests = query_start_loc.size(0) - 1;
    int num_tokens = probe_scores.size(0);
    int num_probes = probe_scores.size(1);
    
    // Get pointers
    const int32_t* query_start_loc_ptr = query_start_loc.data_ptr<int32_t>();
    const float* scores_ptr = probe_scores.data_ptr<float>();
    void* ring_base = (void*)ring_base_addr;
    uint64_t* head_ptr = (uint64_t*)ring_base;

    // Launch kernel
    int threads = 256;
    int total_elements = num_requests + 1 + (num_tokens * num_probes);
    int blocks = std::min((total_elements + threads - 1) / threads, 1024);

    write_batch_kernel<<<blocks, threads>>>(
        (uint64_t)batch_id,
        query_start_loc_ptr,
        num_requests,
        scores_ptr,
        num_tokens,
        num_probes,
        ring_base,
        ring_size,
        slot_size_bytes,
        head_ptr
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("write_batch", &launch_ipc_write_batch, "IPC Ring Buffer Write with Batch Info (CUDA)");
}