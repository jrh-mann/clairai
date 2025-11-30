#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// -------------------------------------------------------------------------
// NEW SLOT LAYOUT (Must match sidecar.py):
// 1. [batch_id: 8 bytes]
// 2. [num_requests: 4 bytes]
// 3. [stable_ids: (num_requests * 2) * 4 bytes]  <-- NEW INSERTION
// 4. [query_start_loc: (num_requests+1) * 4 bytes]
// 5. [probe_scores: num_tokens * num_probes * 4 bytes]
// -------------------------------------------------------------------------

__global__ void write_batch_kernel(
    uint64_t batch_id,
    const int32_t* __restrict__ query_start_loc, 
    const int32_t* __restrict__ stable_ids,       // <-- NEW ARGUMENT
    int num_requests,
    const float* __restrict__ probe_scores,      
    int num_tokens,
    int num_probes,
    void* ring_base_ptr,
    int ring_size,
    int slot_size_bytes,
    uint64_t* write_head_ptr
) {
    // 1. READ HEAD
    // We read the current head to determine where to write
    uint64_t current_seq = *write_head_ptr;
    int slot_idx = current_seq % ring_size;

    // 2. CALCULATE SLOT ADDRESS
    char* base_char = (char*)ring_base_ptr;
    // Skip the first 128 bytes (reserved for ring header/metadata)
    char* dst_slot = base_char + 128 + (slot_idx * slot_size_bytes);

    // 3. WRITE HEADER (Single thread)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *((uint64_t*)dst_slot) = batch_id;      // Offset 0
        *((int32_t*)(dst_slot + 8)) = num_requests; // Offset 8
    }
    __syncthreads();

    // --- CALCULATE OFFSETS ---
    // Offset 12: Start of Stable IDs
    char* stable_ids_dst = dst_slot + 12;
    int stable_ids_count = num_requests * 2;
    int stable_ids_bytes = stable_ids_count * sizeof(int32_t);

    // Offset 12 + stable_ids_bytes: Start of Query Start Loc
    char* query_start_loc_dst = stable_ids_dst + stable_ids_bytes;
    int query_start_loc_count = num_requests + 1;
    int query_start_loc_bytes = query_start_loc_count * sizeof(int32_t);

    // Offset after Query Start Loc: Start of Scores
    char* scores_dst = query_start_loc_dst + query_start_loc_bytes;
    int total_scores = num_tokens * num_probes;

    // 4. WRITE STABLE IDs (Grid-Stride Loop) [NEW]
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < stable_ids_count; 
         i += blockDim.x * gridDim.x) {
        ((int32_t*)stable_ids_dst)[i] = stable_ids[i];
    }

    // 5. WRITE QUERY_START_LOC (Grid-Stride Loop)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < query_start_loc_count; 
         i += blockDim.x * gridDim.x) {
        ((int32_t*)query_start_loc_dst)[i] = query_start_loc[i];
    }

    // 6. WRITE PROBE SCORES (Grid-Stride Loop)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < total_scores; 
         i += blockDim.x * gridDim.x) {
        ((float*)scores_dst)[i] = probe_scores[i];
    }

    // 7. MEMORY FENCE
    // Ensure all global memory writes are visible before updating head
    __threadfence_system();

    // 8. UPDATE HEAD
    // Only one thread updates the head
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // atomicAdd checks against race conditions if multiple kernels run
        atomicAdd((unsigned long long*)write_head_ptr, 1ULL);
        __threadfence_system();
    }
}

void launch_ipc_write_batch(
    int64_t batch_id,
    torch::Tensor query_start_loc,  // [num_requests + 1]
    torch::Tensor stable_ids,       // [num_requests, 2] <-- NEW INPUT
    torch::Tensor probe_scores,     // [num_tokens, num_probes]
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
    const int32_t* stable_ids_ptr = stable_ids.data_ptr<int32_t>(); // <-- NEW POINTER
    const float* scores_ptr = probe_scores.data_ptr<float>();
    
    void* ring_base = (void*)ring_base_addr;
    uint64_t* head_ptr = (uint64_t*)ring_base;

    // Launch kernel parameters
    int threads = 256;
    // Count total items to process to approximate grid size
    int total_elements = (num_requests * 2) + (num_requests + 1) + (num_tokens * num_probes);
    int blocks = std::min((total_elements + threads - 1) / threads, 1024);

    write_batch_kernel<<<blocks, threads>>>(
        (uint64_t)batch_id,
        query_start_loc_ptr,
        stable_ids_ptr,      // Pass the new pointer
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
    m.def("write_batch", &launch_ipc_write_batch, "IPC Ring Buffer Write (Updated for Stable IDs)");
}