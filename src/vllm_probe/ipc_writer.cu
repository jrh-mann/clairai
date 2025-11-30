#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// -------------------------------------------------------------------------
// KERNEL: The low-level writer (unchanged, but crucial)
// -------------------------------------------------------------------------
__global__ void write_batch_kernel(
    uint64_t batch_id,
    const int32_t* __restrict__ query_start_loc, 
    const int32_t* __restrict__ stable_ids,
    int num_requests,
    const float* __restrict__ probe_scores,      
    int num_tokens,
    int num_probes,
    void* ring_base_ptr,
    int ring_size,
    int slot_size_bytes,
    uint64_t* write_head_ptr
) {
    uint64_t current_seq = *write_head_ptr;
    int slot_idx = current_seq % ring_size;
    char* base_char = (char*)ring_base_ptr;
    char* dst_slot = base_char + 128 + (slot_idx * slot_size_bytes);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *((uint64_t*)dst_slot) = batch_id;
        *((int32_t*)(dst_slot + 8)) = num_requests;
    }
    __syncthreads();

    char* stable_ids_dst = dst_slot + 12;
    int stable_ids_count = num_requests * 2;
    int stable_ids_bytes = stable_ids_count * sizeof(int32_t);

    char* query_start_loc_dst = stable_ids_dst + stable_ids_bytes;
    int query_start_loc_count = num_requests + 1;
    int query_start_loc_bytes = query_start_loc_count * sizeof(int32_t);

    char* scores_dst = query_start_loc_dst + query_start_loc_bytes;
    int total_scores = num_tokens * num_probes;

    // Grid-Stride loops for data copy
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < stable_ids_count; i += blockDim.x * gridDim.x) {
        ((int32_t*)stable_ids_dst)[i] = stable_ids[i];
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < query_start_loc_count; i += blockDim.x * gridDim.x) {
        ((int32_t*)query_start_loc_dst)[i] = query_start_loc[i];
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_scores; i += blockDim.x * gridDim.x) {
        ((float*)scores_dst)[i] = probe_scores[i];
    }

    __threadfence_system(); // The A40 handles this fast (20us), so keep it.

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd((unsigned long long*)write_head_ptr, 1ULL);
        __threadfence_system();
    }
}

// Helper to launch the kernel
void launch_ipc_write_batch(
    int64_t batch_id,
    torch::Tensor query_start_loc,
    torch::Tensor stable_ids,
    torch::Tensor probe_scores,
    int64_t ring_base_addr,
    int ring_size,
    int slot_size_bytes
) {
    int num_requests = query_start_loc.size(0) - 1;
    int num_tokens = probe_scores.size(0);
    int num_probes = probe_scores.size(1);
    
    int threads = 256;
    int total_elements = (num_requests * 2) + (num_requests + 1) + (num_tokens * num_probes);
    int blocks = std::min((total_elements + threads - 1) / threads, 1024);

    write_batch_kernel<<<blocks, threads>>>(
        (uint64_t)batch_id,
        query_start_loc.data_ptr<int32_t>(),
        stable_ids.data_ptr<int32_t>(),
        num_requests,
        probe_scores.data_ptr<float>(), // Expects FP32 here!
        num_tokens,
        num_probes,
        (void*)ring_base_addr,
        ring_size,
        slot_size_bytes,
        (uint64_t*)ring_base_addr
    );
}

// -------------------------------------------------------------------------
// NEW: THE FUSED "ALL-IN-ONE" OPERATOR
// -------------------------------------------------------------------------
void fused_probe_forward(
    torch::Tensor hidden_states,
    torch::Tensor probe_dirs,   // [Hidden, Probes]
    torch::Tensor probe_biases, // [1, Probes]
    int64_t batch_id,
    torch::Tensor query_start_loc,
    torch::Tensor stable_ids,
    int64_t ring_base_addr,
    int ring_size,
    int slot_size_bytes
) {
    // 1. MATH: Matrix Multiply (Using PyTorch C++ API, no Python overhead)
    //    Calculates: Hidden @ Probes
    auto scores = torch::matmul(hidden_states, probe_dirs);
    
    // 2. MATH: Add Bias
    //    In-place add is faster: scores += bias
    scores.add_(probe_biases);
    
    // 3. CAST: Ensure FP32 and Contiguous for the writer kernel
    //    This handles the BF16 -> FP32 conversion if necessary
    auto final_scores = scores.to(torch::kFloat32).contiguous();

    // 4. WRITE: Launch the kernel immediately
    launch_ipc_write_batch(
        batch_id,
        query_start_loc,
        stable_ids,
        final_scores,
        ring_base_addr,
        ring_size,
        slot_size_bytes
    );
}

// Register both functions (Legacy + New)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("write_batch", &launch_ipc_write_batch, "Legacy Writer");
    m.def("fused_forward", &fused_probe_forward, "Fused MatMul+Add+Write");
}