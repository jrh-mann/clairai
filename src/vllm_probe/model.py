from transformers import AutoConfig
from vllm.model_executor.models import ModelRegistry
import importlib
import os
import torch
import glob
import json
import ctypes
import vllm_ipc  # Ensure this is installed

TARGET_LAYER = int(os.environ.get("TARGET_LAYER", "19"))
PROBE_MODEL = os.environ.get("PROBE_MODEL", None)
PROBES_DIR = "/root/clairai/src/probes"

RING_SIZE = 2000
MAX_TOKENS_PER_BATCH = 128

def hf_name_into_model_name(hf_name: str) -> str:
    return hf_name.split("/")[-1]

def load_probes_for_layer(layer_idx: int):
    # (Same loading logic as before, abbreviated for clarity)
    probe_tensors = []
    probe_names = []
    
    path = f"{PROBES_DIR}/{hf_name_into_model_name(PROBE_MODEL)}/"
    print(path)
    try:
        json_files = sorted(os.listdir(path))
    except FileNotFoundError:
        return None, []

    for filepath in json_files:
        if not filepath.endswith('.json'): continue
        try:
            with open(os.path.join(path, filepath), 'r') as f:
                data = json.load(f)
            if 'vector' in data:
                vector = data['vector']
                tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(1)
                probe_tensors.append(tensor)
                # Naive naming
                probe_names.append(f"probe_{len(probe_names)}")
        except:
            continue
    
    if not probe_tensors:
        return None, []
    
    return torch.cat(probe_tensors, dim=1), probe_names

def get_probed_class(target_model):
    print(f">> [PLUGIN] 🚀 Inspecting architecture for: {target_model}")
    hf_config = AutoConfig.from_pretrained(target_model)
    target_arch_name = hf_config.architectures[0]
    
    entry = ModelRegistry.models.get(target_arch_name)
    if entry:
        if hasattr(entry, "module_name") and hasattr(entry, "class_name"):
            mod = importlib.import_module(entry.module_name)
            BaseClass = getattr(mod, entry.class_name)
        else:
            BaseClass = entry

    class ProbedModel(BaseClass):
        def __init__(self, vllm_config=None, prefix="", **kwargs):
            super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
            self.target_layer_idx = TARGET_LAYER
            
            loaded_probes, probe_names = load_probes_for_layer(self.target_layer_idx)
            if loaded_probes is None:
                raise ValueError(f"No probes found for layer {self.target_layer_idx}")

            self.num_probes = loaded_probes.shape[1]
            self.register_buffer("probe_dirs", loaded_probes.to(torch.float32))
            
            # --- IPC SETUP (FIXED) ---
            # Slot layout: [num_tokens (uint32), slot_seq (uint64), probe_scores...]
            # Metadata: 4 bytes (num_tokens) + 8 bytes (slot_seq) = 12 bytes
            self.metadata_size_bytes = 12
            self.slot_floats = self.num_probes * MAX_TOKENS_PER_BATCH
            self.slot_size_bytes = self.metadata_size_bytes + (self.slot_floats * 4)
            if self.slot_size_bytes % 512 != 0:
                self.slot_size_bytes += (512 - (self.slot_size_bytes % 512))
            
            total_bytes = 128 + (RING_SIZE * self.slot_size_bytes)
            
            # 1. Allocate Buffer using cudaMalloc for IPC compatibility
            # PyTorch tensors may not be allocated in IPC-shareable memory
            cudart = ctypes.CDLL('libcudart.so')
            cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            cudart.cudaMalloc.restype = ctypes.c_int
            
            ipc_ptr = ctypes.c_void_p()
            result = cudart.cudaMalloc(ctypes.byref(ipc_ptr), total_bytes)
            if result != 0:
                raise RuntimeError(f"cudaMalloc failed: {result}")
            self.ipc_ptr = ipc_ptr.value
            
            # 2. Zero the memory
            cudart.cudaMemset(ctypes.c_void_p(self.ipc_ptr), 0, total_bytes)
            torch.cuda.synchronize()
            
            # 3. Create a PyTorch tensor that directly views the CUDA memory
            # Use torch's as_tensor with a memory pointer to avoid copying
            import cupy as cp
            mem = cp.cuda.UnownedMemory(self.ipc_ptr, total_bytes, None)
            memptr = cp.cuda.MemoryPointer(mem, 0)
            cp_array = cp.ndarray((total_bytes,), dtype=cp.uint8, memptr=memptr)
            # Use DLPack to share memory between CuPy and PyTorch without copying
            self.ipc_buffer = torch.utils.dlpack.from_dlpack(cp_array.toDlpack()).view(torch.uint8)

            self._export_ipc_handle()
            self._probe_dirs_casted = None
            
            # 3. [FIX] CALL THE HOOK ATTACHMENT
            self._attach_hook()

        def _export_ipc_handle(self):
            # (Export logic same as before)
            cudart = ctypes.CDLL('libcudart.so')
            class IpcMemHandle(ctypes.Structure):
                _fields_ = [("reserved", ctypes.c_char * 64)]
            handle = IpcMemHandle()
            
            # [DEBUG] Verify pointer before creating handle
            print(f">> [IPC] Creating handle for pointer: {hex(self.ipc_ptr)}")
            
            # Check return value
            result = cudart.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(self.ipc_ptr))
            if result != 0:
                error_str = ctypes.c_char_p()
                cudart.cudaGetErrorString(result, ctypes.byref(error_str))
                raise RuntimeError(f"cudaIpcGetMemHandle failed: {result} ({error_str.value})")
            
            with open("/tmp/vllm_probe.ipc", "wb") as f:
                f.write(bytes(handle))
            with open("/tmp/vllm_probe_meta.json", "w") as f:
                json.dump({
                    "ring_size": RING_SIZE, "num_probes": self.num_probes,
                    "slot_size_bytes": self.slot_size_bytes, "max_batch": MAX_TOKENS_PER_BATCH,
                    "vllm_ptr": hex(self.ipc_ptr)  # [DEBUG] Export the pointer for comparison
                }, f)
            
            # [DEBUG] Verify head is still 0 after export
            head_check = self.ipc_buffer[:8].view(torch.uint64)[0].cpu().item()
            # [DEBUG] Write a test pattern to verify IPC works
            test_pattern = torch.tensor([0xDEADBEEFCAFEBABE], dtype=torch.uint64, device="cuda")
            self.ipc_buffer[8:16].copy_(test_pattern.view(torch.uint8))
            torch.cuda.synchronize()
            print(f">> [IPC] Handle exported. Buffer ptr: {hex(self.ipc_ptr)}, Head value: {head_check}")
            print(f">> [IPC] Test pattern written at offset 8: 0xDEADBEEFCAFEBABE")

        def _attach_hook(self):
            # [FIX] Actually attach the hook to the correct layer
            layers = self.model.layers
            if self.target_layer_idx >= len(layers):
                raise ValueError(f"Target layer {self.target_layer_idx} out of bounds")
            
            layers[self.target_layer_idx].register_forward_hook(self._probe_hook)
            print(f">> [PROBE] 🪝 Hook attached to Layer {self.target_layer_idx}")

        def _probe_hook(self, module, input, output):
            # [DEBUG] Verify hook is being called
            print(">> [HOOK] 🎣 Probe hook called!")
            # Capture the output of the layer
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Try to extract sequence metadata from the input
            # In vLLM, input may contain kwargs with sequence info
            seq_ids = None
            token_positions = None
            if isinstance(input, tuple) and len(input) > 0:
                # Check if input has sequence metadata
                # vLLM often passes kwargs as part of input
                try:
                    # Access the model's forward kwargs if available
                    if hasattr(module, '_last_forward_kwargs'):
                        kwargs = module._last_forward_kwargs
                        if 'seq_ids' in kwargs:
                            seq_ids = kwargs['seq_ids']
                        if 'positions' in kwargs:
                            token_positions = kwargs['positions']
                except:
                    pass
            
            self._run_probes_logic(hidden_states, seq_ids, token_positions)

        def _run_probes_logic(self, hidden_states, seq_ids=None, token_positions=None):
            target_device = hidden_states.device
            target_dtype = hidden_states.dtype

            if self._probe_dirs_casted is None or self._probe_dirs_casted.dtype != target_dtype:
                self._probe_dirs_casted = self.probe_dirs.to(device=target_device, dtype=target_dtype)

            scores = torch.matmul(hidden_states, self._probe_dirs_casted)

            # [FIX] Ensure contiguous memory for C++ kernel
            scores_float = scores.float().contiguous()
            
            # Extract batch info
            batch_size, num_probes = scores_float.shape
            if len(scores_float.shape) == 3:  # [batch, seq_len, num_probes]
                batch_size, seq_len, num_probes = scores_float.shape
                scores_float = scores_float.reshape(-1, num_probes)  # Flatten to [batch*seq_len, num_probes]
                batch_size = batch_size * seq_len

            print(f">> [PROBE] 🔍 Probing scores: {scores_float.shape}, batch_size: {batch_size}")
            
            # Create metadata: [num_tokens, seq_ids..., token_positions...]
            # For now, use sequence number (head counter) as seq_id if not available
            # Token positions default to 0, 1, 2, ... if not available
            if seq_ids is None:
                # Use a simple counter - in practice you'd get this from vLLM
                seq_ids = torch.zeros(batch_size, dtype=torch.int32, device=target_device)
            if token_positions is None:
                # Default to sequential positions
                token_positions = torch.arange(batch_size, dtype=torch.int32, device=target_device)
            
            # Ensure metadata is the right size
            if isinstance(seq_ids, torch.Tensor):
                seq_ids = seq_ids[:batch_size].to(dtype=torch.int32, device=target_device)
            else:
                seq_ids = torch.tensor(seq_ids[:batch_size], dtype=torch.int32, device=target_device)
                
            if isinstance(token_positions, torch.Tensor):
                token_positions = token_positions[:batch_size].to(dtype=torch.int32, device=target_device)
            else:
                token_positions = torch.tensor(token_positions[:batch_size], dtype=torch.int32, device=target_device)

            print(f">> [PROBE] 📝 Calling vllm_ipc.write() with {scores_float.numel()} elements, batch_size: {batch_size}")
            
            # [DEBUG] Read head value BEFORE kernel call (this will be the slot_seq)
            head_view_before = self.ipc_buffer[:8].view(torch.uint64)
            head_before_val = int(head_view_before[0].cpu().item())
            slot_seq = head_before_val  # Use current head as slot sequence number
            print(f">> [PROBE] 📊 Head BEFORE: {head_before_val} (slot_seq: {slot_seq})")

            # Prepare metadata: [num_tokens (uint32), slot_seq (uint64 stored as 2 uint32s)]
            metadata = torch.zeros(3, dtype=torch.int32, device=target_device)  # 3 int32s = 12 bytes (signed for easier casting)
            metadata[0] = int(batch_size)  # num_tokens
            # Store slot_seq as two uint32s (little-endian)
            slot_seq_low = int(slot_seq & 0xFFFFFFFF)
            slot_seq_high = int((slot_seq >> 32) & 0xFFFFFFFF)
            metadata[1] = slot_seq_low
            metadata[2] = slot_seq_high
            # Convert to uint32 view for kernel
            metadata_uint32 = metadata.to(torch.uint32)

            vllm_ipc.write(
                scores_float,
                metadata_uint32,
                self.ipc_ptr, 
                RING_SIZE, 
                self.slot_size_bytes,
                self.metadata_size_bytes
            )
            
            # [DEBUG] Add synchronization to ensure kernel completes
            torch.cuda.synchronize()
            
            # [DEBUG] Read head value AFTER kernel call
            head_view_after = self.ipc_buffer[:8].view(torch.uint64)
            head_after_val = int(head_view_after[0].cpu().item())
            delta = head_after_val - head_before_val
            print(f">> [PROBE] 📊 Head AFTER: {head_after_val} (Delta: {delta})")
            if delta != 1:
                print(f">> [PROBE] ⚠️ WARNING: Head should increment by 1, but delta is {delta}!")
            print(f">> [PROBE] ✅ Kernel write completed")

        # [FIX] Removed the 'forward' override entirely. 
        # We rely on the hook attached in __init__ now.

    return ProbedModel, BaseClass, target_arch_name