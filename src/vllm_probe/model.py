from transformers import AutoConfig
from vllm.model_executor.models import ModelRegistry
import importlib
import os
import torch
import glob
import json

TARGET_LAYER = os.environ.get("TARGET_LAYER", None)
if TARGET_LAYER is None:
    raise ValueError("TARGET_LAYER environment variable is not set")

PROBE_MODEL = os.environ.get("PROBE_MODEL", None)
if PROBE_MODEL is None:
    raise ValueError("PROBE_MODEL environment variable is not set")

PROBES_DIR = "/root/clairai/src/probes"

def hf_name_into_model_name(hf_name: str) -> str:
    return hf_name.split("/")[-1]

def parse_layer_from_neuronpedia(layer_str: str) -> int:
    """Parse layer number from Neuronpedia format like '19-resid-post-aa'"""
    try:
        # Extract first number from string
        parts = layer_str.split('_')
        return int(parts[1])
    except:
        return -1

def load_probes_for_layer(layer_idx: int):
    """
    Load all probe files (.pt and .json) for a specific layer from the probes/ directory.
    Returns a tensor of shape [hidden_dim, num_probes] or None if no probes found.
    """
    probe_tensors = []
    probe_names = []
        
    # Search for ALL .json files (we'll filter by layer inside)
    path = f"{PROBES_DIR}/{hf_name_into_model_name(PROBE_MODEL)}/"
    json_files = os.listdir(path)

    print(json_files)
    
    # Load .json files (Neuronpedia format)
    for filepath in sorted(json_files):
        try:
            with open(os.path.join(path, filepath), 'r') as f:
                data = json.load(f)
            
            # Check if this is a Neuronpedia format file
            if 'vector' not in data:
                raise ValueError(f"No vector found in {filepath}")
                        # Convert vector to tensor
            vector = data['vector']
            tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(1)  # [dim] -> [dim, 1]
            
            # Generate probe name from index or filename
            if 'index' in data:
                probe_name = f"np_{data.get('layer', 'unk')}_{data['index']}"
            else:
                probe_name = os.path.splitext(os.path.basename(filepath))[0]
            
            # Add label if available
            if data.get('vectorLabel'):
                probe_name = f"{probe_name}_{data['vectorLabel'][:20]}"
            
            probe_tensors.append(tensor)
            probe_names.append(probe_name)
            print(f">> [PROBE] Loaded JSON {probe_name}: shape {tensor.shape}")
        except Exception as e:
            print(f">> [PROBE] Failed to load JSON {filepath}: {e}")
    
    if not probe_tensors:
        print(f">> [PROBE] No probe files found for layer {layer_idx} in {PROBES_DIR}")
        return None, []
    
    # Stack all probes: each is [dim, 1], concatenate along dim 1
    combined = torch.cat(probe_tensors, dim=1)
    print(f">> [PROBE] Combined probe matrix: {combined.shape} ({len(probe_names)} probes)")
    return combined, probe_names

def get_probed_class(target_model):
    print(f">> [PLUGIN] 🚀 Inspecting architecture for: {target_model}")

    hf_config = AutoConfig.from_pretrained(target_model)
    target_arch_name = hf_config.architectures[0]
    
    entry = ModelRegistry.models.get(target_arch_name)
    if entry:
        # entry can be a class or a lazy loader object
        if hasattr(entry, "module_name") and hasattr(entry, "class_name"):
            # Lazy Loader logic
            mod = importlib.import_module(entry.module_name)
            BaseClass = getattr(mod, entry.class_name)
        else:
            BaseClass = entry

    class ProbedModel(BaseClass):
        def __init__(self, vllm_config=None, prefix="", **kwargs):
            super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
            self.target_layer_idx = TARGET_LAYER
            print(f">> [PLUGIN] 🎯 Target layer index: {self.target_layer_idx}")

            loaded_probes, probe_names = load_probes_for_layer(self.target_layer_idx)
            
            if loaded_probes is None:
                raise ValueError(f"No probes found for layer {self.target_layer_idx}")

            self.num_probes = loaded_probes.shape[1]  # shape is [dim, num_probes]
            self.probe_names = probe_names

            self.register_buffer("probe_dirs", loaded_probes.to(torch.float32))
            self.probe_norms = torch.norm(loaded_probes, dim=0)
            print(f">> [PROBE] Probe norms: {self.probe_norms.tolist()}")

            self._probe_dirs_casted = None

        def forward(self, *args, **kwargs):
            return super().forward(*args, **kwargs)

        def _attach_hook(self):
            layers = self.model.layers

            layers[self.target_layer_idx].register_forward_hook(self._probe_hook)
            print(f">> [PROBE] 🪝 Hook attached to Layer {self.target_layer_idx}")

        def _probe_hook(self, module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            self._run_probes_logic(hidden_states)

        def _run_probes_logic(self, hidden_states):
            target_device = hidden_states.device
            target_dtype = hidden_states.dtype

            self._probe_dirs_casted = self.probe_dirs.to(device=target_device, dtype=target_dtype)

            scores = torch.matmul(hidden_states, self._probe_dirs_casted)

            # Store on the model instance (stays on GPU)
            self.last_probe_scores = scores  # [batch_tokens, num_probes]
            self.last_input_ids = self.current_input_ids
            self.last_positions = self.current_positions

        def forward(self, *args, **kwargs):
            hidden_states = super().forward(*args, **kwargs)
            self._run_probes_logic(hidden_states)
            return hidden_states

        def __del__(self):
            self.running = False

    return ProbedModel, BaseClass, target_arch_name