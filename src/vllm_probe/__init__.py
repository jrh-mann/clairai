import os
from vllm.model_executor.models import ModelRegistry
from .model import get_probed_class

def register_plugin():
    target_model = os.environ.get("PROBE_MODEL", None)

    if target_model is None:
        raise ValueError("PROBE_MODEL environment variable is not set")
    
    print(f">> [PLUGIN] Initializing Probe for: {target_model}")
    
    ProbedClass, BaseClass, TargetArch = get_probed_class(target_model)
        
    # Register under the ARCHITECTURE name (e.g. Qwen2ForCausalLM)
    # This allows vLLM to find our class even if we wrapped a fallback class (Llama)
    ModelRegistry.register_model(TargetArch, ProbedClass)
        
    print(f">> [PLUGIN] 🎯 Hijacked {TargetArch} (using {BaseClass.__name__} base)")
        
    return ProbedClass, BaseClass, TargetArch