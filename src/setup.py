from setuptools import setup, find_packages

setup(
    name="vllm_probe",
    version="0.1",
    packages=find_packages(),
    entry_points={
        # vLLM explicitly looks for plugins in this group:
        "vllm.general_plugins": [
            "register_probe = vllm_probe:register_plugin"
        ]
    }
)