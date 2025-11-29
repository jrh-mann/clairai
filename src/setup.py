# src/setup.py
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
    cmdclass={
        'build_ext': BuildExtension
    },
    entry_points={
        "vllm.general_plugins": [
            "register_probe = vllm_probe:register_plugin"
        ]
    }
)