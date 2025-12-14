from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gpu_cpu_benchmark',
    ext_modules=[
        CUDAExtension(
            name='gpu_cpu_benchmark',
            sources=['torch_bindings.cpp','copy_kernels.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-Xptxas', '-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
