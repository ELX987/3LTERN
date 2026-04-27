from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='aq_kernel',
    ext_modules=[
        CUDAExtension(
            name='aq_kernel',
            sources=['aq_wrapper.cpp', 'aq_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math', '-arch=sm_100']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
