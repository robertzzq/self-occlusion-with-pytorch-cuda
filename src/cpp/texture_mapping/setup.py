from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='texture_mapping',
    ext_modules=[
        CUDAExtension('texture_mapping', [
            'texture_mapping_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
