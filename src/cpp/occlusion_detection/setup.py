from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='occlusion_detection',
    ext_modules=[
        CUDAExtension('occlusion_detection', [
            'occlusion_detection_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
