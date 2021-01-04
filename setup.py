from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

extra_compile_args = ['-march=native',  '-mtune=native', '-ftree-vectorize', '-O3']

packageName = 'cy_ppo'
ext_modules = [
	Extension(packageName, ['cy_ppo.pyx'],
	libraries=[],
	library_dirs=[],
	runtime_library_dirs=[],
	extra_link_args=[],
	extra_compile_args=extra_compile_args,
	include_dirs=[]
	)
]


setup(name = 'cy_ppo',
  version='1.0.0',
  ext_modules=cythonize(ext_modules),
  description='cython functions for ppo',
  install_requires=['cython'],
)
