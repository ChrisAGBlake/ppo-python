# uncomment top 2 lines to generate html coverage report
# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

extra_compile_args = ["-march=native", "-mtune=native", "-ftree-vectorize", "-O3"]
packageName = "gym_cython"
ext_modules = [
    Extension(
        packageName,
        ["gym_cython.pyx"],
        libraries=[],
        library_dirs=[],
        runtime_library_dirs=[],
        extra_link_args=[],
        extra_compile_args=extra_compile_args,
        include_dirs=[],
    )
]


setup(
    name="gym_cython",
    version="1.0.0",
    ext_modules=cythonize(ext_modules),
    description="cython based gym environment for one player prestart",
    install_requires=["cython"],
)
