# compile with:
# python setup_cython.py build_ext --inplace


import numpy
from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

extensions = [
    Extension('utils.im2col_cython', ['utils/im2col_cython.pyx'],
              include_dirs=[numpy.get_include()]
              ),
]

setup(
    ext_modules=cythonize(extensions),
)
