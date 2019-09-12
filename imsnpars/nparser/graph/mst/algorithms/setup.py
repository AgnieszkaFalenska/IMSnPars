from distutils.core import setup
import numpy

from Cython.Build import cythonize

setup(ext_modules=cythonize("cyEisner.pyx", include_path=[numpy.get_include()]))
setup(ext_modules=cythonize("cyEisnerO2g.pyx", include_path=[numpy.get_include()]))
setup(ext_modules=cythonize("cyEisnerO2sib.pyx", include_path=[numpy.get_include()]))
