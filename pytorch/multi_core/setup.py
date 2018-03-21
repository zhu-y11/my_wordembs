from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('data_producer.pyx'),
      include_dirs = [numpy.get_include()])

setup(ext_modules = cythonize('data_cython.pyx'),
      include_dirs = [numpy.get_include()])
