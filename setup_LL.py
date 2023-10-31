from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
setup(name="LebwohlLasher",
      ext_modules=cythonize("LebwohlLasher.pyx"),
      include_dirs=[numpy.get_include()]
)

