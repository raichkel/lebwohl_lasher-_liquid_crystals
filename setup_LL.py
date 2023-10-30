from distutils.core import setup
from Cython.Build import cythonize

setup(name="LebwohlLasher",
      ext_modules=cythonize("LebwohlLasher.pyx"))

