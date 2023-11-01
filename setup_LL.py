from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "LebwohlLasher",
        ["LebwohlLasher.pyx"],
        extra_compile_args=['/openmp','-v'],
#        extra_link_args=['/openmp'],
        extra_link_args=['-lgomp', '-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/13/'],
    )
]
setup(name="LebwohlLasher",
      ext_modules=cythonize(ext_modules),
      include_dirs=[numpy.get_include()]
)

