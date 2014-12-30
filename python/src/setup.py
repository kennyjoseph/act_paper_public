__author__ = 'kjoseph'

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("functions",["functions.pyx"], include_dirs=[numpy.get_include()],libraries=['m'],language="c++"),
    Extension("deflection",["deflection.pyx"], include_dirs=[numpy.get_include()],libraries=['m'],language="c++")
]
setup(
    name = "ACT Model",
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()],  # accepts a glob pattern
)