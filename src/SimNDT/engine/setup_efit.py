from distutils.core import setup
from distutils.extension import Extension

import numpy

try:
    from Cython.Distutils import build_ext
except ImportError:
    source_file = "efit2dcython.c"
    cmdclass = {}
else:
    source_file = "EFIT2Dcython.pyx"
    cmdclass = {"build_ext": build_ext}

ext = Extension("efit2dcython", [source_file], include_dirs=[numpy.get_include()])

setup(ext_modules=[ext], cmdclass=cmdclass)
