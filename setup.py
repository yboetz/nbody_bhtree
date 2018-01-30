# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:53:12 2015

@author: yboetz
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension(name="cython.octree",
                        sources=["cython/octree.pyx"],
                        include_dirs=[np.get_include()],
                        extra_compile_args=["-std=c++11", "-O3", "-mfma", "-mavx2", "-fopenmp"],
                        extra_link_args=["-O3", "-lgomp", "-lpthread"],
                        language="c++")]

setup(ext_modules = cythonize(extensions))
