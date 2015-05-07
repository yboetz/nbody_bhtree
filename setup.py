# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:53:12 2015

@author: somebody
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [Extension("Quadtree", ["Quadtree.pyx"],
                        extra_compile_args=["-std=c++11", "-O3"],
                        language="c++")]

setup(ext_modules = cythonize(extensions))