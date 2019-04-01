#! /usr/bin/python3
# -*- coding:utf-8 -*-

# Run this file with :
# python3 interface_setup.py build_ext --inplace


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

flappy_extension = Extension(
    name="interface",
    sources=["interface.pyx"],
    libraries=["_flappy", "SDL", "SDL_image", "SDL_mixer", "SDL_ttf"],
    library_dirs=["lib_flappy"],
    include_dirs=["lib_flappy"]
)

setup(
    name="interface",
    ext_modules=cythonize(flappy_extension)
)
