#!/bin/bash

make
rm interface.c interface.cpython-36m-x86_64-linux-gnu.so
python3 interface_setup.py build_ext --inplace
