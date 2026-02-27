#!/bin/bash
# macro for compiling and runing project, must be setup first
rm -r _bin/Debug/tfg
cmake --build build --parallel && ./_bin/Debug/tfg
