#!/bin/bash
# macro for compiling and runing project, must be setup first
cmake --build build --parallel && ./_bin/tfg
