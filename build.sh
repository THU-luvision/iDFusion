#!/bin/bash


cd CHISEL

mkdir build
cd build
cmake ../src
make -j
cd ../../build
cmake ..
make -j

