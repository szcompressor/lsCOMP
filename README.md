# lsCOMP

<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>

lsCOMP (<u>l</u>ight <u>s</u>ource <u>COMP</u>ression) is a user-friendly and fast GPU lossy/lossless compressor for light source data and unsigned integers (both ```uint32``` and ```uint16```).
Both compression and decompression in lsCOMP is fully executed in a single NVIDIA GPU kernel without CPU intervention, guaranteeing high end-to-end performance.
Supporting both configurable lossy and lossless compression modes, lsCOMP can be used in diverse scenarios that require different level of data fidelity.
- Developer: Yafan Huang, Sheng Di, Robert Underwood
- Contributors: Peco Myint, Miaoqi Chu, Guanpeng Li, Nicholas Schwarz, and Franck Cappello

## Environmental Requirements
- Linux OS with NVIDIA GPUs
- Git >= 2.15
- CMake >= 3.21
- CUDA Toolkit >= 11.0
- GCC >= 7.3.0

## Compile lsCOMP
You can compile lsCOMP by following commands:
```shell
$ git clone https://github.com/szcompressor/lsCOMP.git
$ cd lsCOMP
$ mkdir build && cd build
$ cmake ..
$ make -j
```
After compilation, you will see 2 executable binaries, ```lsCOMP_uint32``` and ```lsCOMP_uint16```, in ```lsCOMP/build/```.
They are used for performing either configurable lossy or lossless compression for ```uint32``` and ```uint16``` data.

## Execute lsCOMP
We use ```uint32``` compression with ```lsCOMP_uint32``` as an example; the usage of ```lsCOMP_uint16``` is similar.