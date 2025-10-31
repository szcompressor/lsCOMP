# lsCOMP

<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>

lsCOMP (<u>l</u>ight <u>s</u>ource <u>COMP</u>ression) is a user-friendly and fast GPU lossy/lossless compressor for light source data and unsigned integers (both ```uint32``` and ```uint16```).
Both compression and decompression in lsCOMP is fully executed in a single NVIDIA GPU kernel without CPU intervention, guaranteeing high end-to-end performance.
Supporting both configurable lossy and lossless compression modes, lsCOMP can be used in diverse scenarios that require different level of data fidelity.
lsCOMP is not only suitable for light source data but also for generic integer compression that requres high speed (e.g., visualization datasets from [Open Scivis Datasets](https://github.com/sci-visus/open-scivis-datasets)).

This work is published in **[SC'25] lsCOMP: Efficient Light Source Compression**.
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
We use ```lsCOMP_uint32``` here to explain; the usage of ```lsCOMP_uint16``` is similar.
```shell
$ ./lsCOMP_uint32 --help
lsCOMP Usage:
   ./lsCOMP_uint32 -i oriFilePath -d dims.x dims.y dims.z -b quantBins.x quantBins.y quantBins.z quantBins.w -p value -x cmpFilePath -o decFilePath
Options:
   -i oriFilePath: Path to the original data file
   -d dims.x dims.y dims.z: Dimensions of the original data, where dim.z is the fastest dimension.
   -b quantBins.x quantBins.y quantBins.z quantBins.w: Quantization bins for the 4 levels, where x is the base one and x<=y<=z<=w.
   -p value: Pooling threshold for a data block.
   -x cmpFilePath: Path to the compressed data file   (optional).
   -o decFilePath: Path to the decompressed data file (optional).
Examples:
   ./lsCOMP_uint32 -i data/cssi.bin -d 600 1813 1558 -b 3 5 10 15 -p 0.5
   ./lsCOMP_uint32 -i data/cssi.bin -d 600 1813 1558 -b 3 5 10 15 -p 0.5 -x data/cssi-cmp.bin
   ./lsCOMP_uint32 -i data/cssi.bin -d 600 1813 1558 -b 3 5 10 15 -p 0.5 -o data/cssi-dec.bin
   ./lsCOMP_uint32 -i data/cssi.bin -d 600 1813 1558 -b 3 5 10 15 -p 0.5 -x data/cssi-cmp.bin -o data/cssi-dec.bin
```
Note that lsCOMP supports configurable lossy modes, consisting of two steps: **Adaptive Scalar Quantization** and **Selective Pooling**.
You may choose to enable either step individually, enable both, or disable both (which makes lsCOMP operate in a lossless mode).
To disable Adaptive Scalar Quantization, set ```-b 1 1 1 1```; to disable Selective Pooling, set ```-p 1```.
More details about these algorithms can be found in the paper.

A sample output (lossless mode) can be found in the below:
```shell
$ ./lsCOMP_uint32 -i xpcs_500_2162_2068.uint32 -d 500 2162 2068 -b 1 1 1 1 -p 1
GPU warmup finished!

Dataset information:
  - dims:       500 x 2162 x 2068
  - length:     2235508000

Input arguments:
  - quantBins:  1 1 1 1
  - poolingTH:  1.000000

Breakdown of time costs:
  - Read data from disk time:   3.413821 s
  - CPU data transfer to GPU:   0.706461 s
  - GPU compression time:       0.015537 s
  - GPU-CPU data tranfer time:  0.486617 s      (optional step, flushing cmpData to 0 for verification)
  - GPU decompression time:     0.022800 s
  - GPU data transfer to CPU:   0.000000 s      (optional step, only used when -x/-o flag is used)
  - Write data to disk time:    0.000000 s      (optional step, only used when -x/-o flag is used)

lsCOMP performance results:
lsCOMP compression   end-to-end speed: 536.004354 GB/s
lsCOMP decompression end-to-end speed: 365.261478 GB/s
lsCOMP compression ratio: 8.228122
  - oriSize: 8942032000 bytes
  - cmpSize: 1086764580 bytes
```
This result is measured under an NVIDIA A100 (40 GB) GPU.

## Citation
If you find lsCOMP is useful, the following paper can be considered for citing.
```bibtex
@inproceedings{huang2025lscomp,
    title={lsCOMP: Efficient Light Source Compression},
    author={Huang, Yafan and Di, Sheng and Underwood, Robert and Myint, Peco and Chu, Miaoqi and and Li, Guanpeng and Schwarz, Nicholas and Cappello, Franck},
    booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
    pages={1--18},
    year={2025}
}
```

## Copyright
(C) 2025 by Argonne National Laboratory and University of Iowa. For more details see [COPYRIGHT](https://github.com/szcompressor/lsCOMP/blob/main/LICENSE).

## Acknowledgement
This work is supported by the U.S. Department of Energy (DOE) Office of Science, Advanced Scientific Computing Research (ASCR) and Basic Energy Sciences (BES) under the award "ILLUMINE - Intelligent Learning for Light Source and Neutron Source User Measurements Including Navigation and Experiment Steering." This work also received support from the DOE Office of Science ASCR Leadership Computing Challenge (ALCC) through the 2025–2026 award "Enhancing APS-Enabled Research through Integrated Research Infrastructure." This research used resources of the Advanced Photon Source (APS) and the Argonne Leadership Computing Facility (ALCF), both U.S. DOE Office of Science user facilities operated by Argonne National Laboratory under Contract No. DE-AC02-06CH11357. Additional computing resources were provided by the National Energy Research Scientific Computing Center (NERSC) under ALCC award ERCAP0030693 and the Oak Ridge Leadership Computing Facility (OLCF) at Oak Ridge National Laboratory under Contract No. DE-AC05-00OR22725. We also acknowledge computing resources provided by ALCF Polaris and the Argonne Laboratory Computing Resource Center (LCRC) Swing. This work was further supported by the U.S. DOE Office of Science, ASCR, under Contracts DE-SC0024559, as well as National Science Foundation (NSF) grants OAC-2104023, OAC-2211538, OAC-2311875, OAC-2514036, and OAC-2513768.
