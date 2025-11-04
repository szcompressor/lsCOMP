# lsCOMP

<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>

This branch contains instructions to setup Podman or Docker container for running lsCOMP compressor.
The original image is created using Docker environments and is compatible with Podman as well.
The prepared image is uploaded in Dockerhub with [link](https://hub.docker.com/repository/docker/hyfshishen/lscomp/general).
Inside this image, lsCOMP compressor and NVIDIA nvCOMP compressors (along with other related software) are configured.

## Prerequisite
- A Linux machine
- An NVIDIA GPU
- Docker or Podman installed

## Launching Container

If you are using Podman 4.0+ and NVIDIA driviers are setup correctly:

```shell
# Pull the image
podman pull docker.io/hyfshishen/lscomp:cuda12-container

# Launch the container with GPU access (NVIDIA runtime must be configured)
podman run --rm -it --hooks-dir=/usr/share/containers/oci/hooks.d \
    --device nvidia.com/gpu=all \
    hyfshishen/lscomp:cuda12-container bash
```

If you are using Docker, please make sure Docker and NVIDIA Container Toolkit are installed and working.

```shell
# Pull the image
docker pull hyfshishen/lscomp:cuda12-container

# Launch an interactive container with GPU access
docker run --rm -it --gpus all hyfshishen/lscomp:cuda12-container bash
```

The executable binaries for lsCOMP and nvCOMP compressors are setup in local path already inside this container.
So in later steps, we assume we are already in ```~/``` path inside this container.

## Executing lsCOMP and nvCOMP Compressors

### Setting Up Datasets

To make sure light source datasets can be executed with lsCOMP and nvCOMP command line interfaces, we need to extract it from the original HDF5 files and save it as binary format.

Taking ```D0131_US-Cup2_a0010_f005000_r00001.h5``` as an example, this is the time-series XPCS light source dataset with dimension (5000, 2162, 2068).
We can use following command to extract it.
```python
import h5py
import numpy as np

data_path = "D0131_US-Cup2_a0010_f005000_r00001.h5"
hf = h5py.File(data_path, 'r')
raw = np.array(hf['entry/data/data'][:])

# Settings
output_prefix = "frame"
slices_per_file = 500
dtype = np.uint32

# Write 500-slice chunks to separate binary files
num_files = raw.shape[0] // slices_per_file  # 10 files

for i in range(num_files):
    chunk = raw[i * slices_per_file : (i + 1) * slices_per_file]
    chunk.tofile(f"{output_prefix}_{i:02d}.bin")

print(f"Saved {num_files} binary files with {slices_per_file} slices each.")
```
Then you split it into a set of binary files with 500 slices in each. Such binary dataset can be used to be compressed using lsCOMP and nvCOMP command line interfaces.

### Using lsCOMP

There are two executable binaries for lsCOMP, ```lsCOMP_uint16``` for uint16 type and ```lsCOMP_uint32``` for uint32 data type.
Taking CSSI dataset with uint32 data type as an example, its usage can be shown as below:

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

Assuming the light source dataset has 3 dimension 600 1813 1558, then it means there are 600 2D images and each image has dimension 1813 x 1558.
There are two lossy modes, adaptive scalar quantization bin ```-b``` and selective pooling ```-p```, they are lossy modes. If you want to use lossless compression, you can set all of them as 1. For example:

```shell
lsCOMP_uint32 -i your-data.bin -d 600 1813 1558 -b 1 1 1 1 -p 1
```

Then it will be compressed using lossless format.
To save the compressed data, you can use ```-x``` flag; to save the reconstructed data (if you are using lossless, the reconstructed data is identical), you can use ```-o``` flag. Note that ```-x``` and ```-o``` flags are optional in execution.

A sample output after compression can be shown as below:

```shell
$ ./lsCOMP_uint32 -i cssi_600_1813_1558.uint32 -d 600 1813 1558 -b 1 1 1 1 -p 1 -x cmp.bin -o dec.bin

Section 0: lsCOMP Input Preparation
  → Read data from disk...
  ✓ Done.
  → Transfer data to GPU...
  ✓ Done.

Section 1: GPU Warmup
  → Performing GPU warmup runs for 3 iterations...
  ✓ Done.

Section 2: lsCOMP Compression and Decompression
  → lsCOMP GPU compression...
  ✓ Done.
  → Verify compressed data correctness via GPU-CPU-GPU transfer (optional step)...
  ✓ Done.
  → lsCOMP GPU decompression...
  ✓ Done.

Section 3: Output Data Writing (optional step)
  → Write compressed data from GPU to CPU...
  ✓ Done.
  → Write compressed data to from CPU to disk...
  ✓ Done.
  → Write decompressed data from GPU to CPU...
  ✓ Done.
  → Write decompressed data from CPU to disk...
  ✓ Done.

====================================
========== lsCOMP Summary ==========
====================================
Dataset information:
  - dims:       600 x 1813 x 1558
  - length:     1694792400

Input arguments:
  - quantBins:  1 1 1 1
  - poolingTH:  1.000000

Breakdown of time costs:
  - Read data from disk time:   2.681619 s
  - CPU data transfer to GPU:   0.530205 s
  - GPU compression time:       0.012865 s
  - GPU-CPU data tranfer time:  0.186769 s      (optional step, flushing cmpData to 0 for verification)
  - GPU decompression time:     0.019442 s
  - GPU data transfer to CPU:   2.447481 s      (optional step, only used when -x/-o flag is used)
  - Write data to disk time:    5.952488 s      (optional step, only used when -x/-o flag is used)

lsCOMP performance results:
lsCOMP compression   end-to-end speed: 490.759780 GB/s
lsCOMP decompression end-to-end speed: 324.740466 GB/s
lsCOMP compression ratio: 16.171549
  - oriSize: 6779169600 bytes
  - cmpSize: 419203488 bytes
```
Detailed breakdown performance and flags are printed.
Above results are tested on an NVIDIA A100 GPU.

```lsCOMP_uint16``` can be executed bin the same way.

### Using nvCOMP Compressors

nvCOMP compressors are also configured in local path and can be executed directly. The commands include:
```shell
benchmark_allgather         benchmark_deflate_chunked   benchmark_lz4_synth
benchmark_ans_chunked       benchmark_gdeflate_chunked  benchmark_snappy_chunked
benchmark_bitcomp_chunked   benchmark_hlif              benchmark_snappy_synth
benchmark_cascaded_chunked  benchmark_lz4_chunked       benchmark_zstd_chunked
```

Assuming we have a dataset ```data.bin```, it can be compressed using commands:

```shell
# compress with cascaded, the fastest one
benchmark_cascaded_chunked -f data.bin

# compress with lz4, high speed and relatively high ratio
benchmark_lz4_chunked -f data.bin

# compress with zstd, the highest ratio one
benchmark_zstd_chunked -f data.bin
```

A sample output can be shown as below:
```shell
benchmark_zstd_chunked -f pawpawsaurus_958x646x1088_uint16.raw 
----------
files: 1
uncompressed (B): 1346656768
comp_size: 1121236226, compressed ratio: 1.2010
compression throughput (GB/s): 1.5662
decompression throughput (GB/s): 27.0891
```
Above results are tested on my local PC with a RTX 3080 GPU.

## Contact
Yafan Huang, yafan-huang@uiowa.edu
