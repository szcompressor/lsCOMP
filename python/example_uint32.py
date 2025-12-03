#!/usr/bin/env python3
"""
Example driver for lsCOMP uint32 compressor using ctypes + pycuda.

Usage (from project root):

  cd python
  python example_uint32.py \
      -i ../data/input_uint32.bin \
      -d 128 128 64 \
      -b 4 8 16 32 \
      -p 0.01 \
      -x cmp_uint32.bin \
      -o dec_uint32.bin

Options:
   -i oriFilePath: Path to the original data file (uint32 raw binary)
   -d dims.x dims.y dims.z: Dimensions of the original data, where dim.z
       is the fastest dimension.
   -b quantBins.x quantBins.y quantBins.z quantBins.w: Quantization bins
       for the 4 levels, where x is the base one and x<=y<=z<=w.
   -p value: Pooling threshold for a data block.
   -x cmpFilePath: Path to the compressed data file   (optional).
   -o decFilePath: Path to the decompressed data file (optional).
"""

import argparse
import time
from pathlib import Path

import numpy as np

import pycuda.autoinit  # noqa: F401  # creates a context
import pycuda.driver as drv

from lsCOMP import lsCOMP


def parse_args():
    parser = argparse.ArgumentParser(description="lsCOMP uint32 Python example")

    parser.add_argument(
        "-i", "--input",
        dest="ori_path",
        required=True,
        help="Path to original uint32 raw data file",
    )
    parser.add_argument(
        "-d", "--dims",
        nargs=3,
        type=int,
        required=True,
        metavar=("DX", "DY", "DZ"),
        help="Dimensions (x y z) of original data (z fastest)",
    )
    parser.add_argument(
        "-b", "--bins",
        nargs=4,
        type=int,
        required=True,
        metavar=("BX", "BY", "BZ", "BW"),
        help="Quantization bins (x y z w), x<=y<=z<=w",
    )
    parser.add_argument(
        "-p", "--pooling",
        type=float,
        required=True,
        help="Pooling threshold for a data block",
    )
    parser.add_argument(
        "-x", "--cmp",
        dest="cmp_path",
        default=None,
        help="Path to compressed file (optional)",
    )
    parser.add_argument(
        "-o", "--output",
        dest="dec_path",
        default=None,
        help="Path to decompressed output file (optional)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dims = tuple(args.dims)
    quant_bins = tuple(args.bins)
    pooling_sh = float(args.pooling)

    nx, ny, nz = dims
    expected_elems = nx * ny * nz

    # ---- load original data (uint32 raw binary) ----
    ori_path = Path(args.ori_path)
    if not ori_path.exists():
        raise FileNotFoundError(f"Input file not found: {ori_path}")

    host_data = np.fromfile(ori_path, dtype=np.uint32)
    if host_data.size != expected_elems:
        raise ValueError(
            f"Input size mismatch: file has {host_data.size} elements, "
            f"but dims product is {expected_elems}."
        )

    # Reshape for convenience; z is fastest in row-major layout as usual.
    host_data = host_data.reshape((nx, ny, nz))

    ori_bytes = host_data.nbytes

    # ---- allocate device memory ----
    d_ori = drv.mem_alloc(ori_bytes)
    drv.memcpy_htod(d_ori, host_data)

    # crude upper bound for compressed buffer: 1.5x original size
    # （可以按经验调大一点，比如 2.0）
    cmp_capacity = int(ori_bytes * 1.5)
    d_cmp = drv.mem_alloc(cmp_capacity)

    # buffer for decompressed data
    d_dec = drv.mem_alloc(ori_bytes)

    # ---- instantiate wrapper ----
    compressor = lsCOMP()

    # ---- compression ----
    t0 = time.perf_counter()
    cmp_size = compressor.compress_uint32(
        d_ori_ptr=int(d_ori),
        d_cmp_ptr=int(d_cmp),
        dims=dims,
        quant_bins=quant_bins,
        pooling_sh=pooling_sh,
        stream_ptr=0,  # default stream
    )
    drv.Context.synchronize()
    t1 = time.perf_counter()

    if cmp_size <= 0:
        raise RuntimeError("Compression returned non-positive cmp_size")

    # copy compressed data back if we need to store / compute ratio
    host_cmp = np.empty(cmp_size, dtype=np.uint8)
    drv.memcpy_dtoh(host_cmp, d_cmp)

    # ---- decompression ----
    t2 = time.perf_counter()
    compressor.decompress_uint32(
        d_dec_ptr=int(d_dec),
        d_cmp_ptr=int(d_cmp),
        cmp_size=cmp_size,
        dims=dims,
        quant_bins=quant_bins,
        pooling_sh=pooling_sh,
        stream_ptr=0,
    )
    drv.Context.synchronize()
    t3 = time.perf_counter()

    # copy decompressed data back to host
    host_dec = np.empty_like(host_data)
    drv.memcpy_dtoh(host_dec, d_dec)

    # ---- optional output files ----
    if args.cmp_path is not None:
        Path(args.cmp_path).write_bytes(host_cmp.tobytes())

    if args.dec_path is not None:
        # flatten when writing, to match input raw layout
        host_dec.ravel().tofile(args.dec_path)

    # ---- metrics: compression ratio & throughput ----
    comp_time = t1 - t0
    decomp_time = t3 - t2

    comp_ratio = ori_bytes / cmp_size
    ori_mib = ori_bytes / (1024**2)
    cmp_mib = cmp_size / (1024**2)

    comp_gbps = (ori_bytes / comp_time) / 1e9 if comp_time > 0 else float("inf")
    decomp_gbps = (ori_bytes / decomp_time) / 1e9 if decomp_time > 0 else float("inf")

    max_abs_diff = np.max(
        np.abs(
            host_dec.astype(np.int64) - host_data.astype(np.int64)
        )
    )

    print("=== lsCOMP uint32 example ===")
    print(f"Input file      : {ori_path}")
    print(f"Dims            : {dims} (z fastest)")
    print(f"Quantization    : {quant_bins}")
    print(f"Pooling SH      : {pooling_sh}")
    print("")
    print(f"Original size   : {ori_mib:.3f} MiB")
    print(f"Compressed size : {cmp_mib:.3f} MiB")
    print(f"Compression ratio (orig/cmp): {comp_ratio:.3f}x")
    print("")
    print(f"Compression time: {comp_time*1e3:.3f} ms")
    print(f"Compression TP  : {comp_gbps:.3f} GB/s")
    print(f"Decompress time : {decomp_time*1e3:.3f} ms")
    print(f"Decompress TP   : {decomp_gbps:.3f} GB/s")
    print("")
    print(f"Max abs diff    : {max_abs_diff}")


if __name__ == "__main__":
    main()
