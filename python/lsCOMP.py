# python/lsCOMP.py
import ctypes
from ctypes import c_uint32, c_uint16, c_size_t, c_float, c_void_p, Structure
from pathlib import Path
from typing import Tuple, Optional


class UInt3(Structure):
    _fields_ = [
        ("x", c_uint32),
        ("y", c_uint32),
        ("z", c_uint32),
    ]


class UInt4(Structure):
    _fields_ = [
        ("x", c_uint32),
        ("y", c_uint32),
        ("z", c_uint32),
        ("w", c_uint32),
    ]


class lsCOMP:
    """
    Thin ctypes wrapper for liblsCOMP.so.

    All data pointers are CUDA *device* pointers (uint16/uint32 or uint8 for compressed bytes).
    dims: (x, y, z), where z is the fastest dimension.
    quant_bins: (x, y, z, w) for the 4 levels, with x <= y <= z <= w.
    """

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            base_dir = Path(__file__).resolve().parent
            default_path = base_dir.parent / "build" / "liblsCOMP.so"
            lib_path = str(default_path)

        self._lib = ctypes.CDLL(lib_path)

        # ---- uint32 version ----
        self._lib.lsCOMP_compression_uint32_bsize64.argtypes = [
            c_void_p,               # d_oriData (uint32_t*)
            c_void_p,               # d_cmpBytes (unsigned char*)
            ctypes.POINTER(c_size_t),  # cmpSize (size_t*)
            UInt3,                  # dims
            UInt4,                  # quantBins
            c_float,                # poolingSH
            c_void_p,               # cudaStream_t (0 for default stream)
        ]
        self._lib.lsCOMP_compression_uint32_bsize64.restype = None

        self._lib.lsCOMP_decompression_uint32_bsize64.argtypes = [
            c_void_p,               # d_decData (uint32_t*)
            c_void_p,               # d_cmpBytes (unsigned char*)
            c_size_t,               # cmpSize (size_t)
            UInt3,                  # dims
            UInt4,                  # quantBins
            c_float,                # poolingSH
            c_void_p,               # cudaStream_t
        ]
        self._lib.lsCOMP_decompression_uint32_bsize64.restype = None

        # ---- uint16 version ----
        self._lib.lsCOMP_compression_uint16_bsize64.argtypes = [
            c_void_p,               # d_oriData (uint16_t*)
            c_void_p,               # d_cmpBytes (unsigned char*)
            ctypes.POINTER(c_size_t),
            UInt3,
            UInt4,
            c_float,
            c_void_p,
        ]
        self._lib.lsCOMP_compression_uint16_bsize64.restype = None

        self._lib.lsCOMP_decompression_uint16_bsize64.argtypes = [
            c_void_p,               # d_decData (uint16_t*)
            c_void_p,               # d_cmpBytes (unsigned char*)
            c_size_t,
            UInt3,
            UInt4,
            c_float,
            c_void_p,
        ]
        self._lib.lsCOMP_decompression_uint16_bsize64.restype = None

    # ---------- low-level wrappers: uint32 ----------

    def compress_uint32(
        self,
        d_ori_ptr: int,
        d_cmp_ptr: int,
        dims: Tuple[int, int, int],
        quant_bins: Tuple[int, int, int, int],
        pooling_sh: float,
        stream_ptr: int = 0,
    ) -> int:
        """
        Wraps:
        void lsCOMP_compression_uint32_bsize64(
            uint32_t* d_oriData, unsigned char* d_cmpBytes,
            size_t* cmpSize, uint3 dims, uint4 quantBins,
            float poolingSH, cudaStream_t stream=0);

        Returns: cmpSize (number of bytes in compressed buffer).
        """
        dims_c = UInt3(*map(int, dims))
        qbins_c = UInt4(*map(int, quant_bins))
        cmp_size_c = c_size_t(0)

        self._lib.lsCOMP_compression_uint32_bsize64(
            c_void_p(d_ori_ptr),
            c_void_p(d_cmp_ptr),
            ctypes.byref(cmp_size_c),
            dims_c,
            qbins_c,
            c_float(pooling_sh),
            c_void_p(stream_ptr),
        )
        return int(cmp_size_c.value)

    def decompress_uint32(
        self,
        d_dec_ptr: int,
        d_cmp_ptr: int,
        cmp_size: int,
        dims: Tuple[int, int, int],
        quant_bins: Tuple[int, int, int, int],
        pooling_sh: float,
        stream_ptr: int = 0,
    ) -> None:
        """
        Wraps:
        void lsCOMP_decompression_uint32_bsize64(
            uint32_t* d_decData, unsigned char* d_cmpBytes,
            size_t cmpSize, uint3 dims, uint4 quantBins,
            float poolingSH, cudaStream_t stream=0);
        """
        dims_c = UInt3(*map(int, dims))
        qbins_c = UInt4(*map(int, quant_bins))

        self._lib.lsCOMP_decompression_uint32_bsize64(
            c_void_p(d_dec_ptr),
            c_void_p(d_cmp_ptr),
            c_size_t(int(cmp_size)),
            dims_c,
            qbins_c,
            c_float(pooling_sh),
            c_void_p(stream_ptr),
        )

    # ---------- low-level wrappers: uint16 ----------

    def compress_uint16(
        self,
        d_ori_ptr: int,
        d_cmp_ptr: int,
        dims: Tuple[int, int, int],
        quant_bins: Tuple[int, int, int, int],
        pooling_sh: float,
        stream_ptr: int = 0,
    ) -> int:
        """
        Wraps:
        void lsCOMP_compression_uint16_bsize64(
            uint16_t* d_oriData, unsigned char* d_cmpBytes,
            size_t* cmpSize, uint3 dims, uint4 quantBins,
            float poolingSH, cudaStream_t stream=0);
        """
        dims_c = UInt3(*map(int, dims))
        qbins_c = UInt4(*map(int, quant_bins))
        cmp_size_c = c_size_t(0)

        self._lib.lsCOMP_compression_uint16_bsize64(
            c_void_p(d_ori_ptr),
            c_void_p(d_cmp_ptr),
            ctypes.byref(cmp_size_c),
            dims_c,
            qbins_c,
            c_float(pooling_sh),
            c_void_p(stream_ptr),
        )
        return int(cmp_size_c.value)

    def decompress_uint16(
        self,
        d_dec_ptr: int,
        d_cmp_ptr: int,
        cmp_size: int,
        dims: Tuple[int, int, int],
        quant_bins: Tuple[int, int, int, int],
        pooling_sh: float,
        stream_ptr: int = 0,
    ) -> None:
        """
        Wraps:
        void lsCOMP_decompression_uint16_bsize64(
            uint16_t* d_decData, unsigned char* d_cmpBytes,
            size_t cmpSize, uint3 dims, uint4 quantBins,
            float poolingSH, cudaStream_t stream=0);
        """
        dims_c = UInt3(*map(int, dims))
        qbins_c = UInt4(*map(int, quant_bins))

        self._lib.lsCOMP_decompression_uint16_bsize64(
            c_void_p(d_dec_ptr),
            c_void_p(d_cmp_ptr),
            c_size_t(int(cmp_size)),
            dims_c,
            qbins_c,
            c_float(pooling_sh),
            c_void_p(stream_ptr),
        )
