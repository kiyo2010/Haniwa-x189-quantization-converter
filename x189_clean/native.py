# SPDX-License-Identifier: Apache-2.0
"""Native IQ quantization and independent checks for the pinned ggml ABI."""
import ctypes as c
from pathlib import Path

import numpy as np

KINDS = {"IQ2_XXS": (16, 256, 66), "IQ3_XXS": (18, 256, 98),
         "IQ1_S": (19, 256, 50), "IQ4_NL": (20, 32, 18)}
ToFloat = c.CFUNCTYPE(None, c.c_void_p, c.c_void_p, c.c_int64)


class Traits(c.Structure):
    _fields_ = [("name", c.c_char_p), ("block", c.c_int64), ("interleave", c.c_int64),
                ("bytes", c.c_size_t), ("quantized", c.c_bool), ("to_float", ToFloat)]


class NativeIQ:
    def __init__(self, library: Path):
        self.library = c.CDLL(str(library))
        self.quantize = self.library.ggml_quantize_chunk
        self.quantize.argtypes = [c.c_int, c.c_void_p, c.c_void_p, c.c_int64,
                                  c.c_int64, c.c_int64, c.c_void_p]
        self.quantize.restype = c.c_size_t
        self.validate = self.library.ggml_validate_row_data
        self.validate.argtypes = [c.c_int, c.c_void_p, c.c_size_t]
        self.validate.restype = c.c_bool
        pointer = self.library.ggml_get_type_traits
        pointer.argtypes = [c.c_int]
        pointer.restype = c.POINTER(Traits)
        self.traits = {}
        for kind, (number, block, size) in KINDS.items():
            t = pointer(number).contents
            if t.block != block or t.bytes != size or not t.quantized or not t.to_float:
                raise ValueError(f"Incompatible ggml type for {kind}")
            self.traits[kind] = t

    def encode(self, kind, weights, diag):
        number, block, size = KINDS[kind]
        source = np.ascontiguousarray(weights, dtype="<f4")
        rows, cols = source.shape
        if (cols % block or diag.shape != (cols,) or not np.isfinite(diag).all() or
                (diag < 0).any() or (diag.max() <= 0) or not np.isfinite(source).all()):
            raise ValueError(f"Invalid input for {kind}")
        importance = np.ascontiguousarray(diag, dtype="<f4")
        data = np.empty((rows, cols // block * size), dtype=np.uint8)
        count = self.quantize(number, source.ctypes.data, data.ctypes.data, 0, rows, cols,
                              importance.ctypes.data)
        if count != data.nbytes or not self.validate(number, data.ctypes.data, count):
            raise ValueError(f"Invalid {kind} encoded extent")
        return data

    def decode(self, kind, data, cols):
        import gguf

        _, block, size = KINDS[kind]
        if data.ndim != 2 or cols % block or data.shape[1] != cols // block * size:
            raise ValueError(f"Invalid {kind} decode extent")
        native = np.empty((len(data), cols), dtype="<f4")
        self.traits[kind].to_float(data.ctypes.data, native.ctypes.data, native.size)
        independent = gguf.dequantize(data, getattr(gguf.GGMLQuantizationType, kind))
        if (native.shape != independent.shape or not np.isfinite(native).all() or
                not np.allclose(native, independent, rtol=4e-6, atol=2e-7)):
            raise ValueError(f"Independent {kind} decode mismatch")
        return native
