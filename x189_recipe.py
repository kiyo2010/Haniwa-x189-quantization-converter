"""Fixed X189 encoder components with historical verification fingerprints."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
from calibrate import FILES, XXS_FP32_SHA, XXS_OWNER, require, sha

TARGET = XXS_OWNER + ".weight"
GGML = "2c6a9014d030948a0c29ed633d8115410fd1350b374e42a8b06c9f9246a096d4"
L17_BODY = "eb5442d157f10b883ffbe8ea66179f48cc85842e039198502f914af7d0728109"
L17_WRAPPED = "3984e89b5503c5b8dc958aaa2cb67d73f9621120e3a862f80f31cc35fe25a260"


def frequency_rows(gain, counts, k=2654):
    """Admit positive observed gains first, then fill by raw gain."""
    require(gain.ndim == 1 and counts.shape == gain.shape and 0 < k <= len(gain) and
            np.isfinite(gain).all() and (counts >= 0).all(), "Invalid vocabulary ranking inputs")
    ids = np.arange(len(gain), dtype=np.int64)
    weighted = counts * gain
    energy = np.lexsort((ids, -gain))
    observed = ids[(counts > 0) & (gain > 0)]
    observed = observed[np.lexsort((observed, -weighted[observed]))]
    seen = np.zeros(len(gain), dtype=np.bool_)
    seen[observed] = True
    return np.concatenate((observed, energy[~seen[energy]]))[:k]


def represented_gain(original, decoded):
    """Use the same FP16 stored scales and FP32 application as the old vocabulary scan."""
    residual = original - decoded
    peaks = np.max(np.abs(residual), axis=1)
    scales = (peaks/127).astype("<f2")
    scales[(peaks > 0) & (scales == 0)] = np.nextafter(np.float16(0), np.float16(1))
    scales[peaks == 0] = 1
    require((scales > 0).all() and np.isfinite(scales).all(), "Unrepresentable row scale")
    scale32 = scales.astype(np.float32)[:, None]
    codes = np.rint(residual / scale32).clip(-127, 127).astype(np.int8)
    before = np.square(residual, dtype=np.float64).sum(axis=1)
    after = np.square(original - (decoded + codes.astype(np.float32)*scale32), dtype=np.float64).sum(axis=1)
    return before - after, scales, codes


def native_iq2(weights, diagonal, library):
    source = ctypes.CDLL(str(library))
    encoder = source.ggml_quantize_chunk
    encoder.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    encoder.restype = ctypes.c_size_t
    rows, cols = weights.shape
    require(cols % 256 == 0 and diagonal.shape == (cols,) and
            weights.dtype == diagonal.dtype == np.float32 and weights.flags.c_contiguous
            and diagonal.flags.c_contiguous, "Incorrect native IQ2 input layout")
    packed = np.empty((rows, cols//256*66), dtype=np.uint8)
    # GGML_TYPE_IQ2_XXS in the pinned ABI.
    count = encoder(16, weights.ctypes.data, packed.ctypes.data, 0, rows, cols, diagonal.ctypes.data)
    require(count == packed.size, "Unexpected native IQ2 size")
    return packed.tobytes()


def encode(args):
    from safetensors import safe_open

    require(not args.output.exists(), "Refusing to replace an existing output")
    require(args.allow_rebuilt_ggml or sha(args.ggml) == GGML, "Pinned ggml-base library differs")
    index = args.checkpoint/"model.safetensors.index.json"
    require(sha(index) == FILES[index.name], "Checkpoint index differs")
    shard = args.checkpoint/(json.loads(index.read_bytes())["weight_map"][TARGET])
    require(shard.parent == args.checkpoint and sha(shard) == FILES[shard.name], "Original BF16 shard differs")
    with safe_open(shard, framework="pt", device="cpu") as model:
        original = model.get_tensor(TARGET)
        require(str(original.dtype) == "torch.bfloat16" and tuple(original.shape) == (2048, 6144),
                "Unexpected L17 BF16 shape")
        weights = original.float().numpy()
    with np.load(args.moments, allow_pickle=False) as data:
        diagonal = np.ascontiguousarray(data["all:"+XXS_OWNER], np.float32)
    require(hashlib.sha256(diagonal.tobytes()).hexdigest() == XXS_FP32_SHA, "L17 calibration moments differ")
    body = native_iq2(weights, diagonal, args.ggml)
    digest = hashlib.sha256(body)
    require(digest.hexdigest() == L17_BODY, "Canonical L17 IQ2 payload differs")
    wire = struct.pack("<8sB3sIIQ32s", b"X180IQ20", 0, bytes(3), *weights.shape,
                       len(body), digest.digest()) + body
    require(hashlib.sha256(wire).hexdigest() == L17_WRAPPED, "Canonical L17 wrapper differs")
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".encode-") as directory:
        staged = Path(directory)/"L17.bin"
        with staged.open("wb") as file:
            file.write(wire)
            file.flush()
            os.fsync(file.fileno())
        os.link(staged, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "moments", "ggml", "output"):
        parser.add_argument("--"+flag, required=True, type=Path)
    parser.add_argument("--allow-rebuilt-ggml", action="store_true",
                        help="Accept a locally built library only if canonical payload and wrapper hashes match")
    encode(parser.parse_args())


if __name__ == "__main__":
    main()
