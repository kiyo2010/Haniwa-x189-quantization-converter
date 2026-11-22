# SPDX-License-Identifier: Apache-2.0
"""Convert the official Qwen3-1.7B BF16 checkpoint to a stock X189-style GGUF."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

from . import calibration
from .roles import EMBEDDING, roles


def check(ok, why):
    if not ok:
        raise ValueError(why)


def records_for(work, seal):
    from .gguf_writer import Record

    root = work / "records"
    for name, kind in roles().items():
        record = root / (name + ".ggml")
        meta = calibration.read(root / (name + ".json"))
        check(meta["seal"] == seal and meta["name"] == name and meta["recipe_type"] == kind and
              record.is_file() and meta["bytes"] == record.stat().st_size and
              meta["sha256"] == calibration.digest(record), f"Record or recipe changed: {name}")
        yield name, Record(record, meta["ggml_type"], tuple(meta["shape"]), meta["sha256"])


def identity(args):
    cap = args.work / "calibration"
    bound = calibration.read(cap / "identity.json")
    files = {str(args.checkpoint/name): value for name,value in calibration.SHARDS.items()}
    files[str(args.kftt)] = calibration.ARCHIVE
    check(bound["sha256"] == files and
          all(calibration.stamp(Path(p)) == bound["stamps"][p] for p in files),
          "The authenticated BF16 checkpoint or KFTT input changed")
    source = Path(__file__).resolve().parent
    modules = ("__init__.py","__main__.py","roles.py","e8.py","native.py","gguf_writer.py","lattice.py")
    return dict(calibration_sha256=calibration.digest(cap/"capture.json"),
                sources={p:calibration.digest(source/p) for p in modules},
                ggml_sha256=calibration.digest(args.ggml_lib), device=args.device,
                torch=str(__import__("torch").__version__))


def float_rows(tensor, first, last):
    return np.ascontiguousarray(tensor[first:last].float().numpy(), dtype="<f4")


def stream_iq(source, kind, diagonal, file, native):
    shape = tuple(source.get_shape())
    digest = hashlib.sha256()
    for first in range(0, shape[0], 128):
        weights = float_rows(source, first, min(first+128,shape[0]))
        body = native.encode(kind, weights, diagonal)
        decoded = native.decode(kind, body, shape[1])
        digest.update(memoryview(decoded).cast("B"))
        file.write(memoryview(body).cast("B"))
    return {"decoded_fp32_sha256":digest.hexdigest(),"independent_iq_decode":True}


def stream_xe82(source, tensor, gram, frequency, file, encoder):
    from .e8 import RANK54_NAMES, apply_rank54, apply_vocabulary, rank54_factors, vocabulary_rank

    shape = tuple(source.get_shape())
    with tempfile.TemporaryFile(dir=Path(file.name).parent) as raw:
        def read(first, stop):
            return float_rows(source, first, stop)

        encoded = encoder.encode(tensor, shape, read, raw, gram)
        raw.seek(0)
        if tensor == EMBEDDING or tensor in RANK54_NAMES:
            file.truncate(4 * int(np.prod(shape)))
            decoded = np.memmap(file.name, mode="r+", dtype="<f4", shape=shape)
            base = encoder.decode(raw, decoded.__setitem__, shape)
            if tensor == EMBEDDING:
                ids = vocabulary_rank(read, decoded, frequency)
                apply_vocabulary(ids, read, decoded)
                correction = dict(kind="x72-int8",rows=len(ids),
                                  rows_sha256=hashlib.sha256(ids.astype("<u4").tobytes()).hexdigest())
            else:
                rank = rank54_factors(encoder,tensor,read(0,shape[0]),decoded,gram)
                apply_rank54(encoder,decoded,rank)
                correction = dict(kind="x148-rank54",bytes=len(rank),sha256=hashlib.sha256(rank).hexdigest())
            for i in range(0, shape[0], 128):
                check(np.isfinite(decoded[i:i+128]).all(), f"Nonfinite corrected tensor {tensor}")
            decoded.flush()
            del decoded
        else:
            correction = None

            def sink(_, value):
                array = np.ascontiguousarray(value, dtype="<f4")
                check(np.isfinite(array).all(), f"Nonfinite decoded tensor {tensor}")
                file.write(memoryview(array).cast("B"))

            base = encoder.decode(raw, sink, shape)
        check(base["sha256"] == encoded["sha256"], f"XE82 roundtrip failed: {tensor}")
    return dict(xe82_sha256=encoded["sha256"],xe82_bytes=encoded["bytes"],correction=correction)


def encode(args, seal):
    import torch
    from safetensors import safe_open
    from .e8 import Encoder
    from .native import NativeIQ

    torch.set_num_threads(args.threads)
    cal = calibration.Calibration(args.work)
    frequencies = cal.frequencies()
    assignment = roles()
    encoder = Encoder(args.device)
    native = NativeIQ(args.ggml_lib)
    root = args.work / "records"
    root.mkdir(mode=0o700, exist_ok=True)
    index = calibration.read(args.checkpoint/"model.safetensors.index.json")["weight_map"]
    check(set(index)==set(assignment) or set(index)==set(assignment)|{"lm_head.weight"},
          "Unexpected checkpoint tensor roster")
    shards = set(index.values())
    check(shards=={"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"},
          "Unexpected checkpoint shards")
    with ExitStack() as opened:
        files = {name:opened.enter_context(safe_open(args.checkpoint/name,framework="pt",device="cpu"))
                 for name in shards}
        for i,(name,kind) in enumerate(assignment.items(),1):
            out = root/(name+".ggml")
            proof = root/(name+".json")
            if proof.exists():
                meta = calibration.read(proof)
                check(meta["seal"] == seal and meta["name"] == name and meta["recipe_type"] == kind and
                      meta["sha256"] == calibration.digest(out), f"Saved record changed: {name}")
                continue
            tensor = files[index[name]].get_slice(name)
            shape = tuple(tensor.get_shape())
            check(tensor.get_dtype()=="BF16", f"Source is not BF16: {name}")
            with tempfile.NamedTemporaryFile(mode="w+b",dir=root,prefix=".weight-",delete=False) as f:
                stage = Path(f.name)
                try:
                    if kind == "BF16":
                        check(len(shape)==1, f"Unexpected norm dimensionality: {name}")
                        weight = files[index[name]].get_tensor(name).float().numpy().astype("<f4", copy=False)
                        check(np.isfinite(weight).all(), f"Nonfinite normalization vector: {name}")
                        f.write(memoryview(weight).cast("B"))
                        detail = {}
                    elif kind == "XE82":
                        detail = stream_xe82(tensor, name, cal.gram(name), frequencies, f, encoder)
                    else:
                        detail = stream_iq(tensor, kind, cal.diagonal(name), f, native)
                    f.flush()
                    os.fsync(f.fileno())
                    count = stage.stat().st_size
                    on_disk = "F32" if kind in ("BF16","XE82") else kind
                    if on_disk=="F32":
                        check(count==4*int(np.prod(shape)), f"F32 size mismatch: {name}")
                    digest = calibration.digest(stage)
                    os.replace(stage,out)
                    calibration.atomic_json(proof,dict(name=name,recipe_type=kind,ggml_type=on_disk,
                        shape=shape,sha256=digest,bytes=count,seal=seal,encoding=detail))
                finally:
                    stage.unlink(missing_ok=True)
            print(f"[{i:03}/310] {kind:7} {name}",flush=True)


def run(args):
    check(1 <= args.threads <= 8, "Choose 1–8 threads")
    args.work.mkdir(mode=0o700,parents=True,exist_ok=True)
    check(not args.work.is_relative_to(args.checkpoint) and not args.output.is_relative_to(args.checkpoint),
          "Keep the source checkpoint separate from the work and output paths")
    with (args.work/"conversion.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase in ("capture","all"):
            capture = calibration.capture(args.checkpoint,args.kftt,args.work,args.device,args.threads)
            print(f"Capture verified: {capture}",flush=True)
            if args.phase=="capture":
                return
            import torch
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        sys.path.insert(0,str(args.llama_source/"gguf-py"))
        from .gguf_writer import checkout, write

        checkout(args.llama_source)
        bound = identity(args)
        if args.phase in ("all","encode"):
            encode(args,bound)
            if args.phase=="encode":
                return
        records = dict(records_for(args.work,bound))
        proof = write(args.checkpoint,args.llama_source,args.work,records,args.output)
        proof.update(recipe=calibration.RECIPE,inputs=bound,
                     peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        calibration.atomic_json(args.work/"gguf-proof.json",proof)
        print(json.dumps({"file":str(args.output),"bytes":args.output.stat().st_size,"records":len(records)},ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","kftt","ggml-lib","llama-source","work","output"):
        parser.add_argument("--"+key,type=lambda value:Path(value).expanduser().resolve(),required=True)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--phase",choices=("all","capture","encode","write"),default="all")
    start = time.monotonic()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(1,f"x189_clean: {exc}\n")
    print(f"Elapsed {time.monotonic()-start:.2f} s",flush=True)


if __name__=="__main__":
    main()
