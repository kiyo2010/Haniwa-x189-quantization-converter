# SPDX-License-Identifier: Apache-2.0
"""Inspect a local BF16 checkpoint and invoke the stock HF-to-GGUF converter."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys


RUN_CONVERTER = """
import runpy
import sys
from pathlib import Path
p = Path(sys.argv.pop(1))
allow = sys.argv.pop(1) == "1"
sys.path.insert(0, str(p.parent))
if not allow:
    from transformers import dynamic_module_utils
    def reject(*args, **kwargs):
        raise ValueError(
            "This checkpoint requires custom model Python. "
            "For a trusted checkpoint, rerun with --allow-model-code."
        )
    dynamic_module_utils.get_cached_module_file = reject
sys.argv[0] = str(p)
runpy.run_path(str(p), run_name="__main__")
"""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected a JSON object in {path.name}")
    return value


def stamp(path):
    s = path.stat()
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def part_dtypes(path, is_safe):
    if is_safe:
        from safetensors import safe_open

        with safe_open(path,framework="np") as f:
            return {k:f.get_slice(k).get_dtype() for k in f.keys()}
    import torch

    weights = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    require(isinstance(weights,dict) and all(isinstance(w,torch.Tensor) for w in weights.values()),
            f"{path.name} must contain a PyTorch tensor state dictionary")
    known = {torch.bfloat16:"BF16",torch.float16:"F16",torch.float32:"F32"}
    return {k:known.get(w.dtype,str(w.dtype)) for k,w in weights.items()}


def inventory(checkpoint):
    cfg = read_object(checkpoint/"config.json")
    require(not cfg.get("quantization_config"), "Provide the original BF16 checkpoint")
    is_safe = (checkpoint/"model.safetensors.index.json").is_file() or any(checkpoint.glob("model*.safetensors"))
    ext = ".safetensors" if is_safe else ".bin"
    stem = "model" if is_safe else "pytorch_model"
    index = checkpoint/(stem+ext+".index.json")
    watched = [checkpoint/"config.json"]
    if index.is_file():
        expected = read_object(index).get("weight_map")
        require(isinstance(expected,dict) and bool(expected), f"Invalid weight map: {index.name}")
        require(all(isinstance(n,str) and Path(n).name==n and n.endswith(ext) for n in expected.values()),
                f"Invalid weight shard name: {index.name}")
        shards = [checkpoint/n for n in sorted(set(expected.values()))]
        watched.append(index)
    else:
        expected = None
        shards = sorted(checkpoint.glob(stem+"*"+ext))
    require(shards and all(p.is_file() for p in shards), "The checkpoint is missing its weight files")
    watched.extend(shards)
    state = {p:stamp(p) for p in watched}
    roster = {}
    counts = Counter()
    for p in shards:
        dtypes = part_dtypes(p,is_safe)
        require(not roster.keys() & dtypes.keys(), f"Repeated tensor in {p.name}")
        roster.update({n:p.name for n in dtypes})
        counts.update(dtypes.values())
    require(expected is None or expected==roster, "The weight index and shard contents disagree")
    require(counts["BF16"]>0, f"No BF16 weights found. Detected dtypes: {dict(counts)}")
    return dict(model_type=cfg.get("model_type"),architectures=cfg.get("architectures"),
                source_shards=len(shards),source_dtypes=dict(counts)),state


def environment(scratch, threads):
    env = dict(os.environ,TMPDIR=str(scratch),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),MALLOC_ARENA_MAX="2")
    env.pop("NO_LOCAL_GGUF",None)
    return env


def converter(source,args,env,allow_model_code=False):
    script = source/"convert_hf_to_gguf.py"
    require(script.is_file() and (source/"gguf-py"/"gguf").is_dir(),
            "--llama-source must point to the llama.cpp source checkout")
    cmd = [sys.executable,"-I","-B","-c",RUN_CONVERTER,str(script),str(int(allow_model_code)),*args]
    subprocess.run(cmd,env=env,check=True)
