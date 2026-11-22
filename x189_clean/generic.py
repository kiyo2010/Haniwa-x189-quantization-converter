# SPDX-License-Identifier: Apache-2.0
"""Apply X189 lattice and IQ quantization to standard GGUF model tensors."""
import argparse
import codecs
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import calibration
from . import portable
from .e8 import COLS, ROWS, Encoder, _int8_rows, _weighted_fit, apply_vocabulary, vocabulary_rank
from .native import NativeIQ

RECIPE = "X189-Generic-v1"
ROUTING = ("ffn_gate_inp", "router", "expert_score")
KEY_VALUE = ("attn_k.weight", "attn_v.weight")
EXPERT_IQ = ("ffn_gate_exps.weight", "ffn_up_exps.weight")
QUERY = re.compile(r"^blk\.(\d+)\.attn_(?:q|qkv)\.weight$")
MAX_CAL_BYTES = 256 * 1024


def binary(path):
    p = Path(path).resolve()
    portable.require(p.is_file() and os.access(p,os.X_OK),f"Expected a llama.cpp executable: {p}")
    return str(p)


def rounded(n,unit):
    return (n+unit-1)//unit*unit


def direct_tensor(tensor):
    shape = tuple(reversed(tuple(int(x) for x in tensor.shape)))
    return (len(shape) not in (2,3) or not tensor.name.endswith(".weight") or
            tensor.tensor_type.name!="F32" or any(key in tensor.name for key in ROUTING))


def encoding(name,shape):
    # Value/key projections and expert expansion keep native IQ block payloads.
    if (name.endswith(KEY_VALUE) or name.endswith(EXPERT_IQ)) and shape[-1]%32==0:
        return "IQ4_NL"
    return "E8"


def calibration_text(source,target):
    with source.open("rb") as stream:
        data = stream.read(MAX_CAL_BYTES+1)
    decoder = codecs.getincrementaldecoder("utf8")()
    text = decoder.decode(data[:MAX_CAL_BYTES],final=len(data)<=MAX_CAL_BYTES)
    portable.require(bool(text.strip()),"Calibration text is empty")
    target.write_text(text,encoding="utf8")
    return hashlib.sha256(text.encode()).hexdigest()


def imatrix(source,output,args,env):
    corpus = output.parent/"calibration.txt"
    digest = calibration_text(args.calibration_text,corpus)
    command = [binary(args.llama_imatrix),"-m",str(source),"-f",str(corpus),"-o",str(output),
               "--output-format","gguf","--process-output","--no-ppl","--no-escape",
               "--chunks",str(args.chunks),"-c",str(args.context),"-b",str(args.context),
               "-ub",str(args.context),"-ngl",str(args.imatrix_gpu_layers),"-t",str(args.threads)]
    subprocess.run(command,env=env,check=True)
    return corpus,digest


class Importance:
    def __init__(self,reader):
        from .gguf_writer import field

        portable.require(field(reader,"general.type")=="imatrix","Calibration file is not a llama.cpp GGUF importance matrix")
        self.entries = {t.name:t for t in reader.tensors}
        self.chunks = int(field(reader,"imatrix.chunk_count"))
        self.context = int(field(reader,"imatrix.chunk_size"))
        portable.require(self.chunks>0 and self.context>0 and
                         any(name.endswith(".in_sum2") for name in self.entries),
                         "The calibration text did not produce a full importance-matrix window")

    def get(self,name,experts,width):
        value = self.entries.get(name+".in_sum2")
        count = self.entries.get(name+".counts")
        if value is None and name=="token_embd.weight":
            value = self.entries.get("output.weight.in_sum2")
            count = self.entries.get("output.weight.counts")
        if value is None or count is None:
            return np.ones((experts,width),np.float32), np.zeros(experts,dtype=np.int64)
        n = np.asarray(count.data,dtype=np.float64).ravel()
        sums = np.asarray(value.data,dtype=np.float64).reshape(-1,width)
        portable.require(n.size==sums.shape[0] and n.size in (1,experts) and
                         np.all(np.isfinite(n)) and np.all(n>=0) and
                         np.all(np.isfinite(sums)) and np.all(sums>=0),
                         f"Invalid calibration second moments for {name}")
        d = np.ones_like(sums)
        for i in np.flatnonzero(n>0):
            avg = np.mean(sums[i])
            if avg>0:
                d[i] = np.maximum(sums[i]/avg,1e-4)
        return np.broadcast_to(d.astype(np.float32),(experts,width)),np.broadcast_to(n.astype(np.int64),(experts,))


def gram(diagonal):
    padded = rounded(len(diagonal),COLS)
    coverage = np.pad(diagonal,(0,padded-len(diagonal)),constant_values=1)
    out = np.zeros((padded//COLS,COLS,COLS),np.float32)
    ix = np.arange(COLS)
    out[:,ix,ix] = coverage.reshape(-1,COLS)
    return out


def transform_e8(source,name,weight,diagonal,encoder,scratch,extra):
    rows,cols = weight.shape
    shape = rounded(rows,ROWS),rounded(cols,COLS)
    moment = gram(diagonal)

    def read(first,stop):
        slab = np.zeros((stop-first,shape[1]),dtype="<f4")
        end = min(stop,rows)
        if first<end:
            slab[:end-first,:cols] = weight[first:end]
        return slab

    def sink(span,output):
        if span.start<rows:
            trimmed = np.ascontiguousarray(output[:rows-span.start,:cols],dtype="<f4")
            portable.require(np.isfinite(trimmed).all(),f"Nonfinite lattice output in {name}")
            source.write(memoryview(trimmed).cast("B"))

    with tempfile.TemporaryFile(dir=scratch) as physical:
        before = encoder.encode(name,shape,read,physical,moment)
        physical.seek(0)
        decoded = encoder.decode(physical,sink,shape)
    portable.require(before["sha256"]==decoded["sha256"],f"Lattice roundtrip failed for {name}")
    if extra=="rank":
        source.flush()
        base = np.memmap(source.name,mode="r+",dtype="<f4",shape=(rows,cols))
        rank_correction(encoder,name,weight,base,moment,shape[1])
        base.flush()
        del base
    return before["bytes"]


def rank_correction(encoder,name,weight,decoded,moment,padded_cols):
    rows,cols = weight.shape
    t = encoder.torch
    residual = np.zeros((rows,padded_cols),dtype=np.float32)
    residual[:,:cols] = weight-decoded
    with t.inference_mode():
        delta = t.from_numpy(residual).to(encoder.device)
        left,right = _weighted_fit(delta,moment,54,name,t)
        left_scale,left_codes = _int8_rows(left.cpu().numpy(),factor=True)
        right_scale,right_codes = _int8_rows(right.cpu().numpy(),factor=True)
        u = t.from_numpy(left_codes.astype(np.float32)*left_scale[:,None].astype(np.float32)).to(encoder.device)
        v = t.from_numpy(right_codes[:,:cols].astype(np.float32)*right_scale[:,None].astype(np.float32)).to(encoder.device)
        for start in range(0,rows,ROWS):
            s = slice(start,min(rows,start+ROWS))
            decoded[s] += (u[s]@v).cpu().numpy()


def token_frequency(source,corpus,args,env,limit,n):
    tokenizer = args.llama_imatrix.parent/"llama-tokenize"
    cmd = [binary(tokenizer),"-m",str(source),"-f",str(corpus),"--ids","--no-bos","--no-escape"]
    run = subprocess.run(cmd,env=env,check=True,stdout=subprocess.PIPE,text=True)
    ids = np.asarray(json.loads(run.stdout)[:limit],dtype=np.int64)
    portable.require(ids.ndim==1 and len(ids)>0 and (ids>=0).all() and (ids<n).all(),
                     "The calibration tokens and embedding vocabulary disagree")
    return np.bincount(ids,minlength=n)


def encode_tensors(image,importance,source,corpus,args,env,scratch):
    from .gguf_writer import Record,sha_file
    import torch

    torch.set_num_threads(args.threads)
    eligible = [t for t in image.tensors if not direct_tensor(t)]
    queries = sorted((t for t in eligible if QUERY.fullmatch(t.name) and len(t.shape)==2
                      and int(min(t.shape))>=54 and int(t.n_bytes)<=64*2**20),
                     key=lambda t:int(QUERY.fullmatch(t.name).group(1)))
    rank_names = {queries[i].name for i in (0,3*(len(queries)-1)//4)} if queries else set()
    encoder = Encoder(args.device)
    native = NativeIQ(args.ggml_lib.resolve())
    replacements = {}
    results = {}
    lattice_bytes = 0
    for i,tensor in enumerate(eligible):
        name = tensor.name
        shape = tuple(int(x) for x in tensor.shape[::-1])
        leaves = tensor.data.reshape((-1,*shape[-2:]))
        diagonal,observed = importance.get(name,len(leaves),shape[-1])
        mode = encoding(name,shape)
        logical_bytes = 0
        payload = scratch/f"{i:06}.payload"
        with payload.open("wb") as f:
            for expert,w in enumerate(leaves):
                if mode=="E8":
                    key = name if len(leaves)==1 else f"{name}/expert/{expert}"
                    logical_bytes += transform_e8(f,key,w,diagonal[expert],encoder,scratch,
                                                  "rank" if name in rank_names else None)
                else:
                    for first in range(0,len(w),ROWS):
                        data = native.encode(mode,w[first:first+ROWS],diagonal[expert])
                        native.decode(mode,data,shape[-1])
                        f.write(memoryview(data).cast("B"))
        correction = "rank54" if name in rank_names else None
        if name=="token_embd.weight" and mode=="E8" and len(shape)==2:
            base = np.memmap(payload,mode="r+",dtype="<f4",shape=shape)
            freq = token_frequency(source,corpus,args,env,importance.chunks*importance.context,shape[0])

            def original(first,stop):
                return tensor.data[first:stop]

            selected = vocabulary_rank(original,base,freq,k=min(2654,shape[0]))
            apply_vocabulary(selected,original,base)
            base.flush()
            del base
            correction = f"int8-vocabulary-{len(selected)}"
        replacements[name] = Record(payload,"F32" if mode=="E8" else mode,shape,sha_file(payload))
        lattice_bytes += logical_bytes
        results[name] = dict(method=mode,matrices=len(leaves),calibrated_matrices=int(np.count_nonzero(observed)),
                             standard_type=replacements[name].kind,correction=correction,shape=shape)
        print(f"[{i+1}/{len(eligible)}] {mode} {name}",flush=True)
    portable.require(lattice_bytes>0,"This model exposes no matrix suitable for X189 lattice encoding")
    return replacements,results,lattice_bytes


def run(args):
    from .gguf_writer import field,write_replacements
    import gguf

    source_dir = args.checkpoint.resolve()
    output = args.output.absolute()
    work = args.work.resolve()
    portable.require(source_dir.is_dir() and output.parent.is_dir(),"Use a local checkpoint and an existing output parent")
    portable.require(not output.exists() and not output.is_symlink() and output.suffix.lower()==".gguf",
                     "Choose a new .gguf output path")
    portable.require(not work.is_relative_to(source_dir) and not output.resolve().is_relative_to(source_dir),
                     "Keep the checkpoint separate from work and output")
    portable.require(1<=args.threads<=8 and 32<=args.context<=4096 and args.chunks>0,"Invalid calibration limits")
    portable.require(args.imatrix_gpu_layers>=0,"Invalid calibration GPU layer count")
    provenance,stamps = portable.inventory(source_dir)
    work.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work,prefix=".x189-generic-") as d:
        scratch = Path(d)
        env = portable.environment(scratch,args.threads)
        stock = scratch/"bf16-source-layout.gguf"
        portable.converter(args.llama_source.resolve(),
            [str(source_dir),"--outfile",str(stock),"--outtype","f32","--use-temp-file"],env,args.allow_model_code)
        source = gguf.GGUFReader(stock,"r")
        portable.require(bool(field(source,"general.architecture")) and bool(source.tensors),"The converter produced no model tensors")
        stats = scratch/"imatrix.gguf"
        corpus,corpus_sha = imatrix(stock,stats,args,env)
        importance = Importance(gguf.GGUFReader(stats,"r"))
        records,results,lattice_bytes = encode_tensors(source,importance,stock,corpus,args,env,scratch)
        portable.require(all(portable.stamp(p)==s for p,s in stamps.items()),"The BF16 source changed")
        report = write_replacements(stock,output,records)
        report.update(recipe=RECIPE,source=provenance,calibration_sha256=corpus_sha,windows=importance.chunks,
            window_size=importance.context,calibration_metric="activation_second_moment_diagonal",
            methods=dict(Counter(r["method"] for r in results.values())),
            preserved_tensors=len(source.tensors)-len(results),lattice_bytes_before_gguf_expansion=lattice_bytes,
            tensor_results=results)
        calibration.atomic_json(work/(output.name+".x189.json"),report)
        print(json.dumps({k:v for k,v in report.items() if k!="tensor_results"},ensure_ascii=False),flush=True)


def cli(action,argv):
    p = argparse.ArgumentParser(prog=f"python -m x189_clean {action}",
        description="Quantize a local HF BF16 checkpoint with X189 and produce a standard GGUF"
        if action=="convert" else "List the model architectures registered by llama.cpp")
    p.add_argument("--llama-source",type=Path,required=True)
    if action=="convert":
        for name in ("checkpoint","calibration-text","llama-imatrix","ggml-lib","output","work"):
            p.add_argument("--"+name,type=Path,required=True)
        p.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
        p.add_argument("--threads",type=int,default=2)
        p.add_argument("--context",type=int,default=256)
        p.add_argument("--chunks",type=int,default=8)
        p.add_argument("--imatrix-gpu-layers",type=int,default=0)
        p.add_argument("--allow-model-code",action="store_true",
                       help="Allow local custom Python supplied by a trusted checkpoint")
    args = p.parse_args(argv)
    start = time.monotonic()
    try:
        if action=="list-models":
            portable.converter(args.llama_source.resolve(),["--print-supported-models"],
                               portable.environment(Path.cwd(),1))
            return
        sys.path.insert(0,str(args.llama_source.resolve()/"gguf-py"))
        run(args)
        print(f"Elapsed {time.monotonic()-start:.2f} s",flush=True)
    except subprocess.CalledProcessError as exc:
        p.exit(1,f"llama.cpp exited with status {exc.returncode}; see the output above.\n")
    except (ValueError,OSError,RuntimeError,TypeError,KeyError) as exc:
        p.exit(1,f"x189_clean: {exc}\n")
