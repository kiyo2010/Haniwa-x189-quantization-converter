# SPDX-License-Identifier: Apache-2.0
"""Public KFTT selection and bounded, resumable Qwen3 Gram capture."""
from contextlib import ExitStack
from collections import defaultdict
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

import numpy as np

from . import RECIPE
from .roles import EMBEDDING, roles

SELECTION_DOMAIN = "X189-public-KFTT-v1"
ARCHIVE = "fcfcaa670d6d59aa691b0e909c0d7c393852dd2fb1d6310fda9b3282dc6d1638"
SHARDS = {
    "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
    "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
    "model.safetensors.index.json": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
    "config.json": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}
MEMBERS = {
    "en": "77cfe62e038a7cfa11b22a5c853f93e902bd3ef2d4ebafe213c635f40d8d9864",
    "ja": "99ede0a61a798c62e5eec37fc9c4c5d5c7d3cc60d8267177bb0e08e2ac362303",
}
WIDTH = 512
VOCAB = 151936
CHUNKS = 5
CAP = 16384
POOL = 8192


def need(ok, description):
    if not ok:
        raise ValueError(description)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def stamp(path):
    s = path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def write(path, fn):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".calibration-", delete=False) as stream:
        pending = Path(stream.name)
        try:
            fn(stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)


def atomic_json(path, value):
    write(path, lambda g: g.write((json.dumps(value, sort_keys=True, allow_nan=False)+"\n").encode()))


def owner(name):
    if name in (EMBEDDING, "lm_head.weight"):
        return "lm_head"
    name = name.removesuffix(".weight")
    stem, sep, role = name.rpartition(".")
    aliases = {"k_proj":"q_proj", "v_proj":"q_proj", "up_proj":"gate_proj"}
    return stem+sep+aliases.get(role, role)


def widths():
    result = {}
    for n, kind in roles().items():
        if kind != "BF16":
            result[owner(n)] = 6144 if n.endswith(".down_proj.weight") else 2048
    need(len(result) == 113, "Invalid shared-input owner roster")
    return dict(sorted(result.items()))


def corpus_pool(path, scratch):
    """Read the two known regular files without extracting any archive paths."""
    with tempfile.TemporaryFile(dir=scratch) as english:
        seen = set()
        heap = []
        with tarfile.open(path, "r|gz") as archive:
            for item in archive:
                for lang, sha in MEMBERS.items():
                    if item.name != f"kftt-data-1.0/data/orig/kyoto-train.{lang}":
                        continue
                    need(item.isfile() and lang not in seen, "Unexpected KFTT tar member")
                    checksum = hashlib.sha256()
                    with archive.extractfile(item) as stream:
                        if lang == "en":
                            for data in iter(lambda: stream.read(1<<20), b""):
                                checksum.update(data)
                                english.write(data)
                            english.seek(0)
                        else:
                            need("en" in seen, "Unexpected KFTT member order")
                            for number, (a,b) in enumerate(zip(english,stream,strict=True),1):
                                checksum.update(b)
                                rank = int.from_bytes(hashlib.sha256(f"{SELECTION_DOMAIN}|{number}".encode()).digest(), "big")
                                if len(heap) == POOL and rank >= -heap[0][0]:
                                    continue
                                en,ja = a.decode().strip(),b.decode().strip()
                                if (not 24 <= len(en) <= 1600 or not 12 <= len(ja) <= 1000
                                    or not re.search(r"[A-Za-z]{3}",en) or not re.search(r"[ぁ-ゖァ-ヺ一-龯]",ja)):
                                    continue
                                item_rank = (-rank, number, en, ja)
                                if len(heap) < POOL:
                                    heapq.heappush(heap,item_rank)
                                else:
                                    heapq.heapreplace(heap,item_rank)
                            need(number == 440288, "Unexpected KFTT aligned training line count")
                    need(checksum.hexdigest() == sha, f"Unpinned KFTT {lang} training member")
                    seen.add(lang)
        need(seen == set(MEMBERS), "Incomplete KFTT members")
        return [(n,en,ja) for _,n,en,ja in sorted(heap, reverse=True)]


def select_pairs(pool, encode, cap=CAP):
    """Take first eligible fixed-rank pair at each step; forbid text reuse."""
    budget = [0,0]
    selected, lines = [], []
    seen = set()
    for number,en,ja in pool:
        if any(abs(old-number)<21 for old in lines):
            continue
        # Unicode case folding and whitespace removal suffice for exact text deduplication.
        keys = ("".join(en.casefold().split()),"".join(ja.casefold().split()))
        if keys[0] == keys[1] or any(key in seen for key in keys):
            continue
        pair = [encode(en),encode(ja)]
        size = list(map(len,pair))
        if (not all(20<=n<=160 for n in size) or min(size)/max(size)<0.72
            or any(min(ids)<0 or max(ids)>=VOCAB for ids in pair)):
            continue
        updated = [a+b for a,b in zip(budget,size,strict=True)]
        if max(updated)>cap or abs(updated[0]-updated[1])>34:
            continue
        selected.extend({"line":number,"lang":lang,"ids":ids} for lang,ids in zip(("en","ja"),pair,strict=True))
        lines.append(number)
        seen.update(keys)
        budget = updated
    return selected,budget


def selection(checkpoint, kftt, base):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(checkpoint/"tokenizer.json"))
    rows, budget = select_pairs(corpus_pool(kftt,base), lambda s: tok.encode(s,add_special_tokens=False).ids)
    need(len(rows)>=CHUNKS*2 and min(budget)>=CAP-160, "Public KFTT selection did not fill both languages")
    return rows,budget


def restore(checkpoint, device, torch):
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, Qwen3Config
    from transformers.modeling_utils import no_init_weights

    config = Qwen3Config.from_json_file(str(checkpoint/"config.json"))
    index = read(checkpoint/"model.safetensors.index.json")["weight_map"]
    need(config.tie_word_embeddings and config.vocab_size == VOCAB and config.num_hidden_layers == 28,
         "Unexpected Qwen3 architecture")
    with torch.inference_mode(), no_init_weights(), torch.device(device):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32, attn_implementation="eager")
    model.tie_weights()
    params = dict(model.named_parameters())
    need(set(params)==set(roles()) and set(index)-set(params) in (set(),{"lm_head.weight"})
         and model.get_input_embeddings().weight is model.get_output_embeddings().weight, "Physical Qwen3 roster differs")
    with ExitStack() as ctx, torch.inference_mode():
        files = {f:ctx.enter_context(safe_open(checkpoint/f, framework="pt",device="cpu")) for f in set(index.values())}
        for name,target in params.items():
            data = files[index[name]].get_slice(name)
            need(data.get_dtype()=="BF16" and tuple(data.get_shape())==tuple(target.shape), f"Unexpected BF16 tensor: {name}")
            tied = files[index["lm_head.weight"]].get_slice("lm_head.weight") if name==EMBEDDING and "lm_head.weight" in index else None
            for first in range(0,len(target),1024):
                part = data[first:first+1024]
                if tied is not None:
                    need(torch.equal(part.view(torch.int16),tied[first:first+1024].view(torch.int16)), "Tied BF16 payload differs")
                target[first:first+len(part)].copy_(part)
    return model.eval()


def capture_chunk(model, rows, path, device, torch):
    selected = widths()
    buffers = {n:torch.zeros((dim//WIDTH,WIDTH,WIDTH),dtype=torch.float64,device=device) for n,dim in selected.items()}
    modules = dict(model.named_modules())
    count = defaultdict(int)
    def observe(name, matrix):
        x = matrix.detach().reshape(-1, selected[name])
        z = x.reshape(len(x),-1,WIDTH).transpose(0,1).double().contiguous()
        buffers[name].baddbmm_(z.transpose(-1,-2),z)
        count[name] += len(x)
    hooks = [modules[n].register_forward_pre_hook(lambda _m,a,key=n: observe(key,a[0]))
             for n in selected if n!="lm_head"]
    try:
        with torch.inference_mode():
            for row in rows:
                hidden = model.model(input_ids=torch.tensor([row["ids"]],device=device),use_cache=False)
                observe("lm_head",hidden.last_hidden_state)
    finally:
        for hook in hooks:
            hook.remove()
    tokens = sum(len(row["ids"]) for row in rows)
    need(set(count)==set(selected) and all(v==tokens for v in count.values()), "Qwen3 activation owner token counts differ")
    content = {}
    def store(stream):
        for name in selected:
            block = buffers[name].cpu().numpy().astype("<f8",copy=False)
            need(np.isfinite(block).all(), "Nonfinite calibration Gram")
            view = memoryview(np.ascontiguousarray(block)).cast("B")
            content[name] = {"offset":stream.tell(),"bytes":len(view),"sha256":hashlib.sha256(view).hexdigest()}
            stream.write(view)
    write(path,store)
    return {"tokens":tokens,"owners":content,"stamp":stamp(path)}


def capture(checkpoint, kftt, work, device, threads):
    import torch

    checkpoint,kftt,work = (Path(p).resolve(strict=True) if i<2 else Path(p).resolve()
                            for i,p in enumerate((checkpoint,kftt,work)))
    base = work/"calibration"
    base.mkdir(mode=0o700,parents=True,exist_ok=True)
    need(not base.is_symlink(),"Calibration work cannot be redirected")
    with (base/"lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(device)
        runtime = {"device":str(device),"torch":str(torch.__version__),"threads":threads}
        if device.type=="cuda":
            runtime["gpu"]=torch.cuda.get_device_name(device)
        assets = {str(checkpoint/n):sha for n,sha in SHARDS.items()} | {str(kftt):ARCHIVE}
        before={f:stamp(Path(f)) for f in assets}
        source={"stamps":before,"sha256":assets,"recipe":RECIPE,"runtime":runtime,"code_sha256":digest(Path(__file__))}
        identity = base/"identity.json"
        if identity.exists():
            need(read(identity)==source, "Calibration inputs or runtime changed")
        else:
            for path,sha in assets.items():
                need(digest(Path(path))==sha, f"Pinned original input differs: {path}")
            need(before=={f:stamp(Path(f)) for f in assets}, "Calibration source changed during authentication")
            atomic_json(identity, source)
        path = base/"capture.json"
        if path.exists():
            Calibration(work)
            return path
        roster=base/"rows.json"
        if roster.exists():
            plan=read(roster)
        else:
            rows,budget=selection(checkpoint,kftt,base)
            plan={"rows":rows,"budget":budget}
            atomic_json(roster,plan)
        rows=plan["rows"]
        freq=np.bincount(np.concatenate([r["ids"] for r in rows]),minlength=VOCAB).astype("<u8")
        slices=np.linspace(0,len(rows)//2,CHUNKS+1,dtype=np.int64)*2
        model=None
        chunks=[]
        for j,(first,stop) in enumerate(zip(slices[:-1],slices[1:],strict=True)):
            marker=base/f"chunk-{j:02}.json"
            binary=base/f"chunk-{j:02}.f64"
            signature={"identity_sha256":digest(identity),"roster_sha256":digest(roster),"span":[int(first),int(stop)]}
            if marker.exists():
                card=read(marker)
                need(card["signature"]==signature and card["stamp"]==stamp(binary),"Calibration checkpoint changed")
            else:
                if model is None:
                    model=restore(checkpoint,device,torch)
                card=capture_chunk(model,rows[first:stop],binary,device,torch)|{"signature":signature}
                atomic_json(marker,card)
            chunks.append({"marker":marker.name,"sha256":digest(marker)})
        need(before=={f:stamp(Path(f)) for f in assets},"Calibration source changed")
        frequency=base/"frequencies.npy"
        write(frequency,lambda f:np.save(f,freq,allow_pickle=False))
        atomic_json(path,{"complete":True,"recipe":RECIPE,"identity_sha256":digest(identity),
                          "tokens":int(freq.sum()),"targets_by_language":plan["budget"],"pairs":len(rows)//2,
                          "frequencies_sha256":digest(frequency),"chunks":chunks})
        return path


class Calibration:
    def __init__(self,work):
        self.base=Path(work)/"calibration"
        doc=read(self.base/"capture.json")
        need(doc["complete"] is True and doc["recipe"]==RECIPE
             and doc["identity_sha256"]==digest(self.base/"identity.json"),"Unsealed public calibration")
        self.tokens=doc["tokens"]
        self.chunks=[]
        for j,card in enumerate(doc["chunks"]):
            marker=self.base/f"chunk-{j:02}.json"
            need(card["marker"]==marker.name and digest(marker)==card["sha256"],"Changed Gram marker")
            chunk=read(marker)
            body=marker.with_suffix(".f64")
            need(stamp(body)==chunk["stamp"],"Changed Gram binary")
            self.chunks.append((body,chunk))
        need(len(self.chunks)==CHUNKS and sum(c["tokens"] for _,c in self.chunks)==self.tokens,
             "Inconsistent capture token count")
        self._frequency_sha=doc["frequencies_sha256"]

    def gram(self,name):
        key=owner(name)
        count=widths()[key]//WIDTH
        result=np.zeros((count,WIDTH,WIDTH),np.float64)
        for path,marker in self.chunks:
            span=marker["owners"][key]
            mapped=np.memmap(path,mode="r",dtype="<f8",offset=span["offset"],shape=result.shape)
            need(mapped.nbytes==span["bytes"] and hashlib.sha256(memoryview(mapped).cast("B")).hexdigest()==span["sha256"]
                 and stamp(path)==marker["stamp"],"Changed authenticated Gram owner span")
            result+=mapped
        return result/self.tokens

    def diagonal(self,name):
        return self.gram(name).diagonal(axis1=1,axis2=2).astype("<f4").ravel()

    def frequencies(self):
        path=self.base/"frequencies.npy"
        need(digest(path)==self._frequency_sha,"Changed KFTT frequency bytes")
        values=np.load(path,allow_pickle=False)
        need(values.shape==(VOCAB,) and values.dtype==np.dtype("<u8") and int(values.sum())==self.tokens,
             "KFTT frequency coverage differs")
        return values
