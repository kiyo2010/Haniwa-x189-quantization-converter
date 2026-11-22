# SPDX-License-Identifier: Apache-2.0
"""Stream standalone X189-layout Qwen3 records into a stock GGUF v3 file."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import zlib

import gguf
import numpy as np


COMMIT = "8777c0c833effb3dec51f8db4f49f5c8b234c644"
TREE = "8fafcf7e7d9d7834ddf91753e3c1df9a050975a8"
SOURCE_TAR_SHA = "a5ea74766fa1ec3ed1341fe6f13c47a242c9b247fc5e6a858d06938aa6dad55b"
CONVERTER_FILES = 154
CONVERTER_TREE_SHA = "eb4592a6c47e85948a4dde38826cb6af89789425b1245c39a57cc2ff5cec857c"
ALIGNMENT = 32
CHUNK = 1 << 20
VOCAB = 151936
ELEMENTS = 1720574976
KINDS = {"F32": 234, "IQ1_S": 2, "IQ2_XXS": 5, "IQ3_XXS": 26, "IQ4_NL": 43}
EMBED = "model.embed_tokens.weight"
DIMS = {
    "embedding_length": "hidden_size",
    "block_count": "num_hidden_layers",
    "feed_forward_length": "intermediate_size",
    "attention.head_count": "num_attention_heads",
    "attention.head_count_kv": "num_key_value_heads",
    "attention.key_length": "head_dim",
    "attention.value_length": "head_dim",
    "context_length": "max_position_embeddings",
    "rope.freq_base": "rope_theta",
    "attention.layer_norm_rms_epsilon": "rms_norm_eps",
}


@dataclass(frozen=True)
class Record:
    """One raw LE GGML payload. Shape follows HF row-major order."""
    path: Path
    kind: str
    shape: tuple[int, ...]
    sha256: str | None = None


def need(condition, what):
    if not condition:
        raise ValueError(what)


def sha_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stamp(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def rounded(size):
    return (size + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def field(meta, key):
    value = meta.get_field(key)
    need(value is not None, f"Missing GGUF metadata field {key}")
    return value.contents()


def expected(cfg):
    need(cfg["model_type"] == "qwen3" and cfg["tie_word_embeddings"] is True and
         tuple(cfg[k] for k in ("vocab_size","hidden_size","head_dim","num_attention_heads",
                                "num_key_value_heads","intermediate_size","num_hidden_layers")) ==
         (VOCAB,2048,128,16,8,6144,28), "Checkpoint is outside the X189 Qwen3-1.7B layout")
    hidden, head, ff = cfg["hidden_size"], cfg["head_dim"], cfg["intermediate_size"]
    q, kv = head * cfg["num_attention_heads"], head * cfg["num_key_value_heads"]
    own = {EMBED: (VOCAB, hidden), "model.norm.weight": (hidden,)}
    attention = {"q_proj": (q, hidden), "k_proj": (kv, hidden), "v_proj": (kv, hidden),
                 "o_proj": (hidden, q), "q_norm": (head,), "k_norm": (head,)}
    mlp = {"gate_proj": (ff, hidden), "up_proj": (ff, hidden), "down_proj": (hidden, ff)}
    for i in range(cfg["num_hidden_layers"]):
        base = f"model.layers.{i}"
        own.update({f"{base}.self_attn.{name}.weight": shape for name, shape in attention.items()})
        own.update({f"{base}.mlp.{name}.weight": shape for name, shape in mlp.items()})
        own[f"{base}.input_layernorm.weight"] = (hidden,)
        own[f"{base}.post_attention_layernorm.weight"] = (hidden,)
    need(len(own) == 310 and sum(map(math.prod, own.values())) == ELEMENTS, "Invalid tied model geometry")
    return own


def _rows(records, owned, require_x189=True):
    need(set(records) == set(owned), "Records must cover exactly the independent HF weights")
    mapper = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, 28)
    result, names = [], set()
    for hf, shape in owned.items():
        rec = records[hf]
        kind = rec.kind
        need(tuple(rec.shape) == shape and kind in KINDS and (len(shape) != 1 or kind == "F32"),
             f"Invalid shape/type for {hf}")
        path = Path(rec.path).resolve()
        stat = path.stat()
        need(path.is_file() and (rec.sha256 is None or re.fullmatch(r"[0-9a-f]{64}", rec.sha256)),
             f"Invalid raw payload or SHA field for {hf}")
        code = getattr(gguf.GGMLQuantizationType, kind)
        block, step = gguf.GGML_QUANT_SIZES[code]
        byte_count = math.prod(shape) // block * step
        gname = mapper.get_name(hf, try_suffixes=(".weight",))
        need(gname is not None and gname not in names and gname != "output.weight" and
             shape[-1] % block == 0 and stat.st_size == byte_count, f"Wrong GGML extent or repeated name for {hf}")
        names.add(gname)
        result.append(dict(hf=hf, name=gname, type=kind, ggml_type=int(code), shape=list(shape),
                           bytes=byte_count, input=path, input_stamp=stamp(stat), expected_sha256=rec.sha256))
    if require_x189:
        from .roles import roles
        assignment={n:("F32" if k in ("BF16","XE82") else k) for n,k in roles().items()}
        need(Counter(x["type"] for x in result) == KINDS and
             {n:r.kind for n,r in records.items()}==assignment,
             "Record types differ from the registered per-tensor X189 assignment")
    return result


def checkout(path):
    """Authenticate the exact converter closure without assuming a Git checkout."""
    files = [path / "convert_hf_to_gguf.py"]
    for name in ("conversion","gguf-py"):
        root = path/name
        need(root.is_dir() and not root.is_symlink(), f"Missing official converter subtree {name}")
        for f in root.rglob("*"):
            if "__pycache__" in f.relative_to(root).parts:
                continue
            need(not f.is_symlink(), f"Unexpected converter symlink {f}")
            if f.is_file():
                files.append(f)
    digest = hashlib.sha256()
    for f in sorted(files):
        need(f.is_file() and not f.is_symlink(), f"Missing official converter file {f}")
        rel = f.relative_to(path).as_posix().encode()
        digest.update(struct.pack("<IQ",len(rel),f.stat().st_size))
        digest.update(rel)
        digest.update(bytes.fromhex(sha_file(f)))
    need(len(files)==CONVERTER_FILES and digest.hexdigest()==CONVERTER_TREE_SHA,
         "The converter and gguf-py trees differ from the pinned llama.cpp b12487 release")
    return dict(commit=COMMIT,tree=TREE,source_archive_expected_sha256=SOURCE_TAR_SHA,
                converter_files=len(files),converter_tree_sha256=digest.hexdigest(),converter_sha256=sha_file(files[0]),
                qwen_conversion_sha256=sha_file(path/"conversion/qwen.py"))


def _converter(checkpoint, llama_src, out):
    script = llama_src / "convert_hf_to_gguf.py"
    # Isolate the process from PYTHONPATH and select the pinned converter's gguf-py.
    run = ("import pathlib,runpy,sys;"
           "p=pathlib.Path(sys.argv.pop(1));sys.path.insert(0,str(p.parent));"
           "sys.argv[0]=str(p);runpy.run_path(str(p),run_name='__main__')")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("NO_LOCAL_GGUF", None)
    subprocess.run([sys.executable, "-I", "-B", "-X", f"pycache_prefix={out.parent/'.pyc'}",
                    "-c", run, str(script), str(checkpoint),
                    "--vocab-only", "--outtype", "f32", "--outfile", str(out)],
                   cwd=llama_src, env=env, check=True)


def metadata(reader, cfg):
    need(not reader.tensors and reader.alignment == ALIGNMENT and
         field(reader, "general.architecture") == "qwen3" and field(reader, "general.file_type") == 0,
         "Official converter did not produce a standard Qwen3 vocab-only image")
    for name, cfg_key in DIMS.items():
        value, expected_value = field(reader, "qwen3." + name), cfg[cfg_key]
        need(isinstance(value, (int, float)) and math.isclose(value, expected_value, rel_tol=1e-6, abs_tol=0),
             f"Qwen3 converter metadata disagrees with checkpoint config for {name}")
    tokens = reader.get_field("tokenizer.ggml.tokens")
    need(tokens is not None and len(tokens.data) == VOCAB and
         reader.get_field("tokenizer.ggml.merges") is not None, "Original Qwen BPE vocabulary geometry")


def _prefix(meta, rows, *, from_model=False):
    stop = max(f.offset + sum(x.nbytes for x in f.parts) for f in meta.fields.values())
    p = bytearray(meta.data[:stop])
    count = len(meta.tensors) if from_model else 0
    need(struct.unpack_from("<4sIQ",p)==(b"GGUF",3,count),"Invalid GGUF metadata prefix")
    if from_model:
        need(meta.alignment==ALIGNMENT and stop<=meta.data_offset and len(rows)==count,
             "The source GGUF directory is incompatible")
    else:
        need(0<=len(meta.data)-stop<ALIGNMENT and not meta.data[stop:].any(),"Invalid vocabulary-only GGUF")
    struct.pack_into("<Q", p, 8, len(rows))
    pos = 0
    for r in rows:
        name = r["name"].encode("utf-8")
        shape = r["shape"]
        p += struct.pack("<Q", len(name)) + name + struct.pack("<I", len(shape))
        p += struct.pack("<" + "Q" * len(shape), *shape[::-1])
        p += struct.pack("<IQ", r["ggml_type"], pos)
        r["offset"] = pos
        pos += rounded(r["bytes"])
    p += bytes(-len(p) % ALIGNMENT)
    return p, pos


def _stream(file, n, out=None, check_f32=False):
    sha, crc = hashlib.sha256(), 0
    while n:
        s = file.read(min(CHUNK, n))
        need(bool(s), "Unexpectedly truncated GGUF payload")
        if check_f32:
            need(len(s)%4==0 and np.isfinite(np.frombuffer(s,"<f4")).all(),"Invalid F32 payload")
        n -= len(s)
        sha.update(s)
        crc = zlib.crc32(s, crc)
        if out is not None:
            out.write(s)
    return sha.hexdigest(), f"{crc:08x}"


def _write_rows(out, rows):
    for row in rows:
        with row["input"].open("rb") as f:
            need(stamp(os.fstat(f.fileno())) == row["input_stamp"], f"Raw payload changed for {row['hf']}")
            sha, crc = _stream(f, row["bytes"], out, check_f32=row["type"]=="F32")
            need(not f.read(1) and stamp(os.fstat(f.fileno())) == row["input_stamp"] ==
                 stamp(row["input"].stat()), f"Raw payload changed while writing {row['hf']}")
        need(row["expected_sha256"] in (None, sha), f"Unexpected raw payload SHA for {row['hf']}")
        row.update(payload_sha256=sha, payload_crc32=crc)
        out.write(bytes(rounded(row["bytes"]) - row["bytes"]))


def _verify(path, expected_meta, rows, prefix_bytes, size):
    image = gguf.GGUFReader(path, "r")
    need(image.data_offset == prefix_bytes and image.alignment == ALIGNMENT and
         path.stat().st_size == size and len(image.tensors) == len(rows),
         "Independent GGUF geometry changed")
    need(tuple(expected_meta.fields) == tuple(image.fields), "Independent GGUF metadata keys changed")
    for key, f in expected_meta.fields.items():
        # Synthetic GGUF.tensor_count is the one header field intentionally replaced.
        if key == "GGUF.tensor_count":
            continue
        need(all(x.tobytes() == y.tobytes() for x,y in zip(f.parts,image.fields[key].parts,strict=True)),
             f"GGUF metadata changed for {key}")
    with path.open("rb") as file:
        for observed, row in zip(image.tensors, rows, strict=True):
            offset = prefix_bytes + row["offset"]
            need(observed.name == row["name"] and int(observed.tensor_type) == row["ggml_type"] and
                 tuple(int(v) for v in observed.shape[::-1]) == tuple(row["shape"]) and
                 int(observed.n_bytes) == row["bytes"] and int(observed.data_offset) == offset,
                 f"Independent GGUF directory readback for {row['name']}")
            file.seek(offset)
            need(_stream(file, row["bytes"]) == (row["payload_sha256"],row["payload_crc32"]),
                 f"Independent GGUF payload SHA or CRC for {row['name']}")
            pad = rounded(row["bytes"]) - row["bytes"]
            need(file.read(pad) == bytes(pad), f"Independent GGUF nonzero padding for {row['name']}")
        need(file.tell() == size and not file.read(1), "GGUF unaccounted trailer")


def write_replacements(source, output, records):
    """Write a full standard GGUF with an unchanged model's names, shapes and metadata."""
    meta = gguf.GGUFReader(source,"r")
    expected = {t.name:t for t in meta.tensors}
    need(bool(records) and records.keys()<=expected.keys(),"Replacement tensor roster differs")
    rows = []
    for name,t in expected.items():
        shape = tuple(int(x) for x in reversed(t.shape))
        if name in records:
            item = records[name]
            path = item.path
            kind = gguf.GGMLQuantizationType[item.kind]
            block,nbytes = gguf.GGML_QUANT_SIZES[kind]
            need(shape==item.shape and shape[-1]%block==0,"Replacement GGUF shape or block width changed")
            count = math.prod(shape)//block*nbytes
            need(path.stat().st_size==count and item.sha256 is not None,"Incomplete GGUF replacement")
            offset, sha = 0, item.sha256
        else:
            path = source
            kind, count, offset = t.tensor_type, int(t.n_bytes), int(t.data_offset)
            sha = None
        rows.append(dict(name=name,shape=shape,type=kind.name,ggml_type=int(kind),bytes=count,
                         source=path,source_offset=offset,source_stamp=stamp(path.stat()),expected_sha256=sha))
    header,extent = _prefix(meta,rows,from_model=True)
    temp = tempfile.NamedTemporaryFile(dir=output.parent,prefix=".x189-model-",delete=False)
    stage = Path(temp.name)
    try:
        with temp as dst:
            dst.write(header)
            for entry in rows:
                with entry["source"].open("rb") as src:
                    need(stamp(os.fstat(src.fileno()))==entry["source_stamp"],"Tensor input changed before publication")
                    src.seek(entry["source_offset"])
                    sha,crc = _stream(src,entry["bytes"],dst,check_f32=entry["type"]=="F32")
                    need(stamp(os.fstat(src.fileno()))==entry["source_stamp"] and
                         entry["expected_sha256"] in (None,sha),"Tensor input changed during publication")
                entry.update(payload_sha256=sha,payload_crc32=crc)
                dst.write(bytes(rounded(entry["bytes"])-entry["bytes"]))
            dst.flush()
            os.fsync(dst.fileno())
            need(dst.tell()==len(header)+extent,"Final GGUF extent differs")
        _verify(stage,meta,rows,len(header),len(header)+extent)
        digest = sha_file(stage)
        os.link(stage,output)
    finally:
        stage.unlink(missing_ok=True)
    return dict(file=str(output),bytes=output.stat().st_size,sha256=digest,
                architecture=field(meta,"general.architecture"),independently_verified_records=len(rows),
                ggml_types=dict(Counter(r["type"] for r in rows)))


def write(checkpoint: Path, llama_src: Path, work: Path, records: dict[str, Record], output: Path) -> dict:
    """Create one no-clobber stock GGUF and return its independently read physical ledger."""
    checkpoint, llama_src, work = [Path(x).resolve() for x in (checkpoint,llama_src,work)]
    output = Path(output).absolute()
    need(not output.is_symlink(),"Output must be a new direct path")
    need(sys.byteorder == "little" and not output.exists() and output.parent.is_dir() and work.is_dir(),
         "A little-endian host, existing directories and new output path are required")
    pin = checkout(llama_src)
    paths = [checkpoint/n for n in ("config.json","tokenizer.json","tokenizer_config.json")]
    input_stamps = {p:stamp(p.stat()) for p in paths}
    inputs = {p.name:sha_file(p) for p in paths}
    cfg = json.loads(paths[0].read_text())
    rows = _rows(records, expected(cfg))
    with tempfile.TemporaryDirectory(dir=work,prefix=".x189-vocab-") as tmp:
        vocab = Path(tmp)/"vocab.gguf"
        _converter(checkpoint, llama_src, vocab)
        need(input_stamps == {p:stamp(p.stat()) for p in paths}, "HF tokenizer or config changed during conversion")
        template = gguf.GGUFReader(vocab,"r")
        metadata(template,cfg)
        header, extent = _prefix(template, rows)
        size = len(header) + extent
        with tempfile.NamedTemporaryFile(dir=output.parent,prefix=".x189-gguf-",delete=False) as f:
            stage = Path(f.name)
            try:
                f.write(header)
                _write_rows(f,rows)
                f.flush()
                os.fsync(f.fileno())
                need(f.tell() == size, "GGUF write byte accounting")
                _verify(stage,template,rows,len(header),size)
                digest = sha_file(stage)
                # An atomic hard link publishes the completed file and fails if the destination exists.
                os.link(stage,output)
            finally:
                stage.unlink(missing_ok=True)
        tokens = sum(sum(x.nbytes for x in f.parts) for k,f in template.fields.items() if k.startswith("tokenizer."))
        vocab_sha = sha_file(vocab)
    public = [{k:v for k,v in row.items() if k not in ("input","input_stamp","expected_sha256")} for row in rows]
    return dict(file=str(output),bytes=size,sha256=digest,format="GGUF v3",alignment=ALIGNMENT,
                payload_bytes=sum(r["bytes"] for r in rows),header_and_directory_bytes=len(header),
                tokenizer_fields_bytes=tokens,weight_bytes=size-tokens,
                padding_bytes=extent-sum(r["bytes"] for r in rows),unique_elements=ELEMENTS,
                tied={"lm_head.weight":EMBED},vocab_size=VOCAB, type_counts=dict(Counter(r["type"] for r in rows)),
                independently_verified_records=len(rows),metadata_source=pin|dict(vocab_only_sha256=vocab_sha),
                hf_metadata_sha256=inputs,records=public)


def fixture():
    """Create tiny stock records, verify raw fields, and reject a corrupted payload."""
    with tempfile.TemporaryDirectory(prefix="x189-gguf-fixture-") as tmp:
        tmp = Path(tmp)
        meta = tmp/"meta.gguf"
        g = gguf.GGUFWriter(meta,"qwen3")
        g.add_file_type(0)
        g.add_token_list(["test","xy"])
        g.write_header_to_file()
        g.write_kv_data_to_file()
        g.write_tensors_to_file()
        g.close()
        reader = gguf.GGUFReader(meta,"r")
        f32, native = tmp/"norm.raw", tmp/"embedding.raw"
        f32.write_bytes(np.array([1.,-2.,.5],dtype="<f4").tobytes())
        record_fmt = gguf.GGMLQuantizationType.IQ2_XXS
        _, step = gguf.GGML_QUANT_SIZES[record_fmt]
        native.write_bytes(bytes(step))
        records = {"model.norm.weight":Record(f32,"F32",(3,),sha_file(f32)),
                   EMBED:Record(native,"IQ2_XXS",(1,256))}
        shapes = {"model.norm.weight":(3,),EMBED:(1,256)}
        rows = _rows(records,shapes,require_x189=False)
        head, charge = _prefix(reader,rows)
        out = tmp/"fixture.gguf"
        with out.open("wb") as f:
            f.write(head)
            _write_rows(f,rows)
        _verify(out,reader,rows,len(head),len(head)+charge)
        with out.open("r+b") as f:
            f.seek(len(head))
            f.write(b"\xff")
        try:
            _verify(out,reader,rows,len(head),len(head)+charge)
        except ValueError as e:
            need("SHA or CRC" in str(e),"Unexpected corrupt payload detection")
        else:
            raise ValueError("GGUF fixture accepted corrupted F32 bytes")
        bad = records | {EMBED:Record(native,"IQ2_XXS",(1,257))}
        try:
            _rows(bad,shapes,require_x189=False)
        except ValueError:
            pass
        else:
            raise ValueError("GGUF fixture accepted incorrect native shape")
    return dict(pass_=True,records=2,native_payload_bytes=step,corrupt_payload_rejected=True)


if __name__ == "__main__":
    print(json.dumps(fixture()))
