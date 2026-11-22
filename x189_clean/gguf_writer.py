# SPDX-License-Identifier: Apache-2.0
"""Stream X189 tensor replacements into a standard GGUF v3 file."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
import zlib

import gguf
import numpy as np


ALIGNMENT = 32
CHUNK = 1 << 20


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


def _prefix(meta, rows, *, recipe):
    stop = max(f.offset + sum(x.nbytes for x in f.parts) for f in meta.fields.values())
    p = bytearray(meta.data[:stop])
    count = len(meta.tensors)
    need(struct.unpack_from("<4sIQ",p)==(b"GGUF",3,count),"Invalid GGUF metadata prefix")
    need(meta.alignment==ALIGNMENT and stop<=meta.data_offset and len(rows)==count,
         "The source GGUF directory is incompatible")
    need(meta.get_field("x189.recipe") is None,"Input already contains X189 recipe metadata")
    kv_count=struct.unpack_from("<Q",p,16)[0]
    struct.pack_into("<Q",p,16,kv_count+1)
    key=b"x189.recipe"
    value=recipe.encode("utf8")
    p+=struct.pack("<Q",len(key))+key+struct.pack("<IQ",int(gguf.GGUFValueType.STRING),len(value))+value
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


def _verify(path, expected_meta, rows, prefix_bytes, size, *, recipe):
    image = gguf.GGUFReader(path, "r")
    need(image.data_offset == prefix_bytes and image.alignment == ALIGNMENT and
         path.stat().st_size == size and len(image.tensors) == len(rows),
         "Independent GGUF geometry changed")
    need(tuple(expected_meta.fields)+("x189.recipe",)==tuple(image.fields),"Independent GGUF metadata keys changed")
    need(field(image,"x189.recipe")==recipe,"Independent X189 recipe marker differs")
    for key, f in expected_meta.fields.items():
        if key == "GGUF.kv_count":
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


def write_replacements(source, output, records, *, recipe):
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
    header,extent = _prefix(meta,rows,recipe=recipe)
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
        _verify(stage,meta,rows,len(header),len(header)+extent,recipe=recipe)
        digest = sha_file(stage)
        os.link(stage,output)
    finally:
        stage.unlink(missing_ok=True)
    return dict(file=str(output),bytes=output.stat().st_size,sha256=digest,
                architecture=field(meta,"general.architecture"),independently_verified_records=len(rows),
                ggml_types=dict(Counter(r["type"] for r in rows)))
