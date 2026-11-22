"""Write standard GGUF from the new X189 QPACK and original Qwen metadata."""
from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import rebuild as ptq
import stock_x189 as stock


def vocab_metadata(checkpoint, work):
    import gguf
    from x19b_model import field

    official = stock.read(stock.OUT/"official-cpu.json")
    ptq.need(stock.clean_commit() == official["source"], "Official converter checkout changed")
    converter = stock.CLEAN/"convert_hf_to_gguf.py"
    path = work/"official-vocabulary-only.gguf"
    sentencepiece = Path.home()/".cache/uv/archive-v0/JEZ24GGu2aqfwSkC"
    dependencies = {
        "sentencepiece/__init__.py": "0ba0cc4e3cb7c6d9c11d87a237412d694e26bc5e7ebd8262f9d5c55b0097200e",
        "sentencepiece/_sentencepiece.cpython-312-x86_64-linux-gnu.so":
            "7de17e6d4869cb5d22c0a3758d3b3efb083dd27af0e45a8432ec64a83f457976",
    }
    for name, expected in dependencies.items():
        ptq.check(sentencepiece/name, expected)
    bootstrap = ("import pathlib,runpy,sys; "
                 "vendor,script=sys.argv[1:3]; "
                 "sys.path[:0]=[str(pathlib.Path(script).parent),vendor]; "
                 "sys.argv=sys.argv[2:]; runpy.run_path(script,run_name='__main__')")
    cmd = [sys.executable, "-I", "-B", "-c", bootstrap, str(sentencepiece), str(converter),
           str(checkpoint), "--vocab-only", "--outtype", "f32", "--outfile"]
    def parse(candidate):
        result = gguf.GGUFReader(candidate)
        ptq.need(not result.tensors and field(result,"general.architecture") == "qwen3"
                 and result.alignment == 32 and field(result,"general.file_type") == 0
                 and len(result.fields["tokenizer.ggml.tokens"].data) == 151936,
                 "Vocab-only official metadata missing native architecture/tokenizer settings")
        return result
    if not path.exists():
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory(dir=work, prefix=".metadata-") as directory:
            staged=Path(directory)/"vocab.gguf"
            subprocess.run([*cmd, str(staged)], check=True, env=env)
            parse(staged)
            os.link(staged,path)
    reader = parse(path)
    return reader, {"argv": [*cmd, "<fresh-staging-directory>/vocab.gguf"], "bytes": path.stat().st_size, "sha256": ptq.digest(path),
                    "converter_sha256": ptq.digest(converter), "official_source": official["source"],
                    "converter_dependency_sha256": dependencies}


def directory(archive):
    import gguf
    import x151_build

    mapper = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, 28)
    rows = []
    for hf, (kind, shape, _) in archive.descriptors.items():
        name = mapper.get_name(hf, try_suffixes=(".weight", ".bias"))
        fmt = stock.native_format(kind) or "F32"
        if fmt == "BF16":
            ptq.need(len(shape) == 1 and "norm.weight" in name, f"Unexpected BF16 tensor {name}")
            fmt = "F32"
        elif kind == "X93-GGML-SYMBOL-ANS":
            _, index, *_ = x151_build.parent_codec.iq4.x93.codec.HEAD.unpack_from(archive.record(hf)[3])
            fmt = tuple(x151_build.parent_codec.iq4.x93.codec.ALPHABETS)[index]
        dtype = getattr(gguf.GGMLQuantizationType, fmt)
        block, stride = gguf.GGML_QUANT_SIZES[dtype]
        ptq.need(name and shape[-1] % block == 0, f"Unmapped HF tensor or unsupported alignment: {hf}")
        rows.append(dict(hf=hf, name=name, shape=shape, kind=kind, type=fmt, ggml_type=int(dtype),
                         bytes=math.prod(shape)//block*stride))
    names = {r["name"] for r in rows}
    ptq.need(len(names) == 310 and "output.weight" not in names and "token_embd.weight" in names
             and archive.meta["aliases"] == {"lm_head.weight":"model.embed_tokens.weight"}
             and Counter(r["type"] for r in rows) ==
             {"F32":234, "IQ1_S":2, "IQ2_XXS":5, "IQ3_XXS":26, "IQ4_NL":43},
             "New standard archive tensor inventory differs")
    return rows


def header(template, rows):
    field_end = max(f.offset+sum(p.nbytes for p in f.parts) for f in template.fields.values())
    ptq.need(field_end == len(template.data), "Unexpected vocab-only metadata tail")
    prefix = bytearray(template.data[:field_end])
    ptq.need(prefix[:4] == b"GGUF" and struct.unpack_from("<Q", prefix, 8) == (0,), "Metadata must be directory-free")
    struct.pack_into("<Q", prefix, 8, len(rows))
    offset = 0
    for row in rows:
        name, shape = row["name"].encode(), row["shape"]
        prefix += struct.pack("<Q",len(name))+name+struct.pack("<I",len(shape))
        prefix += struct.pack("<"+"Q"*len(shape), *shape[::-1])
        prefix += struct.pack("<IQ", row["ggml_type"], offset)
        row["offset"] = offset
        offset += stock.source.aligned(row["bytes"])
    prefix += bytes(-len(prefix) % 32)
    return prefix, offset


def emit(args, work):
    import gguf
    import numpy as np
    import torch
    import x151_build as history
    from x19b_model import field

    path = work/"XXS-stock.gguf"
    ptq.need(not path.exists(), "Refusing to replace an existing standard GGUF")
    archive = ptq.RecordView(ptq.check(work/"XXS.qpack", ptq.X189))
    reader, meta_proof = vocab_metadata(args.checkpoint, work)
    config = archive.meta["x87"]["config"]
    config_fields = {"embedding_length":"hidden_size", "block_count":"num_hidden_layers",
                     "feed_forward_length":"intermediate_size", "attention.head_count":"num_attention_heads",
                     "attention.head_count_kv":"num_key_value_heads", "attention.key_length":"head_dim",
                     "attention.value_length":"head_dim", "context_length":"max_position_embeddings",
                     "rope.freq_base":"rope_theta", "attention.layer_norm_rms_epsilon":"rms_norm_eps"}
    for name, key in config_fields.items():
        value = field(reader, "qwen3."+name)
        ptq.need(math.isclose(value, config[key], rel_tol=1e-6), f"Official metadata geometry differs: {name}")
    rows = directory(archive)
    prefix, charge = header(reader, rows)
    lib = stock.libdir()/"libggml-base.so"
    official = stock.read(stock.OUT/"official-cpu.json")
    ptq.need(official["source"] == stock.clean_commit() and
             official["library"][lib.name] == stock.source.describe(lib), "Vanilla CPU dequantizer differs")
    decode = stock.native_dequant(lib)
    codecs = type("Codecs", (), {})()
    codecs.ans = history.voc.entropy.Ans()
    codecs.old = history.parent_codec.codec.old.Ans(history.parent_codec.codec.old.LIB)
    codecs.native = history.rank.x87.x70.native.Native()
    previous, previous_path = stock.ck_fp32("cpu")
    torch.set_num_threads(2)
    with tempfile.NamedTemporaryFile(dir=work, prefix=".standard-", delete=False) as output, torch.inference_mode():
        temporary = Path(output.name)
        try:
            output.write(prefix)
            for row in rows:
                if row["kind"] == "bf16":
                    wire = archive.record(row["hf"])[3]
                    data = (np.frombuffer(wire, "<u2").astype("<u4") << 16).tobytes()
                    output.write(data)
                    raw = decoded = hashlib.sha256(data).hexdigest()
                else:
                    raw, decoded = stock.write_tensor(output, archive, row, codecs, torch, decode)
                ptq.need(decoded == previous["weights_fp32"][row["hf"]],
                         f"CPU logical value differs from historical X189: {row['hf']}")
                row.update(payload_sha256=raw, cpu_reference_fp32_sha256=decoded)
                output.write(bytes(stock.source.aligned(row["bytes"])-row["bytes"]))
            output.flush()
            os.fsync(output.fileno())
            ptq.need(output.tell() == len(prefix)+charge, "Standard GGUF write extent differs")
            image = gguf.GGUFReader(temporary)
            ptq.need(image.data_offset == len(prefix) and field(image,"general.architecture") == "qwen3",
                     "Independent GGUF header mismatch")
            stock.verify_image(temporary, [r | {"shape":list(r["shape"])} for r in rows], image, lib)
            ptq.need(all(n in image.fields and all(a.tobytes()==b.tobytes()
                      for a,b in zip(f.parts,image.fields[n].parts,strict=True))
                      for n,f in stock.tokenizer_fields(reader).items()), "Independent GGUF tokenizer changed")
            file_hash=ptq.digest(temporary)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"file": str(path), "sha256": file_hash, "bytes": path.stat().st_size, "verified_records": 310,
            "earlier_cpu_reference_sha256": ptq.digest(previous_path), "metadata": meta_proof, "emitter_sha256": ptq.digest(Path(__file__)),
            "fp32_sha256": {r["hf"]:r["cpu_reference_fp32_sha256"] for r in rows}}


def emit_checked(work):
    from types import SimpleNamespace

    source = ptq.read(work/"source-build/inputs.json")
    for name, expected in source["stamps"].items():
        ptq.need(ptq.stamp(Path(name)) == expected, "Original source changed after verified prepare")
    checkpoint = Path(source["paths"]["checkpoint"])
    expected = [ptq.stamp(checkpoint/n) for n in ("tokenizer.json", "tokenizer_config.json", "config.json")]
    proof = emit(SimpleNamespace(checkpoint=checkpoint), work)
    ptq.need(expected == [ptq.stamp(checkpoint/n) for n in ("tokenizer.json", "tokenizer_config.json", "config.json")],
             "Original configuration or tokenizer changed during GGUF metadata export")
    return proof


def qualify_checked(work, device):
    prior = ptq.read(work/"receipts/stock.json")
    proof = prior["proof"]
    ptq.need(prior["complete"] and proof["verified_records"] == 310 and
             proof["file"] == str(work/"XXS-stock.gguf"), "No independently verified fresh standard GGUF")
    ptq.STOCK = proof["sha256"]
    return qualify_original(work, device)


qualify_original = ptq.qualify
if __name__ == "__main__":
    ptq.emit_stock = emit_checked
    ptq.qualify = qualify_checked
    ptq.main()
