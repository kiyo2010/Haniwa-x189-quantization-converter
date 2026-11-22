#!/usr/bin/env python3
"""Rebuild the fixed X189-XXS recipe from Qwen3-1.7B BF16."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT/"code"), str(ROOT/"private-x189-xxs-stock-gguf"), str(ROOT/"private-x189-github-prep")]

import x190_chain as chain

TARGET = "model.layers.17.mlp.down_proj.weight"
SHAPE = (2048, 6144)
X123 = "8aaa27c1b2238721746cb0e00a5ea5371e3f104f53ea8abb1254ec2bd10fcd1d"
X175 = "b44a66aa8bb35ac18b7613edafe104bc904d9fe2fa3c1a7ab15429195faad8f5"
X189 = "bfbec47794ba561cdcf6b42f172d7579fd82daea97160294e13ba944d303bf06"
XXS = "3984e89b5503c5b8dc958aaa2cb67d73f9621120e3a862f80f31cc35fe25a260"
INDEX = "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d"
LIB = "2c6a9014d030948a0c29ed633d8115410fd1350b374e42a8b06c9f9246a096d4"
CAL = "bf98f3e911ace8f09bd9b6570f64cb90ad41258fd976a5eebfa33eaebec91306"
DIAG = "c1613dc3b5fef01cfeaac59c0f7056835d7ce86a26212992c198c77540199347"
TEMPLATE_HEADER = "6bd0b33194c092b2b454be792a22abb73db0489ff61260f431db690a3e872c76"
STOCK = "342c4970676ca59d3801216ea92603f18a098f64b11f204e456749f761edd64d"
PHASES = ("prepare", *(f"capture-{i:02}" for i in range(5)),
          *(f"xe82-{i:02}" for i in range(11)), "records", "assemble", "encode", "qualify",
          "factor-00", "factor-20", "seal", "entropy", "xxs", "combine", "stock", "cpu", "cuda")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(path, expected):
    need(path.is_file() and digest(path) == expected, f"Unexpected input or checkpoint SHA: {path}")
    return path


def read(path):
    return json.loads(path.read_bytes())


def publish(path, value):
    chain.write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save(path, buffer):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(buffer)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def launch(script, *args):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1")
    for option in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        env.pop(option, None)
    subprocess.run([sys.executable, "-I", "-B", str(script), *map(str, args)], check=True, env=env)


def gpu_options(stage):
    flag = ("--gpu-capture" if stage.startswith("capture-") else
            "--gpu-encode" if stage.startswith(("xe82-", "factor-")) or stage == "records" else None)
    if flag is None:
        return []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    need(visible and "," not in visible, f"{stage} needs exactly one managed CUDA device")
    return [flag, visible]


def stamp(path):
    info = path.stat()
    return {"bytes": info.st_size, "inode": info.st_ino, "device": info.st_dev,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def weight_and_diagonal(args, work):
    import numpy as np
    from safetensors import safe_open

    source = read(check(args.calibration, CAL))
    group = source["owners"][TARGET]
    with np.load(check(args.diagonals, DIAG), allow_pickle=False) as saved:
        diagonal = np.ascontiguousarray(saved[group], dtype=np.float32)
    index = read(check(args.checkpoint/"model.safetensors.index.json", INDEX))
    basename = index["weight_map"][TARGET]
    need(basename in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"), "Unexpected shard")
    path = args.checkpoint/basename
    source_pin = read(work/"source-build/inputs.json")["stamps"][str(path)]
    need(stamp(path) == source_pin, "Original L17 source shard differs from verified prepare")
    with safe_open(path, framework="pt", device="cpu") as shard:
        matrix = shard.get_tensor(TARGET)
        need(str(matrix.dtype) == "torch.bfloat16", "Expected original BF16")
        weights = matrix.float().numpy()
    need(stamp(path) == source_pin and weights.shape == SHAPE and diagonal.shape == (SHAPE[1],)
         and np.isfinite(diagonal).all() and (diagonal >= 0).all(), "X189 L17 input changed")
    return weights, diagonal, group


def encode_xxs(args, work):
    import x180_native as native
    import x189_recipe as portable

    weights, diagonal, group = weight_and_diagonal(args, work)
    packed = portable.native_iq2(weights, diagonal, check(args.ggml, LIB))
    need(hashlib.sha256(packed).hexdigest() == portable.L17_BODY, "Native L17 IQ2_XXS payload differs")
    wire = native.pack("XXS", SHAPE, packed)
    need(hashlib.sha256(wire).hexdigest() == XXS, "Rebuilt native L17 XXS differs from historical payload SHA")
    path = work/"L17-XXS.bin"
    if path.exists():
        check(path, XXS)
    else:
        save(path, wire)
    return {"sha256": XXS, "bytes": len(wire), "group": group,
            "fp32_diagonal_sha256": hashlib.sha256(diagonal.tobytes()).hexdigest(),
            "source_fp32_sha256": hashlib.sha256(weights.tobytes()).hexdigest()}


def combine(work):
    import x104_transport as io
    import x180_native as native

    carrier = check(work/"best.qpack", X175)
    archive = native.store().Archive(carrier)
    wire = check(work/"L17-XXS.bin", XXS).read_bytes()

    def records():
        replaced = 0
        for name, kind, shape, raw in archive.rows():
            if name == TARGET:
                need(shape == SHAPE and kind == "X53-XE82-EC", "Unexpected X175 target record")
                replaced += 1
                yield name, native.KIND, shape, wire
            else:
                yield name, kind, shape, raw
        need(replaced == 1, "Missing unique X189 replacement")

    def validate(path):
        result = native.store().Archive(check(path, X189))
        need(result.meta == archive.meta and len(result.names) == 310, "Changed X189 archive geometry")
        for expected, actual in zip(records(), result.rows(), strict=True):
            need(expected == actual, f"Physical X189 record mismatch: {actual[0]}")

    path = work/"XXS.qpack"
    io.create(path, archive.meta, records, validate)
    # Historical QPACK is consulted only after the new archive has passed its own audits.
    reference = check(ROOT/"x189-qwen-layer17-retest/XXS.qpack", X189)
    need(not path.samefile(reference), "Rebuilt file aliases historical reference")
    old = native.store().Archive(reference)
    rebuilt = native.store().Archive(path)
    need(old.meta == rebuilt.meta, "Historical X189 manifest differs")
    compared = 0
    for a, b in zip(old.rows(), rebuilt.rows(), strict=True):
        need(a == b, f"Historical X189 physical record differs: {a[0]}")
        compared += 1
    need(compared == 310, "Historical X189 comparison is incomplete")
    return {"sha256": X189, "bytes": path.stat().st_size, "verified_records": compared,
            "unchanged_from_fresh_x175": 309, "historical_reference_all_records_exact": True}


class RecordView:
    """Expose fresh RQPK directory and records to the internal standard writer."""
    def __init__(self, path):
        import x180_native as native
        self.store = native.store()
        self.archive = self.store.Archive(path)
        self.names = self.archive.names
        self.meta = self.archive.meta
        self.descriptors = {}
        for name, kind, shape, wire in self.archive.rows():
            self.descriptors[name] = (kind, shape, len(wire))

    def record(self, name):
        offset, length = self.archive.entries[name]
        with self.archive.path.open("rb") as stream:
            stream.seek(offset)
            kind, shape, data = self.store.extract(self.store.read(stream, length))
        return name, kind, shape, data


def emit_stock(work):
    import gguf
    import torch
    import stock_x189 as stock
    import x151_build as history

    path = work/"XXS-stock.gguf"
    if path.exists():
        check(path, STOCK)
        return {"sha256": STOCK, "bytes": path.stat().st_size, "verified_records": 310}
    archive = RecordView(check(work/"XXS.qpack", X189))
    # Only this verified prefix is used. None of the template tensor bodies is read.
    template = gguf.GGUFReader(stock.TEMPLATE)
    need(template.data_offset == 5_952_064 and
         hashlib.sha256(memoryview(template.data[:template.data_offset])).hexdigest() == TEMPLATE_HEADER,
         "Pinned standard container metadata template differs")
    config = archive.meta["x87"]["config"]
    rows = stock.prepare_rows(archive, template)
    for row in rows:
        if row["type"] == "BF16":
            row.update(type="F32", ggml_type=0, bytes=2*row["bytes"])
    prefix, charge = stock.header(template, rows)
    lib = stock.libdir()/"libggml-base.so"
    official = stock.read(stock.OUT/"official-cpu.json")
    need(official["source"] == stock.clean_commit() and
         official["library"][lib.name] == stock.source.describe(lib), "Pinned vanilla CPU dequantizer differs")
    decoder = stock.native_dequant(lib)
    codecs = type("Codecs", (), {})()
    codecs.ans = history.voc.entropy.Ans()
    codecs.old = history.parent_codec.codec.old.Ans(history.parent_codec.codec.old.LIB)
    codecs.native = history.rank.x87.x70.native.Native()
    torch.set_num_threads(2)
    with tempfile.NamedTemporaryFile(dir=work, prefix=".stock-", delete=False) as stream, torch.inference_mode():
        temporary = Path(stream.name)
        try:
            stream.write(prefix)
            for row in rows:
                if row["kind"] == "bf16":
                    import numpy as np
                    wire = archive.record(row["hf"])[3]
                    data = (np.frombuffer(wire, "<u2").astype("<u4") << 16).tobytes()
                    stream.write(data)
                    raw = decoded = hashlib.sha256(data).hexdigest()
                else:
                    raw, decoded = stock.write_tensor(stream, archive, row, codecs, torch, decoder)
                row.update(payload_sha256=raw, cpu_reference_fp32_sha256=decoded)
                stream.write(bytes(stock.source.aligned(row["bytes"])-row["bytes"]))
            stream.flush()
            os.fsync(stream.fileno())
            need(stream.tell() == len(prefix)+charge and len(rows) == 310, "Unexpected GGUF extent")
            stock.verify_image(temporary, rows, template, lib)
            check(temporary, STOCK)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return {"sha256": STOCK, "bytes": path.stat().st_size, "verified_records": 310, "config": config, "template_header_sha256": TEMPLATE_HEADER,
            "fp32_sha256": {row["hf"]: row["cpu_reference_fp32_sha256"] for row in rows}}


def qualify(work, device):
    import numpy as np
    import stock_x189 as stock
    import x212_runtime

    start = time.monotonic()
    model = check(work/"XXS-stock.gguf", STOCK)
    official_path = stock.OUT/f"official-{device}.json"
    official = read(official_path)
    probe = stock.OUT/f"official-probe-{device}"
    libs = stock.libdir(device)
    need(official["source"] == stock.clean_commit() and official["probe"] == stock.source.describe(probe)
         and all(v == stock.source.describe(Path(v["file"])) for v in official["library"].values()),
         "Pinned unmodified runtime differs")
    env, deps = x212_runtime.environment(device, {"libraries": str(libs)})
    previous = read(stock.OUT/f"qualify-{device}.json")["input_and_short_generation"]
    log = work/f"native-{device}.log"
    with tempfile.TemporaryDirectory(dir=work, prefix=".logits-") as directory, log.open("w") as error:
        basename = Path(directory)/"step"
        stdout = subprocess.check_output([str(probe), str(model), "99" if device == "cuda" else "0",
                                          str(basename), str(libs)], stderr=error, env=env, text=True)
        traces = {}
        for line in stdout.splitlines():
            label, i, *ids = line.split()
            traces[label, int(i)] = list(map(int, ids))
        results = {}
        for i, lang in enumerate(("en", "ja")):
            expected = previous[lang]
            ids, tokens = traces["prompt", i], traces["greedy", i]
            data = np.fromfile(f"{basename}-{i}.f32", dtype="<f4")
            ends = np.cumsum([(len(ids)+step)*stock.V for step in range(3)])
            need(data.size == ends[-1] and np.isfinite(data).all(), "Native generation logits missing or nonfinite")
            hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in np.split(data, ends[:-1])]
            need(ids == expected["ids"] and tokens == expected["greedy_ids"]
                 and hashes == [step["sha256"] for step in expected["logits"]], "Prior same-device stock trace differs")
            results[lang] = {"ids": ids, "greedy_ids": tokens, "logits_sha256": hashes}
    if device == "cuda":
        need("NVIDIA GeForce RTX 5090" in log.read_text() and "offloaded 29/29 layers" in log.read_text(),
             "Native run did not prove full RTX5090 offload")
    return {"device": device, "archive_sha256": STOCK, "runtime_sha256": digest(official_path), "prompts": results,
            "dependencies": deps, "log_sha256": digest(log), "seconds": time.monotonic()-start,
            "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}


def require_completed(job):
    events = ROOT/"logs/memory-managed/adaptive-events.jsonl"
    with events.open() as source:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (item.get("id"), item.get("event"), item.get("exit"), item.get("target_verified")) == (
                    job, "finish", 0, True):
                return
    raise ValueError(f"Managed predecessor {job} lacks a successful exit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=PHASES)
    parser.add_argument("--after-managed")
    for option in ("checkpoint", "kftt", "calibration", "diagonals", "ggml", "gguf-python-path", "work"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    need(not args.work.is_symlink(), "Work directory cannot be a symlink")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    work = args.work
    need(not any(path.is_relative_to(work) for path in (args.checkpoint, args.kftt, args.calibration,
                                                        args.diagonals, args.ggml)), "Inputs must be outside work")
    need(work == HERE or HERE in work.parents, f"Use a new work directory under {HERE}")
    need(work != HERE, "Work directory must be distinct from source")
    work.mkdir(parents=True, exist_ok=True)
    import x104_transport as io
    import stock_x189 as stock

    stock.source.managed()
    if args.after_managed:
        require_completed(args.after_managed)
    with io.lock(work/"rebuild.lock"):
        stage = args.stage
        start = time.monotonic()
        index = PHASES.index(stage)
        receipts = work/"receipts"
        receipts.mkdir(exist_ok=True)
        identity = {key: str(value) for key, value in vars(args).items() if key not in ("stage", "after_managed")}
        identity.update(driver=digest(Path(__file__)), x153=chain.X153_SHA, x190=chain.R2_SHA,
                        portable_iq_encoder=digest(ROOT/"private-x189-github-prep/x189_recipe.py"),
                        portable_iq_calibration=digest(ROOT/"private-x189-github-prep/calibrate.py"))
        if index:
            previous = read(receipts/f"{PHASES[index-1]}.json")
            need(previous["complete"] and previous["identity"] == identity, "Resume identity differs")
        target = receipts/f"{stage}.json"
        if target.exists():
            current = read(target)
            need(current["complete"] and current["identity"] == identity, "Existing checkpoint identity differs")
            return
        kits = chain.release()
        x190 = chain.OUT
        x153 = ROOT/"x153-product-workflow"
        if index <= PHASES.index("seal"):
            if stage == "prepare":
                check(args.ggml, LIB)
                check(args.calibration, CAL)
                check(args.diagonals, DIAG)
            check(x153/"x153-source.tar.gz", chain.X153_SHA)
            need(digest(x153/"x153.py") == kits["x153_manifest"]["x153.py"]["sha256"], "Pinned X153 launcher differs")
            launch(x153/"x153.py", "build", "--work", work/"source-build", "--checkpoint", args.checkpoint,
                   "--kftt", args.kftt, "--ggml", args.ggml, "--gguf-python-path", args.gguf_python_path,
                   "--stage", stage, *gpu_options(stage))
            proof = read(work/"source-build/receipts"/f"{stage}.json")
            need(proof["stage"] == stage and proof["pass"] is True, "Pinned source stage is incomplete")
        elif stage == "entropy":
            check(work/"source-build/output/marginal.qpack", X123)
            need(digest(x190/"x190.py") == kits["x190_manifest"]["x190.py"]["sha256"], "Pinned X190 launcher differs")
            launch(x190/"x190.py", "from-parent", "--work", work/"entropy-build", "--parent",
                   work/"source-build/output/marginal.qpack", "--rqpk", work/"best.qpack",
                   "--gguf", work/"best-private.gguf", "--gguf-python-path", args.gguf_python_path,
                   "--receipt", work/"entropy.json")
            proof = {"sha256": digest(check(work/"best.qpack", X175)), "receipt_sha256": digest(work/"entropy.json")}
        elif stage == "xxs":
            proof = encode_xxs(args, work)
        elif stage == "combine":
            proof = combine(work)
        elif stage == "stock":
            proof = emit_stock(work)
        else:
            proof = qualify(work, stage)
        publish(target, {"complete": True, "stage": stage, "identity": identity, "proof": proof,
                         "wall_seconds": time.monotonic()-start,
                         "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss})


if __name__ == "__main__":
    main()
