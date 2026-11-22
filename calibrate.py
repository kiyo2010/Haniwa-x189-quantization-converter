"""Offline reconstruction of the original Qwen3 IQ calibration moments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FILES = {
    "config.json": "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197",
    "model-00001-of-00002.safetensors": "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
    "model-00002-of-00002.safetensors": "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
    "model.safetensors.index.json": "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}
PARQUET_SHA = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
WIKI_SHA = "c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8"
TOKEN_SHA = "130d0cb7a2c73b23e9464c244468136ce82953df72b2379f4ead20ed73280534"
XXS_OWNER = "model.layers.17.mlp.down_proj"
XXS_FP32_SHA = "dc4488e2c969f995d99baa85bfbab1e0b7a4c0f75dab4ba9ade41ac1ff59aec5"


def require(test, reason):
    if not test:
        raise ValueError(reason)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def export_wiki(path):
    import pyarrow.parquet as pq

    require(sha(path) == PARQUET_SHA, "WikiText2 parquet identity differs")
    rows = pq.read_table(path, columns=["text"])["text"].to_pylist()
    data = "\n".join(rows).encode("utf-8")
    require(hashlib.sha256(data).hexdigest() == WIKI_SHA, "WikiText2 exported text fingerprint differs")
    return data


def token_hash(sequences):
    digest = hashlib.sha256()
    for ids in sequences:
        digest.update(np.asarray([len(ids), *ids], "<u4").tobytes())
    return digest.hexdigest()


def sequences(args, tokenizer):
    original = json.loads(args.original_calibration.read_bytes())
    require(set(original) == {"en", "ja", "es"} and all(
        isinstance(original[lang], list) and len(original[lang]) == 6 and
        all(isinstance(text, str) and text for text in original[lang]) for lang in original),
        "Supply only the original calibration texts as six strings per language")
    if args.wiki_parquet:
        data = export_wiki(args.wiki_parquet)
    else:
        require(sha(args.wiki_text) == WIKI_SHA, "WikiText2 exported text fingerprint differs")
        data = args.wiki_text.read_bytes()
    candidates = [(i, line) for i, line in enumerate(data.decode("utf-8").split("\n"))
                  if len(line.strip()) >= 768]
    rows, texts = zip(*(candidates[i] for i in np.linspace(0, len(candidates)-1, 8, dtype=int)), strict=True)
    require(rows == (14, 8904, 17309, 25442, 34207, 43567, 51799, 60480), "WikiText sequence positions differ")
    docs = [text for lang in ("en", "ja", "es") for text in original[lang]]
    result = [tokenizer.encode(text, add_special_tokens=False, verbose=False)[:256]
              for text in [*docs, *texts]]
    require(len(result) == 26 and sum(map(len, result)) == 2758 and token_hash(result) == TOKEN_SHA,
            "Original calibration sequence identity differs")
    return result


def capture(model, ids, scratch, torch):
    owners, handles, outputs = {}, [], {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        owner = "model.embed_tokens.weight" if name == "lm_head" else name + ".weight"
        group = name.replace(".k_proj", ".q_proj").replace(".v_proj", ".q_proj").replace(".up_proj", ".gate_proj")
        owners[owner] = group
        if group != name:
            continue
        saved = np.lib.format.open_memmap(scratch/(name+".npy"), "w+", np.float32, (2758, module.in_features))
        outputs[name] = [saved, 0]

        def prehook(current, args, key=name):
            buf, offset = outputs[key]
            x = args[0].detach().float().reshape(-1, current.in_features).cpu().contiguous().numpy()
            buf[offset:offset+len(x)] = x
            outputs[key][1] += len(x)

        handles.append(module.register_forward_pre_hook(prehook))
    require(len(owners) == 197 and len(outputs) == 113, "Unexpected Qwen3 owner layout")
    try:
        with torch.inference_mode():
            for seq in ids:
                model(torch.tensor([seq], device="cuda:0"), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def moments(outputs, ids, torch):
    even = np.repeat(np.arange(len(ids)), list(map(len, ids))) % 2 == 0
    arrays = {}
    for name, (saved, count) in outputs.items():
        require(count == len(even), "Incomplete calibration positions")
        all_values = torch.from_numpy(saved).double().square().mean(0).numpy()
        require(np.isfinite(all_values).all() and (all_values >= 0).all(), "Nonfinite moments")
        diagonal = np.empty(saved.shape[1], "<f4")
        for first in range(0, saved.shape[1], 128):
            values = saved[even, first:first+128].astype(np.float64)
            diagonal[first:first+values.shape[1]] = np.square(values).mean(axis=0)
        arrays["all:"+name] = all_values
        arrays["even:"+name] = diagonal
    cast = arrays["all:"+XXS_OWNER].astype("<f4")
    require(hashlib.sha256(cast.tobytes()).hexdigest() == XXS_FP32_SHA, "Original L17 importance differs")
    return arrays


def run(args):
    require(not args.output.exists(), "Refusing to replace existing calibration output")
    require(args.checkpoint.is_dir(), "Original Qwen3 BF16 checkpoint directory is missing")
    require(args.original_calibration.is_file(),
            "Original authored calibration JSON is missing: requires the exact 18 authorized texts")
    require((args.wiki_parquet or args.wiki_text).is_file(), "Pinned WikiText2 training input is missing")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for name, expected in FILES.items():
        require(sha(args.checkpoint/name) == expected, f"Pinned Qwen3 file differs: {name}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    ids = sequences(args, tokenizer)
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090"
            and str(torch.__version__) == "2.10.0+cu128", "This exact capture requires the original RTX5090 and Torch build")
    torch.manual_seed(17)
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".capture-") as directory:
        scratch = Path(directory)
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.float32,
                     attn_implementation="eager", local_files_only=True).to("cuda:0").eval()
        outputs = capture(model, ids, scratch, torch)
        del model
        arrays = moments(outputs, ids, torch)
        staged = scratch/"moments.npz"
        with staged.open("wb") as output:
            np.savez(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        os.link(staged, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--original-calibration", type=Path, required=True)
    wiki = parser.add_mutually_exclusive_group(required=True)
    wiki.add_argument("--wiki-parquet", type=Path)
    wiki.add_argument("--wiki-text", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    try:
        run(parser.parse_args())
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
