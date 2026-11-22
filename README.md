# Haniwa X189 BF16 to GGUF converter

English · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

Convert Hugging Face BF16 language models to standard GGUF with Haniwa X189 quantization. The desktop GUI and CLI use the same conversion pipeline. Model support depends on whether the chosen llama.cpp version can convert the architecture, collect importance statistics, and run inference. The output includes the tokenizer.

## Install

Use Linux x86-64 and Python 3.12. Allow enough disk space for an F32 copy of the model, intermediate tensors, and the finished GGUF. CUDA conversion requires an NVIDIA driver compatible with CUDA 12.8.

From the repository root, create an environment and install the dependencies.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --resume-retries 30 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r requirements.txt
```

## Convert in the desktop app

With the environment active, install the GUI dependency and launch the app.

```sh
python -m pip install -r requirements-gui.txt
python -m x189_clean gui
```

1. Select the BF16 model folder, a UTF-8 calibration text file, and a new GGUF output path.
2. In the llama.cpp settings, select its source folder and the Linux x64 CPU binaries from the same version. Choose CPU or CUDA for conversion. The app remembers these settings. [Tested llama.cpp version b12487](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)
3. Start the conversion. The app shows progress and makes the output folder available once the GGUF passes readback verification.

On X11, if Qt cannot load the xcb plugin, install the distribution's XCB cursor library. The package is `libxcb-cursor0` on Ubuntu and `xcb-util-cursor` on Arch Linux. [Qt requirements for X11](https://doc.qt.io/qt-6.10/linux-requirements.html)

## Convert from the CLI

Set `$BF16` to the checkpoint directory containing `config.json`, tokenizer files, and the BF16 weights. The converter reads sharded safetensors and supported PyTorch state dictionaries. Set `$CALIBRATION_TEXT` to the calibration text, `$WORK` to a writable directory, and `$OUTPUT` to a new GGUF path. `$LLAMA_SRC` points to the llama.cpp source, and `$LLAMA_CPU` points to the matching directory with `llama-imatrix`, `llama-tokenize`, and `libggml-base.so`.

```sh
python -m x189_clean list-models --llama-source "$LLAMA_SRC"

CUDA_VISIBLE_DEVICES=0 python -m x189_clean convert \
  --checkpoint "$BF16" --calibration-text "$CALIBRATION_TEXT" \
  --llama-source "$LLAMA_SRC" --llama-imatrix "$LLAMA_CPU/llama-imatrix" \
  --ggml-lib "$LLAMA_CPU/libggml-base.so" \
  --work "$WORK" --output "$OUTPUT" --device cuda:0
```

Use `--device cpu` for CPU conversion. Tune calibration with `--context` and `--chunks`.

Haniwa X189 uses diagonal second moments of input activations and applies eight-dimensional lattice quantization, standard IQ, low-rank correction, and vocabulary residuals based on tensor shape and role. The GGUF stores lattice reconstruction values as F32 and IQ blocks in IQ format. The file can be larger than the input BF16 weights. The GGUF identifies the format as `Haniwa X189` in `x189.recipe`. CLI runs write a per-tensor report named `<output-name>.x189.json` to the work directory.

## Run inference

```sh
"$LLAMA_CPU/llama-completion" -m "$OUTPUT" -ngl 0 -t 2 \
  -p 'The capital of France is' -n 3 --temp 0 --no-conversation
```

For GPU inference, use the matching CUDA build with `-ngl 99`. [Official llama.cpp binaries](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)

## Verification and license

The source is released under the [Apache License 2.0](LICENSE). External inputs and dependencies have their own licenses. [Verification results](VERIFICATION.md) · [Dependency notices](NOTICE)
