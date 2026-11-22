# Haniwa X189 general-purpose BF16 to GGUF converter

English · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

Convert Hugging Face BF16 language models to standard GGUF with Haniwa X189 quantization. The desktop GUI and CLI use the same conversion pipeline. Model support depends on whether the chosen llama.cpp version can convert the architecture, collect importance statistics, and run inference. The output includes the tokenizer.

## Install

Use Python 3.12 on Linux x86-64. Open a terminal and run the block below. It creates `x189-quantization-converter` in your home directory and installs the tools.

```sh
cd "$HOME"
git clone https://github.com/kiyo2010/Haniwa-x189-quantization-converter.git x189-quantization-converter
cd x189-quantization-converter
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r requirements.txt -r requirements-gui.txt
mkdir -p vendor/llama-source vendor/llama-bin
curl -fLsS https://github.com/ggml-org/llama.cpp/archive/refs/tags/b12487.tar.gz \
  | tar -xz --strip-components=1 -C vendor/llama-source
curl -fLsS https://github.com/ggml-org/llama.cpp/releases/download/b12487/llama-b12487-bin-ubuntu-x64.tar.gz \
  | tar -xz --strip-components=1 -C vendor/llama-bin
```

## Run

**Open the GUI from the terminal**

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
LLAMA_SRC="$PWD/vendor/llama-source" LLAMA_CPU="$PWD/vendor/llama-bin" \
  python -m x189_clean gui
```

In the window that opens, choose the BF16 model folder and the GGUF output path, then click **変換を開始**. Calibration defaults to **汎用校正テキスト**, the included 16-language sample. You can switch to **日英サンプル** for Japanese and English or click **選択…** to use your own UTF-8 text.

**Convert in the terminal**

Replace `$HOME/models/my-model` with your model folder, then run the block below in the terminal. The output is `haniwa-x189.gguf` in your home directory.

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
python -m x189_clean convert \
  --checkpoint "$HOME/models/my-model" \
  --calibration-text "x189_clean/data/calibration-general.txt" \
  --llama-source "vendor/llama-source" \
  --llama-imatrix "vendor/llama-bin/llama-imatrix" \
  --ggml-lib "vendor/llama-bin/libggml-base.so" \
  --work "./work" --output "$HOME/haniwa-x189.gguf" --device cpu
```
