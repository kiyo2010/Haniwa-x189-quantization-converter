# Haniwa X189 汎用BF16 → GGUF変換ソフト

[English](README.md) · 日本語 · [简体中文](README.zh-CN.md)

Hugging FaceのBF16言語モデルをHaniwa X189で量子化し、標準GGUFに変換するソフトです。GUIとCLIで同じ汎用レシピを実行できます。対応アーキテクチャは使用するllama.cppの変換器と推論ランタイムで決まります。生成したGGUFはトークナイザーを内蔵します。

## インストール

Linux（x86-64）で Python 3.12 を用意します。端末を開き、下のコマンドをまとめて実行します。ホームフォルダに `x189-quantization-converter` が作られ、必要なツールが入ります。

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

## コマンド実行方法

**GUIを開く（端末で実行）**

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
LLAMA_SRC="$PWD/vendor/llama-source" LLAMA_CPU="$PWD/vendor/llama-bin" \
  python -m x189_clean gui
```

開いた画面でBF16モデルのフォルダとGGUFの保存先を選び、**変換を開始**を押します。校正には同梱の **汎用校正テキスト**（16言語）が使われます。同じ欄で **日英サンプル** に切り替えたり、**選択…** から自分のUTF-8テキストを指定したりできます。

**端末だけで変換する**

下の `$HOME/models/my-model` を自分のモデルフォルダに書き換え、端末で実行します。出力はホームフォルダの `haniwa-x189.gguf` です。

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
