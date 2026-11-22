# Haniwa X189

[English](#english) · [日本語](#日本語) · [简体中文](#简体中文)

## English

Convert Hugging Face BF16 language models to standard GGUF with Haniwa X189 quantization. The desktop GUI and CLI use the same conversion pipeline. Model support depends on whether the chosen llama.cpp version can convert the architecture, collect importance statistics, and run inference. The output includes the tokenizer.

### Install

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

### Run

**Open the GUI from the terminal**

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
LLAMA_SRC="$PWD/vendor/llama-source" LLAMA_CPU="$PWD/vendor/llama-bin" \
  python -m x189_clean gui
```

In the window that opens, choose the BF16 model folder, a UTF-8 calibration text file, and the GGUF output path. Click **変換を開始** to start.

**Convert in the terminal**

Replace `$HOME/models/my-model` and `$HOME/calibration.txt` with your model folder and calibration file, then run the block below in the terminal. The output is `haniwa-x189.gguf` in your home directory.

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
python -m x189_clean convert \
  --checkpoint "$HOME/models/my-model" \
  --calibration-text "$HOME/calibration.txt" \
  --llama-source "vendor/llama-source" \
  --llama-imatrix "vendor/llama-bin/llama-imatrix" \
  --ggml-lib "vendor/llama-bin/libggml-base.so" \
  --work "./work" --output "$HOME/haniwa-x189.gguf" --device cpu
```

---

## 日本語

Hugging FaceのBF16言語モデルをHaniwa X189で量子化し、標準GGUFに変換するソフトです。GUIとCLIで同じ汎用レシピを実行できます。対応アーキテクチャは使用するllama.cppの変換器と推論ランタイムで決まります。生成したGGUFはトークナイザーを内蔵します。

### インストール

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

### コマンド実行方法

**GUIを開く（端末で実行）**

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
LLAMA_SRC="$PWD/vendor/llama-source" LLAMA_CPU="$PWD/vendor/llama-bin" \
  python -m x189_clean gui
```

開いた画面でBF16モデルのフォルダ、校正用のUTF-8テキスト、GGUFの保存先を選び、**変換を開始**を押します。

**端末だけで変換する**

下の `$HOME/models/my-model` と `$HOME/calibration.txt` を自分のモデルフォルダと校正ファイルの場所に書き換え、端末で実行します。出力はホームフォルダの `haniwa-x189.gguf` です。

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
python -m x189_clean convert \
  --checkpoint "$HOME/models/my-model" \
  --calibration-text "$HOME/calibration.txt" \
  --llama-source "vendor/llama-source" \
  --llama-imatrix "vendor/llama-bin/llama-imatrix" \
  --ggml-lib "vendor/llama-bin/libggml-base.so" \
  --work "./work" --output "$HOME/haniwa-x189.gguf" --device cpu
```

---

## 简体中文

使用 Haniwa X189 量化将 Hugging Face 的 BF16 语言模型转换为标准 GGUF。桌面界面和命令行使用同一套转换流程。模型支持范围取决于所用的 llama.cpp 版本能否转换该架构、采集重要性统计数据并运行推理。输出文件包含分词器。

### 安装

在 Linux（x86-64）上准备好 Python 3.12。打开终端并运行以下整段命令。命令会在主目录下创建 `x189-quantization-converter` 并安装所需工具。

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

### 运行命令

**在终端中打开图形界面**

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
LLAMA_SRC="$PWD/vendor/llama-source" LLAMA_CPU="$PWD/vendor/llama-bin" \
  python -m x189_clean gui
```

窗口打开后，选择 BF16 模型目录、UTF-8 校准文本文件和 GGUF 保存位置，然后点击 **変換を開始** 开始转换。

**在终端中转换**

将下面的 `$HOME/models/my-model` 和 `$HOME/calibration.txt` 替换为实际的模型目录和校准文件，然后在终端运行。输出文件为主目录下的 `haniwa-x189.gguf`。

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
python -m x189_clean convert \
  --checkpoint "$HOME/models/my-model" \
  --calibration-text "$HOME/calibration.txt" \
  --llama-source "vendor/llama-source" \
  --llama-imatrix "vendor/llama-bin/llama-imatrix" \
  --ggml-lib "vendor/llama-bin/libggml-base.so" \
  --work "./work" --output "$HOME/haniwa-x189.gguf" --device cpu
```
