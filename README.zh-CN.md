# Haniwa X189 通用 BF16 转 GGUF 转换器

[English](README.md) · [日本語](README.ja.md) · 简体中文

使用 Haniwa X189 量化将 Hugging Face 的 BF16 语言模型转换为标准 GGUF。桌面界面和命令行使用同一套转换流程。模型支持范围取决于所用的 llama.cpp 版本能否转换该架构、采集重要性统计数据并运行推理。输出文件包含分词器。

## 安装

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

## 运行命令

**在终端中打开图形界面**

```sh
cd "$HOME/x189-quantization-converter"
. .venv/bin/activate
LLAMA_SRC="$PWD/vendor/llama-source" LLAMA_CPU="$PWD/vendor/llama-bin" \
  python -m x189_clean gui
```

窗口打开后，选择 BF16 模型目录和 GGUF 保存位置，然后点击 **変換を開始**。程序默认使用内置的 **汎用校正テキスト**（16 种语言）。可以切换到 **日英サンプル** 使用日英双语样本，也可以点击 **選択…** 选用自己的 UTF-8 文本。

**在终端中转换**

将下面的 `$HOME/models/my-model` 替换为实际的模型目录，然后在终端运行。输出文件为主目录下的 `haniwa-x189.gguf`。

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
