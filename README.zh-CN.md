# Haniwa X189 通用 BF16 转 GGUF 转换器

[English](README.md) · [日本語](README.ja.md) · 简体中文

使用 Haniwa X189 量化将 Hugging Face 的 BF16 语言模型转换为标准 GGUF。桌面界面和命令行使用同一套转换流程。模型支持范围取决于所用的 llama.cpp 版本能否转换该架构、采集重要性统计数据并运行推理。输出文件包含分词器。

## 安装

使用 Linux x86-64 和 Python 3.12。请预留足够的磁盘空间，用于保存模型的 F32 副本、中间张量和最终的 GGUF。使用 CUDA 转换时，需要支持 CUDA 12.8 的 NVIDIA 驱动程序。

在仓库根目录创建虚拟环境并安装依赖。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --resume-retries 30 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r requirements.txt
```

## 使用桌面界面转换

激活虚拟环境后，安装图形界面依赖并启动程序。

```sh
python -m pip install -r requirements-gui.txt
python -m x189_clean gui
```

1. 选择 BF16 模型目录、UTF-8 校准文本文件和新的 GGUF 输出路径。
2. 在 llama.cpp 设置中选择源码目录，以及同版本的 Linux x64 CPU 二进制文件目录。转换设备可选 CPU 或 CUDA，程序会保存这些设置。 [已验证的 llama.cpp b12487](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)
3. 开始转换。程序会显示进度，并在 GGUF 通过回读验证后提供打开输出目录的按钮。

在 X11 下，如果 Qt 无法加载 xcb 插件，请安装发行版提供的 XCB 光标库。Ubuntu 中的包名为 `libxcb-cursor0`，Arch Linux 中为 `xcb-util-cursor`。 [Qt 的 X11 要求](https://doc.qt.io/qt-6.10/linux-requirements.html)

## 使用命令行转换

将 `$BF16` 设为包含 `config.json`、分词器文件和 BF16 权重的模型目录。转换器支持分片 safetensors 和兼容的 PyTorch state dictionary。将 `$CALIBRATION_TEXT` 设为校准文本路径，`$WORK` 设为可写的工作目录，`$OUTPUT` 设为新的 GGUF 路径。`$LLAMA_SRC` 指向 llama.cpp 源码目录，`$LLAMA_CPU` 指向同版本的二进制文件目录，其中包含 `llama-imatrix`、`llama-tokenize` 和 `libggml-base.so`。

```sh
python -m x189_clean list-models --llama-source "$LLAMA_SRC"

CUDA_VISIBLE_DEVICES=0 python -m x189_clean convert \
  --checkpoint "$BF16" --calibration-text "$CALIBRATION_TEXT" \
  --llama-source "$LLAMA_SRC" --llama-imatrix "$LLAMA_CPU/llama-imatrix" \
  --ggml-lib "$LLAMA_CPU/libggml-base.so" \
  --work "$WORK" --output "$OUTPUT" --device cuda:0
```

使用 `--device cpu` 可在 CPU 上转换。可使用 `--context` 和 `--chunks` 调整校准范围。

Haniwa X189 使用输入激活的对角二阶矩，并根据张量形状和用途应用八维格量化、标准 IQ、低秩校正和词表残差。GGUF 以 F32 保存格量化后的重建值，并以 IQ 格式保存 IQ 块。输出文件可能比原始 BF16 权重更大。GGUF 的 `x189.recipe` 字段记录 `Haniwa X189`。命令行转换会在工作目录中生成逐张量报告 `<输出文件名>.x189.json`。

## 运行推理

```sh
"$LLAMA_CPU/llama-completion" -m "$OUTPUT" -ngl 0 -t 2 \
  -p 'The capital of France is' -n 3 --temp 0 --no-conversation
```

在 GPU 上推理时，请使用对应的 CUDA 版本并指定 `-ngl 99`。 [llama.cpp 官方二进制文件](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)

## 验证和许可证

本仓库源码使用 [Apache License 2.0](LICENSE)。外部输入和依赖项适用各自的许可证。 [验证结果](VERIFICATION.md) · [依赖项声明](NOTICE)
