# Verification

The Haniwa X189 converter was tested on Linux x86-64 with Python 3.12, CUDA conversion, and llama.cpp b12487.

| BF16 checkpoint | Entry point | GGUF tensors | Output size (bytes) |
| --- | --- | ---: | ---: |
| SmolLM2-135M (Llama) | Desktop GUI | 272 | 517,035,680 |
| TinyMoE (Mixtral family) | CLI | 103 | 232,723,584 |

Both files passed tensor readback and contained `Haniwa X189` in `x189.recipe`. The SmolLM2 GGUF also generated text with the upstream llama.cpp CPU binary. For TinyMoE, the public FP32 checkpoint was cast to BF16 before conversion.

The GUI was tested with file dialogs, a Japanese output filename containing spaces, cancellation and retry, and an existing destination file. The CLI architecture listing was checked for Llama, Qwen2, and Mixtral.

[Conversion instructions](README.en.md) · [llama.cpp b12487](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)
