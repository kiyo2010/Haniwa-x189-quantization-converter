# 動作確認

2026年11月23日 JST。Linux x86-64、Python 3.12、PyTorch 2.10.0+cu128で、公式Qwen3-1.7B BF16から標準GGUFへの変換を確認しました。

| 項目 | 結果 |
|---|---|
| 校正 | KFTT 399対、32,744トークン |
| 標準GGUF | 5,493,071,744 B |
| 読み戻し | 310テンソルの名称・型・形状・保存バイトを照合 |
| 型の内訳 | F32 234、IQ1_S 2、IQ2_XXS 5、IQ3_XXS 26、IQ4_NL 43 |
| CPU・GPU | 無改造llama.cpp b12487で英語・日本語の出力が一致。RTX 5090で29/29層をオフロード |

実測GGUFのSHA-256は `330a0f7117ac3b6af9c5cd75cacc4764e103a849625ceac04849176edb02d844` です。

[変換手順](README.md) · [公式llama.cpp b12487](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)
