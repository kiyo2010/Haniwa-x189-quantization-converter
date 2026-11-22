# X189-KFTT-v2

公式Qwen3-1.7BのBF16チェックポイントを、公開KFTTの英日訓練データで校正し、8次元格子コードと標準IQを使って標準Qwen3 GGUFに変換するCLIです。校正データと元モデルは利用者が取得します。生成したGGUFはトークナイザーを内蔵し、無改造のllama.cppで読み込めます。

## インストール

Linux x86-64とPython 3.12を使用します。入力と依存関係に加えて、作業用ディスクに25 GB以上の空きを用意してください。CUDAで変換する場合はCUDA 12.8対応のNVIDIAドライバーが必要です。

リポジトリのルートで仮想環境を作ります。次の例はCUDA用の依存関係をインストールします。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --resume-retries 30 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r requirements.txt
```

## 公式の入力を用意する

1. Qwen3-1.7Bの固定リビジョンを取得します。下の例は、仮想環境に導入済みのHugging Face CLIを使用します。[公式スナップショット](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)
2. KFTT Data v1.0をアーカイブのまま取得します。SHA-256は `fcfcaa670d6d59aa691b0e909c0d7c393852dd2fb1d6310fda9b3282dc6d1638` です。[公式アーカイブ](https://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz)
3. llama.cpp b12487のソースとLinux x64 CPU版を展開します。ソースの `convert_hf_to_gguf.py` と `gguf-py`、CPU版の `libggml-base.so` を使用します。[固定版ソース](https://github.com/ggml-org/llama.cpp/archive/8777c0c833effb3dec51f8db4f49f5c8b234c644.tar.gz)・[Linux x64 CPU版](https://github.com/ggml-org/llama.cpp/releases/download/b12487/llama-b12487-bin-ubuntu-x64.tar.gz)

以下のパスは手元の展開先に置き換えてください。入力・中間ファイル・出力はこのリポジトリの外に置く構成です。

```sh
ASSETS="$HOME/.local/share/x189-kftt-v2"
BF16="$ASSETS/qwen3-1.7b-bf16"
KFTT="$ASSETS/kftt-data-1.0.tar.gz"
LLAMA_SRC="$ASSETS/llama.cpp-8777c0c833effb3dec51f8db4f49f5c8b234c644"
LLAMA_CPU="$ASSETS/llama-b12487-cpu"
WORK="$ASSETS/work"
OUTPUT="$ASSETS/qwen3-1.7b-x189-kftt-v2.gguf"
mkdir -p "$ASSETS"

hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  config.json generation_config.json tokenizer.json tokenizer_config.json \
  model.safetensors.index.json model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors --local-dir "$BF16"
```

## 変換する

```sh
CUDA_VISIBLE_DEVICES=0 python -m x189_clean \
  --checkpoint "$BF16" --kftt "$KFTT" \
  --llama-source "$LLAMA_SRC" --ggml-lib "$LLAMA_CPU/libggml-base.so" \
  --work "$WORK" --output "$OUTPUT" \
  --device cuda:0 --threads 2 --phase all
```

`CUDA_VISIBLE_DEVICES` には使う物理GPUを指定します。出力先の親ディレクトリには書き込み権限が必要です。中断した場合は、同じ入力と設定、同じ作業先で再開してください。

| 引数 | 内容 |
| --- | --- |
| `--checkpoint` | 固定Qwen3-1.7Bを展開したディレクトリ |
| `--kftt` | KFTT v1.0の元tar.gz |
| `--llama-source` | 公式llama.cppのソースディレクトリ |
| `--ggml-lib` | 対応する `libggml-base.so` |
| `--work` | 初回は新規の専用作業先。校正・符号化の中間ファイルを保存 |
| `--output` | 生成するGGUFのパス |
| `--device` | `cuda:0` または `cpu`。既定は `cuda:0` |
| `--threads` | CPUスレッド数。既定は `2` |
| `--phase` | `all`（既定）、`capture`、`encode`、`write` |
| `-h`, `--help` | CLIヘルプ |

段階ごとに実行する場合は、同一コマンドの `--phase` を `capture`、`encode`、`write` の順に切り替えます。最終書き出しでGGUFを読み戻し、全テンソルの型・形状・保存バイトを照合します。

## llama.cppで生成する

```sh
"$LLAMA_CPU/llama-completion" -m "$OUTPUT" -ngl 0 -t 2 \
  -p 'The capital of France is' -n 3 --temp 0 --no-conversation
"$LLAMA_CPU/llama-completion" -m "$OUTPUT" -ngl 0 -t 2 \
  -p '日本の首都は' -n 3 --temp 0 --no-conversation
```

CUDAで推論する場合は同じリリースのCUDA版を使い、`-ngl 99` を指定します。[公式ランタイム](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)

## ライセンス

本リポジトリの配布ソースは [Apache License 2.0](LICENSE) です。外部入力と依存ライブラリにはそれぞれのライセンスが適用されます。[NOTICE](NOTICE)
