# X189 BF16 → GGUF

Hugging FaceのBF16言語モデルをX189で量子化し、標準GGUFに変換するソフトです。GUIとCLIで同じ汎用レシピを実行できます。対応アーキテクチャは使用するllama.cppの変換器と推論ランタイムで決まります。生成したGGUFはトークナイザーを内蔵します。

## インストール

Linux x86-64とPython 3.12を使用します。作業用ディスクには、F32へ展開したモデル、書き出し途中のテンソル、完成GGUFを置く空きが必要です。CUDAで変換する場合はCUDA 12.8対応のNVIDIAドライバーが必要です。

リポジトリのルートで仮想環境を作ります。次の例はCUDA用の依存関係をインストールします。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --resume-retries 30 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r requirements.txt
```

## GUIでBF16モデルを変換する

上の仮想環境を有効にした状態で、デスクトップ用の依存関係を追加して起動します。

```sh
python -m pip install -r requirements-gui.txt
python -m x189_clean gui
```

X11でQtのxcbプラグインを読み込めない場合は、ディストリビューションのXCBカーソルライブラリを導入してください。Ubuntuでは `libxcb-cursor0`、Arch Linuxでは `xcb-util-cursor` です。[QtのX11要件](https://doc.qt.io/qt-6.10/linux-requirements.html)

1. **BF16モデル** のフォルダ、UTF-8の **校正テキスト**、**GGUFの保存先** を選びます。保存先には新しいファイル名を指定します。
2. **llama.cpp 設定** で、公式の変換スクリプトを含むソースフォルダと、同じ版のLinux x64 CPUバイナリフォルダを選びます。変換デバイスはCPUかGPU（CUDA）を選べます。設定は次回起動時にも使えます。[検証版 llama.cpp b12487](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)
3. **変換を開始** を押します。校正と量子化の進行状況が表示されます。検証が完了すると、選んだ場所にGGUFが保存され、**保存先を開く** が使えるようになります。

## CLIでBF16モデルを変換する

元のHugging Faceチェックポイントのディレクトリ、校正用UTF-8テキスト、同じ版のllama.cppソース・バイナリを用意します。バイナリのディレクトリには `llama-imatrix`、`llama-tokenize`、`libggml-base.so` が必要です。[検証版 llama.cpp b12487](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)

```sh
python -m x189_clean list-models --llama-source "$LLAMA_SRC"

CUDA_VISIBLE_DEVICES=0 python -m x189_clean convert \
  --checkpoint "$BF16" --calibration-text "$CALIBRATION_TEXT" \
  --llama-source "$LLAMA_SRC" --llama-imatrix "$LLAMA_CPU/llama-imatrix" \
  --ggml-lib "$LLAMA_CPU/libggml-base.so" \
  --work "$WORK" --output "$OUTPUT" --device cuda:0
```

`$BF16` は `config.json`、トークナイザー、BF16重みが揃ったディレクトリです。分割safetensorsと対応するPyTorch state dictionaryも処理します。`$CALIBRATION_TEXT` はモデルの用途に合ったテキスト、`$WORK` は書き込み可能な作業先、`$OUTPUT` は新規GGUFのファイル名です。GPUのない環境では `--device cpu` を指定できます。校正の範囲は `--context` と `--chunks` で調整します。対象モデルは、llama.cppでGGUFへの変換と重要度の計測の両方が可能な言語モデルです。

汎用レシピはモデルの入力活性の対角二次モーメントを採り、行列形状と役割に応じて8次元格子量子化、標準IQ、低ランク補正、語彙残差を適用します。格子で量子化したテンソルは、その復元値を標準GGUFのF32として格納します。IQブロックを割り当てたテンソルはIQ形式で格納します。F32で格納した割合に応じて、GGUFの容量は元のBF16重みより大きくなる場合があります。GGUFの `x189.recipe` に `X189-Generic-v1` を保存し、テンソル別の処理は作業先の `<出力名>.x189.json` に記録します。

## Qwen3-1.7Bの固定レシピ

X189-KFTT-v2の固定配分で変換する場合は、次の入力とコマンドを使用します。

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

### 固定レシピで変換する

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

## 動作確認

[変換とCPU・GPUでの確認結果](VERIFICATION.md)

## ライセンス

本リポジトリの配布ソースは [Apache License 2.0](LICENSE) です。外部入力と依存ライブラリにはそれぞれのライセンスが適用されます。[NOTICE](NOTICE)
