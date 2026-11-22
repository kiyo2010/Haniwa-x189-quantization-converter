# Haniwa X189 汎用BF16 → GGUF変換ソフト

[English](README.md) · 日本語 · [简体中文](README.zh-CN.md)

Hugging FaceのBF16言語モデルをHaniwa X189で量子化し、標準GGUFに変換するソフトです。GUIとCLIで同じ汎用レシピを実行できます。対応アーキテクチャは使用するllama.cppの変換器と推論ランタイムで決まります。生成したGGUFはトークナイザーを内蔵します。

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

元のHugging Faceチェックポイントのディレクトリ、校正用UTF-8テキスト、同じ版のllama.cppソース・バイナリを用意します。`$LLAMA_SRC` は変換スクリプトを含むソースフォルダ、`$LLAMA_CPU` はLinux x64 CPUバイナリのフォルダです。バイナリのディレクトリには `llama-imatrix`、`llama-tokenize`、`libggml-base.so` が必要です。[検証版 llama.cpp b12487](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)

```sh
python -m x189_clean list-models --llama-source "$LLAMA_SRC"

CUDA_VISIBLE_DEVICES=0 python -m x189_clean convert \
  --checkpoint "$BF16" --calibration-text "$CALIBRATION_TEXT" \
  --llama-source "$LLAMA_SRC" --llama-imatrix "$LLAMA_CPU/llama-imatrix" \
  --ggml-lib "$LLAMA_CPU/libggml-base.so" \
  --work "$WORK" --output "$OUTPUT" --device cuda:0
```

`$BF16` は `config.json`、トークナイザー、BF16重みが揃ったディレクトリです。分割safetensorsと対応するPyTorch state dictionaryも処理します。`$CALIBRATION_TEXT` はモデルの用途に合ったテキスト、`$WORK` は書き込み可能な作業先、`$OUTPUT` は新規GGUFのファイル名です。GPUのない環境では `--device cpu` を指定できます。校正の範囲は `--context` と `--chunks` で調整します。対象モデルは、llama.cppでGGUFへの変換と重要度の計測の両方が可能な言語モデルです。

汎用レシピはモデルの入力活性の対角二次モーメントを採り、行列形状と役割に応じて8次元格子量子化、標準IQ、低ランク補正、語彙残差を適用します。格子で量子化したテンソルは、その復元値を標準GGUFのF32として格納します。IQブロックを割り当てたテンソルはIQ形式で格納します。F32で格納した割合に応じて、GGUFの容量は元のBF16重みより大きくなる場合があります。GGUFの `x189.recipe` に `Haniwa X189` を保存し、テンソル別の処理は作業先の `<出力名>.x189.json` に記録します。

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
