# 元BF16から標準GGUFまでのローカル変換

このディレクトリは、検証済みのX189-XXS固定レシピで使った3個の段階ドライバーを収録しています。既存のX189研究環境が対象です。

## 準備

- 既存研究ルート内の `private-x189-from-bf16/` に3個のPythonファイルを同じ名前で配置します。リポジトリ直下の `calibrate.py` と `x189_recipe.py` は研究ルート内の `private-x189-github-prep/` に配置します。
- 研究ルートには `code/`、X153およびX190の固定実装、`private-x189-xxs-stock-gguf/` の補助コードと無改造llama.cpp b12487、`sources/env.sh` が必要です。照合用の元X189は、新しい実物を作った後の検証で使います。
- 元のQwen3 BF16と設定、固定KFTT 1.0、固定校正資料・統計、ggml-base共有ライブラリ、gguf-pyを用意します。ドライバーが入力をSHA-256で照合します。

## 実行する段階

メモリ25 GiB通常・30 GiB上限の管理枠で、既存研究ルートを作業ディレクトリとして動かします。6 TB HDDなど、書き込み可能な作業先を準備してください。出力先にはこのディレクトリ直下の新しい `work` を使います。

```sh
"$PY" -B private-x189-from-bf16/rebuild.py --stage prepare \
  --work "$PWD/private-x189-from-bf16/work" --checkpoint "$BF16" \
  --kftt "$KFTT_ARCHIVE" --calibration "$CALIBRATION_JSON" \
  --diagonals "$DIAGONALS_NPZ" --ggml "$GGML_LIB" \
  --gguf-python-path "$GGUF_PY"
```

同じ引数で `capture-00`〜`capture-04`、`xe82-00`〜`xe82-10`、`records`、`assemble`、`encode`、`qualify`、`factor-00`、`factor-20`、`seal`、`entropy` の順に進めます。`xxs` は `retry_xxs.py`、`combine` は `rebuild.py`、`stock`・`cpu`・`cuda` は `fresh_gguf.py` を使います。捕捉、格子符号化、CUDA生成の該当段階は、管理枠から1台の対応GPUを割り当てます。

標準書き出しは元のBF16スナップショットの設定・トークナイザーから無改造コンバーターでメタデータを再生成し、新規QPACKの310レコードを読み直します。研究環境ではこの経路で生成した標準GGUFを無改造llama.cppでCPUとRTX 5090から読み込み、英語と日本語の生成を確認しました。

このソース一式で利用できるのはローカル環境向けの変換と、ルートREADMEの条件に従う校正・L17符号化です。第三者が原18校正文と研究用の固定依存物を独立取得できる公開版は準備途中です。
