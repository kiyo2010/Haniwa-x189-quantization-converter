# Qwen3-1.7B BF16から標準GGUFを生成する

**X189-public-KFTT-v1** は、公開KFTTの訓練対から校正データを選び、公式Qwen3-1.7BのBF16から全310テンソルを再計算します。X189の層ごとの型割当、XE82、語彙2,654行の残差、2つのrank-54補正を使います。標準GGUFではXE82と残差をFP32に展開し、76件を標準IQブロックに保存します。固定校正資料が異なるため、研究時のX189とは別レシピ・別の復元値になります。

## 必要な入力

- 64-bit little-endian Linux、Python 3.12、NVIDIAドライバーとCUDA 12.8対応GPU。変換先に25 GB以上の空き容量を用意します。検証手順ではRTX 5090を一台割り当てます。
- [公式Qwen3-1.7Bの固定スナップショット](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)。2つのsafetensors、safetensors index、config、tokenizer、tokenizer_configが必須です。
- 公式KFTT 1.0の元アーカイブ。SHA-256は `fcfcaa670d6d59aa691b0e909c0d7c393852dd2fb1d6310fda9b3282dc6d1638` です。[配布元](https://www.phontron.com/kftt?lang=en)
- [無改造llama.cpp b12487](https://github.com/ggml-org/llama.cpp/releases/tag/b12487) の[ソースアーカイブ](https://github.com/ggml-org/llama.cpp/archive/8777c0c833effb3dec51f8db4f49f5c8b234c644.tar.gz)と[Ubuntu x64 CPU版](https://github.com/ggml-org/llama.cpp/releases/download/b12487/llama-b12487-bin-ubuntu-x64.tar.gz)。ソース側は `convert_hf_to_gguf.py` と `gguf-py`、CPU版は `libggml-base.so` と読み戻しツールを使います。実行時に両方の依存を指定します。

ソースアーカイブのSHA-256は `a5ea74766fa1ec3ed1341fe6f13c47a242c9b247fc5e6a858d06938aa6dad55b`、CPU版アーカイブのSHA-256は `43a5be706de133d9193b3be7eaaf4123c7e79007e4aac78a2a99c95ece9e2c5b` です。

## 変換

リポジトリのルートで独立したPython環境を用意します。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --resume-retries 30 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r requirements-standalone.txt
```

`BF16` は元スナップショット、`KFTT` は元アーカイブ、`LLAMA_SRC` はソース展開先、`LLAMA_CPU` はCPU版の実行ファイルと共有ライブラリのあるディレクトリの絶対パスです。 `WORK` と `OUTPUT` は空の専用作業先と、別名の新しい出力先です。出力先の親ディレクトリを用意します。

```sh
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  .venv/bin/python -B -m x189_clean \
  --checkpoint "$BF16" --kftt "$KFTT" \
  --llama-source "$LLAMA_SRC" --ggml-lib "$LLAMA_CPU/libggml-base.so" \
  --work "$WORK" --output "$OUTPUT" --device cuda:0 --threads 2
```

`CUDA_VISIBLE_DEVICES` には使う物理GPUを一台だけ指定します。容量の大きい埋め込みは行を分け、5回分の校正統計と各テンソルの変換結果を作業先で再開可能に保存します。必要に応じて同じ引数に `--phase capture`、`--phase encode`、`--phase write` を順に追加して実行できます。 `work/gguf-proof.json` に全レコードと物理容量の照合を記録します。

```sh
"$LLAMA_CPU/llama-completion" -m "$OUTPUT" -ngl 0 -t 2 \
  -p 'The capital of France is' -n 3 --temp 0 --no-conversation
```

GPU推論を確認する場合は b12487 のCUDA版と対応ランタイムを使い、 `-ngl 99` を指定します。[b12487のCUDA版](https://github.com/ggml-org/llama.cpp/releases/tag/b12487)

歴史的な固定レシピは研究環境での再現用です。この公開KFTT版は `x189_clean/` のソースと上記の公式入力・依存物から実行します。E8Pの由来とGPLv3の条件は [付属のNOTICE](x189_clean/_vendor/NOTICE.md) を参照してください。
