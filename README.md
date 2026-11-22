# X189-XXS 変換ソフト

**研究環境なしで元BF16から標準GGUFを作る場合は [独立変換の手順](STANDALONE.md) を使用してください。** 公開KFTTで校正する X189-public-KFTT-v1 を提供します。

## 歴史的レシピの互換コンポーネント

このリポジトリには、校正・L17のIQ2_XXS符号化CLIに加え、検証済みの元BF16→標準GGUFの段階ドライバーを収めています。モデル本体、トークナイザー、校正入力・統計、評価資料、出力レコードは既存のローカル環境または正規の配布元から用意します。

- [研究環境での元BF16から標準GGUFへの変換](private-x189-from-bf16/README.md)
- [校正・L17符号化CLIの入力と実行方法](#入力と環境)


Qwen3-1.7Bの公式BF16から、元のnative IQ用の入力2次モーメントを再捕捉し、L17 downのIQ2_XXSレコードを符号化するオフラインコード。元QPACKや量子化済みレコード、保存済み校正統計を入力に取らない。共有語彙の固定順位計算も含む。全モデルの第三者再構築には、下記の原資料と追加の組立工程が必要。

## 入力と環境

- Linux、Python 3.12、対応NVIDIAドライバー、RTX 5090、`requirements.txt`。Torchは元のCUDA 12.8ビルド `2.10.0+cu128`。ローカルに用意してからオフラインで実行する。
- Qwen/Qwen3-1.7Bの公式スナップショット `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。Apache-2.0。固定revisionを指定して取得し、シャード、index、設定、トークナイザーのSHA-256をコードが確認する。[公式リビジョン](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e)
- WikiText-2の `wikitext-2-raw-v1/train` parquet。ファイルSHA-256は `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`。同じバイトのファイルを取得する。コードは `text` 列を改行1字で連結し、末尾改行を加えず、展開SHAも確認する。CC BY-SA 3.0およびGFDL。[配布者の固定リビジョン](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1) · [配布者のカード](https://huggingface.co/datasets/Salesforce/wikitext)
- 元の自作校正18文書。入手済みで利用が認められた原文だけを、英・日・西それぞれ元順の6文字列とし、JSONキー `en`、`ja`、`es` で指定する。これらの公開取得先と単独の再配布許可は確認できていない。代用テキストでは固定X189を再現できない。
- 元のggml-base共有ライブラリ（llama.cpp派生ソース `864934c78136ad75274f202b8eb1e7c751d81fb1`、実ライブラリSHAはコードの `GGML`）。既定は元ライブラリのSHAを要求する。独立ビルドでは `--allow-rebuilt-ggml` で試すことができ、生成したネイティブブロックとラッパーの両方が過去のSHAに一致して初めて書き出す。元のバイナリを一般利用者が独立取得できる配布先は未確認。モデル推論向けの無改造llama.cppとは別の要件。[記録上の元forkのリビジョン](https://github.com/kuraneko1/Qwen3.8-27B-QSA-llama.cpp/commit/864934c78136ad75274f202b8eb1e7c751d81fb1)

## 実行

```sh
python calibrate.py --checkpoint "$BF16" --original-calibration "$ORIGINAL_JSON" \
  --wiki-parquet "$WIKI_TRAIN_PARQUET" --output "$OUT/iq-moments.npz"
python x189_recipe.py --checkpoint "$BF16" --moments "$OUT/iq-moments.npz" \
  --ggml "$GGML" --output "$OUT/L17-XXS.bin"
```

捕捉順は自作18件の後にWikiTextの0始まり8行。256トークンで切り、全26系列2,758トークンの境界込みSHAを検証する。歴史的IQ 75件は偶数番の系列の2次モーメントからFP32へ丸める。L17 IQ2_XXSは全26系列の2次モーメントからFP32へ丸める。後者の列重要度と書き出した3,244,092 Bのレコードを個別に認証する。出力ファイルは既存ファイルを置き換えない。

## 全モデルの残り

残り309件の再構築は、KFTT v1.0の固定trainから求めるXE82・低ランク因子、元の固定型割当と輸送設定、語彙1件の圧縮を含む。KFTTの公式targzは99,246,893 B、SHA-256 `fcfcaa670d6d59aa691b0e909c0d7c393852dd2fb1d6310fda9b3282dc6d1638`。語彙はKFTT校正の32,759入力トークンと新規XE82復元値から、正の頻度×表現利得を優先し、不足分を利得順で補う。同点時は小さい語彙ID。上位2,654を格納する。元のKFTT校正対の選定には過去の非公開除外資料が使われており、独立実行には同じ順序の公開可能な訓練行座標レシピが必要。[KFTT配布](https://www.phontron.com/kftt?lang=en)

段階ドライバーは [ローカル全変換の手順](private-x189-from-bf16/README.md) にあります。IQ全件・XE82・因子・語彙残差の固定実装や認証済み校正入力は既存研究環境から用意します。第三者向けには、正規に取得できる資料から全310件と無改造CPU/CUDA推論までを再現する検証が残っています。
