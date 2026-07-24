# Character Segmentation with GNN

フォント輪郭パスから文字区間を識別する、GNNエッジ分類システム。

## 構成

Rust(データ生成・推論)と Python(学習)のハイブリッド構成。

| コンポーネント       | 言語   | 役割                                                                      |
| -------------------- | ------ | ------------------------------------------------------------------------- |
| `crates/glyph-core`  | Rust   | パス抽出・弧長リサンプリング・特徴量生成・KD木グラフ構築(学習/推論で共有) |
| `crates/dataset-gen` | Rust   | 合成データセット生成(rustybuzz + ttf-parser + rayon)→ MessagePack         |
| `training/glyph_gnn` | Python | PyG データローダー・エッジ特徴量つき GAT・Focal Loss・ONNX エクスポート   |
| `crates/glyph-infer` | Rust   | ort による ONNX 推論・閾値判定・Union-Find 連結成分抽出                   |

## データフロー

```text
fonts/*.ttf ──> dataset-gen ──> dataset/*.msgpack ──> train.py ──> best.pt
                                                          │
demo text ──> glyph-core 前処理 ──> ort 推論 <── model.onnx <── export_onnx.py
                                       │
                                閾値判定 + Union-Find ──> 文字グループ
```

## セットアップ

```bash
# Rust 側(ビルドのみ)
cargo build --release

# Python 側 (uv)。Linux/Windows は CUDA 12.6 版 torch を解決
uv sync
```

## 使い方

### 1. データセット生成(フェーズ1)

```bash
cargo run --release -p dataset-gen -- \
    --body-fonts fonts/body --deco-fonts fonts/deco \
    --count 100000 --out dataset/train
```

日本語データセット例（`charsets/` のプリセットを利用）:

```bash
cargo run --release -p dataset-gen -- \
    --body-fonts fonts/ja-train --deco-fonts fonts/ja-train \
    --charset-file charsets/japanese-mixed.txt \
    --count 100000 --out dataset/ja-train
```

- `--charset` … インラインで文字集合を指定（デフォルトは英数字）
- `--charset-file` … UTF-8 テキストファイルから文字集合を読み込み（`#` 行頭コメントと空行は無視。同一文字の重複は出現率を上げる）
- 同梱プリセット: `charsets/ascii.txt`, `hiragana.txt`, `katakana.txt`, `kana.txt`, `japanese-basic.txt`, `japanese-mixed.txt`

- 本文フォント:装飾フォント = 7:3(`--deco-ratio` で変更可)
- データ拡張: 負のトラッキング(パス交差の生成)、ベースライン上下動、
  アスペクト比変更、ポイントノイズ
- `--knn` / `--radius` / `--spacing` でグラフ構築パラメータを調整

### 2. 学習(フェーズ2)

```bash
uv run python -m glyph_gnn.train --data dataset/train --out training/checkpoints/run --epochs 30
```

- 検証分割はデフォルトでフォント単位のホールドアウト(未学習フォント汎化を測定)
- Focal Loss(`--focal-alpha` / `--focal-gamma`)でクラスインバランスを補正
- ベストチェックポイントは境界クラス(負例)F1 で選択
- 学習成果物は `training/checkpoints/<run>/` 配下にまとめる(ディレクトリごと gitignore)
- 起動時にデータセット統計・モデル規模・ハイパーパラメータのサマリを表示し、
  シャード読み込み / エポック / バッチ / 検証の進捗バーを出す
  (パイプ出力時は自動で無効。`--no-progress` で明示的に抑止)
- エポックごとに loss・学習率・経過時間・スループット・検証指標を 1 行で出力

### 3. ONNX エクスポート

```bash
uv run python -m glyph_gnn.export_onnx --checkpoint training/checkpoints/run/best.pt --out model.onnx
```

動的軸(ノード数・エッジ数)でエクスポートし、onnxruntime と PyTorch の
出力一致を自動検証する。

### 4. 推論(フェーズ3)

```bash
# デモ: テキストを密着レイアウトして分割
cargo run --release -p glyph-infer -- --model model.onnx \
    demo --font fonts/body/arial.ttf --text "Overlap" --tracking=-0.12

# 評価: 生成済みシャードで精度・レイテンシ計測(フェーズ4)
cargo run --release -p glyph-infer -- --model model.onnx \
    eval --shard dataset/eval/shard_00000.msgpack
```

- 既定で CUDA Execution Provider を登録(RTX 3060)。`--cpu` で CPU 実行
- `--threshold` でエッジ結合確率の判定閾値を調整(Union-Find は偽陽性1本で
  グループが結合するため高めが安全。12kサンプル学習モデルの実測では
  0.7 が最良: 未学習フォントで誤結合ゼロ・完全グルーピング 94%)

### CUDA 実行の要件

- ort は CUDA ≥ 13.2(または ≥ 12.8)+ cuDNN ≥ 9 のバイナリを自動ダウンロード。
  CUDA 12/13 が併存する環境ではビルド時に `ORT_CUDA_VERSION=13` を指定
- 実行時に cudart / cublas / cuDNN の DLL が PATH 上に必要。cuDNN を別途
  インストールしていない場合は、CUDA 版 PyTorch が同梱する DLL を流用できる:

```bash
PATH=".venv/Lib/site-packages/torch/lib:$PATH" \
    cargo run --release -p glyph-infer -- --model model.onnx eval --shard ...
```

- 起動時に `execution provider: CUDA` / `CPU` が表示される。CUDA 登録に
  失敗した場合は警告を出して CPU にフォールバックする
- 初回推論は CUDA カーネル初期化で数百 ms かかるため、レイテンシ計測は
  ウォームアップ後に行うこと(`eval` モードは自動でウォームアップする)

## モデルアーキテクチャ

- 初期レイヤー: ノード特徴量(13次元)→ hidden への MLP + LayerNorm
- GAT レイヤー x N: エッジ特徴量(4次元)を考慮した GATv2 スタイルの
  マルチヘッド注意。**ONNX エクスポートを保証するため PyG の conv ではなく
  gather / scatter_add のみで実装**(PyG の scatter カーネルは torch.onnx で
  不安定なため)
- エッジ分類器: `[h_src, h_dst, |h_src - h_dst|, edge_attr]` → MLP → ロジット

### グラフ表現

- ノード = 輪郭を弧長等間隔でリサンプリングした点
  (座標・接線・重心オフセット・輪郭重心・バウンディングボックス・弧長・輪郭位相)
- エッジ = KD木による半径つき kNN(既定: k=8, r=0.25em)、無向(双方向格納)
- ラベル = 両端ノードが同一文字(シェーピングクラスタ)に属するか

## 日本語対応

パイプライン(rustybuzz シェーピング → 輪郭抽出 → グラフ化)は文字体系に
依存しないため、**学習データに日本語を含めることで対応する**。

- `dataset-gen --charset-file`（または `--charset`）にかな・漢字を指定し、日本語フォントのプールを
  `--body-fonts` に渡す(`.ttc` コレクションはフェイス0で読み込み対応済み)
- 漢字は1文字が多数の輪郭(ストローク)で構成されるためグラフが大きく、
  英数字より推論コストが高い(実測 約6ms/件 @ RTX 3060)
- 完全グルーピングの難度も高い: 1文字内の全ストロークを結合できて初めて
  正解となるため、境界 F1 の要求水準が英数字より厳しい
- 英数字+日本語の混合データセットで学習しても英数字側の精度は劣化しない
  (実測ではむしろ向上)

## リスク対応(計画書より)

- **特徴次元の可変化**: `NODE_DIM` / `EDGE_DIM` は `glyph-core` の定数と
  モデルの `--hidden` 等ハイパーパラメータで一元管理。チェックポイントに
  hparams を保存し、エクスポート時に自動復元
- **推論前処理のボトルネック**: 前処理は現状 1〜2ms(10文字・CPU)。
  ボトルネック化した場合は KD木を均一グリッド(空間ハッシュ)に置換可能な
  よう `graph.rs` の近傍探索を分離済み
