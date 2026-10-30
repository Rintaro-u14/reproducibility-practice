# 脳波データの解析

架空の脳波データ（被験者3人 × 測定2回）から、目を開けた時と閉じた時の後頭部のアルファ波の強さを比べます。また、目の開閉を推定するモデルの設定を探索します。

## 結果の作り直し方

このリポジトリを複製した直後のディレクトリで、1行ずつ実行します。

```
uv sync
uv run python src/analyze.py
```

脳波の解析と探索は、生データを `data/raw/` にコピーしてから、次のコマンドで作り直せます。

```
uv run dvc add --glob "data/raw/sub-0?/ses-0?"
uv run dvc repro
```

## ディレクトリの中身

| 場所 | 中身 | Gitに入れるか |
|---|---|---|
| `src/analyze.py` | 乱数を使う短いサンプルコード（乱数の種を固定して作った数値の平均と標準偏差を表示する） | 入れる |
| `src/preprocess.py`・`features.py`・`summarize.py` | 脳波の解析のコード（前処理・特徴量・集計） | 入れる |
| `src/tune.py` | Optuna でモデルの設定を探すコード | 入れる |
| `params.yaml` | 解析の設定の値（しきい値 `reject_uv`、乱数の種 `seed` など） | 入れる |
| `dvc.yaml` | 処理の手順（どのコードを、どの入力から、どの順番で実行するか） | 入れる |
| `dvc.lock` | 実際に使った入力と出力のハッシュ値の記録（`uv run dvc repro` が書く） | 入れる |
| `data/raw/sub-XX/ses-YY.dvc` | 生データの目印のファイル（ハッシュ値） | 入れる |
| `data/raw/sub-XX/ses-YY/` | 生データの本体 | 入れない（DVCで管理） |
| `data/derived/`、`results/alpha/` | 処理の途中と結果 | 入れない（`uv run dvc repro` で作り直せる） |
| `results/tuning/best.json` | Optuna で見つけた最良の設定 | 入れる |
| `results/tuning/optuna.db` | Optuna の全試行の記録 | 入れない（実行した時刻が入り、毎回変わるため） |
| `pyproject.toml`、`uv.lock`、`.python-version` | Python とパッケージのバージョン | 入れる |
| `scripts/check.py` | 記録と処理結果の状態を確かめるスクリプト | 入れる |

データはすべて架空です。実在する人、試料、装置、大学、研究室とは関係ありません。

## MLflowの記録

探索を実行すると、`results/tuning/mlflow.db` に探索ごとの設定・成績、`mlruns/` に成果物が保存されます。各自のPCに残すためGit・DVCの管理対象外です。

```bash
uv run mlflow server --backend-store-uri sqlite:///results/tuning/mlflow.db --host 127.0.0.1 --port 5000
```

ブラウザで http://127.0.0.1:5000/ を開き、`eyes-open-closed` を選びます。


### ワーク4の記録を照合する

親RunのArtifactsで `provenance.json` を開くと、Gitコミット、入力features.csvのSHA-256、Study名、Run ID、探索・評価条件を確認できます。`inputs/`には設定・探索コード・環境のロックファイルを保存します。SHA-256は入力を直接計算した値で、DVCのMD5とは異なります。実行後の `dvc.lock` と `best.json` をGitに記録し、採用した結果と入力・処理を対応付けます。

この教材はSQLiteへ直接記録するため、MLflowのサーバーは閲覧時だけ起動すれば十分です。Git・DVCだけではMLflowの履歴は共有されません。履歴の復元にはDBと成果物の両方が必要です。
