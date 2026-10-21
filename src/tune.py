"""モデルの設定探索：目を開けていたか閉じていたかを当てるモデルの設定を、Optunaで探す。

探すもの：使う周波数帯の下限と幅、ロジスティック回帰の正則化の強さ C。
評価：test_subject 以外の被験者で、測定回を1つずつ外して確かめる（交差検証）。
test_subject は探索に一切使わず、最後に一度だけ正答率を確かめる。

出力：
    results/tuning/best.json   最良の設定と正答率。小さい文字のファイルなので、Gitで記録する
    results/tuning/optuna.db   全試行の記録（optuna-dashboard で見る）。
                               実行した時刻も記録されるので、同じ探索でも毎回中身が変わる。
                               そのためGitにもDVCにも入れず、各自のPCで作る

使い方（dvc.yaml の tune の段が、このとおりに実行する）：
    python src/tune.py data/derived/features.csv results/tuning
探す範囲・試行の回数・乱数の種は params.yaml の tuning から読む。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def band_features(rows: list[dict[str, str]], channels: list[str], low: int, width: int) -> np.ndarray:
    """各試行・各チャンネルについて、low Hz から width Hz 分のパワーの平均（log10）を求める。"""
    freqs = range(low, low + width)
    return np.array([[np.log10(np.mean([float(r[f"{ch}_{f}hz"]) for f in freqs])) for ch in channels] for r in rows])


def model(c: float):
    """値の大きさをそろえてから、ロジスティック回帰で2つに分ける。"""
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))


def main() -> int:
    features, output = Path(sys.argv[1]), Path(sys.argv[2])
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["tuning"]
    with features.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    channels = list(dict.fromkeys(c.split("_")[0] for c in rows[0] if c.endswith("hz")))
    max_hz = max(int(c.split("_")[1][:-2]) for c in rows[0] if c.endswith("hz"))
    train = [r for r in rows if r["subject"] != params["test_subject"]]
    test = [r for r in rows if r["subject"] == params["test_subject"]]
    y_train = np.array([r["condition"] == "eyes_closed" for r in train])
    y_test = np.array([r["condition"] == "eyes_closed" for r in test])
    groups = [f"{r['subject']}_{r['session']}" for r in train]

    output.mkdir(parents=True, exist_ok=True)
    database = output / "optuna.db"
    database.unlink(missing_ok=True)  # 前回の記録を消してから、探索をやり直す

    def objective(trial: optuna.Trial) -> float:
        low = trial.suggest_int("band_low_hz", *params["band_low_hz"])
        width = trial.suggest_int("band_width_hz", *params["band_width_hz"])
        c = trial.suggest_float("regularization_c", *params["regularization_c"], log=True)
        if low + width - 1 > max_hz:
            raise optuna.TrialPruned()  # 特徴量にない周波数を使う組み合わせは、試さずに飛ばす
        x = band_features(train, channels, low, width)
        return float(cross_val_score(model(c), x, y_train, groups=groups, cv=LeaveOneGroupOut()).mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="eyes-open-closed",
        storage=f"sqlite:///{database.as_posix()}",
        direction="maximize",
        # 乱数の種を固定する。同じ種・同じデータなら、誰が実行しても同じ順番で同じ設定を試す
        sampler=optuna.samplers.TPESampler(seed=params["seed"]),
    )
    study.optimize(objective, n_trials=params["n_trials"])

    best = study.best_params
    # 計算機によって小数点以下のずっと先の桁が変わることがあるので、6桁に丸めて記録する
    c = float(f"{best['regularization_c']:.6g}")
    final = model(c).fit(band_features(train, channels, best["band_low_hz"], best["band_width_hz"]), y_train)
    test_accuracy = final.score(band_features(test, channels, best["band_low_hz"], best["band_width_hz"]), y_test)
    result = {
        "best_params": {"band_low_hz": best["band_low_hz"], "band_width_hz": best["band_width_hz"], "regularization_c": c},
        "cross_validation_accuracy": round(study.best_value, 4),
        "test_subject": params["test_subject"],
        "test_accuracy": round(float(test_accuracy), 4),
        "n_trials": params["n_trials"],
        "seed": params["seed"],
    }
    # 改行をLFに固定する（WindowsでもMacでも同じファイルになるように）
    (output / "best.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
    print(f"{params['n_trials']} 回試しました。最良の設定：{result['best_params']}")
    print(f"交差検証の正答率 {result['cross_validation_accuracy']}、"
          f"{params['test_subject']} での正答率 {result['test_accuracy']}")
    print(f"全試行の記録：{database.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
