"""特徴量：前処理後の脳波を2秒ずつの試行に切り、チャンネルごと・1Hzごとのパワーを計算する。

出力は1行が1試行の表（CSV）。列は
    subject, session, trial, condition（eyes_open：目を開けた／eyes_closed：閉じた）,
    各チャンネルの 1Hz〜max_hz のパワー（µV²/Hz）

使い方（dvc.yaml の features の段が、このとおりに実行する）：
    python src/features.py data/derived/filtered data/derived/features.csv
試行の長さなどの設定は params.yaml の features から読む。
振幅の最大が reject_uv（µV）を超えた試行は、まばたきなどのノイズが大きいとみなして除く。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import welch

SAMPLING_RATE = 250


def is_data_file(path: Path) -> bool:
    """OSが自動で作るファイル（Macの「.DS_Store」「._」で始まるもの、Windowsの「Thumbs.db」など）を除く。"""
    return path.is_file() and not path.name.startswith(".") and path.name.lower() not in {"thumbs.db", "desktop.ini"}


def main() -> int:
    filtered, output = Path(sys.argv[1]), Path(sys.argv[2])
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["features"]
    epoch = int(params["epoch_s"] * SAMPLING_RATE)  # 1つの試行の長さ（測定点の数）
    edge = int(params["edge_s"] * SAMPLING_RATE)    # 区間の切り替わりの前後で使わない長さ
    freqs_kept = np.arange(1, params["max_hz"] + 1)

    rows = []
    rejected = 0
    channels: list[str] = []
    paths = sorted(p for p in filtered.glob("sub-*/ses-*/*_desc-filtered_eeg.csv") if is_data_file(p))
    for path in paths:
        subject, session = path.parent.parent.name, path.parent.name
        channels = path.open(encoding="utf-8").readline().strip().split(",")[1:]
        eeg = np.loadtxt(path, delimiter=",", skiprows=1)[:, 1:]
        with (path.parent / f"{subject}_{session}_events.csv").open(encoding="utf-8", newline="") as handle:
            events = list(csv.DictReader(handle))
        trial = 0
        for event in events:
            start = int(float(event["onset_s"]) * SAMPLING_RATE) + edge
            stop = int((float(event["onset_s"]) + float(event["duration_s"])) * SAMPLING_RATE) - edge
            for begin in range(start, stop - epoch + 1, epoch):
                if np.abs(eeg[begin:begin + epoch]).max() > params["reject_uv"]:
                    rejected += 1  # ノイズが大きい試行は使わない
                    continue
                # Welch法で、周波数ごとのパワーを計算する
                freqs, power = welch(eeg[begin:begin + epoch], fs=SAMPLING_RATE, nperseg=SAMPLING_RATE, axis=0)
                picked = power[np.isin(freqs, freqs_kept)]  # 形：周波数 × チャンネル
                trial += 1
                rows.append([subject, session, trial, event["condition"],
                             *(f"{value:.6g}" for value in picked.T.ravel())])

    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["subject", "session", "trial", "condition"] + [f"{ch}_{f}hz" for ch in channels for f in freqs_kept]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")  # 改行をLFに固定する
        writer.writerow(header)
        writer.writerows(rows)
    print(f"{len(paths)} 回の測定から、{len(rows)} 試行の特徴量を {output.as_posix()} に保存しました"
          f"（振幅が {params['reject_uv']} µV を超えた {rejected} 試行を除きました）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
