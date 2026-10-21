"""前処理：生の脳波に帯域通過フィルタをかけ、別の場所に保存する。

生データ（data/raw）は書き換えない。測定回ごとに、次の2つを出力先に作る。
    出力先/sub-XX/ses-YY/sub-XX_ses-YY_desc-filtered_eeg.csv   フィルタをかけた脳波
    出力先/sub-XX/ses-YY/sub-XX_ses-YY_events.csv              区間の記録（生データからそのまま写す）

使い方（dvc.yaml の preprocess の段が、このとおりに実行する）：
    python src/preprocess.py data/raw data/derived/filtered
フィルタの設定は params.yaml の preprocess から読む。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, sosfiltfilt

# 脳波を記録したときの、1秒あたりの測定点の数（Hz）。お手本のデータはすべて250Hz。
SAMPLING_RATE = 250


def is_data_file(path: Path) -> bool:
    """OSが自動で作るファイル（Macの「.DS_Store」「._」で始まるもの、Windowsの「Thumbs.db」など）を除く。"""
    return path.is_file() and not path.name.startswith(".") and path.name.lower() not in {"thumbs.db", "desktop.ini"}


def main() -> int:
    raw, output = Path(sys.argv[1]), Path(sys.argv[2])
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["preprocess"]
    # 帯域通過フィルタ（low_hz〜high_hz の周波数だけを通す）を作る
    sos = butter(params["filter_order"], [params["low_hz"], params["high_hz"]],
                 btype="bandpass", fs=SAMPLING_RATE, output="sos")

    eeg_files = sorted(p for p in raw.glob("sub-*/ses-*/*_eeg.csv") if is_data_file(p))
    if not eeg_files:
        print(f"{raw} の中に脳波のファイル（sub-XX/ses-YY/..._eeg.csv）が見つかりません。")
        return 1
    for path in eeg_files:
        subject, session = path.parent.parent.name, path.parent.name
        header = path.open(encoding="utf-8").readline().strip()
        data = np.loadtxt(path, delimiter=",", skiprows=1)  # 1列目は時刻、2列目からが各チャンネル
        filtered = sosfiltfilt(sos, data[:, 1:], axis=0)

        destination = output / subject / session
        destination.mkdir(parents=True, exist_ok=True)
        # 改行をLFに固定する。OSによって改行が変わると、同じ結果でもハッシュ値が変わるため
        with (destination / f"{subject}_{session}_desc-filtered_eeg.csv").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(header + "\n")
            for t, row in zip(data[:, 0], filtered):
                handle.write(f"{t:.3f}," + ",".join(f"{value:.3f}" for value in row) + "\n")
        events = path.with_name(f"{subject}_{session}_events.csv")
        shutil.copyfile(events, destination / events.name)
        print(f"{subject} {session}: フィルタをかけて保存しました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
