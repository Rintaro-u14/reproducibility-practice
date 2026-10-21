"""集計：被験者・測定回ごとに、目を開けた時と閉じた時の、後頭部のアルファ波の強さをまとめる。

出力：
    結果のディレクトリ/alpha_summary.csv     表（値は log10 のパワー。差が大きいほど、目を閉じた時にアルファ波が強い）
    結果のディレクトリ/alpha_by_condition.png  図

使い方（dvc.yaml の summarize の段が、このとおりに実行する）：
    python src/summarize.py data/derived/features.csv results/alpha
周波数帯とチャンネルは params.yaml の summary から読む。
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面を開かずに、図をファイルに保存する
import matplotlib.pyplot as plt
import numpy as np
import yaml


def main() -> int:
    features, output = Path(sys.argv[1]), Path(sys.argv[2])
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))["summary"]
    columns = [f"{ch}_{f}hz" for ch in params["channels"]
               for f in range(params["alpha_low_hz"], params["alpha_high_hz"] + 1)]

    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with features.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            alpha = np.log10(np.mean([float(row[c]) for c in columns]))
            values[(row["subject"], row["session"], row["condition"])].append(alpha)

    output.mkdir(parents=True, exist_ok=True)
    table = []
    for subject, session in sorted({(s, t) for s, t, _ in values}):
        open_ = np.mean(values[(subject, session, "eyes_open")])
        closed = np.mean(values[(subject, session, "eyes_closed")])
        table.append([subject, session, f"{open_:.4f}", f"{closed:.4f}", f"{closed - open_:.4f}"])
    with (output / "alpha_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")  # 改行をLFに固定する
        writer.writerow(["subject", "session", "log10_alpha_eyes_open", "log10_alpha_eyes_closed", "difference"])
        writer.writerows(table)

    # 測定回ごとに「目を開けた時 → 閉じた時」を線で結ぶ
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
    colors = {"sub-01": "#2e6f95", "sub-02": "#c8702a", "sub-03": "#3a8a44"}
    for subject, session, open_, closed, _ in table:
        ax.plot([0, 1], [float(open_), float(closed)], marker="o", color=colors.get(subject, "#52606d"),
                linestyle="-" if session == "ses-01" else "--", label=f"{subject} {session}")
    ax.set_xticks([0, 1], ["eyes open", "eyes closed"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(f"log10 alpha power ({params['alpha_low_hz']}-{params['alpha_high_hz']} Hz)")
    ax.set_title(f"Occipital alpha ({', '.join(params['channels'])})", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "alpha_by_condition.png")

    for row in table:
        print(f"{row[0]} {row[1]}  目を開けた時 {row[2]}  閉じた時 {row[3]}  差 {row[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
