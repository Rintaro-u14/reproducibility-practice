"""Gitの記録と解析結果の状態を確かめる。

使い方（リポジトリのディレクトリで実行する）：
    uv run python scripts/check.py 2
    uv run python scripts/check.py 3
    uv run python scripts/check.py 4

どの項目も、できていなければ [まだ] と、確認する内容を表示する。
このスクリプトは何も書き換えない。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ["sub-01", "sub-02", "sub-03"]
SESSIONS = ["ses-01", "ses-02"]
RAW_DVC_FILES = [f"data/raw/{s}/{t}.dvc" for s in SUBJECTS for t in SESSIONS]
FIRST_THRESHOLD = 100  # 初期状態の params.yaml の features.reject_uv


def run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def git(*args: str) -> str:
    return run("git", *args).stdout.strip()


def dvc_up_to_date(stage: str) -> bool:
    """uv run dvc status 段の名前 で、その段（と前の段）が記録と一致していれば True。"""
    status = run(sys.executable, "-m", "dvc", "status", "--json", stage)
    try:
        return status.returncode == 0 and not json.loads(status.stdout or "{}")
    except ValueError:
        return False


def recorded_threshold() -> object:
    """最後に記録した params.yaml の features.reject_uv。読めなければ None。"""
    try:
        return yaml.safe_load(git("show", "HEAD:params.yaml"))["features"]["reject_uv"]
    except (TypeError, KeyError, yaml.YAMLError):
        return None


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"2", "3", "4"}:
        print("使い方：uv run python scripts/check.py 確認の番号（2、3、4）")
        print("例：    uv run python scripts/check.py 2")
        return 2
    work = sys.argv[1]
    results: list[tuple[bool, str, str]] = []

    def item(ok: bool, label: str, hint: str) -> None:
        results.append((ok, label, hint))

    # --- 共通の確認 ---
    item(bool(git("config", "user.name")) and bool(git("config", "user.email")),
         "Gitに名前とメールアドレスが設定されている", "git config --global user.name / user.email を設定する")
    item(run("git", "rev-parse", "--git-dir").returncode == 0,
         "Gitの記録があるディレクトリ（git clone で複製したもの）で実行している",
         "履歴が必要な場合は git clone で複製したディレクトリを使う")
    item(not git("diff", "--name-only", "--diff-filter=U"), "変更のぶつかり（CONFLICT）が残っていない", "競合しているファイルを確認し、解決する")

    # --- しきい値の変更が記録されているか ---
    threshold = recorded_threshold()
    item(threshold is not None and threshold != FIRST_THRESHOLD,
         f"params.yaml のしきい値（reject_uv）を {FIRST_THRESHOLD} から変え、記録してある"
         + (f"（いまの記録は {threshold}）" if threshold not in (None, FIRST_THRESHOLD) else ""),
         "params.yaml の reject_uv を変更して Git に記録する")
    item(not git("status", "--porcelain", "--", "params.yaml", "dvc.yaml"),
         "params.yaml・dvc.yaml が、記録されたものから変わっていない",
         "git diff で違いを見る。試しに変えただけなら git restore params.yaml")

    if work in {"3", "4"}:
        tracked = set(git("ls-files", "data/raw").splitlines())
        missing = [p for p in RAW_DVC_FILES if p not in tracked]
        item(not missing, "6回分の目印のファイル（data/raw/sub-0?/ses-0?.dvc）がGitに記録されている",
             "生データを DVC に登録して Git に記録する（" + "、".join(missing) + "）")
        item("\n  summarize:" in git("show", "HEAD:dvc.lock"), "dvc.lock に、結果を作った記録がある",
             "uv run dvc repro summarize を実行し、dvc.lock を Git に記録する")
        item(dvc_up_to_date("summarize"), "uv run dvc status summarize で、データと処理の流れが記録と一致している",
             "設定を確認し、uv run dvc repro summarize を実行する")
        item((ROOT / "results" / "alpha" / "alpha_summary.csv").is_file(), "結果の表 results/alpha/alpha_summary.csv がある",
             "uv run dvc repro summarize を実行する")

    if work == "4":
        item("\n  tune:" in git("show", "HEAD:dvc.lock"), "dvc.lock に、探索（tune）を実行した記録がある", "uv run dvc repro tune を実行し、dvc.lock を Git に記録する")
        item("results/tuning/best.json" in git("ls-files", "results/tuning")
             and not git("status", "--porcelain", "--", "results/tuning/best.json", "dvc.lock"),
             "採用した設定 results/tuning/best.json と dvc.lock が記録され、変わっていない", "best.json と dvc.lock を確認し、Git に記録する")
        item(dvc_up_to_date("tune"), "uv run dvc status tune で、探索が記録と一致している", "uv run dvc repro tune を実行する")
        item((ROOT / "results" / "tuning" / "optuna.db").is_file(), "自分のPCに全試行の記録 results/tuning/optuna.db がある",
             "uv run dvc repro --force tune を実行する")

    for ok, label, hint in results:
        print(("[OK]   " if ok else "[まだ] ") + label + ("" if ok else f"\n         → {hint}"))
    remaining = sum(not ok for ok, _, _ in results)
    print(f"すべて確認できました。確認 {work} は完了です。" if remaining == 0 else f"あと {remaining} 項目です。")
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
