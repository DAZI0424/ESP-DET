"""Create a concise Markdown report from an Ultralytics detection training run."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path


METRIC_COLUMNS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, skipinitialspace=True))


def metric(row: dict[str, str], name: str) -> float:
    return float(row.get(name, "nan"))


def generate(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Training results not found: {results_path}")

    rows = read_results(results_path)
    if not rows:
        raise RuntimeError("results.csv contains no epochs")

    best = max(rows, key=lambda row: metric(row, "metrics/mAP50-95(B)"))
    last = rows[-1]
    best_epoch = int(float(best["epoch"])) + 1
    last_epoch = int(float(last["epoch"])) + 1

    artifacts = [
        name
        for name in ("weights/best.pt", "weights/last.pt", "results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png")
        if (run_dir / name).exists()
    ]
    lines = [
        "# ESPDet Pico 训练报告",
        "",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- 训练目录：`{run_dir}`",
        f"- 已完成轮数：{last_epoch}",
        f"- 最佳轮数（按 mAP50-95）：{best_epoch}",
        "",
        "## 最佳验证指标",
        "",
        "| Precision | Recall | mAP50 | mAP50-95 |",
        "| ---: | ---: | ---: | ---: |",
        "| " + " | ".join(f"{metric(best, column):.4f}" for column in METRIC_COLUMNS) + " |",
        "",
        "## 最后一轮验证指标",
        "",
        "| Precision | Recall | mAP50 | mAP50-95 |",
        "| ---: | ---: | ---: | ---: |",
        "| " + " | ".join(f"{metric(last, column):.4f}" for column in METRIC_COLUMNS) + " |",
        "",
        "## 产物",
        "",
        *[f"- `{name}`" for name in artifacts],
        "",
        "说明：若未完成 500 轮，训练由早停机制在验证指标连续 75 轮没有改善后结束。",
        "",
    ]
    output = run_dir / "training_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: generate_training_report.py <run_dir>")
    print(generate(Path(sys.argv[1])))
