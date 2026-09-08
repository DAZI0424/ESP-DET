"""Train ESPDet-Pico on the Welland v05 YOLO dataset and create a report."""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from train import Train  # noqa: E402


RUN_NAME = "welland_v05_yolo_espdet_pico_224_600e_p50"
RUN_DIR = ROOT / "runs" / "detect" / RUN_NAME
DATA_YAML = ROOT / "datasets" / "welland-v05-yolo" / "data-local.yaml"
DATA_SUMMARY = ROOT / "datasets" / "welland-v05-yolo" / "conversion_summary.json"
METRICS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def validate_dataset() -> dict[str, object]:
    """Fail fast on missing image/label pairs or malformed YOLO annotations."""
    dataset_root = DATA_YAML.parent
    class_names = (dataset_root / "classes.txt").read_text(encoding="utf-8").splitlines()
    split_summary: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    total_boxes = 0

    for split in ("train", "val"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = {path.stem: path for path in image_dir.iterdir() if path.is_file()}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        missing_labels = sorted(images.keys() - labels.keys())
        orphan_labels = sorted(labels.keys() - images.keys())
        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} 张图像缺少标签文件")
        if orphan_labels:
            errors.append(f"{split}: {len(orphan_labels)} 个标签没有对应图像")

        boxes = 0
        empty_labels = 0
        for label_path in labels.values():
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                empty_labels += 1
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                fields = line.split()
                location = f"{label_path}:{line_number}"
                if len(fields) != 5:
                    errors.append(f"{location}: 应有 5 列，实际 {len(fields)} 列")
                    continue
                try:
                    class_id = int(fields[0])
                    box = [float(value) for value in fields[1:]]
                except ValueError as exc:
                    errors.append(f"{location}: 无法解析数值（{exc}）")
                    continue
                if not 0 <= class_id < len(class_names):
                    errors.append(f"{location}: 类别 ID {class_id} 越界")
                if not all(0.0 <= value <= 1.0 for value in box):
                    errors.append(f"{location}: 坐标不在 [0, 1] 范围内")
                if box[2] <= 0.0 or box[3] <= 0.0:
                    errors.append(f"{location}: 宽高必须大于 0")
                boxes += 1

        total_boxes += boxes
        split_summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": boxes,
            "empty_labels": empty_labels,
        }

    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"数据集校验失败，共 {len(errors)} 个问题：\n{preview}")
    result = {"classes": len(class_names), "boxes": total_boxes, "splits": split_summary}
    print("DATASET_VALIDATION=" + json.dumps(result, ensure_ascii=False))
    return result


def _read_rows() -> list[dict[str, str]]:
    with (RUN_DIR / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, skipinitialspace=True))


def _value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def write_report(validation_results) -> Path:
    rows = _read_rows()
    if not rows:
        raise RuntimeError("results.csv 中没有训练轮次")

    best = max(rows, key=lambda row: _value(row, "metrics/mAP50-95(B)"))
    final = rows[-1]
    best_epoch = int(float(best["epoch"])) + 1
    completed_epochs = int(float(final["epoch"])) + 1
    stopped_early = completed_epochs < 600
    summary = json.loads(DATA_SUMMARY.read_text(encoding="utf-8"))

    names = validation_results.names
    maps = list(validation_results.box.maps)
    precision = list(validation_results.box.p)
    recall = list(validation_results.box.r)
    map50 = list(validation_results.box.ap50)
    class_indices = list(validation_results.box.ap_class_index)
    class_metrics = []
    for result_index, class_index in enumerate(class_indices):
        class_index = int(class_index)
        class_metrics.append(
            {
                "class_id": class_index,
                "class_name": names[class_index],
                "precision": float(precision[result_index]),
                "recall": float(recall[result_index]),
                "map50": float(map50[result_index]),
                "map50_95": float(maps[class_index]),
            }
        )
    class_metrics.sort(key=lambda item: item["map50_95"], reverse=True)

    total_seconds = float(final.get("time", "0") or 0)
    artifacts = [
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
    ]
    artifacts = [name for name in artifacts if (RUN_DIR / name).exists()]

    report_data = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(RUN_DIR),
        "model": str(ROOT / "cfg/models/espdet_pico.yaml"),
        "dataset": str(DATA_YAML),
        "planned_epochs": 600,
        "completed_epochs": completed_epochs,
        "patience": 50,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_metrics": {key: _value(best, key) for key in METRICS},
        "final_metrics": {key: _value(final, key) for key in METRICS},
        "training_seconds": total_seconds,
        "dataset_summary": summary,
        "class_metrics": class_metrics,
        "artifacts": artifacts,
    }
    (RUN_DIR / "training_report.json").write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# ESPDet-Pico Welland v05 模型训练报告",
        "",
        f"- 生成时间：{report_data['generated_at']}",
        f"- 模型架构：`{report_data['model']}`",
        f"- 数据集：`{report_data['dataset']}`",
        f"- 训练输出：`{RUN_DIR}`",
        "",
        "## 训练配置",
        "",
        "| 项目 | 配置 |",
        "| --- | --- |",
        "| 输入尺寸 | 224 × 224 |",
        "| 最大轮数 | 600 |",
        "| 早停耐心值 | 50 |",
        "| 批量大小 | AutoBatch（约 60% 可用显存） |",
        "| 设备 | NVIDIA CUDA GPU |",
        "| AMP | 开启 |",
        "| 优化器 | Ultralytics auto |",
        "| 学习率计划 | Cosine |",
        "| 随机种子 | 0（deterministic） |",
        "",
        "## 数据集概况",
        "",
        f"- 图像总数：{summary['images']}（训练 {summary['splits']['train']}，验证 {summary['splits']['val']}）",
        f"- 类别数：{len(summary['classes'])}",
        f"- 标注框：{summary['boxes']}；无目标背景图：{summary['images_without_boxes']}",
        f"- 无效框跳过：{summary['invalid_boxes_skipped']}；下载失败：{summary['download_failures']}",
        "",
        "## 训练结果",
        "",
        f"- 实际完成：{completed_epochs} 轮（{'触发早停' if stopped_early else '完成最大轮数'}）",
        f"- 最佳轮次：第 {best_epoch} 轮（按验证集 mAP50-95）",
        f"- 累计训练时间：{total_seconds / 3600:.2f} 小时",
        "",
        "| 检查点 | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| 最佳轮次 | " + " | ".join(f"{_value(best, key):.4f}" for key in METRICS) + " |",
        "| 最终轮次 | " + " | ".join(f"{_value(final, key):.4f}" for key in METRICS) + " |",
        "",
        "## 各类别验证指标（best.pt）",
        "",
        "| 类别 | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in class_metrics:
        lines.append(
            f"| {item['class_name']} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['map50']:.4f} | {item['map50_95']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 结论与使用建议",
            "",
            "- 部署、量化和后续测试应使用 `weights/best.pt`，而不是 `weights/last.pt`。",
            "- 当前结果来自训练期间使用的验证集；在正式上线前，仍应使用独立测试集和真实设备图像做最终验收。",
            "- 重点复核 mAP50-95 最低类别的漏检、误检和边界框质量，并针对困难场景补充数据。",
            "",
            "## 产物",
            "",
            *[f"- `{name}`" for name in artifacts],
            "- `training_report.md`",
            "- `training_report.json`",
            "",
            "## 运行环境",
            "",
            f"- Python：{platform.python_version()}",
            f"- 操作系统：{platform.platform()}",
            "",
        ]
    )
    output = RUN_DIR / "training_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


if __name__ == "__main__":
    multiprocessing.freeze_support()
    validate_dataset()
    results = Train(
        dataset=str(DATA_YAML),
        epochs=600,
        patience=50,
        imgsz=224,
        device=0,
        batch=-1,
        workers=0,
        cache=False,
        amp=True,
        cos_lr=True,
        close_mosaic=30,
        optimizer="auto",
        project=str(ROOT / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=False,
        save=True,
        save_period=25,
        plots=True,
        verbose=True,
        seed=0,
        deterministic=True,
    )
    report = write_report(results)
    print(f"TRAIN_SAVE_DIR={RUN_DIR}")
    print(f"TRAINING_REPORT={report}")
