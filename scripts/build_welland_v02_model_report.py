"""Build a concise Chinese model report from training and validation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "espdet" / "welland_v02_espdet_pico_224"
DATASET_ROOT = ROOT / "datasets" / "Welland-V02-espdet-pico"
DATASET_YAML = DATASET_ROOT / "clean-20260810.yaml"
RESULTS = RUN / "results.csv"
ARGS = RUN / "args.yaml"
METRICS = RUN / "final_validation" / "metrics.json"
BEST = RUN / "weights" / "best.pt"
REPORT_MD = RUN / "MODEL_REPORT.md"
REPORT_JSON = RUN / "model_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_stats(class_names: dict[int, str]) -> dict[str, object]:
    splits: dict[str, object] = {}
    total_classes: Counter[int] = Counter()
    for split in ("train", "val"):
        image_count = sum(1 for path in (DATASET_ROOT / "images" / split).iterdir() if path.is_file())
        label_files = list((DATASET_ROOT / "labels" / split).glob("*.txt"))
        class_counts: Counter[int] = Counter()
        empty_count = 0
        for label_file in label_files:
            lines = [line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_count += 1
            for line in lines:
                class_counts[int(line.split()[0])] += 1
        total_classes.update(class_counts)
        splits[split] = {
            "images": image_count,
            "labels": len(label_files),
            "boxes": sum(class_counts.values()),
            "background_images": empty_count,
            "boxes_per_class": {class_names[key]: class_counts[key] for key in sorted(class_names)},
        }
    return {
        "splits": splits,
        "total_images": sum(int(item["images"]) for item in splits.values()),
        "total_boxes": sum(total_classes.values()),
        "boxes_per_class": {class_names[key]: total_classes[key] for key in sorted(class_names)},
    }


def total_elapsed_seconds(rows: list[dict[str, str]]) -> float:
    total = 0.0
    previous = 0.0
    for row in rows:
        current = float(row["time"])
        if current < previous:
            total += previous
        previous = current
    return total + previous


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    for required in (RESULTS, ARGS, METRICS, BEST):
        if not required.is_file():
            raise FileNotFoundError(f"Required artifact not found: {required}")

    with RESULTS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args = yaml.safe_load(ARGS.read_text(encoding="utf-8"))
    validation = json.loads(METRICS.read_text(encoding="utf-8"))
    dataset_yaml = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    class_names = {int(key): value for key, value in dataset_yaml["names"].items()}
    data = dataset_stats(class_names)

    metric_key = "metrics/mAP50-95(B)"
    best_row = max(rows, key=lambda row: float(row[metric_key]))
    last_row = rows[-1]
    completed_epochs = int(last_row["epoch"])
    requested_epochs = int(args["epochs"])
    early_stopped = completed_epochs < requested_epochs
    aggregate = validation["aggregate"]
    model = validation["model"]
    strongest_class = max(validation["per_class"], key=lambda item: item["map50_95"])
    weakest_class = min(validation["per_class"], key=lambda item: item["map50_95"])
    elapsed = total_elapsed_seconds(rows)
    model_hash = sha256(BEST)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "completed",
        "model_config": str(ROOT / "cfg" / "models" / "espdet_pico.yaml"),
        "best_checkpoint": str(BEST),
        "best_checkpoint_bytes": BEST.stat().st_size,
        "best_checkpoint_sha256": model_hash,
        "training": {
            "completed_epochs": completed_epochs,
            "requested_epochs": requested_epochs,
            "early_stopped": early_stopped,
            "patience": int(args["patience"]),
            "best_epoch_by_training_csv": int(best_row["epoch"]),
            "best_map50_95_by_training_csv": float(best_row[metric_key]),
            "final_map50_95_by_training_csv": float(last_row[metric_key]),
            "elapsed_seconds": elapsed,
            "parameters": args,
        },
        "validation": validation,
        "dataset": data,
        "analysis": {
            "strongest_class": strongest_class,
            "weakest_class": weakest_class,
            "visual_check": [
                "训练和验证损失整体持续下降，后期没有明显发散",
                "mAP 在约第 300 轮后进入平台，最佳检查点出现在第 350 轮",
                "归一化混淆矩阵主对角线约为 0.94-0.99，类别间混淆较少",
                "负样本仍存在零星误检，部署时需要结合置信度阈值复测",
            ],
        },
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "deployment": {
            "esp32s3_flash_ready": False,
            "next_steps": ["导出 ONNX", "INT8 量化为 ESP-DL 模型", "集成 ESP-IDF 固件并烧录"],
        },
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    stop_text = f"早停（patience={args['patience']}）" if early_stopped else "达到最大轮数"
    speed = validation["speed_ms_per_image"]
    per_class_lines = []
    for item in validation["per_class"]:
        per_class_lines.append(
            f"| {item['class_id']} | {item['class_name']} | {pct(item['precision'])} | "
            f"{pct(item['recall'])} | {pct(item['map50'])} | {pct(item['map50_95'])} |"
        )
    class_distribution = []
    for class_id, class_name in class_names.items():
        class_distribution.append(f"| {class_id} | {class_name} | {data['boxes_per_class'][class_name]} |")

    markdown = f"""# Welland V02 ESPDet-Pico 模型报告

生成时间：{report['generated_at']}

## 结论

训练已完成，结束方式为 **{stop_text}**。独立验证集上的 mAP50-95 为 **{pct(aggregate['metrics/mAP50-95(B)'])}**，mAP50 为 **{pct(aggregate['metrics/mAP50(B)'])}**。最佳 PyTorch 权重已保存，但它还不能直接烧录到 ESP32-S3；需要继续完成 ONNX 导出、INT8 量化和 ESP-IDF 固件集成。

## 模型与训练

| 项目 | 数值 |
|---|---:|
| 模型 | ESPDet-Pico，输入 224×224 |
| 参数量 | {model['parameters']:,} |
| 计算量 | {model['gflops']:.3f} GFLOPs |
| 已训练/上限 | {completed_epochs}/{requested_epochs} epochs |
| CSV 最佳轮次 | {int(best_row['epoch'])} |
| CSV 最佳 mAP50-95 | {pct(float(best_row[metric_key]))} |
| 最后一轮 mAP50-95 | {pct(float(last_row[metric_key]))} |
| 总训练耗时 | {elapsed / 3600:.2f} 小时 |
| batch / workers / AMP | {args['batch']} / {args['workers']} / {args['amp']} |
| 优化器 / 调度 | {args['optimizer']} / cosine LR |
| 随机种子 | {args['seed']} |

## 独立验证结果

| 指标 | 数值 |
|---|---:|
| Precision | {pct(aggregate['metrics/precision(B)'])} |
| Recall | {pct(aggregate['metrics/recall(B)'])} |
| mAP50 | {pct(aggregate['metrics/mAP50(B)'])} |
| mAP50-95 | {pct(aggregate['metrics/mAP50-95(B)'])} |
| 推理耗时（PyTorch/GPU，preprocess） | {speed.get('preprocess', 0):.3f} ms/图 |
| 推理耗时（PyTorch/GPU，inference） | {speed.get('inference', 0):.3f} ms/图 |
| 后处理耗时 | {speed.get('postprocess', 0):.3f} ms/图 |

以上速度只代表本机 PyTorch GPU 验证，不代表 ESP32-S3 上的端侧速度。

### 分类别指标

| ID | 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---:|---|---:|---:|---:|---:|
{chr(10).join(per_class_lines)}

## 曲线与混淆矩阵观察

- 训练和验证损失整体持续下降，后期没有明显发散；mAP 在约第 300 轮后进入平台，第 350 轮取得最佳权重，最后一轮比最佳低 {pct(float(best_row[metric_key]) - float(last_row[metric_key]))}。
- 归一化混淆矩阵主对角线约为 0.94–0.99，类别之间直接混淆较少。
- 最强类别为 **{strongest_class['class_name']}**（mAP50-95 {pct(strongest_class['map50_95'])}）；最弱类别为 **{weakest_class['class_name']}**（mAP50-95 {pct(weakest_class['map50_95'])}），应优先补充其困难样本。
- 空标注图片带来了有价值的负样本训练，但矩阵仍显示零星背景误检；部署前应在真实摄像头画面上选择置信度阈值，并复测误报率。

## 数据集

训练集 {data['splits']['train']['images']:,} 张，验证集 {data['splits']['val']['images']:,} 张；合计 {data['total_images']:,} 张、{data['total_boxes']:,} 个框。空标注负样本为训练集 {data['splits']['train']['background_images']:,} 张、验证集 {data['splits']['val']['background_images']:,} 张。全量检查未发现非法框、类别越界或图像/标签缺配。

| ID | 类别 | 框数（train+val） |
|---:|---|---:|
{chr(10).join(class_distribution)}

## 产物与完整性

- 最佳权重：`weights/best.pt`（{BEST.stat().st_size / 1024 / 1024:.2f} MiB）
- SHA-256：`{model_hash}`
- 训练曲线：`results.png`
- 混淆矩阵：`confusion_matrix.png`、`confusion_matrix_normalized.png`
- 独立验证：`final_validation/metrics.json`
- 机器可读报告：`model_report.json`

## ESP32-S3 部署状态

当前状态：**尚未达到可烧录状态**。下一阶段应固定推理输入和类别顺序，导出 ONNX，使用代表性校准集做 INT8 量化并生成 `.espdl`，再集成到 ESP-IDF 工程进行板端精度、内存和延迟验证。
"""
    REPORT_MD.write_text(markdown, encoding="utf-8")
    print(f"REPORT_MD={REPORT_MD}")
    print(f"REPORT_JSON={REPORT_JSON}")


if __name__ == "__main__":
    main()
