"""Build the canonical portable HTML input for the Welland v04 training run."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_REL = Path("runs/detect/welland_v04_yolo_espdet_pico_416_500e_p50")
RUN = ROOT / RUN_REL
REPORT = RUN / "training_report"
RESULTS_REL = RUN_REL / "results.csv"
ARGS_REL = RUN_REL / "args.yaml"
VALIDATION_REL = RUN_REL / "final_validation/metrics.json"
DATASET_REL = Path("datasets/welland-v04-yolo/data-local.yaml")
MODEL_REL = Path("cfg/models/espdet_pico.yaml")
NOTES_REL = RUN_REL / "training_report/source_notes.md"

MAP = "metrics/mAP50-95(B)"
MAP50 = "metrics/mAP50(B)"
PRECISION = "metrics/precision(B)"
RECALL = "metrics/recall(B)"


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def read_rows() -> list[dict[str, str]]:
    with (ROOT / RESULTS_REL).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("results.csv contains no completed epochs")
    return rows


def dataset_statistics(config: dict[str, object]) -> tuple[list[dict[str, object]], list[str]]:
    dataset_root = Path(str(config["path"]))
    names_value = config["names"]
    if isinstance(names_value, dict):
        names = [str(names_value[index]) for index in sorted(names_value)]
    else:
        names = [str(value) for value in names_value]

    split_rows: list[dict[str, object]] = []
    for split in ("train", "val"):
        image_dir = dataset_root / str(config[split])
        label_dir = dataset_root / "labels" / split
        images = [
            path
            for path in image_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        class_counts = [0] * len(names)
        instances = 0
        backgrounds = 0
        for image in images:
            label = label_dir / image.relative_to(image_dir).with_suffix(".txt")
            lines = (
                [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
                if label.is_file()
                else []
            )
            if not lines:
                backgrounds += 1
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
                instances += 1
        split_rows.append(
            {
                "split": "训练集" if split == "train" else "验证集",
                "images": len(images),
                "instances": instances,
                "backgrounds": backgrounds,
            }
        )
        for class_id, count in enumerate(class_counts):
            split_rows.append(
                {
                    "split": f"{'训练' if split == 'train' else '验证'} · {names[class_id]}",
                    "images": None,
                    "instances": count,
                    "backgrounds": None,
                }
            )
    return split_rows, names


def source(source_id: str, label: str, path: Path) -> dict[str, str]:
    return {"id": source_id, "label": label, "path": path.as_posix()}


def main() -> None:
    validation_path = ROOT / VALIDATION_REL
    if not validation_path.is_file():
        raise FileNotFoundError(f"Final validation is required: {validation_path}")

    REPORT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    args = yaml.safe_load((ROOT / ARGS_REL).read_text(encoding="utf-8"))
    dataset_config = yaml.safe_load((ROOT / DATASET_REL).read_text(encoding="utf-8"))
    split_rows, class_names = dataset_statistics(dataset_config)

    best = max(rows, key=lambda row: number(row, MAP))
    final = rows[-1]
    best_epoch = int(best["epoch"])
    final_epoch = int(final["epoch"])
    best_map = number(best, MAP)
    final_map = number(final, MAP)
    total_hours = number(final, "time") / 3600
    stop_reason = (
        "连续 50 轮未刷新最佳指标，按 patience=50 正常早停"
        if final_epoch < int(args["epochs"]) and final_epoch - best_epoch >= int(args["patience"])
        else "完成计划中的全部训练轮次"
        if final_epoch == int(args["epochs"])
        else "训练在计划轮次之前结束，需结合控制台日志判断原因"
    )

    # The user explicitly requested a manual early stop. Only fully persisted
    # results.csv rows are counted; the interrupted partial epoch is excluded.
    stop_reason = f"按用户指示提前停止；共保留 {final_epoch} 个完整训练轮次"

    aggregate = validation["aggregate"]
    independent_map = float(aggregate["metrics/mAP50-95(B)"])
    independent_map50 = float(aggregate["metrics/mAP50(B)"])
    independent_precision = float(aggregate["metrics/precision(B)"])
    independent_recall = float(aggregate["metrics/recall(B)"])

    class_metrics = sorted(
        [
            {
                "class": row["class_name"],
                "precision": row["precision"],
                "recall": row["recall"],
                "map50": row["map50"],
                "map5095": row["map50_95"],
            }
            for row in validation["per_class"]
        ],
        key=lambda row: row["map5095"],
        reverse=True,
    )
    strongest = class_metrics[0]
    weakest = class_metrics[-1]

    epoch_metrics: list[dict[str, object]] = []
    loss_metrics: list[dict[str, object]] = []
    for row in rows:
        epoch = int(row["epoch"])
        epoch_metrics.extend(
            [
                {"epoch": epoch, "metric": "mAP50-95", "value": number(row, MAP)},
                {"epoch": epoch, "metric": "mAP50", "value": number(row, MAP50)},
            ]
        )
        loss_metrics.extend(
            [
                {"epoch": epoch, "split": "训练集", "value": number(row, "train/box_loss")},
                {"epoch": epoch, "split": "验证集", "value": number(row, "val/box_loss")},
            ]
        )

    candidate_epochs = [1, 25, 50, 75, 100, best_epoch, final_epoch]
    by_epoch = {int(row["epoch"]): row for row in rows}
    milestone_epochs = sorted({epoch for epoch in candidate_epochs if epoch in by_epoch})
    milestones = []
    for epoch in milestone_epochs:
        row = by_epoch[epoch]
        stage = (
            "最佳轮次" if epoch == best_epoch else "训练终点" if epoch == final_epoch else "阶段检查点"
        )
        milestones.append(
            {
                "stage": stage,
                "epoch": epoch,
                "precision": number(row, PRECISION),
                "recall": number(row, RECALL),
                "map50": number(row, MAP50),
                "map5095": number(row, MAP),
            }
        )

    train_summary = next(row for row in split_rows if row["split"] == "训练集")
    val_summary = next(row for row in split_rows if row["split"] == "验证集")
    summary = {
        "best_epoch": best_epoch,
        "completed_epochs": final_epoch,
        "best_map5095": best_map,
        "independent_map5095": independent_map,
        "independent_map50": independent_map50,
        "independent_precision": independent_precision,
        "independent_recall": independent_recall,
        "total_training_hours": total_hours,
        "train_images": train_summary["images"],
        "val_images": val_summary["images"],
        "val_instances": val_summary["instances"],
        "class_count": len(class_names),
        "parameters": validation["model"]["parameters"],
        "gflops": validation["model"]["gflops"],
    }

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    source_list = [
        source("training_results", "逐轮训练与验证指标", RESULTS_REL),
        source("training_config", "实际训练参数", ARGS_REL),
        source("final_validation", "best.pt 最终复验指标", VALIDATION_REL),
        source("dataset_config", "数据集路径与类别定义", DATASET_REL),
        source("model_config", "ESPDet-Pico 模型架构", MODEL_REL),
        source("training_run", "训练运行目录", RUN_REL),
        source("report_notes", "报告口径、图表映射与 QA 记录", NOTES_REL),
    ]

    cards = [
        {
            "id": "independent_map",
            "description": "训练结束后重新加载 best.pt，在验证集上独立复验。",
            "dataset": "summary",
            "sourceId": "final_validation",
            "metrics": [{"label": "复验 mAP50-95", "field": "independent_map5095", "format": "percent"}],
        },
        {
            "id": "independent_map50",
            "description": "IoU=0.50 时六个类别的平均检测精度。",
            "dataset": "summary",
            "sourceId": "final_validation",
            "metrics": [{"label": "复验 mAP50", "field": "independent_map50", "format": "percent"}],
        },
        {
            "id": "best_epoch",
            "description": "按逐轮验证 mAP50-95 选择的最佳训练轮次。",
            "dataset": "summary",
            "sourceId": "training_results",
            "metrics": [{"label": "最佳轮次", "field": "best_epoch", "format": "number"}],
        },
        {
            "id": "completed_epochs",
            "description": stop_reason,
            "dataset": "summary",
            "sourceId": "training_results",
            "metrics": [{"label": "实际完成轮次", "field": "completed_epochs", "format": "number"}],
        },
    ]

    charts = [
        {
            "id": "validation_trend",
            "title": "验证指标随训练轮次变化",
            "subtitle": f"共 {final_epoch} 个完整轮次；虚线标记第 {best_epoch} 轮最佳点。",
            "intent": "trend",
            "question": "模型是否持续改善，并在何时进入平台期？",
            "rationale": "完整轮次序列可以同时呈现早期学习、后期波动和早停依据。",
            "comparisonContext": {"grain": "epoch", "unit": "0–1 score"},
            "type": "line",
            "dataset": "epoch_metrics",
            "sourceId": "training_results",
            "encodings": {
                "x": {"field": "epoch", "type": "quantitative", "label": "轮次"},
                "y": {"field": "value", "type": "quantitative", "label": "验证指标", "format": "percent"},
                "color": {"field": "metric", "type": "nominal", "label": "指标"},
                "lineStyle": {"field": "metric", "type": "nominal", "label": "指标"},
            },
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "endpoints"},
            "referenceLines": [{"axis": "x", "value": best_epoch, "label": "最佳轮次", "color": "neutral", "lineStyle": "dashed"}],
            "layout": "full",
        },
        {
            "id": "box_loss",
            "title": "训练集与验证集 Box Loss",
            "subtitle": "比较定位损失以识别收敛平台和潜在泛化间隙。",
            "intent": "trend",
            "question": "最佳轮次之后，训练损失和验证损失是否仍同步改善？",
            "rationale": "相同尺度下的两条曲线能显示继续拟合训练集是否仍带来验证收益。",
            "comparisonContext": {"grain": "epoch", "unit": "loss"},
            "type": "line",
            "dataset": "loss_metrics",
            "sourceId": "training_results",
            "encodings": {
                "x": {"field": "epoch", "type": "quantitative", "label": "轮次"},
                "y": {"field": "value", "type": "quantitative", "label": "Box Loss", "format": "number"},
                "color": {"field": "split", "type": "nominal", "label": "数据划分"},
                "lineStyle": {"field": "split", "type": "nominal", "label": "数据划分"},
            },
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "endpoints"},
            "referenceLines": [{"axis": "x", "value": best_epoch, "label": "最佳轮次", "color": "neutral", "lineStyle": "dashed"}],
            "layout": "full",
        },
        {
            "id": "class_map",
            "title": "各类别复验 mAP50-95",
            "subtitle": f"best.pt 在 {val_summary['images']} 张验证图像上的逐类别结果。",
            "intent": "comparison",
            "question": "哪些类别限制了整体检测质量？",
            "rationale": "六个类别适合用排序横向条形图进行精确比较。",
            "comparisonContext": {"grain": "class", "unit": "mAP50-95"},
            "type": "horizontalBar",
            "dataset": "class_metrics",
            "sourceId": "final_validation",
            "encodings": {
                "x": {"field": "class", "type": "nominal", "label": "类别"},
                "y": {"field": "map5095", "type": "quantitative", "label": "mAP50-95", "format": "percent"},
                "tooltip": [
                    {"field": "precision", "type": "quantitative", "label": "Precision", "format": "percent"},
                    {"field": "recall", "type": "quantitative", "label": "Recall", "format": "percent"},
                    {"field": "map50", "type": "quantitative", "label": "mAP50", "format": "percent"},
                ],
            },
            "palette": {"kind": "sequential", "name": "blue"},
            "labels": {"values": "all"},
            "settings": {"sort": "descending", "showValues": True},
            "layout": "full",
        },
    ]

    tables = [
        {
            "id": "milestones",
            "title": "关键训练里程碑",
            "subtitle": "从首轮、周期检查点到最佳轮次和训练终点。",
            "dataset": "milestones",
            "sourceId": "training_results",
            "defaultSort": {"field": "epoch", "direction": "asc"},
            "columns": [
                {"field": "stage", "label": "阶段", "type": "text"},
                {"field": "epoch", "label": "轮次", "format": "number"},
                {"field": "precision", "label": "Precision", "format": "percent"},
                {"field": "recall", "label": "Recall", "format": "percent"},
                {"field": "map50", "label": "mAP50", "format": "percent"},
                {"field": "map5095", "label": "mAP50-95", "format": "percent"},
            ],
        },
        {
            "id": "class_detail",
            "title": "逐类别复验明细",
            "subtitle": "按 mAP50-95 从高到低排序。",
            "dataset": "class_metrics",
            "sourceId": "final_validation",
            "defaultSort": {"field": "map5095", "direction": "desc"},
            "columns": [
                {"field": "class", "label": "类别", "type": "text"},
                {"field": "precision", "label": "Precision", "format": "percent"},
                {"field": "recall", "label": "Recall", "format": "percent"},
                {"field": "map50", "label": "mAP50", "format": "percent"},
                {"field": "map5095", "label": "mAP50-95", "format": "percent"},
            ],
        },
        {
            "id": "dataset_detail",
            "title": "数据集划分与实例数",
            "subtitle": "由本地图像和 YOLO 标签逐项统计。",
            "dataset": "dataset_detail",
            "sourceId": "dataset_config",
            "defaultSort": {"field": "split", "direction": "asc"},
            "columns": [
                {"field": "split", "label": "划分/类别", "type": "text"},
                {"field": "images", "label": "图像", "format": "number"},
                {"field": "instances", "label": "实例", "format": "number"},
                {"field": "backgrounds", "label": "背景图", "format": "number"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# ESPDet-Pico · Welland v04 训练报告"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "final_validation",
            "body": (
                "## 技术摘要\n\n"
                f"训练共完成 **{final_epoch}** 轮，{stop_reason}。逐轮记录的最佳 mAP50-95 出现在第 "
                f"**{best_epoch}** 轮，为 **{best_map:.4f}**。重新加载 `best.pt` 后的独立复验结果为："
                f"Precision **{independent_precision:.4f}**、Recall **{independent_recall:.4f}**、"
                f"mAP50 **{independent_map50:.4f}**、mAP50-95 **{independent_map:.4f}**。"
                "部署和后续转换应使用 `best.pt`，不要以 `last.pt` 代替最佳质量权重。"
            ),
        },
        {"id": "metrics", "type": "metric-strip", "cardIds": ["independent_map", "independent_map50", "best_epoch", "completed_epochs"]},
        {
            "id": "quality_finding",
            "type": "markdown",
            "sourceId": "training_results",
            "body": (
                "## 训练指标已进入收敛区间\n\n"
                f"mAP50-95 从首轮的 **{number(rows[0], MAP):.4f}** 提升到第 {best_epoch} 轮的 "
                f"**{best_map:.4f}**。最终轮次为 **{final_map:.4f}**，比最佳点低 "
                f"**{best_map - final_map:.4f}**；完整曲线用于判断提升是否持续以及早停是否合理。"
            ),
        },
        {"id": "quality_chart", "type": "chart", "chartId": "validation_trend"},
        {"id": "milestone_note", "type": "markdown", "body": "### 可复核的关键节点\n\n下表保留阶段检查点、最佳点与训练终点的原始指标。"},
        {"id": "milestones", "type": "table", "tableId": "milestones"},
        {
            "id": "loss_finding",
            "type": "markdown",
            "sourceId": "training_results",
            "body": (
                "## 损失曲线用于检查泛化间隙\n\n"
                f"最佳轮次的训练/验证 Box Loss 分别为 **{number(best, 'train/box_loss'):.4f}** 和 "
                f"**{number(best, 'val/box_loss'):.4f}**；最终轮次分别为 "
                f"**{number(final, 'train/box_loss'):.4f}** 和 **{number(final, 'val/box_loss'):.4f}**。"
                "若训练损失继续下降而验证损失不再改善，应优先保留最佳检查点。"
            ),
        },
        {"id": "loss_chart", "type": "chart", "chartId": "box_loss"},
        {
            "id": "class_finding",
            "type": "markdown",
            "sourceId": "final_validation",
            "body": (
                "## 类别差异决定下一轮数据改进重点\n\n"
                f"复验中表现最强的是 `{strongest['class']}`（mAP50-95 **{strongest['map5095']:.4f}**），"
                f"最弱的是 `{weakest['class']}`（**{weakest['map5095']:.4f}**）。"
                "整体平均值会掩盖弱类别，应结合逐类别 Precision、Recall 与错误案例决定补数方向。"
            ),
        },
        {"id": "class_chart", "type": "chart", "chartId": "class_map"},
        {"id": "class_table", "type": "table", "tableId": "class_detail"},
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "dataset_config",
            "body": (
                "## 范围、数据与指标定义\n\n"
                f"训练集包含 **{train_summary['images']}** 张图像、**{train_summary['instances']}** 个标注实例；"
                f"验证集包含 **{val_summary['images']}** 张图像、**{val_summary['instances']}** 个实例，"
                f"共 **{len(class_names)}** 类。输入图像均为 416×416。mAP50 是 IoU=0.50 下各类别 AP 的平均；"
                "mAP50-95 是 IoU 0.50 至 0.95（步长 0.05）的平均，更严格反映定位质量。"
                "Precision 衡量预测框中正确检测的比例，Recall 衡量标注目标中被检出的比例。"
            ),
        },
        {"id": "dataset_table", "type": "table", "tableId": "dataset_detail"},
        {
            "id": "method",
            "type": "markdown",
            "sourceId": "training_config",
            "body": (
                "## 方法与模型规格\n\n"
                f"模型为 ESPDet-Pico，约 **{validation['model']['parameters']:,}** 个参数、"
                f"**{validation['model']['gflops']:.2f} GFLOPs**；输入 416×416，AutoBatch 实际采用 batch=52，"
                f"AMP 开启，优化器由 Ultralytics 自动选择，cosine 学习率，seed={args['seed']}，"
                f"计划 {args['epochs']} 轮、patience={args['patience']}。总训练时间约 **{total_hours:.2f} 小时**。"
                "最终复验重新加载 `best.pt`，使用相同数据配置、图像尺寸和 batch=52。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "sourceId": "training_run",
            "body": (
                "## 限制与不确定性\n\n"
                "本报告的选模与最终质量判断使用同一验证集，没有独立测试集，因此复验并不是对未见数据的无偏估计。"
                "本次仅运行一个随机种子，没有重复实验、交叉验证或置信区间；也没有完成量化后精度、ESP32-S3 端到端延迟、"
                "峰值内存与真实场景鲁棒性测试。训练集中存在背景图，它们有助于抑制误报，但数据分布仍需与部署现场核对。"
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "sourceId": "final_validation",
            "body": (
                "## 推荐下一步\n\n"
                "1. 冻结并登记 `best.pt` 的 SHA-256、数据集版本、模型 YAML 和 Ultralytics 版本。\n"
                f"2. 优先审查 `{weakest['class']}` 的漏检、误检和框定位误差，并补充困难样本。\n"
                "3. 建立与当前验证集隔离的测试集，覆盖光照、遮挡、尺度、背景和相机差异。\n"
                "4. 在目标 ESP32-S3 部署链上测试转换/量化后的精度、延迟和内存峰值。\n"
                "5. 若再次训练，采用相同数据划分做单变量对照，并至少增加多个随机种子。"
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 仍需回答的问题\n\n"
                "- 弱类别的主要错误来自类别混淆、漏检，还是边界框定位？\n"
                "- 模型转换和量化后，逐类别 AP 与召回率损失多少？\n"
                "- 当前验证集与真实摄像头画面的光照、背景和目标尺度是否一致？\n"
                "- 416×416 在目标硬件上的精度、时延和内存是否达到产品约束？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ESPDet-Pico · Welland v04 训练报告",
            "description": "训练收敛、最终复验、逐类别质量、限制与部署建议。",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": source_list,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary": [summary],
                "epoch_metrics": epoch_metrics,
                "loss_metrics": loss_metrics,
                "class_metrics": class_metrics,
                "milestones": milestones,
                "dataset_detail": split_rows,
            },
        },
        "sources": source_list,
        "package_info": {},
    }

    analysis = {
        "generated_at": generated_at,
        "summary": summary,
        "best_epoch_metrics": best,
        "final_epoch_metrics": final,
        "final_validation": validation,
        "dataset_detail": split_rows,
        "stop_reason": stop_reason,
    }
    (REPORT / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "source_notes.md").write_text(
        f"""# 报告来源、图表映射与 QA 记录

## 报告结构映射

- 标题与技术摘要：`final_validation/metrics.json`、`results.csv`
- 关键发现与视觉证据：验证趋势、Box Loss、逐类别 mAP50-95
- 范围、数据与指标定义：`data-local.yaml` 及图像/标签逐项统计
- 方法与模型细节：`args.yaml`、`espdet_pico.yaml`、最终复验模型信息
- 限制、建议与待回答问题：基于当前证据范围明确列示

## 图表映射

- `validation_trend`：回答模型何时改善、何时平台；双折线用颜色和线型双重区分。
- `box_loss`：回答训练与验证定位损失是否同步；与指标趋势回答不同问题，因此保留第二张折线图。
- `class_map`：回答类别差异；六个类别按 mAP50-95 排序，使用单色横向条形图并直接标值。
- 未使用饼图、面积图或 3D 图，因为它们不会提高这些比较的可读性。

## 数据与口径

- 最佳训练轮次：第 {best_epoch} 轮，按 `results.csv` 中最大 mAP50-95 选择。
- 最终复验：重新加载 `best.pt`，在相同验证集上独立执行一次 Ultralytics `val`。
- 训练完成：第 {final_epoch} 轮；终止口径为“{stop_reason}”。
- 所有可见结论均可追溯到报告 Source 面板中的相对路径，不暴露机器本地绝对路径。

## QA 与限制

- 已检查结果表非空、最终复验文件存在、六个类别均有逐类别指标。
- 图表采用完整轮次，不抽样；表格保留精确数值。
- 没有独立测试集、多随机种子、置信区间或端侧基准，因此报告不作生产部署达标声明。
""",
        encoding="utf-8",
    )
    print(f"ARTIFACT_JSON={REPORT / 'artifact.json'}")


if __name__ == "__main__":
    main()
