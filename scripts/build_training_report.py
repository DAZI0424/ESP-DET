"""Build the canonical portable training report artifact for the Welland run."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_REL = Path("runs/espdet/welland_v01_espdet_pico_224")
RUN = ROOT / RUN_REL
REPORT = RUN / "training_report"
RESULTS_REL = RUN_REL / "results.csv"
ARGS_REL = RUN_REL / "args.yaml"
LOG_REL = RUN_REL / "resume-safe-20260806-154053.out.log"
DATASET_REL = Path("datasets/welland-v01-espdet-pico/welland-v01.yaml")
NOTES_REL = RUN_REL / "training_report/monitoring_notes.md"
EVIDENCE_REL = RUN_REL / "training_report/evidence.sqlite"

EPOCH_SQL = """SELECT epoch, map5095, map50, precision, recall,
       train_box_loss, val_box_loss
FROM epoch_results
ORDER BY epoch"""
CLASS_SQL = """SELECT class, images, instances, precision, recall, map50, map5095
FROM class_metrics
ORDER BY map5095 DESC"""

METRIC_MAP = "metrics/mAP50-95(B)"
METRIC_MAP50 = "metrics/mAP50(B)"
METRIC_PRECISION = "metrics/precision(B)"
METRIC_RECALL = "metrics/recall(B)"


def f(row: dict[str, str], field: str) -> float:
    return float(row[field])


def load_rows() -> list[dict[str, str]]:
    with (ROOT / RESULTS_REL).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_final_class_metrics() -> list[dict[str, object]]:
    text = (ROOT / LOG_REL).read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    final_validation = text.rsplit("Validating ", 1)[-1]
    names = [
        "apple",
        "strawberry",
        "cherry tomato",
        "egg",
        "lettuce",
        "banana",
        "chicken breast",
    ]
    parsed: list[dict[str, object]] = []
    for name in names:
        match = re.search(
            rf"^\s*{re.escape(name)}\s+(\d+)\s+(\d+)\s+"
            r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
            final_validation,
            re.MULTILINE,
        )
        if not match:
            raise RuntimeError(f"Could not parse final validation metrics for {name}")
        images, instances, precision, recall, map50, map5095 = match.groups()
        parsed.append(
            {
                "class": name,
                "images": int(images),
                "instances": int(instances),
                "precision": float(precision),
                "recall": float(recall),
                "map50": float(map50),
                "map5095": float(map5095),
            }
        )
    return parsed


def source(source_id: str, label: str, path: Path) -> dict[str, str]:
    return {"id": source_id, "label": label, "path": path.as_posix()}


def sql_source(
    source_id: str, label: str, path: Path, sql: str, description: str, generated_at: str
) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "path": path.as_posix(),
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "sql": sql,
            "description": description,
            "executed_at": generated_at,
        },
    }


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    by_epoch = {int(row["epoch"]): row for row in rows}
    best = max(rows, key=lambda row: f(row, METRIC_MAP))
    best_epoch = int(best["epoch"])
    final = rows[-1]
    final_epoch = int(final["epoch"])
    pre_resume = max(
        (row for row in rows if int(row["epoch"]) <= 129),
        key=lambda row: f(row, METRIC_MAP),
    )
    pre_resume_epoch = int(pre_resume["epoch"])
    class_metrics = parse_final_class_metrics()

    best_map = f(best, METRIC_MAP)
    pre_resume_map = f(pre_resume, METRIC_MAP)
    final_map = f(final, METRIC_MAP)
    absolute_gain = best_map - pre_resume_map
    relative_gain = absolute_gain / pre_resume_map
    final_gap = best_map - final_map
    initial_seconds = f(by_epoch[129], "time")
    resume_seconds = f(final, "time")
    total_seconds = initial_seconds + resume_seconds
    resumed_epochs = final_epoch - 129

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    evidence_path = ROOT / EVIDENCE_REL
    if evidence_path.exists():
        evidence_path.unlink()
    connection = sqlite3.connect(evidence_path)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE epoch_results (
            epoch INTEGER PRIMARY KEY,
            map5095 REAL, map50 REAL, precision REAL, recall REAL,
            train_box_loss REAL, val_box_loss REAL
        )"""
    )
    connection.executemany(
        "INSERT INTO epoch_results VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                int(row["epoch"]),
                f(row, METRIC_MAP),
                f(row, METRIC_MAP50),
                f(row, METRIC_PRECISION),
                f(row, METRIC_RECALL),
                f(row, "train/box_loss"),
                f(row, "val/box_loss"),
            )
            for row in rows
        ],
    )
    connection.execute(
        """CREATE TABLE class_metrics (
            class TEXT PRIMARY KEY,
            images INTEGER, instances INTEGER,
            precision REAL, recall REAL, map50 REAL, map5095 REAL
        )"""
    )
    connection.executemany(
        "INSERT INTO class_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row["class"], row["images"], row["instances"], row["precision"],
                row["recall"], row["map50"], row["map5095"],
            )
            for row in class_metrics
        ],
    )
    connection.commit()
    queried_epochs = [dict(row) for row in connection.execute(EPOCH_SQL)]
    class_metrics = [dict(row) for row in connection.execute(CLASS_SQL)]
    connection.close()

    epoch_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    for row in queried_epochs:
        epoch = int(row["epoch"])
        epoch_rows.extend(
            [
                {"epoch": epoch, "metric": "mAP50-95", "value": row["map5095"]},
                {"epoch": epoch, "metric": "mAP50", "value": row["map50"]},
            ]
        )
        loss_rows.extend(
            [
                {"epoch": epoch, "split": "训练集", "value": row["train_box_loss"]},
                {"epoch": epoch, "split": "验证集", "value": row["val_box_loss"]},
            ]
        )

    milestone_epochs = [1, 129, 130, 150, 200, best_epoch, final_epoch]
    milestone_labels = {
        1: "首次完整轮次",
        129: "中断前检查点",
        130: "安全续训开始",
        150: "续训首次显著突破",
        200: "训练过半",
        best_epoch: "最佳轮次",
        final_epoch: "早停轮次",
    }
    milestones = [
        {
            "stage": milestone_labels[epoch],
            "epoch": epoch,
            "map5095": f(by_epoch[epoch], METRIC_MAP),
            "map50": f(by_epoch[epoch], METRIC_MAP50),
            "precision": f(by_epoch[epoch], METRIC_PRECISION),
            "recall": f(by_epoch[epoch], METRIC_RECALL),
        }
        for epoch in milestone_epochs
    ]

    summary = {
        "best_epoch": best_epoch,
        "completed_epochs": final_epoch,
        "resumed_epochs": resumed_epochs,
        "best_map5095": best_map,
        "best_map50": f(best, METRIC_MAP50),
        "best_precision": f(best, METRIC_PRECISION),
        "best_recall": f(best, METRIC_RECALL),
        "pre_resume_best_epoch": pre_resume_epoch,
        "pre_resume_best_map5095": pre_resume_map,
        "absolute_map_gain": absolute_gain,
        "relative_map_gain": relative_gain,
        "final_map5095": final_map,
        "final_gap_to_best": final_gap,
        "total_training_hours": total_seconds / 3600,
        "resume_training_hours": resume_seconds / 3600,
        "resume_seconds_per_epoch": resume_seconds / resumed_epochs,
        "best_train_box_loss": f(best, "train/box_loss"),
        "best_val_box_loss": f(best, "val/box_loss"),
        "final_train_box_loss": f(final, "train/box_loss"),
        "final_val_box_loss": f(final, "val/box_loss"),
        "train_images": 7420,
        "val_images": 1309,
        "val_instances": 1366,
        "class_count": 7,
        "model_parameters": 358897,
        "model_gflops": 1.3,
    }

    source_list = [
        sql_source(
            "training_results",
            "逐轮训练与验证指标证据库",
            EVIDENCE_REL,
            EPOCH_SQL,
            "由 results.csv 装载并实际执行的逐轮指标查询；长表图表数据仅做确定性重排。",
            generated_at,
        ),
        source("raw_training_results", "原始逐轮训练结果", RESULTS_REL),
        source("training_config", "训练配置", ARGS_REL),
        sql_source(
            "final_validation",
            "最终类别验证证据库",
            EVIDENCE_REL,
            CLASS_SQL,
            "从最终 best.pt 验证日志解析后装载并实际执行的类别指标查询。",
            generated_at,
        ),
        source("raw_final_validation", "续训、早停与最终验证日志", LOG_REL),
        source("dataset_config", "数据集类别定义", DATASET_REL),
        source("monitoring_notes", "续训监控与风险调整记录", NOTES_REL),
        source("training_run", "完整训练运行目录", RUN_REL),
    ]

    cards = [
        {
            "id": "best_map",
            "description": "验证集上所有 IoU=0.50:0.95 阈值的平均检测精度。",
            "dataset": "summary",
            "sourceId": "training_results",
            "metrics": [
                {"label": "最佳 mAP50-95", "field": "best_map5095", "format": "percent"}
            ],
        },
        {
            "id": "best_epoch",
            "description": "按照验证集 mAP50-95 选择的模型检查点。",
            "dataset": "summary",
            "sourceId": "training_results",
            "metrics": [{"label": "最佳轮次", "field": "best_epoch", "format": "number"}],
        },
        {
            "id": "gain",
            "description": "相对中断前最佳检查点的绝对 mAP50-95 增量。",
            "dataset": "summary",
            "sourceId": "training_results",
            "metrics": [
                {"label": "续训带来的绝对增益", "field": "absolute_map_gain", "format": "percent"}
            ],
        },
        {
            "id": "completed_epochs",
            "description": "计划 400 轮；因连续 50 轮未改善而正常早停。",
            "dataset": "summary",
            "sourceId": "training_results",
            "metrics": [
                {"label": "实际完成轮次", "field": "completed_epochs", "format": "number"}
            ],
        },
    ]

    charts = [
        {
            "id": "validation_metrics",
            "title": "验证集检测指标随轮次变化",
            "subtitle": "共 291 个完整轮次；竖线标记中断前检查点与最佳轮次。",
            "intent": "trend",
            "question": "安全续训是否继续提升了验证集检测质量？",
            "rationale": "完整时间序列能够显示中断、续训、提升和最终平台期。",
            "comparisonContext": {"grain": "epoch", "unit": "0-1 score"},
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
            "referenceLines": [
                {"axis": "x", "value": 129, "label": "中断前检查点", "color": "neutral", "lineStyle": "dotted"},
                {"axis": "x", "value": best_epoch, "label": "最佳轮次", "color": "neutral", "lineStyle": "dashed"},
            ],
            "layout": "full",
        },
        {
            "id": "box_loss",
            "title": "训练与验证 Box Loss",
            "subtitle": "训练损失持续下降，而验证损失在最佳轮次后进入平台。",
            "intent": "trend",
            "question": "模型是否在最佳轮次后继续泛化改善，还是只继续拟合训练集？",
            "rationale": "训练与验证损失的同尺度对比用于识别平台期和轻微过拟合。",
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
            "referenceLines": [
                {"axis": "x", "value": best_epoch, "label": "最佳轮次", "color": "neutral", "lineStyle": "dashed"}
            ],
            "layout": "full",
        },
        {
            "id": "class_map",
            "title": "各类别最终 mAP50-95",
            "subtitle": "基于 best.pt 对 1,309 张验证图像、1,366 个实例的最终验证。",
            "intent": "comparison",
            "question": "哪些类别限制了整体检测质量？",
            "rationale": "7 个类别适合用排序横向条形图比较，并保留实例数用于判断稳定性。",
            "comparisonContext": {"grain": "class", "unit": "mAP50-95", "denominator": "validation instances"},
            "type": "horizontalBar",
            "dataset": "class_metrics",
            "sourceId": "final_validation",
            "encodings": {
                "x": {"field": "class", "type": "nominal", "label": "类别"},
                "y": {"field": "map5095", "type": "quantitative", "label": "mAP50-95", "format": "percent"},
                "tooltip": [
                    {"field": "instances", "type": "quantitative", "label": "验证实例"},
                    {"field": "precision", "type": "quantitative", "label": "Precision", "format": "percent"},
                    {"field": "recall", "type": "quantitative", "label": "Recall", "format": "percent"},
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
            "subtitle": "从首次完整轮次、中断前检查点、安全续训到正常早停。",
            "dataset": "milestones",
            "sourceId": "training_results",
            "defaultSort": {"field": "epoch", "direction": "asc"},
            "density": "spacious",
            "columns": [
                {"field": "stage", "label": "阶段", "type": "text"},
                {"field": "epoch", "label": "轮次", "format": "number"},
                {"field": "map5095", "label": "mAP50-95", "format": "percent"},
                {"field": "map50", "label": "mAP50", "format": "percent"},
                {"field": "precision", "label": "Precision", "format": "percent"},
                {"field": "recall", "label": "Recall", "format": "percent"},
            ],
        },
        {
            "id": "class_detail",
            "title": "各类别验证明细",
            "subtitle": "best.pt 的最终验证结果；按 mAP50-95 从高到低排序。",
            "dataset": "class_metrics",
            "sourceId": "final_validation",
            "defaultSort": {"field": "map5095", "direction": "desc"},
            "density": "spacious",
            "columns": [
                {"field": "class", "label": "类别", "type": "text"},
                {"field": "images", "label": "图像", "format": "number"},
                {"field": "instances", "label": "实例", "format": "number"},
                {"field": "precision", "label": "Precision", "format": "percent"},
                {"field": "recall", "label": "Recall", "format": "percent"},
                {"field": "map50", "label": "mAP50", "format": "percent"},
                {"field": "map5095", "label": "mAP50-95", "format": "percent"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": "# ESPDet-Pico 食材检测训练报告"},
        {
            "id": "summary",
            "type": "markdown",
            "sourceId": "training_run",
            "body": (
                "## 技术摘要\n\n"
                f"模型在第 **{best_epoch}** 轮取得最佳验证结果：mAP50-95 为 **{best_map:.5f}** "
                f"（{best_map:.2%}），mAP50 为 **{f(best, METRIC_MAP50):.5f}**。相较中断前第 "
                f"{pre_resume_epoch} 轮的最佳 mAP50-95 **{pre_resume_map:.5f}**，安全续训带来 "
                f"**+{absolute_gain:.5f}** 的绝对增益（相对约 **+{relative_gain:.1%}**）。\n\n"
                f"训练在第 **{final_epoch}** 轮按 patience=50 正常早停，而非异常退出。最终轮次指标比最佳值低 "
                f"**{final_gap:.5f}**，结合损失曲线说明模型已进入收敛平台。推荐部署与后续验证使用 `best.pt`，"
                "不要使用 `last.pt` 作为质量最优权重。"
            ),
        },
        {
            "id": "quality_finding",
            "type": "markdown",
            "sourceId": "training_results",
            "body": (
                "## 安全续训带来了持续且可验证的质量提升\n\n"
                f"从第 129 轮恢复后，mAP50-95 从 **{pre_resume_map:.5f}** 上升到第 {best_epoch} 轮的 "
                f"**{best_map:.5f}**。提升不是单轮尖峰：第 200、213、224、230 和 241 轮连续刷新阶段最佳，"
                "说明恢复的优化器、EMA 与 AMP 状态确实延续了原训练轨迹。曲线在第 241 轮后趋于平坦，"
                "因此 patience=50 的终止时点有充分依据。"
            ),
        },
        {"id": "quality_chart", "type": "chart", "chartId": "validation_metrics"},
        {
            "id": "milestone_note",
            "type": "markdown",
            "sourceId": "training_results",
            "body": "### 关键节点可复核\n\n下表保留了中断、恢复、最佳和早停节点的精确验证指标，便于复现实验与检查模型选择逻辑。",
        },
        {"id": "milestone_table", "type": "table", "tableId": "milestones"},
        {
            "id": "class_finding",
            "type": "markdown",
            "sourceId": "final_validation",
            "body": (
                "## 类别表现不均衡是当前模型的主要质量风险\n\n"
                "`lettuce` 的 mAP50-95 达到 **0.919**，`apple` 为 **0.701**；但 `cherry tomato` 仅 "
                "**0.393**、`banana` 为 **0.423**、`chicken breast` 为 **0.460**。这意味着整体平均值会掩盖"
                "弱类别问题。`lettuce` 只有 58 个验证实例，其高分不一定代表在更多场景下同样稳定；后续采样应"
                "优先补足弱类别与低样本类别。"
            ),
        },
        {"id": "class_chart", "type": "chart", "chartId": "class_map"},
        {"id": "class_table", "type": "table", "tableId": "class_detail"},
        {
            "id": "loss_finding",
            "type": "markdown",
            "sourceId": "training_results",
            "body": (
                "## 最佳轮次之后出现轻微泛化间隙，早停选择合理\n\n"
                f"第 {best_epoch} 轮训练/验证 Box Loss 分别为 **{f(best, 'train/box_loss'):.5f}** 和 "
                f"**{f(best, 'val/box_loss'):.5f}**；到第 {final_epoch} 轮，训练损失继续降至 "
                f"**{f(final, 'train/box_loss'):.5f}**，验证损失却为 **{f(final, 'val/box_loss'):.5f}**。"
                "训练集仍在改善而验证集不再同步改善，符合轻微过拟合/平台期特征；继续硬跑至 400 轮的收益"
                "证据不足。"
            ),
        },
        {"id": "loss_chart", "type": "chart", "chartId": "box_loss"},
        {
            "id": "scope",
            "type": "markdown",
            "sourceId": "training_run",
            "body": (
                "## 评估范围与指标定义\n\n"
                "训练集包含 **7,420** 张图像，验证集包含 **1,309** 张图像和 **1,366** 个标注实例，覆盖 7 个"
                "食材类别。mAP50 是 IoU=0.50 下各类别平均精度；mAP50-95 是 IoU 从 0.50 到 0.95（步长 0.05）"
                "的平均值，更严格地反映定位质量。Precision 衡量预测框中正确检测的比例，Recall 衡量标注目标"
                "中被检出的比例。所有最佳轮次判断均基于同一验证集的 mAP50-95。"
            ),
        },
        {
            "id": "method",
            "type": "markdown",
            "sourceId": "training_config",
            "body": (
                "## 训练方法与模型规格\n\n"
                "模型为 ESPDet-Pico，输入尺寸 224×224，batch=128，AMP 开启，优化器由 Ultralytics 自动选择，"
                "使用 cosine 学习率、seed=42 和确定性训练；计划 400 轮、patience=50。最终验证载入的模型约 "
                "**358.9k 参数、1.3 GFLOPs**。续训从 `last.pt` 恢复完整状态，并将 Windows DataLoader "
                "worker 从 8 降到 2；162 个续训轮次耗时约 **0.864 小时**，平均约 "
                f"**{resume_seconds / resumed_epochs:.1f} 秒/轮**。"
            ),
        },
        {
            "id": "robustness",
            "type": "markdown",
            "sourceId": "training_run",
            "body": (
                "## 结论可信，但仍缺少独立测试集与重复实验\n\n"
                "训练过程无 CUDA OOM、NaN 或错误日志；安全续训后检查点持续写入，最终由框架正常早停并重新"
                "验证 `best.pt`。不过，本报告的模型选择和质量判断都来自同一验证集，没有独立测试集、交叉验证、"
                "多随机种子重复实验或真实 ESP32-S3 端侧性能测试。类别指标只保留到日志显示的三位小数，类别间"
                "差异可用于排序，但不应解读为高精度置信区间。"
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "sourceId": "training_run",
            "body": (
                "## 推荐下一步\n\n"
                "1. **冻结并登记 `best.pt`**：记录第 241 轮、数据版本和 Ultralytics 8.4.0，作为部署候选基线。\n"
                "2. **建立独立测试集**：至少覆盖不同光照、遮挡、尺度和背景，避免继续使用当前验证集做最终验收。\n"
                "3. **优先改善弱类别**：补充 cherry tomato、banana、chicken breast 的困难样本，并检查类别标注一致性。\n"
                "4. **执行端侧验证**：量化 ESP32-S3 上的模型转换误差、延迟、内存峰值和各类别召回率。\n"
                "5. **如需再训练，先做受控实验**：比较类别重采样/增强和输入尺寸，不建议仅提高 epochs 或关闭早停。"
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## 仍需回答的问题\n\n"
                "- 弱类别的主要误差来自相互混淆、漏检，还是边界框定位不准？\n"
                "- 部署转换为 ESP-DL 格式后，mAP50-95 会损失多少？\n"
                "- lettuce 的高分在更大、更复杂的独立样本上能否保持？\n"
                "- 当前 224×224 输入是否是端侧延迟与弱类别精度之间的最佳平衡？"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "ESPDet-Pico 食材检测训练报告",
            "description": "安全续训、最佳轮次、收敛、类别质量与部署建议的技术报告。",
            "generatedAt": generated_at,
            "cards": [],
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
                "epoch_metrics": epoch_rows,
                "loss_metrics": loss_rows,
                "class_metrics": class_metrics,
                "milestones": milestones,
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
        "pre_resume_best_metrics": pre_resume,
        "class_metrics": class_metrics,
        "milestones": milestones,
    }
    (REPORT / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT / "monitoring_notes.md").write_text(
        """# 续训监控记录

- 2026-08-06 15:37：首次续训按原 workers=8 启动；Windows/Ultralytics 派生约 24 个 DataLoader 子进程，形成明显 RAM 风险。在任何新轮次写入前停止该进程树。
- 2026-08-06 15:40：修正续训入口为先加载 last.pt、再传 resume=True；以 workers=2 安全重启，从第 130 轮继续。
- 训练期间：错误日志保持为空；抽样观测 GPU 温度约 65–78°C、训练显存约 3.7–3.9 GB；未观察到热降频、CUDA OOM、NaN 或检查点停写。
- 2026-08-06 16:33：第 291 轮后因连续 50 轮未刷新最佳而正常早停；框架重新验证 best.pt 并保存最终图表。
- 风险调整原则：只降低数据加载并发，不改变 batch、学习率、增强、随机种子或模型结构，以保持续训轨迹可比。
""",
        encoding="utf-8",
    )
    (REPORT / "source_notes.md").write_text(
        """# 报告源与图表说明

## 报告主线

- 问题：安全续训是否成功、最佳模型质量如何、是否收敛、主要风险和下一步是什么。
- 决策答案：续训有效；使用第 241 轮 best.pt；训练在第 291 轮合理早停；弱类别与独立测试集缺失是主要风险。
- 受众：技术受众。
- 比较基线：中断前第 129 轮最佳检查点。

## 必需结构映射

- 标题：标题块。
- 技术摘要：技术摘要块与指标卡。
- 带视觉证据的关键发现：验证指标趋势、类别横向条形图、Box Loss 趋势。
- 范围、数据与指标定义：评估范围与指标定义。
- 方法与模型规格：训练方法与模型规格。
- 局限、稳健性：轻微泛化间隙与独立测试集限制。
- 建议：推荐下一步。
- 后续问题：仍需回答的问题。

## 图表地图

1. 验证指标趋势：line；epoch × mAP50/mAP50-95；回答续训是否持续提升。蓝色实线与橙色虚线，不只依赖颜色。
2. Box Loss：line；epoch × train/val；回答最佳轮次后是否继续泛化。蓝色实线与橙色虚线。
3. 类别表现：horizontalBar；class × mAP50-95；回答哪些类别限制整体质量。单根蓝色顺序色板，带直接数值标签。

三张时间序列图中只使用两张 line，因为两者分别回答质量趋势与泛化间隙；第三张采用类别比较条形图。没有绘制 Precision/Recall 趋势，避免与 mAP 趋势重复；精确值保留在里程碑和类别表中。

## 证据限制

- 没有独立测试集、交叉验证或多随机种子实验。
- 每类别指标来自最终日志，显示精度为三位小数。
- 没有端侧 ESP32-S3 推理基准或量化后精度。
""",
        encoding="utf-8",
    )
    print(json.dumps({"report_dir": str(REPORT), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
