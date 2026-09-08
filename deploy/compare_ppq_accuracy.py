import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from esp_ppq.api import load_native_graph, load_onnx_graph
from esp_ppq.executor import TorchExecutor
from torch import nn
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.head import Detect

# NumPy 2.x writes ``numpy._core`` module names into dataset cache pickles;
# NumPy 1.x exposes the same implementation as ``numpy.core``.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


class PPQDetectionModel(nn.Module):
    """Expose a PPQ graph as an Ultralytics-compatible detection model."""

    def __init__(self, graph_path: Path, names: dict[int, str], quantized: bool):
        super().__init__()
        graph = load_native_graph(str(graph_path)) if quantized else load_onnx_graph(str(graph_path))
        # TorchExecutor behaves like a module but is not initialized as an nn.Module;
        # keep it outside PyTorch's child-module registry.
        object.__setattr__(self, "executor", TorchExecutor(graph=graph, device="cpu"))
        self.names = names
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.yaml = {"channels": 3}
        self.end2end = False
        self.decoder = Detect(nc=len(names), ch=(32, 64, 128))
        self.decoder.reg_max = 1
        self.decoder.no = len(names) + 4
        self.decoder.dfl = nn.Identity()
        self.decoder.stride = self.stride
        self.decoder.eval()

    def fuse(self, verbose=True):
        return self

    def forward(self, images, *args, **kwargs):
        outputs = self.executor(images)
        box_features = list(outputs[0::2])
        score_features = list(outputs[1::2])
        batch_size = images.shape[0]
        predictions = {
            "boxes": torch.cat(
                [feature.view(batch_size, 4, -1) for feature in box_features], dim=-1
            ),
            "scores": torch.cat(
                [feature.view(batch_size, len(self.names), -1) for feature in score_features], dim=-1
            ),
            # Ultralytics 8.4 uses the feature-map shapes to construct anchors.
            "feats": box_features,
        }
        return self.decoder._inference(predictions)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    model: nn.Module,
    data: Path,
    imgsz: int,
    batch: int,
    fraction: float,
    conf: float,
):
    validator = DetectionValidator(
        args={
            "data": str(data),
            "imgsz": imgsz,
            "batch": batch,
            "device": "cpu",
            "workers": 0,
            "plots": False,
            "save": False,
            "save_json": False,
            "save_txt": False,
            "verbose": True,
            "rect": False,
            "fraction": fraction,
            "conf": conf,
            "agnostic_nms": True,
            "max_det": 1,
        }
    )
    metrics = validator(model=model)
    aggregate = {key: float(metrics[key]) for key in METRIC_KEYS}
    per_class = []
    for class_id, class_name in model.names.items():
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(validator.metrics.box.p[class_id]),
                "recall": float(validator.metrics.box.r[class_id]),
                "map50": float(validator.metrics.box.ap50[class_id]),
                "map50_95": float(validator.metrics.box.maps[class_id]),
            }
        )
    return aggregate, per_class


def main():
    parser = argparse.ArgumentParser(description="Compare an FP32 ONNX graph with an INT8 PPQ Native graph")
    parser.add_argument("--pt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--espdl", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--bits", type=int, default=8, choices=(8, 16))
    parser.add_argument("--precision-label", default=None)
    args = parser.parse_args()

    data_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    raw_names = data_config["names"]
    if isinstance(raw_names, list):
        names = {index: name for index, name in enumerate(raw_names)}
    else:
        names = {int(index): name for index, name in raw_names.items()}

    print("Validating FP32 ONNX through PPQ TorchExecutor...")
    float_metrics, float_per_class = validate(
        PPQDetectionModel(args.onnx, names, quantized=False),
        args.data,
        args.imgsz,
        args.batch,
        args.fraction,
        args.conf,
    )
    print(f"Validating INT{args.bits} PPQ Native graph through PPQ TorchExecutor...")
    int8_metrics, int8_per_class = validate(
        PPQDetectionModel(args.native, names, quantized=True),
        args.data,
        args.imgsz,
        args.batch,
        args.fraction,
        args.conf,
    )

    absolute_drop = {key: float_metrics[key] - int8_metrics[key] for key in METRIC_KEYS}
    relative_drop = {
        key: absolute_drop[key] / float_metrics[key] * 100 if float_metrics[key] else None
        for key in METRIC_KEYS
    }
    per_class_drop = []
    for float_row, int8_row in zip(float_per_class, int8_per_class):
        per_class_drop.append(
            {
                "class_id": float_row["class_id"],
                "class_name": float_row["class_name"],
                "map50_95_fp32": float_row["map50_95"],
                "map50_95_int8": int8_row["map50_95"],
                "map50_95_quantized": int8_row["map50_95"],
                "map50_95_absolute_drop": float_row["map50_95"] - int8_row["map50_95"],
            }
        )

    best_pt_metrics = None
    best_pt_to_int8_drop = None
    best_pt_to_int8_relative_drop = None
    validation_images = None
    if args.baseline_report:
        baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        validation_section = baseline_report.get("validation")
        if isinstance(validation_section, dict) and "aggregate" in validation_section:
            baseline_metrics = validation_section["aggregate"]
        elif isinstance(baseline_report.get("best_metrics"), dict):
            baseline_metrics = baseline_report["best_metrics"]
        else:
            baseline_metrics = baseline_report.get("aggregate", baseline_report)
        best_pt_metrics = {
            key: float(baseline_metrics[key])
            for key in METRIC_KEYS
        }
        best_pt_to_int8_drop = {
            key: best_pt_metrics[key] - int8_metrics[key]
            for key in METRIC_KEYS
        }
        best_pt_to_int8_relative_drop = {
            key: best_pt_to_int8_drop[key] / best_pt_metrics[key] * 100
            if best_pt_metrics[key]
            else None
            for key in METRIC_KEYS
        }
        dataset_section = baseline_report.get("dataset_summary", baseline_report.get("dataset", {}))
        if isinstance(dataset_section, dict):
            validation_split = dataset_section.get("splits", {}).get("val")
            validation_images = (
                validation_split.get("images")
                if isinstance(validation_split, dict)
                else validation_split
            )
        if validation_images is None:
            validation_images = baseline_report.get("validation_images")
    if validation_images is None:
        dataset_root = Path(data_config.get("path", args.data.parent))
        validation_path = Path(data_config["val"])
        if not validation_path.is_absolute():
            validation_path = dataset_root / validation_path
        image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
        validation_images = sum(
            1
            for path in validation_path.rglob("*")
            if path.is_file() and path.suffix.lower() in image_suffixes
        )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "comparison_basis": "same PPQ executor, preprocessing, decoding, confidence threshold and NMS",
        "confidence_threshold": args.conf,
        "agnostic_nms": True,
        "max_detections": 1,
        "validation_fraction": args.fraction,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "quantized_bits": args.bits,
        "precision_label": args.precision_label or f"INT{args.bits}",
        "validation_images": validation_images,
        "best_pt_reported_metrics": best_pt_metrics,
        "fp32_metrics": float_metrics,
        "int8_metrics": int8_metrics,
        "quantized_metrics": int8_metrics,
        "absolute_drop": absolute_drop,
        "relative_drop_percent": relative_drop,
        "best_pt_to_int8_absolute_drop": best_pt_to_int8_drop,
        "best_pt_to_int8_relative_drop_percent": best_pt_to_int8_relative_drop,
        "per_class_map50_95": per_class_drop,
        "artifacts": {
            str(args.pt): {"bytes": args.pt.stat().st_size, "sha256": sha256(args.pt)},
            str(args.onnx): {"bytes": args.onnx.stat().st_size, "sha256": sha256(args.onnx)},
            str(args.native): {"bytes": args.native.stat().st_size, "sha256": sha256(args.native)},
            str(args.espdl): {"bytes": args.espdl.stat().st_size, "sha256": sha256(args.espdl)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
