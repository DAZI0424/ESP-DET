"""Run an independent validation pass for the trained Welland V02 model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from nn.esp_tasks import custom_parse_model  # noqa: E402
import ultralytics.nn.tasks as tasks  # noqa: E402
from ultralytics.utils.torch_utils import model_info  # noqa: E402


RUN = ROOT / "runs" / "espdet" / "welland_v02_espdet_pico_224"
WEIGHTS = RUN / "weights" / "best.pt"
DATASET = ROOT / "datasets" / "Welland-V02-espdet-pico" / "clean-20260810.yaml"


def as_float_list(values: object) -> list[float]:
    return [float(value) for value in values]


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {WEIGHTS}")

    tasks.parse_model = custom_parse_model
    model = YOLO(str(WEIGHTS))
    info = model_info(model.model, verbose=True, imgsz=224)
    metrics = model.val(
        data=str(DATASET),
        imgsz=224,
        batch=128,
        device=0,
        workers=0,
        plots=True,
        save_json=False,
        project=str(RUN),
        name="final_validation",
        exist_ok=True,
        verbose=True,
    )

    names = {int(key): value for key, value in metrics.names.items()}
    class_indices = [int(value) for value in metrics.box.ap_class_index]
    precision = as_float_list(metrics.box.p)
    recall = as_float_list(metrics.box.r)
    ap50 = as_float_list(metrics.box.ap50)
    map5095 = as_float_list(metrics.box.maps)
    class_metrics = []
    for position, class_index in enumerate(class_indices):
        class_metrics.append(
            {
                "class_id": class_index,
                "class_name": names[class_index],
                "precision": precision[position],
                "recall": recall[position],
                "map50": ap50[position],
                "map50_95": map5095[class_index],
            }
        )

    payload = {
        "weights": str(WEIGHTS),
        "dataset": str(DATASET),
        "imgsz": 224,
        "batch": 128,
        "aggregate": {key: float(value) for key, value in metrics.results_dict.items()},
        "per_class": class_metrics,
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "model": {
            "layers": int(info[0]),
            "parameters": int(info[1]),
            "gradients": int(info[2]),
            "gflops": float(info[3]),
        },
        "validation_dir": str(metrics.save_dir),
    }
    output = Path(metrics.save_dir) / "metrics.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"METRICS_JSON={output}")


if __name__ == "__main__":
    main()
