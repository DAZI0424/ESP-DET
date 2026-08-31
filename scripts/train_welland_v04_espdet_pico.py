"""Train ESPDet-Pico on the Welland v04 dataset."""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from train import Train  # noqa: E402


if __name__ == "__main__":
    multiprocessing.freeze_support()
    results = Train(
        dataset=str(ROOT / "datasets" / "welland-v04-yolo" / "data-local.yaml"),
        epochs=500,
        patience=50,
        imgsz=416,
        device=0,
        # AutoBatch targets about 60% of the available VRAM. This is safer than
        # a fixed batch on the 8 GiB RTX 5060 Laptop GPU while desktop apps are open.
        batch=-1,
        # Avoid Windows worker-process duplication; the same machine completed
        # the previous 416 px run stably with this setting.
        workers=0,
        cache=False,
        amp=True,
        cos_lr=True,
        close_mosaic=30,
        optimizer="auto",
        project=str(ROOT / "runs" / "detect"),
        name="welland_v04_yolo_espdet_pico_416_500e_p50",
        exist_ok=False,
        save=True,
        save_period=25,
        plots=True,
        verbose=True,
        seed=0,
        deterministic=True,
    )
    print(f"TRAIN_SAVE_DIR={results.save_dir}")
