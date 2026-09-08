"""Capture one raw GC2145 frame and compare FP32 with board INT8 on one input."""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from nn.esp_tasks import custom_parse_model  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import ultralytics.nn.tasks as tasks  # noqa: E402


NAMES = {
    0: "apple",
    1: "banana",
    2: "strawberry",
    3: "lettuce",
    4: "egg",
    5: "orange",
    6: "eggplant",
    7: "cucumber",
    8: "carrot",
    9: "corn",
}
MODEL_SIZE = 224


def get_bytes(url: str, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} returned HTTP {error.code}: {message}") from error


def post_rgb565(url: str, payload: bytes, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned HTTP {error.code}: {message}") from error


def prepare_model_input(jpeg: bytes) -> tuple[Image.Image, bytes, dict]:
    with Image.open(io.BytesIO(jpeg)) as source:
        source_rgb = source.convert("RGB")
    crop_size = min(source_rgb.size)
    left = (source_rgb.width - crop_size) // 2
    top = (source_rgb.height - crop_size) // 2
    cropped = source_rgb.crop((left, top, left + crop_size, top + crop_size))
    resized = cropped.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR)

    rgb = np.asarray(resized, dtype=np.uint16)
    packed = ((rgb[..., 0] >> 3) << 11) | ((rgb[..., 1] >> 2) << 5) | (rgb[..., 2] >> 3)
    payload = packed.astype(">u2", copy=False).tobytes()

    # Feed FP32 the RGB565 round-trip so both paths receive the same color precision.
    red = ((packed >> 11) & 0x1F).astype(np.uint8)
    green = ((packed >> 5) & 0x3F).astype(np.uint8)
    blue = (packed & 0x1F).astype(np.uint8)
    exact_rgb = np.stack(
        ((red << 3) | (red >> 2), (green << 2) | (green >> 4), (blue << 3) | (blue >> 2)),
        axis=-1,
    )
    model_input = Image.fromarray(exact_rgb, mode="RGB")
    geometry = {
        "source_size": [source_rgb.width, source_rgb.height],
        "crop_xywh": [left, top, crop_size, crop_size],
        "model_size": [MODEL_SIZE, MODEL_SIZE],
        "format": "RGB565 big-endian",
    }
    return model_input, payload, geometry


def fp32_predict(weights: Path, image: Image.Image, confidence: float) -> list[dict]:
    tasks.parse_model = custom_parse_model
    model = YOLO(str(weights))
    result = model.predict(
        source=image,
        imgsz=MODEL_SIZE,
        conf=confidence,
        iou=0.70,
        agnostic_nms=True,
        max_det=10,
        device="cpu",
        verbose=False,
    )[0]
    detections = []
    for box, score, category in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        class_id = int(category.item())
        detections.append(
            {
                "category": class_id,
                "label": NAMES.get(class_id, str(class_id)),
                "score": float(score.item()),
                "box": [float(value) for value in box.tolist()],
            }
        )
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://172.17.4.66")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "runs/detect/welland_v05_yolo_espdet_pico_224_600e_p50/weights/best.pt",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "camera_frame_comparison")
    parser.add_argument("--fp32-conf", type=float, default=0.001)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    timestamp = datetime.now().astimezone()
    run_dir = args.output / f"compare_{timestamp:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    base_url = args.base_url.rstrip("/")

    jpeg = get_bytes(base_url + "/capture.jpg", args.timeout)
    capture_path = run_dir / "capture.jpg"
    capture_path.write_bytes(jpeg)
    model_input, payload, geometry = prepare_model_input(jpeg)
    input_path = run_dir / "model_input_rgb565.png"
    model_input.save(input_path)

    board = post_rgb565(base_url + "/validate", payload, args.timeout)
    fp32 = fp32_predict(args.weights.resolve(), model_input, args.fp32_conf)
    for detection in board.get("detections", []):
        category = int(detection["category"])
        detection["label"] = NAMES.get(category, str(category))

    report = {
        "captured_at": timestamp.isoformat(timespec="seconds"),
        "board": base_url,
        "weights": str(args.weights.resolve()),
        "geometry": geometry,
        "comparison_input": str(input_path.resolve()),
        "fp32_confidence_floor": args.fp32_conf,
        "production_confidence_threshold": 0.75,
        "fp32_detections": fp32,
        "int8_board_response": board,
        "note": "FP32 and INT8 used the same center-cropped, resized, RGB565-round-trip pixels.",
    }
    report_path = run_dir / "comparison.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
