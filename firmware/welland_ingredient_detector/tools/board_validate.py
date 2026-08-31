"""Run held-out YOLO validation images through the flashed ESP32-S3 model."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def load_dataset(dataset_yaml: Path, split: str):
    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    split_path = root / config[split]
    names_raw = config["names"]
    names = dict(enumerate(names_raw)) if isinstance(names_raw, list) else {
        int(key): value for key, value in names_raw.items()
    }
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        images = [
            Path(line) if Path(line).is_absolute() else root / line
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    else:
        images = sorted(path for path in split_path.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = []
    for image in images:
        parts = list(image.parts)
        try:
            parts[parts.index("images")] = "labels"
        except ValueError as error:
            raise ValueError(f"Image path must contain an 'images' directory: {image}") from error
        labels.append(Path(*parts).with_suffix(".txt"))
    return images, labels, names


def letterbox_rgb565be(image_path: Path, size: int = 224):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized_width = max(1, round(image.width * scale))
    resized_height = max(1, round(image.height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (left, top))
    rgb = np.asarray(canvas, dtype=np.uint16)
    rgb565 = ((rgb[..., 0] >> 3) << 11) | ((rgb[..., 1] >> 2) << 5) | (rgb[..., 2] >> 3)
    payload = rgb565.astype(">u2", copy=False).tobytes()
    return payload, image.size, scale, left, top


def load_ground_truth(label_path: Path, original_size, scale: float, left: int, top: int):
    width, height = original_size
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id, center_x, center_y, box_width, box_height = map(float, fields[:5])
        x1 = (center_x - box_width / 2) * width * scale + left
        y1 = (center_y - box_height / 2) * height * scale + top
        x2 = (center_x + box_width / 2) * width * scale + left
        y2 = (center_y + box_height / 2) * height * scale + top
        boxes.append({"category": int(class_id), "box": [x1, y1, x2, y2]})
    return boxes


def request_inference(endpoint: str, payload: bytes, timeout: float):
    request = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Board returned HTTP {error.code}: {message}") from error


def box_iou(box_a, box_b):
    intersection_width = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    intersection_height = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(recalls, precisions):
    if not recalls:
        return 0.0
    recall_curve = np.concatenate(([0.0], np.asarray(recalls), [1.0]))
    precision_curve = np.concatenate(([0.0], np.asarray(precisions), [0.0]))
    precision_curve = np.maximum.accumulate(precision_curve[::-1])[::-1]
    points = np.linspace(0.0, 1.0, 101)
    return float(np.mean([
        np.max(precision_curve[recall_curve >= point]) if np.any(recall_curve >= point) else 0.0
        for point in points
    ]))


def evaluate(predictions, ground_truth, names):
    predictions_by_class = defaultdict(list)
    truth_by_class_image = defaultdict(lambda: defaultdict(list))
    for image_index, items in enumerate(predictions):
        for item in items:
            predictions_by_class[item["category"]].append(
                (float(item["score"]), image_index, item["box"])
            )
    for image_index, items in enumerate(ground_truth):
        for item in items:
            truth_by_class_image[item["category"]][image_index].append(item["box"])

    per_class = []
    aggregate_tp = aggregate_fp = aggregate_gt = 0
    for class_id, class_name in sorted(names.items()):
        class_predictions = sorted(predictions_by_class[class_id], reverse=True)
        truth_for_images = truth_by_class_image[class_id]
        num_truth = sum(len(items) for items in truth_for_images.values())
        aps = []
        precision_50 = recall_50 = 0.0
        tp_50 = fp_50 = 0
        for threshold in IOU_THRESHOLDS:
            matched = defaultdict(set)
            true_positives = []
            false_positives = []
            for _, image_index, predicted_box in class_predictions:
                candidates = truth_for_images.get(image_index, [])
                best_iou = 0.0
                best_index = -1
                for candidate_index, truth_box in enumerate(candidates):
                    if candidate_index in matched[image_index]:
                        continue
                    iou = box_iou(predicted_box, truth_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_index = candidate_index
                is_match = best_index >= 0 and best_iou >= threshold
                if is_match:
                    matched[image_index].add(best_index)
                true_positives.append(1 if is_match else 0)
                false_positives.append(0 if is_match else 1)
            cumulative_tp = np.cumsum(true_positives)
            cumulative_fp = np.cumsum(false_positives)
            recalls = (cumulative_tp / max(num_truth, 1)).tolist()
            precisions = (cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)).tolist()
            aps.append(average_precision(recalls, precisions) if num_truth else float("nan"))
            if abs(float(threshold) - 0.5) < 1e-6:
                tp_50 = int(cumulative_tp[-1]) if len(cumulative_tp) else 0
                fp_50 = int(cumulative_fp[-1]) if len(cumulative_fp) else 0
                precision_50 = tp_50 / max(tp_50 + fp_50, 1)
                recall_50 = tp_50 / max(num_truth, 1)
        aggregate_tp += tp_50
        aggregate_fp += fp_50
        aggregate_gt += num_truth
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth": num_truth,
                "predictions": len(class_predictions),
                "precision_at_iou50": precision_50,
                "recall_at_iou50": recall_50,
                "map50": aps[0] if aps else 0.0,
                "map50_95": float(np.nanmean(aps)) if num_truth else None,
            }
        )
    valid_classes = [item for item in per_class if item["ground_truth"] > 0]
    return {
        "precision_at_iou50": aggregate_tp / max(aggregate_tp + aggregate_fp, 1),
        "recall_at_iou50": aggregate_tp / max(aggregate_gt, 1),
        "map50": statistics.mean(item["map50"] for item in valid_classes),
        "map50_95": statistics.mean(item["map50_95"] for item in valid_classes),
        "per_class": per_class,
    }


def summarize_latency(responses):
    summary = {}
    for phase in ("preprocess", "inference", "postprocess", "total"):
        values_ms = np.asarray([item["timing_us"][phase] / 1000.0 for item in responses])
        summary[f"{phase}_ms"] = {
            "mean": float(np.mean(values_ms)),
            "p50": float(np.percentile(values_ms, 50)),
            "p90": float(np.percentile(values_ms, 90)),
            "p95": float(np.percentile(values_ms, 95)),
            "min": float(np.min(values_ms)),
            "max": float(np.max(values_ms)),
        }
    return summary


def summarize_memory(responses):
    first = responses[0]
    heaps = [item["heap_bytes"] for item in responses]
    return {
        "model_memory_bytes": first["model_memory_bytes"],
        "minimum_observed": {
            key: min(heap[key] for heap in heaps)
            for key in (
                "internal_free_after",
                "internal_min_free",
                "internal_largest_after",
                "psram_free_after",
                "psram_min_free",
                "psram_largest_after",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://192.168.4.1")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the complete split")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("board_validation_report.json"))
    args = parser.parse_args()

    images, labels, names = load_dataset(args.dataset.resolve(), args.split)
    if args.limit > 0 and len(images) > args.limit:
        indices = np.linspace(0, len(images) - 1, args.limit, dtype=int)
        images = [images[index] for index in indices]
        labels = [labels[index] for index in indices]
    if not images:
        raise ValueError("No validation images found")
    endpoint = args.base_url.rstrip("/") + "/validate"

    first_payload, _, _, _, _ = letterbox_rgb565be(images[0])
    print(f"Warming up {args.warmup} inference(s) at {endpoint} ...")
    for _ in range(args.warmup):
        request_inference(endpoint, first_payload, args.timeout)

    responses = []
    predictions = []
    ground_truth = []
    started = time.perf_counter()
    for index, (image_path, label_path) in enumerate(zip(images, labels), 1):
        payload, original_size, scale, left, top = letterbox_rgb565be(image_path)
        response = request_inference(endpoint, payload, args.timeout)
        responses.append(response)
        predictions.append(response["detections"])
        ground_truth.append(
            load_ground_truth(label_path, original_size, scale, left, top)
        )
        if index == 1 or index % 10 == 0 or index == len(images):
            print(f"[{index}/{len(images)}] {image_path.name}: {response['timing_us']['total'] / 1000:.2f} ms")

    report = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "board_endpoint": endpoint,
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "images_evaluated": len(images),
        "warmup_inferences": args.warmup,
        "wall_seconds": time.perf_counter() - started,
        "note": "Metrics use the production firmware score/NMS thresholds.",
        "accuracy": evaluate(predictions, ground_truth, names),
        "latency": summarize_latency(responses),
        "memory": summarize_memory(responses),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"accuracy": report["accuracy"], "latency": report["latency"], "memory": report["memory"]}, indent=2, ensure_ascii=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
