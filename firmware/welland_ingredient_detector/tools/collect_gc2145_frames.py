"""Collect unannotated GC2145 JPEG frames from a flashed detector."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://192.168.4.1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    endpoint = args.base_url.rstrip("/") + "/capture.jpg"
    captured = []
    for index in range(args.count):
        timestamp = datetime.now().astimezone()
        with urllib.request.urlopen(endpoint, timeout=30) as response:
            jpeg = response.read()
        filename = f"gc2145_{timestamp:%Y%m%d_%H%M%S_%f}_{index:05d}.jpg"
        (args.output / filename).write_bytes(jpeg)
        captured.append(
            {
                "file": filename,
                "captured_at": timestamp.isoformat(timespec="milliseconds"),
                "bytes": len(jpeg),
            }
        )
        print(f"{index + 1}/{args.count}: {filename}")
        if index + 1 < args.count:
            time.sleep(args.interval)

    manifest = {
        "source": endpoint,
        "frames": captured,
        "labeling_status": "pending",
    }
    (args.output / "capture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
