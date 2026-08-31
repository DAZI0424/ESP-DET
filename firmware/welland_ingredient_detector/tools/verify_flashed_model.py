"""Verify that the ESP32-S3 Flash contains the model embedded in the planned app."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zlib
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_DIR
    / "models"
    / "welland_v04_yolo_espdet_pico_224_esp32s3_int8_model0_int16.espdl"
)
DEFAULT_APP = PROJECT_DIR / "build" / "welland_ingredient_detector.bin"
DEFAULT_READBACK = PROJECT_DIR / "build" / "app_readback.bin"


def integer(value: str) -> int:
    return int(value, 0)


def measurements(data: bytes) -> dict[str, str | int]:
    return {
        "size": len(data),
        "crc32": f"{zlib.crc32(data) & 0xFFFFFFFF:08X}",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def print_measurements(label: str, data: bytes) -> None:
    values = measurements(data)
    print(
        f"{label}: size={values['size']} crc32={values['crc32']} "
        f"sha256={values['sha256']}"
    )


def find_all(container: bytes, value: bytes) -> list[int]:
    offsets = []
    start = 0
    while True:
        offset = container.find(value, start)
        if offset < 0:
            return offsets
        offsets.append(offset)
        start = offset + 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the planned model, app-embedded model and ESP32 Flash bytes."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--app-offset", type=integer, default=0x10000)
    parser.add_argument("--port", help="Serial port, for example COM5")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--chip", default="esp32s3")
    parser.add_argument("--esptool", default="esptool")
    parser.add_argument(
        "--flash-dump",
        type=Path,
        help="Previously read full app-range dump; mutually exclusive with --port",
    )
    parser.add_argument("--readback", type=Path, default=DEFAULT_READBACK)
    args = parser.parse_args()

    if args.port and args.flash_dump:
        parser.error("use either --port or --flash-dump, not both")

    model = args.model.resolve().read_bytes()
    app = args.app.resolve().read_bytes()
    print_measurements("planned_model", model)
    print_measurements("planned_app", app)

    offsets = find_all(app, model)
    if len(offsets) != 1:
        print(
            f"FAIL: planned model occurs {len(offsets)} times in the app; expected exactly once",
            file=sys.stderr,
        )
        return 1

    app_model_offset = offsets[0]
    flash_model_offset = args.app_offset + app_model_offset
    print(f"app_offset=0x{args.app_offset:X}")
    print(f"model_offset_in_app=0x{app_model_offset:X}")
    print(f"model_flash_address=0x{flash_model_offset:X}")
    print("PASS: app contains exactly one byte-identical copy of the planned model")

    flash_dump = args.flash_dump
    if args.port:
        flash_dump = args.readback.resolve()
        flash_dump.parent.mkdir(parents=True, exist_ok=True)
        command = [
            args.esptool,
            "--chip",
            args.chip,
            "--port",
            args.port,
            "--baud",
            str(args.baud),
            "read-flash",
            hex(args.app_offset),
            hex(len(app)),
            str(flash_dump),
        ]
        print("reading_flash:", subprocess.list2cmdline(command))
        subprocess.run(command, check=True)

    if flash_dump is None:
        print(
            "DEVICE VERIFICATION NOT PERFORMED: no serial port or Flash dump was supplied",
            file=sys.stderr,
        )
        return 2

    app_readback = flash_dump.resolve().read_bytes()
    print_measurements("device_app_readback", app_readback)
    if app_readback != app:
        print("FAIL: device app bytes differ from the planned app", file=sys.stderr)
        return 1

    model_readback = app_readback[app_model_offset : app_model_offset + len(model)]
    print_measurements("device_model_readback", model_readback)
    if model_readback != model:
        print("FAIL: device model bytes differ from the planned model", file=sys.stderr)
        return 1

    print("PASS: device app and embedded model match the planned bytes, CRC32 and SHA-256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
