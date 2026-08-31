"""Quantize an ESPDet raw-head ONNX model and keep reproducibility artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
from esp_ppq.IR import QuantableOperation
from esp_ppq.parser import NativeExporter
from esp_ppq.quantization.analyse import layerwise_error_analyse
from torch.utils.data import DataLoader

from deploy.quantize import CaliDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--target", default="esp32s3", choices=("esp32s3", "esp32p4"))
    parser.add_argument("--bits", type=int, default=8, choices=(8, 16))
    parser.add_argument(
        "--int16-layer",
        action="append",
        default=[],
        help=(
            "Exact ONNX operation name to dispatch to the target's INT16 PIE "
            "backend. Repeat for multiple layers; all unlisted layers keep --bits."
        ),
    )
    parser.add_argument(
        "--int16-companion-layer",
        action="append",
        default=[],
        help=(
            "Exact passive/fused ONNX operation name that must share the INT16 "
            "boundary with --int16-layer (for example its following activation)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=0,
        help="0 uses every available calibration batch",
    )
    parser.add_argument("--calib-algorithm", default="kl")
    parser.add_argument("--tqt-steps", type=int, default=0)
    parser.add_argument("--bias-correct", action="store_true")
    parser.add_argument("--equalization", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--collecting-device",
        default="cpu",
        help="Device used to cache TQT/bias-correction tensors",
    )
    parser.add_argument(
        "--layer-report",
        type=Path,
        help="Write per-computing-layer SNR loss as JSON and a sibling CSV",
    )
    parser.add_argument(
        "--analysis-steps",
        type=int,
        default=16,
        help="Calibration batches used for per-layer loss analysis",
    )
    parser.add_argument(
        "--large-loss-threshold",
        type=float,
        default=0.01,
        help="Flag a layer when noise/signal power ratio reaches this value",
    )
    args = parser.parse_args()

    if args.bits != 8 and (args.int16_layer or args.int16_companion_layer):
        raise ValueError("Mixed INT16 overrides are only valid when base --bits is 8")

    import onnx

    onnx_model = onnx.load(str(args.onnx))
    onnx_operation_names = {node.name for node in onnx_model.graph.node}
    normalized_operation_names = {
        name.replace("\\", "/"): name for name in onnx_operation_names
    }
    requested_dispatch_layers = args.int16_layer + args.int16_companion_layer
    resolved_int16_layer_map = {}
    for requested_name in requested_dispatch_layers:
        if requested_name in onnx_operation_names:
            resolved_int16_layer_map[requested_name] = requested_name
        elif requested_name.replace("\\", "/") in normalized_operation_names:
            resolved_int16_layer_map[requested_name] = normalized_operation_names[
                requested_name.replace("\\", "/")
            ]
    missing_int16_layers = sorted(
        set(requested_dispatch_layers) - set(resolved_int16_layer_map)
    )
    if missing_int16_layers:
        nearby = sorted(name for name in onnx_operation_names if "model.0" in name)
        raise KeyError(
            f"INT16 operation(s) not found in ONNX: {missing_int16_layers}. "
            f"Nearby model.0 operations: {nearby}"
        )
    resolved_int16_layers = [
        resolved_int16_layer_map[name] for name in requested_dispatch_layers
    ]

    dataset = CaliDataset(args.calib_dir, img_shape=args.imgsz)
    available_steps = math.ceil(len(dataset) / args.batch_size)
    calib_steps = args.calib_steps or available_steps
    if calib_steps > available_steps:
        raise ValueError(
            f"Requested {calib_steps} calibration batches, but only "
            f"{available_steps} are available"
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(args.device)

    setting = QuantizationSettingFactory.espdl_setting(num_of_bits=args.bits)
    setting.quantize_activation_setting.calib_algorithm = args.calib_algorithm
    int16_platform = get_target_platform(args.target, 16)
    for operation_name in resolved_int16_layers:
        setting.dispatching_table.append(operation_name, int16_platform)

    setting.bias_correct = args.bias_correct
    if args.bias_correct:
        setting.bias_correct_setting.interested_layers = []
        setting.bias_correct_setting.block_size = 4
        setting.bias_correct_setting.steps = 32
        setting.bias_correct_setting.collecting_device = args.collecting_device

    setting.equalization = args.equalization
    if args.equalization:
        setting.equalization_setting.iterations = 10
        setting.equalization_setting.value_threshold = 0.5
        setting.equalization_setting.opt_level = 2
        setting.equalization_setting.interested_layers = None

    setting.tqt_optimization = args.tqt_steps > 0
    if setting.tqt_optimization:
        tqt = setting.tqt_optimization_setting
        tqt.lr = 1e-5
        tqt.steps = args.tqt_steps
        tqt.block_size = 4
        tqt.is_scale_trainable = True
        tqt.gamma = 0.0
        tqt.int_lambda = 0.0
        tqt.collecting_device = args.collecting_device

    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph = espdl_quantize_onnx(
        onnx_import_file=str(args.onnx),
        espdl_export_file=str(args.output),
        calib_dataloader=dataloader,
        calib_steps=calib_steps,
        input_shape=[1, 3, args.imgsz, args.imgsz],
        target=args.target,
        num_of_bits=args.bits,
        collate_fn=collate_fn,
        setting=setting,
        device=args.device,
        error_report=False,
        skip_export=False,
        export_test_values=False,
        verbose=0,
        inputs=None,
    )

    precision_audit = {}
    for operation_name in resolved_int16_layers:
        operation = graph.operations[operation_name]
        if not isinstance(operation, QuantableOperation):
            raise TypeError(f"Mixed-precision operation is not quantized: {operation_name}")
        input_configs = []
        for variable, config in zip(operation.inputs, operation.input_quant_config):
            input_configs.append(
                {
                    "variable": variable.name,
                    "parameter": bool(variable.is_parameter),
                    "bits": int(config.num_of_bits),
                    "quant_min": int(config.quant_min),
                    "quant_max": int(config.quant_max),
                    "state": config.state.name,
                }
            )
        output_configs = [
            {
                "variable": variable.name,
                "bits": int(config.num_of_bits),
                "quant_min": int(config.quant_min),
                "quant_max": int(config.quant_max),
                "state": config.state.name,
            }
            for variable, config in zip(operation.outputs, operation.output_quant_config)
        ]
        non_bias_bits = {
            row["bits"]
            for row in input_configs
            if not row["parameter"] or row["bits"] <= 16
        } | {row["bits"] for row in output_configs}
        if non_bias_bits != {16}:
            raise RuntimeError(
                f"Expected INT16 activation/weight configs for {operation_name}, "
                f"got bit widths {sorted(non_bias_bits)}"
            )
        precision_audit[operation_name] = {
            "type": operation.type,
            "platform": operation.platform.name,
            "inputs": input_configs,
            "outputs": output_configs,
        }

    native_path = args.output.with_suffix(".native")
    NativeExporter().export(file_path=str(native_path), graph=graph)

    layer_report = None
    if args.layer_report:
        if not 1 <= args.analysis_steps <= available_steps:
            raise ValueError(
                f"analysis-steps must be in [1, {available_steps}], got "
                f"{args.analysis_steps}"
            )
        # esp-ppq 1.3.6 treats ``steps`` as the last zero-based batch index.
        losses = layerwise_error_analyse(
            graph=graph,
            dataloader=dataloader,
            collate_fn=collate_fn,
            running_device=args.device,
            method="snr",
            steps=args.analysis_steps - 1,
            verbose=True,
        )
        rows = [
            {
                "operation": name,
                "type": graph.operations[name].type,
                "snr_loss_ratio": float(loss),
                "snr_loss_percent": float(loss) * 100.0,
                "large_loss": float(loss) >= args.large_loss_threshold,
                "next_version_int16_candidate": float(loss)
                >= args.large_loss_threshold,
            }
            for name, loss in losses.items()
        ]
        rows.sort(key=lambda row: row["snr_loss_ratio"], reverse=True)
        layer_report = {
            "metric": "noise_power / signal_power at the six model outputs",
            "interpretation": "lower is better; each row enables only that operation's quantization",
            "analysis_batches": args.analysis_steps,
            "analysis_images_max": min(
                len(dataset), args.analysis_steps * args.batch_size
            ),
            "large_loss_threshold_ratio": args.large_loss_threshold,
            "large_loss_threshold_percent": args.large_loss_threshold * 100.0,
            "quantized_computing_layers": len(rows),
            "large_loss_layers": sum(row["large_loss"] for row in rows),
            "layers": rows,
        }
        args.layer_report.parent.mkdir(parents=True, exist_ok=True)
        args.layer_report.write_text(
            json.dumps(layer_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        csv_path = args.layer_report.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    recipe = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "onnx": str(args.onnx.resolve()),
        "calibration_dir": str(args.calib_dir.resolve()),
        "calibration_images": len(dataset),
        "calibration_batches": calib_steps,
        "batch_size": args.batch_size,
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "target": args.target,
        "bits": args.bits,
        "precision_policy": {
            "default_bits": args.bits,
            "int16_layers_requested": args.int16_layer,
            "int16_layers_resolved": [
                resolved_int16_layer_map[name] for name in args.int16_layer
            ],
            "int16_companion_layers_requested": args.int16_companion_layer,
            "int16_companion_layers_resolved": [
                resolved_int16_layer_map[name]
                for name in args.int16_companion_layer
            ],
            "int16_layer_name_map": resolved_int16_layer_map,
            "int16_platform": int16_platform.name,
            "audit": precision_audit,
        },
        "device": args.device,
        "collecting_device": args.collecting_device,
        "calibration_algorithm": args.calib_algorithm,
        "bias_correct": args.bias_correct,
        "equalization": args.equalization,
        "tqt_steps": args.tqt_steps,
        "espdl": str(args.output.resolve()),
        "native": str(native_path.resolve()),
        "layer_loss_report": str(args.layer_report.resolve())
        if args.layer_report
        else None,
        "layer_loss_analysis_batches": args.analysis_steps
        if args.layer_report
        else None,
        "large_loss_threshold_ratio": args.large_loss_threshold
        if args.layer_report
        else None,
        "large_loss_layers": layer_report["large_loss_layers"]
        if layer_report
        else None,
    }
    recipe_path = args.output.with_suffix(".recipe.json")
    recipe_path.write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(recipe, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
