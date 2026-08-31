import argparse
import math
import os
from pathlib import Path

import numpy as np
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from PIL import Image
from onnxsim import simplify
import onnx


class CaliDataset(Dataset):
    """Deterministic RGB calibration dataset using YOLO-style letterboxing."""

    def __init__(self, path, img_shape=640, pad_value=114):
        super().__init__()
        height, width = img_shape if isinstance(img_shape, (list, tuple)) else (img_shape, img_shape)
        self.height = height
        self.width = width
        self.pad_value = pad_value
        self.path = Path(path)
        self.imgs_path = []
        for img_name in sorted(os.listdir(self.path)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img_path = self.path / img_name
            self.imgs_path.append(img_path)
        if not self.imgs_path:
            raise ValueError(f"No calibration images found in {self.path}")

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        with Image.open(self.imgs_path[idx]) as source:
            image = source.convert("RGB")
        scale = min(self.width / image.width, self.height / image.height)
        resized_width = max(1, round(image.width * scale))
        resized_height = max(1, round(image.height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.width, self.height), (self.pad_value,) * 3)
        left = (self.width - resized_width) // 2
        top = (self.height - resized_height) // 2
        canvas.paste(resized, (left, top))
        array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
        return torch.from_numpy(array)


def report_hook(blocknum, blocksize, total):
    downloaded = blocknum * blocksize
    percent = downloaded / total * 100
    print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


def quant_espdet(
    onnx_path,
    target,
    num_of_bits,
    device,
    batchsz,
    imgsz,
    calib_dir,
    espdl_model_path,
    calib_steps=32,
):
    INPUT_SHAPE = [3, *imgsz] if isinstance(imgsz, (list, tuple)) else [3, imgsz, imgsz]
    model = onnx.load(onnx_path)
    sim = True
    if sim:
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx.shape_inference.infer_shapes(model), onnx_path)

    calibration_dataset = CaliDataset(calib_dir, img_shape=imgsz)
    dataloader = DataLoader(
        dataset=calibration_dataset, batch_size=batchsz, shuffle=False
    )
    available_steps = math.ceil(len(calibration_dataset) / batchsz)
    if available_steps < calib_steps:
        raise ValueError(
            f"Calibration set provides {available_steps} batches, fewer than "
            f"the requested {calib_steps}; add images or reduce batch size/steps"
        )

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(device)

    # default setting
    quant_setting = QuantizationSettingFactory.espdl_setting()

    # Equalization
    quant_setting.equalization = True
    quant_setting.equalization_setting.iterations = 4
    quant_setting.equalization_setting.value_threshold = .4
    quant_setting.equalization_setting.opt_level = 2
    quant_setting.equalization_setting.interested_layers = None


    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_model_path,
        calib_dataloader=dataloader,
        calib_steps=calib_steps,
        input_shape=[1] + INPUT_SHAPE,
        target=target,
        num_of_bits=num_of_bits,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=device,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=0,
        inputs=None,
    )
    return quant_ppq_graph  # , selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ESPDet ONNX for ESP-DL")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="esp32s3", choices=("esp32s3", "esp32p4"))
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--calib-steps", type=int, default=32)
    args = parser.parse_args()
    quant_espdet(
        onnx_path=args.onnx,
        target=args.target,
        num_of_bits=args.bits,
        device=args.device,
        batchsz=args.batch_size,
        imgsz=args.imgsz,
        calib_dir=args.calib_dir,
        espdl_model_path=args.output,
        calib_steps=args.calib_steps,
    )
