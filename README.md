# Welland ESPDet

用于 ESPDet-Pico 模型训练、ESP32-S3 INT8 量化和板端部署。

## 目录

```text
cfg/        模型配置
nn/         ESPDet 网络实现
scripts/    训练、数据划分和校准集选择
deploy/     ONNX 导出、ESP-PPQ 量化和精度比较
firmware/   ESP32-S3 固件、当前部署模型和 Flash 核验工具
datasets/   原始/整理后的数据集（受保护）
runs/       训练、量化和评估产物（受保护）
```

`.venv-train` 用于 CUDA 训练，`.venv` 用于导出、量化及辅助工具。校准集等中间产物统一写入 `runs/`，不要再放进 `deploy/`。

## 常用命令

```powershell
# 训练
.\.venv-train\Scripts\python.exe .\scripts\train_welland_v04_espdet_pico.py

# 导出 ONNX
.\.venv\Scripts\python.exe .\deploy\export.py --model .\runs\detect\<run>\weights\best.pt --imgsz 416

# 生成 500 张均衡校准集
.\.venv\Scripts\python.exe .\scripts\select_espdl_calibration.py `
  --dataset .\datasets\welland-v04-espdet-pico\welland-v04-grouped.yaml `
  --split train --output .\runs\calibration\welland-v04 `
  --per-class 64 --background 52

# INT8 量化
.\.venv\Scripts\python.exe .\deploy\quantize_espdet_esp32s3.py `
  --onnx .\runs\detect\<run>\weights\best.onnx `
  --calib-dir .\runs\calibration\welland-v04 `
  --output .\runs\detect\<run>\quantized_esp32s3\model.espdl `
  --imgsz 416 --target esp32s3 --bits 8

# 构建 ESP32-S3 固件
.\firmware\welland_ingredient_detector\tools\idf.ps1 build

# 连接板卡后读回并核验 Flash（替换 COM6）
.\.venv\Scripts\python.exe `
  .\firmware\welland_ingredient_detector\tools\verify_flashed_model.py `
  --port COM6
```

当前固件部署模型：
`firmware/welland_ingredient_detector/models/welland_v04_espdet_pico_416_esp32s3_int8_optimized.espdl`。
