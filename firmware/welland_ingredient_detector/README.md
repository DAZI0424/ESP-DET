# ESP32-S3 GC2145 食材识别

目标硬件：HW-679 / ESP32-S3-N16R8 + GC2145。

当前固件配置：

- ESPDet-Pico v04，默认 INT8、仅 `/model.0/conv/Conv` 使用 INT16，输入 `1×224×224×3`，6 类食材。
- GC2145 以 10 MHz XCLK 采集 VGA RGB565，中心裁剪 480×480 后缩放到 224×224。
- GC2145 启用灰世界 AWB、AEC 目标 68，并丢弃前 12 帧等待自动白平衡和曝光稳定。
- 默认水平镜像开启、垂直翻转关闭；这些选项可在 `menuconfig` 中修改。
- 双摄像头帧缓冲、`CAMERA_GRAB_WHEN_EMPTY`；调试配置关闭 PSRAM 直连 DMA。
- 每个 MJPEG 帧先完成本帧推理，再把本帧结果画到本帧上，避免旧框落到新画面。
- 端口 80 提供 GUI、状态、原始帧和验证接口；端口 81 提供 MJPEG。

类别索引：

1. apple
2. strawberry
3. cherry tomato
4. banana
5. egg
6. Lettuce

生产阈值为 score `0.75`、NMS `0.70`。实时链路采用逐帧同步推理，因此画面帧率由单次采集、推理和 JPEG 编码耗时共同决定。

## 识别锁定流程

GUI 打开后进入“识别中”状态，但识别响应计时只从第一个合格且实际进入证据累计的推理帧开始。每个新鲜摄像头帧先经过运动、清晰度、曝光和内容重复门控；不合格帧不会执行或累计为有效推理证据。首个证据前只检测画面中央 70% 展示区，首个证据后切换为首帧目标框向外扩展 20% 的固定 ROI；运动采用进入 45‰、退出 80‰ 的滞回阈值，并要求连续 2 个稳定采样，避免用户从背景经过时反复触发或延长识别计时。只有全画面连续 2 帧变化像素比例达到 600‰、同时食材 ROI 运动超过 80‰ 才判定场景切换，不把单次或 ROI 外的背景运动作为食材识别触发。

合格帧保留六类置信度、Top-1/Top-2、差值和目标框，并按 `quality_weight × class_confidence` 累计每类证据。第一帧同时满足类别高置信度、差值和目标框阈值时立即以高可信度锁定；否则第二个有效帧的归一化累计证据及差值达标时以中可信度锁定；否则第三个有效帧完成后强制以累计证据 Top-1 作为低可信度结果。最多只执行三次有效推理，不使用简单多数投票，也不会用第三帧 Top-1 覆盖前两帧。

多帧证据仅在目标框位置和面积连续、没有明显换物或场景切换的同一次展示中累计；目标消失、目标框异常跳变或明显换物会清空本轮证据。类别锁定后停止模型推理，用户把食材移向秤面不会触发重新识别或改变类别。称重由显式 `POST /weighing/start` 进入，`POST /weighing/complete` 或 `POST /recognition/cancel` 才清除锁定并复位；识别路径不读取重量变化，也不使用秤面目标作为触发条件。

GUI 实时展示质量门控、稳定确认、ROI 状态、全局变化比例、质量权重、运动量、清晰度、亮度、三步有效证据进度、累计证据 Top-1/Top-2、累计差值、一帧/两帧/三帧输出比例和平均有效推理帧数。锁定后通过“开始称重”显式进入称重状态，通过“称重完成”或“取消本次”清除锁定；“重新识别”会清除累计证据并恢复模型推理。页面及导出的 Excel 分别记录从首个有效证据到锁定的“识别响应耗时”和从本轮开始到结束的“完整操作耗时”，以及识别序列号、类别、可信度等级、实际有效推理帧数和人工真实食材。低分辨率运动算法只用于当前展示 ROI 的画面质量门控；背景运动不会启动识别计时，也不是识别或称重触发信号。

GUI 提供“选择保存文件夹”“开启/关闭识别框”和“拍照保存”。识别框默认开启，关闭仅停止在视频输出上画框，不停止模型推理、质量门控、证据累计或类别锁定；切换后拍照会保存当前显示的无框 MJPEG 帧。支持 File System Access API 的浏览器可以在当前页面会话中选择本地目录，照片以 `gc2145_YYYYMMDD_HHMMSS_mmm_cN.jpg` 命名；受非安全 HTTP 页面限制而不能直接写入目录时，自动回退到浏览器下载或另存为流程。拍照不暂停视频流，也不产生额外推理证据。

## 模型

固件嵌入：

```text
models/welland_v04_yolo_espdet_pico_224_esp32s3_int8_model0_int16.espdl
bytes: 498096
sha256: 59441a53d3c9c1f108552bb4d068d4d97db53503daf2c67714cc12d0e7039b93
```

该模型使用 500 张 224×224 校准图、percentile 校准、bias correction 和 TQT 500。在完整 599 张验证集和一致的 `conf=0.25` 下，混合量化模型的 mAP50 为 `0.988931`，对应 FP32 ONNX 为 `0.988409`；mAP50-95 分别为 `0.872550` 和 `0.887618`。

新的无泄漏划分位于：

```text
datasets/welland-v04-yolo/data-local.yaml
```

当前数据集包含训练图 4,163 张、验证图 599 张。板端验证必须使用这里的 6 类顺序，不能继续使用此前版本的报告作为本固件的验证依据。

## 构建与烧录

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\idf.ps1 build
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\idf.ps1 -p COM6 flash monitor
```

当前固件：

```text
build/welland_ingredient_detector.bin
bytes: 3251376
sha256: 44cf84f9cba3a558e90277e121fc47aba6abc09eb69115be56b67c0049f390e9
```

启动日志会输出暖机结束时的 RGB 通道均值和高低光截断比例，例如：

```text
settled RGB mean=.../.../... clipped_low=...% clipped_high=...%
```

推荐在均匀白光下放置白纸或灰卡进行判断。三个通道越接近越好；如果仍有明显偏色，先记录这行日志和 `/capture.jpg`，再调整 GC2145 颜色矩阵，不要直接改变模型输入归一化。

## 采集 GC2145 实机数据

关闭浏览器中的 MJPEG 页面，然后执行：

```powershell
..\..\.venv\Scripts\python.exe .\tools\collect_gc2145_frames.py `
  --base-url http://192.168.4.1 `
  --output ..\..\datasets\gc2145-captures\session-001 `
  --count 100
```

`GET /capture.jpg` 返回未画框的 VGA JPEG。采集结果必须人工标注为 YOLO 检测格式后才能加入训练集；不要使用模型自动预测作为最终标注。

## 板端验证

关闭 MJPEG 页面，先执行 100 张冒烟测试：

```powershell
..\..\.venv\Scripts\python.exe .\tools\board_validate.py `
  --base-url http://192.168.4.1 `
  --dataset ..\..\datasets\welland-v04-yolo\data-local.yaml `
  --limit 100 `
  --output .\board_validation_report_100.json
```

冒烟测试通过后移除 `--limit 100`，运行完整 599 张验证集。报告包含板端 mAP、精确率、召回率、预处理/推理/后处理延迟和内存水位，使用固件生产阈值。

Wi-Fi 默认连接 `guest`，备用热点为 `GC2145-CAM / gc2145cam`，热点地址为 `http://192.168.4.1/`。
