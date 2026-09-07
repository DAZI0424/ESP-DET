# ESP32-S3 N16R8 LVGL 基础工程

当前目标：点亮已接入 FPC1 的长方屏背光。`main/CMakeLists.txt` 仅编译已有的 `backlight_test.c` 作为入口，GPIO48 持续输出高电平，不发送屏幕初始化命令、不启动 LVGL。两路背光共用此控制信号。

## 基础配置

| 项目 | 配置 |
|---|---|
| 主芯片 | 用户指定 ESP32-S3 N16R8；容量以实物为准 |
| Flash | 16 MB，QIO，80 MHz |
| PSRAM | 8 MB Octal，80 MHz，启动初始化与内存测试 |
| CPU / FreeRTOS | 240 MHz / 1000 Hz |
| 调试串口 | UART0，115200，TX GPIO43 / RX GPIO44 |
| 软件 | 本机 ESP-IDF 6.0.2；LVGL ~9.4.0；esp_lvgl_port ^2.9.0 |
| 分区 | NVS 24 KiB、PHY 4 KiB、factory 7 MiB、storage 9152 KiB；无 OTA |

当前固件版本为 `backlight-on-v1`，启动应输出 `GPIO48 HIGH, physical level=1`。该日志表示 GPIO 输入回读为高，实际背光是否点亮仍需观察屏幕。PSRAM 启动失败会由 IDF 报错停止。

## 原理图核对

依据 `ESP32-S3-CAM原理图.pdf` 与 `Zhuanjie_Project.pdf`：

| 转接板信号 | GPIO / 连接 |
|---|---|
| FPC1 MOSI / SCLK / DC / CS | 47 / 21 / 20 / 19 |
| 共用背光 LED_EN | 48，高电平开启；同时连接主板 WS2812 数据端，不运行 RGB LED 驱动 |
| 屏幕 RST | 接 EN，不能作为独立 GPIO 控制 |
| FPC2 MOSI / SCLK / DC / CS | 1 / 2 / 41 / 42；目前不启用 |

使用主板标注 **ESP32S3-CH340** 的 USB 接口下载和查看串口日志。GPIO19/20 接到 FPC1，也接到主板原生 USB/OTG，已关闭次级 USB Serial/JTAG 控制台；接屏时不要将 OTG 接口连接 USB 主机。

N16R8 的 GPIO35/36/37 被 Octal PSRAM 使用，不作为外设 GPIO。主板 P2 的 1/2 脚标为 RXD0/TXD0，而转接板 J2 的 1/2 脚标为 TXD0/RXD0，标签顺序相反；目前这两根线没有接屏幕，后续从转接板引出 UART 时按主板实际连线核对。

参考：https://docs.espressif.com/projects/esp-idf/en/v6.0/esp32s3/api-reference/peripherals/gpio.html

## 屏幕初始化资料

用户提供 `IVO2.86＋GC9B72NA_V1_20250320_MIPI_2COL(1).txt`，标注 200×648、40 Hz、VCI/IOVCC 2.8 V。现有 `main/gc9b72_init.c` 已包含这份表的命令序列，暂不重复替换。

文件名标注 MIPI，待到货屏资料标注 SPI，不能仅凭同为 GC9B72NA 就认定接口设置及供电参数适用。后续需对照实际模组资料验证；目前初始化表和屏幕驱动均不执行。现有代码使用 10 MHz SPI、18 位像素传输，LVGL 内部为 RGB565，尚未完成实屏验证。

## 编译与下载

在 ESP-IDF 6.0.2 命令行环境执行：

```powershell
cd D:\ESP-Project\20260729ESP-DET\LVGL
idf.py build
idf.py -p COMx flash monitor
```

将 COMx 替换为 CH340 实际端口，退出监视器使用 Ctrl+]。项目已固定 esp32s3，无需反复执行 set-target。

本次为纯背光测试，menuconfig 中的 FPC1 开关不影响此入口。后续需要显示画面时，再恢复 app_main.c、board_display.c、gc9b72_init.c、ui_demo.c 的编译入口及相应依赖。
