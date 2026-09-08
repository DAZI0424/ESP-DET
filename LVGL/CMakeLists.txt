cmake_minimum_required(VERSION 3.16)

# This board always uses ESP32-S3. Keeping the target here prevents accidental
# builds for the default ESP32 target when the project is opened for the first time.
set(IDF_TARGET "esp32s3" CACHE STRING "ESP-IDF target")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
set(PROJECT_VER "ui-native-v1")
project(zhuanjie_lvgl)
