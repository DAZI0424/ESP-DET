idf_component_register(
    SRCS "main.c" "camera_web.c" "ingredient_detection.cpp" "recognition_decision.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        esp-dl
        esp32-camera
        esp_app_format
        esp_driver_gpio
        esp_event
        esp_http_server
        esp_netif
        esp_timer
        esp_wifi
        mbedtls
        nvs_flash
)

idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)
set(ingredient_model_path
    ${PROJECT_DIR}/models/welland_v04_yolo_espdet_pico_224_esp32s3_int8_model0_int16.espdl)
file(SIZE ${ingredient_model_path} ingredient_model_expected_bytes)
file(SHA256 ${ingredient_model_path} ingredient_model_sha256)
target_compile_definitions(
    ${COMPONENT_LIB}
    PRIVATE
        WELLAND_MODEL_VERSION="welland-v04-yolo-espdet-pico-224-int8-model0-int16"
        WELLAND_MODEL_EXPECTED_BYTES=${ingredient_model_expected_bytes}
        WELLAND_MODEL_SHA256="${ingredient_model_sha256}"
)
target_add_aligned_binary_data(
    ${COMPONENT_LIB}
    ${ingredient_model_path}
    BINARY
    RENAME_TO ingredient_model_espdl
)
