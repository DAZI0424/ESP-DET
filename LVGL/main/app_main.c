#include "board_display.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "ui_demo.h"
#include "board_config.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "sdkconfig.h"

static const char *TAG = "zhuanjie_lvgl";
#define FIRMWARE_MARKER "N16R8-BASE-NO-SCREEN-V1"

void app_main(void)
{
    ESP_LOGI(TAG, "firmware=%s", FIRMWARE_MARKER);
    ESP_LOGI(TAG, "ESP32-S3-WROOM-1-N16R8 LVGL bring-up");
    ESP_LOGI(TAG, "IDF %s; reset reason=%d", esp_get_idf_version(), esp_reset_reason());
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    uint32_t flash_size = 0;
    ESP_ERROR_CHECK(esp_flash_get_size(NULL, &flash_size));
    ESP_LOGI(TAG, "cores=%d; revision=%d; Flash=%lu bytes", chip.cores,
             chip.revision, (unsigned long)flash_size);
    ESP_LOGI(TAG, "PSRAM detected: %u bytes", (unsigned int)esp_psram_get_size());
    if (flash_size != 16 * 1024 * 1024 || esp_psram_get_size() != 8 * 1024 * 1024) {
        ESP_LOGE(TAG, "Memory size differs from expected N16R8; check physical module");
    }
    ESP_LOGI(TAG, "Free internal RAM=%u; free PSRAM=%u bytes",
             (unsigned int)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
             (unsigned int)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    /* Shared backlight transistor: keep both displays dark during bring-up. */
    ESP_ERROR_CHECK(gpio_set_level(BOARD_LCD_BACKLIGHT_GPIO, BOARD_LCD_BACKLIGHT_OFF_LEVEL));
    const gpio_config_t backlight = {
        .pin_bit_mask = 1ULL << BOARD_LCD_BACKLIGHT_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&backlight));

#if CONFIG_BOARD_ENABLE_FPC1
    board_displays_t displays;
    ESP_ERROR_CHECK(board_display_init(&displays));
    ui_demo_create(displays.main_display);

    ESP_LOGI(TAG, "LVGL demo started on FPC1 display");
#else
    ESP_LOGI(TAG, "Base bring-up ready; FPC1/FPC2 disabled, backlight off");
#endif
}
