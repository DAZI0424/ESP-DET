#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define BACKLIGHT_GPIO GPIO_NUM_48

static const char *TAG = "backlight_test";

void app_main(void)
{
    const gpio_config_t config = {
        .pin_bit_mask = 1ULL << BACKLIGHT_GPIO,
        .mode = GPIO_MODE_INPUT_OUTPUT,
        .intr_type = GPIO_INTR_DISABLE,
    };

    ESP_ERROR_CHECK(gpio_set_level(BACKLIGHT_GPIO, 1));
    ESP_ERROR_CHECK(gpio_config(&config));
    ESP_LOGI(TAG, "firmware=BACKLIGHT-DIAGNOSTIC-V1; SPI/LVGL disabled");
    while (true) {
        ESP_ERROR_CHECK(gpio_set_level(BACKLIGHT_GPIO, 1));
        ESP_LOGI(TAG, "Backlight ON: GPIO48 readback=%d; hold 3 seconds",
                 gpio_get_level(BACKLIGHT_GPIO));
        vTaskDelay(pdMS_TO_TICKS(3000));
        ESP_ERROR_CHECK(gpio_set_level(BACKLIGHT_GPIO, 0));
        ESP_LOGI(TAG, "Backlight OFF: GPIO48 readback=%d; hold 3 seconds",
                 gpio_get_level(BACKLIGHT_GPIO));
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}
