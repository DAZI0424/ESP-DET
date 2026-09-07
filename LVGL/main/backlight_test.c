#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_log.h"

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
    ESP_LOGI(TAG, "GPIO48 HIGH, physical level=%d", gpio_get_level(BACKLIGHT_GPIO));
}
