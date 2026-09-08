#include "lcd.h"
#include "product_ui_app.h"
#include "sdkconfig.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#if !CONFIG_BOARD_LCD_COLOR_TEST
static void ui_task(void *argument)
{
    (void)argument;
    product_ui_run();
    vTaskDelete(NULL);
}
#endif

void app_main(void)
{
    ESP_LOGI("lcd_demo", "firmware=UI-NATIVE-V1; FPC1 200x648; UI 648x200");
    LCD_Init();
#if !CONFIG_BOARD_LCD_COLOR_TEST
    LCD_SetPixelClock(CONFIG_BOARD_ANIMATION_SPI_MHZ * 1000000U);
    /* LVGL rendering and formatted diagnostics exceed IDF's small main stack. */
    const BaseType_t created = xTaskCreate(ui_task, "lvgl_ui", 12288,
                                           NULL, 5, NULL);
    configASSERT(created == pdPASS);
#else
    const unsigned int colors[] = {0xffff, 0xf800, 0x07e0, 0x001f, 0x0000};
    const char *names[] = {"WHITE", "RED", "GREEN", "BLUE", "BLACK"};
    while (1) {
        for (unsigned int i = 0; i < 5; ++i) {
            DispColor(colors[i]);
            ESP_LOGI("lcd_demo", "Pattern sent: %s", names[i]);
            vTaskDelay(pdMS_TO_TICKS(2000));
        }
        DispBand();
        ESP_LOGI("lcd_demo", "Pattern sent: COLOR BANDS");
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
#endif
}
