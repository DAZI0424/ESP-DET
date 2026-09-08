#include "product_ui_app.h"
#include "ui/product_ui.h"
#include "ui/ui_category_scene.h"
#include "sdkconfig.h"
#include "lcd.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/* Logical landscape coordinates; rotate only at the panel boundary. */
static uint8_t rotated_strip[200 * 8 * 3];
#if CONFIG_BOARD_UI_CATEGORY_DEMO
static ui_category_scene_t category_scene;
#else
static product_ui_t ui;
#endif
static uint32_t tick_ms(void) { return (uint32_t)(esp_timer_get_time() / 1000); }

static void flush(lv_display_t *display, const lv_area_t *area, uint8_t *pixels)
{
    const unsigned width = lv_area_get_height(area);
    for (int x = area->x1; x <= area->x2; x += 8) {
        const unsigned rows = area->x2 - x + 1 < 8 ? area->x2 - x + 1 : 8;
        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned col = 0; col < width; ++col) {
                const uint8_t *src = pixels + ((area->y2 - col) * UI_WIDTH + x + row) * 3;
                uint8_t *dst = rotated_strip + (row * width + col) * 3;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            }
        }
        LCD_DrawBGR888(UI_HEIGHT - 1 - area->y2, x, width, rows, rotated_strip, width * 3);
    }
    lv_display_flush_ready(display);
}

void product_ui_run(void)
{
    const size_t bytes = UI_WIDTH * UI_HEIGHT * 3;
    void *buffer = heap_caps_calloc(1, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    configASSERT(buffer);
    lv_init();
#if CONFIG_BOARD_UI_CATEGORY_DEMO
    /* Transform layers are larger than the internal 64KB LVGL heap. */
    void *ui_pool = heap_caps_malloc(512 * 1024, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    configASSERT(ui_pool);
    lv_mem_pool_t added_pool = lv_mem_add_pool(ui_pool, 512 * 1024);
    configASSERT(added_pool);
#endif
    lv_tick_set_cb(tick_ms);
    lv_display_t *display = lv_display_create(UI_WIDTH, UI_HEIGHT);
    configASSERT(display);
    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB888);
    lv_display_set_buffers_with_stride(display, buffer, NULL, bytes, UI_WIDTH * 3,
                                       LV_DISPLAY_RENDER_MODE_DIRECT);
    lv_display_set_flush_cb(display, flush);
#if CONFIG_BOARD_UI_CATEGORY_DEMO
    ui_category_scene_create(&category_scene, display);
#else
    product_ui_create(&ui, display, &PRODUCT_UI_FIRST_FRAME);
#endif
    lv_refr_now(display);
#if CONFIG_BOARD_UI_CATEGORY_DEMO
    ESP_LOGI("product_ui", "M013-NATIVE ready: 648x200; category sequence=5117ms; PSRAM framebuffer=%u B", (unsigned)bytes);
    bool reported = false;
#else
    ESP_LOGI("product_ui", "UI-NATIVE-V1 ready: 648x200; MiSans; loading cubic-bezier(0.5,0,0,1)/1500ms; PSRAM framebuffer=%u B", (unsigned)bytes);
#endif
    while (1) {
        lv_timer_handler();
#if CONFIG_BOARD_UI_CATEGORY_DEMO
        if (category_scene.finished && !reported) {
            ESP_LOGI("product_ui", "M013 complete: elapsed=%lu ms applied=%u/307 final=sodium; holding",
                     (unsigned long)lv_tick_elaps(category_scene.started), category_scene.applied_frames);
            reported = true;
        }
#endif
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
