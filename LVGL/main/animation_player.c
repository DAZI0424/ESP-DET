#include "animation_player.h"
#include "animation_decode.h"
#include "assets/animation_index.h"
#include "lcd.h"
#include "lvgl.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_rom_crc.h"
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#define FRAME_BYTES (ANIMATION_WIDTH * ANIMATION_HEIGHT * 3U)
static const char *TAG = "animation";
extern const uint8_t asset_start[] asm("_binary_animation_bin_start");
extern const uint8_t asset_end[] asm("_binary_animation_bin_end");
static int64_t frame_spi_us;
static uint32_t frame_pixels;
typedef struct {
    uint64_t loop;
    int64_t elapsed_us, max_us, total_us, decode_us, spi_us;
    uint32_t pixels, overruns, stack_free;
} playback_stats_t;
static QueueHandle_t stats_queue;

static void diagnostics_task(void *argument)
{
    (void)argument;
    playback_stats_t s;
    while (true) {
        if (xQueueReceive(stats_queue, &s, portMAX_DELAY) != pdTRUE) continue;
        ESP_LOGI(TAG, "loop=%" PRIu64 " elapsed=%" PRId64 "ms fps=%.2f max_frame=%" PRId64
                 "us schedule_overruns=%" PRIu32 " stack_free=%" PRIu32 "B",
                 s.loop, s.elapsed_us / 1000, ANIMATION_FRAME_COUNT * 1000000.0 / s.elapsed_us,
                 s.max_us, s.overruns, s.stack_free);
        ESP_LOGI(TAG, "profile avg_total=%" PRId64 "us avg_decode=%" PRId64
                 "us avg_spi=%" PRId64 "us avg_pixels=%" PRIu32,
                 s.total_us / ANIMATION_FRAME_COUNT, s.decode_us / ANIMATION_FRAME_COUNT,
                 s.spi_us / ANIMATION_FRAME_COUNT, s.pixels / ANIMATION_FRAME_COUNT);
    }
}

static uint32_t tick_ms(void) { return (uint32_t)(esp_timer_get_time() / 1000); }

/* LVGL DIRECT mode passes the complete framebuffer base, not a packed region.
 * A transparent LVGL screen preserves the externally decoded video pixels. */
static void flush(lv_display_t *display, const lv_area_t *area, uint8_t *pixels)
{
    const int64_t started = esp_timer_get_time();
    const unsigned int width = lv_area_get_width(area);
    const unsigned int height = lv_area_get_height(area);
    const unsigned int stride = ANIMATION_WIDTH * 3U;
    for (unsigned int row = 0; row < height; row += 8) {
        const unsigned int rows = height - row < 8 ? height - row : 8;
        LCD_DrawBGR888(area->x1, area->y1 + row, width, rows,
                       pixels + (area->y1 + row) * stride + area->x1 * 3U, stride);
    }
    frame_pixels += width * height;
    frame_spi_us += esp_timer_get_time() - started;
    lv_display_flush_ready(display);
}

static void *frame_alloc(void)
{
    void *result = heap_caps_malloc(FRAME_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!result) {
        ESP_LOGE(TAG, "Cannot allocate %u bytes in PSRAM", (unsigned int)FRAME_BYTES);
        abort();
    }
    return result;
}

static void apply_frame(uint8_t *canvas, unsigned int frame)
{
    const uint32_t begin = animation_offsets[frame], end = animation_offsets[frame + 1];
    if (end < begin || end > (size_t)(asset_end - asset_start) ||
        !animation_decode(canvas, ANIMATION_WIDTH * ANIMATION_HEIGHT,
                          asset_start + begin, end - begin)) {
        ESP_LOGE(TAG, "Invalid animation packet at frame %u", frame);
        abort();
    }
}

void animation_player_run(void)
{
    uint8_t *canvas_pixels = frame_alloc();
    lv_init();
    lv_tick_set_cb(tick_ms);
    lv_display_t *display = lv_display_create(ANIMATION_WIDTH, ANIMATION_HEIGHT);
    configASSERT(display);
    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB888);
    lv_display_set_buffers_with_stride(display, canvas_pixels, NULL, FRAME_BYTES,
                                        ANIMATION_WIDTH * 3U, LV_DISPLAY_RENDER_MODE_DIRECT);
    lv_display_set_flush_cb(display, flush);
    lv_obj_t *screen = lv_display_get_screen_active(display);
    lv_obj_remove_style_all(screen);
    lv_obj_set_style_bg_opa(screen, LV_OPA_TRANSP, 0);
    lv_obj_remove_flag(screen, LV_OBJ_FLAG_SCROLLABLE);
    apply_frame(canvas_pixels, 0);
    const uint32_t initial_crc = esp_rom_crc32_le(0, canvas_pixels, FRAME_BYTES);
    lv_refr_now(display);
    /* Fail visibly if a future theme/overlay accidentally paints over video. */
    configASSERT(esp_rom_crc32_le(0, canvas_pixels, FRAME_BYTES) == initial_crc);
    stats_queue = xQueueCreate(1, sizeof(playback_stats_t));
    configASSERT(stats_queue);
    configASSERT(xTaskCreate(diagnostics_task, "video_stats", 6144, NULL, 1, NULL) == pdPASS);
    ESP_LOGI(TAG, "110 lossless frames, source 30 fps, target %d fps, %.3f ms loop; clockwise, no scaling",
             CONFIG_BOARD_ANIMATION_FPS, ANIMATION_FRAME_COUNT * 1000.0 / CONFIG_BOARD_ANIMATION_FPS);
    ESP_LOGI(TAG, "LVGL DIRECT framebuffer=%u PSRAM bytes; no duplicate render; startup CRC verified",
             (unsigned int)FRAME_BYTES);

    /* Absolute source timeline: an expensive frame never shifts subsequent
     * deadlines. Keep every frame and use the following light frames to catch
     * up. Sustained throughput must exceed the configured playback rate. */
    int64_t origin = esp_timer_get_time(), loop_start = origin;
    uint64_t number = 1;
    int64_t max_render_us = 0;
    unsigned int overruns = 0;
    int64_t total_spi_us = 0, total_render_us = 0, total_decode_us = 0;
    uint32_t total_pixels = 0;
    while (true) {
        const int64_t deadline = origin + (int64_t)(number * 1000000ULL / CONFIG_BOARD_ANIMATION_FPS);
        int64_t remaining;
        while ((remaining = deadline - esp_timer_get_time()) > 0) {
            TickType_t ticks = pdMS_TO_TICKS((remaining + 999) / 1000);
            vTaskDelay(ticks ? ticks : 1);
        }
        const int64_t started = esp_timer_get_time();
        if (started - deadline > 1000000 / CONFIG_BOARD_ANIMATION_FPS) {
            ++overruns;
        }
        const unsigned int frame = number % ANIMATION_FRAME_COUNT;
        frame_spi_us = 0;
        frame_pixels = 0;
        apply_frame(canvas_pixels, frame == 0 ? ANIMATION_LOOP_PACKET : frame);
        total_decode_us += esp_timer_get_time() - started;
        unsigned int pending = 0;
        for (uint32_t i = animation_rect_offsets[frame]; i < animation_rect_offsets[frame + 1]; ++i) {
            const animation_rect_t *rect = &animation_rects[i];
            const lv_area_t area = {rect->x1, rect->y1, rect->x2, rect->y2};
            lv_obj_invalidate_area(screen, &area);
            /* Batch below LVGL's 32 invalid-area capacity to amortize its
             * refresh setup without falling back to full-screen refresh. */
            if (++pending == 16) {
                lv_refr_now(display);
                pending = 0;
            }
        }
        if (pending) lv_refr_now(display);
        const int64_t finished = esp_timer_get_time();
        const int64_t render_us = finished - started;
        total_render_us += render_us;
        total_spi_us += frame_spi_us;
        total_pixels += frame_pixels;
        if (render_us > max_render_us) max_render_us = render_us;
        if (number % ANIMATION_FRAME_COUNT == 0) {
            const playback_stats_t stats = {
                .loop = number / ANIMATION_FRAME_COUNT,
                .elapsed_us = finished - loop_start, .max_us = max_render_us,
                .total_us = total_render_us, .decode_us = total_decode_us, .spi_us = total_spi_us,
                .pixels = total_pixels, .overruns = overruns,
                .stack_free = uxTaskGetStackHighWaterMark(NULL),
            };
            xQueueOverwrite(stats_queue, &stats);
            loop_start = finished;
            max_render_us = 0;
            overruns = 0;
            total_render_us = total_decode_us = total_spi_us = 0;
            total_pixels = 0;
        }
        ++number;
        vTaskDelay(1); /* Let idle run even when the bus cannot keep up. */
    }
}
