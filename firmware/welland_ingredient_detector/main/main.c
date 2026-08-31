#include <stdbool.h>
#include <stdint.h>

#include "esp_camera.h"
#include "esp_check.h"
#include "esp_err.h"
#include "esp_log.h"
#include "sdkconfig.h"

#include "camera_web.h"
#include "ingredient_detection.h"

static const char *TAG = "gc2145";

#ifdef CONFIG_WELLAND_CAM_HMIRROR
#define WELLAND_CAM_HMIRROR_ENABLED 1
#else
#define WELLAND_CAM_HMIRROR_ENABLED 0
#endif

#ifdef CONFIG_WELLAND_CAM_VFLIP
#define WELLAND_CAM_VFLIP_ENABLED 1
#else
#define WELLAND_CAM_VFLIP_ENABLED 0
#endif

#ifdef CONFIG_WELLAND_CAM_AWB_GRAY_WORLD
#define WELLAND_CAM_AWB_GRAY_WORLD_ENABLED 1
#else
#define WELLAND_CAM_AWB_GRAY_WORLD_ENABLED 0
#endif

enum {
    GC2145_PAGE_SELECT = 0xfe,
    GC2145_PAGE_0 = 0x00,
    GC2145_PAGE_1 = 0x01,
    GC2145_ISP_AUTO_CONTROL = 0x82,
    GC2145_AEC_ENABLE = 0xb6,
    GC2145_AEC_TARGET = 0x13,
    GC2145_AWB_GRAY_MODE = 0x63,
};

static const camera_config_t CAMERA_CONFIG = {
    .pin_pwdn = CONFIG_WELLAND_CAM_PIN_PWDN,
    .pin_reset = CONFIG_WELLAND_CAM_PIN_RESET,
    .pin_xclk = CONFIG_WELLAND_CAM_PIN_XCLK,
    .pin_sccb_sda = CONFIG_WELLAND_CAM_PIN_SIOD,
    .pin_sccb_scl = CONFIG_WELLAND_CAM_PIN_SIOC,
    .pin_d0 = CONFIG_WELLAND_CAM_PIN_D0,
    .pin_d1 = CONFIG_WELLAND_CAM_PIN_D1,
    .pin_d2 = CONFIG_WELLAND_CAM_PIN_D2,
    .pin_d3 = CONFIG_WELLAND_CAM_PIN_D3,
    .pin_d4 = CONFIG_WELLAND_CAM_PIN_D4,
    .pin_d5 = CONFIG_WELLAND_CAM_PIN_D5,
    .pin_d6 = CONFIG_WELLAND_CAM_PIN_D6,
    .pin_d7 = CONFIG_WELLAND_CAM_PIN_D7,
    .pin_vsync = CONFIG_WELLAND_CAM_PIN_VSYNC,
    .pin_href = CONFIG_WELLAND_CAM_PIN_HREF,
    .pin_pclk = CONFIG_WELLAND_CAM_PIN_PCLK,
    .xclk_freq_hz = CONFIG_WELLAND_CAM_XCLK_MHZ * 1000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,
    .pixel_format = PIXFORMAT_RGB565,
    .frame_size = FRAMESIZE_VGA,
    .jpeg_quality = 12,
    .fb_count = 2,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    .sccb_i2c_port = -1,
};

static esp_err_t gc2145_write_register(sensor_t *sensor,
                                       uint8_t page,
                                       uint8_t reg,
                                       uint8_t mask,
                                       uint8_t value)
{
    if (sensor->set_reg == NULL ||
        sensor->set_reg(sensor, GC2145_PAGE_SELECT, 0xff, page) != 0) {
        return ESP_FAIL;
    }
    const int write_result = sensor->set_reg(sensor, reg, mask, value);
    const int restore_result =
        sensor->set_reg(sensor, GC2145_PAGE_SELECT, 0xff, GC2145_PAGE_0);
    return write_result == 0 && restore_result == 0 ? ESP_OK : ESP_FAIL;
}

static esp_err_t configure_gc2145(sensor_t *sensor)
{
    /* Horizontal mirroring is applied once in the Core 0 RGB565 capture path. */
    if (sensor->set_hmirror(sensor, 0) != 0 ||
        sensor->set_vflip(sensor, WELLAND_CAM_VFLIP_ENABLED) != 0) {
        return ESP_FAIL;
    }

    ESP_RETURN_ON_ERROR(gc2145_write_register(sensor,
                                               GC2145_PAGE_0,
                                               GC2145_ISP_AUTO_CONTROL,
                                               0x02,
                                               0x02),
                        TAG,
                        "enable AWB");
    ESP_RETURN_ON_ERROR(gc2145_write_register(sensor,
                                               GC2145_PAGE_0,
                                               GC2145_AEC_ENABLE,
                                               0x01,
                                               0x01),
                        TAG,
                        "enable AEC");
    ESP_RETURN_ON_ERROR(gc2145_write_register(sensor,
                                               GC2145_PAGE_1,
                                               GC2145_AEC_TARGET,
                                               0xff,
                                               CONFIG_WELLAND_CAM_AEC_TARGET),
                        TAG,
                        "set AEC target");
    ESP_RETURN_ON_ERROR(gc2145_write_register(sensor,
                                               GC2145_PAGE_1,
                                               GC2145_AWB_GRAY_MODE,
                                               0x80,
                                               WELLAND_CAM_AWB_GRAY_WORLD_ENABLED ? 0x80 : 0x00),
                        TAG,
                        "set AWB gray-world mode");
    return ESP_OK;
}

static void log_rgb565_statistics(const camera_fb_t *frame)
{
    uint64_t red_sum = 0;
    uint64_t green_sum = 0;
    uint64_t blue_sum = 0;
    size_t low_channels = 0;
    size_t high_channels = 0;
    const size_t pixels = frame->len / 2U;

    for (size_t index = 0; index < pixels; ++index) {
        const uint16_t pixel =
            ((uint16_t)frame->buf[index * 2U] << 8U) | frame->buf[index * 2U + 1U];
        const uint8_t red = (uint8_t)((((pixel >> 11U) & 0x1fU) * 255U) / 31U);
        const uint8_t green = (uint8_t)((((pixel >> 5U) & 0x3fU) * 255U) / 63U);
        const uint8_t blue = (uint8_t)(((pixel & 0x1fU) * 255U) / 31U);
        red_sum += red;
        green_sum += green;
        blue_sum += blue;
        low_channels += (red < 8U) + (green < 8U) + (blue < 8U);
        high_channels += (red > 247U) + (green > 247U) + (blue > 247U);
    }

    const uint64_t channel_samples = pixels * 3U;
    ESP_LOGI(TAG,
             "settled RGB mean=%u/%u/%u clipped_low=%u.%02u%% clipped_high=%u.%02u%%",
             (unsigned)(red_sum / pixels),
             (unsigned)(green_sum / pixels),
             (unsigned)(blue_sum / pixels),
             (unsigned)((low_channels * 100U) / channel_samples),
             (unsigned)(((low_channels * 10000U) / channel_samples) % 100U),
             (unsigned)((high_channels * 100U) / channel_samples),
             (unsigned)(((high_channels * 10000U) / channel_samples) % 100U));
}

static esp_err_t warm_up_gc2145(void)
{
    for (int index = 0; index < CONFIG_WELLAND_CAM_WARMUP_FRAMES; ++index) {
        camera_fb_t *frame = esp_camera_fb_get();
        if (frame == NULL) {
            return ESP_FAIL;
        }
        if (index == CONFIG_WELLAND_CAM_WARMUP_FRAMES - 1) {
            log_rgb565_statistics(frame);
        }
        esp_camera_fb_return(frame);
    }
    return ESP_OK;
}

void app_main(void)
{
    ESP_ERROR_CHECK(camera_web_start());
    camera_web_set_camera_status(false, 0, "initializing");

    const esp_err_t err = esp_camera_init(&CAMERA_CONFIG);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed: %s (0x%x)", esp_err_to_name(err), err);
        camera_web_set_camera_status(false, 0, "camera_init_failed");
        return;
    }

    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor == NULL || sensor->id.PID != GC2145_PID) {
        const uint16_t pid = sensor == NULL ? 0 : sensor->id.PID;
        ESP_LOGE(TAG, "expected GC2145 PID 0x%04x, got 0x%04x", GC2145_PID, pid);
        esp_camera_deinit();
        camera_web_set_camera_status(false, pid, "unexpected_sensor");
        return;
    }

    ESP_ERROR_CHECK(configure_gc2145(sensor));
    camera_web_set_camera_status(false, sensor->id.PID, "awb_aec_warmup");
    ESP_ERROR_CHECK(warm_up_gc2145());
    ESP_ERROR_CHECK(ingredient_detection_start());
    ESP_ERROR_CHECK(camera_web_start_pipeline());
    ESP_LOGI(TAG,
             "GC2145 ready: PID=0x%04x, Core0 capture/display/Wi-Fi + Core1 AI pipeline, "
             "RGB565 640x480 center-crop 480x480 -> 224x224, "
             "XCLK=%d MHz, software_hmirror=%d, vflip=%d, gray_world=%d, AEC_target=%d",
             sensor->id.PID,
             CONFIG_WELLAND_CAM_XCLK_MHZ,
             WELLAND_CAM_HMIRROR_ENABLED,
             WELLAND_CAM_VFLIP_ENABLED,
             WELLAND_CAM_AWB_GRAY_WORLD_ENABLED,
             CONFIG_WELLAND_CAM_AEC_TARGET);
    camera_web_set_camera_status(true, sensor->id.PID, "ready");
}
