#include "board_display.h"

#include <stddef.h>
#include <string.h>

#include "board_config.h"
#include "gc9b72_init.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_attr.h"
#include "esp_check.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ops.h"
#include "esp_lcd_panel_st7789.h"
#include "esp_log.h"
#include "esp_lvgl_port.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "board_display";

static DMA_ATTR uint8_t s_startup_test_buffer[
    BOARD_MAIN_LCD_H_RES * BOARD_LCD_DRAW_BUFFER_LINES * BOARD_LCD_BYTES_PER_PIXEL];

typedef struct {
    const char *name;
    spi_host_device_t host;
    gpio_num_t mosi_gpio;
    gpio_num_t sclk_gpio;
    gpio_num_t dc_gpio;
    gpio_num_t cs_gpio;
    uint16_t hres;
    uint16_t vres;
    uint16_t x_gap;
    uint16_t y_gap;
} lcd_config_t;

static const lcd_config_t s_main_lcd = {
    .name = "main",
    .host = BOARD_MAIN_LCD_SPI_HOST,
    .mosi_gpio = BOARD_MAIN_LCD_MOSI_GPIO,
    .sclk_gpio = BOARD_MAIN_LCD_SCLK_GPIO,
    .dc_gpio = BOARD_MAIN_LCD_DC_GPIO,
    .cs_gpio = BOARD_MAIN_LCD_CS_GPIO,
    .hres = BOARD_MAIN_LCD_H_RES,
    .vres = BOARD_MAIN_LCD_V_RES,
    .x_gap = BOARD_MAIN_LCD_X_GAP,
    .y_gap = BOARD_MAIN_LCD_Y_GAP,
};

void board_display_set_backlight(bool on)
{
    gpio_set_level(BOARD_LCD_BACKLIGHT_GPIO,
                   on ? BOARD_LCD_BACKLIGHT_ON_LEVEL : BOARD_LCD_BACKLIGHT_OFF_LEVEL);
}

static esp_err_t backlight_init(void)
{
    ESP_LOGI(TAG, "Turn off LCD backlight: GPIO%d", BOARD_LCD_BACKLIGHT_GPIO);
    const gpio_config_t config = {
        .pin_bit_mask = 1ULL << BOARD_LCD_BACKLIGHT_GPIO,
        /* Input is enabled so gpio_get_level() reports the physical pin level. */
        .mode = GPIO_MODE_INPUT_OUTPUT,
        .intr_type = GPIO_INTR_DISABLE,
    };

    ESP_RETURN_ON_ERROR(gpio_config(&config), TAG, "configure backlight GPIO");
    board_display_set_backlight(false);
    return ESP_OK;
}

static esp_err_t show_startup_test(esp_lcd_panel_handle_t panel, const lcd_config_t *lcd)
{
    memset(s_startup_test_buffer, 0xFF, sizeof(s_startup_test_buffer));
    ESP_LOGI(TAG, "RAW white-screen test for %d ms (LVGL bypassed)",
             BOARD_LCD_STARTUP_TEST_MS);

    for (int y = 0; y < lcd->vres; y += BOARD_LCD_DRAW_BUFFER_LINES) {
        const int y_end = (y + BOARD_LCD_DRAW_BUFFER_LINES < lcd->vres)
                        ? y + BOARD_LCD_DRAW_BUFFER_LINES : lcd->vres;
        ESP_RETURN_ON_ERROR(
            esp_lcd_panel_draw_bitmap(panel, 0, y, lcd->hres, y_end,
                                      s_startup_test_buffer),
            TAG, "draw raw white test at y=%d", y);
    }

    vTaskDelay(pdMS_TO_TICKS(BOARD_LCD_STARTUP_TEST_MS));
    ESP_LOGI(TAG, "RAW white-screen test finished");
    return ESP_OK;
}

static esp_err_t add_lcd(const lcd_config_t *lcd, lv_display_t **lvgl_display)
{
    const size_t draw_buffer_pixels = lcd->hres * BOARD_LCD_DRAW_BUFFER_LINES;
    ESP_LOGI(TAG, "Initialize %s SPI bus", lcd->name);
    const spi_bus_config_t bus_config = {
        .mosi_io_num = lcd->mosi_gpio,
        .miso_io_num = GPIO_NUM_NC,
        .sclk_io_num = lcd->sclk_gpio,
        .quadwp_io_num = GPIO_NUM_NC,
        .quadhd_io_num = GPIO_NUM_NC,
        .max_transfer_sz = draw_buffer_pixels * BOARD_LCD_BYTES_PER_PIXEL,
    };
    ESP_RETURN_ON_ERROR(spi_bus_initialize(lcd->host, &bus_config, SPI_DMA_CH_AUTO),
                        TAG, "initialize %s SPI bus", lcd->name);

    const esp_lcd_panel_io_spi_config_t io_config = {
        .dc_gpio_num = lcd->dc_gpio,
        .cs_gpio_num = lcd->cs_gpio,
        .pclk_hz = BOARD_LCD_PIXEL_CLOCK_HZ,
        .lcd_cmd_bits = BOARD_LCD_CMD_BITS,
        .lcd_param_bits = BOARD_LCD_PARAM_BITS,
        .spi_mode = 0,
        .trans_queue_depth = 10,
    };
    ESP_LOGI(TAG, "Install %s panel IO", lcd->name);
    esp_lcd_panel_io_handle_t io = NULL;
    ESP_RETURN_ON_ERROR(
        esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)lcd->host, &io_config, &io),
        TAG, "create %s panel IO", lcd->name);

    const esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = BOARD_LCD_RESET_GPIO,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB,
        .data_endian = LCD_RGB_DATA_ENDIAN_BIG,
        .bits_per_pixel = BOARD_LCD_BITS_PER_PIXEL,
    };
    esp_lcd_panel_handle_t panel = NULL;
    /*
     * GC9B72 uses the standard CASET/RASET/RAMWR/MADCTL command set, so the
     * mature ST7789 operation layer can be reused for drawing and orientation.
     * Its ST7789 initialization is deliberately skipped; gc9b72_panel_init()
     * sends the controller-specific power-on sequence instead.
     */
    ESP_RETURN_ON_ERROR(esp_lcd_new_panel_st7789(io, &panel_config, &panel),
                        TAG, "create %s GC9B72 operation layer", lcd->name);
    ESP_RETURN_ON_ERROR(esp_lcd_panel_reset(panel), TAG, "reset %s panel", lcd->name);
    /* The operation layer waits 20 ms after SWRESET; use 120 ms for GC9B72 startup. */
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_RETURN_ON_ERROR(gc9b72_panel_init(io), TAG, "initialize %s GC9B72 panel", lcd->name);
    ESP_RETURN_ON_ERROR(esp_lcd_panel_set_gap(panel, lcd->x_gap, lcd->y_gap),
                        TAG, "set %s panel gap", lcd->name);

    board_display_set_backlight(true);
    ESP_LOGI(TAG, "Backlight enabled before RAW test: GPIO%d physical level=%d",
             BOARD_LCD_BACKLIGHT_GPIO, gpio_get_level(BOARD_LCD_BACKLIGHT_GPIO));
    ESP_RETURN_ON_ERROR(show_startup_test(panel, lcd), TAG, "run raw LCD test");

    const lvgl_port_display_cfg_t display_config = {
        .io_handle = io,
        .panel_handle = panel,
        .buffer_size = draw_buffer_pixels,
        .double_buffer = true,
        .hres = lcd->hres,
        .vres = lcd->vres,
        .monochrome = false,
        .color_format = LV_COLOR_FORMAT_RGB888,
        .rotation = {
            .swap_xy = false,
            .mirror_x = false,
            .mirror_y = false,
        },
        .flags = {
            .buff_dma = false,
            .buff_spiram = false,
            .swap_bytes = false,
            .sw_rotate = false,
        },
    };
    *lvgl_display = lvgl_port_add_disp(&display_config);
    ESP_RETURN_ON_FALSE(*lvgl_display != NULL, ESP_ERR_NO_MEM, TAG,
                        "register %s display with LVGL", lcd->name);

    ESP_LOGI(TAG, "%s LCD ready: %ux%u, SPI%d, MOSI=%d, SCLK=%d, DC=%d, CS=%d",
             lcd->name, lcd->hres, lcd->vres, lcd->host + 1,
             lcd->mosi_gpio, lcd->sclk_gpio, lcd->dc_gpio, lcd->cs_gpio);
    return ESP_OK;
}

esp_err_t board_display_init(board_displays_t *displays)
{
    ESP_RETURN_ON_FALSE(displays != NULL, ESP_ERR_INVALID_ARG, TAG, "displays is NULL");
    displays->main_display = NULL;

    ESP_RETURN_ON_ERROR(backlight_init(), TAG, "initialize backlight");

    lvgl_port_cfg_t lvgl_config = ESP_LVGL_PORT_INIT_CONFIG();
    lvgl_config.task_priority = 4;
    lvgl_config.task_stack = 6144;
    lvgl_config.task_max_sleep_ms = 20;
    ESP_RETURN_ON_ERROR(lvgl_port_init(&lvgl_config), TAG, "initialize LVGL port");

    ESP_RETURN_ON_ERROR(add_lcd(&s_main_lcd, &displays->main_display),
                        TAG, "initialize main LCD");
    board_display_set_backlight(true);
    ESP_LOGI(TAG, "Turn on LCD backlight: GPIO%d physical level=%d",
             BOARD_LCD_BACKLIGHT_GPIO, gpio_get_level(BOARD_LCD_BACKLIGHT_GPIO));
    return ESP_OK;
}
