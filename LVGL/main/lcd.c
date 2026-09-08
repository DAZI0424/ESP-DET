/* Port of ../lcd/lcd.c for ESP32-S3 FPC1.
 * Original C8051 GPIO/CLKSEL/code/reentrant constructs are replaced by IDF.
 * The supplied example only implements GC9305 initialization (RGB565).
 * This board uses GC9B72NA: the separate vendor SPI&MCU table is used verbatim.
 */
#include "lcd.h"
#include "board_config.h"
#include <assert.h>
#include <stdint.h>
#include "esp_err.h"
#include "esp_log.h"
#include "esp_attr.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static spi_device_handle_t lcd_spi;
static uint8_t line_buffer[BOARD_MAIN_LCD_H_RES * 3];
static DMA_ATTR uint8_t pixel_buffer[BOARD_MAIN_LCD_H_RES * 8 * 3];

/* Blocking transfers: DC and CS cannot change before the last bit is sent. */
static void send_bytes(const void *data, size_t length)
{
    spi_transaction_t transaction = {
        .length = length * 8,
        .tx_buffer = data,
    };
    ESP_ERROR_CHECK(spi_device_polling_transmit(lcd_spi, &transaction));
}

void SPI_SendData(unsigned char value)
{
    send_bytes(&value, 1);
}

static void write_byte(int dc, unsigned char value)
{
    ESP_ERROR_CHECK(gpio_set_level(BOARD_MAIN_LCD_DC_GPIO, dc));
    ESP_ERROR_CHECK(gpio_set_level(BOARD_MAIN_LCD_CS_GPIO, 0));
    SPI_SendData(value);
    ESP_ERROR_CHECK(gpio_set_level(BOARD_MAIN_LCD_CS_GPIO, 1));
}

void SPI_WriteComm(unsigned char value) { write_byte(0, value); }
void SPI_WriteData(unsigned char value) { write_byte(1, value); }

static void write_pixels(const uint8_t *data, size_t bytes)
{
    ESP_ERROR_CHECK(gpio_set_level(BOARD_MAIN_LCD_DC_GPIO, 1));
    ESP_ERROR_CHECK(gpio_set_level(BOARD_MAIN_LCD_CS_GPIO, 0));
    send_bytes(data, bytes);
    ESP_ERROR_CHECK(gpio_set_level(BOARD_MAIN_LCD_CS_GPIO, 1));
}

/* 3A=06: expand RGB565 into three high-aligned six-bit channel bytes. */
static void encode_color(uint16_t color, uint8_t out[3])
{
    unsigned int r = (color >> 11) & 31;
    unsigned int g = (color >> 5) & 63;
    unsigned int b = color & 31;
    out[0] = ((r << 1) | (r >> 4)) << 2;
    out[1] = g << 2;
    out[2] = ((b << 1) | (b >> 4)) << 2;
}

void WriteDispData(unsigned char high, unsigned char low)
{
    uint8_t pixel[3];
    encode_color(((uint16_t)high << 8) | low, pixel);
    write_pixels(pixel, sizeof(pixel));
}

void BlockWrite(unsigned int x_start, unsigned int x_end,
                unsigned int y_start, unsigned int y_end)
{
    assert(x_start <= x_end && x_end < BOARD_MAIN_LCD_H_RES);
    assert(y_start <= y_end && y_end < BOARD_MAIN_LCD_V_RES);
    const uint8_t columns[] = {x_start >> 8, x_start & 0xff, x_end >> 8, x_end & 0xff};
    const uint8_t rows[] = {y_start >> 8, y_start & 0xff, y_end >> 8, y_end & 0xff};
    SPI_WriteComm(0x2a);
    write_pixels(columns, sizeof(columns));
    SPI_WriteComm(0x2b);
    write_pixels(rows, sizeof(rows));
    SPI_WriteComm(0x2c);
}

static void fill_rows(unsigned int first, unsigned int end, uint16_t color)
{
    for (unsigned int x = 0; x < BOARD_MAIN_LCD_H_RES; ++x) {
        encode_color(color, &line_buffer[x * 3]);
    }
    BlockWrite(0, BOARD_MAIN_LCD_H_RES - 1, first, end);
    for (unsigned int y = first; y <= end; ++y) {
        write_pixels(line_buffer, sizeof(line_buffer));
        if ((y & 15) == 15) vTaskDelay(1);
    }
}

void DispColor(unsigned int color)
{
    fill_rows(0, BOARD_MAIN_LCD_V_RES - 1, color);
}

void DispBand(void)
{
    const uint16_t colors[] = {0xffff, 0xffe0, 0x07ff, 0x07e0,
                               0xf81f, 0xf800, 0x001f, 0x0000};
    for (unsigned int i = 0; i < 8; ++i) {
        fill_rows(i * BOARD_MAIN_LCD_V_RES / 8,
                  (i + 1) * BOARD_MAIN_LCD_V_RES / 8 - 1, colors[i]);
    }
}

void LCD_SetPixelClock(unsigned int hz)
{
    ESP_ERROR_CHECK(spi_bus_remove_device(lcd_spi));
    const spi_device_interface_config_t device = {
        .clock_speed_hz = hz, .mode = 0, .spics_io_num = -1, .queue_size = 1,
    };
    ESP_ERROR_CHECK(spi_bus_add_device(BOARD_MAIN_LCD_SPI_HOST, &device, &lcd_spi));
    ESP_LOGI("lcd", "Animation SPI pixel clock: %u Hz", hz);
}

void LCD_DrawBGR888(unsigned int x, unsigned int y, unsigned int width,
                   unsigned int height, const uint8_t *pixels, size_t stride)
{
    assert(pixels && width && height && height <= 8);
    assert(x + width <= BOARD_MAIN_LCD_H_RES && y + height <= BOARD_MAIN_LCD_V_RES);
    assert(stride >= width * 3);
    for (unsigned int row = 0; row < height; ++row) {
        const uint8_t *src = pixels + row * stride;
        uint8_t *dst = pixel_buffer + row * width * 3;
        for (unsigned int col = 0; col < width; ++col) {
            dst[col * 3] = src[col * 3 + 2] & 0xfc;
            dst[col * 3 + 1] = src[col * 3 + 1] & 0xfc;
            dst[col * 3 + 2] = src[col * 3] & 0xfc;
        }
    }
    ESP_ERROR_CHECK(spi_device_acquire_bus(lcd_spi, portMAX_DELAY));
    BlockWrite(x, x + width - 1, y, y + height - 1);
    write_pixels(pixel_buffer, width * height * 3);
    spi_device_release_bus(lcd_spi);
}

void LCD_Init(void)
{
    const gpio_config_t gpio = {
        .pin_bit_mask = (1ULL << BOARD_MAIN_LCD_DC_GPIO) |
                        (1ULL << BOARD_MAIN_LCD_CS_GPIO) |
                        (1ULL << BOARD_LCD_BACKLIGHT_GPIO),
        .mode = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&gpio));
    ESP_ERROR_CHECK(gpio_set_direction(BOARD_LCD_BACKLIGHT_GPIO, GPIO_MODE_INPUT_OUTPUT));
    ESP_ERROR_CHECK(gpio_set_level(BOARD_MAIN_LCD_CS_GPIO, 1));
    ESP_ERROR_CHECK(gpio_set_level(BOARD_MAIN_LCD_DC_GPIO, 1));
    ESP_ERROR_CHECK(gpio_set_level(BOARD_LCD_BACKLIGHT_GPIO, BOARD_LCD_BACKLIGHT_ON_LEVEL));
    ESP_LOGI("lcd", "Backlight GPIO48 pad readback=%d (expected 1)",
             gpio_get_level(BOARD_LCD_BACKLIGHT_GPIO));
    const spi_bus_config_t bus = {
        .mosi_io_num = BOARD_MAIN_LCD_MOSI_GPIO,
        .miso_io_num = -1,
        .sclk_io_num = BOARD_MAIN_LCD_SCLK_GPIO,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = sizeof(pixel_buffer),
    };
    ESP_ERROR_CHECK(spi_bus_initialize(BOARD_MAIN_LCD_SPI_HOST, &bus, SPI_DMA_CH_AUTO));
    const spi_device_interface_config_t device = {
        .clock_speed_hz = BOARD_LCD_PIXEL_CLOCK_HZ,
        .mode = 0,
        .spics_io_num = -1,
        .queue_size = 1,
    };
    ESP_ERROR_CHECK(spi_bus_add_device(BOARD_MAIN_LCD_SPI_HOST, &device, &lcd_spi));
    /* RST is physically tied to ESP EN. Never drive EN from firmware.
     * Allow power to settle, then software reset for app-only restarts. */
    vTaskDelay(pdMS_TO_TICKS(100));
    SPI_WriteComm(0x01);
    vTaskDelay(pdMS_TO_TICKS(800));
    ESP_LOGI("lcd", "FPC1 MOSI=47 SCLK=21 DC=20 CS=19 BL=48; SPI 1MHz RGB666");
    // BEGIN VENDOR GC9B72 INIT
    SPI_WriteComm(0xFE);
    SPI_WriteComm(0xEF);
    SPI_WriteComm(0x80);
    SPI_WriteData(0x19);
    SPI_WriteComm(0x81);
    SPI_WriteData(0x30);
    SPI_WriteComm(0x82);
    SPI_WriteData(0x09);
    SPI_WriteComm(0x83);
    SPI_WriteData(0x03);
    SPI_WriteComm(0x84);
    SPI_WriteData(0x20);
    SPI_WriteComm(0x86);
    SPI_WriteData(0x18);
    SPI_WriteComm(0x87);
    SPI_WriteData(0x0A);
    SPI_WriteComm(0x89);
    SPI_WriteData(0x38);
    SPI_WriteComm(0x8A);
    SPI_WriteData(0x40);
    SPI_WriteComm(0x8B);
    SPI_WriteData(0x0A);
    SPI_WriteComm(0x8E);
    SPI_WriteData(0x0F);
    SPI_WriteComm(0x8F);
    SPI_WriteData(0x10);
    SPI_WriteComm(0x3A);
    SPI_WriteData(0x06);
    SPI_WriteComm(0x36);
    SPI_WriteData(0x00);
    SPI_WriteComm(0xEC);
    SPI_WriteData(0x07);
    SPI_WriteComm(0x98);
    SPI_WriteData(0x3E);
    SPI_WriteComm(0x99);
    SPI_WriteData(0x3E);
    SPI_WriteComm(0xA1);
    SPI_WriteData(0x01);
    SPI_WriteData(0x04);
    SPI_WriteComm(0xA2);
    SPI_WriteData(0x01);
    SPI_WriteData(0x04);
    SPI_WriteComm(0xCB);
    SPI_WriteData(0x02);
    SPI_WriteComm(0xB5);
    SPI_WriteData(0x15);
    SPI_WriteData(0x15);
    SPI_WriteComm(0xEB);
    SPI_WriteData(0x02);
    SPI_WriteData(0x87);
    SPI_WriteComm(0x60);
    SPI_WriteData(0x58);
    SPI_WriteData(0x22);
    SPI_WriteData(0x01);
    SPI_WriteData(0x58);
    SPI_WriteComm(0x63);
    SPI_WriteData(0x2d);
    SPI_WriteData(0x39);
    SPI_WriteData(0x01);
    SPI_WriteData(0x52);
    SPI_WriteComm(0x64);
    SPI_WriteData(0x38);
    SPI_WriteData(0x24);
    SPI_WriteData(0x75);
    SPI_WriteData(0x3c);
    SPI_WriteData(0x04);
    SPI_WriteData(0x58);
    SPI_WriteComm(0x65);
    SPI_WriteData(0x38);
    SPI_WriteData(0x28);
    SPI_WriteData(0x75);
    SPI_WriteData(0x40);
    SPI_WriteData(0x04);
    SPI_WriteData(0x58);
    SPI_WriteComm(0x66);
    SPI_WriteData(0x38);
    SPI_WriteData(0x24);
    SPI_WriteData(0x75);
    SPI_WriteData(0x3C);
    SPI_WriteData(0x04);
    SPI_WriteData(0x58);
    SPI_WriteComm(0x6A);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteComm(0x6C);
    SPI_WriteData(0xCC);
    SPI_WriteData(0x0C);
    SPI_WriteData(0xCC);
    SPI_WriteData(0x00);
    SPI_WriteData(0xCC);
    SPI_WriteData(0x04);
    SPI_WriteData(0x5F);
    SPI_WriteComm(0x6E);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x02);
    SPI_WriteData(0x0A);
    SPI_WriteData(0x0C);
    SPI_WriteData(0x0E);
    SPI_WriteData(0x10);
    SPI_WriteData(0x08);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x07);
    SPI_WriteData(0x0F);
    SPI_WriteData(0x0D);
    SPI_WriteData(0x0B);
    SPI_WriteData(0x09);
    SPI_WriteData(0x01);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteComm(0x74);
    SPI_WriteData(0x00);
    SPI_WriteData(0xe6);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteComm(0x7D);
    SPI_WriteData(0x72);
    SPI_WriteComm(0x7E);
    SPI_WriteData(0x1C);
    SPI_WriteComm(0x7C);
    SPI_WriteData(0xB6);
    SPI_WriteData(0x2b);
    SPI_WriteComm(0xAC);
    SPI_WriteData(0x30);
    SPI_WriteComm(0x70);
    SPI_WriteData(0x02);
    SPI_WriteData(0x03);
    SPI_WriteData(0x03);
    SPI_WriteData(0x06);
    SPI_WriteData(0x03);
    SPI_WriteData(0x03);
    SPI_WriteData(0x09);
    SPI_WriteData(0x07);
    SPI_WriteData(0x09);
    SPI_WriteData(0x03);
    SPI_WriteComm(0x90);
    SPI_WriteData(0x06);
    SPI_WriteData(0x06);
    SPI_WriteData(0x01);
    SPI_WriteData(0x01);
    SPI_WriteComm(0x93);
    SPI_WriteData(0x45);
    SPI_WriteData(0xFF);
    SPI_WriteData(0x00);
    SPI_WriteComm(0xBE);
    SPI_WriteData(0x11);
    SPI_WriteComm(0xC3);
    SPI_WriteData(0x39);
    SPI_WriteComm(0xC4);
    SPI_WriteData(0x39);
    SPI_WriteComm(0xC9);
    SPI_WriteData(0x2a);
    SPI_WriteComm(0xED);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteComm(0xF0);
    SPI_WriteData(0x02);
    SPI_WriteData(0x06);
    SPI_WriteData(0x07);
    SPI_WriteData(0x05);
    SPI_WriteData(0x04);
    SPI_WriteData(0x27);
    SPI_WriteComm(0xF1);
    SPI_WriteData(0x3F);
    SPI_WriteData(0x74);
    SPI_WriteData(0x74);
    SPI_WriteData(0x28);
    SPI_WriteData(0x32);
    SPI_WriteData(0x8F);
    SPI_WriteComm(0xF2);
    SPI_WriteData(0x03);
    SPI_WriteData(0x06);
    SPI_WriteData(0x07);
    SPI_WriteData(0x05);
    SPI_WriteData(0x04);
    SPI_WriteData(0x27);
    SPI_WriteComm(0xF3);
    SPI_WriteData(0x3F);
    SPI_WriteData(0x74);
    SPI_WriteData(0x74);
    SPI_WriteData(0x28);
    SPI_WriteData(0x32);
    SPI_WriteData(0x8F);
    SPI_WriteComm(0xF6);
    SPI_WriteData(0x80);
    SPI_WriteComm(0xF9);
    SPI_WriteData(0x70);
    SPI_WriteComm(0xFB);
    SPI_WriteData(0x70);
    SPI_WriteData(0x70);
    SPI_WriteComm(0xFD);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteComm(0xB4);
    SPI_WriteData(0x0A);
    SPI_WriteComm(0x35);
    SPI_WriteData(0x00);
    SPI_WriteComm(0xFE);
    SPI_WriteComm(0xEE);
    SPI_WriteComm(0x11);
    vTaskDelay(pdMS_TO_TICKS(20));
    SPI_WriteComm(0x29);
    vTaskDelay(pdMS_TO_TICKS(20));
    SPI_WriteComm(0x2C);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteComm(0x2C);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    SPI_WriteData(0x00);
    vTaskDelay(pdMS_TO_TICKS(20));
    // END VENDOR GC9B72 INIT
    ESP_LOGI("lcd", "Vendor GC9B72 initialization sent; backlight ON");
}
