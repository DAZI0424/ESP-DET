#pragma once

#include "driver/gpio.h"
#include "driver/spi_master.h"

/*
 * JKC319I002 (FPC1) parameters come from the supplied module specification:
 * GC9B72NA, 200(RGB) x 648, four-wire SPI, 18-bit pixel interface.
 */
#define BOARD_LCD_PIXEL_CLOCK_HZ       (10 * 1000 * 1000)
#define BOARD_LCD_CMD_BITS             8
#define BOARD_LCD_PARAM_BITS           8
#define BOARD_LCD_BITS_PER_PIXEL       18
#define BOARD_LCD_BYTES_PER_PIXEL      3
#define BOARD_LCD_DRAW_BUFFER_LINES    40
#define BOARD_LCD_STARTUP_TEST_MS      3000

/* Both backlight cathodes are switched by Q2; GPIO48 high turns them on. */
#define BOARD_LCD_BACKLIGHT_GPIO       GPIO_NUM_48
#define BOARD_LCD_BACKLIGHT_ON_LEVEL   1
#define BOARD_LCD_BACKLIGHT_OFF_LEVEL  0

/* Connected rectangular LCD: J2/FPC1 nets MSDA, MSCL, MD/C and MCS. */
#define BOARD_MAIN_LCD_SPI_HOST        SPI2_HOST
#define BOARD_MAIN_LCD_MOSI_GPIO       GPIO_NUM_47
#define BOARD_MAIN_LCD_SCLK_GPIO       GPIO_NUM_21
#define BOARD_MAIN_LCD_DC_GPIO         GPIO_NUM_20
#define BOARD_MAIN_LCD_CS_GPIO         GPIO_NUM_19
#define BOARD_MAIN_LCD_H_RES           200
#define BOARD_MAIN_LCD_V_RES           648
#define BOARD_MAIN_LCD_X_GAP           0
#define BOARD_MAIN_LCD_Y_GAP           0

/*
 * LCD RST is tied to the module EN/reset net on the adapter, not to a separate
 * ESP32-S3 GPIO. The driver therefore also sends a software reset.
 */
#define BOARD_LCD_RESET_GPIO           GPIO_NUM_NC
