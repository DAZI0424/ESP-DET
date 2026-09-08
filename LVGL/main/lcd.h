#pragma once
#include <stddef.h>
#include <stdint.h>
/* ESP-IDF port of the supplied C8051 lcd/lcd.h bring-up API.
 * Single task only. Drawing colors are RGB565, sent as RGB666.
 */
void SPI_SendData(unsigned char value);
void SPI_WriteComm(unsigned char value);
void SPI_WriteData(unsigned char value);
void LCD_Init(void);
void WriteDispData(unsigned char high, unsigned char low);
void BlockWrite(unsigned int x_start, unsigned int x_end,
                unsigned int y_start, unsigned int y_end);
void DispColor(unsigned int color);
void DispBand(void);
void LCD_SetPixelClock(unsigned int hz);
/* Packed LVGL BGR888; width <= 200, height <= 8; output is RGB666. */
void LCD_DrawBGR888(unsigned int x, unsigned int y, unsigned int width,
                   unsigned int height, const uint8_t *pixels, size_t stride);
