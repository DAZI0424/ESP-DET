#pragma once

#include "esp_err.h"
#include "esp_lcd_panel_io.h"

/* Send the GC9B72 controller power-on sequence over an existing panel IO. */
esp_err_t gc9b72_panel_init(esp_lcd_panel_io_handle_t io);
