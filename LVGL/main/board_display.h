#pragma once

#include "esp_err.h"
#include "lvgl.h"

typedef struct {
    lv_display_t *main_display;
} board_displays_t;

esp_err_t board_display_init(board_displays_t *displays);
void board_display_set_backlight(bool on);
