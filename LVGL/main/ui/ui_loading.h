#pragma once
#include "ui_theme.h"
#define UI_LOADING_PERIOD_MS 1500U

typedef struct {
    lv_obj_t *root;
    lv_obj_t *dots[8];
    lv_timer_t *timer;
    uint32_t started;
    uint8_t frame;
} ui_loading_t;

/* Repeats the reference's 1.5-second appearance / orbit / collapse sequence. */
void ui_loading_create(ui_loading_t *loading, lv_obj_t *parent);
void ui_loading_stop(ui_loading_t *loading);
