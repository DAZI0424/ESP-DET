#pragma once
#include "ui_theme.h"

/* Runtime LVGL glyph canvas with projected gradient colors. No text assets. */
typedef struct {
    lv_obj_t *object;
    lv_draw_buf_t *buffer;
    lv_grad_dsc_t gradient;
} ui_gradient_text_t;

void ui_gradient_text_create(ui_gradient_text_t *text, lv_obj_t *parent,
                              const char *value, const lv_font_t *font,
                              ui_accent_t accent, int x, int y, int width);
/* Call only after the containing object tree has been deleted. */
void ui_gradient_text_release(ui_gradient_text_t *text);
