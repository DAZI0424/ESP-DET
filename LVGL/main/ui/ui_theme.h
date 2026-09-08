#pragma once
#include "lvgl.h"

#define UI_WIDTH 648
#define UI_HEIGHT 200
#define UI_WEIGHT_X 224
#define UI_CALORIES_X 467
#define UI_METRIC_WIDTH 181

LV_FONT_DECLARE(ui_font_demibold_27);
LV_FONT_DECLARE(ui_font_medium_27);
LV_FONT_DECLARE(ui_font_normal_28);
LV_FONT_DECLARE(ui_font_normal_27);
LV_FONT_DECLARE(ui_font_light_85);
LV_FONT_DECLARE(ui_font_light_84);
LV_FONT_DECLARE(ui_font_demibold_28);
LV_FONT_DECLARE(ui_font_demibold_36);
LV_IMAGE_DECLARE(ui_asset_eyes);
LV_IMAGE_DECLARE(ui_asset_apple);

typedef enum { UI_ACCENT_WEIGHT, UI_ACCENT_CALORIES, UI_ACCENT_PROTEIN,
               UI_ACCENT_FAT, UI_ACCENT_CARBS, UI_ACCENT_SODIUM } ui_accent_t;
void ui_theme_gradient(lv_grad_dsc_t *gradient, ui_accent_t accent);
lv_obj_t *ui_container(lv_obj_t *parent, int x, int y, int width, int height);
lv_obj_t *ui_text(lv_obj_t *parent, const char *text, const lv_font_t *font,
                  int x, int y, int width);
