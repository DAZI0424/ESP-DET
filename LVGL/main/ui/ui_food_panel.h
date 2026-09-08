#pragma once
#include "ui_theme.h"
#include "ui_loading.h"

typedef enum { UI_FOOD_EYES, UI_FOOD_APPLE_PLACEHOLDER } ui_food_picture_t;
typedef struct { lv_obj_t *root; lv_obj_t *picture; lv_obj_t *name; ui_loading_t loading; } ui_food_panel_t;
void ui_food_panel_create(ui_food_panel_t *panel, lv_obj_t *parent, ui_food_picture_t picture);
