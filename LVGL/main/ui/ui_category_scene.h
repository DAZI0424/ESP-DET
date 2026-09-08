#pragma once
#include "ui_food_panel.h"
#include "ui_metric_panel.h"

#define UI_CATEGORY_DURATION_MS 5117U
typedef struct {
    lv_obj_t *root;
    ui_food_panel_t food;
    ui_metric_panel_t weight;
    ui_metric_panel_t panels[5];
    lv_timer_t *timer;
    uint32_t started;
    unsigned frame, category, applied_frames;
    uint16_t scale;
    bool finished;
} ui_category_scene_t;

void ui_category_scene_create(ui_category_scene_t *scene, lv_display_t *display);
void ui_category_scene_restart(ui_category_scene_t *scene);
void ui_category_scene_destroy(ui_category_scene_t *scene);
