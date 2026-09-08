#pragma once
#include "ui_gradient_text.h"

typedef struct {
    const char *total;
    const char *value;
    const char *title;
    const char *unit;
    ui_accent_t accent;
} ui_metric_content_t;

typedef struct {
    lv_obj_t *root;
    lv_obj_t *total;
    lv_obj_t *value;
    lv_obj_t *unit;
    ui_gradient_text_t total_title;
    ui_gradient_text_t title;
    lv_grad_dsc_t rule_gradient;
} ui_metric_panel_t;

void ui_metric_panel_create(ui_metric_panel_t *panel, lv_obj_t *parent,
                            int x, const ui_metric_content_t *content);
void ui_metric_panel_release(ui_metric_panel_t *panel);
