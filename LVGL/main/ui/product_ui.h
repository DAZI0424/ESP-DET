#pragma once
#include "ui_metric_panel.h"
#include "ui_food_panel.h"

/* Initial content; only the eyes' loading indicator animates. No model binding. */
typedef struct {
    const char *total_weight;
    const char *weight;
    const char *total_calories;
    const char *calories;
    ui_food_picture_t picture;
} product_ui_content_t;

typedef struct {
    lv_obj_t *root;
    ui_food_panel_t food;
    ui_metric_panel_t weight;
    ui_metric_panel_t calories;
    lv_grad_dsc_t divider_gradient;
} product_ui_t;

extern const product_ui_content_t PRODUCT_UI_FIRST_FRAME;
void product_ui_create(product_ui_t *ui, lv_display_t *display, const product_ui_content_t *content);
void product_ui_destroy(product_ui_t *ui);
