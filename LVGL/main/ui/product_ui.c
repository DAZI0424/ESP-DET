#include "product_ui.h"
#include <assert.h>
#include <string.h>

const product_ui_content_t PRODUCT_UI_FIRST_FRAME = {"0", "0", "0", "0", UI_FOOD_EYES};

void product_ui_create(product_ui_t *ui, lv_display_t *display, const product_ui_content_t *content)
{
    assert(ui && display && content);
    assert(lv_display_get_horizontal_resolution(display) == UI_WIDTH);
    assert(lv_display_get_vertical_resolution(display) == UI_HEIGHT);
    memset(ui, 0, sizeof(*ui));
    lv_obj_t *screen = lv_display_get_screen_active(display);
    lv_obj_remove_style_all(screen);
    lv_obj_set_style_bg_color(screen, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(screen, LV_OPA_COVER, 0);
    lv_obj_remove_flag(screen, LV_OBJ_FLAG_SCROLLABLE | LV_OBJ_FLAG_CLICKABLE);
    ui->root = ui_container(screen, 0, 0, UI_WIDTH, UI_HEIGHT);
    ui_food_panel_create(&ui->food, ui->root, content->picture);
    const ui_metric_content_t weight = {content->total_weight, content->weight, "重量", "g", UI_ACCENT_WEIGHT};
    const ui_metric_content_t calories = {content->total_calories, content->calories, "热量", "kcal", UI_ACCENT_CALORIES};
    ui_metric_panel_create(&ui->weight, ui->root, UI_WEIGHT_X, &weight);
    ui_metric_panel_create(&ui->calories, ui->root, UI_CALORIES_X, &calories);
    ui->divider_gradient.dir = LV_GRAD_DIR_HOR;
    ui->divider_gradient.stops_count = 3;
    ui->divider_gradient.stops[0] = (lv_gradient_stop_t){.color = lv_color_hex(0x777777), .opa = 255, .frac = 0};
    ui->divider_gradient.stops[1] = (lv_gradient_stop_t){.color = lv_color_white(), .opa = 255, .frac = 128};
    ui->divider_gradient.stops[2] = (lv_gradient_stop_t){.color = lv_color_hex(0xb0b0b0), .opa = 255, .frac = 255};
    lv_obj_t *divider = ui_container(ui->root, 434, 2, 3, 195);
    lv_obj_set_style_bg_opa(divider, LV_OPA_COVER, 0);
    lv_obj_set_style_bg_grad(divider, &ui->divider_gradient, 0);
    /* Rebuilding the tree reuses local style storage; refresh inherited caches. */
    lv_obj_report_style_change(NULL);
}

void product_ui_destroy(product_ui_t *ui)
{
    if (!ui || !ui->root) return;
    ui_loading_stop(&ui->food.loading);
    lv_obj_delete(ui->root);
    ui_metric_panel_release(&ui->weight);
    ui_metric_panel_release(&ui->calories);
    memset(ui, 0, sizeof(*ui));
}
