#include "ui_loading.h"
#include "animation/ui_loading_animation.h"
#include <string.h>

void ui_loading_create(ui_loading_t *loading, lv_obj_t *parent)
{
    memset(loading, 0, sizeof(*loading));
    loading->root = ui_container(parent, 48, 165, 39, 35);
    for (unsigned i = 0; i < 8; ++i) {
        loading->dots[i] = ui_container(loading->root, 0, 0, 5, 5);
        lv_obj_set_style_radius(loading->dots[i], LV_RADIUS_CIRCLE, 0);
        lv_obj_set_style_bg_color(loading->dots[i], lv_color_white(), 0);
        lv_obj_add_flag(loading->dots[i], LV_OBJ_FLAG_HIDDEN);
    }
    loading->started = lv_tick_get();
    loading->frame = 255;
    loading->timer = lv_timer_create(ui_loading_animation_update, 10, loading);
    ui_loading_animation_update(loading->timer);
}

void ui_loading_stop(ui_loading_t *loading)
{
    if (loading->timer) lv_timer_delete(loading->timer);
    loading->timer = NULL;
}
