/******************************************************************************
 * @file    ui_loading_animation.c
 * @author  lichunjiang
 * @version V1.0.0
 * @date    2026-09-08
 * @brief   圆点加载动画定时器回调实现
 ******************************************************************************/
#include "ui_loading_animation.h"

#include "../ui_loading.h"
#include "../ui_loading_motion.h"

/**
 * @brief  根据经过时间更新圆点加载动画
 * @param  timer LVGL 加载动画定时器
 * @retval 无
 * @note   贝塞尔曲线只映射关键帧时间，不对单个圆点做额外插值。
 */
void ui_loading_animation_update(lv_timer_t *timer)
{
    ui_loading_t *loading = lv_timer_get_user_data(timer);
    const uint32_t elapsed = lv_tick_elaps(loading->started) % UI_LOADING_PERIOD_MS;
    /* CSS/Figma cubic-bezier(0.5, 0, 0, 1): solve x(t), then evaluate y(t).
     * This controls the sequence timeline, not a second per-dot animation. */
    const int32_t progress = lv_cubic_bezier(
        elapsed * LV_BEZIER_VAL_MAX / UI_LOADING_PERIOD_MS,
        LV_BEZIER_VAL_FLOAT(0.5), 0, 0, LV_BEZIER_VAL_MAX);
    unsigned frame = (uint32_t)progress * 30U / LV_BEZIER_VAL_MAX;
    if (frame > 29) frame = 29;
    if (frame == loading->frame) return;
    loading->frame = frame;
    for (unsigned i = 0; i < 8; ++i) {
        const ui_loading_dot_key_t *key = &loading_keys[frame][i];
        if (!key->diameter) {
            lv_obj_add_flag(loading->dots[i], LV_OBJ_FLAG_HIDDEN);
            continue;
        }
        lv_obj_remove_flag(loading->dots[i], LV_OBJ_FLAG_HIDDEN);
        lv_obj_set_pos(loading->dots[i], key->x, key->y);
        lv_obj_set_size(loading->dots[i], key->diameter, key->diameter);
        lv_obj_set_style_bg_opa(loading->dots[i], key->opacity, 0);
    }
}
