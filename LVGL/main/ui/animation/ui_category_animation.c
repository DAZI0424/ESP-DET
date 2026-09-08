/******************************************************************************
 * @file    ui_category_animation.c
 * @author  lichunjiang
 * @version V1.0.0
 * @date    2026-09-08
 * @brief   M013 品类切换动画定时器回调实现
 ******************************************************************************/
#include "ui_category_animation.h"

#include "../ui_category_scene.h"
#include "../ui_category_timing.h"

/**
 * @brief  根据绝对经过时间更新 M013 品类和缩放状态
 * @param  timer LVGL 品类切换动画定时器
 * @retval 无
 * @note   关键帧仅包含品类索引和 LVGL 缩放值，不包含图像像素。
 */
void ui_category_animation_update(lv_timer_t *timer)
{
    ui_category_scene_t *scene = lv_timer_get_user_data(timer);
    const uint32_t elapsed = lv_tick_elaps(scene->started);
    unsigned frame = elapsed * 60U / 1000U;
    if (frame > 306) frame = 306;
    if (frame != scene->frame) {
        const ui_category_key_t *key = &category_keys[frame];
        /* Static holds must not invalidate a full panel every timer tick. */
        if (key->category != scene->category || key->scale != scene->scale) {
            for (unsigned i = 0; i < 5; ++i) {
                lv_obj_t *panel = scene->panels[i].root;
                if (i != key->category || !key->scale) {
                    lv_obj_add_flag(panel, LV_OBJ_FLAG_HIDDEN);
                } else {
                    lv_obj_set_style_transform_scale(panel, key->scale, 0);
                    lv_obj_remove_flag(panel, LV_OBJ_FLAG_HIDDEN);
                }
            }
            scene->scale = key->scale;
        }
        scene->frame = frame;
        scene->category = key->category;
        scene->applied_frames++;
    }
    if (elapsed >= UI_CATEGORY_DURATION_MS) {
        scene->finished = true;
        lv_timer_pause(timer);
    }
}
