/******************************************************************************
 * @file    ui_category_animation.h
 * @author  lichunjiang
 * @version V1.0.0
 * @date    2026-09-08
 * @brief   M013 品类切换动画定时器回调接口
 ******************************************************************************/
#pragma once

#include "lvgl.h"

/**
 * @brief  更新 M013 品类切换动画的当前关键帧
 * @param  timer LVGL 品类切换动画定时器
 * @retval 无
 */
void ui_category_animation_update(lv_timer_t *timer);
