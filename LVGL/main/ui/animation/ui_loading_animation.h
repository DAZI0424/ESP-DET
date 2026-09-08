/******************************************************************************
 * @file    ui_loading_animation.h
 * @author  lichunjiang
 * @version V1.0.0
 * @date    2026-09-08
 * @brief   圆点加载动画定时器回调接口
 ******************************************************************************/
#pragma once

#include "lvgl.h"

/**
 * @brief  更新圆点加载动画的当前关键帧
 * @param  timer LVGL 加载动画定时器
 * @retval 无
 */
void ui_loading_animation_update(lv_timer_t *timer);
