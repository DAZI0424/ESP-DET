#include "ui_demo.h"

#include "esp_lvgl_port.h"

static void set_bar_value(void *object, int32_t value)
{
    lv_bar_set_value((lv_obj_t *)object, value, LV_ANIM_OFF);
}

static void create_screen(lv_display_t *display, const char *title, uint32_t color,
                          uint32_t animation_time_ms)
{
    lv_obj_t *screen = lv_display_get_screen_active(display);
    lv_obj_set_style_bg_color(screen, lv_color_hex(0x10131a), 0);

    lv_obj_t *label = lv_label_create(screen);
    lv_label_set_long_mode(label, LV_LABEL_LONG_SCROLL_CIRCULAR);
    lv_label_set_text(label, title);
    lv_obj_set_width(label, lv_pct(84));
    lv_obj_set_style_text_color(label, lv_color_hex(0xffffff), 0);
    lv_obj_set_style_text_font(label, &lv_font_montserrat_14, 0);
    lv_obj_set_style_text_align(label, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(label, LV_ALIGN_TOP_MID, 0, 24);

    lv_obj_t *bar = lv_bar_create(screen);
    lv_obj_set_size(bar, lv_pct(72), 18);
    lv_obj_align(bar, LV_ALIGN_CENTER, 0, 15);
    lv_obj_set_style_bg_color(bar, lv_color_hex(0x303744), LV_PART_MAIN);
    lv_obj_set_style_bg_color(bar, lv_color_hex(color), LV_PART_INDICATOR);
    lv_obj_set_style_radius(bar, LV_RADIUS_CIRCLE, LV_PART_MAIN);
    lv_obj_set_style_radius(bar, LV_RADIUS_CIRCLE, LV_PART_INDICATOR);
    lv_bar_set_range(bar, 0, 100);

    lv_anim_t animation;
    lv_anim_init(&animation);
    lv_anim_set_var(&animation, bar);
    lv_anim_set_exec_cb(&animation, set_bar_value);
    lv_anim_set_values(&animation, 0, 100);
    lv_anim_set_duration(&animation, animation_time_ms);
    lv_anim_set_playback_duration(&animation, animation_time_ms);
    lv_anim_set_repeat_count(&animation, LV_ANIM_REPEAT_INFINITE);
    lv_anim_set_path_cb(&animation, lv_anim_path_ease_in_out);
    lv_anim_start(&animation);
}

void ui_demo_create(lv_display_t *display)
{
    if (!lvgl_port_lock(0)) {
        return;
    }

    lv_display_set_rotation(display, LV_DISPLAY_ROTATION_0);
    create_screen(display, "ESP32-S3  GC9B72  200 x 648  LVGL", 0x22c55e, 1200);
    lvgl_port_unlock();
}
