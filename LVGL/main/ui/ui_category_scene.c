#include "ui_category_scene.h"
#include "animation/ui_category_animation.h"
#include <assert.h>
#include <string.h>

typedef struct {
    const char *title, *total, *value, *unit;
    ui_accent_t accent;
} category_t;

static const category_t categories[] = {
    {"热量", "405", "321", "kcal", UI_ACCENT_CALORIES},
    {"蛋白质", "24.1", "3.2", "g", UI_ACCENT_PROTEIN},
    {"脂肪", "12.7", "1.4", "g", UI_ACCENT_FAT},
    {"碳水化合物", "89.2", "67.6", "g", UI_ACCENT_CARBS},
    {"钠", "23", "9", "mg", UI_ACCENT_SODIUM},
};

static void create_panel(ui_metric_panel_t *panel, lv_obj_t *parent, int x,
                         const category_t *content, bool weight)
{
    memset(panel, 0, sizeof(*panel));
    const int width = weight ? 196 : 181;
    panel->root = ui_container(parent, x, 0, width, UI_HEIGHT);
    ui_gradient_text_create(&panel->total_title, panel->root, "总计", &ui_font_demibold_27,
                              content->accent, 0, 3, 56);
    panel->total = ui_text(panel->root, content->total, &ui_font_normal_28, 0, 7, 177);
    lv_obj_set_style_text_align(panel->total, LV_TEXT_ALIGN_RIGHT, 0);
    lv_obj_t *rule = ui_container(panel->root, 0, 44, 181, 1);
    lv_obj_set_style_bg_color(rule, lv_color_hex(0x646464), 0);
    lv_obj_set_style_bg_opa(rule, LV_OPA_COVER, 0);
    int title_width = content->accent == UI_ACCENT_CARBS ? 136 : content->accent == UI_ACCENT_PROTEIN ? 83 : 56;
    ui_gradient_text_create(&panel->title, panel->root, content->title,
                              weight ? &ui_font_medium_27 : &ui_font_demibold_27,
                              content->accent, 0, 61, title_width);
    const int unit_width = weight ? 64 : content->accent == UI_ACCENT_CALORIES ? 56 : 58;
    panel->unit = ui_text(panel->root, content->unit, &ui_font_normal_27, 115, 64, unit_width);
    lv_obj_set_style_text_align(panel->unit, LV_TEXT_ALIGN_RIGHT, 0);
    panel->value = ui_text(panel->root, content->value, &ui_font_light_84, 0, 125, weight ? 191 : 181);
    lv_obj_set_style_text_align(panel->value, LV_TEXT_ALIGN_RIGHT, 0);
    lv_obj_set_style_transform_pivot_x(panel->root, 90, 0);
    lv_obj_set_style_transform_pivot_y(panel->root, 80, 0);
}

void ui_category_scene_restart(ui_category_scene_t *scene)
{
    scene->started = lv_tick_get();
    scene->frame = 307;
    scene->scale = 257;
    scene->finished = false;
    scene->applied_frames = 0;
    lv_timer_resume(scene->timer);
    ui_category_animation_update(scene->timer);
}

void ui_category_scene_create(ui_category_scene_t *scene, lv_display_t *display)
{
    assert(scene && display);
    memset(scene, 0, sizeof(*scene));
    lv_obj_t *screen = lv_display_get_screen_active(display);
    lv_obj_remove_style_all(screen);
    lv_obj_set_style_bg_color(screen, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(screen, LV_OPA_COVER, 0);
    lv_obj_remove_flag(screen, LV_OBJ_FLAG_SCROLLABLE);
    scene->root = ui_container(screen, 0, 0, UI_WIDTH, UI_HEIGHT);
    ui_food_panel_create(&scene->food, scene->root, UI_FOOD_APPLE_PLACEHOLDER);
    lv_obj_set_style_text_font(scene->food.name, &ui_font_demibold_36, 0);
    lv_obj_set_pos(scene->food.name, 32, 159);
    const category_t weight = {"重量", "194", "541", "g", UI_ACCENT_WEIGHT};
    create_panel(&scene->weight, scene->root, 224, &weight, true);
    for (unsigned i = 0; i < 5; ++i) {
        create_panel(&scene->panels[i], scene->root, 467, &categories[i], false);
        lv_obj_add_flag(scene->panels[i].root, LV_OBJ_FLAG_HIDDEN);
    }
    lv_obj_t *divider = ui_container(scene->root, 434, 2, 3, 195);
    lv_obj_set_style_radius(divider, LV_RADIUS_CIRCLE, 0);
    lv_obj_set_style_bg_color(divider, lv_color_white(), 0);
    lv_obj_set_style_bg_opa(divider, LV_OPA_COVER, 0);
    lv_obj_report_style_change(NULL);
    scene->timer = lv_timer_create(ui_category_animation_update, 8, scene);
    ui_category_scene_restart(scene);
}

void ui_category_scene_destroy(ui_category_scene_t *scene)
{
    if (!scene || !scene->root) return;
    lv_timer_delete(scene->timer);
    lv_obj_delete(scene->root);
    ui_metric_panel_release(&scene->weight);
    for (unsigned i = 0; i < 5; ++i) ui_metric_panel_release(&scene->panels[i]);
    memset(scene, 0, sizeof(*scene));
}
