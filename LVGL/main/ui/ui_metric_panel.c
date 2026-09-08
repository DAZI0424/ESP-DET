#include "ui_metric_panel.h"
#include <string.h>

void ui_metric_panel_create(ui_metric_panel_t *p, lv_obj_t *parent,
                            int x, const ui_metric_content_t *c)
{
    memset(p, 0, sizeof(*p));
    /* The last panel reaches x=648; text geometry matches the reference. */
    p->root = ui_container(parent, x, 0, c->accent == UI_ACCENT_WEIGHT ? 196 : 181, UI_HEIGHT);
    ui_gradient_text_create(&p->total_title, p->root, "总计", &ui_font_demibold_27,
                              c->accent, 0, 3, 56);
    p->total = ui_text(p->root, c->total, &ui_font_normal_28,
                       c->accent == UI_ACCENT_WEIGHT ? 160 : 155, 7, 22);
    p->rule_gradient.dir = LV_GRAD_DIR_HOR;
    p->rule_gradient.stops_count = 3;
    p->rule_gradient.stops[0] = (lv_gradient_stop_t){.color = lv_color_hex(0x444444), .opa = 255, .frac = 0};
    p->rule_gradient.stops[1] = (lv_gradient_stop_t){.color = lv_color_hex(0x646464), .opa = 255, .frac = 128};
    p->rule_gradient.stops[2] = (lv_gradient_stop_t){.color = lv_color_hex(0x333333), .opa = 255, .frac = 255};
    lv_obj_t *rule = ui_container(p->root, 0, 44, 183, 1);
    lv_obj_set_style_bg_opa(rule, LV_OPA_COVER, 0);
    lv_obj_set_style_bg_grad(rule, &p->rule_gradient, 0);
    const lv_font_t *title_font = c->accent == UI_ACCENT_WEIGHT ?
        &ui_font_medium_27 : &ui_font_demibold_27;
    ui_gradient_text_create(&p->title, p->root, c->title, title_font,
                              c->accent, 0, 61, 56);
    p->unit = ui_text(p->root, c->unit, &ui_font_normal_27,
                      c->accent == UI_ACCENT_WEIGHT ? 162 : 122, 64, 59);
    p->value = ui_text(p->root, c->value, &ui_font_light_85,
                       c->accent == UI_ACCENT_WEIGHT ? 139 : 128, 125, 57);
}

void ui_metric_panel_release(ui_metric_panel_t *p)
{
    ui_gradient_text_release(&p->total_title);
    ui_gradient_text_release(&p->title);
}
