#include "ui_theme.h"
#include <string.h>

void ui_theme_gradient(lv_grad_dsc_t *g, ui_accent_t accent)
{
    static const uint32_t green[] = {0x82fc3b, 0x94fa84, 0xb4f6af};
    static const uint32_t gold[] = {0xff9730, 0xffca7f};
    static const uint32_t protein[] = {0x97b7f0, 0xb4ccff};
    static const uint32_t fat[] = {0xf0a0ae, 0xfdc4cc};
    static const uint32_t carbs[] = {0x9dfff6, 0xc3ffff};
    static const uint32_t sodium[] = {0xbc98e8, 0xccaaf7};
    const uint32_t *palettes[] = {green, gold, protein, fat, carbs, sodium};
    const uint32_t *colors = palettes[accent];
    memset(g, 0, sizeof(*g));
    /* The glyph canvas applies the screenshot's oblique axis. */
    g->dir = LV_GRAD_DIR_HOR;
    g->stops_count = accent == UI_ACCENT_WEIGHT ? 3 : 2;
    for (unsigned i = 0; i < g->stops_count; ++i) {
        g->stops[i].color = lv_color_hex(colors[i]);
        g->stops[i].opa = LV_OPA_COVER;
        g->stops[i].frac = accent == UI_ACCENT_WEIGHT ? (i == 0 ? 0 : i == 1 ? 99 : 255) : (i == 0 ? 0 : 255);
    }
}

lv_obj_t *ui_container(lv_obj_t *parent, int x, int y, int width, int height)
{
    lv_obj_t *obj = lv_obj_create(parent);
    lv_obj_remove_style_all(obj);
    lv_obj_set_pos(obj, x, y);
    lv_obj_set_size(obj, width, height);
    lv_obj_remove_flag(obj, LV_OBJ_FLAG_CLICKABLE | LV_OBJ_FLAG_SCROLLABLE);
    return obj;
}

lv_obj_t *ui_text(lv_obj_t *parent, const char *text, const lv_font_t *font,
                  int x, int y, int width)
{
    lv_obj_t *label = lv_label_create(parent);
    lv_obj_remove_style_all(label);
    lv_obj_set_style_text_font(label, font, 0);
    lv_obj_set_style_text_color(label, lv_color_hex(0xfafafa), 0);
    lv_obj_set_style_text_letter_space(label, 0, 0);
    lv_label_set_long_mode(label, LV_LABEL_LONG_MODE_CLIP);
    lv_label_set_text(label, text);
    lv_obj_set_pos(label, x, y);
    if (width > 0) lv_obj_set_width(label, width);
    lv_obj_remove_flag(label, LV_OBJ_FLAG_CLICKABLE | LV_OBJ_FLAG_SCROLLABLE);
    return label;
}
