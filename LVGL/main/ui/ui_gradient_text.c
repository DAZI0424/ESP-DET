#include "ui_gradient_text.h"
#include <assert.h>
#include <string.h>

/* LVGL rasterizes the actual MiSans glyphs at runtime. Only their RGB is
 * colored here; the native antialiasing alpha is preserved. No text asset. */
void ui_gradient_text_create(ui_gradient_text_t *text, lv_obj_t *parent,
                              const char *value, const lv_font_t *font,
                              ui_accent_t accent, int x, int y, int width)
{
    memset(text, 0, sizeof(*text));
    const int height = lv_font_get_line_height(font);
    text->buffer = lv_draw_buf_create(width, height, LV_COLOR_FORMAT_ARGB8888, LV_STRIDE_AUTO);
    assert(text->buffer);
    text->object = lv_canvas_create(parent);
    lv_obj_remove_style_all(text->object);
    lv_obj_remove_flag(text->object, LV_OBJ_FLAG_CLICKABLE | LV_OBJ_FLAG_SCROLLABLE);
    lv_canvas_set_draw_buf(text->object, text->buffer);
    lv_obj_set_pos(text->object, x, y);
    lv_canvas_fill_bg(text->object, lv_color_white(), LV_OPA_TRANSP);
    lv_layer_t layer;
    lv_canvas_init_layer(text->object, &layer);
    lv_draw_label_dsc_t glyphs;
    lv_draw_label_dsc_init(&glyphs);
    glyphs.font = font;
    glyphs.text = value;
    glyphs.color = lv_color_white();
    const lv_area_t bounds = {0, 0, width - 1, height - 1};
    lv_draw_label(&layer, &glyphs, &bounds);
    lv_canvas_finish_layer(text->object, &layer);
    ui_theme_gradient(&text->gradient, accent);
    for (int row = 0; row < height; ++row) {
        uint8_t *pixels = text->buffer->data + row * text->buffer->header.stride;
        for (int col = 0; col < width; ++col) {
            /* Screenshot handle endpoints mapped to a 56x27 title box:
             * green (-4,2)->(82,22), orange (-4,1)->(70,22).
             * Both end beyond the glyphs, as shown in Figma. */
            const bool green = accent == UI_ACCENT_WEIGHT;
            const float dx = green ? 86.0f : 74.0f;
            const float dy = green ? 20.0f : 21.0f;
            const float start_y = green ? 2.0f : 1.0f;
            const float px = col * 56.0f / width;
            const float py = row * 27.0f / height;
            float position = ((px + 4.0f) * dx + (py - start_y) * dy) / (dx*dx + dy*dy);
            if (position < 0) position = 0;
            if (position > 1) position = 1;
            /* Preserve the screenshot's exact 39% stop, not a rounded 8-bit position. */
            const unsigned index = green && position >= 0.39f ? 1 : 0;
            const float begin = index == 1 ? 0.39f : 0.0f;
            const float end = green && index == 0 ? 0.39f : 1.0f;
            const lv_gradient_stop_t *a = &text->gradient.stops[index];
            const lv_gradient_stop_t *b = &text->gradient.stops[index+1];
            const float mix = (position - begin) / (end - begin);
            pixels[col*4+0] = (uint8_t)(a->color.blue + (b->color.blue - a->color.blue) * mix + 0.5f);
            pixels[col*4+1] = (uint8_t)(a->color.green + (b->color.green - a->color.green) * mix + 0.5f);
            pixels[col*4+2] = (uint8_t)(a->color.red + (b->color.red - a->color.red) * mix + 0.5f);
        }
    }
    lv_obj_invalidate(text->object);
}

void ui_gradient_text_release(ui_gradient_text_t *text)
{
    if (text->buffer) lv_draw_buf_destroy(text->buffer);
    memset(text, 0, sizeof(*text));
}
