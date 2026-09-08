#include "product_ui.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
static void flush(lv_display_t *display, const lv_area_t *area, uint8_t *pixels)
{
    (void)area; (void)pixels;
    lv_display_flush_ready(display);
}
static void save(const char *path, const uint8_t *pixels)
{
    FILE *f = fopen(path, "wb"); assert(f);
    fprintf(f, "P6\n648 200\n255\n");
    for (int i = 0; i < UI_WIDTH * UI_HEIGHT; ++i) {
        fputc(pixels[i*3+2], f); fputc(pixels[i*3+1], f); fputc(pixels[i*3], f);
    }
    fclose(f);
}
int main(int argc, char **argv)
{
    assert(argc == 3);
    lv_init();
    const size_t bytes = UI_WIDTH * UI_HEIGHT * 3;
    uint8_t *pixels = calloc(1, bytes);
    uint8_t *reference = malloc(bytes);
    assert(pixels && reference);
    lv_display_t *display = lv_display_create(UI_WIDTH, UI_HEIGHT);
    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB888);
    lv_display_set_buffers_with_stride(display, pixels, NULL, bytes,
                                       UI_WIDTH * 3, LV_DISPLAY_RENDER_MODE_DIRECT);
    lv_display_set_flush_cb(display, flush);
    product_ui_t ui;
    for (int variant = 0; variant < 2; ++variant) {
        product_ui_content_t content = PRODUCT_UI_FIRST_FRAME;
        if (variant) content.picture = UI_FOOD_APPLE_PLACEHOLDER;
        product_ui_create(&ui, display, &content);
        lv_refr_now(display);
        memcpy(reference, pixels, bytes);
        for (int repaint = 0; repaint < 3; ++repaint) {
            lv_tick_inc(40);
            lv_obj_invalidate(lv_display_get_screen_active(display));
            lv_refr_now(display);
            assert(memcmp(reference, pixels, bytes) == 0);
        }
        save(argv[variant + 1], pixels);
        product_ui_destroy(&ui);
    }
    product_ui_create(&ui, display, &PRODUCT_UI_FIRST_FRAME);
    lv_refr_now(display);
    memcpy(reference, pixels, bytes);
    uint32_t previous_ms = 0;
    unsigned changed_frames = 0;
    for (unsigned frame = 0; frame < 30; ++frame) {
        const uint32_t elapsed_ms = (frame * UI_LOADING_PERIOD_MS + 29) / 30;
        lv_tick_inc(elapsed_ms - previous_ms);
        previous_ms = elapsed_ms;
        lv_timer_handler();
        lv_refr_now(display);
        bool changed = false;
        for (int y = 0; y < UI_HEIGHT; ++y) {
            for (int x = 0; x < UI_WIDTH; ++x) {
                const size_t offset = (y * UI_WIDTH + x) * 3;
                if (memcmp(reference + offset, pixels + offset, 3)) {
                    assert(x >= 48 && x < 87 && y >= 165 && y < 200);
                    changed = true;
                }
            }
        }
        changed_frames += changed;
        char path[1024];
        snprintf(path, sizeof(path), "%s.loading-%02u.ppm", argv[1], frame);
        save(path, pixels);
    }
    assert(changed_frames >= 18);
    lv_tick_inc(UI_LOADING_PERIOD_MS - previous_ms);
    lv_timer_handler();
    lv_refr_now(display);
    assert(memcmp(reference, pixels, bytes) == 0);
    /* Independently calculated Bezier milestones: y(.25)=.22457,
     * y(.5)=.85080, y(.75)=.97382; allow only keyframe quantization. */
    const uint8_t expected_phases[] = {6, 25, 29};
    for (unsigned i = 0; i < 3; ++i) {
        lv_tick_inc(UI_LOADING_PERIOD_MS / 4);
        lv_timer_handler();
        assert(ui.food.loading.frame == expected_phases[i]);
    }
    product_ui_destroy(&ui);
    lv_tick_inc(100);
    lv_timer_handler(); /* Deleted page must have no live loading callback. */
    lv_display_delete(display);
    free(reference);
    free(pixels);
    lv_deinit();
    puts("PASS: two static variants; three identical repaints each; same-display destroy/recreate.");
    puts("PASS: loading confined to 39x35; 30 time samples; 1500ms wrap; Bezier milestones; timer cleanup.");
    return 0;
}
