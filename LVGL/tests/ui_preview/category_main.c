#include "ui_category_scene.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void flush(lv_display_t *display, const lv_area_t *area, uint8_t *pixels)
{ (void)area; (void)pixels; lv_display_flush_ready(display); }
static void save(const char *directory, unsigned frame, const uint8_t *pixels)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%03u.ppm", directory, frame);
    FILE *f = fopen(path,"wb"); assert(f);
    fprintf(f,"P6\n648 200\n255\n");
    for (int i=0;i<UI_WIDTH*UI_HEIGHT;++i) {
        fputc(pixels[i*3+2],f); fputc(pixels[i*3+1],f); fputc(pixels[i*3],f);
    }
    fclose(f);
}
int main(int argc, char **argv)
{
    assert(argc==2);
    lv_init();
    void *pool = malloc(512 * 1024);
    assert(pool);
    lv_mem_pool_t added_pool = lv_mem_add_pool(pool,512 * 1024);
    assert(added_pool);
    size_t bytes = UI_WIDTH*UI_HEIGHT*3;
    uint8_t *pixels = calloc(1,bytes), *reference = malloc(bytes);
    assert(pixels && reference);
    lv_display_t *display = lv_display_create(UI_WIDTH, UI_HEIGHT);
    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB888);
    lv_display_set_buffers_with_stride(display,pixels,NULL,bytes,UI_WIDTH*3,LV_DISPLAY_RENDER_MODE_DIRECT);
    lv_display_set_flush_cb(display,flush);
    ui_category_scene_t scene;
    ui_category_scene_create(&scene,display);
    lv_refr_now(display);
    memcpy(reference,pixels,bytes);
    uint32_t previous=0;
    unsigned states=0;
    for (unsigned frame=0;frame<307;++frame) {
        uint32_t ms=(frame*1000+59)/60;
        lv_tick_inc(ms-previous); previous=ms;
        lv_timer_handler(); lv_refr_now(display);
        assert(scene.frame==frame);
        states |= 1U<<scene.category;
        for (int y=0;y<UI_HEIGHT;++y)
            assert(memcmp(reference+y*UI_WIDTH*3,pixels+y*UI_WIDTH*3,440*3)==0);
        save(argv[1],frame,pixels);
    }
    assert(states==31);
    lv_tick_inc(UI_CATEGORY_DURATION_MS-previous);
    lv_timer_handler(); lv_refr_now(display);
    assert(scene.finished && scene.category==4);
    memcpy(reference,pixels,bytes);
    lv_tick_inc(2000); lv_timer_handler(); lv_refr_now(display);
    assert(memcmp(reference,pixels,bytes)==0);
    ui_category_scene_restart(&scene); lv_refr_now(display);
    assert(scene.category==0 && !scene.finished);
    ui_category_scene_destroy(&scene);
    lv_tick_inc(100); lv_timer_handler();
    lv_display_delete(display); free(reference); free(pixels); lv_deinit(); free(pool);
    puts("PASS: 307 native frames; all five categories; left area invariant; final hold; restart; cleanup.");
}
