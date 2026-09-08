#include "ui_food_panel.h"
#include <string.h>

void ui_food_panel_create(ui_food_panel_t *p, lv_obj_t *parent, ui_food_picture_t picture)
{
    memset(p, 0, sizeof(*p));
    p->root = ui_container(parent, 0, 0, 200, 200);
    p->picture = lv_image_create(p->root);
    if (picture == UI_FOOD_EYES) {
        lv_image_set_src(p->picture, &ui_asset_eyes);
        lv_obj_set_pos(p->picture, 0, 38);
        ui_loading_create(&p->loading, p->root);
        return;
    }
    lv_image_set_src(p->picture, &ui_asset_apple);
    lv_obj_set_pos(p->picture, 20, 19);
    static const lv_point_precise_t points[4][5] = {
        {{1,18},{1,5},{2,2},{5,1},{18,1}},
        {{113,1},{126,1},{129,2},{130,5},{130,18}},
        {{1,113},{1,126},{2,129},{5,130},{18,130}},
        {{113,130},{126,130},{129,129},{130,126},{130,113}},
    };
    for (unsigned i = 0; i < 4; ++i) {
        lv_obj_t *corner = lv_line_create(p->root);
        lv_obj_remove_style_all(corner);
        lv_line_set_points(corner, points[i], 5);
        lv_obj_set_style_line_width(corner, 3, 0);
        lv_obj_set_style_line_color(corner, lv_color_hex(0x91ff52), 0);
        lv_obj_set_style_line_rounded(corner, true, 0);
    }
    p->name = ui_text(p->root, "苹果", &ui_font_demibold_28, 32, 161, 80);
}
