idf_component_register(
    SRCS "app_main.c" "lcd.c" "product_ui_app.c"
         "ui/product_ui.c" "ui/ui_theme.c" "ui/ui_gradient_text.c"
         "ui/ui_metric_panel.c" "ui/ui_food_panel.c" "ui/ui_loading.c"
         "ui/ui_category_scene.c"
         "ui/animation/ui_loading_animation.c"
         "ui/animation/ui_category_animation.c"
         "ui/fonts/ui_font_demibold_27.c" "ui/fonts/ui_font_medium_27.c"
         "ui/fonts/ui_font_normal_28.c" "ui/fonts/ui_font_normal_27.c"
         "ui/fonts/ui_font_light_85.c" "ui/fonts/ui_font_demibold_28.c"
         "ui/fonts/ui_font_demibold_36.c"
         "ui/fonts/ui_font_light_84.c"
         "ui/assets/ui_asset_eyes.c" "ui/assets/ui_asset_apple.c"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_gpio esp_driver_spi esp_psram esp_timer lvgl)
target_compile_definitions(${COMPONENT_LIB} PRIVATE LV_LVGL_H_INCLUDE_SIMPLE)
