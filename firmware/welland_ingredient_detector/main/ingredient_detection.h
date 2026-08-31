#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INGREDIENT_MAX_DETECTIONS 7
#define INGREDIENT_MODEL_WIDTH 224
#define INGREDIENT_MODEL_HEIGHT 224

esp_err_t ingredient_detection_start(void);

typedef struct {
    uint8_t category;
    float score;
    int16_t box[4];
} ingredient_detection_item_t;

typedef struct {
    size_t count;
    uint32_t elapsed_ms;
    uint32_t sequence;
    ingredient_detection_item_t items[INGREDIENT_MAX_DETECTIONS];
} ingredient_detection_result_t;

typedef enum {
    INGREDIENT_RECOGNITION_SEARCHING = 0,
    INGREDIENT_RECOGNITION_CONFIRMING,
    INGREDIENT_RECOGNITION_LOCKED,
    INGREDIENT_RECOGNITION_WEIGHING,
} ingredient_recognition_state_t;

typedef struct {
    ingredient_recognition_state_t state;
    bool inference_active;
    uint8_t candidate_category;
    uint8_t candidate_second_category;
    uint8_t collected_frames;
    uint8_t candidate_votes;
    uint8_t window_frames;
    float mean_score;
    float weighted_evidence;
    float evidence_margin;
    uint8_t confidence_level;
    uint8_t lock_reason;
    uint8_t effective_inferences;
    uint32_t completed_locks;
    uint32_t high_confidence_locks;
    uint32_t medium_confidence_locks;
    uint32_t low_confidence_locks;
    float average_lock_inferences;
    uint32_t quality_us;
    uint32_t preprocess_us;
    uint32_t inference_us;
    uint32_t postprocess_us;
    uint32_t decision_us;
    uint32_t total_us;
    uint16_t quality_weight_permille;
    uint16_t motion_permille;
    uint16_t sharpness;
    uint8_t mean_luma;
    uint16_t dark_permille;
    uint16_t bright_permille;
    uint16_t changed_permille;
    bool frame_fresh;
    bool frame_quality_ok;
    bool frame_stable;
    bool frame_roi_changed;
    bool frame_scene_change;
    uint32_t elapsed_ms;
    bool motion_candidate_active;
    bool experience_started;
    bool experience_locked;
    uint32_t experience_elapsed_ms;
} ingredient_recognition_status_t;

typedef struct {
    uint32_t quality_us;
    uint32_t preprocess_us;
    uint32_t inference_us;
    uint32_t postprocess_us;
    uint32_t decision_us;
    uint32_t total_us;
    uint8_t effective_inferences;
    size_t model_internal_bytes;
    size_t model_psram_bytes;
    size_t model_flash_bytes;
    size_t internal_free_before;
    size_t internal_free_after;
    size_t internal_min_free;
    size_t internal_largest_before;
    size_t internal_largest_after;
    size_t psram_free_before;
    size_t psram_free_after;
    size_t psram_min_free;
    size_t psram_largest_before;
    size_t psram_largest_after;
} ingredient_performance_t;

ingredient_detection_result_t ingredient_detection_run(const uint8_t *source_rgb565_be,
                                                        uint16_t source_width,
                                                        uint16_t source_height,
                                                        uint8_t *output_rgb565_be,
                                                        uint16_t output_width,
                                                        uint16_t output_height);

esp_err_t ingredient_detection_restart(void);
esp_err_t ingredient_detection_cancel(void);
esp_err_t ingredient_detection_start_weighing(void);
esp_err_t ingredient_detection_complete_weighing(void);
esp_err_t ingredient_detection_get_recognition_status(
    ingredient_recognition_status_t *status);
void ingredient_detection_set_box_overlay_enabled(bool enabled);
bool ingredient_detection_box_overlay_enabled(void);

esp_err_t ingredient_detection_validate_rgb565_be(
    const uint8_t *input_rgb565_be,
    uint16_t width,
    uint16_t height,
    ingredient_detection_result_t *result,
    ingredient_performance_t *performance);

#ifdef __cplusplus
}
#endif
