#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef RD_CONSTEXPR
#define RD_CONSTEXPR
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RECOGNITION_CLASS_COUNT 10
#define RECOGNITION_FRAME_WIDTH 224U
#define RECOGNITION_FRAME_HEIGHT 224U
#define RECOGNITION_QUALITY_GRID 52
#define RECOGNITION_MAX_EVIDENCE_FRAMES 3U
#define RECOGNITION_CANDIDATE_MAX_AGE_MS 500U
#define RECOGNITION_TARGET_LOST_MS 2000U
#define RECOGNITION_MOTION_SEARCH_RADIUS 3

typedef struct {
    uint16_t motion_max_permille;
    uint16_t motion_exit_permille;
    uint8_t stable_frames_required;
    uint8_t presentation_roi_percent;
    uint8_t target_roi_expand_percent;
    uint8_t scene_change_pixel_delta;
    uint16_t scene_change_changed_permille;
    uint8_t scene_change_required_frames;
    uint16_t sharpness_min;
    uint16_t sharpness_good;
    uint8_t exposure_mean_min;
    uint8_t exposure_mean_max;
    uint8_t dark_luma_max;
    uint8_t bright_luma_min;
    uint16_t clipped_max_permille;
    uint16_t weight_floor_permille;
} recognition_quality_config_t;

typedef struct {
    uint16_t left;
    uint16_t top;
    uint16_t right;
    uint16_t bottom;
} recognition_quality_roi_t;

typedef struct {
    bool fresh;
    bool accepted;
    bool stable;
    bool roi_changed;
    bool scene_change;
    uint16_t motion_permille;
    uint16_t changed_permille;
    uint16_t sharpness;
    uint8_t mean_luma;
    uint16_t dark_permille;
    uint16_t bright_permille;
    uint16_t weight_permille;
    uint64_t signature;
    uint16_t raw_motion_permille;
    int16_t displacement_x;
    int16_t displacement_y;
    bool tracking_reliable;
} recognition_quality_result_t;

typedef struct {
    bool has_reference;
    bool has_motion_reference;
    bool previous_tracking_reliable;
    uint16_t previous_compensated_motion;
    bool stable;
    uint8_t stable_frames;
    uint8_t scene_change_frames;
    recognition_quality_roi_t roi;
    uint64_t signature;
    uint8_t samples[RECOGNITION_QUALITY_GRID * RECOGNITION_QUALITY_GRID];
    uint8_t previous_delta[RECOGNITION_QUALITY_GRID * RECOGNITION_QUALITY_GRID];
} recognition_quality_state_t;

typedef struct {
    bool started;
    bool locked;
    uint32_t started_ms;
    uint32_t locked_elapsed_ms;
} recognition_experience_state_t;

typedef struct {
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
} recognition_box_t;

typedef enum {
    RECOGNITION_DECISION_SEARCHING = 0,
    RECOGNITION_DECISION_CONFIRMING,
    RECOGNITION_DECISION_LOCKED,
    RECOGNITION_DECISION_WEIGHING,
    RECOGNITION_DECISION_UNCERTAIN,
} recognition_decision_state_id_t;

typedef enum {
    RECOGNITION_EVENT_NONE = 0,
    RECOGNITION_EVENT_EVIDENCE_RESET,
    RECOGNITION_EVENT_LOCKED,
    RECOGNITION_EVENT_SESSION_TIMEOUT,
} recognition_decision_event_t;

typedef enum {
    RECOGNITION_CONFIDENCE_NONE = 0,
    RECOGNITION_CONFIDENCE_HIGH,
    RECOGNITION_CONFIDENCE_MEDIUM,
    RECOGNITION_CONFIDENCE_LOW,
    RECOGNITION_CONFIDENCE_COUNT,
} recognition_confidence_t;

typedef enum {
    RECOGNITION_LOCK_REASON_NONE = 0,
    RECOGNITION_LOCK_REASON_SINGLE_HIGH,
    RECOGNITION_LOCK_REASON_TWO_FRAME_EVIDENCE,
    RECOGNITION_LOCK_REASON_THREE_FRAME_EVIDENCE,
} recognition_lock_reason_t;

typedef struct {
    float single_high_threshold;
    float single_margin_threshold;
    float multi_accept_evidence;
    float multi_margin_evidence;
    float class_accept_thresholds[RECOGNITION_CLASS_COUNT];
    uint8_t max_valid_inferences;
    uint8_t target_missing_valid_frames;
    uint16_t box_min_area_permille;
    uint16_t box_max_area_permille;
    uint16_t box_min_side_pixels;
    uint16_t box_max_aspect_permille;
    uint16_t consistency_min_iou_permille;
    uint16_t consistency_max_center_shift_permille;
    uint16_t consistency_min_area_ratio_permille;
    uint16_t consistency_max_area_ratio_permille;
    float obvious_switch_threshold;
    float obvious_switch_margin;
    uint32_t session_timeout_ms;
} recognition_decision_config_t;

typedef struct {
    uint32_t sequence;
    uint32_t now_ms;
    uint16_t quality_weight_permille;
    float class_scores[RECOGNITION_CLASS_COUNT];
    recognition_box_t target_box;
    bool has_prediction;
    recognition_box_t predicted_box;
    bool ambiguous;
} recognition_observation_t;

typedef struct {
    uint32_t sequence;
    uint16_t quality_weight_permille;
    float class_scores[RECOGNITION_CLASS_COUNT];
    uint8_t top1_category;
    uint8_t top2_category;
    float top1_score;
    float top2_score;
    float top1_margin;
    recognition_box_t target_box;
    uint32_t target_area;
} recognition_frame_evidence_t;

typedef struct {
    recognition_decision_state_id_t state;
    float evidence[RECOGNITION_CLASS_COUNT];
    float normalized_evidence[RECOGNITION_CLASS_COUNT];
    float total_quality_weight;
    recognition_frame_evidence_t frames[RECOGNITION_MAX_EVIDENCE_FRAMES];
    uint8_t valid_inferences;
    uint8_t last_round_valid_inferences;
    uint8_t missing_valid_frames;
    bool has_target;
    recognition_box_t last_box;
    uint32_t last_sequence;
    uint32_t session_started_ms;
    uint32_t last_seen_ms;
    uint32_t lock_started_ms;
    uint32_t locked_decision_elapsed_ms;
    uint8_t candidate_category;
    uint8_t candidate_second_category;
    float candidate_score;
    float candidate_evidence;
    float candidate_margin;
    uint8_t locked_category;
    float locked_score;
    recognition_box_t locked_box;
    recognition_confidence_t locked_confidence;
    recognition_lock_reason_t lock_reason;
    uint32_t completed_locks;
    uint32_t total_lock_inferences;
    uint32_t confidence_lock_counts[RECOGNITION_CONFIDENCE_COUNT];
    uint32_t frame_lock_counts[RECOGNITION_MAX_EVIDENCE_FRAMES];
} recognition_decision_state_t;

RD_CONSTEXPR bool recognition_boxes_associate(const recognition_box_t *previous,
    const recognition_box_t *current, const recognition_decision_config_t *config);
RD_CONSTEXPR bool recognition_candidate_replace(uint32_t now_ms, uint32_t old_ms,
    uint16_t new_weight, uint16_t old_weight, bool new_accepted, bool old_accepted);

RD_CONSTEXPR void recognition_quality_reset(recognition_quality_state_t *state);
RD_CONSTEXPR recognition_quality_result_t recognition_quality_evaluate(
    recognition_quality_state_t *state,
    const uint8_t *rgb565_be,
    uint16_t width,
    uint16_t height,
    const recognition_quality_config_t *config,
    const recognition_quality_roi_t *roi);

RD_CONSTEXPR void recognition_experience_reset(recognition_experience_state_t *state);
RD_CONSTEXPR bool recognition_experience_confirm_target(
    recognition_experience_state_t *state,
    uint32_t fallback_started_ms);
RD_CONSTEXPR void recognition_experience_lock(recognition_experience_state_t *state,
                                              uint32_t now_ms);
RD_CONSTEXPR uint32_t recognition_experience_elapsed_ms(
    const recognition_experience_state_t *state,
    uint32_t now_ms);

RD_CONSTEXPR void recognition_decision_init(recognition_decision_state_t *state);
RD_CONSTEXPR void recognition_decision_cancel(recognition_decision_state_t *state);
RD_CONSTEXPR void recognition_decision_complete_weighing(recognition_decision_state_t *state);
RD_CONSTEXPR bool recognition_decision_start_weighing(recognition_decision_state_t *state);
RD_CONSTEXPR recognition_decision_event_t recognition_decision_tick(
    recognition_decision_state_t *state,
    const recognition_decision_config_t *config,
    uint32_t now_ms);
RD_CONSTEXPR recognition_decision_event_t recognition_decision_bad_frame(
    recognition_decision_state_t *state,
    const recognition_decision_config_t *config,
    uint32_t now_ms,
    bool scene_change);
RD_CONSTEXPR recognition_decision_event_t recognition_decision_target_missing(
    recognition_decision_state_t *state,
    const recognition_decision_config_t *config,
    uint32_t now_ms);
RD_CONSTEXPR recognition_decision_event_t recognition_decision_observe(
    recognition_decision_state_t *state,
    const recognition_decision_config_t *config,
    const recognition_observation_t *observation);

#ifdef __cplusplus
}
#endif
