#include "recognition_decision.h"

#define TEST_FRAME_SIZE 52U

static const recognition_quality_config_t QUALITY = {
    .motion_max_permille = 45,
    .motion_exit_permille = 80,
    .stable_frames_required = 2,
    .presentation_roi_percent = 70,
    .target_roi_expand_percent = 20,
    .scene_change_pixel_delta = 32,
    .scene_change_changed_permille = 600,
    .scene_change_required_frames = 2,
    .sharpness_min = 20,
    .sharpness_good = 80,
    .exposure_mean_min = 35,
    .exposure_mean_max = 220,
    .dark_luma_max = 8,
    .bright_luma_min = 247,
    .clipped_max_permille = 200,
    .weight_floor_permille = 500,
};

static const recognition_quality_roi_t FULL_ROI = {
    .left = 0,
    .top = 0,
    .right = TEST_FRAME_SIZE,
    .bottom = TEST_FRAME_SIZE,
};

static const recognition_decision_config_t DECISION = {
    .single_high_threshold = 0.92f,
    .single_margin_threshold = 0.18f,
    .multi_accept_evidence = 0.85f,
    .multi_margin_evidence = 0.15f,
    .class_accept_thresholds = {0.80f, 0.80f, 0.80f, 0.80f, 0.80f, 0.80f},
    .max_valid_inferences = 3,
    .target_missing_valid_frames = 1,
    .box_min_area_permille = 30,
    .box_max_area_permille = 850,
    .box_min_side_pixels = 22,
    .box_max_aspect_permille = 4000,
    .consistency_min_iou_permille = 150,
    .consistency_max_center_shift_permille = 220,
    .consistency_min_area_ratio_permille = 500,
    .consistency_max_area_ratio_permille = 2000,
    .obvious_switch_threshold = 0.90f,
    .obvious_switch_margin = 0.25f,
    .session_timeout_ms = 8000,
};

static recognition_observation_t observation(uint32_t sequence,
                                             uint32_t now_ms,
                                             uint8_t top1,
                                             float top1_score,
                                             uint8_t top2,
                                             float top2_score)
{
    recognition_observation_t value = {0};
    value.sequence = sequence;
    value.now_ms = now_ms;
    value.quality_weight_permille = 1000;
    value.class_scores[top1] = top1_score;
    if (top2 < RECOGNITION_CLASS_COUNT) {
        value.class_scores[top2] = top2_score;
    }
    value.target_box.left = 43;
    value.target_box.top = 43;
    value.target_box.right = 178;
    value.target_box.bottom = 178;
    return value;
}

static void set_gray(uint8_t *image, size_t pixel, uint8_t gray)
{
    const uint16_t red = ((uint16_t)gray * 31U) / 255U;
    const uint16_t green = ((uint16_t)gray * 63U) / 255U;
    const uint16_t blue = ((uint16_t)gray * 31U) / 255U;
    const uint16_t packed = (uint16_t)((red << 11U) | (green << 5U) | blue);
    image[pixel * 2U] = (uint8_t)(packed >> 8U);
    image[pixel * 2U + 1U] = (uint8_t)packed;
}

static void fill_gray(uint8_t *image, uint8_t gray)
{
    size_t pixel;
    for (pixel = 0; pixel < TEST_FRAME_SIZE * TEST_FRAME_SIZE; ++pixel) {
        set_gray(image, pixel, gray);
    }
}

static void fill_checker(uint8_t *image)
{
    size_t y;
    size_t x;
    for (y = 0; y < TEST_FRAME_SIZE; ++y) {
        for (x = 0; x < TEST_FRAME_SIZE; ++x) {
            set_gray(image,
                     y * TEST_FRAME_SIZE + x,
                     ((x / 2U + y / 2U) & 1U) ? 192 : 64);
        }
    }
}

static void fill_outside_roi(uint8_t *image,
                             const recognition_quality_roi_t *roi,
                             uint8_t gray)
{
    size_t y;
    size_t x;
    for (y = 0; y < TEST_FRAME_SIZE; ++y) {
        for (x = 0; x < TEST_FRAME_SIZE; ++x) {
            if (x < roi->left || x >= roi->right || y < roi->top || y >= roi->bottom) {
                set_gray(image, y * TEST_FRAME_SIZE + x, gray);
            }
        }
    }
}

static int test_single_high_lock(void)
{
    recognition_decision_state_t state;
    recognition_observation_t value = observation(1, 100, 0, 0.96f, 1, 0.70f);
    recognition_decision_init(&state);
    return recognition_decision_observe(&state, &DECISION, &value) == RECOGNITION_EVENT_LOCKED &&
           state.locked_category == 0 && state.valid_inferences == 1 &&
           state.locked_confidence == RECOGNITION_CONFIDENCE_HIGH &&
           state.lock_reason == RECOGNITION_LOCK_REASON_SINGLE_HIGH &&
           state.frames[0].top1_category == 0 && state.frames[0].top2_category == 1 &&
           state.frames[0].top1_margin > 0.25f && state.frames[0].target_area > 0 ? 0 : 1;
}

static int test_blur_or_motion_not_counted(void)
{
    uint8_t image[TEST_FRAME_SIZE * TEST_FRAME_SIZE * 2U] = {0};
    recognition_quality_state_t quality_state;
    recognition_decision_state_t decision_state;
    recognition_quality_result_t first;
    recognition_quality_result_t blurred;
    recognition_quality_result_t moving;
    recognition_observation_t value = observation(1, 100, 0, 0.85f, 1, 0.80f);
    recognition_quality_reset(&quality_state);
    recognition_decision_init(&decision_state);
    fill_gray(image, 120);
    first = recognition_quality_evaluate(&quality_state,
                                         image,
                                         TEST_FRAME_SIZE,
                                         TEST_FRAME_SIZE,
                                         &QUALITY,
                                         &FULL_ROI);
    set_gray(image, 8U * TEST_FRAME_SIZE + 8U, 122);
    blurred = recognition_quality_evaluate(&quality_state,
                                           image,
                                           TEST_FRAME_SIZE,
                                           TEST_FRAME_SIZE,
                                           &QUALITY,
                                           &FULL_ROI);
    recognition_decision_observe(&decision_state, &DECISION, &value);
    recognition_decision_bad_frame(&decision_state, &DECISION, 150, blurred.scene_change);
    fill_checker(image);
    moving = recognition_quality_evaluate(&quality_state,
                                          image,
                                          TEST_FRAME_SIZE,
                                          TEST_FRAME_SIZE,
                                          &QUALITY,
                                          &FULL_ROI);
    recognition_decision_bad_frame(&decision_state, &DECISION, 175, false);
    return !first.accepted && blurred.fresh && !blurred.accepted &&
           blurred.sharpness < QUALITY.sharpness_min && moving.fresh &&
           !moving.accepted && moving.motion_permille > QUALITY.motion_max_permille &&
           decision_state.valid_inferences == 1 ? 0 : 1;
}

static int test_background_motion_outside_roi_is_ignored(void)
{
    uint8_t image[TEST_FRAME_SIZE * TEST_FRAME_SIZE * 2U] = {0};
    recognition_quality_state_t quality_state;
    const recognition_quality_roi_t roi = {10, 10, 42, 42};
    recognition_quality_result_t quality;
    recognition_quality_reset(&quality_state);
    fill_checker(image);
    recognition_quality_evaluate(&quality_state,
                                 image,
                                 TEST_FRAME_SIZE,
                                 TEST_FRAME_SIZE,
                                 &QUALITY,
                                 &roi);
    set_gray(image, 20U * TEST_FRAME_SIZE + 20U, 180);
    quality = recognition_quality_evaluate(&quality_state,
                                           image,
                                           TEST_FRAME_SIZE,
                                           TEST_FRAME_SIZE,
                                           &QUALITY,
                                           &roi);
    if (!quality.accepted || !quality.stable) {
        return 1;
    }
    fill_outside_roi(image, &roi, 240);
    set_gray(image, 21U * TEST_FRAME_SIZE + 21U, 181);
    quality = recognition_quality_evaluate(&quality_state,
                                           image,
                                           TEST_FRAME_SIZE,
                                           TEST_FRAME_SIZE,
                                           &QUALITY,
                                           &roi);
    if (!quality.accepted || quality.scene_change ||
        quality.motion_permille > QUALITY.motion_exit_permille) {
        return 1;
    }
    fill_outside_roi(image, &roi, 20);
    set_gray(image, 22U * TEST_FRAME_SIZE + 22U, 182);
    quality = recognition_quality_evaluate(&quality_state,
                                           image,
                                           TEST_FRAME_SIZE,
                                           TEST_FRAME_SIZE,
                                           &QUALITY,
                                           &roi);
    return quality.accepted && !quality.scene_change &&
           quality.changed_permille >= QUALITY.scene_change_changed_permille &&
           quality.motion_permille <= QUALITY.motion_exit_permille ? 0 : 1;
}

static int test_two_same_low_not_locked(void)
{
    recognition_decision_state_t state;
    recognition_observation_t first = observation(1, 100, 0, 0.76f, 1, 0.0f);
    recognition_observation_t second = observation(2, 200, 0, 0.76f, 1, 0.0f);
    recognition_decision_init(&state);
    recognition_decision_observe(&state, &DECISION, &first);
    recognition_decision_observe(&state, &DECISION, &second);
    return state.state == RECOGNITION_DECISION_CONFIRMING &&
           state.valid_inferences == 2 ? 0 : 1;
}

static int test_two_conflict_weighted_winner(void)
{
    recognition_decision_state_t state;
    recognition_observation_t first = observation(1, 100, 0, 0.91f, 1, 0.10f);
    recognition_observation_t second = observation(2, 200, 1, 0.86f, 0, 0.85f);
    second.quality_weight_permille = 600;
    recognition_decision_init(&state);
    recognition_decision_observe(&state, &DECISION, &first);
    return recognition_decision_observe(&state, &DECISION, &second) ==
               RECOGNITION_EVENT_LOCKED &&
           state.locked_category == 0 && state.valid_inferences == 2 &&
           state.locked_confidence == RECOGNITION_CONFIDENCE_MEDIUM &&
           state.lock_reason == RECOGNITION_LOCK_REASON_TWO_FRAME_EVIDENCE &&
           state.candidate_score > 0.88f && state.candidate_score < 0.90f &&
           state.candidate_margin > DECISION.multi_margin_evidence ? 0 : 1;
}

static int test_three_different_forced_weighted_winner(void)
{
    recognition_decision_state_t state;
    recognition_observation_t first = observation(1, 100, 0, 0.84f, 1, 0.10f);
    recognition_observation_t second = observation(2, 200, 1, 0.82f, 0, 0.20f);
    recognition_observation_t third = observation(3, 300, 2, 0.80f, 0, 0.70f);
    recognition_decision_event_t event;
    second.quality_weight_permille = 500;
    recognition_decision_init(&state);
    recognition_decision_observe(&state, &DECISION, &first);
    recognition_decision_observe(&state, &DECISION, &second);
    event = recognition_decision_observe(&state, &DECISION, &third);
    return event == RECOGNITION_EVENT_LOCKED &&
           state.state == RECOGNITION_DECISION_LOCKED &&
           state.valid_inferences == 3 && state.locked_category == 0 &&
           state.locked_confidence == RECOGNITION_CONFIDENCE_LOW &&
           state.lock_reason == RECOGNITION_LOCK_REASON_THREE_FRAME_FORCED &&
           state.frames[0].top1_category == 0 && state.frames[1].top1_category == 1 &&
           state.frames[2].top1_category == 2 &&
           state.evidence[0] > state.evidence[1] && state.evidence[0] > state.evidence[2] ?
        0 : 1;
}

static int test_third_frame_stops_future_inference(void)
{
    recognition_decision_state_t state;
    recognition_observation_t fourth = observation(4, 400, 5, 0.99f, 0, 0.0f);
    float evidence_before;
    recognition_decision_init(&state);
    for (uint8_t index = 0; index < 3; ++index) {
        recognition_observation_t value = observation(index + 1,
                                                      100U * (index + 1U),
                                                      0,
                                                      0.76f,
                                                      6,
                                                      0.0f);
        recognition_decision_observe(&state, &DECISION, &value);
    }
    evidence_before = state.evidence[0];
    recognition_decision_observe(&state, &DECISION, &fourth);
    return state.state == RECOGNITION_DECISION_LOCKED &&
           state.valid_inferences == 3 && state.locked_category == 0 &&
           state.evidence[0] == evidence_before ? 0 : 1;
}

static int test_duplicate_frame_not_evidence(void)
{
    uint8_t image[TEST_FRAME_SIZE * TEST_FRAME_SIZE * 2U] = {0};
    recognition_quality_state_t quality_state;
    recognition_decision_state_t decision_state;
    recognition_quality_result_t quality;
    recognition_observation_t value = observation(1, 100, 0, 0.85f, 1, 0.80f);
    recognition_quality_reset(&quality_state);
    recognition_decision_init(&decision_state);
    fill_checker(image);
    recognition_quality_evaluate(&quality_state,
                                 image,
                                 TEST_FRAME_SIZE,
                                 TEST_FRAME_SIZE,
                                 &QUALITY,
                                 &FULL_ROI);
    set_gray(image, 8U * TEST_FRAME_SIZE + 8U, 180);
    quality = recognition_quality_evaluate(&quality_state,
                                           image,
                                           TEST_FRAME_SIZE,
                                           TEST_FRAME_SIZE,
                                           &QUALITY,
                                           &FULL_ROI);
    if (!quality.accepted) {
        return 1;
    }
    recognition_decision_observe(&decision_state, &DECISION, &value);
    recognition_decision_observe(&decision_state, &DECISION, &value);
    if (decision_state.valid_inferences != 1) {
        return 1;
    }
    quality = recognition_quality_evaluate(&quality_state,
                                           image,
                                           TEST_FRAME_SIZE,
                                           TEST_FRAME_SIZE,
                                           &QUALITY,
                                           &FULL_ROI);
    recognition_decision_bad_frame(&decision_state, &DECISION, 200, quality.scene_change);
    return !quality.fresh && !quality.accepted && decision_state.valid_inferences == 1 ? 0 : 1;
}

static int test_missing_and_switch_reset(void)
{
    recognition_decision_state_t state;
    recognition_observation_t first = observation(1, 100, 0, 0.85f, 1, 0.80f);
    recognition_observation_t switched = observation(2, 300, 1, 0.91f, 0, 0.60f);
    recognition_decision_init(&state);
    recognition_decision_observe(&state, &DECISION, &first);
    recognition_decision_target_missing(&state, &DECISION, 200);
    if (state.valid_inferences != 0 || state.state != RECOGNITION_DECISION_SEARCHING) {
        return 1;
    }
    recognition_decision_observe(&state, &DECISION, &first);
    return recognition_decision_observe(&state, &DECISION, &switched) ==
               RECOGNITION_EVENT_EVIDENCE_RESET &&
           state.valid_inferences == 1 && state.candidate_category == 1 ? 0 : 1;
}

static int test_box_jump_resets_display(void)
{
    recognition_decision_state_t state;
    recognition_observation_t first = observation(1, 100, 0, 0.85f, 1, 0.80f);
    recognition_observation_t jumped = observation(2, 200, 0, 0.86f, 1, 0.79f);
    jumped.target_box.left = 0;
    jumped.target_box.top = 0;
    jumped.target_box.right = 50;
    jumped.target_box.bottom = 50;
    recognition_decision_init(&state);
    recognition_decision_observe(&state, &DECISION, &first);
    return recognition_decision_observe(&state, &DECISION, &jumped) ==
               RECOGNITION_EVENT_EVIDENCE_RESET &&
           state.valid_inferences == 1 && state.frames[0].sequence == 2 ? 0 : 1;
}

static int test_locked_move_does_not_change(void)
{
    recognition_decision_state_t state;
    recognition_observation_t lock = observation(1, 100, 0, 0.96f, 1, 0.70f);
    recognition_observation_t other = observation(2, 200, 3, 0.99f, 0, 0.0f);
    recognition_decision_init(&state);
    state.session_started_ms = 50;
    recognition_decision_observe(&state, &DECISION, &lock);
    recognition_decision_bad_frame(&state, &DECISION, 150, true);
    recognition_decision_target_missing(&state, &DECISION, 175);
    recognition_decision_observe(&state, &DECISION, &other);
    recognition_decision_tick(&state, &DECISION, UINT32_MAX - 1U);
    if (state.locked_category != 0 || state.state != RECOGNITION_DECISION_LOCKED ||
        state.locked_decision_elapsed_ms != 50) {
        return 1;
    }
    return recognition_decision_start_weighing(&state) &&
           state.state == RECOGNITION_DECISION_WEIGHING &&
           state.locked_category == 0 ? 0 : 1;
}

static int test_weighing_complete_resets(void)
{
    recognition_decision_state_t state;
    recognition_observation_t lock = observation(1, 100, 0, 0.96f, 1, 0.70f);
    recognition_decision_init(&state);
    recognition_decision_observe(&state, &DECISION, &lock);
    recognition_decision_start_weighing(&state);
    recognition_decision_complete_weighing(&state);
    return state.state == RECOGNITION_DECISION_SEARCHING &&
           state.locked_category == UINT8_MAX && state.valid_inferences == 0 &&
           state.locked_decision_elapsed_ms == 0 ? 0 : 1;
}

static int test_cancel_resets_locked_result(void)
{
    recognition_decision_state_t state;
    recognition_observation_t lock = observation(1, 100, 0, 0.96f, 1, 0.70f);
    recognition_decision_init(&state);
    recognition_decision_observe(&state, &DECISION, &lock);
    recognition_decision_cancel(&state);
    return state.state == RECOGNITION_DECISION_SEARCHING &&
           state.locked_category == UINT8_MAX && state.valid_inferences == 0 &&
           state.locked_decision_elapsed_ms == 0 ? 0 : 1;
}

static int test_experience_starts_at_first_evidence(void)
{
    recognition_experience_state_t state;
    recognition_experience_reset(&state);
    if (!recognition_experience_confirm_target(&state, 650) ||
        !state.started || state.started_ms != 650) {
        return 1;
    }
    if (recognition_experience_elapsed_ms(&state, 900) != 250) {
        return 1;
    }
    recognition_experience_lock(&state, 1000);
    return state.locked && state.locked_elapsed_ms == 350 &&
           recognition_experience_elapsed_ms(&state, 5000) == 350 ? 0 : 1;
}

static int test_unrelated_motion_cannot_backdate_experience(void)
{
    recognition_experience_state_t state;
    recognition_experience_reset(&state);
    if (!recognition_experience_confirm_target(&state, 3500)) {
        return 1;
    }
    return state.started_ms == 3500 &&
           recognition_experience_elapsed_ms(&state, 4100) == 600 ? 0 : 1;
}

static int test_experience_resets_only_when_explicitly_reset(void)
{
    recognition_experience_state_t state;
    recognition_experience_reset(&state);
    recognition_experience_confirm_target(&state, 400);
    if (!state.started || state.started_ms != 400 ||
        recognition_experience_elapsed_ms(&state, 900) != 500) {
        return 1;
    }
    recognition_experience_reset(&state);
    return !state.started && !state.locked &&
           recognition_experience_elapsed_ms(&state, 1000) == 0 ? 0 : 1;
}

#ifndef RD_COMPILE_TIME_TEST
__attribute__((export_name("run_tests")))
#endif
static int run_tests(void)
{
    int failures = 0;
    failures += test_single_high_lock();
    failures += test_blur_or_motion_not_counted();
    failures += test_background_motion_outside_roi_is_ignored();
    failures += test_two_same_low_not_locked();
    failures += test_two_conflict_weighted_winner();
    failures += test_three_different_forced_weighted_winner();
    failures += test_third_frame_stops_future_inference();
    failures += test_duplicate_frame_not_evidence();
    failures += test_missing_and_switch_reset();
    failures += test_box_jump_resets_display();
    failures += test_locked_move_does_not_change();
    failures += test_weighing_complete_resets();
    failures += test_cancel_resets_locked_result();
    failures += test_experience_starts_at_first_evidence();
    failures += test_unrelated_motion_cannot_backdate_experience();
    failures += test_experience_resets_only_when_explicitly_reset();
    return failures;
}
