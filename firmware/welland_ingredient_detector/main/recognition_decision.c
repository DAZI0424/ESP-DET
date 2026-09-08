#include "recognition_decision.h"

#define QUALITY_SCALE 1000U

RD_CONSTEXPR static uint16_t clamp_u16(uint32_t value)
{
    return value > 65535U ? 65535U : (uint16_t)value;
}

RD_CONSTEXPR static uint8_t rgb565_luma(const uint8_t *pixel)
{
    const uint16_t packed = ((uint16_t)pixel[0] << 8U) | pixel[1];
    const uint32_t red = (((packed >> 11U) & 0x1fU) * 255U) / 31U;
    const uint32_t green = (((packed >> 5U) & 0x3fU) * 255U) / 63U;
    const uint32_t blue = ((packed & 0x1fU) * 255U) / 31U;
    return (uint8_t)((77U * red + 150U * green + 29U * blue) >> 8U);
}

RD_CONSTEXPR void recognition_quality_reset(recognition_quality_state_t *state)
{
    size_t index;
    state->has_reference = false;
    state->has_motion_reference = false;
    state->previous_tracking_reliable = false;
    state->previous_compensated_motion = 0;
    state->stable = false;
    state->stable_frames = 0;
    state->scene_change_frames = 0;
    state->roi = (recognition_quality_roi_t){0};
    state->signature = 0;
    for (index = 0; index < sizeof(state->samples); ++index) {
        state->samples[index] = 0;
        state->previous_delta[index] = 0;
    }
}

RD_CONSTEXPR static bool quality_roi_equal(const recognition_quality_roi_t *left,
                                           const recognition_quality_roi_t *right)
{
    return left->left == right->left && left->top == right->top &&
           left->right == right->right && left->bottom == right->bottom;
}

RD_CONSTEXPR recognition_quality_result_t recognition_quality_evaluate(
    recognition_quality_state_t *state,
    const uint8_t *rgb565_be,
    uint16_t width,
    uint16_t height,
    const recognition_quality_config_t *config,
    const recognition_quality_roi_t *requested_roi)
{
    recognition_quality_result_t result = {0};
    recognition_quality_roi_t roi = {0, 0, width, height};
    uint8_t current[RECOGNITION_QUALITY_GRID * RECOGNITION_QUALITY_GRID];
    uint64_t signature = UINT64_C(1469598103934665603);
    uint32_t luma_sum = 0;
    uint32_t dark_count = 0;
    uint32_t bright_count = 0;
    uint32_t motion_sum = 0;
    uint32_t previous_motion_sum = 0;
    uint64_t previous_signature = UINT64_C(1469598103934665603);
    uint32_t sharpness_sum = 0;
    uint32_t roi_sample_count = 0;
    uint32_t sharpness_count = 0;
    uint32_t changed_count = 0;
    const uint32_t full_sample_count = RECOGNITION_QUALITY_GRID * RECOGNITION_QUALITY_GRID;
    size_t y;
    size_t x;

    if (state == NULL || rgb565_be == NULL || config == NULL || width == 0 || height == 0) {
        return result;
    }
    if (requested_roi != NULL && requested_roi->left < requested_roi->right &&
        requested_roi->top < requested_roi->bottom && requested_roi->left < width &&
        requested_roi->top < height) {
        roi = *requested_roi;
        if (roi.right > width) {
            roi.right = width;
        }
        if (roi.bottom > height) {
            roi.bottom = height;
        }
    }
    result.roi_changed = state->has_reference && !quality_roi_equal(&state->roi, &roi);

    for (y = 0; y < RECOGNITION_QUALITY_GRID; ++y) {
        const size_t source_y = (y * height + RECOGNITION_QUALITY_GRID / 2U) /
                                RECOGNITION_QUALITY_GRID;
        for (x = 0; x < RECOGNITION_QUALITY_GRID; ++x) {
            const size_t source_x = (x * width + RECOGNITION_QUALITY_GRID / 2U) /
                                    RECOGNITION_QUALITY_GRID;
            const size_t source_index = 2U *
                ((source_y < height ? source_y : height - 1U) * width +
                 (source_x < width ? source_x : width - 1U));
            const size_t index = y * RECOGNITION_QUALITY_GRID + x;
            const uint8_t luma = rgb565_luma(&rgb565_be[source_index]);
            current[index] = luma;
            if (state->has_reference) {
                const int difference = (int)luma - state->samples[index];
                const uint32_t absolute_difference =
                    (uint32_t)(difference < 0 ? -difference : difference);
                changed_count += absolute_difference >= config->scene_change_pixel_delta;
                if (source_x >= roi.left && source_x < roi.right &&
                    source_y >= roi.top && source_y < roi.bottom) {
                    motion_sum += absolute_difference;
                    previous_motion_sum += state->previous_delta[index];
                }
                state->previous_delta[index] = (uint8_t)absolute_difference;
            }
            if (source_x >= roi.left && source_x < roi.right &&
                source_y >= roi.top && source_y < roi.bottom) {
                luma_sum += luma;
                dark_count += luma <= config->dark_luma_max;
                bright_count += luma >= config->bright_luma_min;
                previous_signature ^= state->samples[index];
                previous_signature *= UINT64_C(1099511628211);
                signature ^= luma;
                signature *= UINT64_C(1099511628211);
                ++roi_sample_count;
            }
        }
    }

    for (y = 1; y + 1 < RECOGNITION_QUALITY_GRID; ++y) {
        for (x = 1; x + 1 < RECOGNITION_QUALITY_GRID; ++x) {
            const size_t source_y = (y * height + RECOGNITION_QUALITY_GRID / 2U) /
                                    RECOGNITION_QUALITY_GRID;
            const size_t source_x = (x * width + RECOGNITION_QUALITY_GRID / 2U) /
                                    RECOGNITION_QUALITY_GRID;
            if (source_x < roi.left || source_x >= roi.right ||
                source_y < roi.top || source_y >= roi.bottom) {
                continue;
            }
            const size_t index = y * RECOGNITION_QUALITY_GRID + x;
            int laplacian = 4 * current[index] - current[index - 1] - current[index + 1] -
                            current[index - RECOGNITION_QUALITY_GRID] -
                            current[index + RECOGNITION_QUALITY_GRID];
            sharpness_sum += (uint32_t)(laplacian < 0 ? -laplacian : laplacian);
            ++sharpness_count;
        }
    }

    result.signature = signature;
    result.fresh = !state->has_reference || signature != previous_signature;
    if (roi_sample_count == 0) {
        return result;
    }
    result.mean_luma = (uint8_t)(luma_sum / roi_sample_count);
    result.dark_permille = (uint16_t)((dark_count * QUALITY_SCALE) / roi_sample_count);
    result.bright_permille = (uint16_t)((bright_count * QUALITY_SCALE) / roi_sample_count);
    result.sharpness = sharpness_count == 0 ? 0 :
        clamp_u16(sharpness_sum / sharpness_count);
    if (state->has_reference) {
        result.motion_permille = clamp_u16(
            (motion_sum * QUALITY_SCALE) / (roi_sample_count * 255U));
        result.raw_motion_permille = result.motion_permille;
        // Fixed search on sparse luma: reject ambiguous/repeating textures and
        // boundary winners rather than inventing a displacement.
        const int radius = RECOGNITION_MOTION_SEARCH_RADIUS;
        uint32_t best = UINT32_MAX, second = UINT32_MAX, count = 0;
        int best_dx = 0, best_dy = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                uint32_t sad = 0, n = 0;
                for (int gy = radius; gy < RECOGNITION_QUALITY_GRID-radius; gy += 2) {
                    for (int gx = radius; gx < RECOGNITION_QUALITY_GRID-radius; gx += 2) {
                        const int px = gx * width / RECOGNITION_QUALITY_GRID;
                        const int py = gy * height / RECOGNITION_QUALITY_GRID;
                        if (px < roi.left || px >= roi.right || py < roi.top || py >= roi.bottom) continue;
                        int d = current[(gy+dy)*RECOGNITION_QUALITY_GRID+gx+dx] -
                            state->samples[gy*RECOGNITION_QUALITY_GRID+gx];
                        sad += d < 0 ? -d : d;
                        ++n;
                    }
                }
                if (n < 16) continue;
                if (sad < best) {
                    second = best; best = sad; best_dx = dx; best_dy = dy; count = n;
                } else if (sad < second) second = sad;
            }
        }
        if (count >= 16 && best_dx > -radius && best_dx < radius &&
            best_dy > -radius && best_dy < radius && second > best + 2U*count &&
            best * QUALITY_SCALE <= count * 255U * config->motion_max_permille) {
            result.tracking_reliable = true;
            result.displacement_x = best_dx * (int)width / RECOGNITION_QUALITY_GRID;
            result.displacement_y = best_dy * (int)height / RECOGNITION_QUALITY_GRID;
            result.motion_permille = (uint16_t)(best * QUALITY_SCALE / (count * 255U));
        }
        result.changed_permille = (uint16_t)((changed_count * QUALITY_SCALE) /
                                             full_sample_count);
        if (result.changed_permille >= config->scene_change_changed_permille &&
            result.motion_permille > config->motion_exit_permille) {
            if (state->scene_change_frames < UINT8_MAX) {
                ++state->scene_change_frames;
            }
        } else {
            state->scene_change_frames = 0;
        }
        result.scene_change = state->scene_change_frames >=
            config->scene_change_required_frames;
    }

    const int roi_dx = (int)roi.left - state->roi.left;
    const int roi_dy = (int)roi.top - state->roi.top;
    const bool translated_roi = result.roi_changed && result.tracking_reliable &&
        state->previous_tracking_reliable &&
        roi.right-roi.left == state->roi.right-state->roi.left &&
        roi.bottom-roi.top == state->roi.bottom-state->roi.top &&
        roi_dx*roi_dx <= (int)(width*width*9U/(RECOGNITION_QUALITY_GRID*RECOGNITION_QUALITY_GRID)) &&
        roi_dy*roi_dy <= (int)(height*height*9U/(RECOGNITION_QUALITY_GRID*RECOGNITION_QUALITY_GRID));
    if (!state->has_reference || result.roi_changed) {
        state->stable = false;
        // Recheck the previous interval inside the NEW ROI; do not inherit
        // stability measured over a different region.
        state->stable_frames = state->has_motion_reference &&
            (previous_motion_sum * QUALITY_SCALE) / (roi_sample_count * 255U) <=
                config->motion_max_permille ? 1 : 0;
        if (translated_roi && state->previous_compensated_motion <= config->motion_max_permille) {
            state->stable_frames = 1;
        }
    }
    if (result.scene_change || result.motion_permille > config->motion_exit_permille) {
        state->stable = false;
        state->stable_frames = 0;
    } else if (result.motion_permille <= config->motion_max_permille) {
        if (state->stable_frames < UINT8_MAX) {
            ++state->stable_frames;
        }
        if (state->stable_frames >= config->stable_frames_required) {
            state->stable = true;
        }
    }
    result.stable = state->stable;

    for (x = 0; x < full_sample_count; ++x) {
        state->samples[x] = current[x];
    }
    state->previous_tracking_reliable = result.tracking_reliable;
    state->previous_compensated_motion = result.motion_permille;
    state->signature = signature;
    state->roi = roi;
    state->has_motion_reference = state->has_reference;
    state->has_reference = true;

    if (!result.fresh || !result.stable || result.scene_change ||
        result.sharpness < config->sharpness_min ||
        result.mean_luma < config->exposure_mean_min ||
        result.mean_luma > config->exposure_mean_max ||
        result.dark_permille + result.bright_permille > config->clipped_max_permille) {
        return result;
    }

    {
        const uint32_t motion_limit = config->motion_exit_permille == 0 ? 1U :
            config->motion_exit_permille;
        const uint32_t motion_quality = result.motion_permille >= motion_limit ? 0U :
            QUALITY_SCALE - (result.motion_permille * QUALITY_SCALE) / motion_limit;
        const uint32_t sharpness_quality = result.sharpness >= config->sharpness_good ?
            QUALITY_SCALE :
            ((uint32_t)(result.sharpness - config->sharpness_min) * QUALITY_SCALE) /
                (config->sharpness_good > config->sharpness_min ?
                    config->sharpness_good - config->sharpness_min : 1U);
        const uint32_t exposure_mid =
            ((uint32_t)config->exposure_mean_min + config->exposure_mean_max) / 2U;
        const uint32_t exposure_radius =
            ((uint32_t)config->exposure_mean_max - config->exposure_mean_min) / 2U;
        const uint32_t exposure_distance = result.mean_luma > exposure_mid ?
            result.mean_luma - exposure_mid : exposure_mid - result.mean_luma;
        const uint32_t exposure_quality = exposure_radius == 0 ? QUALITY_SCALE :
            QUALITY_SCALE - (exposure_distance * QUALITY_SCALE) / exposure_radius;
        const uint32_t clipped = result.dark_permille + result.bright_permille;
        const uint32_t clip_quality = config->clipped_max_permille == 0 ? QUALITY_SCALE :
            QUALITY_SCALE - (clipped * QUALITY_SCALE) / config->clipped_max_permille;
        uint32_t weight =
            (motion_quality + sharpness_quality + exposure_quality + clip_quality) / 4U;
        if (weight < config->weight_floor_permille) {
            weight = config->weight_floor_permille;
        }
        result.weight_permille = (uint16_t)(weight > QUALITY_SCALE ? QUALITY_SCALE : weight);
    }
    result.accepted = true;
    return result;
}

RD_CONSTEXPR void recognition_experience_reset(recognition_experience_state_t *state)
{
    state->started = false;
    state->locked = false;
    state->started_ms = 0;
    state->locked_elapsed_ms = 0;
}

RD_CONSTEXPR bool recognition_experience_confirm_target(
    recognition_experience_state_t *state,
    uint32_t fallback_started_ms)
{
    if (state->started) {
        return false;
    }
    state->started = true;
    state->locked = false;
    state->started_ms = fallback_started_ms == 0 ? 1 : fallback_started_ms;
    state->locked_elapsed_ms = 0;
    return true;
}

RD_CONSTEXPR void recognition_experience_lock(recognition_experience_state_t *state,
                                              uint32_t now_ms)
{
    if (!state->started || state->locked) {
        return;
    }
    state->locked_elapsed_ms = now_ms - state->started_ms;
    state->locked = true;
}

RD_CONSTEXPR uint32_t recognition_experience_elapsed_ms(
    const recognition_experience_state_t *state,
    uint32_t now_ms)
{
    if (!state->started) {
        return 0;
    }
    return state->locked ? state->locked_elapsed_ms : now_ms - state->started_ms;
}

RD_CONSTEXPR static void clear_evidence(recognition_decision_state_t *state)
{
    size_t index;
    size_t frame_index;
    for (index = 0; index < RECOGNITION_CLASS_COUNT; ++index) {
        state->evidence[index] = 0.0f;
        state->normalized_evidence[index] = 0.0f;
    }
    for (frame_index = 0; frame_index < RECOGNITION_MAX_EVIDENCE_FRAMES; ++frame_index) {
        recognition_frame_evidence_t *frame = &state->frames[frame_index];
        frame->sequence = 0;
        frame->quality_weight_permille = 0;
        frame->top1_category = UINT8_MAX;
        frame->top2_category = UINT8_MAX;
        frame->top1_score = 0.0f;
        frame->top2_score = 0.0f;
        frame->top1_margin = 0.0f;
        frame->target_box.left = 0;
        frame->target_box.top = 0;
        frame->target_box.right = 0;
        frame->target_box.bottom = 0;
        frame->target_area = 0;
        for (index = 0; index < RECOGNITION_CLASS_COUNT; ++index) {
            frame->class_scores[index] = 0.0f;
        }
    }
    state->total_quality_weight = 0.0f;
    state->valid_inferences = 0;
    state->missing_valid_frames = 0;
    state->has_target = false;
    state->last_sequence = 0;
    state->session_started_ms = 0;
    state->last_seen_ms = 0;
    state->candidate_category = UINT8_MAX;
    state->candidate_second_category = UINT8_MAX;
    state->candidate_score = 0.0f;
    state->candidate_evidence = 0.0f;
    state->candidate_margin = 0.0f;
}

RD_CONSTEXPR static void reset_session(recognition_decision_state_t *state)
{
    const uint32_t completed_locks = state->completed_locks;
    const uint32_t total_lock_inferences = state->total_lock_inferences;
    uint32_t confidence_lock_counts[RECOGNITION_CONFIDENCE_COUNT];
    size_t index;
    for (index = 0; index < RECOGNITION_CONFIDENCE_COUNT; ++index) {
        confidence_lock_counts[index] = state->confidence_lock_counts[index];
    }
    state->state = RECOGNITION_DECISION_SEARCHING;
    clear_evidence(state);
    state->last_round_valid_inferences = 0;
    state->lock_started_ms = 0;
    state->locked_decision_elapsed_ms = 0;
    state->locked_category = UINT8_MAX;
    state->locked_score = 0.0f;
    state->locked_box.left = 0;
    state->locked_box.top = 0;
    state->locked_box.right = 0;
    state->locked_box.bottom = 0;
    state->locked_confidence = RECOGNITION_CONFIDENCE_NONE;
    state->lock_reason = RECOGNITION_LOCK_REASON_NONE;
    state->completed_locks = completed_locks;
    state->total_lock_inferences = total_lock_inferences;
    for (index = 0; index < RECOGNITION_CONFIDENCE_COUNT; ++index) {
        state->confidence_lock_counts[index] = confidence_lock_counts[index];
    }
}

RD_CONSTEXPR void recognition_decision_init(recognition_decision_state_t *state)
{
    state->completed_locks = 0;
    state->total_lock_inferences = 0;
    for (size_t index = 0; index < RECOGNITION_CONFIDENCE_COUNT; ++index) {
        state->confidence_lock_counts[index] = 0;
    }
    for (size_t index = 0; index < RECOGNITION_MAX_EVIDENCE_FRAMES; ++index) {
        state->frame_lock_counts[index] = 0;
    }
    reset_session(state);
}

RD_CONSTEXPR void recognition_decision_cancel(recognition_decision_state_t *state)
{
    reset_session(state);
}

RD_CONSTEXPR void recognition_decision_complete_weighing(recognition_decision_state_t *state)
{
    reset_session(state);
}

RD_CONSTEXPR bool recognition_decision_start_weighing(recognition_decision_state_t *state)
{
    if (state->state != RECOGNITION_DECISION_LOCKED) {
        return false;
    }
    state->state = RECOGNITION_DECISION_WEIGHING;
    return true;
}

RD_CONSTEXPR static uint32_t elapsed_ms(uint32_t now_ms, uint32_t started_ms)
{
    return now_ms - started_ms;
}

RD_CONSTEXPR recognition_decision_event_t recognition_decision_tick(
    recognition_decision_state_t *state,
    const recognition_decision_config_t *config,
    uint32_t now_ms)
{
    if (state->state == RECOGNITION_DECISION_LOCKED ||
        state->state == RECOGNITION_DECISION_WEIGHING) return RECOGNITION_EVENT_NONE;
    if ((state->has_target && elapsed_ms(now_ms, state->last_seen_ms) >= RECOGNITION_TARGET_LOST_MS) ||
        (state->session_started_ms != 0 &&
         elapsed_ms(now_ms, state->session_started_ms) >= config->session_timeout_ms)) {
        clear_evidence(state);
        state->state = RECOGNITION_DECISION_SEARCHING;
        return RECOGNITION_EVENT_SESSION_TIMEOUT;
    }
    return RECOGNITION_EVENT_NONE;
}

RD_CONSTEXPR recognition_decision_event_t recognition_decision_bad_frame(
    recognition_decision_state_t *state,
    const recognition_decision_config_t *config,
    uint32_t now_ms,
    bool scene_change)
{
    recognition_decision_event_t event = recognition_decision_tick(state, config, now_ms);
    if (state->state == RECOGNITION_DECISION_LOCKED ||
        state->state == RECOGNITION_DECISION_WEIGHING) {
        return event;
    }
    if (scene_change && state->has_target) {
        clear_evidence(state);
        state->state = RECOGNITION_DECISION_SEARCHING;
        return RECOGNITION_EVENT_EVIDENCE_RESET;
    }
    return event;
}

RD_CONSTEXPR recognition_decision_event_t recognition_decision_target_missing(
    recognition_decision_state_t *state,
    const recognition_decision_config_t *config,
    uint32_t now_ms)
{
    recognition_decision_event_t event = recognition_decision_tick(state, config, now_ms);
    if (state->state == RECOGNITION_DECISION_LOCKED ||
        state->state == RECOGNITION_DECISION_WEIGHING) {
        return event;
    }
    if (state->missing_valid_frames < UINT8_MAX) {
        ++state->missing_valid_frames;
    }
    return event;
}

RD_CONSTEXPR static uint32_t box_area(const recognition_box_t *box)
{
    const int32_t width = (int32_t)box->right - box->left;
    const int32_t height = (int32_t)box->bottom - box->top;
    return width > 0 && height > 0 ? (uint32_t)(width * height) : 0;
}

RD_CONSTEXPR bool recognition_candidate_replace(uint32_t now_ms, uint32_t old_ms,
    uint16_t new_weight, uint16_t old_weight, bool new_accepted, bool old_accepted)
{
    if (elapsed_ms(now_ms, old_ms) >= RECOGNITION_CANDIDATE_MAX_AGE_MS) return true;
    if (new_accepted != old_accepted) return new_accepted;
    return new_weight >= old_weight || elapsed_ms(now_ms, old_ms) >= 150U;
}

RD_CONSTEXPR bool recognition_boxes_associate(const recognition_box_t *previous,
    const recognition_box_t *current, const recognition_decision_config_t *config)
{
    const uint32_t a = box_area(previous), b = box_area(current);
    if (a == 0 || b == 0 || b * QUALITY_SCALE < a * config->consistency_min_area_ratio_permille ||
        b * QUALITY_SCALE > a * config->consistency_max_area_ratio_permille) return false;
    const int left = previous->left > current->left ? previous->left : current->left;
    const int top = previous->top > current->top ? previous->top : current->top;
    const int right = previous->right < current->right ? previous->right : current->right;
    const int bottom = previous->bottom < current->bottom ? previous->bottom : current->bottom;
    const uint32_t intersection = right > left && bottom > top ? (right-left)*(bottom-top) : 0;
    int dx = previous->left + previous->right - current->left - current->right;
    int dy = previous->top + previous->bottom - current->top - current->bottom;
    if (dx < 0) dx = -dx;
    if (dy < 0) dy = -dy;
    return intersection * QUALITY_SCALE >= (a+b-intersection) * config->consistency_min_iou_permille &&
        (uint32_t)dx * QUALITY_SCALE <= 2U * RECOGNITION_FRAME_WIDTH * config->consistency_max_center_shift_permille &&
        (uint32_t)dy * QUALITY_SCALE <= 2U * RECOGNITION_FRAME_HEIGHT * config->consistency_max_center_shift_permille;
}

RD_CONSTEXPR static bool box_reasonable(const recognition_box_t *box,
                           const recognition_decision_config_t *config)
{
    const uint32_t width = box->right > box->left ? (uint32_t)(box->right - box->left) : 0;
    const uint32_t height = box->bottom > box->top ? (uint32_t)(box->bottom - box->top) : 0;
    const uint32_t area = width * height;
    const uint32_t frame_area = RECOGNITION_FRAME_WIDTH * RECOGNITION_FRAME_HEIGHT;
    const uint32_t short_side = width < height ? width : height;
    const uint32_t long_side = width > height ? width : height;
    if (short_side < config->box_min_side_pixels || area * QUALITY_SCALE <
        frame_area * config->box_min_area_permille || area * QUALITY_SCALE >
        frame_area * config->box_max_area_permille) {
        return false;
    }
    return short_side != 0 && long_side * QUALITY_SCALE <=
        short_side * config->box_max_aspect_permille;
}

RD_CONSTEXPR static void find_top_two(const float *scores, uint8_t *top1, uint8_t *top2)
{
    uint8_t category;
    *top1 = UINT8_MAX;
    *top2 = UINT8_MAX;
    for (category = 0; category < RECOGNITION_CLASS_COUNT; ++category) {
        if (*top1 == UINT8_MAX || scores[category] > scores[*top1]) {
            *top2 = *top1;
            *top1 = category;
        } else if (*top2 == UINT8_MAX || scores[category] > scores[*top2]) {
            *top2 = category;
        }
    }
}

RD_CONSTEXPR static recognition_decision_event_t lock_category(
    recognition_decision_state_t *state,
    uint8_t category,
    float score,
    const recognition_box_t *box,
    uint32_t now_ms,
    recognition_confidence_t confidence,
    recognition_lock_reason_t reason)
{
    state->locked_decision_elapsed_ms = state->session_started_ms == 0 ? 0 :
        elapsed_ms(now_ms, state->session_started_ms);
    state->state = RECOGNITION_DECISION_LOCKED;
    state->locked_category = category;
    state->locked_score = score;
    state->locked_box = *box;
    state->locked_confidence = confidence;
    state->lock_reason = reason;
    state->lock_started_ms = now_ms == 0 ? 1 : now_ms;
    ++state->completed_locks;
    ++state->frame_lock_counts[state->valid_inferences - 1];
    state->total_lock_inferences += state->valid_inferences;
    if (confidence > RECOGNITION_CONFIDENCE_NONE &&
        confidence < RECOGNITION_CONFIDENCE_COUNT) {
        ++state->confidence_lock_counts[confidence];
    }
    return RECOGNITION_EVENT_LOCKED;
}

RD_CONSTEXPR static recognition_confidence_t classify_weighted_confidence(
    const recognition_decision_config_t *config,
    uint8_t category,
    float score,
    float margin)
{
    if (score >= config->single_high_threshold &&
        score >= config->class_accept_thresholds[category] &&
        margin >= config->single_margin_threshold) {
        return RECOGNITION_CONFIDENCE_HIGH;
    }
    if (score >= config->multi_accept_evidence &&
        score >= config->class_accept_thresholds[category] &&
        margin >= config->multi_margin_evidence) {
        return RECOGNITION_CONFIDENCE_MEDIUM;
    }
    return RECOGNITION_CONFIDENCE_LOW;
}

RD_CONSTEXPR recognition_decision_event_t recognition_decision_observe(
    recognition_decision_state_t *state,
    const recognition_decision_config_t *config,
    const recognition_observation_t *observation)
{
    uint8_t frame_top1;
    uint8_t frame_top2;
    uint8_t evidence_top1;
    uint8_t evidence_top2;
    float quality_weight;
    float frame_margin;
    float evidence_margin;
    uint8_t maximum_inferences;
    size_t category;
    recognition_decision_event_t initial_event;

    if (state == NULL || config == NULL || observation == NULL) {
        return RECOGNITION_EVENT_NONE;
    }
    initial_event = recognition_decision_tick(state, config, observation->now_ms);
    if (state->state == RECOGNITION_DECISION_LOCKED ||
        state->state == RECOGNITION_DECISION_WEIGHING) {
        return initial_event;
    }
    if (observation->sequence == state->last_sequence) {
        return initial_event;
    }

    maximum_inferences = config->max_valid_inferences;
    if (maximum_inferences == 0 || maximum_inferences > RECOGNITION_MAX_EVIDENCE_FRAMES) {
        maximum_inferences = RECOGNITION_MAX_EVIDENCE_FRAMES;
    }
    if (state->valid_inferences >= maximum_inferences ||
        observation->quality_weight_permille == 0) {
        return initial_event;
    }

    find_top_two(observation->class_scores, &frame_top1, &frame_top2);
    if (frame_top1 == UINT8_MAX || observation->class_scores[frame_top1] <= 0.0f) {
        return recognition_decision_target_missing(state, config, observation->now_ms);
    }
    frame_margin = observation->class_scores[frame_top1] -
        (frame_top2 == UINT8_MAX ? 0.0f : observation->class_scores[frame_top2]);
    if (!box_reasonable(&observation->target_box, config)) {
        /* Skip this inference without restarting the active evidence window. */
        return initial_event;
    }

    if (state->has_target && !recognition_boxes_associate(
            observation->has_prediction ? &observation->predicted_box : &state->last_box,
            &observation->target_box, config)) {
        clear_evidence(state);
        state->state = RECOGNITION_DECISION_SEARCHING;
        return RECOGNITION_EVENT_EVIDENCE_RESET;
    }
    if (state->has_target && frame_top1 != state->candidate_category &&
        observation->class_scores[frame_top1] >= config->obvious_switch_threshold &&
        frame_margin >= config->obvious_switch_margin &&
        state->candidate_score >= config->multi_accept_evidence) {
        clear_evidence(state);
        state->state = RECOGNITION_DECISION_SEARCHING;
        return RECOGNITION_EVENT_EVIDENCE_RESET;
    }

    if (state->session_started_ms == 0) {
        state->session_started_ms = observation->now_ms == 0 ? 1 : observation->now_ms;
    }
    state->last_sequence = observation->sequence;
    state->last_seen_ms = observation->now_ms;
    state->last_box = observation->target_box;
    state->has_target = true;
    state->missing_valid_frames = 0;
    quality_weight = (observation->quality_weight_permille > QUALITY_SCALE ?
        QUALITY_SCALE : observation->quality_weight_permille) / (float)QUALITY_SCALE;
    {
        recognition_frame_evidence_t *frame = &state->frames[state->valid_inferences];
        frame->sequence = observation->sequence;
        frame->quality_weight_permille = observation->quality_weight_permille;
        frame->top1_category = frame_top1;
        frame->top2_category = frame_top2;
        frame->top1_score = observation->class_scores[frame_top1];
        frame->top2_score = frame_top2 == UINT8_MAX ? 0.0f :
            observation->class_scores[frame_top2];
        frame->top1_margin = frame_margin;
        frame->target_box = observation->target_box;
        frame->target_area = box_area(&observation->target_box);
        for (category = 0; category < RECOGNITION_CLASS_COUNT; ++category) {
            frame->class_scores[category] = observation->class_scores[category];
        }
    }
    state->total_quality_weight += quality_weight;
    for (category = 0; category < RECOGNITION_CLASS_COUNT; ++category) {
        state->evidence[category] += quality_weight * observation->class_scores[category];
        state->normalized_evidence[category] = state->total_quality_weight > 0.0f ?
            state->evidence[category] / state->total_quality_weight : 0.0f;
    }
    if (state->valid_inferences < UINT8_MAX) {
        ++state->valid_inferences;
    }
    state->last_round_valid_inferences = state->valid_inferences;
    state->state = RECOGNITION_DECISION_CONFIRMING;

    find_top_two(state->normalized_evidence, &evidence_top1, &evidence_top2);
    state->candidate_category = evidence_top1;
    state->candidate_second_category = evidence_top2;
    state->candidate_evidence = state->evidence[evidence_top1];
    state->candidate_score = state->normalized_evidence[evidence_top1];
    evidence_margin = state->normalized_evidence[evidence_top1] -
        (evidence_top2 == UINT8_MAX ? 0.0f : state->normalized_evidence[evidence_top2]);
    state->candidate_margin = evidence_margin;

    if (!observation->ambiguous && state->valid_inferences == 1 &&
        observation->quality_weight_permille >= QUALITY_SCALE * 85U / 100U &&
        observation->class_scores[frame_top1] >= config->single_high_threshold &&
        observation->class_scores[frame_top1] >= config->class_accept_thresholds[frame_top1] &&
        frame_margin >= config->single_margin_threshold) {
        return lock_category(state,
                             frame_top1,
                             observation->class_scores[frame_top1],
                             &observation->target_box,
                             observation->now_ms,
                             RECOGNITION_CONFIDENCE_HIGH,
                             RECOGNITION_LOCK_REASON_SINGLE_HIGH);
    }

    if (!observation->ambiguous && state->valid_inferences >= 2 &&
        state->frames[state->valid_inferences - 2].top1_category == frame_top1 &&
        frame_top1 == evidence_top1 &&
        state->normalized_evidence[evidence_top1] >= config->multi_accept_evidence &&
        state->normalized_evidence[evidence_top1] >=
            config->class_accept_thresholds[evidence_top1] &&
        evidence_margin >= config->multi_margin_evidence) {
        return lock_category(state,
                             evidence_top1,
                             state->normalized_evidence[evidence_top1],
                             &observation->target_box,
                             observation->now_ms,
                             classify_weighted_confidence(
                                 config,
                                 evidence_top1,
                                 state->normalized_evidence[evidence_top1],
                                 evidence_margin),
                             state->valid_inferences == 2 ?
                                 RECOGNITION_LOCK_REASON_TWO_FRAME_EVIDENCE :
                                 RECOGNITION_LOCK_REASON_THREE_FRAME_EVIDENCE);
    }

    if (state->valid_inferences >= maximum_inferences) {
        state->state = RECOGNITION_DECISION_UNCERTAIN;
    }
    return initial_event;
}
