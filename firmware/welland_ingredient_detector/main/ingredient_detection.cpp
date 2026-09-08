#include "ingredient_detection.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <limits>
#include <new>

#include "dl_image_draw.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_image_process.hpp"
#include "dl_math.hpp"
#include "dl_model_base.hpp"
#include "esp_app_desc.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "psa/crypto.h"
#include "recognition_decision.h"
#include "sdkconfig.h"

extern const uint8_t ingredient_model[] asm("_binary_ingredient_model_espdl_start");
extern const uint8_t ingredient_model_end[] asm("_binary_ingredient_model_espdl_end");

namespace {
const char *TAG = "ingredient_detect";
std::atomic<bool> s_box_overlay_enabled{true};
uint32_t s_frame_log_counter = 0;
constexpr int INGREDIENT_SENSOR_CROP_SIZE = 480;
constexpr int MODEL_INTERNAL_MEMORY_BYTES = 96 * 1024;
const char *LABELS[] = {
    "apple",
    "banana",
    "strawberry",
    "lettuce",
    "egg",
    "orange",
    "eggplant",
    "cucumber",
    "carrot",
    "corn",
};

static_assert(sizeof(LABELS) / sizeof(LABELS[0]) == RECOGNITION_CLASS_COUNT);
static_assert(INGREDIENT_CLASS_COUNT == RECOGNITION_CLASS_COUNT);
static_assert((MODEL_INTERNAL_MEMORY_BYTES & 0x0f) == 0,
              "ESP-DL internal tensor memory must be 16-byte aligned");

void bytes_to_hex(const uint8_t *bytes, size_t size, char *hex)
{
    static constexpr char DIGITS[] = "0123456789abcdef";
    for (size_t index = 0; index < size; ++index) {
        hex[index * 2] = DIGITS[bytes[index] >> 4];
        hex[index * 2 + 1] = DIGITS[bytes[index] & 0x0f];
    }
    hex[size * 2] = '\0';
}

void validate_and_log_embedded_model(size_t embedded_model_bytes)
{
    if (embedded_model_bytes != WELLAND_MODEL_EXPECTED_BYTES) {
        ESP_LOGE(TAG,
                 "embedded model size mismatch: expected %zu, got %zu",
                 static_cast<size_t>(WELLAND_MODEL_EXPECTED_BYTES),
                 embedded_model_bytes);
        std::abort();
    }

    uint8_t model_sha256[32];
    size_t model_sha256_bytes = 0;
    const psa_status_t init_status = psa_crypto_init();
    const psa_status_t hash_status =
        init_status == PSA_SUCCESS
            ? psa_hash_compute(PSA_ALG_SHA_256,
                               ingredient_model,
                               embedded_model_bytes,
                               model_sha256,
                               sizeof(model_sha256),
                               &model_sha256_bytes)
            : init_status;
    if (hash_status != PSA_SUCCESS || model_sha256_bytes != sizeof(model_sha256)) {
        ESP_LOGE(TAG, "unable to hash embedded model: psa_status=%d", (int)hash_status);
        std::abort();
    }

    char model_sha256_hex[65];
    bytes_to_hex(model_sha256, sizeof(model_sha256), model_sha256_hex);
    if (std::strcmp(model_sha256_hex, WELLAND_MODEL_SHA256) != 0) {
        ESP_LOGE(TAG,
                 "embedded model SHA-256 mismatch: expected %.16s..., got %.16s...",
                 WELLAND_MODEL_SHA256,
                 model_sha256_hex);
        std::abort();
    }

    char firmware_sha256_prefix[17];
    esp_app_get_elf_sha256(firmware_sha256_prefix, sizeof(firmware_sha256_prefix));
    const esp_app_desc_t *app = esp_app_get_description();
    ESP_LOGI(TAG,
             "firmware=%s firmware_sha256=%.16s model=%s classes=%zu "
             "model_bytes=%zu model_sha256=%.16s",
             app->version,
             firmware_sha256_prefix,
             WELLAND_MODEL_VERSION,
             sizeof(LABELS) / sizeof(LABELS[0]),
             embedded_model_bytes,
             model_sha256_hex);
}

SemaphoreHandle_t s_model_mutex;
uint32_t s_sequence;
recognition_quality_state_t s_quality_state;
recognition_decision_state_t s_decision_state;
recognition_experience_state_t s_experience_state;
ingredient_detection_result_t s_locked_result;
ingredient_recognition_status_t s_recognition_status = {};

// Capture only reads this small snapshot, never waits for model inference.
SemaphoreHandle_t s_snapshot_mutex;
struct CaptureSnapshot {
    uint32_t epoch;
    bool has_target;
    bool locked;
    recognition_box_t box;
    int32_t motion_x;
    int32_t motion_y;
};
CaptureSnapshot s_capture_snapshot = {};
std::atomic<uint32_t> s_epoch{0};
uint32_t s_model_epoch = 0;
int32_t s_target_motion_x = 0, s_target_motion_y = 0;
uint64_t s_last_evidence_signature = 0;
uint16_t s_uncertain_sharpness = 0;

const recognition_quality_config_t QUALITY_CONFIG = {
    .motion_max_permille = CONFIG_WELLAND_QUALITY_MOTION_MAX_PERMILLE,
    .motion_exit_permille = CONFIG_WELLAND_QUALITY_MOTION_EXIT_PERMILLE,
    .stable_frames_required = CONFIG_WELLAND_QUALITY_STABLE_FRAMES_REQUIRED,
    .presentation_roi_percent = CONFIG_WELLAND_QUALITY_PRESENTATION_ROI_PERCENT,
    .target_roi_expand_percent = CONFIG_WELLAND_QUALITY_TARGET_ROI_EXPAND_PERCENT,
    .scene_change_pixel_delta = CONFIG_WELLAND_QUALITY_SCENE_CHANGE_PIXEL_DELTA,
    .scene_change_changed_permille =
        CONFIG_WELLAND_QUALITY_SCENE_CHANGE_CHANGED_PERMILLE,
    .scene_change_required_frames =
        CONFIG_WELLAND_QUALITY_SCENE_CHANGE_REQUIRED_FRAMES,
    .sharpness_min = CONFIG_WELLAND_QUALITY_SHARPNESS_MIN,
    .sharpness_good = CONFIG_WELLAND_QUALITY_SHARPNESS_GOOD,
    .exposure_mean_min = CONFIG_WELLAND_QUALITY_EXPOSURE_MEAN_MIN,
    .exposure_mean_max = CONFIG_WELLAND_QUALITY_EXPOSURE_MEAN_MAX,
    .dark_luma_max = CONFIG_WELLAND_QUALITY_DARK_LUMA_MAX,
    .bright_luma_min = CONFIG_WELLAND_QUALITY_BRIGHT_LUMA_MIN,
    .clipped_max_permille = CONFIG_WELLAND_QUALITY_CLIPPED_MAX_PERMILLE,
    .weight_floor_permille = CONFIG_WELLAND_QUALITY_WEIGHT_FLOOR_PERMILLE,
};

const recognition_decision_config_t DECISION_CONFIG = {
    .single_high_threshold = CONFIG_WELLAND_RECOGNITION_SINGLE_HIGH_PERCENT / 100.0f,
    .single_margin_threshold = CONFIG_WELLAND_RECOGNITION_SINGLE_MARGIN_PERCENT / 100.0f,
    .multi_accept_evidence = CONFIG_WELLAND_RECOGNITION_MULTI_EVIDENCE_PERCENT / 100.0f,
    .multi_margin_evidence = CONFIG_WELLAND_RECOGNITION_MULTI_MARGIN_PERCENT / 100.0f,
    .class_accept_thresholds = {
        CONFIG_WELLAND_RECOGNITION_CLASS_APPLE_PERCENT / 100.0f,
        CONFIG_WELLAND_RECOGNITION_CLASS_BANANA_PERCENT / 100.0f,
        CONFIG_WELLAND_RECOGNITION_CLASS_STRAWBERRY_PERCENT / 100.0f,
        CONFIG_WELLAND_RECOGNITION_CLASS_LETTUCE_PERCENT / 100.0f,
        CONFIG_WELLAND_RECOGNITION_CLASS_EGG_PERCENT / 100.0f,
        CONFIG_WELLAND_RECOGNITION_CLASS_ORANGE_PERCENT / 100.0f,
        CONFIG_WELLAND_RECOGNITION_CLASS_EGGPLANT_PERCENT / 100.0f,
        CONFIG_WELLAND_RECOGNITION_CLASS_CUCUMBER_PERCENT / 100.0f,
        CONFIG_WELLAND_RECOGNITION_CLASS_CARROT_PERCENT / 100.0f,
        CONFIG_WELLAND_RECOGNITION_CLASS_CORN_PERCENT / 100.0f,
    },
    .max_valid_inferences = CONFIG_WELLAND_RECOGNITION_MAX_VALID_INFERENCES,
    .target_missing_valid_frames = CONFIG_WELLAND_RECOGNITION_TARGET_MISSING_VALID_FRAMES,
    .box_min_area_permille = CONFIG_WELLAND_RECOGNITION_BOX_MIN_AREA_PERMILLE,
    .box_max_area_permille = CONFIG_WELLAND_RECOGNITION_BOX_MAX_AREA_PERMILLE,
    .box_min_side_pixels = CONFIG_WELLAND_RECOGNITION_BOX_MIN_SIDE_PIXELS,
    .box_max_aspect_permille = CONFIG_WELLAND_RECOGNITION_BOX_MAX_ASPECT_PERMILLE,
    .consistency_min_iou_permille = CONFIG_WELLAND_RECOGNITION_CONSISTENCY_IOU_PERMILLE,
    .consistency_max_center_shift_permille =
        CONFIG_WELLAND_RECOGNITION_CONSISTENCY_CENTER_SHIFT_PERMILLE,
    .consistency_min_area_ratio_permille =
        CONFIG_WELLAND_RECOGNITION_CONSISTENCY_AREA_MIN_PERMILLE,
    .consistency_max_area_ratio_permille =
        CONFIG_WELLAND_RECOGNITION_CONSISTENCY_AREA_MAX_PERMILLE,
    .obvious_switch_threshold =
        CONFIG_WELLAND_RECOGNITION_OBVIOUS_SWITCH_PERCENT / 100.0f,
    .obvious_switch_margin =
        CONFIG_WELLAND_RECOGNITION_OBVIOUS_SWITCH_MARGIN_PERCENT / 100.0f,
    .session_timeout_ms = CONFIG_WELLAND_RECOGNITION_SESSION_TIMEOUT_MS,
};

uint32_t now_ms()
{
    return static_cast<uint32_t>(esp_timer_get_time() / 1000);
}

bool recognition_is_locked()
{
    return s_decision_state.state == RECOGNITION_DECISION_LOCKED ||
           s_decision_state.state == RECOGNITION_DECISION_WEIGHING;
}

ingredient_recognition_state_t public_state()
{
    switch (s_decision_state.state) {
    case RECOGNITION_DECISION_UNCERTAIN:
        return INGREDIENT_RECOGNITION_UNCERTAIN;
    case RECOGNITION_DECISION_CONFIRMING:
        return INGREDIENT_RECOGNITION_CONFIRMING;
    case RECOGNITION_DECISION_LOCKED:
        return INGREDIENT_RECOGNITION_LOCKED;
    case RECOGNITION_DECISION_WEIGHING:
        return INGREDIENT_RECOGNITION_WEIGHING;
    default:
        return INGREDIENT_RECOGNITION_SEARCHING;
    }
}

uint32_t recognition_elapsed_ms(uint32_t current_ms)
{
    if (recognition_is_locked()) {
        return s_decision_state.locked_decision_elapsed_ms;
    }
    const uint32_t started = s_decision_state.session_started_ms;
    return started == 0 ? 0 : current_ms - started;
}

const char *confidence_name(recognition_confidence_t confidence)
{
    switch (confidence) {
    case RECOGNITION_CONFIDENCE_HIGH:
        return "high";
    case RECOGNITION_CONFIDENCE_MEDIUM:
        return "medium";
    case RECOGNITION_CONFIDENCE_LOW:
        return "low";
    default:
        return "none";
    }
}

const char *lock_reason_name(recognition_lock_reason_t reason)
{
    switch (reason) {
    case RECOGNITION_LOCK_REASON_SINGLE_HIGH:
        return "single_high";
    case RECOGNITION_LOCK_REASON_TWO_FRAME_EVIDENCE:
        return "two_frame_evidence";
    case RECOGNITION_LOCK_REASON_THREE_FRAME_EVIDENCE:
        return "three_frame_evidence";
    default:
        return "none";
    }
}

void log_quality_rejection(const recognition_quality_result_t &quality)
{
    ESP_LOGD(TAG,
             "decision rejected=quality quality={fresh:%d,stable:%d,roi_changed:%d,"
             "accepted:%d,scene_change:%d,changed:%u,weight:%u,motion:%u,"
             "sharpness:%u,luma:%u,dark:%u,bright:%u} "
             "scores=not_inferred evidence=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
             "normalized=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] actual_inferences=%u",
             quality.fresh,
             quality.stable,
             quality.roi_changed,
             quality.accepted,
             quality.scene_change,
             quality.changed_permille,
             quality.weight_permille,
             quality.motion_permille,
             quality.sharpness,
             quality.mean_luma,
             quality.dark_permille,
             quality.bright_permille,
             s_decision_state.evidence[0], s_decision_state.evidence[1],
             s_decision_state.evidence[2], s_decision_state.evidence[3],
             s_decision_state.evidence[4], s_decision_state.evidence[5],
             s_decision_state.evidence[6], s_decision_state.evidence[7],
             s_decision_state.evidence[8], s_decision_state.evidence[9],
             s_decision_state.normalized_evidence[0],
             s_decision_state.normalized_evidence[1],
             s_decision_state.normalized_evidence[2],
             s_decision_state.normalized_evidence[3],
             s_decision_state.normalized_evidence[4],
             s_decision_state.normalized_evidence[5],
             s_decision_state.normalized_evidence[6],
             s_decision_state.normalized_evidence[7],
             s_decision_state.normalized_evidence[8],
             s_decision_state.normalized_evidence[9],
             s_decision_state.valid_inferences);
}

void log_rejected_inference(const char *reason,
                            const recognition_quality_result_t &quality,
                            const recognition_observation_t &observation)
{
    uint8_t top1 = UINT8_MAX;
    uint8_t top2 = UINT8_MAX;
    for (uint8_t category = 0; category < RECOGNITION_CLASS_COUNT; ++category) {
        if (top1 == UINT8_MAX ||
            observation.class_scores[category] > observation.class_scores[top1]) {
            top2 = top1;
            top1 = category;
        } else if (top2 == UINT8_MAX ||
                   observation.class_scores[category] > observation.class_scores[top2]) {
            top2 = category;
        }
    }
    const float top1_score = top1 == UINT8_MAX ? 0.0f : observation.class_scores[top1];
    const float top2_score = top2 == UINT8_MAX ? 0.0f : observation.class_scores[top2];
    const uint32_t box_width = observation.target_box.right > observation.target_box.left ?
        static_cast<uint32_t>(observation.target_box.right - observation.target_box.left) : 0U;
    const uint32_t box_height = observation.target_box.bottom > observation.target_box.top ?
        static_cast<uint32_t>(observation.target_box.bottom - observation.target_box.top) : 0U;
    ESP_LOGD(TAG,
             "decision sequence=%" PRIu32 " rejected=%s "
             "quality={fresh:%d,stable:%d,roi_changed:%d,accepted:%d,weight:%u,"
             "motion:%u,sharpness:%u,luma:%u} "
             "scores=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
             "frame_top1=%d:%.3f frame_top2=%d:%.3f frame_margin=%.3f "
             "box=[%d,%d,%d,%d] area=%" PRIu32 " "
             "evidence=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
             "normalized=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
             "evidence_top1=%d evidence_top2=%d evidence_margin=%.3f "
             "lock_reason=%s confidence=%s actual_inferences=%u",
             observation.sequence,
             reason,
             quality.fresh,
             quality.stable,
             quality.roi_changed,
             quality.accepted,
             quality.weight_permille,
             quality.motion_permille,
             quality.sharpness,
             quality.mean_luma,
             observation.class_scores[0], observation.class_scores[1],
             observation.class_scores[2], observation.class_scores[3],
             observation.class_scores[4], observation.class_scores[5],
             observation.class_scores[6], observation.class_scores[7],
             observation.class_scores[8], observation.class_scores[9],
             top1 == UINT8_MAX ? -1 : top1,
             top1_score,
             top2 == UINT8_MAX ? -1 : top2,
             top2_score,
             top1_score - top2_score,
             observation.target_box.left,
             observation.target_box.top,
             observation.target_box.right,
             observation.target_box.bottom,
             box_width * box_height,
             s_decision_state.evidence[0], s_decision_state.evidence[1],
             s_decision_state.evidence[2], s_decision_state.evidence[3],
             s_decision_state.evidence[4], s_decision_state.evidence[5],
             s_decision_state.evidence[6], s_decision_state.evidence[7],
             s_decision_state.evidence[8], s_decision_state.evidence[9],
             s_decision_state.normalized_evidence[0],
             s_decision_state.normalized_evidence[1],
             s_decision_state.normalized_evidence[2],
             s_decision_state.normalized_evidence[3],
             s_decision_state.normalized_evidence[4],
             s_decision_state.normalized_evidence[5],
             s_decision_state.normalized_evidence[6],
             s_decision_state.normalized_evidence[7],
             s_decision_state.normalized_evidence[8],
             s_decision_state.normalized_evidence[9],
             s_decision_state.candidate_category == UINT8_MAX ? -1 :
                 s_decision_state.candidate_category,
             s_decision_state.candidate_second_category == UINT8_MAX ? -1 :
                 s_decision_state.candidate_second_category,
             s_decision_state.candidate_margin,
             lock_reason_name(s_decision_state.lock_reason),
             confidence_name(s_decision_state.locked_confidence),
             s_decision_state.valid_inferences);
}

void log_evidence_round(const recognition_quality_result_t &quality)
{
    if (s_decision_state.valid_inferences == 0) {
        return;
    }
    const recognition_frame_evidence_t &frame =
        s_decision_state.frames[s_decision_state.valid_inferences - 1U];
    ESP_LOGD(TAG,
             "decision round=%u sequence=%" PRIu32
             " quality={fresh:%d,stable:%d,roi_changed:%d,accepted:%d,weight:%u,"
             "motion:%u,sharpness:%u,luma:%u} "
             "scores=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
             "frame_top1=%u:%.3f frame_top2=%u:%.3f frame_margin=%.3f "
             "box=[%d,%d,%d,%d] area=%" PRIu32 " "
             "evidence=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
             "normalized=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f] "
             "evidence_top1=%u evidence_top2=%u evidence_margin=%.3f "
             "lock_reason=%s confidence=%s actual_inferences=%u",
             s_decision_state.valid_inferences,
             frame.sequence,
             quality.fresh,
             quality.stable,
             quality.roi_changed,
             quality.accepted,
             frame.quality_weight_permille,
             quality.motion_permille,
             quality.sharpness,
             quality.mean_luma,
             frame.class_scores[0], frame.class_scores[1], frame.class_scores[2],
             frame.class_scores[3], frame.class_scores[4], frame.class_scores[5],
             frame.class_scores[6], frame.class_scores[7], frame.class_scores[8],
             frame.class_scores[9],
             frame.top1_category,
             frame.top1_score,
             frame.top2_category,
             frame.top2_score,
             frame.top1_margin,
             frame.target_box.left,
             frame.target_box.top,
             frame.target_box.right,
             frame.target_box.bottom,
             frame.target_area,
             s_decision_state.evidence[0], s_decision_state.evidence[1],
             s_decision_state.evidence[2], s_decision_state.evidence[3],
             s_decision_state.evidence[4], s_decision_state.evidence[5],
             s_decision_state.evidence[6], s_decision_state.evidence[7],
             s_decision_state.evidence[8], s_decision_state.evidence[9],
             s_decision_state.normalized_evidence[0],
             s_decision_state.normalized_evidence[1],
             s_decision_state.normalized_evidence[2],
             s_decision_state.normalized_evidence[3],
             s_decision_state.normalized_evidence[4],
             s_decision_state.normalized_evidence[5],
             s_decision_state.normalized_evidence[6],
             s_decision_state.normalized_evidence[7],
             s_decision_state.normalized_evidence[8],
             s_decision_state.normalized_evidence[9],
             s_decision_state.candidate_category,
             s_decision_state.candidate_second_category,
             s_decision_state.candidate_margin,
             lock_reason_name(s_decision_state.lock_reason),
             confidence_name(s_decision_state.locked_confidence),
             s_decision_state.valid_inferences);
}

void sync_recognition_status(const recognition_quality_result_t *quality,
                             const ingredient_performance_t *performance,
                             uint32_t current_ms)
{
    xSemaphoreTake(s_snapshot_mutex, portMAX_DELAY);
    if (s_model_epoch != s_epoch.load()) {
        recognition_decision_cancel(&s_decision_state);
        recognition_experience_reset(&s_experience_state);
        s_locked_result = {};
        s_last_evidence_signature = 0;
        s_model_epoch = s_epoch.load();
    }
    s_capture_snapshot = {s_epoch.load(), s_decision_state.has_target, recognition_is_locked(),
                          s_decision_state.last_box, s_target_motion_x, s_target_motion_y};
    xSemaphoreGive(s_snapshot_mutex);
    s_recognition_status.state = public_state();
    s_recognition_status.inference_active = !recognition_is_locked() &&
        s_decision_state.state != RECOGNITION_DECISION_UNCERTAIN;
    s_recognition_status.candidate_category = s_decision_state.candidate_category;
    s_recognition_status.candidate_second_category =
        s_decision_state.candidate_second_category;
    s_recognition_status.collected_frames = s_decision_state.valid_inferences;
    s_recognition_status.candidate_votes = 0;
    s_recognition_status.window_frames = DECISION_CONFIG.max_valid_inferences;
    s_recognition_status.mean_score = s_decision_state.candidate_score;
    s_recognition_status.weighted_evidence = s_decision_state.candidate_evidence;
    s_recognition_status.evidence_margin = s_decision_state.candidate_margin;
    s_recognition_status.confidence_level = s_decision_state.locked_confidence;
    s_recognition_status.lock_reason = s_decision_state.lock_reason;
    s_recognition_status.effective_inferences = s_decision_state.valid_inferences;
    s_recognition_status.completed_locks = s_decision_state.completed_locks;
    s_recognition_status.high_confidence_locks =
        s_decision_state.confidence_lock_counts[RECOGNITION_CONFIDENCE_HIGH];
    s_recognition_status.medium_confidence_locks =
        s_decision_state.confidence_lock_counts[RECOGNITION_CONFIDENCE_MEDIUM];
    s_recognition_status.low_confidence_locks =
        s_decision_state.confidence_lock_counts[RECOGNITION_CONFIDENCE_LOW];
    s_recognition_status.average_lock_inferences =
        s_decision_state.completed_locks == 0 ? 0.0f :
            s_decision_state.total_lock_inferences /
                static_cast<float>(s_decision_state.completed_locks);
    for (size_t i = 0; i < 3; ++i) s_recognition_status.frame_lock_counts[i] = s_decision_state.frame_lock_counts[i];
    s_recognition_status.elapsed_ms = recognition_elapsed_ms(current_ms);
    s_recognition_status.motion_candidate_active = false;
    s_recognition_status.experience_started = s_experience_state.started;
    s_recognition_status.experience_locked = s_experience_state.locked;
    s_recognition_status.experience_elapsed_ms =
        recognition_experience_elapsed_ms(&s_experience_state, current_ms);
    if (quality != nullptr) {
        s_recognition_status.quality_weight_permille = quality->weight_permille;
        s_recognition_status.motion_permille = quality->motion_permille;
        s_recognition_status.sharpness = quality->sharpness;
        s_recognition_status.mean_luma = quality->mean_luma;
        s_recognition_status.dark_permille = quality->dark_permille;
        s_recognition_status.bright_permille = quality->bright_permille;
        s_recognition_status.changed_permille = quality->changed_permille;
        s_recognition_status.frame_fresh = quality->fresh;
        s_recognition_status.frame_quality_ok = quality->accepted;
        s_recognition_status.frame_stable = quality->stable;
        s_recognition_status.frame_roi_changed = quality->roi_changed;
        s_recognition_status.frame_scene_change = quality->scene_change;
    }
    if (performance != nullptr) {
        s_recognition_status.quality_us = performance->quality_us;
        s_recognition_status.preprocess_us = performance->preprocess_us;
        s_recognition_status.inference_us = performance->inference_us;
        s_recognition_status.postprocess_us = performance->postprocess_us;
        s_recognition_status.decision_us = performance->decision_us;
        s_recognition_status.total_us = performance->total_us;
    }
}

void reset_recognition_state()
{
    ++s_epoch;
    s_last_evidence_signature = 0;
    recognition_decision_init(&s_decision_state);
    recognition_experience_reset(&s_experience_state);
    s_locked_result = {};
    s_recognition_status = {};
    s_recognition_status.state = INGREDIENT_RECOGNITION_SEARCHING;
    s_recognition_status.inference_active = true;
    s_recognition_status.candidate_category = UINT8_MAX;
    s_recognition_status.candidate_second_category = UINT8_MAX;
    s_recognition_status.window_frames = CONFIG_WELLAND_RECOGNITION_MAX_VALID_INFERENCES;
    sync_recognition_status(nullptr, nullptr, now_ms());
}

bool validate_tensor_contract(const char *name,
                              dl::TensorBase *tensor,
                              dl::dtype_t expected_dtype,
                              const std::array<int, 4> &expected_shape)
{
    if (tensor == nullptr) {
        ESP_LOGE(TAG, "model tensor '%s' is missing", name);
        return false;
    }

    const bool shape_matches =
        tensor->shape.size() == expected_shape.size() &&
        std::equal(tensor->shape.begin(), tensor->shape.end(), expected_shape.begin());
    ESP_LOGI(TAG,
             "tensor=%s dtype=%s shape=%s exponent=%d",
             name,
             dl::dtype_to_string(tensor->dtype),
             dl::vector_to_string(tensor->shape).c_str(),
             tensor->exponent.get());
    if (tensor->dtype != expected_dtype || !shape_matches) {
        ESP_LOGE(TAG,
                 "model tensor '%s' contract mismatch: expected dtype=%s shape=[%d, %d, %d, %d]",
                 name,
                 dl::dtype_to_string(expected_dtype),
                 expected_shape[0],
                 expected_shape[1],
                 expected_shape[2],
                 expected_shape[3]);
        return false;
    }
    return true;
}

bool validate_model_contract(dl::Model *model)
{
    return validate_tensor_contract(
        "images",
        model->get_input("images"),
        dl::DATA_TYPE_INT16,
        {1, INGREDIENT_MODEL_HEIGHT, INGREDIENT_MODEL_WIDTH, 3}) &&
        validate_tensor_contract(
            "box0", model->get_output("box0"), dl::DATA_TYPE_INT8, {1, 28, 28, 4}) &&
        validate_tensor_contract(
            "score0", model->get_output("score0"), dl::DATA_TYPE_INT8, {1, 28, 28, INGREDIENT_CLASS_COUNT}) &&
        validate_tensor_contract(
            "box1", model->get_output("box1"), dl::DATA_TYPE_INT8, {1, 14, 14, 4}) &&
        validate_tensor_contract(
            "score1", model->get_output("score1"), dl::DATA_TYPE_INT8, {1, 14, 14, INGREDIENT_CLASS_COUNT}) &&
        validate_tensor_contract(
            "box2", model->get_output("box2"), dl::DATA_TYPE_INT8, {1, 7, 7, 4}) &&
        validate_tensor_contract(
            "score2", model->get_output("score2"), dl::DATA_TYPE_INT8, {1, 7, 7, INGREDIENT_CLASS_COUNT});
}

class IngredientDetector final {
public:
    IngredientDetector()
    {
        const size_t embedded_model_bytes =
            static_cast<size_t>(ingredient_model_end - ingredient_model);
        validate_and_log_embedded_model(embedded_model_bytes);

        m_model = new (std::nothrow) dl::Model(reinterpret_cast<const char *>(ingredient_model),
                                               fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                                               MODEL_INTERNAL_MEMORY_BYTES);
        if (m_model == nullptr) {
            ESP_LOGE(TAG, "model object allocation failed");
            return;
        }
        if (!validate_model_contract(m_model)) {
            delete m_model;
            m_model = nullptr;
            m_init_status = ESP_ERR_NO_MEM;
            return;
        }
        m_model->minimize();
        m_image_preprocessor = new (std::nothrow) dl::image::ImagePreprocessor(
            m_model,
            {0, 0, 0},
            {255, 255, 255});
        if (m_image_preprocessor == nullptr) {
            ESP_LOGE(TAG, "image preprocessor allocation failed");
            delete m_model;
            m_model = nullptr;
            return;
        }
        m_image_preprocessor->enable_letterbox({114, 114, 114});
        m_model_memory = m_model->get_memory_info().at("total");
        m_init_status = ESP_OK;
    }

    esp_err_t init_status() const { return m_init_status; }

    void run_profiled(
        const dl::image::img_t &image,
        ingredient_detection_result_t *output,
        ingredient_performance_t *performance,
        const recognition_box_t *target = nullptr, bool live = false)
    {
        const int64_t start_us = esp_timer_get_time();
        m_image_preprocessor->preprocess(image);
        const int64_t preprocess_end_us = esp_timer_get_time();
        m_model->run(dl::RUNTIME_MODE_AUTO);
        const int64_t inference_end_us = esp_timer_get_time();
        if (live) decode_target(output, target);
        else decode_top_categories(output);
        const int64_t postprocess_end_us = esp_timer_get_time();
        if (performance != nullptr) {
            performance->preprocess_us = (uint32_t)(preprocess_end_us - start_us);
            performance->inference_us = (uint32_t)(inference_end_us - preprocess_end_us);
            performance->postprocess_us = (uint32_t)(postprocess_end_us - inference_end_us);
            performance->model_internal_bytes = m_model_memory.internal;
            performance->model_psram_bytes = m_model_memory.psram;
            performance->model_flash_bytes = m_model_memory.flash;
        }
    }

    bool ambiguous() const { return m_ambiguous; }

private:
    bool m_ambiguous = false;
    dl::Model *m_model = nullptr;
    dl::image::ImagePreprocessor *m_image_preprocessor = nullptr;
    esp_err_t m_init_status = ESP_ERR_NO_MEM;
    struct Stage {
        const char *score_name;
        const char *box_name;
        int stride;
        int offset;
    };

    static void insert_sorted(const ingredient_detection_item_t &item,
                              ingredient_detection_result_t *output)
    {
        size_t position = 0;
        while (position < output->count &&
               output->items[position].score >= item.score) {
            ++position;
        }
        if (position >= INGREDIENT_MAX_DETECTIONS) {
            return;
        }
        const size_t new_count = std::min(output->count + 1,
                                          static_cast<size_t>(INGREDIENT_MAX_DETECTIONS));
        for (size_t index = new_count - 1; index > position; --index) {
            output->items[index] = output->items[index - 1];
        }
        output->items[position] = item;
        output->count = new_count;
    }

    void decode_stage(const Stage &stage,
                      std::array<ingredient_detection_item_t, RECOGNITION_CLASS_COUNT> *best,
                      std::array<bool, RECOGNITION_CLASS_COUNT> *found,
                      std::array<float, RECOGNITION_CLASS_COUNT> *best_logits)
    {
        dl::TensorBase *score = m_model->get_output(stage.score_name);
        dl::TensorBase *box = m_model->get_output(stage.box_name);
        const int height = score->shape[1];
        const int width = score->shape[2];
        const int channels = score->shape[3];
        const auto *score_ptr = static_cast<const int8_t *>(score->data);
        const auto *box_ptr = static_cast<const int8_t *>(box->data);
        const float score_scale = DL_SCALE(score->exponent);
        const float box_scale = DL_SCALE(box->exponent);
        const float inverse_scale_x = m_image_preprocessor->get_resize_scale_x(true);
        const float inverse_scale_y = m_image_preprocessor->get_resize_scale_y(true);
        const int border_left = m_image_preprocessor->get_border_left();
        const int border_top = m_image_preprocessor->get_border_top();

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int category = 0; category < channels; ++category, ++score_ptr) {
                    const float logit = dl::dequantize(*score_ptr, score_scale);
                    if (logit <= (*best_logits)[category]) {
                        continue;
                    }
                    ingredient_detection_item_t item = {};
                    item.category = static_cast<uint8_t>(category);
                    item.score = dl::math::sigmoid(logit);
                    const float left = dl::dequantize(box_ptr[0], box_scale);
                    const float top = dl::dequantize(box_ptr[1], box_scale);
                    const float right = dl::dequantize(box_ptr[2], box_scale);
                    const float bottom = dl::dequantize(box_ptr[3], box_scale);
                    const int center_x = x * stage.stride + stage.offset;
                    const int center_y = y * stage.stride + stage.offset;
                    item.box[0] = static_cast<int16_t>(std::clamp(
                        static_cast<int>(((center_x - left * stage.stride) - border_left) *
                                         inverse_scale_x),
                        0,
                        INGREDIENT_MODEL_WIDTH - 1));
                    item.box[1] = static_cast<int16_t>(std::clamp(
                        static_cast<int>(((center_y - top * stage.stride) - border_top) *
                                         inverse_scale_y),
                        0,
                        INGREDIENT_MODEL_HEIGHT - 1));
                    item.box[2] = static_cast<int16_t>(std::clamp(
                        static_cast<int>(((center_x + right * stage.stride) - border_left) *
                                         inverse_scale_x),
                        0,
                        INGREDIENT_MODEL_WIDTH - 1));
                    item.box[3] = static_cast<int16_t>(std::clamp(
                        static_cast<int>(((center_y + bottom * stage.stride) - border_top) *
                                         inverse_scale_y),
                        0,
                        INGREDIENT_MODEL_HEIGHT - 1));
                    (*best)[category] = item;
                    (*found)[category] = true;
                    (*best_logits)[category] = logit;
                }
                box_ptr += 4;
            }
        }
    }

    void decode_target(ingredient_detection_result_t *output, const recognition_box_t *target)
    {
        const char *scores[] = {"score0", "score1", "score2"};
        const char *boxes[] = {"box0", "box1", "box2"};
        const int strides[] = {8, 16, 32};
        float best_rank = -1.0f, winner_score = 0.0f;
        recognition_box_t winner = {};
        const int8_t *winner_scores = nullptr;
        float winner_scale = 0;
        m_ambiguous = false;
        // One anchor supplies all ten class scores. Never mix unrelated locations.
        // Second pass detects competing objects before allowing a quick lock.
        for (int pass = 0; pass < 2; ++pass) {
            for (int stage = 0; stage < 3; ++stage) {
                auto *score_tensor = m_model->get_output(scores[stage]);
                auto *box_tensor = m_model->get_output(boxes[stage]);
                const auto *sp = static_cast<const int8_t *>(score_tensor->data);
                const auto *bp = static_cast<const int8_t *>(box_tensor->data);
                const float ss = DL_SCALE(score_tensor->exponent), bs = DL_SCALE(box_tensor->exponent);
                for (int y = 0; y < score_tensor->shape[1]; ++y) {
                    for (int x = 0; x < score_tensor->shape[2]; ++x, sp += INGREDIENT_CLASS_COUNT, bp += 4) {
                        int8_t max_logit = sp[0];
                        for (int c = 1; c < INGREDIENT_CLASS_COUNT; ++c) max_logit = std::max(max_logit, sp[c]);
                        const float confidence = dl::math::sigmoid(dl::dequantize(max_logit, ss));
                        if (confidence < CONFIG_WELLAND_DETECTION_SCORE_PERCENT / 100.0f) continue;
                        const int stride = strides[stage], cx = x*stride+stride/2, cy = y*stride+stride/2;
                        recognition_box_t box = {
                            (int16_t)std::clamp<int>(cx-dl::dequantize(bp[0],bs)*stride,0,223),
                            (int16_t)std::clamp<int>(cy-dl::dequantize(bp[1],bs)*stride,0,223),
                            (int16_t)std::clamp<int>(cx+dl::dequantize(bp[2],bs)*stride,0,223),
                            (int16_t)std::clamp<int>(cy+dl::dequantize(bp[3],bs)*stride,0,223)};
                        const int w = box.right-box.left, h = box.bottom-box.top;
                        if (std::min(w,h) < DECISION_CONFIG.box_min_side_pixels ||
                            w*h*1000 < 224*224*DECISION_CONFIG.box_min_area_permille ||
                            w*h*1000 > 224*224*DECISION_CONFIG.box_max_area_permille ||
                            std::max(w,h)*1000 > std::min(w,h)*DECISION_CONFIG.box_max_aspect_permille) continue;
                        if (target && !recognition_boxes_associate(target, &box, &DECISION_CONFIG)) continue;
                        const int distance = std::abs(box.left+box.right-224)+std::abs(box.top+box.bottom-224);
                        const float rank = confidence - (target ? 0.0f : 0.10f*distance/448.0f);
                        if (pass == 0 && rank > best_rank) {
                            best_rank = rank; winner_score = confidence; winner = box;
                            winner_scores = sp; winner_scale = ss;
                        } else if (pass == 1 && winner_scores && confidence >= winner_score-0.05f &&
                            !recognition_boxes_associate(&winner, &box, &DECISION_CONFIG)) {
                            m_ambiguous = true;
                        }
                    }
                }
            }
        }
        output->count = 0;
        if (!winner_scores) return;
        for (int c = 0; c < INGREDIENT_CLASS_COUNT; ++c) {
            ingredient_detection_item_t item = {};
            item.category = c;
            item.score = dl::math::sigmoid(dl::dequantize(winner_scores[c], winner_scale));
            item.box[0] = winner.left; item.box[1] = winner.top;
            item.box[2] = winner.right; item.box[3] = winner.bottom;
            insert_sorted(item, output);
        }
    }

    void decode_top_categories(ingredient_detection_result_t *output)
    {
        static constexpr Stage STAGES[] = {
            {"score0", "box0", 8, 4},
            {"score1", "box1", 16, 8},
            {"score2", "box2", 32, 16},
        };
        std::array<ingredient_detection_item_t, RECOGNITION_CLASS_COUNT> best = {};
        std::array<bool, RECOGNITION_CLASS_COUNT> found = {};
        std::array<float, RECOGNITION_CLASS_COUNT> best_logits;
        best_logits.fill(-std::numeric_limits<float>::infinity());
        output->count = 0;
        for (const Stage &stage : STAGES) {
            decode_stage(stage, &best, &found, &best_logits);
        }
        for (size_t category = 0; category < found.size(); ++category) {
            if (found[category]) {
                insert_sorted(best[category], output);
            }
        }
    }

    dl::mem_info_t m_model_memory = {};
};

IngredientDetector &detector()
{
    static IngredientDetector instance;
    return instance;
}

} // namespace

extern "C" void ingredient_detection_set_box_overlay_enabled(bool enabled)
{
    s_box_overlay_enabled.store(enabled, std::memory_order_release);
}

extern "C" bool ingredient_detection_box_overlay_enabled(void)
{
    return s_box_overlay_enabled.load(std::memory_order_acquire);
}

extern "C" esp_err_t ingredient_detection_start(void)
{
    if (s_model_mutex != nullptr) {
        return ESP_OK;
    }
    IngredientDetector &model = detector();
    if (model.init_status() != ESP_OK) {
        return model.init_status();
    }
    s_snapshot_mutex = xSemaphoreCreateMutex();
    if (s_snapshot_mutex == nullptr) return ESP_ERR_NO_MEM;
    s_model_mutex = xSemaphoreCreateMutex();
    if (s_model_mutex == nullptr) {
        vSemaphoreDelete(s_snapshot_mutex);
        s_snapshot_mutex = nullptr;
        return ESP_ERR_NO_MEM;
    }
    reset_recognition_state();
    return ESP_OK;
}

extern "C" esp_err_t ingredient_detection_restart(void)
{
    if (s_model_mutex == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreTake(s_model_mutex, portMAX_DELAY);
    reset_recognition_state();
    xSemaphoreGive(s_model_mutex);
    ESP_LOGI(TAG, "recognition restarted");
    return ESP_OK;
}

extern "C" esp_err_t ingredient_detection_cancel(void)
{
    if (s_model_mutex == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreTake(s_model_mutex, portMAX_DELAY);
    recognition_decision_cancel(&s_decision_state);
    ++s_epoch;
    s_last_evidence_signature = 0;
    recognition_experience_reset(&s_experience_state);
    s_locked_result = {};
    sync_recognition_status(nullptr, nullptr, now_ms());
    xSemaphoreGive(s_model_mutex);
    ESP_LOGI(TAG, "recognition cancelled");
    return ESP_OK;
}

extern "C" esp_err_t ingredient_detection_start_weighing(void)
{
    if (s_model_mutex == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreTake(s_model_mutex, portMAX_DELAY);
    const bool started = recognition_decision_start_weighing(&s_decision_state);
    sync_recognition_status(nullptr, nullptr, now_ms());
    xSemaphoreGive(s_model_mutex);
    return started ? ESP_OK : ESP_ERR_INVALID_STATE;
}

extern "C" esp_err_t ingredient_detection_complete_weighing(void)
{
    if (s_model_mutex == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreTake(s_model_mutex, portMAX_DELAY);
    if (!recognition_is_locked()) {
        xSemaphoreGive(s_model_mutex);
        return ESP_ERR_INVALID_STATE;
    }
    recognition_decision_complete_weighing(&s_decision_state);
    ++s_epoch;
    s_last_evidence_signature = 0;
    recognition_experience_reset(&s_experience_state);
    s_locked_result = {};
    sync_recognition_status(nullptr, nullptr, now_ms());
    xSemaphoreGive(s_model_mutex);
    ESP_LOGI(TAG, "weighing completed; recognition session reset");
    return ESP_OK;
}

extern "C" esp_err_t ingredient_detection_get_recognition_status(
    ingredient_recognition_status_t *status)
{
    if (status == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    if (s_model_mutex == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreTake(s_model_mutex, portMAX_DELAY);
    const uint32_t current_ms = now_ms();
    if (recognition_decision_tick(&s_decision_state, &DECISION_CONFIG, current_ms) != RECOGNITION_EVENT_NONE) {
        ++s_epoch;
        recognition_experience_reset(&s_experience_state);
    }
    sync_recognition_status(nullptr, nullptr, current_ms);
    *status = s_recognition_status;
    xSemaphoreGive(s_model_mutex);
    return ESP_OK;
}

extern "C" void ingredient_detection_prepare(
    const uint8_t *source_rgb565_be, uint16_t source_width, uint16_t source_height,
    uint8_t *output_rgb565_be, uint32_t captured_ms, ingredient_frame_t *frame)
{
    static dl::image::ImageTransformer transformer;
    const int64_t frame_started_us = esp_timer_get_time();
    const uint16_t output_width = INGREDIENT_MODEL_WIDTH, output_height = INGREDIENT_MODEL_HEIGHT;

    const int crop_left = (source_width - INGREDIENT_SENSOR_CROP_SIZE) / 2;
    const int crop_top = (source_height - INGREDIENT_SENSOR_CROP_SIZE) / 2;

    dl::image::img_t source = {
        .data = const_cast<uint8_t *>(source_rgb565_be),
        .width = source_width,
        .height = source_height,
        .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE,
    };
    dl::image::img_t image = {
        .data = output_rgb565_be,
        .width = output_width,
        .height = output_height,
        .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE,
    };

    ESP_ERROR_CHECK(transformer.set_src_img(source)
                        .set_dst_img(image)
                        .set_src_img_crop_area({crop_left,
                                                crop_top,
                                                crop_left + INGREDIENT_SENSOR_CROP_SIZE,
                                                crop_top + INGREDIENT_SENSOR_CROP_SIZE})
                        .set_dst_img_border({})
                        .transform());


    static uint32_t capture_epoch = UINT32_MAX;
    static int32_t motion_x = 0, motion_y = 0;
    CaptureSnapshot snapshot;
    xSemaphoreTake(s_snapshot_mutex, portMAX_DELAY);
    snapshot = s_capture_snapshot;
    xSemaphoreGive(s_snapshot_mutex);
    if (capture_epoch != snapshot.epoch) {
        recognition_quality_reset(&s_quality_state);
        motion_x = motion_y = 0;
        capture_epoch = snapshot.epoch;
    }
    recognition_box_t predicted = snapshot.box;
    if (snapshot.has_target) {
        const int dx = motion_x - snapshot.motion_x, dy = motion_y - snapshot.motion_y;
        predicted.left += dx; predicted.right += dx;
        predicted.top += dy; predicted.bottom += dy;
    }
    const int pad_x = (predicted.right-predicted.left)*QUALITY_CONFIG.target_roi_expand_percent/100;
    const int pad_y = (predicted.bottom-predicted.top)*QUALITY_CONFIG.target_roi_expand_percent/100;
    const int margin = INGREDIENT_MODEL_WIDTH*(100-QUALITY_CONFIG.presentation_roi_percent)/200;
    recognition_quality_roi_t roi = snapshot.has_target ? recognition_quality_roi_t{
        (uint16_t)std::clamp<int>(predicted.left-pad_x, 0, 223),
        (uint16_t)std::clamp<int>(predicted.top-pad_y, 0, 223),
        (uint16_t)std::clamp<int>(predicted.right+pad_x, 1, 224),
        (uint16_t)std::clamp<int>(predicted.bottom+pad_y, 1, 224)} :
        recognition_quality_roi_t{(uint16_t)margin, (uint16_t)margin,
            (uint16_t)(224-margin), (uint16_t)(224-margin)};
    *frame = {};
    frame->captured_ms = captured_ms;
    frame->epoch = snapshot.epoch;
    frame->quality = recognition_quality_evaluate(&s_quality_state, output_rgb565_be,
        output_width, output_height, &QUALITY_CONFIG, &roi);
    if (snapshot.has_target && frame->quality.tracking_reliable) {
        motion_x += frame->quality.displacement_x;
        motion_y += frame->quality.displacement_y;
        predicted.left += frame->quality.displacement_x;
        predicted.right += frame->quality.displacement_x;
        predicted.top += frame->quality.displacement_y;
        predicted.bottom += frame->quality.displacement_y;
    }
    if (frame->quality.scene_change && !snapshot.locked) {
        xSemaphoreTake(s_snapshot_mutex, portMAX_DELAY);
        if (s_epoch.load() == snapshot.epoch) {
            s_capture_snapshot.epoch = ++s_epoch;
            s_capture_snapshot.has_target = false;
        }
        xSemaphoreGive(s_snapshot_mutex);
    }
    frame->motion_x = motion_x;
    frame->motion_y = motion_y;
    frame->has_target = snapshot.has_target;
    frame->predicted_box = predicted;
    frame->quality_us = (uint32_t)(esp_timer_get_time()-frame_started_us);
}

extern "C" ingredient_detection_result_t ingredient_detection_run(
    uint8_t *output_rgb565_be, const ingredient_frame_t *frame)
{
    const int64_t frame_started_us = esp_timer_get_time();
    dl::image::img_t image = {.data = output_rgb565_be,
        .width = INGREDIENT_MODEL_WIDTH, .height = INGREDIENT_MODEL_HEIGHT,
        .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE};
    ingredient_detection_result_t current = {};
    ingredient_performance_t performance = {};
    recognition_quality_result_t quality = frame->quality;
    bool inference_ran = false;
    bool locked = false;
    xSemaphoreTake(s_model_mutex, portMAX_DELAY);
    const uint32_t current_ms = now_ms();
    if (s_model_epoch != s_epoch.load()) sync_recognition_status(nullptr, nullptr, current_ms);
    if (frame->epoch != s_epoch || current_ms-frame->captured_ms > RECOGNITION_CANDIDATE_MAX_AGE_MS) {
        xSemaphoreGive(s_model_mutex);
        return current;
    }
    // While awaiting a new view, a tracked, clear target is still present even
    // though no additional model evidence is requested.
    if (s_decision_state.state == RECOGNITION_DECISION_UNCERTAIN && frame->has_target &&
        quality.stable && !quality.scene_change &&
        quality.sharpness >= QUALITY_CONFIG.sharpness_min &&
        quality.mean_luma >= QUALITY_CONFIG.exposure_mean_min &&
        quality.mean_luma <= QUALITY_CONFIG.exposure_mean_max &&
        quality.dark_permille + quality.bright_permille <= QUALITY_CONFIG.clipped_max_permille) {
        s_decision_state.last_seen_ms = current_ms;
    }
    recognition_decision_event_t event =
        recognition_decision_tick(&s_decision_state, &DECISION_CONFIG, current_ms);

    if (recognition_is_locked()) {
        current = s_locked_result;
        current.elapsed_ms = 0;
    } else {
        performance.quality_us = frame->quality_us;
        if (event == RECOGNITION_EVENT_SESSION_TIMEOUT) {
            ++s_epoch;
            recognition_experience_reset(&s_experience_state);
            quality.accepted = false;
        }
        if (s_decision_state.state == RECOGNITION_DECISION_UNCERTAIN) {
            // Only a changed view or substantially improved sharpness starts a new attempt.
            if (quality.accepted && (quality.raw_motion_permille >= 20 ||
                quality.sharpness > s_uncertain_sharpness * 12U / 10U)) {
                recognition_decision_cancel(&s_decision_state);
                ++s_epoch;
                s_last_evidence_signature = 0;
                recognition_experience_reset(&s_experience_state);
            }
            quality.accepted = false;
        }
        if (quality.signature == s_last_evidence_signature) quality.accepted = false;
        if (!quality.accepted) {
            const int64_t decision_started_us = esp_timer_get_time();
            event = recognition_decision_bad_frame(&s_decision_state,
                                                   &DECISION_CONFIG,
                                                   current_ms,
                                                   quality.scene_change);
            performance.decision_us = static_cast<uint32_t>(
                esp_timer_get_time() - decision_started_us);
            log_quality_rejection(quality);
        } else {
            if (s_decision_state.session_started_ms == 0) {
                s_decision_state.session_started_ms = current_ms == 0 ? 1 : current_ms;
            }
            IngredientDetector &model = detector();
            model.run_profiled(image, &current, &performance, frame->has_target ? &frame->predicted_box : nullptr, true);
            if (frame->epoch != s_epoch.load()) {
                sync_recognition_status(nullptr, nullptr, now_ms());
                xSemaphoreGive(s_model_mutex);
                return {};
            }
            current.sequence = ++s_sequence;
            current.elapsed_ms =
                (performance.preprocess_us + performance.inference_us +
                 performance.postprocess_us + 999) / 1000;
            inference_ran = true;

            const int64_t decision_started_us = esp_timer_get_time();
            const uint32_t decision_now_ms = now_ms();
            recognition_observation_t observation = {};
            observation.sequence = current.sequence;
            observation.now_ms = decision_now_ms;
            observation.quality_weight_permille = quality.weight_permille;
            observation.has_prediction = frame->has_target;
            observation.predicted_box = frame->predicted_box;
            observation.ambiguous = model.ambiguous();
            if (current.count > 0) {
                observation.target_box.left = current.items[0].box[0];
                observation.target_box.top = current.items[0].box[1];
                observation.target_box.right = current.items[0].box[2];
                observation.target_box.bottom = current.items[0].box[3];
            }
            for (size_t index = 0; index < current.count; ++index) {
                const ingredient_detection_item_t &item = current.items[index];
                observation.class_scores[item.category] = item.score;
            }
            if (current.count == 0 ||
                current.items[0].score <
                    CONFIG_WELLAND_DETECTION_SCORE_PERCENT / 100.0f) {
                event = recognition_decision_target_missing(&s_decision_state,
                                                            &DECISION_CONFIG,
                                                            decision_now_ms);
                log_rejected_inference("target_missing", quality, observation);
            } else {
                event = recognition_decision_observe(&s_decision_state,
                                                     &DECISION_CONFIG,
                                                     &observation);
                if (s_decision_state.last_sequence == observation.sequence) {
                    s_last_evidence_signature = quality.signature;
                    s_target_motion_x = frame->motion_x;
                    s_target_motion_y = frame->motion_y;
                    s_uncertain_sharpness = quality.sharpness;
                    recognition_experience_confirm_target(&s_experience_state,
                                                          frame->captured_ms);
                    log_evidence_round(quality);
                } else {
                    log_rejected_inference("target_box", quality, observation);
                }
            }
            performance.decision_us = static_cast<uint32_t>(
                esp_timer_get_time() - decision_started_us);
        }

        if (event == RECOGNITION_EVENT_EVIDENCE_RESET ||
            event == RECOGNITION_EVENT_SESSION_TIMEOUT) {
            if (s_model_epoch == s_epoch.load()) ++s_epoch;
            s_last_evidence_signature = 0;
            recognition_experience_reset(&s_experience_state);
            ESP_LOGI(TAG, "recognition evidence reset: event=%d", static_cast<int>(event));
        }
        if (recognition_is_locked()) {
            recognition_experience_lock(&s_experience_state, now_ms());
            s_locked_result = {};
            s_locked_result.count = 1;
            s_locked_result.elapsed_ms = current.elapsed_ms;
            s_locked_result.sequence = current.sequence;
            ingredient_detection_item_t &locked_item = s_locked_result.items[0];
            locked_item.category = s_decision_state.locked_category;
            locked_item.score = s_decision_state.locked_score;
            locked_item.box[0] = s_decision_state.locked_box.left;
            locked_item.box[1] = s_decision_state.locked_box.top;
            locked_item.box[2] = s_decision_state.locked_box.right;
            locked_item.box[3] = s_decision_state.locked_box.bottom;
            current = s_locked_result;
            ESP_LOGI(TAG,
                     "locked %s score=%.3f evidence=%.3f reason=%s confidence=%s "
                     "valid_inferences=%u elapsed=%" PRIu32 " ms",
                     LABELS[locked_item.category],
                     locked_item.score,
                     s_decision_state.candidate_evidence,
                     lock_reason_name(s_decision_state.lock_reason),
                     confidence_name(s_decision_state.locked_confidence),
                     s_decision_state.valid_inferences,
                     recognition_elapsed_ms(current_ms));
        }
    }
    locked = recognition_is_locked();
    performance.effective_inferences = s_decision_state.valid_inferences;
    performance.total_us = static_cast<uint32_t>(esp_timer_get_time() - frame_started_us);
    sync_recognition_status(&quality, &performance, now_ms());
    if (frame->epoch != s_epoch.load()) { current = {}; locked = false; }
    xSemaphoreGive(s_model_mutex);

    for (size_t index = 0;
         s_box_overlay_enabled.load(std::memory_order_acquire) &&
             index < std::min(current.count, static_cast<size_t>(1));
         ++index) {
        const ingredient_detection_item_t &item = current.items[index];
        if (item.box[2] > item.box[0] && item.box[3] > item.box[1]) {
            if (locked) {
                dl::image::draw_hollow_rectangle(image,
                                                 item.box[0],
                                                 item.box[1],
                                                 item.box[2],
                                                 item.box[3],
                                                 {0x07, 0xE0},
                                                 2);
            } else {
                dl::image::draw_hollow_rectangle(image,
                                                 item.box[0],
                                                 item.box[1],
                                                 item.box[2],
                                                 item.box[3],
                                                 {0xF8, 0x00},
                                                 2);
            }
        }
    }
    const bool log_frame = (++s_frame_log_counter % 10U) == 1U;
    if (inference_ran && log_frame) {
        ESP_LOGI(TAG,
                 "frame=%" PRIu32 " top_k=%zu quality=%" PRIu32 "us preprocess=%" PRIu32
                 "us inference=%" PRIu32 "us postprocess=%" PRIu32 "us decision=%" PRIu32
                 "us total=%" PRIu32 "us valid_inferences=%u motion=%u sharpness=%u luma=%u",
                 current.sequence,
                 current.count,
                 performance.quality_us,
                 performance.preprocess_us,
                 performance.inference_us,
                 performance.postprocess_us,
                 performance.decision_us,
                 performance.total_us,
                 performance.effective_inferences,
                 quality.motion_permille,
                 quality.sharpness,
                 quality.mean_luma);
    } else if (!locked && log_frame) {
        ESP_LOGI(TAG,
                 "quality_gate fresh=%d stable=%d roi_changed=%d accepted=%d "
                 "scene_change=%d changed=%u motion=%u sharpness=%u luma=%u "
                 "clipped=%u/%u quality=%" PRIu32 "us decision=%" PRIu32
                 "us total=%" PRIu32 "us valid_inferences=%u",
                 quality.fresh,
                 quality.stable,
                 quality.roi_changed,
                 quality.accepted,
                 quality.scene_change,
                 quality.changed_permille,
                 quality.motion_permille,
                 quality.sharpness,
                 quality.mean_luma,
                 quality.dark_permille,
                 quality.bright_permille,
                 performance.quality_us,
                 performance.decision_us,
                 performance.total_us,
                 performance.effective_inferences);
    }
    return current;
}

extern "C" esp_err_t ingredient_detection_validate_rgb565_be(
    const uint8_t *input_rgb565_be,
    uint16_t width,
    uint16_t height,
    ingredient_detection_result_t *result,
    ingredient_performance_t *performance)
{
    if (input_rgb565_be == nullptr || result == nullptr || performance == nullptr ||
        width != INGREDIENT_MODEL_WIDTH || height != INGREDIENT_MODEL_HEIGHT ||
        (reinterpret_cast<uintptr_t>(input_rgb565_be) & 0x0fU) != 0 ||
        s_model_mutex == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }

    *result = {};
    *performance = {};
    dl::image::img_t image = {
        .data = const_cast<uint8_t *>(input_rgb565_be),
        .width = width,
        .height = height,
        .pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE,
    };

    xSemaphoreTake(s_model_mutex, portMAX_DELAY);
    IngredientDetector &model = detector();
    const uint32_t internal_caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
    const uint32_t psram_caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
    performance->internal_free_before = heap_caps_get_free_size(internal_caps);
    performance->internal_largest_before = heap_caps_get_largest_free_block(internal_caps);
    performance->psram_free_before = heap_caps_get_free_size(psram_caps);
    performance->psram_largest_before = heap_caps_get_largest_free_block(psram_caps);

    const esp_err_t monitor_status = heap_caps_monitor_local_minimum_free_size_start();
    model.run_profiled(image, result, performance);
    result->count = result->count > 0 &&
                            result->items[0].score >=
                                CONFIG_WELLAND_DETECTION_SCORE_PERCENT / 100.0f ?
                        1 : 0;
    performance->internal_min_free = heap_caps_get_minimum_free_size(internal_caps);
    performance->psram_min_free = heap_caps_get_minimum_free_size(psram_caps);
    if (monitor_status == ESP_OK) {
        ESP_ERROR_CHECK_WITHOUT_ABORT(heap_caps_monitor_local_minimum_free_size_stop());
    }
    performance->internal_free_after = heap_caps_get_free_size(internal_caps);
    performance->internal_largest_after = heap_caps_get_largest_free_block(internal_caps);
    performance->psram_free_after = heap_caps_get_free_size(psram_caps);
    performance->psram_largest_after = heap_caps_get_largest_free_block(psram_caps);

    performance->total_us = performance->preprocess_us + performance->inference_us +
                            performance->postprocess_us;
    result->elapsed_ms = (performance->total_us + 999) / 1000;
    xSemaphoreGive(s_model_mutex);
    return ESP_OK;
}
