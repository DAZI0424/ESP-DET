#define RD_CONSTEXPR constexpr
#include "../main/recognition_decision.c"

#define RD_COMPILE_TIME_TEST 1
#define static constexpr static
#include "recognition_decision_test.c"
#undef static

static_assert(test_single_high_locks_first_frame() == 0,
              "single high locks first frame");
static_assert(test_blur_or_motion_not_counted() == 0, "blur and motion rejected");
static_assert(test_background_motion_outside_roi_is_ignored() == 0,
              "background motion outside presentation ROI is ignored");
static_assert(test_two_same_low_not_locked() == 0, "two same low");
static_assert(test_two_same_high_locks_second_frame() == 0,
              "two same high locks second frame");
static_assert(test_conflicting_latest_frame_is_uncertain() == 0, "conflict requires new view");
static_assert(test_three_different_are_uncertain() == 0,
              "three different stay uncertain");
static_assert(test_uncertain_stops_evidence_until_reset() == 0,
              "uncertain does not accumulate more evidence");
static_assert(test_duplicate_frame_not_evidence() == 0, "duplicate frame");
static_assert(test_brief_missing_preserves_but_switch_resets() == 0,
              "brief occlusion preserves, scene switch resets");
static_assert(test_box_jump_resets_evidence() == 0,
              "box jump clears evidence");
static_assert(test_locked_move_does_not_change() == 0, "locked move");
static_assert(test_weighing_complete_resets() == 0, "weighing complete");
static_assert(test_cancel_resets_locked_result() == 0, "cancel resets locked result");
static_assert(test_experience_starts_at_first_evidence() == 0,
              "experience starts at first accepted evidence");
static_assert(test_unrelated_motion_cannot_backdate_experience() == 0,
              "unrelated motion cannot backdate experience");
static_assert(test_experience_resets_only_when_explicitly_reset() == 0,
              "experience resets only explicitly");

static_assert(test_roi_switch_rechecks_history() == 0, "ROI history, motion and duplicate checks");

static_assert(test_third_frame_can_lock_with_evidence() == 0, "test_third_frame_can_lock_with_evidence");
static_assert(test_ambiguous_objects_do_not_lock() == 0, "test_ambiguous_objects_do_not_lock");
static_assert(test_prediction_and_switch() == 0, "test_prediction_and_switch");
static_assert(test_candidate_policy() == 0, "test_candidate_policy");
static_assert(test_motion_compensation() == 0, "test_motion_compensation");

static_assert(test_association_rejects_background() == 0, "unrelated boxes excluded");
static_assert(test_fast_motion_not_compensated() == 0, "large motion rejected");
