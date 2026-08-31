#define RD_CONSTEXPR constexpr
#include "../main/recognition_decision.c"

#define RD_COMPILE_TIME_TEST 1
#define static constexpr static
#include "recognition_decision_test.c"
#undef static

static_assert(test_single_high_lock() == 0, "single high lock");
static_assert(test_blur_or_motion_not_counted() == 0, "blur and motion rejected");
static_assert(test_background_motion_outside_roi_is_ignored() == 0,
              "background motion outside presentation ROI is ignored");
static_assert(test_two_same_low_not_locked() == 0, "two same low");
static_assert(test_two_conflict_weighted_winner() == 0, "weighted conflict");
static_assert(test_three_different_forced_weighted_winner() == 0,
              "three different forced weighted winner");
static_assert(test_third_frame_stops_future_inference() == 0,
              "third frame is terminal");
static_assert(test_duplicate_frame_not_evidence() == 0, "duplicate frame");
static_assert(test_missing_and_switch_reset() == 0, "missing and switch reset");
static_assert(test_box_jump_resets_display() == 0, "box jump reset");
static_assert(test_locked_move_does_not_change() == 0, "locked move");
static_assert(test_weighing_complete_resets() == 0, "weighing complete");
static_assert(test_cancel_resets_locked_result() == 0, "cancel resets locked result");
static_assert(test_experience_starts_at_first_evidence() == 0,
              "experience starts at first accepted evidence");
static_assert(test_unrelated_motion_cannot_backdate_experience() == 0,
              "unrelated motion cannot backdate experience");
static_assert(test_experience_resets_only_when_explicitly_reset() == 0,
              "experience resets only explicitly");
