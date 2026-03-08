"""Tests for curriculum learning manager."""
import pytest

from training.curriculum import CurriculumManager, WINDOW_SIZE


def test_starts_at_phase_0():
    cm = CurriculumManager()
    assert cm.phase_name == "single_ops"
    assert cm.current_phase_idx == 0


def test_get_problem_returns_dict():
    cm = CurriculumManager()
    p = cm.get_problem()
    assert isinstance(p, dict)
    assert "prompt" in p


def test_status_returns_correct_structure():
    cm = CurriculumManager()
    s = cm.status()
    assert s["phase"] == "single_ops"
    assert s["phase_idx"] == 0
    assert s["total_phases"] == 4
    assert s["window_size"] == 0


def test_record_reward_returns_none_when_window_not_full():
    cm = CurriculumManager()
    result = cm.record_reward(1.0)
    assert result is None
    assert len(cm.reward_history) == 1


def test_promotion():
    """6/10 rewards >= target (1.0) → promoted."""
    cm = CurriculumManager()
    for _ in range(6):
        cm.record_reward(1.0)  # hits target
    for _ in range(3):
        cm.record_reward(-1.0)  # misses target
    # 9 rewards, window not full yet
    assert cm.phase_name == "single_ops"
    result = cm.record_reward(1.0)  # 7/10 >= 1.0, > 50% → promoted
    assert result == "promoted"
    assert cm.phase_name == "fusion_2op"


def test_demotion():
    """<20% positive rewards → demoted."""
    cm = CurriculumManager()
    # Start at phase 1 so we can demote
    cm.current_phase_idx = 1
    for _ in range(9):
        cm.record_reward(-1.0)  # negative
    result = cm.record_reward(-1.0)  # 0/10 positive, < 20% → demoted
    assert result == "demoted"
    assert cm.phase_name == "single_ops"


def test_cannot_promote_past_last_phase():
    cm = CurriculumManager()
    cm.current_phase_idx = len(cm.phases) - 1  # last phase
    for _ in range(WINDOW_SIZE):
        result = cm.record_reward(3.0)
    assert result is None  # cannot promote further


def test_cannot_demote_past_first_phase():
    cm = CurriculumManager()
    cm.current_phase_idx = 0
    for _ in range(WINDOW_SIZE):
        result = cm.record_reward(-1.0)
    assert result is None  # cannot demote below 0


def test_phase_history_tracks_transitions():
    cm = CurriculumManager()
    for _ in range(WINDOW_SIZE):
        cm.record_reward(1.0)
    assert len(cm.phase_history) == 1
    assert cm.phase_history[0]["action"] == "promoted"
    assert cm.phase_history[0]["from"] == "single_ops"
    assert cm.phase_history[0]["to"] == "fusion_2op"


def test_reward_history_clears_on_transition():
    cm = CurriculumManager()
    for _ in range(WINDOW_SIZE):
        cm.record_reward(1.0)
    # After promotion, history should be cleared
    assert len(cm.reward_history) == 0


def test_add_problems():
    cm = CurriculumManager()
    initial_count = len(cm.phases[0].problems)
    cm.add_problems("single_ops", [{"prompt": "test", "ops": ["test"], "difficulty": 1}])
    assert len(cm.phases[0].problems) == initial_count + 1


def test_add_problems_unknown_phase():
    cm = CurriculumManager()
    with pytest.raises(ValueError, match="Unknown phase"):
        cm.add_problems("nonexistent_phase", [{"prompt": "test"}])
