import pytest

from tbots.skills import GoToPoint, build_skill, skill_names


def test_builtin_skill_registry_round_trips_go_to_point():
    assert "go_to_point" in skill_names()
    skill = build_skill("go_to_point", target=(1.0, -0.5))
    assert isinstance(skill, GoToPoint)


def test_unknown_skill_has_actionable_error():
    with pytest.raises(KeyError, match="unknown skill"):
        build_skill("does_not_exist")


def test_go_to_point_avoidance_kwargs_round_trip_and_keep_spec_equality():
    from tbots.tactics.base import SkillSpec

    skill = build_skill("go_to_point", target=(1.0, -0.5), avoid=True, avoid_ball=True,
                        predictor="constant_acceleration")
    assert isinstance(skill, GoToPoint)
    kwargs = {"target": (1.0, -0.5), "avoid": True, "predictor": "constant_velocity"}
    assert SkillSpec("go_to_point", dict(kwargs)) == SkillSpec("go_to_point", dict(kwargs))


def test_go_to_point_rejects_an_unknown_predictor():
    with pytest.raises(ValueError, match="predictor"):
        build_skill("go_to_point", target=(0.0, 0.0), predictor="crystal_ball")
