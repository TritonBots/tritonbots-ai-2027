from __future__ import annotations

from dataclasses import dataclass

import pytest

from tbots.core.command import RobotCommand
from tbots.core.gamestate import GameState, Play
from tbots.core.geometry import DIV_B
from tbots.core.state import BallState, RobotState, WorldState
from tbots.skills.base import SkillStatus, register_skill
from tbots.tactics.base import Assignment, SkillSpec, Tactic
from tbots.tactics.coach import Coach
from tbots.tactics.scripted import ScriptedTactic


@register_skill("test_constant_command")
class ConstantCommand:
    instances: list[ConstantCommand] = []

    def __init__(self, vx: float = 0.0, vy: float = 0.0, done: bool = False) -> None:
        self.vx = vx
        self.vy = vy
        self.done = done
        self.reset_calls = 0
        self.step_calls = 0
        self.instances.append(self)

    def reset(self, world: WorldState, robot_id: int) -> None:
        self.reset_calls += 1

    def step(self, world: WorldState, robot_id: int) -> RobotCommand:
        self.step_calls += 1
        return RobotCommand(robot_id, vx=self.vx, vy=self.vy)

    def status(self) -> SkillStatus:
        return "success" if self.done else "running"


@dataclass
class FixedTactic(Tactic):
    assignments: list[Assignment]
    calls: int = 0

    def decide(self, world: WorldState) -> list[Assignment]:
        self.calls += 1
        return self.assignments


def world(*, t: float = 0.0, game: GameState | None = None,
          ball: tuple[float, float] = (0.0, 0.0),
          robots: dict[int, tuple[float, float]] | None = None) -> WorldState:
    robots = robots or {0: (-1.0, 0.0), 1: (-2.0, 0.5)}
    return WorldState(
        t=t,
        ball=BallState(*ball),
        us={robot_id: RobotState(robot_id, *pos, 0.0) for robot_id, pos in robots.items()},
        game=game or GameState(play=Play.RUN, can_move=True, min_ball_distance=0.0),
    )


@pytest.fixture(autouse=True)
def clear_skill_instances():
    ConstantCommand.instances.clear()


def test_scripted_tactic_is_deterministic_and_assigns_present_robots():
    tactic = ScriptedTactic(DIV_B)
    state = world(robots={0: (-4.0, 0.0), 2: (-1.0, 0.0), 7: (-2.0, 1.0)})
    first = tactic.decide(state)
    assert tactic.decide(state) == first
    assert {assignment.robot_id for assignment in first} == {0, 2, 7}
    assert all(assignment.skill.name == "go_to_point" for assignment in first)
    keeper = next(assignment for assignment in first if assignment.robot_id == 0)
    assert keeper.skill.kwargs["target"][0] < -4.0


def test_coach_decides_at_4hz_and_preserves_equal_skill_specs():
    spec = SkillSpec("test_constant_command", {"vx": 1.0})
    tactic = FixedTactic([Assignment(0, spec)])
    coach = Coach(tactic, DIV_B)

    assert coach.tick(world(t=0.0))[0].vx == 1.0
    assert coach.tick(world(t=0.10))[0].vx == 1.0
    assert coach.tick(world(t=0.24))[0].vx == 1.0
    assert tactic.calls == 1
    assert len(ConstantCommand.instances) == 1
    assert ConstantCommand.instances[0].reset_calls == 1

    coach.tick(world(t=0.25))
    assert tactic.calls == 2
    assert len(ConstantCommand.instances) == 1

    tactic.assignments = [Assignment(0, SkillSpec("test_constant_command", {"vx": 2.0}))]
    assert coach.tick(world(t=0.50))[0].vx == 2.0
    assert len(ConstantCommand.instances) == 2
    assert ConstantCommand.instances[1].reset_calls == 1


def test_finished_skill_stops_until_next_decision():
    tactic = FixedTactic([Assignment(0, SkillSpec("test_constant_command", {"done": True}))])
    coach = Coach(tactic, DIV_B)
    command = coach.tick(world())
    assert command[0] == RobotCommand(0)


@pytest.mark.parametrize("play", list(Play))
def test_coach_applies_safe_commands_for_every_play(play: Play):
    game = GameState(
        play=play,
        can_move=play not in (Play.HALT, Play.TIMEOUT),
        min_ball_distance=0.0,
    )
    coach = Coach(FixedTactic([Assignment(0, SkillSpec("test_constant_command", {"vx": 3.0}))]),
                  DIV_B, stop_speed=0.2)
    command = coach.tick(world(game=game))[0]
    if play in (Play.HALT, Play.TIMEOUT):
        assert command == RobotCommand(0)
    elif play is Play.STOP:
        assert command.vx == pytest.approx(0.2)
    else:
        assert command.vx == 3.0


def test_coach_blocks_ball_and_defense_area_violations():
    forward = FixedTactic([Assignment(0, SkillSpec("test_constant_command", {"vx": 4.0}))])
    running = GameState(play=Play.RUN, can_move=True, min_ball_distance=0.5)
    ball_coach = Coach(forward, DIV_B)
    assert ball_coach.tick(
        world(game=running, ball=(0.0, 0.0), robots={0: (-0.51, 0.0)})
    )[0] == RobotCommand(0)

    area_game = GameState(
        play=Play.RUN,
        can_move=True,
        min_ball_distance=0.0,
        our_goalkeeper=1,
    )
    own_area_coach = Coach(forward, DIV_B)
    assert own_area_coach.tick(world(game=area_game, robots={0: (-3.6, 0.0)}))[0] == RobotCommand(0)

    stopped = GameState(play=Play.STOP, can_move=True, min_ball_distance=0.0)
    their_area_coach = Coach(forward, DIV_B)
    assert their_area_coach.tick(world(game=stopped, robots={0: (3.35, 0.0)}))[0] == RobotCommand(0)


def test_coach_rejects_bad_tactic_assignments():
    coach = Coach(FixedTactic([Assignment(9, SkillSpec("test_constant_command", {}))]), DIV_B)
    with pytest.raises(ValueError, match="unknown robot"):
        coach.tick(world())


# -- TASK-086: the roles drive through the MPC planner ---------------------------

class CountingCoach(Coach):
    """Counts the commands the rule filter had to replace with a stop."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stopped_by_filter = 0

    def _filter(self, world: WorldState, command: RobotCommand) -> RobotCommand:
        out = super()._filter(world, command)
        if out == RobotCommand(command.robot_id) and command != out:
            self.stopped_by_filter += 1
        return out


def _scripted_6v6(game: GameState, seconds: float = 10.0, avoid: bool = True):
    from tbots.backends.base import Scenario
    from tests.kinematic import KinematicWorld, Opponent

    class Tactic(ScriptedTactic):
        def decide(self, world: WorldState) -> list[Assignment]:
            return [Assignment(a.robot_id, SkillSpec(a.skill.name,
                                                     {**a.skill.kwargs, "avoid": avoid}))
                    for a in super().decide(world)]

    kickoff = Scenario.kickoff()
    sim = KinematicWorld(dict(enumerate(kickoff.us)),
                         {i: Opponent(x, y) for i, (x, y, _) in enumerate(kickoff.them)},
                         ball=(0.0, 0.0), game=game)
    coach = CountingCoach(Tactic(DIV_B), DIV_B)
    world = sim.world()
    for _ in range(int(seconds * 60)):
        world = sim.step(coach.tick(world))
    return sim, coach


def test_scripted_6v6_through_the_coach_never_collides():
    from tbots.core.geometry import ROBOT_RADIUS

    running = GameState(play=Play.RUN, can_move=True, min_ball_distance=0.0, our_goalkeeper=0)
    sim, _ = _scripted_6v6(running)
    assert sim.min_clearance >= 2.0 * ROBOT_RADIUS


def test_planner_keeps_the_rule_filter_idle_on_stop():
    stop = GameState(play=Play.STOP, can_move=True, min_ball_distance=0.5, our_goalkeeper=0)
    _, planned = _scripted_6v6(stop, seconds=4.0, avoid=True)
    _, straight = _scripted_6v6(stop, seconds=4.0, avoid=False)
    assert straight.stopped_by_filter > 0
    assert planned.stopped_by_filter == 0
