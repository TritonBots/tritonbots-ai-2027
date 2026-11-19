"""General-purpose terms for skill tasks. All read ground truth only."""

from __future__ import annotations

import math

from tbots.core.transition import Done, Transition
from tbots.rl.rewards.registry import register_reward


def _ego_ball_distance(world, ego_id) -> float | None:
    me = world.us.get(ego_id)
    if me is None:
        return None
    return math.hypot(me.x - world.ball.x, me.y - world.ball.y)


@register_reward("ball_progress")
class BallProgress:
    """Meters the ego robot closed on the ball this tick.

    Summed over an episode this is start distance minus end distance, so it
    cannot be farmed by standing still or by oscillating.
    """

    def reset(self) -> None:
        pass

    def __call__(self, tr: Transition) -> float:
        now = _ego_ball_distance(tr.world, tr.ego_id)
        was = _ego_ball_distance(tr.prev, tr.ego_id)
        if now is None or was is None:
            return 0.0
        return was - now


@register_reward("outcome")
class Outcome:
    """A one-off payment when the episode ends, by reason. Timeouts pay
    nothing: truncation is not an outcome."""

    def __init__(self, success: float = 1.0, failure: float = -1.0,
                 out_of_bounds: float = 0.0) -> None:
        self.table = {Done.SKILL_SUCCESS: success, Done.SKILL_FAILURE: failure,
                      Done.OUT_OF_BOUNDS: out_of_bounds}

    def reset(self) -> None:
        pass

    def __call__(self, tr: Transition) -> float:
        return self.table.get(tr.done, 0.0) if tr.done is not None else 0.0


@register_reward("time_cost")
class TimeCost:
    """-1 per second of simulated time. Pays for finishing sooner."""

    def reset(self) -> None:
        pass

    def __call__(self, tr: Transition) -> float:
        return -tr.dt
