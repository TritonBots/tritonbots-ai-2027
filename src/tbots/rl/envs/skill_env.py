"""Single-robot skill training env: a thin env over a `SkillTask`.

One robot (the task's `ego_id`) learns; everyone else stands still. The
task supplies everything that differs between skills -- scenario, success
and failure, observation builder, codec, time limit -- so this file does
not change when someone adds a skill. Spaces come from the codec and the
builder; out-of-bounds terminates (ARCHITECTURE section 9).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from tbots.backends.base import SimBackend
from tbots.core.command import RobotCommand
from tbots.core.state import WorldState
from tbots.core.transition import Done
from tbots.perception.sim import Perception
from tbots.rl.codec import build_codec
from tbots.rl.envs.base import SSLEnv
from tbots.rl.obs.builders import build_obs, is_egocentric, obs_size
from tbots.rl.rewards.registry import CompositeReward
from tbots.rl.tasks.registry import SkillTask


class SkillEnv(SSLEnv):
    def __init__(
        self,
        backend: SimBackend,
        task: SkillTask,
        reward: CompositeReward,
        *,
        n_us: int,
        n_them: int,
        perception: Perception | None = None,
        codec_overrides: Mapping[str, Any] | None = None,
        out_of_bounds: str = "terminate",
        render_mode: str | None = None,
    ) -> None:
        if n_us < 1:
            raise ValueError("a skill env needs at least the ego robot (n_us >= 1)")
        if not is_egocentric(task.obs_builder):
            raise ValueError(f"skill task '{task.name}' uses '{task.obs_builder}', "
                             "which is not an egocentric builder")
        self.task = task
        self.codec = build_codec(task.codec, dt=backend.dt,
                                 **{**task.codec_kwargs, **(codec_overrides or {})})
        super().__init__(
            backend, reward,
            lambda rng: task.scenario(rng, n_us, n_them),
            perception=perception,
            max_episode_ticks=max(1, round(task.time_limit_s / backend.dt)),
            ego_id=task.ego_id,
            out_of_bounds=out_of_bounds,
            render_mode=render_mode,
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_size(task.obs_builder),), np.float32)
        self.action_space = self.codec.action_space

    def _on_reset(self, truth: WorldState) -> None:
        self.codec.reset()

    def _observe(self, world: WorldState) -> np.ndarray:
        return build_obs(self.task.obs_builder, world, self.task.ego_id)

    def _decode(self, action: Any, world: WorldState) -> list[list[RobotCommand]]:
        return [[c] for c in self.codec.decode(action, self.task.ego_id)]

    def _task_done(self, truth: WorldState, prev: WorldState) -> Done | None:
        if self.task.success(truth, self.task.ego_id):
            return Done.SKILL_SUCCESS
        if self.task.failure(truth, self.task.ego_id):
            return Done.SKILL_FAILURE
        return None
