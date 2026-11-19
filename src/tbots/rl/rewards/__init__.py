from tbots.rl.rewards import (
    example,  # noqa: F401  (self-registers "alive")
    skill,  # noqa: F401  (ball_progress, outcome, time_cost)
)
from tbots.rl.rewards.registry import (  # noqa: F401
    CompositeReward,
    RewardTerm,
    register_reward,
    reward_names,
)
