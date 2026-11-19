"""Evaluate a checkpoint: fixed seeds, deterministic policy, recorded.

    python -m tbots.apps.eval --checkpoint runs/NAME/latest.pt
    python -m tbots.apps.eval --checkpoint runs/NAME/latest.pt --render vision --realtime

Runs the policy deterministically on a FIXED seed set, in `tracked`
perception, and writes next to the checkpoint:

    runs/NAME/eval.json          per-episode return, length, done reason, goal
                                 difference, per-term reward totals; and the
                                 distributions (mean/std/min/max) per term
    runs/NAME/rollouts/*.npz     one rollout per episode; the newest ten kept

`--render vision --realtime` streams every tick to ssl-vision-client
(http://localhost:8083, port 10006) at wall-clock speed as it plays. To watch
a recorded episode again later, use `python -m tbots.apps.replay`.

The environment is rebuilt from the run's own recorded config.yaml through
`make_env`, the same path training used. Trailing `key=value` arguments
override it (`env.n_them=1`). Opponents: none exist yet in the skill env --
everyone but the ego robot stands still (TASK-057 adds the opponent pool).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from tbots.rl.artifacts import load_checkpoint
from tbots.rl.evaluate import (
    config_from_run,
    eval_config,
    eval_seeds,
    evaluate,
    load_policy,
    summarize,
)
from tbots.rl.factory import make_env

_VISION_GROUP = "224.5.23.2"
_VISION_PORT = 10006


def run_eval(checkpoint: str | Path, *, episodes: int = 10, perception: str = "tracked",
             keep: int = 10, render: str | None = None, realtime: bool = False,
             overrides: list[str] | None = None) -> dict:
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent
    cfg = config_from_run(run_dir)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    cfg = eval_config(cfg, perception)

    probe = make_env(cfg)
    try:
        agent = load_policy(checkpoint, probe, cfg)
    finally:
        probe.close()
    results = evaluate(agent, cfg, eval_seeds(episodes), record_dir=run_dir / "rollouts",
                       keep=keep, label="eval",
                       render_mode="vision" if render == "vision" else None,
                       realtime=realtime)
    state = load_checkpoint(checkpoint)
    report = {"checkpoint": str(checkpoint), "global_step": state.get("global_step"),
              "task": str(cfg.task), "perception_mode": perception,
              "seeds": eval_seeds(episodes), **summarize(results)}
    out = run_dir / "eval.json"
    out.write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=10, help="size of the fixed seed set")
    p.add_argument("--perception", default="tracked", choices=["truth", "noisy", "tracked"],
                   help="tracked unless you are debugging: evaluate what a match sees")
    p.add_argument("--keep", type=int, default=10, help="rollout files kept in run_dir/rollouts")
    p.add_argument("--render", choices=["vision"], default=None,
                   help="stream to ssl-vision-client on 224.5.23.2:10006 (localhost:8083)")
    p.add_argument("--realtime", action="store_true", help="play at wall-clock speed")
    p.add_argument("overrides", nargs="*", help="config overrides, key=value")
    args = p.parse_args(argv)

    if args.render == "vision":
        from tbots.net.multicast import senders
        others = senders(_VISION_GROUP, _VISION_PORT)
        if others:
            src = ", ".join(f"{h}:{pt}" for h, pt in sorted(others))
            sys.exit(f"eval: {_VISION_GROUP}:{_VISION_PORT} already has a vision publisher "
                     f"({src}). Stop it first; see `python -m tbots.apps.viz_rsim --help`.")

    r = run_eval(args.checkpoint, episodes=args.episodes, perception=args.perception,
                 keep=args.keep, render=args.render, realtime=args.realtime,
                 overrides=args.overrides)
    print(f"{r['task']} @ step {r['global_step']}, {len(r['episodes'])} episodes, "
          f"{r['perception_mode']} perception")
    for e in r["episodes"]:
        print(f"  seed {e['seed']}  return {e['ret']:8.3f}  {e['seconds']:6.2f} s  "
              f"{e['done']}  goal diff {e['goal_difference']:+d}")
    print(f"mean return {r['mean_return']:.3f} +- {r['std_return']:.3f}   "
          f"goal difference {r['goal_difference']:+d}   done {r['done_reasons']}")
    for name, d in r["reward_terms"].items():
        print(f"  {name:20s} mean {d['mean']:8.3f}  std {d['std']:7.3f}  "
              f"[{d['min']:.3f}, {d['max']:.3f}]")
    print(f"wrote {Path(args.checkpoint).parent / 'eval.json'}")


if __name__ == "__main__":
    main()
