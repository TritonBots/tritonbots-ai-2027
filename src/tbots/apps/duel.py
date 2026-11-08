"""Goalkeep against Attack in rSim: our keeper defends -x, their attacker shoots.

    python -m tbots.apps.duel                         # 20 duels, headless, tally
    python -m tbots.apps.duel --watch --n 3 --seconds 40

`--watch` runs in real time and streams to ssl-vision-client: open
http://localhost:8083 (the compose stack's `vision-client-rsim`, port 10006),
the same view `viz_rsim` uses. Stop any other vision publisher on that port
first.

The attacker is the opponent, driven through `as_opponent()`, so it plays in
its own frame attacking +x, which is our goal. Each duel starts with the ball
somewhere in midfield and the attacker about a metre behind it on their side.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.geometry import DIV_B
from tbots.core.perspective import as_opponent
from tbots.core.state import WorldState
from tbots.skills import Attack, Goalkeep

_VISION_GROUP = "224.5.23.2"


def _outcome(world: WorldState) -> str | None:
    g = DIV_B
    if world.ball.x < -g.half_length:
        return "conceded" if abs(world.ball.y) < g.goal_width / 2.0 else "wide"
    if world.ball.x > g.half_length or abs(world.ball.y) > g.half_width:
        return "cleared_out"
    return None


def run_duels(n: int = 20, seconds: float = 20.0, seed: int = 1,
              publisher=None, realtime: bool = False,
              verbose: bool = True) -> Counter[str]:
    """Play ``n`` duels and return the tally of outcomes.

    Outcomes: ``conceded`` (attacker scored), ``wide`` (crossed our goal line
    outside the posts), ``cleared_out`` (ball left the field anywhere else),
    ``timeout`` (still in play when time ran out).
    """
    backend = RSimBackend(n_us=1, n_them=1, geometry=DIV_B)
    rng = random.Random(seed)
    tally: Counter[str] = Counter()
    try:
        for k in range(n):
            bx, by = rng.uniform(-2.0, 1.0), rng.uniform(-2.0, 2.0)
            world = backend.reset(Scenario(
                ball=(bx, by, 0.0, 0.0),
                us=((DIV_B.our_goal[0] + 0.2, 0.0, 0.0),),
                them=((bx + 1.0, by + rng.uniform(-0.5, 0.5), 3.14159),),
            ))
            keeper, attacker = Goalkeep(), Attack()
            keeper.reset(world, 0)
            attacker.reset(as_opponent(world), 0)

            outcome = "timeout"
            ticks = int(seconds / backend.dt)
            t0 = time.time()
            tick = 0
            for tick in range(ticks):
                world = backend.step([keeper.step(world, 0)],
                                     [attacker.step(as_opponent(world), 0)])
                if publisher is not None:
                    publisher.publish(world)
                    if tick % 300 == 0:
                        publisher.publish_geometry()
                if realtime:
                    time.sleep(max(0.0, t0 + (tick + 1) * backend.dt - time.time()))
                result = _outcome(world)
                if result is not None:
                    outcome = result
                    break
            tally[outcome] += 1
            if verbose:
                print(f"duel {k + 1:2d}: {outcome:12s} ball=({bx:+.2f}, {by:+.2f}) "
                      f"t={(tick + 1) * backend.dt:5.1f}s keeper={keeper.mode} "
                      f"attacker={attacker.last_action}")
    finally:
        backend.close()
    return tally


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=20, help="number of duels")
    p.add_argument("--seconds", type=float, default=20.0,
                   help="simulated time limit per duel")
    p.add_argument("--seed", type=int, default=1, help="seed for the starting positions")
    p.add_argument("--watch", action="store_true",
                   help="stream vision in real time to watch at http://localhost:8083")
    p.add_argument("--port", type=int, default=10006,
                   help="vision multicast port for --watch (default 10006, as viz_rsim)")
    args = p.parse_args()

    pub = None
    if args.watch:
        from tbots.net.multicast import senders
        from tbots.net.vision_publisher import VisionPublisher

        others = senders(_VISION_GROUP, args.port)
        if others:
            src = ", ".join(f"{host}:{port}" for host, port in sorted(others))
            sys.exit(f"duel: {_VISION_GROUP}:{args.port} already has a vision publisher "
                     f"({src}). Stop it first; see `python -m tbots.apps.viz_rsim --help`.")
        pub = VisionPublisher(group=_VISION_GROUP, geometry=DIV_B, port=args.port)
        pub.publish_geometry()

    try:
        tally = run_duels(args.n, args.seconds, args.seed,
                          publisher=pub, realtime=args.watch)
    finally:
        if pub is not None:
            pub.close()
    print(dict(tally))


if __name__ == "__main__":
    main()
