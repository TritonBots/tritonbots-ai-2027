"""Run rSim and stream it to ssl-vision-client. The architecture proof.

    python -m tbots.apps.viz_rsim --realtime

Open http://localhost:8083 and you are watching an in-process training
simulator through the same browser tool you use for live matches. That is
the compose stack's `vision-client-rsim`, on 10006. The match view on :8082
listens on 10020, which the ER-Force simulator already publishes to, so rSim
cannot share it: both send camera 0 and the browser flickers between them.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from tbots.backends.base import Scenario
from tbots.backends.rsim import RSimBackend
from tbots.core.geometry import DIV_B
from tbots.net.multicast import senders
from tbots.net.vision_publisher import VisionPublisher
from tbots.skills.go_to_point import GoToPoint

_VISION_GROUP = "224.5.23.2"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--realtime", action="store_true",
                   help="sleep to match wall clock (needed to actually watch)")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--port", type=int, default=10006,
                   help="vision multicast port. Default 10006 is what the "
                        "compose stack's vision-client-rsim "
                        "(http://localhost:8083) and a native "
                        "ssl-vision-client listen on. Not 10020: the "
                        "ER-Force simulator publishes there, and the app "
                        "refuses any port that already has a publisher.")
    args = p.parse_args()

    # Before we send anything, so whatever this hears is someone else.
    others = senders(_VISION_GROUP, args.port)
    if others:
        src = ", ".join(f"{host}:{port}" for host, port in sorted(others))
        sys.exit(
            f"viz_rsim: {_VISION_GROUP}:{args.port} already has a vision "
            f"publisher ({src}).\n"
            "ssl-vision-client would flicker between that world and this "
            "one. Stop the other publisher. On the compose stack 10020 is "
            "the ER-Force simulator; rSim belongs on the default 10006, "
            "watched at http://localhost:8083.")

    backend = RSimBackend(n_us=6, n_them=6, dt=1.0 / 60.0, geometry=DIV_B)
    pub = VisionPublisher(group=_VISION_GROUP, geometry=DIV_B, port=args.port)
    pub.publish_geometry()

    world = backend.reset(Scenario.kickoff())
    skills = {i: GoToPoint(target=(0.0, 0.0)) for i in range(6)}
    for i, s in skills.items():
        s.reset(world, i)

    n_ticks = int(args.seconds / backend.dt)
    t0 = time.time()
    for tick in range(n_ticks):
        # Make the robots orbit, so there is obviously something happening.
        phase = tick * backend.dt * 0.5
        for i, s in skills.items():
            a = phase + i * (2 * math.pi / 6)
            s.target = (2.0 * math.cos(a), 1.5 * math.sin(a))

        cmds = [skills[i].step(world, i) for i in range(6)]
        world = backend.step(cmds)
        pub.publish(world)

        if tick % 300 == 0:
            pub.publish_geometry()
        if args.realtime:
            target = t0 + (tick + 1) * backend.dt
            time.sleep(max(0.0, target - time.time()))

    elapsed = time.time() - t0
    print(f"{n_ticks} ticks in {elapsed:.2f}s "
          f"= {n_ticks / elapsed:,.0f} steps/s "
          f"({n_ticks * backend.dt / elapsed:.1f}x realtime)")
    pub.close()
    backend.close()


if __name__ == "__main__":
    main()
