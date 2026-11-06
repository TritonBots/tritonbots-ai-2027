"""Drive robot 0 in a circle in the MATCH backend. Proves the UDP path.

    python -m tbots.apps.wiggle --color blue --sim-host 127.0.0.1

Depends on TASK-011 (robot_control) and TASK-010 (vision).

Commands go out over the SSL simulation protocol; the robot's position comes
back over SSL-Vision. Nothing here uses the tracker or the referee: the
colour is the flag, and positions are raw detections in the field frame. That
is the point -- if this works, the two wires work, and anything that fails
above it is above it.

It exits non-zero if the robot did not move, so it doubles as a smoke test.
The fitted circle radius should be close to --speed / --turn.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from tbots.core.command import RobotCommand, stop
from tbots.net.robot_control import BLUE_PORT, YELLOW_PORT, RobotControlSender
from tbots.net.vision import VisionReceiver


def fit_circle(xy: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle through points: (cx, cy, r)."""
    x, y = xy[:, 0], xy[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    (c0, c1, c2), *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = c0 / 2, c1 / 2
    return float(cx), float(cy), float(math.sqrt(c2 + cx * cx + cy * cy))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--color", choices=("blue", "yellow"), default="blue")
    p.add_argument("--robot", type=int, default=0)
    p.add_argument("--sim-host", default="127.0.0.1")
    # 10020 is the ER-Force simulator's vision port (configs/net/dev.yaml);
    # real SSL-Vision is 10006.
    p.add_argument("--vision-group", default="224.5.23.2")
    p.add_argument("--vision-port", type=int, default=10020)
    p.add_argument("--speed", type=float, default=0.5, help="forward, m/s")
    p.add_argument("--turn", type=float, default=1.0, help="rad/s")
    p.add_argument("--seconds", type=float, default=8.0)
    args = p.parse_args(argv)

    port = YELLOW_PORT if args.color == "yellow" else BLUE_PORT
    control = RobotControlSender(host=args.sim_host, port=port)
    vision = VisionReceiver(args.vision_group, args.vision_port)
    track: list[tuple[float, float]] = []
    cmd = RobotCommand(args.robot, vx=args.speed, vtheta=args.turn)

    def ours(frame):
        return frame.robots_yellow if args.color == "yellow" else frame.robots_blue

    print(f"driving {args.color} {args.robot} at {args.speed} m/s, {args.turn} rad/s "
          f"-> {args.sim_host}:{port}; watching {args.vision_group}:{args.vision_port}")
    try:
        t_end = time.monotonic() + args.seconds
        while time.monotonic() < t_end:
            control.send([cmd])
            vision.wait_for_next_frame(timeout=0.05)
            for f in vision.frames():
                track += [(r.x, r.y) for r in ours(f) if r.robot_id == args.robot]
    finally:
        for _ in range(5):
            control.send([stop(args.robot)])
            time.sleep(0.02)
        feedback = control.feedback()
        control.close()
        vision.close()

    if not track:
        print(f"FAIL: no vision of {args.color} {args.robot} on "
              f"{args.vision_group}:{args.vision_port}", file=sys.stderr)
        return 1
    xy = np.array(track)
    path = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    print(f"{len(xy)} sightings, start ({xy[0, 0]:+.2f}, {xy[0, 1]:+.2f}) m, "
          f"path {path:.2f} m, feedback for robots {sorted(feedback)}")
    if path < 0.25 * args.speed * args.seconds:
        print("FAIL: the robot barely moved -- commands are not arriving, "
              "or are going to the other colour", file=sys.stderr)
        return 1
    cx, cy, r = fit_circle(xy)
    print(f"circle: centre ({cx:+.2f}, {cy:+.2f}) m, radius {r:.2f} m "
          f"(commanded {args.speed / args.turn:.2f} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
