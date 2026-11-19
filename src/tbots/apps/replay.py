"""Play a rollout file back to ssl-vision-client, pausable and seekable.

    python -m tbots.apps.replay runs/NAME/rollouts/eval_seed1000000.npz

Open http://localhost:8083 (the compose stack's vision-client-rsim, port
10006) and the episode plays at real time, from ground truth, through the
same VisionPublisher rSim and eval use.

Keys (in this terminal):

    space        pause / resume
    left, h      back 1 s          right, l    forward 1 s
    ,  .         one frame back / forward (pauses)
    -  +         half / double speed
    0            back to the start
    q            quit

Without a terminal (piped, CI) it just plays to the end.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from tbots.core.state import WorldState
from tbots.rl.rollout import Rollout, load_rollout

_VISION_GROUP = "224.5.23.2"
GEOMETRY_EVERY_S = 5.0      # re-send geometry so a browser opened late gets it


class Player:
    """Maps wall-clock time onto rollout frames. No I/O of its own: it calls
    `publish(world)` for each frame it shows, and reads time from `clock`."""

    def __init__(self, rollout: Rollout, publish: Callable[[WorldState], None],
                 clock: Callable[[], float] = time.monotonic, speed: float = 1.0) -> None:
        if len(rollout) == 0:
            raise ValueError("empty rollout")
        self.r = rollout
        self.publish = publish
        self.clock = clock
        self.speed = speed
        self.paused = False
        self.index = 0
        self.shown: int | None = None
        self._anchor_wall = clock()
        self._anchor_index = 0

    @property
    def dt(self) -> float:
        return self.r.dt

    @property
    def at_end(self) -> bool:
        return self.index >= len(self.r) - 1

    def _reanchor(self) -> None:
        self._anchor_wall = self.clock()
        self._anchor_index = self.index

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._reanchor()

    def seek_to(self, index: int) -> None:
        self.index = min(max(0, index), len(self.r) - 1)
        self._reanchor()
        self._show()

    def seek(self, seconds: float) -> None:
        self.seek_to(self.index + round(seconds / self.dt))

    def step_frames(self, n: int) -> None:
        self.paused = True
        self.seek_to(self.index + n)

    def set_speed(self, speed: float) -> None:
        self.speed = min(max(speed, 0.125), 16.0)
        self._reanchor()

    def _show(self) -> None:
        if self.shown != self.index:
            self.publish(self.r.world(self.index))
            self.shown = self.index

    def update(self) -> float:
        """Advance to the frame wall-clock time says, show it, and return
        how long to wait before the next frame is due."""
        if not self.paused:
            elapsed = (self.clock() - self._anchor_wall) * self.speed
            self.index = min(self._anchor_index + int(math.floor(elapsed / self.dt + 1e-9)),
                             len(self.r) - 1)
        self._show()
        if self.paused or self.at_end:
            return 0.05
        next_due = (self.index + 1 - self._anchor_index) * self.dt / self.speed
        return max(0.0, self._anchor_wall + next_due - self.clock())


@contextmanager
def _raw_terminal() -> Iterator[Callable[[], str]]:
    """Non-blocking single-key reads from stdin, if it is a terminal."""
    if not sys.stdin.isatty():
        yield lambda: ""
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def read() -> str:
        if not select.select([fd], [], [], 0)[0]:
            return ""
        s = os.read(fd, 8).decode(errors="ignore")
        return {"\x1b[D": "left", "\x1b[C": "right"}.get(s, s[:1])

    try:
        yield read
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def handle_key(p: Player, key: str) -> bool:
    """Apply one key. Returns False to quit."""
    if key == "q":
        return False
    actions: dict[str, Callable[[], None]] = {
        " ": p.toggle_pause,
        "left": lambda: p.seek(-1.0), "h": lambda: p.seek(-1.0),
        "right": lambda: p.seek(1.0), "l": lambda: p.seek(1.0),
        ",": lambda: p.step_frames(-1), ".": lambda: p.step_frames(1),
        "-": lambda: p.set_speed(p.speed / 2), "+": lambda: p.set_speed(p.speed * 2),
        "=": lambda: p.set_speed(p.speed * 2), "0": lambda: p.seek_to(0),
    }
    if key in actions:
        actions[key]()
    return True


def play(rollout: Rollout, port: int = 10006, speed: float = 1.0, loop: bool = False,
         interactive: bool = True) -> None:
    from tbots.core.geometry import DIV_B
    from tbots.net.vision_publisher import VisionPublisher

    pub = VisionPublisher(group=_VISION_GROUP, port=port, geometry=DIV_B)
    pub.publish_geometry()
    last_geom = time.monotonic()
    player = Player(rollout, lambda w: pub.publish(w), speed=speed)
    n = len(rollout)
    try:
        with _raw_terminal() as read_key:
            while True:
                wait = player.update()
                if time.monotonic() - last_geom > GEOMETRY_EVERY_S:
                    pub.publish_geometry()
                    last_geom = time.monotonic()
                if interactive:
                    state = "paused " if player.paused else f"x{player.speed:g}   "
                    print(f"\r{state} {player.index * player.dt:7.2f} / "
                          f"{(n - 1) * player.dt:.2f} s   frame {player.index + 1}/{n} ",
                          end="", flush=True)
                    if not handle_key(player, read_key()):
                        break
                if player.at_end and not player.paused:
                    if loop:
                        player.seek_to(0)
                    elif not (interactive and sys.stdin.isatty()):
                        break
                time.sleep(min(wait, 0.02) if interactive else wait)
    finally:
        if interactive:
            print()
        pub.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rollout", type=Path)
    p.add_argument("--port", type=int, default=10006,
                   help="vision multicast port (default 10006: http://localhost:8083)")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--loop", action="store_true", help="start again at the end")
    args = p.parse_args(argv)

    from tbots.net.multicast import senders
    others = senders(_VISION_GROUP, args.port)
    if others:
        src = ", ".join(f"{h}:{pt}" for h, pt in sorted(others))
        sys.exit(f"replay: {_VISION_GROUP}:{args.port} already has a vision publisher ({src}). "
                 "Stop it first; see `python -m tbots.apps.viz_rsim --help`.")
    r = load_rollout(args.rollout)
    print(f"{args.rollout}: {len(r)} frames, {(len(r) - 1) * r.dt:.2f} s, "
          f"done {r.done or '-'}, {r.meta.get('task', '?')} seed {r.meta.get('seed', '?')}")
    print("space pause  left/right 1 s  , . frame  - + speed  0 start  q quit")
    play(r, port=args.port, speed=args.speed, loop=args.loop)


if __name__ == "__main__":
    main()
