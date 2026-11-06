"""A global route around the STATIC keep-out regions, for the MPC to follow.

The MPC's horizon is one second. A defense area is two metres wide, so a
robot that has to get round one sees every short-horizon option as "move
away from the target" and stops at the edge -- a local minimum, not a bug
in the sampler. The rule regions do not move, so a shortest path round them
is cheap: a visibility graph over their inflated corners (and an octagon
round each circle), searched with Dijkstra. The planner then aims at the
first waypoint, with the rest of the route's length counted as distance
still to go, so it does not slow down at a corner.

Moving obstacles are NOT routed here. They are the MPC's job, every tick.

Called only when the straight segment to the goal is blocked, which in play
is rare; a region that contains the start or the goal is left to the MPC
(it cannot be routed out of, and the MPC's penalty pushes the robot out).
"""

from __future__ import annotations

import heapq
import math

from tbots.motion.obstacles import Circle, KeepOuts, Rect

Point = tuple[float, float]
CORNER_OFFSET: float = 0.03          # m, nodes sit just outside the regions


def _segment_hits_rect(a: Point, b: Point, r: Rect) -> bool:
    # Liang-Barsky clip of the segment against the rectangle.
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for p, q in ((-dx, a[0] - r.x_min), (dx, r.x_max - a[0]),
                 (-dy, a[1] - r.y_min), (dy, r.y_max - a[1])):
        if abs(p) < 1e-15:
            if q < 0.0:
                return False
            continue
        t = q / p
        if p < 0.0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _segment_hits_circle(a: Point, b: Point, c: Circle) -> bool:
    dx, dy = b[0] - a[0], b[1] - a[1]
    den = dx * dx + dy * dy
    s = 0.0 if den < 1e-15 else max(0.0, min(1.0, ((c.x - a[0]) * dx + (c.y - a[1]) * dy) / den))
    return math.hypot(a[0] + s * dx - c.x, a[1] + s * dy - c.y) < c.radius


def _inside_rect(p: Point, r: Rect) -> bool:
    return r.x_min <= p[0] <= r.x_max and r.y_min <= p[1] <= r.y_max


def _inside_circle(p: Point, c: Circle) -> bool:
    return math.hypot(p[0] - c.x, p[1] - c.y) < c.radius


def _in_bounds(p: Point, b: Rect | None) -> bool:
    return b is None or _inside_rect(p, b)


def route(start: Point, goal: Point, keepouts: KeepOuts) -> list[Point]:
    """Waypoints from ``start`` (exclusive) to ``goal`` (inclusive)."""
    rects = [r for r in keepouts.rects
             if not _inside_rect(start, r) and not _inside_rect(goal, r)]
    circles = [c for c in keepouts.circles
               if not _inside_circle(start, c) and not _inside_circle(goal, c)]

    def clear(a: Point, b: Point) -> bool:
        return (not any(_segment_hits_rect(a, b, r) for r in rects)
                and not any(_segment_hits_circle(a, b, c) for c in circles))

    if clear(start, goal):
        return [goal]

    nodes: list[Point] = [start, goal]
    e = CORNER_OFFSET
    for r in rects:
        nodes += [(r.x_min - e, r.y_min - e), (r.x_min - e, r.y_max + e),
                  (r.x_max + e, r.y_min - e), (r.x_max + e, r.y_max + e)]
    for c in circles:
        # Circumscribed octagon, so its edges stay outside the circle.
        rad = c.radius / math.cos(math.pi / 8.0) + e
        nodes += [(c.x + rad * math.cos(k * math.pi / 4.0), c.y + rad * math.sin(k * math.pi / 4.0))
                  for k in range(8)]
    nodes = nodes[:2] + [
        p for p in nodes[2:]
        if _in_bounds(p, keepouts.bounds)
        and not any(_inside_rect(p, r) for r in rects)
        and not any(_inside_circle(p, c) for c in circles)
    ]

    # Dijkstra from start (0) to goal (1) on the visibility graph.
    best = {0: 0.0}
    parent: dict[int, int] = {}
    frontier = [(0.0, 0)]
    done: set[int] = set()
    while frontier:
        d, i = heapq.heappop(frontier)
        if i in done:
            continue
        done.add(i)
        if i == 1:
            break
        for j in range(len(nodes)):
            if j in done or j == i:
                continue
            step = math.dist(nodes[i], nodes[j])
            if d + step < best.get(j, math.inf) and clear(nodes[i], nodes[j]):
                best[j] = d + step
                parent[j] = i
                heapq.heappush(frontier, (d + step, j))

    if 1 not in parent:
        return [goal]          # no route: leave it to the MPC
    path = [1]
    while path[-1] != 0:
        path.append(parent[path[-1]])
    return [nodes[i] for i in reversed(path[:-1])]
