"""Pure geometry primitives used by line crossing and AOI counting."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Rect = Tuple[float, float, float, float]


def point_in_rect(point: Point, rect: Rect) -> bool:
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def point_in_aoi(point: Point, aoi: Any) -> bool:
    if aoi.type == "polygon":
        return point_in_polygon(point, aoi.coordinates)
    return point_in_rect(point, aoi.coordinates)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for idx, current in enumerate(polygon):
        x1, y1 = current
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        if (y1 > y) != (y2 > y):
            intersect_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersect_x:
                inside = not inside
    return inside


def line_points(line: Any) -> List[Point]:
    if getattr(line, "points", None):
        return list(line.points)
    return [line.p1, line.p2]


def polyline_side(points: Sequence[Point], point: Point, epsilon: float) -> int:
    if len(points) < 2:
        return 0
    best_segment = min(
        zip(points, points[1:]),
        key=lambda pair: _point_to_segment_distance_sq(pair[0], pair[1], point),
    )
    return signed_side(best_segment[0], best_segment[1], point, epsilon)


def polyline_crossing_sign(
    previous_point: Point,
    current_point: Point,
    points: Sequence[Point],
    epsilon: float,
) -> Optional[int]:
    for start, end in zip(points, points[1:]):
        if not segments_intersect(previous_point, current_point, start, end):
            continue
        before = signed_side(start, end, previous_point, epsilon)
        after = signed_side(start, end, current_point, epsilon)
        if before < 0 and after > 0:
            return 1
        if before > 0 and after < 0:
            return -1
    return None


def signed_side(a: Point, b: Point, point: Point, epsilon: float) -> int:
    cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    if abs(cross) <= epsilon:
        return 0
    return 1 if cross > 0 else -1


def _point_to_segment_distance_sq(a: Point, b: Point, point: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0 and dy == 0:
        return (point[0] - a[0]) ** 2 + (point[1] - a[1]) ** 2
    t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    nearest = (a[0] + t * dx, a[1] + t * dy)
    return (point[0] - nearest[0]) ** 2 + (point[1] - nearest[1]) ** 2


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], b[0]) - 1e-6 <= c[0] <= max(a[0], b[0]) + 1e-6
        and min(a[1], b[1]) - 1e-6 <= c[1] <= max(a[1], b[1]) + 1e-6
    )


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    o1, o2 = _orientation(p1, p2, q1), _orientation(p1, p2, q2)
    o3, o4 = _orientation(q1, q2, p1), _orientation(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _on_segment(p1, p2, q1))
        or (o2 == 0 and _on_segment(p1, p2, q2))
        or (o3 == 0 and _on_segment(q1, q2, p1))
        or (o4 == 0 and _on_segment(q1, q2, p2))
    )


# Backward-compatible private names used by the current detector.
_line_points = line_points
_polyline_side = polyline_side
_polyline_crossing_sign = polyline_crossing_sign
_signed_side = signed_side
_segments_intersect = segments_intersect
