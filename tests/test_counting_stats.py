import unittest
from dataclasses import dataclass

from counting_stats import (
    CountingAoi,
    CountingConfig,
    CountingLine,
    analyze_tracks,
    detect_aoi_events,
    counting_config_from_dict,
    detect_line_crossings,
)


@dataclass
class Detection:
    frame_idx: int
    centroid: tuple[float, float]
    area: int = 5
    score: float = 0.5


@dataclass
class Track:
    track_id: int
    detections: list[Detection]


def make_track(track_id, points):
    return Track(track_id, [Detection(frame, point) for frame, point in enumerate(points)])


class CountingStatsTest(unittest.TestCase):
    def test_line_crossing_left_to_right_negative_direction(self):
        cfg = CountingConfig(
            lines=[CountingLine("mid", "Middle", (0, 0), (0, 10), positive_label="right_to_left", negative_label="left_to_right")],
            line_crossing_epsilon=0.0,
        )
        track = make_track(1, [(-5, 5), (5, 5)])

        events = detect_line_crossings([track], fps=10.0, cfg=cfg)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].line_id, "mid")
        self.assertEqual(events[0].direction, "left_to_right")
        self.assertEqual(events[0].frame, 1)

    def test_line_crossing_right_to_left_positive_direction(self):
        cfg = CountingConfig(
            lines=[CountingLine("mid", "Middle", (0, 0), (0, 10), positive_label="right_to_left", negative_label="left_to_right")],
            line_crossing_epsilon=0.0,
        )
        track = make_track(1, [(5, 5), (-5, 5)])

        events = detect_line_crossings([track], fps=10.0, cfg=cfg)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].direction, "right_to_left")

    def test_line_near_but_not_crossing_does_not_count(self):
        cfg = CountingConfig(lines=[CountingLine("mid", "Middle", (0, 0), (0, 10))])
        track = make_track(1, [(-5, 5), (-2, 6), (-1, 7)])

        self.assertEqual(detect_line_crossings([track], fps=10.0, cfg=cfg), [])

    def test_line_crossing_debounce_suppresses_fast_jitter(self):
        cfg = CountingConfig(
            lines=[CountingLine("mid", "Middle", (0, 0), (0, 10))],
            line_crossing_epsilon=0.0,
            min_frames_between_same_line_crossing=3,
        )
        track = make_track(1, [(-2, 5), (2, 5), (-2, 5), (2, 5), (-2, 5)])

        events = detect_line_crossings([track], fps=10.0, cfg=cfg)

        self.assertEqual([event.frame for event in events], [1, 4])

    def test_aoi_entry_and_exit_events(self):
        cfg = CountingConfig(aois=[CountingAoi("box", "Box", (10, 10, 20, 20))])
        track = make_track(1, [(0, 0), (15, 15), (20, 20), (25, 25), (40, 40)])

        events = detect_aoi_events([track], fps=10.0, cfg=cfg)

        self.assertEqual([(event.event_type, event.frame) for event in events], [("entry", 1), ("exit", 4)])

    def test_analyze_tracks_counts_valid_tracks_only_by_default(self):
        cfg = CountingConfig(lines=[CountingLine("mid", "Middle", (0, 0), (0, 10))])
        valid_track = make_track(1, [(-5, 5), (5, 5)])
        invalid_track = make_track(2, [(-5, 5), (5, 5)])

        results = analyze_tracks(
            [valid_track, invalid_track],
            fps=10.0,
            cfg=cfg,
            is_valid_track=lambda track: track.track_id == 1,
        )

        self.assertEqual(len(results.crossings), 1)
        self.assertEqual(results.crossings[0].track_id, 1)
        self.assertEqual(results.run_summary["valid_tracks"], 1)

    def test_polyline_crossing_uses_drawn_segments(self):
        cfg = CountingConfig(
            lines=[
                CountingLine(
                    "bent",
                    "Bent",
                    (0, 0),
                    (10, 10),
                    positive_label="A_to_B",
                    negative_label="B_to_A",
                    points=[(0, 0), (0, 10), (10, 10)],
                )
            ],
            line_crossing_epsilon=0.0,
        )
        track = make_track(1, [(-5, 5), (5, 5)])

        events = detect_line_crossings([track], fps=10.0, cfg=cfg)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].line_id, "bent")
        self.assertEqual(events[0].direction, "B_to_A")

    def test_polygon_aoi_entry_and_exit_from_json(self):
        cfg = counting_config_from_dict(
            {
                "aois": [
                    {
                        "id": "poly",
                        "name": "Polygon",
                        "type": "polygon",
                        "coordinates": [[10, 10], [30, 10], [30, 30], [10, 30]],
                    }
                ]
            }
        )
        cfg.aoi_boundary_debounce_frames = 0
        track = make_track(1, [(0, 0), (20, 20), (40, 40)])

        events = detect_aoi_events([track], fps=10.0, cfg=cfg)

        self.assertEqual([(event.event_type, event.frame) for event in events], [("entry", 1), ("exit", 2)])

    def test_cvc_drawn_config_format_loads_lines_and_zones(self):
        cfg = counting_config_from_dict(
            {
                "lines": [{"name": "Gate", "a": [0, 0], "b": [0, 10]}],
                "zones": [{"name": "Roost", "pts": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
            }
        )

        self.assertEqual(cfg.lines[0].id, "Gate")
        self.assertEqual(cfg.lines[0].points, [(0.0, 0.0), (0.0, 10.0)])
        self.assertEqual(cfg.aois[0].type, "polygon")
        self.assertEqual(cfg.aois[0].name, "Roost")


if __name__ == "__main__":
    unittest.main()
