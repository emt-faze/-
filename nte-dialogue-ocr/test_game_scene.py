"""Offline scene tests; these never capture or interact with a game."""
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import game_scene as scene


def detected(text, score=.99, bbox=(100, 60, 240, 100)):
    x0, y0, x1, y1 = bbox
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], text, score]


class Engine:
    def __init__(self, values):
        self.values = iter(values)
        self.shapes = []
    def __call__(self, image, **kwargs):
        self.shapes.append(image.shape)
        return next(self.values), None


class SceneTests(unittest.TestCase):
    def frame(self, width=1280, height=800):
        return np.random.default_rng(17).integers(0, 90, (height, width, 3), dtype=np.uint8)

    def test_entry_uses_lower_middle_and_returns_client_coordinates(self):
        frame = self.frame()
        engine = Engine([[detected("进入游戏")]])
        result = scene.locate_enter_button(frame, engine)
        self.assertEqual(result["center"], [554, 648])
        self.assertEqual(engine.shapes, [(184, 512, 3)])

    def test_entry_rejects_ambiguous_low_score_and_unrelated_text(self):
        for detections in ([detected("进入游戏", .89)], [detected("开始游戏")],
                           [detected("进入游戏"), detected("进入游戏", bbox=(260, 60, 400, 100))]):
            with self.subTest(detections=detections):
                self.assertIsNone(scene.locate_enter_button(self.frame(), Engine([detections])))

    def test_entry_supports_traditional_chinese(self):
        self.assertIsNotNone(scene.locate_enter_button(self.frame(), Engine([[detected("進入遊戲")]])))

    def test_title_needs_independent_settings_marker(self):
        engine = Engine([[detected("进入游戏")]])
        with patch.object(scene, "_match_feature", return_value=None):
            result = scene.classify_frame(self.frame(), engine)
        self.assertEqual(result["state"], "unknown")
        self.assertIsNotNone(result["enter_button"])

    def test_title_cannot_be_called_in_game(self):
        engine = Engine([[detected("进入游戏")]])
        with patch.object(scene, "_match_feature", return_value={"feature": "login_setting", "score": .99}):
            result = scene.classify_frame(self.frame(), engine)
        self.assertEqual(result["state"], "title")

    def test_absent_entry_is_not_success(self):
        with patch.object(scene, "_match_feature", return_value=None):
            result = scene.classify_frame(self.frame(), Engine([None, None]))
        self.assertEqual(result["state"], "unknown")

    def test_two_spatially_independent_hud_features_required(self):
        def feature(frame, name, threshold):
            return {"feature": name, "score": .99, "bbox": [10, 10, 40, 40]} if name == "mini_map_arrow" else None
        with patch.object(scene, "_match_feature", side_effect=feature):
            result = scene.classify_frame(self.frame(), Engine([None, None]))
        self.assertEqual(result["state"], "unknown")
        def two(frame, name, threshold):
            return {"feature": name, "score": .99} if name in ("mini_map_arrow", "health_bar_slash") else None
        with patch.object(scene, "_match_feature", side_effect=two):
            result = scene.classify_frame(self.frame(), Engine([None]))
        self.assertEqual(result["state"], "in_game")

    def test_dialogue_controls_without_text_are_not_success(self):
        def feature(frame, name, threshold):
            return {"feature": name, "score": .99} if name in ("auto_play", "dialog_history") else None
        with patch.object(scene, "_match_feature", side_effect=feature):
            result = scene.classify_frame(self.frame(), Engine([None, None, None]))
        self.assertEqual(result["state"], "unknown")

    def test_story_requires_controls_and_actual_dialogue(self):
        def feature(frame, name, threshold):
            return {"feature": name, "score": .99} if name in ("auto_play", "dialog_history") else None
        with patch.object(scene, "_match_feature", side_effect=feature):
            result = scene.classify_frame(self.frame(), Engine([None, [detected("我们一起出发吧。")]]))
        self.assertEqual(result["state"], "in_game")

    def test_explicit_login_request_is_blocked(self):
        with patch.object(scene, "_match_feature", return_value=None):
            result = scene.classify_frame(self.frame(), Engine([None, [detected("请输入验证码")]]))
        self.assertEqual(result["state"], "blocked")

    def test_black_frame_is_unknown_without_running_ocr(self):
        engine = Engine([])
        self.assertEqual(scene.classify_frame(np.zeros((800, 1280, 3), np.uint8), engine)["state"], "unknown")
        self.assertEqual(engine.shapes, [])

    def test_template_match_works_on_16_by_10_and_16_by_9(self):
        template = scene._template("login_setting")
        self.assertIsNotNone(template)
        for height in (1440, 1600):
            frame = self.frame(2560, height)
            frame[36:36+template.shape[0], 2467:2467+template.shape[1]] = template
            match = scene._match_feature(frame, "login_setting", .9)
            self.assertIsNotNone(match)
            self.assertGreaterEqual(match["score"], .99)

    def test_actual_saved_title_has_settings_marker(self):
        path = Path(__file__).resolve().parent / "validation" / "title-baseline.png"
        if not path.exists():
            self.skipTest("No historical title image")
        frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        match = scene._match_feature(frame, "login_setting", .78)
        self.assertIsNotNone(match)

    def test_actual_world_requires_rotated_map_and_independent_hud(self):
        path = Path(__file__).resolve().parent / "validation" / "in-game-baseline.jpg"
        if not path.exists():
            self.skipTest("No captured in-game baseline")
        frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        result = scene.classify_frame(frame, Engine([None]))
        self.assertEqual(result["state"], "in_game")
        features = {item["feature"]: item for item in result["evidence"]}
        self.assertGreater(features["mini_map_arrow"]["score"], .9)
        self.assertGreater(features["health_bar_slash"]["score"], .9)
        self.assertIn("char_1_text", features)
        self.assertIn("char_2_text", features)

    def test_actual_title_without_entry_ocr_is_still_not_world(self):
        path = Path(__file__).resolve().parent / "validation" / "title-baseline.png"
        if not path.exists():
            self.skipTest("No captured title baseline")
        frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        result = scene.classify_frame(frame, Engine([None, None]))
        self.assertEqual(result["state"], "unknown")

    def test_world_with_map_marker_removed_is_not_success(self):
        path = Path(__file__).resolve().parent / "validation" / "in-game-baseline.jpg"
        if not path.exists():
            self.skipTest("No captured in-game baseline")
        frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        region, _, _ = scene._crop(frame, scene.SEARCH_REGIONS["mini_map_arrow"])
        region[:] = 0
        result = scene.classify_frame(frame, Engine([None, None]))
        self.assertEqual(result["state"], "unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
