"""想定 JSON 读写与校验测试。"""

import unittest

from c2sim.engine import Engine
from c2sim.scenario_io import ScenarioError, from_dict, to_dict
from c2sim.scenarios import build_point_defense_scenario


def _minimal() -> dict:
    return {
        "asset": [0, 0, 0],
        "spotters": [{"station_id": "S", "position": [0, 0, 20]}],
        "pads": [{"pad_id": "P", "position": [2500, 0, 15]}],
        "targets": [
            {"target_id": "T1", "position": [7000, 0, 500], "aim": [0, 0, 0],
             "cruise_speed": 45.0, "kind": "fixed_wing_uav", "rcs": 0.15},
        ],
    }


class TestRoundTrip(unittest.TestCase):
    def test_roundtrip_reproduces_run(self):
        original = build_point_defense_scenario(seed=2026)
        rebuilt = from_dict(to_dict(original))
        # 重建想定应与原想定跑出一致结果(种子一致)。
        r1 = Engine(build_point_defense_scenario(seed=2026)).run()
        r2 = Engine(rebuilt).run()
        self.assertEqual(r1.summary(), r2.summary())

    def test_from_dict_builds_units(self):
        s = from_dict(_minimal())
        self.assertEqual(len(s.targets), 1)
        self.assertEqual(s.targets[0].kind.value, "fixed_wing_uav")
        self.assertEqual(s.pads[0].pad_id, "P")


class TestValidation(unittest.TestCase):
    def test_missing_targets(self):
        d = _minimal()
        del d["targets"]
        with self.assertRaises(ScenarioError):
            from_dict(d)

    def test_empty_targets(self):
        d = _minimal()
        d["targets"] = []
        with self.assertRaises(ScenarioError):
            from_dict(d)

    def test_bad_kind(self):
        d = _minimal()
        d["targets"][0]["kind"] = "f35"
        with self.assertRaises(ScenarioError):
            from_dict(d)

    def test_bad_vec3(self):
        d = _minimal()
        d["asset"] = [0, 0]
        with self.assertRaises(ScenarioError):
            from_dict(d)

    def test_unknown_pad_field(self):
        d = _minimal()
        d["pads"][0]["frobnicate"] = 1
        with self.assertRaises(ScenarioError):
            from_dict(d)

    def test_negative_cruise_speed_rejected(self):
        d = _minimal()
        d["targets"][0]["cruise_speed"] = -5.0
        with self.assertRaises(ScenarioError):
            from_dict(d)

    def test_nonpositive_rcs_rejected(self):
        d = _minimal()
        d["targets"][0]["rcs"] = 0.0
        with self.assertRaises(ScenarioError):
            from_dict(d)


if __name__ == "__main__":
    unittest.main()
