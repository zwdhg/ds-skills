"""态势可视化(SVG)冒烟测试。"""

import os
import tempfile
import unittest

from c2sim.engine import Engine
from c2sim.scenarios import build_point_defense_scenario
from c2sim.viz import render_svg


class TestViz(unittest.TestCase):
    def test_render_svg_writes_valid_file(self):
        engine = Engine(build_point_defense_scenario(seed=2026))
        engine.run()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "map.svg")
            svg = render_svg(engine, path, title="测试")
            self.assertTrue(os.path.exists(path))
            self.assertTrue(svg.startswith("<svg"))
            self.assertIn("</svg>", svg)
            # 应包含安全穹顶、航迹等关键元素。
            self.assertIn("polyline", svg)
            self.assertIn("circle", svg)


if __name__ == "__main__":
    unittest.main()
