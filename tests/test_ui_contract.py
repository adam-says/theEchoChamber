from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OfflineUiContractTests(unittest.TestCase):
    def test_monitor_has_no_remote_dependencies(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertNotIn('src="http://', html)
        self.assertNotIn('src="https://', html)
        self.assertNotIn('href="http://', html)
        self.assertNotIn('href="https://', html)
        self.assertIn("assets/vendor/uplot/uPlot.iife.min.js", html)
        self.assertTrue((ROOT / "assets/vendor/uplot/uPlot.iife.min.js").is_file())
        self.assertTrue((ROOT / "assets/vendor/uplot/uPlot.min.css").is_file())
        self.assertTrue((ROOT / "assets/vendor/uplot/LICENSE").is_file())

    def test_safety_and_acquisition_controls_are_present(self) -> None:
        html = (ROOT / "index.html").read_text(encoding="utf-8")
        for token in (
            'id="btn-acq"',
            'id="btn-arm"',
            'id="fault-panel"',
            'id="pulse-polarity"',
            "window.location.hostname",
            "command: 'set_armed'",
            "setTimeout(send, 100)",
        ):
            self.assertIn(token, html)


if __name__ == "__main__":
    unittest.main()
