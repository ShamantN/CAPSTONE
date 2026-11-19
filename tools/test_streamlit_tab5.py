"""
Unit & Integration Test for Streamlit Tab 5 (TEST-3 Evaluation Viewer).
Uses streamlit.testing.v1.AppTest to ensure Tab 5 mounts, discovers pilot runs,
renders metadata, primary metrics, anchors, and tables without errors.
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest


class TestStreamlitTab5(unittest.TestCase):
    def test_apptest_tab5_mount_and_render(self):
        """Tests that app.py loads cleanly and Tab 5 renders evaluation runs without unhandled exceptions."""
        app_file = str(PROJECT_ROOT / "app.py")
        at = AppTest.from_file(app_file, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App crashed with exception: {at.exception}")

        # Check tabs exist
        # app.py defines 5 tabs: [Prompt & Settings, Scene Board, Face Bank, Contact Sheets, Evaluation Runs]
        # Verify markdown or text from render_eval_tab is present
        # e.g., 'TEST-3 Pre-Registered Evaluation Dashboard'
        has_eval_header = any("TEST-3 Pre-Registered Evaluation Dashboard" in m.value for m in at.header)
        self.assertTrue(has_eval_header, "Tab 5 Header 'TEST-3 Pre-Registered Evaluation Dashboard' not found in rendered app!")

        # Verify run selection box exists
        self.assertTrue(len(at.selectbox) >= 1, "Expected selectbox for run selection in AppTest!")

        print("[TEST_PASS] AppTest successfully verified Tab 5 mounting and rendering with zero exceptions.")


if __name__ == "__main__":
    unittest.main()
