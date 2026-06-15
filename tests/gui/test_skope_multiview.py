"""Tests for multi-view SkopeRules GUI helper."""

from unittest import TestCase

import pandas as pd

from antakia.gui.helpers.skope_multiview import (
    format_multiview_status,
    multiview_available,
)


class TestSkopeMultiviewHelper(TestCase):
    def test_multiview_available(self):
        self.assertFalse(multiview_available(None))
        self.assertFalse(multiview_available(pd.DataFrame()))
        self.assertTrue(multiview_available(pd.DataFrame({"a": [1.0]})))

    def test_format_multiview_status_classic(self):
        self.assertEqual(format_multiview_status({"multiview": False}), "")

    def test_format_multiview_status_multiview(self):
        text = format_multiview_status(
            {
                "multiview": True,
                "multiview_mode": "conjoint",
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.85,
            }
        )
        self.assertIn("VS∧ES conjoint", text)
        self.assertIn("F1=0.85", text)
