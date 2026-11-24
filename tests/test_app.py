"""Smoke test: the Streamlit report renders every tab without exceptions.

    python -m pytest tests
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
TABS = ["Verdict", "Momentum, 1927-2026", "Volume replication, 2009-2017", "Why it changed", "Data & method"]


def test_app_renders():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs][:5] == TABS
    labels = [m.label for m in at.metric]
    assert "Winners minus losers, 1965-1995" in labels
    assert "Same strategy, 1996-2026" in labels


def test_app_reruns_with_other_settings():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    formation = next(s for s in at.selectbox if s.label.startswith("Formation period J"))
    formation.set_value(12).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("J=12" in s.value for s in at.subheader)
