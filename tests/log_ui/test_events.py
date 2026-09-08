from fastapi.testclient import TestClient
from waf_proxy.events import EventStore
from log_ui.app import create_app


def _seed(db):
    st = EventStore(db)
    st.record(method="GET", path="/search", param="q", value="mouse",
              scores={"cnn": 0.01}, active_model="cnn", score=0.01,
              threshold=0.5, blocked=False, secure_mode=False)
    st.record(method="GET", path="/search", param="q", value="' OR 1=1 -- ",
              scores={"cnn": 0.98}, active_model="cnn", score=0.98,
              threshold=0.5, blocked=True, secure_mode=False)
    return st


def test_events_endpoint_returns_newest_first(tmp_path):
    db = str(tmp_path / "e.db")
    _seed(db)
    c = TestClient(create_app(db))
    ev = c.get("/events").json()["events"]
    assert ev[0]["value"] == "' OR 1=1 -- "
    assert ev[0]["blocked"] == 1


def test_summary_counts(tmp_path):
    db = str(tmp_path / "e.db")
    _seed(db)
    c = TestClient(create_app(db))
    s = c.get("/summary").json()
    assert s["total"] == 2
    assert s["blocked"] == 1
    assert abs(s["block_rate"] - 0.5) < 1e-9
    assert s["active_model"] == "cnn"


def test_index_served(tmp_path):
    db = str(tmp_path / "e.db")
    _seed(db)
    c = TestClient(create_app(db))
    r = c.get("/")
    assert r.status_code == 200
    assert b"<table" in r.content or b"<div" in r.content
