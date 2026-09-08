import respx
import httpx
from fastapi.testclient import TestClient

from waf_proxy.config import Settings
from waf_proxy.events import EventStore
from waf_proxy.decision import decide, create_app


def test_decide_blocks_on_any_high_score():
    results = [{"score": 0.1, "value": "a"}, {"score": 0.8, "value": "b"}]
    blocked, worst = decide(results, 0.5)
    assert blocked is True
    assert worst["value"] == "b"


def test_decide_allows_when_all_low():
    blocked, worst = decide([{"score": 0.2, "value": "a"}], 0.5)
    assert blocked is False


@respx.mock
def test_malicious_request_is_blocked_and_recorded(tmp_path):
    respx.post("http://inf:9000/predict").mock(return_value=httpx.Response(
        200, json={"results": [{"value": "' OR 1=1 -- ",
                                "scores": {"cnn": 0.97}, "active_model": "cnn",
                                "score": 0.97, "decision": "malicious",
                                "threshold": 0.5}]}))
    st = EventStore(str(tmp_path / "e.db"))
    s = Settings(inference_url="http://inf:9000", upstream_url="http://up:8000",
                 events_db=str(tmp_path / "e.db"))
    c = TestClient(create_app(s, st))
    r = c.get("/search?q=' OR 1=1 -- ")
    assert r.status_code == 403
    assert r.json()["blocked_by"] == "ai-waf"
    assert st.recent()[0]["blocked"] == 1


@respx.mock
def test_benign_request_is_forwarded(tmp_path):
    respx.post("http://inf:9000/predict").mock(return_value=httpx.Response(
        200, json={"results": [{"value": "mouse", "scores": {"cnn": 0.01},
                                "active_model": "cnn", "score": 0.01,
                                "decision": "benign", "threshold": 0.5}]}))
    respx.get("http://up:8000/search").mock(return_value=httpx.Response(
        200, text="results page"))
    st = EventStore(str(tmp_path / "e.db"))
    s = Settings(inference_url="http://inf:9000", upstream_url="http://up:8000",
                 events_db=str(tmp_path / "e.db"))
    c = TestClient(create_app(s, st))
    r = c.get("/search?q=mouse")
    assert r.status_code == 200
    assert r.text == "results page"
    assert st.recent()[0]["blocked"] == 0


@respx.mock
def test_inference_down_fails_open(tmp_path):
    respx.post("http://inf:9000/predict").mock(side_effect=httpx.ConnectError("x"))
    respx.get("http://up:8000/search").mock(return_value=httpx.Response(
        200, text="ok"))
    st = EventStore(str(tmp_path / "e.db"))
    s = Settings(inference_url="http://inf:9000", upstream_url="http://up:8000",
                 events_db=str(tmp_path / "e.db"))
    c = TestClient(create_app(s, st))
    r = c.get("/search?q=' OR 1=1 -- ")
    assert r.status_code == 200
    assert st.recent()[0]["blocked"] == 0
