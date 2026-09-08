from waf_proxy.events import EventStore


def test_record_and_recent(tmp_path):
    st = EventStore(str(tmp_path / "e.db"))
    rid = st.record(method="GET", path="/search", param="q",
                    value="' OR 1=1 -- ", scores={"cnn": 0.9},
                    active_model="cnn", score=0.9, threshold=0.5, blocked=True,
                    secure_mode=False)
    assert rid == 1
    rows = st.recent()
    assert rows[0]["blocked"] == 1
    assert rows[0]["param"] == "q"
    assert rows[0]["scores"] == {"cnn": 0.9}
    assert rows[0]["threshold"] == 0.5


def test_value_is_truncated(tmp_path):
    st = EventStore(str(tmp_path / "e.db"))
    st.record(method="GET", path="/x", param="q", value="a" * 500,
              scores={}, active_model="cnn", score=0.0, threshold=0.5,
              blocked=False, secure_mode=True)
    assert len(st.recent()[0]["value"]) == 200


def test_after_returns_new_rows_oldest_first(tmp_path):
    st = EventStore(str(tmp_path / "e.db"))
    for i in range(5):
        st.record(method="GET", path=f"/{i}", param="q", value="xyz",
                  scores={}, active_model="cnn", score=0.0, threshold=0.5,
                  blocked=False, secure_mode=False)
    rows = st.after(2)
    assert [r["id"] for r in rows] == [3, 4, 5]
