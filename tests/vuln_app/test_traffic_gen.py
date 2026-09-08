from vuln_app.traffic_gen import build_requests, SQL_LOOKING_BENIGN


def test_build_requests_count_and_shape():
    reqs = build_requests(40, seed=1)
    assert len(reqs) == 40
    for method, path, payload in reqs:
        assert method in {"GET", "POST"}
        assert path in {"/search", "/product", "/login"}
        assert isinstance(payload, dict)


def test_build_requests_includes_sql_looking_values():
    reqs = build_requests(200, seed=2)
    flat = " ".join(str(p) for _, _, p in reqs)
    assert any(v in flat for v in SQL_LOOKING_BENIGN)


def test_build_requests_deterministic():
    assert build_requests(30, seed=5) == build_requests(30, seed=5)
