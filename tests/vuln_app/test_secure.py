import pytest
from vuln_app.seed import init_db
from vuln_app.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "app.db")
    init_db(db)
    monkeypatch.setenv("SECURE_MODE", "1")
    app = create_app(db)
    app.config.update(TESTING=True)
    return app.test_client()


def test_secure_normal_search_still_works(client):
    r = client.get("/search?q=mouse")
    assert r.status_code == 200
    assert b"mouse" in r.data.lower()


def test_secure_union_select_returns_no_user_data(client):
    payload = "' UNION SELECT id, username, password, 0 FROM users -- "
    r = client.get("/search", query_string={"q": payload})
    assert r.status_code == 200
    assert b"s3cr3t-admin" not in r.data
    assert b"admin" not in r.data


def test_secure_login_bypass_fails(client):
    r = client.post("/login", data={"username": "admin' -- ", "password": "x"})
    assert r.status_code == 401


def test_secure_product_injection_is_rejected(client):
    r = client.get("/product", query_string={"id": "5 OR 1=1"})
    # int() coercion raises -> generic 500, never a boolean-true result set
    assert r.status_code == 500
    assert b"internal error" in r.data
