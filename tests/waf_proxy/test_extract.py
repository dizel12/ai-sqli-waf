import json

from waf_proxy.extract import extract_candidates


def test_extracts_query_values():
    c = extract_candidates("GET", "q=mouse&x=ab&p=hello", {}, b"", 3)
    vals = {x.value for x in c}
    assert "mouse" in vals and "hello" in vals
    assert "ab" not in vals  # too short


def test_extracts_form_body():
    body = b"username=admin%27%20--%20&password=x"
    hdrs = {"content-type": "application/x-www-form-urlencoded"}
    c = extract_candidates("POST", "", hdrs, body, 3)
    vals = {x.value for x in c}
    assert "admin' --" in vals
    assert all(x.location in {"query", "form"} for x in c)


def test_extracts_json_scalars_recursively():
    body = json.dumps({"a": "select desk lamp",
                       "nested": {"b": "' OR 1=1 -- ", "n": 5}}).encode()
    hdrs = {"content-type": "application/json"}
    c = extract_candidates("POST", "", hdrs, body, 3)
    vals = {x.value for x in c}
    assert "select desk lamp" in vals
    assert "' OR 1=1 --" in vals


def test_extracts_cookie_values():
    hdrs = {"cookie": "sid=abc123def; theme=dark"}
    c = extract_candidates("GET", "", hdrs, b"", 3)
    vals = {x.value for x in c}
    assert "abc123def" in vals
