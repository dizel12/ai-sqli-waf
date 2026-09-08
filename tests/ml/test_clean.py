from ml.data.clean import (
    normalize_value, is_inspectable, extract_values_from_qs, dedupe_exact,
)


def test_normalize_url_decodes_once():
    assert normalize_value("%27%20OR%201%3D1") == "' OR 1=1"


def test_normalize_collapses_whitespace_and_strips():
    assert normalize_value("  a\t\tb\n c  ") == "a b c"


def test_normalize_blank_becomes_empty():
    assert normalize_value("   \t ") == ""


def test_is_inspectable_min_len():
    assert is_inspectable("abc") is True
    assert is_inspectable("ab") is False
    assert is_inspectable("  x  ") is False


def test_extract_values_from_qs_keeps_order_drops_empty():
    assert extract_values_from_qs("q=mouse&sort=&page=2") == ["mouse", "2"]


def test_dedupe_exact_preserves_order():
    assert dedupe_exact(["a", "b", "a", "  b ", "c"]) == ["a", "b", "c"]
