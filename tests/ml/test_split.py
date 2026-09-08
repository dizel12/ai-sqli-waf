from ml.data.split import remove_near_duplicates, source_disjoint_split


def _rows(pairs):
    return [{"text": t, "label": l, "source": s} for t, l, s in pairs]


def test_remove_near_duplicates_drops_trivial_variants():
    rows = _rows([
        ("SELECT * FROM users WHERE id=1", 1, "a"),
        ("SELECT * FROM users WHERE id=2", 1, "a"),   # near-dup of first
        ("completely different benign text here", 0, "b"),
    ])
    out = remove_near_duplicates(rows, threshold=0.8)
    texts = [r["text"] for r in out]
    assert "SELECT * FROM users WHERE id=1" in texts
    assert "SELECT * FROM users WHERE id=2" not in texts
    assert "completely different benign text here" in texts


def test_split_is_source_disjoint():
    rows = []
    for i in range(10):
        src = f"src{i}"
        for j in range(20):
            rows.append({"text": f"{src} sample {j}", "label": i % 2, "source": src})
    parts = source_disjoint_split(rows, seed=13)
    seen = {}
    for name, part in parts.items():
        for r in part:
            assert r["source"] not in seen or seen[r["source"]] == name
            seen[r["source"]] = name
    assert sum(len(p) for p in parts.values()) == len(rows)


def test_split_is_deterministic():
    rows = [{"text": f"s{i} t{j}", "label": 0, "source": f"s{i}"}
            for i in range(8) for j in range(15)]
    a = source_disjoint_split(rows, seed=13)
    b = source_disjoint_split(rows, seed=13)
    assert [r["text"] for r in a["train"]] == [r["text"] for r in b["train"]]
