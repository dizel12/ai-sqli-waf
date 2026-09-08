# INTENTIONALLY VULNERABLE - DO NOT DEPLOY
from __future__ import annotations


def search_sql(q: str, secure: bool):
    if secure:
        return ("SELECT id, name, category, price FROM products "
                "WHERE name LIKE ?", (f"%{q}%",))
    return (f"SELECT id, name, category, price FROM products "
            f"WHERE name LIKE '%{q}%'", ())


def login_sql(u: str, p: str, secure: bool):
    if secure:
        return ("SELECT id, username FROM users "
                "WHERE username = ? AND password = ?", (u, p))
    return (f"SELECT id, username FROM users "
            f"WHERE username = '{u}' AND password = '{p}'", ())


def product_sql(pid: str, secure: bool):
    if secure:
        return ("SELECT id, name, category, price FROM products "
                "WHERE id = ?", (int(pid),))
    return (f"SELECT id, name, category, price FROM products "
            f"WHERE id = {pid}", ())
