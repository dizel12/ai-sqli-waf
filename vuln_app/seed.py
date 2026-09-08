# INTENTIONALLY VULNERABLE - DO NOT DEPLOY
from __future__ import annotations
import sqlite3

_PRODUCTS = [
    ("Wireless Mouse", "peripherals", 19.99),
    ("Mechanical Keyboard", "peripherals", 89.00),
    ("Laptop Stand", "accessories", 34.50),
    ("USB-C Hub", "accessories", 45.00),
    ("Desk Lamp", "lighting", 22.00),
    ("27-inch Monitor", "displays", 249.99),
    ("HD Webcam", "peripherals", 59.95),
    ("Noise-Cancelling Headphones", "audio", 199.00),
    ("Bluetooth Speaker", "audio", 74.50),
    ("Ergonomic Chair", "furniture", 329.00),
    ("Standing Desk", "furniture", 459.00),
    ("Laptop Sleeve", "accessories", 24.99),
    ("Portable SSD 1TB", "storage", 119.00),
    ("MicroSD Card 256GB", "storage", 39.99),
    ("Gaming Mousepad", "peripherals", 14.99),
    ("Wireless Charger", "accessories", 29.99),
    ("Smart LED Bulb", "lighting", 17.49),
    ("Cable Organizer Kit", "accessories", 12.99),
    ("Mechanical Numpad", "peripherals", 44.00),
    ("Studio Microphone", "audio", 129.00),
    ("Graphics Tablet", "peripherals", 89.99),
    ("Docking Station", "accessories", 179.00),
    ("Surge Protector", "power", 27.95),
    ("Laptop Cooling Pad", "accessories", 32.50),
    ("4K Action Camera", "cameras", 279.00),
]
_USERS = [
    ("admin", "s3cr3t-admin"),
    ("alice", "password123"),
    ("bob", "hunter2"),
]


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "DROP TABLE IF EXISTS products; DROP TABLE IF EXISTS users;"
        "CREATE TABLE products(id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " name TEXT, category TEXT, price REAL);"
        "CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " username TEXT, password TEXT);")
    conn.executemany("INSERT INTO products(name,category,price) VALUES(?,?,?)",
                     _PRODUCTS)
    conn.executemany("INSERT INTO users(username,password) VALUES(?,?)", _USERS)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    import sys
    init_db(sys.argv[1] if len(sys.argv) > 1 else "app.db")
