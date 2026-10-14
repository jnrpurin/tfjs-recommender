CREATE TABLE stores (
  id      INTEGER PRIMARY KEY,
  country TEXT NOT NULL
);

CREATE TABLE items (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  stock_code  TEXT NOT NULL UNIQUE,
  description TEXT,
  category    TEXT NOT NULL,
  unit_price  REAL
);

CREATE TABLE orders (
  id         TEXT PRIMARY KEY,
  store_id   INTEGER NOT NULL REFERENCES stores(id),
  order_date TEXT NOT NULL
);

CREATE TABLE order_lines (
  order_id TEXT    NOT NULL REFERENCES orders(id),
  item_id  INTEGER NOT NULL REFERENCES items(id),
  quantity INTEGER NOT NULL,
  price    REAL    NOT NULL,
  PRIMARY KEY (order_id, item_id)
);

CREATE INDEX ix_orders_store ON orders(store_id);
CREATE INDEX ix_lines_item   ON order_lines(item_id);
