import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';
import { parse } from 'csv-parse';

// paths relative to this script, so it works no matter where `npm run seed` is called from
const CSV = new URL('../raw/online_retail_II.csv', import.meta.url);
const SCHEMA = new URL('./schema.sql', import.meta.url);
const DB_DIR = new URL('../db/', import.meta.url);
const DB = new URL('store.db', DB_DIR);

// service codes, not products: postage, manual, discount, carriage, samples, fees, tests, vouchers
const NON_PRODUCT = /^(POST|DOT|M|D|C\d|S|B|CRUK|PADS|ADJUST\d*|BANK CHARGES|AMAZONFEE|TEST\d*|GIFT|GIFT_.*)$/;

const CATEGORIES = [
    ['CHRISTMAS', /CHRISTMAS|XMAS/],
    ['CANDLE', /CANDLE|T-LIGHT|TEALIGHT/],
    ['BAG', /\bBAG/],
    ['KITCHEN', /MUG|CUP|PLATE|BOWL|JUG|TEAPOT|CAKE|BAKING|LUNCH BOX/],
    ['LIGHTING', /LIGHT|LANTERN|LAMP/],
    ['STATIONERY', /CARD|PAPER|NOTEBOOK|PENCIL|WRAP|STICKER/],
    ['DECOR', /SIGN|FRAME|HEART|CUSHION|DOORMAT|CLOCK|VASE/],
    ['TOYS', /TOY|GAME|DOLL|PUZZLE/],
];
const categoryOf = description =>
    (CATEGORIES.find(([, re]) => re.test(description.toUpperCase())) ?? ['OTHER'])[0];

// only numeric invoices: "C..." are cancellations, "A..." are bad-debt adjustments
const isValid = r =>
    r['Customer ID'] &&
    /^\d+$/.test(r.Invoice) &&
    Number(r.Quantity) > 0 &&
    Number(r.Price) > 0 &&
    !NON_PRODUCT.test(r.StockCode);

fs.mkdirSync(DB_DIR, { recursive: true });
for (const suffix of ['', '-wal', '-shm']) fs.rmSync(new URL(`store.db${suffix}`, DB_DIR), { force: true });
const db = new Database(fileURLToPath(DB));
db.pragma('journal_mode = WAL');
db.exec(fs.readFileSync(SCHEMA, 'utf8'));

const insStore = db.prepare('INSERT OR IGNORE INTO stores (id, country) VALUES (?, ?)');
const insItem = db.prepare('INSERT INTO items (stock_code, description, category) VALUES (?, ?, ?)');
const insOrder = db.prepare('INSERT OR IGNORE INTO orders (id, store_id, order_date) VALUES (?, ?, ?)');
const insLine = db.prepare(`
    INSERT INTO order_lines (order_id, item_id, quantity, price) VALUES (?, ?, ?, ?)
    ON CONFLICT (order_id, item_id) DO NOTHING`);

const itemIds = new Map();

const flush = db.transaction(rows => {
    for (const r of rows) {
        const storeId = parseInt(r['Customer ID']);
        insStore.run(storeId, r.Country);

        let itemId = itemIds.get(r.StockCode);
        if (!itemId) {
            itemId = insItem.run(r.StockCode, r.Description, categoryOf(r.Description)).lastInsertRowid;
            itemIds.set(r.StockCode, itemId);
        }

        insOrder.run(r.Invoice, storeId, r.InvoiceDate);
        insLine.run(r.Invoice, itemId, Number(r.Quantity), Number(r.Price));
    }
});

let buffer = [], total = 0, kept = 0;
const parser = fs.createReadStream(CSV).pipe(parse({ columns: true, trim: true }));
for await (const row of parser) {
    total++;
    row.StockCode = row.StockCode.toUpperCase(); // same item appears as "72349b" and "72349B"
    if (!isValid(row)) continue;
    kept++;
    buffer.push(row);
    if (buffer.length === 10_000) { flush(buffer); buffer = []; }
}
flush(buffer);

db.exec(`
    UPDATE items SET unit_price = (
        SELECT ROUND(SUM(price * quantity) / SUM(quantity), 2)
        FROM order_lines WHERE item_id = items.id
    )`);

const count = table => db.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get().n;
console.log(`linhas lidas: ${total} | válidas: ${kept}`);
console.log(`stores: ${count('stores')} | items: ${count('items')} | orders: ${count('orders')} | order_lines: ${count('order_lines')}`);
db.close();
