import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { fileURLToPath } from 'node:url';

const DB = fileURLToPath(new URL('../data/db/store.db', import.meta.url));

// orders before this date are history (training); later ones are used only for evaluation
export const CUTOFF = '2011-09-01';

// read-only connection for the catalog and order queries
const db = new Database(DB, { readonly: true, fileMustExist: true });
sqliteVec.load(db);

// separate connection used only to (re)write the vector table after each training
const writeDb = new Database(DB, { fileMustExist: true });
sqliteVec.load(writeDb);

// popularity = number of distinct stores that bought the item before the cutoff
const productsQuery = db.prepare(`
    SELECT i.id, i.description AS name, i.category, i.unit_price AS price,
           COALESCE(p.popularity, 0) AS popularity
    FROM items i
    LEFT JOIN (
        SELECT l.item_id, COUNT(DISTINCT o.store_id) AS popularity
        FROM order_lines l
        JOIN orders o ON o.id = l.order_id
        WHERE o.order_date < @cutoff
        GROUP BY l.item_id
    ) p ON p.item_id = i.id
    ORDER BY i.id`);

const storesQuery = db.prepare(`
    SELECT s.id, s.country,
           COUNT(DISTINCT o.id) AS orderCount,
           ROUND(SUM(l.quantity * l.price) / COUNT(DISTINCT o.id), 2) AS avgTicket
    FROM stores s
    JOIN orders o ON o.store_id = s.id AND o.order_date < @cutoff
    JOIN order_lines l ON l.order_id = o.id
    GROUP BY s.id
    ORDER BY s.id
    LIMIT @limit`);

const purchasesQuery = db.prepare(`
    SELECT DISTINCT i.id, i.description AS name, i.category, i.unit_price AS price
    FROM orders o
    JOIN order_lines l ON l.order_id = o.id
    JOIN items i ON i.id = l.item_id
    WHERE o.store_id = @storeId AND o.order_date < @cutoff
    ORDER BY i.id`);

// items the store bought for the first time after the cutoff
const testPurchasesQuery = db.prepare(`
    SELECT DISTINCT l.item_id AS id
    FROM orders o
    JOIN order_lines l ON l.order_id = o.id
    WHERE o.store_id = @storeId AND o.order_date >= @cutoff
      AND l.item_id NOT IN (
          SELECT l2.item_id
          FROM orders o2
          JOIN order_lines l2 ON l2.order_id = o2.id
          WHERE o2.store_id = @storeId AND o2.order_date < @cutoff
      )
    ORDER BY l.item_id`).pluck();

// the database is read-only at runtime, so the slow popularity aggregation (~1.4s) runs once
let products;
export function getProducts() {
    return products ??= productsQuery.all({ cutoff: CUTOFF });
}

// Vector table (sqlite-vec): one row per product, rowid = item id.
// It is recreated on every training because the vector size depends on the
// number of categories, and cosine distance is what the nearest-items search uses.
export const saveProductVectors = writeDb.transaction(vectors => {
    const dimensions = vectors[0].vector.length;
    writeDb.exec('DROP TABLE IF EXISTS vec_products');
    writeDb.exec(`CREATE VIRTUAL TABLE vec_products USING vec0(embedding float[${dimensions}] distance_metric=cosine)`);

    const insert = writeDb.prepare('INSERT INTO vec_products (rowid, embedding) VALUES (?, ?)');
    vectors.forEach(({ id, vector }) => insert.run(BigInt(id), new Float32Array(vector)));
    return { count: vectors.length, dimensions };
});

export function getStores(limit) {
    return storesQuery.all({ cutoff: CUTOFF, limit }).map(store => ({
        ...store,
        name: `Store ${store.id}`,
        purchases: purchasesQuery.all({ storeId: store.id, cutoff: CUTOFF }),
        testPurchases: testPurchasesQuery.all({ storeId: store.id, cutoff: CUTOFF }),
    }));
}
