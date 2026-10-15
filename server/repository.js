import Database from 'better-sqlite3';
import { fileURLToPath } from 'node:url';

const DB = fileURLToPath(new URL('../data/db/store.db', import.meta.url));

// orders before this date are history (training); later ones are used only for evaluation
export const CUTOFF = '2011-09-01';

const db = new Database(DB, { readonly: true, fileMustExist: true });

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

export function getStores(limit) {
    return storesQuery.all({ cutoff: CUTOFF, limit }).map(store => ({
        ...store,
        name: `Store ${store.id}`,
        purchases: purchasesQuery.all({ storeId: store.id, cutoff: CUTOFF }),
        testPurchases: testPurchasesQuery.all({ storeId: store.id, cutoff: CUTOFF }),
    }));
}
