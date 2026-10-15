import express from 'express';
import { fileURLToPath } from 'node:url';
import { getProducts, getStores } from './repository.js';

const PORT = process.env.PORT || 3000;
const root = path => fileURLToPath(new URL(`../${path}`, import.meta.url));

const app = express();

// only the front-end files are public: data/db, data/raw and node_modules stay private
app.get('/', (req, res) => res.sendFile(root('index.html')));
app.get('/style.css', (req, res) => res.sendFile(root('style.css')));
app.use('/src', express.static(root('src')));

// legacy JSON files, still used by the front end until it switches to the API (step 4)
app.get('/data/users.json', (req, res) => res.sendFile(root('data/users.json')));
app.get('/data/products.json', (req, res) => res.sendFile(root('data/products.json')));

app.get('/api/products', (req, res) => {
    res.json(getProducts());
});

app.get('/api/users', (req, res) => {
    const limit = Math.min(Math.max(parseInt(req.query.limit) || 100, 1), 1000);
    res.json(getStores(limit));
});

app.listen(PORT, () => console.log(`http://localhost:${PORT}`));
