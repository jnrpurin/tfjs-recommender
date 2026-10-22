import express from 'express';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { getProducts, getStores, saveProductVectors } from './repository.js';

const PORT = process.env.PORT || 3000;
const root = path => fileURLToPath(new URL(`../${path}`, import.meta.url));
const MODEL_DIR = root('data/model');

const app = express();
app.use(express.json({ limit: '10mb' }));

// only the front-end files are public: data/db, data/raw and node_modules stay private
app.get('/', (req, res) => res.sendFile(root('index.html')));
app.get('/style.css', (req, res) => res.sendFile(root('style.css')));
app.use('/src', express.static(root('src')));

// saved model (model.json + weights.bin) and encoder context, read by the worker on page load
app.use('/model', express.static(MODEL_DIR));

app.get('/api/products', (req, res) => {
    res.json(getProducts());
});

app.get('/api/users', (req, res) => {
    const limit = Math.min(Math.max(parseInt(req.query.limit) || 100, 1), 1000);
    res.json(getStores(limit));
});

// Receives the artifacts from model.save() in the worker and writes them in the
// TensorFlow.js "layers-model" format, so tf.loadLayersModel('/model/model.json') can read it back
app.post('/api/model', (req, res) => {
    const { modelTopology, weightSpecs, weightData, encoder } = req.body;

    fs.mkdirSync(MODEL_DIR, { recursive: true });
    fs.writeFileSync(`${MODEL_DIR}/weights.bin`, Buffer.from(weightData, 'base64'));
    fs.writeFileSync(`${MODEL_DIR}/model.json`, JSON.stringify({
        format: 'layers-model',
        modelTopology,
        weightsManifest: [{ paths: ['weights.bin'], weights: weightSpecs }],
    }));
    fs.writeFileSync(`${MODEL_DIR}/encoder.json`, JSON.stringify(encoder));

    res.json({ saved: true });
});

app.post('/api/product-vectors', (req, res) => {
    res.json(saveProductVectors(req.body));
});

app.listen(PORT, () => console.log(`http://localhost:${PORT}`));
