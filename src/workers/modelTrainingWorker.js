import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
let _globalCtx = {};
let _model = null

// Peso de cada feature no vetor (quanto ela "pesa" na comparação)
const WEIGHTS = {
    // item
    category: 0.4,
    popularity: 0.3,
    price: 0.2,
    // loja
    country: 0.1,
    orders: 0.2,
    ticket: 0.2,
};

// 🎯 Negative sampling: em vez de todos os pares loja × item (milhões),
// usamos até MAX_POSITIVES compras por loja (label 1) e, para cada uma,
// NEGATIVES_PER_POSITIVE itens sorteados que a loja não comprou (label 0)
const MAX_POSITIVES = 50
const NEGATIVES_PER_POSITIVE = 4

// países com mais lojas viram one-hot; o resto cai em OTHER
const TOP_COUNTRIES = 10

// 🔢 Normalize continuous values to 0–1 range
// Why? Keeps all features balanced so no one dominates training
// Formula: (val - min) / (max - min)
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

// Preço, popularidade, pedidos e ticket têm outliers enormes
// (ex.: ticket médio de 25.784 numa loja e ~300 na maioria).
// O log "achata" a escala antes de normalizar.
const logNormalize = (value, min, max) =>
    normalize(Math.log1p(value), Math.log1p(min), Math.log1p(max))

const minMax = values => ({ min: Math.min(...values), max: Math.max(...values) })

const uniquePurchaseIds = user => [...new Set(user.purchases.map(p => p.id))]

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]]
    }
    return array
}

function makeContext(products, users) {
    const categories = [...new Set(products.map(p => p.category))]
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    )

    const storesByCountry = {}
    users.forEach(user => {
        storesByCountry[user.country] = (storesByCountry[user.country] || 0) + 1
    })
    const countries = [
        ...Object.entries(storesByCountry)
            .sort((a, b) => b[1] - a[1])
            .slice(0, TOP_COUNTRIES)
            .map(([country]) => country),
        'OTHER'
    ]
    const countriesIndex = Object.fromEntries(
        countries.map((country, index) => [country, index])
    )

    const productDimentions = 2 + categories.length // price + popularity + categories

    return {
        products,
        users,
        categoriesIndex,
        countriesIndex,
        price: minMax(products.map(p => p.price)),
        popularity: minMax(products.map(p => p.popularity)),
        orders: minMax(users.map(u => u.orderCount)),
        ticket: minMax(users.map(u => u.avgTicket)),
        numCategories: categories.length,
        numCountries: countries.length,
        productDimentions,
        // média dos itens comprados + country + orders + ticket
        userDimentions: productDimentions + countries.length + 2,
    }
}

const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

// Retorna um array JS: os tensores intermediários são liberados pelo tf.tidy
function encodeProduct(product, context) {
    return tf.tidy(() => {
        const price = tf.tensor1d([
            logNormalize(product.price, context.price.min, context.price.max) * WEIGHTS.price
        ])

        const popularity = tf.tensor1d([
            logNormalize(product.popularity, context.popularity.min, context.popularity.max) * WEIGHTS.popularity
        ])

        const category = oneHotWeighted(
            context.categoriesIndex[product.category],
            context.numCategories,
            WEIGHTS.category
        )

        return tf.concat1d([price, popularity, category]).arraySync()
    })
}

function encodeUser(user, context) {
    return tf.tidy(() => {
        const purchaseIds = uniquePurchaseIds(user)

        // perfil de compra = média dos vetores dos itens comprados
        // (loja sem compras: zeros, sobram só as features da própria loja)
        const purchasesProfile = purchaseIds.length ?
            tf.tensor2d(purchaseIds.map(id => context.productVectorsById[id])).mean(0) :
            tf.zeros([context.productDimentions])

        const country = oneHotWeighted(
            context.countriesIndex[user.country] ?? context.countriesIndex.OTHER,
            context.numCountries,
            WEIGHTS.country
        )

        const orders = tf.tensor1d([
            logNormalize(user.orderCount, context.orders.min, context.orders.max) * WEIGHTS.orders
        ])

        const ticket = tf.tensor1d([
            logNormalize(user.avgTicket, context.ticket.min, context.ticket.max) * WEIGHTS.ticket
        ])

        return tf.concat1d([purchasesProfile, country, orders, ticket]).arraySync()
    })
}

function createTrainingData(context) {
    const inputs = []
    const labels = []
    const productIds = context.products.map(p => p.id)

    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            const userVector = encodeUser(user, context)
            const purchased = new Set(uniquePurchaseIds(user))
            const positives = shuffle([...purchased]).slice(0, MAX_POSITIVES)

            const addPair = (productId, label) => {
                // combinar user + product
                inputs.push([...userVector, ...context.productVectorsById[productId]])
                labels.push(label)
            }

            positives.forEach(productId => {
                addPair(productId, 1)

                for (let n = 0; n < NEGATIVES_PER_POSITIVE; n++) {
                    let negativeId
                    do {
                        negativeId = productIds[Math.floor(Math.random() * productIds.length)]
                    } while (purchased.has(negativeId))
                    addPair(negativeId, 0)
                }
            })
        })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.userDimentions + context.productDimentions
        // tamanho = userVector + productVector
    }
}

// ====================================================================
// 📌 Exemplo de como uma loja é ANTES da codificação
// ====================================================================
/*
const exampleStore = {
    id: 12347,
    name: 'Store 12347',
    country: 'Iceland',
    orderCount: 5,
    avgTicket: 615.71,
    purchases: [
        { id: 1, name: '15CM CHRISTMAS GLASS BALL 20 LIGHTS', category: 'CHRISTMAS', price: 7.37 },
        { id: 8, name: 'FANCY FONT HOME SWEET HOME DOORMAT', category: 'DECOR', price: 6.24 }
    ]
};
*/

// ====================================================================
// 📌 Após a codificação, o modelo NÃO vê nomes ou palavras.
// Ele vê um VETOR NUMÉRICO (todos normalizados entre 0–1).
//
// Vetor do item:
// [preço_log_normalizado, popularidade_log_normalizada, categoria_one_hot...]
//
// Vetor da loja:
// [média dos vetores dos itens comprados..., país_one_hot..., pedidos, ticket]
//
// Entrada da rede = vetor da loja + vetor do item.
// São esses números que vão para a rede neural.
// ====================================================================



// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential()
    // Camada de entrada
    // - inputShape: Número de features por exemplo de treino (trainData.inputDim)
    //   Exemplo: Se o vetor produto + usuário = 20 números, então inputDim = 20
    // - units: 128 neurônios (muitos "olhos" para detectar padrões)
    // - activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    )
    // Camada oculta 1
    // - 64 neurônios (menos que a primeira camada: começa a comprimir informação)
    // - activation: 'relu' (ainda extraindo combinações relevantes de features)
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    // Camada oculta 2
    // - 32 neurônios (mais estreita de novo, destilando as informações mais importantes)
    //   Exemplo: De muitos sinais, mantém apenas os padrões mais fortes
    // - activation: 'relu'
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    // Camada de saída
    // - 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // - activation: 'sigmoid' comprime o resultado para o intervalo 0–1
    //   Exemplo: 0.9 = recomendação forte, 0.1 = recomendação fraca
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

    // Com milhares de exemplos, poucos epochs e batches maiores bastam
    // (100 epochs com batch 32 levariam muitos minutos no navegador)
    await model.fit(trainData.xs, trainData.ys, {
        epochs: 15,
        batchSize: 256,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model
}
async function trainModel({ users }) {
    console.log('Training model with users:', users.length);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
    const products = await (await fetch('/api/products')).json()

    const context = makeContext(products, users)
    context.productVectorsById = Object.fromEntries(
        products.map(product => [product.id, encodeProduct(product, context)])
    )
    // mesma ordem de products: usada para pontuar todos os itens de uma vez
    context.productMatrix = tf.tensor2d(products.map(product => context.productVectorsById[product.id]))

    _globalCtx.productMatrix?.dispose()
    _globalCtx = context

    const trainData = createTrainingData(context)
    console.log(`Training examples: ${trainData.ys.shape[0]}, input size: ${trainData.inputDimention}`)

    _model?.dispose()
    let start = performance.now()
    _model = await configureNeuralNetAndTrain(trainData)
    console.log(`Training: ${((performance.now() - start) / 1000).toFixed(1)}s`)

    // libera a memória dos dados de treino
    trainData.xs.dispose()
    trainData.ys.dispose()

    start = performance.now()
    const evaluation = evaluate(context)
    console.log(`Evaluation (${((performance.now() - start) / 1000).toFixed(1)}s):`, JSON.stringify(evaluation))

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete, evaluation });
}

// ====================================================================
// 📏 Avaliação: precision@K
// O modelo treinou só com compras ANTES da data de corte (2011-09-01).
// Para cada loja, testPurchases são os itens que ela comprou pela
// primeira vez DEPOIS do corte. Das K primeiras recomendações,
// quantas a loja realmente comprou depois?
//
// Baseline: recomendar os mais populares que a loja ainda não comprou.
// Se o modelo não ganhar da baseline, as features precisam melhorar.
// ====================================================================
const EVAL_K = 10

const precisionAtK = (rankedIds, relevantIds) =>
    rankedIds.slice(0, EVAL_K).filter(id => relevantIds.has(id)).length / EVAL_K

function evaluate(context) {
    const stores = context.users.filter(u => u.purchases.length && u.testPurchases?.length)
    const byPopularity = context.products.toSorted((a, b) => b.popularity - a.popularity)

    let model = 0
    let baseline = 0
    stores.forEach(user => {
        const relevantIds = new Set(user.testPurchases)
        const purchased = new Set(uniquePurchaseIds(user))

        model += precisionAtK(rankCandidates(user, context).map(p => p.id), relevantIds)
        baseline += precisionAtK(
            byPopularity.filter(p => !purchased.has(p.id)).map(p => p.id),
            relevantIds
        )
    })

    return {
        k: EVAL_K,
        stores: stores.length,
        model: stores.length ? model / stores.length : 0,
        baseline: stores.length ? baseline / stores.length : 0,
    }
}

// Pontua com o modelo todos os itens que a loja ainda não comprou
// (é o que interessa para descobrir produtos novos) e ordena do maior para o menor
function rankCandidates(user, context) {
    // 1️⃣ Converta a loja fornecida no vetor de features codificadas
    //    (média dos itens comprados, país, pedidos e ticket)
    //    Isso transforma as informações da loja no mesmo formato numérico
    //    que foi usado para treinar o modelo.

    const userVector = encodeUser(user, context)

    // Em aplicações reais:
    //  Armazene todos os vetores de produtos em um banco de dados vetorial (como Postgres, Neo4j ou Pinecone)
    //  Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
    //  Execute _model.predict() apenas nesses produtos

    // 2️⃣ Crie pares de entrada (loja + item) para todos os itens de uma vez:
    //    o vetor da loja é repetido (tile) ao lado da matriz de produtos,
    //    que já está pronta no contexto. O tf.tidy libera os tensores ao final.
    //    Formato: [numProdutos, inputDim]
    const scores = tf.tidy(() => {
        const userRows = tf.tensor2d([userVector]).tile([context.products.length, 1])
        const inputTensor = tf.concat([userRows, context.productMatrix], 1)
        return _model.predict(inputTensor).dataSync()
    })

    // 3️⃣ Candidatos: só itens que a loja ainda não comprou
    const purchased = new Set(uniquePurchaseIds(user))
    return context.products
        .map((product, index) => ({
            ...product,
            score: scores[index] // previsão do modelo para este produto
        }))
        .filter(product => !purchased.has(product.id))
        .sort((a, b) => b.score - a.score)
}

function recommend({ user }) {
    if (!_model) return;
    const sortedItems = rankCandidates(user, _globalCtx)

    // 4️⃣ Envie a lista ordenada de produtos recomendados
    //    para a thread principal (a UI pode exibi-los agora).
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });

}
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
