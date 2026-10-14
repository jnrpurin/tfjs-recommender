# E-commerce Recommendation System

A web application that displays user profiles and product listings, with the ability to track user purchases for future machine learning recommendations using TensorFlow.js.

## Fork from UniPDS repo

- Project link to my studies [UniPDS - Software Engineering with Applied AI](https://github.com/unipds-engenharia-de-ia-aplicada/engenharia-de-software-com-ia-aplicada/tree/main/modulo01-fundamentos-de-ia-e-llms-para-programadores/exemplo-01-ecommerce-recomendations-z/parte05-ecommerce-recomendations-with-tensorflow)

## Project Structure

- `index.html` - Main HTML file for the application
- `style.css` - Application styles
- `src/index.js` - Entry point for the application
- `src/view/` - Classes for managing the DOM, plus HTML templates in `src/view/templates/`
- `src/controller/` - Controllers connecting views, services and the training worker
- `src/service/` - Business logic for data handling
- `src/events/` - Event names and the event bus shared between UI and worker
- `src/workers/modelTrainingWorker.js` - Web Worker that encodes data and trains the TensorFlow.js model
- `data/` - JSON files with user and product data

## Setup and Run

1. Install dependencies:
```
npm install
```

2. Start the application:
```
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Features

- User profile selection with details display
- Past purchase history display
- Product listing with "Buy Now" functionality
- Purchase tracking using sessionStorage

## Future Enhancements

- Replace static JSON files with a SQLite database seeded from the [Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) dataset (B2B wholesale orders)
- Node.js API (Express) serving stores, items and order history to the front end
- Use ids instead of names to identify items
- Negative sampling instead of all user × product pairs, and `tf.tidy()` to avoid tensor leaks
- Train/test split and precision@10 evaluation against a "best sellers" baseline
- (Optional) Train with `@tensorflow/tfjs-node` and load the saved model in the browser
