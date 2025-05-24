team Game Recommendation System
📋 Overview
This project implements a sophisticated recommendation system for Steam games utilizing machine learning techniques. The system analyzes game descriptions, tags, developers, and other metadata to provide personalized game recommendations based on content similarity and user preferences.

🚀 Features
Data Collection: Automated scraping of Steam game information
Content Analysis: Text processing of game descriptions and tags
Similarity Engine: Measures game similarity using cosine similarity on TF-IDF vectors
Clustering: Groups similar games using K-Means clustering
Dimensionality Reduction: Visualizes game clusters using PCA and t-SNE
Comprehensive Evaluation: Multiple metrics including precision, recall, F1 score, NDCG and content overlap
Data Visualization: Interactive visualizations of game relationships and recommendation quality
🔍 Recommendation Methods
The system uses two primary recommendation approaches:

Content-Based Filtering: Recommends games based on similarity to games you already like
Cluster-Based Recommendations: Suggests games from the same cluster as your favorite games
📊 Performance Metrics
Our evaluation demonstrates strong recommendation performance:

Precision: 0.77 @k=5, 0.66 @k=15
Recall: 0.29 @k=5, 0.64 @k=15
F1 Score: 0.40 @k=5, 0.60 @k=15
NDCG: 0.81 @k=5, 0.80 @k=15
Average Tag Overlap: 0.32
Cluster Prediction Accuracy: 0.85

🧰 Project Structure
Steam-Game-Recommendation-System/
├── src/                        # Core modules
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature extraction and processing
│   ├── model_builder.py        # ML models for clustering
│   ├── recommender.py          # Recommendation engine
│   ├── evaluator.py            # System evaluation utilities
│   └── visualization.py        # Data visualization components
├── data/                       # Data storage
│   ├── steam_games.csv         # Processed game dataset
│   └── game_data.pkl           # Serialized game data
├── models/                     # Trained models
├── visualizations/             # Generated visualizations
├── main.py                     # Data collection script
├── newmain.py                  # Main application entry point
└── test.py                     # Unit tests



🏁 Getting Started
Clone the repository
Install dependencies: pip install -r requirements.txt
Run the data collection (optional): python main.py
Run the recommendation system: python newmain.py
✍️ Authors
Aziz Masmoudi PROJET DS
