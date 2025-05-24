import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time

# Suppression des warnings pour une sortie plus propre
warnings.filterwarnings('ignore')

# Import des classes personnalisées
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineering
from src.visualization import Visualizer
from src.model_builder import ModelBuilder
from src.recommender import Recommender
from src.evaluator import Evaluator

def main():
    """Point d'entrée principal du système de recommandation de jeux Steam."""
    
    start_time = time.time()
    print("===== STEAM GAME RECOMMENDATION SYSTEM =====")
    
    # 1. Chargement et prétraitement des données
    print("\n1. Loading and preprocessing data...")
    data_loader = DataLoader(data_path='data_backup/game_data.pkl')
    data_loader.create_output_directories()
    df = data_loader.load_data()
    df = data_loader.preprocess_data()
    
    # 2. Feature Engineering
    print("\n2. Performing feature engineering...")
    feature_engineering = FeatureEngineering(df)
    tfidf_matrix = feature_engineering.create_tfidf_vectors()
    cosine_sim = feature_engineering.calculate_similarity_matrix()
    
    # 3. Visualisations de base
    print("\n3. Creating basic visualizations...")
    visualizer = Visualizer(df)
    visualizer.plot_top_tags()
    visualizer.create_wordcloud()
    visualizer.plot_top_developers()
    visualizer.plot_similarity_distribution(cosine_sim)
    
    # 4. Construction et entrainement des modèles
    print("\n4. Building and training models...")
    model_builder = ModelBuilder(df, tfidf_matrix)
    
    # Réduction de dimensionnalité avec PCA
    reduced_features = model_builder.build_pca_model(n_components=50)
    
    # Trouver le nombre optimal de clusters
    wcss = model_builder.find_optimal_clusters(min_clusters=2, max_clusters=15)
    visualizer.plot_elbow_method(wcss)
    
    # Clustering avec K-means
    optimal_clusters = 8  # Basé sur la méthode du coude
    cluster_labels = model_builder.build_kmeans_model(n_clusters=optimal_clusters)
    
    # Visualisation des clusters avec t-SNE
    tsne_results = model_builder.build_tsne_model(perplexity=30)
    tsne_df = visualizer.visualize_clusters(reduced_features)
    visualizer.create_cluster_wordclouds()
    
    # 5. Système de recommandation
    print("\n5. Testing recommendation system...")
    recommender = Recommender(df, cosine_sim)
    
    # Tester des recommandations sur quelques jeux
    sample_games = df['Name'].sample(3).tolist()  # Prendre 3 jeux aléatoires comme exemples
    for game in sample_games:
        recommender.display_recommendations(game, n=5)
    
    # 6. Évaluation du système
    print("\n6. Evaluating recommendation system...")
    evaluator = Evaluator(df, cosine_sim, recommender)
    
    # Évaluer avec différentes valeurs de k
    k_values = [5, 10, 15, 20]
    print("\nEvaluating recommendation accuracy using different k values:")
    print("K\tPrecision\tRecall\t\tF1 Score\tNDCG")
    print("-" * 60)
    
    for k in k_values:
        precision, recall, f1, ndcg = evaluator.evaluate_recommendations(k=k, n_splits=5)
        print(f"{k}\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}\t\t{ndcg:.4f}")
    
    # Évaluation de la qualité des recommandations
    tag_overlaps, dev_consistencies = evaluator.evaluate_content_quality(sample_size=10)
    visualizer.plot_recommendation_quality(tag_overlaps)
    
    # Évaluation de la prédiction de clusters
    y_true, y_pred, cluster_accuracy = evaluator.evaluate_cluster_prediction(sample_size=len(df))
    visualizer.plot_confusion_matrix(y_true, y_pred)
    
    # 7. Enregistrement des modèles et des données
    print("\n7. Saving models and processed data...")
    model_builder.save_models()
    feature_engineering.save_models()
    data_loader.save_processed_data()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n===== STEAM GAME RECOMMENDATION SYSTEM COMPLETED =====")
    print(f"All models, data, and visualizations saved successfully.")
    print(f"The system can recommend games with an average tag overlap of {np.mean(tag_overlaps):.4f}")
    print(f"Cluster prediction accuracy: {cluster_accuracy:.4f}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()