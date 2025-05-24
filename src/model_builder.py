import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import os

class ModelBuilder:
    """Classe pour créer et entraîner les modèles de clustering et de dimensionnalité."""
    
    def __init__(self, df=None, tfidf_matrix=None):
        """
        Initialise la classe ModelBuilder.
        
        Args:
            df: DataFrame Pandas contenant les données des jeux
            tfidf_matrix: Matrice TF-IDF des caractéristiques des jeux
        """
        self.df = df
        self.tfidf_matrix = tfidf_matrix
        self.pca_model = None
        self.reduced_features = None
        self.kmeans_model = None
        self.tsne_model = None
        self.tsne_results = None
        
    def build_pca_model(self, n_components=50):
        """
        Construit et entraîne un modèle PCA pour la réduction de dimensionnalité.
        
        Args:
            n_components: Nombre de composantes principales à garder
            
        Returns:
            Les caractéristiques réduites
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not provided")
        
        print(f"Building PCA model with {n_components} components...")
        self.pca_model = PCA(n_components=n_components)
        self.reduced_features = self.pca_model.fit_transform(self.tfidf_matrix.toarray())
        
        # Afficher la variance expliquée
        explained_variance = self.pca_model.explained_variance_ratio_.sum()
        print(f"Total explained variance: {explained_variance:.4f}")
        
        return self.reduced_features
    
    def build_kmeans_model(self, n_clusters=8):
        """
        Construit et entraîne un modèle K-means pour le clustering des jeux.
        
        Args:
            n_clusters: Nombre de clusters à créer
            
        Returns:
            Les labels de cluster assignés à chaque jeu
        """
        if self.reduced_features is None:
            raise ValueError("Reduced features not available. Call build_pca_model first.")
        
        print(f"Building KMeans model with {n_clusters} clusters...")
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(self.reduced_features)
        
        # Ajouter les labels de cluster au DataFrame
        if self.df is not None:
            self.df['cluster'] = cluster_labels
            
            # Analyser la distribution des clusters
            cluster_counts = self.df['cluster'].value_counts().sort_index()
            print("\nCluster distribution:")
            for cluster, count in cluster_counts.items():
                print(f"Cluster {cluster}: {count} games")
        
        return cluster_labels
    
    def find_optimal_clusters(self, min_clusters=2, max_clusters=15):
        """
        Trouve le nombre optimal de clusters en utilisant la méthode du coude.
        
        Args:
            min_clusters: Nombre minimum de clusters à tester
            max_clusters: Nombre maximum de clusters à tester
            
        Returns:
            Liste des valeurs WCSS pour chaque nombre de clusters
        """
        if self.reduced_features is None:
            raise ValueError("Reduced features not available. Call build_pca_model first.")
        
        print("Finding optimal number of clusters...")
        wcss = []
        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(self.reduced_features)
            wcss.append(kmeans.inertia_)
            print(f"  Clusters: {n_clusters}, WCSS: {kmeans.inertia_:.2f}")
        
        return wcss
    
    def build_tsne_model(self, perplexity=30):
        """
        Construit et entraîne un modèle t-SNE pour la visualisation des clusters.
        
        Args:
            perplexity: Paramètre de perplexité pour t-SNE
            
        Returns:
            Les résultats t-SNE en 2D
        """
        if self.reduced_features is None:
            raise ValueError("Reduced features not available. Call build_pca_model first.")
        
        print(f"Building t-SNE model with perplexity={perplexity}...")
        self.tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        self.tsne_results = self.tsne_model.fit_transform(self.reduced_features)
        
        return self.tsne_results
    
    def save_models(self, models_dir='models'):
        """
        Sauvegarde tous les modèles entraînés.
        
        Args:
            models_dir: Répertoire où sauvegarder les modèles
        """
        os.makedirs(models_dir, exist_ok=True)
        
        # Sauvegarder le modèle PCA
        if self.pca_model is not None:
            with open(f"{models_dir}/pca_model.pkl", 'wb') as f:
                pickle.dump(self.pca_model, f)
            print(f"PCA model saved to {models_dir}/pca_model.pkl")
        
        # Sauvegarder le modèle KMeans
        if self.kmeans_model is not None:
            with open(f"{models_dir}/kmeans_model.pkl", 'wb') as f:
                pickle.dump(self.kmeans_model, f)
            print(f"KMeans model saved to {models_dir}/kmeans_model.pkl")
        
        # Sauvegarder les caractéristiques réduites
        if self.reduced_features is not None:
            with open(f"{models_dir}/reduced_features.pkl", 'wb') as f:
                pickle.dump(self.reduced_features, f)
            print(f"Reduced features saved to {models_dir}/reduced_features.pkl")
        
        # Sauvegarder les résultats t-SNE
        if self.tsne_results is not None:
            with open(f"{models_dir}/tsne_results.pkl", 'wb') as f:
                pickle.dump(self.tsne_results, f)
            print(f"t-SNE results saved to {models_dir}/tsne_results.pkl")