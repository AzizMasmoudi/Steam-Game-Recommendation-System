import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class FeatureEngineering:
    """Classe pour la création et transformation des features."""
    
    def __init__(self, df=None):
        """
        Initialise la classe FeatureEngineering.
        
        Args:
            df: DataFrame Pandas contenant les données des jeux
        """
        self.df = df
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.pca = None
        self.reduced_features = None
        self.kmeans = None
        
    def create_tfidf_vectors(self, max_features=5000):
        """Crée les vecteurs TF-IDF à partir des caractéristiques combinées."""
        if self.df is None:
            raise ValueError("DataFrame not provided. Initialize with a DataFrame or load data first.")
        
        print("Creating TF-IDF vectors...")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=max_features,
            ngram_range=(1, 2),  # Inclure les bigrammes
            min_df=2,  # Ignorer les termes qui apparaissent dans moins de 2 documents
            max_df=0.8  # Ignorer les termes qui apparaissent dans plus de 80% des documents
        )
        
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            print(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
            return self.tfidf_matrix
        except Exception as e:
            print(f"Erreur lors de la création des vecteurs TF-IDF: {e}")
            return None
    
    def calculate_similarity_matrix(self):
        """Calcule la matrice de similarité cosinus entre les jeux."""
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not created. Call create_tfidf_vectors() first.")
        
        print("Calculating cosine similarity matrix...")
        try:
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
            print(f"Cosine similarity matrix shape: {self.cosine_sim.shape}")
            
            # Statistiques sur la similarité
            similarities = self.cosine_sim.flatten()
            similarities = similarities[similarities < 0.999]  # Exclure les auto-similarités
            print(f"Average similarity: {np.mean(similarities):.4f}")
            print(f"Similarity std: {np.std(similarities):.4f}")
            
            return self.cosine_sim
        except Exception as e:
            print(f"Erreur lors du calcul de la similarité: {e}")
            return None
    
    def reduce_dimensions(self, n_components=50):
        """Réduit la dimensionnalité des vecteurs TF-IDF avec PCA."""
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not created. Call create_tfidf_vectors() first.")
        
        print(f"Reducing dimensions to {n_components} components with PCA...")
        try:
            # S'assurer que n_components ne dépasse pas le nombre de features
            max_components = min(n_components, self.tfidf_matrix.shape[1], self.tfidf_matrix.shape[0])
            
            self.pca = PCA(n_components=max_components, random_state=42)
            self.reduced_features = self.pca.fit_transform(self.tfidf_matrix.toarray())
            
            # Calculer la variance expliquée
            explained_variance = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            print(f"Reduced features shape: {self.reduced_features.shape}")
            print(f"Explained variance ratio: {cumulative_variance[-1]:.4f}")
            print(f"Top 5 components explain {cumulative_variance[4]:.4f} of variance")
            
            return self.reduced_features
        except Exception as e:
            print(f"Erreur lors de la réduction de dimensionnalité: {e}")
            return None
    
    def find_optimal_clusters(self, max_clusters=15):
        """Trouve le nombre optimal de clusters en utilisant la méthode du coude."""
        if self.reduced_features is None:
            raise ValueError("Features not reduced. Call reduce_dimensions() first.")
        
        print("Finding optimal number of clusters using the elbow method...")
        wcss = []
        
        try:
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(
                    n_clusters=i, 
                    init='k-means++', 
                    max_iter=300, 
                    n_init=10, 
                    random_state=42
                )
                kmeans.fit(self.reduced_features)
                wcss.append(kmeans.inertia_)
                
            print(f"WCSS calculated for {max_clusters} clusters")
            return wcss
        except Exception as e:
            print(f"Erreur lors de la recherche du nombre optimal de clusters: {e}")
            return None
    
    def cluster_games(self, n_clusters=8):
        """Regroupe les jeux en clusters avec K-means."""
        if self.reduced_features is None:
            raise ValueError("Features not reduced. Call reduce_dimensions() first.")
        
        print(f"Clustering games into {n_clusters} clusters...")
        try:
            self.kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=42,
                init='k-means++',
                max_iter=300,
                n_init=10
            )
            cluster_labels = self.kmeans.fit_predict(self.reduced_features)
            
            # Ajouter les clusters au DataFrame
            if self.df is not None:
                self.df['cluster'] = cluster_labels
                
                # Analyser la distribution des clusters
                cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                print("\nCluster distribution:")
                for i, count in cluster_counts.items():
                    print(f"Cluster {i}: {count} games ({count/len(cluster_labels)*100:.1f}%)")
            
            print(f"Clustering completed. Silhouette score can be calculated for evaluation.")
            return cluster_labels
        except Exception as e:
            print(f"Erreur lors du clustering: {e}")
            return None
    
    def save_models(self, 
                   tfidf_path='models/tfidf_vectorizer.pkl', 
                   similarity_path='data/similarity.pkl',
                   pca_path='models/pca_model.pkl', 
                   kmeans_path='models/kmeans_model.pkl'):
        """Sauvegarde tous les modèles."""
        models_saved = []
        
        # Sauvegarde du vectoriseur TF-IDF
        if self.tfidf_vectorizer is not None:
            try:
                with open(tfidf_path, 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
                print(f"TF-IDF vectorizer saved to {tfidf_path}")
                models_saved.append('TF-IDF')
            except Exception as e:
                print(f"Erreur lors de la sauvegarde du TF-IDF: {e}")
        
        # Sauvegarde de la matrice de similarité
        if self.cosine_sim is not None:
            try:
                with open(similarity_path, 'wb') as f:
                    pickle.dump(self.cosine_sim, f)
                print(f"Similarity matrix saved to {similarity_path}")
                models_saved.append('Similarity Matrix')
            except Exception as e:
                print(f"Erreur lors de la sauvegarde de la matrice de similarité: {e}")
        
        # Sauvegarde du modèle PCA
        if self.pca is not None:
            try:
                with open(pca_path, 'wb') as f:
                    pickle.dump(self.pca, f)
                print(f"PCA model saved to {pca_path}")
                models_saved.append('PCA')
            except Exception as e:
                print(f"Erreur lors de la sauvegarde du PCA: {e}")
                
        # Sauvegarde du modèle K-means
        if self.kmeans is not None:
            try:
                with open(kmeans_path, 'wb') as f:
                    pickle.dump(self.kmeans, f)
                print(f"KMeans model saved to {kmeans_path}")
                models_saved.append('KMeans')
            except Exception as e:
                print(f"Erreur lors de la sauvegarde du K-means: {e}")
                
        return models_saved
    
    def load_models(self, 
                   tfidf_path='models/tfidf_vectorizer.pkl', 
                   similarity_path='data/similarity.pkl',
                   pca_path='models/pca_model.pkl', 
                   kmeans_path='models/kmeans_model.pkl'):
        """Charge tous les modèles sauvegardés."""
        models_loaded = []
        
        try:
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            models_loaded.append('TF-IDF')
        except FileNotFoundError:
            print(f"TF-IDF model not found at {tfidf_path}")
        
        try:
            with open(similarity_path, 'rb') as f:
                self.cosine_sim = pickle.load(f)
            models_loaded.append('Similarity Matrix')
        except FileNotFoundError:
            print(f"Similarity matrix not found at {similarity_path}")
            
        try:
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)
            models_loaded.append('PCA')
        except FileNotFoundError:
            print(f"PCA model not found at {pca_path}")
            
        try:
            with open(kmeans_path, 'rb') as f:
                self.kmeans = pickle.load(f)
            models_loaded.append('KMeans')
        except FileNotFoundError:
            print(f"KMeans model not found at {kmeans_path}")
            
        print(f"Loaded models: {', '.join(models_loaded)}")
        return models_loaded