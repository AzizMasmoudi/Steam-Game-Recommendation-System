import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    """Classe pour évaluer les performances du système de recommandation."""
    
    def __init__(self, df=None, similarity_matrix=None, recommender=None):
        """
        Initialise la classe Evaluator.
        
        Args:
            df: DataFrame Pandas contenant les données des jeux
            similarity_matrix: Matrice de similarité entre les jeux
            recommender: Instance de la classe Recommender
        """
        self.df = df
        self.similarity_matrix = similarity_matrix
        self.recommender = recommender
        
    def evaluate_recommendations(self, k=10, n_splits=5):
        """
        Évalue la précision des recommandations en utilisant la validation croisée.
        
        Args:
            k: Nombre de recommandations à générer
            n_splits: Nombre de divisions pour la validation croisée
            
        Returns:
            Tuple (precision, recall, f1, ndcg) de scores moyens
        """
        if self.df is None or self.similarity_matrix is None:
            raise ValueError("DataFrame and similarity matrix must be provided")
        
        # Créer des divisions pour la validation croisée
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Stocker les métriques pour chaque pli
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ndcg_scores = []
        
        # Pour chaque jeu, on va considérer les jeux du même cluster comme pertinents
        clusters = self.df['cluster'].values
        
        # Obtenir les indices pour chaque division
        indices = np.arange(len(self.df))
        
        for train_idx, test_idx in kf.split(indices):
            # Pour chaque jeu de test
            for idx in test_idx:
                game_title = self.df.iloc[idx]['Name']
                actual_cluster = self.df.iloc[idx]['cluster']
                
                # Trouver les jeux pertinents (même cluster) en excluant le jeu lui-même
                relevant_indices = set(self.df[self.df['cluster'] == actual_cluster].index) - {idx}
                
                # Obtenir les k meilleures recommandations
                sim_scores = list(enumerate(self.similarity_matrix[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:k+1]  # Ignorer le premier (le jeu lui-même)
                rec_indices = [i[0] for i in sim_scores]
                
                # Calculer la précision et le rappel
                relevant_recs = set(rec_indices).intersection(relevant_indices)
                
                precision = len(relevant_recs) / len(rec_indices) if len(rec_indices) > 0 else 0
                recall = len(relevant_recs) / len(relevant_indices) if len(relevant_indices) > 0 else 0
                
                # Calculer le score F1
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                # Calculer NDCG (Normalized Discounted Cumulative Gain)
                dcg = 0
                idcg = 0
                
                # Calculer DCG
                for i, rec_idx in enumerate(rec_indices):
                    rel = 1 if rec_idx in relevant_indices else 0
                    dcg += rel / np.log2(i + 2)  # i+2 car i commence à 0
                    
                # Calculer IDCG
                relevant_count = min(len(relevant_indices), k)
                for i in range(relevant_count):
                    idcg += 1 / np.log2(i + 2)
                    
                ndcg = dcg / idcg if idcg > 0 else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                ndcg_scores.append(ndcg)
        
        # Calculer les moyennes
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        avg_ndcg = np.mean(ndcg_scores)
        
        return avg_precision, avg_recall, avg_f1, avg_ndcg
    
    def calculate_tag_overlap(self, game_title, recommendations):
        """
        Calcule le pourcentage de tags partagés entre un jeu et ses recommandations.
        
        Args:
            game_title: Titre du jeu de référence
            recommendations: Liste de titres de jeux recommandés
            
        Returns:
            Score de chevauchement moyen des tags (similarité de Jaccard)
        """
        if self.df is None:
            raise ValueError("DataFrame must be provided")
            
        # Check if game_title exists in the DataFrame
        if not any(self.df['Name'] == game_title):
            print(f"Warning: Game '{game_title}' not found in DataFrame.")
            return 0
        
        try:
            game_tags = set(self.df[self.df['Name'] == game_title]['Tags'].iloc[0].split(','))
            game_tags = {tag.strip() for tag in game_tags}
            
            overlap_scores = []
            for rec in recommendations:
                # Check if recommendation exists in the DataFrame
                if not any(self.df['Name'] == rec):
                    print(f"Warning: Recommended game '{rec}' not found in DataFrame.")
                    continue
                
                try:
                    rec_tags = set(self.df[self.df['Name'] == rec]['Tags'].iloc[0].split(','))
                    rec_tags = {tag.strip() for tag in rec_tags}
                    
                    # Calculer la similarité de Jaccard
                    if len(game_tags.union(rec_tags)) > 0:
                        overlap = len(game_tags.intersection(rec_tags)) / len(game_tags.union(rec_tags))
                    else:
                        overlap = 0
                        
                    overlap_scores.append(overlap)
                except (IndexError, ValueError, AttributeError) as e:
                    print(f"Error processing tags for game '{rec}': {e}")
                    continue
            
            # Return 0 if no valid recommendations were found
            if not overlap_scores:
                return 0
                
            return np.mean(overlap_scores)
            
        except (IndexError, ValueError, AttributeError) as e:
            print(f"Error processing tags for game '{game_title}': {e}")
            return 0
    
    def calculate_developer_consistency(self, game_title, recommendations):
        """
        Vérifie si les recommandations incluent des jeux du même développeur.
        
        Args:
            game_title: Titre du jeu de référence
            recommendations: Liste de titres de jeux recommandés
            
        Returns:
            Proportion de recommandations du même développeur
        """
        if self.df is None:
            raise ValueError("DataFrame must be provided")
            
        # Check if game_title exists in the DataFrame
        if not any(self.df['Name'] == game_title):
            print(f"Warning: Game '{game_title}' not found in DataFrame.")
            return 0
        
        try:
            game_dev = self.df[self.df['Name'] == game_title]['Developer'].iloc[0]
            
            same_dev_count = 0
            valid_recommendations = 0
            
            for rec in recommendations:
                # Check if recommendation exists in the DataFrame
                if not any(self.df['Name'] == rec):
                    print(f"Warning: Recommended game '{rec}' not found in DataFrame.")
                    continue
                
                try:
                    rec_dev = self.df[self.df['Name'] == rec]['Developer'].iloc[0]
                    valid_recommendations += 1
                    
                    # Compter les recommandations du même développeur
                    if rec_dev == game_dev:
                        same_dev_count += 1
                except (IndexError, ValueError) as e:
                    print(f"Error processing developer for game '{rec}': {e}")
                    continue
            
            # Return 0 if no valid recommendations were found
            if valid_recommendations == 0:
                return 0
                
            return same_dev_count / valid_recommendations
            
        except (IndexError, ValueError) as e:
            print(f"Error processing developer for game '{game_title}': {e}")
            return 0
    
    def evaluate_content_quality(self, sample_size=10):
        """
        Évalue la qualité des recommandations basées sur le contenu.
        
        Args:
            sample_size: Nombre de jeux à échantillonner pour l'évaluation
            
        Returns:
            Tuple (tag_overlaps, dev_consistencies) des scores moyens
        """
        if self.df is None or self.recommender is None:
            raise ValueError("DataFrame and recommender must be provided")
            
        try:
            # Sélectionner des jeux aléatoires pour l'évaluation
            np.random.seed(42)
            eval_games = np.random.choice(self.df['Name'].values, size=min(sample_size, len(self.df)), replace=False)
            
            print("Evaluating recommendation content quality...")
            print("Tag overlap between games and their recommendations:")
            print("Game\t\t\t\t\tTag Overlap")
            print("-" * 60)
            
            tag_overlaps = []
            dev_consistencies = []
            
            for game in eval_games:
                try:
                    # Get recommendations - handle different return types
                    result = self.recommender.get_recommendations(game)
                    recommendations = result[0] if isinstance(result, tuple) else result
                    
                    if not recommendations:  # Check if recommendations is empty
                        print(f"{game[:30]}...\tNo valid recommendations found")
                        continue
                    
                    # Calculer le chevauchement des tags
                    overlap = self.calculate_tag_overlap(game, recommendations)
                    if overlap > 0:  # Only add valid overlap scores
                        tag_overlaps.append(overlap)
                    
                    # Calculer la cohérence des développeurs
                    consistency = self.calculate_developer_consistency(game, recommendations)
                    if consistency > 0 or consistency == 0:  # Only add valid consistency scores
                        dev_consistencies.append(consistency)
                    
                    # Formater le nom du jeu pour l'affichage
                    display_name = game[:30] + "..." if len(game) > 30 else game.ljust(30)
                    print(f"{display_name}\t{overlap:.4f}")
                    
                except Exception as e:
                    print(f"Error evaluating {game}: {e}")
            
            if not tag_overlaps:
                print("Warning: No valid tag overlaps collected.")
                tag_overlaps = [0]  # Return a list with a single zero to avoid errors
                
            print(f"\nAverage tag overlap: {np.mean(tag_overlaps):.4f}")
            
            print("\nDeveloper consistency in recommendations:")
            print("Game\t\t\t\t\tDeveloper Consistency")
            print("-" * 60)
            
            for i, game in enumerate(eval_games):
                if i < len(dev_consistencies):
                    display_name = game[:30] + "..." if len(game) > 30 else game.ljust(30)
                    print(f"{display_name}\t{dev_consistencies[i]:.4f}")
            
            if not dev_consistencies:
                print("Warning: No valid developer consistencies collected.")
                dev_consistencies = [0]  # Return a list with a single zero to avoid errors
            else:
                print(f"\nAverage developer consistency: {np.mean(dev_consistencies):.4f}")
            
            return tag_overlaps, dev_consistencies
            
        except Exception as e:
            print(f"Error in evaluate_content_quality: {e}")
            return [0], [0]  # Return lists with a single zero to avoid errors
    
    def evaluate_cluster_prediction(self, sample_size=100):
        """
        Évalue la précision de la prédiction de clusters.
        
        Args:
            sample_size: Nombre d'échantillons à utiliser pour l'évaluation
            
        Returns:
            Tuple (y_true, y_pred, accuracy) des vrais clusters, clusters prédits et précision
        """
        if self.df is None or self.recommender is None:
            raise ValueError("DataFrame and recommender must be provided")
        
        try:
            print("\nEvaluating cluster prediction...")
            
            y_true = []
            y_pred = []
            
            # Utiliser un échantillon de jeux pour l'évaluation
            sample_size = min(sample_size, len(self.df))
            eval_indices = np.random.choice(range(len(self.df)), size=sample_size, replace=False)
            
            for idx in eval_indices:
                try:
                    true_cluster = self.df.iloc[idx]['cluster']
                    pred_cluster = self.recommender.predict_cluster(idx)
                    
                    y_true.append(true_cluster)
                    y_pred.append(pred_cluster)
                except Exception as e:
                    print(f"Error predicting cluster for game at index {idx}: {e}")
            
            if not y_true or not y_pred:
                print("Warning: No valid cluster predictions collected.")
                return [0], [0], 0
                
            # Calculer la précision globale du modèle
            accuracy = np.mean(np.array(y_true) == np.array(y_pred))
            
            return y_true, y_pred, accuracy
            
        except Exception as e:
            print(f"Error in evaluate_cluster_prediction: {e}")
            return [0], [0], 0