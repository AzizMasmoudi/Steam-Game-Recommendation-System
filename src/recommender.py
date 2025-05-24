import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    """Classe pour générer des recommandations de jeux."""
    
    def __init__(self, df=None, similarity_matrix=None):
        """
        Initialise la classe Recommender.
        
        Args:
            df: DataFrame Pandas contenant les données des jeux
            similarity_matrix: Matrice de similarité entre les jeux
        """
        self.df = df
        self.similarity_matrix = similarity_matrix
        self.game_to_index = None
        self.index_to_game = None
        
        if self.df is not None:
            self._create_indices()
    
    def _create_indices(self):
        """Crée des dictionnaires pour la correspondance jeu-index."""
        self.game_to_index = {game: idx for idx, game in enumerate(self.df['Name'])}
        self.index_to_game = {idx: game for idx, game in enumerate(self.df['Name'])}
    
    def get_recommendations(self, game_title, n=10, method='cosine'):
        """
        Génère des recommandations basées sur un titre de jeu.
        
        Args:
            game_title: Titre du jeu pour lequel générer des recommandations
            n: Nombre de recommandations à générer
            method: Méthode de recommandation ('cosine', 'cluster')
            
        Returns:
            Tuple (recommendations, similarity_scores, additional_info)
        """
        if self.df is None or self.similarity_matrix is None:
            raise ValueError("DataFrame and similarity matrix must be provided")
        
        # Vérifier si le jeu existe dans le DataFrame
        if game_title not in self.df['Name'].values:
            # Chercher des jeux similaires par nom
            similar_names = self._find_similar_game_names(game_title)
            raise ValueError(
                f"Game '{game_title}' not found in the dataset. "
                f"Did you mean: {', '.join(similar_names[:3])}?"
            )
        
        if method == 'cosine':
            return self._get_cosine_recommendations(game_title, n)
        elif method == 'cluster':
            return self._get_cluster_recommendations(game_title, n)
        else:
            raise ValueError(f"Method '{method}' not supported")
    
    def _get_cosine_recommendations(self, game_title, n):
        """Génère des recommandations basées sur la similarité cosinus."""
        # Trouver l'index du jeu
        idx = self.df[self.df['Name'] == game_title].index[0]
        
        # Obtenir les scores de similarité
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]  # Exclure le jeu lui-même
        
        # Obtenir les indices des jeux recommandés
        game_indices = [i[0] for i in sim_scores]
        similarity_scores = [sim_scores[i][1] for i in range(len(sim_scores))]
        
        # Récupérer les informations des jeux recommandés
        recommendations = []
        for idx in game_indices:
            game_info = {
                'name': self.df.iloc[idx]['Name'],
                'developer': self.df.iloc[idx]['Developer'],
                'tags': self.df.iloc[idx]['Tags'],
                'about': self.df.iloc[idx]['About'][:100] + "..." if len(self.df.iloc[idx]['About']) > 100 else self.df.iloc[idx]['About']
            }
            recommendations.append(game_info)
        
        additional_info = {
            'method': 'cosine_similarity',
            'base_game': game_title,
            'total_games_in_db': len(self.df)
        }
        
        return recommendations, similarity_scores, additional_info
    
    def _get_cluster_recommendations(self, game_title, n):
        """Génère des recommandations basées sur les clusters."""
        if 'cluster' not in self.df.columns:
            raise ValueError("Cluster information not available. Run clustering first.")
        
        # Trouver le cluster du jeu
        game_cluster = self.df[self.df['Name'] == game_title]['cluster'].iloc[0]
        
        # Obtenir tous les jeux du même cluster (exclure le jeu original)
        cluster_games = self.df[
            (self.df['cluster'] == game_cluster) & 
            (self.df['Name'] != game_title)
        ]
        
        if len(cluster_games) == 0:
            # Fallback vers la méthode cosinus
            return self._get_cosine_recommendations(game_title, n)
        
        # Si on a plus de jeux que demandé, utiliser la similarité pour choisir les meilleurs
        if len(cluster_games) > n:
            game_idx = self.df[self.df['Name'] == game_title].index[0]
            cluster_indices = cluster_games.index.tolist()
            
            # Calculer les similarités avec les jeux du cluster
            similarities = [(idx, self.similarity_matrix[game_idx][idx]) for idx in cluster_indices]
            similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Prendre les n meilleurs
            selected_indices = [idx for idx, _ in similarities[:n]]
            similarity_scores = [score for _, score in similarities[:n]]
            cluster_games = self.df.loc[selected_indices]
        else:
            # Calculer les scores de similarité pour tous les jeux du cluster
            game_idx = self.df[self.df['Name'] == game_title].index[0]
            similarity_scores = [self.similarity_matrix[game_idx][idx] for idx in cluster_games.index]
        
        # Récupérer les informations des jeux recommandés
        recommendations = []
        for _, game in cluster_games.iterrows():
            game_info = {
                'name': game['Name'],
                'developer': game['Developer'],
                'tags': game['Tags'],
                'about': game['About'][:100] + "..." if len(game['About']) > 100 else game['About']
            }
            recommendations.append(game_info)
        
        additional_info = {
            'method': 'cluster_based',
            'base_game': game_title,
            'cluster_id': game_cluster,
            'games_in_cluster': len(self.df[self.df['cluster'] == game_cluster])
        }
        
        return recommendations, similarity_scores, additional_info
    
    def _find_similar_game_names(self, game_title, threshold=0.6):
        """Trouve des noms de jeux similaires pour suggérer des corrections."""
        from difflib import SequenceMatcher
        
        similarities = []
        for game in self.df['Name']:
            similarity = SequenceMatcher(None, game_title.lower(), game.lower()).ratio()
            if similarity > threshold:
                similarities.append((game, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [game for game, _ in similarities[:5]]
    
    def display_recommendations(self, game_title, n=10, method='cosine', show_details=True):
        """Affiche les recommandations pour un jeu avec des informations détaillées."""
        try:
            recommendations, scores, additional_info = self.get_recommendations(game_title, n, method)
            
            print(f"\n{'='*60}")
            print(f"RECOMMENDATIONS FOR: {game_title}")
            print(f"Method: {additional_info['method'].replace('_', ' ').title()}")
            if 'cluster_id' in additional_info:
                print(f"Cluster ID: {additional_info['cluster_id']}")
            print(f"{'='*60}")
            
            for i, (rec, score) in enumerate(zip(recommendations, scores)):
                print(f"\n{i+1}. {rec['name']}")
                print(f"   Similarity Score: {score:.4f}")
                
                if show_details:
                    print(f"   Developer: {rec['developer']}")
                    
                    # Afficher les premiers tags
                    tags = rec['tags'].split(',')[:3] if rec['tags'] else ['No tags']
                    tags_str = ', '.join([tag.strip() for tag in tags])
                    print(f"   Main Tags: {tags_str}")
                    
                    # Afficher un extrait de la description
                    if rec['about']:
                        print(f"   Description: {rec['about']}")
                    
                print(f"   {'-'*50}")
            
            print(f"\nTotal games in database: {additional_info['total_games_in_db']}")
            
            return recommendations, scores
            
        except ValueError as e:
            print(f"Error: {e}")
            return None, None
    
    def get_game_details(self, game_title):
        """Retourne les détails complets d'un jeu."""
        if game_title not in self.df['Name'].values:
            return None
        
        game_data = self.df[self.df['Name'] == game_title].iloc[0]
        
        details = {
            'name': game_data['Name'],
            'developer': game_data['Developer'],
            'tags': game_data['Tags'],
            'about': game_data['About'],
            'release_date': game_data.get('ReleaseDate', 'Unknown'),
            'price': game_data.get('Price', 'Unknown'),
            'reviews': game_data.get('Reviews', 'Unknown')
        }
        
        if 'cluster' in game_data:
            details['cluster'] = game_data['cluster']
        
        return details
    
    def predict_cluster(self, game_idx):
        """Prédit le cluster d'un jeu basé sur ses jeux similaires."""
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix not available")
        
        sim_scores = list(enumerate(self.similarity_matrix[game_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Obtenir les 10 jeux les plus similaires (excluant soi-même)
        similar_indices = [i[0] for i in sim_scores]
        
        if 'cluster' not in self.df.columns:
            return None
        
        # Obtenir les clusters des jeux similaires
        similar_clusters = self.df.iloc[similar_indices]['cluster'].values
        
        # Retourner le cluster le plus commun
        most_common_cluster = np.bincount(similar_clusters).argmax()
        return most_common_cluster
    
    def get_recommendations_by_tags(self, tags, n=10):
        """Génère des recommandations basées sur des tags spécifiques."""
        if self.df is None:
            raise ValueError("DataFrame not provided")
        
        # Normaliser les tags d'entrée
        input_tags = [tag.strip().lower() for tag in tags if tag.strip()]
        
        if not input_tags:
            raise ValueError("No valid tags provided")
        
        # Calculer un score pour chaque jeu basé sur la correspondance des tags
        game_scores = []
        
        for idx, row in self.df.iterrows():
            game_tags = [tag.strip().lower() for tag in row['Tags'].split(',') if tag.strip()]
            
            # Calculer le score de correspondance (Jaccard similarity)
            tag_intersection = len(set(input_tags) & set(game_tags))
            tag_union = len(set(input_tags) | set(game_tags))
            
            score = tag_intersection / tag_union if tag_union > 0 else 0
            
            if score > 0:  # Seulement les jeux avec au moins un tag en commun
                game_scores.append((idx, score, row['Name']))
        
        # Trier par score décroissant
        game_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Prendre les n meilleurs
        top_games = game_scores[:n]
        
        recommendations = []
        scores = []
        
        for idx, score, name in top_games:
            game_info = {
                'name': self.df.iloc[idx]['Name'],
                'developer': self.df.iloc[idx]['Developer'],
                'tags': self.df.iloc[idx]['Tags'],
                'about': self.df.iloc[idx]['About'][:100] + "..." if len(self.df.iloc[idx]['About']) > 100 else self.df.iloc[idx]['About']
            }
            recommendations.append(game_info)
            scores.append(score)
        
        additional_info = {
            'method': 'tag_based',
            'input_tags': tags,
            'games_found': len(game_scores)
        }
        
        return recommendations, scores, additional_info
    
    def save_recommender(self, path='models/recommender.pkl'):
        """Sauvegarde l'état du recommender."""
        recommender_state = {
            'similarity_matrix': self.similarity_matrix,
            'game_to_index': self.game_to_index,
            'index_to_game': self.index_to_game
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(recommender_state, f)
            print(f"Recommender saved to {path}")
            return True
        except Exception as e:
            print(f"Error saving recommender: {e}")
            return False
    
    def load_recommender(self, path='models/recommender.pkl'):
        """Charge l'état du recommender."""
        try:
            with open(path, 'rb') as f:
                recommender_state = pickle.load(f)
            
            self.similarity_matrix = recommender_state['similarity_matrix']
            self.game_to_index = recommender_state['game_to_index']
            self.index_to_game = recommender_state['index_to_game']
            
            print(f"Recommender loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading recommender: {e}")
            return False