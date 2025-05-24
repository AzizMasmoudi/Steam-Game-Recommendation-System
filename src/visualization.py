import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import os

class Visualizer:
    """Classe pour générer des visualisations des données et des résultats."""
    
    def __init__(self, df=None):
        """
        Initialise la classe Visualizer.
        
        Args:
            df: DataFrame Pandas contenant les données des jeux
        """
        self.df = df
        # Définir le style de graphique
        plt.style.use('fivethirtyeight')
        sns.set(font_scale=1.2)
        
        # Créer le dossier de visualisations si nécessaire
        os.makedirs('visualizations', exist_ok=True)
        
    def plot_top_tags(self, n=20, save_path='visualizations/top_tags.png'):
        """Visualise les n tags les plus fréquents."""
        if self.df is None:
            raise ValueError("DataFrame not provided")
            
        print(f"Creating top {n} tags visualization...")
        
        # Compter les occurrences des tags
        tag_counts = {}
        for tags_str in self.df['Tags']:
            if isinstance(tags_str, str) and tags_str:
                for tag in tags_str.split(','):
                    tag = tag.strip()
                    if tag:  # Éviter les tags vides
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        if not tag_counts:
            print("Aucun tag trouvé dans les données.")
            return None
            
        # Créer un DataFrame pour la visualisation
        tag_df = pd.DataFrame({
            'Tag': list(tag_counts.keys()), 
            'Count': list(tag_counts.values())
        })
        tag_df = tag_df.sort_values('Count', ascending=False).head(n)
        
        # Créer le graphique
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='Tag', data=tag_df, palette='viridis')
        plt.title(f'Top {n} Game Tags')
        plt.xlabel('Number of Games')
        plt.tight_layout()
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Top tags visualization saved to {save_path}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            plt.close()
        
        return tag_df
    
    def create_wordcloud(self, column='About', save_path='visualizations/description_wordcloud.png'):
        """Crée un nuage de mots à partir des descriptions des jeux."""
        if self.df is None:
            raise ValueError("DataFrame not provided")
            
        print(f"Generating word cloud of game {column.lower()}s...")
        
        # Combiner tout le texte
        text = ' '.join(self.df[column].dropna().astype(str))
        
        if not text.strip():
            print(f"Aucun texte trouvé dans la colonne {column}")
            return
            
        try:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white', 
                max_words=150,
                colormap='viridis',
                relative_scaling=0.5,
                min_font_size=10
            ).generate(text)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud of Game {column}s')
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Word cloud saved to {save_path}")
            
        except Exception as e:
            print(f"Erreur lors de la création du nuage de mots: {e}")
            plt.close()
        
    def plot_top_developers(self, n=15, save_path='visualizations/top_developers.png'):
        """Visualise les n développeurs les plus fréquents."""
        if self.df is None:
            raise ValueError("DataFrame not provided")
            
        print(f"Creating top {n} developers visualization...")
        
        try:
            plt.figure(figsize=(12, 8))
            top_devs = self.df['Developer'].value_counts().head(n)
            
            if top_devs.empty:
                print("Aucun développeur trouvé dans les données.")
                return None
                
            sns.barplot(x=top_devs.values, y=top_devs.index, palette='plasma')
            plt.title(f'Top {n} Game Developers')
            plt.xlabel('Number of Games')
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Top developers visualization saved to {save_path}")
            
            return top_devs
            
        except Exception as e:
            print(f"Erreur lors de la création du graphique des développeurs: {e}")
            plt.close()
            return None
        
    def plot_similarity_distribution(self, cosine_sim, save_path='visualizations/similarity_distribution.png'):
        """Visualise la distribution des scores de similarité."""
        print("Creating similarity distribution visualization...")
        
        try:
            plt.figure(figsize=(10, 6))
            # Aplatir la matrice de similarité et supprimer les auto-similarités
            similarities = cosine_sim.flatten()
            similarities = similarities[similarities < 0.999]  # Supprimer les auto-similarités
            
            sns.histplot(similarities, bins=50, kde=True, color='skyblue')
            plt.axvline(similarities.mean(), color='red', linestyle='--', 
                       label=f'Mean: {similarities.mean():.3f}')
            plt.axvline(similarities.median(), color='orange', linestyle='--', 
                       label=f'Median: {similarities.median():.3f}')
            
            plt.title('Distribution of Game Similarity Scores')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Similarity distribution saved to {save_path}")
            
        except Exception as e:
            print(f"Erreur lors de la création de la distribution de similarité: {e}")
            plt.close()
        
    def plot_elbow_method(self, wcss, save_path='visualizations/elbow_method.png'):
        """Visualise les résultats de la méthode du coude pour KMeans."""
        print("Creating elbow method visualization...")
        
        try:
            plt.figure(figsize=(10, 6))
            k_range = range(1, len(wcss) + 1)
            plt.plot(k_range, wcss, marker='o', linewidth=2, markersize=8)
            
            # Ajouter des annotations pour les points clés
            for i, w in enumerate(wcss):
                if i % 2 == 0:  # Annoter un point sur deux pour éviter l'encombrement
                    plt.annotate(f'{w:.0f}', (i+1, w), textcoords="offset points", 
                               xytext=(0,10), ha='center')
            
            plt.title('Elbow Method for Optimal Number of Clusters')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Elbow method visualization saved to {save_path}")
            
        except Exception as e:
            print(f"Erreur lors de la création du graphique elbow: {e}")
            plt.close()
        
    def visualize_clusters(self, reduced_features, save_path='visualizations/tsne_clusters.png'):
        """Visualise les clusters avec t-SNE."""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("DataFrame with cluster labels not provided")
        
        print("Visualizing clusters with t-SNE...")
        
        try:
            # Adapter perplexity au nombre d'échantillons
            n_samples = reduced_features.shape[0]
            perplexity = min(30, (n_samples - 1) // 3)
            
            tsne = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=perplexity,
                n_iter=1000,
                learning_rate=200
            )
            tsne_results = tsne.fit_transform(reduced_features)
            
            # Créer un DataFrame pour la visualisation
            tsne_df = pd.DataFrame({
                'x': tsne_results[:, 0],
                'y': tsne_results[:, 1],
                'cluster': self.df['cluster'],
                'game': self.df['Name']
            })
            
            # Créer le graphique
            plt.figure(figsize=(12, 10))
            scatter = sns.scatterplot(
                x='x', y='y', hue='cluster', 
                data=tsne_df, 
                palette='tab10', 
                alpha=0.7,
                s=60
            )
            
            plt.title('t-SNE Visualization of Game Clusters')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            # Améliorer la légende
            handles, labels = scatter.get_legend_handles_labels()
            plt.legend(handles, [f'Cluster {l}' for l in labels], 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"t-SNE cluster visualization saved to {save_path}")
            
            return tsne_df
            
        except Exception as e:
            print(f"Erreur lors de la visualisation t-SNE: {e}")
            plt.close()
            return None
    
    def create_cluster_wordclouds(self, save_dir='visualizations'):
        """Crée des nuages de mots pour chaque cluster."""
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("DataFrame with cluster labels not provided")
            
        print("Creating word clouds for each cluster...")
        
        try:
            cluster_analysis = self.df.groupby('cluster')['Tags'].apply(
                lambda x: ' '.join(x.astype(str))
            ).reset_index()
            
            for i, row in cluster_analysis.iterrows():
                cluster_id = row['cluster']
                tags = row['Tags']
                
                if not tags.strip():
                    continue
                    
                # Générer un nuage de mots pour chaque cluster
                plt.figure(figsize=(8, 6))
                
                try:
                    wordcloud = WordCloud(
                        width=600, 
                        height=300, 
                        background_color='white', 
                        max_words=50,
                        colormap='viridis',
                        relative_scaling=0.5,
                        min_font_size=8
                    ).generate(tags)
                    
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Cluster {cluster_id} Tag Cloud')
                    plt.tight_layout()
                    
                    cluster_path = f'{save_dir}/cluster_{cluster_id}_tags.png'
                    plt.savefig(cluster_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"Erreur pour le cluster {cluster_id}: {e}")
                    plt.close()
                    
            print(f"Cluster word clouds saved to {save_dir}/")
            
        except Exception as e:
            print(f"Erreur lors de la création des nuages de mots de cluster: {e}")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='visualizations/cluster_confusion_matrix.png'):
        """Visualise la matrice de confusion pour la prédiction de clusters."""
        print("Creating confusion matrix visualization...")
        
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       square=True, linewidths=0.5)
            plt.title('Confusion Matrix for Cluster Prediction')
            plt.xlabel('Predicted Cluster')
            plt.ylabel('True Cluster')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix saved to {save_path}")
            
            # Afficher le rapport de classification
            print("\nCluster prediction classification report:")
            print(classification_report(y_true, y_pred))
            
            # Calculer la précision globale
            accuracy = np.mean(np.array(y_true) == np.array(y_pred))
            print(f"\nCluster prediction accuracy: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            print(f"Erreur lors de la création de la matrice de confusion: {e}")
            plt.close()
            return None
        
    def plot_recommendation_quality(self, tag_overlaps, save_path='visualizations/recommendation_quality.png'):
        """Visualise la qualité des recommandations."""
        print("Creating recommendation quality visualization...")
        
        try:
            plt.figure(figsize=(10, 6))
            
            sns.histplot(tag_overlaps, bins=min(10, len(tag_overlaps)), kde=True, color='lightcoral')
            plt.axvline(np.mean(tag_overlaps), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(tag_overlaps):.3f}')
            plt.axvline(np.median(tag_overlaps), color='orange', linestyle='--', 
                       label=f'Median: {np.median(tag_overlaps):.3f}')
            
            plt.title('Distribution of Tag Overlap in Recommendations')
            plt.xlabel('Tag Overlap Score (Jaccard Similarity)')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Recommendation quality visualization saved to {save_path}")
            
        except Exception as e:
            print(f"Erreur lors de la visualisation de la qualité des recommandations: {e}")
            plt.close()

    def create_summary_dashboard(self, save_path='visualizations/summary_dashboard.png'):
        """Crée un tableau de bord résumé avec plusieurs métriques."""
        if self.df is None:
            return
            
        print("Creating summary dashboard...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Steam Games Recommendation System - Summary Dashboard', fontsize=16)
            
            # 1. Distribution des clusters
            if 'cluster' in self.df.columns:
                cluster_counts = self.df['cluster'].value_counts().sort_index()
                axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
                axes[0, 0].set_title('Games Distribution by Cluster')
                axes[0, 0].set_xlabel('Cluster ID')
                axes[0, 0].set_ylabel('Number of Games')
            
            # 2. Top 10 tags
            tag_counts = {}
            for tags_str in self.df['Tags']:
                if isinstance(tags_str, str) and tags_str:
                    for tag in tags_str.split(','):
                        tag = tag.strip()
                        if tag:
                            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            if tag_counts:
                top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                tags, counts = zip(*top_tags)
                axes[0, 1].barh(range(len(tags)), counts, color='lightgreen')
                axes[0, 1].set_yticks(range(len(tags)))
                axes[0, 1].set_yticklabels(tags)
                axes[0, 1].set_title('Top 10 Game Tags')
                axes[0, 1].set_xlabel('Number of Games')
            
            # 3. Top développeurs
            top_devs = self.df['Developer'].value_counts().head(8)
            if not top_devs.empty:
                axes[1, 0].pie(top_devs.values, labels=top_devs.index, autopct='%1.1f%%')
                axes[1, 0].set_title('Top Developers Distribution')
            
            # 4. Statistiques générales
            axes[1, 1].axis('off')
            stats_text = f"""
            Dataset Statistics:
            
            Total Games: {len(self.df)}
            Unique Developers: {self.df['Developer'].nunique()}
            Unique Tags: {len(tag_counts) if tag_counts else 0}
            
            Clusters: {self.df['cluster'].nunique() if 'cluster' in self.df.columns else 'N/A'}
            
            Data Completeness:
            Names: {(~self.df['Name'].isna()).sum()}/{len(self.df)}
            Descriptions: {(~self.df['About'].isna()).sum()}/{len(self.df)}
            Tags: {(~self.df['Tags'].isna()).sum()}/{len(self.df)}
            """
            
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                           verticalalignment='center', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Summary dashboard saved to {save_path}")
            
        except Exception as e:
            print(f"Erreur lors de la création du tableau de bord: {e}")
            plt.close()