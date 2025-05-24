import pandas as pd
import os
import numpy as np

class DataLoader:
    """Classe pour charger et prétraiter les données des jeux Steam."""
    
    def __init__(self, data_path="data_backup/game_data.pkl"):
        """
        Initialise le DataLoader.
        
        Args:
            data_path: Chemin vers le fichier de données
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """Charge les données à partir du fichier pickle."""
        try:
            self.df = pd.read_pickle(self.data_path)
            print(f"Original data shape: {self.df.shape}")
            print(f"Original columns: {self.df.columns.tolist()}")
            return self.df
        except FileNotFoundError:
            print(f"Erreur: Fichier {self.data_path} non trouvé.")
            return None
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return None
            
    def preprocess_data(self):
        """Nettoie et prétraite les données."""
        if self.df is None:
            print("Aucune donnée chargée. Appelez d'abord load_data().")
            return None
            
        # Supprimer les lignes avec erreurs
        initial_shape = self.df.shape
        self.df = self.df[~self.df['Name'].str.contains('Error', na=False)]
        print(f"After removing error rows: {self.df.shape}")
        print(f"Removed {initial_shape[0] - self.df.shape[0]} error rows")
        
        # Vérifier les valeurs manquantes
        missing_values = self.df.isnull().sum()
        print("\nMissing values by column:")
        print(missing_values)
        
        # Remplir les valeurs manquantes
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna('')
            else:
                self.df[col] = self.df[col].fillna(0)
                
        # Créer une feature combinée pour le calcul de similarité
        self.df['combined_features'] = (
            self.df['Name'] + ' ' + 
            self.df['About'] + ' ' + 
            self.df['Developer'] + ' ' + 
            self.df['Tags']
        )
        
        print("\nData preprocessing completed successfully.")
        return self.df
    
    def create_output_directories(self):
        """Crée les répertoires de sortie nécessaires."""
        directories = ['data', 'visualizations', 'models']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Directory '{directory}' created/verified.")
        
    def save_processed_data(self, output_path='data/cleaned_games.pkl'):
        """Sauvegarde les données prétraitées."""
        if self.df is None:
            raise ValueError("No data to save. Call load_data() and preprocess_data() first.")
            
        try:
            self.df.to_pickle(output_path)
            print(f"Processed data saved to {output_path}")
            return True
        except Exception as e:
            print(f"Erreur lors de la sauvegarde: {e}")
            return False
        
    def get_data_info(self):
        """Retourne des informations sur les données."""
        if self.df is None:
            return "Aucune donnée chargée."
            
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'missing_values': self.df.isnull().sum().sum()
        }
        return info