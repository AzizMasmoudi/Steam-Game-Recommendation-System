import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the scraped data
df = pd.read_pickle('data/game_data.pkl')

# Data preprocessing
print("Original data shape:", df.shape)

# Remove rows with errors or missing crucial data
df = df[~df['Name'].str.contains('Error')]
print("After removing error rows:", df.shape)

# Create a combined text feature for similarity calculation
df['combined_features'] = df['Name'] + ' ' + df['About'] + ' ' + df['Developer'] + ' ' + df['Tags']

# Fill any NaN values to avoid issues with vectorization
df['combined_features'] = df['combined_features'].fillna('')

# Save a copy of the cleaned dataframe
df.to_pickle('data/cleaned_games.pkl')

# Create TF-IDF vectors from the combined features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print("Cosine similarity matrix shape:", cosine_sim.shape)

# Save the similarity matrix
with open('data/similarity.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f)

print("Feature engineering complete. Files saved to data directory.")