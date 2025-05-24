import streamlit as st
import pandas as pd
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    games_df = pd.read_pickle('data/cleaned_games.pkl')
    with open('data/similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
    return games_df, similarity

try:
    games_df, similarity = load_data()
    
    # Function to get game recommendations
    def get_recommendations(game_title, cosine_sim=similarity):
        # Get the index of the game that matches the title
        idx = games_df[games_df['Name'] == game_title].index[0]
        
        # Get the pairwise similarity scores of all games with that game
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the games based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the 10 most similar games (excluding the first which is the game itself)
        sim_scores = sim_scores[1:11]
        
        # Get the game indices
        game_indices = [i[0] for i in sim_scores]
        
        # Return the top 10 most similar games with their similarity scores
        return [(games_df.iloc[i]['Name'], 
                games_df.iloc[i]['HeaderImage'], 
                games_df.iloc[i]['About'],
                games_df.iloc[i]['Tags'],
                sim_scores[j][1]) 
                for j, i in enumerate(game_indices)]
    
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.info("This recommender system suggests Steam games based on your preferences. It uses cosine similarity to find games with similar features.")
        
        st.title("How It Works")
        st.write("1. Select a game you like")
        st.write("2. Click 'Get Recommendations'")
        st.write("3. Explore similar games you might enjoy")
        
        st.title("Data")
        st.write(f"Total games in database: {len(games_df)}")
        
    # Main content
    st.title("🎮 Steam Game Recommendation System")
    
    # Game selection
    game_list = games_df['Name'].tolist()
    selected_game = st.selectbox("Select a game you like:", game_list)
    
    col1, col2 = st.columns([1, 4])
    
    # Display selected game info
    with col1:
        selected_game_data = games_df[games_df['Name'] == selected_game].iloc[0]
        st.image(selected_game_data['HeaderImage'], caption=selected_game, use_column_width=True)
    
    with col2:
        st.subheader(selected_game)
        st.write(selected_game_data['About'])
        st.write(f"**Developer:** {selected_game_data['Developer']}")
        st.write(f"**Release Date:** {selected_game_data['ReleaseDate']}")
        st.write(f"**Tags:** {selected_game_data['Tags']}")
        
    # Get recommendations
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(selected_game)
        
        st.subheader("Games You Might Like")
        
        # Display recommendations in rows of 3
        for i in range(0, len(recommendations), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(recommendations):
                    with cols[j]:
                        st.image(recommendations[i+j][1], use_column_width=True)
                        st.subheader(recommendations[i+j][0])
                        st.write(recommendations[i+j][2][:150] + "..." if len(recommendations[i+j][2]) > 150 else recommendations[i+j][2])
                        st.write(f"**Tags:** {recommendations[i+j][3]}")
                        st.progress(float(recommendations[i+j][4]))  # Similarity score as progress bar
                        
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please make sure you've run the feature_engineering.py script to prepare the data files.")