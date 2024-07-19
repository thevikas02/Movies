
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import streamlit as st

# Load the data
movies = pd.read_csv("C:/Users/NAND/Desktop/data")
ratings = pd.read_csv("C:/Users/NAND/Desktop/data")

# Prepare the data
movies_users = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
mat_movies = csr_matrix(movies_users.values)

# Train the KNN model
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model.fit(mat_movies)

# Define the recommender function
def recommender(movie_name, data, n=10):
    idx = process.extractOne(movie_name, movies['title'])[2]
    movie_id = movies.loc[idx, 'movieId']
    st.write('Movie Selected:', movies.loc[idx, 'title'])
    st.write('Genre:', movies.loc[idx, 'genres'])
    st.write('Rating:', ratings.loc[ratings['movieId'] == movie_id, 'rating'].mean())
    st.write('Searching for recommendations...')
    distances, indices = model.kneighbors(data[idx], n_neighbors=n)
    recommended_movies = [(movies.loc[i, 'title'], ratings.loc[ratings['movieId'] == movies.loc[i, 'movieId'], 'rating'].mean()) for i in indices.flatten() if i != idx]
    return recommended_movies

# Streamlit UI
st.title('Movie Recommender System')

# Dropdown menu for selecting the movie (alphabetically sorted)
movie_list = sorted(movies['title'].tolist())
user_input = st.selectbox('Select a movie:', movie_list)
num_recommendations = st.slider('Number of recommendations:', 1, 20, 10)

if user_input:
    recommendations = recommender(user_input, mat_movies, num_recommendations)
    st.write('Recommendations:')
    for movie, rating in recommendations:
        st.write(f"{movie} - Rating: {rating:.2f}")

# Run the Streamlit app
# Command: streamlit run app.py

