# streamlit run app.py

import pickle
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up Streamlit page configuration
st.set_page_config(page_title="Music Recommendation System", page_icon=":guitar:")

# Load models and vectorizers
with open('classification_model.pkl', 'rb') as f:
    classification_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
# pickle its use serialise and deserilise the data the object can convetr into bit which can saved in file or tanstmitted over net 
# Load the music dataset and similarity matrix
music = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Define CSS styles
background_style = """
<style>
    .stApp {
        background-image: url('data:image/jpeg;base64,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');
        background-size: cover;
        background-position: center;
    }
    .user-input {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }
</style>
"""

# Apply CSS
st.markdown(background_style, unsafe_allow_html=True)

# Spotify API Credentials
CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(song):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    for i in distances[1:6]:
        artist = music.iloc[i[0]].artist
        recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song, artist))
        recommended_music_names.append(music.iloc[i[0]].song)

    return recommended_music_names, recommended_music_posters

# Function to classify the song genre based on lyrics
def classify_song(lyrics):
    input_vector = tfidf_vectorizer.transform([lyrics])
    genre = classification_model.predict(input_vector)
    return genre[0]

# Streamlit app layout
st.title('Music Recommendation and Genre Prediction System')
st.header('Predict the Genre of Your Lyrics')

# User input for song lyrics
user_input = st.text_area("Enter the song lyrics here:")

if st.button('Predict Genre'):
    if user_input:
        predicted_genre = classify_song(user_input)
        st.write(f'Predicted Genre: **{predicted_genre}**')
    else:
        st.warning("Please enter some lyrics to predict the genre.")

st.header('Music Recommendations')
music_list = music['song'].values
selected_song = st.selectbox("Select a song to get recommendations", music_list)

if st.button('Recommend Similar Music'):
    recommended_music_names, recommended_music_posters = recommend(selected_song)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_music_names[0])
        st.image(recommended_music_posters[0])
    with col2:
        st.text(recommended_music_names[1])
        st.image(recommended_music_posters[1])
    with col3:
        st.text(recommended_music_names[2])
        st.image(recommended_music_posters[2])
    with col4:
        st.text(recommended_music_names[3])
        st.image(recommended_music_posters[3])
    with col5:
        st.text(recommended_music_names[4])
        st.image(recommended_music_posters[4])