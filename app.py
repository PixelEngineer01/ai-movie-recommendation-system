"""
Netflix-style Movie Recommendation System
"""

# -------------------- IMPORTS --------------------
import streamlit as st
import pickle
import re
import requests
import numpy as np
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Netflix Movie Recommender", page_icon="🎬", layout="wide"
)


# -------------------- SESSION STATE --------------------
if "search_input" not in st.session_state:
    st.session_state.search_input = ""

if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = "All Genres"


# -------------------- UI STYLING --------------------
st.markdown(
    """
<style>
html, body {
    background-color: #0f0f0f;
    color: white;
    font-family: Arial, sans-serif;
}

.block-container {
    padding: 1.5rem 2.5rem;
}

.stButton > button {
    background-color: #e50914;
    color: white;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.45rem 1rem;
    width: 100%;
}

.stButton > button:hover {
    background-color: #f40612;
}

.movie-card {
    background-color: #141414;
    border-radius: 6px;
    padding: 6px;
    transition: transform 0.2s ease;
}

.movie-card:hover {
    transform: scale(1.05);
}

.movie-title {
    font-size: 13px;
    font-weight: 500;
    margin-top: 4px;
}

.similarity-badge {
    background-color: #1f7a1f;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 12px;
    display: inline-block;
    margin-top: 3px;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_artifacts():
    movies = pickle.load(open("movies.pkl", "rb"))
    vectors = pickle.load(open("vectors.pkl", "rb"))
    return movies, vectors


movies, vectors = load_artifacts()
movies["clean_title"] = movies["title"].str.lower().str.strip()
title_list = movies["clean_title"].tolist()


# -------------------- API KEY --------------------
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]


# -------------------- HELPERS --------------------
def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(show_spinner=False)
def fetch_poster(title: str):
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        poster = data.get("Poster")
        return poster if poster not in [None, "N/A"] else None
    except (requests.exceptions.RequestException, ValueError):
        return None


# -------------------- SMART FUZZY SEARCH --------------------
def smart_match_movie(user_input: str):
    user_input = user_input.lower().strip()

    match = process.extractOne(user_input, title_list, scorer=fuzz.WRatio)
    if match and match[1] >= 80:
        return match[0]

    match = process.extractOne(user_input, title_list, scorer=fuzz.partial_ratio)
    if match and match[1] >= 85:
        return match[0]

    return None


# -------------------- ML RECOMMENDATION --------------------
def robust_recommend(movie_name: str, top_n: int = 10):
    movie_name = normalize_text(movie_name)
    if not movie_name:
        return []

    matched = movie_name if movie_name in title_list else smart_match_movie(movie_name)

    if not matched:
        return []

    idx = movies[movies["clean_title"] == matched].index[0]

    scores = cosine_similarity(vectors[idx], vectors)[0]
    scores[idx] = 0.0

    top_indices = np.argsort(scores)[::-1][:30]
    mean_score = np.mean(scores[top_indices])
    threshold = max(mean_score * 0.8, 0.12)

    results = []
    for i in top_indices:
        if scores[i] >= threshold:
            results.append((movies.iloc[i].title, round(scores[i] * 100, 2)))
        if len(results) == top_n:
            break

    return results


# -------------------- UI COMPONENT --------------------
def movie_card(title, poster, score):
    st.markdown(
        f"""
    <div class="movie-card">
        <img src="{poster}" style="width:100%; border-radius:4px;">
        <div class="movie-title">{title}</div>
        <span class="similarity-badge">{score}% match</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


# -------------------- UI HEADER --------------------
st.markdown(
    "<h1 style='text-align:center;'>🎬 Netflix Movie Recommender</h1>",
    unsafe_allow_html=True,
)


# -------------------- UI INPUT SECTION --------------------
left, right, reset = st.columns([3, 2, 1])

with left:
    st.markdown("### 🔍 Search Movie")
    search_input = st.text_input(
        "Movie name",
        value=st.session_state.search_input,
        placeholder="Inception, Batman, Harry Potter",
    )
    search_btn = st.button("Search Recommendations")

with right:
    st.markdown("### 🎭 Browse by Genre")
    all_genres = ["All Genres"] + sorted({g for sub in movies["genres"] for g in sub})
    selected_genre = st.selectbox(
        "Genre", all_genres, index=all_genres.index(st.session_state.selected_genre)
    )
    genre_btn = st.button("Recommend by Genre")

with reset:
    st.markdown("### 🔄 Reset")
    reset_btn = st.button("Reset All")


# -------------------- RESET --------------------
if reset_btn:
    st.session_state.search_input = ""
    st.session_state.selected_genre = "All Genres"
    st.rerun()


# -------------------- SEARCH RESULTS --------------------
if search_btn:
    st.session_state.search_input = search_input
    st.session_state.selected_genre = "All Genres"

    recs = robust_recommend(search_input)
    if not recs:
        st.error("No strong recommendations found.")
    else:
        cols = st.columns(5)
        for i, (title, score) in enumerate(recs):
            with cols[i % 5]:
                poster = fetch_poster(title)
                if poster:
                    movie_card(title, poster, score)


# -------------------- GENRE RESULTS --------------------
if genre_btn:
    st.session_state.search_input = ""
    st.session_state.selected_genre = selected_genre

    if selected_genre == "All Genres":
        sample = movies.sample(10)
    else:
        sample = movies[movies["genres"].apply(lambda x: selected_genre in x)]
        sample = sample.sample(min(10, len(sample)))

    cols = st.columns(5)
    for i, row in enumerate(sample.itertuples()):
        with cols[i % 5]:
            poster = fetch_poster(row.title)
            if poster:
                movie_card(row.title, poster, "—")
