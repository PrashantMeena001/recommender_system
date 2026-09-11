"""
utils.py — App helpers for the Streamlit recommender UI

Loads the saved model artifacts once (cached) and exposes a clean
recommend() function so app.py doesn't need to deal with file paths
or raw similarity matrices.
"""

import os
import sys
import pandas as pd
import joblib

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_artifacts():
    """Load all saved artifacts needed for recommendations."""
    collab_sim = joblib.load(os.path.join(MODELS_DIR, "item_similarity_matrix.pkl"))
    content_sim = joblib.load(os.path.join(MODELS_DIR, "content_similarity_matrix.pkl"))
    movies = pd.read_csv(os.path.join(PROCESSED_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(PROCESSED_DIR, "ratings.csv"))

    for col in ["title", "directors", "actors", "genres"]:
        movies[col] = movies[col].fillna("")

    return collab_sim, content_sim, movies, ratings


def recommend(item_id, collab_sim, content_sim, movies, top_n=5, alpha=0.6):
    """
    Thin wrapper that imports and delegates to src.hybrid.recommend().
    Keeps the Streamlit app decoupled from the pipeline internals.
    """
    from src.hybrid import recommend as hybrid_recommend
    return hybrid_recommend(item_id, collab_sim, content_sim, movies, top_n, alpha)


def get_movie_stats(ratings):
    """Compute per-movie rating stats for display in the app."""
    stats = ratings.groupby("item_id").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean"),
    ).reset_index()
    return stats
