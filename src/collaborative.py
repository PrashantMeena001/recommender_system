"""
collaborative.py — Item-Based Collaborative Filtering

Uses the user-item ratings matrix to compute item-item cosine similarity.
The idea: two items are similar if users tend to rate them similarly.
A movie rated highly by the same cluster of users as another movie
will have a high cosine similarity score.

This is "item-based" (not user-based) because we compare items' rating
profiles across all users, which is more stable and scalable for
typical recommendation scenarios.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_user_item_matrix():
    """Load the user-item matrix built in data_prep."""
    matrix = pd.read_csv(
        os.path.join(PROCESSED_DIR, "user_item_matrix.csv"), index_col=0
    )
    # Column names come back as strings from CSV — convert to int
    matrix.columns = matrix.columns.astype(int)
    return matrix


def compute_item_similarity(user_item_matrix):
    """
    Compute item-item cosine similarity.

    We transpose the matrix so rows = items, columns = users.
    Each item's "feature vector" is how all 943 users rated it.
    Cosine similarity then measures the angle between two items'
    rating vectors — items rated by similar users get high scores.
    """
    item_matrix = user_item_matrix.T  # shape: (items, users)
    similarity = cosine_similarity(item_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )
    return similarity_df


def get_collaborative_similar_items(item_id, similarity_df, top_n=10):
    """
    Return the top-N most similar items to the given item, ranked by
    collaborative cosine similarity. Excludes the item itself.
    """
    if item_id not in similarity_df.index:
        return pd.DataFrame(columns=["item_id", "similarity_score"])

    scores = similarity_df[item_id].drop(item_id).sort_values(ascending=False)
    top = scores.head(top_n)
    result = pd.DataFrame({
        "item_id": top.index,
        "similarity_score": top.values,
    }).reset_index(drop=True)
    return result


def run():
    """Build the collaborative similarity matrix, save it, and run sanity checks."""

    print("Loading user-item matrix...")
    user_item_matrix = load_user_item_matrix()
    print(f"Matrix shape: {user_item_matrix.shape}")

    print("Computing item-item cosine similarity...")
    similarity_df = compute_item_similarity(user_item_matrix)
    print(f"Similarity matrix shape: {similarity_df.shape}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "item_similarity_matrix.pkl")
    joblib.dump(similarity_df, save_path)
    print(f"Saved to {save_path}")

    # Load movie titles for readable output
    movies = pd.read_csv(os.path.join(PROCESSED_DIR, "movies.csv"))
    title_map = dict(zip(movies["movie_id"], movies["title"]))

    # Sanity check on a few well-known movies
    sample_items = [1, 50, 181]  # Toy Story, Star Wars, Return of the Jedi
    sample_names = ["Toy Story", "Star Wars", "Return of the Jedi"]

    for item_id, name in zip(sample_items, sample_names):
        print(f"\n{'=' * 60}")
        print(f"TOP 10 SIMILAR TO: {name} (ID={item_id})")
        print(f"{'=' * 60}")
        results = get_collaborative_similar_items(item_id, similarity_df, top_n=10)
        results["title"] = results["item_id"].map(title_map)
        # Also show genres for validation
        genre_map = dict(zip(movies["movie_id"], movies["genres"]))
        results["genres"] = results["item_id"].map(genre_map)
        print(results.to_string(index=False))


if __name__ == "__main__":
    run()
