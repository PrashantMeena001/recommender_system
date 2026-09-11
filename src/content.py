"""
content.py — Content-Based Filtering

Uses movie metadata (genres, directors, actors) to compute item-item
similarity via TF-IDF + cosine similarity.

The idea: two movies are "content-similar" if they share genres, directors,
or actors. This is completely independent of user behavior — it works even
for brand-new movies nobody has rated yet (solving cold-start).

We concatenate all text features into a single string per movie, then use
TF-IDF to vectorize, then cosine similarity to compare. TF-IDF naturally
down-weights common terms (actors who appear in dozens of films) and
up-weights distinctive ones.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_movies():
    movies = pd.read_csv(os.path.join(PROCESSED_DIR, "movies.csv"))
    for col in ["title", "directors", "actors", "genres"]:
        movies[col] = movies[col].fillna("")
    return movies


def build_content_features(movies):
    """
    Create a single text blob per movie by combining genres, directors, and actors.
    Genres are the strongest content signal, so we repeat them to give extra weight.
    """
    movies["content_blob"] = (
        movies["genres"].str.replace(" ", " ", regex=False) + " "
        + movies["genres"].str.replace(" ", " ", regex=False) + " "  # double-weight genres
        + movies["directors"].str.replace(" ", " ", regex=False) + " "
        + movies["actors"].str.replace(" ", " ", regex=False)
    )
    return movies


def compute_content_similarity(movies):
    """
    Vectorize content blobs with TF-IDF then compute pairwise cosine similarity.
    Returns a DataFrame indexed/columned by movie_id.
    """
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies["content_blob"])

    similarity = cosine_similarity(tfidf_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=movies["movie_id"].values,
        columns=movies["movie_id"].values,
    )
    return similarity_df


def get_content_similar_items(item_id, similarity_df, top_n=10):
    """
    Return the top-N most similar items by content similarity.
    Same signature as the collaborative version for easy swapping.
    """
    if item_id not in similarity_df.index:
        return pd.DataFrame(columns=["item_id", "similarity_score"])

    scores = similarity_df.loc[item_id].drop(item_id).sort_values(ascending=False)
    top = scores.head(top_n)
    result = pd.DataFrame({
        "item_id": top.index,
        "similarity_score": top.values,
    }).reset_index(drop=True)
    return result


def run():
    """Build content similarity matrix, save it, and compare with collaborative results."""

    print("Loading movie metadata...")
    movies = load_movies()
    print(f"Movies: {len(movies)}")

    print("Building content features...")
    movies = build_content_features(movies)
    # Show a couple examples of the content blob
    print("\nSample content blobs:")
    for _, row in movies.head(3).iterrows():
        blob_preview = row["content_blob"][:120] + "..."
        print(f"  [{row['movie_id']}] {row['title']}: {blob_preview}")

    print("\nComputing TF-IDF + cosine similarity...")
    similarity_df = compute_content_similarity(movies)
    print(f"Similarity matrix shape: {similarity_df.shape}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "content_similarity_matrix.pkl")
    joblib.dump(similarity_df, save_path)
    print(f"Saved to {save_path}")

    title_map = dict(zip(movies["movie_id"], movies["title"]))
    genre_map = dict(zip(movies["movie_id"], movies["genres"]))

    # Same sample items as collaborative.py for side-by-side comparison
    sample_items = [1, 50, 181]
    sample_names = ["Toy Story", "Star Wars", "Return of the Jedi"]

    for item_id, name in zip(sample_items, sample_names):
        print(f"\n{'=' * 60}")
        print(f"TOP 10 CONTENT-SIMILAR TO: {name} (ID={item_id})")
        print(f"{'=' * 60}")
        results = get_content_similar_items(item_id, similarity_df, top_n=10)
        results["title"] = results["item_id"].map(title_map)
        results["genres"] = results["item_id"].map(genre_map)
        print(results.to_string(index=False))


if __name__ == "__main__":
    run()
