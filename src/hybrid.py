"""
hybrid.py — Hybrid Recommender (Collaborative + Content Blending)

Combines both similarity signals into a single recommendation:
  hybrid_score = alpha * collaborative_score + (1 - alpha) * content_score

alpha controls the blend:
  - alpha=1.0 → pure collaborative (what users similar to you liked)
  - alpha=0.0 → pure content-based (what shares genres/actors/directors)
  - alpha=0.6 → default, favoring collaborative but using content as backup

Cold-start handling: if a movie has no collaborative data (e.g., no users
have rated it, so its entire column in the user-item matrix is zero),
the collaborative similarity to all other items is 0. In that case,
we fall back to content-only and flag it.
"""

import os
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_artifacts():
    """Load both similarity matrices and movie metadata."""
    collab_sim = joblib.load(os.path.join(MODELS_DIR, "item_similarity_matrix.pkl"))
    content_sim = joblib.load(os.path.join(MODELS_DIR, "content_similarity_matrix.pkl"))
    movies = pd.read_csv(os.path.join(PROCESSED_DIR, "movies.csv"))
    return collab_sim, content_sim, movies


def recommend(item_id, collab_sim, content_sim, movies, top_n=5, alpha=0.6):
    """
    Generate hybrid recommendations for a given item.

    Returns a DataFrame with columns:
      item_id, title, hybrid_score, collab_score, content_score, mode

    'mode' is either 'hybrid' (both signals used) or 'content-only'
    (cold-start fallback when collaborative data is missing/empty).
    """
    has_collab = item_id in collab_sim.index
    has_content = item_id in content_sim.index

    if not has_content and not has_collab:
        return pd.DataFrame(columns=[
            "item_id", "title", "hybrid_score", "collab_score", "content_score", "mode"
        ])

    # Get content scores for all other items
    if has_content:
        content_scores = content_sim.loc[item_id].copy()
        content_scores = content_scores.drop(item_id, errors="ignore")
    else:
        content_scores = pd.Series(0.0, index=content_sim.columns)

    # Get collaborative scores — check if the item has meaningful data
    cold_start = False
    if has_collab:
        collab_scores = collab_sim[item_id].copy()
        collab_scores = collab_scores.drop(item_id, errors="ignore")

        # Cold-start detection: if max collaborative similarity is near zero,
        # this item has essentially no collaborative signal
        if collab_scores.max() < 0.01:
            cold_start = True
    else:
        cold_start = True
        collab_scores = pd.Series(0.0, index=collab_sim.columns)

    # Align indices — only score items present in both matrices
    common_items = content_scores.index.intersection(collab_scores.index)
    content_aligned = content_scores.reindex(common_items, fill_value=0.0)
    collab_aligned = collab_scores.reindex(common_items, fill_value=0.0)

    if cold_start:
        # Pure content-based fallback
        hybrid_scores = content_aligned
        mode = "content-only"
    else:
        hybrid_scores = alpha * collab_aligned + (1 - alpha) * content_aligned
        mode = "hybrid"

    # Rank and take top N
    top_items = hybrid_scores.sort_values(ascending=False).head(top_n)

    title_map = dict(zip(movies["movie_id"], movies["title"]))

    result = pd.DataFrame({
        "item_id": top_items.index,
        "title": [title_map.get(i, f"Unknown ({i})") for i in top_items.index],
        "hybrid_score": top_items.values,
        "collab_score": [collab_aligned.get(i, 0.0) for i in top_items.index],
        "content_score": [content_aligned.get(i, 0.0) for i in top_items.index],
        "mode": mode,
    }).reset_index(drop=True)

    return result


def run():
    """Demo the hybrid recommender on a popular item and a sparse/cold item."""

    print("Loading artifacts...")
    collab_sim, content_sim, movies = load_artifacts()

    title_map = dict(zip(movies["movie_id"], movies["title"]))
    ratings = pd.read_csv(os.path.join(PROCESSED_DIR, "ratings.csv"))
    rating_counts = ratings["item_id"].value_counts()

    # --- Test 1: Popular movie (Toy Story, ID=1) ---
    item_id = 1
    n_ratings = rating_counts.get(item_id, 0)
    print(f"\n{'=' * 70}")
    print(f"HYBRID RECOMMENDATIONS FOR: {title_map.get(item_id)} (ID={item_id})")
    print(f"Rated by {n_ratings} users — should use HYBRID mode")
    print(f"{'=' * 70}")
    results = recommend(item_id, collab_sim, content_sim, movies, top_n=10, alpha=0.6)
    print(results.to_string(index=False))

    # --- Test 2: Find a sparse/rarely-rated item ---
    # Items with very few ratings
    sparse_items = rating_counts[rating_counts <= 2].index.tolist()
    if sparse_items:
        item_id = sparse_items[0]
        n_ratings = rating_counts.get(item_id, 0)
        print(f"\n{'=' * 70}")
        print(f"HYBRID RECOMMENDATIONS FOR: {title_map.get(item_id, 'Unknown')} (ID={item_id})")
        print(f"Rated by only {n_ratings} user(s) — may trigger COLD-START fallback")
        print(f"{'=' * 70}")
        results = recommend(item_id, collab_sim, content_sim, movies, top_n=10, alpha=0.6)
        print(results.to_string(index=False))

    # --- Test 3: Item that exists in metadata but has zero ratings ---
    all_meta_items = set(movies["movie_id"].values)
    all_rated_items = set(ratings["item_id"].unique())
    unrated = all_meta_items - all_rated_items
    if unrated:
        item_id = sorted(unrated)[0]
        print(f"\n{'=' * 70}")
        print(f"HYBRID RECOMMENDATIONS FOR: {title_map.get(item_id, 'Unknown')} (ID={item_id})")
        print(f"ZERO ratings — should use CONTENT-ONLY mode (cold start)")
        print(f"{'=' * 70}")
        results = recommend(item_id, collab_sim, content_sim, movies, top_n=10, alpha=0.6)
        if results.empty:
            print("  (No recommendations — item has no content or collaborative data)")
        else:
            print(results.to_string(index=False))


if __name__ == "__main__":
    run()
