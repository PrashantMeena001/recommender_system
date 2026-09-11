"""
data_prep.py — Data Inspection & Preparation

Loads the MovieLens 100K dataset, which comes in two parts:
  1. u.data: 100,000 user-item ratings (user_id, item_id, rating, timestamp)
  2. movielens_100k.csv: movie metadata (movie_id, title, year, directors, actors, genres)

Builds a user-item ratings matrix (rows=users, cols=items, values=ratings).
Saves cleaned data to data/processed/ for downstream use.
"""

import os
import pandas as pd
import numpy as np

# All paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def load_ratings():
    """Load the ratings from the original MovieLens u.data file (tab-separated)."""
    ratings_path = os.path.join(RAW_DIR, "ml-100k", "u.data")
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    return ratings


def load_movie_metadata():
    """
    Load movie metadata from movielens_100k.csv.
    This file has: movie_id, title, year, directors, actors, genres.
    Genres are space-separated (e.g., "Animation Adventure Comedy").
    """
    meta_path = os.path.join(RAW_DIR, "movielens_100k.csv")
    movies = pd.read_csv(meta_path)

    # Fill NaN in text columns with empty strings so downstream code doesn't break
    for col in ["title", "directors", "actors", "genres"]:
        movies[col] = movies[col].fillna("")

    # Year can be NaN for some entries — fill with 0 (or leave as-is)
    movies["year"] = movies["year"].fillna(0).astype(int)

    return movies


def build_user_item_matrix(ratings):
    """
    Pivot ratings into a user×item matrix.
    Missing ratings are filled with 0 (needed for cosine similarity — NaN would
    propagate through the computation and give us garbage results).
    """
    matrix = ratings.pivot_table(
        index="user_id", columns="item_id", values="rating", fill_value=0
    )
    return matrix


def compute_sparsity(matrix):
    """What fraction of the user-item matrix is zero (unrated)?"""
    total_cells = matrix.shape[0] * matrix.shape[1]
    nonzero_cells = (matrix != 0).sum().sum()
    sparsity = 1 - (nonzero_cells / total_cells)
    return sparsity


def run():
    """Run the full data preparation pipeline and print diagnostics."""

    # --- Load ratings ---
    ratings = load_ratings()
    print("=" * 60)
    print("RATINGS DATA")
    print("=" * 60)
    print(f"Shape: {ratings.shape}")
    print(f"Columns: {list(ratings.columns)}")
    print(f"Dtypes:\n{ratings.dtypes}\n")
    print(ratings.head(10))

    # --- Load movie metadata ---
    movies = load_movie_metadata()
    print("\n" + "=" * 60)
    print("MOVIE METADATA")
    print("=" * 60)
    print(f"Shape: {movies.shape}")
    print(f"Columns: {list(movies.columns)}")
    print(f"Dtypes:\n{movies.dtypes}\n")
    print(movies.head(10).to_string())

    # --- Validation checks ---
    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print(f"Unique users:  {n_users}")
    print(f"Unique items:  {n_items}")
    print(f"Total ratings: {len(ratings)}")
    print(f"Rating range:  {ratings['rating'].min()} to {ratings['rating'].max()}")

    # Check coverage: how many rated items actually have metadata?
    rated_items = set(ratings["item_id"].unique())
    meta_items = set(movies["movie_id"].unique())
    missing_meta = rated_items - meta_items
    print(f"Items with ratings but no metadata: {len(missing_meta)}")
    if missing_meta:
        print(f"  IDs: {sorted(missing_meta)[:20]}...")

    # --- Build user-item matrix ---
    user_item_matrix = build_user_item_matrix(ratings)
    sparsity = compute_sparsity(user_item_matrix)
    print(f"\nUser-item matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {sparsity:.4f} ({sparsity * 100:.2f}%)")

    # --- Save processed data ---
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ratings.to_csv(os.path.join(PROCESSED_DIR, "ratings.csv"), index=False)
    movies.to_csv(os.path.join(PROCESSED_DIR, "movies.csv"), index=False)
    user_item_matrix.to_csv(os.path.join(PROCESSED_DIR, "user_item_matrix.csv"))

    print(f"\nSaved to {PROCESSED_DIR}:")
    print("  - ratings.csv")
    print("  - movies.csv")
    print("  - user_item_matrix.csv")

    return ratings, movies, user_item_matrix


if __name__ == "__main__":
    run()
