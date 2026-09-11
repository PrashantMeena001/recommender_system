"""
app.py — Streamlit UI for the Hybrid Movie Recommender

A simple, interactive app where you:
  1. Pick a movie from a dropdown (searchable by title)
  2. Adjust how many recommendations you want (top_n)
  3. Optionally tune the blend weight (alpha) between collaborative and content
  4. Click "Recommend" and see results in a clean table

All expensive loads (similarity matrices, movie data) are cached so the app
responds instantly after the first load.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils import load_artifacts, recommend, get_movie_stats


# --- Page config ---
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


# --- Cached data loading ---
@st.cache_resource
def cached_load():
    """Load artifacts once and cache across reruns."""
    collab_sim, content_sim, movies, ratings = load_artifacts()
    stats = get_movie_stats(ratings)
    return collab_sim, content_sim, movies, ratings, stats


collab_sim, content_sim, movies, ratings, stats = cached_load()

# Build the title→id mapping for the dropdown
movie_options = movies[["movie_id", "title", "year", "genres"]].copy()
movie_options = movie_options.merge(stats, left_on="movie_id", right_on="item_id", how="left")
movie_options["num_ratings"] = movie_options["num_ratings"].fillna(0).astype(int)
movie_options["avg_rating"] = movie_options["avg_rating"].fillna(0).round(2)
movie_options["display"] = (
    movie_options["title"].str.title()
    + " (" + movie_options["year"].astype(str) + ")"
    + " — " + movie_options["num_ratings"].astype(str) + " ratings"
)
# Sort by popularity (most-rated first) so common movies are easy to find
movie_options = movie_options.sort_values("num_ratings", ascending=False).reset_index(drop=True)

id_from_display = dict(zip(movie_options["display"], movie_options["movie_id"]))


# --- UI ---
st.title("🎬 Hybrid Movie Recommender")
st.markdown(
    "Find similar movies using a blend of **collaborative filtering** "
    "(what users with similar tastes liked) and **content-based filtering** "
    "(shared genres, directors, actors)."
)

# Sidebar controls
st.sidebar.header("⚙️ Settings")

selected_display = st.sidebar.selectbox(
    "Pick a movie",
    options=movie_options["display"].tolist(),
    index=0,
    help="Movies sorted by popularity. Start typing to search.",
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=1,
    max_value=20,
    value=10,
)

alpha = st.sidebar.slider(
    "Blend weight (α)",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="α=1.0 → pure collaborative, α=0.0 → pure content-based",
)

recommend_btn = st.sidebar.button("🎯 Recommend", type="primary", use_container_width=True)


# --- Show selected movie info ---
selected_id = id_from_display[selected_display]
selected_movie = movies[movies["movie_id"] == selected_id].iloc[0]
selected_stats = stats[stats["item_id"] == selected_id]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎬 Title", selected_movie["title"].title())
with col2:
    st.metric("📅 Year", int(selected_movie["year"]) if selected_movie["year"] else "N/A")
with col3:
    if not selected_stats.empty:
        st.metric(
            "⭐ Avg Rating",
            f"{selected_stats.iloc[0]['avg_rating']:.1f} / 5",
            f"{int(selected_stats.iloc[0]['num_ratings'])} ratings"
        )
    else:
        st.metric("⭐ Avg Rating", "No ratings", "Cold start")

st.markdown(f"**Genres:** {selected_movie['genres'] if selected_movie['genres'] else 'N/A'}")
if selected_movie["directors"]:
    st.markdown(f"**Director(s):** {selected_movie['directors']}")


# --- Recommendations ---
if recommend_btn:
    st.divider()
    st.subheader(f"Top {top_n} Recommendations")

    with st.spinner("Computing recommendations..."):
        results = recommend(
            selected_id, collab_sim, content_sim, movies, top_n=top_n, alpha=alpha
        )

    if results.empty:
        st.warning("No recommendations found for this movie.")
    else:
        mode = results["mode"].iloc[0]
        if mode == "content-only":
            st.info(
                "⚠️ **Cold-start fallback active** — this movie has insufficient "
                "collaborative data, so recommendations are based purely on content "
                "(genres, directors, actors)."
            )
        else:
            st.success(
                f"✅ **Hybrid mode** — blending collaborative ({alpha:.0%}) "
                f"and content ({1-alpha:.0%}) signals."
            )

        # Format the results table
        display_df = results[["title", "hybrid_score", "collab_score", "content_score", "mode"]].copy()
        display_df.columns = ["Movie", "Hybrid Score", "Collab Score", "Content Score", "Mode"]
        display_df["Movie"] = display_df["Movie"].str.title()
        display_df.index = range(1, len(display_df) + 1)
        display_df.index.name = "Rank"

        # Round scores for readability
        for col in ["Hybrid Score", "Collab Score", "Content Score"]:
            display_df[col] = display_df[col].round(4)

        st.dataframe(display_df, use_container_width=True)

        # Add genres column for context
        genre_map = dict(zip(movies["movie_id"], movies["genres"]))
        results["genres"] = results["item_id"].map(genre_map)

        st.markdown("#### Genre Breakdown")
        genre_df = results[["title", "genres"]].copy()
        genre_df.columns = ["Movie", "Genres"]
        genre_df["Movie"] = genre_df["Movie"].str.title()
        genre_df.index = range(1, len(genre_df) + 1)
        st.dataframe(genre_df, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works:** This system combines item-based collaborative filtering "
    "(cosine similarity on user-item ratings) with content-based filtering "
    "(TF-IDF on genres, directors, actors). The α slider controls the blend."
)
