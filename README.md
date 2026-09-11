# 🎬 Hybrid Movie Recommender System

A clean, explainable movie recommendation engine that blends **item-based collaborative filtering** with **content-based filtering**, served through an interactive **Streamlit app** with real-time inference.

No deep learning, no neural networks — just classic, interpretable ML techniques that work well and are easy to explain.

## Approach

### 1. Item-Based Collaborative Filtering
- Builds a **user-item ratings matrix** (943 users × 1,682 movies) from MovieLens 100K
- Computes **item-item cosine similarity** on the transposed matrix — two movies are similar if they're rated similarly by the same users
- Captures behavioral patterns: "users who liked X also liked Y"

### 2. Content-Based Filtering
- Uses movie metadata (genres, directors, actors) from a combined CSV
- Vectorizes each movie's metadata with **TF-IDF** (genres are double-weighted)
- Computes **cosine similarity** on the TF-IDF vectors
- Works even for brand-new movies with zero ratings (cold-start resilient)

### 3. Hybrid Blending
- Combines both signals: `hybrid_score = α × collab_score + (1-α) × content_score`
- Default α = 0.6 (favoring collaborative, supplemented by content)
- **Cold-start fallback**: if a movie lacks collaborative data, automatically switches to content-only mode

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings (1–5) from 943 users on 1,682 movies.

- **Ratings**: `data/raw/ml-100k/u.data` (user_id, item_id, rating, timestamp)
- **Metadata**: `data/raw/movielens_100k.csv` (movie_id, title, year, directors, actors, genres)
- **Sparsity**: 93.7% of the user-item matrix is unrated

## Project Structure

```
├── app/
│   ├── app.py              # Streamlit app
│   └── utils.py            # artifact loading + recommendation wrapper
├── data/
│   ├── raw/                # original dataset files
│   └── processed/          # cleaned ratings, movies, user-item matrix
├── models/                 # saved similarity matrices (.pkl)
├── outputs/                # sample recommendation CSVs
├── src/
│   ├── data_prep.py        # data loading, validation, matrix building
│   ├── collaborative.py    # item-based cosine similarity on ratings
│   ├── content.py          # TF-IDF + cosine similarity on metadata
│   └── hybrid.py           # blending + cold-start handling
└── requirements.txt
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data (builds user-item matrix, saves to `data/processed/`)
```bash
python -m src.data_prep
```

### 3. Build collaborative similarity matrix
```bash
python -m src.collaborative
```

### 4. Build content similarity matrix
```bash
python -m src.content
```

### 5. Test the hybrid recommender
```bash
python -m src.hybrid
```

### 6. Launch the Streamlit app
```bash
streamlit run app/app.py
```

Then open http://localhost:8501 in your browser.

## Example Results

### Toy Story (α=0.6, top 10)

| Rank | Movie | Hybrid Score | Mode |
|------|-------|-------------|------|
| 1 | Star Wars | 0.482 | hybrid |
| 2 | Aladdin | 0.472 | hybrid |
| 3 | Return of the Jedi | 0.466 | hybrid |
| 4 | Independence Day | 0.448 | hybrid |
| 5 | The Rock | 0.439 | hybrid |

### Star Wars (α=0.6, top 5)

| Rank | Movie | Hybrid Score | Mode |
|------|-------|-------------|------|
| 1 | Return of the Jedi | 0.659 | hybrid |
| 2 | Empire Strikes Back | 0.578 | hybrid |
| 3 | Raiders of the Lost Ark | 0.550 | hybrid |
| 4 | Independence Day | 0.487 | hybrid |
| 5 | Toy Story | 0.482 | hybrid |

## Key Design Decisions

- **Item-based over user-based CF**: Item similarity is more stable (items don't change; user tastes do) and more scalable
- **Cosine similarity over Pearson**: Works well with sparse data and doesn't require mean-centering
- **TF-IDF over one-hot encoding**: Naturally handles multi-word genre labels and down-weights overly common terms (e.g., actors in 50+ movies)
- **Configurable α**: Lets you demo the spectrum from pure collaborative to pure content-based in real time
- **Cold-start detection**: If max collaborative similarity < 0.01, falls back gracefully

## Tech Stack

- Python 3.11
- pandas, NumPy — data manipulation
- scikit-learn — TF-IDF vectorization, cosine similarity
- joblib — model serialization
- Streamlit — interactive web UI
