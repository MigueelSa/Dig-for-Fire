# Dig-for-Fire

**Dig-for-Fire** is a content-based music recommendation engine built on top of **MusicBrainz metadata**.
It enriches a local album library with structured genre and tag information, builds multiple embedding spaces, and explores adjacent musical regions to generate recommendations.

---

## Installation
1. Clone the repo

```bash
git clone https://github.com/MigueelSa/dig-for-fire.git
cd dig-for-fire
```

2. Create a virtual environment (Python ≥3.11)

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -e .
```

## Usage

| Flag | Arguments | Description | Example |
|------|-----------|-------------|---------|
|`--import_spotify` | (no arguments) | Import Spotify Library | `--import_spotify` |
| `--library_path` | Path to JSON library | Load a local library for enrichment | `--library_path=data/Spotify-Dig-for-Fire.json` |
| `--add_album` | TITLE ARTIST | Add a single album to the library | `--add_album "Lá Vem a Morte" "Boogarins"` |
| `--recommend` | (no arguments) | Generate album recommendations | `--recommend` |
| `--k` | Integer | Number of recommendations to generate (default 2) | `--recommend --k 5` |

- The library file must be a JSON file containing your albums.

**Note**: The first run may take a while due to MusicBrainz’s API rate limits. Subsequent runs are much faster.

**Examples**

1. Enrich local library with MusicBrainz
```python
digforfire --library_path data/path/to/local/library.json
```
- **Optional**: Import Spotify library
  - Copy `.env.example` to `.env` and fill in Spotify API values
  - Run the command:
```python
digforfire --import_spotify
```

2. Fetch recommendations
- Obtain a Last.fm API key and write it in `.env`
- Run the command:
```python
digforfire --recommend
```

3. Add albums to your library
```python
digforfire --add_album "Lá Vem a Morte" "Boogarins"
```

**Example JSON library**

**Minimal (only required fields)**

```json
[
  {
    "artist": ["Boogarins"],
    "album": "Lá Vem a Morte"
  },
  {
    "artist": ["Fausto"],
    "album": "O despertar dos alquimistas"
  },
  {
    "artist": ["Bob Dylan"],
    "album": "New Morning"
  }
]
```

---

## Project structure

```
Dig-for-Fire/
│
├── digforfire/
│   ├── __init__.py
|   ├── config.py
|   ├── main.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── analysis.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── genre_space.py
│   │   └── tag_space.py
│   │
│   ├── libraries/
│   │   ├── __init__.py
│   │   └── libraries.py
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   └── predictor.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py
│   │
│   ├── recommender/
│   │   ├── __init__.py
│   │   ├── explorer.py
│   │   ├── fetcher.py
│   │   └── recommend.py
│   │
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── add_album.py
│   │   └── import_spotify_library.py
│   │
│   ├── tags/
│   │   ├── __init__.py
│   │   ├── tags.py
│   │   └── MusicBrainz-genres_repo.json
│   │
│   └── utils/
│       ├── __init__.py
│       ├── albums.py
│       ├── loading.py
│       └── paths.py
│
├── .env.example
├── .gitignore
├── README.md
└── pyproject.toml
```
---

## Core Pipeline

### 1. Library Enrichment

Input: local JSON library (artist + album).

The system:

- Queries MusicBrainz  
- Fetches:
  - Release-group metadata  
  - Tag list  
  - Artist credits  
- Normalizes genres via a directed ontology  
- Stores enriched library as:
  - `MusicBrainz-Dig-for-Fire.json`
  - `MusicBrainz-Dig-for-Fire.pkl`

Incremental updates are supported.

### 2. Genre System (Directed Ontology)

Genres are:

- Normalized (case, separators, aliases)
- Linked via ancestor relationships
- Grouped into **root genres**
- Stored with minimal ancestor distance

Each album stores:

```python
{
  "genres": { "psychedelic_rock": 0, "rock": 1 },
  "tags": ["lofi", "experimental"]
}
```

Distance represents depth in the ontology graph.

### 3. Embedding Spaces

Two independent embedding spaces are constructed.

#### GenreSpace

Supported embedding methods:

| Method | Definition |
|--------|------------|
| `cooc` | Row-normalized co-occurrence matrix |
| `pmi`  | `log( P(i,j) / (P(i)P(j)) )` |
| `ppmi` | `max(PMI, 0)` |
| `svd`  | Truncated SVD of the co-occurrence matrix |


#### Optional Smoothing

Smoothing modifies the raw co-occurrence signal:

- **Ancestor smoothing**  
  Propagates weight to ontology parents.

- **Neighborhood smoothing**  
  Blends nearby genres in co-occurrence space.

- **Node2Vec smoothing**  
  Learns graph embeddings over the genre graph.

#### TagSpace

Tags are embedded semantically using: `sentence-transformers/all-MiniLM-L6-v2`

Process:

1. Encode all unique tags
2. Cluster embeddings with KMeans
3. Use cluster centroids as semantic regions

This creates a **mood / descriptor layer** orthogonal to structural genre space.

### Why Two Spaces?

Genres and tags capture different signals:

| GenreSpace | TagSpace |
|------------|----------|
| Structural style | Descriptive semantics |
| Graph-driven | Language-driven |

### 4. Taste Vector Construction

For each album:

```python
album_embedding = genre_vector ⊕ tag_vector
```

User taste vector is derived from the full library.

Cosine similarity is used for ranking.

### 5. Exploration Strategy

The `Explorer` module probabilistically samples:

- Seen genres (weighted by frequency)
- Roots → unseen children
- Tags
- Random artists

Used tokens are tracked per run.

### 6. Candidate Fetching

Candidates are retrieved live from MusicBrainz:

- Genre-based search
- Tag-based search
- Artist similarity

Already-owned and previously-recommended albums are excluded.

### 7. Ranking

Candidates are ranked by:
```python
score = weighted cosine(genre_space) + weighted cosine(tag_space)
```

Threshold filtering is applied.

Recommendation history is persisted to:

```python
data/recommendation-history-Dig-for-Fire.json
```

## Optional ML Layer: Predictor

A RandomForest classifier can be trained on:
- Previously recommended albums
- Owned library

Features:
- Concatenated genre + tag embeddings

Outputs:
- Macro-F1-derived alpha weight
- Safety checks for class imbalance

A **visual scheme of the pipeline** can be seen in the following graph. 

```python
Library JSON
    ↓
MusicBrainz Enrichment
    ↓
Genre Ontology + Tags
    ↓
GenreSpace      TagSpace
      ↓            ↓
   Album Embeddings
          ↓
     Taste Vector
          ↓
    Explorer → Candidates
          ↓
       Ranking
          ↓
  Final Recommendations
```

---

## Embedding Caching

Embeddings are saved as:
```python
data/embeddings-<hash>-Dig-for-Fire.npz
```

The hash includes:
- Vocabulary
- Library snapshot
- Method
- Token type
- Embedding dimension

## Future Plans

- Web interface for interactive exploration


