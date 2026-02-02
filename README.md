# Dig-for-Fire

**Dig-for-Fire** is a content-based music recommendation system built on top of **MusicBrainz** metadata.
It analyzes a user’s existing music library, builds semantic embeddings from genres and tags, and explores adjacent musical space to recommend new albums.

> Work-in-progress. Machine learning integration and web interface planned.

---

## What this project does

1. Start from a **local library**.
2. **Enrich albums with MusicBrainz** (genres, tags, hierarchy).
3. Normalize genres into a **directed ontology** (roots → subgenres).
4. Embed albums into **genre space** and **tag space**.
5. Aggregate user taste vectors from the full library.
6. Randomly explore **weighted genres/tags**.
7. Fetch candidate albums from MusicBrainz.
8. Rank candidates by **cosine similarity** to user taste.

The result is a recommender that:

- explores beyond obvious genres
- favors stylistic proximity
- remains interpretable

The system is designed to be **incremental**:
already-recommended albums are tracked and excluded in future runs.

---

## Project structure

```
Dig-for-Fire/
├── analysis/           # Exploratory analysis & plotting
├── embeddings/         # Album → vector embeddings
├── libraries/          # MusicBrainz ingestion
├── models/             # Core data types and method enums
├── recommender/        # Recommendation logic
├── tags/               # Genre normalization & ontology
├── utils/              # Paths, album helpers
├── main.py             # CLI entry point

```
---

## Genre system

- Genres are normalized (case, separators, aliases).
- A **directed ancestor graph** encodes relationships.
- Root genres (e.g. `rock`, `jazz`, `electronic`) are treated separately.
- Distance from leaf genres to ancestors is computed and stored.

---

## Embeddings

Each album is embedded twice:

- **Genre vector**
- **Tag vector**

Supported methods:

- `cooc`
- `pmi`
- `ppmi`
- `svd`

Embeddings are aggregated across the user’s library to form **taste vectors**, which are then compared against candidate albums using cosine similarity.

---

## Recommendation strategy

- Tokens (genres / roots / tags) are sampled **probabilistically**, weighted by frequency.
- Previously seen albums and past recommendations are excluded.
- Candidate albums are fetched live from MusicBrainz.
- Final ranking uses a weighted genre/tag similarity score.

This balances:

- exploitation (known tastes)
- exploration (unseen but related styles)

---

## Installation
1. Clone the repo

```bash
git clone https://github.com/MigueelSa/dig-for-fire.git
cd dig-for-fire
```

2. Create a virtual environment (Python ≥3.11)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --library_path path/to/Spotify-Dig-for-Fire.json
```

- The library file must be a JSON file containing your albums.

**Note**: The first run may take a while due to MusicBrainz’s API rate limits. Subsequent runs are much faster.


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

Optional: Build standalone executable

If you want to run Dig-for-Fire **without installing Python or dependencies**, you can create a standalone executable using PyInstaller.

1. Install PyInstaller in your virtual environment

```bash
pip install pyinstaller
```
2. Build a single-file executable

```bash
pyinstaller --onefile --add-data tags/MusicBrainz-genres_repo.json:tags --name dig-for-fire main.py
```
3. After building, the executable will be in the `dist/` folder. Run it like this:

```bash
./dist/dig-for-fire --library_path path/to/library.json
```

- The executable will only work on the platform you built it on (Windows/Linux/macOS).

---

## Future Plans

- **Genre graph**: nodes = genres; edges = taxonomy (sub → parent) + co-occurrence.
- **Mood layer**: aggregate genres into moods.
- **Graph-based recommendations**: explore related genres systematically.
- **Web UI**: visualize graph and recommendations.   
