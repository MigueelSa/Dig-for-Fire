# Dig-for-Fire

Prototype music recommendation system.

> Work-in-progress. Machine learning integration and web interface planned.

---

## What this project does

Dig-for-Fire takes a personal music library and:

1. Enriches albums with high-quality metadata from MusicBrainz  
2. Normalizes genres using a canonical genre + parent hierarchy  
3. Builds a taste profile from the user’s library  
4. Discovers new albums by querying MusicBrainz and ranking results
   via similarity to the user’s taste

The system is designed to be **incremental**:
already-recommended albums are tracked and excluded in future runs.

---

## Recommendation approach (current)

- **Representation**
  - Genres and tags are embedded using TF-IDF
  - Separate genre and tag spaces are maintained
  - Parent genres are pruned to avoid redundancy

- **User taste model**
  - Taste vectors are computed as normalized sums over the library
  - This makes the model interpretable and stable

- **Discovery**
  - Candidate albums are fetched from MusicBrainz using weighted
    random sampling over existing genres/tags
  - Albums are ranked via cosine similarity
  - Thresholding avoids low-quality matches
 
---

## Project structure

```
libraries/      Fetch and normalize data from music platforms
tags/           Canonical genre system with aliases and parent relations
recommender/    Similarity-based recommendation engine
analysis/       Exploratory analysis and visualization tools
utils/          Path and IO utilities
```
---

## Current state

- Fully functional CLI pipeline
- MusicBrainz enrichment implemented
- Recommendation history persistence
- Actively evolving recommender logic

Known limitations (by design, for now):
- No collaborative filtering
- No deep learning models yet
- CLI-first interface

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

- Improved genre–parent weighting
- Smarter candidate sampling strategies
- Optional ML-based embeddings
- Web interface for interaction and visualization   
