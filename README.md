<table width="100%">
  <tr>
    <td width="30%">
      <img src="digforfire/images/logo.png" alt="logo" width="200">
    </td>
    <td width="70%">
      <p>
        <strong>Dig-for-Fire</strong> is a content-based music recommendation engine built on top of <strong>MusicBrainz metadata</strong>.
        It imports your album library from <strong>Spotify</strong> or a local JSON file, enriches it with structured genre and tag
        information, builds multiple embedding spaces, and explores adjacent musical regions to recommend albums you don't own yet.
      </p>
      <p>
        It ships with both a <strong>command-line interface</strong> and a <strong>FastAPI web app</strong>.
      </p>
    </td>
  </tr>
</table>

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI](#cli)
  - [Web app](#web-app)
  - [Example JSON library](#example-json-library)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Development](#development)
- [Contact](#contact)

---

## Features

- **Library import** — from Spotify (via the Spotify Web API) or a local JSON file. Spotify export files are auto-detected and converted.
- **MusicBrainz enrichment** — release-group metadata, tag lists, and artist credits, with genres normalized through a directed ontology.
- **Two embedding spaces** — a graph-driven **GenreSpace** (co-occurrence / PMI / PPMI / SVD, with optional smoothing) and a language-driven **TagSpace** (sentence-transformer embeddings clustered with KMeans).
- **Taste-vector recommendations** — candidates fetched live from MusicBrainz and Last.fm, ranked by cosine similarity against your library's taste vector.
- **Optional ML layer** — a RandomForest classifier trained on your recommendation-feedback history to reweight the ranking.
- **Two interfaces** — a CLI (`digforfire`) and a FastAPI web app (`digforfire-api`).
- **Caching** — enriched libraries and embedding matrices are cached to disk so only the first run pays the MusicBrainz rate-limit cost.

---

## Installation

Requires **Python ≥ 3.11**.

1. Clone the repo

```bash
git clone https://github.com/MigueelSa/dig-for-fire.git
cd dig-for-fire
```

2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

3. Install the package (editable)

```bash
pip install -e .
```

This exposes two console commands:

| Command | Description |
|---------|-------------|
| `digforfire`     | CLI entry point |
| `digforfire-api` | Launches the FastAPI web app with uvicorn |

For development tooling (tests), install the extra:

```bash
pip install -e ".[dev]"
```

---

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Required for | Where to get it |
|----------|--------------|-----------------|
| `APP_NAME`, `APP_EMAIL` | MusicBrainz identification | You set these; MusicBrainz asks clients to identify themselves |
| `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI` | Spotify import | [Spotify Developer Dashboard](https://developer.spotify.com/documentation/web-api) |
| `LASTFM_API_KEY` | Recommendations (artist similarity) | [Last.fm API](https://www.last.fm/api) |
| `API_HOST`, `API_PORT` | Web app | Defaults to `127.0.0.1:8000` |

> **Note:** Spotify import and Last.fm-backed recommendations only need their credentials filled in when you actually use those features. Enriching a local JSON library works without any API keys beyond MusicBrainz identification.

---

## Usage

### CLI

| Flag | Arguments | Description | Example |
|------|-----------|-------------|---------|
| `--library_path` | Path to a JSON library | Load & enrich a local library. Spotify export files are auto-converted. | `--library_path data/YourLibrary.json` |
| `--add_album` | `TITLE ARTIST` | Add a single album to the enriched library | `--add_album "Lá Vem a Morte" "Boogarins"` |
| `--recommend` | (none) | Generate recommendations | `--recommend` |
| `--k` | Integer | Number of recommendations (default `2`) | `--recommend --k 5` |

**1. Enrich a local library with MusicBrainz**

```bash
digforfire --library_path data/path/to/library.json
```

This queries MusicBrainz for every album and writes the enriched library to
`data/MusicBrainz-Dig-for-Fire.json` (and a `.pkl` mirror). The first run can be
slow because of MusicBrainz's rate limits; subsequent runs reuse the cache.

**2. Import your Spotify library (optional)**

Fill the Spotify values in `.env`, then run enrichment against the exported file — Spotify JSON is detected and converted automatically before enrichment.

**3. Fetch recommendations**

```bash
digforfire --recommend --k 5
```

Requires an enriched library and a `LASTFM_API_KEY` in `.env`. Recommendations are
appended to `data/recommendation-history-Dig-for-Fire.json` so the same albums
aren't suggested twice.

**4. Add a single album**

```bash
digforfire --add_album "Lá Vem a Morte" "Boogarins"
```

### Web app

```bash
digforfire-api
```

Then open `http://127.0.0.1:8000`. On first launch, `.env` is created from
`.env.example` if it doesn't exist. The web app serves:

| Route | Method | Purpose |
|-------|--------|---------|
| `/`                | GET  | Recommendations page (redirects to `/user` if no library exists) |
| `/user`            | GET  | Onboarding / library-import page |
| `/recommend`       | GET  | JSON recommendations |
| `/library`         | GET  | JSON of the enriched library |
| `/history`         | GET  | JSON of past recommendations |
| `/enrich-library`  | POST | Upload a JSON library; enrichment runs as a background task |
| `/progress/{id}`   | GET  | Poll enrichment progress (current / total / ETA) |
| `/user/spotify/*`  | GET/POST | Save & check credentials, import Spotify library |

Album covers are pulled from the [Cover Art Archive](https://coverartarchive.org/).

### Example JSON library

A library is a JSON array of albums. Minimal required fields:

```json
[
  { "artist": ["Boogarins"], "album": "Lá Vem a Morte" },
  { "artist": ["Fausto"],    "album": "O despertar dos alquimistas" },
  { "artist": ["Bob Dylan"], "album": "New Morning" }
]
```

---

## How it works

```
Library JSON  ──►  MusicBrainz Enrichment  ──►  Genre Ontology + Tags
                                                       │
                                        ┌──────────────┴──────────────┐
                                   GenreSpace                     TagSpace
                                   (graph/PMI/SVD)            (sentence-transformers)
                                        └──────────────┬──────────────┘
                                                Album Embeddings
                                                       │
                                                  Taste Vector
                                                       │
                                          Explorer  ──►  Candidates
                                          (MusicBrainz + Last.fm)
                                                       │
                                            Ranking (cosine + optional ML)
                                                       │
                                             Final Recommendations
```

### 1. Library enrichment
Each `(artist, album)` pair is looked up on MusicBrainz to fetch release-group
metadata, tags, and artist credits. Genres are normalized (case, separators,
aliases) and linked into a **directed ontology**: each album stores its genres
with a minimal ancestor distance, plus its raw tags.

```python
{
  "genres": { "psychedelic_rock": 0, "rock": 1 },   # value = depth in the ontology
  "tags":   ["lofi", "experimental"]
}
```

### 2. Embedding spaces
Two complementary spaces are built from the enriched library:

| GenreSpace | TagSpace |
|------------|----------|
| Structural style | Descriptive semantics |
| Graph-driven co-occurrence | Language-driven |

**GenreSpace** methods: `cooc` (row-normalized co-occurrence), `pmi`
(`log P(i,j)/(P(i)P(j))`), `ppmi` (`max(PMI, 0)`), `svd` (truncated SVD).
Optional smoothing propagates weight to ontology parents (ancestor), blends
nearby genres (neighborhood), or learns graph embeddings (Node2Vec).

**TagSpace** encodes unique tags with `sentence-transformers/all-MiniLM-L6-v2`,
clusters them with KMeans, and treats cluster centroids as semantic regions.

### 3. Taste vector & recommendation
Each album embedding is the concatenation of its genre and tag vectors. The
user's **taste vector** is derived from the whole library. The `Explorer`
probabilistically samples seen genres (by frequency), unseen children of root
genres, tags, and artists; the `Fetcher` retrieves live candidates from
MusicBrainz and Last.fm, excludes owned and previously-recommended albums, and
ranks the rest:

```python
score = weighted cosine(genre_space) + weighted cosine(tag_space)
```

with threshold filtering applied.

### 4. Optional ML layer
`ml/predictor.py` trains a **RandomForest** on your recommendation-feedback
history (owned vs. recommended albums, using concatenated genre+tag embeddings)
and derives a macro-F1-based `alpha` weight to blend into the ranking, with
safety checks for class imbalance.

### Caching
Enriched libraries are cached as `data/MusicBrainz-Dig-for-Fire.{json,pkl}`.
Embedding matrices are cached as `data/embeddings-<hash>-Dig-for-Fire.npz`,
where the hash covers the vocabulary, library snapshot, method, token type, and
embedding dimension — so embeddings are only recomputed when something relevant
changes.

---

## Project structure

```
Dig-for-Fire/
├── digforfire/
│   ├── main.py               # CLI entry point
│   ├── api.py                # FastAPI app
│   ├── analysis/             # Library analysis helpers
│   ├── config/               # Config from .env + pyproject.toml
│   │   ├── config_base.py    # env/pyproject loading, contact info
│   │   ├── config.py         # CLI config
│   │   └── config_api.py     # web-app config
│   ├── libraries/            # Spotify + MusicBrainz fetching (Library ABC)
│   │   ├── libraries.py
│   │   └── converters.py     # Spotify JSON → standard format
│   ├── embeddings/
│   │   ├── embeddings.py     # abstract base (caching, hashing)
│   │   ├── genre_space.py    # co-occurrence, PMI, SVD, Node2Vec
│   │   └── tag_space.py      # tag clustering embeddings
│   ├── recommender/
│   │   ├── recommend.py      # orchestrator
│   │   ├── fetcher.py        # fetches & scores candidates
│   │   ├── explorer.py       # random token sampling
│   │   └── history.py        # recommendation-history persistence
│   ├── ml/predictor.py       # RandomForest classifier
│   ├── tags/tags.py          # genre taxonomy, normalization, ancestor graph
│   ├── models/models.py      # type aliases (AlbumData, LibraryData, …)
│   ├── scripts/              # add_album, import_spotify_library, run_api
│   ├── static/ · templates/  # web-app assets (JS/CSS + Jinja2)
│   ├── utils/                # path helpers, loading animation, album helpers
│   └── tests/                # pytest suite
├── data/                     # cached libraries, embeddings, history (gitignored)
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Development

Install dev dependencies and run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

Tests live under `digforfire/tests/` and cover the Spotify converter and tag
normalization. Continuous integration runs `pytest` on every push and pull
request via GitHub Actions (`.github/workflows/ci.yml`, Python 3.13).

---

## Contact

Maintained by **Miguel Sá** — [miguel.bcmsa@proton.me](mailto:miguel.bcmsa@proton.me) ·
[GitHub](https://github.com/MigueelSa) ·
[LinkedIn](https://www.linkedin.com/in/miguel-sa-47077023b)

Licensed under the **MIT License**.
