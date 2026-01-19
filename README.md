# Dig-for-Fire

Prototype music recommendation system.

> Work-in-progress. Machine learning integration planned.

## Features

- Fetch and enrich music libraries.
- Generate recommendations based on genre/tag similarity.
- Save recommendation history for incremental updates.
- Visualize top tags in the library.

## Installation
1. Clone the repo:

```bash
git clone https://github.com/MigueelSa/dig-for-fire.git
cd dig-for-fire
```

2. Create a virtual environment

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

## Future Plans

- Add machine learning for smarter recommendations.
- Implement a web interface for easier interaction.     
