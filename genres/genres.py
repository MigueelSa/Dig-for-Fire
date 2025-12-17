import json, os, re
import numpy as np
from tqdm import tqdm

class Genres:

    def __init__(self):
        self.repo_file                                      =   os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MusicBrainz-genres_repo.json"))
        self.canonical_genres, self.aliases, self.parents   =   self._load_repo()
        self.skipped_genres: set[str]                       =   set()


    def _load_repo(self) -> tuple[set, dict, dict]:
        if os.path.exists(self.repo_file):
            with open(self.repo_file, "r") as f:
                data = json.load(f)
        else:
            data = {}
        return set(data.get("canonical_genres", [])), data.get("aliases", {}), data.get("parents", {})

    def _save_repo(self) -> None:
        data = {
            "canonical_genres": list(self.canonical_genres),
            "aliases": self.aliases,
            "parents": self.parents
        }
        with open(self.repo_file, "w") as f:
            json.dump(data, f, indent=4)

    def build_repo_from_txt(self, genre_repo_txt_file: str) -> None:
        txt = os.path.abspath(genre_repo_txt_file)
        if not os.path.exists(txt):
            return

        with open(txt, "r") as t:
            for line in t:
                line = line.strip()
                if "(" in line and ")" in line:
                    genre_name, description = line.split("(", 1)
                    canonical_genre = genre_name.strip()
                    if description[-1] == ")": # maybe do something with description later
                        description = description[:-1]
                else:
                    canonical_genre, description = line, None

                canonical_genre_split, parents, has_parents = canonical_genre.split(), [], False
                # this misses things like rock and rockabilly but avoids things rap and trap
                for cg in canonical_genre_split:
                    if cg in self.parents:
                        parents.append(cg)
                        has_parents = True
                if canonical_genre in self.parents:
                    parents = canonical_genre
                    has_parents = True
                canonical_genre = canonical_genre.replace("/", " ").replace("-", " ").replace(" ", "_").replace("&", "and").strip()
                if has_parents:
                    self.edit_repo_genre(canonical_genre, alias = canonical_genre, parent = parents)
                else:
                    self.edit_repo_genre(canonical_genre, alias = canonical_genre)

        self._save_repo()

    def normalize_genre(self, genre_name: str) -> str | None:
        genre_key = genre_name.lower().replace("/", " ").replace("-", " ").replace(" ", "_").replace("&", "and").strip()

        if genre_key in self.canonical_genres:
            return genre_key
        elif genre_key in self.aliases:
            return self.aliases[genre_key]
        else:
            has_dots                =   "." in genre_key
            is_year                 =   re.match(r'^(\d{4})$', genre_key.strip())
            is_decade               =   re.match(r'^\d{2,4}s$', genre_key.strip())
            has_digits_and_letters  =   re.match(r'^(?=.*\d)(?=.*[a-zA-Z]).+$', genre_key.strip())
            if has_dots or has_digits_and_letters:
                return None
            elif is_year:
                genre_key = int(genre_key.strip())
                genre_key = genre_key // 10
                genre_key = str(genre_key) + "0s"
            elif is_decade:
                if int(genre_key[:-1].strip()) > 30:
                    genre_key = "19" + genre_key
                else:
                    genre_key = "20" + genre_key
            return genre_key
        #elif genre_key in self.skipped_genres:
            #return None
        '''
        else:
            genre_user_name = input(f"{genre_key} is not recognized. Do you want to add it anyway? If so, type the name you want to save it as. If not, just press enter. ")
            if genre_user_name != "":
                if genre_user_name in self.aliases:
                    genre_user_name = self.aliases[genre_user_name]
                    return genre_user_name
                genre_user_name = genre_user_name.lower().replace("/", " ").replace("-", " ").replace(" ", "_").replace("&", "and").strip()
                self.edit_repo_genre(genre_user_name)
                tqdm.write(f"{genre_user_name} was added.")
                return genre_user_name
            else:
                self.skipped_genres.add(genre_key)
                return None
        '''

    def edit_repo_genre(self, canonical_genre: str, alias: str | None = None, parent: str | list[str] | None = None) -> None:
        canonical_genres, aliases, parents   =   self.canonical_genres, self.aliases, self.parents
        
        if canonical_genre not in canonical_genres:
            canonical_genres.add(canonical_genre)
        if alias is not None:
            aliases[alias] = canonical_genre
        if parent is not None:
            if isinstance(parent, str):
                parent = [parent]
            parents.setdefault(canonical_genre, [])
            for p in parent:
                if p not in parents[canonical_genre]:
                    parents[canonical_genre].append(p)

        self.canonical_genres, self.aliases, self.parents   =   canonical_genres, aliases, parents

if __name__ == "__main__":
    genre = Genres()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.abspath(os.path.join(script_dir, "MusicBrainz-genres_repo.txt"))
    genre.build_repo_from_txt(repo_path)
