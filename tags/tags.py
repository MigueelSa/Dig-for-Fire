import json, os, re
from functools import lru_cache
from typing import Any
from utils.paths import resource_path

class Tags:

    def __init__(self):
        self.repo_file                                      =   resource_path("tags", "MusicBrainz-genres_repo.json")
        self.canonical_genres, self.aliases, self.parents   =   self._load_repo()
        #self.skipped_genres: set[str]                       =   set()


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

                canonical_genre = self._normalize_tag(canonical_genre)
                canonical_genre_split, parents, has_parents = canonical_genre.split("_"), [], False
                # this misses things like rock and rockabilly but avoids things rap and trap
                if canonical_genre not in self.parents:
                    for cg in canonical_genre_split:
                        if cg in self.parents:
                            parents.append(cg)
                            has_parents = True
                if has_parents:
                    self.edit_repo_genre(canonical_genre, alias = canonical_genre, parent = parents)
                else:
                    self.edit_repo_genre(canonical_genre, alias = canonical_genre)

        self._save_repo()

    @lru_cache(maxsize=None)
    def _normalize_tag(self, tag_name: str) -> str | None:
        if tag_name is None:
            return
        tag_key = tag_name.lower().replace("/", " ").replace("-", " ").replace(" ", "_").replace("&", "and").strip()

        if tag_key in self.canonical_genres:
            return tag_key
        elif tag_key in self.aliases:
            return self.aliases[tag_key]
        else:
            #has_dots                =   "." in tag_key
            is_year                 =   re.match(r'^(\d{4})$', tag_key.strip())
            is_decade               =   re.match(r'^\d{2}s$', tag_key.strip())
            #has_digits_and_letters  =   re.match(r'^(?=.*\d)(?=.*[a-zA-Z]).+$', tag_key.strip())
            #if has_dots or has_digits_and_letters:
                #return
            if is_year:
                tag_key = int(tag_key.strip())
                tag_key = tag_key // 10
                tag_key = str(tag_key) + "0s"
            elif is_decade:
                if int(tag_key[:-1].strip()) > 30:
                    tag_key = "19" + tag_key
                else:
                    tag_key = "20" + tag_key
            return tag_key

    def get_decade(self, date: str | None) -> str | None:
        if date is None:
            return None
        year = None
        if isinstance(date, str) and len(date) >= 4:
            y = date[:4]
            if y.isdigit():
                year = int(y)
        if year is not None:
            return str(year//10) + "0s"
        else:
            return None 

    def genres_tags(self, tag_names: list[Any]) -> tuple[list[str], list[str]]:
        genres, tags = [], []
        for tag_name in tag_names:
            normalized = self._normalize_tag(tag_name)
            if normalized is None:
                continue
            if normalized in self.canonical_genres or normalized in self.aliases.values():
                genres.append(normalized)
            else:
                tags.append(normalized)
        return genres, tags


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

    def _get_parents_children(self) -> dict[str, list[str]]:
        parents_children = {}
        for genre in self.canonical_genres:
            parents = self.parents.get(genre, [])
            for p in parents:
                parents_children.setdefault(p, [])
                if genre not in parents_children[p]:
                    parents_children[p].append(genre)
        return parents_children

if __name__ == "__main__":
    genres = Tags()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_txt = os.path.abspath(os.path.join(script_dir, "MusicBrainz-genres_repo.txt"))
    #genres.build_repo_from_txt(repo_txt)
    print(genres._get_parents_children())
