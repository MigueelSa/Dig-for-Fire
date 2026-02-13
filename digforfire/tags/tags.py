import json, os, re
from functools import lru_cache
from typing import Any

from digforfire.utils.paths import resource_path

class Tags:

    def __init__(self):
        self.repo_file                                      =   resource_path("digforfire", "tags", "MusicBrainz-genres_repo.json")
        self.canonical_genres, self.aliases, self.ancestors =   self._load_repo()
        self.roots                                          =   self._roots()
        self.ancestors_children                             =   self._get_ancestors_children()
        #self.skipped_genres: set[str]                       =   set()


    def _load_repo(self) -> tuple[set, dict, dict]:
        if os.path.exists(self.repo_file):
            with open(self.repo_file, "r") as f:
                data = json.load(f)
        else:
            data = {}
        return set(data.get("canonical_genres", [])), data.get("aliases", {}), data.get("ancestors", {})

    def _save_repo(self) -> None:
        data = {
            "canonical_genres": list(self.canonical_genres),
            "aliases": self.aliases,
            "ancestors": self.ancestors
        }
        with open(self.repo_file, "w") as f:
            json.dump(data, f, indent=4)

    def build_repo_from_txt(self, genre_repo_txt_file: str) -> None:
        txt = os.path.abspath(genre_repo_txt_file)
        if not os.path.exists(txt):
            return
        genres = set()
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
                genres.add(canonical_genre)

        for genre in genres:
                ancestors = {a for a in genres if "_" + a + "_" in genre or genre.startswith(a + "_") or genre.endswith("_" + a)}
                if ancestors:
                    self.edit_repo_genre(genre, alias = genre, ancestor = list(ancestors))
                else:
                    self.edit_repo_genre(genre, alias = genre)

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
        """
        Separate tag names into genres and other tags,
        and compute minimal distance of ancestors from genres.
        
        :param self: Instance of the Tags class.
        :param tag_names: List of tag names to be categorized.
        :type tag_names: list[Any]
        :return: A tuple containing two lists: genres, and other tags.
        :rtype: tuple[list[str], list[str]]
        """

        distance_dict, tags = {}, set()
        for tag_name in tag_names:
            normalized = self._normalize_tag(tag_name)
            if normalized is None:
                continue
            elif normalized in self.canonical_genres or normalized in self.aliases.values():
                    distance_dict[normalized] = 0
            else:
                tags.add(normalized)


        stack = list(distance_dict.keys())
        while stack:
            current = stack.pop()
            for a in self.ancestors.get(current, []):
                new_distance = distance_dict[current] + 1
                if a not in distance_dict or new_distance < distance_dict[a]:
                    distance_dict[a] = new_distance
                    stack.append(a)

        return distance_dict, list(tags)


    def edit_repo_genre(self, canonical_genre: str, alias: str | None = None, ancestor: str | list[str] | None = None) -> None:
        canonical_genres, aliases, ancestors   =   self.canonical_genres, self.aliases, self.ancestors
        
        if canonical_genre not in canonical_genres:
            canonical_genres.add(canonical_genre)
        if alias is not None:
            aliases[alias] = canonical_genre
        if ancestor is not None:
            if isinstance(ancestor, str):
                ancestor = [ancestor]
            ancestors.setdefault(canonical_genre, [])
            for a in ancestor:
                if a not in ancestors[canonical_genre]:
                    ancestors[canonical_genre].append(a)

        self.canonical_genres, self.aliases, self.ancestors   =   canonical_genres, aliases, ancestors

    def _get_ancestors_children(self) -> dict[str, list[str]]:
        """
        Get a mapping of ancestor genres to their child genres.
        
        :param self: Instance of the Tags class.
        :return: A dictionary mapping ancestor genres to lists of their child genres.
        :rtype: dict[str, list[str]]
        """
        ancestors_children = {}
        for genre in self.canonical_genres:
            ancestors = self.ancestors.get(genre, [])
            for a in ancestors:
                ancestors_children.setdefault(a, [])
                if genre not in ancestors_children[a]:
                    ancestors_children[a].append(genre)
        return ancestors_children
    
    def _roots(self) -> set[str]:
        """
        Return the set of root genres in the library.

        A root genre is defined as a genre that has no direct ancestors
        in the `ancestors` mapping (its ancestor list is empty).

        :return: Set of root genre names.
        :rtype: set[str]
        """
        roots = set(genre for genre, ancestors in self.ancestors.items() if not ancestors)
        return roots
    
    def get_library_artists(self, library: list[dict]) -> dict[str, str]:
        artists = {} 
        for album in library:
            for id, artist in enumerate(album.get("artist", [])):
                artists[artist] = album.get("artist_id", [None])[id]
        return artists

if __name__ == "__main__":
    genres = Tags()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_txt = os.path.abspath(os.path.join(script_dir, "MusicBrainz-genres_repo.txt"))

    # manual labor here
    roots = ["rock", "punk", "blues", "classical", "electronic", "folk", "hip_hop", "jazz", "metal", "pop", "soul", "reggae"]
    for root in roots:
        root = genres._normalize_tag(root)
        genres.edit_repo_genre(root, ancestor = [])
    genres.edit_repo_genre("trap", ancestor = ["gangsta_rap", "hardcore_hip_hop", "electronic"])
    genres.edit_repo_genre("progressive", alias="prog")
    #genres.edit_repo_genre("progressive_rock", alias="prog_rock")

    genres.build_repo_from_txt(repo_txt)
