import random
from collections import Counter

from digforfire.utils.albums import get_album_genres, get_album_tags
from digforfire.models.models import LibraryData

class Explorer:
    def __init__(self, library: LibraryData, roots: set[str], ancestors_children: dict[str, list[str]]):
        self.library: LibraryData                                                   =   library
        self.roots: set[str]                                                        =   roots
        self.ancestors_children: dict[str, list[str]]                               =   ancestors_children
        self.genre_counts, self.root_counts, self.tag_counts, self.unseen_siblings  =   self._pools()
        self.used_tokens                                                            =   set()

    def _pools(self) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, list[str]]]:

        all_genres, all_roots, all_tags = [], [], []
        for album in self.library:
            album_genres, album_tags = get_album_genres(album), get_album_tags(album)
            for genre in album_genres.keys():
                if genre in self.roots:
                    all_roots.append(genre)
                else:
                    all_genres.append(genre)
            all_tags.extend(album_tags)
        genre_counts, root_counts, tag_counts = dict(Counter(all_genres)), dict(Counter(all_roots)), dict(Counter(all_tags))

        ancestor_children = self.ancestors_children
        unseen_siblings = {}
        for genre in genre_counts.keys():
            children = ancestor_children.get(genre, [])
            unseen = [c for c in children if c not in genre_counts]
            if unseen:
                unseen_siblings[genre] = unseen

        return genre_counts, root_counts, tag_counts, unseen_siblings

    def _genre_tag_randomizer(self, k: int, **kwargs) -> list[tuple[str, str]]:
        g_probability = kwargs.get("g_probability", 0.6)
        r_probability = kwargs.get("r_probability", 0.25)
        
        genres, roots, tags, unseen_siblings = self.genre_counts, self.root_counts, self.tag_counts, self.unseen_siblings

        g_items, g_weights = [], []
        for genre, weight in genres.items():
            if genre not in self.used_tokens:
                g_items.append(genre)
                g_weights.append(weight)

        r_items, r_weights = [], []
        for root, weight in roots.items():
            if unseen_siblings.get(root) and root not in self.used_tokens:
                r_items.append(root)
                r_weights.append(weight)

        t_items, t_weights = [], []
        for tag, weight in tags.items():
            if tag not in self.used_tokens:
                t_items.append(tag)
                t_weights.append(weight)

        random_tokens, tokens_set = [], set()
        while len(random_tokens) < k:
            selected, r = None, random.random()
            if r < g_probability and g_items:
                selected = random.choices(g_items, weights=g_weights, k=1)[0], "genre"
            elif r < g_probability + r_probability and r_items:
                root = random.choices(r_items, weights=r_weights, k=1)[0]
                selected = random.choice(unseen_siblings.get(root, [])), "genre"
            elif t_items:
                selected = random.choices(t_items, weights=t_weights, k=1)[0], "tag"

            if selected is None:
                available_items = [(x, "genre") for x in g_items + r_items] + [(x, "tag") for x in t_items]
                if not available_items:
                    break
                selected = random.choice(available_items)

            token, _ = selected
            if token not in self.used_tokens and token not in tokens_set:
                random_tokens.append(selected)
                tokens_set.add(token)
        self.used_tokens.update(token for token, _ in random_tokens)
        return random_tokens
    
    def _random_artist_generator(self, k: int) -> list[str]:
        num_albums = len(self.library)
        artists = []
        for _ in range(k):
            irandom_artist = random.randint(0, num_albums - 1)
            random_artist = self.library[irandom_artist].get("artist", [])
            if random_artist:
                if isinstance(random_artist, list):
                    artists.extend(random_artist)
                else:
                    artists.append(random_artist)
        return [(artist, "artist") for artist in artists[:k]]
    
    def _pick_random_similar(self, similars: list[dict[str, str | float]]) -> dict[str, str | float] | None:
        if not similars:
            return None

        weights = [s.get("match", 0.0) for s in similars]
        # fallback if all matches are zero
        if sum(weights) == 0:
            return random.choice(similars)

        return random.choices(similars, weights=weights, k=1)[0].get("name")
    
    def _random_artist_genre_tag_generator(self, k: int, a_probability: float = 1.0, **kwargs) -> list[tuple[str, str]]:
        if random.random() < a_probability:
            random_artist = self._random_artist_generator(k)
            if random_artist:
                return random_artist
        else:
            return self._genre_tag_randomizer(k, **kwargs)