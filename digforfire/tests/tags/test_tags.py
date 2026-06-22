from digforfire.tags.tags import Tags


def test__normalize_tag():
    tags = Tags()
    assert tags._normalize_tag(None) is None
    assert tags._normalize_tag("mood_kayō") in tags.canonical_genres
    assert tags._normalize_tag("ryūkōka") in tags.aliases.values()
    assert tags._normalize_tag("2023") == "2020s"
    assert tags._normalize_tag("90s") == "1990s"
