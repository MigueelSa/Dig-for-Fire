import pytest

from digforfire.models.models import LibraryData
from digforfire.recommender import JsonHistoryRepository
from digforfire.recommender.recommend import Recommender


@pytest.fixture
def fake_library() -> LibraryData:
    library = [
        {
            "id": "5605441c-135d-48f7-aca0-ef9592192110",
            "title": "Lá Vem a Morte",
            "artist": ["Boogarins"],
            "artist_id": ["46713d05-4143-4704-a9c3-71b482a13944"],
            "date": "2017-06-09",
            "genres": {
                "psychedelic_rock": 0,
                "rock": 0,
                "psychedelic": 1,
            },
        }
    ]
    return library


class FakeFetcher:
    def _fetch_recommendations(self, lastfm_api_key: str) -> LibraryData:
        new_recommendations = [
            {
                "id": "7ac705a1-5fba-3a2e-a37e-f3849f201bb9",
                "title": "Permanent: Joy Division 1995",
                "artist": ["Joy Division"],
                "date": "1995-06-12",
                "genres": {
                    "electronic": 0,
                    "new_wave": 0,
                    "post_punk": 0,
                    "rock": 0,
                    "punk": 1,
                    "wave": 1,
                },
                "tags": ["alternative_and_punk", "alt_punk"],
                "source": "MusicBrainz",
                "score": 0.8599435882701352,
            }
        ]

        return new_recommendations


@pytest.fixture
def fake_recommendations() -> LibraryData:
    recommendation_history = [
        {
            "id": "d6a52dc7-f4fa-4d10-89d9-5a99d178dae9",
            "title": "McCartney",
            "artist": ["Paul McCartney"],
            "date": "1970-04-17",
            "genres": {
                "blues_rock": 0,
                "classic_rock": 0,
                "experimental": 0,
                "folk_pop": 0,
                "folk_rock": 0,
                "pop": 0,
                "pop_rock": 0,
                "rock": 0,
                "singer_songwriter": 0,
                "soft_rock": 0,
                "folk": 1,
                "blues": 1,
            },
            "tags": [
                "contemporary_pop_rock",
                "pop_male_singer_songwriter",
                "late_60's_early_70's",
                "beatles",
                "1970s",
                "male_vocalists",
                "club_house",
            ],
            "source": "MusicBrainz",
            "score": 0.9240371188792884,
        },
        {
            "id": "e6a8726d-3a60-4de4-b60c-7f7b9c90b059",
            "title": "Crazy Horse",
            "artist": ["Phil Woods", "Chris Swansen"],
            "date": "1980",
            "genres": {"jazz": 0},
            "tags": ["fusion"],
            "source": "MusicBrainz",
            "score": 0.7265643444185437,
        },
    ]

    return recommendation_history


def test_saving_recommendations_adds_recommendations_to_history(
    tmp_path,
    fake_library: LibraryData,
    fake_recommendations: LibraryData,
):
    fake_history_path = tmp_path / "fake-history.json"
    fake_history_repository: JsonHistoryRepository = JsonHistoryRepository(
        path=fake_history_path
    )
    fake_fetcher = FakeFetcher()
    recommendation_history = fake_recommendations
    fake_history_repository.add_recommendations(recommendation_history)
    rec = Recommender(
        fake_library,
        "dig4fire-mail@proton.me",
        "fake_lastfm_api_key",
        k=1,
        threshold=0.0,
        history_repository=fake_history_repository,
        fetcher=fake_fetcher,
    )
    rec.recommend()
    updated_recommendation_history = fake_history_repository.load_recommendations()
    assert (
        updated_recommendation_history
        == recommendation_history
        + fake_fetcher._fetch_recommendations("fake_lastfm_api_key")
    )
