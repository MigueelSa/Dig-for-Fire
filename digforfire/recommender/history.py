from abc import ABC, abstractmethod
import json, os
from digforfire.models.models import LibraryData
from digforfire.utils.paths import output_path


class HistoryRepository(ABC):
    """
    @classmethod
    def load_recommendations(cls, path=output_path("data", "recommendation-history-Dig-for-Fire.json")) -> LibraryData:
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as file:
                json.dump([], file, ensure_ascii=False, indent=2)

        with open(path, "r", encoding='utf-8') as file:
            recommendations = json.load(file)
        return recommendations
    """

    @abstractmethod
    def load_recommendations(self) -> LibraryData:
        pass

    @abstractmethod
    def save_recommendations(self) -> None:
        pass


class JsonHistoryRepository(HistoryRepository):

    def __init__(
        self, path=output_path("data", "recommendation-history-Dig-for-Fire.json")
    ):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as file:
                json.dump([], file, ensure_ascii=False, indent=2)

    def load_recommendations(self) -> LibraryData:
        with open(self.path, "r", encoding="utf-8") as file:
            recommendations = json.load(file)
        return recommendations

    def save_recommendations(self, recommendations: LibraryData) -> None:
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(recommendations, file, ensure_ascii=False, indent=4)
