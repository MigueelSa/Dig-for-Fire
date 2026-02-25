import json, os
from digforfire.models.models import LibraryData
from digforfire.utils.paths import output_path

class HistoryManager:

    @classmethod
    def load_recommendations(cls, path=output_path("data", "recommendation-history-Dig-for-Fire.json")) -> LibraryData:
        if not os.path.exists(path):
            with open(path, "w", encoding='utf-8') as file:
                json.dump([], file, ensure_ascii=False, indent=2)
                
        with open(path, "r", encoding='utf-8') as file:
            recommendations = json.load(file)
        return recommendations