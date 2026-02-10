import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from models.models import AlbumData, LibraryData, EmbeddingData
from utils.paths import output_path

class Predictor:
    def __init__(self, history: LibraryData, library: LibraryData, genre_embeddings: EmbeddingData, tag_embeddings: EmbeddingData, stratify: bool = True, n_estimators: int = 300, balanced: bool = False):
        self.history                                =   history
        self.library                                =   library
        self.genre_embeddings, self.tag_embeddings  =   genre_embeddings, tag_embeddings
        self.model                                  =   self._train_model(stratify=stratify, n_estimators=n_estimators, balanced=balanced)
        self.alpha                                  =   self._get_alpha()
        self.is_safe                                =   self._safety_check()

    def _album_label(self, album: AlbumData) -> int:
        album_id = album["id"]
        for lib_album in self.library:
            if lib_album["id"] == album_id:
                return 1
        return 0
    
    def _build_dataset(self) -> tuple[EmbeddingData, list[int]]:
        embeddings, labels = {},[]
        merged = {album["id"]: album for album in self.history + self.library}
        for album in merged.values():
            genre_embeddings, tag_embeddings = self.genre_embeddings.get_album_embeddings(album), self.tag_embeddings.get_album_embeddings(album)
            if genre_embeddings is not None and tag_embeddings is not None:
                embeddings[album["id"]] = np.concatenate([genre_embeddings, tag_embeddings])
                labels.append(self._album_label(album))
        return embeddings, labels
    
    def _split_dataset(self, dataset: EmbeddingData, labels: list[int], test_size: float = 0.2, random_state: int = 42, stratify: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, y = np.vstack(list(dataset.values())), np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None)

        return X_train, X_test, y_train, y_test
    
    def _train_split(self, X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 300, balanced: bool = True) -> RandomForestClassifier:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1, class_weight='balanced' if balanced else None)
        model.fit(X_train, y_train)
        return model
    
    def _train_model(self, stratify: bool = True, n_estimators: int = 300, balanced: bool = True) -> RandomForestClassifier:
        dataset, labels = self._build_dataset()
        X_train, X_test, y_train, y_test = self._split_dataset(dataset, labels, stratify=stratify)
        model = self._train_split(X_train, y_train, n_estimators=n_estimators, balanced=balanced)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        return model
    
    def _get_alpha(self) -> float:
        preds = self.model.predict(self.X_test)
        report = classification_report(self.y_test, preds, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]
        alpha = macro_f1 / 2
        return alpha
    
    def _safety_check(self, min_samples: int = 50, min_ratio: float = 0.2) -> bool:
        _, labels = self._build_dataset()
        if not labels:
            return False
        num_positive = sum(labels)
        num_negative = len(labels) - num_positive
        smaller_class = min(num_positive, num_negative)
        return smaller_class >= min_samples and smaller_class / len(labels) >= min_ratio
    
if __name__ == "__main__":
    from libraries.libraries import MusicBrainz
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import classification_report

    from embeddings.genre_space import GenreSpace
    from embeddings.tag_space import TagSpace

    history_path = output_path("data/recommendation-history-Dig-for-Fire.json")
    library_path = output_path("data/MusicBrainz-Dig-for-Fire.json")

    mb = MusicBrainz("Dig-for-Fire", "0.1", "miguel@dig-for-fire.com")
    history, library = mb._load_library(history_path), mb._load_library(library_path)

    balanced = False
    if balanced:
        print("Using balanced dataset.")
    else:
        print("Using imbalanced dataset.")

    genre_embeddings, tag_embeddings = GenreSpace(library, method="pmi"), TagSpace(library, n_clusters=50)

    predictor = Predictor(history, library, genre_embeddings, tag_embeddings, balanced=balanced)
    X_train, X_test, y_train, y_test = predictor.X_train, predictor.X_test, predictor.y_train, predictor.y_test

    preds = predictor.model.predict(X_test)
    probs = predictor.model.predict_proba(X_test)[:,1]

    print("Accuracy:", accuracy_score(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))

    scores = predictor.model.predict_proba(X_test)[:, 1]
    ranking = np.argsort(scores)[::-1]

    top10 = list(zip(ranking[:10], y_test[ranking][:10]))
    print("Top 10 predictions (index, true label):", top10)

    report = classification_report(y_test, preds, digits=3, zero_division=0)
    print("Classification Report:\n", report)

