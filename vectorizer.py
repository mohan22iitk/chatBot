from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from typing import List, Tuple

def train_vectorizer(texts: List[str], max_features: int = 1000) -> Tuple[TfidfVectorizer, any]:

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_vectors = vectorizer.fit_transform(texts)
    return vectorizer, X_vectors


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = "vectorizer.pkl") -> None:

    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(path: str = "vectorizer.pkl") -> TfidfVectorizer:

    with open(path, "rb") as f:
        return pickle.load(f)
