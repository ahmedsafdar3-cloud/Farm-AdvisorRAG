"""
Embedding backend for the RAG pipeline.

Primary path: sentence-transformers (all-MiniLM-L6-v2) — real dense embeddings,
what you'd use in production and what interviewers expect to hear about.

Fallback path: TF-IDF (scikit-learn) — used automatically if the transformer
model can't be downloaded (e.g. no internet / offline environment). This keeps
the pipeline runnable everywhere and is worth mentioning in an interview as a
deliberate graceful-degradation design choice, not a limitation you missed.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


class DenseEmbedder:
    """Wraps sentence-transformers for real dense embeddings."""

    def __init__(self, model_name: str = _MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.backend = "sentence-transformers"

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


class TfidfEmbedder:
    """Sparse fallback embedder using TF-IDF, sklearn-only, no downloads required."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=4096)
        self._fitted = False
        self.backend = "tfidf-fallback"

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts):
        if not self._fitted:
            # Fit-on-first-use so single query embedding still works after ingest.
            self.fit(texts)
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


def get_embedder():
    """
    Try to load the real sentence-transformer model. If that fails for any
    reason (no internet, model not cached, etc.), fall back to TF-IDF so the
    rest of the pipeline still runs end-to-end.
    """
    try:
        embedder = DenseEmbedder()
        logger.info("Using sentence-transformers embedder (%s)", _MODEL_NAME)
        return embedder
    except Exception as e:
        logger.warning(
            "Could not load sentence-transformers model (%s). "
            "Falling back to TF-IDF embeddings. To use real dense embeddings, "
            "run this with internet access so the model can download once.",
            e,
        )
        return TfidfEmbedder()
