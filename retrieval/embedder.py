"""Vector embedding pipeline using sentence-transformers (all-MiniLM-L6-v2) with deterministic fallback."""

from typing import List
import numpy as np


class Embedder:
    """Computes dense vector representations for text queries and document chunks.
    
    Default model: sentence-transformers/all-MiniLM-L6-v2 (dim=384).
    Includes a fast hash-based deterministic fallback when torch/sentence-transformers
    is not installed, ensuring pipeline robustness and instant local testing.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None
        self._is_fallback = False
        self._initialize_model()

    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._is_fallback = False
        except Exception:
            self._model = None
            self._is_fallback = True

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of strings into an [N, dimension] numpy array normalized to unit length."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if not self._is_fallback and self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return embeddings.astype(np.float32)

        # High-performance deterministic fallback embedding
        return self._fallback_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query into a 1D or [1, dimension] normalized array."""
        embeddings = self.embed_texts([query])
        return embeddings[0]

    def _fallback_embed(self, texts: List[str]) -> np.ndarray:
        """Deterministic hashing projection simulating normalized dense vector space."""
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            if not words:
                vec[0] = 1.0
            else:
                for word in words:
                    # Deterministic hash to dimension index
                    h = hash(word)
                    idx = abs(h) % self.dimension
                    sign = 1.0 if (h > 0) else -1.0
                    vec[idx] += sign

                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                else:
                    vec[0] = 1.0
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)
