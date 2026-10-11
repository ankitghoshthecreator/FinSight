"""Vector store abstraction supporting FAISS IndexFlatIP with in-memory persistence."""

import os
import json
from typing import List, Tuple, Optional
import numpy as np
from ingestion.models import DocumentChunk, DocumentMetadata


class VectorStore:
    """Vector store index for dense retrieval using FAISS or an in-memory numpy index."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.chunks: List[DocumentChunk] = []
        self._index = None
        self._vectors: Optional[np.ndarray] = None
        self._use_faiss = False
        self._init_index()

    def _init_index(self):
        try:
            import faiss
            # Inner product for normalized embeddings is equivalent to Cosine Similarity
            self._index = faiss.IndexFlatIP(self.dimension)
            self._use_faiss = True
        except ImportError:
            self._index = None
            self._use_faiss = False
            self._vectors = np.empty((0, self.dimension), dtype=np.float32)

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Adds document chunks and their dense embeddings to the store."""
        if len(chunks) == 0:
            return

        assert len(chunks) == len(embeddings), "Number of chunks must equal number of embeddings"
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self._use_faiss and self._index is not None:
            self._index.add(embeddings)
        else:
            if self._vectors is None or len(self._vectors) == 0:
                self._vectors = embeddings
            else:
                self._vectors = np.vstack([self._vectors, embeddings])

        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Returns top_k (DocumentChunk, score) tuples ranked by descending cosine similarity."""
        if len(self.chunks) == 0:
            return []

        top_k = min(top_k, len(self.chunks))
        query_vec = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)

        if self._use_faiss and self._index is not None:
            scores, indices = self._index.search(query_vec, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Cosine similarity via inner product on normalized vectors
            sims = np.dot(self._vectors, query_vec.T).flatten()
            indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[indices]

        results = []
        for idx, score in zip(indices, scores):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def count(self) -> int:
        return len(self.chunks)

    def save(self, directory: str) -> None:
        """Serializes chunks metadata and index to disk."""
        os.makedirs(directory, exist_ok=True)
        # Save chunks
        chunks_file = os.path.join(directory, "chunks.json")
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        if self._use_faiss and self._index is not None:
            import faiss
            index_file = os.path.join(directory, "faiss.index")
            faiss.write_index(self._index, index_file)
        elif self._vectors is not None:
            np_file = os.path.join(directory, "vectors.npy")
            np.save(np_file, self._vectors)

    def load(self, directory: str) -> None:
        """Loads chunks metadata and vector index from disk."""
        chunks_file = os.path.join(directory, "chunks.json")
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Missing {chunks_file}")

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        self.chunks = [DocumentChunk(**item) for item in chunks_data]

        faiss_file = os.path.join(directory, "faiss.index")
        npy_file = os.path.join(directory, "vectors.npy")

        if os.path.exists(faiss_file) and self._use_faiss:
            import faiss
            self._index = faiss.read_index(faiss_file)
        elif os.path.exists(npy_file):
            self._vectors = np.load(npy_file)
            self._use_faiss = False
