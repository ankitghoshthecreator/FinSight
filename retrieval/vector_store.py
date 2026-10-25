"""Vector store abstraction: FAISS (local + disk-persist) and Qdrant (server mode).

Backend is selected by ``settings.vector_store_backend``:
  - ``"faiss"``  — FAISS IndexFlatIP in-memory; auto-saved to
                   ``settings.faiss_persist_dir`` on app shutdown and reloaded
                   on startup so the index survives restarts.
  - ``"qdrant"`` — Qdrant server via ``qdrant-client``; named collection
                   ``settings.qdrant_collection``; fully persistent by default.
"""

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from ingestion.models import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


# ── FAISS vector store ────────────────────────────────────────────────────────

class VectorStore:
    """FAISS-backed dense vector store with optional disk persistence.

    On first construction the index is empty.  Call ``try_load(directory)`` to
    reload a previously saved index, or wire the FastAPI lifespan hooks (see
    ``api/main.py``) to do this automatically.
    """

    def __init__(self, dimension: int = 384, persist_dir: Optional[str] = None):
        self.dimension = dimension
        self.persist_dir = persist_dir
        self.chunks: List[DocumentChunk] = []
        self._index = None
        self._vectors: Optional[np.ndarray] = None
        self._use_faiss = False
        self._init_index()

        if persist_dir:
            self.try_load(persist_dir)

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_index(self) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            self._use_faiss = True
        except ImportError:
            self._index = None
            self._use_faiss = False
            self._vectors = np.empty((0, self.dimension), dtype=np.float32)

    # ── Write ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Adds document chunks and their dense embeddings to the store."""
        if not chunks:
            return
        assert len(chunks) == len(embeddings), \
            "Number of chunks must equal number of embeddings"
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self._use_faiss and self._index is not None:
            self._index.add(embeddings)
        else:
            if self._vectors is None or len(self._vectors) == 0:
                self._vectors = embeddings
            else:
                self._vectors = np.vstack([self._vectors, embeddings])

        self.chunks.extend(chunks)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Removes all chunks belonging to *doc_id*.

        FAISS does not support selective deletion; we rebuild the index from
        the surviving chunks and their stored numpy vectors.

        Returns the number of chunks removed.
        """
        surviving_indices = [
            i for i, c in enumerate(self.chunks) if c.metadata.doc_id != doc_id
        ]
        removed = len(self.chunks) - len(surviving_indices)
        if removed == 0:
            return 0

        surviving_chunks = [self.chunks[i] for i in surviving_indices]

        # Rebuild index from surviving vectors
        if self._use_faiss and self._index is not None:
            # Re-extract vectors — FAISS IndexFlatIP stores them internally
            try:
                import faiss
                all_vecs = np.zeros(
                    (self._index.ntotal, self.dimension), dtype=np.float32
                )
                self._index.reconstruct_n(0, self._index.ntotal, all_vecs)
                surviving_vecs = all_vecs[surviving_indices]
                new_index = faiss.IndexFlatIP(self.dimension)
                if len(surviving_vecs):
                    new_index.add(surviving_vecs)
                self._index = new_index
            except Exception:
                # Fallback: switch to numpy store
                self._use_faiss = False
                self._vectors = np.empty((0, self.dimension), dtype=np.float32)
        else:
            if self._vectors is not None and len(self._vectors):
                self._vectors = self._vectors[surviving_indices]

        self.chunks = surviving_chunks
        logger.info("Deleted %d chunks for doc_id=%s", removed, doc_id)
        return removed

    def reset(self) -> None:
        """Wipes all chunks and re-initialises the index."""
        self.chunks = []
        self._init_index()
        logger.info("VectorStore reset — index cleared")

    # ── Read ──────────────────────────────────────────────────────────────

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """Returns top_k (DocumentChunk, score) tuples by descending cosine similarity."""
        if not self.chunks:
            return []

        top_k = min(top_k, len(self.chunks))
        query_vec = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )

        if self._use_faiss and self._index is not None:
            scores, indices = self._index.search(query_vec, top_k)
            scores, indices = scores[0], indices[0]
        else:
            sims = np.dot(self._vectors, query_vec.T).flatten()
            indices = np.argsort(sims)[::-1][:top_k]
            scores = sims[indices]

        return [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices, scores)
            if 0 <= idx < len(self.chunks)
        ]

    def count(self) -> int:
        return len(self.chunks)

    def get_doc_ids(self) -> List[str]:
        """Returns the list of unique doc_ids currently indexed."""
        return list({c.metadata.doc_id for c in self.chunks})

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Serialises chunks metadata and FAISS/numpy index to *directory*."""
        os.makedirs(directory, exist_ok=True)

        chunks_file = os.path.join(directory, "chunks.json")
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self.chunks], f, indent=2)

        if self._use_faiss and self._index is not None:
            import faiss
            faiss.write_index(self._index, os.path.join(directory, "faiss.index"))
        elif self._vectors is not None:
            np.save(os.path.join(directory, "vectors.npy"), self._vectors)

        logger.info(
            "VectorStore saved to %s (%d chunks)", directory, len(self.chunks)
        )

    def load(self, directory: str) -> None:
        """Loads chunks metadata and index from *directory*. Raises if missing."""
        chunks_file = os.path.join(directory, "chunks.json")
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"No saved index at {chunks_file!r}")

        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = [DocumentChunk(**item) for item in json.load(f)]

        faiss_file = os.path.join(directory, "faiss.index")
        npy_file   = os.path.join(directory, "vectors.npy")

        if os.path.exists(faiss_file) and self._use_faiss:
            import faiss
            self._index = faiss.read_index(faiss_file)
        elif os.path.exists(npy_file):
            self._vectors = np.load(npy_file)
            self._use_faiss = False

        logger.info(
            "VectorStore loaded from %s (%d chunks)", directory, len(self.chunks)
        )

    def try_load(self, directory: str) -> bool:
        """Attempts to load a saved index; returns True on success, False if none exists."""
        try:
            self.load(directory)
            return True
        except FileNotFoundError:
            logger.info(
                "No persisted index found at %s — starting with empty store", directory
            )
            return False
