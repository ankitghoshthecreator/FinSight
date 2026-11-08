"""ColBERT late-interaction reranking for financial documents.

When enabled, this uses the official ``colbert-ai`` checkpoint implementation.
The deterministic token encoder remains available as a lightweight fallback for
offline development and installations without the optional model dependencies.
"""

import hashlib
import logging
import os
import re
from typing import List, Optional

import numpy as np

from config.settings import settings
from ingestion.models import SearchResult

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """Rerank a hybrid-search candidate set with ColBERT MaxSim."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_dim: int = 128,
        use_fallback: Optional[bool] = None,
    ):
        self.model_name = model_name or settings.colbert_model
        self.embedding_dim = embedding_dim
        self._checkpoint = None
        self._is_fallback = True
        enabled = settings.colbert_enabled if use_fallback is None else not use_fallback
        if enabled:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Load the official ColBERT checkpoint (downloads if not cached)."""
        try:
            from colbert.infra import ColBERTConfig, Run, RunConfig
            from colbert.modeling.checkpoint import Checkpoint

            # Checkpoint provides query/document token embeddings and the same
            # MaxSim scoring used by ColBERT search, without building a second index.
            with Run().context(RunConfig(nranks=1, experiment="finsight-reranker")):
                self._checkpoint = Checkpoint(
                    self.model_name,
                    colbert_config=ColBERTConfig(root=os.environ.get("COLBERT_ROOT", ".colbert")),
                )
            self._is_fallback = False
        except Exception as exc:
            logger.warning("Could not load ColBERT model %s; using local fallback: %s", self.model_name, exc)
            self._checkpoint = None
            self._is_fallback = True

    @property
    def is_fallback(self) -> bool:
        return self._is_fallback

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9_\-.$%]+", text.lower())
        return tokens if tokens else [text.lower().strip()]

    def _token_embedding(self, token: str) -> np.ndarray:
        """Stable pseudo-embedding used only by the offline fallback."""
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        for i in range(4):
            idx = int.from_bytes(digest[i * 4 : i * 4 + 4], "little") % self.embedding_dim
            vec[idx] += 1.0 if digest[12 + i] % 2 == 0 else -1.0
        for k in range(max(0, len(token) - 2)):
            gram = hashlib.blake2b(token[k : k + 3].encode("utf-8"), digest_size=4).digest()
            vec[int.from_bytes(gram, "little") % self.embedding_dim] += 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def _encode_sequence(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
        return np.stack([self._token_embedding(token) for token in tokens])

    def compute_maxsim(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Compute mean token-level maximum cosine similarity (fallback only)."""
        if not query_tokens or not doc_tokens:
            return 0.0
        similarities = self._encode_sequence(query_tokens) @ self._encode_sequence(doc_tokens).T
        return float(np.clip(np.max(similarities, axis=1).mean(), 0.0, 1.0))

    def score(self, query: str, document_text: str) -> float:
        if self._checkpoint is not None:
            return self._model_scores(query, [document_text])[0]
        return self.compute_maxsim(self._tokenize(query), self._tokenize(document_text))

    def _model_scores(self, query: str, documents: List[str]) -> List[float]:
        """Return official ColBERT MaxSim scores for a query and documents."""
        from colbert.modeling.colbert import colbert_score

        q = self._checkpoint.queryFromText([query], bsize=1)
        d, d_mask = self._checkpoint.docFromText(
            documents, bsize=len(documents), keep_dims="return_mask"
        )
        # The upstream helper applies ColBERT's document-token mask and supports
        # broadcasting one query embedding over the whole candidate batch.
        scores = colbert_score(q, d, d_mask, config=self._checkpoint.colbert_config)
        return [float(value) for value in scores.detach().cpu().reshape(-1).tolist()]

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: Optional[int] = None,
        fusion_weight: float = 0.65,
    ) -> List[SearchResult]:
        if not candidates:
            return []
        fusion_weight = min(1.0, max(0.0, float(fusion_weight)))
        if self._checkpoint is not None:
            raw_scores = self._model_scores(query, [item.chunk.content for item in candidates])
            low, high = min(raw_scores), max(raw_scores)
            late_scores = [((score - low) / (high - low)) if high > low else 0.5 for score in raw_scores]
        else:
            late_scores = [self.compute_maxsim(self._tokenize(query), self._tokenize(item.chunk.content)) for item in candidates]

        reranked = []
        for candidate, late_score in zip(candidates, late_scores):
            fused_score = fusion_weight * late_score + (1.0 - fusion_weight) * candidate.score
            reranked.append(SearchResult(
                chunk=candidate.chunk,
                score=round(float(fused_score), 4),
                dense_score=candidate.dense_score,
                bm25_score=candidate.bm25_score,
            ))
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k] if top_k is not None else reranked
