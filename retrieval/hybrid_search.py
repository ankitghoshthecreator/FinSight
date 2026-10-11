"""Hybrid retrieval combining dense vector search and BM25 with Reciprocal Rank Fusion (RRF)."""

import math
import re
from typing import List, Dict
from ingestion.models import DocumentChunk, SearchResult
from retrieval.vector_store import VectorStore
from retrieval.embedder import Embedder


class HybridRetriever:
    """Executes dense vector search and sparse BM25 keyword search, combining scores with RRF.
    
    Critical for financial documents where exact regulatory terms (e.g. 'Regulation T',
    'Tier 1 Capital Ratio', 'ISDA Schedule') must be matched precisely alongside semantic queries.
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder, bm25_weight: float = 0.5, dense_weight: float = 0.5):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self._bm25_corpus: List[List[str]] = []
        self._bm25_model = None
        self._rebuild_bm25()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize preserving acronyms, hyphens, and numbers."""
        return re.findall(r'[a-zA-Z0-9_\-\.\$]+', text.lower())

    def _rebuild_bm25(self):
        """Indexes or re-indexes current chunks from the vector store."""
        if not self.vector_store.chunks:
            self._bm25_corpus = []
            self._bm25_model = None
            return

        self._bm25_corpus = [self._tokenize(c.content) for c in self.vector_store.chunks]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25_model = BM25Okapi(self._bm25_corpus)
        except ImportError:
            self._bm25_model = None

    def index_updated(self):
        """Notify retriever that new documents were added to the vector store."""
        self._rebuild_bm25()

    def _custom_bm25_search(self, tokenized_query: List[str], top_k: int) -> List[Dict[str, float]]:
        """Lightweight BM25 implementation if rank_bm25 is not present."""
        N = len(self._bm25_corpus)
        if N == 0:
            return []

        doc_lengths = [len(doc) for doc in self._bm25_corpus]
        avgdl = sum(doc_lengths) / N if N > 0 else 1.0
        k1 = 1.5
        b = 0.75

        # Document frequencies
        df = {}
        for token in set(tokenized_query):
            count = sum(1 for doc in self._bm25_corpus if token in doc)
            df[token] = count

        scores = []
        for doc_idx, doc in enumerate(self._bm25_corpus):
            doc_len = doc_lengths[doc_idx]
            score = 0.0
            term_counts = {}
            for t in doc:
                term_counts[t] = term_counts.get(t, 0) + 1

            for token in tokenized_query:
                if token in term_counts:
                    freq = term_counts[token]
                    idf = math.log(1 + (N - df[token] + 0.5) / (df[token] + 0.5))
                    num = freq * (k1 + 1)
                    denom = freq + k1 * (1 - b + b * (doc_len / avgdl))
                    score += idf * (num / denom)

            scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[SearchResult]:
        """Hybrid search with Reciprocal Rank Fusion (RRF) and min-max score normalization."""
        if self.vector_store.count() == 0:
            return []

        # 1. Dense retrieval
        query_emb = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_emb, top_k=top_k * 2)

        # 2. BM25 retrieval
        tokenized_query = self._tokenize(query)
        bm25_scores = []

        if self._bm25_model is not None and tokenized_query:
            raw_scores = self._bm25_model.get_scores(tokenized_query)
            ranked_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:top_k * 2]
            bm25_results = [(self.vector_store.chunks[i], float(raw_scores[i])) for i in ranked_indices]
        else:
            raw_ranked = self._custom_bm25_search(tokenized_query, top_k=top_k * 2)
            bm25_results = [(self.vector_store.chunks[i], score) for i, score in raw_ranked]

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_Score = dense_weight * (1 / (rrf_k + rank_dense)) + bm25_weight * (1 / (rrf_k + rank_bm25))
        chunk_ranks: Dict[str, Dict[str, Any]] = {}

        for rank, (chunk, score) in enumerate(dense_results, start=1):
            if chunk.chunk_id not in chunk_ranks:
                chunk_ranks[chunk.chunk_id] = {
                    "chunk": chunk,
                    "dense_rank": rank,
                    "dense_score": score,
                    "bm25_rank": None,
                    "bm25_score": 0.0,
                }
            else:
                chunk_ranks[chunk.chunk_id]["dense_rank"] = rank
                chunk_ranks[chunk.chunk_id]["dense_score"] = score

        for rank, (chunk, score) in enumerate(bm25_results, start=1):
            if chunk.chunk_id not in chunk_ranks:
                chunk_ranks[chunk.chunk_id] = {
                    "chunk": chunk,
                    "dense_rank": None,
                    "dense_score": 0.0,
                    "bm25_rank": rank,
                    "bm25_score": score,
                }
            else:
                chunk_ranks[chunk.chunk_id]["bm25_rank"] = rank
                chunk_ranks[chunk.chunk_id]["bm25_score"] = score

        # Compute fused score
        fused_list: List[SearchResult] = []
        for c_id, data in chunk_ranks.items():
            d_rank = data["dense_rank"]
            b_rank = data["bm25_rank"]

            d_rrf = (1.0 / (rrf_k + d_rank)) if d_rank is not None else 0.0
            b_rrf = (1.0 / (rrf_k + b_rank)) if b_rank is not None else 0.0

            final_score = (self.dense_weight * d_rrf) + (self.bm25_weight * b_rrf)
            # Scale to 0..1 approximately
            normalized_score = min(1.0, final_score * (rrf_k + 1) * 2.0)

            fused_list.append(
                SearchResult(
                    chunk=data["chunk"],
                    score=round(normalized_score, 4),
                    dense_score=round(data["dense_score"], 4),
                    bm25_score=round(data["bm25_score"], 4),
                )
            )

        fused_list.sort(key=lambda x: x.score, reverse=True)
        return fused_list[:top_k]
