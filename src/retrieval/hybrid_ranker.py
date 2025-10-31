import logging
from typing import List, Tuple, Optional, Dict

from langchain_core.documents import Document

from src.config import CHROMA_PATH, BM25_PATH
from src.retrieval.bm25_store import BM25Store
from src.retrieval.vector_store import VectorStore
from src.retrieval.query_expander import QueryExpander


logger = logging.getLogger(__name__)


class HybridRanker:
    def __init__(self):
        self.vector_store = VectorStore(CHROMA_PATH)
        self.bm25_store = BM25Store(BM25_PATH)
        self.vector_store.load_index()
        self.bm25_store.load_index()
        self.query_expander = QueryExpander()

    def search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        metadata_filter: Optional[Dict[str, str]] = None,
        use_query_expansion: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search combining vector and BM25 with normalized scores.

        Args:
            query: Search query
            k: Number of results
            alpha: Weight for vector search (0-1), BM25 gets (1-alpha)
            metadata_filter: Optional metadata filter
            use_query_expansion: Whether to expand the query

        Returns:
            List of (content, combined_score) tuples
        """
        queries = [query]

        if use_query_expansion and self.query_expander:
            queries = self.query_expander.expand_multi_strategy(query)
            logger.info(f"[Query Expansion] Generated {len(queries)} queries")

        all_results = {}

        for q in queries:
            vector_results = self.vector_store.search(q, k=k)
            bm25_results = self.bm25_store.search(q, k=k)

            vector_scores = self._normalize_scores(
                [score for _, score in vector_results]
            )
            bm25_scores = self._normalize_scores([score for _, score in bm25_results])

            for (doc, _), norm_score in zip(vector_results, vector_scores):
                if self._matches_filter(doc, metadata_filter):
                    content = doc.page_content
                    if content not in all_results:
                        all_results[content] = 0.0
                    all_results[content] += alpha * norm_score / len(queries)

            for (doc, _), norm_score in zip(bm25_results, bm25_scores):
                if self._matches_filter(doc, metadata_filter):
                    content = doc.page_content
                    if content not in all_results:
                        all_results[content] = 0.0
                    all_results[content] += (1 - alpha) * norm_score / len(queries)

        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalization."""
        if not scores or max(scores) == min(scores):
            return [1.0] * len(scores)

        min_score = min(scores)
        max_score = max(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _matches_filter(
        self, doc: Document, metadata_filter: Optional[Dict[str, str]]
    ) -> bool:
        """Check if document matches metadata filter."""
        if not metadata_filter:
            return True

        for key, value in metadata_filter.items():
            if doc.metadata.get(key) != value:
                return False

        return True
