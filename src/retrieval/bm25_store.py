"""BM25 keyword search index."""
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Tuple

from langchain_core.documents import Document


class BM25Store:
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self.bm25 = None
        self.documents = None
    
    def create_index(self, chunks: List[Document]) -> None:
        self.documents = chunks
        tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        with open(self.persist_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)
    
    def load_index(self) -> None:
        with open(self.persist_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.documents[i], scores[i]) for i in top_indices]

