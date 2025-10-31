from typing import Dict, List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class VectorStore:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.db = None

    def create_index(self, chunks: List[Document]) -> None:
        self.db = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.persist_directory
        )
        self.db.persist()

    def load_index(self) -> None:
        self.db = Chroma(
            persist_directory=self.persist_directory, embedding_function=self.embeddings
        )

    def search(
        self, query: str, k: int = 5, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        return self.db.similarity_search_with_score(query, k=k, filter=filter)
