"""
Document ingestion module.
Handles loading documents from various sources and formats.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    source: str
    filename: str
    file_type: str
    file_size: int
    page_count: Optional[int] = None
    created_at: Optional[str] = None


class DocumentLoader:
    def __init__(self):
        pass

    def load_directory(self, directory: str) -> List[Document]:
        """Load all documents from a directory."""
        if not os.path.exists(directory):
            logger.error(f"Directory not found: {directory}")
            return []

        documents = []

        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                doc = self._load_file(file_path)
                documents.append(doc)

        return documents

    def get_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        if not documents:
            return {"total_documents": 0, "by_type": {}, "total_size": 0}

        stats = {
            "total_documents": len(documents),
            "by_type": {},
            "total_size": 0,
            "total_characters": 0,
        }

        for doc in documents:
            file_type = doc.metadata.get("file_type", "unknown")
            stats["by_type"][file_type] = stats["by_type"].get(file_type, 0) + 1

            stats["total_size"] += doc.metadata.get("file_size", 0)
            stats["total_characters"] += len(doc.page_content)

        return stats

    def _load_file(self, file_path: str) -> Optional[Document]:
        """Load a single file and create a Document object."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        metadata = self._extract_metadata(file_path)

        return Document(page_content=content, metadata=metadata)

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        stat = path.stat()

        return {
            "filename": path.name,
            "file_type": path.suffix.lower().replace(".", ""),
            "file_size": stat.st_size,
            "created_at": str(stat.st_ctime),
        }
