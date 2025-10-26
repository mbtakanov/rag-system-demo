"""
Document ingestion module.
Handles loading documents from various sources and formats.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class DocumentMetadata:
    source: str
    filename: str
    file_type: str
    file_size: int
    page_count: Optional[int] = None
    created_at: Optional[str] = None


class DocumentLoader:
    def __init__(self, supported_formats: tuple = ('.pdf', '.docx', '.md', '.txt')):
        self.supported_formats = supported_formats
    
    def load_file(self, file_path: str) -> Optional[Document]:
        """
        Load a single file and create a Document object.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object with content and metadata, or None if failed
        """
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return None
        
        extension = Path(file_path).suffix.lower()
        if extension not in self.supported_formats:
            print(f"[WARN] Unsupported format: {extension}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = self._extract_metadata(file_path)
            
            return Document(
                page_content=content,
                metadata=metadata
            )
        
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            return None
    
    def load_directory(self, 
                       directory: str,
                       recursive: bool = False,
                       file_pattern: Optional[str] = None) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories
            file_pattern: Optional glob pattern (e.g., "report_*.pdf")
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(directory):
            print(f"[ERROR] Directory not found: {directory}")
            return []
        
        documents = []
        
        if recursive:
            for root, _, files in os.walk(directory):
                for filename in files:
                    if self._should_process(filename, file_pattern):
                        file_path = os.path.join(root, filename)
                        doc = self.load_file(file_path)
                        if doc:
                            documents.append(doc)
        else:
            files = os.listdir(directory)
            for filename in files:
                if self._should_process(filename, file_pattern):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        doc = self.load_file(file_path)
                        if doc:
                            documents.append(doc)
        
        print(f"[INFO] Loaded {len(documents)} documents from {directory}")
        return documents
    
    def _should_process(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.supported_formats
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        stat = path.stat()
        
        return {
            'source': str(path.absolute()),
            'filename': path.name,
            'file_type': path.suffix.lower().replace('.', ''),
            'file_size': stat.st_size,
            'created_at': str(stat.st_ctime)
        }
    
    def get_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        if not documents:
            return {
                'total_documents': 0,
                'by_type': {},
                'total_size': 0
            }
        
        stats = {
            'total_documents': len(documents),
            'by_type': {},
            'total_size': 0,
            'total_characters': 0
        }
        
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            stats['by_type'][file_type] = stats['by_type'].get(file_type, 0) + 1
            
            stats['total_size'] += doc.metadata.get('file_size', 0)
            stats['total_characters'] += len(doc.page_content)
        
        return stats


class RawDocumentLoader(DocumentLoader):
    """
    Specialized loader for raw documents (PDF, DOCX).
    These need preprocessing before use.
    """
    
    def __init__(self):
        super().__init__(supported_formats=('.pdf', '.docx'))
    
    def load_for_preprocessing(self, directory: str) -> List[str]:
        if not os.path.exists(directory):
            print(f"[ERROR] Directory not found: {directory}")
            return []
        
        file_paths = []
        
        for filename in os.listdir(directory):
            extension = Path(filename).suffix.lower()
            if extension in self.supported_formats:
                file_paths.append(os.path.join(directory, filename))
        
        print(f"[INFO] Found {len(file_paths)} raw documents to process")
        return file_paths


class ProcessedDocumentLoader(DocumentLoader):
    """
    Specialized loader for processed documents (markdown, text).
    These are ready for chunking and indexing.
    """
    
    def __init__(self):
        super().__init__(supported_formats=('.md', '.txt'))
