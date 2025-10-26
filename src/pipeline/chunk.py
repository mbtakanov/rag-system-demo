"""
Document chunking module.
Implements multiple chunking strategies for different document types.
"""


from typing import List, Literal
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


class DocumentChunker:
    def __init__(self, 
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 embedding_model: str = EMBEDDING_MODEL):
        """
        Initialize the DocumentChunker with configuration parameters.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            embedding_model: Name of the OpenAI embedding model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
    
    def chunk_documents(self, 
                       documents: List[Document],
                       strategy: Literal["auto", "semantic", "markdown", "recursive"] = "auto"
                       ) -> List[Document]:
        """
        Chunk a list of documents using the specified strategy.
        
        Args:
            documents: List of documents to chunk
            strategy: Chunking strategy to use
                - "auto": Automatically select strategy based on document type
                - "semantic": Use semantic similarity to split documents
                - "markdown": Split based on markdown headers
                - "recursive": Recursively split text by separators
                
        Returns:
            List of chunked documents
        """
        all_chunks = []
        
        for document in documents:
            if strategy == "auto":
                doc_type = self._get_document_type(document)
                chunks = self._chunk_by_type(document, doc_type)
            elif strategy == "semantic":
                chunks = self._semantic_chunk(document)
            elif strategy == "markdown":
                chunks = self._markdown_chunk(document)
            elif strategy == "recursive":
                chunks = self._recursive_chunk(document)
            else:
                raise ValueError(f"Unknown chunking strategy: {strategy}")
            
            all_chunks.extend(chunks)
        
        print(f"[INFO] Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _get_document_type(self, document: Document) -> str:
        """
        Determine the document type based on metadata.
        
        Args:
            document: Document to analyze
            
        Returns:
            Document type: 'pdf', 'docx', 'markdown', or 'txt'
        """
        source = document.metadata.get('source', '').lower()
        file_type = document.metadata.get('file_type', '').lower()
        
        if 'pdf' in source or file_type == 'pdf':
            return 'pdf'
        elif 'docx' in source or file_type == 'docx':
            return 'docx'
        elif file_type in ['md', 'markdown']:
            return 'markdown'
        else:
            return 'txt'
    
    def _chunk_by_type(self, document: Document, doc_type: str) -> List[Document]:
        """
        Chunk a document using a strategy based on its type.
        
        For testing purposes, we're using different chunking strategies for different doc types.
        It would probably be a good idea to add LLM chunking as well.
        TODO: The current chunking strategy is not very good. Improve it.
        
        Args:
            document: Document to chunk
            doc_type: Type of document ('pdf', 'docx', 'markdown', or 'txt')
            
        Returns:
            List of chunked documents
        """
        if doc_type == 'pdf':
            return self._semantic_chunk(document)
        elif doc_type in ['docx', 'markdown']:
            return self._markdown_chunk(document)
        else:
            return self._recursive_chunk(document)
    
    def _semantic_chunk(self, document: Document) -> List[Document]:
        """
        Chunk a document using semantic similarity-based splitting.
        
        Uses embeddings to identify semantic boundaries and split at natural
        content breakpoints rather than fixed character counts.
        
        Args:
            document: Document to chunk semantically
            
        Returns:
            List of semantically-chunked documents
        """
        chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85
        )
        
        chunks = chunker.split_documents([document])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_strategy'] = 'semantic'
        
        return chunks
    
    def _markdown_chunk(self, document: Document) -> List[Document]:
        """
        Chunk a markdown document by headers, then by size.
        
        First splits by markdown headers (H1, H2, H3), then further splits
        large sections using recursive character splitting to fit within chunk_size.
        
        Args:
            document: Markdown document to chunk
            
        Returns:
            List of markdown-chunked documents
        """
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        
        header_chunks = header_splitter.split_text(document.page_content)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        final_chunks = []
        for i, header_chunk in enumerate(header_chunks):
            header_chunk.metadata.update(document.metadata)
            
            if len(header_chunk.page_content) > self.chunk_size:
                sub_chunks = text_splitter.split_documents([header_chunk])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(header_chunk)
        
        for i, chunk in enumerate(final_chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_strategy'] = 'markdown'
        
        return final_chunks
    
    def _recursive_chunk(self, document: Document) -> List[Document]:
        """
        Chunk a document using recursive character text splitting.
        
        Splits text hierarchically by trying separators in order: paragraphs,
        sentences, words, and finally characters. This approach preserves
        as much semantic structure as possible while fitting within chunk_size.
        
        Args:
            document: Document to chunk recursively
            
        Returns:
            List of recursively-chunked documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents([document])
        
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
            chunk.metadata['chunk_strategy'] = 'recursive'
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        Calculate and return statistics about the chunked documents.
        
        Args:
            chunks: List of chunked documents
            
        Returns:
            Dictionary containing statistics:
                - total_chunks: Total number of chunks
                - avg_chunk_length: Average character length of chunks
                - min_chunk_length: Minimum chunk length
                - max_chunk_length: Maximum chunk length
                - strategies_used: Count of chunks by strategy type
        """
        if not chunks:
            return {'total_chunks': 0}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        strategies = {}
        
        for chunk in chunks:
            strategy = chunk.metadata.get('chunk_strategy', 'unknown')
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'strategies_used': strategies
        }


# TODO: Add LLM-based contextualization
class ContextualChunker(DocumentChunker):
    """
    Advanced chunker that adds context to each chunk.
    Uses LLM to generate contextual information.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the ContextualChunker.
        
        Extends DocumentChunker with LLM-based contextualization capabilities.
        """
        super().__init__(*args, **kwargs)
        # self.llm = ChatOpenAI(model=config.MODEL)
    
    def add_context(self, chunks: List[Document], full_document: str) -> List[Document]:
        """
        Add contextual information to each chunk.
        
        Args:
            chunks: List of chunks
            full_document: The full document text for context
            
        Returns:
            Chunks with added context in metadata
        """
        # TODO: Use CONTEXTUALIZER_PROMPT from config.py
        
        for chunk in chunks:
            chunk.metadata['document_summary'] = self._generate_summary(full_document)
        
        return chunks
    
    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate brief summary of text."""
        # TODO: Simple truncation for now - replace with LLM call
        return text[:max_length] + "..." if len(text) > max_length else text
