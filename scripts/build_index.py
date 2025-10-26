"""Main script to build both indices."""
from src.pipeline.ingest import ProcessedDocumentLoader
from src.pipeline.chunk import DocumentChunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
import config

def main():
    loader = ProcessedDocumentLoader()
    documents = loader.load_directory(config.PROCESSED_DATA_DIR)
    
    chunker = DocumentChunker(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = chunker.chunk_documents(documents, strategy="auto")
    
    print("Building vector store...")
    vector_store = VectorStore(config.CHROMA_PATH)
    vector_store.create_index(chunks)
    
    print("Building BM25 index...")
    bm25_store = BM25Store(config.BM25_PATH)
    bm25_store.create_index(chunks)
    
    print("Indexing complete!")

if __name__ == "__main__":
    main()
