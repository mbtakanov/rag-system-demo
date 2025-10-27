"""
Unified document processing and index building pipeline.
"""

import os
import logging
import argparse

from dotenv import load_dotenv

from src.pipeline.preprocess import DocumentPreprocessor
from src.pipeline.chunk import DocumentChunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.pipeline.ingest import DocumentLoader
from src.config import (
    RAW_DATA_DIR,
    SAMPLES_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_PATH,
    BM25_PATH,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build(use_samples: bool = False):
    """
    Build indexes from documents.

    Args:
        use_samples: If True, use sample documents. If False, use full dataset.
    """
    input_dir = SAMPLES_DATA_DIR if use_samples else RAW_DATA_DIR
    dataset_name = "sample" if use_samples else "full"

    logger.info(f"Processing {dataset_name} dataset from {input_dir}")

    logger.info("Step 1/4: Extracting text from documents...")
    preprocessor = DocumentPreprocessor(use_ocr=True, use_picture_description=True)
    stats = preprocessor.process_directory(input_dir, PROCESSED_DATA_DIR)

    if stats["successful"] == 0:
        logger.error(f"Error: No documents were successfully processed")
        return False

    logger.info(
        f"\nSuccessfully processed {stats['successful']}/{stats['total']} documents"
    )

    logger.info("Step 2/4: Loading processed documents...")
    loader = DocumentLoader()
    documents = loader.load_directory(PROCESSED_DATA_DIR)

    if not documents:
        logger.error("Error: No processed documents found")
        return False

    doc_stats = loader.get_statistics(documents)
    logger.info(f"Loaded {doc_stats['total_documents']} documents")
    logger.info(f"Total characters: {doc_stats['total_characters']:,}")
    logger.info(f"Document types: {doc_stats['by_type']}")

    logger.info("Step 3/4: Chunking documents...")
    chunker = DocumentChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = chunker.chunk_documents(documents, strategy="auto")

    chunk_stats = chunker.get_chunk_statistics(chunks)
    logger.info(f"Created {chunk_stats['total_chunks']} chunks")
    logger.info(f"Average chunk size: {chunk_stats['avg_chunk_length']:.0f} characters")

    logger.info("Step 4/4: Building search indexes...")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    vector_store = VectorStore(CHROMA_PATH)
    vector_store.create_index(chunks)
    logger.info("Vector index created")

    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
    bm25_store = BM25Store(BM25_PATH)
    bm25_store.create_index(chunks)
    logger.info("BM25 index created")

    logger.info(f"{dataset_name.capitalize()} dataset processing complete!")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG indexes")
    parser.add_argument(
        "--samples",
        action="store_true",
        help="Use sample documents instead of full dataset",
    )
    args = parser.parse_args()

    success = build(use_samples=args.samples)
    exit(0 if success else 1)
