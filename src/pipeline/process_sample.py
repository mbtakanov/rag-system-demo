"""
Process sample documents and create the database.
This is a quick start script for users who want to test the system
without generating the full 100 documents.
"""

import config
from pipeline.process_documents import process_all_documents
from pipeline.create_database import generate_data_store


def process_sample_documents():
    """Process sample documents from data/sample and create the database."""
    print("--- Processing Sample Documents ---")
    
    source_dir = config.SAMPLE_DATA_DIR
    process_all_documents(raw_dir=source_dir)
    
    print("--- Creating Vector Database and BM25 Index ---")
    try:
        generate_data_store()
    except Exception as e:
        print(f"Error creating vector database and BM25 index: {e}")
        return
    
    print("--- Sample documents processed and database created ---")


if __name__ == "__main__":
    process_sample_documents()

# python -m pipeline.process_sample
