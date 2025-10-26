import os
import shutil
import pickle
from typing import List

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from config import CHROMA_PATH, PROCESSED_DATA_DIR, BM25_DIR, BM25_PATH, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    save_bm25_index(chunks)


def load_documents() -> List[Document]:
    loader = DirectoryLoader(
        PROCESSED_DATA_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True,
        use_multithreading=True
    )
    return loader.load()


def get_document_type(source_path: str) -> str:
    if "pdf" in source_path.lower():
        return "pdf"
    elif "docx" in source_path.lower():
        return "docx"
    else:
        return "unknown"


def split_text(documents: List[Document]) -> List[Document]:
    chunks = []
    
    print("Splitting documents..." + str(len(documents)) + " documents found.")
    for document in documents:
        source_path = document.metadata.get('source', '')
        doc_type = get_document_type(source_path)
        
        if doc_type == "pdf":
            print("Splitting PDF document...")
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large"
            )
            chunker = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85
            )
            doc_chunks = chunker.split_documents([document])
            chunks.extend(doc_chunks)
            
        elif doc_type == "docx":
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            )
            
            header_chunks = header_splitter.split_text(document.page_content)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
                separators=["\n\n", "\n", ". ", " "]
            )
            
            for header_chunk in header_chunks:
                if len(header_chunk.page_content) > CHUNK_SIZE:
                    chunks.extend(text_splitter.split_documents([header_chunk]))
                else:
                    chunks.append(header_chunk)
            print("Split docx document into chunks.")
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )
            chunks.extend(text_splitter.split_documents([document]))
            
    return chunks


def save_to_chroma(chunks: List[Document]) -> None:
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    os.makedirs(CHROMA_PATH, exist_ok=True)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def save_bm25_index(chunks: List[Document]) -> None:
    os.makedirs(BM25_DIR, exist_ok=True)
    
    tokenized_corpus = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_data = {
        "bm25": bm25,
        "documents": chunks
    }
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_data, f)
    print(f"BM25 index built and saved to {BM25_PATH}.")


if __name__ == "__main__":
    main()
