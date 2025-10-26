import os
import json
import pickle
import numpy as np
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Iterator
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from config import RAG_PROMPT, CHROMA_PATH, BM25_PATH

load_dotenv()

MODEL = os.getenv("MODEL_NAME", "qwen3:4b")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.chunks = []
    app.vectorizer = None
    app.chunk_vectors = None
    app.llm = init_chat_model(MODEL, model_provider=MODEL_PROVIDER, temperature=0.0)
    
    app.db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings()
    )
    
    with open(BM25_PATH, 'rb') as f:
        bm25_data = pickle.load(f)
        app.bm25 = bm25_data['bm25']
        app.bm25_docs = bm25_data['documents']
    
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


def retrieve(query: str, k: int = 5, alpha: float = 0.5) -> list[str]:
    """Hybrid search: vector + BM25"""
    
    # Vector search
    vector_results = app.db.similarity_search_with_score(query, k=k)
    vector_docs = {doc.page_content: (1 - score) for doc, score in vector_results}
    
    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = app.bm25.get_scores(tokenized_query)
    top_bm25_idx = np.argsort(bm25_scores)[-k:][::-1]
    
    bm25_docs = {}
    for idx in top_bm25_idx:
        doc = app.bm25_docs[idx]
        bm25_docs[doc.page_content] = bm25_scores[idx]
    
    # Normalize and combine scores
    max_bm25 = max(bm25_docs.values()) if bm25_docs else 1
    
    combined = {}
    all_content = set(vector_docs.keys()) | set(bm25_docs.keys())
    
    for content in all_content:
        vec_score = vector_docs.get(content, 0)
        bm25_score = bm25_docs.get(content, 0) / max_bm25
        combined[content] = alpha * vec_score + (1 - alpha) * bm25_score
    
    # Top-k by combined score
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [content for content, _ in sorted_results[:k]]


def generate_answer(query: str) -> Iterator[dict]:
    """Keep as-is, it already uses retrieve()"""
    context_chunks = retrieve(query)
    
    context = "\n".join(f"<context>{c}</context>" for c in context_chunks)
    prompt = RAG_PROMPT.format(context=context, query=query)
    
    for chunk in app.llm.stream(prompt):
        yield {"type": "content", "data": chunk.content}
        

def generate_answer(query: str) -> Iterator[dict]:
    context_chunks = retrieve(query)

    context = "\n".join(f"<context>{c}</context>" for c in context_chunks)
    prompt = RAG_PROMPT.format(context=context, query=query)

    for chunk in app.llm.stream(prompt):
        yield {"type": "content", "data": chunk.content}


@app.get("/ask", description="Ask a question")
async def ask(query: str = Query(...)):
    def response_stream():
        for json_chunk in generate_answer(query):
            yield f"{json.dumps(json_chunk)}\n"
        yield f"{json.dumps({'type': 'done'})}\n"

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/")
async def root():
    return {
        "message": "RAG API",
        "endpoints": {
            "ask": "GET /ask?query=... - Ask a questions",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
