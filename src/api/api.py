import os
import sys
import json
import random
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Iterator, Optional

from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, Query
from langchain.chat_models import init_chat_model

from src.retrieval.hybrid_ranker import HybridRanker
from src.config import MODEL_NAME, MODEL_PROVIDER, RAG_PROMPT
from src.api.models import (
    HealthResponse,
    SearchResponse,
    SearchResultItem,
    MetadataResponse,
    RootResponse,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL = os.getenv("MODEL_NAME", MODEL_NAME)
PROVIDER = os.getenv("MODEL_PROVIDER", MODEL_PROVIDER)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        app.llm = init_chat_model(
            MODEL,
            model_provider=PROVIDER,
            temperature=0.0,
        )
        app.ranker = HybridRanker()
    except Exception as e:
        logger.exception(f"Failed to initialize resources during startup: {e}")
        raise

    yield

    logger.info("Shutting down application resources...")


app = FastAPI(title="RAG API", lifespan=lifespan)


def generate_answer(
    query: str, k: int = 5, alpha: float = 0.5, expand_query: bool = False
) -> Iterator[dict]:
    try:
        results = app.ranker.search(
            query,
            k=k,
            alpha=alpha,
            metadata_filter=None,
            use_query_expansion=expand_query,
        )

        if not results:
            yield {"type": "error", "data": "No relevant context found"}
            return

        scores_data = [
            {
                "chunk_index": i,
                "score": round(float(score), 2),
                "content_preview": content[:100] + "...",
            }
            for i, (content, score) in enumerate(results)
        ]
        yield {"type": "metadata", "data": {"retrieval_scores": scores_data}}

        context = "\n".join(
            f"<context>{content}</context>" for content, _score in results
        )
        prompt = RAG_PROMPT.format(context=context, query=query)

        for chunk in app.llm.stream(prompt):
            yield {"type": "content", "data": chunk.content}

    except Exception as e:
        logger.exception(f"Error in generate_answer: {e}")
        yield {"type": "error", "data": str(e)}


@app.get("/health", description="Health check", response_model=HealthResponse)
async def health():
    try:
        if not hasattr(app, "llm") or not hasattr(app, "ranker"):
            return HealthResponse(status="unhealthy", error="Resources not loaded")

        return HealthResponse(
            status="healthy",
            model=MODEL,
            provider=MODEL_PROVIDER,
            vector_db="chroma",
            documents_indexed=app.ranker.vector_store.db._collection.count(),
        )
    except Exception as e:
        return HealthResponse(status="unhealthy", error=str(e))


@app.get(
    "/ask",
    summary="Ask a question",
    description="""
    Ask a question and get a streaming answer with source attribution.
    
    The response streams newline-delimited JSON chunks:
    - {"type": "metadata", "data": {"retrieval_scores": [...]}} - retrieval scores
    - {"type": "content", "data": "..."} - answer text chunks
    - {"type": "done"} - completion signal
    - {"type": "error", "data": "..."} - error message
    """,
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Streaming JSON response",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": '{"type":"metadata","data":{"retrieval_scores":[...]}}\n{"type":"content","data":"Answer..."}\n{"type":"done"}',
                    }
                }
            },
        }
    },
)
async def ask(
    query: str = Query(..., min_length=1, max_length=500, description="Your question"),
    k: int = Query(5, ge=1, le=10, description="Number of context chunks"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Vector search weight (0-1)"),
    expand_query: bool = Query(False, description="Enable query expansion"),
):
    """
    Ask a question and get a streaming answer.

    - **query**: Your question
    - **k**: Number of context chunks to retrieve (1-10)
    - **alpha**: Weight for vector search vs BM25 (0=only BM25, 1=only vector)
    - **expand_query**: Enable query expansion

    The response streams in JSON format:
    - `{"type": "metadata", "data": {...}}` - retrieval scores
    - `{"type": "content", "data": "..."}` - answer chunks
    - `{"type": "done"}` - completion signal
    """
    if not query.strip():
        raise HTTPException(400, "Query cannot be empty")

    def response_stream():
        try:
            for json_chunk in generate_answer(
                query, k=k, alpha=alpha, expand_query=expand_query
            ):
                yield f"{json.dumps(json_chunk)}\n\n"
            yield f"{json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"{json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(
        response_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get(
    "/search",
    summary="Search for relevant chunks",
    response_model=SearchResponse,
)
async def search(
    query: str = Query(..., min_length=1, max_length=500),
    k: int = Query(5, ge=1, le=10),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    file_type: Optional[str] = Query(
        None, description="Filter by file type (pdf, docx, md)"
    ),
    expand_query: bool = Query(False, description="Enable query expansion"),
):
    """
    Search for relevant document chunks without generating an answer.
    """
    try:
        # TODO: Storing and filtering by metadata must be improved.
        metadata_filter = {"file_type": file_type} if file_type else None
        results = app.ranker.search(
            query,
            k=k,
            alpha=alpha,
            metadata_filter=metadata_filter,
            use_query_expansion=expand_query,
        )

        formatted_results = [
            SearchResultItem(content=content, score=round(score, 2))
            for content, score in results
        ]

        return SearchResponse(
            query=query,
            num_results=len(formatted_results),
            results=formatted_results,
            expanded=expand_query,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/", summary="Root endpoint", response_model=RootResponse)
async def root():
    return RootResponse(
        message="RAG API",
        version="1.0.0",
        endpoints={
            "GET /": "This documentation",
            "GET /health": "Health check",
            "ask": "GET /ask?query=... - Ask a questions",
            "GET /search": "Search for chunks (no LLM)",
            "GET /random-metadata": "Get metadata from a random document",
        },
        examples={
            "root": "/",
            "health": "/health",
            "ask": "/ask?query=ML&expand_query=True",
            "ask": "/ask?query=What is machine learning&k=5&alpha=0.5",
            "search": "/search?query=neural networks&file_type=md",
            "random-metadata": "/random-metadata",
        },
    )


@app.get(
    "/random-metadata",
    summary="Get metadata from a random document",
    response_model=MetadataResponse,
)
async def random_metadata():
    """Get metadata from a random document."""
    try:
        if not app.ranker.bm25_store.documents:
            raise HTTPException(500, "No documents indexed")

        random_doc = random.choice(app.ranker.bm25_store.documents)

        return MetadataResponse(
            metadata=random_doc.metadata,
            content_preview=random_doc.page_content[:200] + "...",
        )
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
