from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: str
    vector_db: str
    documents_indexed: int
    error: Optional[str] = None


class SearchResultItem(BaseModel):
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    num_results: int
    results: List[SearchResultItem]
    expanded: bool


class MetadataResponse(BaseModel):
    metadata: Dict[str, Any]
    content_preview: str


class RootResponse(BaseModel):
    message: str
    version: str
    endpoints: Dict[str, str]
    examples: Dict[str, str]
