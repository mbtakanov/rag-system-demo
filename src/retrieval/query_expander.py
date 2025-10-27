"""
Query expansion and rewriting module for improved retrieval.
Implements multiple strategies to enhance search queries.
"""

import os
import logging
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.config import MODEL_NAME, MODEL_PROVIDER

load_dotenv()
logger = logging.getLogger(__name__)


MODEL = os.getenv("MODEL_NAME", MODEL_NAME)


class QueryExpander:
    def __init__(self):
        self.llm = ChatOpenAI(model=MODEL, temperature=0.3)

    def expand_with_synonyms(self, query: str) -> List[str]:
        prompt = f"""Generate 3 alternative phrasings of this query that maintain the same meaning:
Query: {query}

Return only the alternatives, one per line."""

        response = self.llm.invoke(prompt)
        variations = [query] + [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip()
        ]
        return variations[:4]

    def decompose_query(self, query: str) -> List[str]:
        # Currently, this is not used.
        prompt = f"""Break this complex query into 2-3 simpler sub-questions:
Query: {query}

Return only the sub-questions, one per line."""

        response = self.llm.invoke(prompt)
        sub_queries = [
            q.strip() for q in response.content.strip().split("\n") if q.strip()
        ]
        return sub_queries if sub_queries else [query]

    def rewrite_query(self, query: str) -> str:
        prompt = f"""Rewrite this query to be more specific and search-friendly. Keep it concise:
Query: {query}

Return only the rewritten query."""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def expand_multi_strategy(self, query: str) -> List[str]:
        results = [query]

        rewritten = self.rewrite_query(query)
        if rewritten != query:
            results.append(rewritten)

        variations = self.expand_with_synonyms(query)
        results.extend([v for v in variations if v not in results])

        logger.info(f"[Query Expansion] Generated {results} queries")
        return results[:5]
