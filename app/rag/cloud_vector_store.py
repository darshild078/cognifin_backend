"""
CogniFin AI - Cloud Vector Store (Qdrant Cloud)
================================================
Lightweight, high-speed cloud vector store client.
Provides dense vector similarity search, metadata filtering (company, year, doc_type),
and Reciprocal Rank Fusion (RRF) with zero PyTorch/FAISS dependencies.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
)

from app.core.config import settings

logger = logging.getLogger("cognifin.rag.vectorstore")


@dataclass
class VectorSearchResult:
    chunk_id: str
    snippet: str
    score: float
    document_label: str
    page_number: int
    pdf_url: str
    company: str = ""
    year: str = ""
    doc_type: str = ""


class CloudVectorStore:
    """
    Managed Cloud Vector Store using Qdrant Cloud.
    """

    def __init__(self):
        self.url = os.getenv("QDRANT_URL") or getattr(settings, "QDRANT_URL", "")
        self.api_key = os.getenv("QDRANT_API_KEY") or getattr(settings, "QDRANT_API_KEY", "")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME") or getattr(settings, "QDRANT_COLLECTION_NAME", "cognifin_corpus")
        self.vector_dim = int(getattr(settings, "EMBEDDING_DIM", 768))

        self.is_configured = bool(self.url and self.api_key)
        self.client: Optional[QdrantClient] = None

        if self.is_configured:
            try:
                self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=60.0)
                self._ensure_collection()
                logger.info(f"CloudVectorStore connected to Qdrant ({self.collection_name})")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant Cloud: {e}")
                self.is_configured = False
        else:
            logger.info("Qdrant Cloud credentials not configured.")

    def _ensure_collection(self):
        if not self.client:
            return
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Could not verify/create Qdrant collection: {e}")

    def count(self) -> int:
        """
        Return the total number of indexed vectors in Qdrant Cloud.
        """
        if not self.is_configured or not self.client:
            return 0
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        company: Optional[str] = None,
        year: Optional[str] = None,
        doc_type: Optional[str] = None,
        threshold: float = 0.30,
    ) -> List[VectorSearchResult]:
        """
        Perform dense vector search with optional metadata filtering.
        """
        if not self.is_configured or not self.client:
            logger.warning("CloudVectorStore is not configured. Returning empty search.")
            return []

        # Build metadata filters
        must_conditions = []
        if company:
            must_conditions.append(
                FieldCondition(key="company", match=MatchValue(value=company.upper()))
            )
        if year:
            must_conditions.append(
                FieldCondition(key="year", match=MatchValue(value=str(year)))
            )
        if doc_type:
            must_conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=threshold,
            )

            results: List[VectorSearchResult] = []
            for hit in response.points:
                payload = hit.payload or {}
                results.append(
                    VectorSearchResult(
                        chunk_id=str(payload.get("chunk_id", hit.id)),
                        snippet=payload.get("snippet", payload.get("text", "")),
                        score=float(hit.score),
                        document_label=payload.get("document_label", ""),
                        page_number=int(payload.get("page_number", 0)),
                        pdf_url=payload.get("pdf_url", ""),
                        company=payload.get("company", ""),
                        year=payload.get("year", ""),
                        doc_type=payload.get("doc_type", ""),
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def upsert_batch(self, points: List[PointStruct]) -> bool:
        """
        Batch upload points to Qdrant Cloud.
        """
        if not self.is_configured or not self.client:
            raise ValueError("Qdrant Cloud is not configured.")
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert points to Qdrant: {e}")
            raise e


# Global singleton instance
cloud_vector_store = CloudVectorStore()

