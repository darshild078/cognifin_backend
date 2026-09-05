import os
import time
import logging
from typing import Optional, List, Dict, Any

from app.core.config import settings
from app.core.exceptions import DependencyUnavailableException, BadRequestException, NotFoundException
from app.core.constants import ErrorCode
from app.schemas.retrieval import RetrieveRequest, RetrieveResponseData, RetrieveResultItem
from app.schemas.chat import ChatRequest, ChatResponseData, EvidenceItem
from app.utils.follow_up import extract_follow_ups
from app.services.conversation_service import create_conversation, append_to_conversation

# RAG Engines
from app.rag.retriever_pipeline import RetrieverPipeline
from app.rag.corpus_manager import CorpusManager
from app.rag.corpus_router import CorpusRouter
from app.rag.llm_client import LLMClient
from app.rag.prompt_builder import build_context, build_prompt, extract_citations
from app.rag.query_orchestrator import retrieve_context, intelligent_retrieve
from app.rag.query_understanding import parse_query
from app.rag.search_plan_builder import build_plan
from app.rag.reranker import init_reranker
from app.rag.retrieval_pipeline_v2 import refine_results
from app.rag.bm25_retriever import init_bm25
from app.rag.multi_query import init_multi_query
from app.rag.intelligent_parser import init_intelligent_parser, is_intelligent_parsing_enabled
from app.rag.context_assembler import assemble_context
from app.rag.confidence_scorer import compute_confidence
from app.rag.response_cache import response_cache
from app.rag.latency_tracker import LatencyTracker, latency_stats
from app.rag.cache_utils import has_leftover_tmp, clean_cache
from app.rag.asset_manager import ensure_index_cache

logger = logging.getLogger("cognifin.service.rag")


class RagService:
    def __init__(self):
        self.pipeline: Optional[RetrieverPipeline] = None
        self.corpus_manager: Optional[CorpusManager] = None
        self.corpus_router: Optional[CorpusRouter] = None
        self.llm_client: Optional[LLMClient] = None
        self.is_ready: bool = False

    def initialize(self):
        start_time = time.time()
        cache_dir = settings.INDEX_CACHE_DIR

        self.pipeline = RetrieverPipeline()
        self.llm_client = LLMClient()
        self.corpus_manager = CorpusManager(self.pipeline)

        ensure_index_cache()

        if not os.path.exists(cache_dir):
            logger.warning(f"Cache directory '{cache_dir}' does not exist.")
            return

        if has_leftover_tmp(cache_dir):
            clean_cache(cache_dir)

        if not self.pipeline.load_index(cache_dir):
            logger.error(f"Failed to load FAISS index from '{cache_dir}'.")
            return

        if not self.corpus_manager.load_registry(cache_dir):
            logger.error(f"Failed to load document registry from '{cache_dir}'.")
            return

        if not self.corpus_manager.validate_cache_integrity(self.pipeline.index.ntotal):
            logger.error("Cache integrity check failed.")
            return

        self.corpus_manager.init_lookup_index(cache_dir, self.pipeline.index.ntotal)
        self.corpus_router = CorpusRouter(self.corpus_manager)

        if init_reranker():
            logger.info("Reranker loaded successfully.")
        if init_bm25(self.pipeline.chunks, cache_dir):
            logger.info(f"BM25 index ready ({len(self.pipeline.chunks)} chunks).")
        if init_multi_query(self.llm_client):
            logger.info("Multi-query initialized.")
        if init_intelligent_parser(self.llm_client):
            logger.info("Intelligent query parser initialized.")

        self.is_ready = True
        elapsed = time.time() - start_time
        logger.info(
            f"action=rag_init status=success duration={elapsed:.2f}s "
            f"vectors={self.pipeline.index.ntotal} docs={len(self.corpus_manager.documents)}"
        )

    def retrieve(self, request: RetrieveRequest) -> RetrieveResponseData:
        if not self.is_ready or self.corpus_manager is None or not self.corpus_manager.is_indexed:
            raise DependencyUnavailableException(
                message="Vector corpus is not ready. Please ensure document index exists.",
                error_code=ErrorCode.CORPUS_NOT_READY,
            )

        top_k = request.top_k or settings.TOP_K
        final_k = settings.FINAL_K

        if request.session_id is not None:
            entities = self.corpus_manager.list_available_entities()
            companies = entities.get("companies", [])
            parsed = parse_query(request.query, companies)
            plan = build_plan(parsed, top_k)
            results = self.corpus_router.execute_plan(
                plan,
                embed_query=lambda q: self.pipeline.embed_query(q),
                session_id=request.session_id,
            )
        else:
            results, parsed = retrieve_context(
                raw_query=request.query,
                corpus_manager=self.corpus_manager,
                embed_query=lambda q: self.pipeline.embed_query(q),
                default_top_k=top_k,
            )

        refined = refine_results(
            results=results,
            query=request.query,
            parsed_query=parsed,
            all_chunks=self.pipeline.chunks,
            chunk_metadata=self.corpus_manager.chunk_metadata,
            final_k=final_k,
        )

        result_items = [
            RetrieveResultItem(chunk_id=r.chunk_id, score=round(r.score, 4), snippet=r.snippet)
            for r in refined
        ]

        filtered_count = top_k - len(result_items) if len(result_items) < top_k else 0
        return RetrieveResponseData(
            query=request.query,
            top_k=len(result_items),
            results=result_items,
            filtered_count=filtered_count,
        )

    def chat(self, request: ChatRequest, user_id: str) -> ChatResponseData:
        if not self.is_ready or self.corpus_manager is None or not self.corpus_manager.is_indexed:
            raise DependencyUnavailableException(
                message="Vector corpus is not ready. Please ensure document index exists.",
                error_code=ErrorCode.CORPUS_NOT_READY,
            )

        if self.llm_client is None or not self.llm_client.is_configured:
            raise DependencyUnavailableException(
                message="LLM API key is not configured.",
                error_code=ErrorCode.LLM_NOT_CONFIGURED,
            )

        tracker = LatencyTracker()
        final_k = settings.FINAL_K
        top_k = request.top_k or settings.TOP_K
        intent = "lookup"
        parsed = None

        cached = response_cache.get(request.question, request.session_id)
        if cached:
            logger.info(f"action=chat_cache_hit query='{request.question[:40]}'")
            cached_meta = cached.get("metadata", {})
            return ChatResponseData(
                answer=cached.get("answer", ""),
                citations=cached.get("citations", []),
                evidence=[EvidenceItem(**e) for e in cached.get("evidence", [])],
                conversation_id=cached.get("conversation_id"),
                metadata={**cached_meta, "cached": True},
                follow_ups=cached.get("follow_ups", []),
            )

        if request.session_id is not None:
            with tracker.track("retrieval"):
                entities = self.corpus_manager.list_available_entities()
                companies = entities.get("companies", [])
                parsed = parse_query(request.question, companies)
                plan = build_plan(parsed, top_k)
                results = self.corpus_router.execute_plan(
                    plan,
                    embed_query=lambda q: self.pipeline.embed_query(q),
                    session_id=request.session_id,
                )
            with tracker.track("reranking"):
                results = refine_results(
                    results=results,
                    query=request.question,
                    parsed_query=parsed,
                    all_chunks=self.pipeline.chunks,
                    chunk_metadata=self.corpus_manager.chunk_metadata,
                    final_k=final_k,
                )
            context, chunk_ids = build_context(results)

        elif is_intelligent_parsing_enabled():
            with tracker.track("intelligent_retrieve"):
                step_results, iq = intelligent_retrieve(
                    raw_query=request.question,
                    corpus_manager=self.corpus_manager,
                    embed_query=lambda q: self.pipeline.embed_query(q),
                    pipeline=self.pipeline,
                    default_top_k=top_k,
                )
            intent = iq.intent
            with tracker.track("context_assembly"):
                context, chunk_ids = assemble_context(step_results, intent=iq.intent)
            results = []
            for step_r in step_results.values():
                results.extend(step_r)

        else:
            with tracker.track("retrieval"):
                results, parsed = retrieve_context(
                    raw_query=request.question,
                    corpus_manager=self.corpus_manager,
                    embed_query=lambda q: self.pipeline.embed_query(q),
                    default_top_k=top_k,
                )
            if parsed:
                intent = parsed.get("intent", "lookup") if isinstance(parsed, dict) else getattr(parsed, "intent", "lookup")
            with tracker.track("reranking"):
                results = refine_results(
                    results=results,
                    query=request.question,
                    parsed_query=parsed,
                    all_chunks=self.pipeline.chunks,
                    chunk_metadata=self.corpus_manager.chunk_metadata,
                    final_k=final_k,
                )
            context, chunk_ids = build_context(results)

        system_prompt, user_message = build_prompt(context, request.question)
        raw_answer = self.llm_client.generate(system_prompt, user_message)
        answer, follow_ups = extract_follow_ups(raw_answer)
        citations = extract_citations(answer, chunk_ids)
        confidence, conf_label = compute_confidence(results, answer, request.question, citations)

        evidence = [
            EvidenceItem(
                chunk_id=r.chunk_id,
                snippet=r.snippet,
                page_number=r.page_number,
                document_label=r.document_label,
                pdf_url=r.pdf_url,
            )
            for r in results
        ]

        total_ms = tracker.get_total_ms()
        breakdown = tracker.get_breakdown()
        latency_stats.record(breakdown)
        top_score = max((r.score for r in results), default=0.0)

        # Persistence to MongoDB
        user_msg = {"role": "user", "content": request.question, "metadata": {}}
        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "metadata": {"citations": citations, "evidence": [e.model_dump() for e in evidence]},
        }

        conv_id = request.conversation_id
        try:
            if conv_id:
                append_to_conversation(conv_id, user_id, user_msg, assistant_msg)
            else:
                title = request.question[:60] + ("..." if len(request.question) > 60 else "")
                conv_id = create_conversation(user_id, title, user_msg, assistant_msg)
        except Exception as persist_err:
            logger.warning(f"action=conversation_persist_failed error='{persist_err}'")

        pipeline_metadata = {
            "confidence": confidence,
            "confidence_label": conf_label,
            "intent": intent,
            "latency_ms": round(total_ms),
            "latency_breakdown": breakdown,
            "sources_used": len(results),
            "top_score": round(top_score, 3),
            "cached": False,
            "model": self.llm_client.model,
        }

        response_data = ChatResponseData(
            answer=answer,
            citations=citations,
            evidence=evidence,
            conversation_id=conv_id,
            metadata=pipeline_metadata,
            follow_ups=follow_ups,
        )

        response_cache.set(
            request.question,
            {
                "answer": answer,
                "citations": citations,
                "evidence": [e.model_dump() for e in evidence],
                "metadata": pipeline_metadata,
                "follow_ups": follow_ups,
            },
            request.session_id,
        )

        logger.info(
            f"action=chat_generate user_id={user_id} intent={intent} "
            f"confidence={confidence:.2f} citations={len(citations)} duration={total_ms:.0f}ms"
        )
        return response_data


rag_service = RagService()
