# -*- coding: utf-8 -*-
"""
answer_generator.py — Phase 5B/C/D/E/F: Grounded RAG Answer Generation

Generates natural-language answers strictly from retrieved official corpus context.
Never generates answers from LLM pre-trained knowledge alone.

Architecture:
    retrieved chunks
        |
        ├── ContextBuilder    (deduplicate, truncate, format for prompt)
        |
        ├── PromptBuilder     (strict grounding prompt)
        |
        ├── BaseLLMProvider   (provider abstraction)
        |   ├── OpenAILLMProvider
        |   └── LocalTemplateLLMProvider  (dev/offline fallback)
        |
        └── AnswerValidator   (empty, hallucination guard, source check)

GROUNDING RULES (enforced here — see also prompt):
  1. Answer only from retrieved context.
  2. Never invent facts: figures, eligibility, amounts, deadlines.
  3. Preserve exact numbers/percentages/amounts from context.
  4. Signal INSUFFICIENT_CONTEXT when context is thin.
  5. Signal CONFLICTING_SOURCES when chunks contradict each other.
  6. Every factual claim must map to a provided source chunk.
  7. Never claim a user IS definitely eligible — use hedged language.
"""
from __future__ import annotations

import datetime
import logging
import os
import re
import textwrap
from abc import ABC, abstractmethod
from typing import List, Optional

from src.rag.hybrid_retriever import HybridResult
from src.rag.schemas import GroundedAnswer, GroundingStatus, SourceCitation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MAX_CONTEXT_CHARS: int = 6000   # ~1500 tokens at 4 chars/token
DEFAULT_MAX_CHUNKS: int = 5
INSUFFICIENT_THRESHOLD_CHARS: int = 100  # fewer than this → INSUFFICIENT_CONTEXT
MIN_ANSWER_CHARS: int = 20


# ---------------------------------------------------------------------------
# Context Builder (Phase 5E)
# ---------------------------------------------------------------------------

def _deduplicate_chunks(chunks: List[HybridResult]) -> List[HybridResult]:
    """Remove near-duplicate chunks by exact text equality."""
    seen_texts: set[str] = set()
    deduped: List[HybridResult] = []
    for chunk in chunks:
        normalized = chunk.chunk_text.strip()
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            deduped.append(chunk)
    return deduped


def build_context_block(
    chunks: List[HybridResult],
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    detected_scheme_id: Optional[str] = None,
) -> tuple[str, List[HybridResult]]:
    """
    Build structured retrieved context string for the LLM prompt.

    Strategy:
      1. Deduplicate by exact text.
      2. Prioritise chunks from detected scheme when a scheme is detected.
      3. Truncate to max_chars budget.
      4. Return formatted context string and the list of chunks actually used.

    Returns:
        (context_string, used_chunks)
    """
    if not chunks:
        return "", []

    deduped = _deduplicate_chunks(chunks)[:max_chunks]

    # If scheme detected, promote scheme-specific chunks to front
    if detected_scheme_id:
        scheme_first = [c for c in deduped if c.scheme_id == detected_scheme_id]
        others = [c for c in deduped if c.scheme_id != detected_scheme_id]
        deduped = scheme_first + others

    used_chunks: List[HybridResult] = []
    parts: List[str] = []
    total_chars = 0

    for i, chunk in enumerate(deduped, start=1):
        # Prefer full chunk text, fall back to text_preview
        content = chunk.chunk_text.strip() if chunk.chunk_text.strip() else chunk.text_preview.strip()

        block = (
            f"[Source {i}]\n"
            f"Scheme ID: {chunk.scheme_id}\n"
            f"Scheme Name: {chunk.scheme_name}\n"
            f"Official URL: {getattr(chunk, 'official_url', '') or ''}\n"
            f"Source File: {chunk.source_filename}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Content:\n{content}\n"
        )

        if total_chars + len(block) > max_chars:
            # Try to include a truncated version of this chunk if budget allows
            remaining = max_chars - total_chars - 200  # leave buffer for header
            if remaining > 200:
                truncated_content = content[:remaining] + "\n[...content truncated...]"
                block = (
                    f"[Source {i}]\n"
                    f"Scheme ID: {chunk.scheme_id}\n"
                    f"Scheme Name: {chunk.scheme_name}\n"
                    f"Official URL: {getattr(chunk, 'official_url', '') or ''}\n"
                    f"Source File: {chunk.source_filename}\n"
                    f"Chunk ID: {chunk.chunk_id}\n"
                    f"Content:\n{truncated_content}\n"
                )
                parts.append(block)
                used_chunks.append(chunk)
            break

        parts.append(block)
        used_chunks.append(chunk)
        total_chars += len(block)

    context_str = "\n---\n".join(parts)
    return context_str, used_chunks


def build_source_citations(used_chunks: List[HybridResult]) -> List[SourceCitation]:
    """
    Build deduplicated SourceCitation list from used chunks.
    Deduplication key: official_url (or source_filename if url missing).
    """
    seen_keys: set[str] = set()
    citations: List[SourceCitation] = []
    sid = 1

    for chunk in used_chunks:
        url = getattr(chunk, "official_url", "") or ""
        dedup_key = url if url else chunk.source_filename

        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            citations.append(
                SourceCitation(
                    source_id=str(sid),
                    scheme_id=chunk.scheme_id,
                    scheme_name=chunk.scheme_name,
                    document_title=getattr(chunk, "document_title", None),
                    source_filename=chunk.source_filename,
                    official_url=url,
                    chunk_id=chunk.chunk_id,
                )
            )
            sid += 1

    return citations


# ---------------------------------------------------------------------------
# Prompt Builder (Phase 5D)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
You are SchemeIQ+, a factual AI assistant for official Indian government scheme information.

Your ONLY job is to answer the user's question using the official source context provided below.

STRICT GROUNDING RULES:
1. Use ONLY the information in the OFFICIAL RETRIEVED CONTEXT below. Do not use your training knowledge about government schemes.
2. Do not invent or guess any fact not explicitly stated in the context.
3. Preserve exact numbers, percentages, monetary amounts (such as ₹50,000 or 35%), dates, and conditions as they appear in the context. Do not round or approximate them.
4. If the retrieved context does not contain sufficient information to answer the question, say clearly: "I could not find sufficient information in the official SchemeIQ+ sources to answer that completely." Then share only what is actually supported.
5. If two or more sources in the context contain conflicting information about the same fact, explicitly state that the retrieved official sources contain differing information and identify which sources disagree.
6. Never claim that a user is definitely eligible for a scheme. Use hedged language such as: "Based on the information available in the retrieved official sources, you may be eligible if..."
7. Structure the answer clearly using plain language. Use bullet points or numbered lists when listing multiple criteria, benefits, or steps.
8. At the end of your answer, list the official sources used in this format:

Sources:
[1] Scheme Name — Document Title
    https://official-url

OFFICIAL RETRIEVED CONTEXT:
{retrieved_context}

USER QUESTION:
{user_query}
""").strip()


def build_prompt(
    user_query: str,
    context_str: str,
) -> str:
    """Inject query and context into the system prompt template."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_context=context_str,
        user_query=user_query,
    )


# ---------------------------------------------------------------------------
# LLM Provider Abstraction (Phase 5B)
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract base for LLM text generation providers."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate a response from the prompt. Raise on failure."""
        ...


class OpenAILLMProvider(BaseLLMProvider):
    """
    OpenAI-compatible LLM provider.
    Reads OPENAI_API_KEY and OPENAI_MODEL from environment.
    OPENAI_BASE_URL can be set for OpenAI-compatible endpoints (e.g. Together AI, local vLLM).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._model = model or os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Please set it in your .env file or as an environment variable. "
                "SchemeIQ+ will not fabricate answers without a valid LLM provider."
            )

        try:
            from openai import OpenAI
            kwargs = {"api_key": api_key}
            if base_url := (base_url or os.environ.get("OPENAI_BASE_URL")):
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        except ImportError:
            raise ImportError("openai package is not installed. Run: pip install openai")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,   # low temperature for factual grounding
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("LLM returned an empty response.")
        return content.strip()


class LocalTemplateLLMProvider(BaseLLMProvider):
    """
    Offline fallback provider for development / testing without an API key.
    Does NOT generate an LLM answer. Instead it:
      - Extracts raw context and presents it as the answer.
      - Marks grounding_status as PARTIALLY_GROUNDED.
      - Is ONLY used when no real LLM provider is configured.

    This provider must NEVER be used in production to answer scheme queries,
    since it does not perform language generation. It exists purely so the
    pipeline can be tested end-to-end without an API key.
    """

    @property
    def model_name(self) -> str:
        return "local-template-fallback"

    @property
    def provider_name(self) -> str:
        return "local"

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Extract the context block from the prompt and return it verbatim
        as a clearly-labelled fallback response.
        """
        # Pull out just the context section for a readable offline output
        context_match = re.search(
            r"OFFICIAL RETRIEVED CONTEXT:\n(.*?)\nUSER QUESTION:",
            prompt,
            re.DOTALL,
        )
        context_text = context_match.group(1).strip() if context_match else "[No context extracted]"

        return (
            "[OFFLINE FALLBACK — No LLM API key configured]\n\n"
            "The following is the raw official government source context retrieved for your query. "
            "No language model has processed or summarised this. "
            "Please configure OPENAI_API_KEY in .env for full answer generation.\n\n"
            f"{context_text}"
        )


def get_llm_provider(
    provider_type: Optional[str] = None,
    model: Optional[str] = None,
    allow_fallback: bool = True,
) -> BaseLLMProvider:
    """
    Factory: returns the configured LLM provider.

    Priority:
      1. Explicit provider_type argument
      2. LLM_PROVIDER env var
      3. If OPENAI_API_KEY is set → OpenAILLMProvider
      4. If allow_fallback=True → LocalTemplateLLMProvider (dev only)
      5. Otherwise → raise EnvironmentError

    Args:
        provider_type: "openai" or "local"
        model: override model name
        allow_fallback: if True, use LocalTemplateLLMProvider when no key present
    """
    ptype = (provider_type or os.environ.get("LLM_PROVIDER", "")).lower()

    if ptype == "openai" or (not ptype and os.environ.get("OPENAI_API_KEY", "")):
        try:
            return OpenAILLMProvider(model=model)
        except EnvironmentError:
            if allow_fallback:
                logger.warning("OPENAI_API_KEY not set — using LocalTemplateLLMProvider fallback.")
                return LocalTemplateLLMProvider()
            raise

    if allow_fallback:
        logger.warning("No LLM provider configured — using LocalTemplateLLMProvider (development only).")
        return LocalTemplateLLMProvider()

    raise EnvironmentError(
        "No LLM provider configured. Set LLM_PROVIDER=openai and OPENAI_API_KEY in .env"
    )


# ---------------------------------------------------------------------------
# Answer Validator (Phase 5I)
# ---------------------------------------------------------------------------

def validate_answer(
    answer: str,
    context_str: str,
    used_chunks: List[HybridResult],
) -> tuple[str, str]:
    """
    Lightweight deterministic validation of the generated answer.

    Checks:
      - Empty answer → GENERATION_FAILED
      - Very short answer → PARTIALLY_GROUNDED
      - No sources used → INSUFFICIENT_CONTEXT
      - Context was empty → INSUFFICIENT_CONTEXT

    Returns:
        (grounding_status, grounding_note)

    IMPORTANT: This validator checks whether an answer was generated and whether
    sources are present. It does NOT guarantee factual correctness of the answer
    content. The answer is "retrieval-grounded" (derived from official sources),
    not "verified factually correct".
    """
    if not answer or not answer.strip():
        return GroundingStatus.GENERATION_FAILED, "LLM returned an empty answer."

    if len(answer.strip()) < MIN_ANSWER_CHARS:
        return GroundingStatus.GENERATION_FAILED, f"Answer is too short ({len(answer.strip())} chars)."

    if not used_chunks:
        return (
            GroundingStatus.INSUFFICIENT_CONTEXT,
            "No relevant chunks were retrieved for this query. "
            "The answer cannot be grounded in official sources.",
        )

    if not context_str.strip() or len(context_str.strip()) < INSUFFICIENT_THRESHOLD_CHARS:
        return (
            GroundingStatus.INSUFFICIENT_CONTEXT,
            "Retrieved context was too thin to support a grounded answer.",
        )

    return GroundingStatus.GROUNDED, "Answer derived from retrieved official government corpus."


# ---------------------------------------------------------------------------
# AnswerGenerator (Phase 5B orchestrator)
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """
    Generates grounded natural-language answers from retrieved chunks.

    Usage::

        generator = AnswerGenerator()
        answer = generator.generate(
            query="Who is eligible for Rythu Bharosa?",
            retrieved_chunks=hybrid_results,
            detected_scheme_id="TS001",
        )
    """

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_tokens: int = 1024,
        allow_fallback: bool = True,
    ) -> None:
        self._llm = llm_provider or get_llm_provider(allow_fallback=allow_fallback)
        self.max_context_chars = max_context_chars
        self.max_chunks = max_chunks
        self.max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        return self._llm.provider_name

    @property
    def model_name(self) -> str:
        return self._llm.model_name

    def generate(
        self,
        query: str,
        retrieved_chunks: List[HybridResult],
        detected_scheme_id: Optional[str] = None,
        detected_scheme_name: Optional[str] = None,
        detection: Optional[dict] = None,
    ) -> GroundedAnswer:
        """
        Generate a grounded answer from the given retrieved chunks.

        Args:
            query: The user's natural language question.
            retrieved_chunks: Ranked HybridResult objects from HybridRetriever.
            detected_scheme_id: scheme_id from query_detector (may be None).
            detected_scheme_name: scheme name corresponding to detected_scheme_id.
            detection: Full detection dict from detect_scheme_with_details().

        Returns:
            GroundedAnswer with answer text, sources, and grounding metadata.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # ---- Step 1: Build context ----
        context_str, used_chunks = build_context_block(
            chunks=retrieved_chunks,
            max_chars=self.max_context_chars,
            max_chunks=self.max_chunks,
            detected_scheme_id=detected_scheme_id,
        )

        retrieved_scheme_ids = sorted(set(c.scheme_id for c in retrieved_chunks))

        # ---- Step 2: Handle empty retrieval early ----
        if not used_chunks:
            return GroundedAnswer(
                question=query,
                answer=(
                    "I could not find sufficient information in the official SchemeIQ+ sources "
                    "to answer that question. No relevant official documents were retrieved."
                ),
                detected_scheme_id=detected_scheme_id,
                detected_scheme_name=detected_scheme_name,
                retrieval_count=len(retrieved_chunks),
                retrieved_scheme_ids=retrieved_scheme_ids,
                sources=[],
                grounding_status=GroundingStatus.INSUFFICIENT_CONTEXT,
                grounding_note="No chunks passed context building — retrieval returned no usable results.",
                model_name=self.model_name,
                provider_name=self.provider_name,
                generation_timestamp=timestamp,
                context_chunks_used=0,
            )

        # ---- Step 3: Build prompt ----
        prompt = build_prompt(query, context_str)

        # ---- Step 4: Call LLM ----
        answer_text = ""
        gen_status = GroundingStatus.GROUNDED
        gen_note = ""

        try:
            answer_text = self._llm.generate(prompt, max_tokens=self.max_tokens)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer_text = (
                "I could not generate an answer at this time due to a technical error. "
                "Please check your LLM provider configuration (OPENAI_API_KEY in .env)."
            )
            gen_status = GroundingStatus.GENERATION_FAILED
            gen_note = f"LLM call failed: {e}"

        # ---- Step 5: Validate ----
        if gen_status == GroundingStatus.GROUNDED:
            gen_status, gen_note = validate_answer(answer_text, context_str, used_chunks)

        # ---- Step 6: Build source citations ----
        citations = build_source_citations(used_chunks)

        # ---- Step 7: Detect LocalTemplateLLMProvider → mark as PARTIALLY_GROUNDED ----
        if self.provider_name == "local" and gen_status == GroundingStatus.GROUNDED:
            gen_status = GroundingStatus.PARTIALLY_GROUNDED
            gen_note = (
                "This answer was produced by the offline template fallback (no LLM API key). "
                "It presents raw retrieved context, not a language-model-generated response. "
                "Configure OPENAI_API_KEY for full RAG answer generation."
            )

        return GroundedAnswer(
            question=query,
            answer=answer_text,
            detected_scheme_id=detected_scheme_id,
            detected_scheme_name=detected_scheme_name,
            retrieval_count=len(retrieved_chunks),
            retrieved_scheme_ids=retrieved_scheme_ids,
            sources=citations,
            grounding_status=gen_status,
            grounding_note=gen_note,
            model_name=self.model_name,
            provider_name=self.provider_name,
            generation_timestamp=timestamp,
            context_chunks_used=len(used_chunks),
        )
