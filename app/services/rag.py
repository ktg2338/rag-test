import logging
from typing import List, Optional, Tuple
from uuid import uuid4

from app.core.config import settings
from app.services import semantic_cache
from app.services.llm import generate_answer
from app.services.memory import memory
from app.services.retriever import retrieve

logger = logging.getLogger(__name__)


def answer_question(
    question: str,
    top_k: int | None = None,
    conversation_id: str | None = None,
    mode: str = "local",
    use_cache: bool = True,
    cache_threshold: Optional[float] = None,
) -> Tuple[str, List[str], str, List[str], bool, Optional[float]]:
    """질문에 답변. (answer, contexts, conversation_id, graph_entities, cache_hit, cache_similarity)"""
    if conversation_id is None:
        conversation_id = str(uuid4())

    history = memory.get(conversation_id)

    cache_enabled = settings.SEMANTIC_CACHE_ENABLED and use_cache
    if cache_enabled:
        try:
            hit = semantic_cache.lookup(question, threshold=cache_threshold)
        except Exception:
            logger.exception("semantic cache lookup failed; bypassing cache")
            hit = None
        if hit is not None:
            memory.append(conversation_id, "user", question)
            memory.append(conversation_id, "assistant", hit["answer"])
            return (
                hit["answer"],
                hit["contexts"],
                conversation_id,
                hit["graph_entities"],
                True,
                hit["similarity"],
            )

    contexts, _metas, graph_entities = retrieve(question, top_k=top_k, mode=mode)
    answer = generate_answer(question, contexts, history)
    memory.append(conversation_id, "user", question)
    memory.append(conversation_id, "assistant", answer)

    if cache_enabled:
        try:
            semantic_cache.store(question, answer, contexts, graph_entities)
        except Exception:
            logger.exception("semantic cache store failed; continuing")

    return answer, contexts, conversation_id, graph_entities, False, None
