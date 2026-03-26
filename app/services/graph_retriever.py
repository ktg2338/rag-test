import logging
from typing import List, Tuple

from openai import AzureOpenAI

from app.core.config import settings
from app.services.graph_store import graph_store
from app.services.community_summarizer import get_global_summaries

logger = logging.getLogger(__name__)

_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)


def extract_query_entities(query: str) -> List[str]:
    """쿼리에서 핵심 엔티티를 LLM으로 추출"""
    prompt = (
        "Extract the key entities (people, organizations, concepts, locations, products) "
        "from the following query. Output ONLY a JSON array of strings.\n\n"
        f"Query: {query}"
    )
    try:
        resp = _client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        import json
        content = resp.choices[0].message.content.strip()
        # ```json ... ``` 처리
        if "```" in content:
            import re
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if match:
                content = match.group(1).strip()
        entities = json.loads(content)
        if isinstance(entities, list):
            return [str(e) for e in entities]
    except Exception as e:
        logger.error("Query entity extraction failed: %s", e)
    return []


def _format_triples(triples: List[Tuple[str, str, str]]) -> List[str]:
    """트리플을 자연어 컨텍스트 문자열로 변환"""
    if not triples:
        return []

    # 주어 기준으로 그룹핑
    groups: dict[str, list[str]] = {}
    for subj, rel, obj in triples:
        groups.setdefault(subj, []).append(f"{rel} → {obj}")

    contexts = []
    for entity, relations in groups.items():
        context = f"[Graph] {entity}: " + "; ".join(relations)
        contexts.append(context)
    return contexts


def retrieve_graph_context(
    query: str, max_triples: int | None = None
) -> Tuple[List[str], List[str]]:
    """
    Local 검색: 쿼리 → 엔티티 추출 → 그래프 탐색 → 컨텍스트 반환.
    Returns: (context_strings, matched_entities)
    """
    if graph_store.node_count == 0:
        return [], []

    max_triples = max_triples or settings.GRAPH_MAX_CONTEXT_TRIPLES

    # 1. 쿼리에서 엔티티 추출
    query_entities = extract_query_entities(query)
    if not query_entities:
        return [], []

    # 2. 그래프에서 매칭
    matched = graph_store.find_entities(query_entities)
    if not matched:
        return [], []

    # 3. 이웃 탐색
    all_triples: List[Tuple[str, str, str]] = []
    seen = set()
    for entity in matched:
        neighbors = graph_store.get_neighbors(entity, depth=settings.GRAPH_NEIGHBOR_DEPTH)
        for triple in neighbors:
            key = (triple[0], triple[1], triple[2])
            if key not in seen:
                seen.add(key)
                all_triples.append(triple)
            if len(all_triples) >= max_triples:
                break
        if len(all_triples) >= max_triples:
            break

    # 4. 자연어 컨텍스트로 변환
    contexts = _format_triples(all_triples)
    return contexts, matched


def retrieve_global_context(query: str, top_k: int = 5) -> List[str]:
    """Global 검색: 커뮤니티 요약 기반 컨텍스트 반환"""
    if graph_store.node_count == 0:
        return []
    summaries = get_global_summaries(query, top_k=top_k)
    return [f"[Graph Summary] {s}" for s in summaries]
