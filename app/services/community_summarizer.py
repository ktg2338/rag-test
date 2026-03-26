import logging
from typing import Dict, FrozenSet, List, Tuple

import community as community_louvain
from openai import AzureOpenAI

from app.core.config import settings
from app.services.embeddings import embed_texts
from app.services.graph_store import graph_store

logger = logging.getLogger(__name__)

_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

# 커뮤니티 요약 캐시 (그래프 업데이트 시 초기화 필요)
_summary_cache: Dict[FrozenSet[str], str] = {}


def invalidate_cache() -> None:
    """그래프 업데이트 후 캐시 초기화"""
    _summary_cache.clear()


def detect_communities() -> List[List[str]]:
    """Louvain 알고리즘으로 커뮤니티 탐지. 최소 크기 이상의 커뮤니티만 반환."""
    graph = graph_store.graph
    if graph.number_of_nodes() < 2:
        return []

    partition = community_louvain.best_partition(
        graph, resolution=settings.GRAPH_COMMUNITY_RESOLUTION
    )

    # community_id → [entities] 그룹핑
    communities: Dict[int, List[str]] = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    # 최소 크기 필터링
    return [
        entities
        for entities in communities.values()
        if len(entities) >= settings.GRAPH_COMMUNITY_MIN_SIZE
    ]


def summarize_community(
    entities: List[str], triples: List[Tuple[str, str, str]]
) -> str:
    """커뮤니티의 엔티티와 트리플을 LLM으로 요약"""
    cache_key = frozenset(entities)
    if cache_key in _summary_cache:
        return _summary_cache[cache_key]

    triple_lines = "\n".join(
        f"- {s} → {r} → {o}" for s, r, o in triples[:50]  # 토큰 제한
    )
    prompt = (
        f"다음은 관련된 엔티티들과 그들 간의 관계입니다.\n\n"
        f"엔티티: {', '.join(entities)}\n\n"
        f"관계:\n{triple_lines}\n\n"
        f"위 정보를 바탕으로 이 그룹이 어떤 주제/토픽에 대한 것인지 "
        f"2-3문장으로 요약해주세요. 한국어로 답변하세요."
    )

    try:
        resp = _client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        summary = resp.choices[0].message.content.strip()
        _summary_cache[cache_key] = summary
        return summary
    except Exception as e:
        logger.error("Community summarization failed: %s", e)
        return f"Community: {', '.join(entities[:10])}"


def get_global_summaries(query: str, top_k: int = 5) -> List[str]:
    """
    Global query: 모든 커뮤니티를 요약 후, 쿼리와 가장 관련 높은 요약 반환.
    1. Louvain 커뮤니티 탐지
    2. 각 커뮤니티 요약 (캐시 활용)
    3. 쿼리 임베딩 vs 요약 임베딩으로 유사도 랭킹
    """
    communities = detect_communities()
    if not communities:
        return []

    # 각 커뮤니티 요약 생성
    summaries = []
    for entities in communities:
        triples = graph_store.get_entity_triples(entities)
        summary = summarize_community(entities, triples)
        summaries.append(summary)

    if not summaries:
        return []

    # top_k가 전체 수 이상이면 전부 반환
    if len(summaries) <= top_k:
        return summaries

    # 임베딩 유사도로 랭킹
    all_texts = [query] + summaries
    embeddings = embed_texts(all_texts)
    query_emb = embeddings[0]
    summary_embs = embeddings[1:]

    # 코사인 유사도 계산
    scores = []
    for i, s_emb in enumerate(summary_embs):
        dot = sum(a * b for a, b in zip(query_emb, s_emb))
        norm_q = sum(a * a for a in query_emb) ** 0.5
        norm_s = sum(a * a for a in s_emb) ** 0.5
        sim = dot / (norm_q * norm_s) if norm_q * norm_s > 0 else 0
        scores.append((i, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [summaries[i] for i, _ in scores[:top_k]]
