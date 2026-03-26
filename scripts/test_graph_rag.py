"""
GraphRAG 통합 테스트 스크립트.

사용법:
    python -m scripts.test_graph_rag          # 전체 테스트 (LLM 호출 포함)
    python -m scripts.test_graph_rag --local   # LLM 없이 GraphStore만 테스트
"""

import argparse
import sys


def test_graph_store():
    """GraphStore 단위 테스트 (LLM 호출 없음)"""
    print("=" * 50)
    print("1. GraphStore 테스트")
    print("=" * 50)

    from app.services.graph_store import GraphStore

    store = GraphStore.__new__(GraphStore)
    import networkx as nx
    store._graph = nx.Graph()

    # 트리플 추가
    triples = [
        ("삼성전자", "본사 위치", "수원"),
        ("삼성전자", "CEO", "이재용"),
        ("삼성전자", "생산", "Galaxy S25"),
        ("이재용", "출생지", "서울"),
        ("수원", "소속", "경기도"),
        ("Galaxy S25", "탑재", "Snapdragon 8 Gen 4"),
    ]
    added = store.add_triples(triples, chunk_id="test-001")
    print(f"  추가된 트리플: {added}")
    print(f"  노드 수: {store.node_count}")
    print(f"  엣지 수: {store.edge_count}")
    assert store.node_count == 7, f"Expected 7 nodes, got {store.node_count}"
    assert store.edge_count == 6, f"Expected 6 edges, got {store.edge_count}"

    # 이웃 탐색
    neighbors = store.get_neighbors("삼성전자", depth=1)
    print(f"\n  삼성전자 depth=1 이웃: {len(neighbors)}개 트리플")
    for t in neighbors:
        print(f"    {t[0]} --[{t[1]}]--> {t[2]}")

    neighbors_d2 = store.get_neighbors("삼성전자", depth=2)
    print(f"\n  삼성전자 depth=2 이웃: {len(neighbors_d2)}개 트리플")
    for t in neighbors_d2:
        print(f"    {t[0]} --[{t[1]}]--> {t[2]}")
    assert len(neighbors_d2) > len(neighbors), "Depth 2 should find more triples"

    # 엔티티 매칭
    matched = store.find_entities(["삼성", "galaxy"])
    print(f"\n  '삼성', 'galaxy' 매칭: {matched}")
    assert len(matched) >= 2, "Should match at least 2 entities"

    # 커뮤니티 탐지
    import community as community_louvain
    partition = community_louvain.best_partition(store.graph)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    print(f"\n  커뮤니티 수: {len(communities)}")
    for cid, members in communities.items():
        print(f"    커뮤니티 {cid}: {members}")

    # JSON 영속화
    import tempfile, os, json
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        store.save(tmp_path)
        size = os.path.getsize(tmp_path)
        print(f"\n  JSON 저장: {tmp_path} ({size} bytes)")

        # 로드 테스트
        store2 = GraphStore.__new__(GraphStore)
        store2._graph = nx.Graph()
        store2.load(tmp_path)
        assert store2.node_count == store.node_count
        assert store2.edge_count == store.edge_count
        print(f"  JSON 로드 OK: {store2.node_count} nodes, {store2.edge_count} edges")
    finally:
        os.unlink(tmp_path)

    print("\n  ✅ GraphStore 테스트 통과!\n")


def test_triple_extraction():
    """LLM 트리플 추출 테스트"""
    print("=" * 50)
    print("2. Triple Extraction 테스트 (LLM 호출)")
    print("=" * 50)

    from app.services.graph_extractor import extract_triples

    text = (
        "삼성전자는 1969년 이병철이 설립한 대한민국의 전자제품 제조 기업이다. "
        "현재 CEO는 경계현이며, 본사는 경기도 수원시에 위치해 있다. "
        "주요 제품으로는 Galaxy 스마트폰, QLED TV, 반도체 등이 있다. "
        "삼성전자는 TSMC와 반도체 파운드리 시장에서 경쟁하고 있다."
    )

    print(f"\n  입력 텍스트: {text[:80]}...")
    triples = extract_triples(text)
    print(f"  추출된 트리플: {len(triples)}개")
    for s, r, o in triples:
        print(f"    {s} --[{r}]--> {o}")

    assert len(triples) > 0, "Should extract at least 1 triple"
    print("\n  ✅ Triple Extraction 테스트 통과!\n")
    return triples


def test_graph_retrieval(triples):
    """Graph Retriever 테스트"""
    print("=" * 50)
    print("3. Graph Retrieval 테스트 (LLM 호출)")
    print("=" * 50)

    from app.services.graph_store import graph_store
    from app.services.graph_retriever import extract_query_entities, retrieve_graph_context

    # 기존 그래프 초기화 후 테스트 데이터 추가
    graph_store.clear()
    graph_store.add_triples(triples, chunk_id="test-001")
    print(f"\n  그래프 상태: {graph_store.node_count} nodes, {graph_store.edge_count} edges")

    # 쿼리 엔티티 추출
    query = "삼성전자의 CEO는 누구인가?"
    entities = extract_query_entities(query)
    print(f"\n  쿼리: '{query}'")
    print(f"  추출 엔티티: {entities}")

    # 그래프 매칭
    matched = graph_store.find_entities(entities)
    print(f"  매칭된 엔티티: {matched}")

    # 컨텍스트 검색
    contexts, matched_entities = retrieve_graph_context(query)
    print(f"\n  Graph 컨텍스트 ({len(contexts)}개):")
    for ctx in contexts:
        print(f"    {ctx}")

    graph_store.clear()
    print("\n  ✅ Graph Retrieval 테스트 통과!\n")


def test_full_pipeline():
    """서버 기동 후 API 호출 테스트 (수동)"""
    print("=" * 50)
    print("4. Full Pipeline API 테스트 방법")
    print("=" * 50)
    print("""
  서버를 먼저 기동하세요:
    uvicorn app.main:app --reload

  그 후 아래 순서로 API를 호출하세요:

  # Step 1: 문서 Ingest (ChromaDB)
  curl -X POST http://localhost:8000/ingest \\
    -H "Content-Type: application/json" \\
    -d '{"texts": [
      "삼성전자는 1969년 이병철이 설립한 대한민국의 전자제품 제조 기업이다. 현재 CEO는 경계현이다.",
      "삼성전자의 본사는 경기도 수원시에 위치해 있다. 주요 제품으로는 Galaxy 스마트폰이 있다.",
      "TSMC는 대만의 반도체 파운드리 기업이다. 삼성전자와 파운드리 시장에서 경쟁하고 있다."
    ]}'

  # Step 2: Graph Ingest (Knowledge Graph 구축)
  curl -X POST http://localhost:8000/graph/ingest \\
    -H "Content-Type: application/json" \\
    -d '{}'

  # Step 3: Graph 통계 확인
  curl http://localhost:8000/graph/stats

  # Step 4: 엔티티 검색
  curl "http://localhost:8000/graph/entities?q=삼성"

  # Step 5: Local 쿼리 (chunk + graph)
  curl -X POST http://localhost:8000/query \\
    -H "Content-Type: application/json" \\
    -d '{"question": "삼성전자의 CEO는 누구인가?", "mode": "local"}'

  # Step 6: Global 쿼리 (커뮤니티 요약)
  curl -X POST http://localhost:8000/query \\
    -H "Content-Type: application/json" \\
    -d '{"question": "전체 문서의 주요 주제를 요약해줘", "mode": "global"}'

  # Step 7: Hybrid 쿼리
  curl -X POST http://localhost:8000/query \\
    -H "Content-Type: application/json" \\
    -d '{"question": "삼성전자와 TSMC의 관계는?", "mode": "hybrid"}'
""")


def main():
    parser = argparse.ArgumentParser(description="GraphRAG 테스트")
    parser.add_argument("--local", action="store_true", help="LLM 호출 없이 GraphStore만 테스트")
    args = parser.parse_args()

    test_graph_store()

    if args.local:
        print("🏁 Local 테스트 완료 (LLM 호출 스킵)")
        return

    triples = test_triple_extraction()
    test_graph_retrieval(triples)
    test_full_pipeline()
    print("🏁 전체 테스트 완료!")


if __name__ == "__main__":
    main()
