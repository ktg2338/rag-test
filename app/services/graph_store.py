import logging
from typing import Dict, List, Tuple

import networkx as nx
from neo4j import GraphDatabase

from app.core.config import settings

logger = logging.getLogger(__name__)


class GraphStore:
    """Neo4j 기반 Knowledge Graph 저장소"""

    def __init__(self):
        self._driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
        )

    def verify_connection(self) -> None:
        """연결 확인 및 제약조건 생성"""
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT entity_name IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
        logger.info("Neo4j connection verified")

    def add_triples(
        self, triples: List[Tuple[str, str, str]], chunk_id: str | None = None
    ) -> int:
        """트리플 (subject, relation, object) 리스트를 그래프에 추가. 추가된 엣지 수 반환."""
        clean = []
        for subj, rel, obj in triples:
            subj, obj = subj.strip(), obj.strip()
            if subj and obj:
                clean.append({"subj": subj, "rel": rel, "obj": obj})

        if not clean:
            return 0

        query = """
        UNWIND $triples AS t
        MERGE (s:Entity {name: t.subj})
          ON CREATE SET s.mentions = 1,
                        s.chunk_ids = CASE WHEN $chunk_id IS NOT NULL THEN [$chunk_id] ELSE [] END
          ON MATCH SET  s.mentions = s.mentions + 1,
                        s.chunk_ids = CASE WHEN $chunk_id IS NOT NULL AND NOT $chunk_id IN s.chunk_ids
                                      THEN s.chunk_ids + $chunk_id ELSE s.chunk_ids END
        MERGE (o:Entity {name: t.obj})
          ON CREATE SET o.mentions = 1,
                        o.chunk_ids = CASE WHEN $chunk_id IS NOT NULL THEN [$chunk_id] ELSE [] END
          ON MATCH SET  o.mentions = o.mentions + 1,
                        o.chunk_ids = CASE WHEN $chunk_id IS NOT NULL AND NOT $chunk_id IN o.chunk_ids
                                      THEN o.chunk_ids + $chunk_id ELSE o.chunk_ids END
        MERGE (s)-[r:RELATED_TO]->(o)
          ON CREATE SET r.relations = [t.rel], r.weight = 1,
                        r.chunk_ids = CASE WHEN $chunk_id IS NOT NULL THEN [$chunk_id] ELSE [] END
          ON MATCH SET  r.weight = r.weight + 1,
                        r.relations = CASE WHEN NOT t.rel IN r.relations
                                      THEN r.relations + t.rel ELSE r.relations END,
                        r.chunk_ids = CASE WHEN $chunk_id IS NOT NULL AND NOT $chunk_id IN r.chunk_ids
                                      THEN r.chunk_ids + $chunk_id ELSE r.chunk_ids END
        """
        with self._driver.session() as session:
            session.run(query, triples=clean, chunk_id=chunk_id)

        return len(clean)

    def get_neighbors(
        self, entity: str, depth: int | None = None
    ) -> List[Tuple[str, str, str]]:
        """BFS로 entity 주변 트리플 탐색"""
        depth = depth or settings.GRAPH_NEIGHBOR_DEPTH
        query = """
        MATCH path = (start:Entity {name: $entity})-[:RELATED_TO*1..""" + str(depth) + """]-(end:Entity)
        UNWIND relationships(path) AS r
        WITH DISTINCT startNode(r) AS s, endNode(r) AS o, r
        UNWIND r.relations AS rel
        RETURN s.name AS subject, rel AS relation, o.name AS object
        """
        with self._driver.session() as session:
            result = session.run(query, entity=entity)
            return [(r["subject"], r["relation"], r["object"]) for r in result]

    def find_entities(self, query_entities: List[str]) -> List[str]:
        """쿼리 엔티티를 그래프 노드에 매칭 (exact → substring)"""
        query = """
        UNWIND $entities AS qe
        OPTIONAL MATCH (exact:Entity) WHERE toLower(exact.name) = toLower(qe)
        WITH qe, exact
        OPTIONAL MATCH (sub:Entity)
          WHERE exact IS NULL
            AND (toLower(sub.name) CONTAINS toLower(qe) OR toLower(qe) CONTAINS toLower(sub.name))
        WITH qe, COALESCE(exact.name, sub.name) AS matched
        WHERE matched IS NOT NULL
        RETURN DISTINCT matched
        """
        with self._driver.session() as session:
            result = session.run(query, entities=query_entities)
            return [r["matched"] for r in result]

    def get_all_triples(self) -> List[Tuple[str, str, str]]:
        query = """
        MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
        UNWIND r.relations AS rel
        RETURN s.name AS subject, rel AS relation, o.name AS object
        """
        with self._driver.session() as session:
            result = session.run(query)
            return [(r["subject"], r["relation"], r["object"]) for r in result]

    def get_entity_triples(self, entities: List[str]) -> List[Tuple[str, str, str]]:
        """주어진 엔티티 리스트와 관련된 트리플만 반환"""
        query = """
        MATCH (s:Entity)-[r:RELATED_TO]-(o:Entity)
        WHERE s.name IN $entities OR o.name IN $entities
        UNWIND r.relations AS rel
        RETURN DISTINCT startNode(r).name AS subject, rel AS relation, endNode(r).name AS object
        """
        with self._driver.session() as session:
            result = session.run(query, entities=entities)
            return [(r["subject"], r["relation"], r["object"]) for r in result]

    def get_entity_names(self, limit: int = 50) -> List[str]:
        """엔티티 이름 목록 반환 (전체 그래프 로드 없이)"""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.name AS name LIMIT $limit",
                limit=limit,
            )
            return [r["name"] for r in result]

    def save(self) -> None:
        """Neo4j는 자동 영속화 — no-op"""
        pass

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    @property
    def node_count(self) -> int:
        with self._driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
            return result.single()["cnt"]

    @property
    def edge_count(self) -> int:
        with self._driver.session() as session:
            result = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS cnt")
            return result.single()["cnt"]

    @property
    def graph(self) -> nx.Graph:
        """커뮤니티 탐지용으로 Neo4j 그래프를 NetworkX로 변환"""
        G = nx.Graph()
        with self._driver.session() as session:
            nodes = session.run("MATCH (e:Entity) RETURN e.name AS name")
            for record in nodes:
                G.add_node(record["name"])

            edges = session.run(
                "MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity) "
                "RETURN s.name AS src, o.name AS dst, r.weight AS weight"
            )
            for record in edges:
                G.add_edge(record["src"], record["dst"], weight=record["weight"])
        return G

    def close(self) -> None:
        self._driver.close()


# 싱글톤 인스턴스
graph_store = GraphStore()
