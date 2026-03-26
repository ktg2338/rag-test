# FastAPI RAG with GraphRAG (pgvector + Neo4j)

Hybrid Search, Reranking, GraphRAG를 적용한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- **Hybrid Search**: Vector Search(pgvector) + BM25 키워드 검색 결합
- **Reranking**: Cross-Encoder를 통한 검색 결과 재정렬
- **GraphRAG**: Knowledge Graph 기반 엔티티-관계 검색 (Neo4j)
- **커뮤니티 요약**: Louvain 커뮤니티 탐지 + LLM 요약 (Global Query)
- **대화 메모리**: conversation_id 기반 멀티턴 대화 지원
- **Azure OpenAI**: Embedding 및 Chat Completion 연동

## 인프라 구성

```
┌─────────────────────────────────────────────────────┐
│                 Docker Compose                       │
│                                                      │
│  ┌──────────┐  ┌────────────────┐  ┌─────────────┐  │
│  │ FastAPI   │  │ PostgreSQL 16  │  │ Neo4j 5     │  │
│  │ :8000     │──│ + pgvector     │  │ :7687 bolt  │  │
│  │           │  │ :5432          │  │ :7474 web   │  │
│  └──────────┘  └────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
```

| 서비스 | 역할 | 이미지 |
|--------|------|--------|
| PostgreSQL + pgvector | 벡터 저장소 (임베딩 + 원문) | `pgvector/pgvector:pg16` |
| Neo4j | Knowledge Graph (엔티티 + 관계) | `neo4j:5-community` |
| FastAPI | API 서버 | 자체 빌드 |

## 실행

### Docker Compose (권장)

```bash
docker compose up -d --build
```

### 로컬 개발 (DB만 Docker)

```bash
# DB만 띄우기
docker compose up -d postgres neo4j

# venv에서 앱 실행
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

## 환경변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=your-chat-deployment
AZURE_OPENAI_EMBED_DEPLOYMENT=your-embedding-deployment

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=ragdb
POSTGRES_USER=rag
POSTGRES_PASSWORD=rag

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme123
```

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | Health check |
| POST | `/ingest` | 문서 저장 (PostgreSQL) |
| POST | `/query` | 질의응답 (mode: local/global/hybrid) |
| GET | `/documents` | 저장된 문서 조회 |
| POST | `/graph/ingest` | Knowledge Graph 구축 (Neo4j) |
| GET | `/graph/stats` | 그래프 통계 (노드/엣지/커뮤니티) |
| GET | `/graph/entities` | 엔티티 검색 |

## 시스템 흐름도

### Ingest Flow (문서 저장)

```
POST /ingest  {"texts": [...]}
    │
    ▼
 vectorstore.py ── embeddings.py ── Azure OpenAI Embedding API
    │                                (text-embedding-3-large, 1536차원)
    ▼
 PostgreSQL (documents 테이블 + HNSW 인덱스)
```

### Graph Ingest Flow (그래프 구축)

```
POST /graph/ingest
    │
    ├─ PostgreSQL에서 전체 문서 로드
    │
    ▼
 graph_extractor.py ── Azure OpenAI gpt-4o
    │                   (텍스트 → 트리플 추출)
    │
    │  예: "삼성전자는 1969년 이병철이 설립한..."
    │   → (Samsung Electronics, founded by, Lee Byung-chul)
    │   → (Samsung Electronics, is CEO of, Kyung Kye-hyun)
    │
    ▼
 Neo4j  (:Entity)-[:RELATED_TO]->(:Entity)
```

### Query Flow (질의응답)

```
POST /query  {"question": "삼성전자 CEO는?", "mode": "local"}
    │
    ▼
 rag.py ── memory.py (대화 이력 조회)
    │
    ▼
 retriever.py ── mode에 따라 분기
    │
    ├─── LOCAL ────────────────────────────────────────────┐
    │                                                      │
    │  STEP 1: Chunk 검색                                  │
    │  ┌────────────┐  ┌────────────┐                      │
    │  │ Vector     │  │ BM25       │                      │
    │  │ Search     │  │ Search     │                      │
    │  │ (pgvector) │  │ (rank-bm25)│                      │
    │  └─────┬──────┘  └─────┬──────┘                      │
    │        │  70%          │  30%                         │
    │        └───────┬───────┘                              │
    │                ▼                                      │
    │         Hybrid Fusion (정규화 + 가중 합산)              │
    │                │                                      │
    │                ▼                                      │
    │         Reranker (Cross-Encoder, Top 4 선별)           │
    │                                                      │
    │  STEP 2: Graph 검색                                  │
    │  ┌──────────────────────────────────┐                │
    │  │ 질문에서 엔티티 추출 (LLM)         │                │
    │  │ → Neo4j 엔티티 매칭               │                │
    │  │ → 이웃 탐색 (BFS, depth=2)        │                │
    │  │ → 자연어 변환                     │                │
    │  └──────────────────────────────────┘                │
    │                                                      │
    │  컨텍스트 합침:                                       │
    │  [Graph 관계] + [Chunk #1~#4]                        │
    ├──────────────────────────────────────────────────────┘
    │
    ├─── GLOBAL ───────────────────────────────────────────┐
    │  Louvain 커뮤니티 탐지 → LLM 요약 → 임베딩 유사도 랭킹  │
    ├──────────────────────────────────────────────────────┘
    │
    ├─── HYBRID ───────────────────────────────────────────┐
    │  [커뮤니티 요약] + [Graph 관계] + [Chunk #1~#4]       │
    ├──────────────────────────────────────────────────────┘
    │
    ▼
 llm.py ── Azure OpenAI gpt-4o (temp=0.2)
    │        컨텍스트 + 대화 이력 기반 답변 생성
    ▼
 Response: { answer, contexts, conversation_id, graph_entities }
```

## 프로젝트 구조

```
app/
├── main.py                      # FastAPI 앱 + lifespan (DB 초기화)
├── api/
│   └── routes.py                # API 엔드포인트 정의
├── core/
│   └── config.py                # 환경변수 설정 (Pydantic Settings)
├── models/
│   └── schemas.py               # Pydantic 요청/응답 모델
└── services/
    ├── rag.py                   # RAG 오케스트레이션
    ├── retriever.py             # Hybrid Search + Graph 통합
    ├── vectorstore.py           # PostgreSQL + pgvector 연동
    ├── graph_store.py           # Neo4j Knowledge Graph
    ├── graph_extractor.py       # LLM 기반 트리플 추출
    ├── graph_retriever.py       # 그래프 검색 (Local/Global)
    ├── community_summarizer.py  # Louvain 커뮤니티 + LLM 요약
    ├── embeddings.py            # Azure OpenAI Embedding
    ├── llm.py                   # Azure OpenAI Chat
    ├── memory.py                # 대화 메모리 (in-memory)
    ├── bm25_index.py            # BM25 키워드 검색
    ├── reranker.py              # Cross-Encoder Reranking
    └── chunker.py               # 텍스트 청킹
```

## 저장소 비교

| | PostgreSQL + pgvector | Neo4j |
|---|---|---|
| **저장 단위** | 텍스트 chunk + 1536차원 벡터 | (Entity)-[Relation]->(Entity) 트리플 |
| **검색 방식** | 코사인 유사도 + BM25 키워드 | 엔티티 매칭 + BFS 이웃 탐색 |
| **강점** | "비슷한 의미" 검색 | "관계/연결" 추적 |
| **인덱스** | HNSW (벡터), 전문검색 (BM25) | 네이티브 그래프 탐색 |
| **예시** | "CEO 관련 문서" → 유사 chunk | "삼성전자→CEO→경계현" 직접 도달 |

## 설정 옵션

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `EMBEDDING_DIMENSION` | `1536` | 임베딩 차원 수 |
| `MAX_CONTEXT_CHUNKS` | `4` | 검색 결과 최대 개수 |
| `HYBRID_SEARCH_ENABLED` | `True` | Hybrid Search 활성화 |
| `BM25_WEIGHT` | `0.3` | BM25 점수 가중치 (Vector: 0.7) |
| `RERANKER_ENABLED` | `True` | Cross-Encoder Reranking 활성화 |
| `GRAPH_ENABLED` | `True` | GraphRAG 활성화 |
| `GRAPH_NEIGHBOR_DEPTH` | `2` | 그래프 이웃 탐색 깊이 (홉) |
| `GRAPH_MAX_TRIPLES_PER_CHUNK` | `20` | chunk당 최대 트리플 추출 수 |
| `GRAPH_MAX_CONTEXT_TRIPLES` | `30` | 컨텍스트에 포함할 최대 트리플 수 |
| `GRAPH_COMMUNITY_MIN_SIZE` | `3` | 커뮤니티 최소 엔티티 수 |

## 테스트

```bash
# 1. 헬스 체크
curl http://localhost:8000/

# 2. 문서 Ingest
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["삼성전자는 1969년 이병철이 설립한 대한민국의 전자제품 제조 기업이다."]}'

# 3. Graph 구축
curl -X POST http://localhost:8000/graph/ingest \
  -H "Content-Type: application/json" -d '{}'

# 4. 그래프 통계
curl http://localhost:8000/graph/stats

# 5. 쿼리 (Local)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "삼성전자의 CEO는?", "mode": "local"}'

# 6. 쿼리 (Global)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "주요 주제는?", "mode": "global"}'

# 7. Neo4j 브라우저에서 그래프 시각화
#    http://localhost:7474 → neo4j/changeme123 로그인
#    MATCH (n) RETURN n

# 8. PostgreSQL 데이터 확인
docker compose exec postgres psql -U rag -d ragdb -c "SELECT id, LEFT(content, 30) FROM documents;"
```
