# FastAPI RAG (Azure OpenAI + ChromaDB)

LangGraph 기반 Agentic RAG 시스템입니다. 단순 파이프라인이 아닌, 에이전트가 스스로 판단하고 검증하며 답변을 생성합니다.

## 주요 기능

- **Agentic RAG**: LangGraph 기반 조건부 분기 워크플로우 (질문 라우팅, 문서 평가, 환각 검출, 쿼리 재작성)
- **Hybrid Search**: Vector Search + BM25 키워드 검색 결합
- **Reranking**: Cross-Encoder를 통한 검색 결과 재정렬
- **대화 메모리**: conversation_id 기반 멀티턴 대화 지원
- **Azure OpenAI**: Embedding 및 Chat Completion 연동

## 설치

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 환경변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=your-chat-deployment
AZURE_OPENAI_EMBED_DEPLOYMENT=your-embedding-deployment
```

## 실행

```bash
uvicorn app.main:app --reload
```

## API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | Health check |
| POST | `/ingest` | 문서 저장 |
| POST | `/query` | 질의응답 |
| GET | `/documents` | 저장된 문서 조회 |

## 시스템 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG SYSTEM FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                              INGEST FLOW (문서 저장)
═══════════════════════════════════════════════════════════════════════════════

    [Client]
        │
        │  POST /ingest
        │  { texts, metadatas }
        ▼
    ┌─────────┐      ┌─────────────┐      ┌─────────────────┐
    │ routes  │ ───▶ │ vectorstore │ ───▶ │ embeddings      │
    │ .py     │      │ .py         │      │ .py             │
    └─────────┘      └─────────────┘      └─────────────────┘
                            │                     │
                            │                     │ Azure OpenAI
                            │                     │ Embedding API
                            ▼                     ▼
                     ┌─────────────────────────────────┐
                     │         ChromaDB                │
                     │   (Vector Database 저장)        │
                     └─────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                        QUERY FLOW (Agentic RAG 질의응답)
═══════════════════════════════════════════════════════════════════════════════

    [Client]
        │
        │  POST /query { question, conversation_id }
        ▼
    ┌─────────┐     ┌─────────────┐
    │ routes  │────▶│   rag.py    │  conversation_id 확인 + memory 조회
    └─────────┘     └──────┬──────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  LangGraph Agent 시작    │
              │  (agent.py + nodes.py)  │
              └────────────┬────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   route_query   │  "검색이 필요한 질문인가?"
                  └────┬───────┬────┘
                       │       │
                 "retrieve"  "direct"
                       │       │
                       │       ▼
                       │  ┌──────────────┐
                       │  │direct_answer │  검색 없이 바로 답변
                       │  └──────┬───────┘
                       │         │
                       │         ▼
                       │       [END]
                       │
                       ▼
            ┌──────────────────────┐
            │ retrieve_documents   │  Hybrid Search + Reranking
            │                      │
            │  ┌────────┬────────┐ │
            │  │ Vector │  BM25  │ │
            │  │ (0.7)  │ (0.3) │ │
            │  └────┬───┴───┬───┘ │
            │       └───┬───┘     │
            │     Hybrid Fusion   │
            │           │         │
            │     ┌───────────┐   │
            │     │ Reranker  │   │
            │     └───────────┘   │
            └──────────┬──────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ grade_documents │  LLM이 문서 관련성 평가
              └────┬───────┬────┘
                   │       │
              관련 있음   관련 없음 (& 재시도 < 2)
                   │       │
                   │       ▼
                   │  ┌──────────────┐
                   │  │rewrite_query │──▶ retrieve_documents (재시도)
                   │  └──────────────┘
                   ▼
              ┌──────────┐
              │ generate │  컨텍스트 + 대화이력 기반 답변 생성
              └────┬─────┘
                   │
                   ▼
         ┌───────────────────┐
         │hallucination_check│  "답변이 문서에 근거하는가?"
         └────┬─────────┬────┘
              │         │
           통과       실패 (& 재시도 < 2)
              │         │
              │         └──▶ rewrite_query (재시도)
              ▼
        ┌───────────┐
        │ memory    │  대화 이력 저장
        └─────┬─────┘
              │
              ▼
        ┌──────────────────────────────────────┐
        │  Response                             │
        │  { answer, contexts,                  │
        │    conversation_id, steps }           │
        └──────────────────────────────────────┘
```

## 프로젝트 구조

```
app/
├── main.py              # FastAPI 앱 진입점
├── api/
│   └── routes.py        # API 엔드포인트 정의
├── core/
│   └── config.py        # 환경변수 설정
├── models/
│   └── schemas.py       # Pydantic 모델
└── services/
    ├── rag.py           # RAG 오케스트레이션
    ├── agent.py         # LangGraph 에이전트 그래프 구성
    ├── nodes.py         # 에이전트 노드 구현 (라우팅, 평가, 생성, 환각검출)
    ├── graph_state.py   # 에이전트 상태 정의 (AgentState)
    ├── retriever.py     # Hybrid Search 로직
    ├── vectorstore.py   # ChromaDB 연동
    ├── embeddings.py    # Azure OpenAI Embedding
    ├── llm.py           # Azure OpenAI Chat
    ├── memory.py        # 대화 메모리
    ├── bm25_index.py    # BM25 키워드 검색
    ├── reranker.py      # Cross-Encoder Reranking
    └── chunker.py       # 텍스트 청킹
```

## Agentic RAG 워크플로우

LangGraph를 사용하여 에이전트가 자율적으로 판단하는 RAG 파이프라인을 구현했습니다.

### 노드 설명

| 노드 | 역할 |
|------|------|
| `route_query` | 질문 분석 후 검색 필요 여부 판단 (retrieve / direct) |
| `retrieve_documents` | Hybrid Search(Vector + BM25) + Reranking으로 문서 검색 |
| `grade_documents` | LLM이 검색된 문서의 관련성을 개별 평가 |
| `rewrite_query` | 관련 문서 부족 시 LLM이 쿼리를 재작성하여 재검색 |
| `generate` | 필터링된 문서 + 대화 이력 기반으로 최종 답변 생성 |
| `hallucination_check` | 생성된 답변이 문서에 근거하는지 검증 |
| `direct_answer` | 인사 등 단순 질문에 검색 없이 직접 답변 |

### 자기 교정 메커니즘

- **문서 관련성 미달** → `rewrite_query` → `retrieve_documents` 재시도
- **환각 검출** → `rewrite_query` → 전체 파이프라인 재시도
- 최대 재시도 횟수: **2회** (무한 루프 방지)

## 설정 옵션

`config.py`에서 다음 설정을 조정할 수 있습니다:

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `HYBRID_SEARCH_ENABLED` | `True` | Hybrid Search 활성화 |
| `BM25_WEIGHT` | `0.3` | BM25 점수 가중치 |
| `RERANKER_ENABLED` | `True` | Reranking 활성화 |
| `MAX_CONTEXT_CHUNKS` | `4` | 검색 결과 최대 개수 |
