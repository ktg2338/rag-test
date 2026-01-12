# FastAPI RAG (Azure OpenAI + ChromaDB)

Hybrid Search와 Reranking을 적용한 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

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
                              QUERY FLOW (질의응답)
═══════════════════════════════════════════════════════════════════════════════

    [Client]
        │
        │  POST /query
        │  { question, conversation_id }
        ▼
    ┌─────────┐
    │ routes  │
    │ .py     │
    └────┬────┘
         │
         ▼
    ┌─────────┐      1. conversation_id 확인
    │ rag.py  │◀────────────────────────────────────┐
    └────┬────┘                                     │
         │                                          │
         │  ┌───────────────────────────────────────┴───┐
         │  │              memory.py                    │
         │  │  (대화 히스토리 조회/저장)                 │
         │  └───────────────────────────────────────────┘
         │
         ▼
    ┌─────────────┐
    │ retriever   │ ◀─── 2. Hybrid Search + Reranking
    │ .py         │
    └──────┬──────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────┐
│ Vector  │  │  BM25   │
│ Search  │  │ Search  │
└────┬────┘  └────┬────┘
     │            │
     ▼            ▼
┌─────────┐  ┌──────────┐
│ChromaDB │  │bm25_index│
│(시맨틱) │  │(키워드)  │
└────┬────┘  └────┬─────┘
     │            │
     └─────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   Hybrid    │  3. 점수 정규화 & 가중 합산
    │   Fusion    │     (Vector: 0.7, BM25: 0.3)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Reranker   │  4. Cross-Encoder로 재정렬
    │  .py        │     (sentence-transformers)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   llm.py    │  5. Azure OpenAI로 답변 생성
    │             │     (contexts + history + question)
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Response   │  { answer, contexts, conversation_id }
    └─────────────┘
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
    ├── retriever.py     # Hybrid Search 로직
    ├── vectorstore.py   # ChromaDB 연동
    ├── embeddings.py    # Azure OpenAI Embedding
    ├── llm.py           # Azure OpenAI Chat
    ├── memory.py        # 대화 메모리
    ├── bm25_index.py    # BM25 키워드 검색
    ├── reranker.py      # Cross-Encoder Reranking
    └── chunker.py       # 텍스트 청킹
```

## 설정 옵션

`config.py`에서 다음 설정을 조정할 수 있습니다:

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `HYBRID_SEARCH_ENABLED` | `True` | Hybrid Search 활성화 |
| `BM25_WEIGHT` | `0.3` | BM25 점수 가중치 |
| `RERANKER_ENABLED` | `True` | Reranking 활성화 |
| `MAX_CONTEXT_CHUNKS` | `4` | 검색 결과 최대 개수 |
