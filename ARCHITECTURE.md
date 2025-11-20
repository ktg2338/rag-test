# RAG 시스템 아키텍처 및 흐름도

## 시스템 개요
Azure OpenAI와 ChromaDB를 활용한 Retrieval-Augmented Generation (RAG) 시스템

---

## 전체 흐름도

### 1. 문서 적재 (INGEST)

```
사용자
  │
  │  POST /ingest
  │  {"texts": ["문서1", "문서2"], "metadatas": [...]}
  │
  ▼
[routes.py] ingest()
  │
  ├─► [vectorstore.py] upsert_texts()
  │     │
  │     ├─► [embeddings.py] embed_texts()
  │     │     │
  │     │     └─► Azure OpenAI (text-embedding-3-large)
  │     │           └─► 임베딩 벡터 생성: [0.123, 0.456, ...]
  │     │
  │     └─► [ChromaDB] 저장
  │           ├─ ID: UUID 자동 생성
  │           ├─ Document: 원본 텍스트
  │           ├─ Metadata: {"src": "seed1"}
  │           └─ Embedding: 벡터 [0.123, 0.456, ...]
  │
  └─► 응답: {"inserted": 2, "ids": ["uuid1", "uuid2"]}
```

**처리 과정:**
1. 사용자가 문서 리스트와 메타데이터 전송
2. 각 문서를 Azure OpenAI Embedding API로 벡터화
3. ChromaDB에 문서, 메타데이터, 임베딩 벡터 저장
4. 생성된 ID 리스트 반환

---

### 2. 질문 응답 (QUERY/RAG)

```
사용자
  │
  │  POST /query
  │  {"question": "서울에 대해 알려줘", "conversation_id": "abc123"}
  │
  ▼
[routes.py] query()
  │
  └─► [rag.py] answer_question()
        │
        ├─► 1) Conversation ID 확인/생성
        │     └─ 없으면 새로 생성: str(uuid4())
        │
        ├─► 2) [memory.py] 대화 히스토리 조회
        │     └─ memory.get(conversation_id)
        │         └─► 이전 대화 반환: [{"role": "user", "content": "..."}, ...]
        │
        ├─► 3) [retriever.py] retrieve() - 관련 문서 검색
        │     │
        │     └─► [vectorstore.py] query_similar()
        │           │
        │           ├─► [embeddings.py] embed_texts([질문])
        │           │     └─► Azure OpenAI로 질문 임베딩 생성
        │           │
        │           └─► [ChromaDB] 코사인 유사도 검색
        │                 └─► top_k개의 가장 유사한 문서 반환
        │                       ├─ documents: ["서울은...", "한강은..."]
        │                       ├─ metadatas: [{"src": "seed1"}, ...]
        │                       └─ distances: [0.12, 0.25, ...]
        │
        ├─► 4) [llm.py] generate_answer() - LLM으로 답변 생성
        │     │
        │     ├─► 프롬프트 구성:
        │     │   ├─ System: "You are a helpful assistant..."
        │     │   ├─ History: 이전 대화 내용 (있으면)
        │     │   └─ User: "# Question\n서울에 대해...\n# Context\n서울은..."
        │     │
        │     └─► Azure OpenAI (gpt-4o)
        │           └─► 컨텍스트 기반 답변 생성
        │
        ├─► 5) [memory.py] 대화 저장
        │     ├─► memory.append(conversation_id, "user", question)
        │     └─► memory.append(conversation_id, "assistant", answer)
        │
        └─► 응답: {
              "answer": "서울은 한국의 수도입니다...",
              "contexts": ["서울은...", "한강은..."],
              "conversation_id": "abc123"
            }
```

**처리 과정:**
1. Conversation ID 확인 (없으면 생성)
2. 대화 히스토리 조회 (이전 대화 컨텍스트)
3. 질문을 임베딩으로 변환
4. ChromaDB에서 유사도 높은 문서 검색
5. 검색된 문서와 대화 히스토리를 LLM에 전달하여 답변 생성
6. 질문과 답변을 메모리에 저장
7. 답변과 사용된 컨텍스트 반환

---

### 3. 문서 조회 (DEBUG)

```
사용자
  │
  │  GET /documents
  │
  ▼
[routes.py] get_all_documents()
  │
  └─► [vectorstore.py] get_all_documents()
        │
        └─► [ChromaDB] _collection.get()
              └─► 모든 문서 반환:
                    {
                      "count": 6,
                      "ids": [...],
                      "documents": [...],
                      "metadatas": [...]
                    }
```

**처리 과정:**
1. ChromaDB에 저장된 모든 문서 조회
2. 문서 개수, ID, 원본 텍스트, 메타데이터 반환

---

## 데이터 저장소

### ChromaDB (Vector Store)
```
파일 위치: data/chroma/

Collection: "docs"
Metric: cosine similarity

Document 구조:
├─ ID: UUID (자동 생성)
├─ Document: 원본 텍스트
├─ Metadata: 사용자 정의 메타데이터
└─ Embedding: 임베딩 벡터 (1536차원)
```

### ConversationMemory (In-Memory)
```
휘발성 메모리 (서버 재시작 시 초기화)

구조:
{
  "conversation_id_1": [
    {"role": "user", "content": "질문1"},
    {"role": "assistant", "content": "답변1"},
    {"role": "user", "content": "질문2"},
    {"role": "assistant", "content": "답변2"}
  ],
  "conversation_id_2": [...]
}
```

---

## Azure OpenAI 사용

### Embedding API
- **Deployment**: `text-embedding-3-large`
- **용도**: 텍스트 → 벡터 변환
- **호출 시점**:
  1. 문서 적재 시: 각 문서를 벡터로 변환
  2. 질문 검색 시: 질문을 벡터로 변환
- **입력**: 텍스트 문자열 또는 리스트
- **출력**: 1536차원 벡터

### Chat Completion API
- **Deployment**: `gpt-4o`
- **용도**: 질문에 대한 답변 생성
- **호출 시점**: 검색된 컨텍스트로 답변 생성
- **입력**:
  - System prompt
  - 대화 히스토리
  - 질문 + 검색된 컨텍스트
- **출력**: 답변 텍스트

---

## 주요 컴포넌트

### API 라우트 (app/api/routes.py)
```python
POST /ingest       # 문서 적재
POST /query        # 질문 응답 (RAG)
GET  /documents    # 저장된 문서 조회 (디버그용)
```

### 서비스 레이어

#### vectorstore.py
- `upsert_texts()`: 문서 저장/업데이트
- `query_similar()`: 유사 문서 검색
- `get_all_documents()`: 전체 문서 조회

#### embeddings.py
- `embed_texts()`: Azure OpenAI Embedding API 호출

#### retriever.py
- `retrieve()`: 질문에 대한 관련 문서 검색

#### llm.py
- `generate_answer()`: Azure OpenAI Chat API로 답변 생성

#### rag.py
- `answer_question()`: RAG 전체 파이프라인 조율

#### memory.py
- `ConversationMemory`: 대화 히스토리 관리
  - `get()`: 대화 조회
  - `append()`: 대화 추가

---

## 핵심 흐름 요약

```
INGEST 흐름:
문서 입력 → 임베딩 변환 → ChromaDB 저장 → ID 반환

QUERY 흐름:
질문 입력 → 질문 임베딩 → 유사 문서 검색 → LLM에 컨텍스트 전달
→ 답변 생성 → 대화 저장 → 답변 반환
```

---

## 설정 (app/core/config.py)

```python
AZURE_OPENAI_API_KEY          # Azure OpenAI API 키
AZURE_OPENAI_ENDPOINT         # Azure OpenAI 엔드포인트
AZURE_OPENAI_API_VERSION      # API 버전 (2024-02-15-preview)
AZURE_OPENAI_DEPLOYMENT       # Chat 모델 배포 이름 (gpt-4o)
AZURE_OPENAI_EMBED_DEPLOYMENT # Embedding 모델 배포 이름 (text-embedding-3-large)
CHROMA_PATH                   # ChromaDB 저장 경로 (data/chroma)
MAX_CONTEXT_CHUNKS            # 검색할 최대 문서 수 (4)
```

---

## API 사용 예시

### 문서 적재
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["서울은 한국의 수도입니다.", "한강은 ���울을 가로지릅니다."],
    "metadatas": [{"src":"seed1"}, {"src":"seed2"}]
  }'
```

### 질문하기
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "서울에 대해 알려줘",
    "top_k": 4,
    "conversation_id": "abc-123"
  }'
```

### 저장된 문서 확인
```bash
curl http://localhost:8000/documents
```

---

## 기술 스택

- **Framework**: FastAPI
- **Vector DB**: ChromaDB (Persistent)
- **LLM**: Azure OpenAI (gpt-4o)
- **Embedding**: Azure OpenAI (text-embedding-3-large)
- **Memory**: In-Memory Dictionary (휘발성)
