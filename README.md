# FastAPI RAG (Azure OpenAI + Chroma)

## 1) 설치
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Azure OpenAI 환경변수 설정

`.env` 파일에는 다음 값들을 설정합니다:

```
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_EMBED_DEPLOYMENT=
```

## 2) 서버 실행
uvicorn app.main:app --reload

## 3) 데이터 적재(옵션)
# data/raw/ 에 txt, md 파일 넣은 뒤:
python -m scripts.ingest

## 4) API 사용
# 문장 적재
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["서울은 한국의 수도입니다.", "한강은 서울을 가로지릅니다."],
    "metadatas": [{"src":"seed1"}, {"src":"seed2"}]
  }'

# 질의
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"한국의 수도는 어디야?"}'
