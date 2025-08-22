<h2>LlamaIndex 학습 가이드: API 서버 배포 - 세상을 향한 창 열기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 로컬 환경에서 개발하고 테스트한 LlamaIndex RAG 애플리케이션(쿼리 엔진)을 외부 서비스나 웹 애플리케이션에서 사용할 수 있도록 REST API 형태로 배포하는 방법을 학습하는 것을 목표로 합니다. FastAPI와 같은 웹 프레임워크를 사용하여 쿼리 엔진을 감싸고, uvicorn을 통해 서버를 실행하는 기본적인 과정을 익힙니다.

<h2>목차</h2>

- [1. 왜 API 배포가 필요한가?](#1-왜-api-배포가-필요한가)
- [2. 배포를 위한 아키텍처](#2-배포를-위한-아키텍처)
- [3. FastAPI를 이용한 API 서버 구축](#3-fastapi를-이용한-api-서버-구축)
  - [3.1. 필요한 라이브러리 설치](#31-필요한-라이브러리-설치)
  - [3.2. 서버 코드 작성](#32-서버-코드-작성)
  - [3.3. 서버 실행](#33-서버-실행)
- [4. API 테스트](#4-api-테스트)

--- 

## 1. 왜 API 배포가 필요한가?

지금까지 우리가 만든 쿼리 엔진은 개발자의 컴퓨터에서만 실행되는 파이썬 객체입니다. 다른 개발자가 만든 웹사이트, 모바일 앱, 또는 다른 백엔드 서비스에서 이 RAG 시스템의 기능을 사용하려면, 표준화된 방식으로 호출할 수 있는 **창구(API)**가 필요합니다.

API(Application Programming Interface) 배포는 우리의 LlamaIndex 애플리케이션을 독립적인 서비스로 만들고, 다른 시스템과의 통합을 가능하게 하는 필수적인 과정입니다.

## 2. 배포를 위한 아키텍처

가장 기본적인 배포 아키텍처는 다음과 같습니다.

1.  **인덱스 로드**: 서버가 시작될 때, 미리 디스크에 저장해 둔 인덱스를 메모리로 불러옵니다. 이 과정은 한 번만 수행되어야 합니다.
2.  **쿼리 엔진 생성**: 로드된 인덱스로부터 쿼리 엔진을 생성합니다.
3.  **API 엔드포인트 정의**: 외부에서 HTTP 요청을 받을 수 있는 경로(예: `/query`)를 정의합니다. 이 엔드포인트는 사용자의 질문을 입력으로 받습니다.
4.  **쿼리 실행 및 응답**: 요청이 들어오면, 미리 생성해 둔 쿼리 엔진을 사용하여 답변을 생성하고, 이를 HTTP 응답(주로 JSON 형식)으로 반환합니다.

## 3. FastAPI를 이용한 API 서버 구축

FastAPI는 현대적이고 빠르며, 사용하기 쉬운 파이썬 웹 프레임워크로, LlamaIndex 애플리케이션을 배포하는 데 매우 적합합니다.

### 3.1. 필요한 라이브러리 설치

```bash
pip install fastapi uvicorn
```

### 3.2. 서버 코드 작성

`main.py`
```python
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage
import uvicorn

# 1. FastAPI 앱 초기화
app = FastAPI()

# 2. 인덱스 및 쿼리 엔진 전역 로드
# 서버 시작 시 한 번만 실행되도록 app 초기화 부분에 위치
storage_context = StorageContext.from_defaults(persist_dir="./my_index_storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

# 3. API 요청/응답 모델 정의
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# 4. API 엔드포인트 정의
@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """사용자의 질문을 받아 RAG 답변을 반환하는 엔드포인트"""
    response = query_engine.query(request.question)
    return QueryResponse(answer=str(response))

@app.get("/")
def read_root():
    return {"message": "LlamaIndex RAG API is running."}

# 5. 서버 실행 (개발용)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3.3. 서버 실행

터미널에서 다음 명령어로 FastAPI 서버를 실행합니다.

```bash
uvicorn main:app --reload
```
- `main`: `main.py` 파일을 의미
- `app`: 파일 안의 `FastAPI()` 객체 이름을 의미
- `--reload`: 코드 변경 시 서버가 자동으로 재시작되도록 하는 개발용 옵션

## 4. API 테스트

서버가 실행되면, `http://127.0.0.1:8000` 주소로 API를 호출할 수 있습니다.

- **자동 문서 확인**: 웹 브라우저에서 `http://127.0.0.1:8000/docs`로 접속하면, FastAPI가 자동으로 생성해주는 대화형 API 문서를 볼 수 있습니다. 여기서 직접 API를 테스트해볼 수 있습니다.
- **curl 사용**: 터미널에서 `curl` 명령어를 사용하여 테스트할 수 있습니다.

  ```bash
  curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What did the author do in college?"}'
  ```

이제 여러분의 RAG 애플리케이션은 다른 어떤 서비스와도 통신할 수 있는 준비를 마쳤습니다. 이 API를 기반으로 웹 챗봇을 만들거나, 다른 백엔드 서비스와 연동하는 등 무한한 확장이 가능합니다.
