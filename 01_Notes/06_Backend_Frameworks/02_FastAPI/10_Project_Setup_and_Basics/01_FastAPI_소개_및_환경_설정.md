<h2>FastAPI 학습 가이드: FastAPI 소개 및 환경 설정 - 고성능 API 개발의 시작</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 FastAPI 프레임워크의 주요 특징과 장점을 이해하고, 개발 환경을 설정하며, 간단한 FastAPI 애플리케이션을 구축하는 것을 목표로 합니다. 고성능 비동기 API 개발을 위한 첫걸음을 내딛습니다.

<h2>목차</h2>

- [1. FastAPI란?](#1-fastapi란)
  - [1.1. 주요 특징](#11-주요-특징)
  - [1.2. 왜 FastAPI를 선택해야 하는가?](#12-왜-fastapi를-선택해야-하는가)
- [2. 개발 환경 설정](#2-개발-환경-설정)
  - [2.1. 가상 환경 생성 및 활성화](#21-가상-환경-생성-및-활성화)
  - [2.2. FastAPI 및 Uvicorn 설치](#22-fastapi-및-uvicorn-설치)
- [3. 첫 FastAPI 애플리케이션 구축](#3-첫-fastapi-애플리케이션-구축)
  - [3.1. 기본 코드 작성](#31-기본-코드-작성)
  - [3.2. 애플리케이션 실행](#32-애플리케이션-실행)
  - [3.3. 자동 API 문서 확인](#33-자동-api-문서-확인)

---

## 1. FastAPI란?

FastAPI는 파이썬 3.7+ 버전에서 고성능 API를 빠르고 쉽게 구축할 수 있도록 설계된 현대적인 웹 프레임워크입니다. Starlette(웹 부분)과 Pydantic(데이터 부분)을 기반으로 하며, 파이썬의 표준 타입 힌트(Type Hint)를 적극적으로 활용하여 개발 생산성과 코드 품질을 동시에 높입니다.

### 1.1. 주요 특징
-   **높은 성능**: Starlette 기반으로 비동기(async/await)를 완벽하게 지원하여 Node.js, Go와 유사한 수준의 매우 높은 성능을 제공합니다.
-   **빠른 개발 속도**: 파이썬 타입 힌트를 통해 코드 자동 완성, 데이터 유효성 검사, 직렬화/역직렬화, 그리고 자동 API 문서 생성을 지원하여 개발 시간을 단축합니다.
-   **자동 API 문서**: OpenAPI(Swagger UI)와 ReDoc을 자동으로 생성하여 API 명세서 작성 부담을 줄이고 프론트엔드 개발자와의 협업을 용이하게 합니다.
-   **데이터 유효성 검사**: Pydantic을 통해 강력한 데이터 유효성 검사와 직렬화를 제공하여 런타임 오류를 줄이고 데이터의 신뢰성을 높입니다.
-   **의존성 주입 (Dependency Injection)**: 복잡한 의존성 관리를 간결하게 처리할 수 있도록 돕습니다. 인증, 데이터베이스 세션 관리 등 다양한 곳에 활용됩니다.
-   **쉬운 학습 곡선**: Flask와 유사한 직관적인 문법을 가지고 있어 파이썬 개발자에게 친숙합니다.

### 1.2. 왜 FastAPI를 선택해야 하는가?
-   **AI/ML 백엔드**: 고성능이 요구되는 AI 모델 서빙 API나 데이터 처리 파이프라인 구축에 매우 적합합니다. 비동기 처리는 I/O 바운드 작업(예: 외부 API 호출, DB 접근)이 많은 AI 서비스에서 특히 유리합니다.
-   **마이크로서비스**: 가볍고 빠르며, 독립적인 API를 구축하기 용이하여 마이크로서비스 아키텍처에 잘 어울립니다.
-   **생산성**: 자동 문서화와 강력한 데이터 유효성 검사 덕분에 개발 초기부터 배포까지의 생산성이 높습니다.

## 2. 개발 환경 설정

FastAPI 프로젝트를 시작하기 전에, 파이썬 프로젝트 관리를 위한 가상 환경을 설정하고 필요한 라이브러리를 설치합니다.

### 2.1. 가상 환경 생성 및 활성화
파이썬 프로젝트에서는 의존성 충돌을 방지하고 프로젝트별 독립적인 환경을 구축하기 위해 가상 환경을 사용하는 것이 필수적입니다.

```bash
# 프로젝트 디렉토리 생성 및 이동
mkdir my_fastapi_app
cd my_fastapi_app

# 가상 환경 생성 (venv 사용)
python -m venv venv

# 가상 환경 활성화
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2.2. FastAPI 및 Uvicorn 설치
FastAPI 애플리케이션을 실행하기 위해서는 ASGI(Asynchronous Server Gateway Interface) 서버가 필요합니다. Uvicorn은 FastAPI와 함께 가장 널리 사용되는 ASGI 서버입니다.

```bash
# FastAPI 및 Uvicorn 설치
pip install "fastapi[all]" uvicorn
```
`fastapi[all]`은 FastAPI의 핵심 기능 외에 Pydantic, Starlette, Uvicorn 등 FastAPI가 의존하는 주요 라이브러리들을 한 번에 설치해 줍니다.

## 3. 첫 FastAPI 애플리케이션 구축

이제 기본적인 FastAPI 애플리케이션을 만들어 보고 실행해 봅시다.

### 3.1. 기본 코드 작성
프로젝트 루트 디렉토리에 `main.py` 파일을 생성하고 다음 코드를 작성합니다.

```python
# main.py
from fastapi import FastAPI

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# 경로 작업(Path Operation) 정의
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}
```

### 3.2. 애플리케이션 실행
터미널에서 프로젝트 루트 디렉토리로 이동하여 Uvicorn을 통해 애플리케이션을 실행합니다.

```bash
uvicorn main:app --reload
```
-   `main:app`: `main.py` 파일 내의 `app` 객체를 의미합니다.
-   `--reload`: 코드 변경 시 자동으로 서버를 재시작하여 개발 편의성을 높입니다. (개발 환경에서만 사용)

서버가 실행되면 터미널에 `INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`와 같은 메시지가 표시됩니다.

### 3.3. 자동 API 문서 확인
웹 브라우저를 열고 다음 주소로 접속하여 FastAPI가 자동으로 생성해주는 API 문서를 확인해 보세요.

-   **Swagger UI**: `http://127.0.0.1:8000/docs`
-   **ReDoc**: `http://127.0.0.1:8000/redoc`

이 문서들은 API 엔드포인트, 요청/응답 스키마, 파라미터 등을 자동으로 보여주며, Swagger UI에서는 직접 API를 테스트해볼 수도 있습니다. 이는 FastAPI의 강력한 기능 중 하나로, API 명세서 작성 시간을 획기적으로 줄여줍니다.