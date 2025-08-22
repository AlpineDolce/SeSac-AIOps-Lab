<h2>LangChain 학습 가이드: LangServe - API 서버 손쉽게 배포하기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 개발이 완료된 LangChain 애플리케이션(Runnable 객체)을 단 몇 줄의 코드로 견고한 REST API 서버로 배포하는 방법을 학습하는 것을 목표로 합니다. LangServe를 사용하여 API 엔드포인트를 자동으로 생성하고, 입력 데이터 검증, 스트리밍, 병렬 처리 등을 손쉽게 구현하는 방법을 익힙니다.

<h2>목차</h2>

- [1. 왜 LangServe가 필요한가?](#1-왜-langserve가-필요한가)
  - [1.1. 직접 API 서버를 만들 때의 어려움](#11-직접-api-서버를-만들-때의-어려움)
- [2. LangServe의 주요 기능](#2-langserve의-주요-기능)
- [3. LangServe를 이용한 배포 단계](#3-langserve를-이용한-배포-단계)
  - [3.1. 필요한 라이브러리 설치](#31-필요한-라이브러리-설치)
  - [3.2. API 서버 코드 작성](#32-api-서버-코드-작성)
  - [3.3. 서버 실행](#33-서버-실행)
- [4. 배포된 API 활용하기](#4-배포된-api-활용하기)
  - [4.1. Playground UI](#41-playground-ui)
  - [4.2. Python 클라이언트](#42-python-클라이언트)

---

## 1. 왜 LangServe가 필요한가?

LangChain으로 멋진 체인이나 에이전트를 만들었다고 해도, 그것은 아직 개발자의 컴퓨터에서만 동작하는 파이썬 객체일 뿐입니다. 이를 웹 서비스나 다른 애플리케이션에서 사용하려면 **API(Application Programming Interface)** 형태로 외부에서 호출할 수 있도록 만들어야 합니다.

### 1.1. 직접 API 서버를 만들 때의 어려움
FastAPI나 Flask와 같은 웹 프레임워크를 사용하여 직접 API 서버를 구축할 수 있지만, 다음과 같은 번거로운 작업들이 필요합니다.

- **입력/출력 데이터 타입 정의 및 검증**
- **스트리밍 응답 처리**
- **비동기 및 배치 요청 처리**
- **API 문서 자동 생성**
- **CORS 설정 등 웹 서버 관련 구성**

LangServe는 이러한 모든 작업을 자동화하여, 개발자가 비즈니스 로직(체인 구현)에만 집중할 수 있도록 도와줍니다.

## 2. LangServe의 주요 기능

- **자동 API 엔드포인트 생성**: LCEL로 만들어진 `Runnable` 객체를 제공하면, `invoke`, `stream`, `batch` 등에 해당하는 API 엔드포인트를 자동으로 생성합니다.
- **입력 스키마 자동 추론**: 체인의 입력 타입을 분석하여 API 요청 시 필요한 데이터 스키마를 자동으로 정의하고 검증합니다.
- **Playground 제공**: 배포된 API를 쉽게 테스트해볼 수 있는 웹 기반의 UI(Playground)를 제공합니다.
- **동시성 처리**: 여러 요청을 효율적으로 처리할 수 있도록 비동기 및 병렬 실행을 지원합니다.

## 3. LangServe를 이용한 배포 단계

### 3.1. 필요한 라이브러리 설치
LangServe는 FastAPI를 기반으로 동작합니다. 필요한 라이브러리를 설치합니다.

```bash
pip install "langserve[server]"
```

### 3.2. API 서버 코드 작성
기존에 만들었던 체인을 가져와 `add_routes` 함수를 사용하여 API 경로를 추가하는 간단한 서버 파일을 작성합니다.

`server.py`
```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. FastAPI 앱 생성
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# 2. 배포할 체인 정의 (LCEL 사용)
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model

# 3. add_routes를 사용하여 체인을 특정 경로에 추가
# 예: http://localhost:8000/joke 경로로 API가 생성됨
add_routes(
    app,
    chain,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

### 3.3. 서버 실행
터미널에서 다음 명령어로 서버를 실행합니다.

```bash
python server.py
```

## 4. 배포된 API 활용하기

서버가 실행되면, 이제 외부에서 이 API를 호출할 수 있습니다.

### 4.1. Playground UI
웹 브라우저에서 `http://localhost:8000/joke/playground/` 로 접속하면, 코드를 작성하지 않고도 API를 테스트할 수 있는 UI가 나타납니다. 입력 필드에 `{"topic": "ice cream"}` 과 같이 JSON 형식으로 데이터를 입력하고 `Start` 버튼을 누르면 결과를 확인할 수 있습니다.

또한, `http://localhost:8000/docs` 로 접속하면 FastAPI가 자동으로 생성해주는 API 문서를 볼 수 있습니다.

### 4.2. Python 클라이언트
LangServe는 배포된 API와 상호작용할 수 있는 편리한 파이썬 클라이언트(`RemoteRunnable`)도 제공합니다.

```python
from langserve import RemoteRunnable

# 배포된 API 주소로 RemoteRunnable 생성
joke_api = RemoteRunnable("http://localhost:8000/joke/")

# 로컬 체인을 사용하듯이 .invoke() 호출
response = joke_api.invoke({"topic": "programmers"})

print(response.content)
```
이처럼 LangServe를 사용하면, 복잡한 웹 서버 구현 없이 단 몇 줄의 코드로 LangChain 애플리케이션을 안정적인 API로 전환하고 다른 서비스와 쉽게 통합할 수 있습니다.