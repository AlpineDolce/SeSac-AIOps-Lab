<h2>FastAPI 학습 가이드: APIRouter 및 미들웨어 - API 구조화 및 요청/응답 처리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 FastAPI 애플리케이션을 모듈화하고 확장 가능하게 만드는 `APIRouter`의 사용법과, 모든 요청/응답에 공통 로직을 적용할 수 있는 미들웨어(Middleware)의 개념 및 구현 방법을 학습하는 것을 목표로 합니다. 이를 통해 대규모 API 프로젝트를 효율적으로 관리하고, 공통 기능을 중앙에서 제어하는 능력을 기릅니다.

<h2>목차</h2>

- [1. APIRouter를 이용한 API 모듈화](#1-apirouter를-이용한-api-모듈화)
  - [1.1. APIRouter의 필요성](#11-apirouter의-필요성)
  - [1.2. APIRouter 기본 사용법](#12-apirouter-기본-사용법)
  - [1.3. 태그(tags)와 접두사(prefix)](#13-태그tags와-접두사prefix)
- [2. 미들웨어 (Middleware)](#2-미들웨어-middleware)
  - [2.1. 미들웨어의 개념 및 필요성](#21-미들웨어의-개념-및-필요성)
  - [2.2. 미들웨어의 작동 방식](#22-미들웨어의-작동-방식)
  - [2.3. FastAPI에서 미들웨어 구현](#23-fastapi에서-미들웨어-구현)
    - [2.3.1. `BaseHTTPMiddleware`를 이용한 미들웨어](#231-basehttpmiddleware를-이용한-미들웨어)
    - [2.3.2. `@app.middleware` 데코레이터를 이용한 미들웨어](#232-appmiddleware-데코레이터를-이용한-미들웨어)
- [3. APIRouter와 미들웨어의 조합](#3-apirouter와-미들웨어의-조합)
  - [3.1. 라우터별 미들웨어 적용](#31-라우터별-미들웨어-적용)
  - [3.2. 미들웨어의 순서](#32-미들웨어의-순서)

---

## 1. APIRouter를 이용한 API 모듈화

FastAPI 애플리케이션이 커지면 모든 경로 작업(Path Operations)을 하나의 `main.py` 파일에 작성하는 것은 비효율적입니다. 코드가 길어지고, 특정 기능(예: 사용자 관리, 상품 관리)과 관련된 엔드포인트를 찾기 어려워지며, 팀 협업 시 코드 충돌이 발생할 가능성이 높아집니다. `APIRouter`는 이러한 문제를 해결하기 위해 API를 모듈화하는 기능을 제공합니다.

### 1.1. APIRouter의 필요성
-   **코드 분리**: 특정 기능이나 리소스와 관련된 경로 작업들을 별도의 파일로 분리하여 관리할 수 있습니다.
-   **재사용성**: 분리된 라우터를 다른 FastAPI 애플리케이션에서 쉽게 재사용할 수 있습니다.
-   **협업**: 여러 개발자가 각자의 기능에 해당하는 라우터를 독립적으로 개발할 수 있어 협업 효율이 높아집니다.
-   **관리 용이성**: API 구조가 명확해지고, 특정 엔드포인트를 찾거나 수정하기 쉬워집니다.

### 1.2. APIRouter 기본 사용법
`APIRouter`는 `FastAPI` 클래스와 거의 동일한 방식으로 경로 작업을 정의할 수 있습니다.

**1단계: 라우터 파일 생성 (예: `app/routers/users.py`)**
```python
# app/routers/users.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/users/")
async def read_users():
    return ["user1", "user2"]

@router.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}
```

**2단계: 메인 애플리케이션에 라우터 포함 (예: `app/main.py`)**
```python
# app/main.py
from fastapi import FastAPI
from app.routers import users # users 라우터 임포트

app = FastAPI()

# 라우터 포함
app.include_router(users.router)

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}
```
이제 `/users/`와 `/users/{user_id}` 엔드포인트는 `main.py`가 아닌 `users.py` 파일에서 정의되었지만, `main.py`를 통해 접근할 수 있습니다.

### 1.3. 태그(tags)와 접두사(prefix)
`APIRouter`를 `include_router`할 때 `tags`와 `prefix` 인자를 사용하여 API 문서화와 URL 구조를 더욱 체계적으로 만들 수 있습니다.

```python
# app/main.py (수정)
from fastapi import FastAPI
from app.routers import users, items # items 라우터도 있다고 가정

app = FastAPI()

# users 라우터 포함: /api/v1/users/ 로 접근, Swagger UI에 "Users" 태그로 표시
app.include_router(users.router, prefix="/api/v1", tags=["Users"])

# items 라우터 포함: /api/v1/items/ 로 접근, Swagger UI에 "Items" 태그로 표시
app.include_router(items.router, prefix="/api/v1", tags=["Items"])

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}
```
-   `prefix="/api/v1"`: `users.py`와 `items.py`에 정의된 모든 경로 앞에 `/api/v1`이 자동으로 붙습니다.
-   `tags=["Users"]`: Swagger UI에서 해당 라우터에 속한 모든 엔드포인트를 "Users"라는 태그로 그룹화하여 보여줍니다.

## 2. 미들웨어 (Middleware)

미들웨어는 요청(Request)이 경로 작업 함수에 도달하기 전과, 응답(Response)이 클라이언트에게 전송되기 전에 공통적인 로직을 적용할 수 있는 계층입니다. 로깅, 인증, CORS 처리, 응답 헤더 추가 등 반복되는 작업을 중앙에서 처리할 때 유용합니다.

### 2.1. 미들웨어의 개념 및 필요성
-   **개념**: 요청-응답 사이클의 중간에 위치하여 요청과 응답을 가로채고 수정할 수 있는 함수 또는 클래스입니다.
-   **필요성**:
    *   **공통 로직 적용**: 모든 요청에 대해 인증, 로깅, 에러 처리, CORS 헤더 추가 등 공통적으로 필요한 로직을 한 곳에서 관리할 수 있습니다.
    *   **코드 중복 제거**: 각 경로 작업 함수에서 반복적으로 작성해야 할 코드를 줄여줍니다.
    *   **모듈성**: 핵심 비즈니스 로직과 공통 인프라 로직을 분리하여 코드의 가독성과 유지보수성을 높입니다.

### 2.2. 미들웨어의 작동 방식
미들웨어는 요청이 들어올 때와 응답이 나갈 때 두 번 실행됩니다.
1.  **요청 처리**: 클라이언트로부터 요청이 들어오면, 미들웨어는 요청을 가로채어 필요한 작업을 수행한 후, 다음 미들웨어 또는 경로 작업 함수로 요청을 전달합니다.
2.  **응답 처리**: 경로 작업 함수가 응답을 반환하면, 미들웨어는 응답을 다시 가로채어 필요한 작업을 수행한 후, 다음 미들웨어 또는 클라이언트에게 응답을 전달합니다.

### 2.3. FastAPI에서 미들웨어 구현

FastAPI는 `app.add_middleware()` 메서드를 통해 미들웨어를 추가할 수 있습니다.

#### 2.3.1. `BaseHTTPMiddleware`를 이용한 미들웨어
`starlette.middleware.base.BaseHTTPMiddleware`를 상속받아 클래스 기반 미들웨어를 구현합니다. `dispatch` 메서드를 오버라이드하여 요청/응답 로직을 작성합니다.

```python
# app/middlewares/timing_middleware.py
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

class TimingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request) # 다음 미들웨어 또는 경로 작업 함수 호출
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time) # 응답 헤더에 처리 시간 추가
        print(f"Request processed in {process_time:.4f} seconds")
        return response

# app/main.py (미들웨어 적용)
from fastapi import FastAPI
from app.middlewares.timing_middleware import TimingMiddleware

app = FastAPI()

app.add_middleware(TimingMiddleware)

@app.get("/slow_operation")
async def slow_operation():
    await asyncio.sleep(1) # 1초 지연
    return {"message": "느린 작업 완료"}
```

#### 2.3.2. `@app.middleware` 데코레이터를 이용한 미들웨어
함수 기반 미들웨어를 구현할 때 사용합니다. `call_next` 인자를 받아 다음 미들웨어 또는 경로 작업 함수를 호출합니다.

```python
# app/main.py (수정)
from fastapi import FastAPI, Request, Response
import time

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request) # 다음 미들웨어 또는 경로 작업 함수 호출
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"Request processed in {process_time:.4f} seconds")
    return response

@app.get("/fast_operation")
async def fast_operation():
    return {"message": "빠른 작업 완료"}
```

## 3. APIRouter와 미들웨어의 조합

### 3.1. 라우터별 미들웨어 적용
`APIRouter`에도 미들웨어를 직접 추가할 수 있습니다. 이는 특정 라우터에 속한 엔드포인트에만 미들웨어를 적용하고 싶을 때 유용합니다.

```python
# app/routers/admin.py
from fastapi import APIRouter, Request, Response
import time

admin_router = APIRouter()

@admin_router.middleware("http")
async def add_admin_log(request: Request, call_next):
    print(f"Admin request received: {request.url}")
    response = await call_next(request)
    return response

@admin_router.get("/dashboard")
async def admin_dashboard():
    return {"message": "관리자 대시보드"}

# app/main.py (미들웨어 적용)
from fastapi import FastAPI
from app.routers import admin

app = FastAPI()

app.add_middleware(TimingMiddleware) # 전역 미들웨어

app.include_router(admin.admin_router, prefix="/admin", tags=["Admin"])
```
위 예시에서 `/admin/dashboard`로 요청이 오면 `TimingMiddleware` (전역)와 `add_admin_log` (라우터별) 미들웨어가 모두 실행됩니다.

### 3.2. 미들웨어의 순서
미들웨어는 `app.add_middleware()`에 추가된 순서대로 실행됩니다. 요청이 들어올 때는 등록된 순서대로, 응답이 나갈 때는 역순으로 실행됩니다. 미들웨어의 순서는 매우 중요하며, 특히 `CORSMiddleware`와 같은 미들웨어는 다른 미들웨어보다 먼저 실행되어야 하는 경우가 많습니다.

```python
# app/main.py (미들웨어 순서 예시)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time

app = FastAPI()

# 1. CORS 미들웨어 (가장 먼저 실행되어야 하는 경우가 많음)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 개발용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 커스텀 로깅/타이밍 미들웨어
@app.middleware("http")
async def custom_logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Request to {request.url} took {process_time:.4f} seconds")
    return response

@app.get("/")
async def read_root():
    return {"message": "Hello, Middleware!"}
```
위 예시에서 요청이 들어오면 `CORSMiddleware`가 먼저 실행되고, 그 다음 `custom_logging_middleware`가 실행됩니다. 응답이 나갈 때는 `custom_logging_middleware`가 먼저 응답 헤더를 추가하고, 그 다음 `CORSMiddleware`가 CORS 관련 헤더를 추가합니다.