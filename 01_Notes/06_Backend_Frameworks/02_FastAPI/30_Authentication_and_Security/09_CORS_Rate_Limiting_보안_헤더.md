<h2>FastAPI 학습 가이드: CORS, Rate Limiting, 보안 헤더 - API 보안 모범 사례</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 FastAPI 애플리케이션에서 CORS(Cross-Origin Resource Sharing) 설정, 요청 횟수 제한(Rate Limiting), 그리고 보안 관련 HTTP 헤더 설정 등 API 보안 모범 사례를 학습하는 것을 목표로 합니다. 이를 통해 안전하고 견고한 API를 구축하는 능력을 기릅니다.

<h2>목차</h2>

- [1. CORS (Cross-Origin Resource Sharing)](#1-cors-cross-origin-resource-sharing)
  - [1.1. 동일 출처 정책 (Same-Origin Policy)](#11-동일-출처-정책-same-origin-policy)
  - [1.2. CORS의 작동 방식](#12-cors의-작동-방식)
  - [1.3. FastAPI에서 CORS 설정](#13-fastapi에서-cors-설정)
- [2. Rate Limiting (요청 횟수 제한)](#2-rate-limiting-요청-횟수-제한)
  - [2.1. Rate Limiting의 필요성](#21-rate-limiting의-필요성)
  - [2.2. FastAPI에서 Rate Limiting 구현](#22-fastapi에서-rate-limiting-구현)
- [3. 보안 관련 HTTP 헤더](#3-보안-관련-http-헤더)
  - [3.1. `SecurityMiddleware`](#31-securitymiddleware)
  - [3.2. `Content-Security-Policy (CSP)`](#32-content-security-policy-csp)
  - [3.3. `X-Frame-Options`](#33-x-frame-options)
  - [3.4. `X-Content-Type-Options`](#34-x-content-type-options)
  - [3.5. `Referrer-Policy`](#35-referrer-policy)
  - [3.6. `Strict-Transport-Security (HSTS)`](#36-strict-transport-security-hsts)
- [4. 기타 API 보안 고려사항](#4-기타-api-보안-고려사항)
  - [4.1. 입력 값 유효성 검사](#41-입력-값-유효성-검사)
  - [4.2. 오류 메시지 최소화](#42-오류-메시지-최소화)
  - [4.3. 의존성 관리 및 업데이트](#43-의존성-관리-및-업데이트)

---

## 1. CORS (Cross-Origin Resource Sharing)

웹 브라우저는 보안상의 이유로 **동일 출처 정책(Same-Origin Policy, SOP)**을 따릅니다. 이 정책은 웹 페이지가 자신과 동일한 출처(프로토콜, 호스트, 포트)에서 로드된 리소스만 접근할 수 있도록 제한합니다. 따라서 프론트엔드(예: React, Vue 앱)가 백엔드(FastAPI API)와 다른 도메인에서 실행될 때, 브라우저는 기본적으로 API 요청을 차단합니다.

### 1.1. 동일 출처 정책 (Same-Origin Policy)
-   **출처(Origin)**: `프로토콜://호스트:포트`의 조합입니다. (예: `https://api.example.com:443`)
-   **제한**: JavaScript 코드가 다른 출처의 리소스에 접근하는 것을 막아, 악의적인 웹사이트가 사용자의 민감한 정보를 탈취하는 것을 방지합니다.

### 1.2. CORS의 작동 방식
**CORS(교차 출처 리소스 공유)**는 이 SOP에 대한 예외를 허용하는 메커니즘입니다. 서버는 특정 출처의 외부 요청을 허용하겠다는 응답 헤더를 보내고, 브라우저는 이 헤더를 확인하여 안전하다고 판단되면 API 요청을 허용합니다.

-   **Preflight Request (사전 요청)**: `GET`, `HEAD`, `POST` 같은 단순 요청이 아닌, `PUT`, `DELETE`나 `Authorization` 같은 특정 헤더를 포함하는 요청을 보낼 때, 브라우저는 본 요청을 보내기 전에 먼저 `OPTIONS` 메서드를 사용해 사전 요청을 보냅니다. 서버가 이 `OPTIONS` 요청에 대해 유효한 CORS 헤더로 응답해야만 브라우저는 본 요청을 보냅니다.

### 1.3. FastAPI에서 CORS 설정
FastAPI는 `CORSMiddleware`를 통해 CORS를 쉽게 설정할 수 있도록 지원합니다.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000", # React 개발 서버
    "https://your-frontend-domain.com", # 운영 환경 프론트엔드 도메인
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # 허용할 출처 목록
    allow_credentials=True, # 자격 증명(쿠키, HTTP 인증)을 포함한 요청 허용
    allow_methods=["*"], # 모든 HTTP 메서드 허용
    allow_headers=["*"], # 모든 HTTP 헤더 허용
)

@app.get("/")
async def read_root():
    return {"message": "Hello, CORS!"}
```
-   `allow_origins`: 요청을 허용할 출처(도메인) 목록입니다. `["*"]`로 설정하면 모든 출처를 허용하지만, **운영 환경에서는 절대 사용해서는 안 됩니다.**
-   `allow_credentials`: `True`로 설정하면 쿠키, HTTP 인증, 클라이언트 SSL 인증서와 같은 자격 증명을 포함한 요청을 허용합니다. `allow_origins`에 `"*"`를 사용할 수 없습니다.
-   `allow_methods`: 허용할 HTTP 메서드 목록입니다. `["*"`는 모든 메서드를 허용합니다.
-   `allow_headers`: 허용할 HTTP 헤더 목록입니다. `["*"`는 모든 헤더를 허용합니다.

## 2. Rate Limiting (요청 횟수 제한)

API Rate Limiting은 특정 클라이언트가 단위 시간당 API를 호출할 수 있는 횟수를 제한하는 기술입니다.

### 2.1. Rate Limiting의 필요성
-   **서비스 안정성**: 특정 사용자의 과도한 요청으로 인한 서버 과부하를 방지합니다.
-   **보안 강화**: 무차별 대입 공격(Brute-force)이나 서비스 거부(DoS) 공격의 영향을 완화합니다.
-   **공정한 사용**: 모든 사용자에게 공평한 리소스 사용 기회를 보장합니다.

### 2.2. FastAPI에서 Rate Limiting 구현
`FastAPI-Limiter`와 같은 라이브러리를 사용하여 Rate Limiting을 구현할 수 있습니다.

**1단계: 라이브러리 설치**
```bash
pip install fastapi-limiter redis
```

**2단계: Redis 연결 및 FastAPILimiter 초기화**
`FastAPI-Limiter`는 Redis를 캐시 백엔드로 사용합니다. 애플리케이션 시작 시 Redis에 연결하고 `FastAPILimiter`를 초기화합니다.

```python
from fastapi import FastAPI, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Redis 연결 설정 (환경 변수 사용 권장)
    redis_connection = redis.from_url("redis://localhost:6379", encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)

@app.get("/limited_route", dependencies=[Depends(RateLimiter(times=1, seconds=5))])
async def limited_route():
    return {"message": "이 경로는 5초에 한 번만 접근 가능합니다."}

@app.get("/user_data", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def user_data():
    return {"message": "이 경로는 1분당 5회만 접근 가능합니다."}
```
-   `RateLimiter(times=N, seconds=M)`: `M`초 동안 `N`회 요청을 허용합니다.
-   요청이 제한을 초과하면 `429 Too Many Requests` 상태 코드를 반환합니다.

## 3. 보안 관련 HTTP 헤더

HTTP 보안 헤더는 웹 애플리케이션의 보안을 강화하고, 특정 유형의 공격(예: 클릭재킹, XSS)을 방지하는 데 도움을 줍니다. FastAPI는 `fastapi.middleware.httpsredirect.HTTPSRedirectMiddleware`와 `fastapi.middleware.trustedhost.TrustedHostMiddleware`와 같은 미들웨어를 통해 일부 보안 헤더를 설정할 수 있습니다. 더 많은 보안 헤더는 `Starlette`의 `SecurityMiddleware`나 `FastAPI-Utils`의 `HTTPStrictTransportSecurityMiddleware` 등을 통해 설정할 수 있습니다.

### 3.1. `SecurityMiddleware` (Starlette)
`Starlette`의 `SecurityMiddleware`는 `X-Frame-Options`, `X-Content-Type-Options`, `X-XSS-Protection` 등 기본적인 보안 헤더를 설정합니다.

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

# app.add_middleware(SecurityHeadersMiddleware)
```

### 3.2. `Content-Security-Policy (CSP)`
CSP는 XSS(Cross-Site Scripting) 공격을 방지하는 데 효과적인 보안 헤더입니다. 브라우저가 로드할 수 있는 리소스(스크립트, 스타일시트, 이미지 등)의 출처를 제한합니다.

```python
# response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
```

### 3.3. `X-Frame-Options`
클릭재킹(Clickjacking) 공격을 방지합니다. 웹 페이지가 `<iframe>`, `<frame>`, `<object>` 태그 내에서 로드되는 것을 제어합니다. `DENY`는 어떤 경우에도 로드를 허용하지 않습니다.

### 3.4. `X-Content-Type-Options`
MIME 타입 스니핑(MIME-sniffing) 공격을 방지합니다. 브라우저가 서버가 보낸 `Content-Type` 헤더를 무시하고 콘텐츠를 추론하는 것을 막습니다.

### 3.5. `Referrer-Policy`
브라우저가 `Referer` 헤더에 어떤 정보를 포함하여 보낼지 제어합니다. 사용자 프라이버시 보호에 기여합니다.

### 3.6. `Strict-Transport-Security (HSTS)`
웹사이트가 HTTPS를 통해서만 접근되도록 브라우저에 지시합니다. 중간자 공격(Man-in-the-Middle attack)을 방지하고, HTTPS 연결을 강제합니다.

```python
# response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
```

## 4. 기타 API 보안 고려사항

### 4.1. 입력 값 유효성 검사
FastAPI는 Pydantic을 통해 강력한 유효성 검사를 제공하지만, 모든 사용자 입력(경로 파라미터, 쿼리 파라미터, 요청 본문)에 대해 항상 유효성 검사를 수행해야 합니다. 이는 SQL Injection, XSS 등 다양한 공격을 방지하는 기본 방어선입니다.

### 4.2. 오류 메시지 최소화
운영 환경에서는 상세한 오류 메시지나 스택 트레이스를 사용자에게 노출하지 않아야 합니다. 이는 공격자에게 시스템의 내부 구조에 대한 힌트를 줄 수 있습니다. 일반적인 오류 메시지를 반환하고, 상세한 오류 정보는 서버 로그에 기록해야 합니다.

### 4.3. 의존성 관리 및 업데이트
프로젝트에 사용되는 모든 라이브러리(FastAPI, Uvicorn, Pydantic 등)를 최신 상태로 유지하고, 알려진 보안 취약점이 있는지 주기적으로 확인합니다. `pip-audit`와 같은 도구를 활용할 수 있습니다.