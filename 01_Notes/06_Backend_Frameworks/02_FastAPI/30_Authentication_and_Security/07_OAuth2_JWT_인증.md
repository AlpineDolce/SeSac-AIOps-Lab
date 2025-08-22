<h2>FastAPI 학습 가이드: OAuth2 JWT 인증 - 안전한 API 접근 제어</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 FastAPI 애플리케이션에서 OAuth2 Password Flow를 이용한 사용자 인증과 JWT(JSON Web Token) 발급 및 검증 방법을 학습하는 것을 목표로 합니다. 이를 통해 안전하고 확장 가능한 API 접근 제어 시스템을 구축하는 능력을 기릅니다.

<h2>목차</h2>

- [1. API 인증의 필요성](#1-api-인증의-필요성)
  - [1.1. 인증(Authentication)과 인가(Authorization)](#11-인증authentication과-인가authorization)
  - [1.2. 토큰 기반 인증의 장점](#12-토큰-기반-인증의-장점)
- [2. OAuth2 Password Flow](#2-oauth2-password-flow)
  - [2.1. 작동 방식](#21-작동-방식)
  - [2.2. FastAPI에서 OAuth2 Password Flow 구현](#22-fastapi에서-oauth2-password-flow-구현)
- [3. JWT (JSON Web Token)](#3-jwt-json-web-token)
  - [3.1. JWT의 구조](#31-jwt의-구조)
  - [3.2. JWT의 장점](#32-jwt의-장점)
  - [3.3. JWT 발급 및 검증 라이브러리](#33-jwt-발급-및-검증-라이브러리)
- [4. FastAPI에서 JWT 인증 구현](#4-fastapi에서-jwt-인증-구현)
  - [4.1. 환경 설정](#41-환경-설정)
  - [4.2. 사용자 모델 및 데이터베이스 연동](#42-사용자-모델-및-데이터베이스-연동)
  - [4.3. 비밀번호 해싱 및 검증](#43-비밀번호-해싱-및-검증)
  - [4.4. JWT 토큰 생성 및 발급](#44-jwt-토큰-생성-및-발급)
  - [4.5. JWT 토큰 검증 및 현재 사용자 가져오기](#45-jwt-토큰-검증-및-현재-사용자-가져오기)
  - [4.6. 인증된 사용자만 접근 가능한 경로 보호](#46-인증된-사용자만-접근-가능한-경로-보호)

---

## 1. API 인증의 필요성

API(Application Programming Interface)는 애플리케이션 간의 통신을 가능하게 하는 중요한 인터페이스입니다. 하지만 모든 사용자에게 API 접근을 허용하면 보안 취약점이 발생하고, 서비스 남용으로 이어질 수 있습니다. 따라서 API에 접근하는 사용자를 식별하고, 그들이 특정 리소스에 접근할 권한이 있는지 확인하는 과정이 필수적입니다.

### 1.1. 인증(Authentication)과 인가(Authorization)
-   **인증(Authentication)**: "당신은 누구인가?"를 확인하는 과정입니다. 사용자가 주장하는 신원(예: 사용자 이름)이 실제와 일치하는지 비밀번호, 토큰 등을 통해 검증합니다.
-   **인가(Authorization)**: "당신은 무엇을 할 수 있는가?"를 확인하는 과정입니다. 인증된 사용자가 특정 리소스(예: 게시글 수정, 관리자 페이지 접근)에 접근하거나 특정 작업을 수행할 권한이 있는지 검증합니다.

### 1.2. 토큰 기반 인증의 장점
FastAPI와 같은 RESTful API에서는 세션 기반 인증보다 토큰 기반 인증이 널리 사용됩니다.
-   **상태 비저장(Stateless)**: 서버가 클라이언트의 세션 상태를 저장할 필요가 없어 서버의 확장성이 좋습니다.
-   **크로스 도메인(Cross-domain)**: CORS(Cross-Origin Resource Sharing) 문제를 쉽게 해결할 수 있습니다.
-   **모바일/SPA 친화적**: 모바일 앱이나 SPA(Single Page Application)와 같은 다양한 클라이언트에서 쉽게 사용할 수 있습니다.
-   **보안**: 토큰은 암호화되거나 서명되어 위변조를 방지합니다.

## 2. OAuth2 Password Flow

OAuth2는 권한 부여를 위한 산업 표준 프레임워크입니다. Password Flow는 사용자가 자신의 사용자 이름과 비밀번호를 클라이언트(예: 모바일 앱)에 직접 입력하고, 클라이언트가 이를 인증 서버(FastAPI 백엔드)로 보내 토큰을 발급받는 방식입니다.

### 2.1. 작동 방식
1.  **사용자 자격 증명 제출**: 사용자가 클라이언트(예: 로그인 폼)에 사용자 이름과 비밀번호를 입력합니다.
2.  **클라이언트 -> API**: 클라이언트가 사용자 이름과 비밀번호를 FastAPI 백엔드의 인증 엔드포인트로 전송합니다.
3.  **API 인증 및 토큰 발급**: FastAPI 백엔드는 사용자 이름과 비밀번호를 검증하고, 인증에 성공하면 Access Token(접근 토큰)과 Refresh Token(갱신 토큰)을 생성하여 클라이언트에게 반환합니다.
4.  **클라이언트 -> API (토큰 사용)**: 클라이언트는 발급받은 Access Token을 모든 후속 API 요청의 `Authorization` 헤더(예: `Bearer <Access Token>`)에 포함하여 보냅니다.
5.  **API 토큰 검증**: FastAPI 백엔드는 요청에 포함된 Access Token을 검증하고, 유효하면 요청을 처리합니다.

### 2.2. FastAPI에서 OAuth2 Password Flow 구현
FastAPI는 `fastapi.security.OAuth2PasswordBearer`를 통해 OAuth2 Password Flow를 쉽게 구현할 수 있도록 돕습니다.

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

app = FastAPI()

# OAuth2PasswordBearer 인스턴스 생성
# tokenUrl은 클라이언트가 토큰을 요청할 엔드포인트의 URL입니다.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 사용자 인증 및 토큰 발급 엔드포인트
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # 실제 사용자 인증 로직 (데이터베이스 조회, 비밀번호 검증 등)
    username = form_data.username
    password = form_data.password
    
    # 여기서는 간단히 하드코딩된 사용자 정보로 인증 (실제로는 DB에서 조회)
    if username == "testuser" and password == "password":
        # JWT 토큰 생성 로직 (다음 섹션에서 다룸)
        access_token = "fake-jwt-token" 
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

# 보호된 경로 (토큰이 필요함)
@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    # token 변수에 Authorization 헤더의 토큰 값이 주입됩니다.
    # 여기서는 토큰 검증 로직 (다음 섹션에서 다룸)
    return {"username": "currentuser"}
```

## 3. JWT (JSON Web Token)

JWT(JSON Web Token)는 정보를 안전하게 전송하기 위한 간결하고 자체 포함적인(self-contained) 방법입니다. 주로 인증된 사용자의 정보를 클라이언트와 서버 간에 교환하는 데 사용됩니다.

### 3.1. JWT의 구조
JWT는 점(.)으로 구분된 세 부분으로 구성됩니다.
-   **Header (헤더)**: 토큰의 타입(JWT)과 사용된 서명 알고리즘(예: HS256, RS256)을 포함합니다.
-   **Payload (페이로드)**: 클레임(Claims)이라고 불리는 사용자 정보나 추가 데이터(예: 사용자 ID, 만료 시간)를 포함합니다.
-   **Signature (서명)**: 헤더와 페이로드를 인코딩한 값과 서버의 비밀 키를 사용하여 생성된 서명입니다. 토큰의 위변조 여부를 검증하는 데 사용됩니다.

### 3.2. JWT의 장점
-   **상태 비저장(Stateless)**: 서버가 세션 정보를 저장할 필요가 없어 확장성이 좋습니다.
-   **자체 포함적(Self-contained)**: 토큰 자체에 필요한 모든 정보가 포함되어 있어 데이터베이스 조회 없이도 사용자 정보를 확인할 수 있습니다.
-   **보안**: 서명되어 있어 위변조를 방지할 수 있습니다.

### 3.3. JWT 발급 및 검증 라이브러리
파이썬에서는 `python-jose`와 `passlib` 라이브러리를 사용하여 JWT를 발급하고 검증하며, 비밀번호를 안전하게 해싱하고 검증합니다.

```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

## 4. FastAPI에서 JWT 인증 구현

FastAPI에서 JWT 인증을 구현하는 전체적인 흐름은 다음과 같습니다.

### 4.1. 환경 설정
JWT 서명에 사용될 비밀 키와 토큰 만료 시간을 환경 변수로 관리합니다.

```python
# config.py (또는 settings.py)
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key") # JWT 서명에 사용될 비밀 키
ALGORITHM = "HS256" # 서명 알고리즘
ACCESS_TOKEN_EXPIRE_MINUTES = 30 # Access Token 만료 시간 (분)
```

### 4.2. 사용자 모델 및 데이터베이스 연동
사용자 정보를 저장할 데이터베이스 모델을 정의합니다. (이전 `04_데이터베이스_설정_및_비동기_DB.md` 문서 참조)

```python
# models.py (예시)
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
```

### 4.3. 비밀번호 해싱 및 검증
사용자 비밀번호는 평문으로 저장해서는 안 됩니다. `passlib`의 `bcrypt`를 사용하여 비밀번호를 해싱하고 검증합니다.

```python
# auth_utils.py
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
```

### 4.4. JWT 토큰 생성 및 발급
사용자 인증에 성공하면 JWT Access Token을 생성하여 반환합니다.

```python
# auth_utils.py (계속)
from datetime import datetime, timedelta
from jose import JWTError, jwt
from .config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# main.py (로그인 엔드포인트)
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_db
from . import models, schemas, auth_utils

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token", response_model=schemas.Token) # schemas.Token은 access_token과 token_type을 가진 Pydantic 모델
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await models.get_user_by_username(db, form_data.username) # DB에서 사용자 조회
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth_utils.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}
```

### 4.5. JWT 토큰 검증 및 현재 사용자 가져오기
요청에 포함된 JWT 토큰을 검증하고, 유효한 경우 현재 로그인된 사용자 정보를 가져옵니다.

```python
# auth_utils.py (계속)
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from .config import SECRET_KEY, ALGORITHM
from . import models, schemas

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = schemas.TokenData(username=username) # TokenData는 username을 가진 Pydantic 모델
    except JWTError:
        raise credentials_exception
    user = await models.get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
```

### 4.6. 인증된 사용자만 접근 가능한 경로 보호
`get_current_user` 의존성을 경로 작업 함수에 주입하여, 해당 경로에 접근하려면 유효한 JWT 토큰이 필요하도록 보호합니다.

```python
# main.py (계속)
@app.get("/users/me/", response_model=schemas.UserResponse)
async def read_users_me(current_user: models.User = Depends(auth_utils.get_current_user)):
    return current_user

@app.get("/protected_route/", response_model=schemas.Message) # schemas.Message는 message를 가진 Pydantic 모델
async def protected_route(current_user: models.User = Depends(auth_utils.get_current_user)):
    return {"message": f"Hello, {current_user.username}! You accessed a protected route."}
```
이처럼 FastAPI의 의존성 주입 시스템과 JWT를 결합하면 안전하고 유연한 API 인증 시스템을 구축할 수 있습니다.