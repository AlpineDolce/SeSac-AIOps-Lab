<h2>FastAPI 학습 가이드: SQLAlchemy ORM 기본 - 파이썬 객체로 데이터베이스 다루기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 SQLAlchemy ORM(Object-Relational Mapping)의 기본 개념을 이해하고, FastAPI 애플리케이션에서 SQLAlchemy ORM을 사용하여 데이터베이스 모델을 정의하고 CRUD(Create, Read, Update, Delete) 작업을 수행하는 방법을 학습하는 것을 목표로 합니다. 파이썬 객체 지향 방식으로 데이터를 효율적으로 관리하는 능력을 기릅니다.

<h2>목차</h2>

- [1. ORM(Object-Relational Mapping)이란?](#1-ormobject-relational-mapping이란)
  - [1.1. ORM의 장점](#11-orm의-장점)
  - [1.2. SQLAlchemy ORM의 특징](#12-sqlalchemy-orm의-특징)
- [2. SQLAlchemy ORM 모델 정의](#2-sqlalchemy-orm-모델-정의)
  - [2.1. DeclarativeBase와 Base](#21-declarativebase와-base)
  - [2.2. 테이블 이름과 컬럼 정의](#22-테이블-이름과-컬럼-정의)
  - [2.3. 필드 타입과 옵션](#23-필드-타입과-옵션)
  - [2.4. 관계 정의 (Relationships)](#24-관계-정의-relationships)
- [3. 세션(Session) 관리](#3-세션session-관리)
  - [3.1. 세션의 역할](#31-세션의-역할)
  - [3.2. 비동기 세션 생성 및 사용](#32-비동기-세션-생성-및-사용)
- [4. 기본 CRUD(Create, Read, Update, Delete) 작업](#4-기본-crudcreate-read-update-delete-작업)
  - [4.1. 데이터 생성 (Create)](#41-데이터-생성-create)
  - [4.2. 데이터 조회 (Read)](#42-데이터-조회-read)
  - [4.3. 데이터 수정 (Update)](#43-데이터-수정-update)
  - [4.4. 데이터 삭제 (Delete)](#44-데이터-삭제-delete)
- [5. FastAPI와 SQLAlchemy ORM 연동 예시](#5-fastapi와-sqlalchemy-orm-연동-예시)
  - [5.1. 프로젝트 구조](#51-프로젝트-구조)
  - [5.2. `database.py` (DB 설정 및 세션)](#52-databasepy-db-설정-및-세션)
  - [5.3. `models.py` (ORM 모델 정의)](#53-modelspy-orm-모델-정의)
  - [5.4. `schemas.py` (Pydantic 스키마 정의)](#54-schemaspy-pydantic-스키마-정의)
  - [5.5. `main.py` (FastAPI 엔드포인트)](#55-mainpy-fastapi-엔드포인트)

---

## 1. ORM(Object-Relational Mapping)이란?

ORM(Object-Relational Mapping)은 객체 지향 프로그래밍 언어의 객체와 관계형 데이터베이스의 데이터를 자동으로 매핑(연결)하는 기술입니다. 개발자는 SQL 쿼리를 직접 작성하는 대신, 파이썬 클래스와 객체를 사용하여 데이터베이스를 조작할 수 있습니다.

### 1.1. ORM의 장점
-   **생산성 향상**: SQL 쿼리 작성 시간을 줄이고 파이썬 코드로 데이터베이스를 다룰 수 있어 개발 속도가 빨라집니다.
-   **유지보수 용이**: SQL과 파이썬 코드가 분리되어 코드의 가독성과 유지보수성이 향상됩니다.
-   **데이터베이스 독립성**: 데이터베이스 종류(PostgreSQL, MySQL 등)가 변경되어도 코드 수정 없이 쉽게 전환할 수 있습니다.
-   **객체 지향적 접근**: 데이터베이스 테이블을 파이썬 클래스로, 레코드를 객체로 다루어 객체 지향 프로그래밍의 장점을 활용할 수 있습니다.

### 1.2. SQLAlchemy ORM의 특징
SQLAlchemy는 파이썬에서 가장 널리 사용되는 ORM 중 하나로, 다음과 같은 특징을 가집니다.
-   **유연성**: ORM 기능 외에 Raw SQL을 직접 실행하거나, SQL Expression Language를 사용하여 SQL을 파이썬 코드로 작성하는 등 다양한 수준의 데이터베이스 접근을 지원합니다.
-   **비동기 지원**: SQLAlchemy 1.4부터 비동기(async/await) 데이터베이스 연동을 지원하여 FastAPI와 같은 비동기 프레임워크와 잘 어울립니다.
-   **강력한 쿼리 빌더**: 복잡한 쿼리를 파이썬 코드로 직관적으로 작성할 수 있습니다.

## 2. SQLAlchemy ORM 모델 정의

SQLAlchemy ORM에서 데이터베이스 테이블은 파이썬 클래스로 정의됩니다. 이를 '모델(Model)'이라고 부릅니다.

### 2.1. DeclarativeBase와 Base
SQLAlchemy 2.0 스타일에서는 `DeclarativeBase`를 상속받는 `Base` 클래스를 정의하여 모든 ORM 모델의 기반으로 사용합니다.

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

### 2.2. 테이블 이름과 컬럼 정의
모델 클래스는 `__tablename__` 속성으로 데이터베이스 테이블 이름을 지정하고, `Mapped`와 `mapped_column`을 사용하여 컬럼을 정의합니다.

```python
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

class User(Base):
    __tablename__ = "users" # 데이터베이스 테이블 이름

    id: Mapped[int] = mapped_column(Integer, primary_key=True) # 기본 키
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True) # 문자열, 고유, 인덱스
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
```

### 2.3. 필드 타입과 옵션
`mapped_column` 함수는 SQLAlchemy의 컬럼 타입을 지정하고, `primary_key`, `unique`, `index`, `nullable` 등 다양한 옵션을 설정할 수 있습니다.

-   **`Integer`**: 정수 타입
-   **`String`**: 문자열 타입 (최대 길이 지정 가능)
-   **`Boolean`**: 불리언 타입
-   **`DateTime`**: 날짜 및 시간 타입
-   **`ForeignKey`**: 다른 테이블과의 관계 정의

```python
from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # ForeignKey 관계 정의
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    # relationship: User 모델과의 관계를 파이썬 객체로 접근할 수 있게 함
    author: Mapped["User"] = relationship("User", backref="posts")
```

### 2.4. 관계 정의 (Relationships)
SQLAlchemy ORM은 `relationship()` 함수를 사용하여 모델 간의 관계(일대다, 다대다 등)를 정의합니다.

-   **일대다 (One-to-Many)**: `ForeignKey`와 `relationship`을 함께 사용합니다.
    *   `Post` 모델의 `user_id`는 `User` 모델의 `id`를 참조하는 `ForeignKey`입니다.
    *   `Post` 모델의 `author` 필드는 `User` 모델과의 관계를 나타내며, `User` 모델에서는 `backref="posts"`를 통해 해당 사용자가 작성한 모든 게시글에 `user.posts`로 접근할 수 있습니다.

-   **다대다 (Many-to-Many)**: 중간 테이블을 정의하고 `relationship`에서 `secondary` 인자를 사용합니다.

## 3. 세션(Session) 관리

SQLAlchemy에서 세션(Session)은 데이터베이스와의 모든 대화를 관리하는 핵심 객체입니다. 데이터베이스에 대한 모든 쿼리, 객체 생성, 수정, 삭제 작업은 세션을 통해 이루어집니다.

### 3.1. 세션의 역할
-   **트랜잭션 관리**: 세션은 기본적으로 하나의 트랜잭션을 나타냅니다. `session.commit()`을 호출하면 변경사항이 데이터베이스에 영구적으로 저장되고, `session.rollback()`을 호출하면 모든 변경사항이 취소됩니다.
-   **객체 상태 관리**: 세션은 로드된 객체들의 상태를 추적하고, 변경사항을 감지하여 데이터베이스에 반영합니다.
-   **데이터베이스 연결 관리**: 세션은 데이터베이스 연결을 획득하고 해제하는 역할을 합니다.

### 3.2. 비동기 세션 생성 및 사용
FastAPI와 같은 비동기 환경에서는 비동기 세션을 사용해야 합니다. `async_sessionmaker`를 통해 세션 팩토리를 생성하고, `async with` 구문을 사용하여 세션을 안전하게 관리합니다.

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

# AsyncSessionLocal은 세션 팩토리
AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine, # 데이터베이스 엔진
    class_=AsyncSession, # 비동기 세션 클래스
)

# 세션 사용 예시
async def get_user_by_id(user_id: int):
    async with AsyncSessionLocal() as session:
        # 쿼리 실행
        result = await session.execute(select(User).filter(User.id == user_id))
        user = result.scalar_one_or_none()
        return user
```

## 4. 기본 CRUD(Create, Read, Update, Delete) 작업

SQLAlchemy ORM을 사용하여 데이터베이스에서 객체를 생성, 조회, 수정, 삭제하는 기본 CRUD 작업을 수행합니다.

### 4.1. 데이터 생성 (Create)
새로운 객체를 생성하고 세션에 추가한 뒤 커밋합니다.

```python
from sqlalchemy.ext.asyncio import AsyncSession
from .models import User # User 모델 임포트

async def create_new_user(session: AsyncSession, username: str, email: str, password: str):
    new_user = User(username=username, email=email, hashed_password=password)
    session.add(new_user) # 세션에 객체 추가
    await session.commit() # 데이터베이스에 저장
    await session.refresh(new_user) # 데이터베이스에서 최신 상태로 객체 갱신 (id 등)
    return new_user
```

### 4.2. 데이터 조회 (Read)
`session.execute(select(Model))`을 사용하여 쿼리를 실행하고 결과를 가져옵니다.

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from .models import User

async def get_users(session: AsyncSession, skip: int = 0, limit: int = 100) -> List[User]:
    result = await session.execute(select(User).offset(skip).limit(limit))
    users = result.scalars().all() # 결과에서 모델 객체만 추출
    return users

async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    result = await session.execute(select(User).filter(User.username == username))
    user = result.scalar_one_or_none() # 결과가 하나이거나 없을 때 사용
    return user
```

### 4.3. 데이터 수정 (Update)
조회한 객체의 속성을 변경하고 세션을 커밋합니다.

```python
async def update_user_email(session: AsyncSession, user_id: int, new_email: str) -> User | None:
    user = await session.get(User, user_id) # 기본 키로 객체 조회
    if user:
        user.email = new_email
        await session.commit()
        await session.refresh(user)
    return user
```

### 4.4. 데이터 삭제 (Delete)
조회한 객체를 세션에서 삭제하고 커밋합니다.

```python
async def delete_user_by_id(session: AsyncSession, user_id: int) -> bool:
    user = await session.get(User, user_id)
    if user:
        await session.delete(user) # 세션에서 객체 삭제
        await session.commit() # 데이터베이스에 반영
        return True
    return False
```

## 5. FastAPI와 SQLAlchemy ORM 연동 예시

`04_데이터베이스_설정_및_비동기_DB.md` 문서에서 다룬 내용을 바탕으로, FastAPI와 SQLAlchemy ORM을 연동하는 전체적인 프로젝트 구조와 코드 예시를 제공합니다.

### 5.1. 프로젝트 구조
```
.
├── main.py
├── database.py
├── models.py
└── schemas.py
```

### 5.2. `database.py` (DB 설정 및 세션)
데이터베이스 연결 설정과 `get_db` 의존성 주입 함수를 정의합니다.

```python
# database.py
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/test_db")

class Base(DeclarativeBase):
    pass

engine = create_async_engine(DATABASE_URL, echo=True)

AsyncSessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession,
)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

### 5.3. `models.py` (ORM 모델 정의)
SQLAlchemy ORM 모델을 정의합니다.

```python
# models.py
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base # database.py에서 Base 임포트

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"
```

### 5.4. `schemas.py` (Pydantic 스키마 정의)
API 요청/응답을 위한 Pydantic 스키마를 정의합니다.

```python
# schemas.py
from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True # SQLAlchemy 모델을 Pydantic 모델로 변환 시 필요
```

### 5.5. `main.py` (FastAPI 엔드포인트)
FastAPI 애플리케이션의 메인 파일로, 엔드포인트를 정의하고 DB 연동 로직을 포함합니다.

```python
# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from .database import get_db, init_db
from . import models, schemas

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    await init_db()

@app.post("/users/", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    hashed_password = user.password + "notreallyhashed" # 실제로는 bcrypt 등 사용
    db_user = models.User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

@app.get("/users/", response_model=List[schemas.UserResponse])
async def read_users(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(models.User).offset(skip).limit(limit))
    users = result.scalars().all()
    return users

@app.get("/users/{user_id}", response_model=schemas.UserResponse)
async def read_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(models.User).filter(models.User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/{user_id}", response_model=schemas.UserResponse)
async def update_user(user_id: int, user: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(models.User).filter(models.User.id == user_id))
    db_user = result.scalar_one_or_none()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    db_user.username = user.username
    db_user.email = user.email
    db_user.hashed_password = user.password + "notreallyhashed" # 실제로는 bcrypt 등 사용
    
    await db.commit()
    await db.refresh(db_user)
    return db_user

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(models.User).filter(models.User.id == user_id))
    db_user = result.scalar_one_or_none()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    await db.delete(db_user)
    await db.commit()
    return
```