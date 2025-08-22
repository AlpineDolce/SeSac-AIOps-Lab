<h2>FastAPI 학습 가이드: 요청 본문 및 Pydantic - 데이터 유효성 검사 및 직렬화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 FastAPI에서 POST, PUT 요청 등 HTTP 요청의 본문(Request Body)을 다루는 방법을 학습하고, Pydantic 라이브러리를 활용하여 데이터 유효성 검사(Validation) 및 직렬화(Serialization)를 수행하는 것을 목표로 합니다. 이를 통해 API의 데이터 신뢰성을 높이고, 자동 문서화 기능을 효과적으로 활용하는 능력을 기릅니다.

<h2>목차</h2>

- [1. 요청 본문 (Request Body)](#1-요청-본문-request-body)
  - [1.1. 기본 사용법](#11-기본-사용법)
  - [1.2. Pydantic 모델 정의](#12-pydantic-모델-정의)
- [2. Pydantic을 이용한 데이터 유효성 검사](#2-pydantic을-이용한-데이터-유효성-검사)
  - [2.1. 필드 유효성 검사 (Field Validation)](#21-필드-유효성-검사-field-validation)
    - [필수 필드와 선택 필드](#필수-필드와-선택-필드)
    - [기본값 설정](#기본값-설정)
    - [유효성 검사 추가 (Field, Query, Path)](#유효성-검사-추가-field-query-path)
  - [2.2. 데이터 타입 검사](#22-데이터-타입-검사)
  - [2.3. 중첩된 Pydantic 모델](#23-중첩된-pydantic-모델)
- [3. 요청 본문과 쿼리/경로 파라미터 조합](#3-요청-본문과-쿼리경로-파라미터-조합)
- [4. Pydantic을 이용한 응답 모델 (Response Model)](#4-pydantic을-이용한-응답-모델-response-model)
  - [4.1. 응답 데이터 필터링](#41-응답-데이터-필터링)
  - [4.2. 응답 데이터 변환](#42-응답-데이터-변환)

---

## 1. 요청 본문 (Request Body)

FastAPI에서 클라이언트가 서버로 데이터를 보낼 때, URL 파라미터(경로 파라미터, 쿼리 파라미터) 외에 HTTP 요청의 본문(Body)에 데이터를 포함하여 보낼 수 있습니다. 주로 `POST`, `PUT`, `PATCH` 요청에서 사용되며, 대량의 데이터나 민감한 데이터를 전송할 때 적합합니다.

### 1.1. 기본 사용법
경로 작업 함수의 인자로 Pydantic 모델을 선언하면, FastAPI는 자동으로 해당 인자를 요청 본문으로 인식하고, 클라이언트로부터 받은 JSON 데이터를 Pydantic 모델 인스턴스로 변환합니다.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Pydantic 모델 정의
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
    # item은 Pydantic 모델 인스턴스
    return item
```
클라이언트가 위 API로 `POST` 요청을 보낼 때, 요청 본문에 다음과 같은 JSON 데이터를 포함할 수 있습니다.
```json
{
  "name": "Book",
  "description": "A very nice book",
  "price": 12.99,
  "tax": 1.04
}
```
FastAPI는 이 JSON 데이터를 `Item` 모델의 인스턴스로 자동 변환하여 `item` 인자로 전달합니다.

### 1.2. Pydantic 모델 정의
Pydantic은 파이썬의 타입 힌트를 기반으로 데이터 유효성 검사와 직렬화를 제공하는 라이브러리입니다. `BaseModel`을 상속받아 모델을 정의하며, 각 필드에 타입 힌트를 지정합니다.

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    age: int | None = None # 선택적 필드
    is_active: bool = True # 기본값 설정
```

## 2. Pydantic을 이용한 데이터 유효성 검사

FastAPI는 Pydantic을 통해 강력한 데이터 유효성 검사 기능을 제공합니다. 클라이언트가 보낸 데이터가 Pydantic 모델의 스키마에 맞지 않으면, FastAPI는 자동으로 `422 Unprocessable Entity` 에러를 반환하고, 에러 메시지에 어떤 필드가 어떤 이유로 유효하지 않은지 상세하게 알려줍니다.

### 2.1. 필드 유효성 검사 (Field Validation)
Pydantic 모델의 필드에 `Field` 함수를 사용하여 추가적인 유효성 검사 규칙을 정의할 수 있습니다.

#### 필수 필드와 선택 필드
-   **필수 필드**: 타입 힌트만 지정하면 기본적으로 필수 필드가 됩니다.
-   **선택 필드**: `Optional` 또는 `| None`을 사용하고 기본값을 `None`으로 설정합니다.

```python
from typing import Optional
from pydantic import BaseModel

class Product(BaseModel):
    name: str # 필수
    description: Optional[str] = None # 선택적, 기본값 None
    price: float # 필수
    quantity: int = 1 # 필수, 기본값 1
```

#### 기본값 설정
필드에 기본값을 할당하면 해당 필드가 요청 본문에 포함되지 않았을 때 기본값이 사용됩니다.

```python
class Order(BaseModel):
    item_id: int
    quantity: int = 1 # 기본값 1
    status: str = "pending" # 기본값 "pending"
```

#### 유효성 검사 추가 (Field, Query, Path)
`Field`, `Query`, `Path` 함수를 사용하여 필드에 대한 추가적인 유효성 검사 규칙을 정의할 수 있습니다.

```python
from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    name: str = Field(min_length=3, max_length=50, description="아이템 이름")
    description: str | None = Field(default=None, min_length=10, max_length=300)
    price: float = Field(gt=0, description="가격은 0보다 커야 합니다.") # gt: greater than
    tax: float | None = Field(default=None, le=10.5) # le: less than or equal

@app.post("/items/{item_id}")
async def create_item(
    item_id: int = Path(ge=1, description="아이템 ID는 1 이상이어야 합니다."), # ge: greater than or equal
    q: str | None = Query(default=None, max_length=50),
    item: Item = Body(examples={ # 자동 문서화를 위한 예시 추가
        "normal": {
            "summary": "정상적인 아이템",
            "value": {
                "name": "책",
                "price": 12.99
            }
        },
        "long_description": {
            "summary": "긴 설명이 있는 아이템",
            "value": {
                "name": "노트북",
                "description": "매우 긴 설명이 여기에 들어갑니다. 최소 10자 이상이어야 합니다.",
                "price": 1200.00,
                "tax": 9.5
            }
        }
    })
):
    results = {"item_id": item_id, "item": item.dict()}
    if q:
        results.update({"q": q})
    return results
```

### 2.2. 데이터 타입 검사
Pydantic은 파이썬의 표준 타입 힌트를 기반으로 강력한 데이터 타입 검사를 수행합니다. `int`, `float`, `str`, `bool`, `list`, `dict` 등 기본 타입은 물론, `datetime`, `UUID` 등 복잡한 타입도 자동으로 변환하고 검증합니다.

```python
from datetime import datetime
from uuid import UUID

class Event(BaseModel):
    event_id: UUID
    name: str
    event_date: datetime
    attendees: list[str] # 문자열 리스트

# 유효하지 않은 데이터가 들어오면 422 에러 발생
# {
#   "event_id": "invalid-uuid",
#   "name": "Test Event",
#   "event_date": "not-a-date",
#   "attendees": ["Alice", 123]
# }
```

### 2.3. 중첩된 Pydantic 모델
요청 본문이 복잡한 계층 구조를 가질 때, Pydantic 모델을 중첩하여 정의할 수 있습니다. 이는 API 스키마를 명확하게 정의하고, 복잡한 JSON 데이터를 쉽게 다룰 수 있도록 합니다.

```python
class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class UserProfile(BaseModel):
    username: str
    email: str
    address: Address # Address 모델을 중첩

@app.post("/profile/")
async def create_profile(profile: UserProfile):
    return profile
```
클라이언트가 다음과 같은 JSON을 보내면 FastAPI는 이를 `UserProfile` 인스턴스로 변환합니다.
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "address": {
    "street": "123 Main St",
    "city": "Anytown",
    "zip_code": "12345"
  }
}
```

## 3. 요청 본문과 쿼리/경로 파라미터 조합

하나의 경로 작업 함수에서 요청 본문, 경로 파라미터, 쿼리 파라미터를 모두 사용할 수 있습니다. FastAPI는 인자의 타입 힌트와 기본값을 보고 어떤 종류의 파라미터인지 자동으로 판단합니다.

-   **경로 파라미터**: 경로에 `{}`로 정의되고, 타입 힌트만 있는 인자.
-   **쿼리 파라미터**: 경로에 없고, 기본값이 있거나 `Query`로 정의된 인자.
-   **요청 본문**: 경로에 없고, 타입 힌트가 Pydantic 모델인 인자.

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None

@app.put("/items/{item_id}")
async def update_item(
    item_id: int, # 경로 파라미터
    item: Item, # 요청 본문
    q: str | None = Query(default=None, max_length=50) # 쿼리 파라미터
):
    results = {"item_id": item_id, **item.dict()}
    if q:
        results.update({"q": q})
    return results
```

## 4. Pydantic을 이용한 응답 모델 (Response Model)

FastAPI의 `response_model` 인자를 사용하면 경로 작업 함수의 응답 스키마를 명시적으로 정의할 수 있습니다. 이는 API 문서화를 정확하게 하고, 실제 반환되는 데이터를 Pydantic 모델에 맞춰 자동으로 필터링하거나 변환하는 강력한 기능입니다.

### 4.1. 응답 데이터 필터링
`response_model`을 사용하면, 경로 작업 함수가 반환하는 데이터 중 `response_model`에 정의된 필드만 클라이언트에게 전송됩니다. 이는 민감한 정보(예: 사용자 비밀번호 해시)가 실수로 노출되는 것을 방지하는 데 매우 유용합니다.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserInDB(BaseModel): # 데이터베이스에 저장된 사용자 모델
    username: str
    email: str
    hashed_password: str # 민감한 정보

class UserPublic(BaseModel): # 클라이언트에게 노출될 사용자 모델
    username: str
    email: str

@app.get("/users/{user_id}", response_model=UserPublic) # 응답 모델을 UserPublic으로 지정
async def get_user(user_id: int):
    # 실제로는 DB에서 사용자 정보를 가져옴
    user_from_db = UserInDB(username="john_doe", email="john@example.com", hashed_password="supersecret_hash")
    
    # 함수는 UserInDB 인스턴스를 반환하지만, FastAPI는 response_model에 따라 UserPublic으로 필터링하여 응답
    return user_from_db
```
위 예시에서 `hashed_password` 필드는 `response_model`인 `UserPublic`에 정의되어 있지 않으므로, 클라이언트에게는 전송되지 않습니다.

### 4.2. 응답 데이터 변환
`response_model`은 단순히 필드를 필터링하는 것을 넘어, 반환되는 데이터를 `response_model`의 타입 힌트에 맞춰 자동으로 변환합니다. 예를 들어, `datetime` 객체를 ISO 8601 형식의 문자열로 변환하거나, `UUID` 객체를 문자열로 변환하는 등의 작업을 자동으로 수행합니다.

```python
from datetime import datetime
from uuid import UUID, uuid4

class EventResponse(BaseModel):
    event_id: UUID
    name: str
    event_date: datetime

@app.get("/events/{event_id}", response_model=EventResponse)
async def get_event(event_id: UUID):
    # 실제로는 DB에서 이벤트 정보를 가져옴
    event_from_db = {
        "event_id": uuid4(),
        "name": "컨퍼런스",
        "event_date": datetime.now()
    }
    return event_from_db
```
`response_model`을 사용하면 API 응답의 일관성과 정확성을 보장하고, 클라이언트 개발자가 API를 더 쉽게 이해하고 사용할 수 있도록 돕습니다.