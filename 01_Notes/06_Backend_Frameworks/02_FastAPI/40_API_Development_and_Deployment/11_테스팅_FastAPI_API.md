<h2>FastAPI 학습 가이드: 테스팅 FastAPI API - 견고한 API 구축을 위한 필수 과정</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 FastAPI 애플리케이션의 API를 효과적으로 테스트하는 방법을 학습하는 것을 목표로 합니다. `pytest`와 `TestClient`를 이용한 API 테스트 작성법, 외부 의존성을 제어하기 위한 Mocking 기법, 그리고 테스트 커버리지 측정 방법을 이해하여 견고하고 신뢰할 수 있는 API를 구축하는 능력을 기릅니다.

<h2>목차</h2>

- [1. API 테스팅의 중요성](#1-api-테스팅의-중요성)
  - [1.1. 왜 API를 테스트해야 하는가?](#11-왜-api를-테스트해야-하는가)
  - [1.2. 테스트의 종류](#12-테스트의-종류)
- [2. `pytest`와 `TestClient`를 이용한 API 테스트](#2-pytest와-testclient를-이용한-api-테스트)
  - [2.1. `pytest` 기본 사용법](#21-pytest-기본-사용법)
  - [2.2. `TestClient`를 이용한 API 요청 시뮬레이션](#22-testclient를-이용한-api-요청-시뮬레이션)
  - [2.3. 테스트 데이터 관리 (Fixtures)](#23-테스트-데이터-관리-fixtures)
- [3. Mocking: 외부 의존성 제어](#3-mocking-외부-의존성-제어)
  - [3.1. Mocking의 필요성](#31-mocking의-필요성)
  - [3.2. `unittest.mock.patch` 사용법](#32-unittestmockpatch-사용법)
- [4. 테스트 커버리지 (Test Coverage)](#4-테스트-커버리지-test-coverage)
  - [4.1. 커버리지 측정의 중요성](#41-커버리지-측정의-중요성)
  - [4.2. `coverage.py` 사용법](#42-coveragepy-사용법)
- [5. 실전 API 테스트 예시](#5-실전-api-테스트-예시)
  - [5.1. 프로젝트 구조](#51-프로젝트-구조)
  - [5.2. `main.py` (FastAPI 애플리케이션)](#52-mainpy-fastapi-애플리케이션)
  - [5.3. `test_main.py` (테스트 코드)](#53-testmainpy-테스트-코드)

---

## 1. API 테스팅의 중요성

API(Application Programming Interface)는 애플리케이션의 핵심 로직을 외부에 노출하는 창구입니다. API의 품질은 서비스의 안정성과 신뢰성에 직접적인 영향을 미치므로, API를 개발할 때는 철저한 테스트가 필수적입니다.

### 1.1. 왜 API를 테스트해야 하는가?
-   **기능 검증**: API가 예상대로 동작하고 올바른 응답을 반환하는지 확인합니다.
-   **회귀 방지**: 코드 변경이나 기능 추가 시 기존 API의 오작동을 방지합니다.
-   **데이터 유효성**: API가 올바른 형식의 데이터를 받고 처리하는지, 유효하지 않은 데이터에 대해 적절히 에러를 반환하는지 검증합니다.
-   **성능 및 확장성**: API의 응답 시간과 처리량을 측정하여 성능 병목을 식별합니다.
-   **문서화**: 테스트 코드는 API의 동작 방식에 대한 살아있는 문서 역할을 합니다.

### 1.2. 테스트의 종류
-   **단위 테스트 (Unit Tests)**: API 엔드포인트 내부의 개별 함수나 모듈을 독립적으로 테스트합니다.
-   **통합 테스트 (Integration Tests)**: API 엔드포인트가 데이터베이스, 외부 서비스 등 다른 컴포넌트와 올바르게 상호작용하는지 테스트합니다.
-   **기능 테스트 (Functional Tests)**: 사용자 관점에서 API의 전체 흐름을 테스트합니다.

## 2. `pytest`와 `TestClient`를 이용한 API 테스트

FastAPI는 `pytest`와 `httpx` 기반의 `TestClient`를 사용하여 API를 쉽게 테스트할 수 있도록 지원합니다.

### 2.1. `pytest` 기본 사용법
`pytest`는 파이썬에서 가장 인기 있는 테스트 프레임워크 중 하나입니다. 간결한 문법과 강력한 기능을 제공합니다.

**설치:**
```bash
pip install pytest
```

**테스트 파일 작성:**
테스트 파일은 `test_`로 시작하거나 `_test.py`로 끝나야 합니다. 테스트 함수는 `test_`로 시작해야 합니다.

```python
# test_example.py
def test_addition():
    assert 1 + 1 == 2

def test_string_concat():
    assert "hello" + "world" == "helloworld"
```

**테스트 실행:**
```bash
pytest
```

### 2.2. `TestClient`를 이용한 API 요청 시뮬레이션
`TestClient`는 FastAPI 애플리케이션에 실제 HTTP 요청을 보내는 것처럼 시뮬레이션할 수 있는 동기 클라이언트입니다. 이를 통해 API 엔드포인트의 동작을 쉽게 테스트할 수 있습니다.

**설치:**
`httpx` 라이브러리가 필요합니다.
```bash
pip install httpx
```

**사용법:**
`TestClient`에 FastAPI 애플리케이션 인스턴스를 전달하여 생성합니다. `get()`, `post()`, `put()`, `delete()` 등 HTTP 메서드에 해당하는 메서드를 제공합니다.

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/items/")
async def create_item(item: dict):
    return {"message": "Item created", "item": item}

client = TestClient(app) # TestClient 인스턴스 생성

def test_read_item():
    response = client.get("/items/123") # GET 요청 시뮬레이션
    assert response.status_code == 200
    assert response.json() == {"item_id": 123}

def test_create_item():
    response = client.post("/items/", json={"name": "Test Item", "price": 10.0}) # POST 요청 시뮬레이션
    assert response.status_code == 200
    assert response.json() == {"message": "Item created", "item": {"name": "Test Item", "price": 10.0}}
```

### 2.3. 테스트 데이터 관리 (Fixtures)
`pytest`의 Fixtures는 테스트 함수가 실행되기 전에 필요한 설정이나 데이터를 준비하고, 테스트가 끝난 후 정리하는 데 사용됩니다. 데이터베이스 연결, 테스트용 사용자 생성 등에 유용합니다.

```python
# conftest.py (테스트 루트 디렉토리에 위치)
import pytest
from fastapi.testclient import TestClient
from app.main import app # FastAPI 애플리케이션 임포트

@pytest.fixture(scope="module") # 모듈 내 모든 테스트에서 한 번만 실행
def client():
    with TestClient(app) as c:
        yield c # 테스트 실행 후 클라이언트 종료

@pytest.fixture(scope="function") # 각 테스트 함수마다 실행
def test_user():
    # 테스트용 사용자 생성 (DB에 저장하거나 Mocking)
    user = {"username": "testuser", "email": "test@example.com"}
    yield user
    # 테스트 후 사용자 데이터 정리 (DB에서 삭제 등)
```

## 3. Mocking: 외부 의존성 제어

단위 테스트를 작성할 때, 테스트 대상 코드가 외부 서비스(API), 데이터베이스, 파일 시스템 등과 같은 복잡한 의존성을 가질 수 있습니다. **Mocking**은 이러한 의존성을 가짜(mock) 객체로 대체하여 테스트 대상 코드만 독립적으로, 빠르게, 그리고 일관되게 테스트할 수 있도록 돕는 기법입니다.

### 3.1. Mocking의 필요성
-   **독립성**: 테스트가 외부 요인(네트워크 지연, DB 상태, 외부 API 응답)에 의존하지 않고 독립적으로 실행될 수 있도록 합니다.
-   **속도**: 실제 데이터베이스 접근이나 네트워크 요청 없이 메모리 내에서 빠르게 테스트를 실행할 수 있습니다.
-   **재현성**: 외부 서비스의 상태 변화에 관계없이 테스트 결과를 일관되게 재현할 수 있습니다.
-   **특정 시나리오 테스트**: 오류 발생, 특정 값 반환 등 실제로는 발생하기 어렵거나 제어하기 어려운 시나리오를 쉽게 시뮬레이션할 수 있습니다.

### 3.2. `unittest.mock.patch` 사용법
파이썬의 내장 `unittest.mock` 모듈을 사용하여 Mock 객체를 생성하고 관리할 수 있습니다. `patch` 데코레이터는 특정 객체나 함수의 동작을 임시로 변경하는 데 사용됩니다.

```python
from unittest.mock import patch, MagicMock
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

app = FastAPI()

# 외부 API 호출을 시뮬레이션하는 함수
def get_external_data():
    # 실제로는 외부 API 호출
    return {"data": "real_external_data"}

@app.get("/external_data")
async def read_external_data():
    data = get_external_data()
    return {"message": "Data from external source", "data": data}

client = TestClient(app)

# get_external_data 함수를 Mocking
@patch('main.get_external_data') # Mocking할 함수의 경로
def test_read_external_data_mocked(mock_get_external_data):
    # Mock 객체가 반환할 값 설정
    mock_get_external_data.return_value = {"data": "mocked_data"}

    response = client.get("/external_data")
    assert response.status_code == 200
    assert response.json() == {"message": "Data from external source", "data": {"data": "mocked_data"}}
    mock_get_external_data.assert_called_once() # 함수가 한 번 호출되었는지 확인
```

## 4. 테스트 커버리지 (Test Coverage)

테스트 커버리지는 테스트 코드가 실제 애플리케이션 코드의 몇 퍼센트를 실행하는지 측정하는 지표입니다. 코드의 어떤 부분이 테스트되지 않았는지 시각적으로 보여주어, 테스트를 보강해야 할 부분을 식별하는 데 도움을 줍니다.

### 4.1. 커버리지 측정의 중요성
-   **코드 품질 지표**: 높은 커버리지는 코드의 신뢰도를 높이는 데 기여합니다.
-   **누락된 테스트 식별**: 테스트되지 않은 코드 영역을 시각적으로 보여주어 잠재적인 버그를 줄입니다.
-   **리팩토링 지원**: 코드를 변경할 때, 테스트 커버리지를 통해 기존 기능이 손상되지 않았음을 확인할 수 있습니다.

### 4.2. `coverage.py` 사용법
`coverage.py`는 파이썬 코드의 커버리지를 측정하는 표준 도구입니다.

**설치:**
```bash
pip install coverage
```

**테스트 실행 및 커버리지 측정:**
```bash
coverage run -m pytest
```
이 명령은 `pytest`를 실행하면서 `coverage.py`가 코드 실행을 추적하도록 합니다. 실행 후 `.coverage` 파일이 생성됩니다.

**보고서 생성:**
```bash
coverage report
```
터미널에 커버리지 요약 보고서가 출력됩니다.

```bash
coverage html
```
`htmlcov/` 디렉토리에 상세한 HTML 보고서가 생성됩니다. 이 보고서를 웹 브라우저로 열어 각 파일별 커버리지와 테스트되지 않은 코드 라인을 확인할 수 있습니다.

**실무적 관점:**
-   **목표 설정**: 100% 커버리지가 항상 목표가 될 필요는 없습니다. 중요한 비즈니스 로직, 복잡한 조건 분기, 에러 처리 로직 등 핵심적인 부분에 대한 커버리지를 높이는 것이 더 중요합니다.
-   **CI/CD 통합**: CI/CD 파이프라인에 커버리지 측정을 통합하고, 특정 임계값(예: 80%) 미만일 경우 빌드를 실패하도록 설정하여 코드 품질을 지속적으로 관리합니다.

## 5. 실전 API 테스트 예시

FastAPI 애플리케이션의 `main.py`와 이를 테스트하는 `test_main.py`의 예시를 통해 실제 API 테스트 과정을 살펴봅니다.

### 5.1. 프로젝트 구조
```
.
├── main.py
└── test_main.py
```

### 5.2. `main.py` (FastAPI 애플리케이션)
```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

items_db = {} # 간단한 인메모리 데이터베이스

@app.post("/items/", status_code=201)
async def create_item(item: Item):
    item_id = len(items_db) + 1
    items_db[item_id] = item
    return {"id": item_id, **item.dict()}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"id": item_id, **items_db[item_id].dict()}

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    items_db[item_id] = item
    return {"id": item_id, **item.dict()}

@app.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_id]
    return
```

### 5.3. `test_main.py` (테스트 코드)
```python
# test_main.py
from fastapi.testclient import TestClient
from main import app, items_db # FastAPI 앱과 인메모리 DB 임포트

# TestClient 인스턴스 생성 (모든 테스트에서 재사용)
client = TestClient(app)

# 각 테스트 함수 실행 전에 DB 초기화 (fixture 사용)
def setup_function():
    items_db.clear()

def test_create_item():
    response = client.post("/items/", json={"name": "Test Item", "price": 10.0})
    assert response.status_code == 201
    assert response.json()["name"] == "Test Item"
    assert response.json()["price"] == 10.0
    assert response.json()["id"] == 1
    assert items_db[1].name == "Test Item"

def test_read_item():
    # 먼저 아이템 생성
    client.post("/items/", json={"name": "Read Item", "price": 20.0})
    
    response = client.get("/items/1")
    assert response.status_code == 200
    assert response.json()["name"] == "Read Item"

def test_read_non_existent_item():
    response = client.get("/items/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Item not found"

def test_update_item():
    # 먼저 아이템 생성
    client.post("/items/", json={"name": "Original Item", "price": 30.0})
    
    response = client.put("/items/1", json={"name": "Updated Item", "price": 35.0})
    assert response.status_code == 200
    assert response.json()["name"] == "Updated Item"
    assert items_db[1].name == "Updated Item"

def test_delete_item():
    # 먼저 아이템 생성
    client.post("/items/", json={"name": "Delete Item", "price": 40.0})
    
    response = client.delete("/items/1")
    assert response.status_code == 204
    assert 1 not in items_db # DB에서 삭제되었는지 확인

def test_delete_non_existent_item():
    response = client.delete("/items/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Item not found"
```