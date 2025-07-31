# 🧩 Python 데이터베이스 연동 심화 가이드 (Day 22)

> 이 문서는 파이썬을 사용하여 데이터베이스와 상호작용하는 두 가지 주요 방법, 즉 **직접 드라이버(pymysql) 사용**과 **ORM(SQLAlchemy) 활용**에 대해 심도 있게 다룹니다. 각 방식의 개념부터 실용적인 예제, 그리고 성능 최적화 팁까지 포괄적으로 정리하여, 견고하고 효율적인 데이터베이스 애플리케이션 개발 역량을 보여주는 것을 목표로 합니다.

---

## 목차

1.  [**DB-API 드라이버를 이용한 직접 연결: `pymysql`**](#1-db-api-드라이버를-이용한-직접-연결-pymysql)
    -   [1.1. 기본 연결 및 쿼리 실행](#11-기본-연결-및-쿼리-실행)
    -   [1.2. (중요) SQL Injection 방지를 위한 파라미터화](#12-중요-sql-injection-방지를-위한-파라미터화)
2.  [**ORM을 이용한 객체 지향적 접근: `SQLAlchemy`**](#2-orm을-이용한-객체-지향적-접근-sqlalchemy)
    -   [2.1. ORM(Object-Relational Mapping)이란?](#21-ormobject-relational-mapping이란)
    -   [2.2. 기본 설정 및 세션 관리](#22-기본-설정-및-세션-관리)
    -   [2.3. CRUD 작업 예제](#23-crud-작업-예제)
3.  [**성능 최적화: 커넥션 풀 (Connection Pool)**](#3-성능-최적화-커넥션-풀-connection-pool)
    -   [3.1. 커넥션 풀의 필요성](#31-커넥션-풀의-필요성)
    -   [3.2. SQLAlchemy 커넥션 풀 설정](#32-sqlalchemy-커넥션-풀-설정)
4.  [**심화: 파이썬 데이터 모델과 ORM**](#4-심화-파이썬-데이터-모델과-orm)
    -   [4.1. 연산자 오버로딩 (Operator Overloading)](#41-연산자-오버로딩-operator-overloading)
5.  [**결론: `pymysql` vs `SQLAlchemy`**](#5-결론-pymysql-vs-sqlalchemy)

---

## 1. DB-API 드라이버를 이용한 직접 연결: `pymysql`

`pymysql`은 파이썬의 데이터베이스 표준 API(DB-API v2) 명세를 따르는 MySQL 드라이버입니다. 이를 통해 개발자는 SQL 쿼리를 문자열 형태로 직접 작성하여 데이터베이스에 전달하고 결과를 받아옵니다.

### 1.1. 기본 연결 및 쿼리 실행

가장 기본적인 데이터 조회 방법입니다. `try...finally` 구문을 사용하여 어떤 상황에서든 데이터베이스 연결이 안전하게 종료되도록 보장하는 것이 중요합니다.

```python
import pymysql

# 1. 데이터베이스 연결 정보
conn = pymysql.connect(
    host='localhost',
    user='user',
    password='password',
    db='test_db',
    charset='utf8mb4'
)

try:
    # 2. 커서(Cursor) 생성: SQL 문을 실행하고 결과를 탐색하는 객체
    with conn.cursor() as cursor:
        # 3. 실행할 SQL 문 정의
        sql = "SELECT id, name FROM users"
        cursor.execute(sql)

        # 4. 결과 가져오기
        rows = cursor.fetchall() # 모든 결과를 튜플의 리스트로 반환
        for row in rows:
            print(f"ID: {row[0]}, Name: {row[1]}")
finally:
    # 5. 데이터베이스 연결 종료
    conn.close()
```

### 1.2. (중요) SQL Injection 방지를 위한 파라미터화

사용자 입력을 받아 쿼리를 구성할 때, 문자열 포맷팅(`f-string` 등)을 사용하면 악의적인 SQL 구문이 주입될 수 있는 **SQL Injection** 공격에 매우 취약합니다. `execute()` 메서드의 두 번째 인자로 파라미터를 전달하여 이 문제를 해결해야 합니다.

```python
# 나쁜 예시: SQL Injection에 취약
user_id = "1 OR 1=1"
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# 좋은 예시: 파라미터화된 쿼리 사용
user_id_to_find = 1
sql = "SELECT * FROM users WHERE id = %s"
cursor.execute(sql, (user_id_to_find,))
user = cursor.fetchone()
```

---

## 2. ORM을 이용한 객체 지향적 접근: `SQLAlchemy`

### 2.1. ORM(Object-Relational Mapping)이란?

ORM은 **객체(Object)와 관계형 데이터베이스(Relational Database)의 데이터를 자동으로 매핑(연결)**해주는 기술입니다. 개발자는 순수 SQL 쿼리 대신, 익숙한 프로그래밍 언어의 객체와 메서드를 통해 데이터베이스를 조작할 수 있습니다.

-   **장점**: 생산성 향상, 유지보수 용이, 특정 데이터베이스에 대한 종속성 감소
-   **단점**: 복잡하고 세밀한 튜닝이 필요한 쿼리 작성의 어려움, ORM 자체의 학습 곡선

### 2.2. 기본 설정 및 세션 관리

SQLAlchemy는 다음과 같은 핵심 구성요소로 동작합니다.

-   **Engine**: 데이터베이스 연결을 관리하는 핵심 인터페이스 (커넥션 풀 포함).
-   **Declarative Base**: 매핑될 클래스들이 상속받는 기본 클래스.
-   **Session**: ORM 작업의 단위. 객체의 변경 사항을 추적하고, 트랜잭션을 관리합니다.

```python
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Engine 생성: "데이터베이스종류+드라이버://사용자:비밀번호@호스트/DB이름"
engine = create_engine("mysql+pymysql://user:password@localhost/test_db", echo=True)

# 2. 매핑 클래스의 기본이 될 Base 클래스 생성
Base = declarative_base()

# 3. 테이블과 매핑될 User 클래스 정의
class User(Base):
    __tablename__ = 'users' # 실제 데이터베이스 테이블 이름
    id = Column(Integer, primary_key=True)
    name = Column(String(50))

# 4. 정의된 클래스 정보를 바탕으로 실제 테이블 생성
Base.metadata.create_all(engine)

# 5. 데이터베이스와 통신하는 세션 생성
Session = sessionmaker(bind=engine)
session = Session()
```

### 2.3. CRUD 작업 예제

ORM을 사용하면 SQL 쿼리 없이 객체지향적으로 데이터를 다룰 수 있습니다.

```python
# C: Create (생성)
user1 = User(name='이몽룡')
user2 = User(name='성춘향')
session.add_all([user1, user2])
session.commit() # 변경사항을 데이터베이스에 최종 반영

# R: Read (조회)
all_users = session.query(User).all()
for user in all_users:
    print(f"ID: {user.id}, Name: {user.name}")

# U: Update (수정)
user_to_update = session.query(User).filter_by(name='이몽룡').first()
if user_to_update:
    user_to_update.name = '변학도'
    session.commit()

# D: Delete (삭제)
user_to_delete = session.query(User).filter_by(name='성춘향').first()
if user_to_delete:
    session.delete(user_to_delete)
    session.commit()
```

---

## 3. 성능 최적화: 커넥션 풀 (Connection Pool)

### 3.1. 커넥션 풀의 필요성

웹 애플리케이션과 같이 다수의 요청을 동시에 처리해야 하는 환경에서는, 매 요청마다 데이터베이스 연결을 새로 생성하고 해제하는 것은 큰 비용을 유발합니다. **커넥션 풀**은 미리 일정 개수의 연결(Connection)을 만들어두고, 필요할 때마다 빌려주고 반납받는 방식으로 이러한 오버헤드를 크게 줄여 성능을 향상시킵니다.

### 3.2. SQLAlchemy 커넥션 풀 설정

`create_engine` 시점에 다양한 옵션으로 커넥션 풀의 동작을 세밀하게 제어할 수 있습니다.

```python
engine = create_engine(
    "mysql+pymysql://user:password@localhost/test_db",
    pool_size=5,          # 풀에 유지할 최소한의 커넥션 개수 (기본값 5)
    max_overflow=10,      # pool_size를 초과하여 추가로 생성할 수 있는 최대 커넥션 수
    pool_timeout=15,      # 풀에서 커넥션을 얻기 위해 대기할 최대 시간 (초)
    pool_recycle=3600     # 커넥션을 재활용할 시간 (초). DB 서버의 타임아웃보다 짧게 설정
)
```

| 옵션 | 설명 |
| :--- | :--- |
| `pool_size` | 풀이 비어있을 때에도 유지하는 최소 연결 수. 동시 요청이 많은 서비스일수록 이 값을 늘리는 것을 고려합니다. |
| `max_overflow` | `pool_size` 만큼의 연결이 모두 사용 중일 때, 추가로 생성할 수 있는 임시 연결의 최대 개수입니다. |
| `pool_timeout` | 모든 연결이 사용 중일 때, 새로운 연결 요청이 반환을 기다릴 최대 시간입니다. 이 시간을 초과하면 오류가 발생합니다. |
| `pool_recycle` | 설정된 시간이 지난 연결을 자동으로 해제하고 다시 연결합니다. MySQL의 `wait_timeout` 설정으로 인해 연결이 끊어지는 것을 방지하기 위해 중요합니다. |

---

## 4. 심화: 파이썬 데이터 모델과 ORM

### 4.1. 연산자 오버로딩 (Operator Overloading)

클래스에 `__add__`, `__mul__`과 같은 특별한 **매직 메서드**를 구현하면, `+`, `*`와 같은 파이썬의 기본 연산자를 사용자 정의 객체에 맞게 재정의할 수 있습니다. 이를 **연산자 오버로딩**이라 합니다.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(2, 3)
v2 = Vector(1, 5)

print("덧셈:", v1 + v2)    # Vector(3, 8)
print("뺄셈:", v1 - v2)    # Vector(1, -2)
print("스칼라곱:", v1 * 3) # Vector(6, 9)
```

> 💡 **ORM과의 관계**: SQLAlchemy가 `session.query(User).filter(User.id > 5)` 와 같은 직관적인 표현을 가능하게 하는 것도 내부적으로는 이러한 연산자 오버로딩을 활용하여 파이썬 표현식을 SQL 구문으로 변환하기 때문입니다.

---

## 5. 결론: `pymysql` vs `SQLAlchemy`

| 구분 | `pymysql` (DB-API Driver) | `SQLAlchemy` (ORM) |
| :--- | :--- | :--- |
| **장점** | - 가볍고 빠름<br>- SQL을 완벽하게 제어 가능 | - 생산성 높음 (SQL 작성 불필요)<br>- 데이터베이스 비종속적 코드<br>- 유지보수 용이 | 
| **단점** | - SQL Injection에 직접 대비해야 함<br>- DB 변경 시 모든 SQL 수정 필요<br>- 반복적인 코드 작성 | - ORM 자체의 학습 곡선<br>- 복잡한 쿼리는 성능 저하 가능성<br>- 내부 동작이 추상화되어 있음 | 
| **사용 시점** | - 간단한 스크립트<br>- 성능이 극도로 중요한 경우<br>- 복잡한 SQL 튜닝이 필요한 경우 | - 대부분의 웹 애플리케이션<br>- 빠른 프로토타이핑<br>- 유지보수가 중요한 대규모 프로젝트 | 

---
[⏮️ 이전 문서](./0521_SQL정리.md) | [다음 문서 ⏭️](./0528_SQLAlchemy정리.md)