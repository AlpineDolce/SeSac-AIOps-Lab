# 🧩 SQLAlchemy 심화 가이드: Core와 ORM 완벽 정복 (Day 23)

> 이 문서는 파이썬의 대표적인 데이터베이스 툴킷인 SQLAlchemy를 활용하는 방법을 **Core**와 **ORM**이라는 두 가지 핵심 패러다임을 중심으로 심도 있게 정리한 가이드입니다. 단순한 기능 나열을 넘어, 각 방식의 철학적 차이를 이해하고 상황에 맞는 최적의 기술을 선택할 수 있는 역량을 기르는 것을 목표로 합니다.

---

## 목차

1.  [**SQLAlchemy 소개**](#1-sqlalchemy-소개)
    -   [1.1. 두 가지 패러다임: Core vs ORM](#11-두-가지-패러다임-core-vs-orm)
    -   [1.2. 설치 및 기본 연결 설정](#12-설치-및-기본-연결-설정)
2.  [**Part 1: SQLAlchemy Core - SQL의 힘을 그대로**](#part-1-sqlalchemy-core---sql의-힘을-그대로)
    -   [2.1. 트랜잭션과 쿼리 실행](#21-트랜잭션과-쿼리-실행)
    -   [2.2. 데이터베이스 메타데이터 작업](#22-데이터베이스-메타데이터-작업)
    -   [2.3. SQL 표현식 언어를 이용한 조회 (SELECT)](#23-sql-표현식-언어를-이용한-조회-select)
    -   [2.4. SQL 표현식 언어를 이용한 데이터 조작 (INSERT, UPDATE, DELETE)](#24-sql-표현식-언어를-이용한-데이터-조작-insert-update-delete)
3.  [**Part 2: SQLAlchemy ORM - 객체지향적 데이터 관리**](#part-2-sqlalchemy-orm---객체지향적-데이터-관리)
    -   [3.1. ORM 테이블 메타데이터 정의](#31-orm-테이블-메타데이터-정의)
    -   [3.2. Session: ORM의 작업 단위](#32-session-orm의-작업-단위)
    -   [3.3. ORM을 이용한 데이터 조작 (CRUD)](#33-orm을-이용한-데이터-조작-crud)
    -   [3.4. 관계(Relationship) 설정과 활용](#34-관계relationship-설정과-활용)
    -   [3.5. 관계 로딩(Relationship Loading) 전략](#35-관계-로딩relationship-loading-전략)

---

## 1. SQLAlchemy 소개

### 1.1. 두 가지 패러다임: Core vs ORM

SQLAlchemy는 파이썬에서 데이터베이스를 효율적으로 다룰 수 있도록 도와주는 라이브러리로, 두 가지 주요 접근 방식을 제공합니다.

-   **SQLAlchemy Core**: SQL 표현식 언어(Expression Language)를 사용하여, 파이썬 코드로 SQL 문을 생성하고 실행합니다. SQL에 가깝게 데이터베이스를 제어할 수 있어 유연하고 강력합니다.
-   **SQLAlchemy ORM (Object Relational Mapper)**: 데이터베이스 테이블을 파이썬 클래스 객체에 매핑하여, SQL 쿼리 없이 객체지향적으로 데이터베이스를 다룹니다. 생산성과 유지보수성이 높습니다.

### 1.2. 설치 및 기본 연결 설정

SQLAlchemy와 MySQL을 연동하기 위해 `pymysql` 드라이버를 함께 설치합니다.

```bash
pip install sqlalchemy pymysql
```

연결의 핵심은 **Engine** 객체입니다. Engine은 데이터베이스 서버와의 통신을 관리하는 시작점입니다.

```python
from sqlalchemy import create_engine

# 연결 문자열 형식: "데이터베이스+드라이버://사용자:비밀번호@호스트:포트/DB이름"
# echo=True는 실행되는 모든 SQL을 콘솔에 출력하여 디버깅에 유용합니다.
engine = create_engine("mysql+pymysql://root:password@localhost:3306/testdb", echo=True)
```

---

## Part 1: SQLAlchemy Core - SQL의 힘을 그대로

Core는 SQL의 구조와 문법을 파이썬 코드로 표현하는 데 중점을 둡니다.

### 2.1. 트랜잭션과 쿼리 실행

Core에서 모든 작업은 **Connection** 객체를 통해 이루어지며, 이 연결 내에서 **Transaction**을 관리합니다.

```python
from sqlalchemy import text

# 1. Engine으로부터 Connection을 얻습니다.
with engine.connect() as conn:
    # 2. 트랜잭션 시작 (명시적)
    trans = conn.begin()
    try:
        # 3. SQL 문 실행 (파라미터 바인딩으로 SQL Injection 방지)
        conn.execute(
            text("INSERT INTO users (name) VALUES (:name)"),
            {"name": "Alice"}
        )
        # 4. 성공 시 트랜잭션 커밋
        trans.commit()
    except:
        # 5. 실패 시 트랜잭션 롤백
        trans.rollback()
        raise

    # 6. 간단한 조회 (자동 커밋)
    result = conn.execute(text("SELECT * FROM users"))
    for row in result:
        print(row)
```

### 2.2. 데이터베이스 메타데이터 작업

`MetaData` 객체는 데이터베이스의 스키마 정보(테이블, 컬럼, 제약 조건 등)를 파이썬 객체로 담아두는 컨테이너입니다.

```python
from sqlalchemy import Table, Column, Integer, String, MetaData

metadata_obj = MetaData()

# "users" 테이블을 파이썬 객체로 정의
users_table = Table(
    "users",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("name", String(50), nullable=False),
    Column("email", String(100), unique=True) # 단순 제약 조건 선언
)

# 정의된 메타데이터를 기반으로 실제 DB에 테이블 생성
metadata_obj.create_all(engine)
```

### 2.3. SQL 표현식 언어를 이용한 조회 (SELECT)

문자열 쿼리 대신 파이썬 객체와 연산자를 사용하여 SQL을 생성합니다. 이를 통해 문법 오류를 줄이고, 동적으로 쿼리를 안전하게 조립할 수 있습니다.

#### 기본 SELECT

```python
from sqlalchemy import select

# SELECT * FROM users
stmt = select(users_table)

# 특정 컬럼만 선택: SELECT id, name FROM users
stmt_cols = select(users_table.c.id, users_table.c.name)

with engine.connect() as conn:
    result = conn.execute(stmt_cols)
    for row in result:
        print(f"ID: {row.id}, Name: {row.name}") # 컬럼 이름으로 접근 가능
```

#### WHERE 절

파이썬의 비교 연산자를 사용하여 `WHERE` 절을 구성합니다.

```python
# WHERE users.name = 'Alice'
stmt_where = select(users_table).where(users_table.c.name == "Alice")

# AND, OR, NOT
from sqlalchemy import and_, or_, not_
stmt_and = select(users_table).where(and_(users_table.c.name == "Alice", users_table.c.id > 1))
```

#### ORDER BY, GROUP BY, HAVING

```python
from sqlalchemy import func, desc

# ORDER BY users.name DESC
stmt_order_by = select(users_table).order_by(desc("name"))

# SELECT name, COUNT(id) FROM users GROUP BY name HAVING COUNT(id) > 1
stmt_group_by = (
    select(users_table.c.name, func.count(users_table.c.id).label("user_count"))
    .group_by(users_table.c.name)
    .having(func.count(users_table.c.id) > 1)
)
```

#### 별칭 (Alias)

`table.alias()` 또는 `select_obj.alias()`를 사용하여 테이블이나 서브쿼리에 별칭을 부여할 수 있습니다.

```python
user_a = users_table.alias("user_a")
address_a = addresses_table.alias("address_a")

# SELECT user_a.name, address_a.email_address FROM users AS user_a JOIN addresses AS address_a ON user_a.id = address_a.user_id
stmt_alias = (
    select(user_a.c.name, address_a.c.email_address)
    .join_from(user_a, address_a, user_a.c.id == address_a.c.user_id)
)
```

#### 서브쿼리와 CTE (Common Table Expressions)

-   **서브쿼리**: `subquery()`를 사용하여 `SELECT` 문을 인라인 뷰로 만듭니다.
-   **CTE**: `cte()`를 사용하여 `WITH` 절을 구성합니다. 복잡한 쿼리를 더 읽기 쉽게 만들어줍니다.

```python
# 서브쿼리: SELECT * FROM users WHERE id IN (SELECT id FROM users WHERE name = 'Alice')
subq = select(users_table.c.id).where(users_table.c.name == "Alice").subquery()
stmt_subq = select(users_table).where(users_table.c.id.in_(subq))

# CTE: WITH regional_users AS (SELECT * FROM users WHERE name = 'Alice') SELECT * FROM regional_users
cte_obj = select(users_table).where(users_table.c.name == "Alice").cte("regional_users")
stmt_cte = select(cte_obj)
```

#### 스칼라 서브쿼리 (Scalar Subquery)

하나의 행, 하나의 열만 반환하는 서브쿼리로, `SELECT` 절 등에서 단일 값처럼 사용할 수 있습니다.

```python
# SELECT name, (SELECT COUNT(addresses.id) FROM addresses WHERE addresses.user_id = users.id) AS address_count FROM users
scalar_subq = (
    select(func.count(addresses_table.c.id))
    .where(addresses_table.c.user_id == users_table.c.id)
    .scalar_subquery()
)

stmt_scalar = select(users_table.c.name, scalar_subq.label("address_count"))
```

#### UNION, UNION ALL

두 개 이상의 `SELECT` 문의 결과를 결합합니다.

```python
from sqlalchemy import union_all

stmt1 = select(users_table).where(users_table.c.name == "Alice")
stmt2 = select(users_table).where(users_table.c.name == "Robert")

# UNION ALL
union_stmt = union_all(stmt1, stmt2)
```

#### EXISTS 서브쿼리

서브쿼리의 결과가 존재하는지 여부를 확인하는 데 사용됩니다.

```python
from sqlalchemy import exists

# SELECT name FROM users WHERE EXISTS (SELECT 1 FROM addresses WHERE addresses.user_id = users.id)
stmt_exists = (
    select(users_table.c.name)
    .where(exists().where(addresses_table.c.user_id == users_table.c.id))
)
```

#### SQL 함수 다루기

`func` 객체를 통해 거의 모든 SQL 함수를 사용할 수 있습니다.

```python
# SELECT COUNT(id) FROM users
stmt_count = select(func.count(users_table.c.id))

# SELECT SUM(amount) FROM orders
# stmt_sum = select(func.sum(orders_table.c.amount))
```

### 2.4. SQL 표현식 언어를 이용한 데이터 조작 (INSERT, UPDATE, DELETE)

```python
from sqlalchemy import insert, update, delete

# INSERT
stmt_insert = insert(users_table).values(name="Bob", email="bob@example.com")

# 여러 값 한 번에 INSERT
params = [{"name": "Charlie", "email": "charlie@example.com"}, {"name": "David", "email": "david@example.com"}]

# UPDATE
stmt_update = update(users_table).where(users_table.c.name == "Bob").values(name="Robert")

# DELETE
stmt_delete = delete(users_table).where(users_table.c.name == "David")

with engine.connect() as conn:
    # INSERT 실행
    result = conn.execute(stmt_insert)
    print(f"Inserted PK: {result.inserted_primary_key}")
    conn.execute(insert(users_table), params)

    # UPDATE 실행
    result = conn.execute(stmt_update)
    print(f"Rows updated: {result.rowcount}")

    # DELETE 실행
    result = conn.execute(stmt_delete)
    print(f"Rows deleted: {result.rowcount}")

    conn.commit() # 잊지 말고 커밋!
```

#### INSERT, UPDATE, DELETE와 함께 RETURNING 사용하기

`returning()` 메서드를 사용하면 INSERT, UPDATE, DELETE 작업 후 영향을 받은 행의 특정 컬럼 값을 반환받을 수 있습니다. (일부 DB 백엔드에서만 지원)

```python
# INSERT 후 새로 생성된 id와 name을 반환
stmt_insert_returning = insert(users_table).values(name="Edward").returning(users_table.c.id, users_table.c.name)

# UPDATE 후 변경된 name을 반환
stmt_update_returning = update(users_table).where(users_table.c.name == "Robert").values(name="Rob").returning(users_table.c.name)

with engine.connect() as conn:
    result = conn.execute(stmt_insert_returning)
    for row in result:
        print(f"Inserted: id={row.id} name={row.name}")
    conn.commit()
```

---

## Part 2: SQLAlchemy ORM - 객체지향적 데이터 관리

ORM은 데이터베이스의 구조와 상호작용을 파이썬 클래스와 객체로 추상화합니다.

### 3.1. ORM 테이블 메타데이터 정의

`declarative_base`를 상속받는 클래스를 만들어 테이블과 매핑합니다.

```python
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    addresses = relationship("Address", back_populates="user", cascade="all, delete-orphan")

class Address(Base):
    __tablename__ = "addresses"
    id = Column(Integer, primary_key=True)
    email_address = Column(String(100), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="addresses")

# 기존 DB 테이블을 ORM 객체로 불러오기 (Reflection)
from sqlalchemy.ext.automap import automap_base

AutoMapBase = automap_base()
AutoMapBase.prepare(autoload_with=engine)
ReflectedUser = AutoMapBase.classes.users
```

### 3.2. Session: ORM의 작업 단위

**Session**은 ORM 객체의 상태를 관리하는 작업 공간입니다. Session에 객체를 추가, 수정, 삭제하면 Session이 해당 변경사항을 추적했다가, `commit()` 시점에まとめて 데이터베이스에 반영합니다.

```python
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()
```

### 3.3. ORM을 이용한 데이터 조작 (CRUD)

```python
# Create
new_user = User(name="Emma", addresses=[Address(email_address="emma@example.com")])
session.add(new_user)
session.commit()

# Read
emma = session.query(User).filter_by(name="Emma").first()
print(emma.name, emma.addresses[0].email_address)

# Update
emma.name = "Emily"
session.commit()

# Delete
session.delete(emma)
session.commit()

# Rollback
# session.rollback()

# Session 종료
session.close()
```

### 3.4. 관계(Relationship) 설정과 활용

`relationship`은 두 ORM 클래스 간의 관계(일대다, 다대일 등)를 정의하여, 객체 탐색을 통해 관련된 다른 객체에 쉽게 접근할 수 있도록 해줍니다. `user.addresses`처럼 말이죠.

-   **`back_populates`**: 양방향 관계를 설정하여, `user.addresses`와 `address.user`가 서로를 참조하고 동기화되도록 합니다.
-   **`cascade`**: 부모 객체에 대한 작업(추가, 삭제 등)이 자식 객체에 어떻게 전파될지를 정의합니다. `all, delete-orphan`은 부모가 저장될 때 자식도 저장되고, 부모로부터 자식이 제거되면 자식 객체도 삭제되도록 하는 강력한 옵션입니다.

### 3.5. 관계 로딩(Relationship Loading) 전략

`user.addresses`에 접근할 때, 관련된 `Address` 객체를 언제, 어떻게 데이터베이스에서 불러올지 결정하는 것은 성능에 큰 영향을 미칩니다.

-   **`select` (Lazy Loading - 기본값)**: `user.addresses`에 실제로 접근하는 시점에 별도의 SELECT 쿼리가 실행됩니다. 간단하지만 N+1 문제를 유발할 수 있습니다.
-   **`joined` (Joined Eager Loading)**: `User`를 조회할 때 `JOIN`을 사용하여 관련된 `Address`를 한 번의 쿼리로 함께 가져옵니다.
-   **`subquery` (Subquery Eager Loading)**: `User`를 조회한 후, 별도의 서브쿼리를 통해 관련된 `Address`를 가져옵니다.
-   **`selectin` (Select IN Eager Loading)**: `User`를 조회한 후, 조회된 `User`의 ID 목록을 사용하여 `WHERE id IN (...)` 형태의 두 번째 쿼리로 `Address`를 가져옵니다. `joined`보다 효율적일 때가 많습니다.

```python
from sqlalchemy.orm import joinedload, selectinload

# Joined Eager Loading 예시
user_with_addresses = session.query(User).options(joinedload(User.addresses)).filter_by(name="Emily").one()

# Select IN Eager Loading 예시
users = session.query(User).options(selectinload(User.addresses)).all()
```

#### 쿼리에서 relationship 사용하기

`relationship`으로 정의된 관계는 `JOIN`의 조건으로도 활용될 수 있어 코드를 더 간결하게 만듭니다.

```python
# Address의 user 관계를 통해 User 테이블과 JOIN하고, User.name으로 필터링
# SELECT addresses.id, addresses.email_address, addresses.user_id 
# FROM addresses JOIN users ON users.id = addresses.user_id 
# WHERE users.name = 'Emily'
addresses_of_emily = session.query(Address).join(Address.user).filter(User.name == "Emily").all()
```
---
[⏮️ 이전 문서](./0527_SQLAlchemy정리.md) | [다음 문서 ⏭️](../../03_Dev/03_CleanCode/01_organize/0523_CleanCode정리.md)