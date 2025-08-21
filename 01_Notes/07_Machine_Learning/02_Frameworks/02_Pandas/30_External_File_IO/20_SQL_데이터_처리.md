<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 Pandas를 사용하여 SQL 데이터베이스와 연동하는 방법을 다룹니다. SQL 쿼리를 통해 데이터를 불러오고, DataFrame을 데이터베이스 테이블로 저장하는 방법을 학습합니다. 다양한 데이터베이스 시스템(SQLite, PostgreSQL, MySQL 등)과의 연결 및 기본적인 데이터 입출력 과정을 실습합니다.

<h2>목차</h2>

- [목차](#목차)
- [1. SQL 데이터베이스 연동 개요](#1-sql-데이터베이스-연동-개요)
- [2. 데이터베이스 연결 설정](#2-데이터베이스-연결-설정)
- [3. SQL 쿼리로 데이터 불러오기 (`read_sql_query`)](#3-sql-쿼리로-데이터-불러오기-read_sql_query)
- [4. 테이블 전체 불러오기 (`read_sql_table`)](#4-테이블-전체-불러오기-read_sql_table)
- [5. DataFrame을 SQL 테이블로 저장 (`to_sql`)](#5-dataframe을-sql-테이블로-저장-to_sql)
- [6. 실전 예제: SQLite 데이터베이스 연동](#6-실전-예제-sqlite-데이터베이스-연동)

---

## 1. SQL 데이터베이스 연동 개요

데이터 과학 및 분석 프로젝트에서 데이터는 다양한 형태로 존재하지만, 그 중에서도 **관계형 데이터베이스(RDB)**는 정형 데이터를 저장하고 관리하는 가장 보편적이고 강력한 시스템입니다. 기업의 핵심 비즈니스 데이터, 고객 정보, 거래 내역 등 대부분의 중요한 데이터는 MySQL, PostgreSQL, Oracle, SQL Server와 같은 RDB에 저장되어 있습니다. 따라서 Pandas와 같은 데이터 분석 라이브러리가 이러한 SQL 데이터베이스와 원활하게 연동되는 것은 데이터 과학 워크플로우에서 필수적인 역량입니다.

Pandas는 SQL 데이터베이스와의 연동을 통해 다음과 같은 핵심 기능을 제공합니다:

*   **데이터 불러오기 (Read)**: SQL 쿼리를 실행하거나 특정 테이블의 데이터를 직접 불러와 Pandas `DataFrame` 객체로 변환합니다. 이를 통해 데이터베이스에 저장된 방대한 데이터를 Python 환경으로 가져와 분석, 전처리, 시각화 등의 작업을 수행할 수 있습니다.
*   **데이터 저장하기 (Write)**: Pandas `DataFrame`에 있는 분석 결과나 새로 생성된 데이터를 SQL 데이터베이스의 테이블로 저장합니다. 이는 분석 결과를 영구적으로 보관하거나, 다른 시스템에서 활용할 수 있도록 데이터베이스에 다시 쓰는 데 사용됩니다.

이러한 연동 기능은 데이터 수집부터 분석, 그리고 결과 배포에 이르는 전 과정에서 데이터의 흐름을 원활하게 하고, 데이터 분석 워크플로우를 자동화하며 효율성을 극대화하는 데 기여합니다.

### 필요 라이브러리

Pandas가 SQL 데이터베이스와 상호작용하기 위해서는 몇 가지 추가적인 Python 라이브러리가 필요합니다. 이 라이브러리들은 데이터베이스와의 연결을 설정하고, SQL 쿼리를 실행하며, 데이터를 효율적으로 주고받는 역할을 합니다.

*   **`SQLAlchemy`**:
    *   **역할**: `SQLAlchemy`는 Python SQL 툴킷이자 ORM(Object-Relational Mapping) 라이브러리입니다. Pandas가 다양한 종류의 SQL 데이터베이스(SQLite, PostgreSQL, MySQL, Oracle 등)와 일관된 방식으로 통신할 수 있도록 추상화 계층을 제공합니다. 즉, 데이터베이스 종류에 상관없이 거의 동일한 코드로 데이터를 처리할 수 있게 해주는 핵심 라이브러리입니다.
    *   **설치**: `pip install SQLAlchemy`

*   **데이터베이스 드라이버 (DB-API 2.0 호환)**:
    *   **역할**: `SQLAlchemy`는 범용적인 인터페이스를 제공하지만, 실제 특정 데이터베이스 시스템에 연결하고 통신하기 위해서는 해당 데이터베이스에 특화된 "드라이버" 또는 "어댑터" 라이브러리가 필요합니다. 이 드라이버들은 Python의 표준 데이터베이스 API(DB-API 2.0)를 준수하여 `SQLAlchemy`와 호환됩니다.
    *   **주요 데이터베이스별 드라이버 예시**:
        *   **SQLite**: Python 표준 라이브러리에 `sqlite3` 모듈이 내장되어 있어 별도 설치 없이 바로 사용할 수 있습니다. 파일 기반의 경량 데이터베이스로, 테스트나 소규모 프로젝트에 매우 적합합니다.
        *   **PostgreSQL**: `psycopg2` 라이브러리가 가장 널리 사용되는 드라이버입니다. `pip install psycopg2-binary`로 설치할 수 있습니다.
        *   **MySQL**: `mysql-connector-python` (MySQL 공식 드라이버) 또는 `pymysql` (순수 Python 구현 드라이버) 등이 사용됩니다. `pip install mysql-connector-python` 또는 `pip install pymysql`로 설치할 수 있습니다.
        *   **SQL Server**: `pyodbc` 라이브러리가 주로 사용됩니다. `pip install pyodbc`로 설치할 수 있습니다.

이 문서에서는 별도의 서버 설정 없이 파일 기반으로 쉽게 사용할 수 있으며, Python에 기본 내장되어 있어 추가 설치 부담이 적은 **SQLite** 데이터베이스를 중심으로 Pandas의 SQL 연동 기능을 설명하고 실습 예제를 제공합니다. 다른 데이터베이스 시스템과의 연동도 연결 문자열과 드라이버만 변경하면 기본적인 사용법은 유사합니다.

## 2. 데이터베이스 연결 설정

Pandas가 SQL 데이터베이스와 상호작용하려면 먼저 데이터베이스에 대한 "연결(Connection)"을 설정해야 합니다. 이 연결은 Pandas가 데이터를 주고받을 수 있는 통로 역할을 합니다. `SQLAlchemy` 라이브러리의 `create_engine` 함수는 이러한 연결을 위한 "엔진(Engine)" 객체를 생성하는 핵심적인 도구입니다.

### 2.1. `create_engine` 함수 개요

`create_engine` 함수는 데이터베이스 연결을 위한 `Engine` 객체를 반환합니다. 이 `Engine` 객체는 데이터베이스와의 연결 풀(connection pool)을 관리하고, SQL 쿼리를 실행하며, 트랜잭션을 처리하는 등의 저수준 작업을 담당합니다. Pandas의 `read_sql_query`, `read_sql_table`, `to_sql` 함수들은 이 `Engine` 객체를 `con` (connection) 파라미터로 받아 데이터베이스와 통신합니다.

`create_engine`의 가장 중요한 인수는 **연결 문자열(Connection String)** 또는 **URI(Uniform Resource Identifier)**입니다. 이 문자열은 어떤 종류의 데이터베이스에, 어떤 자격 증명으로, 어디에 위치한 데이터베이스에 연결할 것인지를 정의합니다.

### 2.2. 연결 문자열(Connection String / URI) 형식

연결 문자열의 일반적인 형식은 다음과 같습니다:

`'데이터베이스종류[+드라이버]://사용자명:비밀번호@호스트:포트/데이터베이스이름'`

각 부분의 의미는 다음과 같습니다:
*   **`데이터베이스종류`**: `sqlite`, `postgresql`, `mysql`, `oracle`, `mssql` 등 사용할 데이터베이스의 종류를 지정합니다.
*   **`+드라이버` (선택 사항)**: 특정 데이터베이스에 여러 드라이버가 있을 경우, 사용할 드라이버를 명시합니다 (예: `mysql+pymysql`). 생략하면 `SQLAlchemy`가 기본 드라이버를 선택합니다.
*   **`사용자명:비밀번호`**: 데이터베이스에 로그인할 사용자명과 비밀번호를 지정합니다. (SQLite와 같이 사용자 인증이 필요 없는 경우 생략)
*   **`호스트:포트`**: 데이터베이스 서버가 실행 중인 호스트 주소(IP 주소 또는 도메인 이름)와 포트 번호를 지정합니다. (SQLite와 같이 파일 기반인 경우 생략)
*   **`/데이터베이스이름`**: 연결할 특정 데이터베이스의 이름을 지정합니다.

#### 주요 데이터베이스별 연결 문자열 예시:

*   **SQLite**:
    *   **인메모리(In-memory) 데이터베이스**: `'sqlite:///:memory:'`
        *   메모리 내에 임시 데이터베이스를 생성합니다. 프로그램이 종료되면 모든 데이터가 사라집니다. 테스트나 임시 데이터 처리에 유용합니다.
    *   **파일 기반 데이터베이스**: `'sqlite:///경로/데이터베이스이름.db'`
        *   지정된 경로에 `.db` 파일을 생성하거나 연결합니다. 파일이 존재하지 않으면 새로 생성하고, 존재하면 해당 파일에 연결합니다. 상대 경로 또는 절대 경로를 사용할 수 있습니다.
        *   예: `'sqlite:///my_database.db'` (현재 디렉토리에 `my_database.db` 파일 생성)
*   **PostgreSQL**:
    *   `'postgresql+psycopg2://user:password@host:port/dbname'`
        *   예: `'postgresql+psycopg2://postgres:mypassword@localhost:5432/mydatabase'`
*   **MySQL**:
    *   `'mysql+mysqlconnector://user:password@host:port/dbname'` (mysql-connector-python 드라이버 사용)
    *   `'mysql+pymysql://user:password@host:port/dbname'` (pymysql 드라이버 사용)
        *   예: `'mysql+pymysql://root:mypassword@localhost:3306/mydb'`
*   **SQL Server**:
    *   `'mssql+pyodbc://user:password@host:port/dbname?driver=ODBC+Driver+17+for+SQL+Server'`
        *   드라이버 이름은 시스템에 설치된 ODBC 드라이버에 따라 달라질 수 있습니다.

### 2.3. 데이터베이스 연결 및 테스트 예시 (SQLite 중심)

이 문서에서는 별도의 서버 설정 없이 파일 기반으로 쉽게 사용할 수 있는 **SQLite**를 중심으로 설명합니다. 다음 코드는 SQLite의 인메모리 및 파일 기반 데이터베이스에 연결하고, 간단한 테이블을 생성하여 연결이 성공했는지 테스트하는 과정을 보여줍니다.

```python
import pandas as pd
from sqlalchemy import create_engine, text # text 함수 임포트
import sqlite3 # SQLite 사용을 위해 필요 (선택 사항이지만 명시적으로 임포트)

# 1. SQLite 인메모리 데이터베이스 연결 (테스트용)
# 'sqlite:///:memory:'는 휘발성 인메모리 데이터베이스를 생성합니다.
# 이 데이터베이스는 Python 스크립트가 실행되는 동안에만 존재하며, 스크립트 종료 시 모든 데이터가 사라집니다.
engine_memory = create_engine('sqlite:///:memory:')
print("인메모리 SQLite 엔진 생성 완료.")

# 2. SQLite 파일 기반 데이터베이스 연결 (영구 저장)
# 'sqlite:///my_database.db'는 현재 스크립트가 실행되는 디렉토리에 'my_database.db' 파일을 생성하거나, 
# 이미 존재한다면 해당 파일에 연결합니다. 데이터는 파일에 영구적으로 저장됩니다.
engine_file = create_engine('sqlite:///my_database.db')
print("파일 기반 SQLite 엔진 생성 완료 (my_database.db).")

# 3. 연결 테스트 및 간단한 테이블 생성 (인메모리 DB에)
# with engine.connect() as conn: 구문은 데이터베이스 연결을 열고, 작업이 완료되면 자동으로 닫아줍니다.
# conn.execute(text("..."))는 SQLAlchemy 2.0 스타일의 권장되는 SQL 실행 방식입니다.
with engine_memory.connect() as conn:
    # users 테이블이 없으면 생성합니다.
    conn.execute(text("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"))
    # 데이터 삽입
    conn.execute(text("INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25)"))
    # 변경사항을 데이터베이스에 반영 (커밋)
    conn.commit()
print("인메모리 DB에 users 테이블 생성 및 데이터 삽입 완료.")

# 4. 생성된 테이블에서 데이터 조회하여 확인
# pd.read_sql_query를 사용하여 데이터가 올바르게 삽입되었는지 확인합니다.
# 이 함수는 engine_memory 객체를 con 파라미터로 받습니다.
df_test_users = pd.read_sql_query("SELECT * FROM users", engine_memory)
print("\n연결 테스트: users 테이블 데이터:\n", df_test_users)

# 파일 기반 DB에 대한 추가적인 연결 테스트 (선택 사항)
# 실제 파일에 테이블을 생성하고 싶다면 아래 주석을 해제하고 실행하세요.
# with engine_file.connect() as conn:
#     conn.execute(text("CREATE TABLE IF NOT EXISTS products (product_id INTEGER PRIMARY KEY, product_name TEXT)"))
#     conn.execute(text("INSERT INTO products (product_name) VALUES ('Laptop'), ('Mouse')"))
#     conn.commit()
# print("파일 기반 DB에 products 테이블 생성 및 데이터 삽입 완료.")
# df_test_products = pd.read_sql_query("SELECT * FROM products", engine_file)
# print("\n파일 기반 DB 테스트: products 테이블 데이터:\n", df_test_products)

# 생성된 my_database.db 파일 삭제 (테스트 후 정리용, 필요시 주석 해제)
# import os
# if os.path.exists('my_database.db'):
#     os.remove('my_database.db')
#     print("my_database.db 파일 삭제 완료.")
```

이처럼 `create_engine`을 통해 데이터베이스 연결을 설정하고, 간단한 SQL 명령을 실행하여 연결을 테스트할 수 있습니다. 이제 이 `Engine` 객체를 사용하여 Pandas의 다양한 SQL 연동 함수들을 활용할 준비가 되었습니다.


## 3. SQL 쿼리로 데이터 불러오기 (`read_sql_query`)

Pandas의 `read_sql_query()` 함수는 SQL 데이터베이스에서 데이터를 `DataFrame`으로 불러오는 가장 유연하고 강력한 방법입니다. 이 함수는 사용자가 직접 작성한 SQL 쿼리 문자열을 데이터베이스에 전달하고, 그 쿼리 결과 셋(Result Set)을 Pandas `DataFrame` 형태로 반환합니다. 이를 통해 복잡한 조인(JOIN), 필터링(WHERE), 집계(GROUP BY) 등 다양한 SQL 연산을 수행하여 원하는 형태의 데이터를 정확하게 가져올 수 있습니다.

### 3.1. `read_sql_query()` 함수 개요

`read_sql_query(sql, con, index_col=None, chunksize=None, **kwargs)`

*   **`sql` (필수)**: 데이터베이스에서 실행할 SQL 쿼리 문자열입니다. `SELECT` 문을 사용하여 데이터를 조회합니다. 예를 들어, `"SELECT * FROM users"`, `"SELECT name, age FROM employees WHERE department = 'IT'"` 등이 될 수 있습니다.
*   **`con` (필수)**: 데이터베이스 연결 객체입니다. `SQLAlchemy`의 `Engine` 객체(예: `create_engine`으로 생성한 `engine_memory` 또는 `engine_file`) 또는 DB-API 2.0 호환 연결 객체를 전달합니다.
*   **`index_col` (선택)**: 불러온 데이터로 생성될 `DataFrame`의 인덱스(행 라벨)로 사용할 컬럼의 이름(문자열) 또는 컬럼 이름 리스트를 지정합니다. 기본값은 `None`이며, 이 경우 Pandas는 0부터 시작하는 기본 정수 인덱스를 생성합니다.
*   **`chunksize` (선택)**: 대용량 데이터를 한 번에 메모리에 로드하기 어려울 때, 데이터를 지정된 크기의 "청크(chunk)" 단위로 나누어 불러올 수 있도록 합니다. 이 경우 `read_sql_query`는 `DataFrame`을 직접 반환하는 대신, 각 청크를 `DataFrame`으로 반환하는 이터레이터(iterator)를 반환합니다. 이를 통해 메모리 효율적으로 대용량 데이터를 처리할 수 있습니다.

### 3.2. `read_sql_query()` 활용 예시

이전 섹션에서 생성한 인메모리 SQLite 데이터베이스(`engine_memory`)를 사용하여 `read_sql_query()`의 다양한 활용법을 살펴보겠습니다. 먼저, 예시를 위해 `users` 테이블에 데이터를 추가합니다.

```python
import pandas as pd
from sqlalchemy import create_engine, text
import sqlite3

# 인메모리 SQLite 데이터베이스 연결 (이전 섹션에서 생성된 engine_memory 재사용 또는 새로 생성)
engine_memory = create_engine('sqlite:///:memory:')

# users 테이블 생성 및 데이터 삽입 (예시를 위해 다시 실행)
with engine_memory.connect() as conn:
    conn.execute(text("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT)"))
    conn.execute(text("INSERT INTO users (name, age, city) VALUES ('Alice', 30, 'New York'), ('Bob', 25, 'London'), ('Charlie', 35, 'New York'), ('David', 28, 'Paris')"))
    conn.commit()
print("인메모리 DB에 users 테이블 초기화 및 데이터 삽입 완료.")
```

**1. 모든 데이터 불러오기 (`SELECT *`)**
가장 기본적인 형태로, 테이블의 모든 컬럼과 모든 행을 불러옵니다.

**코드**
```python
df_users_all = pd.read_sql_query("SELECT * FROM users", engine_memory)
print("\n--- 1. read_sql_query로 불러온 모든 users 데이터 ---")
print(df_users_all)
```
**결과:**
```
--- 1. read_sql_query로 불러온 모든 users 데이터 ---
   id     name  age      city
0   1    Alice   30  New York
1   2      Bob   25    London
2   3  Charlie   35  New York
3   4    David   28     Paris
```
**결과 설명**
*   `users` 테이블의 모든 데이터가 `DataFrame`으로 성공적으로 로드되었습니다.

**2. 특정 컬럼만 선택하여 불러오기 (`SELECT column1, column2`)**
필요한 컬럼만 명시적으로 지정하여 불러올 수 있습니다. 이는 메모리 효율성을 높이고 불필요한 데이터를 가져오지 않도록 합니다.

**코드**
```python
df_users_selected = pd.read_sql_query("SELECT name, age FROM users", engine_memory)
print("\n--- 2. read_sql_query로 불러온 특정 컬럼 데이터 (name, age) ---")
print(df_users_selected)
```
**결과:**
```
--- 2. read_sql_query로 불러온 특정 컬럼 데이터 (name, age) ---
      name  age
0    Alice   30
1      Bob   25
2  Charlie   35
3    David   28
```
**결과 설명**
*   `name`과 `age` 컬럼만 선택적으로 불러와 `DataFrame`이 생성되었습니다.

**3. 조건부 쿼리로 데이터 필터링 (`WHERE`)**
`WHERE` 절을 사용하여 특정 조건을 만족하는 행만 불러올 수 있습니다. 이는 데이터 분석의 첫 단계에서 관심 있는 서브셋을 추출할 때 매우 유용합니다.

**코드**
```python
df_users_filtered = pd.read_sql_query("SELECT * FROM users WHERE age > 28", engine_memory)
print("\n--- 3. read_sql_query로 불러온 필터링된 users 데이터 (age > 28) ---")
print(df_users_filtered)
```
**결과:**
```
--- 3. read_sql_query로 불러온 필터링된 users 데이터 (age > 28) ---
   id     name  age      city
0   1    Alice   30  New York
1   3  Charlie   35  New York
```
**결과 설명**
*   나이가 28세 초과인 사용자(`Alice`, `Charlie`)의 데이터만 불러와졌습니다.

**4. 데이터 정렬 (`ORDER BY`)**
`ORDER BY` 절을 사용하여 특정 컬럼을 기준으로 데이터를 정렬하여 불러올 수 있습니다.

**코드**
```python
df_users_sorted = pd.read_sql_query("SELECT * FROM users ORDER BY age DESC", engine_memory)
print("\n--- 4. read_sql_query로 불러온 정렬된 users 데이터 (age 내림차순) ---")
print(df_users_sorted)
```
**결과:**
```
--- 4. read_sql_query로 불러온 정렬된 users 데이터 (age 내림차순) ---
   id     name  age      city
0   3  Charlie   35  New York
1   1    Alice   30  New York
2   4    David   28     Paris
3   2      Bob   25    London
```
**결과 설명**
*   `age` 컬럼을 기준으로 내림차순으로 정렬된 데이터가 로드되었습니다.

**5. 데이터 집계 (`GROUP BY`, `SUM`, `COUNT` 등)**
SQL의 집계 함수와 `GROUP BY` 절을 사용하여 데이터베이스 레벨에서 바로 집계된 결과를 불러올 수 있습니다. 이는 대량의 데이터를 Python으로 모두 가져와서 집계하는 것보다 훨씬 효율적일 수 있습니다.

**코드**
```python
df_users_city_count = pd.read_sql_query("SELECT city, COUNT(*) as user_count FROM users GROUP BY city", engine_memory)
print("\n--- 5. read_sql_query로 불러온 도시별 사용자 수 ---")
print(df_users_city_count)
```
**결과:**
```
--- 5. read_sql_query로 불러온 도시별 사용자 수 ---
       city  user_count
0    London           1
1  New York           2
2     Paris           1
```
**결과 설명**
*   도시별 사용자 수가 SQL 쿼리 내에서 집계되어 `DataFrame`으로 반환되었습니다.

**6. `index_col` 옵션 활용**
불러온 데이터의 특정 컬럼을 `DataFrame`의 인덱스로 바로 지정할 수 있습니다. 이는 데이터 접근을 용이하게 합니다.

**코드**
```python
df_users_indexed = pd.read_sql_query("SELECT id, name, age FROM users", engine_memory, index_col='id')
print("\n--- 6. read_sql_query로 불러온 데이터 (id를 인덱스로) ---")
print(df_users_indexed)
```
**결과:**
```
--- 6. read_sql_query로 불러온 데이터 (id를 인덱스로) ---
        name  age
id               
1      Alice   30
2        Bob   25
3    Charlie   35
4      David   28
```
**결과 설명**
*   `id` 컬럼이 `DataFrame`의 인덱스로 설정되어, `df_users_indexed.loc[1]`과 같이 인덱스를 통해 데이터를 조회할 수 있습니다.

**7. `chunksize` 옵션 활용 (대용량 데이터 처리)**
`chunksize` 옵션은 대용량 데이터를 한 번에 메모리에 로드하는 대신, 지정된 크기의 청크(chunk)로 나누어 불러올 때 사용합니다. 이는 메모리 부족 문제를 방지하고, 데이터를 부분적으로 처리할 수 있게 합니다. `read_sql_query`는 이 경우 `DataFrame` 이터레이터를 반환합니다.

**코드**
```python
# 예시를 위해 더 많은 데이터를 삽입
with engine_memory.connect() as conn:
    conn.execute(text("INSERT INTO users (name, age, city) VALUES ('Eve', 22, 'Tokyo'), ('Frank', 40, 'Berlin'), ('Grace', 29, 'London')"))
    conn.commit()

print("\n--- 7. read_sql_query로 청크 단위 데이터 불러오기 (chunksize=3) ---")
for i, chunk_df in enumerate(pd.read_sql_query("SELECT * FROM users", engine_memory, chunksize=3)):
    print(f"\n--- 청크 {i+1} ---")
    print(chunk_df)
```
**결과:**
```
--- 7. read_sql_query로 청크 단위 데이터 불러오기 (chunksize=3) ---

--- 청크 1 ---
   id     name  age      city
0   1    Alice   30  New York
1   2      Bob   25    London
2   3  Charlie   35  New York

--- 청크 2 ---
   id   name  age    city
0   4  David   28   Paris
1   5    Eve   22   Tokyo
2   6  Frank   40  Berlin

--- 청크 3 ---
   id   name  age    city
0   7  Grace   29  London
```
**결과 설명**
*   `chunksize=3`으로 설정했기 때문에, `read_sql_query`는 3개의 행을 포함하는 `DataFrame` 청크를 순차적으로 반환합니다. 이를 통해 전체 데이터를 한 번에 로드하지 않고도 처리할 수 있습니다.

`read_sql_query()`는 SQL의 모든 강력한 기능을 활용하여 데이터베이스에서 원하는 데이터를 정확하고 효율적으로 불러올 수 있게 해주는 Pandas의 핵심 함수입니다. 복잡한 데이터 추출 및 전처리 작업을 데이터베이스 레벨에서 수행한 후, 그 결과를 Pandas `DataFrame`으로 가져와 추가 분석을 진행할 때 매우 유용합니다.


## 4. 테이블 전체 불러오기 (`read_sql_table`)

Pandas의 `read_sql_table()` 함수는 SQL 데이터베이스 내의 특정 테이블 전체를 `DataFrame`으로 불러오는 데 사용됩니다. 이 함수는 `read_sql_query()`와 달리 SQL 쿼리 문자열을 직접 작성할 필요 없이, 테이블 이름만 지정하면 해당 테이블의 모든 데이터를 가져옵니다. 따라서 테이블의 전체 내용을 빠르게 확인하거나, 복잡한 쿼리 없이 단순히 테이블 데이터를 로드할 때 매우 편리합니다.

### 4.1. `read_sql_table()` 함수 개요

`read_sql_table(table_name, con, schema=None, index_col=None, chunksize=None, **kwargs)`

*   **`table_name` (필수)**: 데이터베이스에서 불러올 테이블의 이름(문자열)입니다. 예를 들어, `"users"`, `"products"` 등이 될 수 있습니다.
*   **`con` (필수)**: 데이터베이스 연결 객체입니다. `SQLAlchemy`의 `Engine` 객체(예: `create_engine`으로 생성한 `engine_memory` 또는 `engine_file`) 또는 DB-API 2.0 호환 연결 객체를 전달합니다.
*   **`schema` (선택)**: 데이터베이스 스키마의 이름(문자열)을 지정합니다. PostgreSQL과 같이 스키마를 사용하는 데이터베이스에서 특정 스키마 내의 테이블을 지정할 때 유용합니다. 기본값은 `None`이며, 이 경우 데이터베이스의 기본 스키마를 사용합니다.
*   **`index_col` (선택)**: 불러온 데이터로 생성될 `DataFrame`의 인덱스(행 라벨)로 사용할 컬럼의 이름(문자열) 또는 컬럼 이름 리스트를 지정합니다. 기본값은 `None`이며, 이 경우 Pandas는 0부터 시작하는 기본 정수 인덱스를 생성합니다.
*   **`chunksize` (선택)**: `read_sql_query()`와 동일하게 대용량 데이터를 청크(chunk) 단위로 나누어 불러올 수 있도록 합니다. 이 경우 `read_sql_table`은 `DataFrame`을 직접 반환하는 대신, 각 청크를 `DataFrame`으로 반환하는 이터레이터(iterator)를 반환합니다.

### 4.2. `read_sql_table()` 활용 예시

이전 섹션에서 생성한 인메모리 SQLite 데이터베이스(`engine_memory`)를 사용하여 `read_sql_table()`의 다양한 활용법을 살펴보겠습니다. 예시를 위해 `users` 테이블에 데이터가 있다고 가정합니다.

```python
import pandas as pd
from sqlalchemy import create_engine, text
import sqlite3

# 인메모리 SQLite 데이터베이스 연결 및 users 테이블 초기화 (예시를 위해 다시 실행)
engine_memory = create_engine('sqlite:///:memory:')
with engine_memory.connect() as conn:
    conn.execute(text("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT)"))
    conn.execute(text("INSERT INTO users (name, age, city) VALUES ('Alice', 30, 'New York'), ('Bob', 25, 'London'), ('Charlie', 35, 'New York'), ('David', 28, 'Paris')"))
    conn.commit()
print("인메모리 DB에 users 테이블 초기화 및 데이터 삽입 완료.")

# products 테이블도 생성하여 예시에 사용
with engine_memory.connect() as conn:
    conn.execute(text("CREATE TABLE IF NOT EXISTS products (product_id INTEGER PRIMARY KEY, product_name TEXT, price INTEGER)"))
    conn.execute(text("INSERT INTO products (product_name, price) VALUES ('Laptop', 1200), ('Mouse', 25), ('Keyboard', 75)"))
    conn.commit()
print("인메모리 DB에 products 테이블 초기화 및 데이터 삽입 완료.")
```

**1. 테이블 전체 데이터 불러오기 (기본 사용법)**
가장 기본적인 형태로, 테이블의 모든 컬럼과 모든 행을 불러옵니다.

**코드**
```python
df_users_table = pd.read_sql_table("users", engine_memory)
print("\n--- 1. read_sql_table로 불러온 users 테이블 전체 데이터 ---")
print(df_users_table)
```
**결과:**
```
--- 1. read_sql_table로 불러온 users 테이블 전체 데이터 ---
   id     name  age      city
0   1    Alice   30  New York
1   2      Bob   25    London
2   3  Charlie   35  New York
3   4    David   28     Paris
```
**결과 설명**
*   `users` 테이블의 모든 데이터가 `DataFrame`으로 성공적으로 로드되었습니다. `read_sql_table`은 내부적으로 `SELECT * FROM table_name` 쿼리를 실행하는 것과 유사하게 작동합니다.

**2. `index_col` 옵션 활용**
불러온 데이터의 특정 컬럼을 `DataFrame`의 인덱스로 바로 지정할 수 있습니다. 이는 데이터 접근을 용이하게 합니다.

**코드**
```python
df_products_indexed = pd.read_sql_table("products", engine_memory, index_col='product_id')
print("\n--- 2. read_sql_table로 불러온 products 테이블 (product_id를 인덱스로) ---")
print(df_products_indexed)
```
**결과:**
```
--- 2. read_sql_table로 불러온 products 테이블 (product_id를 인덱스로) ---
            product_name  price
product_id                     
101               Laptop   1200
102                Mouse     25
103             Keyboard     75
```
**결과 설명**
*   `product_id` 컬럼이 `DataFrame`의 인덱스로 설정되어, `df_products_indexed.loc[101]`과 같이 인덱스를 통해 데이터를 조회할 수 있습니다.

**3. `chunksize` 옵션 활용 (대용량 데이터 처리)**
`read_sql_table()`도 `chunksize` 옵션을 지원하여 대용량 테이블을 한 번에 메모리에 로드하는 대신, 지정된 크기의 청크(chunk)로 나누어 불러올 수 있습니다. 이는 메모리 부족 문제를 방지하고, 데이터를 부분적으로 처리할 수 있게 합니다.

**코드**
```python
# 예시를 위해 users 테이블에 더 많은 데이터 삽입
with engine_memory.connect() as conn:
    conn.execute(text("INSERT INTO users (name, age, city) VALUES ('Eve', 22, 'Tokyo'), ('Frank', 40, 'Berlin'), ('Grace', 29, 'London')"))
    conn.commit()

print("\n--- 3. read_sql_table로 청크 단위 데이터 불러오기 (chunksize=3) ---")
for i, chunk_df in enumerate(pd.read_sql_table("users", engine_memory, chunksize=3)):
    print(f"\n--- 청크 {i+1} ---")
    print(chunk_df)
```
**결과:**
```
--- 3. read_sql_table로 청크 단위 데이터 불러오기 (chunksize=3) ---

--- 청크 1 ---
   id     name  age      city
0   1    Alice   30  New York
1   2      Bob   25    London
2   3  Charlie   35  New York

--- 청크 2 ---
   id   name  age    city
0   4  David   28   Paris
1   5    Eve   22   Tokyo
2   6  Frank   40  Berlin

--- 청크 3 ---
   id   name  age    city
0   7  Grace   29  London
```
**결과 설명**
*   `chunksize=3`으로 설정했기 때문에, `read_sql_table`은 3개의 행을 포함하는 `DataFrame` 청크를 순차적으로 반환합니다. 이를 통해 전체 테이블 데이터를 한 번에 로드하지 않고도 처리할 수 있습니다.

`read_sql_table()`은 특정 테이블의 전체 데이터를 빠르게 `DataFrame`으로 가져와야 할 때 매우 유용합니다. 복잡한 SQL 쿼리가 필요하지 않고, 테이블의 모든 컬럼과 행이 필요한 경우에 `read_sql_query()`보다 간결하게 사용할 수 있습니다.


## 5. DataFrame을 SQL 테이블로 저장 (`to_sql`)

Pandas `DataFrame`에 있는 데이터를 SQL 데이터베이스의 테이블로 저장하는 것은 데이터 분석 결과를 영구적으로 보관하거나, 다른 애플리케이션에서 활용할 수 있도록 데이터베이스에 다시 쓰는 데 필수적인 기능입니다. `DataFrame` 객체의 `to_sql()` 메서드는 이러한 작업을 효율적으로 수행할 수 있도록 다양한 옵션을 제공합니다.

### 5.1. `to_sql()` 메서드 개요

`DataFrame.to_sql(name, con, if_exists='fail', index=True, index_label=None, chunksize=None, dtype=None, method=None)`

*   **`name` (필수)**: 데이터를 저장할 SQL 테이블의 이름(문자열)입니다. 이 이름으로 데이터베이스에 새로운 테이블이 생성되거나, 기존 테이블이 참조됩니다.
*   **`con` (필수)**: 데이터베이스 연결 객체입니다. `SQLAlchemy`의 `Engine` 객체(예: `create_engine`으로 생성한 `engine_memory` 또는 `engine_file`) 또는 DB-API 2.0 호환 연결 객체를 전달합니다.
*   **`if_exists` (선택)**: 지정된 `name`의 테이블이 데이터베이스에 이미 존재할 경우 어떤 동작을 취할지 정의합니다. 다음 세 가지 옵션이 있습니다:
    *   `'fail'` (기본값): 테이블이 이미 존재하면 `ValueError`를 발생시켜 작업을 중단합니다. 안전한 기본값으로, 실수로 기존 데이터를 덮어쓰는 것을 방지합니다.
    *   `'replace'`: 테이블이 이미 존재하면 기존 테이블을 삭제하고 새로운 테이블을 생성한 후 데이터를 저장합니다. 기존 데이터가 모두 사라지므로 주의해서 사용해야 합니다.
    *   `'append'`: 테이블이 이미 존재하면 기존 테이블의 끝에 새로운 데이터를 추가합니다. 테이블 스키마가 `DataFrame`의 스키마와 일치해야 합니다.
*   **`index` (선택)**: `DataFrame`의 인덱스(행 라벨)를 SQL 테이블의 컬럼으로 저장할지 여부를 `True` 또는 `False`로 지정합니다. 기본값은 `True`이며, 이 경우 인덱스는 `index`라는 이름의 컬럼으로 저장됩니다. 인덱스를 저장하고 싶지 않다면 `False`로 설정합니다.
*   **`index_label` (선택)**: `index=True`로 인덱스를 저장할 때, 해당 인덱스 컬럼의 이름을 지정합니다. 기본값은 `None`이며, 이 경우 `DataFrame` 인덱스에 이름이 있다면 그 이름이 사용되고, 없다면 'index'라는 기본 이름이 사용됩니다.
*   **`chunksize` (선택)**: 대용량 `DataFrame`을 한 번에 데이터베이스에 삽입하기 어려울 때, 데이터를 지정된 크기의 "청크(chunk)" 단위로 나누어 삽입할 수 있도록 합니다. 이는 메모리 효율성을 높이고, 데이터베이스의 부하를 줄이는 데 도움이 됩니다.
*   **`dtype` (선택)**: `DataFrame`의 컬럼별로 SQL 데이터베이스의 특정 데이터 타입으로 매핑하고 싶을 때 딕셔너리 형태로 지정합니다(예: `{'column_name': sqlalchemy.types.VARCHAR(50)}`). Pandas는 기본적으로 Python 데이터 타입을 가장 적절한 SQL 타입으로 추론하지만, 명시적인 제어가 필요할 때 사용합니다.
*   **`method` (선택)**: 데이터를 데이터베이스에 삽입하는 방법을 지정합니다. 대용량 데이터 삽입 시 성능을 최적화할 수 있습니다. 기본값은 한 행씩 삽입하는 방식이며, `'multi'`를 사용하면 여러 행을 한 번에 삽입하여 성능을 향상시킬 수 있습니다.

### 5.2. `to_sql()` 활용 예시

이전 섹션에서 생성한 인메모리 SQLite 데이터베이스(`engine_memory`)를 사용하여 `to_sql()`의 다양한 활용법을 살펴보겠습니다.

```python
import pandas as pd
from sqlalchemy import create_engine, text, types # types 모듈 임포트
import sqlite3

# 인메모리 SQLite 데이터베이스 연결 (이전 섹션에서 생성된 engine_memory 재사용 또는 새로 생성)
engine_memory = create_engine('sqlite:///:memory:')
print("인메모리 SQLite 엔진 생성 완료.")

# 예시를 위한 초기 users 테이블 생성 (to_sql 테스트를 위해 비워둠)
with engine_memory.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS users")) # 기존 users 테이블이 있다면 삭제
    conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT)"))
    conn.commit()
print("인메모리 DB에 users 테이블 초기화 완료.")
```

**1. 새로운 테이블로 저장 (`if_exists='fail'`)**
가장 안전한 방법으로, 지정된 이름의 테이블이 이미 존재하면 에러를 발생시켜 데이터 손실을 방지합니다. 새로운 테이블을 생성할 때 주로 사용됩니다.

**코드**
```python
# 새로운 DataFrame 생성
df_products = pd.DataFrame({
    'product_id': [101, 102, 103],
    'product_name': ['Laptop', 'Mouse', 'Keyboard'],
    'price': [1200, 25, 75]
})

# 'products' 테이블로 저장 (테이블이 존재하면 에러 발생)
# 이 코드는 products 테이블이 이미 존재하면 에러를 발생시킵니다.
try:
    df_products.to_sql('products', engine_memory, if_exists='fail', index=False)
    print("\n1. 'products' 테이블 생성 및 데이터 저장 완료 (if_exists='fail').")
except ValueError as e:
    print(f"\n1. 'products' 테이블 생성 실패 (if_exists='fail'): {e}")

# 저장된 데이터 확인
df_check_products = pd.read_sql_query("SELECT * FROM products", engine_memory)
print("저장된 products 테이블 데이터:\n", df_check_products)
```
**결과 설명**
*   `products` 테이블이 성공적으로 생성되고 `df_products`의 데이터가 저장됩니다. 만약 이 코드를 두 번 실행하면 `ValueError`가 발생합니다.

**2. 기존 테이블 교체 (`if_exists='replace'`)**
지정된 이름의 테이블이 이미 존재하면 기존 테이블을 삭제하고 새로운 테이블을 생성한 후 데이터를 저장합니다. 기존 데이터가 완전히 사라지므로 매우 신중하게 사용해야 합니다.

**코드**
```python
# 새로운 데이터로 products 테이블을 교체
df_products_v2 = pd.DataFrame({
    'product_id': [201, 202],
    'product_name': ['Monitor', 'Webcam'],
    'price': [300, 50]
})

df_products_v2.to_sql('products', engine_memory, if_exists='replace', index=False)
print("\n2. 'products' 테이블 데이터 교체 완료 (if_exists='replace').")

# 교체된 데이터 확인
df_check_products_v2 = pd.read_sql_query("SELECT * FROM products", engine_memory)
print("교체된 products 테이블 데이터:\n", df_check_products_v2)
```
**결과 설명**
*   기존 `products` 테이블의 모든 데이터가 삭제되고, `df_products_v2`의 데이터로 완전히 대체됩니다.

**3. 기존 테이블에 데이터 추가 (`if_exists='append'`)**
지정된 이름의 테이블이 이미 존재하면 기존 테이블의 끝에 새로운 데이터를 추가합니다. 테이블 스키마(컬럼명, 데이터 타입)가 `DataFrame`의 스키마와 일치해야 합니다.

**코드**
```python
# products 테이블에 새 데이터 추가
df_new_product = pd.DataFrame({
    'product_id': [203],
    'product_name': ['Headphones'],
    'price': [150]
})

df_new_product.to_sql('products', engine_memory, if_exists='append', index=False)
print("\n3. 'products' 테이블에 새 데이터 추가 완료 (if_exists='append').")

# 추가된 데이터 확인
df_products_appended = pd.read_sql_query("SELECT * FROM products", engine_memory)
print("추가 후 products 테이블 최종 데이터:\n", df_products_appended)
```
**결과 설명**
*   `df_new_product`의 한 행이 기존 `products` 테이블의 끝에 성공적으로 추가됩니다.

**4. `index` 옵션 활용 (인덱스 저장 여부)**
`DataFrame`의 인덱스를 SQL 테이블에 컬럼으로 저장할지 여부를 제어합니다. 기본값은 `True`입니다.

**코드**
```python
df_indexed_example = pd.DataFrame({
    'value': [10, 20, 30]
}, index=['A', 'B', 'C'])
df_indexed_example.index.name = 'my_index_col' # 인덱스에 이름 부여

# 인덱스를 컬럼으로 저장 (기본 동작)
df_indexed_example.to_sql('indexed_table_true', engine_memory, if_exists='replace', index=True)
print("\n4. 'indexed_table_true' (인덱스 포함):\n", pd.read_sql_query("SELECT * FROM indexed_table_true", engine_memory))

# 인덱스를 컬럼으로 저장하지 않음
df_indexed_example.to_sql('indexed_table_false', engine_memory, if_exists='replace', index=False)
print("\n4. 'indexed_table_false' (인덱스 제외):\n", pd.read_sql_query("SELECT * FROM indexed_table_false", engine_memory))
```
**결과 설명**
*   `indexed_table_true`에는 `my_index_col`이라는 이름으로 인덱스가 저장됩니다.
*   `indexed_table_false`에는 인덱스 컬럼 없이 `value` 컬럼만 저장됩니다.

**5. `dtype` 옵션 활용 (SQL 데이터 타입 명시)**
Pandas는 `DataFrame`의 데이터 타입을 기반으로 SQL 테이블의 컬럼 타입을 자동으로 추론하지만, 때로는 명시적으로 SQL 데이터 타입을 지정해야 할 필요가 있습니다. 예를 들어, 문자열 컬럼의 최대 길이를 제한하거나, 특정 숫자 컬럼을 정밀한 실수형으로 저장하고 싶을 때 유용합니다. `SQLAlchemy`의 `types` 모듈을 사용합니다.

**코드**
```python
df_data_types = pd.DataFrame({
    'text_col': ['short', 'longer_string'],
    'int_col': [1, 2],
    'float_col': [1.12345, 2.67890]
})

# 컬럼별 SQL 데이터 타입 명시
dtype_mapping = {
    'text_col': types.VARCHAR(20), # 최대 20자 문자열
    'int_col': types.INTEGER,
    'float_col': types.DECIMAL(10, 5) # 총 10자리, 소수점 이하 5자리
}

df_data_types.to_sql('typed_table', engine_memory, if_exists='replace', index=False, dtype=dtype_mapping)
print("\n5. 'typed_table' 생성 및 데이터 저장 완료 (dtype 명시).")

# 저장된 테이블의 스키마 확인 (SQLite의 경우 PRAGMA table_info 사용)
with engine_memory.connect() as conn:
    result = conn.execute(text("PRAGMA table_info(typed_table)")).fetchall()
    print("\n'typed_table' 스키마 정보:\n", result)

# 데이터 확인
print("\n'typed_table' 데이터:\n", pd.read_sql_query("SELECT * FROM typed_table", engine_memory))
```
**결과 설명**
*   `typed_table`이 생성될 때 `dtype_mapping`에 따라 컬럼의 SQL 데이터 타입이 명시적으로 지정됩니다. SQLite의 `PRAGMA table_info`를 통해 실제 저장된 타입을 확인할 수 있습니다 (SQLite는 내부적으로 TEXT, INTEGER, REAL 등으로 매핑).

**6. `chunksize` 및 `method='multi'` 옵션 활용 (대용량 데이터 효율적 저장)**
대용량 `DataFrame`을 데이터베이스에 저장할 때, `chunksize`를 사용하여 데이터를 작은 덩어리로 나누어 삽입하고, `method='multi'`를 사용하여 여러 행을 한 번의 SQL INSERT 문으로 삽입하면 성능을 크게 향상시킬 수 있습니다. 이는 특히 네트워크 지연이 있거나 데이터베이스에 많은 부하를 주지 않아야 할 때 유용합니다.

**코드**
```python
# 대용량 DataFrame 생성 (예시를 위해 10000행)
df_large = pd.DataFrame({
    'data_id': range(1, 10001),
    'value': [i * 10 for i in range(1, 10001)],
    'category': ['A' if i % 2 == 0 else 'B' for i in range(1, 10001)]
})

print(f"\n6. 대용량 DataFrame 생성 완료: {len(df_large)} 행")

# chunksize와 method='multi'를 사용하여 저장
# SQLite는 기본적으로 'multi'를 지원하며, 다른 DB는 드라이버에 따라 다를 수 있습니다.
# 이 작업은 시간이 다소 걸릴 수 있습니다.
print("대용량 데이터 저장 중... (chunksize=1000, method='multi')")
import time
start_time = time.time()
df_large.to_sql('large_data_table', engine_memory, if_exists='replace', index=False, chunksize=1000, method='multi')
end_time = time.time()
print(f"대용량 데이터 저장 완료. 소요 시간: {end_time - start_time:.2f} 초")

# 저장된 데이터의 일부 확인
df_check_large = pd.read_sql_query("SELECT COUNT(*) FROM large_data_table", engine_memory)
print("저장된 총 행 수:\n", df_check_large)
```
**결과 설명**
*   `large_data_table`이 생성되고 10000개의 행이 1000개씩 청크로 나뉘어 효율적으로 삽입됩니다. `method='multi'`는 각 청크 내에서 여러 행을 하나의 `INSERT` 문으로 묶어 데이터베이스와의 통신 횟수를 줄여 성능을 향상시킵니다.

`to_sql()` 메서드는 Pandas `DataFrame`의 데이터를 SQL 데이터베이스로 내보내는 데 있어 매우 강력하고 유연한 도구입니다. `if_exists` 옵션을 통해 데이터 손실을 방지하거나 데이터를 효율적으로 관리하고, `dtype`, `chunksize`, `method`와 같은 고급 옵션을 통해 저장 성능과 데이터 무결성을 최적화할 수 있습니다.


## 6. 실전 예제: SQLite 데이터베이스 연동

이 섹션에서는 앞서 학습한 Pandas의 SQL 연동 기능(`create_engine`, `to_sql`, `read_sql_table`, `read_sql_query`)을 종합적으로 활용하여 SQLite 데이터베이스와 상호작용하는 실전 예제를 다룹니다. SQLite는 별도의 서버 설정 없이 파일 기반으로 쉽게 사용할 수 있어 테스트 및 학습에 매우 적합합니다.

이 예제에서는 다음과 같은 과정을 수행합니다:
1.  파일 기반 SQLite 데이터베이스를 생성하고 연결합니다.
2.  가상의 판매 데이터를 포함하는 Pandas `DataFrame`을 생성합니다.
3.  `DataFrame`을 데이터베이스의 새로운 테이블로 저장합니다.
4.  저장된 테이블의 모든 데이터를 불러와 확인합니다.
5.  SQL 쿼리를 사용하여 특정 조건의 데이터를 필터링하여 불러옵니다.
6.  SQL 쿼리를 사용하여 데이터를 집계합니다.
7.  새로운 데이터를 기존 테이블에 추가하고, 최종 데이터를 확인합니다.

```python
import pandas as pd
from sqlalchemy import create_engine, text
import os # 파일 시스템 작업을 위해 os 모듈 임포트

# 1. 파일 기반 SQLite 데이터베이스 생성 및 연결
# 데이터베이스 파일 경로 설정
db_file_path = 'sales_data.db'

# 기존에 sales_data.db 파일이 있다면 삭제하여 깨끗한 상태에서 시작 (테스트용)
if os.path.exists(db_file_path):
    os.remove(db_file_path)
    print(f"기존 '{db_file_path}' 파일 삭제 완료.")

# SQLAlchemy의 create_engine을 사용하여 SQLite 데이터베이스에 연결 엔진 생성
# 'sqlite:///' 뒤에 파일 경로를 지정하면 해당 파일에 데이터베이스가 생성되거나 연결됩니다.
engine_sales = create_engine(f'sqlite:///{db_file_path}')
print(f"\n--- '{db_file_path}' 파일 기반 DB 연결 엔진 생성 완료 ---")

# 2. 판매 데이터 DataFrame 생성
# 실제 시나리오를 모방한 가상의 판매 데이터를 DataFrame으로 만듭니다.
# sale_date 컬럼은 pd.to_datetime을 사용하여 datetime 타입으로 미리 변환합니다.
df_sales_data = pd.DataFrame({
    'sale_id': [1, 2, 3, 4, 5],
    'product': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse'],
    'region': ['East', 'West', 'East', 'North', 'West'],
    'amount': [1200, 25, 1500, 75, 30],
    'sale_date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03'])
})
print("\n2. 판매 데이터 DataFrame 생성 완료:")
print(df_sales_data)
print("\nDataFrame 정보:")
df_sales_data.info()

# 3. DataFrame을 'sales' 테이블로 저장
# to_sql() 메서드를 사용하여 df_sales_data를 'sales'라는 이름의 테이블로 데이터베이스에 저장합니다.
# if_exists='replace': 테이블이 이미 존재하면 기존 테이블을 삭제하고 새로 생성합니다.
# index=False: DataFrame의 인덱스를 SQL 테이블의 컬럼으로 저장하지 않습니다.
df_sales_data.to_sql('sales', engine_sales, if_exists='replace', index=False)
print("\n3. 'sales' 테이블에 데이터 저장 완료 (if_exists='replace').")

# 4. 저장된 'sales' 테이블의 모든 데이터 불러오기 (`read_sql_table`)
# read_sql_table() 함수를 사용하여 'sales' 테이블의 모든 데이터를 DataFrame으로 불러옵니다.
df_loaded_sales = pd.read_sql_table('sales', engine_sales)
print("\n4. 'sales' 테이블 전체 데이터 (read_sql_table로 불러옴):\n", df_loaded_sales)

# 5. 특정 조건의 데이터만 SQL 쿼리로 불러오기 (`read_sql_query`)
# read_sql_query() 함수를 사용하여 SQL 쿼리를 직접 실행하여 원하는 조건의 데이터만 가져옵니다.
# 예: 'Laptop' 제품의 판매 데이터만 조회
df_laptop_sales = pd.read_sql_query("SELECT * FROM sales WHERE product = 'Laptop'", engine_sales)
print("\n5. 'Laptop' 제품 판매 데이터 (SQL 쿼리로 필터링):\n", df_laptop_sales)

# 6. 지역별 총 판매액 계산 (SQL 쿼리 사용)
# SQL의 GROUP BY와 SUM 함수를 사용하여 데이터베이스 레벨에서 집계된 결과를 가져옵니다.
df_region_sales = pd.read_sql_query("SELECT region, SUM(amount) as total_amount FROM sales GROUP BY region", engine_sales)
print("\n6. 지역별 총 판매액 (SQL 쿼리로 집계):\n", df_region_sales)

# 7. 새로운 데이터 추가 후 확인
# 새로운 판매 데이터를 포함하는 DataFrame을 생성합니다.
df_new_sales = pd.DataFrame({
    'sale_id': [6, 7],
    'product': ['Monitor', 'Webcam'],
    'region': ['East', 'North'],
    'amount': [300, 50],
    'sale_date': [pd.to_datetime('2023-01-04'), pd.to_datetime('2023-01-04')]
})

# to_sql()의 if_exists='append' 옵션을 사용하여 기존 'sales' 테이블에 데이터를 추가합니다.
df_new_sales.to_sql('sales', engine_sales, if_exists='append', index=False)
print("\n7. 새로운 판매 데이터 추가 완료 (if_exists='append').")

# 데이터 추가 후 'sales' 테이블의 모든 데이터를 다시 불러와 최종 상태를 확인합니다.
df_all_sales_after_append = pd.read_sql_query("SELECT * FROM sales", engine_sales)
print("데이터 추가 후 'sales' 테이블 최종 데이터:\n", df_all_sales_after_append)

# 데이터베이스 연결 종료 (선택 사항)
# 파일 기반 SQLite 데이터베이스는 스크립트 종료 시 자동으로 저장되므로, 
# 명시적으로 engine.dispose()를 호출할 필요는 없지만, 
# 다른 종류의 데이터베이스에서는 연결 풀을 정리하기 위해 유용할 수 있습니다.
# engine_sales.dispose()
# print("\n데이터베이스 연결 종료.")

# 최종적으로 생성된 sales_data.db 파일 삭제 (테스트 후 정리용, 필요시 주석 해제)
# if os.path.exists(db_file_path):
#     os.remove(db_file_path)
#     print(f"\n'{db_file_path}' 파일 최종 삭제 완료.")
```

이 실전 예제를 통해 Pandas의 SQL 연동 기능을 사용하여 데이터베이스를 생성하고, 데이터를 저장하며, 다양한 방식으로 데이터를 불러오고 조작하는 전체 워크플로우를 이해할 수 있습니다. 이는 실제 데이터 과학 프로젝트에서 데이터베이스와 상호작용하는 데 필요한 핵심적인 기술입니다.
