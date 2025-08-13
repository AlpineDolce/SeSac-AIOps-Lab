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

데이터 과학 프로젝트에서 데이터는 종종 관계형 데이터베이스(RDB)에 저장되어 있습니다. Pandas는 이러한 SQL 데이터베이스와 쉽게 연동하여 데이터를 불러오고, 분석 결과를 다시 저장할 수 있는 강력한 기능을 제공합니다. 이는 데이터 분석 워크플로우를 효율적으로 구축하는 데 필수적입니다.

**필요 라이브러리:**
Pandas는 데이터베이스 연동을 위해 `SQLAlchemy` 라이브러리를 사용합니다. 또한, 특정 데이터베이스에 연결하기 위한 드라이버가 필요합니다.

*   **`SQLAlchemy`**: 다양한 데이터베이스를 동일한 방식으로 다룰 수 있게 해주는 ORM(Object-Relational Mapping) 및 SQL 툴킷.
    ```bash
pip install SQLAlchemy
    ```
*   **데이터베이스 드라이버**: 사용하려는 데이터베이스에 맞는 드라이버를 설치해야 합니다.
    *   SQLite: Python 내장 (`sqlite3`)
    *   PostgreSQL: `psycopg2` (`pip install psycopg2-binary`)
    *   MySQL: `mysql-connector-python` 또는 `pymysql` (`pip install mysql-connector-python`)

이 문서에서는 별도의 서버 설정 없이 파일 기반으로 쉽게 사용할 수 있는 **SQLite**를 중심으로 설명합니다.

## 2. 데이터베이스 연결 설정

데이터베이스에 연결하기 위해서는 `SQLAlchemy`의 `create_engine` 함수를 사용하여 연결 엔진을 생성해야 합니다. 연결 문자열(Connection String 또는 URI)은 데이터베이스의 종류, 사용자 정보, 호스트, 포트, 데이터베이스 이름 등을 포함합니다.

```python
import pandas as pd
from sqlalchemy import create_engine
import sqlite3 # SQLite 사용을 위해 필요

# SQLite 인메모리 데이터베이스 연결 (테스트용)
# 'sqlite:///:memory:'는 휘발성 인메모리 데이터베이스를 생성합니다.
engine_memory = create_engine('sqlite:///:memory:')
print("인메모리 SQLite 엔진 생성 완료.")

# SQLite 파일 기반 데이터베이스 연결 (영구 저장)
# 'sqlite:///my_database.db'는 현재 디렉토리에 my_database.db 파일을 생성합니다.
engine_file = create_engine('sqlite:///my_database.db')
print("파일 기반 SQLite 엔진 생성 완료 (my_database.db).")

# 연결 테스트를 위해 간단한 테이블 생성 (인메모리 DB에)
with engine_memory.connect() as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.execute("INSERT INTO users (name, age) VALUES ('Alice', 30), ('Bob', 25)")
    conn.commit()
print("인메모리 DB에 users 테이블 생성 및 데이터 삽입 완료.")
```

## 3. SQL 쿼리로 데이터 불러오기 (`read_sql_query`)

`read_sql_query()` 함수는 SQL 쿼리 문자열을 실행하여 그 결과를 `DataFrame`으로 반환합니다. 가장 유연하게 데이터를 불러올 수 있는 방법입니다.

**주요 `read_sql_query()` 옵션:**

| 옵션 | 설명 | 필수 여부 |
| --- | --- | --- |
| `sql` | 실행할 SQL 쿼리 문자열. | 필수 |
| `con` | 데이터베이스 연결 객체 (SQLAlchemy 엔진 또는 DB API 2.0 연결). | 필수 |
| `index_col` | `DataFrame`의 인덱스로 사용할 컬럼 이름 또는 리스트. | 선택 |
| `chunksize` | 대용량 데이터를 청크(chunk) 단위로 불러올 때 사용. | 선택 |

```python
# 인메모리 DB에서 모든 users 데이터 불러오기
df_users_all = pd.read_sql_query("SELECT * FROM users", engine_memory)
print("\n--- read_sql_query로 불러온 모든 users 데이터 ---")
print(df_users_all)

# 조건부 쿼리로 데이터 불러오기
df_users_filtered = pd.read_sql_query("SELECT name, age FROM users WHERE age > 28", engine_memory)
print("\n--- read_sql_query로 불러온 필터링된 users 데이터 ---")
print(df_users_filtered)
```

## 4. 테이블 전체 불러오기 (`read_sql_table`)

`read_sql_table()` 함수는 데이터베이스 내의 특정 테이블 전체를 `DataFrame`으로 불러옵니다. 테이블 이름만 알면 되므로 간단하게 사용할 수 있습니다.

**주요 `read_sql_table()` 옵션:**

| 옵션 | 설명 | 필수 여부 |
| --- | --- | --- |
| `table_name` | 불러올 테이블의 이름. | 필수 |
| `con` | 데이터베이스 연결 객체. | 필수 |
| `schema` | 스키마 이름 (PostgreSQL 등에서 사용). | 선택 |
| `index_col` | `DataFrame`의 인덱스로 사용할 컬럼 이름 또는 리스트. | 선택 |

```python
# 인메모리 DB에서 users 테이블 전체 불러오기
df_users_table = pd.read_sql_table("users", engine_memory)
print("\n--- read_sql_table로 불러온 users 테이블 ---")
print(df_users_table)
```

## 5. DataFrame을 SQL 테이블로 저장 (`to_sql`)

`DataFrame`을 SQL 데이터베이스의 테이블로 저장할 때는 `to_sql()` 메서드를 사용합니다. 새로운 테이블을 생성하거나 기존 테이블에 데이터를 추가/교체할 수 있습니다.

**주요 `to_sql()` 옵션:**

| 옵션 | 설명 | 필수 여부 |
| --- | --- | --- |
| `name` | 저장할 테이블의 이름. | 필수 |
| `con` | 데이터베이스 연결 객체. | 필수 |
| `if_exists` | 테이블이 이미 존재할 경우의 동작. (`'fail'`, `'replace'`, `'append'`). | 선택 (`'fail'`) |
| `index` | `DataFrame`의 인덱스를 테이블의 컬럼으로 저장할지 여부. | 선택 (`True`) |
| `dtype` | 컬럼별 SQL 데이터 타입 매핑 (딕셔너리). | 선택 |

```python
# 새로운 DataFrame 생성
df_products = pd.DataFrame({
    'product_id': [101, 102, 103],
    'product_name': ['Laptop', 'Mouse', 'Keyboard'],
    'price': [1200, 25, 75]
})

# 1. 새로운 테이블로 저장 (테이블이 존재하면 에러 발생)
# df_products.to_sql('products', engine_memory, if_exists='fail', index=False)
# print("\nproducts 테이블 생성 및 데이터 저장 완료 (if_exists='fail').")

# 2. 새로운 테이블로 저장 (테이블이 존재하면 교체)
df_products.to_sql('products', engine_memory, if_exists='replace', index=False)
print("\nproducts 테이블 생성 및 데이터 저장 완료 (if_exists='replace').")

# 3. 기존 테이블에 데이터 추가
df_new_product = pd.DataFrame({
    'product_id': [104],
    'product_name': ['Monitor'],
    'price': [300]
})
df_new_product.to_sql('products', engine_memory, if_exists='append', index=False)
print("\nproducts 테이블에 새 데이터 추가 완료 (if_exists='append').")

# 추가된 데이터 확인
df_products_check = pd.read_sql_query("SELECT * FROM products", engine_memory)
print("\nproducts 테이블 최종 데이터:\n", df_products_check)
```

## 6. 실전 예제: SQLite 데이터베이스 연동

이 섹션에서는 SQLite 데이터베이스를 사용하여 Pandas의 SQL 연동 기능을 종합적으로 실습합니다. 파일 기반 데이터베이스를 생성하고, 데이터를 저장하며, 다양한 쿼리로 데이터를 불러오는 과정을 보여줍니다.

```python
# 1. 파일 기반 SQLite 데이터베이스 생성 및 연결
# 기존 파일이 있다면 덮어쓰기 위해 삭제 (테스트용)
import os
if os.path.exists('sales_data.db'):
    os.remove('sales_data.db')

engine_sales = create_engine('sqlite:///sales_data.db')
print("\n--- sales_data.db 파일 기반 DB 연결 ---")

# 2. 판매 데이터 DataFrame 생성
df_sales_data = pd.DataFrame({
    'sale_id': [1, 2, 3, 4, 5],
    'product': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Mouse'],
    'region': ['East', 'West', 'East', 'North', 'West'],
    'amount': [1200, 25, 1500, 75, 30],
    'sale_date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03'])
})

# 3. DataFrame을 'sales' 테이블로 저장 (기존 테이블이 있다면 교체)
df_sales_data.to_sql('sales', engine_sales, if_exists='replace', index=False)
print("\n'sales' 테이블에 데이터 저장 완료.")

# 4. 'sales' 테이블의 모든 데이터 불러오기
df_loaded_sales = pd.read_sql_table('sales', engine_sales)
print("\n'sales' 테이블 전체 데이터:\n", df_loaded_sales)

# 5. 특정 조건의 데이터만 SQL 쿼리로 불러오기 (예: 'Laptop' 제품의 판매 데이터)
df_laptop_sales = pd.read_sql_query("SELECT * FROM sales WHERE product = 'Laptop'", engine_sales)
print("\n'Laptop' 제품 판매 데이터:\n", df_laptop_sales)

# 6. 지역별 총 판매액 계산 (SQL 쿼리 사용)
df_region_sales = pd.read_sql_query("SELECT region, SUM(amount) as total_amount FROM sales GROUP BY region", engine_sales)
print("\n지역별 총 판매액:\n", df_region_sales)

# 7. 새로운 데이터 추가 후 확인
df_new_sales = pd.DataFrame({
    'sale_id': [6],
    'product': ['Monitor'],
    'region': ['East'],
    'amount': [300],
    'sale_date': [pd.to_datetime('2023-01-04')]
})
df_new_sales.to_sql('sales', engine_sales, if_exists='append', index=False)
print("\n새로운 판매 데이터 추가 완료.")

df_all_sales_after_append = pd.read_sql_query("SELECT * FROM sales", engine_sales)
print("\n데이터 추가 후 'sales' 테이블:\n", df_all_sales_after_append)

# 데이터베이스 연결 종료 (선택 사항, 파일 기반 DB는 자동으로 저장됨)
# engine_sales.dispose()
# print("\n데이터베이스 연결 종료.")
```
