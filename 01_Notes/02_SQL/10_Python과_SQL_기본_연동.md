<h2>Python과 SQL 연동: DB-API 및 Pandas 활용 (실무 활용 중심)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-28

<h2>문서 목표</h2>
<p>이 문서는 <strong>Python과 SQL 데이터베이스를 연동하는 핵심 방법</strong>인 DB-API와 Pandas 활용법을 심도 있게 다룹니다. 각 개념의 정의, 실제 코드에서의 활용법, 그리고 <strong>데이터 분석 및 AI 실무에서 발생할 수 있는 주의사항과 활용 팁</strong>을 상세한 예제와 함께 설명하여, 파이썬을 활용한 안전하고 효율적인 데이터베이스 연동의 견고한 기초를 다지는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. DB-API 드라이버를 이용한 직접 연결](#1-db-api-드라이버를-이용한-직접-연결)
  - [1.1. 설치 및 기본 연결 설정](#11-설치-및-기본-연결-설정)
  - [1.2. 쿼리 실행](#12-쿼리-실행)
  - [1.3. SQL Injection 방지를 위한 파라미터화](#13-sql-injection-방지를-위한-파라미터화)
  - [1.4. 트랜잭션 관리](#14-트랜잭션-관리)
  - [1.5. 에러 처리 및 연결 종료](#15-에러-처리-및-연결-종료)
- [2. Pandas와 SQL 연동](#2-pandas와-sql-연동)
  - [2.1. `pd.read_sql()`: SQL 쿼리 결과를 DataFrame으로 로드](#21-pdread_sql-sql-쿼리-결과를-dataframe으로-로드)
  - [2.2. `df.to_sql()`: DataFrame을 데이터베이스 테이블로 저장](#22-dfto_sql-dataframe을-데이터베이스-테이블로-저장)
  - [2.3. 데이터 분석 파이프라인에서의 활용 예시](#23-데이터-분석-파이프라인에서의-활용-예시)
- [3. 파이썬-SQL 데이터 타입 매핑 및 변환](#3-파이썬-sql-데이터-타입-매핑-및-변환)
  - [3.1. 주요 데이터 타입 매핑](#31-주요-데이터-타입-매핑)
  - [3.2. 일반적인 문제점 및 해결 방안](#32-일반적인-문제점-및-해결-방안)

---

## 1. DB-API 드라이버를 이용한 직접 연결

파이썬에서 MySQL 데이터베이스에 직접 연결하여 SQL 쿼리를 실행하려면 DB-API 2.0 표준을 따르는 드라이버가 필요합니다. `pymysql`은 순수 파이썬으로 작성된 MySQL 클라이언트 라이브러리이며, `mysql-connector-python`은 MySQL 공식 드라이버입니다. 여기서는 `pymysql`을 예시로 설명합니다.

### 1.1. 설치 및 기본 연결 설정

먼저 `pymysql` 라이브러리를 설치합니다.

```bash
pip install pymysql
```

기본적인 연결 설정은 다음과 같습니다.

```python
import pymysql
import os
from dotenv import load_dotenv # .env 파일에서 환경 변수를 로드하기 위함

load_dotenv() # .env 파일 로드

# 데이터베이스 연결 설정 (환경 변수 사용 권장)
conn = pymysql.connect(
    host=os.getenv('DB_HOST', 'localhost'), # 환경 변수에서 가져오거나 기본값 사용
    user=os.getenv('DB_USER', 'root'),
    password=os.getenv('DB_PASSWORD'), # 보안상 코드에 직접 노출하지 않음
    db=os.getenv('DB_NAME', 'company_db'),
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor, # 결과를 딕셔너리 형태로 받기 위함
    autocommit=False # 명시적으로 자동 커밋 비활성화
)

try:
    with conn.cursor() as cursor:
        # 쿼리 실행 예시
        sql = "SELECT VERSION();"
        cursor.execute(sql)
        result = cursor.fetchone()
        print(f"MySQL Version: {result['VERSION()']}")

finally:
    conn.close()
```

**실무적 관점: 민감 정보 관리 (보안 모범 사례)**
데이터베이스 접속 정보(호스트, 사용자 이름, 비밀번호 등)는 매우 민감한 정보이므로, 코드 내에 직접 하드코딩하는 것은 절대 피해야 합니다. 대신 **환경 변수**나 **비밀 관리 서비스(AWS Secrets Manager, Google Secret Manager 등)**를 사용하여 안전하게 관리해야 합니다. `.env` 파일을 사용하여 개발 환경에서 환경 변수를 편리하게 관리할 수 있습니다 (`python-dotenv` 라이브러리 활용).

### 1.2. 쿼리 실행

`cursor.execute()` 메서드를 사용하여 SQL 쿼리를 실행합니다. `SELECT` 쿼리의 결과는 `fetchone()`, `fetchall()`, `fetchmany()` 등으로 가져올 수 있습니다.

```python
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

conn = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'), charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

try:
    with conn.cursor() as cursor:
        # INSERT 예시
        insert_sql = "INSERT INTO employees (first_name, last_name, hire_date, job_id, salary, department_id) VALUES (%s, %s, %s, %s, %s, %s)"
        cursor.execute(insert_sql, ('Grace', 'Hopper', '2024-01-01', 'DEV', 80000.00, 1))
        conn.commit() # 변경사항 커밋
        print(f"Inserted new employee with ID: {cursor.lastrowid}")

        # SELECT 예시
        select_sql = "SELECT employee_id, first_name, last_name, salary FROM employees WHERE job_id = %s"
        cursor.execute(select_sql, ('DEV',))
        employees = cursor.fetchall()
        print("\nDevelopers:")
        for emp in employees:
            print(f"  {emp['first_name']} {emp['last_name']} (Salary: {emp['salary']})")

        # UPDATE 예시
        update_sql = "UPDATE employees SET salary = %s WHERE employee_id = %s"
        cursor.execute(update_sql, (85000.00, 1))
        conn.commit()
        print(f"\nUpdated {cursor.rowcount} row(s).")

        # DELETE 예시
        delete_sql = "DELETE FROM employees WHERE first_name = %s"
        cursor.execute(delete_sql, ('Grace',))
        conn.commit()
        print(f"Deleted {cursor.rowcount} row(s).")

finally:
    conn.close()
```

### 1.3. SQL Injection 방지를 위한 파라미터화

사용자 입력을 직접 SQL 쿼리 문자열에 삽입하는 것은 **SQL Injection** 공격에 취약합니다. `pymysql`과 같은 DB-API 드라이버는 쿼리 파라미터화(Parameterized Queries)를 지원하여 이를 방지합니다. **절대 문자열 포매팅(f-string, `%`)으로 SQL 쿼리를 만들지 마십시오.**

```python
# 안전한 방법 (파라미터화)
user_input_name = "Robert'; DROP TABLE employees; --"
sql = "SELECT * FROM employees WHERE first_name = %s"
cursor.execute(sql, (user_input_name,)) # %s 플레이스홀더 사용

# 위험한 방법 (SQL Injection 취약)
# sql = f"SELECT * FROM employees WHERE first_name = '{user_input_name}'"
# cursor.execute(sql)
```

### 1.4. 트랜잭션 관리

데이터의 일관성과 무결성을 위해 트랜잭션을 명시적으로 관리하는 것이 중요합니다. `conn.commit()`으로 변경사항을 확정하고, `conn.rollback()`으로 취소할 수 있습니다.

```python
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

conn = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'), charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

try:
    with conn.cursor() as cursor:
        # 트랜잭션 시작 (기본적으로 자동 커밋이 아님)
        cursor.execute("SET autocommit = 0;") # 명시적으로 자동 커밋 끄기 (pymysql은 기본적으로 autocommit=True)

        # 첫 번째 작업
        cursor.execute("UPDATE employees SET salary = salary + 1000 WHERE employee_id = 1;")
        print("Employee 1 salary increased.")

        # 두 번째 작업 (오류 발생 가정)
        # raise Exception("Simulated error")
        cursor.execute("UPDATE employees SET salary = salary + 500 WHERE employee_id = 999;") # 존재하지 않는 ID
        print("Employee 999 salary increased.")

        conn.commit() # 모든 작업 성공 시 커밋
        print("Transaction committed.")

except Exception as e:
    conn.rollback() # 오류 발생 시 롤백
    print(f"Transaction rolled back due to error: {e}")

finally:
    conn.close()
```

### 1.5. 에러 처리 및 연결 종료

데이터베이스 작업 중 발생할 수 있는 다양한 예외를 처리하고, 사용 후에는 반드시 연결을 닫아 리소스 누수를 방지해야 합니다. `try...finally` 블록을 사용하여 연결이 항상 닫히도록 보장합니다.

```python
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

conn = None # 초기화
try:
    conn = pymysql.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'), db=os.getenv('DB_NAME'), charset='utf8mb4')
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM non_existent_table;")
        print(cursor.fetchall())
except pymysql.err.OperationalError as e:
    print(f"Database connection or operation error: {e}")
except pymysql.err.ProgrammingError as e:
    print(f"SQL syntax or programming error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if conn:
        conn.close()
        print("Database connection closed.")
```

## 2. Pandas와 SQL 연동

Pandas는 파이썬 데이터 분석의 핵심 라이브러리이며, SQL 데이터베이스와의 연동 기능을 강력하게 제공합니다. 이를 통해 데이터베이스의 데이터를 Pandas DataFrame으로 가져와 분석하거나, DataFrame의 데이터를 데이터베이스로 저장할 수 있습니다.

**실무적 관점:** Pandas와 SQL 연동은 데이터 수집, 전처리, 분석, 결과 저장 등 데이터 분석 파이프라인의 여러 단계에서 활용될 수 있습니다. 특히 대용량 데이터 처리 시 메모리 문제를 해결하기 위한 `chunksize` 활용과 `df.to_sql()`의 성능 최적화 기법은 실무에서 반드시 숙지해야 할 필수 요소입니다.

### 2.1. `pd.read_sql()`: SQL 쿼리 결과를 DataFrame으로 로드

`pandas.read_sql()` 함수는 SQL 쿼리의 결과를 직접 Pandas DataFrame으로 로드합니다. `read_sql_query`와 `read_sql_table`의 기능을 통합한 함수입니다.

**실무 팁: 대용량 데이터 조회 시 메모리 문제와 `chunksize` 활용**
`pd.read_sql`은 기본적으로 쿼리의 모든 결과를 메모리로 불러옵니다. 만약 수백만, 수천만 건의 대용량 데이터를 한 번에 조회하면 **메모리 부족(Out of Memory) 오류**가 발생하거나, 시스템 전체의 성능 저하를 유발할 수 있습니다. 이때 `chunksize` 파라미터를 사용하면 데이터를 지정된 크기의 덩어리(chunk)로 나누어 처리할 수 있습니다. 이 방식은 메모리 사용량을 크게 줄여주어 대용량 데이터 처리 시 필수적인 기법입니다.

```python
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

# SQLAlchemy 엔진 생성 시 pool_pre_ping=True 옵션 추가
# 이는 데이터베이스 연결이 유효한지 주기적으로 확인하여, 
# 오랜 시간 사용하지 않아 끊어진 연결로 인한 오류를 방지합니다.
DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}"
    f