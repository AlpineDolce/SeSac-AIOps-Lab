<h2>Python과 SQL 기본 연동 및 Pandas 활용</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-28

<h2>문서 목표</h2>
<p>이 문서는 데이터 분석가가 가장 빈번하게 사용하는 DB-API와 Pandas를 활용하여 데이터베이스와 상호작용하는 기본적인 방법을 학습합니다. 안전한 쿼리 작성법과 데이터프레임 변환을 중심으로 실무적인 데이터 처리 능력을 기르는 것을 목표로 합니다.</p>

> **현업 데이터 분석가의 관점 한 줄 요약:**
> 파이썬과 SQL의 연동은 데이터 분석가의 핵심 역량입니다. `pymysql`과 같은 DB-API를 통해 데이터베이스와 직접 소통하고, `Pandas`를 활용하여 데이터를 효율적으로 가져오고 저장하는 능력은 데이터 분석 파이프라인 구축의 기본입니다. 특히 SQL Injection 방지와 대용량 데이터 처리 기법은 실무에서 반드시 숙지해야 할 안전하고 효율적인 데이터 관리의 필수 요소입니다.

<h2>목차</h2>

- [1. DB-API 드라이버를 이용한 직접 연결: `pymysql` (또는 `mysql-connector-python`)](#1-db-api-드라이버를-이용한-직접-연결-pymysql-또는-mysql-connector-python)
  - [1.1. 설치 및 기본 연결 설정](#11-설치-및-기본-연결-설정)
  - [1.2. 쿼리 실행 (SELECT, INSERT, UPDATE, DELETE)](#12-쿼리-실행-select-insert-update-delete)
  - [1.3. (중요) SQL Injection 방지를 위한 파라미터화](#13-중요-sql-injection-방지를-위한-파라미터화)
  - [1.4. 트랜잭션 관리 (COMMIT, ROLLBACK)](#14-트랜잭션-관리-commit-rollback)
  - [1.5. 에러 처리 및 연결 종료](#15-에러-처리-및-연결-종료)
- [2. Pandas와 SQL 연동: 데이터 분석 워크플로우](#2-pandas와-sql-연동-데이터-분석-워크플로우)
  - [2.1. `pd.read_sql()`: SQL 쿼리 결과를 DataFrame으로 로드](#21-pdread_sql-sql-쿼리-결과를-dataframe으로-로드)
  - [2.2. `df.to_sql()`: DataFrame을 데이터베이스 테이블로 저장](#22-dfto_sql-dataframe을-데이터베이스-테이블로-저장)
  - [2.3. 데이터 분석 파이프라인에서의 활용 예시](#23-데이터-분석-파이프라인에서의-활용-예시)
- [3. 파이썬-SQL 데이터 타입 매핑 및 변환](#3-파이썬-sql-데이터-타입-매핑-및-변환)
  - [3.1. 주요 데이터 타입 매핑 (Python to SQL)](#31-주요-데이터-타입-매핑-python-to-sql)
  - [3.2. 일반적인 문제점 및 해결 방안 (NULL, 날짜/시간, 불리언)](#32-일반적인-문제점-및-해결-방안-null-날짜시간-불리언)

---

## 1. DB-API 드라이버를 이용한 직접 연결: `pymysql` (또는 `mysql-connector-python`)

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

**보안 모범 사례: 민감 정보 관리 (실무적 관점):**
데이터베이스 접속 정보(호스트, 사용자 이름, 비밀번호 등)는 매우 민감한 정보이므로, 코드 내에 직접 하드코딩하는 것은 절대 피해야 합니다. 이는 보안 취약점을 만들고, 코드를 공유하거나 버전 관리 시스템에 올릴 때 정보가 유출될 위험이 있습니다. 대신 다음과 같은 방법을 사용하여 안전하게 관리해야 합니다.

1.  **환경 변수 (Environment Variables):**
    가장 일반적이고 권장되는 방법입니다. 운영체제 환경 변수에 민감 정보를 저장하고, 파이썬 코드에서는 `os.getenv()`를 사용하여 이 값을 읽어옵니다. `.env` 파일을 사용하여 개발 환경에서 환경 변수를 편리하게 관리할 수 있습니다 (`python-dotenv` 라이브러리 활용).
    *   `.env` 파일 예시:
        ```
        DB_HOST=localhost
        DB_USER=your_db_user
        DB_PASSWORD=your_strong_password
        DB_NAME=your_database
        ```
    *   `.gitignore`에 `.env` 파일을 추가하여 버전 관리 시스템에 포함되지 않도록 합니다.
    *   **장점:** 코드와 설정 분리, 버전 관리 시스템에 민감 정보 노출 방지, 배포 환경별 설정 용이.
    *   **단점:** 환경 변수 자체의 보안(서버 접근 권한)에 의존, 대규모/복잡한 설정 관리 어려움.

2.  **비밀 관리 서비스 (Secret Management Services):**
    클라우드 환경(AWS Secrets Manager, Google Secret Manager, Azure Key Vault)이나 온프레미스 환경(HashiCorp Vault)에서는 비밀 관리 서비스를 사용하여 민감 정보를 중앙에서 안전하게 관리하고, 애플리케이션이 필요할 때 동적으로 가져오도록 설정할 수 있습니다. 이는 대규모 애플리케이션이나 마이크로서비스 아키텍처에서 특히 중요합니다.
    *   **장점:** 중앙 집중식 관리, 강력한 보안(암호화, 접근 제어, 감사 로깅), 비밀 정보 자동 로테이션.
    *   **단점:** 추가적인 인프라 및 설정 필요, 서비스 비용 발생.

3.  **설정 파일 (Configuration Files) - 암호화 필수:**
    `config.ini` 또는 `config.json`과 같은 설정 파일을 사용할 수도 있지만, 이 경우에도 설정 파일 자체를 버전 관리 시스템에 포함하지 않거나, **반드시 암호화하여 저장하는 등의 추가적인 보안 조치가 필요합니다.** 평문으로 저장된 설정 파일은 환경 변수보다 더 큰 보안 위험을 초래할 수 있습니다. 환경 변수 방식이 더 선호됩니다.

이 문서에서는 `dotenv` 라이브러리를 사용하여 `.env` 파일에서 환경 변수를 로드하는 방식을 예시로 들고 있습니다. 이는 개발 편의성과 보안을 동시에 고려한 좋은 방법입니다. **실무에서는 프로젝트의 규모, 보안 요구사항, 배포 환경 등을 종합적으로 고려하여 가장 적절한 민감 정보 관리 전략을 선택해야 합니다.**

### 1.2. 쿼리 실행 (SELECT, INSERT, UPDATE, DELETE)

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

### 1.4. 트랜잭션 관리 (COMMIT, ROLLBACK)

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

## 2. Pandas와 SQL 연동: 데이터 분석 워크플로우

Pandas는 파이썬 데이터 분석의 핵심 라이브러리이며, SQL 데이터베이스와의 연동 기능을 강력하게 제공합니다. 이를 통해 데이터베이스의 데이터를 Pandas DataFrame으로 가져와 분석하거나, DataFrame의 데이터를 데이터베이스로 저장할 수 있습니다.

### 2.1. `pd.read_sql()`: SQL 쿼리 결과를 DataFrame으로 로드

`pandas.read_sql()` 함수는 SQL 쿼리의 결과를 직접 Pandas DataFrame으로 로드합니다. `read_sql_query`와 `read_sql_table`의 기능을 통합한 함수입니다.

**대용량 데이터 조회 시 메모리 문제와 `chunksize` 활용:**
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
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:3306/"
    f"{os.getenv('DB_NAME')}"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

try:
    # SQL 쿼리를 사용하여 데이터 로드
    sql_query = "SELECT employee_id, first_name, last_name, salary FROM employees"
    
    # chunksize를 사용하여 대용량 데이터 처리
    # chunk_iter는 DataFrame 객체가 아닌 이터레이터(Iterator)를 반환합니다.
    chunk_iter = pd.read_sql(sql_query, engine, chunksize=1000) # 1000개 로우씩 나누어 읽기
    
    print("\nProcessing large dataset in chunks:")
    for i, chunk_df in enumerate(chunk_iter):
        print(f"  Processing chunk {i+1} with {len(chunk_df)} rows")
        # 각 chunk에 대한 데이터 처리 로직 (예: 분석, 저장, 필터링 등)
        # print(chunk_df.head())

except Exception as e:
    print(f"Error loading data with Pandas: {e}")
finally:
    if engine:
        engine.dispose() # 연결 풀 닫기 (매우 중요!)
```

**`pd.read_sql_query()`와 `pd.read_sql_table()` (명확한 의도 전달):**
`pd.read_sql()`은 `pd.read_sql_query()`와 `pd.read_sql_table()`의 기능을 통합한 함수입니다. 하지만 코드의 의도를 더 명확히 하고, 특정 상황에서 성능상 이점을 가질 수 있으므로 이들을 구분하여 사용하는 것이 좋습니다.

*   **`pd.read_sql_query(sql, con)`:** SQL 쿼리 문자열을 직접 실행하여 결과를 DataFrame으로 가져옵니다. 복잡한 `JOIN`, `WHERE` 조건, 집계 등이 포함된 쿼리에 적합합니다.
*   **`pd.read_sql_table(table_name, con, columns=None, index_col=None, ...)`:** 데이터베이스의 특정 테이블 전체를 DataFrame으로 가져옵니다. `columns` 인자를 사용하여 필요한 컬럼만 선택할 수 있습니다. 테이블의 모든 데이터를 가져올 때 코드의 가독성이 높습니다.

```python
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:3306/"
    f"{os.getenv('DB_NAME')}"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

try:
    # pd.read_sql_query 예시 (SQL 쿼리 사용)
    df_query = pd.read_sql_query("SELECT employee_id, first_name, salary FROM employees WHERE department_id = 1", engine)
    print("\nDataFrame from read_sql_query:")
    print(df_query.head())

    # pd.read_sql_table 예시 (테이블 이름 사용, 필요한 컬럼만 지정)
    df_table = pd.read_sql_table('departments', engine, columns=['department_id', 'department_name'])
    print("\nDataFrame from read_sql_table:")
    print(df_table.head())

finally:
    if engine:
        engine.dispose() # 연결 풀 닫기
```


### 2.2. `df.to_sql()`: DataFrame을 데이터베이스 테이블로 저장

`DataFrame.to_sql()` 메서드는 Pandas DataFrame의 데이터를 SQL 데이터베이스 테이블로 저장합니다. 새로운 테이블을 생성하거나 기존 테이블에 데이터를 추가/교체할 수 있습니다.

```python
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}"
    f"{os.getenv('DB_HOST')}:3306/"
    f"{os.getenv('DB_NAME')}"
)
engine = create_engine(DATABASE_URL)

try:
    # 새로운 데이터프레임 생성
    new_employees_data = pd.DataFrame({
        'first_name': ['Frank', 'Gina'],
        'last_name': ['White', 'Black'],
        'hire_date': ['2024-05-01', '2024-05-10'],
        'job_id': ['DEV', 'HR'],
        'salary': [70000.00, 60000.00],
        'department_id': [1, 2]
    })

    # 데이터베이스에 저장
    # if_exists='append': 테이블이 존재하면 데이터 추가
    # if_exists='replace': 테이블이 존재하면 삭제 후 새로 생성
    # if_exists='fail': 테이블이 존재하면 에러 발생
    new_employees_data.to_sql('employees', engine, if_exists='append', index=False, dtype={'hire_date': Date}) # dtype 명시적 지정 예시
    print("\nNew employees data inserted into database.")

    # 저장된 데이터 확인
    df_check = pd.read_sql("SELECT * FROM employees ORDER BY employee_id DESC LIMIT 2", engine)
    print(df_check)

except Exception as e:
    print(f"Error saving data with Pandas: {e}")
finally:
    engine.dispose() # 연결 풀 닫기
```

**`df.to_sql()` 성능 최적화 (대량 데이터 삽입 시 실무 필수):**
`df.to_sql()`은 Pandas DataFrame의 데이터를 SQL 데이터베이스 테이블로 저장하는 매우 편리한 메서드입니다. 하지만 대량의 데이터를 삽입할 때는 기본 동작 방식(로우별 `INSERT` 문 실행)으로 인해 성능 문제가 발생할 수 있습니다. 다음 방법들을 통해 성능을 획기적으로 개선할 수 있습니다.

1.  **`method='multi'` 사용 (가장 먼저 고려):**
    Pandas 0.23.0부터 `method='multi'` 옵션을 지원합니다. 이는 여러 로우를 하나의 `INSERT` 문으로 묶어 실행(multi-value insert)하여 데이터베이스와의 네트워크 왕복 횟수를 줄여줍니다. 대량 삽입 시 가장 먼저 고려해야 할 옵션이며, 대부분의 경우 상당한 성능 향상을 가져옵니다.
    ```python
    # 예시: method='multi' 사용
    new_employees_data.to_sql('employees', engine, if_exists='append', index=False, method='multi')
    ```
    **주의:** `method='multi'`는 데이터베이스의 `max_allowed_packet` 설정에 따라 한 번에 삽입할 수 있는 로우 수에 제한이 있을 수 있습니다. 너무 많은 로우를 한 번에 삽입하려 하면 오류가 발생할 수 있으므로, 적절한 `chunksize`와 함께 사용하는 것이 좋습니다.

2.  **`LOAD DATA INFILE` 활용 (MySQL 특화, 최고 성능):**
    MySQL에서 가장 빠른 대량 데이터 로드 방법 중 하나입니다. DataFrame을 CSV 파일로 저장한 후, MySQL의 `LOAD DATA LOCAL INFILE` 명령을 사용하여 직접 데이터베이스로 로드합니다. 이 방법은 `pymysql` 드라이버를 통해 실행할 수 있습니다.
    ```python
    import csv

    # DataFrame을 임시 CSV 파일로 저장
    temp_csv_path = 'temp_employees.csv'
    new_employees_data.to_csv(temp_csv_path, index=False, encoding='utf8', quoting=csv.QUOTE_MINIMAL)

    try:
        with engine.connect() as connection:
            # LOAD DATA LOCAL INFILE 쿼리 실행
            # LOCAL 키워드는 클라이언트에서 서버로 파일을 전송하도록 허용합니다.
            # MySQL 서버와 클라이언트 모두 local_infile=1 설정이 필요할 수 있습니다.
            load_data_sql = f"""
            LOAD DATA LOCAL INFILE '{temp_csv_path.replace('\', '/')}' -- Windows 경로 문제 해결
            INTO TABLE employees
            FIELDS TERMINATED BY ','
            ENCLOSED BY '"' -- Corrected escaping for double quote
            LINES TERMINATED BY '\n' -- Corrected escaping for newline
            IGNORE 1 ROWS; -- 헤더 로우 무시
            """
            connection.execute(load_data_sql)
            connection.commit()
            print(f"\nData loaded using LOAD DATA LOCAL INFILE from {temp_csv_path}")
    except Exception as e:
        print(f"Error using LOAD DATA LOCAL INFILE: {e}")
    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path) # 임시 파일 삭제
    ```
    **주의:** `LOAD DATA LOCAL INFILE`은 보안상의 이유로 MySQL 서버와 클라이언트 모두 `local_infile` 설정이 활성화되어 있어야 합니다. 운영 환경에서는 보안 정책을 검토 후 사용해야 합니다.

3.  **`executemany()` 또는 ORM의 `bulk_insert_mappings()` 사용 (더 세밀한 제어):**
    `pd.to_sql()` 대신 `pymysql`의 `cursor.executemany()`나 SQLAlchemy ORM의 `session.bulk_insert_mappings()`를 직접 사용하여 대량 삽입을 구현할 수도 있습니다. 이들은 Pandas DataFrame을 리스트 오브 딕셔너리(list of dictionaries) 또는 튜플(list of tuples) 형태로 변환하여 전달하는 방식입니다. 이 방법은 `pd.to_sql()`의 내부 구현보다 더 세밀한 제어가 가능하며, 특정 상황에서 성능 이점을 제공할 수 있습니다.
    ```python
    # 예시: pymysql의 executemany() 사용
    # data_to_insert = new_employees_data.values.tolist() # DataFrame을 리스트의 리스트로 변환
    # insert_sql = "INSERT INTO employees (...) VALUES (%s, %s, ...)"
    # cursor.executemany(insert_sql, data_to_insert)
    ```

**`dtype` 파라미터의 중요성:**
`df.to_sql()` 사용 시 `dtype` 파라미터를 명시적으로 지정하는 것이 좋습니다. Pandas의 데이터 타입과 SQL 데이터베이스의 데이터 타입 간의 정확한 매핑을 보장하여, 데이터 손실이나 예상치 못한 오류를 방지할 수 있습니다. 특히 날짜/시간, 불리언, 그리고 정수형의 크기(INT, BIGINT 등)를 명확히 지정하는 것이 중요합니다.

대량 데이터 삽입 시에는 `method='multi'`를 우선적으로 고려하고, 더 높은 성능이 필요하거나 특정 DBMS의 기능을 활용해야 할 경우 `LOAD DATA INFILE` 또는 `executemany()`/`bulk_insert_mappings()`를 직접 구현하는 것을 검토할 수 있습니다.



### 2.3. 데이터 분석 파이프라인에서의 활용 예시

Pandas와 SQL 연동은 데이터 수집, 전처리, 분석, 결과 저장 등 데이터 분석 파이프라인의 여러 단계에서 활용될 수 있습니다.

1.  **데이터 수집:** 데이터베이스에서 필요한 데이터를 `pd.read_sql()`로 가져와 DataFrame으로 만듭니다.
2.  **데이터 전처리/분석:** Pandas의 강력한 기능을 사용하여 DataFrame에서 데이터 정제, 변환, 집계, 모델링 등을 수행합니다.
3.  **결과 저장:** 분석 결과를 새로운 테이블로 `df.to_sql()`을 사용하여 데이터베이스에 저장하거나, 기존 테이블을 업데이트합니다.
4.  **자동화:** 이 모든 과정을 파이썬 스크립트로 작성하여 정기적으로 실행되도록 자동화할 수 있습니다.

## 3. 파이썬-SQL 데이터 타입 매핑 및 변환 (데이터 무결성 및 호환성)

파이썬 객체와 SQL 데이터베이스의 데이터 타입은 서로 다릅니다. 이들 간의 올바른 매핑과 변환을 이해하는 것은 데이터 연동 시 발생할 수 있는 오류를 방지하고, 데이터 무결성을 유지하며, 쿼리 성능을 최적화하는 데 매우 중요합니다.

### 3.1. 주요 데이터 타입 매핑 (Python to SQL)

| 파이썬 타입 | 일반적인 SQL 타입 (MySQL 기준) | 비고 |
| :--- | :--- | :--- |
| `int` | `INT`, `BIGINT` | 파이썬 `int`는 크기 제한이 없으므로, SQL에서는 저장될 값의 범위에 따라 적절한 정수형 선택 (`TINYINT`, `SMALLINT`, `MEDIUMINT`, `INT`, `BIGINT`) |
| `float` | `FLOAT`, `DOUBLE`, `DECIMAL` | `float`는 부동 소수점 오차 가능성이 있으므로, 정확한 금융 계산 등에는 `DECIMAL` 사용을 강력히 권장 |
| `str` | `VARCHAR`, `TEXT`, `CHAR` | 문자열 길이와 인코딩 (`utf8mb4` 권장) 고려. `TEXT`는 대용량 텍스트 저장용 |
| `bool` | `TINYINT(1)` (0 또는 1) | MySQL은 `BOOLEAN` 타입을 `TINYINT(1)`로 처리하며, `True`는 1, `False`는 0으로 매핑됨 |
| `None` | `NULL` | SQL의 `NULL`에 매핑. `NOT NULL` 제약조건이 있는 컬럼에는 `None` 삽입 불가 |
| `datetime.date` | `DATE` | 날짜 정보만 저장 |
| `datetime.datetime` | `DATETIME`, `TIMESTAMP` | `DATETIME`은 고정 길이, `TIMESTAMP`는 타임존 고려 및 범위 제한 있음 |
| `list`, `dict` | `JSON` (MySQL 5.7+), `TEXT` | `JSON` 타입이 지원되면 가장 좋음. 없다면 파이썬에서 `json.dumps()`로 직렬화하여 `TEXT` 컬럼에 저장 후 `json.loads()`로 역직렬화 |

### 3.2. 일반적인 문제점 및 해결 방안 (NULL, 날짜/시간, 불리언, 인코딩)

*   **`NULL` 처리:**
    *   **문제:** 파이썬의 `None`은 SQL의 `NULL`로 잘 매핑되지만, SQL에서 `NULL`이 허용되지 않는 (`NOT NULL` 제약조건이 있는) 컬럼에 `None`을 삽입하려 하면 오류가 발생합니다.
    *   **해결:** `CREATE TABLE` 시 `NOT NULL` 제약조건을 명확히 정의하고, 파이썬 코드에서 `None`이 아닌 유효한 값을 제공하도록 유효성 검사를 수행하거나, 기본값(Default Value)을 설정합니다.

*   **날짜/시간 형식 불일치:**
    *   **문제:** 파이썬의 `datetime` 객체와 SQL의 날짜/시간 타입 간의 형식 불일치로 오류가 발생할 수 있습니다. 특히 문자열로 날짜를 다룰 때 흔하며, 타임존(Timezone) 문제도 발생할 수 있습니다.
    *   **해결:** `datetime` 객체를 직접 전달하는 것이 가장 좋습니다. 대부분의 DB-API 드라이버나 ORM은 이를 자동으로 처리합니다. 만약 문자열로 변환해야 한다면, SQL 드라이버/ORM이 지원하는 표준 형식(예: `YYYY-MM-DD HH:MM:SS`)으로 변환하여 전달합니다. SQL의 `DATE_FORMAT` 함수나 파이썬의 `strftime`/`strptime` 함수를 활용하여 형식을 맞춥니다.

*   **불리언(Boolean) 값:**
    *   **문제:** MySQL은 `BOOLEAN` 타입을 `TINYINT(1)`로 처리하므로, 파이썬의 `True`/`False`가 1/0으로 올바르게 변환되는지 확인해야 합니다.
    *   **해결:** 대부분의 드라이버/ORM은 자동으로 처리하지만, 간혹 문제가 발생하면 명시적으로 1 또는 0으로 변환하여 저장합니다. (예: `int(python_boolean_value)`)

*   **인코딩 문제:**
    *   **문제:** 파이썬 문자열과 데이터베이스 문자열 인코딩(예: `utf8mb4`)이 일치하지 않으면 한글 깨짐 현상이나 삽입 오류가 발생할 수 있습니다.
    *   **해결:** 데이터베이스 연결 시 `charset='utf8mb4'`와 같이 명시적으로 인코딩을 지정하고, 파이썬 코드에서도 문자열을 처리할 때 동일한 인코딩을 사용합니다.

*   **데이터 유효성 검사 (Python 단에서 선행):**
데이터베이스에 데이터를 삽입하기 전에 파이썬 코드 단에서 미리 데이터 유효성 검사를 수행하는 것이 좋습니다. 이는 데이터베이스 제약조건에만 의존하는 것보다 오류를 더 빠르게 감지하고, 사용자에게 더 친절한 피드백을 제공하며, 데이터베이스의 부하를 줄이는 데 도움이 됩니다. (관련 내용은 [01_Notes/01_Python/0512정리.md의 '4. 데이터 유효성 검사 (Data Validation)'](../01_Python/0512정리.md#4-데이터-유효성-검사-data-validation) 섹션을 참조하세요.)
