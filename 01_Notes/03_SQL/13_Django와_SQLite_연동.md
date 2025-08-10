<h2>SQL 핵심 문법: Django와 SQLite 연동 (실무 활용 중심)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
<p>이 문서는 서버 없이 동작하는 경량 데이터베이스 <strong>SQLite의 특징과 파이썬 <code>sqlite3</code> 모듈을 활용하는 방법</strong>을 심도 있게 다룹니다. 특히 풀스택 프레임워크인 Django에서 SQLite를 기본 개발 데이터베이스로 사용하는 방법과 연동 과정을 상세한 예제와 함께 설명하여, <strong>프로토타이핑 및 소규모 애플리케이션 개발 능력</strong>을 기르고, <strong>운영 환경 전환 시 고려사항</strong>을 이해하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. SQLite: 서버 없는 경량 데이터베이스](#1-sqlite-서버-없는-경량-데이터베이스)
  - [1.1. SQLite 재정의: 데이터베이스를 넘어 '애플리케이션 파일 포맷'으로](#11-sqlite-재정의-데이터베이스를-넘어-애플리케이션-파일-포맷으로)
  - [1.2. SQLite의 핵심 특징과 그 이면의 동작 원리](#12-sqlite의-핵심-특징과-그-이면의-동작-원리)
  - [1.3. SQLite의 한계와 반드시 피해야 할 시나리오](#13-sqlite의-한계와-반드시-피해야-할-시나리오)
  - [1.4. SQLite, 언제 사용해야 하는가? (전략적 선택 가이드)](#14-sqlite-언제-사용해야-하는가-전략적-선택-가이드)
- [2. 파이썬과 SQLite 연동: `sqlite3` 모듈](#2-파이썬과-sqlite-연동-sqlite3-모듈)
  - [2.1. 연결과 설정: 단순함을 넘어서](#21-연결과-설정-단순함을-넘어서)
  - [2.2. `with` 문을 활용한 안전한 리소스 및 트랜잭션 관리](#22-with-문을-활용한-안전한-리소스-및-트랜잭션-관리)
  - [2.3. `sqlite3` 모듈 고급 활용](#23-sqlite3-모듈-고급-활용)
- [3. Django와 SQLite 연동](#3-django와-sqlite-연동)
  - [3.1. Django의 철학: 설정보다 관례, 그리고 빠른 시작](#31-django의-철학-설정보다-관례-그리고-빠른-시작)
  - [3.2. Django ORM: SQL을 파이썬 객체로 대체하다](#32-django-orm-sql을-파이썬-객체로-대체하다)
  - [3.3. 마이그레이션 시스템: 코드와 DB 스키마의 동기화](#33-마이그레이션-시스템-코드와-db-스키마의-동기화)
  - [3.4. Django가 SQLite의 한계를 보완하는 방법](#34-django가-sqlite의-한계를-보완하는-방법)
- [4. 실무적 고려사항: 개발에서 프로덕션까지](#4-실무적-고려사항-개발에서-프로덕션까지)
  - [4.1. 개발용(SQLite)과 운영용(PostgreSQL/MySQL) DB 분리](#41-개발용sqlite과-운영용postgresqlmysql-db-분리)
  - [4.2. SQLite의 데이터 타입과 Django 필드](#42-sqlite의-데이터-타입과-django-필드)
  - [4.3. 동시성 문제와 파일 잠금](#43-동시성-문제와-파일-잠금)

---

## 1. SQLite: 서버 없는 경량 데이터베이스 (심층 탐구)

### 1.1. SQLite 재정의: 데이터베이스를 넘어 '애플리케이션 파일 포맷'으로

SQLite를 단순히 '가벼운 데이터베이스'로만 이해하는 것은 그 잠재력을 과소평가하는 것입니다. SQLite의 창시자인 D. Richard Hipp는 SQLite를 **"애플리케이션 파일 포맷(Application File Format)"** 이라고 설명합니다. 즉, `JSON`, `CSV`, `XML`처럼 데이터를 디스크에 저장하는 '파일 형식'의 일종이지만, 여기에 **강력한 SQL 쿼리 엔진과 트랜잭션 기능이 내장된 형태**라는 것입니다.

이 관점은 SQLite의 본질을 이해하는 데 매우 중요합니다. 우리는 `mysqld`나 `postgres` 같은 별도의 서버 프로세스에 접속하는 것이 아니라, 내 애플리케이션이 직접 라이브러리를 통해 `.sqlite3` 파일을 읽고 쓰는 것입니다. 모든 작업은 프로세스 내 함수 호출로 이루어지므로 네트워크 지연이 없으며, 설정과 관리가 극도로 단순해집니다.

### 1.2. SQLite의 핵심 특징과 그 이면의 동작 원리

-   **서버리스 (Serverless) & 제로 설정 (Zero-configuration)**: 별도의 서버 설치, 설정, 관리 작업이 전혀 필요 없습니다. `import sqlite3` 코드 한 줄이면 모든 준비가 끝납니다. 이는 개발 환경 구축 시간을 획기적으로 단축시키고, 배포를 매우 단순하게 만듭니다.

-   **단일 파일과 트랜잭션 원자성 (Atomicity)**: 데이터베이스의 모든 것(스키마, 데이터, 인덱스 등)이 단일 파일에 저장됩니다. 파일 복사만으로 완벽한 백업이 가능합니다. 더 중요한 것은, SQLite는 **ACID(원자성, 일관성, 고립성, 지속성)**를 완벽하게 지원한다는 점입니다. 쓰기 작업 중 정전이나 시스템 충돌이 발생해도, **저널(Journal) 파일** 시스템을 통해 데이터베이스 파일이 손상되지 않도록 보장합니다.
    -   **Rollback Journal (기본)**: 변경 전 원본 데이터를 `-journal` 파일에 백업한 후, 데이터베이스 파일을 수정합니다. 커밋 시 저널 파일이 삭제됩니다. 충돌이 발생하면, 재시작 시 저널 파일을 보고 원본 데이터로 복구합니다.
    -   **WAL (Write-Ahead Logging)**: 아래 동시성 섹션에서 자세히 다룹니다.

-   **엄격한 표준 준수와 이식성**: SQLite는 표준 SQL-92 문법 대부분을 지원합니다. 또한, 데이터베이스 파일 형식은 플랫폼 간(Windows, macOS, Linux, 모바일 등) 완벽하게 호환되므로, 파일을 복사하는 것만으로 어떤 환경에서든 데이터를 동일하게 사용할 수 있습니다.

### 1.3. SQLite의 한계와 반드시 피해야 할 시나리오

SQLite의 단순함은 강력한 장점이지만, 특정 상황에서는 명백한 한계로 작용합니다.

#### 1.3.1. 심층 탐구: 동시성(Concurrency)과 잠금 메커니즘

SQLite의 가장 중요한 한계는 동시 쓰기 처리 능력입니다.

-   **기본 잠금 (Rollback Mode)**: 기본 모드에서 SQLite는 쓰기 작업을 시작하면 **데이터베이스 파일 전체에 잠금(Lock)**을 겁니다. 이 시간 동안 다른 모든 연결은 쓰기 작업을 할 수 없으며, 일정 시간 대기하다 `database is locked` 오류를 발생시킵니다. 즉, **동시에 여러 곳에서 쓰기 작업을 처리할 수 없습니다.**

-   **해결책: WAL (Write-Ahead Logging) 모드**
    WAL은 SQLite의 동시성을 크게 향상시키는 저널링 모드입니다.
    -   **동작 방식**: 변경사항을 즉시 데이터베이스 파일에 쓰지 않고, 별도의 `-wal` 파일에 순차적으로 기록합니다. 읽기 작업은 데이터베이스 파일을 직접 읽고, 쓰기 작업은 `-wal` 파일에만 기록합니다. 주기적으로 `-wal` 파일의 변경사항을 데이터베이스 파일에 병합(Checkpointing)합니다.
    -   **장점**: 이 방식 덕분에 **한 명의 쓰기(Writer)와 여러 명의 읽기(Reader)가 동시에 작업을 수행**할 수 있게 됩니다. 웹 애플리케이션처럼 읽기 작업이 쓰기 작업보다 훨씬 빈번한 환경에서 성능을 크게 향상시킵니다.
    -   **활성화**: `PRAGMA journal_mode=WAL;` 쿼리를 실행하여 간단히 활성화할 수 있습니다. Django 4.2부터는 SQLite 사용 시 WAL 모드가 기본으로 활성화됩니다.

    ```python
    import sqlite3
    conn = sqlite3.connect('my_wal_database.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;") # WAL 모드 활성화
    # ... 이후 작업 수행 ...
    conn.close()
    ```
    **결론**: WAL 모드를 사용하더라도, 여전히 **동시 쓰기는 한 번에 하나만 가능**합니다. 동시 쓰기 요청이 많은 서비스에는 SQLite가 적합하지 않습니다.

#### 1.3.2. 심층 탐구: 동적 타이핑과 타입 선호도(Type Affinity)

대부분의 RDBMS는 정적 타이핑(Static Typing)을 사용하여 컬럼에 지정된 타입의 데이터만 저장할 수 있도록 강제합니다. 하지만 SQLite는 **동적 타이핑(Dynamic Typing)**을 사용하며, 컬럼 타입은 강제 사항이 아닌 **타입 선호도(Type Affinity)**로 동작합니다.

-   **5가지 타입 선호도**: `TEXT`, `NUMERIC`, `INTEGER`, `REAL`, `BLOB`
-   **동작 방식**: `INTEGER` 선호도를 가진 컬럼에 문자열 '123'을 넣으면, SQLite는 이를 정수 123으로 변환하려고 시도합니다. 하지만 변환할 수 없는 문자열 'hello'를 넣어도 오류 없이 그대로 저장합니다.

    ```python
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER, data INTEGER)")
    
    # INTEGER 선호도 컬럼에 정수 저장 (정상)
    cursor.execute("INSERT INTO test VALUES (1, 123)")
    
    # INTEGER 선호도 컬럼에 문자열 '456' 저장 -> 정수 456으로 변환되어 저장됨
    cursor.execute("INSERT INTO test VALUES (2, '456')")
    
    # INTEGER 선호도 컬럼에 문자열 'hello' 저장 -> 변환 없이 문자열 그대로 저장됨!
    cursor.execute("INSERT INTO test VALUES (3, 'hello')")

    results = cursor.execute("SELECT id, data, typeof(data) FROM test").fetchall()
    # [(1, 123, 'integer'), (2, 456, 'integer'), (3, 'hello', 'text')]
    print(results)
    ```
-   **실무적 의미**: 유연성을 제공하지만, 애플리케이션 레벨에서 데이터 유효성 검사를 철저히 하지 않으면 데이터 무결성이 깨질 수 있습니다. Django와 같은 ORM은 모델 필드 레벨에서 유효성 검사를 수행하여 이 문제를 완화해 줍니다.

#### 1.3.3. 기타 주요 한계

-   **네트워크 파일 시스템에서의 사용 금지**: NFS, SMB 등 네트워크 공유 폴더에 데이터베이스 파일을 두고 여러 클라이언트가 접근하는 방식은 파일 잠금 메커니즘의 한계로 인해 데이터 손상을 유발할 수 있어 절대 권장되지 않습니다.
-   **고급 기능 부재**: `RIGHT JOIN`, `FULL OUTER JOIN`, 저장 프로시저, 세분화된 사용자 권한 관리 등 대규모 RDBMS가 제공하는 고급 기능들이 없습니다.

### 1.4. SQLite, 언제 사용해야 하는가? (전략적 선택 가이드)

-   **애플리케이션 개발 및 프로토타이핑**: 별도 DB 서버 없이 즉시 개발을 시작할 수 있어 생산성이 극대화됩니다.
-   **모바일, 데스크톱, 임베디드 시스템의 내장 DB**: 경량성과 안정성 덕분에 로컬 데이터 저장소로 가장 널리 사용됩니다.
-   **데이터 분석을 위한 '임시 SQL 쿼리 엔진'**:
    -   대용량 CSV나 여러 개의 파일을 Pandas로 다루기 복잡할 때, 이들을 메모리 내 SQLite DB에 적재하면 복잡한 `JOIN`이나 집계 쿼리를 SQL로 손쉽게 처리할 수 있습니다.
        ```python
        import pandas as pd
        import sqlite3

        # 1. CSV를 DataFrame으로 읽기
        df = pd.read_csv('my_large_data.csv')
        
        # 2. 인메모리 SQLite DB에 적재
        conn = sqlite3.connect(':memory:')
        df.to_sql('my_data', conn, index=False)
        
        # 3. Pandas로 하기 복잡한 쿼리를 SQL로 실행
        query = "SELECT category, AVG(price) FROM my_data GROUP BY category;"
        result_df = pd.read_sql_query(query, conn)
        
        conn.close()
        ```
-   **테스트 자동화**: 각 테스트 케이스마다 독립적인 인메모리 DB(`:memory:`)를 빠르게 생성하고 폐기할 수 있어, 빠르고 격리된 테스트 환경을 구축하는 데 이상적입니다.
-   **데이터 아카이빙 및 교환 포맷**: 잘 정의된 스키마, 인덱스, 트랜잭션을 지원하는 단일 파일이므로, 복잡한 구조의 데이터를 교환하거나 장기 보관할 때 `CSV`나 `JSON`보다 훨씬 안정적이고 효율적인 포맷이 될 수 있습니다.

## 2. 파이썬과 SQLite 연동: `sqlite3` 모듈 (심화)

파이썬은 `sqlite3` 모듈을 표준 라이브러리로 제공하므로, 별도의 드라이버 설치 없이 SQLite의 모든 기능을 활용할 수 있습니다. 기본 CRUD를 넘어, 견고하고 효율적인 애플리케이션을 만들기 위한 고급 기능과 모범 사례를 알아봅니다.

### 2.1. 연결과 설정: 단순함을 넘어서

`sqlite3.connect()` 함수는 단순한 파일 연결 이상의 기능을 제공합니다.

```python
import sqlite3
import datetime

# 실무에서 권장되는 연결 설정
conn = sqlite3.connect(
    'my_database.db',
    timeout=10,  # 10초간 잠금 대기 (동시성 문제 완화)
    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    isolation_level=None # autocommit 모드 활성화 (아래 설명 참조)
)
conn.execute("PRAGMA foreign_keys = ON;") # 외래 키 제약조건 활성화
conn.execute("PRAGMA journal_mode = WAL;") # WAL 모드 활성화로 읽기/쓰기 동시성 향상
```
- **`timeout`**: 다른 트랜잭션이 데이터베이스를 잠그고 있을 때, 지정된 시간(초)만큼 대기합니다. 기본값은 5초이며, 이 시간을 초과하면 `OperationalError: database is locked` 예외가 발생합니다.
- **`detect_types`**: 파이썬 타입과 SQL 타입 간의 자동 변환을 활성화합니다.
    - `sqlite3.PARSE_DECLTYPES`: `CREATE TABLE` 문에 명시된 타입(예: `TIMESTAMP`)을 인식하여 데이터를 변환합니다.
    - `sqlite3.PARSE_COLNAMES`: `SELECT` 문의 컬럼명(예: `col as "col [timestamp]"`)을 보고 타입을 변환합니다.
- **`isolation_level`**: 트랜잭션 격리 수준을 제어합니다. `None`으로 설정하면 **autocommit 모드**로 동작하여, `INSERT/UPDATE/DELETE` 문이 실행 즉시 반영됩니다. 명시적인 `conn.commit()` 호출이 필요 없어 편리하지만, 여러 작업을 하나의 트랜잭션으로 묶으려면 수동으로 `BEGIN/COMMIT`을 관리해야 합니다.

### 2.2. `with` 문을 활용한 안전한 리소스 및 트랜잭션 관리

`with` 문은 리소스 누수와 트랜잭션 오류를 방지하는 가장 파이썬스러운 방법입니다.

- **`with sqlite3.connect(...) as conn:`**: 이 블록이 끝나면 연결(`conn`)이 **자동으로 닫힙니다.**
- **`with conn:`**: 이 블록은 **트랜잭션 범위**를 정의합니다. 블록이 성공적으로 완료되면 자동으로 `COMMIT`되고, 블록 내에서 예외가 발생하면 자동으로 `ROLLBACK`됩니다. (`isolation_level`이 `None`이 아닐 때 동작)

```python
# with 문을 사용한 가장 안전하고 간결한 패턴
try:
    # autocommit 모드를 비활성화하기 위해 isolation_level을 명시적으로 설정
    with sqlite3.connect('my_database.db', isolation_level='DEFERRED') as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        
        # with conn: 블록으로 원자적 트랜잭션 관리
        with conn:
            conn.execute("INSERT INTO users (username, email) VALUES (?, ?)", ('diana', 'diana@example.com'))
            # 의도적 오류 발생 (UNIQUE 제약조건 위반)
            conn.execute("INSERT INTO users (username, email) VALUES (?, ?)", ('diana', 'diana_duplicate@example.com'))
            
except sqlite3.IntegrityError as e:
    # UNIQUE, NOT NULL, FOREIGN KEY 등 제약조건 위반 시 발생하는 구체적인 예외 처리
    print(f"Transaction failed due to data integrity issue: {e}")
except sqlite3.Error as e:
    print(f"A database error occurred: {e}")
finally:
    print("Connection is guaranteed to be closed.")
```

### 2.3. `sqlite3` 모듈 고급 활용

#### 2.3.1. 사용자 정의 SQL 함수 (User-Defined Functions)

SQLite에는 없는 함수(예: 정규표현식)를 파이썬 함수로 직접 만들어 SQL 쿼리 내에서 사용할 수 있습니다.

```python
import re

def regexp(pattern, text):
    """정규표현식 매칭을 위한 함수"""
    return re.search(pattern, text) is not None

# 인자 2개, 함수 이름 'REGEXP'로 사용자 정의 함수 등록
conn.create_function("REGEXP", 2, regexp)

# SQL 쿼리에서 파이썬 함수 사용
cursor = conn.execute("SELECT username FROM users WHERE email REGEXP ?", (r'@example\.com$',))
for row in cursor:
    print(f"Found user with example.com email: {row['username']}")
```

#### 2.3.2. 커스텀 Row Factory: 결과를 객체로 바로 받기

기본 `sqlite3.Row`는 딕셔너리처럼 동작하지만, 진정한 파이썬 객체는 아닙니다. `conn.row_factory`를 커스터마이징하면 쿼리 결과를 원하는 `dataclass`나 일반 클래스 객체로 즉시 변환할 수 있습니다.

```python
from dataclasses import dataclass

@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime.date

# 연결 설정 시 row_factory를 지정
conn.row_factory = lambda cursor, row: User(*row)

cursor = conn.execute("SELECT id, username, email, created_at FROM users")
users: list[User] = cursor.fetchall()

for user in users:
    # 이제 user는 완전한 User 객체!
    print(f"User object: id={user.id}, name={user.username}, created_at_type={type(user.created_at)}")
```

#### 2.3.3. 자동 타입 변환 (Adapters and Converters)

`detect_types`를 활성화하면, `sqlite3`는 특정 타입 이름(컬럼 선언 또는 별칭)을 보고 파이썬 객체와 자동으로 변환합니다.

```python
# 1. 테이블 생성 시 타입 명시
conn.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    event_name TEXT,
    event_date DATE, -- 'DATE' 타입 명시
    created_at TIMESTAMP -- 'TIMESTAMP' 타입 명시
)
""")

# 2. 파이썬 datetime 객체를 직접 사용
today = datetime.date.today()
now = datetime.datetime.now()
conn.execute("INSERT INTO events (event_name, event_date, created_at) VALUES (?, ?, ?)",
             ('product launch', today, now))
conn.commit()

# 3. 조회 시 자동으로 파이썬 datetime 객체로 변환됨
event = conn.execute("SELECT * FROM events").fetchone()
print(f"Event date type: {type(event['event_date'])}")     # <class 'datetime.date'>
print(f"Created_at type: {type(event['created_at'])}") # <class 'datetime.datetime'>
```
이처럼 `sqlite3` 모듈은 단순한 DB 연동 도구를 넘어, 파이썬의 강력한 기능을 SQL과 결합하여 생산성을 크게 높일 수 있는 다양한 고급 기능들을 제공합니다.

## 3. Django와 SQLite 연동 (심화)

Django가 개발 환경의 기본 데이터베이스로 SQLite를 채택한 것은 **'개발자 경험'**과 **'빠른 프로토타이핑'**을 최우선으로 고려한 탁월한 설계 결정입니다. Django의 강력한 추상화 계층(ORM, 마이그레이션)이 어떻게 SQLite의 단순성과 결합하여 시너지를 내는지, 그 내부 동작을 중심으로 심도 있게 알아봅니다.

### 3.1. Django의 철학: 설정보다 관례, 그리고 빠른 시작

Django 프로젝트를 시작하면 `settings.py`에 다음과 같은 설정이 자동으로 생성됩니다.

```python
# myproject/settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```
이 설정은 단순한 파일 경로 지정 이상의 의미를 가집니다. `ENGINE`에 명시된 `'django.db.backends.sqlite3'`는 Django가 SQLite와 통신하기 위해 사용할 **데이터베이스 백엔드(어댑터)**입니다. 이 백엔드는 내부적으로 파이썬의 `sqlite3` 모듈을 사용하며, Django ORM이 생성한 쿼리를 SQLite가 이해할 수 있는 SQL 문법으로 변환하고 실행하는 모든 복잡한 과정을 처리합니다.

**실무적 관점**: Django는 이처럼 데이터베이스 백엔드만 교체하면(예: `...postgresql`), 코드 변경 거의 없이 다른 데이터베이스로 전환할 수 있는 **느슨한 결합(Loose Coupling)** 구조를 가지고 있습니다. 개발 초기에는 SQLite로 빠르게 시작하고, 서비스가 성장하면 PostgreSQL 등으로 손쉽게 확장할 수 있는 기반이 바로 이 설정에 담겨 있습니다.

### 3.2. Django ORM: SQL을 파이썬 객체로 대체하다

Django ORM(Object-Relational Mapper)은 개발자가 SQL 쿼리를 직접 작성하는 대신, 파이썬 클래스와 메서드를 사용하여 데이터베이스와 상호작용하게 해주는 강력한 도구입니다.

**시나리오**: `Post`라는 모델(테이블)을 만들고 데이터를 추가한 뒤 조회하는 작업

<details>
<summary><b>1. `sqlite3` 모듈 직접 사용 시</b></summary>

```python
# 1. SQL로 테이블 구조 정의
create_sql = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT,
    published_at TIMESTAMP
);
"""
# 2. SQL로 데이터 삽입
insert_sql = "INSERT INTO posts (title, content, published_at) VALUES (?, ?, ?);"
# 3. SQL로 데이터 조회
select_sql = "SELECT id, title, content FROM posts WHERE title LIKE ?;"

# ... conn.execute(create_sql), conn.execute(insert_sql, ...), ...
```
- **문제점**: 모든 스키마와 로직이 SQL 문자열에 의존합니다. 오타가 발생하기 쉽고, 모델 구조가 변경되면 모든 SQL 문자열을 찾아 수정해야 합니다.
</details>

<details>
<summary><b>2. Django ORM 사용 시</b></summary>

```python
# myapp/models.py
from django.db import models

# 1. 파이썬 클래스로 테이블 구조 정의
class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    published_at = models.DateTimeField(auto_now_add=True)

# python manage.py shell
from myapp.models import Post
from django.utils import timezone

# 2. 파이썬 객체로 데이터 삽입
Post.objects.create(title="My First Post", content="Hello, Django!", published_at=timezone.now())

# 3. 파이썬 메서드로 데이터 조회
posts = Post.objects.filter(title__startswith="My First")
for post in posts:
    print(post.title)
```
- **장점**: SQL이 완전히 사라졌습니다. `models.py`라는 단일 진실 공급원(Single Source of Truth)을 통해 스키마를 관리하며, 모든 데이터 조작은 직관적인 파이썬 코드로 이루어집니다.
</details>

### 3.3. 마이그레이션 시스템: 코드와 DB 스키마의 동기화

Django의 가장 강력한 기능 중 하나는 마이그레이션 시스템입니다. 이는 `models.py`의 변경사항을 추적하여, 데이터베이스 스키마를 변경하는 SQL을 **자동으로 생성하고 적용**하는 메커니즘입니다.

1.  **`python manage.py makemigrations`**: 이 명령은 현재 모델 상태와 마지막 마이그레이션 상태를 비교하여 변경사항을 감지하고, `myapp/migrations/0002_add_author_to_post.py`와 같은 **마이그레이션 파일**을 생성합니다. 이 파일은 단순한 SQL이 아닌, 스키마 변경 작업을 서술하는 파이썬 코드입니다.

    ```python
    # myapp/migrations/0002_...py (예시)
    from django.db import migrations, models
    
    class Migration(migrations.Migration):
        dependencies = [('myapp', '0001_initial')]
        operations = [
            migrations.AddField(
                model_name='post',
                name='author',
                field=models.CharField(max_length=100, default='anonymous'),
            ),
        ]
    ```

2.  **`python manage.py migrate`**: 이 명령은 `django_migrations`라는 특별한 테이블을 참조하여, 아직 적용되지 않은 모든 마이그레이션 파일을 순서대로 실행합니다. 각 마이그레이션 파일의 `operations` 목록이 SQLite(또는 다른 DB)가 이해할 수 있는 `ALTER TABLE ...`과 같은 SQL로 변환되어 실행됩니다.

**내부 동작**: `django_migrations` 테이블에는 적용된 모든 마이그레이션의 이름(예: `myapp.0002_...`)이 기록됩니다. `migrate` 명령은 이 테이블에 없는 최신 마이그레이션만 찾아 적용함으로써, 데이터베이스 스키마가 항상 코드와 일관된 상태를 유지하도록 보장합니다.

### 3.4. Django가 SQLite의 한계를 보완하는 방법

Django ORM은 데이터베이스 백엔드의 차이점을 추상화하여, 개발자가 SQLite의 특정 한계를 직접 신경 쓰지 않도록 돕습니다.

-   **타입 문제**: SQLite의 '타입 선호도'와 달리, Django의 모델 필드(예: `CharField(max_length=200)`)는 **애플리케이션 레벨에서 데이터 유효성 검사를 강제**합니다. 200자가 넘는 문자열을 저장하려고 하면, 데이터베이스에 도달하기 전에 Django의 `ValidationError`가 발생하여 데이터 무결성을 지켜줍니다.

-   **동시성 문제**: Django 4.2 버전부터, SQLite를 사용할 때 **WAL(Write-Ahead Logging) 모드를 자동으로 활성화**합니다. 이는 SQLite의 동시성 성능을 크게 개선하여, 여러 읽기 작업과 하나의 쓰기 작업이 동시에 이루어질 수 있도록 합니다. 이 덕분에 개발 서버나 소규모 트래픽 환경에서 `database is locked` 오류를 마주할 확률이 크게 줄어들었습니다.

-   **기능 제약**: SQLite는 `JSONField`와 같은 고급 데이터 타입을 직접 지원하지 않지만, Django는 해당 필드에 저장될 파이썬 딕셔너리를 `TEXT` 타입으로 직렬화(serialize)하여 저장하고, 조회 시 다시 파이썬 객체로 변환해주는 방식으로 기능을 에뮬레이션합니다. 이를 통해 개발자는 DB 종류에 상관없이 일관된 방식으로 `JSONField`를 사용할 수 있습니다.

## 4. 실무적 고려사항: 개발에서 프로덕션까지 (전환 전략)

SQLite는 훌륭한 개발용 데이터베이스이지만, 대부분의 상용 서비스는 결국 PostgreSQL이나 MySQL과 같은 강력한 서버 기반 RDBMS를 운영 환경(Production)으로 선택합니다. 이 전환 과정은 단순히 `settings.py` 파일 수정만으로 끝나지 않으며, 미리 고려하지 않으면 예상치 못한 장애로 이어질 수 있습니다. 이 섹션에서는 개발에서 프로덕션으로의 순조로운 전환을 위한 핵심 전략을 다룹니다.

### 4.1. 환경 분리: 설정 관리의 첫걸음

가장 먼저 해야 할 일은 개발(Development), 테스트(Test), 운영(Production) 환경의 설정을 분리하는 것입니다. 데이터베이스 접속 정보, 시크릿 키, 디버그 모드 등 환경마다 달라져야 하는 값들을 코드에 하드코딩하는 것은 매우 위험합니다.

-   **모범 사례**: `python-dotenv`와 `django-environ` 라이브러리 활용
    1.  **`.env` 파일 생성**: 프로젝트 루트에 `.env` 파일을 만들고 민감한 정보를 저장합니다. 이 파일은 `.gitignore`에 추가하여 절대 Git에 커밋하지 않습니다.
        ```ini
        # .env
        SECRET_KEY='your-production-secret-key'
        DEBUG=False
        DATABASE_URL='postgres://user:password@host:port/dbname'
        ```
    2.  **`settings.py` 수정**: `django-environ`을 사용하여 `.env` 파일의 값을 읽어옵니다.
        ```python
        # settings.py
        import environ
        import os

        env = environ.Env(
            # set casting, default value
            DEBUG=(bool, False)
        )
        
        # .env 파일 읽기
        environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

        SECRET_KEY = env('SECRET_KEY')
        DEBUG = env('DEBUG')

        # DATABASE_URL 환경 변수를 파싱하여 DATABASES 설정에 자동으로 반영
        DATABASES = {'default': env.db()}
        
        # 개발 환경에서는 SQLite를 기본값으로 사용
        if DATABASES['default']['ENGINE'] == 'django.db.backends.sqlite3':
            print("Running with development SQLite database.")
        ```
    이제 로컬 개발 환경에서는 `.env` 파일 없이 SQLite를 사용하고, 서버 환경에서는 `.env` 파일을 통해 프로덕션 DB 정보를 주입할 수 있습니다.

### 4.2. 기능 불일치: "내 컴퓨터에선 됐는데..." 문제의 주범

Django ORM이 많은 부분을 추상화해주지만, 데이터베이스 간의 근본적인 기능 차이까지 모두 해결해주지는 못합니다. 개발(SQLite)에서는 잘 동작하던 코드가 운영(PostgreSQL/MySQL) 환경에서 오류를 일으키는 주된 원인은 다음과 같습니다.

-   **대소문자 구분**:
    -   **SQLite**: `LIKE` 연산이 기본적으로 대소문자를 구분하지 않습니다. `Post.objects.filter(title__contains='django')`는 'Django', 'django', 'DJANGO'를 모두 찾아냅니다.
    -   **PostgreSQL**: `LIKE`는 대소문자를 구분합니다. 위 쿼리는 'django'만 찾아냅니다. 대소문자를 구분하지 않으려면 `__icontains`를 사용해야 합니다.
-   **데이터 타입 강제성**:
    -   **SQLite**: `CharField`의 `max_length`를 초과하는 데이터를 저장해도 오류가 발생하지 않습니다.
    -   **PostgreSQL/MySQL**: `VARCHAR` 길이를 초과하면 즉시 데이터베이스 오류가 발생합니다. 개발 중에 이 문제를 발견하지 못하면, 운영 환경에서 데이터가 잘리는 심각한 버그로 이어질 수 있습니다.
-   **날짜/시간 함수**:
    -   **SQLite**: 날짜/시간 관련 함수가 제한적입니다. `ExtractWeekDay` 같은 Django의 일부 함수는 SQLite에서 지원되지 않습니다.
    -   **PostgreSQL**: `DATE_TRUNC`, `EXTRACT` 등 풍부한 날짜/시간 처리 함수를 제공합니다.
-   **JSONField**:
    -   **SQLite**: `JSONField`를 `TEXT` 타입으로 에뮬레이션합니다. 복잡한 JSON 내부 키에 대한 쿼리 성능이 떨어집니다.
    -   **PostgreSQL**: 네이티브 `JSONB` 타입을 지원하여, JSON 내부를 인덱싱하고 쿼리하는 성능이 매우 뛰어납니다.

**해결 전략**:
1.  **CI/CD 파이프라인에서 테스트**: 지속적 통합(CI) 과정에서 SQLite뿐만 아니라, PostgreSQL이나 MySQL을 사용하여 통합 테스트를 반드시 실행해야 합니다. Docker를 사용하면 테스트 환경에 실제 운영 DB와 동일한 환경을 쉽게 구축할 수 있습니다.
2.  **DB 특화 기능 사용 자제**: 가급적 Django ORM이 제공하는 표준 기능만 사용하고, 특정 데이터베이스에만 의존하는 기능(예: PostgreSQL의 `ArrayField`)은 꼭 필요한 경우에만 신중하게 사용합니다.

### 4.3. 데이터 이전: `dumpdata`와 `loaddata`

개발 중 `db.sqlite3`에 쌓인 데이터를 프로덕션 데이터베이스로 이전해야 할 때가 있습니다. Django는 이를 위한 `dumpdata`와 `loaddata` 관리 명령어를 제공합니다.

1.  **데이터 덤프 (JSON 형식으로 내보내기)**:
    ```bash
    # 특정 앱(myapp)의 데이터를 myapp_data.json 파일로 저장
    python manage.py dumpdata myapp --output myapp_data.json --indent 2
    
    # contenttypes와 auth 앱을 제외하고 모든 데이터를 덤프 (일반적으로 유용)
    python manage.py dumpdata --exclude auth --exclude contenttypes --indent 2 > all_data.json
    ```
    - **`--exclude`**: 불필요한 Django 내부 모델(권한, 컨텐츠 타입 등)을 제외하여 충돌을 방지합니다.

2.  **프로덕션 환경 설정**: `settings.py`가 프로덕션 DB를 바라보도록 설정합니다.

3.  **스키마 마이그레이션**: 데이터를 로드하기 전에, 프로덕션 DB에 최신 스키마가 먼저 적용되어 있어야 합니다.
    ```bash
    python manage.py migrate
    ```

4.  **데이터 로드 (JSON 데이터를 DB로 가져오기)**:
    ```bash
    python manage.py loaddata all_data.json
    ```
    - **주의**: 이 과정은 데이터 양이 많을 경우 시간이 오래 걸릴 수 있습니다. 또한, 외래 키(Foreign Key) 관계가 복잡하게 얽혀있으면 순서 문제로 로드에 실패할 수 있습니다. 이 경우, 데이터를 여러 파일로 나누어 순서대로 로드해야 할 수도 있습니다.

**결론**: SQLite는 Django 개발의 생산성을 비약적으로 향상시키는 최고의 파트너입니다. 하지만 그 한계를 명확히 인지하고, 개발 초기부터 프로덕션 환경과의 차이를 염두에 둔 전략적인 개발(환경 분리, CI/CD 테스트 등)을 통해, '개발의 편리함'이 '운영의 재앙'으로 이어지지 않도록 관리하는 것이 성공적인 Django 프로젝트의 핵심입니다.
