<h2>SQL 핵심 문법: Django와 SQLite 연동 (실무 활용 중심)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
<p>이 문서는 서버 없이 동작하는 경량 데이터베이스 <strong>SQLite의 특징과 파이썬 <code>sqlite3</code> 모듈을 활용하는 방법</strong>을 심도 있게 다룹니다. 특히 풀스택 프레임워크인 Django에서 SQLite를 기본 개발 데이터베이스로 사용하는 방법과 연동 과정을 상세한 예제와 함께 설명하여, <strong>프로토타이핑 및 소규모 애플리케이션 개발 능력</strong>을 기르고, <strong>운영 환경 전환 시 고려사항</strong>을 이해하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. SQLite: 서버 없는 경량 데이터베이스](#1-sqlite-서버-없는-경량-데이터베이스)
  - [1.1. SQLite의 특징 및 장점](#11-sqlite의-특징-및-장점)
  - [1.2. SQLite의 한계와 사용 시 고려사항](#12-sqlite의-한계와-사용-시-고려사항)
  - [1.3. SQLite 사용이 적합한 경우](#13-sqlite-사용이-적합한-경우)
- [2. 파이썬과 SQLite 연동: `sqlite3` 모듈](#2-파이썬과-sqlite-연동-sqlite3-모듈)
  - [2.1. 연결 및 커서 생성](#21-연결-및-커서-생성)
  - [2.2. 테이블 생성 및 데이터 조작 (CRUD)](#22-테이블-생성-및-데이터-조작-crud)
  - [2.3. 트랜잭션 관리](#23-트랜잭션-관리)
- [3. Django와 SQLite 연동](#3-django와-sqlite-연동)
  - [3.1. Django가 SQLite를 기본으로 사용하는 이유](#31-django가-sqlite를-기본으로-사용하는-이유)
  - [3.2. `settings.py`의 데이터베이스 설정](#32-settingspy의-데이터베이스-설정)
  - [3.3. 마이그레이션(Migration)과 스키마 관리](#33-마이그레이션migration과-스키마-관리)
  - [3.4. Django ORM을 통한 데이터 조작](#34-django-orm을-통한-데이터-조작)
- [4. 실무적 고려사항: 개발에서 프로덕션까지](#4-실무적-고려사항-개발에서-프로덕션까지)
  - [4.1. 개발용(SQLite)과 운영용(PostgreSQL/MySQL) DB 분리](#41-개발용sqlite과-운영용postgresqlmysql-db-분리)
  - [4.2. SQLite의 데이터 타입과 Django 필드](#42-sqlite의-데이터-타입과-django-필드)
  - [4.3. 동시성 문제와 파일 잠금](#43-동시성-문제와-파일-잠금)

---

## 1. SQLite: 서버 없는 경량 데이터베이스

SQLite는 별도의 서버 프로세스가 필요 없는 **서버리스(Serverless)**, **파일 기반**의 관계형 데이터베이스 관리 시스템(RDBMS)입니다. C언어 라이브러리로 구현되어 있으며, 애플리케이션에 쉽게 내장하여 사용할 수 있습니다.

### 1.1. SQLite의 특징 및 장점

SQLite는 별도의 서버 프로세스가 필요 없는 **서버리스(Serverless)**, **파일 기반**의 관계형 데이터베이스 관리 시스템(RDBMS)입니다. C언어 라이브러리로 구현되어 있으며, 애플리케이션에 쉽게 내장하여 사용할 수 있습니다. 이러한 특성 덕분에 개발 및 테스트 환경에서 매우 강력한 장점을 가집니다.

*   **서버리스 (Serverless):** 별도의 데이터베이스 서버를 설치하거나 설정할 필요가 없습니다. 데이터베이스가 애플리케이션의 일부로 동작하며, 프로세스 간 통신이 아닌 직접 파일 접근 방식으로 데이터를 처리합니다. 설치 및 설정이 매우 간편하여 개발 환경 구축 시간을 단축하고, 운영 비용이 발생하지 않습니다.

*   **단일 파일 저장:** 전체 데이터베이스(테이블, 인덱스, 데이터, 스키마 등)가 하나의 `.sqlite3` 또는 `.db` 파일에 저장됩니다. 데이터베이스 관리가 매우 간편합니다. 파일 복사만으로 백업/복원이 가능하고, Git과 같은 버전 관리 시스템으로 데이터베이스 스키마와 초기 데이터를 함께 관리하기 용이합니다.

*   **설정 불필요 (Zero-configuration):** 복잡한 설정 파일이나 네트워크 설정 없이 바로 사용할 수 있습니다. 개발자가 데이터베이스 설정에 시간을 낭비하지 않고 핵심 개발에 집중할 수 있습니다.

*   **경량성 및 내장 가능:** 라이브러리 크기가 매우 작고, 메모리 사용량이 적어 모바일 기기나 임베디드 시스템에 적합합니다. 모바일 앱(Android, iOS), 데스크톱 애플리케이션(Electron, PyQt 등), IoT 기기 등 다양한 환경에 내장되어 로컬 데이터 저장소로 활용됩니다.

*   **이식성:** 데이터베이스 파일 하나만 복사하면 운영체제나 하드웨어에 관계없이 다른 시스템에서도 동일하게 사용할 수 있습니다. 개발 환경과 테스트 환경 간의 데이터베이스 공유가 용이하며, 배포 시에도 데이터베이스 파일을 함께 배포할 수 있습니다.

*   **표준 SQL 지원:** 대부분의 표준 SQL-92 문법을 지원합니다. `SELECT`, `INSERT`, `UPDATE`, `DELETE`, `JOIN`, `GROUP BY` 등 기본적인 SQL 기능은 모두 사용할 수 있습니다. SQL 학습 및 실습에 용이하며, 다른 RDBMS로의 전환 시에도 SQL 문법 학습 부담이 적습니다.

### 1.2. SQLite의 한계와 사용 시 고려사항

SQLite는 많은 장점을 가지고 있지만, 모든 상황에 적합한 데이터베이스는 아닙니다. 특히 다중 사용자 환경이나 고성능이 요구되는 운영 환경에서는 다음과 같은 한계점을 명확히 인지하고 사용을 지양해야 합니다.

*   **동시성(Concurrency) 제한:** SQLite는 쓰기(Write) 작업을 수행할 때 데이터베이스 파일 전체에 잠금(Lock)을 겁니다. 이로 인해 여러 사용자가 동시에 쓰기 작업을 시도하면 다른 트랜잭션이 대기하거나 `database is locked` 오류가 발생할 수 있습니다. 이는 쓰기 작업이 빈번하거나 동시 사용자가 많은 웹 애플리케이션의 운영 환경에는 치명적인 성능 병목을 유발합니다.

*   **제한된 데이터 타입 및 동적 타이핑:** SQLite는 `VARCHAR`, `DATETIME` 등 다른 RDBMS에 있는 일부 데이터 타입을 엄격하게 구분하지 않고, 동적 타이핑(Dynamic Typing)을 사용합니다. 즉, 컬럼에 어떤 타입의 데이터든 저장할 수 있습니다. 이는 유연성을 제공하지만, 데이터 무결성 측면에서는 취약할 수 있습니다. 데이터 타입 불일치로 인한 예상치 못한 오류나 데이터 손실이 발생할 수 있으며, 다른 RDBMS로 마이그레이션 시 데이터 타입 변환 문제가 발생할 수 있습니다.

*   **고급 기능 부재:** 윈도우 함수, 저장 프로시저, `RIGHT JOIN`, `FULL OUTER JOIN` 등 일부 고급 SQL 기능이 지원되지 않거나 제한적으로 지원됩니다. 복잡한 비즈니스 로직이나 고급 분석 쿼리를 데이터베이스 레벨에서 구현하기 어렵습니다.

*   **성능:** 대용량 데이터 처리나 복잡한 쿼리 실행 시, 서버 기반 RDBMS(PostgreSQL, MySQL 등)에 비해 성능이 떨어질 수 있습니다. 특히 디스크 I/O가 많은 작업에서 성능 한계가 명확합니다.

**실무적 관점:** SQLite는 개발 및 테스트, 소규모 애플리케이션, 임베디드 시스템 등 특정 목적에 매우 적합하지만, **다중 사용자 환경이나 높은 동시성, 대용량 데이터 처리가 필요한 운영 환경에는 절대 사용해서는 안 됩니다.** 프로젝트의 요구사항을 명확히 분석하여 적절한 데이터베이스를 선택하는 것이 중요합니다.

### 1.3. SQLite 사용이 적합한 경우

SQLite는 앞서 언급된 한계점에도 불구하고, 특정 사용 사례에서는 매우 강력하고 효율적인 선택이 될 수 있습니다.

*   **애플리케이션 개발 및 프로토타이핑:** 새로운 웹 애플리케이션이나 서비스의 초기 개발 단계에서 데이터베이스 스키마를 빠르게 정의하고 테스트해야 할 때 적합합니다. 별도의 데이터베이스 서버 설치나 설정 없이 즉시 개발을 시작할 수 있어 개발 속도를 극대화합니다.

*   **소규모 웹사이트 및 개인 프로젝트:** 트래픽이 많지 않고 동시 쓰기 작업이 거의 없는 개인 블로그, 포트폴리오 웹사이트, 또는 소규모 내부 관리 도구 등에 적합합니다. 서버 관리 부담이 없고, 유지보수 비용이 거의 들지 않습니다.

*   **모바일 및 데스크톱 애플리케이션:** 안드로이드, iOS 앱의 로컬 데이터 저장, 웹 브라우저의 북마크/히스토리 관리, 오프라인에서 동작하는 데스크톱 소프트웨어 등에 적합합니다. 경량성, 내장 가능성, 단일 파일 관리의 장점 덕분에 애플리케이션 내부에 데이터를 저장하고 관리하는 데 최적입니다.

*   **데이터 분석 및 임시 데이터 저장:** CSV, JSON 등 파일 기반 데이터 대신 SQL 쿼리로 데이터를 처리하고 싶을 때, 또는 복잡한 분석을 위한 중간 결과 데이터를 임시로 저장해야 할 때 적합합니다. `sqlite3` 모듈을 통해 파이썬에서 쉽게 접근하고 조작할 수 있어, 데이터 전처리, 탐색적 데이터 분석(EDA), 소규모 ETL 작업 등에 임시 데이터베이스로 활용하기 좋습니다.

*   **테스트 및 교육 목적:** 데이터베이스 관련 코드의 단위 테스트, 통합 테스트, 또는 SQL 교육 환경 구축에 적합합니다. 메모리 기반 데이터베이스(`:memory:`)를 사용하여 테스트 환경을 빠르게 구축하고 해체할 수 있으며, 테스트 간 데이터 독립성을 보장하기 용이합니다.

**실무적 관점:** SQLite는 특정 목적에 최적화된 '틈새 시장' 데이터베이스입니다. 그 한계를 명확히 이해하고, 적절한 사용 사례에 적용한다면 매우 효율적이고 편리한 도구가 될 수 있습니다. 특히 개발 초기 단계의 빠른 프로토타이핑, 로컬 데이터 관리, 또는 간단한 분석 작업에 활용하면 그 진가를 발휘합니다.

## 2. 파이썬과 SQLite 연동: `sqlite3` 모듈

파이썬은 `sqlite3` 모듈을 기본 라이브러리로 제공하므로, 별도의 드라이버 설치 없이 바로 SQLite를 사용할 수 있습니다.

### 2.1. 연결 및 커서 생성

`sqlite3.connect()` 함수를 사용하여 데이터베이스에 연결합니다. 파일 경로를 지정하면 해당 파일이 데이터베이스가 되며, 파일이 없으면 새로 생성됩니다. `:memory:`를 사용하면 메모리 내 임시 데이터베이스를 생성할 수 있습니다. 연결 객체는 데이터베이스와의 통신을 담당하며, 커서(Cursor) 객체는 SQL 쿼리를 실행하고 결과를 가져오는 데 사용됩니다.

```python
import sqlite3

# 데이터베이스 연결 (파일이 없으면 생성됨)
# :memory: 를 사용하면 메모리 내 임시 데이터베이스 생성 (테스트 용이)
conn = sqlite3.connect('my_database.db')

# 결과 로우를 딕셔너리 형태로 가져오기 위한 설정 (선택 사항, 가독성 향상)
# 기본적으로는 튜플 형태로 반환됩니다.
conn.row_factory = sqlite3.Row 

# 커서 생성
cursor = conn.cursor()

# 작업 완료 후 연결 종료
cursor.close()
conn.close()
```

**실무 팁: `with` 문을 사용한 연결 및 커서 관리**
파이썬에서는 `with` 문을 사용하여 파일이나 네트워크 연결과 같은 리소스를 안전하게 관리하는 것이 일반적인 모범 사례입니다. `sqlite3` 모듈의 연결 객체(`conn`)와 커서 객체(`cursor`)도 `with` 문을 지원하므로, 이를 활용하면 `close()` 메서드를 명시적으로 호출하지 않아도 자동으로 리소스가 해제됩니다.

```python
import sqlite3

# with 문을 사용하여 연결 및 커서 자동 관리
try:
    with sqlite3.connect('my_database.db') as conn:
        conn.row_factory = sqlite3.Row # 딕셔너리 형태 결과 설정
        with conn.cursor() as cursor:
            # SQL 쿼리 실행
            cursor.execute("SELECT SQLITE_VERSION();")
            version = cursor.fetchone()
            print(f"SQLite Version: {version[0]}") # 튜플 접근
            print(f"SQLite Version (dict-like): {version['SQLITE_VERSION()']}") # 딕셔너리 접근

except sqlite3.Error as e:
    print(f"Database error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("Database connection automatically closed.")
```

### 2.2. 테이블 생성 및 데이터 조작 (CRUD)

SQL 쿼리를 문자열로 작성하고 `cursor.execute()`를 사용하여 실행합니다. 이때 **SQL Injection 공격을 방지하기 위해 사용자 입력을 직접 문자열에 삽입하는 대신 반드시 파라미터화된 쿼리를 사용해야 합니다.**

```python
import sqlite3

conn = sqlite3.connect('my_database.db')
conn.row_factory = sqlite3.Row # 딕셔너리 형태 결과 설정
cursor = conn.cursor()

try:
    # 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 데이터 삽입 (SQL Injection 방지를 위해 파라미터화 사용: ? 플레이스홀더)
    cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", ('alice', 'alice@example.com'))
    user_id = cursor.lastrowid
    print(f"Inserted user with ID: {user_id}")

    # 대량 데이터 삽입 (executemany() 활용)
    users_to_insert = [
        ('bob', 'bob@example.com'),
        ('charlie', 'charlie@example.com'),
    ]
    cursor.executemany("INSERT INTO users (username, email) VALUES (?, ?)", users_to_insert)
    print(f"Inserted {cursor.rowcount} users using executemany.")

    # 데이터 조회
    cursor.execute("SELECT * FROM users WHERE username = ?", ('alice',))
    user = cursor.fetchone()
    if user:
        print(f"Fetched user: {dict(user)}") # sqlite3.Row 객체를 딕셔너리로 변환

    # 모든 사용자 조회
    cursor.execute("SELECT * FROM users")
    all_users = cursor.fetchall()
    print("\nAll users:")
    for u in all_users:
        print(dict(u))

    # 데이터 수정
    cursor.execute("UPDATE users SET email = ? WHERE username = ?", ('alice_new@example.com', 'alice'))
    print(f"Updated {cursor.rowcount} row(s).")

    # 데이터 삭제
    cursor.execute("DELETE FROM users WHERE username = ?", ('charlie',))
    print(f"Deleted {cursor.rowcount} row(s).")

    # 변경사항 커밋
    conn.commit()
    print("Transaction committed.")

except sqlite3.Error as e:
    print(f"Database error: {e}")
    conn.rollback() # 오류 발생 시 롤백
    print("Transaction rolled back.")
finally:
    conn.close()
    print("Database connection closed.")
```

### 2.3. 트랜잭션 관리

`sqlite3` 모듈은 기본적으로 DML(`INSERT`, `UPDATE`, `DELETE`) 문 실행 시 자동으로 트랜잭션을 시작합니다. `conn.commit()`을 호출하여 변경사항을 영구 저장하고, `conn.rollback()`으로 취소할 수 있습니다. `SELECT` 문은 기본적으로 트랜잭션에 영향을 주지 않습니다.

**실무 팁: 명시적인 트랜잭션 제어**
`sqlite3`는 기본적으로 `BEGIN DEFERRED TRANSACTION` 모드로 동작합니다. 이는 첫 번째 쓰기 작업이 발생할 때 트랜잭션을 시작하고, `COMMIT` 또는 `ROLLBACK` 시 트랜잭션을 종료합니다. 명시적으로 `BEGIN`, `COMMIT`, `ROLLBACK`을 사용하여 트랜잭션을 제어하는 것이 데이터 일관성을 보장하는 데 중요합니다.

```python
import sqlite3

conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

try:
    # 명시적으로 트랜잭션 시작
    cursor.execute("BEGIN;") # 또는 conn.execute("BEGIN;")

    # 첫 번째 작업
    cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", ('diana', 'diana@example.com'))
    print("User Diana inserted.")

    # 두 번째 작업 (오류 발생 가정)
    # cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", ('diana', 'diana_duplicate@example.com')) # UNIQUE 제약조건 위반

    # 모든 작업 성공 시 커밋
    conn.commit()
    print("Transaction committed.")

except sqlite3.Error as e:
    print(f"Transaction failed: {e}")
    conn.rollback() # 오류 발생 시 롤백
    print("Transaction rolled back.")
finally:
    conn.close()
    print("Database connection closed.")
```

**실무 팁: `with` 문을 사용한 트랜잭션 관리**
`sqlite3` 연결 객체는 `with` 문을 지원하여 트랜잭션 관리를 더욱 간결하고 안전하게 할 수 있습니다. `with` 블록이 성공적으로 종료되면 자동으로 `commit()`이 호출되고, 예외 발생 시 `rollback()`이 호출됩니다.

```python
import sqlite3

try:
    with sqlite3.connect('my_database.db') as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", ('eve', 'eve@example.com'))
        print("User Eve inserted.")

        # 의도적인 오류 발생
        # cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", ('eve', 'eve_duplicate@example.com'))

        print("All operations within with block completed.")

except sqlite3.Error as e:
    print(f"Transaction failed (via with statement): {e}")
    # with 문이 rollback을 자동으로 처리

finally:
    print("Database connection automatically closed by with statement.")
```

## 3. Django와 SQLite 연동

Django는 새로운 프로젝트를 생성할 때 기본 데이터베이스로 SQLite를 사용하도록 설정되어 있습니다. 이는 개발자가 복잡한 데이터베이스 설정 없이 즉시 개발을 시작할 수 있도록 돕습니다.

### 3.1. Django가 SQLite를 기본으로 사용하는 이유

Django는 새로운 프로젝트를 생성할 때 기본 데이터베이스로 SQLite를 사용하도록 설정되어 있습니다. 이는 개발자가 복잡한 데이터베이스 설정 없이 즉시 개발을 시작하고, 초기 단계에서 높은 생산성을 확보할 수 있도록 돕기 위함입니다.

*   **간편한 시작 (Zero-configuration):** SQLite는 별도의 DB 서버 설치나 설정이 필요 없는 파일 기반 데이터베이스입니다. Django 프로젝트를 생성하는 순간 `db.sqlite3` 파일이 자동으로 생성되며, 추가적인 설정 없이 바로 데이터베이스 작업을 시작할 수 있습니다. 개발 환경 구축에 드는 시간과 노력을 최소화하여, 개발자가 핵심 애플리케이션 로직 개발에 집중할 수 있게 합니다.

*   **뛰어난 이식성 및 버전 관리 용이성:** 데이터베이스 전체가 단일 파일(`db.sqlite3`)로 저장되므로, 이 파일 하나만 복사하면 다른 개발 환경이나 테스트 환경으로 쉽게 이동할 수 있습니다. Git과 같은 버전 관리 시스템으로 데이터베이스 스키마 변경 이력(`migrations` 폴더)과 초기 데이터(fixture)를 함께 관리하기 매우 편리합니다.

*   **개발 및 테스트에 충분한 기능:** Django의 강력한 ORM(Object-Relational Mapping)은 SQLite의 기본적인 SQL 기능만으로도 대부분의 개발 및 테스트 요구사항을 충족시킬 수 있도록 추상화 계층을 제공합니다. 복잡한 SQL 쿼리를 직접 작성할 필요 없이 파이썬 객체 지향적인 방식으로 데이터베이스를 조작할 수 있습니다.

**실무적 관점:** Django가 SQLite를 기본으로 사용하는 것은 개발자의 생산성과 편의성을 최우선으로 고려한 설계 결정입니다. 이를 통해 개발자는 데이터베이스 설정의 복잡성에서 벗어나 애플리케이션의 핵심 기능 구현에 집중할 수 있습니다. 하지만 이는 개발 단계에 한정되며, 운영 환경에서는 SQLite의 한계를 고려하여 적절한 RDBMS로 전환해야 합니다.

### 3.2. `settings.py`의 데이터베이스 설정

Django 프로젝트의 `settings.py` 파일에서 데이터베이스 설정을 확인할 수 있습니다.

```python
# myproject/settings.py

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```
*   `ENGINE`: 사용할 데이터베이스 백엔드를 지정합니다.
*   `NAME`: 데이터베이스 파일의 경로를 지정합니다. `BASE_DIR`는 프로젝트의 루트 디렉토리입니다.

### 3.3. 마이그레이션(Migration)과 스키마 관리

Django ORM을 사용하는 핵심적인 장점 중 하나는 **마이그레이션(Migration)** 시스템입니다. 마이그레이션은 데이터베이스 스키마의 변경 이력을 관리하고, 파이썬 모델 정의와 데이터베이스 스키마를 동기화하는 Django의 강력한 기능입니다. 이를 통해 개발자는 SQL을 직접 작성하지 않고도 데이터베이스 스키마를 안전하게 변경하고 관리할 수 있습니다.

1.  **마이그레이션 파일 생성 (`makemigrations`):** `models.py` 파일에 정의된 모델의 변경사항을 감지하여, 해당 변경사항을 데이터베이스에 적용하기 위한 파이썬 스크립트 파일(마이그레이션 파일)을 자동으로 생성합니다.

    ```bash
    python manage.py makemigrations [app_name] # 특정 앱만 지정 가능
    ```

2.  **마이그레이션 적용 (`migrate`):** 생성된 마이그레이션 파일을 데이터베이스에 적용하여 실제 데이터베이스 스키마를 변경합니다. 이 과정에서 Django는 `settings.py`에 설정된 데이터베이스에 접속하여 필요한 SQL 쿼리를 실행합니다.

    ```bash
    python manage.py migrate [app_name] [migration_name] # 특정 앱 또는 마이그레이션 지정 가능
    ```

**실무 팁: 마이그레이션 관리 시 고려사항**
*   **버전 관리:** 마이그레이션 파일은 Git과 같은 버전 관리 시스템에 포함되어야 합니다.
*   **충돌 해결:** 여러 개발자가 동시에 모델을 변경하고 마이그레이션 파일을 생성할 경우, 마이그레이션 충돌이 발생할 수 있습니다.
*   **운영 환경 배포:** 운영 환경에 마이그레이션을 배포할 때는 항상 백업을 수행하고, 서비스 중단 시간을 최소화하기 위한 전략을 고려해야 합니다.
*   **데이터 마이그레이션:** 스키마 변경과 함께 데이터 자체를 변환해야 하는 경우, `RunPython` 작업을 포함하는 데이터 마이그레이션을 작성해야 합니다.

### 3.4. Django ORM을 통한 데이터 조작

Django 셸이나 뷰 코드에서 ORM을 사용하여 데이터를 조작할 수 있습니다. Django ORM은 내부적으로 `settings.py`에 설정된 데이터베이스에 맞는 SQL 쿼리를 생성하여 실행합니다.

```python
# Django 셸 실행
# python manage.py shell

from myapp.models import MyModel

# 데이터 생성
MyModel.objects.create(name='Test', description='This is a test.')

# 데이터 조회
all_objects = MyModel.objects.all()
print(all_objects)

# 데이터 필터링
filtered_objects = MyModel.objects.filter(name='Test')
print(filtered_objects)
```

## 4. 실무적 고려사항: 개발에서 프로덕션까지

SQLite는 개발 및 테스트 환경에서 매우 유용하지만, 대부분의 상용 서비스 운영 환경에는 적합하지 않습니다. 따라서 Django 프로젝트를 개발할 때는 SQLite의 한계를 명확히 이해하고, 개발 단계에서 운영 환경으로 전환할 때 발생할 수 있는 문제들을 미리 고려해야 합니다.

### 4.1. 개발용(SQLite)과 운영용(PostgreSQL/MySQL) DB 분리

SQLite는 동시성 문제와 성능 한계로 인해 다중 사용자 환경이나 대용량 데이터 처리가 필요한 운영 환경에는 적합하지 않습니다. 따라서 일반적인 Django 프로젝트에서는 다음과 같은 전략을 사용합니다.

*   **개발 환경:** SQLite를 사용하여 빠르고 간편하게 개발합니다. `db.sqlite3` 파일을 Git으로 관리하여 팀원 간의 환경 동기화를 용이하게 합니다.
*   **운영(프로덕션) 환경:** PostgreSQL이나 MySQL과 같은 강력하고 확장성 있는 서버 기반 RDBMS를 사용합니다. 이들은 높은 동시성, 트랜잭션 안정성, 고급 기능 등을 제공하여 운영 환경의 요구사항을 충족시킵니다.

**실무 팁: 환경별 `settings.py` 관리**
환경에 따라 다른 데이터베이스 설정을 사용하기 위해 `settings.py` 파일을 분리하거나, 환경 변수를 사용하여 동적으로 설정을 변경하는 방법을 사용합니다. (예: `settings/dev.py`, `settings/prod.py`)

### 4.2. SQLite의 데이터 타입과 Django 필드

Django ORM은 데이터베이스 간의 차이를 추상화해주지만, SQLite의 동적 타이핑 특성 때문에 간혹 예상치 못한 문제가 발생할 수 있습니다. 예를 들어, Django 모델에서 `CharField`의 `max_length`는 SQLite에서 엄격하게 강제되지 않을 수 있습니다.

*   **문제점:** SQLite는 `TEXT` 컬럼에 `max_length`를 초과하는 문자열을 저장해도 오류를 발생시키지 않습니다. 이는 개발 단계에서 데이터 유효성 검사 로직의 버그를 놓치게 할 수 있습니다.
*   **해결 방안:**
    *   **운영 DB와 동일한 종류의 DB 사용:** 가장 이상적인 방법은 Docker 등을 사용하여 개발 환경에서도 운영 환경과 동일한 종류의 데이터베이스(예: PostgreSQL/MySQL)를 사용하는 것입니다. 이는 개발/운영 환경 간의 데이터베이스 불일치로 인한 문제를 원천적으로 방지합니다.
    *   **Django의 유효성 검사 활용:** Django 모델 필드의 `max_length`나 `validators`를 적극적으로 활용하여 애플리케이션 레벨에서 데이터 유효성 검사를 강화합니다.

### 4.3. 동시성 문제와 파일 잠금

SQLite는 쓰기 작업을 수행할 때 데이터베이스 파일 전체에 잠금을 겁니다. 이로 인해 여러 웹 요청이 동시에 쓰기 작업을 시도하면 `database is locked` 오류가 발생할 수 있습니다. Django는 이 문제를 완화하기 위해 기본 `timeout` 값을 설정하지만, 이는 근본적인 해결책이 아닙니다.

*   **문제점:**
    *   **낮은 동시성:** 동시 쓰기 요청이 많은 웹 애플리케이션에서는 SQLite의 파일 잠금 방식이 심각한 병목 현상을 유발합니다.
    *   **`database is locked` 오류:** 여러 프로세스나 스레드가 동시에 쓰기 작업을 시도할 때 이 오류가 발생하며, 이는 사용자 경험을 저하시키고 서비스 안정성을 해칩니다.
*   **해결 방안:**
    *   **서버 기반 RDBMS로 전환:** 동시 사용자가 많거나 쓰기 작업이 빈번한 서비스라면 반드시 PostgreSQL, MySQL과 같은 서버 기반 RDBMS로 전환해야 합니다. 이들은 로우 레벨 잠금(Row-level Locking)을 지원하여 높은 동시성을 제공합니다.
    *   **트랜잭션 최적화:** 트랜잭션의 길이를 짧게 유지하고, 필요한 최소한의 데이터에만 잠금을 걸도록 쿼리를 최적화합니다.

**실무적 관점:** 데이터 분석가는 SQLite의 이러한 한계점을 명확히 이해하고, 분석 프로젝트의 규모와 요구사항에 따라 적절한 데이터베이스를 선택할 수 있어야 합니다. 특히 운영 환경으로의 전환 시 발생할 수 있는 데이터베이스 관련 문제들을 예측하고, 개발팀과 협력하여 해결 방안을 모색하는 데 기여할 수 있어야 합니다.