<h2>SQLite와 Django 연동</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
<p>이 문서는 서버 없이 동작하는 경량 데이터베이스 SQLite의 특징을 이해하고, 파이썬의 기본 라이브러리인 <code>sqlite3</code>를 활용하는 방법을 학습합니다. 특히 풀스택 프레임워크인 Django에서 SQLite를 기본 개발 데이터베이스로 사용하는 방법과 연동 과정을 익혀, 프로토타이핑 및 소규모 애플리케이션 개발 능력을 기르는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. SQLite: 서버 없는 경량 데이터베이스](#1-sqlite-서버-없는-경량-데이터베이스)
  - [1.1. SQLite의 특징 및 장점](#11-sqlite의-특징-및-장점)
  - [1.2. SQLite의 한계와 사용 시 고려사항](#12-sqlite의-한계와-사용-시-고려사항)
  - [1.3. SQLite 사용이 적합한 경우](#13-sqlite-사용이-적합한-경우)
- [2. 파이썬과 SQLite 연동: `sqlite3` 모듈](#2-파이썬과-sqlite-연동-sqlite3-모듈)
  - [2.1. 연결 및 커서 생성](#21-연결-및-커서-생성)
  - [2.2. 테이블 생성 및 데이터 조작 (CRUD)](#22-테이블-생성-및-데이터-조작-crud)
  - [2.3. 트랜잭션 관리](#23-트랜잭션-관리)
- [3. Django와 SQLite 연동: 개발의 시작](#3-django와-sqlite-연동-개발의-시작)
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

*   **서버리스 (Serverless):** 별도의 데이터베이스 서버를 설치하거나 설정할 필요가 없습니다. 데이터베이스가 애플리케이션의 일부로 동작합니다.
*   **단일 파일 저장:** 전체 데이터베이스(테이블, 인덱스, 데이터 등)가 하나의 `.sqlite3` 또는 `.db` 파일에 저장되어 관리가 매우 간편합니다.
*   **설정 불필요 (Zero-configuration):** 복잡한 설정 과정 없이 바로 사용할 수 있습니다.
*   **경량성:** 라이브러리 크기가 매우 작고, 메모리 사용량이 적어 모바일 기기나 임베디드 시스템에 적합합니다.
*   **이식성:** 데이터베이스 파일 하나만 복사하면 다른 시스템에서도 동일하게 사용할 수 있습니다.
*   **표준 SQL 지원:** 대부분의 표준 SQL-92 문법을 지원합니다.

### 1.2. SQLite의 한계와 사용 시 고려사항

*   **동시성(Concurrency) 제한:** 여러 사용자가 동시에 쓰기(Write) 작업을 수행하는 데 한계가 있습니다. 데이터베이스 파일 전체에 잠금(Lock)이 걸리는 방식으로 동작하므로, 쓰기 작업이 많은 웹 애플리케이션의 운영 환경에는 적합하지 않습니다.
*   **제한된 데이터 타입:** `VARCHAR`, `DATETIME` 등 다른 RDBMS에 있는 일부 데이터 타입을 엄격하게 구분하지 않고, 동적 타이핑(Dynamic Typing)을 사용합니다. (예: 모든 문자열은 `TEXT`, 숫자는 `INTEGER` 또는 `REAL`로 처리)
*   **고급 기능 부재:** 윈도우 함수, 저장 프로시저, `RIGHT JOIN`, `FULL OUTER JOIN` 등 일부 고급 SQL 기능이 지원되지 않습니다.
*   **성능:** 대용량 데이터 처리나 복잡한 쿼리 실행 시, 서버 기반 RDBMS(PostgreSQL, MySQL 등)에 비해 성능이 떨어질 수 있습니다.

### 1.3. SQLite 사용이 적합한 경우

*   **애플리케이션 개발 및 프로토타이핑:** 빠르고 간편하게 데이터베이스 환경을 구축할 수 있어 초기 개발 단계에 매우 유용합니다.
*   **소규모 웹사이트:** 트래픽이 많지 않고 동시 쓰기 작업이 거의 없는 개인 블로그나 소규모 웹사이트에 사용할 수 있습니다.
*   **모바일 애플리케이션:** 안드로이드, iOS 등 모바일 앱에 내장되어 데이터를 저장하는 용도로 널리 사용됩니다.
*   **데스크톱 애플리케이션:** 로컬 데이터를 저장하는 데 사용됩니다. (예: 웹 브라우저의 북마크, 히스토리 관리)
*   **데이터 분석:** CSV나 JSON 파일 대신, SQL로 데이터를 처리하고 싶을 때 임시 데이터베이스로 활용하기 좋습니다.

## 2. 파이썬과 SQLite 연동: `sqlite3` 모듈

파이썬은 `sqlite3` 모듈을 기본 라이브러리로 제공하므로, 별도의 드라이버 설치 없이 바로 SQLite를 사용할 수 있습니다.

### 2.1. 연결 및 커서 생성

`sqlite3.connect()` 함수를 사용하여 데이터베이스에 연결합니다. 파일 경로를 지정하면 해당 파일이 데이터베이스가 되며, 파일이 없으면 새로 생성됩니다. `:memory:`를 사용하면 메모리 내 임시 데이터베이스를 생성할 수 있습니다.

```python
import sqlite3

# 데이터베이스 연결 (파일이 없으면 생성됨)
conn = sqlite3.connect('my_database.db')

# 커서 생성
cursor = conn.cursor()

# 작업 완료 후 연결 종료
cursor.close()
conn.close()
```

### 2.2. 테이블 생성 및 데이터 조작 (CRUD)

SQL 쿼리를 문자열로 작성하고 `cursor.execute()`를 사용하여 실행합니다.

```python
import sqlite3

conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

# 테이블 생성
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL
    )
''')

# 데이터 삽입 (SQL Injection 방지를 위해 파라미터화 사용)
cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", ('alice', 'alice@example.com'))
user_id = cursor.lastrowid
print(f"Inserted user with ID: {user_id}")

# 데이터 조회
cursor.execute("SELECT * FROM users WHERE username = ?", ('alice',))
user = cursor.fetchone()
print(f"Fetched user: {user}")

# 데이터 수정
cursor.execute("UPDATE users SET email = ? WHERE username = ?", ('alice_new@example.com', 'alice'))

# 데이터 삭제
cursor.execute("DELETE FROM users WHERE username = ?", ('alice',))

# 변경사항 커밋
conn.commit()

# 연결 종료
conn.close()
```

### 2.3. 트랜잭션 관리

`sqlite3`는 기본적으로 DML(`INSERT`, `UPDATE`, `DELETE`) 문 실행 시 자동으로 트랜잭션을 시작합니다. `conn.commit()`을 호출하여 변경사항을 영구 저장하고, `conn.rollback()`으로 취소할 수 있습니다.

## 3. Django와 SQLite 연동: 개발의 시작

Django는 새로운 프로젝트를 생성할 때 기본 데이터베이스로 SQLite를 사용하도록 설정되어 있습니다. 이는 개발자가 복잡한 데이터베이스 설정 없이 즉시 개발을 시작할 수 있도록 돕습니다.

### 3.1. Django가 SQLite를 기본으로 사용하는 이유

*   **간편함:** 별도의 DB 서버 설치나 설정이 필요 없어 개발 환경을 빠르게 구축할 수 있습니다.
*   **이식성:** `db.sqlite3` 파일 하나만 있으면 되므로, 다른 개발자와 프로젝트를 공유하거나 버전 관리하기 편리합니다.
*   **충분한 기능:** Django의 ORM이 제공하는 대부분의 기능을 문제없이 지원하여, 개발 및 테스트 용도로는 충분합니다.

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

Django ORM을 사용하여 `models.py`에 모델을 정의한 후, 다음 명령어를 통해 데이터베이스 스키마를 생성하고 관리합니다.

1.  **마이그레이션 파일 생성:** 모델의 변경사항을 감지하여 마이그레이션 파일을 생성합니다.
    ```bash
    python manage.py makemigrations
    ```
2.  **마이그레이션 적용:** 생성된 마이그레이션 파일을 데이터베이스에 적용하여 테이블을 생성하거나 변경합니다.
    ```bash
    python manage.py migrate
    ```
이 과정에서 Django는 `settings.py`에 설정된 SQLite 데이터베이스에 접속하여 필요한 테이블을 자동으로 생성합니다.

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

### 4.1. 개발용(SQLite)과 운영용(PostgreSQL/MySQL) DB 분리

SQLite는 개발용으로는 훌륭하지만, 동시성 문제와 성능 한계로 인해 대부분의 상용 서비스 운영 환경에는 적합하지 않습니다. 따라서 일반적인 Django 프로젝트에서는 다음과 같은 전략을 사용합니다.

*   **개발 환경:** SQLite를 사용하여 빠르고 간편하게 개발합니다.
*   **운영(프로덕션) 환경:** PostgreSQL이나 MySQL과 같은 강력한 서버 기반 RDBMS를 사용합니다.

환경에 따라 다른 설정을 사용하기 위해 `settings.py` 파일을 분리하거나, 환경 변수를 사용하여 동적으로 설정을 변경하는 방법을 사용합니다.

### 4.2. SQLite의 데이터 타입과 Django 필드

Django ORM은 데이터베이스 간의 차이를 추상화해주지만, SQLite의 동적 타이핑 특성 때문에 간혹 문제가 발생할 수 있습니다. 예를 들어, Django 모델에서 `CharField`의 `max_length`는 SQLite에서 엄격하게 강제되지 않을 수 있습니다. 운영 환경에서 사용할 데이터베이스와 동일한 종류의 데이터베이스(예: Docker를 이용한 PostgreSQL/MySQL)를 개발 환경에서도 사용하는 것이 가장 이상적입니다.

### 4.3. 동시성 문제와 파일 잠금

SQLite는 쓰기 작업을 수행할 때 데이터베이스 파일 전체에 잠금을 겁니다. 이로 인해 여러 웹 요청이 동시에 쓰기 작업을 시도하면 `database is locked` 오류가 발생할 수 있습니다. Django는 이 문제를 완화하기 위해 기본 `timeout` 값을 설정하지만, 근본적인 해결책은 아닙니다. 동시 사용자가 많은 서비스라면 반드시 서버 기반 RDBMS로 전환해야 합니다.
