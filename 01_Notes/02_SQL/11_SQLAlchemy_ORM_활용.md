<h2>Python과 SQL 연동: SQLAlchemy ORM 활용 (실무 활용 중심)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-29

<h2>문서 목표</h2>
<p>이 문서는 파이썬에서 가장 널리 사용되는 ORM(Object-Relational Mapping) 라이브러리인 <strong>SQLAlchemy</strong>를 활용하여 데이터베이스와 상호작용하는 방법을 심도 있게 다룹니다. 특히 ORM의 핵심 개념인 객체 지향적 데이터 접근 방식과 마이그레이션 도구인 Alembic을 활용한 스키마 관리 방법을 상세한 예제와 함께 설명하여, 파이썬 기반의 견고한 데이터베이스 애플리케이션 개발의 기초를 다지는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. ORM (Object-Relational Mapping)](#1-orm-object-relational-mapping)
  - [1.1. ORM의 개념과 장점](#11-orm의-개념과-장점)
  - [1.2. ORM의 단점](#12-orm의-단점)
- [2. SQLAlchemy 핵심 개념](#2-sqlalchemy-핵심-개념)
  - [2.1. 설치 및 기본 설정](#21-설치-및-기본-설정)
  - [2.2. 엔진 (Engine)](#22-엔진-engine)
  - [2.3. 메타데이터 (Metadata)](#23-메타데이터-metadata)
  - [2.4. 테이블 정의 (Table Definition)](#24-테이블-정의-table-definition)
  - [2.5. 세션 (Session)](#25-세션-session)
  - [2.6. 선언적 베이스 (Declarative Base)](#26-선언적-베이스-declarative-base)
- [3. SQLAlchemy ORM을 이용한 CRUD 작업](#3-sqlalchemy-orm을-이용한-crud-작업)
  - [3.1. 모델 정의](#31-모델-정의)
  - [3.2. 테이블 생성](#32-테이블-생성)
  - [3.3. 데이터 삽입 (Create)](#33-데이터-삽입-create)
  - [3.4. 데이터 조회 (Read)](#34-데이터-조회-read)
  - [3.5. 데이터 수정 (Update)](#35-데이터-수정-update)
  - [3.6. 데이터 삭제 (Delete)](#36-데이터-삭제-delete)
  - [3.7. 관계 설정 (Relationships)](#37-관계-설정-relationships)
- [4. Alembic을 이용한 데이터베이스 마이그레이션](#4-alembic을-이용한-데이터베이스-마이그레이션)
  - [4.1. 마이그레이션의 개념과 필요성](#41-마이그레이션의-개념과-필요성)
  - [4.2. Alembic 설치 및 초기화](#42-alembic-설치-및-초기화)
  - [4.3. 마이그레이션 스크립트 생성](#43-마이그레이션-스크립트-생성)
  - [4.4. 마이그레이션 실행](#44-마이그레이션-실행)
  - [4.5. 마이그레이션 되돌리기](#45-마이그레이션-되돌리기)

---

## 1. ORM (Object-Relational Mapping)

ORM(Object-Relational Mapping)은 객체 지향 프로그래밍 언어의 객체와 관계형 데이터베이스의 데이터를 자동으로 매핑(Mapping)하는 기술입니다. 개발자는 SQL 쿼리를 직접 작성하는 대신, 프로그래밍 언어의 객체를 사용하여 데이터베이스를 조작할 수 있게 됩니다.

### 1.1. ORM의 개념과 장점

*   **개념:** 객체 지향 언어의 클래스와 객체를 관계형 데이터베이스의 테이블과 로우에 매핑하여, 객체 지향적인 방식으로 데이터베이스를 조작할 수 있도록 돕는 기술입니다.
*   **장점:**
    *   **생산성 향상:** SQL 쿼리 작성 시간을 줄이고, 개발자가 비즈니스 로직에 더 집중할 수 있게 합니다.
    *   **객체 지향적 개발:** 데이터베이스 작업을 객체 지향적인 방식으로 수행하여 코드의 일관성과 가독성을 높입니다.
    *   **DBMS 독립성:** 대부분의 ORM은 다양한 데이터베이스 시스템(MySQL, PostgreSQL, SQLite 등)을 지원하여, 필요에 따라 DBMS를 쉽게 변경할 수 있습니다.
    *   **보안:** SQL Injection과 같은 보안 취약점을 자동으로 방지합니다.
    *   **유지보수 용이성:** 데이터베이스 스키마 변경 시 관련 코드 수정이 용이합니다.

### 1.2. ORM의 단점

*   **성능 오버헤드:** 복잡한 쿼리나 대량의 데이터 처리 시, ORM이 생성하는 SQL 쿼리가 수동으로 작성한 최적화된 SQL보다 비효율적일 수 있습니다.
*   **학습 곡선:** ORM의 개념과 사용법을 익히는 데 시간이 필요합니다.
*   **복잡한 쿼리 한계:** 매우 복잡하거나 특정 DBMS에 특화된 쿼리는 ORM으로 표현하기 어렵거나 불가능할 수 있습니다. 이 경우 원시 SQL(Raw SQL)을 사용해야 합니다.
*   **추상화로 인한 제어 부족:** 데이터베이스의 세부적인 동작을 직접 제어하기 어렵습니다.

**실무적 관점:** ORM은 대부분의 CRUD(Create, Read, Update, Delete) 작업에서 생산성을 크게 높여주지만, 성능이 중요한 특정 쿼리나 복잡한 분석 쿼리에서는 원시 SQL을 혼용하는 전략이 필요합니다. 데이터 분석가는 ORM의 장점을 활용하되, 단점을 인지하고 적절한 상황에 맞게 사용하는 지혜가 필요합니다.

## 2. SQLAlchemy 핵심 개념

SQLAlchemy는 파이썬에서 가장 널리 사용되는 ORM 라이브러리 중 하나입니다. SQL을 직접 다루는 SQL Expression Language와 객체 지향적인 ORM 컴포넌트를 모두 제공하여 유연한 데이터베이스 접근을 가능하게 합니다.

### 2.1. 설치 및 기본 설정

SQLAlchemy와 사용할 데이터베이스 드라이버(예: `pymysql`)를 설치합니다.

```bash
pip install sqlalchemy pymysql
```

### 2.2. 엔진 (Engine)

엔진(Engine)은 데이터베이스와의 연결을 관리하는 핵심 객체입니다. 데이터베이스 URL을 사용하여 생성하며, 데이터베이스 드라이버와 연결 풀(Connection Pool)을 설정합니다.

```python
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

# 데이터베이스 URL 형식: dialect+driver://user:password@host:port/database
DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}"
    f