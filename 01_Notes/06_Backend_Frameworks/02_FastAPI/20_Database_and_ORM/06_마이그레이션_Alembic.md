<h2>FastAPI 학습 가이드: 마이그레이션 Alembic - 데이터베이스 스키마 버전 관리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 Alembic을 사용하여 FastAPI 애플리케이션의 데이터베이스 스키마를 효율적으로 관리하는 방법을 학습하는 것을 목표로 합니다. Alembic의 기본 개념, 초기 설정, 마이그레이션 파일 생성 및 적용, 그리고 실무적인 마이그레이션 전략을 이해하여 데이터베이스 변경 사항을 안전하게 관리하는 능력을 기릅니다.

<h2>목차</h2>

- [1. 데이터베이스 마이그레이션이란?](#1-데이터베이스-마이그레이션이란)
  - [1.1. 왜 마이그레이션이 필요한가?](#11-왜-마이그레이션이-필요한가)
  - [1.2. Alembic의 역할](#12-alembic의-역할)
- [2. Alembic 초기 설정](#2-alembic-초기-설정)
  - [2.1. Alembic 설치](#21-alembic-설치)
  - [2.2. Alembic 프로젝트 초기화](#22-alembic-프로젝트-초기화)
  - [2.3. `env.py` 설정](#23-envpy-설정)
- [3. 마이그레이션 파일 생성 및 적용](#3-마이그레이션-파일-생성-및-적용)
  - [3.1. 마이그레이션 파일 생성 (`alembic revision --autogenerate`)](#31-마이그레이션-파일-생성-alembic-revision---autogenerate)
  - [3.2. 마이그레이션 적용 (`alembic upgrade`)](#32-마이그레이션-적용-alembic-upgrade)
  - [3.3. 마이그레이션 되돌리기 (`alembic downgrade`)](#33-마이그레이션-되돌리기-alembic-downgrade)
- [4. 실무적인 마이그레이션 전략](#4-실무적인-마이그레이션-전략)
  - [4.1. 데이터 마이그레이션 (Data Migrations)](#41-데이터-마이그레이션-data-migrations)
  - [4.2. 스쿼싱 (Squashing Migrations)](#42-스쿼싱-squashing-migrations)
  - [4.3. 무중단 배포를 위한 마이그레이션](#43-무중단-배포를-위한-마이그레이션)

---

## 1. 데이터베이스 마이그레이션이란?

데이터베이스 마이그레이션은 데이터베이스 스키마(테이블 구조, 컬럼, 제약 조건 등)의 변경 이력을 관리하는 프로세스입니다. 애플리케이션이 발전함에 따라 데이터 모델이 변경되는 것은 자연스러운 일이며, 마이그레이션 도구는 이러한 변경 사항을 추적하고 데이터베이스에 안전하게 적용할 수 있도록 돕습니다.

### 1.1. 왜 마이그레이션이 필요한가?
-   **버전 관리**: 데이터베이스 스키마의 변경 사항을 파일 형태로 관리하여 버전 관리가 가능합니다.
-   **협업**: 여러 개발자가 동시에 작업할 때 데이터베이스 스키마 변경으로 인한 충돌을 방지하고, 모든 팀원이 동일한 스키마를 유지하도록 돕습니다.
-   **배포 자동화**: 개발 환경에서 테스트된 스키마 변경 사항을 운영 환경에 안전하고 자동화된 방식으로 적용할 수 있습니다.
-   **롤백**: 문제가 발생했을 때 이전 스키마 상태로 쉽게 되돌릴 수 있습니다.

### 1.2. Alembic의 역할
Alembic은 SQLAlchemy를 위한 경량 데이터베이스 마이그레이션 도구입니다. SQLAlchemy ORM 모델의 변경 사항을 감지하여 마이그레이션 스크립트를 자동으로 생성하고, 이를 데이터베이스에 적용하거나 되돌리는 기능을 제공합니다.

## 2. Alembic 초기 설정

Alembic을 사용하기 위한 초기 설정 과정을 학습합니다.

### 2.1. Alembic 설치
먼저 Alembic 라이브러리를 설치합니다.

```bash
pip install alembic
```

### 2.2. Alembic 프로젝트 초기화
FastAPI 프로젝트의 루트 디렉토리에서 다음 명령어를 실행하여 Alembic 프로젝트를 초기화합니다.

```bash
alembic init migrations
```
이 명령은 `migrations`라는 디렉토리를 생성하고, 그 안에 Alembic 설정 파일(`alembic.ini`)과 마이그레이션 스크립트가 저장될 `versions` 디렉토리, 그리고 `env.py` 파일을 생성합니다.

-   `alembic.ini`: Alembic의 전역 설정 파일입니다. 데이터베이스 연결 정보, 스크립트 디렉토리 경로 등을 정의합니다.
-   `migrations/env.py`: 마이그레이션 스크립트가 실행될 때 호출되는 파이썬 스크립트입니다. 데이터베이스 연결 설정, SQLAlchemy 모델 메타데이터 로드 등을 담당합니다.

### 2.3. `env.py` 설정
`migrations/env.py` 파일을 수정하여 SQLAlchemy 모델의 메타데이터를 Alembic이 인식할 수 있도록 설정해야 합니다. 이 파일은 마이그레이션이 실행될 때마다 호출됩니다.

```python
# migrations/env.py

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# FastAPI 프로젝트의 루트 디렉토리를 sys.path에 추가
# 이렇게 해야 Alembic이 models.py와 같은 파일을 찾을 수 있습니다.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# FastAPI 프로젝트의 데이터베이스 설정 및 모델 임포트
# from app.config import DATABASE_URL, Base # 예시: app/config.py에 정의된 경우
# from app.models import User, Item # 예시: app/models.py에 정의된 모델들

# --- 실제 프로젝트에 맞게 수정해야 할 부분 ---
# 1. 데이터베이스 URL 설정
# config.set_main_option("sqlalchemy.url", DATABASE_URL) # 또는 직접 URL 문자열 지정
config.set_main_option("sqlalchemy.url", os.environ.get("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/test_db"))

# 2. SQLAlchemy Base 메타데이터 임포트
# 모든 SQLAlchemy 모델의 Base 클래스를 여기에 임포트해야 합니다.
# Alembic은 이 Base.metadata를 사용하여 현재 모델의 스키마를 파악합니다.
from app.database import Base # app/database.py에 Base가 정의되어 있다고 가정
target_metadata = Base.metadata
# --- 수정해야 할 부분 끝 ---

# ... (나머지 env.py 내용은 그대로 유지) ...

def run_migrations_offline():
    # ...
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    # ...

def run_migrations_online():
    # ...
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        url=config.get_main_option("sqlalchemy.url"),
        # ...
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # ...
        )
        # ...
```

## 3. 마이그레이션 파일 생성 및 적용

모델을 변경할 때마다 마이그레이션 파일을 생성하고 데이터베이스에 적용하는 과정을 학습합니다.

### 3.1. 마이그레이션 파일 생성 (`alembic revision --autogenerate`)
SQLAlchemy 모델(`models.py`)을 변경한 후, 다음 명령어를 실행하여 변경 사항을 감지하고 마이그레이션 스크립트를 자동으로 생성합니다.

```bash
alembic revision --autogenerate -m "Add new user fields"
```
-   `-m "메시지"`: 마이그레이션 파일에 포함될 설명 메시지입니다.
-   `--autogenerate`: 현재 데이터베이스 스키마와 `target_metadata`에 정의된 모델 스키마를 비교하여 변경 사항을 자동으로 감지하고 스크립트를 생성합니다.

이 명령은 `migrations/versions/` 디렉토리에 새로운 파이썬 파일(예: `xxxxxxxxxxxx_add_new_user_fields.py`)을 생성합니다. 이 파일에는 `upgrade()` 함수(스키마 변경 적용)와 `downgrade()` 함수(스키마 변경 되돌리기)가 포함됩니다.

**생성된 마이그레이션 파일 예시:**
```python
# migrations/versions/xxxxxxxxxxxx_add_new_user_fields.py

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'xxxxxxxxxxxx'
down_revision = 'yyyyyyyyyyyy' # 이전 마이그레이션의 revision
branch_labels = None
depends_on = None

def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('users', sa.Column('phone_number', sa.String(length=20), nullable=True))
    op.add_column('users', sa.Column('address', sa.String(length=255), nullable=True))
    # ### end Alembic commands ###

def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('users', 'address')
    op.drop_column('users', 'phone_number')
    # ### end Alembic commands ###
```
**주의:** `autogenerate`는 완벽하지 않을 수 있습니다. 생성된 마이그레이션 파일을 **반드시 검토**하고, 필요한 경우 수동으로 수정해야 합니다. 특히 데이터 마이그레이션(데이터 내용 변경)은 `autogenerate`가 처리하지 못하므로 수동으로 작성해야 합니다.

### 3.2. 마이그레이션 적용 (`alembic upgrade`)
생성된 마이그레이션 파일을 데이터베이스에 적용합니다.

```bash
alembic upgrade head
```
-   `head`: 아직 적용되지 않은 모든 마이그레이션을 최신 버전까지 적용합니다.
-   `alembic upgrade +1`: 다음 마이그레이션 하나만 적용합니다.
-   `alembic upgrade <revision_id>`: 특정 `revision_id`까지 적용합니다.

### 3.3. 마이그레이션 되돌리기 (`alembic downgrade`)
적용된 마이그레이션을 이전 상태로 되돌립니다.

```bash
alembic downgrade -1
```
-   `-1`: 가장 최근에 적용된 마이그레이션 하나를 되돌립니다.
-   `base`: 모든 마이그레이션을 초기 상태로 되돌립니다.
-   `alembic downgrade <revision_id>`: 특정 `revision_id`까지 되돌립니다.

## 4. 실무적인 마이그레이션 전략

대규모 애플리케이션에서는 마이그레이션을 더욱 신중하게 다루어야 합니다.

### 4.1. 데이터 마이그레이션 (Data Migrations)
스키마 변경 외에, 데이터 자체를 조작해야 할 때 사용합니다. 예를 들어, 기존 컬럼의 데이터를 새로운 컬럼으로 옮기거나, 특정 조건에 따라 데이터를 업데이트하는 경우입니다. `autogenerate`는 데이터 마이그레이션을 생성하지 않으므로 수동으로 작성해야 합니다.

```python
# migrations/versions/xxxxxxxxxxxx_populate_new_field.py (수동 생성)

from alembic import op
import sqlalchemy as sa

revision = 'xxxxxxxxxxxx'
down_revision = 'yyyyyyyyyyyy'

def upgrade() -> None:
    # 데이터 마이그레이션 로직
    # op.execute()를 사용하여 Raw SQL을 실행하거나,
    # SQLAlchemy ORM을 사용하여 데이터를 조작할 수 있습니다.
    op.execute("UPDATE users SET new_field = old_field * 2 WHERE condition = true")

def downgrade() -> None:
    # 롤백 로직 (데이터를 이전 상태로 되돌림)
    op.execute("UPDATE users SET old_field = new_field / 2 WHERE condition = true")
```

### 4.2. 스쿼싱 (Squashing Migrations)
프로젝트가 오래되어 마이그레이션 파일이 너무 많아지면, 테스트 데이터베이스 구축 시간이 길어지거나 관리하기 어려워집니다. 스쿼싱은 여러 마이그레이션 파일을 하나의 파일로 압축하는 기능입니다.

```bash
alembic revision --squash <start_revision>:<end_revision> -m "Squashed migrations"
```
**주의:** 이미 배포된 마이그레이션을 스쿼싱하는 것은 매우 위험합니다. 주로 새로운 메이저 버전을 배포하기 전이나, 아직 다른 팀원과 공유되지 않은 기능 브랜치 내에서 수행하는 것이 안전합니다.

### 4.3. 무중단 배포를 위한 마이그레이션
운영 중인 서비스에서는 배포 중 서버가 잠시라도 멈추거나 오류를 내뱉는 것을 최소화해야 합니다(Zero-Downtime Deployment). 마이그레이션은 이 과정에서 가장 큰 위험 요소 중 하나입니다.

-   **핵심 전략: 모든 변경을 하위 호환 가능하게 만들기 (Two-Phase Deploy)**
    *   **1단계 (코드 배포):** 새로운 필드를 추가할 때는 `nullable=True` 또는 `default` 값을 설정하여 기존 코드와 호환되도록 합니다. 이 필드를 사용하는 코드는 아직 배포하지 않습니다.
    *   **2단계 (마이그레이션 적용):** 새로운 필드가 추가된 마이그레이션을 적용합니다.
    *   **3단계 (코드 배포):** 새로운 필드를 사용하는 코드를 배포합니다.
    *   **4단계 (정리):** 필요하다면 `nullable=False`로 변경하거나 `default` 값을 제거하는 마이그레이션을 적용합니다.

이러한 전략을 통해 데이터베이스 스키마 변경을 안전하고 효율적으로 관리할 수 있습니다.