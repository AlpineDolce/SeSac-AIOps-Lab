<h2>Django Backend: ORM 및 데이터베이스 관리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django의 ORM(Object-Relational Mapping)을 통해 데이터베이스와 상호작용하는 방법을 이해하고, 모델 정의, 데이터베이스 설정, 마이그레이션, 그리고 쿼리셋(QuerySet) API를 활용한 데이터 조작 방법을 학습하는 것을 목표로 합니다.</p>

<h2>목차</h2> 

- [1. Django Model (ORM)](#1-django-model-orm)
  - [1.1. Model 정의 (models.Model 상속) - 실무적 접근](#11-model-정의-modelsmodel-상속---실무적-접근)
    - [**1. 추상 기본 클래스를 통한 공통 필드 관리**](#1-추상-기본-클래스를-통한-공통-필드-관리)
    - [**2. `Meta` 클래스를 이용한 모델 동작 제어**](#2-meta-클래스를-이용한-모델-동작-제어)
    - [**3. 필드 옵션을 통한 가독성 및 성능 향상**](#3-필드-옵션을-통한-가독성-및-성능-향상)
    - [**4. 실무적인 모델 정의 종합 예시**](#4-실무적인-모델-정의-종합-예시)
  - [1.2. `__str__` 메서드 오버라이딩: 객체에 이름표 붙이기](#12-__str__-메서드-오버라이딩-객체에-이름표-붙이기)
    - [**`__str__`의 역할과 중요성**](#__str__의-역할과-중요성)
    - [**좋은 `__str__` 구현을 위한 원칙**](#좋은-__str__-구현을-위한-원칙)
    - [**좋은 예시 vs. 나쁜 예시**](#좋은-예시-vs-나쁜-예시)
- [2. 데이터베이스 설정 및 마이그레이션](#2-데이터베이스-설정-및-마이그레이션)
  - [2.1. `settings.py`에서 데이터베이스 설정 - 실무적 접근](#21-settingspy에서-데이터베이스-설정---실무적-접근)
    - [**1. 데이터베이스 엔진 선택: 개발 vs 운영**](#1-데이터베이스-엔진-선택-개발-vs-운영)
    - [**2. 환경 변수를 이용한 설정 (`python-decouple` 활용)**](#2-환경-변수를-이용한-설정-python-decouple-활용)
    - [**3. 성능 및 보안을 위한 추가 옵션**](#3-성능-및-보안을-위한-추가-옵션)
    - [**4. `dj-database-url` 라이브러리 활용**](#4-dj-database-url-라이브러리-활용)
  - [2.2. 마이그레이션 (`makemigrations`, `migrate`) - 데이터베이스 스키마의 버전 관리](#22-마이그레이션-makemigrations-migrate---데이터베이스-스키마의-버전-관리)
    - [**실무 워크플로우: 2단계 프로세스**](#실무-워크플로우-2단계-프로세스)
    - [**팀 협업 시나리오와 문제 해결**](#팀-협업-시나리오와-문제-해결)
    - [**권장 마이그레이션 워크플로우 요약**](#권장-마이그레이션-워크플로우-요약)
  - [2.3. 데이터베이스 마이그레이션 고급 주제: 실무 심화](#23-데이터베이스-마이그레이션-고급-주제-실무-심화)
    - [2.3.1. 마이그레이션 스쿼싱 (Squashing Migrations)](#231-마이그레이션-스쿼싱-squashing-migrations)
    - [2.3.2. 데이터 마이그레이션 (Data Migrations)](#232-데이터-마이그레이션-data-migrations)
    - [2.3.3. 무중단 배포를 위한 마이그레이션 전략](#233-무중단-배포를-위한-마이그레이션-전략)
- [3. Model 필드 타입과 옵션: 실무 가이드](#3-model-필드-타입과-옵션-실무-가이드)
  - [3.1. 기본 필드 타입](#31-기본-필드-타입)
  - [3.2. 날짜 및 시간 필드](#32-날짜-및-시간-필드)
  - [3.3. 관계 필드 (Relationships)](#33-관계-필드-relationships)
  - [3.4. 특수 목적 필드](#34-특수-목적-필드)
  - [3.5. 모든 필드의 공통 옵션](#35-모든-필드의-공통-옵션)
- [4. 쿼리셋(QuerySet) API](#4-쿼리셋queryset-api)
  - [4.1. 데이터 조회: 쿼리셋(QuerySet)의 이해와 활용](#41-데이터-조회-쿼리셋queryset의-이해와-활용)
    - [**쿼리셋의 핵심: 지연 평가 (Lazy Evaluation)**](#쿼리셋의-핵심-지연-평가-lazy-evaluation)
    - [**1. 모든 객체 조회: `all()`**](#1-모든-객체-조회-all)
    - [**2. 단 하나의 객체 조회: `get()`**](#2-단-하나의-객체-조회-get)
    - [**3. 조건에 맞는 객체 필터링: `filter()` 와 `exclude()`**](#3-조건에-맞는-객체-필터링-filter-와-exclude)
    - [**4. 기타 유용한 조회 메서드**](#4-기타-유용한-조회-메서드)
  - [4.2. 데이터 생성: `create()`, `save()`, `get_or_create()`, `bulk_create()`](#42-데이터-생성-create-save-get_or_create-bulk_create)
    - [**방법 1: `create()` - 가장 간단한 한 줄 생성**](#방법-1-create---가장-간단한-한-줄-생성)
    - [**방법 2: `save()` - 유연한 2단계 생성 (실무 핵심)**](#방법-2-save---유연한-2단계-생성-실무-핵심)
    - [**방법 3: `get_or_create()` - 중복 방지 생성**](#방법-3-get_or_create---중복-방지-생성)
    - [**방법 4: `bulk_create()` - 대량 데이터 고속 생성**](#방법-4-bulk_create---대량-데이터-고속-생성)
  - [4.3. 데이터 수정: `save()`, `update()`, 그리고 `F()` 표현식](#43-데이터-수정-save-update-그리고-f-표현식)
    - [**방법 1: `save()` - 객체 단위 수정 (실무 핵심)**](#방법-1-save---객체-단위-수정-실무-핵심)
    - [**`save()` 사용 시 주의점: Race Condition**](#save-사용-시-주의점-race-condition)
    - [**방법 2: `F()` 표현식 - 원자적 연산으로 Race Condition 해결**](#방법-2-f-표현식---원자적-연산으로-race-condition-해결)
    - [**방법 3: `update()` - 여러 객체를 한 번에 효율적으로 수정**](#방법-3-update---여러-객체를-한-번에-효율적으로-수정)
    - [**성능 최적화: `save(update_fields=[...])`**](#성능-최적화-saveupdate_fields)
  - [4.4. 데이터 삭제: 하드 삭제(Hard Delete) vs 소프트 삭제(Soft Delete)](#44-데이터-삭제-하드-삭제hard-delete-vs-소프트-삭제soft-delete)
    - [**방법 1: 하드 삭제 (Hard Delete) - 기본 `delete()`**](#방법-1-하드-삭제-hard-delete---기본-delete)
    - [**방법 2: 소프트 삭제 (Soft Deletion) - 실무 권장 패턴**](#방법-2-소프트-삭제-soft-deletion---실무-권장-패턴)
    - [**소프트 삭제된 데이터 관리 (휴지통 기능)**](#소프트-삭제된-데이터-관리-휴지통-기능)
  - [4.5. 고급 쿼리셋 및 최적화: 실무 심화](#45-고급-쿼리셋-및-최적화-실무-심화)
    - [**4.5.1. 쿼리 성능 최적화**](#451-쿼리-성능-최적화)
    - [**4.5.2. 복잡한 쿼리 작성**](#452-복잡한-쿼리-작성)
    - [**4.5.3. 코드 추상화 및 재사용**](#453-코드-추상화-및-재사용)
    - [4.5.4. 트랜잭션 (Transactions)](#454-트랜잭션-transactions)
    - [4.5.5. 커스텀 매니저 (Custom Managers)](#455-커스텀-매니저-custom-managers)
  - [4.6. 직접 데이터베이스 접근 및 Raw 쿼리: 최후의 수단](#46-직접-데이터베이스-접근-및-raw-쿼리-최후의-수단)
    - [**언제 Raw 쿼리를 사용해야 하는가?**](#언제-raw-쿼리를-사용해야-하는가)
    - [**방법 1: `Model.objects.raw()` - 가장 안전하고 권장되는 방법**](#방법-1-modelobjectsraw---가장-안전하고-권장되는-방법)
    - [**방법 2: `connection.cursor()` - 가장 낮은 수준의 직접 접근**](#방법-2-connectioncursor---가장-낮은-수준의-직접-접근)
    - [**상황별 Raw 쿼리 사용법 요약**](#상황별-raw-쿼리-사용법-요약)

---

## 1. Django Model (ORM)

Django의 Model은 데이터베이스의 테이블을 파이썬 클래스로 추상화한 것입니다. ORM(Object-Relational Mapping)을 통해 개발자는 SQL 쿼리를 직접 작성하지 않고도 파이썬 코드로 데이터베이스를 조작할 수 있습니다.

### 1.1. Model 정의 (models.Model 상속) - 실무적 접근

Django 모델은 **"데이터에 대한 단 하나의 진정한 출처(the single source of truth)"**입니다. 즉, 모델 클래스는 애플리케이션 데이터의 구조, 제약 조건, 관계, 그리고 동작을 모두 정의하는 중심적인 역할을 합니다.

모든 모델은 `django.db.models.Model`을 상속받는 파이썬 클래스이며, 각 클래스는 데이터베이스 테이블에, 클래스의 각 속성(필드)은 테이블의 컬럼에 매핑됩니다. Django는 별도로 지정하지 않는 한, `id = models.AutoField(primary_key=True)` 필드를 자동으로 추가하여 모든 테이블이 고유한 기본 키를 갖도록 보장합니다.

단순히 필드를 나열하는 것을 넘어, 잘 설계된 모델은 다음과 같은 실무적 기법들을 포함합니다.

---

#### **1. 추상 기본 클래스를 통한 공통 필드 관리**

많은 모델들이 생성일시(`created_at`)나 수정일시(`updated_at`) 같은 공통 필드를 가집니다. 이 필드들을 모델마다 반복해서 작성하는 대신, **추상 기본 클래스(Abstract Base Class)**로 분리하면 코드 중복을 줄이고 유지보수성을 크게 향상시킬 수 있습니다.

- **`class Meta`** 내부에 **`abstract = True`**를 설정하면, Django는 이 모델을 위한 데이터베이스 테이블을 생성하지 않습니다. 대신 다른 모델들이 상속받아 필드를 물려주기 위한 용도로만 사용됩니다.

**`common/models.py` 예시:**
```python
from django.db import models

class TimestampedModel(models.Model):
    """
    생성 및 수정일시를 자동으로 기록하는 추상 기본 클래스 모델
    """
    created_at = models.DateTimeField("생성일시", auto_now_add=True)
    updated_at = models.DateTimeField("수정일시", auto_now=True)

    class Meta:
        abstract = True
```

---

#### **2. `Meta` 클래스를 이용한 모델 동작 제어**

모델 클래스 내부에 `class Meta`를 정의하여 모델의 다양한 동작을 제어할 수 있습니다. 이는 모델의 메타데이터를 설정하는 강력한 방법입니다.

- **`ordering`**: 쿼리 시 특별히 순서를 지정하지 않았을 때 사용될 기본 정렬 순서를 정의합니다. `['-created_at']`처럼 필드 이름 앞에 `-`를 붙이면 내림차순으로 정렬됩니다. 이를 통해 불필요한 `.order_by()` 호출을 줄일 수 있습니다.
- **`verbose_name`** / **`verbose_name_plural`**: Django 관리자 페이지 등에서 표시될 모델의 단수형/복수형 이름을 지정합니다. (예: `verbose_name="게시글"`, `verbose_name_plural="게시글 목록"`)
- **`db_table`**: Django가 자동으로 생성하는 테이블 이름(예: `blog_post`) 대신, 사용할 테이블 이름을 직접 지정합니다. (레거시 데이터베이스 연동 시 유용)
- **`constraints`**: 둘 이상의 필드를 조합한 복합 제약 조건(예: 복합 기본 키, 복합 유니크 제약)을 정의합니다. `UniqueConstraint`를 사용하여 특정 필드 조합이 테이블 내에서 유일하도록 강제할 수 있습니다.

---

#### **3. 필드 옵션을 통한 가독성 및 성능 향상**

- **`verbose_name`**: 필드의 레이블을 지정합니다. (예: `models.CharField("게시글 제목", ...)`)
- **`help_text`**: 관리자 페이지나 폼에서 필드 아래에 표시될 도움말 텍스트를 제공하여 사용자 입력을 돕습니다.
- **`db_index=True`**: 해당 컬럼에 데이터베이스 인덱스를 생성합니다. `filter()`, `exclude()`, `order_by()` 등에서 자주 사용되는 필드(특히 `ForeignKey`)에 인덱스를 추가하면 조회 성능이 크게 향상됩니다.
- **`related_name`**: `ForeignKey`나 `ManyToManyField`에서 역참조 시 사용할 이름을 지정합니다. 예를 들어 `Post` 모델의 `author` 필드에 `related_name='posts'`를 지정하면, `user.posts.all()`과 같이 직관적인 코드로 특정 유저가 작성한 모든 게시글을 조회할 수 있습니다.

---

#### **4. 실무적인 모델 정의 종합 예시**

위 개념들을 모두 적용하여 블로그의 `Post`(게시글)와 `Comment`(댓글) 모델을 정의한 예시입니다.

```python
# blog/models.py

from django.db import models
from django.conf import settings

# 1. 다른 앱(예: common)에 정의된 추상 모델을 임포트
# from common.models import TimestampedModel 

# (예시를 위해 여기에 TimestampedModel을 다시 정의)
class TimestampedModel(models.Model):
    created_at = models.DateTimeField("생성일시", auto_now_add=True)
    updated_at = models.DateTimeField("수정일시", auto_now=True)
    class Meta:
        abstract = True

class Post(TimestampedModel):
    # 2. 사용자 모델 참조 시 settings.AUTH_USER_MODEL 사용 권장
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        verbose_name="작성자",
        on_delete=models.CASCADE,
        related_name="posts", # User 모델에서 post들을 조회할 때 사용할 이름
        db_index=True, # 작성자로 검색하는 경우가 많으므로 인덱스 추가
    )
    title = models.CharField("제목", max_length=255)
    content = models.TextField("내용")
    is_published = models.BooleanField("공개 여부", default=False, db_index=True)

    # 3. Meta 클래스를 통한 상세 설정
    class Meta:
        ordering = ["-created_at"] # 기본 정렬을 최신순으로
        verbose_name = "게시글"
        verbose_name_plural = "게시글 목록"
        constraints = [
            # 한 명의 유저는 동일한 제목의 게시글을 중복해서 작성할 수 없도록 제약
            models.UniqueConstraint(fields=['author', 'title'], name='unique_post_title_per_author')
        ]

    def __str__(self):
        return self.title

class Comment(TimestampedModel):
    post = models.ForeignKey(
        Post,
        verbose_name="게시글",
        on_delete=models.CASCADE,
        related_name="comments", # Post 모델에서 댓글들을 조회할 때 사용할 이름
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        verbose_name="작성자",
        on_delete=models.CASCADE,
        related_name="comments",
    )
    content = models.TextField("댓글 내용", help_text="여기에 댓글을 입력하세요.")

    class Meta:
        ordering = ["created_at"] # 댓글은 오래된 순으로 정렬
        verbose_name = "댓글"
        verbose_name_plural = "댓글 목록"

    def __str__(self):
        return f"{self.author}의 댓글: {self.content[:20]}..."
```

### 1.2. `__str__` 메서드 오버라이딩: 객체에 이름표 붙이기

`__str__` 메서드는 파이썬의 "매직 메서드(magic method)" 중 하나로, 특정 객체를 사람이 읽을 수 있는 형태의 문자열로 표현하는 방법을 정의합니다. Django 모델에서 이 메서드를 오버라이딩하는 것은 작지만 매우 중요한 습관입니다.

#### **`__str__`의 역할과 중요성**

`__str__`을 정의하지 않으면, Django는 객체를 `<Post object (1)>`과 같이 모호한 형태로 표시합니다. 이는 개발과 관리에 큰 불편을 초래합니다.

1.  **Django 관리자 페이지 가독성 향상**: `__str__`의 가장 큰 수혜자는 관리자 페이지입니다. 객체 목록이나 다른 모델과의 관계를 설정하는 드롭다운 메뉴에 `__str__`이 반환하는 값이 표시되어, 개발자나 운영자가 데이터를 훨씬 쉽게 식별하고 관리할 수 있게 됩니다.
2.  **디버깅 및 로깅 효율 증가**: `print(my_post)`를 실행하거나 로그 메시지에 객체를 포함할 때, `__str__`의 결과가 출력됩니다. "<Post: 첫 번째 게시글>" 처럼 명확한 정보는 디버깅 속도를 크게 향상시킵니다.
3.  **템플릿 및 폼에서의 활용**: 템플릿에서 `{{ post }}`처럼 객체를 직접 출력하거나, 폼의 `ModelChoiceField` 등에서 객체를 선택하는 위젯에 `__str__`의 결과가 사용됩니다.

#### **좋은 `__str__` 구현을 위한 원칙**

- **간결하고 명확하게**: 객체를 대표할 수 있는 가장 핵심적인 정보를 담되, 너무 길지 않아야 합니다. 보통 제목, 이름, 아이디 등 고유성을 나타내는 필드가 주로 사용됩니다.
- **F-string 활용**: 파이썬 3.6 이상부터 도입된 f-string을 사용하면 여러 필드를 조합하여 문자열을 만들 때 가장 직관적이고 가독성이 좋습니다.
- **긴 내용은 반드시 생략**: `TextField`와 같이 내용이 긴 필드를 `__str__`에 그대로 포함하면 안 됩니다. 관리자 페이지 목록이 깨지거나 로그가 지저분해집니다. 포함해야 한다면 `self.content[:30]...` 와 같이 일부만 잘라서 표시합니다.
- **관련 객체 정보 활용**: `ForeignKey`로 연결된 객체의 정보를 함께 표시하면 더 풍부한 컨텍스트를 제공할 수 있습니다. (예: `f'{self.post.title}에 달린 {self.author.username}의 댓글'`)

#### **좋은 예시 vs. 나쁜 예시**

이전 `1.1` 섹션의 `Post`와 `Comment` 모델을 기준으로 비교해 보겠습니다.

**Post 모델**
```python
class Post(TimestampedModel):
    # ... 필드 정의 ...

    # 좋은 예시: 게시글의 제목을 반환하여 명확하게 식별
    def __str__(self):
        return self.title

    # 나쁜 예시: 너무 많은 정보를 담고 있으며, 긴 내용(content)을 포함
    # def __str__(self):
    #     return f'{self.id} - {self.title} by {self.author.username} / {self.content}'
```

**Comment 모델**
```python
class Comment(TimestampedModel):
    # ... 필드 정의 ...

    # 좋은 예시: 어떤 게시글에 누가 쓴 댓글인지 컨텍스트를 명확히 함
    def __str__(self):
        return f"{self.author.username}의 댓글 (게시글: {self.post.title[:15]}...)"

    # 나쁜 예시: 댓글 내용만 반환하여 어떤 댓글인지 식별하기 어려움
    # def __str__(self):
    #     return self.content
```

이처럼 `__str__` 메서드를 잘 정의하는 것은 Django 애플리케이션의 사용성과 유지보수성을 높이는 기본적이면서도 중요한 단계입니다.

## 2. 데이터베이스 설정 및 마이그레이션

### 2.1. `settings.py`에서 데이터베이스 설정 - 실무적 접근

데이터베이스 설정은 Django 프로젝트의 심장과도 같습니다. 이 설정은 **절대로 소스 코드에 직접 하드코딩해서는 안 되며**, 이전 문서에서 설명한 **환경 변수**를 통해 관리하는 것이 철칙입니다.

#### **1. 데이터베이스 엔진 선택: 개발 vs 운영**

- **개발 환경: SQLite**
    - **장점**: 별도의 서버 설치나 설정 없이, 단일 파일(`db.sqlite3`)로 데이터베이스를 구성할 수 있어 매우 편리합니다. 새로운 프로젝트를 시작하거나 간단한 기능을 테스트할 때 이상적입니다.
    - **단점**: 동시성 처리(여러 요청이 동시에 DB에 접근)에 취약하고, 실제 운영 환경에서 사용되는 PostgreSQL이나 MySQL과 기능적 차이가 있어, 개발 환경에서는 문제가 없던 코드가 운영 환경에서 오류를 일으키는 원인이 될 수 있습니다.

- **운영 환경: PostgreSQL (권장) 또는 MySQL/MariaDB**
    - **PostgreSQL**: Django 커뮤니티에서 가장 널리 권장되는 데이터베이스입니다. 데이터 무결성, 동시성 처리 능력이 뛰어나며, JSONB, 배열 필드 등 Django의 고급 기능들을 완벽하게 지원합니다. (`pip install psycopg2-binary` 필요)
    - **MySQL/MariaDB**: PostgreSQL과 함께 가장 많이 사용되는 오픈소스 데이터베이스로, 역시 안정적이고 성능이 뛰어납니다. 오랜 기간 검증되었으며 관련 자료를 찾기 쉽다는 장점이 있습니다. (`pip install mysqlclient` 필요)

**실무 Tip**: 최종 운영 환경에서 사용할 데이터베이스와 **동일한 종류의 데이터베이스를 개발 환경에서도 사용하는 것**이 가장 이상적입니다. Docker를 사용하면 로컬 환경에서도 PostgreSQL이나 MySQL 서버를 손쉽게 구성할 수 있습니다.

#### **2. 환경 변수를 이용한 설정 (`python-decouple` 활용)**

`settings/production.py` (또는 `settings/base.py`) 파일에 다음과 같이 `decouple`을 사용하여 데이터베이스 설정을 구성합니다.

```python
# settings/production.py
from decouple import config

DATABASES = {
    'default': {
        # .env 파일에서 DB 종류, 이름, 계정 정보 등을 읽어옴
        'ENGINE': config('DB_ENGINE', default='django.db.backends.sqlite3'),
        'NAME': config('DB_NAME', default='db.sqlite3'),
        'USER': config('DB_USER', default=''),
        'PASSWORD': config('DB_PASSWORD', default=''),
        'HOST': config('DB_HOST', default='localhost'),
        'PORT': config('DB_PORT', default='', cast=int),
    }
}
```

이에 해당하는 `.env` 파일은 다음과 같이 작성할 수 있습니다.

```ini
# .env 파일

# PostgreSQL 사용 시
DB_ENGINE=django.db.backends.postgresql
DB_NAME=my_project_db
DB_USER=my_user
DB_PASSWORD=supersecretpassword
DB_HOST=localhost
DB_PORT=5432
```

#### **3. 성능 및 보안을 위한 추가 옵션**

- **`CONN_MAX_AGE` (커넥션 재사용)**: 매 요청마다 데이터베이스에 새로 연결하는 대신, 기존 연결을 재사용하여 성능을 향상시킵니다. `0`(기본값)은 매번 새로운 연결을, `None`은 무한정 유지, 양수 값은 해당 시간(초)만큼 연결을 유지합니다. 일반적으로 `60` ~ `300` 사이의 값을 설정하는 것이 권장됩니다.

- **SSL 연결 강제**: 운영 환경에서는 데이터베이스와 Django 서버 간의 통신을 암호화하는 것이 안전합니다. `OPTIONS` 딕셔너리를 통해 SSL 설정을 추가할 수 있습니다.

**모든 옵션을 적용한 예시:**
```python
# settings/production.py
DATABASES = {
    'default': {
        'ENGINE': config('DB_ENGINE'),
        'NAME': config('DB_NAME'),
        'USER': config('DB_USER'),
        'PASSWORD': config('DB_PASSWORD'),
        'HOST': config('DB_HOST'),
        'PORT': config('DB_PORT', cast=int),
        # 커넥션 수명을 300초(5분)로 설정
        'CONN_MAX_AGE': config('DB_CONN_MAX_AGE', default=300, cast=int),
        # SSL 연결을 위한 옵션
        'OPTIONS': {
            'sslmode': config('DB_SSL_MODE', default='require') # 'require', 'verify-ca', 'verify-full'
        }
    }
}
```

#### **4. `dj-database-url` 라이브러리 활용**

Heroku와 같은 클라우드 플랫폼에서는 데이터베이스 접속 정보를 `DATABASE_URL`이라는 단일 환경 변수로 제공하는 경우가 많습니다. `dj-database-url` 라이브러리는 이 URL을 Django의 `DATABASES` 설정 포맷으로 자동 변환해주는 매우 편리한 도구입니다.

1.  **설치**: `pip install dj-database-url`
2.  **`.env` 파일**: `DATABASE_URL` 변수를 정의합니다.
    ```ini
    # postgresql://[사용자이름]:[비밀번호]@[호스트]:[포트]/[DB이름]
    DATABASE_URL=postgresql://my_user:supersecretpassword@localhost:5432/my_project_db
    ```
3.  **`settings.py` 설정**:
    ```python
    # settings/production.py
    import dj_database_url
    from decouple import config

    DATABASES = {
        # DATABASE_URL 환경 변수를 읽어와 Django 설정으로 변환
        # conn_max_age, ssl_require 등 추가 옵션도 함께 설정 가능
        'default': dj_database_url.config(
            default=config('DATABASE_URL'),
            conn_max_age=600,
            ssl_require=True
        )
    }
    ```
이 방식을 사용하면 여러 개의 DB 관련 환경 변수를 하나로 통합하여 관리가 더욱 간편해집니다.

### 2.2. 마이그레이션 (`makemigrations`, `migrate`) - 데이터베이스 스키마의 버전 관리

Django의 마이그레이션 시스템은 **데이터베이스 스키마를 위한 버전 관리 시스템(VCS)**과 같습니다. Git이 소스 코드의 변경 이력을 관리하듯, 마이그레이션은 `models.py`의 변경 이력을 파일로 만들어 관리하고, 이를 통해 모든 개발팀원과 운영 서버가 동일한 데이터베이스 구조를 갖도록 보장합니다.

**주요 장점:**
- **자동화**: 모델 변경에 필요한 SQL DDL(Data Definition Language)을 자동으로 생성합니다.
- **플랫폼 독립성**: 생성된 마이그레이션 파일 하나로 PostgreSQL, MySQL, SQLite 등 다양한 데이터베이스에 동일한 스키마를 적용할 수 있습니다.
- **이력 관리**: 모든 스키마 변경 사항이 파일로 기록되어 추적이 용이하며, 필요시 이전 상태로 롤백할 수 있습니다.

#### **실무 워크플로우: 2단계 프로세스**

모델 변경부터 데이터베이스 적용까지의 과정은 항상 다음의 2단계를 따릅니다.

**1단계: `makemigrations` - 변경 사항 설계도 생성**

`python manage.py makemigrations [app_name]` 명령어는 현재의 모델 코드와 마지막 마이그레이션 파일의 상태를 비교하여, 변경된 내용에 대한 **마이그레이션 파일(설계도)**을 생성합니다.

- **무엇을 하는가?**: `models.py`의 변경점을 감지하여 `0002_add_field_to_post.py`와 같은 파이썬 파일을 `migrations/` 디렉토리에 생성합니다. 이 파일 안에는 `migrations.CreateModel`, `migrations.AddField` 등 스키마 변경에 필요한 작업들이 파이썬 코드로 기술되어 있습니다.
- **실무 Tip 1: 특정 앱 지정**: `python manage.py makemigrations blog`처럼 변경이 발생한 앱 이름을 지정하는 것이 좋습니다. 이렇게 하면 관련 없는 다른 앱의 변경사항이 섞이지 않아 마이그레이션 관리가 깔끔해집니다.
- **실무 Tip 2: 이름 지정**: `makemigrations blog --name add_is_published_to_post` 와 같이 `--name` 옵션으로 마이그레이션의 목적을 파일명에 명시하면, 나중에 파일 이름만 보고도 어떤 변경이었는지 쉽게 파악할 수 있습니다.
- **대화형 프롬프트**: `null=False`인 필드를 기존 데이터가 있는 테이블에 추가하는 경우처럼, Django가 스스로 결정할 수 없는 변경이 생기면 터미널에 질문을 표시합니다. 예를 들어, "새로운 필드에 어떤 기본값을 채워넣을까요?"라고 물을 수 있습니다. 상황에 맞게 적절히 답해야 합니다.

**2단계: `migrate` - 설계도를 실제 데이터베이스에 시공**

`python manage.py migrate [app_name] [migration_name]` 명령어는 생성된 마이그레이션 파일을 읽어 실제 데이터베이스에 적용합니다.

- **무엇을 하는가?**: 아직 적용되지 않은 마이그레이션 파일들을 순서대로 실행하여, 실제 SQL 명령을 데이터베이스에 전송합니다. 어떤 마이그레이션까지 적용되었는지는 `django_migrations`라는 별도의 테이블에 기록됩니다.
- **실무 Tip 3: 실행될 SQL 사전 검토**: 특히 중요한 운영 데이터베이스에 `migrate`를 실행하기 전에는, **반드시 `sqlmigrate` 명령어로 어떤 SQL이 실행될지 미리 확인**하는 습관을 들이는 것이 안전합니다. 이는 예상치 못한 데이터 손실이나 스키마 변경을 막아주는 중요한 안전장치입니다.
  ```bash
  # blog 앱의 0002번 마이그레이션이 실행할 SQL 문을 미리 확인
  python manage.py sqlmigrate blog 0002
  ```
- **특정 마이그레이션으로의 적용 및 롤백**: 
    - `migrate blog 0002`: `blog` 앱의 마이그레이션을 0002번까지 적용합니다.
    - `migrate blog zero`: `blog` 앱의 **모든** 마이그레이션을 되돌립니다(롤백). 데이터가 손실될 수 있으므로 운영 환경에서는 절대 사용해서는 안 되며, 로컬 개발 환경에서 스키마를 완전히 초기화하고 싶을 때만 제한적으로 사용해야 합니다.

#### **팀 협업 시나리오와 문제 해결**

- **마이그레이션 충돌**: 여러 개발자가 동일한 앱의 모델을 동시에 수정하고 각자 `makemigrations`를 실행하면, 마이그레이션 파일들의 순서가 꼬이거나 의존성 문제가 발생할 수 있습니다. 이 경우, Git 브랜치를 합치기(merge) 전에 한쪽 브랜치를 재정렬(rebase)하고 마이그레이션을 다시 생성하거나, 마이그레이션 파일의 `dependencies` 속성을 수동으로 수정하여 해결해야 합니다.
- **"No changes detected"**: 모델을 수정했는데도 `makemigrations`가 변경을 감지하지 못한다면, `models.py` 파일이 제대로 저장되었는지, 혹은 앱이 `INSTALLED_APPS`에 올바르게 등록되었는지 확인해야 합니다.

#### **권장 마이그레이션 워크플로우 요약**

1.  `models.py` 파일의 모델 클래스를 수정합니다.
2.  `python manage.py makemigrations <app_name>` 명령으로 마이그레이션 파일을 생성합니다.
3.  (권장) `python manage.py sqlmigrate <app_name> <migration_number>` 명령으로 생성될 SQL을 검토합니다.
4.  `python manage.py migrate` 명령으로 변경사항을 데이터베이스에 적용합니다.
5.  **변경된 `models.py` 파일과 새로 생성된 마이그레이션 파일을 함께 Git에 커밋합니다.**

### 2.3. 데이터베이스 마이그레이션 고급 주제: 실무 심화

단순한 모델 변경을 넘어, 운영 중인 대규모 서비스를 다룰 때는 마이그레이션을 훨씬 더 신중하고 전략적으로 다루어야 합니다. 이 섹션에서는 실무에서 마주할 수 있는 복잡한 시나리오와 그 해결책을 다룹니다.

#### 2.3.1. 마이그레이션 스쿼싱 (Squashing Migrations)

- **문제점 (Why?):** 프로젝트가 오래되어 앱 하나에 수백 개의 마이그레이션 파일이 쌓이면, `python manage.py test` 실행 시 테스트 데이터베이스를 구축하는 시간이 매우 길어집니다. 또한, 스키마의 변경 이력을 파악하기가 점점 어려워집니다.
- **해결책 (How?):** `squashmigrations`는 여러 마이그레이션 파일을 하나의 최적화된 파일로 압축하여 이 문제를 해결합니다.
  ```bash
  # myapp의 모든 마이그레이션을 하나의 파일로 압축
  python manage.py squashmigrations myapp <마지막_마이그레이션_번호>
  # 예: python manage.py squashmigrations blog 0015
  ```
- **작업 흐름 및 주의사항:**
    1.  `squashmigrations` 명령을 실행하여 새로운 스쿼시 마이그레이션 파일을 생성합니다.
    2.  Django는 생성된 파일 안에 순환 의존성 등 자동 해결이 어려운 부분을 주석으로 남겨둘 수 있습니다. 파일을 열어 주석을 확인하고 필요시 수동으로 수정해야 합니다.
    3.  기존의 원본 마이그레이션 파일들(압축 대상이 된 파일들)을 삭제합니다.
    4.  새로 생성된 스쿼시 파일과 삭제된 파일 내역을 함께 Git에 커밋합니다.
    5.  **경고**: 이미 여러 운영 환경(프로덕션, 스테이징 등)에 배포된 마이그레이션을 스쿼싱하는 것은 매우 위험합니다. 각 환경의 `django_migrations` 테이블 상태와 불일치가 발생할 수 있습니다. 주로 새로운 메이저 버전을 배포하기 전이나, 아직 다른 팀원과 공유되지 않은 기능 브랜치 내에서 수행하는 것이 안전합니다.

#### 2.3.2. 데이터 마이그레이션 (Data Migrations)

스키마 구조 변경(`AddField`, `RemoveField` 등)과 별개로, **데이터 자체를 조작**해야 할 때 데이터 마이그레이션을 사용합니다. 

- **주요 사용 사례:**
    - 기존 `full_name` 필드를 `first_name`과 `last_name`으로 분리
    - 새로운 필드를 추가하고, 기존 필드의 값을 기반으로 계산하여 채워넣기
    - 외부 API에서 데이터를 가져와 초기 데이터로 설정(seeding)

- **안전한 구현 단계 (예: `full_name` -> `first_name`, `last_name` 분리):**

    1.  **1단계 (필드 추가):** 먼저 새로운 필드들을 `null=True` 옵션과 함께 추가하는 스키마 마이그레이션을 생성하고 적용합니다. 기존 코드에 영향을 주지 않기 위함입니다.
        ```python
        # models.py
        first_name = models.CharField(max_length=50, null=True)
        last_name = models.CharField(max_length=50, null=True)
        ```
        ```bash
        python manage.py makemigrations myapp --name add_name_fields
        python manage.py migrate
        ```

    2.  **2단계 (데이터 이전):** 빈 데이터 마이그레이션 파일을 생성하고, 데이터를 이전하는 로직과 **롤백을 위한 역방향 로직**을 함께 작성합니다.
        ```bash
        python manage.py makemigrations myapp --empty --name populate_split_names
        ```

        ```python
        # 생성된 000X_populate_split_names.py 파일
        from django.db import migrations

        def split_names_forward(apps, schema_editor):
            # 마이그레이션 파일에서는 반드시 apps.get_model을 사용해야 함
            User = apps.get_model('myapp', 'User')
            for user in User.objects.all():
                # 매우 단순한 예시. 실제로는 더 정교한 분리 로직 필요
                parts = user.full_name.split(' ', 1)
                user.first_name = parts[0]
                user.last_name = parts[1] if len(parts) > 1 else ''
                user.save(update_fields=['first_name', 'last_name'])

        def combine_names_backward(apps, schema_editor):
            User = apps.get_model('myapp', 'User')
            for user in User.objects.all():
                user.full_name = f'{user.first_name} {user.last_name}'.strip()
                user.save(update_fields=['full_name'])

        class Migration(migrations.Migration):
            dependencies = [
                ('myapp', '000X_add_name_fields'),
            ]
            operations = [
                migrations.RunPython(split_names_forward, reverse_code=combine_names_backward),
            ]
        ```
        `python manage.py migrate`를 실행하여 데이터를 이전합니다.

    3.  **3단계 (기존 필드 제거):** 데이터 이전이 모든 환경에 성공적으로 적용된 것을 확인한 후, 기존 `full_name` 필드를 모델에서 제거하고, `first_name`, `last_name`의 `null=True` 옵션을 제거하는 마지막 스키마 마이그레이션을 생성하고 적용합니다.

#### 2.3.3. 무중단 배포를 위한 마이그레이션 전략

운영 중인 서비스에서는 배포 중 서버가 잠시라도 멈추거나 오류를 내뱉는 것을 최소화해야 합니다(Zero-Downtime). 마이그레이션은 이 과정에서 가장 큰 위험 요소 중 하나입니다.

- **핵심 문제**: 배포가 진행되는 동안, **새로운 코드**가 **이전 데이터베이스 스키마**를 대상으로 실행되거나, **이전 코드**가 **새로운 데이터베이스 스키마**를 대상으로 실행되는 시점이 발생하여 충돌이 일어날 수 있습니다.

- **핵심 전략: 모든 변경을 하위 호환 가능하게 만들기**

    가장 일반적이고 강력한 전략은 모든 변경을 여러 단계로 나누어, 각 단계가 배포 전후의 코드 및 스키마와 모두 호환되도록 하는 **2단계 배포(Two-Phase Deploy)** 방식입니다.

    - **시나리오 1: 새로운 필드 추가**
        1.  **배포 1단계**: `models.py`에 새 필드를 `null=True` 또는 `default` 값을 설정하여 추가합니다. 이 필드를 사용하는 코드는 아직 추가하지 않습니다. 이 상태로 배포하면, 새 코드는 이 필드를 모르므로 문제가 없고, 이전 코드도 이 필드를 모르므로 문제가 없습니다. `migrate`를 실행합니다.
        2.  **배포 2단계**: 새로운 필드를 사용하는 비즈니스 로직을 코드에 추가하여 배포합니다. 이제 모든 코드가 새로운 필드의 존재를 알고 있으므로 안전합니다.
        3.  (선택) **배포 3단계**: 해당 필드가 필수(`null=False`)가 되어야 한다면, 데이터 마이그레이션을 통해 모든 기존 레코드에 값을 채워 넣은 뒤, `null=False`로 변경하는 스키마 마이그레이션을 마지막으로 적용합니다.

    - **시나리오 2: 필드 삭제 (가장 위험!)**
        1.  **배포 1단계**: 코드에서 해당 필드를 사용하는 모든 부분을 제거합니다 (`views.py`, `serializers.py`, `forms.py` 등). 모델에서는 아직 필드를 지우지 않습니다. 이 코드를 배포합니다. 이제 더 이상 어떤 코드도 이 필드에 접근하지 않습니다.
        2.  **배포 2단계**: `models.py`에서 해당 필드를 삭제하고, `makemigrations`와 `migrate`를 실행합니다. 이제 코드는 이미 해당 필드를 사용하지 않으므로 안전합니다.

    - **시나리오 3: 필드 타입 변경 또는 이름 변경**
        - 필드 삭제와 추가를 조합한 방식으로 접근합니다. 예를 들어 `title` 필드의 이름을 `headline`으로 바꾸려면, `headline` 필드를 새로 추가하고, 데이터 마이그레이션으로 `title`의 데이터를 `headline`으로 복사한 뒤, 코드에서 `title` 대신 `headline`을 사용하도록 변경하고, 마지막으로 `title` 필드를 삭제하는 여러 단계의 배포를 거쳐야 안전합니다.

이러한 전략은 번거로워 보이지만, 서비스의 안정성을 보장하기 위한 필수적인 절차입니다.


## 3. Model 필드 타입과 옵션: 실무 가이드

Django 모델 필드는 데이터베이스 컬럼의 타입과 제약조건, 그리고 Django가 데이터를 다루는 방식을 정의하는 핵심 요소입니다. 적절한 필드 타입을 선택하는 것은 데이터의 무결성을 보장하고 성능을 최적화하는 첫걸음입니다.

---

### 3.1. 기본 필드 타입

가장 흔하게 사용되는 기본적인 데이터 타입 필드들입니다.

- **`CharField`**: 짧은 길이의 문자열을 위한 필드입니다. `max_length` 인자가 반드시 필요합니다.
    - **사용 예**: 제목, 이름, 사용자 아이디, 상품 코드 등
    - **실무 Tip**: 자주 검색되는 `CharField`(예: `username`)에는 `db_index=True` 옵션을 추가하여 조회 성능을 향상시키세요.
    - `title = models.CharField("제목", max_length=200, db_index=True)`

- **`TextField`**: 긴 길이의 텍스트를 위한 필드입니다. `max_length` 옵션이 없습니다.
    - **사용 예**: 게시글 본문, 상품 상세 설명, 긴 댓글 등

- **`BooleanField`**: `True`/`False` 값을 저장합니다.
    - **사용 예**: 공개 여부(`is_published`), 활성화 상태(`is_active`)
    - **Tip**: `null=True`를 허용하는 `NullBooleanField`는 폐지(deprecated)되었습니다. `BooleanField(null=True, default=None)`와 같이 사용하여 `True`/`False`/`None` 세 가지 상태를 표현할 수 있습니다.

- **`IntegerField`**, **`BigAutoField`**, **`SmallIntegerField`**: 정수를 저장합니다.
    - `BigAutoField`: Django 3.2부터 기본 키(`id`)에 자동으로 사용되는 64비트 정수입니다.
    - `IntegerField`: 일반적인 32비트 정수입니다.
    - `SmallIntegerField`: 더 작은 범위의 정수를 저장하여 공간을 절약합니다.

- **`FloatField`** vs **`DecimalField`**: 소수점을 다루는 필드입니다. **이 둘의 구분은 매우 중요합니다.**
    - **`DecimalField`**: **금액, 환율, 이자율 등 재무 관련 데이터를 다룰 때 반드시 사용해야 합니다.** 고정 소수점 숫자로, 부동 소수점 연산에서 발생하는 미세한 오차를 방지합니다. `max_digits`(총 자릿수)와 `decimal_places`(소수부 자릿수) 인자가 필수입니다.
        - `price = models.DecimalField("가격", max_digits=10, decimal_places=2)`
    - **`FloatField`**: 부동 소수점 숫자를 저장합니다. 과학 계산이나 그래픽 처리 등 오차에 덜 민감한 분야에 사용됩니다.

---

### 3.2. 날짜 및 시간 필드

- **`DateTimeField`**: 날짜와 시간을 함께 저장합니다.
    - **`auto_now_add=True`**: 객체가 **처음 생성될 때**의 시각이 자동으로 저장됩니다. (수정 불가)
    - **`auto_now=True`**: 객체가 **저장될 때마다**(`save()`가 호출될 때마다) 현재 시각으로 자동 업데이트됩니다.
    - **주의**: `auto_now`나 `auto_now_add`가 `True`이면, 해당 필드는 `editable=False`와 `blank=True`가 자동으로 설정되어 관리자 페이지 등에서 직접 수정할 수 없게 됩니다.

- **`DateField`**: 날짜만 저장합니다. `auto_now_add`, `auto_now` 옵션을 동일하게 사용할 수 있습니다.

- **`DurationField`**: 시간의 간격(기간)을 저장합니다. (예: 동영상 재생 시간, 작업 소요 시간)

---

### 3.3. 관계 필드 (Relationships)

모델 간의 관계를 정의하며, Django ORM의 가장 강력한 기능 중 하나입니다.

- **`ForeignKey(to, on_delete, ...)`**: 다대일(Many-to-One) 관계를 정의합니다. (예: 여러 개의 `Post`는 하나의 `User`에 속함)
    - **`to`**: 관계를 맺을 대상 모델을 지정합니다.
    - **`on_delete`**: 참조하는 객체가 삭제될 때 어떻게 행동할지 정의하는 **필수 옵션**입니다.
        - `models.CASCADE`: **가장 흔하게 사용됩니다.** 참조된 객체(예: `User`)가 삭제되면, 해당 객체를 참조하던 객체들(예: `Post`)도 함께 삭제됩니다.
        - `models.PROTECT`: 참조하는 객체가 하나라도 남아있으면, 참조된 객체의 삭제를 막고 `ProtectedError`를 발생시킵니다. (예: 주문 내역이 있는 상품은 삭제 불가)
        - `models.SET_NULL`: 참조된 객체가 삭제되면, 이 필드를 `NULL`로 설정합니다. 필드에 `null=True` 옵션이 반드시 함께 설정되어야 합니다. (예: 작성자가 탈퇴해도 게시글은 남기고 작성자 정보만 비움)
        - `models.SET_DEFAULT`: 지정된 기본값으로 설정합니다. `default` 옵션이 필요합니다.
        - `models.DO_NOTHING`: 아무 행동도 하지 않습니다. 데이터베이스 레벨에서 무결성 오류를 발생시킬 수 있어 **사용을 권장하지 않습니다.**

- **`ManyToManyField(to, ...)`**: 다대다(Many-to-Many) 관계를 정의합니다. (예: 하나의 `Post`는 여러 `Tag`를, 하나의 `Tag`는 여러 `Post`에 속할 수 있음)
    - Django가 중간 테이블(intermediate table)을 자동으로 생성하여 관계를 관리합니다.
    - **`through`**: 관계에 추가적인 데이터(예: `Post`와 `Tag` 관계가 맺어진 날짜)를 저장해야 할 경우, 중간 테이블을 직접 모델로 만들어 `through` 옵션으로 지정할 수 있습니다.

- **`OneToOneField(to, on_delete, ...)`**: 일대일(One-to-One) 관계를 정의합니다. `ForeignKey`에 `unique=True` 제약조건이 걸린 것과 유사합니다.
    - **사용 예**: Django의 기본 `User` 모델을 확장하는 `UserProfile` 모델을 만들 때 주로 사용됩니다.

---

### 3.4. 특수 목적 필드

- **`EmailField`**, **`URLField`**: `CharField`를 상속받지만, 각각 이메일과 URL 형식에 대한 유효성 검사를 기본적으로 포함합니다.
- **`UUIDField`**: Auto-incrementing `id` 대신, 충돌 확률이 극히 낮은 범용 고유 식별자(UUID)를 저장합니다. 외부로 노출되는 API 엔드포인트에서 추측하기 어려운 `id`를 사용하고 싶을 때 유용합니다.
- **`FileField`**, **`ImageField`**: 파일이나 이미지를 업로드하기 위한 필드입니다.
    - `upload_to` 인자를 통해 파일이 저장될 경로를 지정할 수 있습니다.
    - `settings.py`에 `MEDIA_ROOT`와 `MEDIA_URL` 설정이 반드시 필요합니다.
    - `ImageField`는 `FileField`를 상속받으며, 업로드된 파일이 유효한 이미지인지 검사하는 기능이 추가됩니다. **`Pillow` 라이브러리(`pip install Pillow`)가 필요합니다.**
- **`JSONField`**: 유연한 JSON 데이터를 저장합니다. PostgreSQL과 함께 사용할 때 가장 강력한 성능과 기능을 발휘합니다.

---

### 3.5. 모든 필드의 공통 옵션

- **`verbose_name`**: 필드의 레이블. 관리자 페이지나 폼에 표시됩니다. (첫 번째 위치 인자로 간단히 지정 가능)
- **`help_text`**: 필드에 대한 도움말. 관리자 페이지나 폼에 표시됩니다.
- **`null` vs `blank`**: 
    - **`null=True`**: **데이터베이스** 관련 설정. DB에 `NULL` 값을 저장하는 것을 허용합니다.
    - **`blank=True`**: **유효성 검사** 관련 설정. Django 폼에서 해당 필드가 비어있는 것을 허용합니다.
    - **실무 Tip**: `CharField`나 `TextField` 같은 문자열 기반 필드에는 `null=True` 사용을 피하는 것이 일반적입니다. '값이 없음'을 표현할 때 `NULL`과 빈 문자열(`''`) 두 가지 상태가 공존하면 혼란을 야기할 수 있으므로, `default=''`를 사용하고 빈 문자열로 통일하는 것이 좋습니다.
- **`default`**: 필드의 기본값을 지정합니다.
- **`unique=True`**: 해당 컬럼의 모든 값이 유일해야 한다는 제약조건을 겁니다.
- **`db_index=True`**: 해당 컬럼에 데이터베이스 인덱스를 생성하여 조회 성능을 향상시킵니다.
- **`editable=False`**: 이 필드는 관리자 페이지나 폼에서 수정할 수 없게 됩니다.
- **`choices`**: 선택지를 제한하여 드롭다운 위젯을 만들 때 사용합니다. (예: 상태 필드)
    ```python
    class Post(models.Model):
        STATUS_CHOICES = [
            ('draft', '초안'),
            ('published', '발행됨'),
            ('archived', '보관됨'),
        ]
        status = models.CharField(
            "상태",
            max_length=10,
            choices=STATUS_CHOICES,
            default='draft'
        )
    ```

## 4. 쿼리셋(QuerySet) API

Django ORM은 `QuerySet` 객체를 통해 데이터베이스에서 데이터를 조회, 생성, 수정, 삭제하는 강력한 API를 제공합니다. 모델 매니저(`objects`)를 통해 쿼리셋을 얻을 수 있습니다.

---
**실습을 위한 모델 정의**

이하 모든 예제는 아래와 같이 정의된 `blog` 앱의 모델들을 기준으로 작성되었습니다. 타임스탬프, 소프트 삭제, 다대다 관계 등 실무적인 요소들을 포함하고 있습니다.

```python
# blog/models.py

from django.db import models
from django.conf import settings
from django.utils import timezone

# --- 공용 추상 모델 ---

class TimestampedModel(models.Model):
    """ 생성 및 수정일시를 자동으로 기록하는 추상 기본 클래스 """
    created_at = models.DateTimeField("생성일시", auto_now_add=True)
    updated_at = models.DateTimeField("수정일시", auto_now=True)

    class Meta:
        abstract = True

class SoftDeletionManager(models.Manager):
    """ 소프트 삭제되지 않은 객체만 조회하는 커스텀 매니저 """
    def get_queryset(self):
        return super().get_queryset().filter(is_deleted=False)

class SoftDeletionModel(models.Model):
    """ 
    소프트 삭제 기능을 위한 추상 기본 클래스.
    delete() 메서드를 오버라이드하여 실제 삭제 대신 is_deleted 플래그를 활성화합니다.
    """
    is_deleted = models.BooleanField("삭제 여부", default=False)
    deleted_at = models.DateTimeField("삭제일시", null=True, blank=True, default=None)

    objects = SoftDeletionManager()  # 기본 매니저를 교체
    all_objects = models.Manager()   # 삭제된 객체 포함, 모든 객체에 접근하기 위한 매니저

    def delete(self, using=None, keep_parents=False):
        """ 소프트 삭제를 수행 """
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save(update_fields=['is_deleted', 'deleted_at'])

    def restore(self):
        """ 소프트 삭제된 객체를 복구 """
        self.is_deleted = False
        self.deleted_at = None
        self.save(update_fields=['is_deleted', 'deleted_at'])

    def hard_delete(self, using=None, keep_parents=False):
        """ 데이터베이스에서 영구적으로 삭제 """
        return super().delete(using, keep_parents)

    class Meta:
        abstract = True

# --- 블로그 앱 모델 ---

class Post(TimestampedModel, SoftDeletionModel):
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        verbose_name="작성자",
        on_delete=models.CASCADE, 
        related_name="posts"
    )
    title = models.CharField("제목", max_length=255)
    content = models.TextField("내용")
    tags = models.ManyToManyField('Tag', verbose_name="태그", related_name='posts', blank=True)
    is_published = models.BooleanField("공개 여부", default=False, db_index=True)
    hit = models.PositiveIntegerField("조회수", default=0)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "게시글"
        verbose_name_plural = "게시글 목록"

    def __str__(self):
        return self.title

class Comment(TimestampedModel, SoftDeletionModel):
    post = models.ForeignKey(Post, verbose_name="게시글", on_delete=models.CASCADE, related_name="comments")
    author = models.ForeignKey(settings.AUTH_USER_MODEL, verbose_name="작성자", on_delete=models.CASCADE, related_name="comments")
    content = models.TextField("댓글 내용")

    class Meta:
        ordering = ["created_at"]
        verbose_name = "댓글"
        verbose_name_plural = "댓글 목록"

    def __str__(self):
        return f"{self.author.username}의 댓글: {self.content[:20]}..."

class Tag(models.Model):
    name = models.CharField("태그명", max_length=50, unique=True)

    def __str__(self):
        return self.name
```
---

### 4.1. 데이터 조회: 쿼리셋(QuerySet)의 이해와 활용

데이터 조회는 ORM의 가장 기본적이면서도 중요한 기능입니다. Django에서는 **쿼리셋(QuerySet)** 이라는 특별한 객체를 통해 데이터베이스와 소통합니다.

#### **쿼리셋의 핵심: 지연 평가 (Lazy Evaluation)**

가장 먼저 이해해야 할 개념은 쿼리셋이 **게으르다(Lazy)**는 것입니다. `Post.objects.filter(is_published=True)` 와 같은 코드를 작성했을 때, Django는 **즉시 데이터베이스에 쿼리를 보내지 않습니다.** 대신, "앞으로 이런 조건으로 데이터를 가져와야지" 라고 계획만 세워둡니다.

- **언제 쿼리가 실행되는가? (Evaluation)**: 실제 데이터베이스 쿼리는 쿼리셋 객체가 "평가"될 때 실행됩니다.
    - `for post in my_queryset:` 처럼 반복문에서 사용될 때
    - `list(my_queryset)` 처럼 명시적으로 리스트로 변환할 때
    - `print(my_queryset)` 또는 템플릿에서 `{{ my_queryset }}` 처럼 결과를 출력할 때
    - `len(my_queryset)` 으로 개수를 셀 때 (하지만 `.count()`가 더 효율적입니다.)

- **왜 중요한가?**: 이 "지연 평가" 특성 덕분에, 여러 `filter`, `exclude`, `order_by` 등을 마치 레고 블록처럼 여러 줄에 걸쳐 조합하더라도, 최종적으로 단 한 번의 효율적인 데이터베이스 쿼리만 실행하게 됩니다.

#### **1. 모든 객체 조회: `all()`**
테이블의 모든 레코드를 포함하는 쿼리셋을 반환합니다. 모델에 `class Meta`의 `ordering` 옵션이 정의되어 있다면 해당 순서로, 없다면 보장되지 않은 순서로 반환됩니다. `.order_by()`를 함께 사용하여 명시적으로 정렬하는 것이 일반적입니다.
- **Django Shell에서 확인하기**
  ```bash
  python manage.py shell
  ```
  ```python
  >>> from blog.models import Post
  # Post.objects는 소프트 삭제되지 않은 모든 Post 객체를 가리킴
  >>> all_posts = Post.objects.all() 
  >>> print(all_posts)
  <QuerySet [<Post: ...>, <Post: ...>]>
  ```

- **View 함수에서 활용하기**
  ```python
  # blog/views.py
  from django.shortcuts import render
  from .models import Post

  def post_list(request):
      # 공개된 게시글만 최신순으로 가져오기
      posts = Post.objects.filter(is_published=True).order_by('-created_at')
      context = {'posts': posts}
      return render(request, 'blog/post_list.html', context)
  ```

#### **2. 단 하나의 객체 조회: `get()`**
주어진 조건과 일치하는 **단 하나의 객체**를 반환합니다. 기본 키(pk)로 조회할 때 가장 많이 사용됩니다.

- **주의할 점**: `get()`은 조건에 맞는 객체가 없으면 `Model.DoesNotExist` 예외를, 두 개 이상이면 `Model.MultipleObjectsReturned` 예외를 발생시킵니다. 따라서 항상 예외 처리를 염두에 두어야 합니다.

- **실무 Best Practice**: 뷰에서는 `try...except` 블록을 직접 사용하는 것보다, `get_object_or_404()` 단축 함수를 사용하는 것이 훨씬 깔끔하고 일반적입니다. 객체가 없으면 자동으로 HTTP 404 (Not Found) 응답을 반환해줍니다.
- **Django Shell에서 확인하기**
  ```python
  >>> from blog.models import Post
  # pk(Primary Key)가 1인 게시글 조회
  >>> post = Post.objects.get(pk=1)
  >>> print(post.title)
  '첫 번째 게시글'
  
  # 없는 pk로 조회 시 예외 발생
  >>> Post.objects.get(pk=999)
  blog.models.Post.DoesNotExist: Post matching query does not exist.
  ```

- **View 함수에서 활용하기 (Best Practice)**
  뷰에서는 `get()`의 예외를 직접 처리하기보다, `get_object_or_404()`를 사용하는 것이 훨씬 간결하고 실용적입니다.
  ```python
  # blog/views.py
  from django.shortcuts import render, get_object_or_404
  from .models import Post

  def post_detail(request, post_id):
      # pk=post_id 조건으로 Post를 찾고, 없으면 404 에러 페이지를 보여줌
      post = get_object_or_404(Post, pk=post_id, is_published=True)
      context = {'post': post}
      return render(request, 'blog/post_detail.html', context)
  ```

#### **3. 조건에 맞는 객체 필터링: `filter()` 와 `exclude()`**

- **`filter(**kwargs)`**: 주어진 조건에 **일치하는** 객체들을 포함하는 새로운 쿼리셋을 반환합니다.
- **`exclude(**kwargs)`**: 주어진 조건에 **일치하지 않는** 객체들을 포함하는 새로운 쿼리셋을 반환합니다.

이 메서드들의 진정한 힘은 **필드 조회(Field Lookups)**, 즉 더블 언더스코어(`__`)를 통해 발휘됩니다.

- **자주 사용되는 필드 조회:**
    - `field__exact`: 정확히 일치 (기본값이므로 보통 생략. 예: `filter(title="My Title")`)
    - `field__iexact`: 대소문자 구분 없이 정확히 일치
    - `field__contains`: 문자열 포함
    - `field__icontains`: 대소문자 구분 없이 문자열 포함
    - `field__startswith` / `field__endswith`: 특정 문자열로 시작/끝
    - `field__gt` / `field__gte`: 보다 큼 / 크거나 같음 (Greater Than)
    - `field__lt` / `field__lte`: 보다 작음 / 작거나 같음 (Less Than)
    - `field__in`: 주어진 리스트 안에 포함된 값
    - `field__range`: 주어진 범위 사이의 값 (예: `(start_date, end_date)`)
    - `field__isnull`: `True` 또는 `False`로 `NULL` 값 여부 확인

- **관계를 넘나드는 조회**: `__`는 `ForeignKey` 관계를 따라 다른 테이블의 필드를 조회하는 데에도 사용됩니다. 이것이 ORM의 가장 강력한 기능 중 하나입니다.

- **Django Shell에서 확인하기**
  ```python
  >>> from blog.models import Post
  >>> from django.contrib.auth.models import User
  >>> user_alice = User.objects.get(username='alice')

  # alice가 작성한 글 중, 제목에 'Django'가 포함된 글만 필터링
  >>> posts = Post.objects.filter(author=user_alice, title__icontains='Django')
  
  # 공개되지 않은 글 제외
  >>> published_posts = Post.objects.exclude(is_published=False)

  # 'python' 태그가 달린 모든 게시글 조회 (관계 필드 필터링)
  >>> Post.objects.filter(tags__name='python')
  ```

- **View 함수에서 활용하기 (검색 기능)**
  ```python
  # blog/views.py
  def search_post(request):
      # URL 쿼리 파라미터에서 검색어 가져오기 (예: /search?q=django)
      query = request.GET.get('q', '')
      if query:
          # 제목 또는 내용에 검색어가 포함된 게시글 검색
          posts = Post.objects.filter(title__icontains=query) | Post.objects.filter(content__icontains=query)
          posts = posts.filter(is_published=True).distinct() # 중복 제거
      else:
          posts = Post.objects.none() # 검색어가 없으면 빈 쿼리셋 반환
      
      context = {'posts': posts, 'query': query}
      return render(request, 'blog/search_results.html', context)
  ```

#### **4. 기타 유용한 조회 메서드**

- **`count()`**: 쿼리셋의 객체 수를 반환합니다. `len()`보다 효율적입니다.
- **`exists()`**: 쿼리셋에 결과가 하나라도 존재하는지 확인합니다. `count() > 0` 보다 효율적입니다.
- **`first()`, `last()`**: 쿼리셋의 첫 번째, 마지막 객체를 반환합니다. 결과가 없으면 `None`을 반환하여 안전합니다.

```python
# 공개된 게시글의 총 개수
num_posts = Post.objects.filter(is_published=True).count()

# 'alice'가 작성한 글이 하나라도 있는지 확인
has_posts = Post.objects.filter(author__username='alice').exists()

# 가장 최근에 작성된 댓글
latest_comment = Comment.objects.latest('created_at')
```

### 4.2. 데이터 생성: `create()`, `save()`, `get_or_create()`, `bulk_create()`

데이터베이스에 새로운 레코드를 추가하는 방법은 여러 가지가 있으며, 각기 다른 상황에 적합합니다.

#### **방법 1: `create()` - 가장 간단한 한 줄 생성**

모델의 매니저(`objects`)가 제공하는 `create()` 메서드는 객체 생성과 데이터베이스 저장을 **한 번에** 처리합니다. 코드가 간결해져 단순한 객체 생성에 가장 많이 사용됩니다.

- **동작**: `Post.objects.create(author=user, title="New Post")`는 `Post` 객체를 만들고, 즉시 `INSERT` SQL 쿼리를 실행하여 데이터베이스에 저장한 뒤, 생성된 객체 인스턴스를 반환합니다.
- **장점**: 코드가 간결하고 직관적입니다.
- **단점**: 객체가 데이터베이스에 저장되기 전에 추가적인 로직을 수행할 수 없습니다.


- **Django Shell에서 확인하기**
  ```python
  >>> from blog.models import Post
  >>> from django.contrib.auth.models import User
  >>> author = User.objects.get(pk=1)
  >>> new_post = Post.objects.create(
  ...     author=author,
  ...     title='새로운 포스트',
  ...     content='create 메서드로 생성되었습니다.',
  ...     is_published=True
  ... )
  >>> print(new_post.id)
  5  # 예시 ID
  ```

#### **방법 2: `save()` - 유연한 2단계 생성 (실무 핵심)**

`ModelForm`과 함께 사용하여 사용자의 입력을 처리하고 저장 전 추가 로직을 적용할 때 가장 많이 사용됩니다.

이 방식은 객체를 메모리에 먼저 생성하고, 원하는 시점에 `save()` 메서드를 호출하여 데이터베이스에 저장합니다.

1.  **객체 인스턴스화**: `my_obj = MyModel(...)` 코드는 아직 데이터베이스에 아무런 영향을 주지 않는, 순수한 파이썬 객체를 메모리에 만듭니다.
2.  **`save()` 호출**: `my_obj.save()`가 호출되는 시점에 비로소 `INSERT` (또는 `UPDATE`) 쿼리가 실행됩니다.

- **장점**: 데이터베이스에 저장하기 전에 모델 인스턴스의 속성을 변경하거나, 모델에 정의된 다른 메서드를 호출하는 등 복잡한 로직을 수행할 수 있는 유연성을 제공합니다. `request` 객체처럼 폼 데이터에 포함되지 않은 추가적인 데이터를 모델에 채워 넣어야 할 때 필수적입니다.
- **단점**: `create()`에 비해 코드가 몇 줄 더 길어집니다.


- **View 함수에서 활용하기 (게시글 작성 뷰)**
  ```python
  # blog/forms.py
  from django import forms
  from .models import Post

  class PostForm(forms.ModelForm):
      class Meta:
          model = Post
          fields = ['title', 'content', 'tags', 'is_published']

  # blog/views.py
  from django.shortcuts import render, redirect
  from .forms import PostForm

  def post_create(request):
      if request.method == 'POST':
          form = PostForm(request.POST)
          if form.is_valid():
              # 1. commit=False: DB 저장을 지연하고, 메모리에 객체만 생성
              post = form.save(commit=False)
              # 2. 저장 전 추가 로직 수행 (작성자 정보 추가)
              post.author = request.user
              # 3. 실제 DB에 저장
              post.save()
              # 4. ManyToMany 필드는 save() 이후에 저장해야 함
              form.save_m2m() 
              return redirect('post_detail', post_id=post.id)
      else:
          form = PostForm()
      
      context = {'form': form}
      return render(request, 'blog/post_form.html', context)
  ```

#### **방법 3: `get_or_create()` - 중복 방지 생성**

"객체가 있으면 가져오고, 없으면 생성하라"는 로직을 한 번에 처리하는 매우 실용적인 메서드입니다.

- **동작**: 주어진 조건으로 `get()`을 시도하고, `DoesNotExist` 예외가 발생하면 `create()`를 실행합니다.
- **반환 값**: `(object, created)` 형태의 튜플을 반환합니다.
    - `object`: 조회되거나 새로 생성된 객체입니다.
    - `created`: 객체가 새로 생성되었으면 `True`, 기존 객체를 가져왔으면 `False`인 불리언 값입니다. 이 값을 통해 후속 처리를 분기할 수 있습니다.
- **사용 사례**: 태그(Tag) 생성, 사용자 프로필 생성 등 중복되면 안 되는 데이터를 처리할 때 매우 유용합니다.


- **View 함수에서 활용하기 (태그 처리)**
  사용자가 입력한 태그 문자열을 분리하여, 기존에 없던 태그만 새로 생성하고 게시글에 연결합니다.
  ```python
  # ... post_create 뷰의 form.is_valid() 블록 내부 ...
  if form.is_valid():
      post = form.save(commit=False)
      post.author = request.user
      post.save()

      # 사용자가 폼에서 입력한 태그 이름들 (예: "django, python")
      tag_names_str = request.POST.get('tags_string', '')
      tag_names = [name.strip() for name in tag_names_str.split(',') if name.strip()]

      for tag_name in tag_names:
          # 태그가 있으면 가져오고(get), 없으면 생성(create)
          tag, created = Tag.objects.get_or_create(name=tag_name)
          post.tags.add(tag) # 게시글과 태그 연결

      return redirect('post_detail', post_id=post.id)
  ```

#### **방법 4: `bulk_create()` - 대량 데이터 고속 생성**

여러 개의 객체를 생성할 때, `for` 루프 안에서 `create()`나 `save()`를 반복 호출하는 것은 매우 비효율적입니다. 매번 DB 쿼리가 발생하기 때문입니다. `bulk_create()`는 객체 리스트를 받아 단 **한 번의 `INSERT` 쿼리**로 모든 객체를 데이터베이스에 삽입하여 성능을 극적으로 향상시킵니다.

- **주의사항**: 성능을 위해 몇 가지를 희생합니다.
    - 각 모델 인스턴스의 `.save()` 메서드가 호출되지 않습니다. 따라서 `save()` 메서드를 오버라이드하여 구현한 커스텀 로직은 실행되지 않습니다.
    - `auto_now_add`나 `auto_now` 같은 필드는 자동으로 채워지지 않을 수 있습니다. (Django 버전에 따라 동작이 다를 수 있음)
    - 일부 데이터베이스(예: 구버전 MySQL)에서는 생성된 객체의 기본 키(`id`)가 반환되지 않을 수 있습니다.


- **Management Command에서 활용하기 (초기 데이터 생성)**
  `python manage.py seed_posts` 와 같이 실행하여 테스트용 데이터를 대량으로 생성할 때 유용합니다.
  ```python
  # blog/management/commands/seed_posts.py
  from django.core.management.base import BaseCommand
  from blog.models import Post
  from django.contrib.auth.models import User

  class Command(BaseCommand):
      help = 'Creates 100 dummy posts for testing.'

      def handle(self, *args, **options):
          user = User.objects.first()
          if not user:
              self.stdout.write(self.style.ERROR('Create a user first.'))
              return

          posts_to_create = [
              Post(author=user, title=f'Test Post {i}', content='...')
              for i in range(100)
          ]
          
          # 단 한 번의 쿼리로 모든 객체 생성
          Post.objects.bulk_create(posts_to_create)
          
          self.stdout.write(self.style.SUCCESS('Successfully created 100 posts.'))
  ```

### 4.3. 데이터 수정: `save()`, `update()`, 그리고 `F()` 표현식

데이터 수정은 단순히 필드 값을 바꾸는 것 이상의 의미를 가집니다. 실무에서는 **데이터 무결성**과 **성능**을 함께 고려해야 합니다.

#### **방법 1: `save()` - 객체 단위 수정 (실무 핵심)**

가장 기본적인 수정 방법은 객체를 데이터베이스에서 불러와, 파이썬 객체의 속성을 변경한 뒤, 다시 `save()`를 호출하는 것입니다.

- **동작 원리**: Django는 객체가 데이터베이스에서 읽혔다는 것(즉, `pk` 값이 존재한다는 것)을 인지하므로, `save()` 호출 시 `INSERT`가 아닌 `UPDATE` SQL 쿼리를 실행합니다.
- **장점**: 모델의 `save()` 메서드를 오버라이드하여 구현한 커스텀 로직이나, `pre_save`, `post_save` 같은 시그널이 정상적으로 호출됩니다. 객체 단위의 비즈니스 로직을 수행하기에 적합합니다.
- **단점**: 데이터를 읽고(SELECT), 수정하고, 저장하는(UPDATE) 과정에서 여러 번의 DB 통신이 필요하며, 동시성 문제(Race Condition)에 취약할 수 있습니다.


- **View 함수에서 활용하기 (게시글 수정 뷰)**
  ```python
  # blog/views.py
  def post_update(request, post_id):
      post = get_object_or_404(Post, pk=post_id)

      # 본인만 수정 가능하도록 권한 확인
      if request.user != post.author:
          return HttpResponseForbidden("수정 권한이 없습니다.")

      if request.method == 'POST':
          # instance=post: 기존 객체 위에 폼 데이터를 덮어씌움
          form = PostForm(request.POST, instance=post)
          if form.is_valid():
              form.save() # pk가 있으므로 UPDATE 쿼리가 실행됨
              return redirect('post_detail', post_id=post.id)
      else:
          # GET 요청 시, 기존 데이터를 채운 폼을 보여줌
          form = PostForm(instance=post)
      
      context = {'form': form, 'post': post}
      return render(request, 'blog/post_form.html', context)
  ```

#### **`save()` 사용 시 주의점: Race Condition**

`save()`를 사용한 "읽고-수정하고-저장하기" 패턴은 여러 요청이 동시에 들어올 때 문제를 일으킬 수 있습니다. 예를 들어, 조회수를 1 증가시키는 로직을 다음과 같이 작성했다고 가정해 봅시다.

```python
# Race Condition에 취약한 코드
def increase_hit_count(post_id):
    post = Post.objects.get(pk=post_id) # (1) 현재 조회수(예: 10)를 읽음
    post.hit = post.hit + 1             # (2) 파이썬에서 1을 더함 (11)
    post.save()                         # (3) 조회수 11을 DB에 저장
```
만약 두 명의 사용자가 거의 동시에 이 로직을 실행하면, 두 요청 모두 (1)번 시점에서 조회수 10을 읽고, 각자 1을 더한 11을 (3)번 시점에서 저장하게 됩니다. 결과적으로 조회수는 2가 증가해야 하지만 1만 증가하는 **데이터 유실**이 발생합니다. 이를 **Race Condition(경쟁 상태)**이라고 합니다.

#### **방법 2: `F()` 표현식 - 원자적 연산으로 Race Condition 해결**

이러한 동시성 문제를 해결하기 위해 `F()` 표현식을 사용합니다. `F()`는 파이썬 메모리가 아닌 **데이터베이스 레벨에서 필드 값을 직접 참조**하여 연산을 수행하도록 합니다.

- **동작 원리**: `post.hit = F('hit') + 1` 코드는 "`posts` 테이블에서 이 레코드의 `hit` 컬럼 값을 가져와서 1을 더한 값으로 업데이트하라"는 단일 SQL `UPDATE` 문으로 변환됩니다. 읽고 쓰는 과정이 데이터베이스 내에서 원자적(atomic)으로 일어나므로 Race Condition이 발생하지 않습니다.

- **View 함수에서 활용하기 (조회수 증가)**
  게시글 상세 페이지에 접근할 때마다 `F()` 표현식을 사용하여 안전하게 조회수를 1 증가시킵니다.
  ```python
  # blog/views.py
  from django.db.models import F

  def post_detail(request, post_id):
      post = get_object_or_404(Post, pk=post_id)
      
      # F() 표현식을 사용하여 DB 레벨에서 원자적으로 조회수 증가
      post.hit = F('hit') + 1
      post.save(update_fields=['hit']) # hit 필드만 업데이트하도록 최적화
      
      # 변경된 값을 즉시 보려면 DB에서 다시 로드
      post.refresh_from_db()

      context = {'post': post}
      return render(request, 'blog/post_detail.html', context)
  ```

#### **방법 3: `update()` - 여러 객체를 한 번에 효율적으로 수정**

쿼리셋에 대해 `update()`를 호출하면, `for` 루프 없이 단 한 번의 `UPDATE` 쿼리로 여러 레코드를 수정할 수 있어 매우 효율적입니다.

- **장점**: 대량의 데이터를 수정할 때 성능상 압도적으로 유리합니다.
- **주의사항**: 이 메서드는 모델의 `save()` 메서드나 `pre_save`/`post_save` 시그널을 발생시키지 않으며, `auto_now` 필드를 자동으로 갱신하지도 않습니다. 순수하게 데이터베이스 레벨에서 `UPDATE` 쿼리만 실행합니다.


- **Management Command에서 활용하기 (일괄 작업)**
  ```python
  # blog/management/commands/publish_all.py
  from django.core.management.base import BaseCommand
  from blog.models import Post
  import datetime

  class Command(BaseCommand):
      help = 'Publishes all posts created before today.'

      def handle(self, *args, **options):
          today = datetime.date.today()
          # 어제까지 작성된 모든 비공개 게시글을 한 번에 공개 처리
          updated_count = Post.objects.filter(
              is_published=False,
              created_at__lt=today
          ).update(is_published=True)
          
          self.stdout.write(self.style.SUCCESS(f'{updated_count} posts published.'))
  ```

#### **성능 최적화: `save(update_fields=[...])`**

`save()` 메서드는 기본적으로 모델의 모든 필드를 `UPDATE` 쿼리에 포함시킵니다. 만약 특정 필드 몇 개만 수정했다는 것을 명확히 안다면, `update_fields` 인자를 사용하여 변경이 필요한 필드만 지정할 수 있습니다. 이는 불필요한 DB 쓰기를 줄여 성능을 개선하는 데 도움이 됩니다.

### 4.4. 데이터 삭제: 하드 삭제(Hard Delete) vs 소프트 삭제(Soft Delete)

실무에서는 복구 및 데이터 분석을 위해 **소프트 삭제**가 강력히 권장됩니다.

#### **방법 1: 하드 삭제 (Hard Delete) - 기본 `delete()`**

모델 인스턴스나 쿼리셋의 `.delete()` 메서드를 호출하면, 해당 레코드는 데이터베이스에서 **영구적으로, 복구할 수 없게 삭제**됩니다. `DELETE FROM ...` SQL 문이 직접 실행됩니다.


- **동작 방식**:
    - `post = Post.objects.get(pk=1); post.delete()`: 단일 객체를 삭제합니다.
    - `Post.objects.filter(author__is_active=False).delete()`: 여러 객체를 한 번의 쿼리로 효율적으로 삭제합니다.
- **연쇄 삭제 (Cascading Delete)**: `delete()`가 호출되면, 해당 객체를 `ForeignKey`로 참조하고 `on_delete=models.CASCADE`로 설정된 다른 모든 객체들도 함께 연쇄적으로 삭제됩니다. 매우 편리하지만, 의도치 않은 대량의 데이터 손실을 유발할 수 있어 항상 주의해야 합니다.
- **문제점**: 
    - **복구 불가**: 실수로 삭제하면 데이터를 되살릴 방법이 없습니다.
    - **데이터 분석의 어려움**: 사용자의 탈퇴나 콘텐츠 삭제 이력을 추적할 수 없어, 서비스의 중요한 통계 및 분석 데이터를 잃게 됩니다.
    - **무결성 문제**: `on_delete` 옵션이 잘못 설정된 경우, 관계가 꼬이거나 오류가 발생할 수 있습니다.

- **View 함수에서 활용하기 (신중하게 사용!)**
  ```python
  # blog/views.py
  from django.views.decorators.http import require_POST

  @require_POST # POST 요청만 허용
  def post_hard_delete(request, post_id):
      post = get_object_or_404(Post, pk=post_id)
      if request.user != post.author:
          return HttpResponseForbidden("삭제 권한이 없습니다.")
      
      # 모델에 정의된 hard_delete() 메서드 호출
      post.hard_delete()
      
      return redirect('post_list')
  ```

#### **방법 2: 소프트 삭제 (Soft Deletion) - 실무 권장 패턴**


소프트 삭제는 레코드를 실제로 지우는 대신, `is_deleted=True` 와 같은 플래그(flag)를 두어 삭제된 것처럼 취급하는 방식입니다. 데이터는 DB에 그대로 남아있지만, 일반적인 조회에서는 나타나지 않습니다.

- **장점**: 데이터 복구가 가능하고, 모든 기록이 보존되어 데이터 분석 및 감사 추적에 유리합니다.

- **구현 방법**: 커스텀 매니저와 `delete()` 메서드 오버라이딩을 조합하여 구현합니다.


- **View 함수에서 활용하기 (안전한 삭제)**
  모델의 `delete()` 메서드를 오버라이드했으므로, 일반적인 `.delete()` 호출이 소프트 삭제를 수행합니다.
  ```python
  # blog/views.py
  @require_POST
  def post_soft_delete(request, post_id):
      post = get_object_or_404(Post, pk=post_id)
      if request.user != post.author:
          return HttpResponseForbidden("삭제 권한이 없습니다.")
      
      # 오버라이드된 delete()가 호출되어 소프트 삭제 실행
      post.delete()
      
      return redirect('post_list')
  ```

#### **소프트 삭제된 데이터 관리 (휴지통 기능)**

- **View 함수에서 활용하기 (휴지통 및 복원)**
  `all_objects` 매니저를 사용하여 삭제된 항목을 보고, `restore()` 메서드로 복원합니다.
  ```python
  # blog/views.py
  def trash_bin(request):
      # all_objects 매니저로 삭제된 게시글만 조회
      deleted_posts = Post.all_objects.filter(is_deleted=True, author=request.user)
      context = {'deleted_posts': deleted_posts}
      return render(request, 'blog/trash_bin.html', context)

  @require_POST
  def post_restore(request, post_id):
      # all_objects로 삭제된 게시글 중에서 찾아야 함
      post = get_object_or_404(Post.all_objects, pk=post_id, is_deleted=True)
      if request.user != post.author:
          return HttpResponseForbidden("복원 권한이 없습니다.")
      
      # 모델에 정의된 restore() 메서드 호출
      post.restore()
      
      return redirect('trash_bin')
  ```


- **사용법**:
    - `post.delete()`: 이제 `Post` 객체의 `delete()`를 호출하면 `is_deleted`가 `True`로 바뀌는 **소프트 삭제**가 일어납니다.
    - `Post.objects.all()`: `SoftDeletionManager` 덕분에, 삭제된 게시글은 이 쿼리 결과에 포함되지 않습니다.
    - `Post.all_objects.all()`: 삭제된 게시글을 포함한 **모든** 게시글을 조회하고 싶을 때 사용합니다. (예: 관리자 페이지의 휴지통 기능)
    - `post.hard_delete()`: 정말로 DB에서 레코드를 영구히 삭제하고 싶을 때 명시적으로 호출합니다.

### 4.5. 고급 쿼리셋 및 최적화: 실무 심화

단순한 CRUD를 넘어, Django ORM은 복잡한 요구사항을 해결하고 애플리케이션 성능을 극대화할 수 있는 강력하고 다채로운 기능들을 제공합니다. 이 섹션에서는 실무에서 반드시 알아야 할 고급 기법들을 다룹니다.

#### **4.5.1. 쿼리 성능 최적화**

**1. N+1 문제 해결: `select_related` 와 `prefetch_related`**

- **N+1 문제란?**: 쿼리셋을 순회하며 관련(related) 객체에 접근할 때, 각 객체마다 추가적인 DB 쿼리가 발생하는 문제입니다. 게시글 목록(쿼리 1번)을 가져와서, 루프 안에서 각 게시글의 작성자 정보(`post.author`)에 접근할 때마다 매번 쿼리(N번)가 발생하여 총 N+1개의 쿼리가 실행되는 것이 대표적인 예입니다. 이는 서비스 성능에 치명적입니다.

- **`select_related(*fields)`**: `ForeignKey`, `OneToOneField`와 같이 **단일 객체를 참조하는 관계**를 SQL의 `JOIN`을 사용해 미리 가져옵니다. 한 번의 쿼리로 관련 객체까지 모두 가져오므로, DB 접근 횟수를 획기적으로 줄입니다.
    ```python
    # N+1 문제 발생: Post 10개를 가져온 뒤, author에 접근할 때마다 쿼리 발생 (총 11번)
    posts = Post.objects.all()[:10]
    for post in posts: 
        print(post.author.username)

    # 해결: select_related로 author 정보를 함께 JOIN하여 가져옴 (총 1번의 쿼리)
    posts = Post.objects.select_related('author').all()[:10]
    for post in posts:
        print(post.author.username)
    ```

- **`prefetch_related(*lookups)`**: `ManyToManyField`, `역참조 ForeignKey` (예: `user.posts.all()`) 와 같이 **여러 객체를 참조하는 관계**를 최적화합니다. `JOIN`을 사용하면 결과 데이터가 중복되어 오히려 비효율적일 수 있는 경우에 사용됩니다. 별도의 쿼리로 관련 객체들을 미리 모두 가져온 뒤, 파이썬 레벨에서 두 쿼리의 결과를 합쳐줍니다.
    ```python
    # N+1 문제 발생: Post 10개를 가져온 뒤, 각 post의 태그에 접근할 때마다 쿼리 발생
    posts = Post.objects.all()[:10]
    for post in posts:
        print([tag.name for tag in post.tags.all()])

    # 해결: prefetch_related로 tags를 미리 가져옴 (총 2번의 쿼리: Post, Tag)
    posts = Post.objects.prefetch_related('tags').all()[:10]
    for post in posts:
        print([tag.name for tag in post.tags.all()])
    ```
- **`Prefetch` 객체**: `prefetch_related`를 더욱 세밀하게 제어하고 싶을 때 사용합니다. 예를 들어, 승인된 댓글만 미리 가져오고 싶을 때 유용합니다.
    ```python
    from django.db.models import Prefetch

    approved_comments = Comment.objects.filter(is_approved=True)
    posts = Post.objects.prefetch_related(
        Prefetch('comments', queryset=approved_comments)
    )
    ```

**2. 필요한 데이터만 가져오기: `values`, `values_list`, `only`, `defer`**

- **`values(*fields)`**: 모델 인스턴스가 아닌, **딕셔너리**들의 쿼리셋을 반환합니다. 특정 필드의 데이터만 필요할 때 사용하여 메모리 사용량을 줄일 수 있습니다.
- **`values_list(*fields, flat=False)`**: `values`와 유사하지만 **튜플**들의 쿼리셋을 반환합니다. `flat=True` 옵션을 주면 단일 필드에 대한 값들을 하나의 리스트로 받을 수 있습니다.
    ```python
    # 모든 게시글의 제목만 리스트로 가져오기
    titles = Post.objects.values_list('title', flat=True)
    ```
- **`only(*fields)`** 와 **`defer(*fields)`**: 전체 모델 인스턴스가 필요하지만, 특정 필드만 미리 로드하거나(`only`), 특정 필드를 나중에 로드하고 싶을 때(`defer`) 사용합니다. 특히 `TextField`처럼 용량이 큰 필드를 목록 페이지에서 제외하여 성능을 향상시킬 때 유용합니다.
    ```python
    # content 필드를 제외한 모든 필드를 가져옴
    posts = Post.objects.defer("content").all()
    ```

**3. 대용량 데이터 처리: `iterator()`**

일반적인 쿼리셋은 평가 시 모든 결과를 메모리에 로드합니다. 수백만 건의 데이터를 처리할 경우 메모리 부족으로 프로그램이 중단될 수 있습니다. `iterator()`는 결과를 한 번에 하나씩 가져와 메모리 사용량을 최소화합니다. 데이터 마이그레이션이나 CSV 파일 생성 등 대용량 데이터를 순회 처리할 때 필수적입니다.

```python
# 메모리 문제 없이 모든 사용자의 이메일을 처리
for user in User.objects.all().iterator():
    send_newsletter(user.email)
```

#### **4.5.2. 복잡한 쿼리 작성**

- **`Q()` 객체**: `filter()` 내에서 `OR`(`|`), `AND`(`&`), `NOT`(`~`) 과 같은 복잡한 논리 조건을 조합할 수 있게 해줍니다.
    ```python
    # 제목에 'Django'가 포함되거나, 작성자의 이름이 'Alice'인 게시글 조회
    from django.db.models import Q
    Post.objects.filter(Q(title__icontains='Django') | Q(author__username='Alice'))
    ```
- **`F()` 객체**: 데이터베이스 레벨에서 모델 필드 값을 참조하여 원자적 연산을 수행합니다. (4.3 데이터 수정 섹션 참고)

- **집계: `aggregate` 와 `annotate`**
    - **`aggregate()`**: 쿼리셋 전체에 대한 집계 값을 계산하여 **딕셔너리**로 반환합니다.
        ```python
        from django.db.models import Avg, Max
        # 전체 게시물의 평균 조회수와 최대 조회수 계산
        stats = Post.objects.aggregate(avg_hit=Avg('hit'), max_hit=Max('hit'))
        # 결과: {'avg_hit': 123.45, 'max_hit': 2000}
        ```
    - **`annotate()`**: 쿼리셋의 **각 객체별로** 집계 값을 계산하여 새로운 필드로 추가합니다. SQL의 `GROUP BY`와 유사하게 동작합니다.
        ```python
        from django.db.models import Count
        # 각 게시글별 댓글 수를 계산하여 `num_comments` 필드로 추가
        posts_with_comment_count = Post.objects.annotate(num_comments=Count('comments'))
        for post in posts_with_comment_count:
            print(f"{post.title}: {post.num_comments}개의 댓글")
        ```

- **조건부 표현식: `Case` 와 `When`**: SQL의 `CASE WHEN` 구문을 ORM으로 표현합니다. 복잡한 조건에 따라 동적인 값을 `annotate`하거나 `update`할 때 매우 강력합니다.
    ```python
    from django.db.models import Case, When, Value, CharField
    # 조회수에 따라 게시글 등급을 매김
    posts = Post.objects.annotate(
        grade=Case(
            When(hit__gte=1000, then=Value('인기글')),
            When(hit__gte=100, then=Value('추천글')),
            default=Value('일반글'),
            output_field=CharField(),
        )
    )
    ```

#### **4.5.3. 코드 추상화 및 재사용**

- **커스텀 매니저 (Custom Managers)**: 특정 모델에 자주 사용되는 쿼리를 메서드로 만들어 재사용성을 높입니다. 예를 들어, `Post.objects.published()` 와 같이 직관적인 코드를 작성할 수 있습니다.
    ```python
    # models.py
    class PublishedManager(models.Manager):
        def get_queryset(self):
            return super().get_queryset().filter(is_published=True)

    class Post(models.Model):
        # ...
        objects = models.Manager() # 기본 매니저
        published = PublishedManager() # 커스텀 매니저
    ```

- **커스텀 쿼리셋 (Custom QuerySets)**: 한 단계 더 나아가, 쿼리셋 자체에 커스텀 메서드를 추가하여 `Post.objects.published().by_author(user)` 와 같이 여러 커스텀 필터를 연쇄적으로(chainable) 호출할 수 있게 합니다.
    ```python
    # models.py
    class PostQuerySet(models.QuerySet):
        def published(self):
            return self.filter(is_published=True)

        def created_in_year(self, year):
            return self.filter(created_at__year=year)

    class Post(models.Model):
        # ...
        objects = PostQuerySet.as_manager()

    # 사용 예: Post.objects.published().created_in_year(2024)
    ```

- **트랜잭션 (Transactions)**: 여러 데이터베이스 작업을 하나의 논리적 단위로 묶어 데이터 무결성을 보장합니다. (자세한 내용은 4.5.4에서 별도 설명)

#### 4.5.4. 트랜잭션 (Transactions)

트랜잭션은 데이터베이스의 무결성(Integrity)을 보장하기 위한 핵심 개념입니다. 여러 데이터베이스 연산(읽기, 쓰기, 수정, 삭제)을 하나의 논리적인 작업 단위로 묶어, 모든 연산이 성공적으로 완료되거나(커밋, Commit) 하나라도 실패하면 모든 연산을 취소(롤백, Rollback)하여 데이터의 일관성을 유지합니다.

*   **`atomic()`:** Django는 `django.db.transaction.atomic()` 데코레이터나 컨텍스트 매니저를 통해 트랜잭션을 쉽게 사용할 수 있도록 합니다.
    *   **예시 (데코레이터):**
        ```python
        from django.db import transaction

        @transaction.atomic
        def transfer_money(sender, receiver, amount):
            sender.balance -= amount
            sender.save()
            receiver.balance += amount
            receiver.save()
            # 만약 여기서 오류 발생 시, 위 두 save() 작업은 모두 롤백됨
        ```
    *   **예시 (컨텍스트 매니저):**
        ```python
        from django.db import transaction

        def process_order(order):
            with transaction.atomic():
                # 주문 생성
                order.save()
                # 재고 감소
                product = order.product
                product.stock -= order.quantity
                product.save()
                # 결제 처리 (외부 API 호출 등)
                # 만약 결제 실패 시, 이 블록 내의 모든 DB 변경사항은 롤백됨
        ```

**실무적 관점:**
은행 거래, 주문 처리, 재고 관리 등 여러 데이터베이스 작업이 서로 의존적이며, 중간에 실패할 경우 데이터 불일치가 발생할 수 있는 중요한 비즈니스 로직에는 반드시 트랜잭션을 적용해야 합니다. 이는 데이터의 신뢰성과 시스템의 안정성을 보장하는 데 필수적입니다.

#### 4.5.5. 커스텀 매니저 (Custom Managers)

Django 모델의 `objects` 매니저는 `all()`, `filter()` 등 기본적인 쿼리셋 메서드를 제공합니다. 커스텀 매니저를 사용하면 특정 모델에 특화된 쿼리 메서드를 추가하거나, 기본 쿼리셋을 오버라이드하여 재사용 가능한 데이터 접근 로직을 정의할 수 있습니다.

*   **구현:** `django.db.models.Manager`를 상속받아 클래스를 정의하고, 이를 모델 클래스의 속성으로 할당합니다.
    *   **예시:**
        ```python
        from django.db import models

        class PublishedManager(models.Manager):
            def get_queryset(self):
                return super().get_queryset().filter(status='published')

        class Post(models.Model):
            title = models.CharField(max_length=100)
            content = models.TextField()
            status = models.CharField(max_length=10, default='draft')

            objects = models.Manager() # 기본 매니저
            published = PublishedManager() # 커스텀 매니저
        ```
        이제 `Post.published.all()`을 호출하면 `status='published'`인 Post 객체만 가져올 수 있습니다.

**실무적 관점:**
자주 사용되는 복잡한 쿼리나 특정 비즈니스 로직이 포함된 데이터 접근 패턴을 커스텀 매니저로 캡슐화하면 코드의 재사용성을 높이고, 뷰나 다른 로직에서 중복 코드를 줄일 수 있습니다. 이는 코드의 가독성과 유지보수성을 향상시킵니다.

### 4.6. 직접 데이터베이스 접근 및 Raw 쿼리: 최후의 수단

Django ORM은 매우 강력하지만, 모든 경우를 완벽하게 처리할 수는 없습니다. 때로는 ORM으로 표현하기 매우 복잡하거나, 특정 데이터베이스의 고유 기능을 사용해야 하거나, 극단적인 성능 최적화가 필요할 때 Raw SQL을 사용해야 합니다. 하지만 이는 ORM의 장점(DB 독립성, 가독성, 보안)을 포기하는 것이므로, **항상 최후의 수단으로 고려해야 합니다.**

#### **언제 Raw 쿼리를 사용해야 하는가?**

- **복잡한 조인 및 서브쿼리**: ORM으로 표현하기 너무 복잡하거나 비효율적인 경우.
- **데이터베이스 고유 기능 활용**: PostgreSQL의 `Window` 함수, `Common Table Expressions (CTE)` 등 특정 DB가 제공하는 고급 기능을 사용해야 할 때.
- **레거시 데이터베이스 연동**: Django 모델로 관리되지 않는 기존 테이블이나 뷰에 접근해야 할 때.
- **성능 최적화**: ORM이 생성하는 SQL보다 더 효율적인, 수작업으로 튜닝된 SQL을 실행해야 하는 매우 드문 경우. (반드시 성능 측정 후 결정해야 합니다.)

#### **방법 1: `Model.objects.raw()` - 가장 안전하고 권장되는 방법**

Raw SQL을 실행하여 그 결과를 **모델 인스턴스**로 받고 싶을 때 사용하는 가장 좋은 방법입니다. Django는 쿼리 결과의 필드 이름을 모델의 필드에 매핑하여 완전한 모델 객체를 돌려줍니다.

- **핵심 조건**: `SELECT` 문에 반드시 모델의 **기본 키(primary key)**가 포함되어야 합니다.
- **장점**: SQL Injection 공격으로부터 안전하게 파라미터를 처리하며, 결과로 모델 인스턴스를 받기 때문에 후속 작업이 편리합니다.

```python
# 특정 사용자가 작성한 게시글을 Raw SQL로 조회
def find_posts_by_author_raw(request, author_id):
    # 쿼리 내에 직접 변수를 넣지 않고, params 인자로 전달하여 SQL Injection을 방지
    query = "SELECT * FROM blog_post WHERE author_id = %s"
    posts = Post.objects.raw(query, [author_id])

    for post in posts:
        # post는 완전한 Post 모델 인스턴스입니다.
        print(post.title, post.author.username)
```

#### **방법 2: `connection.cursor()` - 가장 낮은 수준의 직접 접근**

모델 인스턴스가 필요 없거나, `SELECT`가 아닌 쿼리(예: `UPDATE`, `INSERT`, Stored Procedure 호출)를 실행할 때 사용합니다. Django의 모델 레이어를 완전히 우회하여 데이터베이스 연결에 직접 접근합니다.

- **가장 중요한 보안 규칙: SQL Injection 방지**
    - **절대로** f-string이나 `%` 포매팅을 사용하여 쿼리 문자열에 변수를 직접 삽입하면 안 됩니다. 이는 심각한 보안 취약점을 만듭니다.
    - **반드시** `cursor.execute()`의 두 번째 인자로 파라미터를 리스트나 튜플 형태로 전달해야 합니다. 데이터베이스 드라이버가 안전하게 값을 이스케이프(escape) 처리해줍니다.

```python
# 나쁜 예: SQL Injection에 매우 취약!
# cursor.execute(f"SELECT * FROM users WHERE username = '{username}'")

# 좋은 예: 파라미터를 안전하게 전달
# cursor.execute("SELECT * FROM users WHERE username = %s", [username])
```

- **사용 예시: `dictfetchall` 헬퍼와 함께 사용하기**

`cursor.fetchall()`은 결과를 튜플의 리스트 `[(값1, 값2), ...]` 형태로 반환합니다. 이를 다루기 쉬운 딕셔너리의 리스트 `[{'컬럼1': 값1, '컬럼2': 값2}, ...]` 형태로 변환해주는 헬퍼 함수를 만들어 사용하면 편리합니다.

```python
from django.db import connection

def dictfetchall(cursor):
    """커서의 모든 결과를 딕셔너리의 리스트로 반환합니다."""
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

def get_author_stats(request):
    with connection.cursor() as cursor:
        # 저자별 게시물 수와 평균 조회수를 계산하는 복잡한 쿼리
        cursor.execute("""
            SELECT 
                u.username, 
                COUNT(p.id) as post_count,
                AVG(p.hit) as average_hits
            FROM auth_user u
            JOIN blog_post p ON u.id = p.author_id
            GROUP BY u.username
            ORDER BY post_count DESC
        """)
        author_stats = dictfetchall(cursor)

    return render(request, 'stats.html', {'author_stats': author_stats})
```

#### **상황별 Raw 쿼리 사용법 요약**

| 목표                                       | 권장 방법                 | 이유                                                 |
| ------------------------------------------ | ------------------------- | ---------------------------------------------------- |
| **모델 객체**를 결과로 받고 싶을 때        | `Model.objects.raw()`     | 가장 안전하고, 결과를 모델 인스턴스로 바로 활용 가능 |
| 모델과 무관한 **데이터**를 받고 싶을 때    | `connection.cursor()`     | 집계, 통계 등 자유로운 `SELECT` 쿼리 실행 가능       |
| `SELECT`가 아닌 명령(`UPDATE` 등) 실행 시 | `connection.cursor()`     | 데이터베이스에 직접 명령을 내려야 할 때            |
| ORM으로 충분히 표현 가능할 때              | **Django ORM 사용**       | 안전성, 가독성, 유지보수성, DB 독립성 확보         |