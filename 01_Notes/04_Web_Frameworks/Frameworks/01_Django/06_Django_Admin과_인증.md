<h2>Django Backend: Admin과 인증</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django의 강력한 관리자 페이지(Admin)를 커스터마이징하는 방법과, 내장된 사용자 인증 및 권한 부여 시스템을 활용하여 안전한 웹 애플리케이션을 구축하는 방법을 학습하는 것을 목표로 합니다.</p>

<h2>목차</h2> 

- [1. Django Admin (관리자 페이지)](#1-django-admin-관리자-페이지)
  - [1.1. 모델 등록 (admin.site.register)](#11-모델-등록-adminsiteregister)
  - [1.2. 관리자 인터페이스 커스터마이징](#12-관리자-인터페이스-커스터마이징)
- [2. 사용자 인증 및 권한 부여](#2-사용자-인증-및-권한-부여)
  - [2.1. 기본 User 모델](#21-기본-user-모델)
  - [2.1.1. Custom User 모델](#211-custom-user-모델)
  - [2.2. 로그인/로그아웃 뷰](#22-로그인로그아웃-뷰)
  - [2.3. 권한 및 그룹 관리](#23-권한-및-그룹-관리)

---

## 1. Django Admin (관리자 페이지)

Django는 강력하고 자동화된 관리자 인터페이스를 기본으로 제공합니다. 이를 통해 개발자는 모델 데이터를 쉽게 관리하고 조작할 수 있습니다.

### 1.1. 모델 등록 (admin.site.register)
관리자 페이지에서 특정 모델의 데이터를 관리하려면 해당 모델을 `admin.py` 파일에 등록해야 합니다.

**예시 (`myhome1/blog/admin.py`):
```python
from django.contrib import admin
from .models import Blog # Blog 모델 임포트

# Blog 모델을 관리자 페이지에 등록
admin.site.register(Blog)
```

**예시 (`myhome1/score/admin.py`):
```python
from django.contrib import admin
from .models import Score # Score 모델 임포트

# Score 모델을 관리자 페이지에 등록
admin.site.register(Score)
```

모델을 등록한 후, `python manage.py createsuperuser` 명령어로 관리자 계정을 생성하고, `python manage.py runserver`로 서버를 실행한 뒤 `/admin` 경로로 접속하면 관리자 페이지를 확인할 수 있습니다.

**관리자 계정 생성 (`장고사이트구축방법.txt` 참고):
```bash
python manage.py createsuperuser
# 프롬프트에 따라 사용자 이름, 이메일, 비밀번호 입력
# 예: admin / admin@myhome2.com / qwer1234
```

### 1.2. 관리자 인터페이스 커스터마이징
`ModelAdmin` 클래스를 사용하여 관리자 페이지의 목록 보기, 검색 필드, 필터, 폼 레이아웃 등을 커스터마이징할 수 있습니다.

**예시 (가상):
```python
from django.contrib import admin
from .models import Blog

class BlogAdmin(admin.ModelAdmin):
    list_display = ('title', 'writer', 'wdate', 'hit') # 목록에 표시할 필드
    list_filter = ('writer', 'wdate') # 필터 옵션
    search_fields = ('title', 'contents') # 검색 가능한 필드
    ordering = ('-wdate',) # 기본 정렬 순서

admin.site.register(Blog, BlogAdmin)
```

**예시 (`장고사이트구축방법.txt` - `board/admin.py`):
```python
# board/admin.py (가상)
from django.contrib import admin
from .models import Board # Board 모델 임포트

class BoardAdmin(admin.ModelAdmin):
    search_fields =['title'] # title 필드로 검색 가능하도록 설정

admin.site.register(Board, BoardAdmin)
```

## 2. 사용자 인증 및 권한 부여

Django는 강력한 사용자 인증 및 권한 부여 시스템을 기본으로 제공합니다. 이를 통해 사용자 계정 관리, 로그인/로그아웃, 권한 설정 등을 쉽게 구현할 수 있습니다.

### 2.1. 기본 User 모델
Django는 `django.contrib.auth.models.User`라는 기본 사용자 모델을 제공합니다. 이 모델은 사용자 이름, 비밀번호, 이메일, 권한 등의 정보를 포함합니다.

### 2.1.1. Custom User 모델
대부분의 실제 웹 애플리케이션에서는 Django의 기본 `User` 모델만으로는 부족합니다. 사용자에게 추가적인 정보(예: 전화번호, 프로필 사진, 주소, 특정 비즈니스 로직에 필요한 필드)를 저장해야 할 필요가 있습니다. Django는 이러한 요구사항을 위해 `AbstractUser`와 `AbstractBaseUser`를 통한 커스텀 사용자 모델 정의를 지원합니다.

**실무적 관점:**
*   **`AbstractUser` 상속 (권장):** Django의 기본 `User` 모델이 제공하는 모든 필드와 기능을 유지하면서, 추가적인 필드를 손쉽게 확장할 수 있습니다. 대부분의 경우 이 방법을 사용하는 것이 가장 편리하고 안전합니다.
*   **`AbstractBaseUser` 상속:** 완전히 새로운 사용자 모델을 처음부터 정의할 때 사용합니다. 이 경우 인증 관련 필드(예: `username`, `password`, `last_login`, `is_active`)를 직접 구현해야 하므로 더 복잡하지만, 최대한의 유연성을 제공합니다.

**구현 방법 (AbstractUser 예시):**

1.  **`accounts` 앱 생성:**
    사용자 모델을 관리할 별도의 앱을 생성하는 것이 일반적입니다.
    ```bash
    python manage.py startapp accounts
    ```

2.  **`settings.py`에 앱 등록:**
    ```python
    # settings.py
    INSTALLED_APPS = [
        # ...
        'accounts',
        # ...
    ]
    ```

3.  **`accounts/models.py`에 Custom User 모델 정의:**
    ```python
    # accounts/models.py
    from django.contrib.auth.models import AbstractUser
    from django.db import models

    class CustomUser(AbstractUser):
        # 기본 User 모델의 필드(username, password, email 등)는 그대로 사용
        # 여기에 추가하고 싶은 필드를 정의합니다.
        phone_number = models.CharField(max_length=15, blank=True, null=True)
        profile_picture = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
        address = models.CharField(max_length=255, blank=True, null=True)

        # __str__ 메서드는 관리자 페이지 등에서 객체를 식별하는 데 사용됩니다.
        def __str__(self):
            return self.username
    ```

4.  **`settings.py`에 Custom User 모델 지정:**
    Django에게 사용할 사용자 모델이 무엇인지 알려줍니다. 이 설정은 **마이그레이션을 실행하기 전에 반드시 추가**해야 합니다.
    ```python
    # settings.py
    AUTH_USER_MODEL = 'accounts.CustomUser'
    ```

5.  **마이그레이션 실행:**
    ```bash
    python manage.py makemigrations accounts
    python manage.py migrate
    ```
    *   **주의:** `AUTH_USER_MODEL`을 변경한 후에는 기존의 `auth_user` 테이블을 사용하는 마이그레이션이 있다면 문제가 발생할 수 있습니다. 일반적으로 프로젝트 초기 단계에서 설정하는 것이 가장 좋습니다. 이미 데이터가 있는 상태에서 변경하려면 데이터 마이그레이션 전략을 신중하게 수립해야 합니다.

### 2.2. 로그인/로그아웃 뷰
Django는 내장된 로그인 및 로그아웃 뷰를 제공하여 인증 기능을 쉽게 추가할 수 있습니다.

**예시 (가상 `urls.py`):
```python
from django.contrib.auth import views as auth_views

urlpatterns = [
    # ...
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    # ...
]
```

### 2.3. 권한 및 그룹 관리
Django의 인증 시스템은 사용자에게 특정 권한을 부여하거나, 여러 사용자를 그룹으로 묶어 권한을 관리할 수 있는 기능을 제공합니다. 이는 관리자 페이지에서 설정할 수 있습니다.
