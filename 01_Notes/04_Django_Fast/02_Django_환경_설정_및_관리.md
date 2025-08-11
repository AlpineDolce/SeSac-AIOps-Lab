<h2>Django Backend: 프로젝트 설정 및 URL 라우팅</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django 프로젝트의 초기 설정, 앱 생성 및 등록, 주요 설정 파일(`settings.py`, `urls.py`)의 역할과 사용법을 이해하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. Django 프로젝트 생성 및 기본 구조](#1-django-프로젝트-생성-및-기본-구조)
  - [1.1. 개발 환경 설정 및 프로젝트 생성 (`django-admin startproject`)](#11-개발-환경-설정-및-프로젝트-생성-django-admin-startproject)
  - [1.2. 프로젝트 디렉토리 구조](#12-프로젝트-디렉토리-구조)
- [2. Django 앱 생성 및 등록](#2-django-앱-생성-및-등록)
  - [2.1. 앱 생성 (`python manage.py startapp`)](#21-앱-생성-python-managepy-startapp)
  - [2.2. 앱 등록 (INSTALLED\_APPS)](#22-앱-등록-installed_apps)
- [3. Django 설정 (settings.py) - 실무적 접근](#3-django-설정-settingspy---실무적-접근)
  - [3.1. 설정 파일 분리 (개발/운영)](#31-설정-파일-분리-개발운영)
  - [3.2. 민감 정보 관리 (환경 변수)](#32-민감-정보-관리-환경-변수)
  - [3.3. 주요 설정 상세](#33-주요-설정-상세)
    - [3.3.1. INSTALLED\_APPS, MIDDLEWARE](#331-installed_apps-middleware)
    - [3.3.2. TEMPLATES, DATABASES](#332-templates-databases)
    - [3.3.3. 정적 파일(Static) 및 미디어 파일(Media) 설정](#333-정적-파일static-및-미디어-파일media-설정)
  - [3.4. 로깅 설정 (Logging Configuration)](#34-로깅-설정-logging-configuration)
- [4. URL 라우팅 (urls.py)](#4-url-라우팅-urlspy)
  - [4.1. 프로젝트 `urls.py` 설정](#41-프로젝트-urlspy-설정)
  - [4.2. 앱 `urls.py` 설정 (`path`, `include`)](#42-앱-urlspy-설정-path-include)

---

## 1. Django 프로젝트 생성 및 기본 구조

### 1.1. 개발 환경 설정 및 프로젝트 생성 (`django-admin startproject`)
Django 프로젝트를 시작하기 전에 개발 환경을 설정하고 프로젝트를 생성합니다.

**1. 가상 환경 만들기 및 활성화:**
Python 프로젝트에서는 의존성 관리를 위해 가상 환경을 사용하는 것이 필수적입니다. `venv` 또는 `conda`를 사용하여 프로젝트별 독립적인 환경을 구축합니다.

```bash
# venv 사용 (Python 3.3+ 내장)
python -m venv venv
source venv/bin/activate # Linux/macOS
# venv\Scripts\activate # Windows

# conda 사용
conda create --name <your_project_env_name> python=3.9 # 예시: python 3.9 버전 지정
conda activate <your_project_env_name>
```
*   가상 환경은 프로젝트별로 독립적인 Python 환경을 제공하여 패키지 충돌을 방지하고, 프로젝트 의존성을 명확하게 관리할 수 있게 합니다.

**2. Django 설치:**
활성화된 가상 환경에 Django를 설치합니다.

```bash
pip install django 
```

**3. 프로젝트 생성:**
Django 프로젝트는 웹 애플리케이션의 전체 설정을 담는 컨테이너입니다. 프로젝트 이름은 해당 서비스의 목적을 명확히 나타내도록 **설명적인 이름**을 사용하는 것이 좋습니다.

```bash
# 현재 디렉토리에 프로젝트 생성 (권장: 불필요한 중첩 폴더 방지)
django-admin startproject <project_name> .

# 예시: AI 모델 서빙을 위한 백엔드 프로젝트
# cd <원하는_작업_디렉토리>
# django-admin startproject ai_model_backend .
```
*   `django-admin startproject <project_name> .` 명령은 현재 디렉토리에 프로젝트를 생성하여 불필요한 중첩 폴더(`myproject/myproject/`) 생성을 방지하고, 프로젝트 루트를 깔끔하게 유지할 수 있도록 합니다.

**4. 초기 마이그레이션 및 서버 실행:**
Django의 기본 데이터베이스 스키마를 적용하고 개발 서버를 실행하여 프로젝트가 정상적으로 생성되었는지 확인합니다.

```bash
python manage.py migrate # 기본 데이터베이스 스키마 적용
python manage.py runserver 
```
*   서버 실행 후 웹 브라우저에서 `http://127.0.0.1:8000/` 또는 `http://localhost:8000/`으로 접속하여 Django의 기본 환영 페이지(`It worked!`)가 나타나는지 확인합니다. 이는 프로젝트 설정이 성공적으로 완료되었음을 의미합니다.

### 1.2. 프로젝트 디렉토리 구조
`django-admin startproject <project_name> .` 명령으로 프로젝트를 생성하면 다음과 같은 기본 디렉토리 구조가 만들어집니다.

```
<project_root>/ # 프로젝트의 최상위 디렉토리 (예: ai_model_backend/)
├── manage.py
└── <project_name>/ # 프로젝트의 핵심 설정 파일들을 담는 Python 패키지 (예: ai_model_backend/)
    ├── __init__.py
    ├── asgi.py
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

*   `<project_root>/`: 프로젝트의 최상위 디렉토리입니다. Git 저장소의 루트가 되며, `manage.py` 파일이 위치합니다.
*   `manage.py`: Django 프로젝트와 상호작용하는 명령줄 유틸리티입니다. 서버 실행, 앱 생성, 마이그레이션, 테스트 실행 등 다양한 개발 및 관리 작업을 수행합니다.
*   `<project_name>/` (내부 패키지): 프로젝트의 실제 Python 패키지입니다. 이 디렉토리 내에 프로젝트의 전역 설정 파일들이 위치합니다.
    *   `__init__.py`: Python에게 이 디렉토리가 패키지임을 알려줍니다.
    *   `settings.py`: 이 Django 프로젝트의 모든 설정이 들어있습니다. 데이터베이스, 설치된 앱, 미들웨어, 템플릿 등 프로젝트의 전반적인 동작을 제어합니다.
    *   `urls.py`: 이 Django 프로젝트의 최상위 URL 선언이 들어있습니다. 모든 URL 요청을 처리하는 "목차" 역할을 하며, 각 앱의 URL을 포함(include)하여 라우팅합니다.
    *   `wsgi.py`: 프로젝트를 서비스하기 위한 WSGI(Web Server Gateway Interface) 호환 웹 서버 진입점입니다. 동기 웹 요청 처리에 사용됩니다.
    *   `asgi.py`: 프로젝트를 서비스하기 위한 ASGI(Asynchronous Server Gateway Interface) 호환 웹 서버 진입점입니다. 웹소켓, 롱 폴링 등 비동기 웹 요청 처리에 사용됩니다.
*   **`BASE_DIR`**: `settings.py` 파일이 위치한 프로젝트의 루트 디렉토리(즉, `manage.py`가 있는 `<project_root>/`)를 나타내는 파이썬 변수입니다. 파일 경로를 설정할 때 상대 경로 대신 `BASE_DIR`을 기준으로 절대 경로를 지정하는 것이 일반적이며 권장됩니다.
    ```python
    # settings.py 예시: BASE_DIR 활용
    import os
    # ...
    TEMPLATES = [
        {
            # ...
            'DIRS': [os.path.join(BASE_DIR, 'templates')],
            # ...
        },
    ]
    # ...
    STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
    ```

## 2. Django 앱 생성 및 등록

Django 프로젝트는 여러 개의 앱으로 구성될 수 있습니다. 각 앱은 특정 기능을 담당하는 모듈로, 프로젝트의 모듈성과 재사용성을 높이는 데 기여합니다.

### 2.1. 앱 생성 (`python manage.py startapp`)
Django 앱은 특정 기능을 수행하는 모듈입니다. 예를 들어, 블로그 기능, 사용자 관리 기능 등은 각각 별도의 앱으로 만들 수 있습니다. 앱 이름은 일반적으로 **단수형, 소문자**로 지정하며, 해당 앱의 기능을 명확히 나타내도록 합니다.

```bash
python manage.py startapp <app_name>
```

**예시:**
```bash
django-admin startapp users # 사용자 관리 앱
django-admin startapp products # 상품 관리 앱
django-admin startapp orders # 주문 관리 앱
```

앱을 생성하면 다음과 같은 기본 디렉토리 구조가 만들어집니다.

```
<app_name>/
├── migrations/ # 데이터베이스 스키마 변경 내역을 관리
│   └── __init__.py
├── __init__.py
├── admin.py    # Django 관리자 페이지에 모델을 등록
├── apps.py     # 앱의 설정 (AppConfig)
├── models.py   # 데이터베이스 모델 정의
├── tests.py    # 앱의 테스트 코드 작성
└── views.py    # 웹 요청을 처리하는 뷰 함수/클래스 정의
```
*   **앱의 역할:** 각 앱은 단일 책임 원칙(Single Responsibility Principle)을 따르는 것이 좋습니다. 즉, 하나의 앱은 하나의 명확한 기능 영역을 담당하도록 설계합니다.

### 2.2. 앱 등록 (INSTALLED_APPS)
생성한 앱을 Django 프로젝트에서 사용하려면, 프로젝트의 `settings.py` 파일에 있는 `INSTALLED_APPS` 리스트에 앱을 등록해야 합니다. 앱을 등록해야 Django가 해당 앱의 모델, 뷰, 템플릿 등을 인식하고 사용할 수 있습니다.

**`AppConfig`를 이용한 앱 등록 (권장):**
Django 1.7부터 도입된 `AppConfig`는 앱별 설정을 관리하고, 앱이 로드될 때 특정 코드를 실행하는 등 앱의 동작을 세밀하게 제어할 수 있게 합니다. `INSTALLED_APPS`에 `'<app_name>.apps.<AppConfigClassName>'` 형태로 등록하는 것이 모범 사례입니다.

**`myproject/settings.py` 예시:**
```python
INSTALLED_APPS = [
    # Django 기본 앱
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # 서드파티 앱 (설치 순서에 따라)
    # "rest_framework",
    # "debug_toolbar",

    # 사용자 정의 앱 (프로젝트 앱)
    "users.apps.UsersConfig",    # users 앱 등록
    "products.apps.ProductsConfig", # products 앱 등록
    "orders.apps.OrdersConfig",  # orders 앱 등록
]
```
*   **등록 순서:** 일반적으로 Django 기본 앱, 서드파티 앱, 사용자 정의 앱 순서로 등록하는 것이 관례입니다. 이는 의존성 문제를 줄이고 가독성을 높입니다.

**앱 생성 후 필수 단계: 마이그레이션**
새로운 앱을 생성하고 `models.py`에 모델을 정의했다면, 해당 모델을 데이터베이스에 반영하기 위해 마이그레이션 과정을 거쳐야 합니다.

```bash
python manage.py makemigrations <app_name> # 특정 앱의 변경사항만 감지
python manage.py migrate # 모든 앱의 마이그레이션 파일들을 데이터베이스에 적용
```
*   `makemigrations`: 모델의 변경사항을 감지하여 마이그레이션 파일(`migrations/0001_initial.py` 등)을 생성합니다.
*   `migrate`: 생성된 마이그레이션 파일들을 데이터베이스에 실제로 적용하여 테이블을 생성하거나 변경합니다.

## 3. Django 설정 (settings.py) - 실무적 접근

`settings.py` 파일은 Django 프로젝트의 핵심 설정 파일입니다. 실무 환경에서는 단순히 기본 설정을 사용하는 것을 넘어, 보안, 확장성, 환경별 분리를 고려한 체계적인 설정 관리가 필수적입니다. 이 섹션에서는 실무에서 권장되는 설정 방법들을 중심으로 설명합니다.

### 3.1. 설정 파일 분리 (개발/운영) 상세 가이드

프로젝트가 커지고 협업이 많아질수록, 단일 `settings.py` 파일은 여러 문제를 야기합니다.

- **보안 문제**: 운영 환경의 비밀 키가 개발 환경에도 노출될 수 있습니다.
- **관리의 어려움**: `if DEBUG:` 와 같은 조건문이 많아져 가독성이 떨어지고 실수가 발생하기 쉽습니다.
- **충돌 발생**: 팀원마다 다른 개발 환경 설정이 Git에서 충돌을 일으킬 수 있습니다.

이를 해결하기 위해 설정 파일을 **역할에 따라 분리**하는 것이 실무의 표준적인 방식입니다.

---

#### **1단계: `settings` 패키지 구조 생성**

먼저, 기존의 `settings.py` 파일을 `settings` 라는 패키지로 재구성합니다.

**변경 전 구조:**
```
<project_name>/
├── settings.py
├── ...
```

**변경 후 구조:**
```
<project_name>/
├── settings/
│   ├── __init__.py    # 이 디렉토리를 파이썬 패키지로 인식시킴
│   ├── base.py        # 모든 환경에 공통적인 설정
│   ├── development.py # 개발 환경 전용 설정
│   └── production.py  # 운영(배포) 환경 전용 설정
├── ...
```

**실행 방법:**

1.  프로젝트 설정 폴더(`<project_name>/`) 안에 `settings` 라는 새 디렉토리를 만듭니다.
2.  기존 `settings.py` 파일의 이름을 `base.py`로 바꾸고, `settings/` 디렉토리 안으로 옮깁니다.
3.  `settings/` 디렉토리 안에 비어있는 `__init__.py` 파일과 `development.py`, `production.py` 파일을 새로 생성합니다.

---

#### **2단계: `base.py` - 공통 설정 정의**

`base.py`에는 어떤 환경에서든 동일하게 적용되는 **핵심 설정**들을 남겨둡니다.

- `INSTALLED_APPS`
- `MIDDLEWARE`
- `TEMPLATES`
- `AUTH_PASSWORD_VALIDATORS`
- `LANGUAGE_CODE`, `TIME_ZONE`, `USE_I18N`, `USE_TZ`
- `STATIC_URL`, `STATICFILES_DIRS`, `MEDIA_URL`, `MEDIA_ROOT`
- `SECRET_KEY`나 `DATABASES` 같은 민감한 정보는 `base.py`에서 제거하거나, `decouple`을 사용한다면 기본값 없이 정의만 해둘 수 있습니다. (환경별 파일에서 반드시 설정하도록 강제)

**`settings/base.py` 예시:**
```python
import os
from pathlib import Path
from decouple import config # 환경변수 관리를 위해 추가

# Build paths inside the project like this: BASE_DIR / 'subdir'.
# settings/base.py 기준이므로 .parent를 한번 더 추가하여 프로젝트 루트를 가리키게 합니다.
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    # ...
]

MIDDLEWARE = [
    # ...
]

ROOT_URLCONF = '<project_name>.urls'

TEMPLATES = [
    # ...
]

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    # ...
]

# Internationalization
LANGUAGE_CODE = 'ko-kr'
TIME_ZONE = 'Asia/Seoul'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
]
```
**중요:** `BASE_DIR`의 경로가 `settings.py` 파일의 위치 변경에 따라 수정되어야 합니다. `Path(__file__).resolve().parent.parent.parent`와 같이 상위 디렉토리를 올바르게 가리키도록 조정합니다.

---

#### **3단계: `development.py` - 개발 환경 설정**

개발 환경에 특화된 설정을 정의합니다. `base.py`의 모든 설정을 가져온(`from .base import *`) 뒤, 필요한 부분을 덮어씁니다.

- `DEBUG = True`
- `ALLOWED_HOSTS = []` 또는 `['localhost', '127.0.0.1']`
- `SECRET_KEY`는 Git에 커밋되어도 상관없는 간단한 값으로 설정합니다.
- 데이터베이스는 가벼운 `SQLite`를 사용하거나, 로컬에 설치된 `PostgreSQL/MySQL`을 연결합니다.
- `django-debug-toolbar`나 `django-extensions` 같이 개발에 유용한 도구들을 `INSTALLED_APPS`에 추가합니다.

**`settings/development.py` 예시:**
```python
from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# 개발 환경에서는 간단한 키 사용
SECRET_KEY = 'django-insecure-this-is-a-dev-key-so-no-problem'

# 개발용 DB (SQLite)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Django Debug Toolbar 설정
INSTALLED_APPS += [
    'debug_toolbar',
]
MIDDLEWARE += [
    'debug_toolbar.middleware.DebugToolbarMiddleware',
]
INTERNAL_IPS = [
    '127.0.0.1',
]
```

---

#### **4단계: `production.py` - 운영 환경 설정**

운영(배포) 환경을 위한 설정입니다. **보안과 성능**에 초점을 맞춥니다.

- `DEBUG = False`
- `SECRET_KEY`, `ALLOWED_HOSTS`, `DATABASES` 등 모든 민감 정보는 **반드시** 환경 변수(`os.environ` 또는 `decouple`)를 통해 가져옵니다.
- HTTPS 관련 보안 설정(`CSRF_COOKIE_SECURE`, `SESSION_COOKIE_SECURE`)을 `True`로 설정합니다.
- 로깅 설정을 더 엄격하게 구성합니다. (예: `INFO` 레벨 이상만 파일에 기록)

**`settings/production.py` 예시:**
```python
from .base import *
from decouple import config, Csv

# 운영 환경이므로 DEBUG는 False
DEBUG = False

# decouple을 사용하여 .env 파일 또는 환경 변수에서 값 로드
SECRET_KEY = config('SECRET_KEY')
ALLOWED_HOSTS = config('ALLOWED_HOSTS', cast=Csv())

# 운영용 데이터베이스
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME'),
        'USER': config('DB_USER'),
        'PASSWORD': config('DB_PASSWORD'),
        'HOST': config('DB_HOST'),
        'PORT': config('DB_PORT', cast=int),
    }
}

# HTTPS 환경을 위한 보안 설정
CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE = True
```

---

#### **5단계: Django에 설정 파일 알려주기**

마지막으로, Django가 어떤 설정 파일을 사용해야 하는지 알려줘야 합니다. `manage.py`와 `wsgi.py`, `asgi.py` 파일에서 `DJANGO_SETTINGS_MODULE` 환경 변수를 지정합니다.

**`manage.py` 수정:**
```python
def main():
    """Run administrative tasks."""
    # 이 부분을 수정합니다.
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', '<project_name>.settings.development')
    # ...
```
- `manage.py`는 주로 로컬 개발 시 사용되므로, 기본값을 `development`로 설정하는 것이 편리합니다.

**`wsgi.py` / `asgi.py` 수정:**
```python
import os
from django.core.wsgi import get_wsgi_application

# 이 부분을 수정합니다.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', '<project_name>.settings.production')

application = get_wsgi_application()
```
- `wsgi/asgi` 파일은 Gunicorn, uWSGI 등 배포용 웹 서버와 연동되므로, 기본값을 `production`으로 설정하는 것이 안전합니다.

**서버 실행 시 설정 선택:**

- **개발 서버:** `python manage.py runserver` 명령을 그대로 사용하면 `development` 설정이 적용됩니다.
- **운영 서버:** 배포 환경에서는 `DJANGO_SETTINGS_MODULE` 환경 변수를 직접 설정하여 `production` 설정을 사용하도록 강제합니다.
  ```bash
  # Gunicorn으로 운영 서버 실행 시
  export DJANGO_SETTINGS_MODULE=<project_name>.settings.production
  gunicorn --workers 3 --bind 0.0.0.0:8000 <project_name>.wsgi:application
  ```

이러한 구조를 통해 각 환경에 맞는 설정을 명확히 분리하고, 안전하고 효율적으로 프로젝트를 관리할 수 있습니다.

### 3.2. 민감 정보 관리 (환경 변수) 상세 가이드

소스 코드에 민감 정보를 직접 작성하는 것(하드코딩)은 심각한 보안 사고로 이어질 수 있습니다. 예를 들어, 실수로 `SECRET_KEY`나 데이터베이스 암호가 포함된 코드를 공용 GitHub 저장소에 푸시하면, 악의적인 봇이 수 분 내에 이를 탈취하여 서버를 공격하거나 데이터를 유출할 수 있습니다.

**민감 정보란?**
- `SECRET_KEY`
- `DEBUG` 상태 값 (운영에서 `True`로 노출되면 안 됨)
- 데이터베이스 연결 정보 (사용자 이름, 암호, 호스트 주소)
- 이메일 서버 접속 정보
- 외부 API 키 (예: AWS, Google Cloud, 결제 모듈 키)

이러한 정보들은 코드와 완전히 분리하여 **환경 변수(Environment Variables)**를 통해 관리하는 것이 필수적입니다.

---

#### **작동 원리 및 라이브러리 선택**

환경 변수는 코드가 실행되는 셸(운영체제)에 설정된 변수입니다. Django는 `os.environ`을 통해 이 변수들을 읽을 수 있습니다.

하지만 매번 `os.environ.get('VAR_NAME')`을 사용하는 것은 다소 번거롭고, 특히 로컬 개발 환경에서 변수를 설정하기가 불편합니다. 이를 해결하기 위해 `python-decouple`이나 `django-environ` 같은 라이브러리를 사용합니다.

- **`python-decouple`의 장점**: 가볍고, Django에 종속적이지 않아 범용적으로 사용할 수 있습니다. `config()` 함수 하나로 환경 변수와 `.env` 파일을 모두 처리해주는 편리함을 제공합니다.

`config()` 함수의 동작 순서는 다음과 같습니다.
1.  먼저 코드 실행 환경(운영체제)에 해당 이름의 환경 변수가 있는지 확인합니다.
2.  없다면, 프로젝트 루트의 `.env` 파일에서 해당 키를 찾습니다.
3.  두 군데 모두 없다면 `UndefinedValueError`를 발생시킵니다. (단, `default` 값이 지정된 경우는 예외)

이러한 원리 덕분에, 개발 환경에서는 `.env` 파일로 편하게 작업하고, 운영 환경에서는 서버에 직접 설정된 환경 변수를 코드가 자동으로 읽게 할 수 있습니다.

---

#### **단계별 적용 가이드**

**1단계: 라이브러리 설치**
```bash
pip install python-decouple
```

**2단계: `.env` 파일 작성 및 `.gitignore` 등록**

프로젝트 최상위 디렉토리(manage.py가 있는 곳)에 `.env` 파일을 생성합니다. 이 파일은 **로컬 개발 환경에서만 사용**됩니다.

**.env 예시:**
```ini
# Django Core
# 주의: 여기에 작성된 키는 개발용이며, 실제 운영에서는 서버에 직접 환경변수를 설정해야 합니다.
SECRET_KEY=django-insecure-my-local-dev-secret-key
DEBUG=True
ALLOWED_HOSTS=.localhost, 127.0.0.1

# Database (PostgreSQL)
DB_ENGINE=django.db.backends.postgresql
DB_NAME=my_local_db
DB_USER=my_user
DB_PASSWORD=my_password
DB_HOST=localhost
DB_PORT=5432

# AWS S3
AWS_ACCESS_KEY_ID=my_local_aws_access_key
AWS_SECRET_ACCESS_KEY=my_local_aws_secret_key
AWS_STORAGE_BUCKET_NAME=my-local-s3-bucket
```

**가장 중요한 단계:** `.env` 파일은 민감 정보를 담고 있으므로 **절대로 Git으로 관리하면 안 됩니다.** 프로젝트의 `.gitignore` 파일에 다음 한 줄을 반드시 추가합니다.

**.gitignore:**
```
# Environment variables
.env
```

**3단계: `settings.py`에 적용하기**

`decouple`의 `config` 함수를 사용하여 `.env` 파일의 값을 읽어옵니다. 이때 `cast` 인자를 활용하면 문자열이 아닌 다른 타입(Boolean, Integer, List 등)으로 깔끔하게 변환할 수 있습니다.

**`settings/production.py` 또는 `settings/base.py` 예시:**
```python
from decouple import config, Csv

# Core
# default 값을 지정하지 않으면, 환경 변수가 없을 때 에러를 발생시켜 설정 누락을 방지합니다.
SECRET_KEY = config('SECRET_KEY') 

# cast=bool: "True", "yes", "1" 등의 문자열을 True 불리언으로 변환합니다.
DEBUG = config('DEBUG', default=False, cast=bool)

# cast=Csv: 콤마로 구분된 문자열을 리스트로 변환합니다. "a,b, c" -> ['a', 'b', 'c']
ALLOWED_HOSTS = config('ALLOWED_HOSTS', cast=Csv())

# Database
DATABASES = {
    'default': {
        'ENGINE': config('DB_ENGINE', default='django.db.backends.sqlite3'),
        'NAME': config('DB_NAME', default=BASE_DIR / 'db.sqlite3'),
        'USER': config('DB_USER', default=''),
        'PASSWORD': config('DB_PASSWORD', default=''),
        'HOST': config('DB_HOST', default=''),
        'PORT': config('DB_PORT', default='', cast=int),
    }
}
```

**4단계: 팀 협업을 위한 `.env.example`**

새로운 팀원이 프로젝트에 참여했을 때 어떤 환경 변수가 필요한지 알려주기 위해, `.env` 파일의 복사본인 `.env.example` 파일을 만들어 Git에 추가하는 것이 좋습니다. 여기에는 실제 값이 아닌, 변수 이름과 설명만 담습니다.

**.env.example 예시:**
```ini
# Django Core
SECRET_KEY=
DEBUG=True
ALLOWED_HOSTS=127.0.0.1, localhost

# Database (PostgreSQL)
DB_ENGINE=django.db.backends.postgresql
DB_NAME=
DB_USER=
DB_PASSWORD=
DB_HOST=localhost
DB_PORT=5432
```
이제 새로운 개발자는 이 파일을 복사하여 `.env` 파일을 만들고, 자신의 로컬 환경에 맞게 값을 채워넣기만 하면 됩니다.

**5단계: 운영 환경에서의 관리**

운영 서버(Heroku, AWS, Docker 등)에서는 `.env` 파일을 사용하지 않습니다. 대신, 각 플랫폼에서 제공하는 방법으로 환경 변수를 직접 설정해야 합니다.

- **AWS Elastic Beanstalk**: `Configuration > Software > Environment properties` 에서 설정
- **Heroku**: `Settings > Config Vars` 에서 설정
- **Docker**: `docker run` 명령어의 `-e` 옵션 또는 `docker-compose.yml`의 `environment` 섹션을 통해 설정

이렇게 함으로써 개발 환경과 운영 환경의 설정을 코드 변경 없이 유연하게 전환할 수 있습니다.

### 3.3. 주요 설정 상세

이 섹션에서는 `settings/base.py`에 주로 위치하는 주요 설정 변수들의 역할과 실무적인 구성 방법을 상세히 다룹니다.

---

#### 3.3.1. `INSTALLED_APPS`와 `MIDDLEWARE`

**`INSTALLED_APPS` - 설치된 앱 관리**

`INSTALLED_APPS`는 Django 프로젝트가 인식하고 관리하는 모든 앱의 목록입니다. Django는 이 목록을 순회하며 모델, URL, 템플릿, 정적 파일 등 앱의 구성 요소를 찾습니다.

- **등록 순서의 중요성**: Django는 목록의 위에서 아래 순서로 앱을 로드하고 처리합니다. 대부분의 경우 순서가 문제 되지 않지만, 특정 앱이 다른 앱의 리소스(특히 템플릿이나 정적 파일)를 덮어써야 하는 경우 순서가 중요해집니다. 권장되는 순서는 다음과 같습니다.
    1.  **Django 프레임워크 기본 앱**: `django.contrib.admin`, `django.contrib.auth` 등
    2.  **서드파티(Third-party) 앱**: 외부에서 설치한 앱. 예: `rest_framework`, `corsheaders`
    3.  **사용자 정의 앱**: 직접 개발한 앱. 예: `users`, `products`

- **실무에서 자주 사용하는 서드파티 앱:**
    - `rest_framework`: Django로 RESTful API를 구축하기 위한 필수 라이브러리.
    - `rest_framework.authtoken`: 토큰 기반 인증 기능을 제공.
    - `corsheaders`: 브라우저의 CORS(Cross-Origin Resource Sharing) 정책을 관리하여 다른 도메인의 프론트엔드와 통신할 수 있게 함.
    - `django_extensions`: `shell_plus`(자동으로 모델 임포트), `runserver_plus`(디버깅 기능 강화) 등 매우 유용한 관리자 명령어를 추가.
    - `debug_toolbar`: 개발 중 SQL 쿼리, 템플릿 컨텍스트 등 상세한 디버깅 정보를 브라우저에 표시.

- **`AppConfig` 사용 권장**: `'users'`처럼 앱 이름만 등록하는 대신, `'users.apps.UsersConfig'`와 같이 `AppConfig` 클래스 경로를 사용하는 것이 좋습니다. 이를 통해 각 앱의 로딩 시점이나 초기화 동작을 세밀하게 제어할 수 있습니다.


**`MIDDLEWARE` - 요청과 응답의 관문**

미들웨어는 요청(Request)이 뷰에 도달하기 전과, 응답(Response)이 브라우저로 전송되기 전에 거치는 처리 계층입니다. 흔히 **'양파 껍질'**에 비유하며, 각 계층이 고유한 역할을 수행합니다.

- **동작 방식**: 
    1.  **요청(Request)**: 목록의 **위에서 아래로** 각 미들웨어를 순서대로 통과하며 처리됩니다.
    2.  **응답(Response)**: 목록의 **아래에서 위로** 각 미들웨어를 역순으로 통과하며 처리됩니다.

- **주요 미들웨어의 역할:**
    - `django.middleware.security.SecurityMiddleware`: HSTS, SSL 리다이렉트 등 여러 보안 관련 설정을 적용합니다. `settings/production.py`에서 `SECURE_HSTS_SECONDS = True` 와 같은 관련 설정을 활성화해야 합니다. 가장 상단에 위치하는 것이 좋습니다.
    - `django.contrib.sessions.middleware.SessionMiddleware`: HTTP 요청 간에 세션을 관리합니다. 사용자의 로그인 상태 등을 유지하는 데 사용됩니다.
    - `corsheaders.middleware.CorsMiddleware`: CORS 헤더를 응답에 추가합니다. 다른 미들웨어보다 먼저 CORS 정책을 확인해야 하므로, 보통 `CommonMiddleware`보다 위에 위치시킵니다.
    - `django.middleware.common.CommonMiddleware`: URL 끝에 슬래시(`/`)를 추가(`APPEND_SLASH=True`)하는 등 기본적인 처리를 수행합니다.
    - `django.middleware.csrf.CsrfViewMiddleware`: POST, PUT, DELETE 등 상태를 변경하는 요청에 대해 CSRF 토큰을 검증하여 사이트 간 요청 위조 공격을 방어합니다. (Stateless한 API 서버에서는 비활성화하기도 합니다.)
    - `django.contrib.auth.middleware.AuthenticationMiddleware`: 세션 정보를 바탕으로 요청에 `user` 객체를 추가합니다. 이 미들웨어 덕분에 `request.user`로 로그인된 사용자에 접근할 수 있습니다.
    - `django.contrib.messages.middleware.MessageMiddleware`: 일회성 알림 메시지 기능을 활성화합니다.

---

#### 3.3.2. `TEMPLATES`와 `DATABASES`

**`TEMPLATES` - 템플릿 엔진 설정**

- `'DIRS'`: Django가 앱별 `templates` 디렉토리 외에 추가로 템플릿 파일을 검색할 경로를 지정합니다. 보통 프로젝트 전역에서 사용되는 `base.html`, `navbar.html` 같은 기본 레이아웃 템플릿을 이곳에(`[BASE_DIR / 'templates']`) 보관합니다.
- `'APP_DIRS': True`: `INSTALLED_APPS`에 등록된 각 앱의 `templates` 하위 디렉토리를 자동으로 검색하도록 설정합니다. 이를 통해 각 앱은 자신만의 템플릿을 가질 수 있어 재사용성이 높아집니다.
- `'OPTIONS': {'context_processors': [...]}`: 모든 템플릿에 기본적으로 전달될 컨텍스트 변수를 정의합니다. 예를 들어, `django.contrib.auth.context_processors.auth` 덕분에 모든 템플릿에서 `{{ user }}` 변수를 사용하여 현재 로그인된 사용자에 접근할 수 있습니다.

**`DATABASES` - 데이터베이스 연결 설정**

- **개발과 운영 환경의 DB 통일**: 사소한 기능 차이로 인해 운영 환경에서만 버그가 발생하는 것을 막기 위해, **가급적 개발 환경과 운영 환경의 데이터베이스 엔진을 동일하게(예: 둘 다 PostgreSQL) 사용하는 것을 강력히 권장**합니다. SQLite는 사용이 간편하지만, 데이터 타입이나 제약 조건 등에서 PostgreSQL/MySQL과 달라 예상치 못한 문제를 일으킬 수 있습니다.
- **`CONN_MAX_AGE` (커넥션 재사용)**: 이 값은 데이터베이스 커넥션의 수명을 초 단위로 지정합니다. `0`이 기본값이며, 매 요청마다 새로운 커넥션을 맺습니다. `60`이나 `300` 같은 값을 설정하면, 한번 맺은 커넥션을 지정된 시간 동안 재사용하여 매번 연결을 새로 맺는 오버헤드를 줄여줍니다. 이는 성능 향상에 큰 도움이 되는 간단하면서도 효과적인 설정입니다.
- **전문 커넥션 풀러(Connection Pooler)**: 트래픽이 매우 많은 대규모 서비스에서는 `CONN_MAX_AGE`만으로는 부족할 수 있습니다. 이 경우, Django 애플리케이션과 데이터베이스 서버 사이에 `PgBouncer` (PostgreSQL용) 같은 전문 커넥션 풀러를 별도로 두어 커넥션을 훨씬 효율적으로 관리하는 방법을 사용합니다.

---

#### 3.3.3. 정적 파일(Static) 및 미디어 파일(Media) 설정

Django에서 이 두 가지는 명확히 구분되어 관리되어야 합니다.

- **정적 파일 (Static Files)**: 개발자가 프로젝트를 위해 미리 준비한 파일입니다. 코드의 일부로 버전 관리(Git)에 포함됩니다. (예: CSS, JavaScript, 로고 이미지, 폰트)
- **미디어 파일 (Media Files)**: 사용자가 애플리케이션을 사용하며 업로드하는 파일입니다. 코드와 무관하며 버전 관리에 포함되지 않습니다. (예: 프로필 사진, 첨부 파일)

**정적 파일 설정 및 운영 워크플로우**

1.  **`STATIC_URL = '/static/'`**: 브라우저에서 정적 파일에 접근할 URL 접두사입니다.
2.  **`STATICFILES_DIRS`**: 프로젝트 전역에서 사용할 정적 파일(예: `main.css`)이 위치한 디렉토리 목록입니다.
3.  **`STATIC_ROOT`**: **운영 환경을 위한 설정**입니다. `python manage.py collectstatic` 명령을 실행하면, Django는 `STATICFILES_DIRS`와 각 앱의 `static` 디렉토리에 흩어져 있는 모든 정적 파일을 찾아 `STATIC_ROOT`로 지정된 단일 디렉토리로 복사합니다.
4.  **운영 서버(Nginx 등) 설정**: 웹 서버는 `/static/`으로 시작하는 모든 요청을 Django로 보내지 않고, `STATIC_ROOT` 디렉토리에서 직접 파일을 찾아 제공하도록 설정합니다. 이는 Django의 부하를 줄여 성능을 크게 향상시킵니다.

**미디어 파일 설정 및 서빙**

1.  **`MEDIA_URL = '/media/'`**: 브라우저에서 미디어 파일에 접근할 URL 접두사입니다.
2.  **`MEDIA_ROOT`**: 업로드된 파일이 서버의 파일 시스템에 실제로 저장될 경로입니다.
3.  **개발 환경 서빙**: 개발 서버는 미디어 파일을 자동으로 서빙하지 않으므로, 프로젝트의 `urls.py`에 `if settings.DEBUG:` 조건을 사용하여 임시로 서빙 규칙을 추가해야 합니다.
4.  **운영 환경 서빙**: 정적 파일과 마찬가지로, 웹 서버(Nginx)가 `/media/` 경로의 요청을 `MEDIA_ROOT` 디렉토리에서 직접 처리하도록 설정합니다.

**실무 Tip: 클라우드 스토리지 활용**

서버에 직접 파일을 저장하는 대신, `django-storages` 라이브러리를 사용하여 AWS S3, Google Cloud Storage 같은 외부 스토리지에 정적/미디어 파일을 저장하는 것이 현대적인 방식입니다.

- **장점**: 서버 용량 부담 감소, CDN을 통한 전송 속도 향상, 서버 스케일 아웃 시 파일 동기화 문제 해결 등
- **설정 예시 (`settings/production.py`):**
    ```python
    # settings/production.py
    DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
    STATICFILES_STORAGE = 'storages.backends.s3boto3.S3StaticStorage'

    AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
    AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
    AWS_S3_REGION_NAME = config('AWS_S3_REGION_NAME')
    ```

### 3.4. 로깅 설정 (Logging Configuration)

체계적인 로깅은 애플리케이션의 동작을 모니터링하고 문제 발생 시 디버깅하는 데 필수적입니다. 실무에서는 몇 가지를 더 고려할 수 있습니다.

- **구조화된 로깅 (Structured Logging):** 로그를 일반 텍스트가 아닌 JSON 형식으로 남기면 Datadog, ELK, Splunk 같은 로그 수집/분석 시스템에서 파싱하고 검색하기 훨씬 용이해집니다. `python-json-logger` 라이브러리를 사용하면 쉽게 구현할 수 있습니다.
- **환경별 로깅 레벨 제어:** 개발 환경에서는 `DEBUG` 레벨의 로그를, 운영 환경에서는 `INFO` 레벨의 로그만 기록하도록 환경 변수로 제어하면 유연성이 높아집니다.

**운영 환경 로깅 설정 개선 예시 (`settings/production.py`):**

```python
# settings/production.py

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        # JSON 포매터 추가
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(lineno)d %(pathname)s",
        },
        "verbose": {
            "format": "{levelname} {asctime} {module} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "level": "INFO", # 운영에서는 INFO 이상만
            "class": "logging.StreamHandler",
            "formatter": "verbose", # 또는 "json"
        },
        "file_prod": { # 운영 환경용 파일 핸들러
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(BASE_DIR, 'logs/prod.log'),
            "maxBytes": 1024 * 1024 * 5, # 5MB
            "backupCount": 5,
            "formatter": "json", # JSON 포맷으로 기록
        },
        "mail_admins": {
            "level": "ERROR",
            "class": "django.utils.log.AdminEmailHandler",
            "formatter": "verbose",
        }
    },
    "loggers": {
        "django": {
            "handlers": ["console", "file_prod"],
            "level": "INFO",
            "propagate": False,
        },
        "django.request": {
            "handlers": ["mail_admins"],
            "level": "ERROR",
            "propagate": False,
        },
        "my_project_apps": { # 프로젝트의 모든 앱에 대한 로거
            "handlers": ["console", "file_prod"],
            "level": "INFO", # 운영 환경에서는 INFO 레벨
            "propagate": False,
        },
    },
}
```

위 설정에서는 `my_project_apps`라는 로거를 정의하여 프로젝트 내 모든 앱의 로그를 일관되게 관리할 수 있습니다. 각 앱의 `views.py`에서는 다음과 같이 로거를 사용합니다.

```python
# myapp/views.py
import logging

# settings.py에 정의된 로거 이름 사용
logger = logging.getLogger('my_project_apps')

def my_view(request):
    logger.info("This is an informational message.")
    try:
        # ... some logic ...
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True) # exc_info=True로 트레이스백 기록
    # ...
```

## 4. URL 라우팅 (urls.py) 상세 가이드

URL 라우팅은 웹사이트의 주소(URL)를 특정 기능(View)과 연결하는, Django의 핵심적인 "교통정리" 시스템입니다. 사용자가 특정 URL로 접속하면, Django는 `settings.py`에 정의된 `ROOT_URLCONF` 파일(보통 `<project_name>/urls.py`)을 시작으로 어떤 뷰가 이 요청을 처리해야 할지 찾아 나섭니다.

효과적인 URL 설계는 애플리케이션을 논리적이고, 재사용 가능하며, 유지보수하기 쉽게 만듭니다.

---

### 4.1. 프로젝트 `urls.py`의 역할: 중앙 관제소

프로젝트의 최상위 `urls.py` 파일은 전체 URL 구조의 진입점 역할을 합니다. 이 파일의 주된 임무는 들어온 요청의 URL 접두사(prefix)를 보고, 해당 요청을 처리할 적절한 **앱(app)**으로 위임하는 것입니다.

**모범 사례:**
- **`include()`를 적극적으로 사용하세요**: 각 앱의 URL은 해당 앱의 `urls.py` 파일에서 관리하도록 하고, 프로젝트 `urls.py`에서는 `include()`를 사용해 연결만 합니다. 이렇게 하면 프로젝트 `urls.py`가 간결하게 유지되고, 각 앱은 독립적으로 URL 구조를 가질 수 있어 모듈성이 극대화됩니다.
- **일관된 URL 접두사 사용**: 각 앱의 URL을 포함할 때, API라면 `api/v1/users/`처럼, 일반 웹 페이지라면 `blog/`, `products/`처럼 일관된 접두사를 붙여주는 것이 좋습니다.

**프로젝트 `urls.py` 예시:**
```python
# <project_name>/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # 1. 관리자 페이지
    path("admin/", admin.site.urls),

    # 2. 각 앱의 urls.py를 include하여 URL 위임
    path("blog/", include("blog.urls")), # "/blog/"로 시작하는 모든 URL은 blog.urls에서 처리
    path("users/", include("users.urls")), # "/users/"로 시작하는 모든 URL은 users.urls에서 처리
    
    # API를 위한 URL 접두사 사용 예시
    # path("api/v1/products/", include("products.api_urls"))
]

# 3. 개발 환경에서 미디어 파일을 서빙하기 위한 설정 (운영 환경에서는 웹서버가 처리)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

---

### 4.2. 앱 `urls.py`의 역할: 기능별 라우팅 테이블

각 앱 내부에 생성하는 `urls.py` 파일은 해당 앱이 제공하는 구체적인 기능들의 URL을 정의합니다.

**`app_name` 설정과 네임스페이스(Namespace)**

파일 최상단에 `app_name = 'blog'`와 같이 네임스페이스를 지정하는 것은 매우 중요합니다. 만약 `users` 앱과 `blog` 앱 양쪽에 `list`라는 이름의 URL이 있다면, Django는 어떤 `list`를 말하는지 구분할 수 없습니다. `app_name`은 이처럼 URL 이름에 소속을 부여하여 충돌을 방지합니다.

**`path()` 함수 상세 분석**

`path(route, view, name=None, kwargs=None)` 함수는 URL 패턴을 정의하는 핵심 요소입니다.

- **`route` (경로 문자열)**: 매칭될 URL 패턴입니다. 경로의 일부를 변수로 캡처하기 위해 **경로 변환기(Path Converter)**를 사용합니다.
    - `<str:username>`: 문자열(공백 제외)을 `username` 변수로 캡처합니다. (기본값)
    - `<int:post_id>`: 정수를 `post_id` 변수로 캡처합니다.
    - `<slug:post_slug>`: 슬러그 문자열(알파벳, 숫자, 하이픈, 밑줄)을 `post_slug` 변수로 캡처합니다.
    - `<uuid:user_uuid>`: UUID를 `user_uuid` 변수로 캡처합니다.
    - `<path:full_path>`: 슬래시(`/`)를 포함한 모든 문자열을 `full_path` 변수로 캡처합니다.

- **`view` (뷰 함수 또는 클래스)**: `route`가 매칭되었을 때 실행될 뷰입니다.
    - **함수 기반 뷰 (FBV)**: `views.post_list` 와 같이 함수 이름을 직접 전달합니다.
    - **클래스 기반 뷰 (CBV)**: `views.PostListView.as_view()` 와 같이 클래스에 `.as_view()` 메소드를 호출한 결과를 전달해야 합니다.

- **`name` (URL 이름)**: 이 URL 패턴에 고유한 이름을 부여합니다. **URL 하드코딩을 피하기 위한 가장 중요한 설정입니다.** 만약 나중에 URL 경로(`route`)를 변경하더라도, 코드에서는 `name`을 사용했기 때문에 아무것도 수정할 필요가 없습니다. (DRY 원칙)

**앱 `urls.py` 종합 예시:**
```python
# blog/urls.py

from django.urls import path
from . import views

# URL 네임스페이스 지정
app_name = "blog"

urlpatterns = [
    # 예: /blog/
    path("", views.PostListView.as_view(), name="post_list"),

    # 예: /blog/5/  (views.post_detail 함수는 post_id 인자를 받음)
    path("<int:post_id>/", views.post_detail, name="post_detail"),

    # 예: /blog/hello-world/  (views.post_detail_by_slug 함수는 post_slug 인자를 받음)
    path("<slug:post_slug>/", views.post_detail_by_slug, name="post_detail_by_slug"),

    # 예: /blog/new/
    path("new/", views.post_create, name="post_create"),

    # 예: /blog/5/edit/
    path("<int:post_id>/edit/", views.post_update, name="post_update"),
]
```

---

### 4.3. URL 이름의 활용: `{% url %}`과 `reverse()`

`path()` 함수에 `name`을 지정하는 이유는 템플릿과 뷰에서 URL을 동적으로 생성하기 위함입니다.

**템플릿에서 (`{% url %}` 태그)**

템플릿에서는 URL을 직접 `/blog/5/` 와 같이 하드코딩하는 대신, `{% url %}` 태그와 URL 이름을 사용해야 합니다.

```html
<!-- 잘못된 방법 (하드코딩) -->
<a href="/blog/{{ post.id }}/">{{ post.title }}</a>

<!-- 올바른 방법 (URL 이름 사용) -->
<!-- 'app_name:url_name' 형식으로 사용 -->
<a href="{% url 'blog:post_detail' post.id %}">{{ post.title }}</a>
```

**뷰에서 (`reverse()` 함수)**

뷰 로직 내에서 특정 URL로 리다이렉트해야 할 때 `reverse()` 함수를 사용합니다.

```python
# blog/views.py
from django.shortcuts import redirect
from django.urls import reverse

def post_create(request):
    # ... (글 생성 로직)
    
    # 글 생성이 성공하면, 방금 만든 글의 상세 페이지로 리다이렉트
    # new_post 객체가 생성되었다고 가정
    return redirect(reverse('blog:post_detail', kwargs={'post_id': new_post.id}))
```

### 4.4. 고급: 정규 표현식을 위한 `re_path()`

`path()` 함수의 경로 변환기로 처리하기 어려운 복잡한 URL 패턴이 필요할 경우, `re_path()`를 사용하여 파이썬 정규 표현식을 직접 활용할 수 있습니다.

```python
from django.urls import re_path

# 예: /articles/2025/
re_path(r"^articles/(?P<year>[0-9]{4})/$ ", views.year_archive, name="year_archive"),
```
- `(?P<year>[0-9]{4})`: `year`라는 이름으로 4자리 숫자를 캡처하는 정규 표현식 그룹입니다.

하지만 `path()`가 더 가독성이 높고 배우기 쉬우므로, 대부분의 경우에는 `path()`를 우선적으로 사용하고 꼭 필요할 때만 `re_path()`를 사용하는 것이 좋습니다.