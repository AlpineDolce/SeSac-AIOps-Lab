<h2>Django Backend: 환경 설정 및 관리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 개발, 테스트, 운영 환경에 따른 Django 설정 관리(환경 변수, `python-decouple` 등) 및 보안 고려사항을 학습하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. Django 설정 (settings.py) - 실무적 접근](#1-django-설정-settingspy---실무적-접근)
- [2. 설정 파일 분리 (개발/운영) 상세 가이드](#2-설정-파일-분리-개발운영-상세-가이드)
  - [2.1. **1단계: `settings` 패키지 구조 생성**](#21-1단계-settings-패키지-구조-생성)
  - [2.2. **2단계: `base.py` - 공통 설정 정의**](#22-2단계-basepy---공통-설정-정의)
  - [2.3. **3단계: `development.py` - 개발 환경 설정**](#23-3단계-developmentpy---개발-환경-설정)
  - [2.4. **4단계: `production.py` - 운영 환경 설정**](#24-4단계-productionpy---운영-환경-설정)
  - [2.5. **5단계: Django에 설정 파일 알려주기**](#25-5단계-django에-설정-파일-알려주기)
- [3. 주요 설정 상세](#3-주요-설정-상세)
  - [3.1. `INSTALLED_APPS`와 `MIDDLEWARE`](#31-installed_apps와-middleware)
  - [3.2. `TEMPLATES`와 `DATABASES`](#32-templates와-databases)
  - [3.3. 정적 파일(Static) 및 미디어 파일(Media) 설정](#33-정적-파일static-및-미디어-파일media-설정)

---
## 1. Django 설정 (settings.py) - 실무적 접근

`settings.py` 파일은 Django 프로젝트의 핵심 설정 파일입니다. 실무 환경에서는 단순히 기본 설정을 사용하는 것을 넘어, 보안, 확장성, 환경별 분리를 고려한 체계적인 설정 관리가 필수적입니다. 이 섹션에서는 실무에서 권장되는 설정 방법들을 중심으로 설명합니다.

## 2. 설정 파일 분리 (개발/운영) 상세 가이드

프로젝트가 커지고 협업이 많아질수록, 단일 `settings.py` 파일은 여러 문제를 야기합니다.

- **보안 문제**: 운영 환경의 비밀 키가 개발 환경에도 노출될 수 있습니다.
- **관리의 어려움**: `if DEBUG:` 와 같은 조건문이 많아져 가독성이 떨어지고 실수가 발생하기 쉽습니다.
- **충돌 발생**: 팀원마다 다른 개발 환경 설정이 Git에서 충돌을 일으킬 수 있습니다.

이를 해결하기 위해 설정 파일을 **역할에 따라 분리**하는 것이 실무의 표준적인 방식입니다.

---

### 2.1. **1단계: `settings` 패키지 구조 생성**

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

### 2.2. **2단계: `base.py` - 공통 설정 정의**

`base.py` 파일은 Django 프로젝트의 모든 환경(개발, 테스트, 운영 등)에서 **동일하게 적용되는 핵심 설정**들을 정의하는 곳입니다. 이 파일은 환경별로 달라지는 민감한 정보나 특정 환경에만 필요한 설정들을 제외하고, 프로젝트의 기본적인 동작 방식과 구조를 결정하는 데 사용됩니다.

**`base.py`에 포함되는 주요 설정 항목들:**

*   **`INSTALLED_APPS`**: 프로젝트에 설치된 모든 Django 앱(기본 앱, 서드파티 앱, 사용자 정의 앱)의 목록입니다. 어떤 환경에서든 동일한 앱들이 사용되므로 `base.py`에 정의합니다.
*   **`MIDDLEWARE`**: 요청과 응답 처리 과정에서 공통적으로 적용되는 미들웨어 목록입니다. 보안, 세션 관리, 메시지 처리 등 대부분의 미들웨어는 환경에 관계없이 동일하게 적용됩니다.
*   **`ROOT_URLCONF`**: 프로젝트의 최상위 URL 설정 파일의 경로를 지정합니다. (예: `'<project_name>.urls'`) 이는 모든 웹 요청의 라우팅 시작점이 됩니다.
*   **`TEMPLATES`**: 템플릿 엔진 설정입니다. 템플릿 파일의 위치(`DIRS`)나 컨텍스트 프로세서(`context_processors`)와 같은 기본적인 템플릿 관련 설정은 환경에 따라 변하지 않으므로 `base.py`에 포함됩니다.
*   **`AUTH_PASSWORD_VALIDATORS`**: 사용자 비밀번호의 유효성 검사 규칙을 정의합니다. 보안과 관련된 설정이지만, 모든 환경에서 동일한 비밀번호 정책을 유지하는 것이 일반적이므로 `base.py`에 정의합니다.
*   **`LANGUAGE_CODE`, `TIME_ZONE`, `USE_I18N`, `USE_TZ`**: 국제화 및 시간대 관련 설정입니다. 서비스의 기본 언어와 시간대는 환경에 따라 변하지 않으므로 `base.py`에 정의합니다.
*   **`STATIC_URL`, `STATICFILES_DIRS`, `MEDIA_URL`, `MEDIA_ROOT`**: 정적 파일 및 미디어 파일의 URL 접두사, 파일이 위치할 디렉토리 등을 정의합니다. 파일 서빙 방식은 환경에 따라 달라질 수 있지만, 기본적인 URL과 디렉토리 구조는 `base.py`에서 정의하는 것이 일반적입니다.

**`base.py`에서 제외하거나 환경 변수로 처리해야 할 설정:**

*   **`SECRET_KEY`**: 매우 민감한 정보이므로 `base.py`에 직접 하드코딩해서는 안 됩니다. `decouple`과 같은 라이브러리를 사용하여 환경 변수에서 로드하도록 합니다.
*   **`DEBUG`**: 개발 환경에서만 `True`로 설정하고, 운영 환경에서는 반드시 `False`로 설정해야 합니다. `base.py`에서는 기본값 없이 정의만 해두거나, 환경별 파일에서 재정의합니다.
*   **`DATABASES`**: 데이터베이스 연결 정보는 환경별로 달라지므로 `base.py`에서는 기본값 없이 정의만 해두거나, 환경별 파일에서 구체적으로 설정합니다.
*   **`ALLOWED_HOSTS`**: 운영 환경에서 서비스가 허용할 호스트 목록입니다. 개발 환경에서는 비워두거나 `localhost`를 허용하지만, 운영 환경에서는 실제 도메인을 명시해야 합니다.

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

ROOT_URLCONF = '<project_name>.urls' # 프로젝트의 최상위 URL 설정 파일 지정

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')], # 프로젝트 전역 템플릿 디렉토리
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
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
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles') # collectstatic 명령 시 파일이 모이는 곳
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"), # 앱 외에 프로젝트 전역 정적 파일 위치
]

# Media files (User uploaded files)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media') # 사용자 업로드 파일이 저장될 곳
```
**중요:** `BASE_DIR`의 경로가 `settings.py` 파일의 위치 변경에 따라 수정되어야 합니다. `Path(__file__).resolve().parent.parent.parent`와 같이 상위 디렉토리를 올바르게 가리키도록 조정합니다.

---

### 2.3. **3단계: `development.py` - 개발 환경 설정**

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

### 2.4. **4단계: `production.py` - 운영 환경 설정**

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

### 2.5. **5단계: Django에 설정 파일 알려주기**

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

## 3. 주요 설정 상세

이 섹션에서는 `settings/base.py`에 주로 위치하는 주요 설정 변수들의 역할과 실무적인 구성 방법을 상세히 다룹니다.

---

### 3.1. `INSTALLED_APPS`와 `MIDDLEWARE`

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

### 3.2. `TEMPLATES`와 `DATABASES`

**`TEMPLATES` - 템플릿 엔진 설정**

- `'DIRS'`: Django가 앱별 `templates` 디렉토리 외에 추가로 템플릿 파일을 검색할 경로를 지정합니다. 보통 프로젝트 전역에서 사용되는 `base.html`, `navbar.html` 같은 기본 레이아웃 템플릿을 이곳에(`[BASE_DIR / 'templates']`) 보관합니다.
- `'APP_DIRS': True`: `INSTALLED_APPS`에 등록된 각 앱의 `templates` 하위 디렉토리를 자동으로 검색하도록 설정합니다. 이를 통해 각 앱은 자신만의 템플릿을 가질 수 있어 재사용성이 높아집니다.
- `'OPTIONS': {'context_processors': [...]}`: 모든 템플릿에 기본적으로 전달될 컨텍스트 변수를 정의합니다. 예를 들어, `django.contrib.auth.context_processors.auth` 덕분에 모든 템플릿에서 `{{ user }}` 변수를 사용하여 현재 로그인된 사용자에 접근할 수 있습니다.

**`DATABASES` - 데이터베이스 연결 설정**

- **개발과 운영 환경의 DB 통일**: 사소한 기능 차이로 인해 운영 환경에서만 버그가 발생하는 것을 막기 위해, **가급적 개발 환경과 운영 환경의 데이터베이스 엔진을 동일하게(예: 둘 다 PostgreSQL) 사용하는 것을 강력히 권장**합니다. SQLite는 사용이 간편하지만, 데이터 타입이나 제약 조건 등에서 PostgreSQL/MySQL과 달라 예상치 못한 문제를 일으킬 수 있습니다.
- **`CONN_MAX_AGE` (커넥션 재사용)**: 이 값은 데이터베이스 커넥션의 수명을 초 단위로 지정합니다. `0`이 기본값이며, 매 요청마다 새로운 커넥션을 맺습니다. `60`이나 `300` 같은 값을 설정하면, 한번 맺은 커넥션을 지정된 시간 동안 재사용하여 매번 연결을 새로 맺는 오버헤드를 줄여줍니다. 이는 성능 향상에 큰 도움이 되는 간단하면서도 효과적인 설정입니다.
- **전문 커넥션 풀러(Connection Pooler)**: 트래픽이 매우 많은 대규모 서비스에서는 `CONN_MAX_AGE`만으로는 부족할 수 있습니다. 이 경우, Django 애플리케이션과 데이터베이스 서버 사이에 `PgBouncer` (PostgreSQL용) 같은 전문 커넥션 풀러를 별도로 두어 커넥션을 훨씬 효율적으로 관리하는 방법을 사용합니다.

---

### 3.3. 정적 파일(Static) 및 미디어 파일(Media) 설정

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