<h2>Django Backend: API 보안 강화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django REST API의 보안을 강화하기 위한 기법들을 학습하는 것을 목표로 합니다. API Rate Limiting, CORS 설정, 민감 데이터 처리 등을 다룹니다.</p>

<h2>목차</h2>

- [1. 민감 정보 관리 (환경 변수)](#1-민감-정보-관리-환경-변수)
    - [1.1. **작동 원리 및 라이브러리 선택**](#11-작동-원리-및-라이브러리-선택)
    - [1.2. **단계별 적용 가이드**](#12-단계별-적용-가이드)
- [2. CORS (Cross-Origin Resource Sharing) 설정](#2-cors-cross-origin-resource-sharing-설정)
  - [2.1. Preflight Request (사전 요청)](#21-preflight-request-사전-요청)
  - [2.2. `django-cors-headers`를 이용한 설정](#22-django-cors-headers를-이용한-설정)
- [3. API Rate Limiting (요청 횟수 제한)](#3-api-rate-limiting-요청-횟수-제한)
  - [3.1. 작동 방식 및 캐시 설정](#31-작동-방식-및-캐시-설정)
  - [3.2. Throttling 설정 방법](#32-throttling-설정-방법)
    - [3.2.1. 기본 Throttling 클래스](#321-기본-throttling-클래스)
    - [3.2.2. `ScopedRateThrottle`을 이용한 세분화된 제어](#322-scopedratethrottle을-이용한-세분화된-제어)
  - [3.3. Throttling 응답](#33-throttling-응답)
- [4. 인증, 인가 및 추가 보안 강화](#4-인증-인가-및-추가-보안-강화)
  - [4.1. 인증 (Authentication): API 접근 자격 증명](#41-인증-authentication-api-접근-자격-증명)
    - [4.1.1. DRF 기본 토큰 인증 (`TokenAuthentication`)](#411-drf-기본-토큰-인증-tokenauthentication)
    - [4.1.2. JWT (JSON Web Token) 인증](#412-jwt-json-web-token-인증)
  - [4.2. 인가 (Authorization): 리소스 접근 제어](#42-인가-authorization-리소스-접근-제어)
    - [4.2.1. DRF 기본 권한 클래스](#421-drf-기본-권한-클래스)
    - [4.2.2. 커스텀 권한 (객체 수준 권한)](#422-커스텀-권한-객체-수준-권한)
  - [4.3. HTTPS 강제 적용](#43-https-강제-적용)
  - [4.4. 입력 값 유효성 검사 (Input Validation)](#44-입력-값-유효성-검사-input-validation)
  - [4.5. 보안 관련 HTTP 헤더 설정](#45-보안-관련-http-헤더-설정)
  - [4.6. 오류 메시지 최소화 및 로깅](#46-오류-메시지-최소화-및-로깅)
  - [4.7. 정기적인 보안 감사 및 업데이트](#47-정기적인-보안-감사-및-업데이트)
- [5. CSRF (Cross-Site Request Forgery) 보호](#5-csrf-cross-site-request-forgery-보호)
  - [5.1. Django의 CSRF 보호 메커니즘 개요](#51-django의-csrf-보호-메커니즘-개요)
  - [5.2. `CsrfViewMiddleware`의 역할](#52-csrfviewmiddleware의-역할)
    - [`settings.py`의 `MIDDLEWARE` 설정](#settingspy의-middleware-설정)
  - [5.3. 템플릿에서 `{% csrf_token %}` 태그 사용](#53-템플릿에서--csrf_token--태그-사용)
  - [5.4. AJAX 요청에서 CSRF 처리](#54-ajax-요청에서-csrf-처리)
    - [jQuery를 사용한 예시](#jquery를-사용한-예시)
    - [Fetch API를 사용한 예시](#fetch-api를-사용한-예시)
  - [5.5. CSRF 보호 비활성화 (경고: 권장하지 않음)](#55-csrf-보호-비활성화-경고-권장하지-않음)
    - [데코레이터 사용](#데코레이터-사용)
    - [주의사항](#주의사항)

---

## 1. 민감 정보 관리 (환경 변수)

소스 코드에 민감 정보를 직접 작성하는 것(하드코딩)은 심각한 보안 사고로 이어질 수 있습니다. 예를 들어, 실수로 `SECRET_KEY`나 데이터베이스 암호가 포함된 코드를 공용 GitHub 저장소에 푸시하면, 악의적인 봇이 수 분 내에 이를 탈취하여 서버를 공격하거나 데이터를 유출할 수 있습니다.

**민감 정보란?**
- `SECRET_KEY`
- `DEBUG` 상태 값 (운영에서 `True`로 노출되면 안 됨)
- 데이터베이스 연결 정보 (사용자 이름, 암호, 호스트 주소)
- 이메일 서버 접속 정보
- 외부 API 키 (예: AWS, Google Cloud, 결제 모듈 키)

이러한 정보들은 코드와 완전히 분리하여 **환경 변수(Environment Variables)**를 통해 관리하는 것이 필수적입니다.

---

#### 1.1. **작동 원리 및 라이브러리 선택**

환경 변수는 코드가 실행되는 셸(운영체제)에 설정된 변수입니다. Django는 `os.environ`을 통해 이 변수들을 읽을 수 있습니다.

하지만 매번 `os.environ.get('VAR_NAME')`을 사용하는 것은 다소 번거롭고, 특히 로컬 개발 환경에서 변수를 설정하기가 불편합니다. 이를 해결하기 위해 `python-decouple`이나 `django-environ` 같은 라이브러리를 사용합니다.

- **`python-decouple`의 장점**: 가볍고, Django에 종속적이지 않아 범용적으로 사용할 수 있습니다. `config()` 함수 하나로 환경 변수와 `.env` 파일을 모두 처리해주는 편리함을 제공합니다.

`config()` 함수의 동작 순서는 다음과 같습니다.
1.  먼저 코드 실행 환경(운영체제)에 해당 이름의 환경 변수가 있는지 확인합니다.
2.  없다면, 프로젝트 루트의 `.env` 파일에서 해당 키를 찾습니다.
3.  두 군데 모두 없다면 `UndefinedValueError`를 발생시킵니다. (단, `default` 값이 지정된 경우는 예외)

이러한 원리 덕분에, 개발 환경에서는 `.env` 파일로 편하게 작업하고, 운영 환경에서는 서버에 직접 설정된 환경 변수를 코드가 자동으로 읽게 할 수 있습니다.

---

#### 1.2. **단계별 적용 가이드**

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

---

## 2. CORS (Cross-Origin Resource Sharing) 설정

웹 브라우저는 보안을 위해 **동일 출처 정책(Same-Origin Policy, SOP)**을 따릅니다. 이 정책은 특정 출처(Origin: Protocol + Host + Port)에서 로드된 문서나 스크립트가 다른 출처의 리소스와 상호작용하는 것을 제한합니다. 예를 들어, `https://my-frontend.com`에서 실행되는 JavaScript 코드는 `https://api.my-backend.com`에 직접 API를 요청할 수 없습니다.

**CORS(교차 출처 리소스 공유)**는 이 정책에 대한 예외를 허용하는 메커니즘입니다. 서버는 특정 출처의 외부 요청을 허용하겠다는 응답 헤더를 보내고, 브라우저는 이 헤더를 확인하여 안전하다고 판단되면 API 요청을 허용합니다. 프론트엔드와 백엔드 API가 다른 도메인(또는 포트)에서 실행되는 현대적인 웹 애플리케이션에서는 CORS 설정이 필수적입니다.

### 2.1. Preflight Request (사전 요청)

`GET`, `HEAD`, `POST` 같은 단순 요청(Simple Request)이 아닌, `PUT`, `DELETE`나 `Authorization` 같은 특정 헤더를 포함하는 요청을 보낼 때, 브라우저는 본 요청을 보내기 전에 먼저 `OPTIONS` 메서드를 사용해 **사전 요청(Preflight Request)**을 보냅니다.

이 사전 요청을 통해 브라우저는 서버가 앞으로 보낼 본 요청(메서드, 헤더 등)을 허용하는지 미리 확인합니다. 서버가 이 `OPTIONS` 요청에 대해 유효한 CORS 헤더로 응답해야만 브라우저는 본 요청을 보냅니다. `django-cors-headers` 라이브러리는 이 과정을 자동으로 처리해줍니다.

### 2.2. `django-cors-headers`를 이용한 설정

Django에서 CORS를 가장 쉽게 설정하는 방법은 `django-cors-headers` 라이브러리를 사용하는 것입니다.

**1단계: 라이브러리 설치**
```bash
pip install django-cors-headers
```

**2단계: `settings.py`에 앱 및 미들웨어 등록**
```python
# settings.py
INSTALLED_APPS = [
    # ...
    'corsheaders', # 앱 등록
    # ...
]

MIDDLEWARE = [
    # CorsMiddleware는 응답을 수정할 수 있는 다른 미들웨어보다 먼저,
    # 특히 SecurityMiddleware, CommonMiddleware보다 위에 위치하는 것이 좋습니다.
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # ...
]
```

**3단계: 허용할 출처(Origin) 및 정책 설정**
`settings.py`에 CORS 관련 설정을 추가합니다. **운영 환경에서는 `CORS_ALLOW_ALL_ORIGINS = True`를 절대 사용해서는 안 됩니다.**

```python
# settings.py

# 1. 특정 출처만 허용 (가장 안전한 방법)
CORS_ALLOWED_ORIGINS = [
    "https://your-frontend-domain.com",
    "http://localhost:3000",       # React 개발 서버
    "http://127.0.0.1:3000",      # React 개발 서버
    "http://localhost:8080",       # Vue 개발 서버
    "http://127.0.0.1:8080",      # Vue 개발 서버
]

# 2. 정규표현식을 이용한 출처 허용
# CORS_ALLOWED_ORIGIN_REGEXES = [
#     r"^https://\w+\.your-domain\.com$",
# ]

# 3. 모든 출처 허용 (⚠️ 개발 환경에서만 사용)
# CORS_ALLOW_ALL_ORIGINS = True

# --- 추가 설정 ---

# 허용할 HTTP 메서드 지정
CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

# Preflight 요청에서 허용할 HTTP 헤더 지정
# 클라이언트에서 커스텀 헤더(예: 'X-Custom-Header')를 보낸다면 여기에 추가해야 합니다.
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization', # 토큰 인증을 위해 필수
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# 쿠키 등 자격 증명(Credentials)을 포함한 요청 허용
# 이 설정이 True이면, 프론트엔드에서도 withCredentials: true 옵션을 주어야 합니다.
# CORS_ALLOWED_ORIGINS에 특정 출처가 명시되어야 하며, CORS_ALLOW_ALL_ORIGINS = True와 함께 사용할 수 없습니다.
CORS_ALLOW_CREDENTIALS = True
```

`CORS_ALLOW_CREDENTIALS = True`는 프론트엔드에서 쿠키나 `Authorization` 헤더를 API 요청에 담아 보낼 때 필요한 중요한 설정입니다. 대부분의 인증 기반 API는 이 설정이 필요합니다.

---

## 3. API Rate Limiting (요청 횟수 제한)

API Rate Limiting은 특정 클라이언트가 단위 시간당 API를 호출할 수 있는 횟수를 제한하는 기술입니다. 이는 다음과 같은 목적으로 필수적입니다.

- **서비스 안정성:** 특정 사용자의 과도한 요청으로 인한 서버 과부하를 방지합니다.
- **보안 강화:** 무차별 대입 공격(Brute-force)이나 서비스 거부(DDoS) 공격의 영향을 완화합니다.
- **공정한 사용:** 모든 사용자에게 공평한 리소스 사용 기회를 보장합니다.

Django REST Framework(DRF)는 유연하고 강력한 Throttling(요청 횟수 제한) 시스템을 제공합니다.

### 3.1. 작동 방식 및 캐시 설정

DRF의 Throttling은 **캐시(Cache)**를 사용하여 작동합니다. 각 요청의 타임스탬프를 캐시에 기록하고, 다음 요청이 왔을 때 이전 요청 기록을 확인하여 제한을 초과했는지 검사합니다.

따라서, 프로덕션 환경에서 여러 서버나 프로세스를 운영하는 경우, **반드시 Redis나 Memcached 같은 공유 캐시를 설정해야 합니다.** Django의 기본 로컬 메모리 캐시(`LocMemCache`)는 각 프로세스별로 독립적이므로 여러 서버 환경에서는 요청 제한이 정확하게 동작하지 않습니다.

### 3.2. Throttling 설정 방법

#### 3.2.1. 기본 Throttling 클래스

DRF는 두 가지 기본 클래스를 제공합니다.

- `AnonRateThrottle`: 인증되지 않은 사용자(IP 주소 기준)의 요청 횟수를 제한합니다.
- `UserRateThrottle`: 인증된 사용자(User ID 기준)의 요청 횟수를 제한합니다.

`settings.py`에서 전역으로 설정할 수 있습니다.

```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',  # 비인증 사용자는 시간당 100회
        'user': '1000/hour'  # 인증 사용자는 시간당 1000회
    }
}
# 사용 가능한 단위: second, minute, hour, day
```

#### 3.2.2. `ScopedRateThrottle`을 이용한 세분화된 제어

실제 애플리케이션에서는 API의 종류에 따라 다른 제한을 두고 싶을 때가 많습니다. 예를 들어, 로그인 API는 더 엄격하게, 일반적인 데이터 조회 API는 더 관대하게 설정할 수 있습니다. 이때 `ScopedRateThrottle`을 사용합니다.

**1단계: `settings.py`에 Scope 정의**
`DEFAULT_THROTTLE_RATES`에 커스텀 스코프(scope)를 정의합니다.

```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.ScopedRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour',
        # 커스텀 스코프 정의
        'login': '5/minute',         # 로그인 시도: 분당 5회
        'uploads': '20/day',         # 파일 업로드: 하루 20회
        'password_reset': '3/hour',  # 비밀번호 찾기: 시간당 3회
    }
}
```

**2단계: 각 View에 Scope 적용**
`throttle_classes`와 함께 `throttle_scope` 속성을 뷰에 지정합니다.

```python
# views.py
from rest_framework.views import APIView
from rest_framework.throttling import ScopedRateThrottle

class LoginView(APIView):
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'login' # 'login' 스코프의 제한(5/minute)을 적용

    def post(self, request, *args, **kwargs):
        # ... 로그인 로직 ...
        pass

class FileUploadView(APIView):
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'uploads' # 'uploads' 스코프의 제한(20/day)을 적용

    def post(self, request, *args, **kwargs):
        # ... 파일 업로드 로직 ...
        pass
```
이렇게 하면 각 API의 중요도와 특성에 맞게 요청 횟수를 유연하게 제어할 수 있습니다.

### 3.3. Throttling 응답

요청이 제한 횟수를 초과하면, DRF는 HTTP 상태 코드 `429 Too Many Requests`와 함께 다음과 같은 응답 본문을 반환합니다.

```json
{
    "detail": "Request was throttled. Expected available in 42 seconds."
}
```
또한, 응답 헤더에 `Retry-After` (재시도까지 남은 시간(초)) 정보를 포함하여 클라이언트가 다음 요청을 언제 보내야 할지 알 수 있도록 돕습니다.

---

## 4. 인증, 인가 및 추가 보안 강화

API 보안을 완성하려면 강력한 인증/인가 체계를 갖추고, 다양한 웹 취약점에 대비해야 합니다. 이 섹션에서는 각 주제를 더 깊이 있게 다룹니다.

### 4.1. 인증 (Authentication): API 접근 자격 증명

API 요청자가 누구인지 식별하는 과정입니다. SPA나 모바일 앱과 통신하는 API는 주로 상태가 없는(stateless) **토큰 기반 인증**을 사용합니다.

#### 4.1.1. DRF 기본 토큰 인증 (`TokenAuthentication`)

DRF는 간단한 토큰 인증 시스템을 내장하고 있습니다.

**1. 설정:**
- `settings.py`의 `INSTALLED_APPS`에 `'rest_framework.authtoken'`을 추가합니다.
- `python manage.py migrate`를 실행하여 토큰 모델을 데이터베이스에 생성합니다.
- `settings.py`의 `REST_FRAMEWORK`에 기본 인증 클래스를 설정합니다.
  ```python
  REST_FRAMEWORK = {
      'DEFAULT_AUTHENTICATION_CLASSES': [
          'rest_framework.authentication.TokenAuthentication',
      ],
      # ...
  }
  ```

**2. 토큰 발급 엔드포인트 생성:**
사용자는 아이디와 비밀번호를 보내 토큰을 발급받아야 합니다. DRF가 제공하는 `obtain_auth_token` 뷰를 사용하면 편리합니다.

```python
# urls.py
from rest_framework.authtoken.views import obtain_auth_token

urlpatterns = [
    # ...
    path('api/token/', obtain_auth_token, name='api_token_auth'),
]
```
이제 클라이언트는 `/api/token/`으로 `username`과 `password`를 POST 요청으로 보내 토큰을 얻을 수 있습니다.

**3. 클라이언트 사용법:**
발급받은 토큰은 모든 후속 API 요청의 HTTP 헤더에 포함시켜야 합니다.
`Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b`

#### 4.1.2. JWT (JSON Web Token) 인증

JWT는 토큰 자체에 사용자 정보(claims)를 담을 수 있고, 토큰 만료 및 갱신 메커니즘을 제공하여 확장성이 더 뛰어납니다. `djangorestframework-simplejwt` 라이브러리가 표준처럼 사용됩니다.

**1. 설치:** `pip install djangorestframework-simplejwt`

**2. 설정:**
- `settings.py`의 `INSTALLED_APPS`에 `'rest_framework_simplejwt'`를 추가합니다.
- `REST_FRAMEWORK`의 기본 인증 클래스를 JWT용으로 변경합니다.
  ```python
  REST_FRAMEWORK = {
      'DEFAULT_AUTHENTICATION_CLASSES': [
          'rest_framework_simplejwt.authentication.JWTAuthentication',
      ],
      # ...
  }
  ```
- `urls.py`에 토큰 발급(access) 및 갱신(refresh) 엔드포인트를 추가합니다.
  ```python
  # urls.py
  from rest_framework_simplejwt.views import (
      TokenObtainPairView,
      TokenRefreshView,
  )

  urlpatterns = [
      # ...
      path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
      path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
  ]
  ```
JWT는 Access Token(수명이 짧음)과 Refresh Token(수명이 긺)을 함께 사용하여 보안을 강화합니다.

### 4.2. 인가 (Authorization): 리소스 접근 제어

인증된 사용자가 특정 리소스에 접근하거나 특정 동작을 수행할 권한이 있는지 확인하는 과정입니다.

#### 4.2.1. DRF 기본 권한 클래스

- `IsAuthenticated`: 인증된 사용자만 접근 가능.
- `IsAdminUser`: 관리자(`is_staff=True`)만 접근 가능.
- `IsAuthenticatedOrReadOnly`: 비인증 사용자는 읽기만, 인증 사용자는 모든 작업 가능.

뷰별로 `permission_classes` 속성에 리스트 형태로 지정합니다.

#### 4.2.2. 커스텀 권한 (객체 수준 권한)

게시물의 작성자만 수정/삭제할 수 있도록 하는 권한을 직접 만들 수 있습니다.

**1. `permissions.py` 작성:**
```python
# permissions.py
from rest_framework import permissions

class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    객체의 소유자만 쓰기 권한을 부여하고, 나머지는 읽기만 허용합니다.
    """
    def has_object_permission(self, request, view, obj):
        # 읽기 요청(GET, HEAD, OPTIONS)은 누구나 허용
        if request.method in permissions.SAFE_METHODS:
            return True

        # 쓰기 요청은 게시물(obj)의 user 필드가 요청을 보낸 사용자(request.user)와
        # 동일한 경우에만 허용합니다.
        return obj.user == request.user
```

**2. `views.py`에 적용:**
DRF의 제네릭 뷰(`RetrieveUpdateDestroyAPIView` 등)는 내부적으로 `has_object_permission`을 호출하여 권한을 검사합니다.

```python
# views.py
from rest_framework import generics
from .models import Post
from .serializers import PostSerializer
from .permissions import IsOwnerOrReadOnly

class PostDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly] # 인증된 사용자이면서, 소유자인 경우에만 쓰기 가능
```
`permission_classes`에 여러 권한을 지정하면 모든 권한을 통과해야 접근이 허용됩니다.

### 4.3. HTTPS 강제 적용

운영 환경에서는 모든 API 통신을 암호화해야 합니다. 리버스 프록시(Nginx 등) 뒤에서 Django를 실행하는 경우를 가정합니다.

```python
# settings/production.py
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000 # 1년
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
```

### 4.4. 입력 값 유효성 검사 (Input Validation)

DRF Serializer는 유효성 검사를 위한 강력한 도구입니다. 필드 수준 검증 외에 여러 필드를 함께 검증할 수도 있습니다.

```python
# serializers.py
from rest_framework import serializers

class EventSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    start_date = serializers.DateField()
    end_date = serializers.DateField()

    # 여러 필드를 함께 검증
    def validate(self, data):
        """
        종료일이 시작일보다 빠른 경우 에러를 발생시킵니다.
        """
        if data['start_date'] > data['end_date']:
            raise serializers.ValidationError("종료일은 시작일보다 빠를 수 없습니다.")
        return data
```

### 4.5. 보안 관련 HTTP 헤더 설정

Django의 `SecurityMiddleware`를 통해 브라우저의 보안 기능을 강화하는 HTTP 헤더를 설정합니다.

```python
# settings.py
SECURE_CONTENT_TYPE_NOSNIFF = True # MIME-sniffing 공격 방지
X_FRAME_OPTIONS = 'DENY' # 클릭재킹(Clickjacking) 방지
```

### 4.6. 오류 메시지 최소화 및 로깅

운영 환경(`DEBUG=False`)에서는 상세한 오류 정보를 사용자에게 노출하지 않아야 합니다. Django는 자동으로 일반 오류 페이지를 보여주며, 실제 오류 내용은 [로깅 설정](path/to/logging/doc)에 따라 파일이나 모니터링 시스템으로 보내야 합니다.

### 4.7. 정기적인 보안 감사 및 업데이트

사용 중인 라이브러리의 보안 취약점은 프로젝트 전체의 보안을 위협합니다. `pip-audit` 같은 도구를 사용하여 알려진 보안 취약점이 있는지 정기적으로 점검해야 합니다.

```bash
pip install pip-audit
pip-audit
```

---

## 5. CSRF (Cross-Site Request Forgery) 보호

CSRF(Cross-Site Request Forgery)는 사용자가 의도하지 않은 요청을 보내도록 유도하여 웹 애플리케이션의 취약점을 악용하는 공격 기법입니다. 공격자는 사용자가 로그인된 상태에서 악성 웹사이트를 방문하도록 유도하고, 해당 웹사이트에서 사용자의 세션을 이용해 원본 웹사이트에 요청을 보냅니다. 이로 인해 비밀번호 변경, 송금, 게시글 삭제 등 사용자가 원치 않는 작업이 수행될 수 있습니다.

Django는 이러한 CSRF 공격으로부터 애플리케이션을 보호하기 위한 강력한 내장 메커니즘을 제공합니다.

### 5.1. Django의 CSRF 보호 메커니즘 개요

Django의 CSRF 보호는 주로 두 가지 구성 요소로 작동합니다.

1.  **`CsrfViewMiddleware`**: `settings.py`의 `MIDDLEWARE`에 포함되어 모든 POST 요청에 대해 CSRF 토큰을 검증합니다.
2.  **`{% csrf_token %}` 템플릿 태그**: HTML 폼에 숨겨진 입력 필드를 추가하여 CSRF 토큰을 포함시킵니다.

이 두 가지가 함께 작동하여, Django는 서버로 들어오는 모든 POST 요청이 해당 웹사이트에서 시작되었음을 확인합니다.

### 5.2. `CsrfViewMiddleware`의 역할

`CsrfViewMiddleware`는 Django의 CSRF 보호의 핵심입니다. 이 미들웨어는 다음 두 가지 주요 작업을 수행합니다.

*   **응답에 토큰 추가**: GET 요청에 대한 응답 시, 쿠키에 CSRF 토큰을 설정하고, `{% csrf_token %}` 태그가 사용된 폼에는 숨겨진 입력 필드로 동일한 토큰을 삽입합니다.
*   **요청 검증**: POST 요청이 들어올 때, 요청 헤더나 폼 데이터에서 CSRF 토큰을 찾아 쿠키의 토큰과 비교합니다. 두 토큰이 일치하지 않거나 토큰이 없으면 `403 Forbidden` 응답을 반환하여 요청을 거부합니다.

#### `settings.py`의 `MIDDLEWARE` 설정

`CsrfViewMiddleware`는 기본적으로 `settings.py`의 `MIDDLEWARE` 리스트에 포함되어 있습니다. 이 미들웨어의 순서는 중요합니다.

```python
# myproject/settings.py

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware', # CSRF 미들웨어
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
```

*   **위치**: `SessionMiddleware` 다음에 위치하는 것이 일반적입니다. `SessionMiddleware`가 요청에 세션 정보를 추가해야 `CsrfViewMiddleware`가 세션 기반의 CSRF 토큰을 생성하고 검증할 수 있기 때문입니다.

### 5.3. 템플릿에서 `{% csrf_token %}` 태그 사용

모든 POST 요청을 보내는 HTML 폼에는 반드시 `{% csrf_token %}` 템플릿 태그를 포함해야 합니다. 이 태그는 숨겨진 `<input>` 필드를 생성하여 현재 세션의 CSRF 토큰을 폼 데이터에 포함시킵니다.

```html
<form method="post" action="/my-action/">
    {% csrf_token %}
    <input type="text" name="data">
    <button type="submit">제출</button>
</form>
```

### 5.4. AJAX 요청에서 CSRF 처리

AJAX(Asynchronous JavaScript and XML)를 사용하여 POST, PUT, DELETE 요청을 보낼 때는 `{% csrf_token %}` 태그를 직접 사용할 수 없습니다. 대신, CSRF 토큰을 JavaScript에서 가져와 요청 헤더에 포함시켜야 합니다.

Django는 CSRF 토큰을 `csrftoken`이라는 이름의 쿠키에 저장합니다. JavaScript에서 이 쿠키 값을 읽어와 `X-CSRFToken` 헤더에 포함시켜 요청을 보냅니다.

#### jQuery를 사용한 예시

```javascript
// CSRF 토큰을 가져오는 함수
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const csrftoken = getCookie('csrftoken');

// 모든 AJAX 요청에 CSRF 토큰 헤더를 자동으로 추가 (jQuery)
$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!/^(GET|HEAD|OPTIONS|TRACE)$/.test(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});

// 예시: AJAX POST 요청
$('#my-form').submit(function(e) {
    e.preventDefault();
    $.ajax({
        url: '/api/some-endpoint/',
        type: 'POST',
        data: $(this).serialize(),
        success: function(response) {
            console.log('Success:', response);
        },
        error: function(xhr, status, error) {
            console.error('Error:', error);
        }
    });
});
```

#### Fetch API를 사용한 예시

```javascript
// CSRF 토큰을 가져오는 함수 (위와 동일)
function getCookie(name) { /* ... */ }
const csrftoken = getCookie('csrftoken');

// 예시: Fetch API POST 요청
fetch('/api/some-endpoint/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken,
    },
    body: JSON.stringify({ key: 'value' }),
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

### 5.5. CSRF 보호 비활성화 (경고: 권장하지 않음)

특정 뷰에 대해 CSRF 보호를 일시적으로 비활성화해야 하는 경우가 있을 수 있습니다 (예: 외부 서비스로부터의 웹훅(webhook) 수신). 하지만 이는 보안 위험을 증가시키므로 **매우 신중하게 사용해야 하며, 가능한 한 피하는 것이 좋습니다.**

#### 데코레이터 사용

`@csrf_exempt` 데코레이터를 사용하여 특정 뷰에 대한 CSRF 검증을 건너뛸 수 있습니다.

```python
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse

@csrf_exempt
def my_webhook_view(request):
    if request.method == 'POST':
        # 웹훅 데이터 처리
        return HttpResponse("Webhook received!")
    return HttpResponse("Only POST requests allowed.")
```

#### 주의사항

*   `@csrf_exempt`는 해당 뷰에 대한 CSRF 보호를 완전히 비활성화합니다. 이 뷰가 민감한 작업을 수행하거나 사용자 세션에 의존하는 경우 심각한 보안 취약점이 될 수 있습니다.
*   가능하다면 `csrf_exempt` 대신 `csrf_protect`를 사용하여 특정 뷰에만 CSRF 보호를 적용하거나, 웹훅의 경우 서명 검증 등 다른 보안 메커니즘을 사용하는 것을 고려해야 합니다.