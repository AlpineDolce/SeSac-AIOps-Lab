<h2>Django Backend: 로깅, 모니터링, 및 헬스 체크</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04 (수정: 2025-08-12)

<h2>문서 목표</h2>
<p>이 문서는 Django 애플리케이션의 로깅 전략, 성능 모니터링 도구, 헬스 체크 및 알림 설정 기법을 학습하여 안정적인 서비스 운영 역량을 강화하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 로깅 설정 (Logging Configuration)](#1-로깅-설정-logging-configuration)
- [2. 성능 모니터링 (Performance Monitoring)](#2-성능-모니터링-performance-monitoring)
- [3. 헬스 체크 (Health Checks)](#3-헬스-체크-health-checks)
- [4. 알림 설정 (Alerting)](#4-알림-설정-alerting)

---

## 1. 로깅 설정 (Logging Configuration)

체계적인 로깅은 애플리케이션의 동작을 추적하고, 에러를 신속하게 진단하며, 성능 문제를 분석하는 데 필수적인 기반입니다. Django는 Python의 표준 `logging` 모듈을 기반으로 강력하고 유연한 로깅 시스템을 제공합니다.

### 1.1. Django 로깅의 구성 요소

Django의 로깅 설정(`LOGGING` 사전)은 네 가지 주요 구성 요소로 이루어집니다.

- **`loggers` (로거):** 로그 메시지를 받는 진입점입니다. 각 로거는 이름(예: `django.request`, `myapp.views`)을 가지며, 계층 구조를 이룹니다. 로거는 메시지의 심각도(level)를 보고 이 메시지를 처리할지 무시할지 결정한 후, 처리하기로 하면 연결된 핸들러로 메시지를 전달합니다.
- **`handlers` (핸들러):** 로거로부터 받은 로그 메시지를 실제로 어디에, 어떻게 처리할지 결정합니다. 예를 들어, 콘솔에 출력(`StreamHandler`), 파일에 저장(`FileHandler`), 또는 관리자에게 이메일을 보낼(`AdminEmailHandler`) 수 있습니다. 핸들러 역시 자체적으로 로그 레벨을 가질 수 있습니다.
- **`formatters` (포매터):** 로그 메시지의 최종 출력 형식을 지정합니다. 타임스탬프, 로거 이름, 로그 레벨, 실제 메시지 등을 원하는 형식으로 조합할 수 있습니다.
- **`filters` (필터):** 로거에서 핸들러로 메시지가 전달될 때, 더 세분화된 조건으로 메시지를 필터링할 수 있습니다. 예를 들어, 특정 요청에서만 발생하는 로그를 기록하거나, 특정 단어가 포함된 로그만 처리할 수 있습니다.

`propagate` 속성은 특정 로거에서 처리된 로그 메시지를 상위(부모) 로거로 전파할지 여부를 결정하는 boolean 값입니다. `False`로 설정하면 해당 로거의 핸들러까지만 메시지가 처리되고 전파가 중단됩니다.

### 1.2. 환경별 로깅 전략

**개발 환경**에서는 즉각적인 피드백이 중요하므로, `DEBUG` 레벨 이상의 모든 로그를 간단한 포맷으로 콘솔에 출력하는 것이 효율적입니다.

**`settings/development.py` 예시:**
```python
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "DEBUG", # 개발 시에는 DEBUG 레벨까지 모두 확인
    },
}
```

**운영 환경**에서는 성능과 비용을 고려하여 `INFO` 레벨 이상의 로그만 기록하고, JSON과 같은 구조화된 형식으로 파일에 저장하여 로그 분석 시스템(Datadog, ELK, Splunk 등)과 연동하는 것이 일반적입니다.

### 1.3. 실용적인 운영 로깅 설정 (심화)

아래는 요청 ID(Request ID)를 모든 로그에 추가하여 특정 요청의 흐름을 추적하기 쉽게 만들고, 구조화된 JSON 포맷을 사용하는 실용적인 운영 환경 로깅 설정입니다.

**1단계: 필요 라이브러리 설치**
```bash
pip install python-json-logger django-log-formatter-request-id
```

**2단계: 미들웨어 추가**
`settings.py`의 `MIDDLEWARE` 목록에 `log_formatter_request_id.middleware.RequestIDMiddleware`를 추가합니다.

**3단계: `settings/production.py` 설정**
```python
# settings/production.py
import os
from .base import BASE_DIR

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    # 필터 정의: 요청 ID를 로그 레코드에 추가
    "filters": {
        "request_id": {
            "()": "log_formatter_request_id.filters.RequestIDFilter"
        }
    },
    # 포매터 정의
    "formatters": {
        # JSON 포매터: 요청 ID(request_id) 필드 추가
        "json_request_id": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s",
        },
        "verbose": {
            "format": "{levelname} {asctime} {module} {message}",
            "style": "{",
        },
    },
    # 핸들러 정의
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
        "file_prod": {
            "level": "INFO",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(BASE_DIR, 'logs/prod.log'),
            "maxBytes": 1024 * 1024 * 5,  # 5MB
            "backupCount": 5,
            "formatter": "json_request_id", # 요청 ID가 포함된 JSON 포매터 사용
            "filters": ["request_id"],      # 요청 ID 필터 적용
        },
        "mail_admins": {
            "level": "ERROR",
            "class": "django.utils.log.AdminEmailHandler",
            "formatter": "verbose",
        },
    },
    # 로거 정의
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
        # 내 앱들을 위한 로거
        "my_apps": {
            "handlers": ["console", "file_prod"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
```

**4단계: 로거 사용**
이제 앱에서는 `my_apps` 로거를 사용하여 로그를 남기면, 모든 로그 메시지에 고유한 요청 ID가 자동으로 포함되어 분산 환경에서도 사용자 요청을 쉽게 추적할 수 있습니다.

```python
# myapp/views.py
import logging

logger = logging.getLogger('my_apps') # settings.py에 정의된 로거 이름

def my_view(request):
    # 이 로그와 아래의 에러 로그는 동일한 request_id를 갖게 됨
    logger.info("My_view processing started.")
    try:
        # ... some logic ...
        result = 1 / 0
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    # ...
```

---

## 2. 성능 모니터링 (Performance Monitoring)

성능 저하는 사용자 경험에 치명적이며, 조용한 장애(silent failure)를 유발할 수 있습니다. 코드 변경이 없더라도 데이터의 양이나 트래픽 패턴 변화만으로 성능 문제가 발생할 수 있으므로, 개발 단계부터 운영까지 지속적인 모니터링이 중요합니다.

### 2.1. 개발 환경에서의 성능 측정

#### 2.1.1. `django-debug-toolbar` 활용 극대화

`django-debug-toolbar`는 단순한 디버깅 도구를 넘어, 성능 병목을 찾는 데 매우 유용한 도구입니다.

- **SQL 패널 활용:**
  - **중복 쿼리(Duplicate Queries) 확인:** 동일한 쿼리가 여러 번 실행되는 것은 N+1 문제의 대표적인 신호입니다. 예를 들어, 게시물 목록을 가져오면서 각 게시물의 작성자 정보를 루프 안에서 조회하면, 게시물 수만큼 사용자 조회 쿼리가 추가로 발생합니다.
  - **N+1 문제 해결:** `select_related` (정방향 ForeignKey)나 `prefetch_related` (역방향 ForeignKey, ManyToMany)를 사용하여 관련된 객체를 한 번의 쿼리로 미리 가져와 N+1 문제를 해결할 수 있습니다.
  - **느린 쿼리(Slow Queries) 식별:** 각 쿼리의 실행 시간을 확인하고, 인덱스가 필요한 쿼리가 있는지 검토합니다.

- **템플릿(Templates) 패널:** 어떤 템플릿이 몇 번 렌더링되고, 컨텍스트에 어떤 데이터가 포함되는지 확인할 수 있어 복잡한 템플릿 상속 구조를 디버깅하는 데 유용합니다.

#### 2.1.2. `connection.queries`를 이용한 수동 쿼리 분석

`DEBUG=True`일 때, Django는 실행된 모든 SQL 쿼리 기록을 `django.db.connection.queries`에 저장합니다. 이를 이용해 특정 코드 블록의 성능을 수동으로 측정할 수 있습니다.

```python
# views.py 또는 test_cases.py
from django.db import connection, reset_queries
import time

def my_complex_view(request):
    reset_queries() # 쿼리 로그 초기화
    start_time = time.time()

    # ... 분석하고 싶은 복잡한 로직 ...
    # 예: posts = Post.objects.all()
    # for post in posts:
    #     print(post.author.name) # N+1 문제 발생 지점
    # ...

    end_time = time.time()
    
    print(f"로직 실행 시간: {end_time - start_time:.2f}초")
    print(f"실행된 쿼리 수: {len(connection.queries)}")
    # 상세 쿼리 내용 출력
    for query in connection.queries:
        print(query['sql'])

    # ...
```

### 2.2. 운영 환경: APM (Application Performance Monitoring)

운영 환경에서는 Sentry, Datadog, New Relic과 같은 APM 도구를 사용하여 성능을 실시간으로 추적하고 병목 현상을 분석합니다. 여기서는 Sentry를 예로 들어 설명합니다.

#### 2.2.1. Sentry 기본 설정

- **설치:** `pip install sentry-sdk[django]`
- **설정 (`settings/production.py`):**
  ```python
  import sentry_sdk
  from sentry_sdk.integrations.django import DjangoIntegration

  sentry_sdk.init(
      dsn="YOUR_SENTRY_DSN",
      integrations=[DjangoIntegration()],
      traces_sample_rate=0.2, # 실제 운영에서는 20% 정도의 트랜잭션만 샘플링하여 성능에 미치는 영향 최소화
      send_default_pii=False,
      environment="production",
      release="my-project@1.0.0", # 배포 시 동적으로 버전 주입
  )
  ```

#### 2.2.2. Sentry 활용도 높이기

- **사용자 정보 추가:** 에러가 발생했을 때 어떤 사용자가 겪은 문제인지 알면 디버깅이 훨씬 쉬워집니다.
  ```python
  import sentry_sdk

  def my_view(request):
      if request.user.is_authenticated:
          sentry_sdk.set_user({"id": request.user.id, "email": request.user.email, "username": request.user.username})
      # ...
  ```

- **커스텀 태그(Tag) 및 컨텍스트(Context) 추가:**
  ```python
  sentry_sdk.set_tag("service.section", "payment")
  sentry_sdk.set_context("order_details", {
      "order_id": 123,
      "amount": 49.99,
  })
  ```

- **커스텀 성능 트랜잭션:** Django 뷰가 아닌, 특정 백그라운드 작업(예: Celery task)의 성능을 측정하고 싶을 때 유용합니다.
  ```python
  import sentry_sdk

  def my_background_task():
      with sentry_sdk.start_transaction(op="task", name="Process Daily Reports"):
          # ... 시간이 오래 걸리는 작업 ...
          pass
  ```
Sentry와 같은 APM 도구를 적극적으로 활용하면, 문제가 발생하기 전에 잠재적인 성능 병목을 식별하고 사용자 경험을 개선하는 데 큰 도움이 됩니다.

---

## 3. 헬스 체크 (Health Checks)

헬스 체크 엔드포인트는 서비스의 현재 상태를 외부에 알리는 통신 규약입니다. 로드 밸런서나 컨테이너 오케스트레이션 시스템(예: Kubernetes)은 이 엔드포인트를 주기적으로 호출하여, 애플리케이션이 정상적으로 요청을 처리할 수 있는 상태인지 확인합니다.

### 3.1. 왜 헬스 체크가 필요한가?

- **로드 밸런싱:** 로드 밸런서는 헬스 체크에 실패한 인스턴스(서버)를 서비스에서 일시적으로 제외하여, 사용자 요청이 실패한 서버로 전달되는 것을 막습니다.
- **무중단 배포:** 쿠버네티스와 같은 환경에서는 새 버전의 앱이 정상적으로 실행되는지 헬스 체크로 확인한 후 트래픽을 새 버전으로 전환합니다. 이를 통해 배포 중단 시간을 없앨 수 있습니다.
- **자동 복구(Auto-healing):** 서비스가 비정상 상태(예: DB 연결 끊김)일 때, 오케스트레이션 시스템이 이를 감지하고 자동으로 해당 인스턴스를 재시작하여 서비스를 복구합니다.

### 3.2. 직접 헬스 체크 엔드포인트 구현하기

서비스가 의존하는 핵심 시스템(DB, 캐시 등)의 상태를 점검하는 엔드포인트를 직접 만들 수 있습니다.

```python
# health_check/views.py
from django.http import JsonResponse
from django.db import connections
from django.db.utils import OperationalError
from django.core.cache import caches, CacheKeyWarning
import warnings

def health_check(request):
    # 서비스 상태를 확인할 항목들을 딕셔너리로 관리
    component_status = {}

    # 1. 데이터베이스 연결 확인
    db_conn = connections['default']
    try:
        db_conn.cursor()
        component_status['database'] = 'ok'
    except OperationalError:
        component_status['database'] = 'error'

    # 2. 캐시(Redis 등) 연결 확인
    cache_conn = caches['default']
    try:
        # CacheKeyWarning을 일시적으로 무시하여, 존재하지 않는 키에 대한 경고를 숨김
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CacheKeyWarning)
            cache_conn.set('health_check', 'ok', timeout=1)
            if cache_conn.get('health_check') == 'ok':
                component_status['cache'] = 'ok'
            else:
                component_status['cache'] = 'error'
    except Exception:
        component_status['cache'] = 'error'

    # 모든 컴포넌트가 'ok' 상태인지 확인
    is_healthy = all(status == 'ok' for status in component_status.values())

    if is_healthy:
        # 모든 것이 정상이면 200 OK 응답
        return JsonResponse(component_status, status=200)
    else:
        # 하나라도 문제가 있으면 503 Service Unavailable 응답
        return JsonResponse(component_status, status=503)

```
이 방식은 간단하지만, 점검할 서비스가 늘어날수록 코드가 복잡해지는 단점이 있습니다.

### 3.3. `django-health-check` 라이브러리 활용

다양한 종류의 서비스를 간편하게 점검하고 확장할 수 있도록 `django-health-check` 라이브러리를 사용하는 것이 좋습니다.

**1단계: 라이브러리 및 플러그인 설치**
```bash
# 기본 라이브러리와 DB, 캐시, 스토리지 확인 플러그인 설치
pip install django-health-check django-health-check[db] django-health-check[cache] django-health-check[storage]
```

**2단계: `settings.py`에 앱 등록**
```python
# settings.py
INSTALLED_APPS = [
    # ...
    'health_check',                             # 기본 앱
    'health_check.db',                          # DB 확인 플러그인
    'health_check.cache',                       # 캐시 확인 플러그인
    'health_check.storage',                     # 스토리지 확인 플러그인
    # 'health_check.contrib.celery',            # Celery 사용 시
    # 'health_check.contrib.redis',             # Redis 사용 시
]
```

**3단계: `urls.py`에 엔드포인트 추가**
```python
# urls.py
from django.urls import path, include

urlpatterns = [
    # ...
    path('health/', include('health_check.urls')),
]
```

이제 `/health/` 엔드포인트로 접속하면, 등록된 모든 플러그인이 각 서비스의 상태를 점검하여 결과를 JSON 형식으로 보여줍니다. 상태가 하나라도 비정상이면 HTTP 503 코드를 반환합니다. 커스텀 점검 로직을 만들어 플러그인으로 쉽게 추가할 수도 있어 확장성이 매우 뛰어납니다.

---

## 4. 알림 설정 (Alerting)

로깅과 모니터링 시스템이 문제를 감지했을 때, 이를 담당자에게 신속하게 전달하는 알림(Alerting) 시스템이 구축되어야 즉각적인 대응이 가능합니다.

### 4.1. Django 기본 기능: `AdminEmailHandler`

가장 간단한 알림 방법은 Django의 `AdminEmailHandler`를 사용하는 것입니다. `ERROR` 레벨 이상의 로그가 발생하면 `settings.py`의 `ADMINS` 목록에 있는 모든 이메일 주소로 에러 리포트를 보냅니다.

- **장점:** 설정이 매우 간단합니다.
- **단점:** 모든 에러에 대해 이메일이 발송되어 '알림 피로(Alert Fatigue)'가 발생하기 쉽고, 긴급도를 구분하기 어렵습니다.

```python
# settings.py
ADMINS = [('관리자 이름', 'admin@example.com')]
MANAGERS = ADMINS

# 실제 운영 환경에서는 SendGrid, AWS SES 같은 트랜잭셔널 이메일 서비스 사용을 권장
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_HOST_USER = 'your_email@gmail.com'
EMAIL_HOST_PASSWORD = 'your_app_password' # 2단계 인증 사용 시 앱 비밀번호
EMAIL_USE_TLS = True
SERVER_EMAIL = 'your_email@gmail.com' # 발신자 이메일
```
`LOGGING` 설정의 `handlers`에 `mail_admins`가 포함되어 있어야 동작합니다.

### 4.2. APM을 이용한 지능형 알림 (Sentry)

Sentry와 같은 APM 도구는 훨씬 더 정교한 알림 기능을 제공합니다.

- **알림 규칙(Alert Rules) 설정:** Sentry 대시보드에서 특정 조건에만 알림을 보내도록 규칙을 만들 수 있습니다.
  - **예시 1:** "결제(payment) 관련 에러가 **새롭게** 발생하면 **즉시** '결제팀' Slack 채널과 PagerDuty로 알림"
  - **예시 2:** "특정 에러가 **5분 동안 100번 이상** 발생하면 '개발팀' Slack 채널로 알림"
  - **예시 3:** "전체 에러 발생률이 평소보다 **300% 이상 증가**하면 '전체' 채널로 알림"
- **워크플로우 연동:** Sentry에서 발생한 이슈를 Jira, Asana 등의 프로젝트 관리 도구와 연동하여 자동으로 티켓을 생성할 수 있습니다.

이러한 지능형 알림을 통해 정말 중요한 문제에만 집중하고, 불필요한 알림은 줄일 수 있습니다.

### 4.3. 커스텀 로깅 핸들러 작성 (Slack 연동)

팀에서 사용하는 메신저(예: Slack)로 직접 알림을 보내고 싶다면, 커스텀 로깅 핸들러를 직접 만들 수 있습니다.

**1단계: `requests` 라이브러리 설치**
```bash
pip install requests
```

**2단계: 커스텀 핸들러 작성**
프로젝트 내의 적절한 위치(예: `core/log_handlers.py`)에 핸들러 클래스를 작성합니다.

```python
# core/log_handlers.py
import requests
import logging

class SlackWebhookHandler(logging.Handler):
    def __init__(self, webhook_url):
        super().__init__()
        self.webhook_url = webhook_url

    def emit(self, record):
        # format() 메서드를 호출하여 로그 메시지를 포매팅
        log_entry = self.format(record)
        
        # Slack 메시지 형식에 맞게 payload 구성
        payload = {
            "text": f"🚨 Django Critical Error! 🚨\n```{log_entry}```",
            "username": "Django Bot",
            "icon_emoji": ":warning:"
        }
        
        try:
            requests.post(self.webhook_url, json=payload, timeout=5)
        except requests.exceptions.RequestException:
            # Slack 전송 실패 시 에러를 출력 (무한 루프 방지를 위해 여기서 다시 로깅하지 않음)
            pass
```

**3단계: `settings.py`에 핸들러 추가**
`LOGGING` 설정에 방금 만든 커스텀 핸들러를 추가합니다. Slack Webhook URL은 환경 변수로 관리하는 것이 안전합니다.

```python
# settings.py
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {pathname}:{lineno} {message}",
            "style": "{",
        },
    },
    "handlers": {
        # ... 기존 핸들러 ...
        "slack_critical": {
            "level": "CRITICAL", # CRITICAL 레벨의 로그만 이 핸들러로 보냄
            "class": "core.log_handlers.SlackWebhookHandler",
            "webhook_url": os.environ.get("SLACK_WEBHOOK_URL"),
            "formatter": "verbose",
        },
    },
    "loggers": {
        "my_apps": {
            "handlers": ["console", "file_prod", "slack_critical"], # 핸들러 목록에 추가
            "level": "INFO",
            "propagate": False,
        },
        # ...
    },
}
```
이제 코드에서 `CRITICAL` 레벨로 로그를 남기면 지정된 Slack 채널로 즉시 알림이 전송됩니다.

```python
# myapp/views.py
import logging
logger = logging.getLogger('my_apps')

def some_critical_process():
    logger.critical("This is a critical error that requires immediate attention!")
```
