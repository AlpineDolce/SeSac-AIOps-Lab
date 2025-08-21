<h2>Django Backend: 배포</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django 애플리케이션을 실제 운영 환경에 배포하는 과정과, 효율적인 설정 관리, 배포 자동화(CI/CD)에 대해 다루는 것을 목표로 합니다. 이를 통해 안정적이고 확장 가능한 Django 서비스를 구축하는 능력을 기릅니다.</p>

<h2>목차</h2> 

- [1. 배포 (Deployment)](#1-배포-deployment)
  - [1.1. 정적 파일 수집 (collectstatic): 운영 환경을 위한 준비](#11-정적-파일-수집-collectstatic-운영-환경을-위한-준비)
  - [1.2. WSGI/ASGI 서버: Django 애플리케이션 실행](#12-wsgiasgi-서버-django-애플리케이션-실행)
  - [1.3. 웹 서버 (Nginx, Apache)](#13-웹-서버-nginx-apache)
    - [1.3.1. Nginx 설정 예시 (리버스 프록시 및 정적 파일 서빙)](#131-nginx-설정-예시-리버스-프록시-및-정적-파일-서빙)
    - [1.3.2. Gunicorn Systemd 서비스 파일 예시](#132-gunicorn-systemd-서비스-파일-예시)
  - [1.4. 환경 변수 관리 (프로덕션)](#14-환경-변수-관리-프로덕션)
    - [1.4.1. 프로덕션 환경에서의 환경 변수 관리](#141-프로덕션-환경에서의-환경-변수-관리)
- [2. 설정 파일 분리 (Settings Management)](#2-설정-파일-분리-settings-management)
- [3. 배포 자동화 (CI/CD)](#3-배포-자동화-cicd)
  - [3.1. CI (Continuous Integration)](#31-ci-continuous-integration)
  - [3.2. CD (Continuous Deployment/Delivery)](#32-cd-continuous-deploymentdelivery)
  - [3.3. 학습 내용](#33-학습-내용)


## 1. 배포 (Deployment)

Django 애플리케이션을 개발 환경에서 벗어나 실제 사용자에게 서비스하려면 배포 과정이 필요합니다.

### 1.1. 정적 파일 수집 (collectstatic): 운영 환경을 위한 준비

운영 환경(`DEBUG=False`)에서는 Django 개발 서버가 정적 파일을 직접 제공하지 않습니다. 따라서 배포 전에 모든 정적 파일을 한 곳으로 모으는 과정이 필수적입니다. 이 과정은 웹 서버(Nginx, Apache)나 CDN(Content Delivery Network)이 정적 파일을 효율적으로 서빙할 수 있도록 준비하는 단계입니다.

-   **`STATIC_ROOT`**: `settings.py`에 정의하는 이 경로는 `collectstatic` 명령을 실행했을 때, 프로젝트 전체에 흩어져 있는 모든 정적 파일들이 최종적으로 수집될 **단일 디렉토리의 절대 경로**입니다. 이 디렉토리는 비어 있어야 하며, 버전 관리 시스템(Git)에 포함되어서는 안 됩니다. (`.gitignore`에 추가)
    ```python
    # settings.py
    STATIC_ROOT = BASE_DIR / 'staticfiles' # 프로젝트 루트에 staticfiles 디렉토리 생성
    ```

-   **`collectstatic` 실행**: 배포 과정에서 다음 명령어를 실행합니다.
    ```bash
    python manage.py collectstatic
    ```
    이 명령은 `STATICFILES_DIRS` (프로젝트 공통 정적 파일)와 각 앱의 `static/` 디렉토리를 포함한 모든 경로를 순회하며 정적 파일들을 `STATIC_ROOT`에 복사합니다.

-   **정적 파일 캐시 무효화 (Cache Busting)**: 브라우저는 정적 파일을 캐시하여 로딩 속도를 높입니다. 하지만 파일이 업데이트되었을 때, 사용자가 이전 버전의 캐시된 파일을 계속 사용하면 문제가 발생할 수 있습니다. Django는 `ManifestStaticFilesStorage`를 통해 이 문제를 해결합니다.
    -   **원리**: 파일 내용의 해시 값을 파일 이름에 포함시킵니다. (예: `style.css` -> `style.2e8d.css`). 파일 내용이 변경되면 해시 값이 바뀌어 파일 이름도 변경되므로, 브라우저는 새로운 파일을 다운로드하게 됩니다.
    -   **설정**: `settings.py`에 `STATICFILES_STORAGE`를 설정합니다.
        ```python
        # settings.py
        STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'
        # Whitenoise를 사용하는 경우: 'whitenoise.storage.CompressedManifestStaticFilesStorage'
        ```
    -   **사용**: 템플릿에서 `{% static %}` 태그를 사용하면 Django가 자동으로 해시된 파일 경로를 생성해줍니다.

---

### 1.2. WSGI/ASGI 서버: Django 애플리케이션 실행

Django 애플리케이션은 웹 서버(Nginx 등)와 직접 통신하지 않습니다. 대신 WSGI(Web Server Gateway Interface) 또는 ASGI(Asynchronous Server Gateway Interface) 서버를 통해 통신합니다. 이들은 웹 서버의 요청을 받아 Django 애플리케이션으로 전달하고, 애플리케이션의 응답을 웹 서버로 다시 전달하는 역할을 합니다.

-   **WSGI (Web Server Gateway Interface)**: 동기(synchronous) 웹 애플리케이션을 위한 표준 인터페이스입니다. 대부분의 전통적인 Django 애플리케이션은 WSGI를 사용합니다.
    -   **Gunicorn**: 파이썬 WSGI 서버 중 가장 널리 사용되고 안정적입니다. 설정이 간단하고 성능이 뛰어납니다.
        -   **설치**: `pip install gunicorn`
        -   **주요 옵션**: 
            -   `--workers N`: 동시에 처리할 수 있는 요청의 수(워커 프로세스 수)를 지정합니다. 일반적으로 `(CPU 코어 수 * 2) + 1`로 설정하는 것이 권장됩니다.
            -   `--bind IP:PORT` 또는 `--bind unix:/path/to/socket`: Gunicorn이 요청을 수신할 주소와 포트 또는 유닉스 소켓 경로를 지정합니다. 웹 서버(Nginx)와 통신할 때 사용됩니다.
            -   `--timeout SECONDS`: 워커가 응답하기까지 기다릴 최대 시간입니다. 장시간 응답이 없는 워커는 재시작됩니다.
            -   `--log-file FILE`: Gunicorn의 로그를 기록할 파일 경로입니다.
        -   **실행 예시**:
            ```bash
            gunicorn <project_name>.wsgi:application --workers 3 --bind 0.0.0.0:8000 --timeout 120 --log-file - # -는 stdout으로 로그 출력
            ```
    -   **uWSGI**: Gunicorn보다 더 많은 기능을 제공하고 매우 높은 성능을 낼 수 있지만, 설정이 복잡하여 초보자에게는 Gunicorn이 더 적합합니다.

-   **ASGI (Asynchronous Server Gateway Interface)**: 비동기(asynchronous) 웹 애플리케이션을 위한 새로운 표준 인터페이스입니다. Django 3.0부터 기본적으로 지원하며, 웹소켓(WebSocket)이나 장시간 I/O 작업이 많은 애플리케이션에서 성능을 크게 향상시킬 수 있습니다.
    -   **Uvicorn**: 파이썬 ASGI 서버 중 가장 빠르고 널리 사용됩니다. Gunicorn과 함께 사용하여 ASGI 워커로 실행할 수도 있습니다.
        -   **설치**: `pip install uvicorn`
        -   **실행 예시**:
            ```bash
            uvicorn <project_name>.asgi:application --host 0.0.0.0 --port 8000 --workers 3
            ```
    -   **Gunicorn + Uvicorn 워커**: Gunicorn을 사용하여 Uvicorn 워커를 실행할 수 있습니다.
        ```bash
        gunicorn <project_name>.asgi:application -k uvicorn.workers.UvicornWorker --workers 3 --bind 0.0.0.0:8000
        ```

-   **WSGI vs ASGI 선택 가이드**: 
    -   **WSGI (Gunicorn)**: 대부분의 전통적인 웹 애플리케이션(주로 동기식 요청 처리)에 적합합니다. 설정이 간단하고 안정적입니다.
    -   **ASGI (Uvicorn, Daphne)**: 웹소켓, 롱 폴링, 서버 센트 이벤트(SSE) 등 실시간 통신이 필요하거나, 외부 API 호출, 대용량 파일 처리 등 I/O 바운드 작업이 많은 비동기 애플리케이션에 적합합니다. 비동기 뷰(`async def`)를 사용하면 성능 이점을 얻을 수 있습니다.

    **실무 Tip**: 애플리케이션의 특성을 고려하여 선택합니다. 대부분의 CRUD 기반 웹 서비스는 WSGI로도 충분하며, 비동기 기능이 필요할 때만 ASGI로 전환하는 것을 고려합니다.

### 1.3. 웹 서버 (Nginx, Apache)

Nginx나 Apache와 같은 웹 서버는 Django 배포 스택의 최전선에 위치하는 핵심 구성 요소입니다. 웹 서버는 클라이언트의 요청을 가장 먼저 받아 처리하며, 다음과 같은 중요한 역할을 수행합니다.

-   **리버스 프록시 (Reverse Proxy)**: 클라이언트의 요청을 받아 내부망에 있는 WSGI/ASGI 서버(Gunicorn, Uvicorn 등)로 전달합니다. 이를 통해 애플리케이션 서버의 IP 주소나 포트를 외부에 노출하지 않고 보안을 강화할 수 있습니다.
-   **정적 파일 서빙 (Static File Serving)**: `collectstatic`으로 수집된 정적 파일(`CSS`, `JavaScript`, 이미지 등)과 사용자가 업로드한 미디어 파일을 Django 대신 직접 서빙합니다. 이는 Django 애플리케이션의 부하를 크게 줄여 성능을 향상시킵니다.
-   **로드 밸런싱 (Load Balancing)**: 여러 대의 애플리케이션 서버에 트래픽을 분산하여, 특정 서버에 장애가 발생하더라도 서비스 중단을 방지하고(고가용성), 전체 처리량을 높입니다(확장성).
-   **HTTPS/SSL 암호화**: SSL/TLS 인증서를 적용하여 클라이언트와 서버 간의 모든 통신을 암호화합니다. Let's Encrypt와 같은 서비스를 통해 무료로 인증서를 발급받고 자동 갱신을 설정하는 것이 일반적입니다.
-   **보안 강화**: 보안 관련 HTTP 헤더를 추가하고, 특정 IP의 과도한 요청을 제한(Rate Limiting)하는 등 다양한 보안 기능을 적용할 수 있습니다.

**일반적인 프로덕션 배포 스택:**
`클라이언트 <-> 웹 서버 (Nginx) <-> 유닉스 소켓 <-> WSGI/ASGI 서버 (Gunicorn) <-> Django 애플리케이션`

#### 1.3.1. Nginx 설정 예시 (실무 중심)

아래는 리버스 프록시, 정적 파일 서빙, SSL 적용, 보안 헤더, 속도 제한(Rate Limiting)을 포함한 실무적인 Nginx 설정 예시입니다.

```nginx
# /etc/nginx/nginx.conf (http 블록 내에 추가)
# IP당 초당 10개의 요청을 처리하는 'one'이라는 이름의 10MB 메모리 존을 설정합니다.
# burst=20은 큐(queue)에 20개의 요청을 추가로 대기시킬 수 있음을 의미합니다.
limit_req_zone $binary_remote_addr zone=one:10m rate=10r/s;

# /etc/nginx/sites-available/your_project_name.conf (새 파일 생성)
server {
    # 80번 포트로 들어오는 HTTP 요청을 HTTPS로 리다이렉트합니다.
    listen 80;
    server_name your_domain.com www.your_domain.com;

    # Let's Encrypt 인증서 갱신을 위한 경로 허용
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    # 443번 포트에서 SSL/TLS 연결을 수신합니다.
    listen 443 ssl http2;
    server_name your_domain.com www.your_domain.com;

    # SSL 인증서 경로 (Let's Encrypt 사용 기준)
    ssl_certificate /etc/letsencrypt/live/your_domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your_domain.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    # 보안 헤더 설정
    add_header X-Frame-Options "SAMEORIGIN"; # 클릭재킹 방지
    add_header X-Content-Type-Options "nosniff"; # MIME 타입 스니핑 방지
    add_header Referrer-Policy "strict-origin-when-cross-origin"; # Referer 정보 전송 제어
    add_header Permissions-Policy "camera=(), microphone=(), geolocation=()"; # 브라우저 기능 접근 제어

    # 파일 업로드 크기 제한 (필요에 따라 조절)
    client_max_body_size 100M;

    # 정적 파일 서빙 설정
    location /static/ {
        # alias는 URL 경로를 파일 시스템 경로에 매핑합니다.
        alias /path/to/your/project/staticfiles/;
        expires 7d; # 브라우저 캐시 만료 기간 설정
    }

    # 미디어 파일 서빙 설정
    location /media/ {
        alias /path/to/your/project/media/;
        expires 7d;
    }

    # 애플리케이션 서버로의 리버스 프록시 설정
    location / {
        # 위에서 설정한 속도 제한 정책 적용
        limit_req zone=one burst=20;

        # Gunicorn과 통신할 유닉스 소켓 경로
        proxy_pass http://unix:/run/gunicorn.sock;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
*   **설정 활성화**: `sudo ln -s /etc/nginx/sites-available/your_project_name.conf /etc/nginx/sites-enabled/` 명령으로 설정을 활성화하고 `sudo nginx -t`로 문법 검사 후 `sudo systemctl restart nginx`로 Nginx를 재시작합니다.
*   **유닉스 소켓 vs TCP 포트**: TCP 포트(`127.0.0.1:8000`) 대신 유닉스 소켓을 사용하면 OS 커널 내에서 통신이 이루어지므로 네트워크 스택을 거치지 않아 약간의 성능 향상과 보안상 이점이 있습니다.

#### 1.3.2. Gunicorn Systemd 서비스 파일 예시

`systemd`는 리눅스 시스템에서 프로세스를 관리하는 표준적인 방법입니다. Gunicorn을 서비스로 등록하면 시스템 부팅 시 자동으로 실행되고, 예기치 않게 종료될 경우 자동으로 재시작되어 안정적인 서비스 운영이 가능합니다.

```ini
# /etc/systemd/system/gunicorn.service (예시)

[Unit]
Description=gunicorn daemon for your_project_name
# Nginx가 실행된 후에 Gunicorn이 실행되도록 순서를 지정합니다.
After=network.target

[Service]
# Gunicorn을 실행할 사용자 및 그룹. 보안을 위해 root가 아닌 별도의 사용자를 권장합니다.
User=your_user
Group=www-data

# Django 프로젝트 루트 경로
WorkingDirectory=/path/to/your/project

# 환경 변수 파일 경로. 이 파일에 SECRET_KEY, DATABASE_URL 등을 저장합니다.
# 이 파일은 버전 관리에 포함되지 않아야 합니다.
EnvironmentFile=/etc/environment/your_project_name.env

# Gunicorn 실행 명령어
# Nginx 설정과 일치하도록 유닉스 소켓을 사용합니다.
ExecStart=/path/to/your/project/venv/bin/gunicorn \
    --workers 3 \
    --bind unix:/run/gunicorn.sock \
    your_project_name.wsgi:application

# --workers: 워커 프로세스 수. 일반적으로 (CPU 코어 수 * 2) + 1 로 설정합니다.
# --bind: Nginx와 통신할 유닉스 소켓 파일 경로.
# your_project_name.wsgi:application: 프로젝트의 WSGI 애플리케이션 경로.

# 프로세스 종료 시 항상 재시작
Restart=always
# 재시작 사이의 대기 시간
RestartSec=5s

[Install]
# 다중 사용자 모드에서 서비스가 활성화되도록 합니다.
WantedBy=multi-user.target
```
*   **서비스 활성화 및 시작**: `sudo systemctl enable gunicorn`으로 부팅 시 자동 실행을 설정하고, `sudo systemctl start gunicorn`으로 서비스를 시작합니다. `sudo systemctl status gunicorn`으로 상태를 확인할 수 있습니다.
*   **로그 확인**: `sudo journalctl -u gunicorn` 명령어로 Gunicorn의 로그를 실시간으로 확인할 수 있습니다.

### 1.4. 환경 변수 관리 (프로덕션)

**"설정은 코드에서 분리되어야 한다"** 는 [The Twelve-Factor App](https://12factor.net/config)의 핵심 원칙입니다. `SECRET_KEY`, 데이터베이스 접속 정보, 외부 API 키 등 민감하거나 환경에 따라 달라지는 정보들을 코드에 직접 하드코딩하는 것은 심각한 보안 위협을 초래하며, 배포 유연성을 저해합니다.

프로덕션 환경에서는 `.env` 파일을 사용하는 대신, 더 안전하고 확장 가능한 방법으로 환경 변수를 관리해야 합니다.

#### 1.4.1. 프로덕션 환경의 환경 변수 관리 전략

환경 변수 관리 방법은 인프라의 복잡성과 보안 요구 수준에 따라 선택할 수 있습니다.

*   **Good: 시스템 환경 변수 및 파일**
    *   **방법**: `systemd` 서비스 파일의 `Environment` 또는 `EnvironmentFile` 지시어를 사용하거나, 서버의 `/etc/environment` 파일에 변수를 직접 설정합니다. `EnvironmentFile`을 사용하는 것이 변수를 그룹화하여 관리하기 용이합니다.
    *   **장점**: 구현이 간단하고, 코드가 저장소에 커밋될 때 민감 정보가 유출될 위험이 없습니다. 단일 서버 환경에 적합합니다.
    *   **단점**: 서버가 여러 대일 경우, 각 서버에 접속하여 수동으로 설정해야 하므로 확장성이 떨어지고 설정 실수가 발생할 수 있습니다.

    ```bash
    # /etc/environment/your_project_name.env 예시
    # 이 파일의 권한은 600 또는 640으로 설정하여 보안을 강화합니다.
    DJANGO_SETTINGS_MODULE=your_project_name.settings.production
    SECRET_KEY='your-super-secret-and-long-key'
    DATABASE_URL='mysql://user:password@host:port/dbname'
    ```

*   **Better: 컨테이너 및 오케스트레이션 도구**
    *   **방법**: Docker를 사용할 경우 `docker-compose.yml`의 `env_file` 옵션을 사용하거나, `docker run` 명령어의 `-e` 플래그로 변수를 주입합니다. Kubernetes 환경에서는 `ConfigMap`과 `Secret` 객체를 사용하여 설정을 관리합니다. `Secret`은 Base64로 인코딩되어 민감 정보를 더 안전하게 다룹니다.
    *   **장점**: 환경 구성을 코드화(IaC)하여 이식성과 확장성을 크게 높입니다. 개발, 스테이징, 프로덕션 환경을 동일한 방식으로 구성할 수 있습니다.
    *   **단점**: Docker 및 Kubernetes에 대한 학습 곡선이 존재합니다.

*   **Best: 중앙화된 비밀 관리 서비스 (Secret Management)**
    *   **방법**: AWS Secrets Manager, Google Cloud Secret Manager, Azure Key Vault, HashiCorp Vault와 같은 전문 서비스를 사용합니다. 애플리케이션은 실행 시점에 API를 통해 이러한 서비스에서 직접 민감 정보를 가져옵니다.
    *   **장점**: 최고의 보안 수준을 제공합니다. 비밀 정보에 대한 접근 제어(IAM), 감사 로깅, 자동 순환(rotation)과 같은 고급 기능을 지원하여 엔터프라이즈급 보안 요구사항을 충족합니다.
    *   **단점**: 클라우드 플랫폼에 종속될 수 있으며, 추가적인 비용과 구현 복잡성이 발생합니다.

#### 1.4.2. Django에서 환경 변수 사용하기

`django-environ`과 같은 라이브러리를 사용하면 어떤 관리 방식을 선택하든 일관된 방식으로 Django `settings.py`에서 환경 변수를 불러올 수 있습니다.

```python
# settings/base.py 또는 settings.py

import environ
import os

# .env 파일을 읽기 위한 기본 경로 설정 (개발 환경용)
# 프로덕션에서는 시스템에 설정된 환경 변수를 직접 읽습니다.
env = environ.Env(
    # 기본값 및 타입 캐스팅 설정
    DEBUG=(bool, False)
)

# .env 파일이 존재할 경우에만 읽도록 설정
# BASE_DIR는 manage.py가 있는 프로젝트 루트를 가리킵니다.
# environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# 이제 os.environ 또는 .env 파일에서 변수를 가져옵니다.
# 프로덕션에서는 systemd나 Docker가 설정한 환경 변수를 os.environ을 통해 읽게 됩니다.
SECRET_KEY = env('SECRET_KEY')

# DEBUG 값은 환경 변수에서 가져오되, 없으면 False를 기본값으로 사용합니다.
DEBUG = env('DEBUG')

# 데이터베이스 URL을 파싱하여 DATABASES 설정 객체로 변환해줍니다.
DATABASES = {
    'default': env.db_url('DATABASE_URL')
}

# 허용할 호스트 목록도 환경 변수로 관리합니다.
# env.list는 콤마로 구분된 문자열을 파이썬 리스트로 변환해줍니다.
# 예: ALLOWED_HOSTS="your_domain.com,www.your_domain.com"
ALLOWED_HOSTS = env.list('ALLOWED_HOSTS', default=[])
```
이러한 방식으로 설정을 관리하면, 코드를 전혀 수정하지 않고도 `systemd` 환경 변수, Docker 컨테이너, 클라우드 비밀 관리 서비스 등 다양한 환경에 맞게 애플리케이션을 유연하게 배포할 수 있습니다.

## 2. 설정 파일 분리 (Settings Management)

프로젝트의 규모가 커지고 개발, 테스트, 운영 등 다양한 환경에 배포될 경우, `settings.py` 파일을 환경별로 분리하여 관리하는 것이 일반적입니다. 이는 설정의 유연성과 보안을 강화하는 데 필수적입니다.

**권장되는 설정 파일 구조:**

```
your_project_name/
├── settings/
│   ├── __init__.py
│   ├── base.py         # 모든 환경에 공통으로 적용되는 기본 설정
│   ├── local.py        # 개발 환경 전용 설정 (로컬 개발, DEBUG=True 등)
│   ├── production.py   # 운영 환경 전용 설정 (DEBUG=False, 실제 DB/SECRET_KEY 등)
│   └── __pycache__/
└── manage.py
└── ...
```

**각 파일의 역할 및 내용:**

*   **`base.py`**:
    *   모든 환경에서 공통으로 사용되는 기본 설정들을 정의합니다.
    *   `INSTALLED_APPS`, `MIDDLEWARE`, `TEMPLATES`, `STATIC_URL`, `MEDIA_URL` 등 환경에 따라 크게 변하지 않는 설정들이 포함됩니다.
    *   민감한 정보(예: `SECRET_KEY`, 데이터베이스 접속 정보)는 포함하지 않습니다.

*   **`local.py`**:
    *   개발 환경에서만 필요한 설정들을 정의합니다.
    *   `DEBUG = True`
    *   `ALLOWED_HOSTS = ['127.0.0.1', 'localhost']`
    *   SQLite 데이터베이스 설정
    *   개발용 로깅 설정
    *   **중요**: 이 파일은 절대로 버전 관리 시스템(Git)에 커밋되어서는 안 됩니다. (`.gitignore`에 추가)

    ```python
    # settings/local.py 예시
    from .base import *

    DEBUG = True

    ALLOWED_HOSTS = ['127.0.0.1', 'localhost']

    # 개발용 데이터베이스 (SQLite)
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

    # 개발용 로깅 설정 등
    ```

*   **`production.py`**:
    *   운영 환경에서만 필요한 설정들을 정의합니다.
    *   `DEBUG = False`
    *   `ALLOWED_HOSTS` (실제 도메인)
    *   실제 데이터베이스(PostgreSQL, MySQL 등) 설정
    *   `SECRET_KEY` 및 기타 민감 정보는 **환경 변수**에서 로드합니다. (섹션 1.4 참조)
    *   운영용 로깅 설정, 캐싱 설정 등

    ```python
    # settings/production.py 예시
    from .base import *
    import environ # django-environ 사용 예시

    # 환경 변수 로드 (섹션 1.4에서 설명된 방식 사용)
    env = environ.Env()
    # .env 파일을 사용하지 않고, 시스템 환경 변수에서 직접 읽도록 설정
    # env.read_env(os.path.join(BASE_DIR, '.env')) # 개발 환경에서만 사용 권장

    DEBUG = False

    # ALLOWED_HOSTS는 환경 변수에서 로드
    ALLOWED_HOSTS = env.list('ALLOWED_HOSTS')

    # SECRET_KEY는 환경 변수에서 로드
    SECRET_KEY = env('SECRET_KEY')

    # 실제 데이터베이스 설정 (환경 변수에서 로드)
    DATABASES = {
        'default': env.db_url('DATABASE_URL')
    }

    # 정적 파일 및 미디어 파일 ROOT 설정
    STATIC_ROOT = BASE_DIR / 'staticfiles'
    MEDIA_ROOT = BASE_DIR / 'media'

    # 운영용 로깅, 캐싱, 보안 설정 등
    ```

**환경별 설정 적용 방법:**

Django는 `DJANGO_SETTINGS_MODULE` 환경 변수를 통해 어떤 설정 파일을 사용할지 결정합니다.

*   **개발 환경**:
    ```bash
    export DJANGO_SETTINGS_MODULE=your_project_name.settings.local
    python manage.py runserver
    ```
*   **운영 환경**:
    ```bash
    export DJANGO_SETTINGS_MODULE=your_project_name.settings.production
    gunicorn your_project_name.wsgi:application
    ```
    (또는 `systemd` 서비스 파일 내 `Environment` 또는 `EnvironmentFile`로 설정)

**장점:**

*   **환경별 유연성**: 개발, 테스트, 운영 환경에 맞는 설정을 쉽게 적용하고 전환할 수 있습니다.
*   **보안 강화**: 민감한 정보(DB 비밀번호, API 키 등)를 운영 환경 설정 파일에만 포함시키고, 이를 환경 변수에서 로드하여 코드 저장소에 노출되지 않도록 합니다. `local.py`와 같은 개발 전용 파일은 `.gitignore`에 추가하여 실수로 커밋되는 것을 방지합니다.
*   **코드 관리 용이성**: 각 환경의 설정이 명확히 분리되어 있어 코드 관리가 용이하며, 협업 시 충돌을 줄일 수 있습니다.

## 3. 배포 자동화 (CI/CD)

CI/CD(Continuous Integration/Continuous Delivery/Deployment)는 소프트웨어 개발의 핵심적인 현대적 관행입니다. 개발자가 작성한 코드를 자동으로 빌드, 테스트, 배포하는 파이프라인을 구축하여, 소프트웨어의 품질을 높이고 배포 주기를 단축하며, 안정적인 서비스 제공을 가능하게 합니다.

**CI/CD의 핵심 목표:**

*   **빠른 피드백**: 코드 변경이 서비스에 미치는 영향을 빠르게 확인하고 문제를 조기에 발견합니다.
*   **수동 오류 감소**: 반복적인 수동 작업을 자동화하여 인적 오류를 최소화합니다.
*   **일관된 배포**: 모든 배포가 동일한 절차와 환경에서 이루어지도록 보장합니다.
*   **빠른 시장 출시**: 새로운 기능이나 버그 수정을 신속하게 사용자에게 제공합니다.

### 3.1. CI (Continuous Integration)

CI는 개발자들이 작성한 코드를 주기적으로 통합(merge)하고, 자동으로 빌드 및 테스트를 실행하여 코드 변경으로 인한 문제를 조기에 발견하는 과정입니다.

**CI 파이프라인의 일반적인 단계:**

1.  **코드 변경 감지**: Git 저장소(예: GitHub, GitLab)에 코드가 푸시되거나 병합될 때 CI 시스템이 이를 감지합니다.
2.  **환경 설정**: 테스트 및 빌드를 위한 격리된 환경(예: Docker 컨테이너)을 준비합니다.
3.  **의존성 설치**: 프로젝트에 필요한 라이브러리 및 패키지(예: `pip install -r requirements.txt`)를 설치합니다.
4.  **코드 품질 검사 (Linting & Formatting)**: 코드 스타일 가이드 준수 여부를 확인하고(예: Black, Flake8), 잠재적인 오류를 검출합니다.
5.  **테스트 실행**: 단위 테스트, 통합 테스트, 기능 테스트 등을 자동으로 실행하여 코드의 정확성과 안정성을 검증합니다. (예: `pytest`)
6.  **보안 스캔**: 정적 분석 도구(SAST)를 사용하여 코드의 보안 취약점을 스캔합니다.
7.  **빌드 아티팩트 생성**: 테스트를 통과한 코드를 배포 가능한 형태로 빌드합니다. Django 애플리케이션의 경우, 주로 Docker 이미지를 생성하는 단계가 포함됩니다.

**주요 CI 도구:** GitHub Actions, GitLab CI/CD, Jenkins, CircleCI, Travis CI 등

### 3.2. CD (Continuous Delivery/Deployment)

CD는 CI 단계를 통과한 코드를 자동으로 스테이징 또는 프로덕션 환경에 배포하는 과정입니다.

*   **Continuous Delivery (지속적 전달)**: CI를 통과한 빌드 아티팩트(예: Docker 이미지)를 언제든지 수동으로 배포할 수 있는 상태로 유지합니다. 배포 여부는 수동으로 결정합니다.
*   **Continuous Deployment (지속적 배포)**: CI를 통과한 빌드 아티팩트를 자동으로 프로덕션 환경에 배포합니다. 사람의 개입 없이 모든 과정이 자동화됩니다.

**CD 파이프라인의 일반적인 단계:**

1.  **배포 환경 준비**: 대상 서버 또는 클라우드 환경에 접속하여 배포를 위한 준비를 합니다.
2.  **데이터베이스 마이그레이션**: Django의 `python manage.py migrate` 명령을 실행하여 데이터베이스 스키마 변경 사항을 적용합니다. **주의**: 마이그레이션은 서비스 중단 없이 이루어지도록 신중하게 계획해야 합니다. (예: `django-zero-downtime-migrations` 라이브러리 고려)
3.  **정적 파일 수집**: `python manage.py collectstatic`을 실행하여 정적 파일을 `STATIC_ROOT`에 모읍니다.
4.  **새로운 버전 배포**:
    *   **컨테이너 환경**: 새로 빌드된 Docker 이미지를 컨테이너 레지스트리(Docker Hub, AWS ECR 등)에서 가져와 배포합니다.
    *   **가상 머신 환경**: 빌드된 코드를 서버에 복사하고, Gunicorn/Nginx 서비스를 재시작합니다.
5.  **배포 전략 적용**: 무중단 배포를 위해 Blue/Green, Canary, Rolling Update 등의 전략을 사용합니다.
6.  **헬스 체크 및 모니터링**: 배포 후 서비스가 정상적으로 작동하는지 확인하고, 문제가 발생할 경우 자동으로 롤백하거나 알림을 보냅니다.

**주요 CD 도구:** Argo CD, Spinnaker, Jenkins, GitLab CI/CD, GitHub Actions 등 (CI/CD 통합 도구가 많음)

### 3.3. 실무적인 CI/CD 워크플로우 예시 (Django + Docker + GitHub Actions)

1.  **개발**: 개발자가 로컬에서 코드를 작성하고 테스트합니다.
2.  **Git Push**: 변경 사항을 GitHub 저장소의 특정 브랜치(예: `develop` 또는 `main`)에 푸시합니다.
3.  **CI 트리거**: GitHub Actions 워크플로우가 자동으로 트리거됩니다.
4.  **CI 실행**:
    *   Python 환경 설정 및 의존성 설치
    *   코드 Linting (Black, Flake8)
    *   단위 및 통합 테스트 실행 (Pytest)
    *   테스트 통과 시, Django 애플리케이션의 Docker 이미지를 빌드하고 Docker Hub 또는 AWS ECR과 같은 컨테이너 레지스트리에 푸시합니다.
5.  **CD 트리거 (선택적)**:
    *   **지속적 전달**: CI가 성공하면 수동 배포를 위한 버튼이 활성화됩니다.
    *   **지속적 배포**: CI가 성공하면 자동으로 프로덕션 서버에 배포를 시작합니다.
6.  **CD 실행**:
    *   대상 서버(VM 또는 Kubernetes 클러스터)에 접속합니다.
    *   새로운 Docker 이미지를 풀(pull)합니다.
    *   **데이터베이스 마이그레이션 실행**: `python manage.py migrate`
    *   **정적 파일 수집 실행**: `python manage.py collectstatic`
    *   새로운 컨테이너를 배포하고 기존 컨테이너를 점진적으로 교체합니다 (무중단 배포).
    *   배포 후 헬스 체크를 수행합니다.

이러한 CI/CD 파이프라인을 구축하면 개발 생산성을 극대화하고, 안정적이고 신뢰할 수 있는 소프트웨어 배포를 보장할 수 있습니다.
