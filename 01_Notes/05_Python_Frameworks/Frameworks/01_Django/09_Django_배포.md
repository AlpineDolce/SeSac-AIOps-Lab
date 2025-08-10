<h2>Django Backend: 배포</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django 애플리케이션을 실제 운영 환경에 배포하는 과정과, 효율적인 설정 관리, 배포 자동화(CI/CD)에 대해 다루는 것을 목표로 합니다. 이를 통해 안정적이고 확장 가능한 Django 서비스를 구축하는 능력을 기릅니다.</p>

<h2>목차</h2> 

- [1. 배포 (Deployment)](#1-배포-deployment)
  - [1.1. 정적 파일 수집 (collectstatic)](#11-정적-파일-수집-collectstatic)
  - [1.2. WSGI/ASGI 서버 (Gunicorn, uWSGI)](#12-wsgiasgi-서버-gunicorn-uwsgi)
    - [1.2.1. ASGI와 비동기 뷰](#121-asgi와-비동기-뷰)
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

---

## 1. 배포 (Deployment)

Django 애플리케이션을 개발 환경에서 벗어나 실제 사용자에게 서비스하려면 배포 과정이 필요합니다.

### 1.1. 정적 파일 수집 (collectstatic)
운영 환경에서는 Django 개발 서버가 정적 파일을 제공하지 않습니다. 따라서 배포 전에 모든 정적 파일을 한 곳으로 모아야 합니다.

```bash
python manage.py collectstatic
```
이 명령은 `settings.py`에 정의된 `STATIC_ROOT` 경로로 모든 정적 파일을 복사합니다. 이후 웹 서버(Nginx, Apache)가 이 경로에서 정적 파일을 직접 제공하도록 설정합니다.

### 1.2. WSGI/ASGI 서버 (Gunicorn, uWSGI)
Django는 WSGI(Web Server Gateway Interface) 또는 ASGI(Asynchronous Server Gateway Interface)를 통해 웹 서버와 통신합니다. 개발 서버는 간단한 테스트용이며, 실제 운영 환경에서는 Gunicorn, uWSGI와 같은 프로덕션용 WSGI/ASGI 서버를 사용해야 합니다.

**예시 (Gunicorn 실행):
```bash
gunicorn <project_name>.wsgi:application --bind 0.0.0.0:8000
```

#### 1.2.1. ASGI와 비동기 뷰
Django 3.0부터 ASGI(Asynchronous Server Gateway Interface)를 기본적으로 지원하며, 이를 통해 비동기(asynchronous) 웹 애플리케이션을 구축할 수 있습니다. 비동기 뷰는 I/O 바운드(I/O-bound) 작업(예: 외부 API 호출, 데이터베이스 쿼리, 파일 I/O)이 많은 애플리케이션에서 성능을 크게 향상시킬 수 있습니다.

**실무적 관점:**
*   **성능 향상:** 동기식 뷰는 I/O 작업이 완료될 때까지 다른 요청을 처리할 수 없지만, 비동기 뷰는 I/O 작업이 진행되는 동안 다른 요청을 처리할 수 있어 동시성(concurrency)을 높이고 응답 시간을 단축시킵니다.
*   **실시간 애플리케이션:** 웹소켓(WebSocket)과 같은 실시간 통신이 필요한 애플리케이션(채팅, 알림 등)을 구축하는 데 필수적입니다.
*   **주의사항:** 모든 뷰를 비동기로 만들 필요는 없습니다. CPU 바운드(CPU-bound) 작업이 많은 뷰는 비동기화해도 성능 이점이 적거나 오히려 오버헤드가 발생할 수 있습니다. 비동기 코드는 동기 코드보다 디버깅이 더 복잡할 수 있습니다.

**구현 방법:**
1.  **ASGI 서버 사용:** `daphne`나 `uvicorn`과 같은 ASGI 서버를 사용해야 합니다. Gunicorn도 `uvicorn` 워커를 통해 ASGI를 지원합니다.
    ```bash
    pip install uvicorn
    # uvicorn 실행 예시
    uvicorn <project_name>.asgi:application --host 0.0.0.0 --port 8000
    ```
2.  **비동기 뷰 작성:** 뷰 함수를 `async def`로 정의합니다. 비동기 I/O 작업을 수행할 때는 `await` 키워드를 사용합니다.
    ```python
    # myapp/views.py
    import asyncio
    from django.http import JsonResponse
    from asgiref.sync import sync_to_async # 동기 함수를 비동기로 실행할 때 사용

    # 비동기 뷰 예시 (외부 API 호출 시)
    async def async_view(request):
        # 비동기 I/O 작업 수행 (예: await asyncio.sleep(1) 또는 외부 API 호출)
        await asyncio.sleep(1) # 1초 대기 (비동기 시뮬레이션)
        data = await sync_to_async(some_blocking_io_function)() # 동기 함수를 비동기 컨텍스트에서 실행
        return JsonResponse({"message": "Hello from async view!", "data": data})

    # urls.py에서 연결
    # from .views import async_view
    # path('async-test/', async_view),
    ```
3.  **ORM 비동기 지원:** Django 3.1부터 ORM도 비동기 인터페이스를 제공합니다. `await` 키워드를 사용하여 비동기적으로 데이터베이스 쿼리를 실행할 수 있습니다.
    ```python
    # myapp/views.py
    from myapp.models import MyModel

    async def get_my_model_data(request):
        # 비동기적으로 쿼리 실행
        obj = await MyModel.objects.aget(id=1) # aget, aall, afilter 등
        return JsonResponse({"title": obj.title})
    ```

### 1.3. 웹 서버 (Nginx, Apache)
Nginx나 Apache와 같은 웹 서버는 클라이언트의 요청을 받아 정적 파일을 직접 제공하고, 동적인 요청(Django 애플리케이션)은 WSGI/ASGI 서버로 프록시(Proxy)합니다. 이는 성능, 보안, 로드 밸런싱 등의 이점을 제공합니다.

**일반적인 배포 스택:**
`클라이언트 <-> 웹 서버 (Nginx/Apache) <-> WSGI/ASGI 서버 (Gunicorn/uWSGI) <-> Django 애플리케이션 <-> 데이터베이스`

**실무적 관점:**
*   **HTTPS/SSL 적용:** 프로덕션 환경에서는 반드시 HTTPS를 사용하여 클라이언트와 서버 간의 통신을 암호화해야 합니다. Let's Encrypt와 같은 서비스를 통해 무료 SSL/TLS 인증서를 발급받을 수 있습니다. Nginx나 Apache에서 SSL 인증서를 설정하고 HTTP 요청을 HTTPS로 리다이렉트하도록 구성합니다.
*   **로드 밸런싱:** 여러 Django 애플리케이션 인스턴스에 트래픽을 분산하여 고가용성과 확장성을 확보합니다.
*   **정적/미디어 파일 서빙:** Nginx나 Apache가 `STATIC_ROOT`와 `MEDIA_ROOT`에 있는 정적/미디어 파일을 직접 서빙하도록 설정하여 Django 애플리케이션의 부하를 줄입니다.
*   **보안 헤더:** X-Frame-Options, X-Content-Type-Options, Strict-Transport-Security 등 보안 관련 HTTP 헤더를 설정하여 웹 취약점을 방어합니다.

**MySQL 데이터베이스 설정 및 권한 부여 (`장고사이트구축방법.txt` 참고):**

Django 프로젝트에서 MySQL 데이터베이스를 사용하려면 `settings.py`에 데이터베이스 연결 정보를 설정하고, 필요한 경우 `mysqlclient` 라이브러리를 설치해야 합니다.

1.  **`mysqlclient` 설치:**
    ```bash
pip install mysqlclient
    ```

2.  **MySQL 사용자 및 권한 설정 (MySQL 콘솔에서):**
    ```sql
    mysql -u root -p 
    # 비밀번호 입력

    use mydb; # 사용할 데이터베이스 선택

    # Django에서 사용할 사용자 생성 및 권한 부여
    grant all privileges on mydb.* to user01@localhost identified by '1234';
    flush privileges;
    ```
    *   `mydb`: 데이터베이스 이름
    *   `user01`: Django에서 사용할 사용자 이름
    *   `1234`: `user01`의 비밀번호

3.  **`settings.py`의 `DATABASES` 설정:**
    `0625정리.md`의 3.3 섹션에 있는 MySQL 설정 예시를 참고하여 `settings.py`에 데이터베이스 정보를 추가합니다.

    ```python
    # settings.py
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': 'mydb',
            'USER': 'user01',
            'PASSWORD':'1234',
            'HOST':'localhost',
            'PORT':'3306'
        }
    }
    ```

4.  **마이그레이션 실행:**
    데이터베이스 설정 후, 모델 변경사항을 데이터베이스에 반영하기 위해 마이그레이션을 실행합니다.
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

#### 1.3.1. Nginx 설정 예시 (리버스 프록시 및 정적 파일 서빙)

```nginx
/etc/nginx/sites-available/your_project_name.conf (예시)

server {
    listen 80;
    server_name your_domain.com www.your_domain.com; # 실제 도메인으로 변경

    HTTPS 리다이렉트 (선택 사항, Let's Encrypt 등으로 SSL 설정 후)
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your_domain.com www.your_domain.com;

    ssl_certificate /etc/letsencrypt/live/your_domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your_domain.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    client_max_body_size 100M; # 파일 업로드 크기 제한

    location /static/ {
        alias /path/to/your/project/staticfiles/; # collectstatic으로 모인 정적 파일 경로
    }

    location /media/ {
        alias /path/to/your/project/media/; # 사용자가 업로드한 미디어 파일 경로
    }

    location / {
        proxy_pass http://127.0.0.1:8000; # Gunicorn/uWSGI 서버 주소 및 포트
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
*   `your_domain.com`: 실제 서비스 도메인으로 변경
*   `/path/to/your/project/staticfiles/`: `settings.py`의 `STATIC_ROOT` 경로
*   `/path/to/your/project/media/`: `settings.py`의 `MEDIA_ROOT` 경로
*   `http://127.0.0.1:8000`: Gunicorn 또는 uWSGI가 실행되는 주소와 포트

#### 1.3.2. Gunicorn Systemd 서비스 파일 예시

```ini
# /etc/systemd/system/your_project_name.service (예시)

[Unit]
Description=Gunicorn instance for your_project_name
After=network.target

[Service]
User=your_user # Gunicorn을 실행할 사용자 (예: www-data 또는 생성한 사용자)
Group=www-data # Gunicorn을 실행할 그룹
WorkingDirectory=/path/to/your/project # Django 프로젝트 루트 경로
ExecStart=/path/to/your/project/venv/bin/gunicorn --workers 3 --bind 0.0.0.0:8000 your_project_name.wsgi:application
# --workers: 워커 프로세스 수 (CPU 코어 수 * 2 + 1 권장)
# --bind: Gunicorn이 바인딩할 주소와 포트
# your_project_name.wsgi:application: 프로젝트의 WSGI 애플리케이션 경로
Restart=always # 프로세스 종료 시 항상 재시작

[Install]
WantedBy=multi-user.target
```
*   `your_user`: Gunicorn을 실행할 시스템 사용자 이름
*   `/path/to/your/project`: Django 프로젝트의 절대 경로
*   `your_project_name`: Django 프로젝트 이름 (예: `mysite`)


### 1.4. 환경 변수 관리 (프로덕션)

`python-decouple`이나 `django-environ`을 사용하여 `settings.py`에서 환경 변수를 로드하는 것은 개발 환경에서 편리합니다. 하지만 프로덕션 환경에서는 `.env` 파일을 직접 사용하는 것보다 더 안전하고 체계적인 방법으로 환경 변수를 관리해야 합니다.

#### 1.4.1. 프로덕션 환경에서의 환경 변수 관리

*   **운영체제 환경 변수:**
    *   가장 기본적인 방법은 서버의 운영체제 레벨에서 환경 변수를 설정하는 것입니다. `~/.bashrc`, `~/.profile`, `/etc/environment` 또는 `systemd` 서비스 파일 등을 통해 설정할 수 있습니다.
    *   **장점:** `.env` 파일이 코드 저장소에 실수로 커밋되는 것을 방지합니다.
    *   **단점:** 여러 서버에 배포할 때 각 서버마다 수동으로 설정해야 하는 번거로움이 있습니다.

*   **컨테이너 환경 변수 (Docker, Kubernetes):**
    *   Docker를 사용할 경우 `docker run -e KEY=VALUE` 옵션이나 `docker-compose.yml` 파일의 `environment` 섹션을 통해 환경 변수를 전달합니다.
    *   Kubernetes와 같은 컨테이너 오케스트레이션 도구에서는 `ConfigMap`이나 `Secret`을 사용하여 환경 변수를 관리합니다. `Secret`은 민감한 정보를 안전하게 저장하는 데 사용됩니다.
    *   **장점:** 컨테이너화된 애플리케이션의 이식성과 확장성을 높입니다.
    *   **단점:** 컨테이너 기술에 대한 이해가 필요합니다.

*   **클라우드 서비스의 비밀 관리 도구:**
    *   AWS Secrets Manager, Google Cloud Secret Manager, Azure Key Vault와 같은 클라우드 제공업체의 비밀 관리 서비스를 사용합니다.
    *   **장점:** 민감한 정보를 중앙에서 안전하게 관리하고, 접근 제어 및 감사 기능을 제공합니다.
    *   **단점:** 클라우드 서비스에 종속되며, 추가 비용이 발생할 수 있습니다.

**`systemd` 서비스 파일 예시 (Linux):
```ini
# /etc/systemd/system/myproject.service
[Unit]
Description=Gunicorn instance for myproject
After=network.target

[Service]
User=myuser
Group=www-data
WorkingDirectory=/path/to/myproject
Environment="DJANGO_SETTINGS_MODULE=myproject.settings.production"
Environment="SECRET_KEY=your_super_secret_key"
Environment="DATABASE_URL=mysql://user:password@host:port/dbname"
ExecStart=/path/to/myproject/venv/bin/gunicorn --workers 3 myproject.wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target
```
이러한 방식으로 환경 변수를 관리하면 코드와 설정이 분리되어 보안이 강화되고 배포가 유연해집니다.

## 2. 설정 파일 분리 (Settings Management)

프로젝트의 규모가 커지고 개발, 테스트, 운영 등 다양한 환경에 배포될 경우, `settings.py` 파일을 환경별로 분리하여 관리하는 것이 일반적입니다. 이는 설정의 유연성과 보안을 강화하는 데 도움이 됩니다.

*   **패턴:** `settings/base.py`, `settings/local.py`, `settings/production.py` 등으로 파일을 분리하고, 각 환경에 맞는 설정을 상속하거나 오버라이드합니다.
*   **장점:**
    *   **환경별 유연성:** 개발 환경에서는 `DEBUG=True`와 SQLite를 사용하고, 운영 환경에서는 `DEBUG=False`와 PostgreSQL/MySQL을 사용하는 등 환경에 맞는 설정을 쉽게 적용할 수 있습니다.
    *   **보안 강화:** 민감한 정보(DB 비밀번호, API 키 등)는 운영 환경 설정 파일에만 포함시키고, `.gitignore`에 추가하여 버전 관리 시스템에 노출되지 않도록 합니다.
    *   **코드 관리 용이성:** 각 환경의 설정이 명확히 분리되어 있어 코드 관리가 용이합니다.

## 3. 배포 자동화 (CI/CD)

개발된 애플리케이션을 안정적으로 배포하기 위해서는 CI/CD(Continuous Integration/Continuous Deployment) 파이프라인 구축이 필수적입니다.

### 3.1. CI (Continuous Integration)

*   개발자들이 작성한 코드를 주기적으로 통합(merge)하고, 자동으로 빌드 및 테스트를 실행하여 코드 변경으로 인한 문제를 조기에 발견합니다.
*   **도구:** GitHub Actions, GitLab CI/CD, Jenkins, CircleCI 등.

### 3.2. CD (Continuous Deployment/Delivery)

*   CI 단계를 통과한 코드를 자동으로 스테이징 또는 프로덕션 환경에 배포합니다.
*   **장점:** 개발 주기를 단축하고, 배포 오류를 줄이며, 안정적이고 빠른 서비스 제공을 가능하게 합니다.

### 3.3. 학습 내용

*   CI/CD 도구의 설정 방법, 테스트 자동화, 빌드 및 배포 스크립트 작성, 컨테이너화(Docker) 및 오케스트레이션(Kubernetes)과의 연동 등을 학습하면 좋습니다.
