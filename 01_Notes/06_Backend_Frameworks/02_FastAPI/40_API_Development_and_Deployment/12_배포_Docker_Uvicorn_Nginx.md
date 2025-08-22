<h2>FastAPI 학습 가이드: 배포 Docker, Uvicorn, Nginx - 프로덕션 환경 구축</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 FastAPI 애플리케이션을 실제 운영 환경에 배포하는 과정을 학습하는 것을 목표로 합니다. Docker를 이용한 컨테이너화, Uvicorn을 이용한 애플리케이션 실행, 그리고 Nginx를 이용한 웹 서버 설정을 통해 안정적이고 확장 가능한 프로덕션 배포 전략을 이해하고 구현하는 능력을 기릅니다.

<h2>목차</h2>

- [1. 배포 스택 개요](#1-배포-스택-개요)
  - [1.1. 왜 이 스택을 사용하는가?](#11-왜-이-스택을-사용하는가)
- [2. Docker를 이용한 컨테이너화](#2-docker를-이용한-컨테이너화)
  - [2.1. Dockerfile 작성](#21-dockerfile-작성)
  - [2.2. Docker 이미지 빌드 및 실행](#22-docker-이미지-빌드-및-실행)
  - [2.3. Docker Compose를 이용한 다중 서비스 관리](#23-docker-compose를-이용한-다중-서비스-관리)
- [3. Uvicorn: ASGI 애플리케이션 서버](#3-uvicorn-asgi-애플리케이션-서버)
  - [3.1. Uvicorn 실행 옵션](#31-uvicorn-실행-옵션)
  - [3.2. Gunicorn과 Uvicorn Worker 조합](#32-gunicorn과-uvicorn-worker-조합)
- [4. Nginx: 웹 서버 및 리버스 프록시](#4-nginx-웹-서버-및-리버스-프록시)
  - [4.1. Nginx의 역할](#41-nginx의-역할)
  - [4.2. Nginx 설정 예시](#42-nginx-설정-예시)
- [5. 배포 자동화 (CI/CD) 개요](#5-배포-자동화-cicd-개요)
  - [5.1. CI (Continuous Integration)](#51-ci-continuous-integration)
  - [5.2. CD (Continuous Delivery/Deployment)](#52-cd-continuous-deliverydeployment)

--- 

## 1. 배포 스택 개요

FastAPI 애플리케이션을 프로덕션 환경에 배포할 때는 일반적으로 다음과 같은 스택을 사용합니다.

`클라이언트 (웹 브라우저/모바일 앱) <-> 웹 서버 (Nginx) <-> ASGI 서버 (Uvicorn) <-> FastAPI 애플리케이션`

### 1.1. 왜 이 스택을 사용하는가?
-   **Nginx (웹 서버)**: 클라이언트의 요청을 가장 먼저 받아 처리합니다. 정적 파일 서빙, 로드 밸런싱, SSL/TLS 암호화, 리버스 프록시(Reverse Proxy) 역할을 수행하여 애플리케이션 서버의 부하를 줄이고 보안을 강화합니다.
-   **Uvicorn (ASGI 서버)**: FastAPI 애플리케이션을 실행하는 서버입니다. Nginx로부터 요청을 받아 FastAPI로 전달하고, FastAPI의 응답을 Nginx로 다시 전달합니다. 비동기 처리에 최적화되어 있습니다.
-   **Docker (컨테이너화)**: 애플리케이션과 그 의존성을 컨테이너로 패키징하여 개발, 테스트, 프로덕션 환경 간의 일관성을 보장하고 배포를 단순화합니다.

## 2. Docker를 이용한 컨테이너화

Docker는 애플리케이션과 그 실행 환경을 컨테이너라는 독립적인 단위로 묶어 관리하는 기술입니다. 이를 통해 "내 컴퓨터에서는 되는데 서버에서는 안 돼요"와 같은 문제를 해결하고, 배포 과정을 표준화할 수 있습니다.

### 2.1. Dockerfile 작성
FastAPI 애플리케이션을 위한 `Dockerfile`을 작성합니다.

```dockerfile
# Dockerfile

# 1. 베이스 이미지 선택 (Python 3.9 Slim Buster 사용)
FROM python:3.9-slim-buster

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (필요한 경우)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 의존성 설치
# requirements.txt를 먼저 복사하여 캐시 레이어를 활용
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 5. 애플리케이션 코드 복사
COPY . /app

# 6. Uvicorn을 사용하여 애플리케이션 실행 (기본 명령어)
# main:app은 main.py 파일 내의 app 인스턴스를 의미
# --host 0.0.0.0은 모든 IP에서 접근 가능하도록 설정
# --port 8000은 8000번 포트 사용
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
`requirements.txt` 예시:
```
fastapi
uvicorn
pydantic
ssqlalchemy
asyncpg
python-dotenv
```

### 2.2. Docker 이미지 빌드 및 실행
`Dockerfile`이 있는 디렉토리에서 다음 명령어를 실행합니다.

```bash
# Docker 이미지 빌드
docker build -t my-fastapi-app .

# Docker 컨테이너 실행
docker run -d --name fastapi-container -p 8000:8000 my-fastapi-app
```
-   `docker build -t my-fastapi-app .`: 현재 디렉토리의 `Dockerfile`을 사용하여 `my-fastapi-app`이라는 이름의 Docker 이미지를 빌드합니다.
-   `docker run -d --name fastapi-container -p 8000:8000 my-fastapi-app`: 빌드된 이미지를 사용하여 `fastapi-container`라는 이름의 컨테이너를 백그라운드(`-d`)에서 실행하고, 호스트의 8000번 포트를 컨테이너의 8000번 포트에 연결(`-p`)합니다.

### 2.3. Docker Compose를 이용한 다중 서비스 관리
FastAPI 애플리케이션이 데이터베이스, Redis 등 여러 서비스와 함께 작동해야 할 경우, Docker Compose를 사용하여 이들을 한 번에 정의하고 관리할 수 있습니다.

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: . # 현재 디렉토리의 Dockerfile 사용
    ports:
      - "8000:8000"
    environment:
      # 환경 변수 설정 (예: 데이터베이스 URL)
      DATABASE_URL: "postgresql+asyncpg://user:password@db:5432/mydb"
    depends_on:
      - db # db 서비스가 먼저 시작되도록 의존성 설정

  db:
    image: "postgres:13-alpine" # PostgreSQL 데이터베이스 이미지
    environment:
      POSTGRES_DB: mydb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - db_data:/var/lib/postgresql/data # 데이터 영속성을 위한 볼륨 마운트

volumes:
  db_data: # db_data 볼륨 정의
```
-   `docker-compose up -d`: `docker-compose.yml`에 정의된 모든 서비스를 백그라운드에서 실행합니다.
-   `docker-compose down`: 모든 서비스를 중지하고 컨테이너를 제거합니다.

## 3. Uvicorn: ASGI 애플리케이션 서버

Uvicorn은 FastAPI 애플리케이션을 실행하는 고성능 ASGI(Asynchronous Server Gateway Interface) 서버입니다.

### 3.1. Uvicorn 실행 옵션
-   `uvicorn main:app`: `main.py` 파일 내의 `app` 인스턴스를 실행합니다.
-   `--host 0.0.0.0`: 모든 네트워크 인터페이스에서 요청을 수신합니다. (프로덕션 환경에서 필수)
-   `--port 8000`: 8000번 포트에서 수신합니다.
-   `--workers N`: 여러 워커 프로세스를 사용하여 동시 처리량을 늘립니다. (일반적으로 `CPU 코어 수 * 2 + 1`로 설정)
-   `--log-level info`: 로그 레벨을 설정합니다.
-   `--reload`: 코드 변경 시 자동으로 서버를 재시작합니다. (개발 환경에서만 사용)

### 3.2. Gunicorn과 Uvicorn Worker 조합
프로덕션 환경에서는 Uvicorn을 직접 실행하기보다, Gunicorn과 같은 프로세스 관리자와 함께 사용하는 것이 일반적입니다. Gunicorn은 워커 프로세스 관리, 로깅, 안정성 등 프로덕션에 필요한 다양한 기능을 제공합니다.

```bash
# Gunicorn 설치
pip install gunicorn

# Gunicorn을 사용하여 Uvicorn 워커 실행
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
-   `--worker-class uvicorn.workers.UvicornWorker`: Gunicorn이 Uvicorn 워커를 사용하도록 지정합니다.
-   `--workers 4`: 4개의 워커 프로세스를 실행하여 동시 처리량을 늘립니다.

## 4. Nginx: 웹 서버 및 리버스 프록시

Nginx는 클라이언트의 요청을 받아 처리하고, 이를 Uvicorn(FastAPI)으로 전달하는 리버스 프록시 역할을 수행합니다. 또한 정적 파일 서빙, SSL/TLS 암호화, 로드 밸런싱 등 다양한 기능을 제공합니다.

### 4.1. Nginx의 역할
-   **리버스 프록시**: 클라이언트의 요청을 받아 Uvicorn으로 전달하고, Uvicorn의 응답을 클라이언트에게 다시 전달합니다.
-   **정적 파일 서빙**: 이미지, CSS, JavaScript 등 정적 파일을 직접 서빙하여 FastAPI 애플리케이션의 부하를 줄입니다.
-   **SSL/TLS 암호화**: HTTPS를 통해 클라이언트와 서버 간의 통신을 암호화합니다.
-   **로드 밸런싱**: 여러 Uvicorn 인스턴스에 트래픽을 분산하여 서비스의 확장성과 안정성을 높입니다.

### 4.2. Nginx 설정 예시
```nginx
# /etc/nginx/sites-available/your_fastapi_app.conf

server {
    listen 80;
    server_name your_domain.com www.your_domain.com;

    # Let's Encrypt 인증서 갱신을 위한 경로 허용 (HTTPS 설정 시)
    # location /.well-known/acme-challenge/ {
    #     root /var/www/html;
    # }

    # 모든 HTTP 요청을 HTTPS로 리다이렉트 (HTTPS 설정 시)
    # location / {
    #     return 301 https://$host$request_uri;
    # }

    # 정적 파일 서빙 (필요한 경우)
    # location /static/ {
    #     alias /path/to/your/static/files/;
    #     expires 7d;
    # }

    # FastAPI 애플리케이션으로의 리버스 프록시
    location / {
        proxy_pass http://127.0.0.1:8000; # Uvicorn이 실행되는 주소와 포트
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
-   `proxy_pass http://127.0.0.1:8000`: Nginx가 8000번 포트에서 실행 중인 Uvicorn으로 요청을 전달하도록 설정합니다.
-   `proxy_set_header`: 클라이언트의 원본 IP 주소, 호스트 정보 등을 Uvicorn으로 전달합니다.

## 5. 배포 자동화 (CI/CD) 개요

FastAPI 애플리케이션의 배포는 CI/CD(Continuous Integration/Continuous Delivery/Deployment) 파이프라인을 통해 자동화할 수 있습니다.

### 5.1. CI (Continuous Integration)
개발자가 작성한 코드를 주기적으로 통합하고, 자동으로 빌드 및 테스트를 실행하여 코드 변경으로 인한 문제를 조기에 발견하는 과정입니다.

-   **주요 단계**: 코드 변경 감지, 환경 설정, 의존성 설치, 코드 품질 검사(Linting), 테스트 실행, Docker 이미지 빌드 및 레지스트리 푸시.

### 5.2. CD (Continuous Delivery/Deployment)
CI 단계를 통과한 코드를 자동으로 스테이징 또는 프로덕션 환경에 배포하는 과정입니다.

-   **주요 단계**: 배포 환경 준비, Docker 이미지 풀(pull), 컨테이너 실행, 헬스 체크, 모니터링.

FastAPI 애플리케이션은 Docker를 통해 컨테이너화되어 있으므로, CI/CD 파이프라인을 통해 Docker 이미지를 빌드하고, 이를 Kubernetes, AWS ECS, Google Cloud Run 등 다양한 컨테이너 오케스트레이션 플랫폼에 배포할 수 있습니다.