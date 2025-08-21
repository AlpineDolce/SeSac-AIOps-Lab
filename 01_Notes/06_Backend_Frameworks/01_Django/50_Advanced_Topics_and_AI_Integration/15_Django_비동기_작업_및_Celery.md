<h2>Django Backend: 비동기 작업 및 Celery</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Celery를 활용한 백그라운드 작업 처리, 장시간 소요되는 AI 작업(예: 모델 학습, 대규모 추론)의 비동기 실행 및 관리 방법을 학습하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 비동기 작업의 필요성](#1-비동기-작업의-필요성)
- [2. Celery 소개 및 기본 개념](#2-celery-소개-및-기본-개념)
  - [2.1. Celery란?](#21-celery란)
  - [2.2. 주요 구성 요소](#22-주요-구성-요소)
- [3. Celery 설치 및 Django 연동](#3-celery-설치-및-django-연동)
  - [3.1. Celery 설치](#31-celery-설치)
  - [3.2. Django 프로젝트 설정](#32-django-프로젝트-설정)
- [4. Celery Task 정의 및 실행](#4-celery-task-정의-및-실행)
  - [4.1. Task 작성](#41-task-작성)
  - [4.2. Task 실행](#42-task-실행)
  - [4.3. Task 결과 확인](#43-task-결과-확인)
  - [4.4. 주기적인 Task 실행 (Celery Beat)](#44-주기적인-task-실행-celery-beat)
- [5. Celery를 활용한 AI 작업 처리](#5-celery를-활용한-ai-작업-처리)
  - [5.1. 모델 학습 비동기화](#51-모델-학습-비동기화)
  - [5.2. 대규모 추론 작업 처리](#52-대규모-추론-작업-처리)
- [6. Celery 모니터링 및 관리 (Flower)](#6-celery-모니터링-및-관리-flower)

---

## 1. 비동기 작업의 필요성

웹 애플리케이션에서 사용자 요청을 처리하는 동안 시간이 오래 걸리는 작업(예: 이미지 처리, 대규모 데이터 분석, 외부 API 호출, AI 모델 학습/추론)이 발생하면, 해당 작업이 완료될 때까지 사용자에게 응답을 줄 수 없어 웹 서비스의 응답성이 저하됩니다. 이는 사용자 경험을 해치고, 웹 서버의 리소스를 비효율적으로 사용하게 만듭니다.

비동기 작업은 이러한 장시간 작업을 백그라운드에서 별도로 처리하여, 웹 서버가 즉시 사용자에게 응답을 반환하고 다른 요청을 처리할 수 있도록 합니다. 이는 웹 서비스의 확장성과 응답성을 크게 향상시킵니다.

## 2. Celery 소개 및 기본 개념

### 2.1. Celery란?

Celery는 Python으로 작성된 분산 Task Queue 시스템입니다. 웹 애플리케이션에서 장시간 작업을 비동기적으로 처리하거나, 주기적인 작업을 스케줄링하는 데 사용됩니다. Celery는 웹 서버와 독립적으로 동작하며, 여러 워커(Worker)를 통해 작업을 병렬로 처리할 수 있어 높은 확장성을 제공합니다.

### 2.2. 주요 구성 요소

Celery는 주로 세 가지 핵심 구성 요소로 이루어집니다.

*   **Celery Client (Producer)**: 웹 애플리케이션(Django)과 같이 작업을 생성하고 Celery에게 전달하는 주체입니다. Task를 정의하고 `delay()` 또는 `apply_async()` 메서드를 호출하여 작업을 큐에 보냅니다.
*   **Broker (메시지 브로커)**: Client가 보낸 작업을 임시로 저장하고, Worker에게 전달하는 중간 매개체입니다. RabbitMQ, Redis, Amazon SQS 등이 주로 사용됩니다. Celery의 핵심적인 부분으로, 작업의 안정적인 전달을 보장합니다.
*   **Celery Worker (Consumer)**: Broker로부터 작업을 가져와 실제로 실행하는 프로세스입니다. 여러 Worker를 실행하여 작업을 병렬로 처리할 수 있으며, Worker의 수를 늘려 처리량을 확장할 수 있습니다.
*   **Result Backend (선택 사항)**: Task의 실행 결과(성공/실패 여부, 반환 값, 예외 정보)를 저장하는 곳입니다. 데이터베이스, Redis, RabbitMQ 등이 사용될 수 있습니다.

## 3. Celery 설치 및 Django 연동

### 3.1. Broker 선택: Redis vs RabbitMQ

Celery를 사용하려면 먼저 메시지 브로커를 선택해야 합니다. 가장 많이 사용되는 두 가지는 Redis와 RabbitMQ입니다.

- **Redis:**
  - **장점:** 설치 및 설정이 매우 간단하고 빠릅니다. In-memory 기반으로 동작하여 성능이 뛰어납니다. Result Backend로도 사용할 수 있어 편리합니다.
  - **단점:** 메시지 유실 가능성이 RabbitMQ보다 높습니다. (예: 서버 장애 시). 복잡한 라우팅 기능이 부족합니다.
  - **추천:** 대부분의 중소 규모 애플리케이션, 개발 환경, 또는 메시지 유실이 치명적이지 않은 작업에 적합합니다.

- **RabbitMQ:**
  - **장점:** 메시지 전달 보증(Message Delivery Guarantee) 기능이 뛰어나 안정성이 높습니다. 유연하고 복잡한 라우팅 규칙(Direct, Topic, Fanout 등)을 설정할 수 있습니다. 대규모 시스템에서 검증되었습니다.
  - **단점:** Redis에 비해 설치 및 설정이 복잡합니다.
  - **추천:** 금융 거래, 주문 처리 등 작업 유실이 절대 발생해서는 안 되는 미션 크리티컬한 시스템에 적합합니다.

이 문서에서는 설정이 간편한 **Redis**를 기준으로 설명합니다.

### 3.2. Celery 및 Broker 클라이언트 설치

```bash
# Celery와 Redis 클라이언트 라이브러리 설치
pip install "celery[redis]"
```
`celery[redis]`와 같이 설치하면 Celery와 Redis 연동에 필요한 모든 의존성이 함께 설치됩니다.

### 3.3. Django 프로젝트 설정

Django 프로젝트에 Celery를 연동하는 표준적인 방법은 다음과 같습니다.

**1단계: `proj/proj/celery.py` 파일 생성**
`settings.py`와 같은 위치에 `celery.py` 파일을 생성하여 Celery 애플리케이션 인스턴스를 정의합니다.

```python
# proj/proj/celery.py
import os
from celery import Celery

# Django 프로젝트의 settings.py를 Celery가 사용할 수 있도록 환경 변수 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proj.settings')

app = Celery('proj')

# 'CELERY' 네임스페이스를 사용하여 Django settings.py에서 Celery 설정을 로드
app.config_from_object('django.conf:settings', namespace='CELERY')

# 등록된 모든 Django 앱의 tasks.py 파일을 자동으로 찾아서 로드
app.autodiscover_tasks()
```

**2단계: `proj/proj/__init__.py` 수정**
Django가 시작될 때 위에서 정의한 Celery 앱이 함께 로드되도록 `__init__.py` 파일을 수정합니다.

```python
# proj/proj/__init__.py
from .celery import app as celery_app

__all__ = ('celery_app',)
```

**3단계: `proj/settings.py`에 Celery 설정 추가**
Broker와 Result Backend의 URL 및 기타 기본 설정을 추가합니다.

```python
# proj/settings.py

# Celery Settings
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'Asia/Seoul'
CELERY_ENABLE_UTC = False
```

### 3.4. Docker Compose를 이용한 개발 환경 구성 (권장)

로컬 환경에서 Redis, Django, Celery를 각각 실행하는 것은 번거롭습니다. Docker Compose를 사용하면 이들을 한 번에 관리할 수 있어 매우 편리합니다.

**`docker-compose.yml` 예시:**
```yaml
version: '3.8'

services:
  # Redis 메시지 브로커
  redis:
    image: "redis:alpine"
    ports:
      - "6379:6379"

  # Django 웹 서버
  web:
    build: .
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/code
    ports:
      - "8000:8000"
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0

  # Celery 워커
  worker:
    build: .
    # -A: Celery 앱 지정, -l: 로그 레벨
    command: celery -A proj worker -l info
    volumes:
      - .:/code
    depends_on:
      - web
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/0

  # Celery Beat (주기적 작업 스케줄러) - 필요 시 주석 해제
  # beat:
  #   build: .
  #   command: celery -A proj beat -l info
  #   volumes:
  #     - .:/code
  #   depends_on:
  #     - web
  #     - redis
  #   environment:
  #     - CELERY_BROKER_URL=redis://redis:6379/0
  #     - CELERY_RESULT_BACKEND=redis://redis:6379/0
```
이제 터미널에서 `docker-compose up` 명령어 하나로 전체 개발 환경을 실행할 수 있습니다.

## 4. Celery Task 정의 및 실행

### 4.1. Task 작성 기본

Celery Task는 `@app.task` 데코레이터가 적용된 Python 함수입니다. Django 프로젝트에서는 각 앱의 `tasks.py` 파일에 Task를 정의하는 것이 일반적입니다. Celery는 `app.autodiscover_tasks()` 설정을 통해 이 파일들을 자동으로 찾습니다.

```python
# myapp/tasks.py
from proj.celery import app
import time

@app.task
def add(x, y):
    time.sleep(5) # 시간이 오래 걸리는 작업을 시뮬레이션
    return x + y
```

### 4.2. 견고한 Task 작성을 위한 모범 사례

단순한 Task는 쉽게 작성할 수 있지만, 실제 운영 환경에서는 예측 불가능한 상황에 대비하여 견고하게 Task를 작성해야 합니다.

#### 4.2.1. 멱등성 (Idempotency)

Task는 **멱등성**을 갖도록 설계하는 것이 매우 중요합니다. 멱등성이란 동일한 입력으로 Task를 여러 번 실행해도 항상 결과가 같은 특성을 의미합니다. 네트워크 문제나 워커의 비정상 종료로 인해 Task가 중복 실행될 수 있기 때문입니다.

- **나쁜 예:** `balance = balance + 10` (실행할 때마다 결과가 바뀜)
- **좋은 예:** `balance = get_latest_balance(); set_balance(balance + 10)` 또는 `UPDATE balance SET amount = 110 WHERE user_id = 1` (여러 번 실행해도 최종 결과는 동일)

#### 4.2.2. Task 인자 전달

Task를 호출할 때 **Django 모델 인스턴스 자체를 인자로 전달해서는 안 됩니다.** Task가 큐에서 대기하는 동안, 웹 프로세스에서 전달한 모델 객체의 상태와 워커가 실제로 Task를 실행하는 시점의 데이터베이스 상태가 달라질 수 있기 때문입니다.

- **나쁜 예:** `send_email_to_user.delay(user_object)`
- **좋은 예:** `send_email_to_user.delay(user_id=user.id)`

Task 내부에서 `user_id`를 받아 `User.objects.get(id=user_id)`와 같이 최신 상태의 객체를 직접 조회해서 사용해야 합니다.

#### 4.2.3. 에러 처리 및 자동 재시도

외부 API 호출이나 네트워크 불안정 등 일시적인 오류는 Task를 실패시킬 수 있습니다. 이런 경우, 즉시 실패 처리하는 대신 잠시 후 자동으로 재시도하도록 만들면 시스템의 안정성을 크게 높일 수 있습니다.

`bind=True` 옵션은 Task 함수가 `self` 인자를 통해 Task 인스턴스 자체에 접근할 수 있게 해줍니다. 이를 통해 `self.retry()` 메서드를 호출할 수 있습니다.

```python
# myapp/tasks.py
from proj.celery import app
import requests

@app.task(bind=True, max_retries=3, default_retry_delay=60) # 최대 3번, 60초 간격으로 재시도
def call_external_api(self, url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # 2xx 상태 코드가 아니면 예외 발생
        return response.json()
    except requests.exceptions.RequestException as exc:
        # 예외(exc)를 전달하여 Celery가 에러를 로깅하고, 재시도 횟수를 관리하도록 함
        print(f"API call failed, retrying in {self.default_retry_delay}s...")
        raise self.retry(exc=exc)
```

### 4.3. Task 실행 및 결과 확인

- **실행:**
  - `my_task.delay(arg1, arg2)`: 인자를 직접 전달하여 실행하는 가장 간단한 방법.
  - `my_task.apply_async(args=[arg1, arg2], kwargs={'key': 'value'}, countdown=10)`: 10초 후에 실행하는 등 다양한 옵션을 제공.

- **결과 확인:** `delay()`나 `apply_async()`는 `AsyncResult` 객체를 반환합니다. 이 객체의 `id` (Task ID)를 저장해두면 나중에 결과를 조회할 수 있습니다.

  ```python
  # views.py
  from .tasks import add
  from celery.result import AsyncResult

  def start_task(request):
      task_result = add.delay(10, 20)
      return JsonResponse({"task_id": task_result.id})

  def check_task_result(request, task_id):
      result = AsyncResult(task_id)
      response_data = {
          'task_id': task_id,
          'state': result.state,
          'result': result.result if result.successful() else str(result.info), # 성공 시 결과, 실패 시 에러 정보
      }
      return JsonResponse(response_data)
  ```
  **주의:** `result.get()` 메서드는 Task가 완료될 때까지 현재 프로세스를 블로킹하므로, 동기적인 HTTP 요청-응답 사이클 내에서 직접 호출하는 것은 피해야 합니다. 대신 위와 같이 별도의 API로 상태를 조회하는 방식을 사용해야 합니다.

### 4.4. 주기적인 Task 실행 (Celery Beat)

`settings.py`에 `CELERY_BEAT_SCHEDULE`을 정의하여 주기적인 작업을 스케줄링합니다.

```python
# proj/settings.py
from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    # 30초마다 실행
    'add-every-30-seconds': {
        'task': 'myapp.tasks.add',
        'schedule': 30.0,
        'args': (16, 16),
    },
    # 매일 아침 7시 30분에 실행
    'send-daily-summary-email': {
        'task': 'myapp.tasks.send_summary',
        'schedule': crontab(hour=7, minute=30),
    },
}
```
Celery Beat를 실행하려면 별도의 워커와 분리된 프로세스가 필요합니다: `celery -A proj beat -l info`

## 5. Celery를 활용한 AI 작업 처리

모델 학습이나 대규모 데이터에 대한 일괄 추론과 같이 수 분에서 수 시간이 걸리는 AI 관련 작업은 Celery를 활용하기에 가장 적합한 사례입니다.

### 5.1. 모델 학습 비동기화 및 진행 상태 보고

사용자가 UI에서 버튼을 클릭하여 모델 재학습을 시작하는 시나리오를 가정해 보겠습니다. 이 작업은 매우 오래 걸리므로, 사용자에게 작업이 시작되었음을 알리고 진행 상태를 시각적으로 보여주는 것이 중요합니다.

**1단계: 진행 상태를 업데이트하는 Task 작성**
`bind=True`를 사용하여 Task 인스턴스에 접근하고, `self.update_state()`를 호출하여 현재 상태를 `meta` 데이터와 함께 Result Backend에 저장합니다.

```python
# myapp/tasks.py
from proj.celery import app
import time

@app.task(bind=True)
def train_model_task(self, user_id, dataset_id):
    total_epochs = 100
    model = ... # 모델 초기화
    dataset = ... # 데이터셋 로드

    for epoch in range(total_epochs):
        # 모델 학습 로직
        time.sleep(2) # 1 에폭 학습에 2초가 걸린다고 가정
        
        # 진행 상태 업데이트
        self.update_state(
            state='PROGRESS',
            meta={
                'current_epoch': epoch + 1,
                'total_epochs': total_epochs,
                'percent_complete': ((epoch + 1) / total_epochs) * 100,
            }
        )
    
    # 최종 결과 저장
    final_accuracy = 0.95 
    # model.save(...)
    
    return {'status': 'completed', 'accuracy': final_accuracy}
```

**2단계: Task를 시작하고 상태를 조회하는 API 작성**

- **Task 시작 API:**
  ```python
  # myapp/views.py
  from rest_framework.views import APIView
  from rest_framework.response import Response
  from .tasks import train_model_task

  class StartTrainingView(APIView):
      def post(self, request, *args, **kwargs):
          user_id = request.user.id
          dataset_id = request.data.get('dataset_id')
          
          # Task를 비동기적으로 실행하고 task_id 반환
          task = train_model_task.delay(user_id, dataset_id)
          
          return Response({"task_id": task.id}, status=202)
  ```

- **Task 상태 조회 API:**
  ```python
  # myapp/views.py
  from celery.result import AsyncResult
  
  class TaskStatusView(APIView):
      def get(self, request, task_id, *args, **kwargs):
          task_result = AsyncResult(task_id)
          
          response_data = {
              'task_id': task_id,
              'state': task_result.state,
              'details': task_result.info, # update_state()의 meta 데이터
          }
          return Response(response_data)
  ```
이제 프론트엔드에서는 `StartTrainingView`를 호출하여 학습을 시작하고 `task_id`를 받은 뒤, `TaskStatusView`를 주기적으로(예: 2초마다) 폴링(polling)하여 `details.percent_complete` 값을 가져와 프로그레스 바를 업데이트할 수 있습니다.

### 5.2. 대규모 추론 작업과 결과 처리

수백만 건의 데이터에 대한 일괄 추론 역시 비슷한 패턴을 따릅니다. Task가 완료된 후, 결과의 크기가 매우 크다면 Result Backend에 직접 저장하는 것은 비효율적입니다. 대신, 결과를 파일(CSV, JSON 등)로 만들어 클라우드 스토리지(S3 등)에 업로드하고, 데이터베이스에는 파일의 경로와 요약 정보만 저장하는 것이 좋습니다.

```python
# myapp/tasks.py
@app.task(bind=True)
def batch_inference_task(self, data_id):
    # 1. 데이터 로드
    data = load_large_data_from_db(data_id)
    
    # 2. 모델 추론
    results = model.predict(data)
    
    # 3. 결과 파일을 S3에 업로드
    result_file_url = save_results_to_s3(results)
    
    # 4. DB에 결과 정보 업데이트
    update_inference_job_in_db(data_id, status='COMPLETED', result_url=result_file_url)
    
    return {'result_file_url': result_file_url}
```
이러한 패턴을 통해 Django와 Celery는 무거운 AI 작업을 안정적으로 처리하고, 사용자에게 좋은 경험을 제공하는 강력한 백엔드 시스템을 구축할 수 있습니다.

## 6. Celery 모니터링 및 관리 (Flower)

Celery 시스템이 복잡해지고 운영 환경에 배포되면, 워커(Worker)의 상태를 추적하고, Task의 실행 현황을 시각적으로 확인하며, 문제가 발생했을 때 이를 관리할 수 있는 도구가 필수적입니다. **Flower**는 Celery를 위한 가장 대표적인 실시간 웹 기반 모니터링 도구입니다.

### 6.1. Flower 설치 및 실행

Flower는 Celery와 별도로 설치하고 실행합니다.

```bash
# 1. Flower 설치
pip install flower

# 2. Flower 실행
# -A: Celery 앱 지정, --port: 웹 UI 포트 지정
# Docker 환경에서는 Django/Celery 컨테이너와 동일한 네트워크에서 실행해야 합니다.
flower -A proj --port=5555
```
이제 웹 브라우저에서 `http://localhost:5555`로 접속하면 Flower 대시보드를 볼 수 있습니다.

### 6.2. Flower 대시보드 활용하기

Flower 대시보드는 여러 탭으로 구성되어 있으며, 각 탭에서 Celery 시스템의 다양한 정보를 확인할 수 있습니다.

- **Dashboard:**
  - 전체 Task의 상태(성공, 실패, 진행 중 등)를 파이 그래프로 한눈에 볼 수 있습니다.
  - 현재 활성화된 워커의 수와 처리된 Task의 총 개수를 확인할 수 있습니다.

- **Workers 탭:**
  - 현재 실행 중인 모든 워커의 상세 정보를 보여줍니다.
  - 각 워커의 상태(Online/Offline), 처리한 Task 수, 현재 동시성(concurrency) 설정 등을 확인할 수 있습니다.
  - 워커를 선택하여 워커 풀(worker pool)을 재시작하거나, 동시성을 조절하고, 특정 큐(queue) 추가/삭제 등 고급 관리 작업을 수행할 수 있습니다.

- **Tasks 탭:**
  - 모든 Task의 실행 기록을 최신순으로 보여줍니다.
  - 각 Task의 이름, UUID, 상태, 인자(arguments), 결과(result), 실행 시간 등을 상세하게 볼 수 있습니다.
  - Task ID를 클릭하면 해당 Task의 더 자세한 정보(재시도 기록, 예외 정보 등)를 확인할 수 있습니다.
  - **가장 강력한 기능 중 하나로, 이 페이지에서 기존 Task와 동일한 인자로 Task를 다시 실행(re-run)하거나, 아직 실행되지 않은 Task를 취소(revoke)할 수 있습니다.**

- **Monitor 탭:**
  - 시간 경과에 따른 성공/실패 Task의 수, 활성 워커의 수 등을 그래프로 보여주어 시스템의 전반적인 부하와 상태 변화를 모니터링하는 데 유용합니다.

### 6.3. 운영 환경에서의 Flower 보안

운영 환경에서 Flower를 외부에 노출할 경우, 누구나 Task 정보(민감한 인자 포함)를 보거나 워커를 제어할 수 있으므로 반드시 인증을 설정해야 합니다.

- **Basic Authentication 설정:**
  ```bash
  flower -A proj --port=5555 --basic_auth=user:password
  ```
- **OAuth 2.0 연동 (Google 등):**
  ```bash
  flower -A proj --auth=google --auth_provider=google --client_id=... --client_secret=... --redirect_uri=...
  ```

Flower를 통해 Celery 시스템을 투명하게 관리하고 신속하게 문제를 해결함으로써, Django 기반의 비동기 AI 서비스를 더욱 안정적으로 운영할 수 있습니다.