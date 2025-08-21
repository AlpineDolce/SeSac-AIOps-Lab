<h2>Django Backend: AI 모델 서빙 심화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 AI 모델의 효율적인 로딩, 버전 관리, 추론 최적화 및 서빙 아키텍처(예: ONNX Runtime, TensorFlow Serving 연동)를 심층적으로 다루는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. AI 모델 서빙 개요](#1-ai-모델-서빙-개요)
- [2. 효율적인 모델 로딩 및 관리](#2-효율적인-모델-로딩-및-관리)
  - [2.1. 모델 로딩 전략](#21-모델-로딩-전략)
  - [2.2. 모델 버전 관리](#22-모델-버전-관리)
- [3. 추론 최적화 기법](#3-추론-최적화-기법)
  - [3.1. 하드웨어 가속 (GPU)](#31-하드웨어-가속-gpu)
  - [3.2. 배치 처리 (Batch Processing)](#32-배치-처리-batch-processing)
  - [3.3. 모델 경량화 (양자화, 가지치기)](#33-모델-경량화-양자화-가지치기)
- [4. AI 모델 서빙 아키텍처 연동](#4-ai-모델-서빙-아키텍처-연동)
  - [4.1. Django와 직접 통합](#41-django와-직접-통합)
  - [4.2. 마이크로서비스 아키텍처 (TensorFlow Serving, ONNX Runtime)](#42-마이크로서비스-아키텍처-tensorflow-serving-onnx-runtime)
  - [4.3. API 게이트웨이 활용](#43-api-게이트웨이-활용)

---

## 1. AI 모델 서빙 개요

AI 모델 서빙은 학습된 머신러닝 모델을 실제 서비스 환경에서 사용자 요청에 따라 추론(Inference)을 수행하고 결과를 반환하는 과정입니다. 이는 단순히 모델 파일을 로드하는 것을 넘어, 모델의 효율적인 관리, 빠른 추론 속도, 높은 확장성, 안정적인 운영 등을 고려해야 하는 복잡한 과정입니다. 특히 Django와 같은 웹 프레임워크를 통해 AI 모델을 서빙할 때는 웹 서비스의 특성(동시 요청 처리, 응답 시간)과 AI 모델의 특성(높은 연산량, 메모리 사용량)을 모두 고려한 설계가 필요합니다.

## 2. 효율적인 모델 로딩 및 관리

AI 모델은 크기가 수백 MB에서 수 GB에 달하며 로딩에 수십 초가 걸릴 수 있습니다. 매 요청마다 모델을 로드하는 것은 비효율적이므로, 애플리케이션의 생명주기에 맞춰 모델을 관리하는 전략이 필수적입니다.

### 2.1. 모델 로딩 전략: 사전 로딩 (Pre-loading)

가장 일반적이고 권장되는 방식은 Django 애플리케이션이 시작될 때 모델을 미리 메모리에 로드하여 전역 변수처럼 사용하는 것입니다. 이를 통해 첫 요청의 지연 시간을 없애고 일관된 응답 속도를 보장할 수 있습니다.

**구현 단계:**

**1단계: 모델 관리 모듈 생성**
모델 로딩 및 캐싱을 전담하는 모듈(예: `ml_models/registry.py`)을 만듭니다.

```python
# ml_models/registry.py
import os
from django.conf import settings
from tensorflow.keras.models import load_model

class ModelRegistry:
    def __init__(self):
        self._models = {}

    def load(self, model_name: str):
        if model_name not in self._models:
            model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'files', f'{model_name}.h5')
            try:
                # 스레드 충돌을 방지하기 위해 모델 로드 후 세션 초기화
                # from tensorflow.keras import backend as K
                # K.clear_session()
                model = load_model(model_path)
                self._models[model_name] = model
                print(f"Model '{model_name}' loaded successfully.")
            except Exception as e:
                print(f"Error loading model '{model_name}': {e}")
                self._models[model_name] = None
        return self._models.get(model_name)

    def get(self, model_name: str):
        return self._models.get(model_name)

# 전역 인스턴스 생성
model_registry = ModelRegistry()
```

**2단계: Django 앱 시작 시 모델 로드 (`apps.py`)**
Django의 `AppConfig`를 사용하여 앱이 준비되는 시점에 모델을 로드합니다.

```python
# ml_models/apps.py
from django.apps import AppConfig

class MlModelsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ml_models'

    def ready(self):
        # 이 메서드는 Django 서버가 시작될 때 한 번만 호출됩니다.
        from .registry import model_registry
        
        # 로드할 모델 목록
        models_to_load = ['sentiment_analyzer_v1', 'image_classifier_v2']
        for model_name in models_to_load:
            model_registry.load(model_name)
```
`__init__.py`에서 `default_app_config`를 설정하거나 `settings.py`의 `INSTALLED_APPS`에 `MlModelsConfig`를 직접 명시해야 `ready()`가 호출됩니다.

### 2.2. 데이터베이스를 이용한 모델 버전 관리

모델이 계속 업데이트되는 환경에서는 어떤 버전의 모델을 서빙할지 동적으로 결정하는 기능이 필요합니다. Django 모델을 사용하여 모델의 메타데이터를 관리하면 이 과정을 체계적으로 처리할 수 있습니다.

**1단계: 모델 메타데이터를 위한 Django 모델 정의**

```python
# ml_models/models.py
from django.db import models

class MLModel(models.Model):
    class Meta:
        verbose_name = "ML Model"
        verbose_name_plural = "ML Models"

    name = models.CharField(max_length=100, unique=True) # 예: sentiment_analyzer
    version = models.CharField(max_length=50) # 예: v1.2.0
    description = models.TextField(blank=True)
    
    # 모델 파일은 미디어 스토리지(S3 등)에 저장하는 것을 가정
    model_file = models.FileField(upload_to='ml_models/')
    
    is_active = models.BooleanField(default=False) # 현재 서빙에 사용할 모델인지 여부
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.version}"
```

**2단계: 활성화된 모델을 동적으로 로드**
`ready()` 메서드를 수정하여, DB에서 `is_active=True`로 설정된 모델만 로드하도록 변경합니다.

```python
# ml_models/apps.py
from django.apps import AppConfig

class MlModelsConfig(AppConfig):
    # ... (이전 내용) ...
    def ready(self):
        from .registry import model_registry
        from .models import MLModel

        # DB에서 현재 활성화된 모델 목록을 가져옴
        active_models = MLModel.objects.filter(is_active=True)
        
        for model_record in active_models:
            # model_record.model_file.path 를 사용하여 모델 로드
            # (실제로는 S3 등에서 파일을 다운로드하는 로직이 필요할 수 있음)
            print(f"Loading active model: {model_record.name} version {model_record.version}")
            # model_registry.load(...) 
        
        # 관리자 페이지에서 is_active 플래그만 변경하면,
        # 서버 재시작 시 새로운 버전의 모델이 자동으로 로드됩니다.
```
이 방식을 사용하면 코드 변경이나 재배포 없이 Django Admin 페이지만으로 서빙할 모델 버전을 안전하게 교체할 수 있습니다.

## 3. 추론 최적화 기법

모델 추론 속도는 AI 서비스의 사용자 경험에 직접적인 영향을 미치며, 서버 비용과도 직결됩니다. 다양한 최적화 기법을 통해 추론 성능을 향상시킬 수 있습니다.

### 3.1. 하드웨어 가속 (GPU)

딥러닝 모델의 행렬 연산은 대규모 병렬 처리에 특화된 GPU에서 CPU보다 훨씬 빠르게 수행됩니다.
- **설정:** 서빙 서버에 NVIDIA 드라이버, CUDA Toolkit, cuDNN 라이브러리를 설치하고, `tensorflow-gpu` 또는 `pytorch`의 CUDA 버전을 설치하면 프레임워크가 자동으로 GPU를 인식하고 사용합니다.
- **고려사항:** GPU 서버는 비용이 높으므로, 트래픽 대비 효용성을 충분히 검토해야 합니다. 모든 모델이 GPU에서 효율적인 것은 아니며, 모델 크기와 연산 복잡도에 따라 효과가 달라집니다.

### 3.2. 배치 처리 (Batch Processing)

GPU는 여러 개의 입력을 한 번에 처리(배치)할 때 최고의 성능을 보입니다. 단일 요청을 개별적으로 처리하는 대신, 짧은 시간 동안 들어온 여러 요청을 모아 배치로 만들어 한 번에 추론하면 전체 처리량(throughput)을 극대화할 수 있습니다.

**구현 아키텍처 예시 (Celery + Redis 활용):**

1.  **요청 접수:** Django 뷰는 사용자 요청을 받으면 직접 추론하지 않고, 입력 데이터를 Redis 같은 빠른 메시지 큐에 넣은 후 사용자에게는 작업 ID를 즉시 반환합니다.
2.  **비동기 작업 실행:** Celery 워커(worker)는 주기적으로(예: 매 100ms 또는 큐에 16개 이상 쌓이면) 큐에서 여러 개의 요청 데이터를 가져와 배치(batch)를 만듭니다.
3.  **배치 추론:** Celery 워커는 생성된 배치를 AI 모델에 전달하여 한 번에 추론을 수행합니다.
4.  **결과 저장:** 추론이 완료되면, 각 요청의 결과는 작업 ID를 키로 하여 Redis나 데이터베이스에 저장됩니다.
5.  **결과 조회:** 사용자는 처음에 받은 작업 ID를 가지고 별도의 API 엔드포인트를 통해 추론 결과를 조회합니다.

이 방식은 실시간 응답이 필수는 아니지만, 전체 처리량이 중요한 서비스(예: 이미지 분석, 텍스트 번역)에 매우 효과적입니다.

### 3.3. 모델 경량화 (Model Optimization)

모델의 구조나 가중치를 변경하여 크기를 줄이고 연산 속도를 높이는 기법입니다.

#### 3.3.1. 양자화 (Quantization)

모델의 가중치를 표현하는 데이터 타입의 정밀도를 낮추는 기술입니다. (예: 32비트 부동소수점 -> 8비트 정수). 모델 크기가 약 1/4로 줄고, CPU/GPU 및 전용 하드웨어(TPU, NPU)에서 연산 속도가 크게 향상됩니다. 약간의 성능 저하가 발생할 수 있습니다.

**TensorFlow Lite를 이용한 Post-Training Quantization 예제:**
```python
import tensorflow as tf

# 기존 Keras 모델 로드
model = tf.keras.models.load_model('my_model.h5')

# TFLite 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 양자화 옵션 설정 (Dynamic Range Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 모델 변환
tflite_quant_model = converter.convert()

# 변환된 모델 저장
with open('model_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)
```

#### 3.3.2. 가지치기 (Pruning) 및 지식 증류 (Knowledge Distillation)

- **가지치기 (Pruning):** 모델 성능에 거의 영향을 주지 않는 작은 가중치들을 0으로 만들어 제거하는 기법입니다. 모델의 복잡도를 낮추고 연산량을 줄일 수 있습니다.
- **지식 증류 (Knowledge Distillation):** 크고 복잡하지만 성능이 좋은 '교사 모델'의 예측 결과를, 작고 빠른 '학생 모델'이 모방하도록 학습시키는 방법입니다. 학생 모델은 교사 모델의 "지식"을 증류받아, 작은 크기에도 불구하고 높은 성능을 낼 수 있습니다.

## 4. AI 모델 서빙 아키텍처 연동

Django와 AI 모델을 통합하는 방법은 서비스의 규모, 복잡성, 요구사항에 따라 달라집니다. 각 아키텍처의 장단점을 이해하고 실제 코드 수준에서 어떻게 연동하는지 살펴보겠습니다.

### 4.1. 아키텍처 1: Django와 직접 통합 (Monolithic)

가장 간단한 방식으로, Django 애플리케이션 내에서 직접 AI 모델을 로드하고 추론 API를 제공합니다.

- **적합한 경우:**
  - 프로토타입 또는 소규모 서비스
  - AI 추론이 웹 서비스의 핵심 로직과 매우 밀접하게 결합된 경우
  - 별도의 인프라를 관리할 여력이 없는 경우
- **장점:** 구현이 간단하고 빠르며, 모든 로직이 한 곳에 있어 관리가 용이합니다.
- **단점:** AI 추론이 웹 서버의 CPU/Memory를 많이 사용하여 웹 요청 처리에 영향을 줄 수 있습니다. 모델이나 의존성 업데이트가 전체 서비스의 재배포로 이어집니다.

**구현 예시 (`views.py`):**
```python
# ml_models/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import numpy as np
from .registry import model_registry # 2.1에서 만든 모델 레지스트리

class SentimentAnalysisView(APIView):
    def post(self, request, *args, **kwargs):
        # 입력 데이터 유효성 검사 (Serializer 사용 권장)
        text_data = request.data.get('text')
        if not text_data:
            return Response({"error": "Text data is required."}, status=status.HTTP_400_BAD_REQUEST)

        # 사전 로드된 모델 가져오기
        model = model_registry.get('sentiment_analyzer_v1')
        if model is None:
            return Response({"error": "Model not available."}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        try:
            # 모델에 맞는 입력 형태로 전처리
            # (예시: 토크나이징, 패딩 등)
            processed_input = np.array([text_data]) # 실제로는 더 복잡한 전처리 필요

            # 추론 수행
            prediction = model.predict(processed_input)
            
            # 결과 후처리
            sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
            
            return Response({"sentiment": sentiment, "score": float(prediction[0][0])})

        except Exception as e:
            # 로깅 필수
            return Response({"error": f"An error occurred during inference: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
```

### 4.2. 아키텍처 2: 전용 서빙 엔진 연동 (Microservice)

AI 모델 서빙을 전담하는 별도의 마이크로서비스를 구축하고, Django는 이 서비스에 API 요청을 보내 추론 결과를 받는 방식입니다.

- **적합한 경우:**
  - 대규모 트래픽을 처리해야 하는 프로덕션 서비스
  - 웹 서비스와 AI 모델의 개발/배포 주기가 다른 경우
  - 다양한 종류의 모델을 독립적으로 확장하고 관리해야 할 때
- **장점:** 웹 서버와 추론 서버의 리소스가 분리되어 안정성이 높습니다. 모델의 독립적인 배포, 버전 관리, A/B 테스트가 용이합니다.
- **단점:** 아키텍처가 복잡해지고 네트워크 지연 시간이 추가됩니다. 별도의 서빙 인프라 구축 및 관리 비용이 발생합니다.

#### 4.2.1. TensorFlow Serving 연동 예시

TensorFlow Serving은 gRPC와 RESTful API 두 가지 엔드포인트를 제공합니다. 여기서는 사용이 간편한 RESTful API를 예로 듭니다.

**Django 뷰 (`views.py`):**
```python
# aiapp/views.py
import requests
import json
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response

class ImageClassificationView(APIView):
    def post(self, request, *args, **kwargs):
        # TF Serving 서버 주소 (환경 변수로 관리)
        TF_SERVING_URL = settings.TF_SERVING_URL # 예: "http://localhost:8501/v1/models/image_classifier:predict"
        
        # 이미지 데이터 전처리 (예시)
        image_data = request.data.get('image_base64')
        # 실제로는 Base64 디코딩, 리사이징, 정규화 등 필요
        processed_input = [image_data] 

        # TF Serving이 요구하는 JSON 형식으로 데이터 구성
        payload = {
            "instances": processed_input
        }

        try:
            # TF Serving에 POST 요청
            response = requests.post(TF_SERVING_URL, data=json.dumps(payload))
            response.raise_for_status() # 2xx 상태 코드가 아니면 예외 발생
            
            predictions = response.json()['predictions']
            return Response(predictions)

        except requests.exceptions.RequestException as e:
            return Response({"error": f"Failed to connect to TF Serving: {e}"}, status=503)
        except Exception as e:
            return Response({"error": f"An error occurred: {e}"}, status=500)
```

#### 4.2.2. ONNX Runtime Server 연동 예시

ONNX Runtime(ORT) 서버도 HTTP/gRPC를 통해 추론 서비스를 제공합니다.

**Django 뷰 (`views.py`):**
```python
# aiapp/views.py
import requests
import json
import numpy as np
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response

class ONNXInferenceView(APIView):
    def post(self, request, *args, **kwargs):
        ORT_SERVER_URL = settings.ORT_SERVER_URL # 예: "http://localhost:8001/v1/models/my_onnx_model/versions/1/run"
        
        # 입력 데이터 전처리
        input_data = np.array(request.data.get('input'), dtype=np.float32)

        # ONNX Runtime 서버가 요구하는 JSON 형식으로 데이터 구성
        payload = {
            "inputs": [
                {
                    "name": "input_name", # ONNX 모델의 입력 텐서 이름
                    "shape": list(input_data.shape),
                    "datatype": "FP32",
                    "data": input_data.flatten().tolist()
                }
            ]
        }

        try:
            response = requests.post(ORT_SERVER_URL, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return Response(result['outputs'])

        except requests.exceptions.RequestException as e:
            return Response({"error": f"Failed to connect to ONNX Runtime Server: {e}"}, status=503)
        except Exception as e:
            return Response({"error": f"An error occurred: {e}"}, status=500)
```

이처럼 아키텍처를 분리하면 Django는 비즈니스 로직, 인증, 데이터 관리에 집중하고, 무거운 AI 추론은 최적화된 전용 서버에 위임하여 확장성 있고 안정적인 시스템을 구축할 수 있습니다.