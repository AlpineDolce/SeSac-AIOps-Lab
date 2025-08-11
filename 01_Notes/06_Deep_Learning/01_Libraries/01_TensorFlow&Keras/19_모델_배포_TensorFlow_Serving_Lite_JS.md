<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 학습된 딥러닝 모델을 실제 서비스 환경에 배포하기 위한 TensorFlow의 다양한 도구들을 상세히 다룹니다. 프로덕션 환경을 위한 TensorFlow Serving, 모바일 및 엣지 디바이스를 위한 TensorFlow Lite, 그리고 웹 브라우저 환경을 위한 TensorFlow.js를 이용한 모델 배포 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 모델 배포 및 MLOps](#1-모델-배포-및-mlops)
  - [1.1. Keras 모델을 TensorFlow Serving, Lite, TF.js 등으로 내보내기](#11-keras-모델을-tensorflow-serving-lite-tfjs-등으로-내보내기)
    - [1.1.1. TensorFlow Serving](#111-tensorflow-serving)
    - [1.1.2. TensorFlow Lite](#112-tensorflow-lite)
    - [1.1.3. TensorFlow.js](#113-tensorflowjs)

---

## 1. 모델 배포 및 MLOps

모델 학습만큼이나 중요한 것이 학습된 모델을 실제 서비스 환경에 배포하고 운영하는 것입니다. MLOps(Machine Learning Operations)는 머신러닝 모델의 개발부터 배포, 운영, 모니터링까지 전체 라이프사이클을 자동화하고 관리하는 방법론입니다. TensorFlow는 MLOps를 위한 다양한 도구와 프레임워크를 제공합니다.

### 1.1. Keras 모델을 TensorFlow Serving, Lite, TF.js 등으로 내보내기

Keras 모델은 TensorFlow의 백엔드를 사용하므로, TensorFlow의 다양한 배포 도구와 호환됩니다.

#### 1.1.1. TensorFlow Serving:
*   **개념:** 프로덕션 환경에서 고성능으로 모델을 서빙하기 위한 시스템입니다. Keras 모델을 SavedModel 형식으로 저장하여 TensorFlow Serving에 배포합니다.
*   **실무 관점:** `model.save('path/to/saved_model', save_format='tf')`를 사용하여 SavedModel로 내보냅니다. TensorFlow Serving은 모델 버전 관리, 배치 처리 등을 지원하여 대규모 서비스에 적합합니다.
    *   **배포 다음 단계:** SavedModel로 저장된 모델은 Docker 컨테이너나 Kubernetes 클러스터를 통해 TensorFlow Serving으로 배포할 수 있습니다. 예를 들어, Docker를 사용하여 로컬에서 모델을 서빙하는 명령어는 다음과 같습니다.
        ```bash
        # SavedModel이 /tmp/my_model/1 에 저장되어 있다고 가정
        # (1은 모델 버전 번호)
        docker run -p 8501:8501 --name tfserving_model \
          --mount type=bind,source=/tmp/my_model,target=/models/my_model \
          -e MODEL_NAME=my_model -t tensorflow/serving &
        
        # 모델 추론 요청 예시 (Python requests)
        # import requests
        # import json
        # data = json.dumps({"signature_name": "serving_default", "instances": input_data.tolist()})
        # headers = {"content-type": "application/json"}
        # json_response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data, headers=headers)
        # predictions = json.loads(json_response.text) 
        ```

#### 1.1.2. TensorFlow Lite:
*   **개념:** 모바일, 임베디드, 엣지 디바이스에서 머신러닝 모델을 실행하기 위한 경량 솔루션입니다. Keras 모델을 `.tflite` 형식으로 변환하여 배포합니다.
*   **실무 관점:** `tf.lite.TFLiteConverter`를 사용하여 Keras 모델을 `.tflite` 형식으로 변환합니다. 이 과정에서 양자화(Quantization)를 적용하여 모델 크기를 줄이고 추론 속도를 높일 수 있습니다. 온디바이스 ML에 필수적입니다.
    *   **변환 및 배포 다음 단계:**
        ```python
        import tensorflow as tf
        # model은 학습된 Keras 모델
        # converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # tflite_model = converter.convert()
        # with open('model.tflite', 'wb') as f:
        #     f.write(tflite_model)
        ```
        변환된 `.tflite` 모델은 Android, iOS, Raspberry Pi 등 다양한 플랫폼의 애플리케이션에 통합하여 사용할 수 있습니다. 각 플랫폼별 TensorFlow Lite 인터프리터 API를 사용하여 모델을 로드하고 추론을 실행합니다.

#### 1.1.3. TensorFlow.js:
*   **개념:** 웹 브라우저나 Node.js 환경에서 머신러닝 모델을 개발하고 실행하기 위한 JavaScript 라이브러리입니다. Keras 모델을 TensorFlow.js 형식으로 변환하여 웹 애플리케이션에 포함합니다.
*   **실무 관점:** `tensorflowjs_converter` 도구를 사용하여 Keras 모델을 TensorFlow.js 형식으로 변환합니다. 클라이언트 측 ML을 통해 개인 정보 보호, 낮은 지연 시간, 서버 부하 감소 등의 이점을 얻을 수 있습니다.
    *   **변환 및 배포 다음 단계:**
        ```bash
        # Keras 모델을 SavedModel 형식으로 먼저 저장
        # model.save('my_tf_model', save_format='tf')
        
        # TensorFlow.js 형식으로 변환
        # pip install tensorflowjs
        # tensorflowjs_converter --input_format=tf_saved_model \
        #                        --output_node_names='output_node_name' \
        #                        --output_format=tfjs_graph_model \
        #                        ./my_tf_model ./web_model
        ```
        변환된 모델은 웹 페이지에서 `<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>`를 통해 TensorFlow.js 라이브러리를 로드한 후, JavaScript 코드로 모델을 로드하고 추론을 실행할 수 있습니다.
