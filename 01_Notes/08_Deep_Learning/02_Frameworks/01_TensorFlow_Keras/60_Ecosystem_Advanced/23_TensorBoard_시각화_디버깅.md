<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 TensorBoard를 활용하여 TensorFlow 및 Keras 모델의 학습 과정을 시각화하고 디버깅하며 최적화하는 방법을 상세히 다룹니다. TensorBoard의 개요, Keras 연동, 학습 과정 모니터링(Scalars, Graphs, Histograms), TensorFlow 그래프 시각화, 그리고 TensorFlow Profiler를 이용한 성능 분석 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. TensorBoard를 활용한 Keras 모델 시각화 및 디버깅](#1-tensorboard를-활용한-keras-모델-시각화-및-디버깅)
  - [1.1. TensorBoard 개요 및 Keras 연동](#11-tensorboard-개요-및-keras-연동)
  - [1.2. Keras 모델 학습 과정 모니터링](#12-keras-모델-학습-과정-모니터링)
  - [1.3. Keras 모델 그래프 시각화](#13-keras-모델-그래프-시각화)
  - [1.4. Keras Profiler를 이용한 성능 분석](#14-keras-profiler를-이용한-성능-분석)

---

## 1. TensorBoard를 활용한 Keras 모델 시각화 및 디버깅

TensorBoard는 TensorFlow와 Keras 모델의 학습 과정을 시각화하고 디버깅하며 최적화하는 데 필수적인 도구입니다.

### 1.1. TensorBoard 개요 및 Keras 연동

*   **개념:** 학습 과정의 다양한 메트릭, 모델 그래프, 이미지, 오디오, 텍스트, 프로파일링 데이터 등을 시각화하여 보여주는 웹 기반 대시보드입니다.
*   **Keras 연동:** `keras.callbacks.TensorBoard` 콜백을 사용하여 Keras 모델 학습 시 자동으로 TensorBoard 로그를 생성할 수 있습니다.
*   **실무 관점:**
    *   **설치 및 실행:** `pip install tensorboard`로 설치하고, `tensorboard --logdir /path/to/logs` 명령어로 실행합니다.
    *   **로그 디렉토리:** 각 실험마다 별도의 로그 디렉토리를 사용하여 여러 실험 결과를 쉽게 비교할 수 있도록 관리하는 것이 중요합니다.
    *   **예시:**

```python
import keras
import numpy as np
import datetime

# 더미 데이터
x_train = np.random.rand(100, 784).astype("float32")
y_train = np.random.randint(0, 10, size=(100,)).astype("int32")

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
# 터미널에서 `tensorboard --logdir logs/fit` 실행
```

### 1.2. Keras 모델 학습 과정 모니터링

TensorBoard의 Scalars 대시보드를 통해 학습 과정의 주요 지표들을 실시간으로 모니터링할 수 있습니다.

*   **개념:** 손실(loss), 정확도(accuracy), 학습률(learning rate) 등 시간에 따라 변하는 단일 숫자 값을 그래프로 시각화합니다.
*   **실무 관점:**
    *   **과적합/과소적합 진단:** 학습 손실과 검증 손실의 추이를 비교하여 모델의 과적합 또는 과소적합 여부를 판단합니다.
    *   **학습 안정성:** 손실 값의 변동성을 확인하여 학습이 안정적으로 진행되는지 파악합니다.
    *   **하이퍼파라미터 튜닝:** 여러 실험의 스칼라 그래프를 겹쳐서 비교하여 최적의 하이퍼파라미터를 찾을 수 있습니다.
    *   **`histogram_freq`:** `TensorBoard` 콜백의 `histogram_freq` 인자를 설정하여 가중치, 편향, 활성화 값의 분포 변화를 히스토그램으로 시각화할 수 있습니다. 이는 기울기 소실/폭주 문제나 Dying ReLU와 같은 문제를 진단하는 데 유용합니다.

### 1.3. Keras 모델 그래프 시각화

TensorBoard의 Graphs 대시보드는 Keras 모델의 내부 구조를 시각적으로 탐색할 수 있게 합니다.

*   **개념:** Keras 모델의 레이어 구성, 연결 관계, 데이터 흐름을 노드와 엣지로 표현된 계산 그래프 형태로 보여줍니다.
*   **실무 관점:**
    *   **모델 이해:** 복잡한 신경망의 아키텍처를 직관적으로 이해하고, 각 레이어의 역할과 연결 방식을 파악하는 데 도움을 줍니다.
    *   **디버깅:** 예상치 못한 연결이나 누락된 연산을 찾아내어 모델 정의 오류를 디버깅할 수 있습니다. 특히 Functional API나 Subclassing API로 구축된 복잡한 모델의 구조를 검증할 때 유용합니다.
    *   **`tf.function`:** `tf.function`으로 래핑된 Custom Training Loop나 Custom Layer의 내부 그래프도 시각화하여 연산 흐름을 분석할 수 있습니다.

### 1.4. Keras Profiler를 이용한 성능 분석

TensorBoard의 Profile 대시보드는 Keras 모델 학습 및 추론 과정의 성능 병목 현상을 진단하고 최적화하는 데 사용되는 강력한 도구입니다.

*   **개념:** CPU, GPU 등 다양한 디바이스에서 Keras/TensorFlow 연산의 실행 시간, 메모리 사용량, 디바이스 간 통신 등을 상세하게 기록하고 분석합니다.
*   **실무 관점:**
    *   **병목 현상 식별:** 어떤 연산이 가장 많은 시간을 소모하는지, CPU와 GPU 간의 데이터 전송이 비효율적인지 등을 파악하여 최적화 포인트를 찾습니다.
    *   **GPU 활용률 분석:** GPU가 얼마나 효율적으로 사용되고 있는지 (예: GPU 유휴 시간, Tensor Core 활용률)를 확인하여 학습 속도 향상 방안을 모색합니다.
    *   **메모리 분석:** 각 연산이 사용하는 메모리 양을 분석하여 메모리 부족 문제를 해결하거나 메모리 사용량을 최적화합니다.
    *   **데이터 파이프라인 분석:** `tf.data` 파이프라인의 각 단계에서 데이터가 얼마나 빠르게 준비되고 모델에 공급되는지 분석하여 입력 파이프라인의 병목을 해결합니다.
    *   **사용법:** `keras.callbacks.TensorBoard` 콜백에 `profile_batch` 인자를 설정하거나, `tf.profiler.experimental.start()` 및 `tf.profiler.experimental.stop()` 함수를 사용하여 프로파일링을 시작하고 중지할 수 있습니다.
