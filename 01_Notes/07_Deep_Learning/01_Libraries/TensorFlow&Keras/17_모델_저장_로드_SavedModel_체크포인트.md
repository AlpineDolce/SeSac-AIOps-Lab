<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 학습된 딥러닝 모델을 저장하고 필요할 때 다시 로드하는 방법을 상세히 다룹니다. Keras 모델 저장 형식(`.keras`), 가중치만/아키텍처만 저장 및 로드, 그리고 프로덕션 배포를 위한 표준 형식인 TensorFlow SavedModel 형식과 학습 중 모델 상태를 저장하는 체크포인트 관리 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 모델 저장 및 로드](#1-모델-저장-및-로드)
  - [1.1. Keras 모델 저장 형식 (.keras)](#11-keras-모델-저장-형식-keras)
  - [1.2. 가중치만 저장 및 로드](#12-가중치만-저장-및-로드)
  - [1.3. 모델 아키텍처만 저장 및 로드](#13-모델-아키텍처만-저장-및-로드)
  - [1.4. TensorFlow SavedModel 형식으로 내보내기 (배포를 위한 준비)](#14-tensorflow-savedmodel-형식으로-내보내기-배포를-위한-준비)
  - [1.5. 체크포인트 (Checkpoints) 관리](#15-체크포인트-checkpoints-관리)

---

## 1. 모델 저장 및 로드

학습된 모델을 저장하고 필요할 때 다시 로드하는 것은 딥러닝 워크플로우의 필수적인 부분입니다. Keras는 다양한 저장 형식을 제공합니다.

### 1.1. Keras 모델 저장 형식 (.keras)

Keras 3.0부터 `.keras` 확장자는 Keras 모델을 저장하는 권장 형식입니다. 이 형식은 모델의 아키텍처, 가중치, 컴파일 설정(옵티마이저, 손실, 메트릭), 그리고 사용자 정의 객체까지 모두 포함하는 포괄적인 형식입니다.

*   **개념:** `model.save('my_model.keras')`를 사용하여 모델을 저장하고, `keras.models.load_model('my_model.keras')`를 사용하여 로드합니다.
*   **실무 관점:**
    *   **간편성:** 모델의 모든 정보를 한 파일에 저장하므로, 모델을 다른 환경으로 이동하거나 재사용할 때 매우 편리합니다.
    *   **재현성:** 학습된 모델을 완벽하게 재현할 수 있습니다.
    *   **사용자 정의 객체:** Custom Layer, Custom Loss, Custom Metric 등을 포함하는 모델도 이 형식으로 저장하고 로드할 수 있습니다. 이때 `custom_objects` 인자를 `load_model`에 전달해야 할 수 있습니다.
    *   **예시:**

```python
import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 더미 학습
# x_train = np.random.rand(100, 784).astype("float32")
# y_train = np.random.randint(0, 10, size=(100,)).astype("int32")
# model.fit(x_train, y_train, epochs=1)

# 모델 저장
# model.save('my_model.keras')

# 모델 로드
# loaded_model = keras.models.load_model('my_model.keras')
# loaded_model.summary()
```

### 1.2. 가중치만 저장 및 로드

모델의 아키텍처는 코드에 정의되어 있고, 학습된 가중치만 저장하거나 로드해야 할 때 사용합니다.

*   **개념:** `model.save_weights('my_model_weights.weights.h5')` 또는 `model.save_weights('my_model_weights')` (Keras 3.0부터 권장되는 SavedModel 형식의 가중치 저장)를 사용하여 가중치만 저장하고, `model.load_weights('my_model_weights.weights.h5')` 또는 `model.load_weights('my_model_weights')`를 사용하여 로드합니다.
*   **실무 관점:**
    *   **전이 학습:** 사전 학습된 모델의 가중치를 로드하여 새로운 모델에 적용하거나, 미세 조정을 위해 사용될 때 유용합니다.
    *   **체크포인트:** 학습 중 주기적으로 가중치를 저장하여 학습 중단 시 재시작하거나, 가장 좋은 성능을 보인 시점의 가중치를 복원하는 데 사용됩니다. `keras.callbacks.ModelCheckpoint` 콜백과 함께 자주 사용됩니다.
    *   **예시:**

```python
import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 더미 학습
# x_train = np.random.rand(100, 784).astype("float32")
# y_train = np.random.randint(0, 10, size=(100,)).astype("int32")
# model.fit(x_train, y_train, epochs=1)

# 가중치만 저장 (HDF5 형식)
# model.save_weights('my_model_weights.weights.h5')

# 새로운 모델 인스턴스 생성 후 가중치 로드
# new_model = keras.Sequential([
#     keras.layers.Dense(64, activation="relu", input_shape=(784,)),
#     keras.layers.Dense(10, activation="softmax")
# ])
# new_model.load_weights('my_model_weights.weights.h5')
# new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

### 1.3. 모델 아키텍처만 저장 및 로드

모델의 구조(레이어 구성)만 저장하고 로드해야 할 때 사용합니다. 가중치는 포함되지 않습니다.

*   **개념:** `model.to_json()` 또는 `model.to_yaml()`을 사용하여 모델 아키텍처를 JSON 또는 YAML 문자열로 저장하고, `keras.models.model_from_json()` 또는 `keras.models.model_from_yaml()`을 사용하여 로드합니다.
*   **실무 관점:**
    *   **경량화된 저장:** 모델의 가중치 없이 구조만 공유하거나 저장할 때 유용합니다.
    *   **재현성:** 모델 아키텍처를 텍스트 파일로 저장하여 버전 관리 시스템에 포함시키기 용이합니다.
    *   **주의:** 이 방법은 `Sequential` 및 `Functional API`로 구축된 모델에만 작동합니다. `Subclassing API`로 구축된 모델은 Python 코드에 아키텍처가 정의되어 있으므로 이 방법으로 저장할 수 없습니다.
    *   **예시:**

```python
import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])

# 모델 아키텍처를 JSON 문자열로 저장
# json_config = model.to_json()
# print(json_config)

# JSON 문자열로부터 모델 로드
# loaded_model_from_json = keras.models.model_from_json(json_config)
# loaded_model_from_json.summary()
```

### 1.4. TensorFlow SavedModel 형식으로 내보내기 (배포를 위한 준비)

Keras 모델은 TensorFlow의 SavedModel 형식으로 내보낼 수 있습니다. SavedModel은 TensorFlow 모델을 프로덕션 환경에 배포하기 위한 표준 형식입니다.

*   **개념:** `model.save('my_saved_model', save_format='tf')`를 사용하여 SavedModel 형식으로 저장합니다. 이 형식은 모델의 아키텍처, 가중치, 컴파일 설정, 그리고 `tf.function`으로 컴파일된 연산 그래프까지 모두 포함합니다.
*   **실무 관점:**
    *   **배포 표준:** TensorFlow Serving, TensorFlow Lite, TensorFlow.js 등 다양한 TensorFlow 생태계 도구를 통해 모델을 배포할 때 필수적으로 사용됩니다.
    *   **언어 독립적:** Python 환경 없이도 로드하고 실행할 수 있습니다.
    *   **서명(Signature):** 모델의 입력 및 출력 텐서의 형태와 데이터 타입을 정의하는 서명을 포함하여, 모델을 쉽게 호출할 수 있도록 합니다.
    *   **`Subclassing` 모델:** `Subclassing API`로 구축된 모델의 경우, `call` 메서드에 `tf.function` 데코레이터를 적용해야 SavedModel로 올바르게 저장되고 추론 시 성능 최적화를 얻을 수 있습니다.
    *   **예시:**

```python
import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 더미 학습
x_train = np.random.rand(100, 784).astype("float32")
y_train = np.random.randint(0, 10, size=(100,)).astype("int32")
model.fit(x_train, y_train, epochs=1)

# SavedModel 형식으로 저장
model.save('my_saved_model_directory', save_format='tf')

# SavedModel 로드 (TensorFlow API 사용)
import tensorflow as tf
loaded_saved_model = tf.saved_model.load('my_saved_model_directory')
print(loaded_saved_model.signatures['serving_default'])
```

### 1.5. 체크포인트 (Checkpoints) 관리

체크포인트는 학습 과정 중 모델의 가중치만을 저장하는 방식입니다. 주로 학습 중단 시 재시작하거나, 학습이 가장 잘 된 시점의 모델을 복원하는 데 사용됩니다.

*   **개념:**
    *   **가중치만 저장:** 모델의 아키텍처는 저장하지 않고, 학습 가능한 변수(가중치)의 값만 저장합니다.
    *   **옵티마이저 상태 포함:** 옵티마이저의 상태(예: Adam 옵티마이저의 모멘텀 값)도 함께 저장하여 학습을 정확히 중단된 지점부터 재개할 수 있도록 합니다.
*   **실무 관점:**
    *   **학습 중단/재개:** 장시간 학습이 필요한 모델의 경우, 주기적으로 체크포인트를 저장하여 시스템 장애나 학습 중단 시에도 학습을 이어서 할 수 있도록 합니다.
    *   **최적 모델 복원:** `tf.keras.callbacks.ModelCheckpoint` 콜백을 사용하여 검증 성능이 가장 좋은 모델의 체크포인트를 자동으로 저장하고, 학습 완료 후 이 체크포인트를 로드하여 최적의 모델을 사용할 수 있습니다.
    *   **저장:** `tf.keras.callbacks.ModelCheckpoint` 콜백을 `model.fit()`에 전달하거나, Custom Training Loop에서 `tf.train.Checkpoint` 객체를 사용하여 수동으로 저장합니다.
    *   **로드:** `model.load_weights('path/to/checkpoint')`를 사용하여 저장된 가중치를 로드합니다.
