<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 Keras에서 가장 유연한 모델 구축 방법인 Subclassing API를 상세히 다룹니다. `keras.Model`을 상속받아 Custom Model을 구현하는 방법과 Custom Layer, Custom Loss Function, Custom Metric을 정의하는 방법을 실제 코드 예제를 통해 학습합니다. 이를 통해 Keras의 기본 API로 표현하기 어려운 복잡한 모델 아키텍처나 평가 지표를 구현하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. 모델 아키텍처 설계](#1-모델-아키텍처-설계)
  - [1.1. Subclassing API를 이용한 Custom Model 구현](#11-subclassing-api를-이용한-custom-model-구현)
  - [1.2. Custom Layer 및 Custom Loss/Metric 구현](#12-custom-layer-및-custom-lossmetric-구현)
    - [1.2.1. Custom Layer](#121-custom-layer)
    - [1.2.2. Custom Loss Function](#122-custom-loss-function)
    - [1.2.3. Custom Metric](#123-custom-metric)

---

## 1. 모델 아키텍처 설계

### 1.1. Subclassing API를 이용한 Custom Model 구현

가장 유연한 모델 구축 방법으로, Python 클래스 상속을 통해 모델의 `__init__` 메서드에서 레이어를 정의하고, `call` 메서드에서 순전파 로직을 직접 구현합니다.

*   **개념:** `keras.Model` 클래스를 상속받아 새로운 클래스를 정의하고, `__init__` 메서드에서 필요한 레이어들을 인스턴스화하며, `call(self, inputs)` 메서드에서 입력 텐서가 레이어들을 통과하는 순전파 로직을 작성합니다.
*   **실무 관점:**
    *   **장점:** 가장 높은 수준의 유연성을 제공하여, Keras의 기본 레이어로는 표현하기 어려운 매우 복잡하거나 동적인 모델 아키텍처(예: 조건부 로직, 루프, 재귀적 구조)를 구현할 때 적합합니다. Python의 모든 기능을 활용할 수 있습니다.
    *   **단점:** 디버깅이 더 어렵고, 모델의 구조가 코드에 숨겨져 있어 가독성이 떨어질 수 있습니다. 또한, `model.save()` 시 모델 아키텍처가 아닌 가중치만 저장될 수 있으므로 주의가 필요합니다 (SavedModel로 저장 시 `call` 메서드에 `tf.function` 데코레이터를 적용하는 것이 권장됨).
    *   **활용 사례:** 연구 목적의 새로운 모델 아키텍처를 구현하거나, 기존 Keras API로 표현하기 어려운 특수한 모델에 사용됩니다.

```python
import keras
from keras import layers
import tensorflow as tf

# Subclassing API 모델 정의 예시
class CustomModel(keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.classifier = layers.Dense(num_classes, activation="softmax")

    @tf.function # 성능 최적화 및 SavedModel 저장을 위해 권장
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.classifier(x)

# 모델 인스턴스 생성 및 사용
model = CustomModel(num_classes=10)
# 모델의 입력 형태를 정의하기 위해 더미 데이터로 빌드
model.build(input_shape=(None, 784)) # None은 배치 크기를 의미
model.summary()
```

### 1.2. Custom Layer 및 Custom Loss/Metric 구현

Keras의 기본 레이어나 손실 함수, 메트릭으로는 표현할 수 없는 특수한 연산이나 평가 지표가 필요할 때 사용자 정의 기능을 구현할 수 있습니다.

#### 1.2.1. Custom Layer:
*   **개념:** `keras.layers.Layer` 클래스를 상속받아 `__init__`, `build`, `call` 메서드를 오버라이드하여 새로운 레이어를 정의합니다.
    *   `__init__(self, **kwargs)`: 레이어의 하이퍼파라미터 등을 초기화합니다.
    *   `build(self, input_shape)`: 입력 형태가 주어졌을 때 레이어의 가중치(변수)를 생성합니다. `self.add_weight()`를 사용합니다.
    *   `call(self, inputs)`: 입력 텐서에 대한 순전파 연산을 정의합니다.
*   **실무 관점:** 특정 도메인 지식을 반영한 새로운 연산, 복잡한 어텐션 메커니즘, 특수한 정규화 기법 등을 구현할 때 사용됩니다. 재사용 가능한 컴포넌트를 만들어 코드의 모듈성을 높일 수 있습니다.

```python
import keras
from keras import layers
import tensorflow as tf

class CustomDense(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)

# Custom Layer 사용 예시
model = keras.Sequential([
    keras.Input(shape=(784,)),
    CustomDense(64, activation="relu"),
    CustomDense(10, activation="softmax")
])
model.summary()
```

#### 1.2.2. Custom Loss Function:
*   **개념:** `keras.losses.Loss` 클래스를 상속받아 `call(self, y_true, y_pred)` 메서드를 오버라이드하거나, 단순히 `y_true`와 `y_pred`를 인자로 받는 함수를 정의합니다.
*   **실무 관점:** 특정 문제에 최적화된 손실 함수(예: Focal Loss for imbalanced data, IoU Loss for object detection)를 구현할 때 사용됩니다.

```python
import keras
import tensorflow as tf

# Custom Loss Function (함수형)
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Custom Loss Function (클래스형)
class CustomHuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, name="huber_loss"):
        super().__init__(name=name)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * (tf.abs(error) - self.threshold / 2)
        return tf.where(is_small_error, squared_loss, linear_loss)

# 모델 컴파일 시 사용
# model.compile(optimizer="adam", loss=custom_mse)
# model.compile(optimizer="adam", loss=CustomHuberLoss(threshold=0.5))
```

#### 1.2.3. Custom Metric:
*   **개념:** `keras.metrics.Metric` 클래스를 상속받아 `__init__`, `update_state`, `result`, `reset_state` 메서드를 오버라이드하여 새로운 메트릭을 정의합니다.
    *   `__init__(self, name='custom_metric', **kwargs)`: 메트릭의 상태 변수를 초기화합니다.
    *   `update_state(self, y_true, y_pred, sample_weight=None)`: 배치별로 메트릭의 상태를 업데이트합니다.
    *   `result(self)`: 현재까지의 상태를 기반으로 최종 메트릭 값을 계산하여 반환합니다.
    *   `reset_state(self)`: 메트릭의 상태를 초기화합니다 (새로운 에포크 시작 시).
*   **실무 관점:** 특정 도메인에 특화된 평가 지표(예: F1-score for classification, IoU for segmentation)를 학습 과정에서 모니터링할 때 사용됩니다.

```python
import keras
import tensorflow as tf

class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool) # 이진 분류 가정

        tp = tf.cast(tf.reduce_sum(tf.cast(y_true & y_pred, tf.float32)), tf.float32)
        fp = tf.cast(tf.reduce_sum(tf.cast(~y_true & y_pred, tf.float32)), tf.float32)
        fn = tf.cast(tf.reduce_sum(tf.cast(y_true & ~y_pred, tf.float32)), tf.float32)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + keras.backend.epsilon())
        f1 = 2 * ((precision * recall) / (precision + recall + keras.backend.epsilon()))
        return f1

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

# 모델 컴파일 시 사용
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[F1Score()])
```
