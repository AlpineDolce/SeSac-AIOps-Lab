<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 Keras에서 딥러닝 모델을 구축하는 가장 일반적인 두 가지 방법인 Sequential API와 Functional API를 상세히 다룹니다. 각 API의 개념, 장단점, 그리고 실제 코드 예시를 통해 모델의 복잡성과 유연성 요구사항에 따라 적절한 구축 방식을 선택하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. 모델 아키텍처 설계](#1-모델-아키텍처-설계)
  - [1.1. Sequential API를 이용한 모델 구축](#11-sequential-api를-이용한-모델-구축)
  - [1.2. Functional API를 이용한 복잡한 모델 구축](#12-functional-api를-이용한-복잡한-모델-구축)

---

## 1. 모델 아키텍처 설계

Keras는 모델을 구축하는 세 가지 주요 방법을 제공하며, 각 방법은 유연성과 사용 편의성 면에서 장단점을 가집니다.

### 1.1. Sequential API를 이용한 모델 구축

가장 간단하고 직관적인 모델 구축 방법으로, 레이어들을 순서대로 쌓아 올리는 선형적인(linear stack) 모델에 적합합니다.

*   **개념:** `keras.Sequential` 클래스를 사용하여 레이어 리스트를 전달하거나, `add()` 메서드를 통해 레이어를 하나씩 추가하여 모델을 정의합니다.
*   **실무 관점:**
    *   **장점:** 코드가 간결하고 이해하기 쉬워 딥러닝 초보자나 간단한 모델을 빠르게 구현할 때 매우 유용합니다.
    *   **단점:** 다중 입력/출력 모델, 공유 레이어, 비선형적인 연결(예: 잔차 연결, 인셉션 모듈)과 같은 복잡한 아키텍처를 표현하기 어렵습니다.
    *   **활용 사례:** 기본적인 MLP(Multi-Layer Perceptron), 간단한 CNN(Convolutional Neural Network), RNN(Recurrent Neural Network) 모델 등에 적합합니다.

```python
import keras
from keras import layers

# Sequential 모델 정의 예시
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(784,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 또는 add() 메서드 사용
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(784,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()
```

### 1.2. Functional API를 이용한 복잡한 모델 구축

Sequential API보다 훨씬 유연하며, 다중 입력/출력, 공유 레이어, 비선형적인 연결 등 복잡한 모델 아키텍처를 구축할 수 있게 합니다.

*   **개념:** 입력 텐서(`keras.Input`)를 정의하고, 이 입력 텐서에 레이어를 함수처럼 적용하여 출력 텐서를 얻는 방식으로 모델을 구성합니다. 최종적으로 입력 텐서와 출력 텐서를 `keras.Model` 클래스에 전달하여 모델을 생성합니다.
*   **실무 관점:**
    *   **장점:** 모델의 구조를 명시적으로 정의하므로 가독성이 좋고, 복잡한 신경망 아키텍처(예: ResNet, Inception, Siamese Network)를 구현하는 데 필수적입니다.
    *   **단점:** Sequential API보다 코드가 길어지고, 개념적으로 약간 더 복잡할 수 있습니다.
    *   **활용 사례:** 대부분의 실제 딥러닝 프로젝트에서 복잡한 모델을 구축할 때 선호되는 방식입니다.

```python
import keras
from keras import layers

# Functional API 모델 정의 예시 (다중 입력/출력)
input_a = keras.Input(shape=(64,), name="branch_a_input")
input_b = keras.Input(shape=(128,), name="branch_b_input")

x = layers.Dense(32, activation="relu")(input_a)
x = layers.Dense(16, activation="relu")(x)

y = layers.Dense(64, activation="relu")(input_b)
y = layers.Dense(16, activation="relu")(y)

# 두 브랜치의 출력을 연결
combined = layers.concatenate([x, y])

# 최종 출력 레이어
output_c = layers.Dense(10, activation="softmax", name="classification_output")(combined)
output_d = layers.Dense(1, activation="sigmoid", name="regression_output")(combined)

model = keras.Model(inputs=[input_a, input_b], outputs=[output_c, output_d])
model.summary()
```
