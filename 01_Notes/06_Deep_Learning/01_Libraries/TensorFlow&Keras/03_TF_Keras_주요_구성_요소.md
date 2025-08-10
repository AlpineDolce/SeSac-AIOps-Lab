<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 Keras를 사용하여 딥러닝 모델을 구성하는 핵심 요소들인 `Models`, `Layers`, `Optimizers`, `Losses`, `Metrics`의 개념과 실무적 중요성을 상세히 다룹니다. 각 구성 요소의 역할과 상호작용을 이해하여 효과적인 딥러닝 모델을 설계하고 구현하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. TensorFlow & Keras 개요 및 기본 개념](#1-tensorflow--keras-개요-및-기본-개념)
  - [1.1. Keras의 주요 구성 요소 (Models, Layers, Optimizers, Losses, Metrics)](#11-keras의-주요-구성-요소-models-layers-optimizers-losses-metrics)

---

## 1. TensorFlow & Keras 개요 및 기본 개념

### 1.1. Keras의 주요 구성 요소 (Models, Layers, Optimizers, Losses, Metrics)

Keras는 딥러닝 모델을 구성하는 핵심 요소들을 직관적인 API로 제공합니다.

*   **Models (`keras.Model`):**
    *   **개념:** 레이어들을 조직화하여 입력으로부터 출력으로의 변환을 정의하는 객체입니다. Keras는 `Sequential` 모델, `Functional API` 모델, `Subclassing` 모델의 세 가지 방식으로 모델을 구축할 수 있습니다.
    *   **실무 관점:** `model.compile()`, `model.fit()`, `model.evaluate()`, `model.predict()`와 같은 메서드를 통해 모델의 학습, 평가, 예측 과정을 쉽게 관리할 수 있습니다. 모델의 복잡성과 유연성 요구사항에 따라 적절한 모델 구축 방식을 선택합니다.

*   **Layers (`keras.layers`):**
    *   **개념:** 신경망의 기본 빌딩 블록입니다. 입력 텐서를 받아 변환을 수행하고 출력 텐서를 반환합니다. `Dense`, `Conv2D`, `MaxPooling2D`, `LSTM`, `Dropout`, `BatchNormalization` 등 다양한 종류의 레이어를 제공합니다.
    *   **실무 관점:** 각 레이어는 학습 가능한 가중치(weights)를 가질 수 있으며, 모델의 복잡성과 표현력을 결정합니다. 적절한 레이어 선택과 조합은 모델 성능에 큰 영향을 미칩니다.

*   **Optimizers (`keras.optimizers`):**
    *   **개념:** 모델의 가중치를 업데이트하여 손실 함수를 최소화하는 알고리즘입니다. `Adam`, `SGD`, `RMSprop`, `Adagrad` 등 다양한 최적화 알고리즘을 제공합니다.
    *   **실무 관점:** 옵티마이저는 학습률(learning rate)과 같은 하이퍼파라미터를 가집니다. 적절한 옵티마이저와 학습률 선택은 모델의 수렴 속도와 최종 성능에 매우 중요합니다.

*   **Losses (`keras.losses`):**
    *   **개념:** 모델의 예측과 실제 정답 간의 오차를 측정하는 함수입니다. 모델이 학습해야 할 목표를 정의합니다. `CategoricalCrossentropy`, `SparseCategoricalCrossentropy`, `MeanSquaredError`, `BinaryCrossentropy` 등 다양한 손실 함수를 제공합니다.
    *   **실무 관점:** 태스크의 특성(분류, 회귀 등)과 출력 데이터의 형태에 따라 적절한 손실 함수를 선택해야 합니다. 예를 들어, 다중 클래스 분류에는 `CategoricalCrossentropy` 또는 `SparseCategoricalCrossentropy`를, 회귀에는 `MeanSquaredError`를 사용합니다.

*   **Metrics (`keras.metrics`):**
    *   **개념:** 모델의 성능을 평가하는 지표입니다. 손실 함수와 달리, 메트릭은 모델 학습에 직접적으로 사용되지 않지만, 모델의 성능을 사람이 이해하기 쉬운 형태로 보여줍니다. `Accuracy`, `Precision`, `Recall`, `AUC` 등 다양한 메트릭을 제공합니다.
    *   **실무 관점:** 학습 과정에서 모델의 성능 변화를 모니터링하고, 최종 모델의 성능을 평가하는 데 사용됩니다. 비즈니스 목표에 부합하는 메트릭을 선택하는 것이 중요합니다.
