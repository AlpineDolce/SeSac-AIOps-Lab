<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 TensorFlow의 기본 데이터 구조인 텐서(Tensor)의 개념과 중요성을 다룹니다. 또한, 딥러닝 모델 학습의 핵심인 자동 미분(Automatic Differentiation)과 이를 구현하는 `tf.GradientTape`의 활용 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. TensorFlow & Keras 개요 및 기본 개념](#1-tensorflow--keras-개요-및-기본-개념)
  - [1.1. 기본 데이터 구조: Tensor](#11-기본-데이터-구조-tensor)
  - [1.2. 자동 미분 (Automatic Differentiation)과 `tf.GradientTape`](#12-자동-미분-automatic-differentiation과-tf-gradienttape)

---

## 1. TensorFlow & Keras 개요 및 기본 개념

### 1.1. 기본 데이터 구조: Tensor

TensorFlow의 모든 연산은 텐서(Tensor)를 중심으로 이루어집니다. 텐서는 다차원 배열로, 스칼라(0차원), 벡터(1차원), 행렬(2차원)을 포함하는 일반화된 개념입니다.

*   **개념:**
    *   **랭크(Rank/Dimension):** 텐서의 차원 수 (예: 스칼라 0, 벡터 1, 행렬 2).
    *   **형태(Shape):** 각 차원의 크기를 나타내는 튜플 (예: (2, 3)은 2행 3열 행렬).
    *   **데이터 타입(Dtype):** 텐서 내 요소들의 데이터 타입 (예: `tf.float32`, `tf.int32`, `tf.string`). 딥러닝에서는 주로 `tf.float32` 또는 `tf.float16` (Mixed Precision 시)을 사용합니다.
*   **실무 관점:**
    *   **데이터 표현:** 이미지(높이, 너비, 채널), 시계열 데이터(샘플 수, 시간 스텝, 특징 수), 텍스트 데이터(배치 크기, 시퀀스 길이, 임베딩 차원) 등 모든 종류의 데이터를 텐서로 표현합니다.
    *   **연산 단위:** TensorFlow의 모든 API는 텐서를 입력으로 받고 텐서를 출력합니다. 텐서의 형태와 데이터 타입을 정확히 이해하는 것이 모델 구축 및 디버깅에 매우 중요합니다.
    *   **GPU 활용:** 텐서는 CPU 메모리뿐만 아니라 GPU 메모리에도 저장될 수 있으며, GPU 상에서 텐서 연산을 수행하여 계산 속도를 극대화합니다. `tf.Tensor` 객체는 자동으로 적절한 디바이스에 할당됩니다.

### 1.2. 자동 미분 (Automatic Differentiation)과 `tf.GradientTape`

딥러닝 모델 학습의 핵심은 손실 함수의 기울기(gradient)를 계산하고 이를 이용하여 모델의 가중치를 업데이트하는 것입니다. TensorFlow는 자동 미분 기능을 통해 이 과정을 효율적으로 처리합니다.

*   **개념:**
    *   **자동 미분:** 주어진 입력 변수에 대한 함수의 출력 변수의 기울기를 자동으로 계산하는 기술입니다. TensorFlow는 역전파(backpropagation) 알고리즘을 구현하기 위해 이 기능을 사용합니다.
    *   **`tf.GradientTape`:** TensorFlow 2.x에서 자동 미분을 수행하는 주요 도구입니다. `tf.GradientTape` 컨텍스트 내에서 실행된 모든 연산을 "기록"하고, 이 기록을 사용하여 나중에 어떤 텐서에 대한 다른 텐서의 기울기를 계산할 수 있습니다.
*   **실무 관점:**
    *   **Custom Training Loop:** Keras `model.fit()`과 같은 고수준 API를 사용하지 않고, 직접 학습 루프를 구현할 때 `tf.GradientTape`는 필수적입니다. 이를 통해 학습 과정을 세밀하게 제어하고, 복잡한 손실 함수나 학습 전략을 적용할 수 있습니다.
    *   **손실 함수 및 최적화:** `tf.GradientTape`를 사용하여 손실 함수에 대한 모델 가중치의 기울기를 계산하고, 이 기울기를 Adam, SGD 등과 같은 옵티마이저(`tf.keras.optimizers`)에 전달하여 가중치를 업데이트합니다.
    *   **디버깅:** 기울기 값이 `NaN`이 되거나 너무 커지거나 작아지는 등의 문제가 발생할 경우, `tf.GradientTape`를 사용하여 특정 지점의 기울기를 확인하고 디버깅하는 데 활용할 수 있습니다.
    *   **메모리 관리:** `tf.GradientTape`는 연산을 기록하므로, 불필요한 연산이 기록되지 않도록 주의해야 합니다. 특히 추론 시에는 `tf.GradientTape`를 사용하지 않거나 `persistent=False` (기본값)로 설정하여 메모리 사용량을 최적화합니다.
