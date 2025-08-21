<h2>NumPy 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
이 문서는 NumPy가 머신러닝 및 딥러닝 알고리즘의 구현과 데이터 처리 과정에서 어떻게 핵심적인 역할을 하는지 실제 사례를 통해 학습합니다. 경사 하강법을 이용한 선형 회귀, 신경망의 순전파와 역전파, 이미지 필터링, 그리고 다양한 데이터 전처리 기법을 NumPy 코드로 직접 구현하며 실무 역량을 강화합니다.

<h2>목차</h2>

- [1. 머신러닝 모델 구현: 선형 회귀](#1-머신러닝-모델-구현-선형-회귀)
  - [1.1. 방법 1: 정규 방정식 (Normal Equation)](#11-방법-1-정규-방정식-normal-equation)
  - [1.2. 방법 2: 경사 하강법 (Gradient Descent)](#12-방법-2-경사-하강법-gradient-descent)
- [2. 딥러닝 모델 구현: 신경망의 순전파와 역전파](#2-딥러닝-모델-구현-신경망의-순전파와-역전파)
  - [2.1. 순전파 (Forward Propagation)](#21-순전파-forward-propagation)
  - [2.2. 역전파 (Backward Propagation)](#22-역전파-backward-propagation)
- [3. 이미지 처리와 컨볼루션](#3-이미지-처리와-컨볼루션)
  - [3.1. 기본 이미지 조작](#31-기본-이미지-조작)
  - [3.2. 컨볼루션 필터 적용](#32-컨볼루션-필터-적용)
- [4. 머신러닝을 위한 데이터 전처리](#4-머신러닝을-위한-데이터-전처리)
  - [4.1. 스케일링: 정규화와 표준화](#41-스케일링-정규화와-표준화)
  - [4.2. 원-핫 인코딩 (One-Hot Encoding)](#42-원-핫-인코딩-one-hot-encoding)

---

## 1. 머신러닝 모델 구현: 선형 회귀

NumPy의 강력한 행렬 연산은 머신러닝 알고리즘을 밑바닥부터 구현하는 데 필수적입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성 (y = 4 + 3x + 노이즈)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 모든 샘플에 x0 = 1 (편향) 추가
X_b = np.c_[np.ones((100, 1)), X]
```

### 1.1. 방법 1: 정규 방정식 (Normal Equation)
정규 방정식은 비용 함수를 최소화하는 파라미터 `theta`를 해석적으로, 즉 한 번의 계산으로 바로 찾는 방법입니다.

- **수식**: $\theta = (X^T X)^{-1} X^T y$

```python
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"정규 방정식으로 찾은 최적 파라미터:\n{theta_best}")
```

### 1.2. 방법 2: 경사 하강법 (Gradient Descent)
경사 하강법은 비용 함수의 경사(gradient)를 계산하여 파라미터를 점진적으로 업데이트하는 반복적인 최적화 알고리즘입니다. 대용량 데이터셋에 더 적합합니다.

```python
learning_rate = 0.1
n_iterations = 1000
m = 100 # 샘플 개수

# 1. 파라미터 랜덤 초기화
theta = np.random.randn(2, 1)

# 2. 경사 하강법 반복
for iteration in range(n_iterations):
    # 2-1. 예측 (h = X * theta)
    predictions = X_b @ theta
    # 2-2. 오차 계산
    error = predictions - y
    # 2-3. 비용 함수의 경사 계산
    gradients = 2/m * X_b.T @ error
    # 2-4. 파라미터 업데이트
    theta = theta - learning_rate * gradients

print(f"\n경사 하강법으로 찾은 최적 파라미터:\n{theta}")

# 시각화
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b @ theta_best

plt.plot(X_new, y_predict, "r-", label="Prediction")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
```

## 2. 딥러닝 모델 구현: 신경망의 순전파와 역전파

NumPy를 사용하면 신경망의 핵심 메커니즘인 순전파와 역전파를 단계별로 구현하며 깊이 있게 이해할 수 있습니다.

```python
# 데이터 및 파라미터 초기화
X = np.array([[1.0, 0.5, -1.0]]) # 입력
y_true = np.array([[0.8]]) # 실제 값

W1 = np.random.rand(3, 2) # (입력 특성, 히든 유닛)
b1 = np.zeros((1, 2))
W2 = np.random.rand(2, 1) # (히든 유닛, 출력 유닛)
b2 = np.zeros((1, 1))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
```

### 2.1. 순전파 (Forward Propagation)
입력 데이터가 네트워크를 통과하여 최종 출력을 계산하는 과정입니다.

```python
# Layer 1
Z1 = X @ W1 + b1
A1 = relu(Z1)

# Layer 2 (Output)
Z2 = A1 @ W2 + b2
output = Z2 # 활성화 함수 없는 출력 레이어

print(f"신경망 최종 출력: {output}")
```

### 2.2. 역전파 (Backward Propagation)
출력층의 오차를 기반으로 각 가중치와 편향에 대한 손실 함수의 그래디언트를 계산하는 과정입니다. (Chain Rule 적용)

```python
# 1. 출력층의 오차 계산 (MSE 손실 함수 사용)
loss_derivative = 2 * (output - y_true)

# 2. Layer 2의 그래디언트 계산
dZ2 = loss_derivative
dW2 = A1.T @ dZ2
db2 = np.sum(dZ2, axis=0, keepdims=True)

# 3. Layer 1의 그래디언트 계산
dA1 = dZ2 @ W2.T
dZ1 = dA1 * relu_derivative(Z1)
dW1 = X.T @ dZ1
db1 = np.sum(dZ1, axis=0, keepdims=True)

print(f"\nW1의 그래디언트:\n{dW1}")
print(f"W2의 그래디언트:\n{dW2}")

# 4. (옵션) 파라미터 업데이트
# W1 -= learning_rate * dW1 ...
```

## 3. 이미지 처리와 컨볼루션

이미지는 픽셀 값으로 이루어진 NumPy 배열이며, 다양한 배열 연산을 통해 이미지를 조작할 수 있습니다.

```python
from scipy.misc import face # 예제용 이미지 로드
from scipy.signal import convolve2d

# 768x1024x3 크기의 컬러 이미지 로드 후 흑백으로 변환
image = face(gray=True)
```

### 3.1. 기본 이미지 조작

```python
# 이미지 자르기 (Slicing)
cropped_image = image[200:500, 400:800]

# 이미지 뒤집기
flipped_image = np.fliplr(image)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(image, cmap='gray'); plt.title('Original')
plt.subplot(1, 3, 2); plt.imshow(cropped_image, cmap='gray'); plt.title('Cropped')
plt.subplot(1, 3, 3); plt.imshow(flipped_image, cmap='gray'); plt.title('Flipped')
plt.show()
```

### 3.2. 컨볼루션 필터 적용
컨볼루션은 이미지의 각 픽셀에 필터(커널)를 적용하여 특징을 추출하거나 이미지를 변형하는 기법입니다. CNN의 핵심 연산입니다.

```python
# 샤프닝 필터
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# 컨볼루션 연산 수행
sharpened_image = convolve2d(image, sharpen_kernel, mode='same', boundary='symm')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(image, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(sharpened_image, cmap='gray'); plt.title('Sharpened')
plt.show()
```

## 4. 머신러닝을 위한 데이터 전처리

모델의 성능을 높이기 위해 데이터를 적절한 형태로 가공하는 것은 매우 중요합니다.

### 4.1. 스케일링: 정규화와 표준화
- **표준화 (Standardization)**: 평균 0, 표준편차 1로 변환합니다. (Z-score)
- **정규화 (Normalization)**: 데이터를 0과 1 사이의 값으로 변환합니다. (Min-Max Scaling)

```python
data = np.array([10, 20, 30, 40, 50], dtype=np.float32).reshape(-1, 1)

# 표준화
mean = np.mean(data)
std = np.std(data)
standardized_data = (data - mean) / std

# 정규화
min_val = np.min(data)
max_val = np.max(data)
normalized_data = (data - min_val) / (max_val - min_val)

print(f"원본 데이터:\n{data.flatten()}")
print(f"표준화된 데이터:\n{standardized_data.flatten()}")
print(f"정규화된 데이터:\n{normalized_data.flatten()}")
```

### 4.2. 원-핫 인코딩 (One-Hot Encoding)
범주형 데이터를 머신러닝 모델이 이해할 수 있는 숫자 형태로 변환하는 기법입니다. 각 범주를 고유한 이진 벡터로 표현합니다.

```python
# 0, 1, 2 세 개의 클래스가 있는 레이블 데이터
labels = np.array([0, 2, 1, 0, 1, 2])
num_classes = 3

# 원-핫 인코딩 구현
one_hot_labels = np.zeros((labels.size, num_classes))
one_hot_labels[np.arange(labels.size), labels] = 1

print(f"\n원본 레이블: {labels}")
print(f"원-핫 인코딩된 레이블:\n{one_hot_labels}")
```