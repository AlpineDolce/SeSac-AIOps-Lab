<h2>NumPy 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
이 문서는 NumPy가 머신러닝 및 딥러닝 알고리즘의 구현과 데이터 처리 과정에서 어떻게 핵심적인 역할을 하는지 실제 사례를 통해 학습합니다. 선형 회귀 구현, 신경망의 순전파, 이미지 처리, 데이터 정규화 등 다양한 ML/DL 적용 사례를 NumPy 코드로 직접 구현하며 실무 역량을 강화합니다.

<h2>목차</h2>

- [1. 실제 ML/DL 적용 사례](#1-실제-mldl-적용-사례)
  - [1.1. 선형 회귀 (Linear Regression) 구현](#11-선형-회귀-linear-regression-구현)
  - [1.2. 신경망의 순전파 (Forward Propagation) 구현](#12-신경망의-순전파-forward-propagation-구현)
  - [1.3. 이미지 처리](#13-이미지-처리)
  - [1.4. 데이터 정규화 (Normalization)](#14-데이터-정규화-normalization)

---

## 1. 실제 ML/DL 적용 사례

NumPy는 머신러닝 및 딥러닝 알고리즘의 구현과 데이터 처리 과정에서 핵심적인 역할을 합니다. 다음은 몇 가지 대표적인 적용 사례입니다.

### 1.1. 선형 회귀 (Linear Regression) 구현

가장 기본적인 머신러닝 알고리즘 중 하나인 선형 회귀는 NumPy를 사용하여 행렬 연산으로 효율적으로 구현할 수 있습니다.

```python
import numpy as np

# 데이터 생성 (특성 X, 타겟 y)
X = 2 * np.random.rand(100, 1) # 100개의 샘플, 1개의 특성
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + 노이즈

# X에 편향(bias) 항을 추가 (모든 값이 1인 열)
X_b = np.c_[np.ones((100, 1)), X]

# 정규 방정식(Normal Equation)을 사용한 선형 회귀 해법
# theta_best = (X_b^T * X_b)^(-1) * X_b^T * y
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print(f"최적의 파라미터 (theta_best):\n{theta_best}")
# 결과는 [4.xxx, 3.xxx]와 유사하게 나와야 합니다.

# 예측
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b @ theta_best
print(f"\n새로운 X에 대한 예측:\n{y_predict}")
```

### 1.2. 신경망의 순전파 (Forward Propagation) 구현

딥러닝의 기본 구성 요소인 신경망의 순전파 과정은 행렬 곱셈과 활성화 함수 적용으로 이루어지며, NumPy로 쉽게 구현할 수 있습니다.

```python
import numpy as np

# 입력 데이터 (배치 크기 1, 특성 3개)
X = np.array([[1.0, 0.5, -1.0]])

# 첫 번째 레이어의 가중치와 편향
W1 = np.array([[0.1, 0.2],
                   [0.3, 0.4],
                   [0.5, 0.6]])
b1 = np.array([[0.1, 0.2]])

# 두 번째 레이어의 가중치와 편향
W2 = np.array([[0.7],
                   [0.8]])
b2 = np.array([[0.3]])

# 활성화 함수 (ReLU)
def relu(x):
    return np.maximum(0, x)

# 순전파
# 첫 번째 레이어
Z1 = X @ W1 + b1
A1 = relu(Z1)
print(f"첫 번째 레이어 출력 (활성화 후):\n{A1}")

# 두 번째 레이어 (출력 레이어)
Z2 = A1 @ W2 + b2
output = Z2
print(f"\n최종 출력:\n{output}")
```

### 1.3. 이미지 처리

이미지는 픽셀 값의 2D 또는 3D 배열로 표현될 수 있으며, NumPy는 이미지 데이터를 로드, 조작, 저장하는 데 사용됩니다. 예를 들어, 이미지의 밝기 조절, 크기 변경, 필터 적용 등에 활용됩니다.

```python
import numpy as np
# from PIL import Image # Pillow 라이브러리가 설치되어 있어야 합니다.

# 예시: 10x10 흑백 이미지 (0-255)
# image_data = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
# print(f"원본 이미지 데이터 (일부):\n{image_data[:3, :3]}")

# 이미지 밝기 20 증가 (클리핑 적용)
# bright_image = np.clip(image_data + 20, 0, 255)
# print(f"\n밝기 조절된 이미지 데이터 (일부):\n{bright_image[:3, :3]}")

# 이미지 저장 (Pillow 사용 예시)
# img = Image.fromarray(image_data)
# img.save('random_image.png')
# print("\n'random_image.png' 저장됨")
```

### 1.4. 데이터 정규화 (Normalization)

머신러닝 모델의 성능 향상을 위해 데이터를 정규화하는 것은 일반적인 전처리 단계입니다. NumPy를 사용하여 평균 0, 표준편차 1로 데이터를 스케일링할 수 있습니다.

```python
import numpy as np

data = np.array([10, 20, 30, 40, 50], dtype=np.float32)
print(f"원본 데이터: {data}")

# 평균과 표준편차 계산
mean = np.mean(data)
std = np.std(data)

# Z-score 정규화: (x - mean) / std
normalized_data = (data - mean) / std

print(f"\n정규화된 데이터: {normalized_data}")
print(f"정규화된 데이터의 평균: {np.mean(normalized_data):.4f}")
print(f"정규화된 데이터의 표준편차: {np.std(normalized_data):.4f}")
```

