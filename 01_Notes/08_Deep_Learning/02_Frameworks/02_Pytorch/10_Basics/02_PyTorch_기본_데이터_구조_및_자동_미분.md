<h2>PyTorch 기본 데이터 구조 및 자동 미분</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-18

<h2>문서 목표</h2>
이 문서는 PyTorch의 핵심 데이터 구조인 Tensor와 자동 미분(Autograd) 시스템에 대해 심층적으로 다룹니다. Tensor의 다양한 생성 방법, 주요 속성, 그리고 다채로운 연산들을 상세히 설명합니다. 또한, 딥러닝 모델 학습의 근간이 되는 Autograd의 작동 원리, `requires_grad` 속성, `backward()` 메서드, 계산 그래프의 개념, 그리고 기울기 추적을 제어하는 `torch.no_grad()`와 `detach()`의 활용법을 명확히 제시하여 PyTorch를 활용한 딥러닝 개발의 견고한 기초를 다지는 데 기여하고자 합니다.

<h2>목차</h2>

- [PyTorch 기본 데이터 구조 및 자동 미분](#pytorch-기본-데이터-구조-및-자동-미분)
  - [1. Tensor: PyTorch의 핵심 데이터 구조](#1-tensor-pytorch의-핵심-데이터-구조)
    - [1.1. Tensor 생성](#11-tensor-생성)
    - [1.2. Tensor의 속성 (Attributes)](#12-tensor의-속성-attributes)
    - [1.3. Tensor 연산](#13-tensor-연산)
      - [1.3.1. 인덱싱 및 슬라이싱](#131-인덱싱-및-슬라이싱)
      - [1.3.2. Tensor 결합 (Concatenation)](#132-tensor-결합-concatenation)
      - [1.3.3. 수학 연산](#133-수학-연산)
      - [1.3.4. 브로드캐스팅 (Broadcasting)](#134-브로드캐스팅-broadcasting)
      - [1.3.5. 인플레이스(In-place) 연산](#135-인플레이스in-place-연산)
      - [1.3.6. 크기 변경 (Reshaping)](#136-크기-변경-reshaping)
    - [1.4. Tensor와 NumPy 변환](#14-tensor와-numpy-변환)
    - [1.5. Tensor의 CPU/GPU 이동](#15-tensor의-cpugpu-이동)
  - [2. 자동 미분 (Autograd)](#2-자동-미분-autograd)
    - [2.1. 자동 미분이란?](#21-자동-미분이란)
    - [2.2. `requires_grad` 속성](#22-requires_grad-속성)
    - [2.3. 기울기 계산 (`backward()` 메서드)](#23-기울기-계산-backward-메서드)
    - [2.4. 계산 그래프 (Computational Graph)와 `grad_fn`](#24-계산-그래프-computational-graph와-grad_fn)
    - [2.5. 기울기 추적 중지: `torch.no_grad()`와 `detach()`](#25-기울기-추적-중지-torchno_grad와-detach)

---

# PyTorch 기본 데이터 구조 및 자동 미분

## 1. Tensor: PyTorch의 핵심 데이터 구조

Tensor는 PyTorch에서 데이터를 표현하는 가장 기본적인 단위입니다. NumPy의 `ndarray`와 매우 유사하지만, GPU를 사용한 연산 가속 기능을 추가로 지원하여 딥러닝 모델의 대규모 수치 연산을 효율적으로 처리할 수 있다는 강력한 장점이 있습니다. Tensor는 스칼라(0차원), 벡터(1차원), 행렬(2차원)뿐만 아니라 더 높은 차원의 데이터를 표현할 수 있는 일반화된 다차원 배열입니다. 딥러닝 모델의 입력 데이터, 모델의 가중치(weights)와 편향(biases), 그리고 모델의 출력값 등 모든 데이터는 Tensor 형태로 표현됩니다.

### 1.1. Tensor 생성

PyTorch에서 Tensor를 생성하는 방법은 매우 다양하며, 필요에 따라 적절한 방법을 선택할 수 있습니다.

```python
import torch
import numpy as np

# 1. 리스트(List) 또는 NumPy 배열로부터 Tensor 생성
# 기존 Python 데이터 구조나 NumPy 배열로부터 Tensor를 생성할 수 있습니다.
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data) # Python 리스트로부터 생성
print(f"Tensor from list:
 {x_data}
")

np_array = np.array(data)
x_np = torch.from_numpy(np_array) # NumPy 배열로부터 생성
print(f"Tensor from NumPy array:
 {x_np}
")

# 2. 다른 Tensor로부터 생성 (속성 유지 또는 덮어쓰기)
# 기존 Tensor의 속성(shape, dtype, device)을 유지하면서 새로운 Tensor를 생성할 수 있습니다.
x_ones = torch.ones_like(x_data) # x_data와 동일한 shape, dtype, device를 가지며 1로 채워짐
print(f"Ones Tensor (like x_data):
 {x_ones}
")

# dtype을 명시하여 기존 Tensor의 속성을 덮어쓸 수 있습니다.
x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data와 동일한 shape, device를 가지며 float 타입의 랜덤 값으로 채워짐
print(f"Random Tensor (like x_data, float dtype):
 {x_rand}
")

# 3. 특정 값을 가진 Tensor 생성
# 원하는 shape(크기)를 지정하여 특정 값으로 채워진 Tensor를 생성합니다.
shape = (2, 3,) # 2행 3열의 Tensor를 생성할 것임을 명시
ones_tensor = torch.ones(shape) # 모든 요소를 1로 채움
zeros_tensor = torch.zeros(shape) # 모든 요소를 0으로 채움
rand_tensor = torch.rand(shape) # 0과 1 사이의 균일 분포에서 무작위 값으로 채움
full_tensor = torch.full(shape, 7.0) # 모든 요소를 특정 값(여기서는 7.0)으로 채움

print(f"Ones Tensor (shape {shape}):
 {ones_tensor}
")
print(f"Zeros Tensor (shape {shape}):
 {zeros_tensor}
")
print(f"Random Tensor (shape {shape}):
 {rand_tensor}
")
print(f"Full Tensor (shape {shape}, value 7.0):
 {full_tensor}
")

# 4. 초기화되지 않은 Tensor 생성
# 메모리만 할당하고 내용은 초기화하지 않습니다. 이전에 해당 메모리에 있던 값이 그대로 남아있을 수 있습니다.
empty_tensor = torch.empty(2, 2)
print(f"Empty Tensor:
 {empty_tensor}
")

# 5. 순차적인 값으로 Tensor 생성
# NumPy의 arange와 유사하게, 특정 범위의 순차적인 값으로 Tensor를 생성합니다.
arange_tensor = torch.arange(0, 10, 2) # 0부터 10 미만까지 2씩 증가
print(f"Arange Tensor:
 {arange_tensor}
")

linspace_tensor = torch.linspace(0, 10, 5) # 0부터 10까지 5개의 균등한 간격의 값
print(f"Linspace Tensor:
 {linspace_tensor}
")
```

### 1.2. Tensor의 속성 (Attributes)

Tensor는 데이터를 담는 것 외에도, 데이터의 특성을 설명하는 중요한 속성들을 가집니다. 이 속성들은 Tensor의 형태, 자료형, 그리고 저장된 장치를 알려줍니다.

*   `shape` (또는 `size()`): Tensor의 각 차원(dimension)의 크기를 나타내는 튜플입니다. 예를 들어, `(3, 4)`는 3행 4열의 2차원 Tensor임을 의미합니다. Tensor의 차원 수를 '랭크(rank)'라고도 합니다.
*   `dtype`: Tensor에 저장된 데이터 요소들의 자료형을 나타냅니다. PyTorch는 `torch.float32` (기본 부동 소수점), `torch.float64` (double), `torch.int64` (long), `torch.bool` 등 다양한 자료형을 지원합니다. 올바른 `dtype`을 사용하는 것은 메모리 효율성과 연산 정확성에 중요합니다.
*   `device`: Tensor가 현재 저장되어 있는 컴퓨팅 장치(CPU 또는 GPU)를 나타냅니다. `cpu`는 CPU 메모리에, `cuda:0`은 첫 번째 GPU 메모리에 Tensor가 저장되어 있음을 의미합니다. GPU를 활용하면 연산 속도를 크게 향상시킬 수 있습니다.

```python
tensor = torch.rand(3, 4) # 3행 4열의 랜덤 Tensor 생성

print(f"Shape of tensor: {tensor.shape}") # Tensor의 형태: (3, 4)
print(f"Datatype of tensor: {tensor.dtype}") # Tensor의 자료형: torch.float32 (기본값)
print(f"Device tensor is stored on: {tensor.device}") # Tensor가 저장된 장치: cpu (기본값)

# Tensor를 GPU로 이동 (CUDA를 사용할 수 있는 경우)
if torch.cuda.is_available():
    tensor = tensor.to("cuda") # Tensor를 GPU 메모리로 이동
    print(f"Device tensor is now stored on: {tensor.device}") # Tensor가 저장된 장치: cuda:0
else:
    print("CUDA is not available. Tensor remains on CPU.")
```

### 1.3. Tensor 연산

Tensor는 NumPy 배열과 유사하게 다양한 연산을 지원하며, 이 연산들은 GPU에서도 효율적으로 수행됩니다.

#### 1.3.1. 인덱싱 및 슬라이싱

Tensor의 특정 요소나 부분 집합에 접근하는 방법은 Python 리스트나 NumPy 배열과 유사합니다.

```python
tensor = torch.ones(4, 4) # 4x4 크기의 모든 요소가 1인 Tensor 생성
print(f"Original Tensor:
 {tensor}
")

# 첫 번째 행 접근
print(f"First row: {tensor[0]}
")

# 첫 번째 열 접근
print(f"First column: {tensor[:, 0]}
")

# 마지막 열 접근 (파이썬의 음수 인덱싱과 동일)
print(f"Last column: {tensor[..., -1]}
")

# 특정 범위의 행과 열 접근 (슬라이싱)
print(f"Rows 1 to 2, columns 1 to 2:
 {tensor[1:3, 1:3]}
")

# 특정 요소 값 변경
tensor[0, 0] = 5 # 첫 번째 행, 첫 번째 열의 값을 5로 변경
print(f"Tensor after changing (0,0) element:
 {tensor}
")

# 특정 행/열의 값 변경
tensor[:, 1] = 0 # 두 번째 열의 모든 값을 0으로 변경
print(f"Tensor after setting second column to 0:
 {tensor}
")

# 불리언 인덱싱: 특정 조건을 만족하는 요소만 선택
mask = tensor > 0 # 0보다 큰 요소는 True, 아니면 False인 불리언 Tensor 생성
print(f"Boolean mask:
 {mask}
")
print(f"Elements greater than 0:
 {tensor[mask]}
") # 마스크가 True인 요소들만 1차원 Tensor로 반환
```

#### 1.3.2. Tensor 결합 (Concatenation)

여러 Tensor를 특정 차원을 따라 결합할 수 있습니다. `torch.cat` 함수를 사용하며, `dim` 인수를 통해 결합할 차원을 지정합니다.

```python
tensor = torch.ones(2, 2) # 2x2 Tensor
t1 = torch.cat([tensor, tensor, tensor], dim=0) # 행(dim=0)을 따라 결합
t2 = torch.cat([tensor, tensor, tensor], dim=1) # 열(dim=1)을 따라 결합

print(f"Original Tensor:
 {tensor}
")
print(f"Concatenated along dim=0 (rows):
 {t1}
") # 6x2 Tensor
print(f"Concatenated along dim=1 (columns):
 {t2}
") # 2x6 Tensor
```

#### 1.3.3. 수학 연산

덧셈, 뺄셈, 곱셈, 나눗셈 등 기본적인 수학 연산과 행렬 곱셈, 통계 연산 등을 지원합니다.

```python
tensor_a = torch.tensor([[1, 2], [3, 4]])
tensor_b = torch.tensor([[5, 6], [7, 8]])

# 덧셈 (element-wise)
print(f"Addition (tensor_a + tensor_b):
 {tensor_a + tensor_b}
")
print(f"Addition (torch.add(tensor_a, tensor_b)):
 {torch.add(tensor_a, tensor_b)}
")

# 곱셈 (element-wise)
print(f"Element-wise product (tensor_a * tensor_b):
 {tensor_a * tensor_b}
")
print(f"Element-wise product (torch.mul(tensor_a, tensor_b)):
 {torch.mul(tensor_a, tensor_b)}
")

# 행렬 곱셈 (Matrix Multiplication)
# torch.matmul 또는 @ 연산자 사용
matrix_a = torch.randn(2, 3) # 2x3 랜덤 Tensor
matrix_b = torch.randn(3, 2) # 3x2 랜덤 Tensor
print(f"Matrix A:
 {matrix_a}
")
print(f"Matrix B:
 {matrix_b}
")
print(f"Matrix multiplication (torch.matmul(matrix_a, matrix_b)):
 {torch.matmul(matrix_a, matrix_b)}
") # 2x2 Tensor
print(f"Matrix multiplication (matrix_a @ matrix_b):
 {matrix_a @ matrix_b}
") # 2x2 Tensor

# 통계 연산
stats_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"Original Tensor for stats:
 {stats_tensor}
")
print(f"Sum of all elements: {stats_tensor.sum()}
")
print(f"Mean of all elements: {stats_tensor.mean()}
")
print(f"Max element: {stats_tensor.max()}
")
print(f"Min element: {stats_tensor.min()}
")
print(f"Standard deviation: {stats_tensor.std()}
")

# 특정 차원(dim)을 따라 연산 수행
print(f"Sum along dim=0 (columns sum up):
 {stats_tensor.sum(dim=0)}
") # 각 열의 합
print(f"Mean along dim=1 (rows mean up):
 {stats_tensor.mean(dim=1)}
") # 각 행의 평균
```

#### 1.3.4. 브로드캐스팅 (Broadcasting)

PyTorch는 NumPy와 유사하게 브로드캐스팅 기능을 제공합니다. 크기가 다른 Tensor 간에도 특정 조건이 만족되면 연산을 수행할 수 있도록 Tensor의 크기를 자동으로 확장하는 메커니즘입니다.

**브로드캐스팅 규칙:**
1.  두 Tensor의 차원 수가 다르면, 더 작은 차원 수의 Tensor의 앞쪽에 1이 추가됩니다.
2.  두 Tensor의 각 차원 크기가 일치하거나, 둘 중 하나의 크기가 1이면 해당 차원은 호환됩니다.
3.  어떤 차원도 호환되지 않으면 오류가 발생합니다.

```python
tensor_a = torch.tensor([[1, 2], [3, 4]]) # shape: (2, 2)
scalar = 10 # 스칼라 값

# Tensor와 스칼라 연산: 스칼라가 Tensor의 모든 요소에 브로드캐스팅됨
print(f"Tensor + Scalar:
 {tensor_a + scalar}
")

vector = torch.tensor([10, 20]) # shape: (2,)

# Tensor와 1차원 Tensor 연산: vector가 (2, 1)로 확장되어 브로드캐스팅됨
# [[1, 2],   +   [[10, 20]]  -> [[1, 2],   +   [[10, 20],
#  [3, 4]]                       [3, 4]]       [10, 20]]
print(f"Tensor + Vector:
 {tensor_a + vector}
")

# 브로드캐스팅 불가능한 예시 (shape 불일치)
# tensor_c = torch.tensor([[1, 2, 3]]) # shape: (1, 3)
# print(tensor_a + tensor_c) # 오류 발생: (2,2)와 (1,3)은 브로드캐스팅 불가
```

#### 1.3.5. 인플레이스(In-place) 연산

일부 연산은 Tensor의 내용을 직접 변경합니다. 이러한 연산은 일반적으로 `_` 접미사로 끝납니다 (예: `add_`, `mul_`). 인플레이스 연산은 새로운 Tensor를 생성하지 않으므로 메모리 사용량을 줄일 수 있지만, 원래 Tensor의 값이 변경되므로 주의해서 사용해야 합니다.

```python
tensor = torch.ones(2, 2)
print(f"Original Tensor:
 {tensor}
")

# 인플레이스 덧셈: tensor의 값이 직접 변경됨
tensor.add_(5) # tensor = tensor + 5 와 동일하지만, tensor의 메모리 주소는 그대로 유지
print(f"Tensor after in-place add (add_):
 {tensor}
")

# 인플레이스 곱셈
tensor.mul_(2)
print(f"Tensor after in-place multiply (mul_):
 {tensor}
")
```

#### 1.3.6. 크기 변경 (Reshaping)

Tensor의 크기나 형태를 변경하는 연산입니다. 데이터는 그대로 유지하면서 Tensor의 차원만 재배열합니다.

```python
tensor = torch.arange(1, 10).reshape(3, 3) # 1부터 9까지의 숫자로 3x3 Tensor 생성
print(f"Original Tensor (3x3):
 {tensor}
")

# view(): Tensor의 데이터를 공유하며 새로운 shape의 Tensor를 반환. 연속된 메모리 블록에만 사용 가능.
view_tensor = tensor.view(9) # 1차원 Tensor로 변경
print(f"Tensor after view(9):
 {view_tensor}
")

view_tensor_2x_ = tensor.view(1, 9) # 1x9 Tensor로 변경
print(f"Tensor after view(1, 9):
 {view_tensor_2x_}
")

view_tensor_3x3 = tensor.view(3, 3) # 3x3 Tensor로 변경
print(f"Tensor after view(3, 3):
 {view_tensor_3x3}
")

view_tensor_auto = tensor.view(-1, 3) # -1은 다른 차원에 맞춰 자동으로 계산 (여기서는 3x3)
print(f"Tensor after view(-1, 3):
 {view_tensor_auto}
")

# reshape(): view와 유사하지만, Tensor의 메모리가 연속적이지 않아도 동작. 필요한 경우 데이터를 복사.
reshape_tensor = tensor.reshape(9)
print(f"Tensor after reshape(9):
 {reshape_tensor}
")

# squeeze(): 차원 중 크기가 1인 차원을 제거
squeezed_tensor = torch.randn(1, 3, 1, 4).squeeze() # (1, 3, 1, 4) -> (3, 4)
print(f"Original for squeeze: {torch.randn(1, 3, 1, 4).shape}, Squeezed: {squeezed_tensor.shape}
")

# unsqueeze(): 특정 위치에 크기가 1인 차원을 추가
unsqueezed_tensor = torch.randn(3, 4).unsqueeze(0) # (3, 4) -> (1, 3, 4)
print(f"Original for unsqueeze: {torch.randn(3, 4).shape}, Unsqueezed: {unsqueezed_tensor.shape}
")
```

### 1.4. Tensor와 NumPy 변환

PyTorch Tensor와 NumPy 배열은 서로 쉽게 변환할 수 있습니다. 이 변환은 매우 효율적이며, **CPU 상의 Tensor와 NumPy 배열은 메모리를 공유합니다.** 즉, 한쪽을 변경하면 다른 쪽도 변경됩니다. GPU Tensor는 NumPy 배열과 직접 메모리를 공유할 수 없으므로, GPU Tensor를 NumPy로 변환하려면 먼저 CPU로 이동시켜야 합니다.

```python
import torch
import numpy as np

# 1. Tensor를 NumPy 배열로 변환
torch_tensor = torch.ones(5) # CPU Tensor 생성
numpy_array = torch_tensor.numpy() # Tensor를 NumPy 배열로 변환
print(f"Tensor to NumPy:
 {numpy_array}
")

# NumPy 배열 변경 시 Tensor도 변경됨 (메모리 공유)
np.add(numpy_array, 1, out=numpy_array) # NumPy 배열의 모든 요소에 1을 더함
print(f"NumPy array after modification:
 {numpy_array}
")
print(f"Tensor after NumPy modification:
 {torch_tensor}
") # torch_tensor도 변경됨

# 2. NumPy 배열을 Tensor로 변환
numpy_array_2 = np.ones(5)
torch_tensor_2 = torch.from_numpy(numpy_array_2) # NumPy 배열로부터 Tensor 생성
print(f"NumPy to Tensor:
 {torch_tensor_2}
")

# Tensor 변경 시 NumPy 배열도 변경됨 (메모리 공유)
torch_tensor_2.add_(1) # Tensor의 모든 요소에 1을 더함 (인플레이스 연산)
print(f"Tensor after modification:
 {torch_tensor_2}
")
print(f"NumPy array after Tensor modification:
 {numpy_array_2}
") # numpy_array_2도 변경됨

# GPU Tensor의 경우
if torch.cuda.is_available():
    gpu_tensor = torch.ones(5, device="cuda")
    # numpy_array_from_gpu = gpu_tensor.numpy() # 오류 발생: GPU Tensor는 직접 NumPy로 변환 불가
    numpy_array_from_gpu = gpu_tensor.cpu().numpy() # 먼저 CPU로 이동 후 변환
    print(f"NumPy array from GPU Tensor (after moving to CPU):
 {numpy_array_from_gpu}
")
```

### 1.5. Tensor의 CPU/GPU 이동

PyTorch는 GPU를 활용하여 딥러닝 연산 속도를 크게 향상시킬 수 있습니다. Tensor는 `to()` 메서드를 사용하여 CPU와 GPU 간에 쉽게 이동할 수 있습니다.

```python
import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# CPU에 Tensor 생성
cpu_tensor = torch.tensor([1, 2, 3])
print(f"CPU Tensor: {cpu_tensor.device}
")

# Tensor를 GPU로 이동
if device.type == 'cuda':
    gpu_tensor = cpu_tensor.to(device) # 또는 cpu_tensor.cuda()
    print(f"GPU Tensor: {gpu_tensor.device}
")
    # GPU에서 연산 수행
    result_gpu = gpu_tensor * 2
    print(f"Result on GPU: {result_gpu}
")

    # GPU Tensor를 다시 CPU로 이동
    result_cpu = result_gpu.to("cpu") # 또는 result_gpu.cpu()
    print(f"Result back on CPU: {result_cpu}
")
else:
    print("GPU not available, all operations remain on CPU.")

# 모델 학습 시에는 일반적으로 모든 Tensor를 동일한 장치로 이동시켜야 합니다.
# model.to(device)
# inputs = inputs.to(device)
# labels = labels.to(device)
```

## 2. 자동 미분 (Autograd)

`torch.autograd`는 PyTorch의 핵심 기능 중 하나로, 신경망 학습에 필수적인 자동 미분 엔진입니다. 딥러닝 모델은 수많은 파라미터(가중치와 편향)를 가지고 있으며, 이 파라미터들을 최적화하기 위해 경사 하강법(Gradient Descent)과 같은 최적화 알고리즘을 사용합니다. 경사 하강법은 손실 함수(Loss Function)에 대한 각 파라미터의 기울기(Gradient)를 계산해야 하는데, `autograd`는 이 복잡한 기울기 계산을 자동으로 수행해줍니다.

### 2.1. 자동 미분이란?

자동 미분은 함수를 구성하는 모든 연산에 대해 미분값을 자동으로 계산하는 기술입니다. PyTorch의 `autograd`는 Tensor에 대한 모든 연산을 추적하여 **계산 그래프(Computational Graph)**를 동적으로 생성합니다. 이 그래프를 통해 순전파(forward pass) 시에는 연산 결과를 계산하고, 역전파(backward pass) 시에는 이 그래프를 역으로 탐색하며 각 파라미터에 대한 기울기를 효율적으로 계산합니다.

### 2.2. `requires_grad` 속성

PyTorch Tensor는 `requires_grad`라는 불리언 속성을 가집니다. 이 속성이 `True`로 설정된 Tensor에 대해 수행되는 모든 연산은 `autograd`에 의해 추적되어 계산 그래프에 기록됩니다. 기본적으로 Tensor는 `requires_grad=False`로 생성됩니다.

*   **`requires_grad=True`**: 이 Tensor에 대한 모든 연산이 추적되며, 역전파 시 이 Tensor에 대한 기울기가 계산됩니다. 신경망의 학습 가능한 파라미터(예: `nn.Linear` 레이어의 `weight`와 `bias`)는 자동으로 `requires_grad=True`로 설정됩니다.
*   **`requires_grad=False`**: 이 Tensor에 대한 연산은 추적되지 않으며, 역전파 시 이 Tensor에 대한 기울기는 계산되지 않습니다. 모델의 입력 데이터나 레이블 등 학습 대상이 아닌 Tensor는 일반적으로 `requires_grad=False`입니다.

```python
import torch

# requires_grad=True로 설정된 Tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"x.requires_grad: {x.requires_grad}") # True

# requires_grad=False로 설정된 Tensor (기본값)
y = torch.tensor([4.0, 5.0, 6.0]) # requires_grad=False
print(f"y.requires_grad: {y.requires_grad}") # False

# x와 y를 이용한 연산
z = x + y # z = [5.0, 7.0, 9.0]
# 연산 결과 Tensor의 requires_grad는 입력 Tensor 중 하나라도 True이면 True가 됩니다.
print(f"z.requires_grad (from x+y): {z.requires_grad}") # True (x가 True이므로)

# y만 이용한 연산
w = y * 2 # w = [8.0, 10.0, 12.0]
print(f"w.requires_grad (from y*2): {w.requires_grad}") # False (y가 False이므로)

# requires_grad를 False로 변경 (기울기 계산을 멈추고 싶을 때)
x.requires_grad_(False) # 인플레이스(in-place)로 requires_grad 속성을 변경
print(f"x.requires_grad after requires_grad_(False): {x.requires_grad}") # False
```

### 2.3. 기울기 계산 (`backward()` 메서드)

계산 그래프가 생성된 후, 최종 결과 Tensor(일반적으로 손실 함수 값)에 대해 `.backward()` 메서드를 호출하면 `autograd`는 그래프를 역방향으로 탐색하며 각 `requires_grad=True`로 설정된 Tensor에 대한 기울기를 계산하고, 그 결과를 해당 Tensor의 `.grad` 속성에 저장합니다.

**중요:**
*   `.backward()`는 스칼라(scalar) Tensor에 대해서만 직접 호출할 수 있습니다. 만약 최종 결과 Tensor가 스칼라가 아닌 경우 (예: 여러 요소로 구성된 손실 벡터), `backward()` 호출 시 `gradient` 인수로 해당 Tensor와 동일한 shape의 Tensor를 전달해야 합니다. 일반적으로는 `sum()`이나 `mean()` 등을 사용하여 스칼라 값으로 만든 후 `backward()`를 호출합니다.
*   기울기는 `.grad` 속성에 **누적**됩니다. 따라서 새로운 역전파를 수행하기 전에 이전 기울기를 `zero_()` 메서드를 사용하여 0으로 초기화해야 합니다. 이는 옵티마이저(Optimizer)의 `zero_grad()` 메서드를 통해 자동으로 처리됩니다.

```python
import torch

# 예시 1: 간단한 스칼라 미분
x = torch.tensor(2.0, requires_grad=True) # 스칼라 Tensor
y = x**2 # y = 4.0
z = y * 3  # z = 12.0

# z는 스칼라 값이므로 바로 backward() 호출 가능
z.backward()

# x에 대한 z의 기울기 (dz/dx) 확인
# z = 3 * y = 3 * (x^2) = 3x^2
# dz/dx = 6x
# x=2.0 이므로 dz/dx = 6 * 2.0 = 12.0
print(f"Gradient of z with respect to x (dz/dx): {x.grad}
") # Tensor(12.)

# 기울기 초기화 (다음 backward 호출을 위해)
x.grad.zero_()
print(f"x.grad after zero_(): {x.grad}
") # Tensor(0.)

# 예시 2: 여러 Tensor에 대한 미분 및 누적 확인
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)

c = a * b # c = 12.0
d = c.sum() # d = 12.0 (스칼라)

d.backward() # d는 스칼라

# a에 대한 d의 기울기 (dd/da)
# d = a * b 이므로 dd/da = b
print(f"Gradient of d with respect to a (dd/da): {a.grad}
") # Tensor(4.)

# b에 대한 d의 기울기 (dd/db)
# d = a * b 이므로 dd/db = a
print(f"Gradient of d with respect to b (dd/db): {b.grad}
") # Tensor(3.)

# 기울기 누적 예시
e = torch.tensor(5.0, requires_grad=True)
f = e * 2 # f = 10.0
g = e * 3 # g = 15.0
h = f + g # h = 25.0

h.backward() # h는 스칼라

# e에 대한 h의 기울기 (dh/de)
# h = 2e + 3e = 5e
# dh/de = 5
print(f"Gradient of h with respect to e (dh/de): {e.grad}
") # Tensor(5.)

# 다시 backward() 호출 시 기울기 누적
h.backward()
print(f"Gradient of h with respect to e (dh/de) after second backward(): {e.grad}
") # Tensor(10.) (5 + 5)
# 따라서 매 학습 스텝마다 optimizer.zero_grad()를 호출하여 기울기를 초기화해야 합니다.
```

### 2.4. 계산 그래프 (Computational Graph)와 `grad_fn`

`autograd`는 Tensor에 대한 연산이 수행될 때마다 **계산 그래프(Computational Graph)**를 동적으로 생성합니다. 이 그래프는 연산의 흐름을 나타내는 **방향성 비순환 그래프(DAG, Directed Acyclic Graph)** 형태를 가집니다.

*   **노드(Node)**: 그래프의 각 노드는 Tensor 또는 연산(Operation)을 나타냅니다.
    *   **잎(Leaf) 노드**: `requires_grad=True`로 직접 생성된 입력 Tensor (예: 모델의 가중치).
    *   **중간/루트 노드**: 연산을 통해 생성된 Tensor.
*   **엣지(Edge)**: 연산의 입력과 출력을 연결합니다.

각 Tensor는 자신을 생성한 연산에 대한 참조를 `.grad_fn` 속성에 저장합니다. 이 `grad_fn`은 역전파 시 해당 연산의 기울기를 계산하는 데 필요한 정보를 가지고 있습니다. `.backward()` 메서드가 호출되면, `autograd`는 이 `grad_fn` 체인을 따라 그래프를 역방향으로 이동하며 각 연산에 대한 기울기를 계산하고, 이를 해당 Tensor의 `.grad` 속성에 누적합니다.

PyTorch의 **동적 계산 그래프(Dynamic Computational Graph)**는 모델을 실행하는 시점(Define-by-Run)에 그래프를 생성합니다. 이는 TensorFlow 1.x의 정적 계산 그래프(Define-and-Run)와 대조되는 특징으로, PyTorch의 유연성과 쉬운 디버깅을 가능하게 하는 핵심 요소입니다.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x + 3 # AddBackward0
z = y * y # MulBackward0
out = z.mean() # MeanBackward0

print(f"x: {x}
")
print(f"y: {y}, grad_fn: {y.grad_fn}
") # AddBackward0
print(f"z: {z}, grad_fn: {z.grad_fn}
") # MulBackward0
print(f"out: {out}, grad_fn: {out.grad_fn}
") # MeanBackward0

# out.grad_fn을 통해 그래프를 역으로 추적할 수 있습니다.
# out -> MeanBackward0 -> MulBackward0 -> AddBackward0 -> x
```

### 2.5. 기울기 추적 중지: `torch.no_grad()`와 `detach()`

모델 학습 시에는 기울기 계산이 필수적이지만, 모델 평가(evaluation)나 추론(inference) 시에는 기울기 계산이 필요 없습니다. 이때 불필요한 기울기 계산을 비활성화하여 메모리 사용량을 줄이고 연산 속도를 높일 수 있습니다.

*   **`torch.no_grad()`**:
    *   이 컨텍스트 매니저 블록 내에서 생성되거나 연산되는 모든 Tensor는 `requires_grad=False`가 됩니다.
    *   주로 모델의 평가(evaluation) 단계나 추론(inference) 단계에서 사용됩니다. 이 단계에서는 모델의 가중치를 업데이트할 필요가 없으므로 기울기 계산이 불필요합니다.
    *   메모리 사용량을 줄이고 연산 속도를 향상시키는 효과가 있습니다.

*   **`detach()`**:
    *   `detach()` 메서드는 현재 Tensor와 계산 그래프로부터 분리된 새로운 Tensor를 반환합니다.
    *   반환된 새로운 Tensor는 `requires_grad=False`이며, 원래 Tensor의 연산 기록에 영향을 주지 않습니다.
    *   원본 Tensor의 값을 사용하되, 그 값에 대한 기울기 계산은 원치 않을 때 유용합니다. 예를 들어, 중간 계산 결과를 로깅하거나 시각화할 때 사용할 수 있습니다.
    *   `detach()`된 Tensor는 원본 Tensor와 메모리를 공유할 수 있습니다. 따라서 `detach()`된 Tensor를 인플레이스(in-place)로 수정하면 원본 Tensor도 변경될 수 있으므로 주의해야 합니다.

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
print(f"Original x: {x}, requires_grad: {x.requires_grad}
")

# 1. torch.no_grad() 사용 예시
with torch.no_grad():
    y = x * 2 # 이 연산은 기울기 추적되지 않음
    print(f"y inside no_grad(): {y}, requires_grad: {y.requires_grad}
") # False

# 2. detach() 사용 예시
z = x.detach() # x로부터 분리된 새로운 Tensor z 생성
print(f"z after detach(): {z}, requires_grad: {z.requires_grad}
") # False

# z를 변경해도 x의 기울기 계산에 영향 없음 (단, z를 인플레이스 수정 시 x도 변경될 수 있음)
z[0] = 10.0
print(f"x after z modification (z[0]=10.0): {x}
") # x는 변경되지 않음 (새로운 Tensor가 생성되었으므로)

# 만약 detach()된 Tensor를 인플레이스 수정하고 싶다면 clone()을 사용하는 것이 안전합니다.
z_clone = x.detach().clone()
z_clone[0] = 20.0
print(f"x after z_clone modification (z_clone[0]=20.0): {x}
") # x는 변경되지 않음

# x를 통해 연산 후 backward() 호출 (y와 z는 그래프에서 분리되었으므로 x의 기울기 계산에 영향 없음)
w = x * 3
w.sum().backward()
print(f"x.grad after backward(): {x.grad}
") # x의 기울기는 정상적으로 계산됨 (Tensor([3., 3.]))

# no_grad()와 detach()의 주요 차이점:
# - no_grad(): 특정 코드 블록 전체의 기울기 계산을 비활성화합니다.
# - detach(): 특정 Tensor 하나를 계산 그래프에서 분리하여 새로운 Tensor를 만듭니다.
#             이 새로운 Tensor는 원본 Tensor의 값을 가지지만, 기울기 계산과는 무관합니다.
```

이제 PyTorch의 기본 데이터 구조인 Tensor와 핵심 기능인 `autograd`에 대해 심층적으로 이해했습니다. 다음 장에서는 이러한 요소들을 조합하여 딥러닝 모델을 구성하는 **신경망 모듈(nn.Module)**과 **옵티마이저(Optimizer)**에 대해 자세히 알아보겠습니다.