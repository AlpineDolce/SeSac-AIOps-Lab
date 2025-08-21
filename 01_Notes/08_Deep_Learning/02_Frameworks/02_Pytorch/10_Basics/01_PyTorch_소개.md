<h2>PyTorch 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-18

<h2>문서 목표</h2>
이 문서는 PyTorch의 핵심 개념을 이해하고 실무에 활용하는 데 필요한 기초 지식을 제공합니다. PyTorch의 정의와 특징, 설치 방법부터 핵심 데이터 구조인 Tensor의 생성 및 연산, 그리고 딥러닝 모델 학습의 기반이 되는 자동 미분(Autograd) 시스템까지 상세히 다룹니다. 이 문서를 통해 PyTorch를 처음 접하는 개발자나 연구자가 딥러닝 프로젝트를 성공적으로 시작할 수 있는 기반을 다지는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. PyTorch란?](#1-pytorch란)
- [2. PyTorch의 주요 특징](#2-pytorch의-주요-특징)
- [3. 설치 및 환경 설정](#3-설치-및-환경-설정)
  - [3.1. 설치](#31-설치)
  - [3.2. 환경 설정 및 확인](#32-환경-설정-및-확인)
- [4. PyTorch의 핵심 데이터 구조: Tensor](#4-pytorch의-핵심-데이터-구조-tensor)
  - [4.1. Tensor란?](#41-tensor란)
  - [4.2. Tensor 생성](#42-tensor-생성)
  - [4.3. Tensor 연산](#43-tensor-연산)
  - [4.4. Tensor와 NumPy 변환](#44-tensor와-numpy-변환)
  - [4.5. Tensor의 CPU/GPU 이동](#45-tensor의-cpugpu-이동)
- [5. 자동 미분 (Autograd)](#5-자동-미분-autograd)
  - [5.1. 자동 미분이란?](#51-자동-미분이란)
  - [5.2. `requires_grad` 속성](#52-requires_grad-속성)
  - [5.3. `backward()` 메서드](#53-backward-메서드)
  - [5.4. `torch.no_grad()`와 `detach()`](#54-torchnograd와-detach)
  - [5.5. 계산 그래프 (Computational Graph)](#55-계산-그래프-computational-graph)

---

# PyTorch 소개

## 1. PyTorch란?

PyTorch는 Facebook의 AI 연구팀(FAIR)이 개발한 오픈소스 머신러닝 라이브러리입니다. Python을 기반으로 하며, 다음과 같은 두 가지 핵심 기능을 제공합니다.

*   **강력한 GPU 가속을 지원하는 Tensor 계산**: NumPy와 유사하지만 GPU를 활용하여 계산 속도를 크게 향상시킬 수 있는 다차원 배열인 Tensor를 제공합니다.
*   **유연하고 직관적인 자동 미분 기반의 딥러닝 연구 플랫폼**: `autograd` 시스템을 통해 동적 계산 그래프(Dynamic Computational Graph)를 지원하여, 복잡한 딥러닝 모델을 쉽고 유연하게 구축하고 학습시킬 수 있습니다.

## 2. PyTorch의 주요 특징

*   **Pythonic**: PyTorch는 Python의 철학을 따라 직관적이고 간결한 API를 제공합니다. Python의 다양한 라이브러리(NumPy, SciPy, Matplotlib 등)와 자연스럽게 통합되어 데이터 처리 및 시각화가 용이합니다.
*   **동적 계산 그래프 (Define-by-Run)**: PyTorch는 모델을 실행하는 시점(runtime)에 계산 그래프를 생성합니다. 이를 통해 가변적인 입력이나 모델 구조를 가진 딥러닝 모델(예: 자연어 처리의 RNN)을 쉽게 구현하고 디버깅할 수 있습니다. 이는 정적 계산 그래프(Define-and-Run)를 사용하는 TensorFlow 1.x 버전과 대조되는 가장 큰 특징입니다.
*   **쉬운 디버깅**: 동적 계산 그래프 덕분에 Python 디버거(e.g., `pdb`)를 사용하여 모델의 어느 지점에서든 텐서의 값이나 기울기를 쉽게 확인할 수 있습니다.
*   **활발한 커뮤니티와 풍부한 생태계**: PyTorch는 전 세계 수많은 연구자와 개발자들이 참여하는 거대한 커뮤니티를 가지고 있습니다. 최신 연구 논문들이 PyTorch로 구현되어 공개되는 경우가 많으며, `TorchVision`, `TorchText`, `PyTorch Geometric` 등 다양한 도메인을 위한 공식 라이브러리와 `Hugging Face Transformers`와 같은 강력한 서드파티 라이브러리 생태계를 자랑합니다.
*   **간편한 모델 배포**: `TorchScript`를 통해 모델을 정적 그래프로 변환하여 Python 런타임에 의존하지 않는 환경(C++, 모바일 등)에 쉽게 배포할 수 있으며, `TorchServe`를 통해 프로덕션 환경에서 모델을 효율적으로 서빙할 수 있습니다.

## 3. 설치 및 환경 설정

### 3.1. 설치

PyTorch는 공식 웹사이트([https://pytorch.org/](https://pytorch.org/))에서 자신의 개발 환경에 맞는 설치 명령어를 쉽게 생성하여 사용할 수 있습니다.

**설치 옵션:**

*   **PyTorch Build**: 안정적인 최신 버전(Stable) 또는 최신 기능이 포함된 실험적인 버전(Preview)을 선택할 수 있습니다.
*   **Your OS**: 운영체제(Linux, Mac, Windows)를 선택합니다.
*   **Package**: 패키지 매니저(Conda, Pip)를 선택합니다.
*   **Language**: 프로그래밍 언어(Python, C++/Java)를 선택합니다.
*   **Compute Platform**: GPU 사용 여부에 따라 CUDA 버전 또는 CPU를 선택합니다.

**예시 (Windows, Pip, Python, CUDA 12.1):**

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3.2. 환경 설정 및 확인

설치가 완료되면 Python 스크립트에서 PyTorch를 `import`하고, 버전 및 GPU 사용 가능 여부를 확인할 수 있습니다.

```python
import torch

# PyTorch 버전 확인
print(f"PyTorch Version: {torch.__version__}")

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    # 현재 사용 가능한 GPU 개수
    print(f"Available GPUs: {torch.cuda.device_count()}")
    # 현재 사용 중인 GPU 이름
    print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    # GPU 사용 설정
    device = torch.device("cuda")
    print("GPU is available.")
else:
    # CPU 사용 설정
    device = torch.device("cpu")
    print("GPU not available, using CPU.")

# 간단한 텐서 연산 테스트
x = torch.rand(5, 3).to(device)
print("
Tensor on device:", x.device)
print(x)
```

이제 PyTorch를 사용할 준비가 모두 끝났습니다. 다음 장에서는 PyTorch의 핵심 데이터 구조인 **Tensor**에 대해 더 자세히 알아보겠습니다.

## 4. PyTorch의 핵심 데이터 구조: Tensor

### 4.1. Tensor란?

**Tensor(텐서)**는 PyTorch의 핵심 데이터 구조로, 다차원 배열입니다. NumPy의 `ndarray`와 매우 유사하지만, GPU 가속을 지원하여 딥러닝 모델의 대규모 수치 연산을 효율적으로 처리할 수 있다는 강력한 장점이 있습니다. 텐서는 스칼라(0차원), 벡터(1차원), 행렬(2차원)을 포함하는 일반화된 개념입니다.

### 4.2. Tensor 생성

PyTorch에서 텐서를 생성하는 방법은 다양합니다.

```python
import torch
import numpy as np

# 1. 초기화되지 않은 텐서 생성 (메모리 할당만)
x = torch.empty(5, 3)
print("Empty Tensor:
", x)

# 2. 무작위로 초기화된 텐서 생성
x = torch.rand(5, 3)
print("
Random Tensor:
", x)

# 3. 0으로 채워진 텐서 생성
x = torch.zeros(5, 3, dtype=torch.long)
print("
Zeros Tensor:
", x)

# 4. 데이터로부터 직접 텐서 생성
x = torch.tensor([[1, 2], [3, 4]])
print("
Tensor from data:
", x)

# 5. NumPy 배열로부터 텐서 생성
numpy_array = np.array([5, 6, 7])
torch_tensor = torch.from_numpy(numpy_array)
print("
Tensor from NumPy array:
", torch_tensor)

# 6. 특정 크기의 텐서 생성 (ones, full 등)
x = torch.ones(2, 2)
print("
Ones Tensor:
", x)

x = torch.full((2, 2), 7.0) # 모든 요소를 7.0으로 채움
print("
Full Tensor:
", x)

# 텐서의 크기 확인
print("
Shape of x:", x.size()) # 또는 x.shape
```

### 4.3. Tensor 연산

텐서는 NumPy 배열과 유사하게 다양한 연산을 지원합니다.

```python
import torch

x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

# 1. 덧셈
print("Addition (x + y):
", x + y)
print("Addition (torch.add(x, y)):
", torch.add(x, y))

# 2. 인플레이스(in-place) 덧셈 (y의 값이 변경됨)
y.add_(x)
print("In-place Addition (y.add_(x)):
", y)

# 3. 곱셈 (요소별 곱셈)
print("Multiplication (x * y):
", x * y)

# 4. 행렬 곱셈
x = torch.randn(2, 3)
y = torch.randn(3, 2)
print("Matrix Multiplication (torch.matmul(x, y)):
", torch.matmul(x, y))
print("Matrix Multiplication (x @ y):
", x @ y)

# 5. 슬라이싱 및 인덱싱
x = torch.rand(4, 4)
print("
Original Tensor for slicing:
", x)
print("First column:
", x[:, 0])
print("First row:
", x[0, :])
print("Element at (1, 2):
", x[1, 2])

# 6. 크기 변경 (Reshaping)
x = torch.randn(4, 4)
y = x.view(16) # 1차원 텐서로 변경
z = x.view(2, 8) # 2x8 텐서로 변경
a = x.view(-1, 8) # -1은 다른 차원에 맞춰 자동으로 계산
print("
Original Tensor for reshaping:
", x)
print("Reshaped to 1D:
", y)
print("Reshaped to 2x8:
", z)
print("Reshaped to (-1, 8):
", a)
```

### 4.4. Tensor와 NumPy 변환

PyTorch 텐서와 NumPy 배열은 서로 쉽게 변환할 수 있습니다. 이들은 메모리를 공유하므로, 한쪽을 변경하면 다른 쪽도 변경됩니다.

```python
import torch
import numpy as np

# 1. Tensor를 NumPy 배열로 변환
torch_tensor = torch.ones(5)
numpy_array = torch_tensor.numpy()
print("Tensor to NumPy:
", numpy_array)

# NumPy 배열 변경 시 Tensor도 변경됨
np.add(numpy_array, 1, out=numpy_array)
print("NumPy array after modification:
", numpy_array)
print("Tensor after NumPy modification:
", torch_tensor)

# 2. NumPy 배열을 Tensor로 변환
numpy_array = np.ones(5)
torch_tensor = torch.from_numpy(numpy_array)
print("
NumPy to Tensor:
", torch_tensor)

# Tensor 변경 시 NumPy 배열도 변경됨
torch_tensor.add_(1)
print("Tensor after modification:
", torch_tensor)
print("NumPy array after Tensor modification:
", numpy_array)
```

### 4.5. Tensor의 CPU/GPU 이동

PyTorch 텐서는 `cpu()` 또는 `cuda()` 메서드를 사용하여 CPU와 GPU 간에 쉽게 이동할 수 있습니다. GPU를 사용하면 연산 속도를 크게 향상시킬 수 있습니다.

```python
import torch

# GPU 사용 가능 여부 확인 (이전 섹션에서 정의된 device 변수 사용)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# CPU에 텐서 생성
cpu_tensor = torch.tensor([1, 2, 3])
print("
CPU Tensor:", cpu_tensor.device)

# 텐서를 GPU로 이동
if device.type == 'cuda':
    gpu_tensor = cpu_tensor.to(device)
    print("GPU Tensor:", gpu_tensor.device)
    # GPU에서 연산 수행
    result_gpu = gpu_tensor * 2
    print("Result on GPU:", result_gpu)

# GPU 텐서를 다시 CPU로 이동
if device.type == 'cuda':
    result_cpu = result_gpu.to("cpu")
    print("Result back on CPU:", result_cpu)
```

다음 장에서는 PyTorch의 핵심 기능 중 하나인 **자동 미분(Autograd)** 시스템에 대해 자세히 알아보겠습니다.

## 5. 자동 미분 (Autograd)

### 5.1. 자동 미분이란?

**자동 미분(Autograd)**은 PyTorch의 핵심 기능 중 하나로, 딥러닝 모델 학습에 필수적인 역전파(Backpropagation) 알고리즘을 효율적으로 구현할 수 있도록 돕습니다. 모델의 파라미터(가중치와 편향)를 업데이트하기 위해서는 손실 함수(Loss Function)에 대한 각 파라미터의 기울기(Gradient)를 계산해야 합니다. Autograd는 사용자가 정의한 연산 그래프를 추적하여 이 기울기를 자동으로 계산해줍니다.

### 5.2. `requires_grad` 속성

PyTorch 텐서는 `requires_grad`라는 속성을 가집니다. 이 속성이 `True`로 설정된 텐서에 대해 수행되는 모든 연산은 추적되어 계산 그래프에 기록됩니다. 기본적으로 텐서는 `requires_grad=False`로 생성됩니다. 모델의 학습 가능한 파라미터(예: `nn.Linear`의 `weight`와 `bias`)는 자동으로 `requires_grad=True`로 설정됩니다.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=False)

print("x.requires_grad:", x.requires_grad)
print("y.requires_grad:", y.requires_grad)

z = x + y
print("z.requires_grad:", z.requires_grad) # x가 True이므로 z도 True

w = y * 2
print("w.requires_grad:", w.requires_grad) # y가 False이므로 w도 False
```

### 5.3. `backward()` 메서드

`backward()` 메서드는 스칼라 값(보통 손실 함수)에 대해 호출됩니다. 이 메서드가 호출되면 Autograd는 계산 그래프를 역방향으로 탐색하며, `requires_grad=True`로 설정된 모든 텐서에 대해 기울기를 계산하고, 그 결과를 해당 텐서의 `.grad` 속성에 저장합니다.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**2 # y = 4.0
z = y * 3  # z = 12.0

# z는 스칼라 값이므로 바로 backward() 호출 가능
z.backward() 

# x에 대한 z의 기울기 (dz/dx) 확인
# z = 3 * x^2 이므로 dz/dx = 6x
# x=2.0 이므로 dz/dx = 6 * 2.0 = 12.0
print("Gradient of z with respect to x (dz/dx):
", x.grad)

# 여러 번 backward() 호출 시 기울기가 누적되므로, 보통 optimizer.zero_grad()로 초기화
x.grad.zero_() # 기울기 초기화

a = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([4.0], requires_grad=True)

c = a * b
d = c.sum()

d.backward() # d는 스칼라

print("Gradient of d with respect to a (dd/da):
", a.grad) # dd/da = b = 4.0
print("Gradient of d with respect to b (dd/db):
", b.grad) # dd/db = a = 3.0
```

### 5.4. `torch.no_grad()`와 `detach()`

모델 학습 시에는 기울기 계산이 필수적이지만, 모델 평가(evaluation)나 추론(inference) 시에는 기울기 계산이 필요 없습니다. 이때 `torch.no_grad()` 컨텍스트 매니저를 사용하면 기울기 계산을 비활성화하여 메모리 사용량을 줄이고 연산 속도를 높일 수 있습니다.

`detach()` 메서드는 현재 텐서와 계산 그래프로부터 분리된 새로운 텐서를 반환합니다. 이 새로운 텐서는 `requires_grad=False`이며, 원래 텐서의 연산 기록에 영향을 주지 않습니다. 이는 특정 텐서의 값을 사용하되, 그 값에 대한 기울기 계산은 원치 않을 때 유용합니다.

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

# 1. torch.no_grad() 사용
with torch.no_grad():
    y = x * 2
    print("y.requires_grad inside no_grad():", y.requires_grad) # False

# 2. detach() 사용
z = x.detach()
print("z.requires_grad after detach():", z.requires_grad) # False

# z를 변경해도 x의 기울기 계산에 영향 없음
z[0] = 10.0
print("x after z modification:", x) # x는 변경되지 않음

# x를 통해 연산 후 backward() 호출
w = x * 3
w.sum().backward()
print("x.grad after backward():", x.grad) # x의 기울기는 정상적으로 계산됨
```

### 5.5. 계산 그래프 (Computational Graph)

Autograd는 텐서 연산이 수행될 때마다 **계산 그래프(Computational Graph)**를 동적으로 생성합니다. 이 그래프는 연산 노드(Operation Node)와 텐서 노드(Tensor Node)로 구성됩니다. 각 연산 노드는 입력 텐서와 출력 텐서, 그리고 역전파 시 기울기를 계산하는 데 필요한 정보를 저장합니다. `backward()` 메서드가 호출되면 이 그래프를 따라 역방향으로 이동하며 기울기를 계산합니다.

PyTorch의 동적 계산 그래프는 모델의 유연한 설계와 쉬운 디버깅을 가능하게 하는 핵심 요소입니다.

---

다음 장에서는 PyTorch의 **신경망 모듈(nn.Module)**과 **옵티마이저(Optimizer)**에 대해 자세히 알아보겠습니다.
