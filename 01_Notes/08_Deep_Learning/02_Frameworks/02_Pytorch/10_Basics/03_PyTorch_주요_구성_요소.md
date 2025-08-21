<h2>PyTorch 주요 구성 요소</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-18

<h2>문서 목표</h2>
이 문서는 PyTorch를 사용하여 딥러닝 모델을 구축하는 데 필수적인 핵심 구성 요소들을 심층적으로 다룹니다. 신경망의 기반이 되는 `nn.Module`과 학습 가능한 파라미터인 `nn.Parameter`의 역할과 사용법을 설명합니다. 또한, 모델 학습의 방향을 제시하는 다양한 손실 함수(`nn.MSELoss`, `nn.CrossEntropyLoss`, `nn.BCEWithLogitsLoss` 등)와 모델 파라미터를 업데이트하는 옵티마이저(`optim.SGD`, `optim.Adam` 등)의 원리 및 활용법을 상세히 제시합니다. 마지막으로, 모델의 성능을 객관적으로 평가하는 메트릭의 중요성과 구현 방법을 다루어, PyTorch를 활용한 효과적인 딥러닝 모델 개발 역량을 강화하는 데 기여하고자 합니다.

<h2>목차</h2>

- [1. `nn.Module`: 모든 신경망의 기초](#1-nnmodule-모든-신경망의-기초)
  - [1.1. `nn.Module`의 주요 기능](#11-nnmodule의-주요-기능)
  - [1.2. `nn.Module` 사용법](#12-nnmodule-사용법)
  - [1.3. `nn.Sequential`: 간편한 모델 구축](#13-nnsequential-간편한-모델-구축)
- [2. `nn.Parameter`: 학습 가능한 Tensor](#2-nnparameter-학습-가능한-tensor)
- [3. 손실 함수 (Loss Function)](#3-손실-함수-loss-function)
  - [3.1. `nn.MSELoss`: 평균 제곱 오차](#31-nnmseloss-평균-제곱-오차)
  - [3.2. `nn.CrossEntropyLoss`: 교차 엔트로피 손실](#32-nncrossentropyLoss-교차-엔트로피-손실)
  - [3.3. `nn.BCELoss` 및 `nn.BCEWithLogitsLoss`: 이진 교차 엔트로피 손실](#33-nnbceloss-및-nnbcewithlogitsloss-이진-교차-엔트로피-손실)
- [4. 옵티마이저 (Optimizer)](#4-옵티마이저-optimizer)
  - [4.1. `torch.optim` 패키지](#41-torchoptim-패키지)
  - [4.2. 주요 옵티마이저](#42-주요-옵티마이저)
  - [4.3. 옵티마이저 사용 흐름](#43-옵티마이저-사용-흐름)
  - [4.4. 학습률 스케줄러 (Learning Rate Scheduler)](#44-학습률-스케줄러-learning-rate-scheduler)
- [5. 메트릭 (Metrics)](#5-메트릭-metrics)
  - [5.1. 메트릭의 중요성](#51-메트릭의-중요성)
  - [5.2. 일반적인 메트릭](#52-일반적인-메트릭)
  - [5.3. 메트릭 구현](#53-메트릭-구현)

--- 

# PyTorch 주요 구성 요소

앞서 Tensor와 `autograd`에 대해 학습했습니다. 이제 이러한 기본 요소들을 바탕으로 딥러닝 모델을 구축하고 학습시키는 데 필요한 PyTorch의 핵심 구성 요소들을 살펴보겠습니다. `torch.nn` 패키지는 신경망을 구축하기 위한 다양한 데이터 구조와 레이어, 손실 함수 등을 제공하며, `torch.optim` 패키지는 모델의 파라미터를 최적화하는 알고리즘들을 포함합니다.

## 1. `nn.Module`: 모든 신경망의 기초

`nn.Module`은 PyTorch에서 모든 신경망 모듈(레이어)과 전체 모델의 기반이 되는 추상 클래스입니다. 사용자 정의 레이어나 전체 신경망 모델을 만들려면 반드시 `nn.Module`을 상속받아야 합니다. `nn.Module`을 상속함으로써 PyTorch의 강력한 기능들을 활용할 수 있게 됩니다.

### 1.1. `nn.Module`의 주요 기능

*   **파라미터 추적**: `nn.Module` 내부에 정의된 `nn.Parameter` 객체(예: `nn.Linear` 레이어의 가중치와 편향)들을 자동으로 추적하고 관리합니다. 이는 `optimizer`가 모델의 학습 가능한 파라미터들을 쉽게 찾고 업데이트할 수 있도록 합니다.
*   **구조 정의 및 관리**: 모델의 계층적인 구조를 정의하고, 서브 모듈(다른 `nn.Module` 인스턴스)들을 포함할 수 있도록 합니다. 이를 통해 복잡한 신경망도 모듈화하여 관리할 수 있습니다.
*   **모드 전환**: 모델의 학습 모드(`model.train()`)와 평가 모드(`model.eval()`)를 전환할 수 있습니다. 이는 `Dropout`이나 `BatchNorm`과 같이 학습 시와 평가 시 다르게 동작하는 레이어들의 동작을 제어하는 데 필수적입니다.
*   **장치 이동**: 모델의 모든 파라미터와 버퍼를 CPU나 GPU로 쉽게 이동시킬 수 있습니다 (`model.to(device)`).
*   **상태 저장 및 로드**: 모델의 현재 상태(파라미터 값)를 저장하고 로드하는 기능을 제공합니다.

### 1.2. `nn.Module` 사용법

`nn.Module`을 상속받아 사용자 정의 모델을 만드는 일반적인 절차는 다음과 같습니다.

1.  `nn.Module`을 상속하는 클래스를 정의합니다.
2.  `__init__` 메서드에서 `super().__init__()`를 호출하여 부모 클래스의 생성자를 초기화합니다. 이 안에서 모델에 필요한 레이어(예: `nn.Linear`, `nn.Conv2d`, `nn.ReLU` 등)들을 정의하고 클래스의 속성으로 할당합니다. `nn.Module`은 클래스 속성으로 할당된 다른 `nn.Module` 인스턴스나 `nn.Parameter`들을 자동으로 인식하고 관리합니다.
3.  `forward` 메서드에서 입력 데이터를 받아 모델의 순전파(forward pass) 로직을 구현합니다. 입력 데이터가 각 레이어를 통과하는 과정을 정의하며, 이 메서드의 반환값이 모델의 최종 출력이 됩니다.

```python
import torch
from torch import nn

# SimpleModel 클래스는 nn.Module을 상속받아 신경망을 정의합니다.
class SimpleModel(nn.Module):
    def __init__(self):
        # 부모 클래스인 nn.Module의 생성자를 호출합니다.
        super(SimpleModel, self).__init__()
        
        # 신경망에 필요한 레이어들을 정의하고 클래스의 속성으로 할당합니다.
        # nn.Linear는 선형 변환(y = Wx + b)을 수행하는 레이어입니다.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # 입력 차원: 28*28 (예: MNIST 이미지), 출력 차원: 512
            nn.ReLU(),             # ReLU 활성화 함수
            nn.Linear(512, 512),   # 입력 차원: 512, 출력 차원: 512
            nn.ReLU(),
            nn.Linear(512, 10)     # 입력 차원: 512, 출력 차원: 10 (예: 10개 클래스 분류)
        )

    # forward 메서드는 모델의 순전파 로직을 정의합니다.
    # 입력 x가 각 레이어를 통과하는 과정을 구현합니다.
    def forward(self, x):
        # 입력 x의 형태를 (batch_size, 28*28)로 평탄화합니다.
        x = x.view(x.size(0), -1) 
        logits = self.linear_relu_stack(x)
        return logits

# 모델 인스턴스 생성
model = SimpleModel()
print(model)

# 모델의 파라미터 확인
# named_parameters()는 모델의 모든 학습 가능한 파라미터(nn.Parameter)를 이름과 함께 반환합니다.
print("\nModel Parameters:")
for name, param in model.named_parameters():
    if param.requires_grad: # requires_grad=True인 파라미터만 출력 (학습 대상)
        print(f"- {name}: {param.data.shape}")

# 가상의 입력 데이터로 모델 테스트
# 배치 크기 64, 이미지 크기 1x28x28 (흑백 이미지)
input_tensor = torch.randn(64, 1, 28, 28)
output = model(input_tensor)
print(f"\nOutput shape: {output.shape}") # 출력 형태: (64, 10)
```

### 1.3. `nn.Sequential`: 간편한 모델 구축

`nn.Sequential`은 여러 모듈(레이어)을 순서대로 연결하여 모델을 구성할 때 매우 유용합니다. `forward` 메서드를 직접 구현할 필요 없이, 모듈들을 리스트처럼 전달하면 입력이 순차적으로 각 모듈을 통과하게 됩니다. 이는 간단하고 순차적인 신경망을 정의할 때 코드를 간결하게 만들어 줍니다.

```python
import torch
from torch import nn

# nn.Sequential을 사용하여 간단한 신경망 정의
# 입력 -> 선형 변환 -> ReLU -> 선형 변환
sequential_model = nn.Sequential(
    nn.Linear(10, 20), # 입력 차원 10, 출력 차원 20
    nn.ReLU(),         # 활성화 함수
    nn.Linear(20, 5)   # 입력 차원 20, 출력 차원 5
)

print(sequential_model)

# 가상의 입력 데이터로 테스트
input_data = torch.randn(1, 10) # 배치 크기 1, 입력 차원 10
output = sequential_model(input_data)
print(f"Output shape from sequential model: {output.shape}") # 출력 형태: (1, 5)
```

## 2. `nn.Parameter`: 학습 가능한 Tensor

`nn.Parameter`는 `torch.Tensor`를 상속하는 특별한 클래스입니다. `nn.Module`의 속성으로 할당될 때, `nn.Parameter`는 자동으로 해당 모듈의 학습 가능한 파라미터로 등록됩니다. 이는 `optimizer`가 이 파라미터들을 찾아 기울기를 계산하고 업데이트할 수 있도록 합니다.

*   **`requires_grad=True`**: `nn.Parameter`는 기본적으로 `requires_grad=True`로 설정되어 있어, 항상 기울기 계산 대상이 됩니다.
*   **자동 등록**: `nn.Module`은 `nn.Parameter` 타입의 속성을 발견하면 이를 자동으로 `_parameters` 딕셔너리에 등록합니다.

일반적으로 `nn.Linear`, `nn.Conv2d`와 같은 PyTorch 내장 레이어를 사용하면 내부적으로 가중치(weight)와 편향(bias)이 `nn.Parameter`로 자동으로 생성되므로, 개발자가 직접 `nn.Parameter`를 정의할 일은 많지 않습니다. 하지만 사용자 정의 레이어를 만들거나, 특정 Tensor를 모델의 학습 가능한 파라미터로 포함시키고 싶을 때는 `nn.Parameter`로 명시적으로 감싸주어야 합니다.

```python
import torch
from torch import nn

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 가중치(weight)와 편향(bias)을 nn.Parameter로 직접 정의
        # nn.Parameter로 감싸주면 이 Tensor들이 모델의 학습 가능한 파라미터로 등록됩니다.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # 수동으로 선형 변환 구현: y = xW^T + b
        return torch.matmul(x, self.weight.T) + self.bias

custom_layer = CustomLinear(in_features=5, out_features=3)
print(custom_layer)

# custom_layer의 파라미터 확인
print("\nCustom Layer Parameters:")
for name, param in custom_layer.named_parameters():
    print(f"- {name}: {param.data.shape}, requires_grad: {param.requires_grad}")
```

## 3. 손실 함수 (Loss Function)

손실 함수(또는 비용 함수, Cost Function)는 모델의 예측값과 실제 정답(target) 사이의 오차(error)를 측정하는 함수입니다. 모델은 이 손실 값을 최소화하는 방향으로 학습을 진행합니다. 손실 함수는 미분 가능해야 하며, `torch.nn` 패키지에서 다양한 손실 함수를 제공합니다.

### 3.1. `nn.MSELoss`: 평균 제곱 오차

*   **용도**: 회귀(regression) 문제에 주로 사용됩니다.
*   **설명**: 예측값과 실제값 사이의 차이를 제곱하여 평균한 값입니다. 오차가 클수록 손실 값이 크게 증가하는 특징이 있습니다.
*   **수식**: $L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$

```python
import torch
from torch import nn

loss_fn_mse = nn.MSELoss()

# 가상의 예측값과 실제값 (회귀 문제)
# predictions_reg: 5개의 샘플, 1차원 출력
predictions_reg = torch.randn(5, 1, requires_grad=True)
targets_reg = torch.randn(5, 1)

loss_mse = loss_fn_mse(predictions_reg, targets_reg)
print(f"MSE Loss: {loss_mse.item()}")
```

### 3.2. `nn.CrossEntropyLoss`: 교차 엔트로피 손실

*   **용도**: 다중 클래스 분류(multi-class classification) 문제에 주로 사용됩니다.
*   **설명**: 모델의 예측(logits)과 실제 클래스 레이블 간의 차이를 측정합니다. 내부적으로 `nn.LogSoftmax`와 `nn.NLLLoss`가 결합되어 있어, 모델의 마지막 레이어에 별도로 `Softmax` 함수를 적용할 필요가 없습니다. 모델의 출력은 각 클래스에 대한 로짓(logit, 정규화되지 않은 예측 점수)이어야 합니다.
*   **입력 형태**: `input`은 `(N, C)` 형태의 로짓(N: 배치 크기, C: 클래스 수), `target`은 `(N)` 형태의 정수형 클래스 인덱스(0부터 C-1까지)여야 합니다.

```python
import torch
from torch import nn

loss_fn_ce = nn.CrossEntropyLoss()

# 가상의 예측값 (로짓)과 실제 정답 (다중 클래스 분류)
# input_tensor: 배치 크기 4, 클래스 수 10에 대한 로짓
input_tensor_ce = torch.randn(4, 10, requires_grad=True)
# target: 배치 크기 4, 각 샘플의 실제 클래스 인덱스 (0-9)
target_ce = torch.randint(10, (4,))

loss_ce = loss_fn_ce(input_tensor_ce, target_ce)
print(f"CrossEntropy Loss: {loss_ce.item()}")

# 손실 함수를 이용한 역전파 (기울기 계산)
loss_ce.backward()
print(f"Gradient of input_tensor_ce after CE loss backward:\n {input_tensor_ce.grad}")
```

### 3.3. `nn.BCELoss` 및 `nn.BCEWithLogitsLoss`: 이진 교차 엔트로피 손실

*   **용도**: 이진 분류(binary classification) 문제에 사용됩니다.
*   **`nn.BCELoss`**: 모델의 출력이 0과 1 사이의 확률 값이어야 합니다 (일반적으로 `Sigmoid` 활성화 함수를 통과한 후).
*   **`nn.BCEWithLogitsLoss`**: `Sigmoid` 활성화 함수와 `BCELoss`를 결합한 버전입니다. 모델의 출력이 로짓(정규화되지 않은 예측 점수)이어야 하며, 내부적으로 `Sigmoid`를 적용하여 수치적으로 안정적입니다. 일반적으로 `nn.BCELoss`보다 이 함수를 사용하는 것이 권장됩니다.

```python
import torch
from torch import nn

loss_fn_bce = nn.BCELoss()
loss_fn_bce_logits = nn.BCEWithLogitsLoss()

# 가상의 예측값과 실제값 (이진 분류)
# predictions_binary: 배치 크기 5, 1차원 출력
predictions_binary = torch.tensor([0.1, 0.9, 0.4, 0.6, 0.2], requires_grad=True) # Sigmoid 통과 후 확률값
targets_binary = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])

loss_bce = loss_fn_bce(predictions_binary, targets_binary)
print(f"BCELoss: {loss_bce.item()}")

# BCEWithLogitsLoss는 로짓(정규화되지 않은 값)을 입력으로 받습니다.
predictions_logits = torch.tensor([-2.0, 3.0, -0.5, 1.0, -1.5], requires_grad=True) # 로짓 값
loss_bce_logits = loss_fn_bce_logits(predictions_logits, targets_binary)
print(f"BCEWithLogitsLoss: {loss_bce_logits.item()}")
```

## 4. 옵티마이저 (Optimizer)

옵티마이저는 손실 함수를 통해 계산된 기울기(gradient)를 사용하여 모델의 파라미터를 업데이트하는 알고리즘입니다. 딥러닝 모델 학습의 핵심적인 부분으로, 손실 함수를 최소화하여 모델의 성능을 향상시키는 역할을 합니다. PyTorch는 `torch.optim` 패키지에 다양한 최적화 알고리즘을 구현하여 제공합니다.

### 4.1. `torch.optim` 패키지

`torch.optim`은 다양한 최적화 알고리즘을 포함하고 있습니다. 옵티마이저를 사용하려면, 먼저 최적화할 파라미터(일반적으로 `model.parameters()`)와 학습률(`lr`)을 전달하여 옵티마이저 인스턴스를 생성해야 합니다.

```python
from torch import optim

# 모델 인스턴스 (이전 SimpleModel 사용)
model = SimpleModel()

# SGD 옵티마이저 생성: 모델의 모든 학습 가능한 파라미터와 학습률 0.001을 전달
optimizer_sgd = optim.SGD(model.parameters(), lr=1e-3)

# Adam 옵티마이저 생성: Adam은 기본 학습률이 0.001로 설정되어 있습니다.
optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)

print("SGD Optimizer created.")
print("Adam Optimizer created.")
```

### 4.2. 주요 옵티마이저

*   **`optim.SGD` (Stochastic Gradient Descent)**:
    *   가장 기본적인 경사 하강법 알고리즘입니다. 각 학습 스텝에서 하나의 샘플 또는 미니 배치(mini-batch)에 대한 기울기를 계산하여 파라미터를 업데이트합니다.
    *   `momentum` (모멘텀): 이전 기울기의 방향을 일정 비율로 반영하여 최적화 과정을 가속화하고 지역 최솟값(local minima)에 갇히는 것을 방지합니다.
    *   `weight_decay` (가중치 감쇠): L2 정규화(regularization)를 적용하여 과적합(overfitting)을 방지합니다.

*   **`optim.Adam` (Adaptive Moment Estimation)**:
    *   현재 가장 널리 사용되는 최적화 알고리즘 중 하나입니다. 각 파라미터마다 적응적인 학습률(adaptive learning rate)을 적용하여 빠르고 안정적인 수렴을 돕습니다.
    *   모멘텀과 RMSprop의 장점을 결합한 형태로, 대부분의 딥러닝 문제에서 좋은 성능을 보입니다.

*   **`optim.RMSprop` (Root Mean Square Propagation)**:
    *   Adam과 유사하게 적응적 학습률을 사용합니다. 과거 기울기의 제곱 평균을 사용하여 학습률을 조정합니다.

*   **`optim.Adagrad`, `optim.Adadelta`**: 이들도 적응적 학습률을 사용하는 옵티마이저입니다. 특정 문제에 따라 Adam보다 좋은 성능을 보일 수도 있습니다.

### 4.3. 옵티마이저 사용 흐름

딥러닝 모델 학습 루프 내에서 옵티마이저는 다음과 같은 순서로 사용됩니다.

1.  **`optimizer.zero_grad()`**: 이전 학습 반복에서 계산된 기울기를 모두 0으로 초기화합니다. PyTorch는 기본적으로 기울기를 누적(accumulate)하기 때문에, 매번 새로운 기울기를 계산하기 전에 반드시 초기화해야 합니다. 그렇지 않으면 이전 기울기와 현재 기울기가 합쳐져 잘못된 업데이트가 발생합니다.
2.  **`loss.backward()`**: 손실 함수에 대해 역전파를 수행하여 모델의 각 학습 가능한 파라미터에 대한 기울기를 계산합니다. 이 기울기들은 각 파라미터의 `.grad` 속성에 저장됩니다.
3.  **`optimizer.step()`**: 계산된 기울기(`param.grad`)를 사용하여 옵티마이저가 관리하는 모든 파라미터를 업데이트합니다. 이 단계에서 선택된 최적화 알고리즘(예: SGD, Adam)에 따라 파라미터 업데이트 규칙이 적용됩니다.

```python
# 모델, 손실 함수, 옵티마이저 정의 (이전 코드에서 이어짐)
model = SimpleModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# 학습 루프 (가상 데이터 사용)
print("\nStarting training loop (virtual data):")
for epoch in range(5):
    # 1. 기울기 초기화
    optimizer.zero_grad()

    # 2. 모델 순전파 및 손실 계산 (가상 입력 데이터와 랜덤 타겟)
    input_data = torch.randn(64, 1, 28, 28) # 배치 크기 64, 1채널 28x28 이미지
    predictions = model(input_data) # 모델의 forward 메서드 호출
    target = torch.randint(10, (64,)) # 0-9 사이의 랜덤 클래스 레이블
    loss = loss_fn(predictions, target)

    # 3. 역전파: 손실에 대한 각 파라미터의 기울기 계산
    loss.backward()

    # 4. 파라미터 업데이트: 계산된 기울기를 사용하여 모델 파라미터 업데이트
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
print("Training loop finished.")
```

### 4.4. 학습률 스케줄러 (Learning Rate Scheduler)

학습률 스케줄러는 학습 과정 중에 학습률(`lr`)을 동적으로 조정하는 메커니즘입니다. 학습 초기에는 큰 학습률로 빠르게 수렴을 유도하고, 학습 후반에는 작은 학습률로 미세 조정을 통해 안정적인 최적화를 돕습니다. `torch.optim.lr_scheduler` 패키지에 다양한 스케줄러가 구현되어 있습니다.

```python
from torch.optim import lr_scheduler

# 모델과 옵티마이저 정의 (이전 코드에서 이어짐)
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# StepLR 스케줄러 생성: 30 에포크마다 학습률을 0.1배로 감소
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 학습 루프 내에서 스케줄러 사용
# for epoch in range(num_epochs):
#     train(...)
#     validate(...)
#     scheduler.step() # 각 에포크 끝에서 스케줄러 업데이트
print("Learning Rate Scheduler created.")
```

## 5. 메트릭 (Metrics)

### 5.1. 메트릭의 중요성

메트릭은 모델의 성능을 객관적으로 평가하는 지표입니다. 손실 함수는 모델 학습을 위해 미분 가능해야 하며, 주로 최적화 과정에서 사용됩니다. 반면, 메트릭은 정확도(accuracy), 정밀도(precision), 재현율(recall), F1-score 등과 같이 모델의 성능을 사람이 직관적으로 이해하고 비교할 수 있는 척도입니다. 손실 값이 낮다고 해서 항상 모델의 실제 성능이 좋은 것은 아니므로, 메트릭을 함께 사용하여 모델을 종합적으로 평가해야 합니다.

### 5.2. 일반적인 메트릭

*   **분류(Classification) 문제**: 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-score, ROC-AUC 등
*   **회귀(Regression) 문제**: 평균 절대 오차(MAE, Mean Absolute Error), 평균 제곱근 오차(RMSE, Root Mean Squared Error), R-제곱(R-squared) 등

### 5.3. 메트릭 구현

PyTorch 자체에는 `torchmetrics`와 같은 공식적인 메트릭 라이브러리가 내장되어 있지 않습니다. 따라서 메트릭은 직접 구현하거나, `scikit-learn`과 같은 외부 라이브러리를 사용하거나, PyTorch 생태계에서 제공하는 `torchmetrics` 라이브러리를 활용하여 계산할 수 있습니다.

다음은 분류 문제에서 정확도(Accuracy)를 계산하는 간단한 예시입니다.

```python
import torch

# 가상의 모델 예측 (로짓)과 실제 레이블
predictions_logits = torch.randn(10, 5) # 배치 크기 10, 5개 클래스에 대한 로짓
true_labels = torch.randint(0, 5, (10,)) # 0-4 사이의 실제 클래스 레이블

# 로짓을 확률로 변환 (Softmax) 후 가장 높은 확률을 가진 클래스 선택
# torch.argmax는 가장 큰 값의 인덱스를 반환합니다.
predicted_classes = torch.argmax(predictions_logits, dim=1)

# 정확도 계산
# 예측된 클래스와 실제 레이블이 일치하는 경우를 세어 전체 샘플 수로 나눕니다.
correct_predictions = (predicted_classes == true_labels).sum().item()
total_samples = true_labels.size(0)
accuracy = correct_predictions / total_samples

print(f"Predicted Classes: {predicted_classes}")
print(f"True Labels: {true_labels}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Total Samples: {total_samples}")
print(f"Accuracy: {accuracy:.4f}")

# torchmetrics 라이브러리 사용 예시 (설치 필요: pip install torchmetrics)
# from torchmetrics import Accuracy
# acc_metric = Accuracy(task="multiclass", num_classes=5)
# accuracy_torchmetrics = acc_metric(predictions_logits, true_labels)
# print(f"Accuracy (torchmetrics): {accuracy_torchmetrics.item():.4f}")
```

이제 모델을 구성하는 주요 요소들을 모두 배웠습니다. 다음 장에서는 이러한 요소들을 조합하여 PyTorch를 이용한 **딥러닝 모델 학습 과정**을 전체적으로 살펴보겠습니다.

