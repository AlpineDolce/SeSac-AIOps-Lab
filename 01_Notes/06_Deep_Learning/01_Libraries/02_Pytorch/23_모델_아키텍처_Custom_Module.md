<h2>PyTorch 모델 아키텍처: 사용자 정의 모듈 (`Custom Module`)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-18

<h2>문서 목표</h2>
이 문서는 PyTorch에서 제공하는 기본 레이어만으로는 구현하기 어려운 복잡하거나 특수한 신경망 구조를 위해 사용자 정의 모듈(`Custom Module`)을 생성하는 방법을 심층적으로 다룹니다. `nn.Module` 클래스를 상속받아 자신만의 레이어나 전체 모델을 구축하는 과정, 학습 가능한 파라미터를 `nn.Parameter`로 정의하는 방법, 그리고 `forward` 메서드를 통해 순전파 로직을 구현하는 과정을 상세한 코드 예시와 함께 설명합니다. 이를 통해 PyTorch의 유연성을 최대한 활용하여 독창적인 딥러닝 모델을 설계하고 구현하는 역량을 강화하는 데 기여하고자 합니다.

<h2>목차</h2>

- [1. 사용자 정의 모듈의 필요성](#1-사용자-정의-모듈의-필요성)
- [2. `nn.Module` 상속의 기본](#2-nnmodule-상속의-기본)
  - [2.1. `__init__` 메서드: 모듈 구성 요소 정의](#21-__init__-메서드-모듈-구성-요소-정의)
  - [2.2. `forward` 메서드: 순전파 로직 구현](#22-forward-메서드-순전파-로직-구현)
- [3. 학습 가능한 파라미터 정의 (`nn.Parameter`)](#3-학습-가능한-파라미터-정의-nnparameter)
- [4. 사용자 정의 레이어 구현 예시](#4-사용자-정의-레이어-구현-예시)
  - [4.1. 예시 1: 사용자 정의 선형 레이어 (`CustomLinear`)](#41-예시-1-사용자-정의-선형-레이어-customlinear)
  - [4.2. 예시 2: 사용자 정의 활성화 함수 (`Swish`)](#42-예시-2-사용자-정의-활성화-함수-swish)
  - [4.3. 예시 3: 여러 서브 모듈을 포함하는 사용자 정의 레이어 (`ResidualBlock`)](#43-예시-3-여러-서브-모듈을-포함하는-사용자-정의-레이어-residualblock)
- [5. 사용자 정의 모델 구현 예시](#5-사용자-정의-모델-구현-예시)
- [6. `nn.Module`의 추가 기능 활용](#6-nnmodule의-추가-기능-활용)
  - [6.1. 학습/평가 모드 전환 (`train()`, `eval()`)](#61-학습평가-모드-전환-train-eval)
  - [6.2. 장치 이동 (`to(device)`)](#62-장치-이동-todevice)
  - [6.3. 모델 상태 저장 및 로드 (`state_dict()`, `load_state_dict()`)](#63-모델-상태-저장-및-로드-state_dict-load_state_dict)
- [7. 결론](#7-결론)

--- 

# PyTorch 모델 아키텍처: 사용자 정의 모듈 (`Custom Module`)

## 1. 사용자 정의 모듈의 필요성

PyTorch는 `nn.Linear`, `nn.Conv2d`, `nn.ReLU` 등 다양한 표준 신경망 레이어와 활성화 함수를 `torch.nn` 패키지를 통해 제공합니다. 하지만 때로는 이러한 내장 모듈만으로는 구현하기 어려운 특수한 연산, 복잡한 데이터 흐름, 또는 독창적인 신경망 구조가 필요할 수 있습니다. 이러한 경우, `nn.Module` 클래스를 상속받아 자신만의 사용자 정의 모듈(`Custom Module`)을 생성함으로써 PyTorch의 유연성을 최대한 활용할 수 있습니다.

사용자 정의 모듈은 다음과 같은 상황에서 유용합니다:
*   **비표준 연산**: PyTorch에 내장되지 않은 새로운 종류의 레이어나 연산을 구현할 때.
*   **복잡한 데이터 흐름**: 분기(branching), 스킵 연결(skip connections), 다중 입력/출력 등 `nn.Sequential`로는 표현하기 어려운 복잡한 데이터 흐름을 가질 때.
*   **재사용 가능한 블록**: 여러 모델에서 반복적으로 사용될 수 있는 특정 구조의 신경망 블록을 정의할 때.
*   **연구 및 실험**: 새로운 아이디어를 가진 신경망 구조를 실험하고 구현할 때.

## 2. `nn.Module` 상속의 기본

PyTorch에서 모든 신경망 모듈(레이어)과 전체 모델은 `nn.Module` 클래스를 상속받아야 합니다. 사용자 정의 모듈을 생성하는 과정은 크게 두 가지 핵심 메서드를 구현하는 것으로 요약됩니다.

### 2.1. `__init__` 메서드: 모듈 구성 요소 정의

`__init__` 메서드는 사용자 정의 모듈의 생성자입니다. 이 메서드에서는 모듈이 가질 구성 요소들(다른 `nn.Module` 인스턴스, 학습 가능한 파라미터 등)을 정의하고 초기화합니다.

*   **`super().__init__()` 호출**: `nn.Module`을 상속받는 모든 클래스는 생성자의 첫 줄에서 반드시 `super().__init__()`를 호출해야 합니다. 이는 부모 클래스인 `nn.Module`의 초기화 로직을 실행하여, 파라미터 등록 및 기타 내부 설정을 올바르게 수행하도록 합니다.
*   **서브 모듈 정의**: `nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d` 등 다른 `nn.Module` 인스턴스들을 클래스의 속성으로 할당합니다 (예: `self.linear = nn.Linear(...)`). 이렇게 할당된 모듈들은 자동으로 상위 모듈의 서브 모듈로 등록되어, `model.parameters()`를 통해 파라미터들이 올바르게 추적됩니다.
*   **`nn.Parameter` 정의**: 직접 학습 가능한 가중치나 편향을 정의해야 할 경우, `nn.Parameter`를 사용하여 Tensor를 감싸 클래스의 속성으로 할당합니다. 이에 대해서는 다음 섹션에서 더 자세히 다룹니다.

### 2.2. `forward` 메서드: 순전파 로직 구현

`forward` 메서드는 모듈의 순전파(forward pass) 로직을 정의하는 곳입니다. 이 메서드는 모듈에 입력 데이터가 주어졌을 때 어떤 연산을 수행하여 출력을 내보낼지를 결정합니다. PyTorch 모델을 호출할 때 (예: `model(input_data)`), 내부적으로 이 `forward` 메서드가 실행됩니다.

*   **입력과 출력**: `forward` 메서드는 일반적으로 하나 이상의 Tensor를 입력으로 받고, 하나 이상의 Tensor를 출력으로 반환합니다.
*   **연산 정의**: 입력 Tensor가 `__init__`에서 정의된 구성 요소들을 어떻게 통과하고 어떤 연산을 거쳐 최종 출력이 되는지를 구현합니다. PyTorch의 Tensor 연산과 `autograd` 시스템이 이 과정에서 자동으로 계산 그래프를 구축합니다.

## 3. 학습 가능한 파라미터 정의 (`nn.Parameter`)

`nn.Parameter`는 `torch.Tensor`를 상속하는 특별한 클래스입니다. `nn.Module`의 속성으로 `nn.Parameter` 타입의 객체가 할당되면, PyTorch는 이를 자동으로 해당 모듈의 학습 가능한 파라미터로 등록합니다. 이렇게 등록된 파라미터들은 `model.parameters()` 메서드를 통해 접근할 수 있으며, `optimizer`에 의해 기울기가 계산되고 업데이트됩니다.

*   **`requires_grad=True`**: `nn.Parameter`는 기본적으로 `requires_grad=True`로 설정되어 있어, 항상 기울기 계산 대상이 됩니다.
*   **자동 등록**: `nn.Module`은 `nn.Parameter` 타입의 속성을 발견하면 이를 자동으로 `_parameters` 딕셔너리에 등록합니다. 일반 `torch.Tensor`를 속성으로 할당하면 등록되지 않습니다.

사용자 정의 레이어를 만들 때, 가중치(weight)나 편향(bias)과 같이 학습을 통해 값이 변경되어야 하는 Tensor들은 반드시 `nn.Parameter`로 감싸주어야 합니다.

```python
import torch
from torch import nn

# nn.Parameter를 사용하지 않은 경우 (파라미터로 등록되지 않음)
class MyModuleWithoutParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_tensor = torch.randn(3, 3) # 일반 Tensor

# nn.Parameter를 사용한 경우 (파라미터로 등록됨)
class MyModuleWithParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_parameter = nn.Parameter(torch.randn(3, 3)) # nn.Parameter

model_no_param = MyModuleWithoutParameter()
model_with_param = MyModuleWithParameter()

print("\n--- Parameters of MyModuleWithoutParameter ---")
for name, param in model_no_param.named_parameters():
    print(f"- {name}: {param.shape}") # 아무것도 출력되지 않음

print("\n--- Parameters of MyModuleWithParameter ---")
for name, param in model_with_param.named_parameters():
    print(f"- {name}: {param.shape}") # my_parameter가 출력됨
```

## 4. 사용자 정의 레이어 구현 예시

### 4.1. 예시 1: 사용자 정의 선형 레이어 (`CustomLinear`)

PyTorch의 `nn.Linear`와 유사하게 동작하는 사용자 정의 선형 레이어를 구현하여 `nn.Parameter`와 `forward` 메서드의 사용법을 이해합니다.

```python
import torch
from torch import nn

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 가중치(weight)와 편향(bias)을 nn.Parameter로 직접 정의
        # 초기화는 일반적으로 무작위 값으로 수행합니다.
        self.weight = nn.Parameter(torch.randn(out_features, in_features)) # (출력 차원, 입력 차원)
        self.bias = nn.Parameter(torch.randn(out_features)) # (출력 차원)

    def forward(self, x):
        # 선형 변환 구현: y = xW^T + b
        # x: (batch_size, in_features)
        # weight.T: (in_features, out_features)
        # 결과: (batch_size, out_features)
        return torch.matmul(x, self.weight.T) + self.bias

# CustomLinear 레이어 인스턴스 생성
custom_linear_layer = CustomLinear(in_features=5, out_features=3)
print("\n--- CustomLinear Layer ---")
print(custom_linear_layer)

# 파라미터 확인
print("\nCustomLinear Parameters:")
for name, param in custom_linear_layer.named_parameters():
    print(f"- {name}: {param.shape}, requires_grad: {param.requires_grad}")

# 가상의 입력 데이터로 테스트
input_data = torch.randn(10, 5) # 배치 크기 10, 입력 차원 5
output = custom_linear_layer(input_data)
print(f"Output shape from CustomLinear: {output.shape}") # (10, 3)
```

### 4.2. 예시 2: 사용자 정의 활성화 함수 (`Swish`)

`Swish`는 `x * sigmoid(x)` 형태의 활성화 함수입니다. `nn.Module`을 상속하여 이를 구현할 수 있습니다. 이 경우 학습 가능한 파라미터는 없습니다.

```python
import torch
from torch import nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

swish_activation = Swish()
print("\n--- Swish Activation Function ---")
print(swish_activation)

# 가상의 입력 데이터로 테스트
input_data = torch.randn(5)
output = swish_activation(input_data)
print(f"Input: {input_data}")
print(f"Output from Swish: {output}")
```

### 4.3. 예시 3: 여러 서브 모듈을 포함하는 사용자 정의 레이어 (`ResidualBlock`)

ResNet에서 사용되는 잔차 블록(Residual Block)과 같이 여러 내장 레이어들을 조합하여 하나의 복합적인 레이어를 만들 수 있습니다. 이 경우 `__init__`에서 내장 레이어들을 정의하고 `forward`에서 이들을 연결합니다.

```python
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 서브 모듈 정의
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 스킵 연결을 위한 다운샘플링 레이어 (입력/출력 채널 또는 스트라이드가 다를 경우)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x # 스킵 연결을 위한 입력 저장

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity) # 스킵 연결 추가
        out = self.relu(out)
        return out

residual_block = ResidualBlock(in_channels=3, out_channels=64, stride=2)
print("\n--- ResidualBlock ---")
print(residual_block)

# 가상의 입력 데이터로 테스트
input_image = torch.randn(1, 3, 32, 32) # 배치 1, 3채널, 32x32 이미지
output = residual_block(input_image)
print(f"Output shape from ResidualBlock: {output.shape}") # (1, 64, 16, 16) (stride=2로 인해 크기 감소)
```

## 5. 사용자 정의 모델 구현 예시

사용자 정의 레이어와 내장 레이어들을 조합하여 전체 신경망 모델을 구축할 수 있습니다. 다음은 `CustomLinear`와 `Swish`를 사용하여 간단한 분류 모델을 만드는 예시입니다.

```python
import torch
from torch import nn

# 위에서 정의한 CustomLinear와 Swish 클래스를 재사용
# class CustomLinear(...)
# class Swish(...)

class CustomClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = CustomLinear(input_dim, hidden_dim) # 사용자 정의 선형 레이어
        self.activation1 = Swish() # 사용자 정의 활성화 함수
        self.layer2 = nn.Linear(hidden_dim, output_dim) # 내장 선형 레이어

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        return x

custom_classifier = CustomClassifier(input_dim=784, hidden_dim=128, output_dim=10)
print("\n--- CustomClassifier Model ---")
print(custom_classifier)

# 모델 파라미터 확인 (CustomLinear의 파라미터도 올바르게 등록됨)
print("\nCustomClassifier Parameters:")
for name, param in custom_classifier.named_parameters():
    print(f"- {name}: {param.shape}")

# 가상의 입력 데이터로 테스트
input_data = torch.randn(64, 784)
output = custom_classifier(input_data)
print(f"Output shape from CustomClassifier: {output.shape}")
```

## 6. `nn.Module`의 추가 기능 활용

`nn.Module`을 상속하여 사용자 정의 모듈을 만들면, PyTorch가 제공하는 다양한 편리한 기능들을 활용할 수 있습니다.

### 6.1. 학습/평가 모드 전환 (`train()`, `eval()`)

`model.train()`과 `model.eval()` 메서드는 모델의 학습 모드와 평가 모드를 전환합니다. 이는 `Dropout`이나 `BatchNorm`과 같이 학습 시와 평가 시 다르게 동작하는 레이어들의 동작을 제어하는 데 필수적입니다.

```python
model = CustomClassifier(784, 128, 10)

model.train() # 모델을 학습 모드로 설정
print(f"Model is in training mode: {model.training}")

model.eval() # 모델을 평가 모드로 설정
print(f"Model is in evaluation mode: {model.training}")
```

### 6.2. 장치 이동 (`to(device)`)

모델의 모든 파라미터와 버퍼를 CPU나 GPU로 쉽게 이동시킬 수 있습니다. 이는 모델을 GPU에서 학습시키거나 추론할 때 매우 중요합니다.

```python
model = CustomClassifier(784, 128, 10)

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device) # 모델을 GPU로 이동
    print(f"Model moved to: {next(model.parameters()).device}")
    # 입력 데이터도 동일한 장치로 이동시켜야 합니다.
    # input_data = input_data.to(device)
else:
    print("CUDA is not available. Model remains on CPU.")
```

### 6.3. 모델 상태 저장 및 로드 (`state_dict()`, `load_state_dict()`)

`state_dict()` 메서드는 모델의 모든 학습 가능한 파라미터(가중치, 편향 등)를 딕셔너리 형태로 반환합니다. 이를 통해 모델의 현재 상태를 저장하고, 나중에 `load_state_dict()`를 사용하여 저장된 상태를 다시 로드할 수 있습니다.

```python
model = CustomClassifier(784, 128, 10)

# 모델의 현재 상태 저장
torch.save(model.state_dict(), "custom_classifier_model.pth")
print("Model state saved to custom_classifier_model.pth")

# 새로운 모델 인스턴스 생성
new_model = CustomClassifier(784, 128, 10)
# 저장된 상태 로드
new_model.load_state_dict(torch.load("custom_classifier_model.pth"))
new_model.eval() # 평가 모드로 설정 (로드 후에는 보통 평가 모드로 전환)
print("Model state loaded successfully.")
```

## 7. 결론

PyTorch에서 `nn.Module`을 상속하여 사용자 정의 모듈을 생성하는 것은 딥러닝 모델을 설계하고 구현하는 데 있어 가장 강력하고 유연한 방법입니다. 이를 통해 내장 레이어만으로는 불가능한 복잡한 연산이나 독창적인 아키텍처를 자유롭게 구현할 수 있습니다. `__init__`에서 구성 요소를 정의하고 `forward`에서 데이터 흐름을 정의하는 기본 원칙을 이해하고, `nn.Parameter`를 적절히 활용한다면, PyTorch의 모든 기능을 활용하여 어떤 종류의 신경망이든 효과적으로 구축할 수 있을 것입니다.

