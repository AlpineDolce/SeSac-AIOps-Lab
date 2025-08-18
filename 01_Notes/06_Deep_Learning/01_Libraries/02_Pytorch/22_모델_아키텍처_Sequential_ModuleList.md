<h2>PyTorch 모델 아키텍처: `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-18

<h2>문서 목표</h2>
이 문서는 PyTorch에서 복잡한 신경망 아키텍처를 효과적으로 구축하고 조직하는 데 사용되는 핵심 도구인 `nn.Sequential`, `nn.ModuleList`, 그리고 `nn.ModuleDict`에 대해 심층적으로 다룹니다. 각 도구의 개념, 사용법, 장단점을 상세히 설명하고, 실제 코드 예시를 통해 이해를 돕습니다. 또한, 각 도구가 어떤 상황에서 가장 적합하게 사용될 수 있는지 비교 분석하여, PyTorch를 활용한 유연하고 효율적인 모델 설계 역량을 강화하는 데 기여하고자 합니다.

<h2>목차</h2>

- [1. 모델 아키텍처 설계의 중요성](#1-모델-아키텍처-설계의-중요성)
- [2. `nn.Sequential`: 순차적인 모델 구축](#2-nnsequential-순차적인-모델-구축)
  - [2.1. `nn.Sequential`의 개념](#21-nnsequential의-개념)
  - [2.2. `nn.Sequential` 사용법](#22-nnsequential-사용법)
  - [2.3. `nn.Sequential`의 장점과 한계](#23-nnsequential의-장점과-한계)
- [3. `nn.ModuleList`: 모듈 리스트 관리](#3-nnmodulelist-모듈-리스트-관리)
  - [3.1. `nn.ModuleList`의 개념](#31-nnmodulelist의-개념)
  - [3.2. `nn.ModuleList` 사용법](#32-nnmodulelist-사용법)
  - [3.3. `nn.ModuleList`의 장점과 Python 리스트와의 차이점](#33-nnmodulelist의-장점과-python-리스트와의-차이점)
- [4. `nn.ModuleDict`: 모듈 딕셔너리 관리](#4-nnmoduledict-모듈-딕셔너리-관리)
  - [4.1. `nn.ModuleDict`의 개념](#41-nnmoduledict의-개념)
  - [4.2. `nn.ModuleDict` 사용법](#42-nnmoduledict-사용법)
  - [4.3. `nn.ModuleDict`의 장점과 Python 딕셔너리와의 차이점](#43-nnmoduledict의-장점과-python-딕셔너리와의-차이점)
- [5. `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict` 비교](#5-nnsequential-nnmodulelist-nnmoduledict-비교)
  - [5.1. 각 도구의 적합한 사용 사례](#51-각-도구의-적합한-사용-사례)
  - [5.2. 주요 차이점 요약](#52-주요-차이점-요약)
- [6. 결론](#6-결론)

--- 

# PyTorch 모델 아키텍처: `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`

## 1. 모델 아키텍처 설계의 중요성

딥러닝 모델을 설계할 때, 단순히 레이어들을 쌓는 것을 넘어 모델의 구조를 명확하고 효율적으로 조직하는 것이 중요합니다. 잘 설계된 아키텍처는 코드의 가독성을 높이고, 디버깅을 용이하게 하며, 복잡한 모델을 유연하게 구성할 수 있도록 돕습니다. PyTorch는 `nn.Module`을 기반으로 다양한 모델 구성 도구를 제공하며, 그 중 `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`는 모델의 레이어들을 효과적으로 관리하고 연결하는 데 핵심적인 역할을 합니다.

## 2. `nn.Sequential`: 순차적인 모델 구축

### 2.1. `nn.Sequential`의 개념

`nn.Sequential`은 여러 `nn.Module`들을 순서대로 연결하여 하나의 큰 모듈처럼 동작하게 만드는 컨테이너입니다. 입력 데이터가 `nn.Sequential`에 전달되면, 내부의 모듈들을 정의된 순서대로 차례로 통과하게 됩니다. 이는 데이터의 흐름이 명확하고 순차적인 모델(예: 다층 퍼셉트론, 간단한 컨볼루션 신경망)을 구축할 때 매우 유용합니다.

### 2.2. `nn.Sequential` 사용법

`nn.Sequential`은 `nn.Module` 인스턴스들을 인자로 직접 전달하거나, `OrderedDict` 형태로 전달하여 각 모듈에 이름을 부여할 수 있습니다.

```python
import torch
from torch import nn

# 1. 모듈들을 인자로 직접 전달하는 방법
model_sequential_1 = nn.Sequential(
    nn.Linear(784, 256), # 입력 784, 출력 256
    nn.ReLU(),           # ReLU 활성화 함수
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)   # 최종 출력 10 (예: 10개 클래스 분류)
)
print("\n--- nn.Sequential (direct arguments) ---")
print(model_sequential_1)

# 2. OrderedDict를 사용하여 모듈에 이름을 부여하는 방법
from collections import OrderedDict
model_sequential_2 = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 128)),
    ('relu2', nn.ReLU()),
    ('output', nn.Linear(128, 10))
]))
print("\n--- nn.Sequential (OrderedDict with names) ---")
print(model_sequential_2)

# 모델 사용 예시
input_data = torch.randn(64, 784) # 배치 크기 64, 입력 차원 784
output = model_sequential_1(input_data)
print(f"\nOutput shape from sequential model: {output.shape}")

# 이름으로 특정 레이어에 접근
print(f"\nAccessing layer by name: {model_sequential_2.fc1}")
```

### 2.3. `nn.Sequential`의 장점과 한계

*   **장점**: 코드가 매우 간결하고 직관적입니다. 순차적인 데이터 흐름을 가진 모델을 빠르게 구축할 수 있습니다. `forward` 메서드를 직접 구현할 필요가 없습니다.
*   **한계**: 데이터의 흐름이 순차적이지 않은 복잡한 아키텍처(예: 분기(branching), 스킵 연결(skip connections), 다중 입력/출력)에는 적합하지 않습니다. 동적으로 레이어를 추가하거나 제거하기 어렵습니다.

## 3. `nn.ModuleList`: 모듈 리스트 관리

### 3.1. `nn.ModuleList`의 개념

`nn.ModuleList`는 `nn.Module` 인스턴스들을 담는 파이썬 리스트와 유사한 컨테이너입니다. `nn.ModuleList`에 포함된 모든 `nn.Module`들은 자동으로 상위 `nn.Module`의 서브모듈로 등록됩니다. 이는 `model.parameters()`와 같은 메서드를 통해 내부 모듈들의 파라미터가 올바르게 인식되고 최적화될 수 있도록 합니다.

### 3.2. `nn.ModuleList` 사용법

`nn.ModuleList`는 주로 반복적인 구조를 가진 모델이나, 레이어의 개수가 동적으로 변하는 모델을 구축할 때 사용됩니다. 파이썬 리스트처럼 인덱싱과 반복문(`for` 루프)을 사용하여 각 모듈에 접근할 수 있습니다.

```python
import torch
from torch import nn

class MLPWithModuleList(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.ModuleList() # nn.ModuleList 초기화
        
        # 입력 레이어
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())

        # 히든 레이어들
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
        
        # 출력 레이어
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# hidden_dims를 변경하여 다양한 깊이의 MLP 생성 가능
model_ml_1 = MLPWithModuleList(input_dim=784, hidden_dims=[256, 128], output_dim=10)
print("\n--- MLPWithModuleList (2 hidden layers) ---")
print(model_ml_1)

model_ml_2 = MLPWithModuleList(input_dim=784, hidden_dims=[512, 256, 128], output_dim=10)
print("\n--- MLPWithModuleList (3 hidden layers) ---")
print(model_ml_2)

# 특정 레이어에 접근
print(f"\nAccessing first linear layer: {model_ml_1.layers[0]}")
```

### 3.3. `nn.ModuleList`의 장점과 Python 리스트와의 차이점

*   **장점**: 레이어의 개수를 동적으로 정의하거나, 반복문을 통해 레이어들을 적용해야 할 때 매우 유용합니다. `nn.ModuleList`에 포함된 모듈들은 `nn.Module`의 `parameters()` 메서드에 의해 자동으로 인식되어 학습 대상이 됩니다.
*   **Python 리스트와의 차이점**: 일반적인 파이썬 리스트(`list`)에 `nn.Module` 인스턴스를 담으면, 해당 모듈들은 상위 `nn.Module`의 서브모듈로 등록되지 않습니다. 따라서 `model.parameters()`를 호출해도 파이썬 리스트 내의 모듈 파라미터들은 인식되지 않아 학습이 불가능합니다. `nn.ModuleList`는 이러한 문제를 해결하여, 리스트 내의 모든 모듈 파라미터가 올바르게 등록되도록 합니다.

```python
import torch
from torch import nn

class ModelWithPythonList(nn.Module):
    def __init__(self):
        super().__init__()
        # 파이썬 리스트에 레이어 저장
        self.layers = [
            nn.Linear(10, 5),
            nn.ReLU()
        ]

class ModelWithModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.ModuleList에 레이어 저장
        self.layers = nn.ModuleList([
            nn.Linear(10, 5),
            nn.ReLU()
        ])

model_py_list = ModelWithPythonList()
model_ml_list = ModelWithModuleList()

print("\n--- Parameters of ModelWithPythonList ---")
# 파이썬 리스트 내의 레이어 파라미터는 인식되지 않음
for name, param in model_py_list.named_parameters():
    print(f"- {name}: {param.shape}") # 아무것도 출력되지 않음

print("\n--- Parameters of ModelWithModuleList ---")
# nn.ModuleList 내의 레이어 파라미터는 올바르게 인식됨
for name, param in model_ml_list.named_parameters():
    print(f"- {name}: {param.shape}")
```

## 4. `nn.ModuleDict`: 모듈 딕셔너리 관리

### 4.1. `nn.ModuleDict`의 개념

`nn.ModuleDict`는 `nn.Module` 인스턴스들을 키-값(key-value) 쌍으로 담는 파이썬 딕셔너리와 유사한 컨테이너입니다. 각 모듈에 고유한 문자열 키를 부여하여 접근할 수 있으며, `nn.ModuleDict`에 포함된 모든 `nn.Module`들은 자동으로 상위 `nn.Module`의 서브모듈로 등록됩니다.

### 4.2. `nn.ModuleDict` 사용법

`nn.ModuleDict`는 주로 모델 내에 여러 개의 독립적인 서브 모듈이 존재하고, 이들을 이름으로 구분하여 접근해야 할 때 유용합니다. 예를 들어, 여러 종류의 입력(텍스트, 이미지, 수치 데이터)을 처리하는 모델에서 각 입력 타입별로 다른 처리 모듈을 가질 때 사용할 수 있습니다.

```python
import torch
from torch import nn

class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({
            'text_encoder': nn.Linear(100, 50),
            'image_encoder': nn.Linear(200, 50),
            'numeric_encoder': nn.Linear(10, 50)
        })
        self.classifier = nn.Linear(50, 2)

    def forward(self, x_text, x_image, x_numeric):
        encoded_text = self.encoders['text_encoder'](x_text)
        encoded_image = self.encoders['image_encoder'](x_image)
        encoded_numeric = self.encoders['numeric_encoder'](x_numeric)
        
        # 인코딩된 특징들을 합치거나 다른 방식으로 처리
        combined_features = encoded_text + encoded_image + encoded_numeric
        output = self.classifier(combined_features)
        return output

model_md = MultiInputModel()
print("\n--- MultiInputModel with nn.ModuleDict ---")
print(model_md)

# 특정 인코더에 접근
print(f"\nAccessing image encoder: {model_md.encoders['image_encoder']}")

# 모델 파라미터 확인 (nn.ModuleDict 내의 파라미터도 올바르게 등록됨)
print("\nModel Parameters:")
for name, param in model_md.named_parameters():
    print(f"- {name}: {param.shape}")
```

### 4.3. `nn.ModuleDict`의 장점과 Python 딕셔너리와의 차이점

*   **장점**: 이름 기반으로 모듈을 관리하고 접근할 수 있어 코드의 가독성과 유지보수성을 높입니다. 특히 여러 개의 독립적인 서브 모듈이나 조건부로 활성화되는 모듈들을 관리할 때 유용합니다. `nn.ModuleDict`에 포함된 모듈들도 `nn.Module`의 `parameters()` 메서드에 의해 자동으로 인식되어 학습 대상이 됩니다.
*   **Python 딕셔너리와의 차이점**: `nn.ModuleList`와 마찬가지로, 일반적인 파이썬 딕셔너리(`dict`)에 `nn.Module` 인스턴스를 담으면 해당 모듈들은 상위 `nn.Module`의 서브모듈로 등록되지 않아 학습이 불가능합니다. `nn.ModuleDict`는 이 문제를 해결하여 딕셔너리 내의 모든 모듈 파라미터가 올바르게 등록되도록 합니다.

## 5. `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict` 비교

이 세 가지 컨테이너는 모두 `nn.Module` 인스턴스들을 관리하는 데 사용되지만, 각각의 목적과 적합한 사용 사례가 다릅니다.

### 5.1. 각 도구의 적합한 사용 사례

*   **`nn.Sequential`**: 
    *   **가장 적합**: 데이터가 순차적으로 여러 레이어를 통과하는 간단한 피드포워드 신경망(Feedforward Neural Network)이나 컨볼루션 신경망(Convolutional Neural Network)의 블록을 정의할 때.
    *   **예시**: `(Linear -> ReLU -> Linear)`, `(Conv2d -> BatchNorm -> ReLU -> MaxPool)`과 같은 기본적인 블록.

*   **`nn.ModuleList`**: 
    *   **가장 적합**: 레이어의 개수가 동적으로 변하거나, 반복문을 통해 여러 레이어를 적용해야 할 때. 또는 여러 레이어가 동일한 구조를 가지고 반복될 때.
    *   **예시**: Transformer의 인코더/디코더 블록처럼 동일한 구조의 레이어가 N번 반복되는 경우, ResNet의 잔차 블록(residual block)을 여러 개 쌓을 때.

*   **`nn.ModuleDict`**: 
    *   **가장 적합**: 모델 내에 여러 개의 독립적인 서브 모듈이 존재하고, 이들을 이름으로 구분하여 접근해야 할 때. 또는 조건부로 특정 모듈을 선택하여 사용해야 할 때.
    *   **예시**: 멀티모달(multi-modal) 입력(텍스트, 이미지, 오디오)을 처리하기 위해 각 모달리티별로 다른 인코더를 가질 때, 또는 A/B 테스트를 위해 여러 버전의 서브 모듈을 정의하고 선택적으로 사용할 때.

### 5.2. 주요 차이점 요약

| 특징           | `nn.Sequential`                               | `nn.ModuleList`                                   | `nn.ModuleDict`                                   |
| :------------- | :-------------------------------------------- | :------------------------------------------------ | :------------------------------------------------ |
| **목적**       | 순차적인 레이어 연결                          | 모듈 리스트 관리 (동적 개수, 반복)                | 이름 기반 모듈 관리 (분기, 선택)                  |
| **데이터 흐름**| 입력이 순서대로 모든 모듈을 통과             | `forward` 메서드에서 수동으로 각 모듈에 적용      | `forward` 메서드에서 키를 통해 특정 모듈에 적용   |
| **접근 방식**  | 인덱스 (`model[0]`) 또는 이름 (`model.fc1`)  | 인덱스 (`model.layers[0]`)                        | 키 (`model.encoders['text']`)                     |
| **유연성**     | 낮음 (순차적 고정)                            | 높음 (동적 레이어 개수, 반복문 활용)              | 높음 (이름 기반 선택, 조건부 사용)                |
| **파라미터 등록**| 자동                                          | 자동                                              | 자동                                              |
| **주요 사용처**| 간단한 MLP, CNN 블록                          | 반복적인 레이어 구조, 가변적인 깊이의 모델        | 멀티모달 모델, 이름 있는 서브 모듈, 조건부 로직   |

## 6. 결론

PyTorch에서 `nn.Sequential`, `nn.ModuleList`, `nn.ModuleDict`는 신경망 아키텍처를 설계하고 구현하는 데 있어 매우 강력하고 유연한 도구들입니다. 각 컨테이너의 특성과 적합한 사용 사례를 이해하고 적절히 활용함으로써, 코드의 가독성과 유지보수성을 높이고, 복잡하고 다양한 형태의 딥러닝 모델을 효율적으로 구축할 수 있습니다. 이들을 조합하여 사용하는 능력은 PyTorch 개발자로서의 역량을 한층 더 강화할 것입니다.
