<h2>PyTorch 모델 저장 및 로드: `state_dict`와 체크포인트 활용</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-18

<h2>문서 목표</h2>
이 문서는 PyTorch에서 딥러닝 모델을 효과적으로 저장하고 로드하는 방법을 심층적으로 다룹니다. `state_dict`의 개념과 이를 활용하여 모델의 파라미터와 옵티마이저 상태를 저장하는 방법을 설명합니다. 또한, 학습 재개를 위한 체크포인트(Checkpoint) 저장 및 로드 방법, 전이 학습(Transfer Learning) 시 사전 학습된 모델을 로드하는 방법, 그리고 분산 학습(Distributed Training) 환경에서의 모델 저장/로드 시 고려사항을 상세한 코드 예시와 함께 제시하여, PyTorch 기반 딥러닝 프로젝트의 효율성과 안정성을 극대화하는 데 기여하고자 합니다.

<h2>목차</h2>

- [1. 모델 저장 및 로드의 중요성](#1-모델-저장-및-로드의-중요성)
- [2. `state_dict` 이해하기](#2-state_dict-이해하기)
  - [2.1. `state_dict`의 개념](#21-state_dict의-개념)
  - [2.2. `nn.Module.state_dict()`: 모델 파라미터 저장](#22-nnmodulestate_dict-모델-파라미터-저장)
  - [2.3. `optimizer.state_dict()`: 옵티마이저 상태 저장](#23-optimizerstate_dict-옵티마이저-상태-저장)
- [3. 모델 저장 (Saving Models)](#3-모델-저장-saving-models)
  - [3.1. `torch.save()`: 일반적인 저장 함수](#31-torchsave-일반적인-저장-함수)
  - [3.2. 모델 파라미터만 저장 (권장)](#32-모델-파라미터만-저장-권장)
  - [3.3. 전체 모델 저장 (비권장)](#33-전체-모델-저장-비권장)
  - [3.4. 체크포인트 저장: 학습 재개를 위한 모든 정보 저장](#34-체크포인트-저장-학습-재개를-위한-모든-정보-저장)
- [4. 모델 로드 (Loading Models)](#4-모델-로드-loading-models)
  - [4.1. `torch.load()`: 일반적인 로드 함수](#41-torchload-일반적인-로드-함수)
  - [4.2. 모델 파라미터만 로드 (권장)](#42-모델-파라미터만-로드-권장)
  - [4.3. 전체 모델 로드 (비권장)](#43-전체-모델-로드-비권장)
  - [4.4. 체크포인트 로드 및 학습 재개](#44-체크포인트-로드-및-학습-재개)
- [5. 전이 학습을 위한 모델 로드](#5-전이-학습을-위한-모델-로드)
  - [5.1. 사전 학습된 가중치 로드](#51-사전-학습된-가중치-로드)
  - [5.2. 불일치하는 키 처리 (`strict=False`)](#52-불일치하는-키-처리-strictfalse)
- [6. 분산 학습 모델 저장 및 로드](#6-분산-학습-모델-저장-및-로드)
  - [6.1. `nn.DataParallel` 모델 저장/로드](#61-nndataparallel-모델-저장로드)
  - [6.2. `nn.DistributedDataParallel` 모델 저장/로드](#62-nndistributeddataparallel-모델-저장로드)
- [7. 결론](#7-결론)

--- 

# PyTorch 모델 저장 및 로드: `state_dict`와 체크포인트 활용

## 1. 모델 저장 및 로드의 중요성

딥러닝 모델을 개발하고 활용하는 과정에서 모델의 상태를 저장하고 필요할 때 다시 로드하는 기능은 매우 중요합니다. 이는 다음과 같은 다양한 시나리오에서 필수적입니다.

*   **학습 재개**: 장시간 소요되는 모델 학습 중 예기치 않은 중단이 발생하거나, 학습을 일시 중지했다가 나중에 다시 시작해야 할 때.
*   **최적 모델 선택**: 학습 과정에서 검증 성능이 가장 좋았던 시점의 모델을 저장하여 최종 모델로 활용할 때.
*   **모델 배포**: 학습이 완료된 모델을 실제 서비스 환경(예: 웹 애플리케이션, 모바일 앱)에 배포하여 추론(inference)에 사용할 때.
*   **전이 학습 (Transfer Learning)**: 대규모 데이터셋으로 사전 학습된 모델의 가중치를 가져와 새로운 작업에 활용할 때.
*   **모델 공유**: 연구 결과나 학습된 모델을 다른 연구자나 개발자와 공유할 때.

PyTorch는 모델의 상태를 효율적으로 저장하고 로드할 수 있는 유연한 메커니즘을 제공합니다.

## 2. `state_dict` 이해하기

### 2.1. `state_dict`의 개념

`state_dict`는 PyTorch에서 `nn.Module` 객체(모델)나 `optimizer` 객체의 **학습 가능한 파라미터(learnable parameters)**와 **등록된 버퍼(registered buffers)**를 Python 딕셔너리 형태로 저장하는 객체입니다. 각 키(key)는 레이어의 이름이나 파라미터의 이름을 나타내고, 값(value)은 해당 파라미터의 Tensor를 나타냅니다.

*   **모델의 `state_dict`**: 모델의 모든 `nn.Parameter` (예: `weight`, `bias`)와 `register_buffer`로 등록된 버퍼(예: `BatchNorm` 레이어의 `running_mean`, `running_var`)를 포함합니다.
*   **옵티마이저의 `state_dict`**: 옵티마이저의 상태(예: Adam 옵티마이저의 모멘텀 버퍼, 학습률)와 옵티마이저가 관리하는 파라미터들의 참조를 포함합니다.

### 2.2. `nn.Module.state_dict()`: 모델 파라미터 저장

`nn.Module` 인스턴스에서 `.state_dict()` 메서드를 호출하면 모델의 현재 학습된 파라미터들을 딕셔너리 형태로 얻을 수 있습니다.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

model = SimpleModel()

# 모델의 state_dict 확인
print("\n--- Model state_dict ---")
for param_tensor in model.state_dict():
    print(f"{param_tensor}\t{model.state_dict()[param_tensor].size()}")
# 출력 예시:
# linear1.weight\ttorch.Size([20, 10])
# linear1.bias\ttorch.Size([20])
# linear2.weight\ttorch.Size([1, 20])
# linear2.bias\ttorch.Size([1])
```

### 2.3. `optimizer.state_dict()`: 옵티마이저 상태 저장

옵티마이저의 `.state_dict()` 메서드를 호출하면 옵티마이저의 현재 상태를 딕셔너리 형태로 얻을 수 있습니다. 이는 학습을 중단했다가 나중에 정확히 동일한 상태에서 재개할 때 필수적입니다.

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 옵티마이저의 state_dict 확인
print("\n--- Optimizer state_dict ---")
for var_name in optimizer.state_dict():
    print(f"{var_name}\t{optimizer.state_dict()[var_name]}")
# 출력 예시:
# state\t{}
# param_groups\t[{'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, ''params': [0, 1, 2, 3]}]
```

## 3. 모델 저장 (Saving Models)

PyTorch에서 모델을 저장하는 가장 일반적인 방법은 `torch.save()` 함수를 사용하는 것입니다.

### 3.1. `torch.save()`: 일반적인 저장 함수

`torch.save()`는 Python의 `pickle` 모듈을 사용하여 객체를 직렬화(serialize)하고 디스크에 저장합니다. `.pth` 또는 `.pt` 확장자를 사용하는 것이 일반적입니다.

### 3.2. 모델 파라미터만 저장 (권장)

가장 권장되는 방법은 모델의 `state_dict()`만 저장하는 것입니다. 이는 모델의 아키텍처와 독립적으로 파라미터만 저장하므로, 모델을 로드할 때 더 유연하게 사용할 수 있습니다.

```python
# 모델의 state_dict 저장
PATH = "model_weights.pth"
torch.save(model.state_dict(), PATH)
print(f"Model weights saved to {PATH}")
```

### 3.3. 전체 모델 저장 (비권장)

`torch.save(model, PATH)`를 사용하여 모델의 아키텍처와 파라미터 전체를 저장할 수도 있습니다. 하지만 이 방법은 권장되지 않습니다.

*   **단점**: 저장된 모델을 로드할 때, 모델 클래스 정의가 저장 시점과 동일한 위치에 있어야 합니다. 코드 구조가 변경되거나 다른 환경에서 로드할 때 문제가 발생할 수 있습니다.

### 3.4. 체크포인트 저장: 학습 재개를 위한 모든 정보 저장

학습을 중단했다가 나중에 재개해야 할 경우, 모델의 파라미터뿐만 아니라 옵티마이저의 상태, 현재 에포크, 손실 값, 학습률 스케줄러의 상태 등 학습에 필요한 모든 정보를 함께 저장해야 합니다. 이를 **체크포인트(Checkpoint)**라고 합니다.

```python
# 가상의 학습 상태
epoch = 10
loss = 0.05

CHECKPOINT_PATH = "checkpoint.pth"
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    # 'best_accuracy': best_accuracy, # 필요에 따라 추가 정보 저장
    # 'scheduler_state_dict': scheduler.state_dict(),
}, CHECKPOINT_PATH)
print(f"Checkpoint saved to {CHECKPOINT_PATH}")
```

## 4. 모델 로드 (Loading Models)

PyTorch에서 모델을 로드하는 가장 일반적인 방법은 `torch.load()` 함수를 사용하는 것입니다.

### 4.1. `torch.load()`: 일반적인 로드 함수

`torch.load()`는 `torch.save()`로 저장된 객체를 역직렬화(deserialize)하여 메모리로 로드합니다.

### 4.2. 모델 파라미터만 로드 (권장)

`state_dict()`만 저장된 모델을 로드하는 방법입니다. 이 방법은 가장 유연하고 권장됩니다.

1.  **모델 아키텍처 정의 및 인스턴스화**: 먼저 모델 클래스를 정의하고, 모델 인스턴스를 생성합니다. 이 모델의 아키텍처는 저장된 `state_dict`와 호환되어야 합니다.
2.  **`state_dict` 로드**: `torch.load()`로 저장된 `state_dict`를 로드한 후, `model.load_state_dict()` 메서드를 사용하여 모델에 파라미터를 적용합니다.
3.  **장치 매핑**: `torch.load(PATH, map_location=device)`를 사용하여 저장된 모델을 특정 장치(CPU 또는 GPU)로 로드할 수 있습니다. 이는 저장된 장치와 현재 장치가 다를 때 유용합니다.

```python
# 모델 아키텍처 정의 (저장 시 사용한 것과 동일)
loaded_model = SimpleModel()

# 모델 파라미터 로드
# map_location='cpu': GPU에서 학습된 모델을 CPU로 로드할 때
# map_location='cuda': CPU에서 학습된 모델을 GPU로 로드할 때
loaded_model.load_state_dict(torch.load(PATH, map_location=device))

loaded_model.eval() # 추론 시에는 모델을 평가 모드로 설정
print("Model weights loaded successfully.")
```

### 4.3. 전체 모델 로드 (비권장)

`model = torch.load(PATH)`를 사용하여 전체 모델을 로드할 수 있습니다. 하지만 이 방법은 저장 시점의 모델 클래스 정의가 현재 환경에 있어야 하므로 유연성이 떨어집니다.

### 4.4. 체크포인트 로드 및 학습 재개

체크포인트를 로드하여 학습을 중단했던 지점부터 재개할 수 있습니다.

1.  **체크포인트 로드**: `torch.load()`로 체크포인트 딕셔너리를 로드합니다.
2.  **모델 및 옵티마이저 인스턴스화**: 모델과 옵티마이저를 새로 생성합니다.
3.  **`state_dict` 로드**: 로드된 체크포인트에서 `model_state_dict`와 `optimizer_state_dict`를 각각 모델과 옵티마이저에 로드합니다.
4.  **학습 상태 복원**: 체크포인트에 저장된 에포크, 손실, 학습률 스케줄러 상태 등을 복원합니다.

```python
# 모델과 옵티마이저를 새로 인스턴스화
resumed_model = SimpleModel().to(device)
resumed_optimizer = optim.Adam(resumed_model.parameters(), lr=0.001)

# 체크포인트 로드
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

resumed_model.load_state_dict(checkpoint['model_state_dict'])
resumed_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
resumed_epoch = checkpoint['epoch']
resumed_loss = checkpoint['loss']

# 학습 재개 시 모델을 학습 모드로 설정
resumed_model.train()

print(f"Model resumed from epoch {resumed_epoch+1} with loss {resumed_loss:.4f}")
```

## 5. 전이 학습을 위한 모델 로드

전이 학습 시 사전 학습된 모델의 가중치를 로드하여 새로운 모델에 적용할 수 있습니다. 이때 사전 학습된 모델의 아키텍처와 현재 모델의 아키텍처가 완전히 일치하지 않을 수 있습니다.

### 5.1. 사전 학습된 가중치 로드

`model.load_state_dict()`는 기본적으로 `strict=True`로 설정되어 있어, `state_dict`의 모든 키가 현재 모델의 키와 정확히 일치해야 합니다. 일치하지 않으면 오류가 발생합니다.

### 5.2. 불일치하는 키 처리 (`strict=False`)

전이 학습 시에는 보통 사전 학습된 모델의 마지막 분류 레이어를 새로운 작업에 맞게 변경하므로, `state_dict`의 키가 일치하지 않는 경우가 많습니다. 이럴 때 `strict=False`를 사용하여 일치하지 않는 키를 무시하고 로드할 수 있습니다.

```python
# 사전 학습된 ResNet-18 모델 로드 (ImageNet 가중치)
import torchvision.models as models
pretrained_resnet = models.resnet18(pretrained=True)

# 새로운 작업에 맞게 마지막 fc 레이어를 변경한 모델
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False) # 가중치 없이 아키텍처만 로드
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

custom_model = CustomResNet(num_classes=2).to(device)

# 사전 학습된 state_dict에서 fc 레이어를 제외하고 로드
pretrained_dict = pretrained_resnet.state_dict()
model_dict = custom_model.state_dict()

# 일치하지 않는 키(fc 레이어)를 필터링
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}

# 현재 모델의 state_dict를 업데이트
model_dict.update(pretrained_dict)

# 업데이트된 state_dict를 모델에 로드 (strict=False를 사용하면 일치하지 않는 키 무시)
custom_model.load_state_dict(model_dict, strict=False)

print("Pre-trained weights loaded for transfer learning.")
```

## 6. 분산 학습 모델 저장 및 로드

분산 학습 환경에서 모델을 저장하고 로드할 때는 `nn.DataParallel`과 `nn.DistributedDataParallel`의 특성을 고려해야 합니다.

### 6.1. `nn.DataParallel` 모델 저장/로드

`nn.DataParallel`로 래핑된 모델은 `module` 속성을 통해 원본 모델에 접근할 수 있습니다. 따라서 `model.module.state_dict()`를 사용하여 원본 모델의 `state_dict`를 저장해야 합니다. 로드 시에는 일반 모델처럼 로드하면 됩니다.

```python
# model = nn.DataParallel(SimpleModel())
# torch.save(model.module.state_dict(), "dp_model.pth")
# loaded_model = SimpleModel()
# loaded_model.load_state_dict(torch.load("dp_model.pth"))
print("DataParallel model saving/loading (conceptual).")
```

### 6.2. `nn.DistributedDataParallel` 모델 저장/로드

`nn.DistributedDataParallel`도 `module` 속성을 통해 원본 모델에 접근합니다. 각 프로세스가 모델의 독립적인 복사본을 가지므로, 일반적으로 랭크 0 (rank 0) 프로세스에서만 모델을 저장하고, 모든 프로세스에서 로드합니다. 로드 시에는 `map_location`을 사용하여 각 프로세스의 GPU로 매핑하는 것이 좋습니다.

```python
# # DDP 모델 저장 (rank 0에서만 저장)
# if dist.get_rank() == 0:
#     torch.save(model.module.state_dict(), "ddp_model.pth")

# # DDP 모델 로드 (모든 프로세스에서 로드)
# map_location = {'cuda:%d' % 0: 'cuda:%d' % dist.get_rank()}
# model.module.load_state_dict(torch.load("ddp_model.pth", map_location=map_location))
print("DistributedDataParallel model saving/loading (conceptual).")
```

## 7. 결론

PyTorch에서 모델을 저장하고 로드하는 것은 딥러닝 프로젝트의 필수적인 부분입니다. `state_dict`를 활용한 파라미터 저장 및 로드는 가장 유연하고 권장되는 방법이며, 체크포인트를 통해 학습 재개 기능을 구현할 수 있습니다. 또한, 전이 학습이나 분산 학습과 같은 특정 시나리오에서 모델을 저장하고 로드하는 방법을 이해하는 것은 모델의 효율적인 개발, 배포 및 활용에 매우 중요합니다. 이러한 지식은 딥러닝 모델의 생명주기 관리에 있어 핵심적인 역량이 됩니다.

