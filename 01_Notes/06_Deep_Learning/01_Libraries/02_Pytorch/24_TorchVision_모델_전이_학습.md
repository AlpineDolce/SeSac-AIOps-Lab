<h2>PyTorch TorchVision: 모델 전이 학습 (Transfer Learning)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-18

<h2>문서 목표</h2>
이 문서는 PyTorch의 컴퓨터 비전 라이브러리인 TorchVision을 활용한 모델 전이 학습(Transfer Learning)의 개념과 실제 적용 방법을 심층적으로 다룹니다. 전이 학습의 정의, 장점, 그리고 주요 전략인 특징 추출(Feature Extraction)과 미세 조정(Fine-tuning)을 상세히 설명합니다. 또한, TorchVision에서 제공하는 사전 학습된 모델을 로드하고 수정하는 방법, 데이터 전처리 파이프라인 구축, 그리고 실제 이미지 분류 문제에 전이 학습을 적용하는 코드 예시를 통해 효율적인 딥러닝 모델 개발 역량을 강화하는 데 기여하고자 합니다.

<h2>목차</h2>

- [1. TorchVision 소개](#1-torchvision-소개)
- [2. 전이 학습 (Transfer Learning) 개념](#2-전이-학습-transfer-learning-개념)
  - [2.1. 전이 학습이란?](#21-전이-학습이란)
  - [2.2. 전이 학습의 장점](#22-전이-학습의-장점)
  - [2.3. 전이 학습의 주요 전략](#23-전이-학습의-주요-전략)
- [3. TorchVision 사전 학습된 모델 활용](#3-torchvision-사전-학습된-모델-활용)
  - [3.1. 사전 학습된 모델 로드](#31-사전-학습된-모델-로드)
  - [3.2. 모델 구조 이해](#32-모델-구조-이해)
- [4. 전이 학습 구현 전략](#4-전이-학습-구현-전략)
  - [4.1. 특징 추출 (Feature Extraction)](#41-특징-추출-feature-extraction)
  - [4.2. 미세 조정 (Fine-tuning)](#42-미세-조정-fine-tuning)
- [5. 데이터 전처리 (Transforms)](#5-데이터-전처리-transforms)
  - [5.1. `torchvision.transforms`의 역할](#51-torchvisiontransforms의-역할)
  - [5.2. 일반적인 이미지 전처리 파이프라인](#52-일반적인-이미지-전처리-파이프라인)
- [6. 전이 학습 예시: 이미지 분류](#6-전이-학습-예시-이미지-분류)
  - [6.1. 데이터셋 준비](#61-데이터셋-준비)
  - [6.2. 모델 로드 및 수정](#62-모델-로드-및-수정)
  - [6.3. 옵티마이저 설정](#63-옵티마이저-설정)
  - [6.4. 학습 루프](#64-학습-루프)
- [7. 결론](#7-결론)

--- 

# PyTorch TorchVision: 모델 전이 학습 (Transfer Learning)

## 1. TorchVision 소개

TorchVision은 PyTorch의 공식 컴퓨터 비전 라이브러리로, 이미지 및 비디오 관련 딥러닝 작업을 위한 다양한 도구와 리소스를 제공합니다. 주요 구성 요소는 다음과 같습니다.

*   **`torchvision.datasets`**: MNIST, CIFAR-10/100, ImageNet 등 널리 사용되는 컴퓨터 비전 데이터셋을 쉽게 다운로드하고 로드할 수 있도록 지원합니다.
*   **`torchvision.models`**: AlexNet, VGG, ResNet, Inception, MobileNet 등 다양한 유명 신경망 아키텍처의 사전 학습된(pre-trained) 모델들을 제공합니다. 이 모델들은 대규모 데이터셋(예: ImageNet)으로 학습되어 강력한 특징 추출 능력을 가지고 있습니다.
*   **`torchvision.transforms`**: 이미지 데이터를 전처리하고 증강(augmentation)하는 데 필요한 다양한 변환(transform) 함수들을 제공합니다. 이를 통해 모델의 입력 요구사항을 충족시키고 학습 데이터의 다양성을 높일 수 있습니다.

TorchVision은 컴퓨터 비전 분야의 딥러닝 개발을 훨씬 효율적이고 편리하게 만들어 줍니다.

## 2. 전이 학습 (Transfer Learning) 개념

### 2.1. 전이 학습이란?

전이 학습은 한 분야에서 학습한 지식이나 모델을 다른 관련 분야의 문제 해결에 재활용하는 머신러닝 기법입니다. 딥러닝에서는 주로 대규모 데이터셋(예: ImageNet)으로 미리 학습된 모델(사전 학습 모델)의 가중치를 가져와 새로운, 더 작은 데이터셋에 대한 학습을 시작하는 것을 의미합니다. 사전 학습 모델은 이미 일반적인 특징(예: 이미지의 에지, 질감, 형태)을 추출하는 방법을 학습했기 때문에, 이 지식을 새로운 작업에 전이하여 활용할 수 있습니다.

### 2.2. 전이 학습의 장점

*   **데이터 부족 문제 해결**: 새로운 작업에 대한 데이터가 충분하지 않을 때, 사전 학습 모델의 풍부한 지식을 활용하여 모델의 성능을 향상시킬 수 있습니다.
*   **학습 시간 단축**: 모델이 이미 어느 정도 학습되어 있기 때문에, 처음부터 학습하는 것보다 훨씬 빠르게 수렴할 수 있습니다.
*   **더 나은 성능 달성**: 대규모 데이터셋으로 학습된 모델은 강력한 특징 추출 능력을 가지고 있어, 작은 데이터셋만으로 학습한 모델보다 더 좋은 성능을 달성할 가능성이 높습니다.
*   **계산 자원 절약**: 대규모 모델을 처음부터 학습시키는 데 필요한 막대한 계산 자원을 절약할 수 있습니다.

### 2.3. 전이 학습의 주요 전략

전이 학습은 크게 두 가지 주요 전략으로 나눌 수 있습니다.

*   **특징 추출 (Feature Extraction)**: 사전 학습된 모델의 특징 추출 부분(Convolutional Base 또는 Backbone)은 고정하고, 새로운 작업에 맞게 분류기 부분(Classifier Head)만 재학습하는 전략입니다.
*   **미세 조정 (Fine-tuning)**: 사전 학습된 모델의 일부 또는 전체 레이어를 새로운 데이터셋에 맞게 미세하게 조정하는 전략입니다.

## 3. TorchVision 사전 학습된 모델 활용

### 3.1. 사전 학습된 모델 로드

`torchvision.models` 모듈을 사용하여 다양한 사전 학습된 모델을 쉽게 로드할 수 있습니다. `pretrained=True` 인자를 설정하면 ImageNet 데이터셋으로 학습된 가중치를 함께 로드합니다.

```python
import torchvision.models as models

# ResNet-18 모델을 사전 학습된 가중치와 함께 로드
resnet18 = models.resnet18(pretrained=True)
print("\n--- ResNet-18 Model (pretrained) ---")
print(resnet18)

# VGG-16 모델을 사전 학습된 가중치와 함께 로드
vgg16 = models.vgg16(pretrained=True)
print("\n--- VGG-16 Model (pretrained) ---")
print(vgg16)
```

### 3.2. 모델 구조 이해

사전 학습된 모델은 일반적으로 크게 두 부분으로 구성됩니다.

*   **특징 추출기 (Feature Extractor / Backbone)**: 입력 이미지로부터 유의미한 특징을 추출하는 부분입니다. 대부분의 컨볼루션 레이어들이 여기에 해당하며, ImageNet과 같은 대규모 데이터셋으로 학습되어 일반적인 이미지 특징을 잘 추출할 수 있습니다.
*   **분류기 (Classifier / Head)**: 특징 추출기에서 나온 특징들을 바탕으로 최종 분류를 수행하는 부분입니다. 일반적으로 하나 이상의 선형(Linear) 레이어로 구성됩니다. 이 부분은 새로운 작업의 클래스 수에 맞게 수정해야 합니다.

예를 들어, ResNet 모델의 경우 `fc` (fully connected) 레이어가 분류기 역할을 합니다. VGG 모델의 경우 `classifier` 모듈이 분류기 역할을 합니다.

## 4. 전이 학습 구현 전략

### 4.1. 특징 추출 (Feature Extraction)

**개념**: 사전 학습된 모델의 특징 추출 부분(backbone)은 고정(freeze)하고, 새로운 작업에 맞게 분류기 부분만 재학습하는 전략입니다. 백본의 가중치는 업데이트되지 않습니다.

**구현**: 백본의 모든 파라미터에 대해 `requires_grad=False`로 설정하여 기울기 계산을 비활성화합니다. 그런 다음, 새로운 분류기 레이어를 정의하고 이 레이어의 파라미터만 옵티마이저에 전달하여 학습시킵니다.

**적용 시점**: 새로운 데이터셋의 크기가 작고, 원본 데이터셋(예: ImageNet)과 새로운 작업의 도메인이 유사할 때 주로 사용됩니다. 백본이 이미 충분히 일반적인 특징을 학습했다고 가정합니다.

```python
import torch.nn as nn

# ResNet-18 모델 로드
model_fe = models.resnet18(pretrained=True)

# 백본의 모든 파라미터 동결 (requires_grad=False 설정)
for param in model_fe.parameters():
    param.requires_grad = False

# 새로운 분류기 레이어 정의
# ResNet-18의 마지막 fc 레이어의 입력 특징 수는 512개입니다.
num_ftrs = model_fe.fc.in_features
model_fe.fc = nn.Linear(num_ftrs, 2) # 새로운 작업의 클래스 수(예: 2개 클래스)

print("\n--- ResNet-18 for Feature Extraction ---")
print(model_fe.fc) # 변경된 fc 레이어 확인

# 옵티마이저에는 새로 추가된 fc 레이어의 파라미터만 전달
# (requires_grad=True인 파라미터만 전달됨)
optimizer_fe = torch.optim.Adam(model_fe.fc.parameters(), lr=0.001)
print(f"Number of parameters to optimize (Feature Extraction): {sum(p.numel() for p in model_fe.parameters() if p.requires_grad)}")
```

### 4.2. 미세 조정 (Fine-tuning)

**개념**: 사전 학습된 모델의 일부 또는 전체 레이어를 새로운 데이터셋에 맞게 미세하게 조정하는 전략입니다. 백본의 가중치도 업데이트될 수 있습니다.

**구현**: 사전 학습된 모델의 일부 또는 전체 레이어의 `requires_grad`를 `True`로 유지합니다. 일반적으로 백본의 초기 레이어는 일반적인 특징을 학습하므로 동결하고, 후반부 레이어는 특정 작업에 특화된 특징을 학습하므로 동결을 해제하여 미세 조정합니다. 이때, 사전 학습된 가중치를 급격히 변경하지 않도록 매우 작은 학습률을 사용하는 것이 중요합니다.

**적용 시점**: 새로운 데이터셋의 크기가 충분히 크고, 원본 데이터셋과 새로운 작업의 도메인이 다소 차이가 있을 때 주로 사용됩니다. 모델이 새로운 작업에 더 잘 적응하도록 합니다.

```python
# ResNet-18 모델 로드
model_ft = models.resnet18(pretrained=True)

# 모든 파라미터의 requires_grad를 True로 유지 (기본값)
# 또는 특정 레이어만 동결 해제 (예: 마지막 컨볼루션 블록만)
# for name, param in model_ft.named_parameters():
#     if "layer4" not in name: # layer4 이전 레이어는 동결
#         param.requires_grad = False

# 새로운 분류기 레이어 정의 (클래스 수 변경)
num_ftrs_ft = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs_ft, 2) # 새로운 작업의 클래스 수

print("\n--- ResNet-18 for Fine-tuning ---")
print(model_ft.fc) # 변경된 fc 레이어 확인

# 옵티마이저에는 모델의 모든 (requires_grad=True인) 파라미터를 전달
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.0001) # 매우 작은 학습률
print(f"Number of parameters to optimize (Fine-tuning): {sum(p.numel() for p in model_ft.parameters() if p.requires_grad)}")
```

## 5. 데이터 전처리 (Transforms)

### 5.1. `torchvision.transforms`의 역할

사전 학습된 모델은 특정 크기와 정규화 방식을 가진 이미지 데이터로 학습되었습니다. 따라서 새로운 이미지 데이터를 모델의 입력 요구사항에 맞게 전처리하는 것이 필수적입니다. `torchvision.transforms`는 이러한 이미지 변환 작업을 효율적으로 수행할 수 있도록 돕습니다.

### 5.2. 일반적인 이미지 전처리 파이프라인

일반적으로 ImageNet으로 사전 학습된 모델을 사용할 때는 다음과 같은 전처리 단계를 따릅니다.

1.  **`transforms.Resize(size)`**: 이미지의 가장 짧은 변을 `size`로 조정하고, 비율을 유지하며 다른 변을 조정합니다.
2.  **`transforms.CenterCrop(size)`**: 이미지 중앙에서 `size` 크기로 이미지를 자릅니다. 사전 학습 모델은 보통 224x224 또는 299x299 크기의 이미지를 입력으로 받습니다.
3.  **`transforms.ToTensor()`**: PIL Image나 NumPy `ndarray`를 PyTorch Tensor로 변환합니다. 이 과정에서 이미지 픽셀 값의 범위가 [0, 255]에서 [0.0, 1.0]으로 자동 정규화됩니다.
4.  **`transforms.Normalize(mean, std)`**: Tensor의 각 채널을 평균(`mean`)과 표준편차(`std`)를 사용하여 정규화합니다. ImageNet으로 사전 학습된 모델의 경우, ImageNet 데이터셋의 평균과 표준편차를 사용하는 것이 일반적입니다.
    *   ImageNet 평균: `[0.485, 0.456, 0.406]`
    *   ImageNet 표준편차: `[0.229, 0.224, 0.225]`

```python
from torchvision import transforms

# 학습 데이터 전처리 (데이터 증강 포함)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224), # 랜덤 크롭 및 리사이즈
    transforms.RandomHorizontalFlip(), # 랜덤 수평 뒤집기
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 검증/테스트 데이터 전처리 (데이터 증강 없이 일관된 전처리)
val_transforms = transforms.Compose([
    transforms.Resize(256), # 먼저 짧은 변을 256으로 리사이즈
    transforms.CenterCrop(224), # 중앙 224x224 크롭
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Image transforms defined.")
```

## 6. 전이 학습 예시: 이미지 분류

다음은 TorchVision의 사전 학습된 ResNet-18 모델을 사용하여 간단한 이미지 분류 작업을 수행하는 전이 학습의 전체적인 흐름 예시입니다. (데이터셋 로드 및 학습 루프는 간략화되어 있습니다.)

### 6.1. 데이터셋 준비

`torchvision.datasets.ImageFolder`를 사용하여 폴더 구조에 따라 이미지를 로드하고, 위에서 정의한 `transforms`를 적용합니다.

```python
from torchvision import datasets
from torch.utils.data import DataLoader
import os

# 가상의 데이터셋 경로 (실제 경로로 변경 필요)
data_dir = './data/hymenoptera_data' # 예시: 꿀벌과 개미 이미지 데이터셋

# 데이터셋 로드
# train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
# val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transforms)

# data_loaders = {
#     'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
#     'val': DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
# }

# dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
# class_names = train_dataset.classes

print("Dataset preparation (conceptual).")
```

### 6.2. 모델 로드 및 수정

사전 학습된 ResNet-18 모델을 로드하고, 특징 추출을 위해 백본을 동결한 후, 새로운 분류기 레이어를 정의합니다.

```python
import torch.nn as nn

# 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 사전 학습된 ResNet-18 모델 로드
model = models.resnet18(pretrained=True)

# 백본 파라미터 동결
for param in model.parameters():
    param.requires_grad = False

# 마지막 완전 연결(fc) 레이어 교체 (새로운 클래스 수에 맞게)
# ImageNet은 1000개 클래스, 새로운 작업은 2개 클래스라고 가정
num_ftrs = model.fc.in_features # 기존 fc 레이어의 입력 특징 수 (512)
model.fc = nn.Linear(num_ftrs, 2) # 새로운 2개 클래스에 대한 출력 레이어

model = model.to(device) # 모델을 GPU로 이동

print("Model loaded and modified for transfer learning.")
```

### 6.3. 옵티마이저 설정

옵티마이저는 `requires_grad=True`인 파라미터만 업데이트합니다. 따라서 특징 추출 전략에서는 새로 추가된 `fc` 레이어의 파라미터만 옵티마이저에 전달하면 됩니다.

```python
import torch.optim as optim

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 옵티마이저 정의: 새로 추가된 (requires_grad=True인) 파라미터만 전달
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

print("Loss function and optimizer defined.")
```

### 6.4. 학습 루프

일반적인 PyTorch 학습 루프와 동일합니다. 각 에포크마다 데이터를 순전파하고, 손실을 계산하며, 역전파를 통해 기울기를 업데이트합니다.

```python
# 가상의 학습 루프 (실제 데이터 로더와 함께 사용)
num_epochs = 3

print("\nStarting conceptual training loop...")
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # 각 에포크는 학습 단계와 검증 단계를 가집니다.
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train() # 모델을 학습 모드로 설정
        else:
            model.eval()  # 모델을 평가 모드로 설정

        running_loss = 0.0
        running_corrects = 0

        # 가상의 데이터 로더 (실제 데이터 로더로 대체)
        # for inputs, labels in data_loaders[phase]:
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)

        #     optimizer.zero_grad() # 기울기 초기화

        #     with torch.set_grad_enabled(phase == 'train'): # 학습 단계에서만 기울기 계산 활성화
        #         outputs = model(inputs)
        #         _, preds = torch.max(outputs, 1)
        #         loss = criterion(outputs, labels)

        #     running_loss += loss.item() * inputs.size(0)
        #     running_corrects += torch.sum(preds == labels.data)

        # epoch_loss = running_loss / dataset_sizes[phase]
        # epoch_acc = running_corrects.double() / dataset_sizes[phase]

        # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # (모델 저장 로직 등)

print("Conceptual training loop finished.")
```

## 7. 결론

전이 학습은 제한된 데이터와 계산 자원으로도 강력한 딥러닝 모델을 구축할 수 있게 해주는 매우 효과적인 기법입니다. TorchVision은 사전 학습된 모델과 편리한 데이터 전처리 도구를 제공하여 이러한 전이 학습 과정을 크게 단순화합니다. 특징 추출과 미세 조정이라는 두 가지 주요 전략을 이해하고 적절히 활용한다면, 다양한 컴퓨터 비전 문제에서 빠르고 효율적으로 높은 성능을 달성할 수 있을 것입니다.
