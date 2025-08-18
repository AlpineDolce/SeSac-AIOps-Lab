<h2>PyTorch 분산 학습: `DataParallel` (DP)와 `DistributedDataParallel` (DDP)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-18

<h2>문서 목표</h2>
이 문서는 PyTorch에서 대규모 딥러닝 모델을 효율적으로 학습시키기 위한 분산 학습(Distributed Training) 기법을 심층적으로 다룹니다. 특히, PyTorch에서 제공하는 두 가지 주요 데이터 병렬화 전략인 `torch.nn.DataParallel` (DP)과 `torch.nn.parallel.DistributedDataParallel` (DDP)의 개념, 작동 원리, 구현 방법, 장단점을 상세히 비교 분석합니다. 이를 통해 단일 머신 내 다중 GPU 환경 및 다중 머신 환경에서 모델 학습을 확장하고 최적화하는 데 필요한 실질적인 지식과 코드 예시를 제공하여, 대규모 딥러닝 프로젝트 개발 역량을 강화하는 데 기여하고자 합니다.

<h2>목차</h2>

- [1. 분산 학습 (Distributed Training)의 필요성](#1-분산-학습-distributed-training의-필요성)
- [2. PyTorch 분산 학습 개요](#2-pytorch-분산-학습-개요)
  - [2.1. 데이터 병렬화 (Data Parallelism) vs. 모델 병렬화 (Model Parallelism)](#21-데이터-병렬화-data-parallelism-vs-모델-병렬화-model-parallelism)
- [3. `torch.nn.DataParallel` (DP)](#3-torchnndataparallel-dp)
  - [3.1. 개념](#31-개념)
  - [3.2. 작동 원리](#32-작동-원리)
  - [3.3. 구현 방법](#33-구현-방법)
  - [3.4. 장점과 한계](#34-장점과-한계)
- [4. `torch.nn.parallel.DistributedDataParallel` (DDP)](#4-torchnnparallelDistributedDataParallel-ddp)
  - [4.1. 개념](#41-개념)
  - [4.2. 작동 원리](#42-작동-원리)
  - [4.3. 구현 방법](#43-구현-방법)
  - [4.4. 장점과 고려사항](#44-장점과-고려사항)
- [5. DP vs. DDP 비교](#5-dp-vs-ddp-비교)
  - [5.1. 주요 차이점 요약](#51-주요-차이점-요약)
  - [5.2. 언제 무엇을 사용할까?](#52-언제-무엇을-사용할까)
- [6. 분산 학습을 위한 데이터 로딩](#6-분산-학습을-위한-데이터-로딩)
- [7. 결론](#7-결론)

---

# PyTorch 분산 학습: `DataParallel` (DP)와 `DistributedDataParallel` (DDP)

## 1. 분산 학습 (Distributed Training)의 필요성

최근 딥러닝 모델은 그 크기와 복잡성이 기하급수적으로 증가하고 있으며, 학습에 필요한 데이터셋의 규모 또한 방대해지고 있습니다. 이러한 대규모 모델과 데이터셋을 단일 GPU나 CPU로 학습시키는 것은 막대한 시간과 메모리 자원을 요구하며, 때로는 불가능하기도 합니다. 분산 학습(Distributed Training)은 여러 개의 GPU나 여러 대의 머신을 활용하여 모델 학습을 병렬화함으로써, 학습 시간을 단축하고 더 큰 모델을 학습시킬 수 있도록 하는 필수적인 기술입니다.

## 2. PyTorch 분산 학습 개요

PyTorch는 분산 학습을 위한 다양한 도구를 제공하며, 그중 가장 널리 사용되는 것은 **데이터 병렬화(Data Parallelism)** 방식입니다. 데이터 병렬화는 모델의 복사본을 여러 장치에 배치하고, 각 장치에 데이터셋의 다른 부분을 할당하여 동시에 학습을 진행하는 방식입니다.

### 2.1. 데이터 병렬화 (Data Parallelism) vs. 모델 병렬화 (Model Parallelism)

*   **데이터 병렬화 (Data Parallelism)**: 
    *   동일한 모델의 복사본을 여러 장치(GPU)에 배치합니다.
    *   전체 데이터 배치를 여러 장치로 나누어 각 장치에서 독립적으로 순전파 및 역전파를 수행합니다.
    *   각 장치에서 계산된 기울기(gradient)를 모아 평균을 낸 후, 이를 사용하여 모델 파라미터를 업데이트합니다.
    *   주로 모델이 단일 GPU 메모리에 들어갈 수 있지만, 배치 크기를 늘려 학습 속도를 높이고 싶을 때 사용됩니다.

*   **모델 병렬화 (Model Parallelism)**: 
    *   하나의 모델을 여러 장치에 나누어 배치합니다 (예: 모델의 절반은 GPU 0에, 나머지 절반은 GPU 1에).
    *   데이터는 한 번에 하나의 장치에서 처리되거나, 파이프라인 형태로 여러 장치를 거쳐 처리됩니다.
    *   주로 모델 자체가 너무 커서 단일 GPU 메모리에 들어가지 않을 때 사용됩니다.

이 문서에서는 PyTorch에서 가장 흔히 사용되는 데이터 병렬화 기법인 `DataParallel` (DP)과 `DistributedDataParallel` (DDP)에 초점을 맞춥니다.

## 3. `torch.nn.DataParallel` (DP)

### 3.1. 개념

`torch.nn.DataParallel` (DP)은 단일 머신 내에서 여러 GPU를 사용하여 모델 학습을 병렬화하는 가장 간단한 방법입니다. 이는 `nn.Module`을 래핑(wrap)하여 모델을 여러 GPU에 복제하고, 입력 데이터를 자동으로 분할하여 각 GPU에 할당합니다.

### 3.2. 작동 원리

1.  **모델 복제**: `DataParallel`은 원본 모델을 각 GPU에 복사합니다.
2.  **데이터 분할**: 입력 배치(batch)를 여러 서브 배치(sub-batch)로 분할하여 각 GPU에 보냅니다.
3.  **순전파**: 각 GPU는 할당된 서브 배치에 대해 독립적으로 순전파를 수행합니다.
4.  **기울기 수집 및 평균**: 각 GPU에서 계산된 기울기를 메인 GPU (기본적으로 GPU 0)로 모아 평균을 냅니다.
5.  **파라미터 업데이트**: 메인 GPU에서 파라미터를 업데이트한 후, 업데이트된 파라미터를 다른 GPU로 브로드캐스트(broadcast)합니다.

### 3.3. 구현 방법

`DataParallel`은 사용하기 매우 간단합니다. 모델을 정의한 후 `nn.DataParallel`로 래핑하기만 하면 됩니다.

```python
import torch
import torch.nn as nn

# 간단한 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# 모델 인스턴스 생성
model = SimpleModel()

# 여러 GPU가 사용 가능한 경우 DataParallel로 래핑
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    model = nn.DataParallel(model) # 기본적으로 모든 가용한 GPU 사용
    # model = nn.DataParallel(model, device_ids=[0, 1]) # 특정 GPU만 지정

model.to('cuda') # 모델을 GPU로 이동 (DataParallel이 알아서 각 GPU로 분배)

# 학습 루프는 단일 GPU 학습과 동일하게 진행
# outputs = model(inputs) # inputs는 여전히 단일 배치
print("DataParallel model setup (conceptual).")
```

### 3.4. 장점과 한계

*   **장점**: 
    *   **구현 용이성**: 기존 단일 GPU 학습 코드에서 모델을 `nn.DataParallel`로 래핑하는 것 외에 거의 코드 변경이 필요 없습니다.
    *   **빠른 테스트**: 단일 머신에서 여러 GPU를 빠르게 활용하여 학습 속도를 높일 수 있습니다.

*   **한계**: 
    *   **GIL (Global Interpreter Lock) 병목 현상**: `DataParallel`은 단일 Python 프로세스 내에서 동작하므로, CPU 바운드(CPU-bound) 작업(예: 데이터 전처리)이 많을 경우 GIL로 인해 병렬화의 이점을 충분히 누리지 못할 수 있습니다.
    *   **불균형한 GPU 활용**: 메인 GPU (GPU 0)가 데이터 분할, 기울기 수집 및 평균, 파라미터 업데이트 등 더 많은 작업을 처리하므로, GPU 0의 활용률이 다른 GPU보다 높아 불균형이 발생할 수 있습니다.
    *   **확장성 부족**: 다중 머신(multi-node) 환경에서는 사용할 수 없습니다.
    *   **Batch Normalization 문제**: 각 GPU가 자신의 서브 배치에 대해서만 배치 정규화 통계량(평균, 분산)을 계산하므로, 배치 크기가 작을 경우 통계량 추정이 불안정해질 수 있습니다. 이는 모델의 성능에 부정적인 영향을 미칠 수 있습니다.

## 4. `torch.nn.parallel.DistributedDataParallel` (DDP)

### 4.1. 개념

`torch.nn.parallel.DistributedDataParallel` (DDP)은 PyTorch에서 권장하는 분산 학습 방법입니다. DP와 달리 DDP는 각 GPU마다 별도의 Python 프로세스를 할당하여 동작합니다. 이는 단일 머신 내 다중 GPU 환경뿐만 아니라 다중 머신 환경에서도 효율적인 분산 학습을 가능하게 합니다.

### 4.2. 작동 원리

1.  **프로세스당 모델 복사본**: 각 프로세스는 자신의 GPU에 모델의 독립적인 복사본을 가집니다.
2.  **데이터 분할**: 각 프로세스는 `DistributedSampler`를 통해 전체 데이터셋의 고유한 부분집합을 할당받습니다.
3.  **순전파**: 각 프로세스는 할당된 데이터에 대해 독립적으로 순전파를 수행합니다.
4.  **기울기 동기화**: 역전파가 완료된 후, 각 프로세스에서 계산된 기울기는 효율적인 `all-reduce` 통신 프리미티브(primitive)를 사용하여 모든 프로세스 간에 평균화됩니다. 이 통신은 백그라운드에서 비동기적으로 이루어져 오버헤드를 최소화합니다.
5.  **파라미터 업데이트**: 각 프로세스는 평균화된 기울기를 사용하여 자신의 모델 파라미터를 독립적으로 업데이트합니다. 모든 프로세스가 동일한 기울기를 사용하여 업데이트하므로, 모델 파라미터는 모든 GPU에서 동기화된 상태를 유지합니다.

### 4.3. 구현 방법

DDP는 DP보다 설정이 복잡하지만, 더 나은 성능과 확장성을 제공합니다. 주요 단계는 다음과 같습니다.

1.  **프로세스 그룹 초기화**: `torch.distributed.init_process_group`을 사용하여 분산 환경을 설정합니다. 백엔드(예: `nccl` for GPU, `gloo` for CPU)와 통신 방식(예: `env://` for environment variables)을 지정합니다.
2.  **모델 래핑**: 모델을 `nn.parallel.DistributedDataParallel`로 래핑하고, 해당 프로세스가 사용할 GPU ID를 지정합니다.
3.  **데이터 로딩**: `torch.utils.data.distributed.DistributedSampler`를 사용하여 각 프로세스가 데이터셋의 고유한 부분집합을 로드하도록 합니다.
4.  **프로세스 실행**: `torch.distributed.launch` 또는 `torchrun` 유틸리티를 사용하여 여러 Python 프로세스를 시작합니다.

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
import os

# 1. 분산 환경 초기화 (실제 실행 시에는 torch.distributed.launch 또는 torchrun이 환경 변수 설정)
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# rank = int(os.environ['RANK']) # 현재 프로세스의 랭크
# world_size = int(os.environ['WORLD_SIZE']) # 총 프로세스 수
# dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# 2. 모델 정의 및 GPU로 이동
model = SimpleModel().cuda() # 각 프로세스는 자신의 GPU로 모델을 이동
# model = DDP(model, device_ids=[rank]) # DDP로 모델 래핑 (단일 GPU 프로세스당)

# 3. DistributedSampler를 사용한 데이터 로딩
# dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,))) # 가상의 데이터셋
# sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
# dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

# 학습 루프는 단일 GPU 학습과 유사하지만, 각 프로세스가 자신의 데이터만 처리
# for inputs, labels in dataloader:
#     inputs, labels = inputs.cuda(), labels.cuda()
#     # ... (순전파, 역전파, 옵티마이저 스텝)

print("DistributedDataParallel (DDP) model setup (conceptual).")
```

### 4.4. 장점과 고려사항

*   **장점**: 
    *   **진정한 병렬화**: 각 프로세스가 독립적으로 실행되므로 GIL 병목 현상이 없습니다.
    *   **균형 잡힌 GPU 활용**: 각 GPU가 동일한 양의 작업을 처리하여 활용률이 균형적입니다.
    *   **확장성**: 단일 머신 내 다중 GPU뿐만 아니라 다중 머신 환경에서도 효율적으로 확장됩니다.
    *   **Batch Normalization 문제 해결**: 각 프로세스가 자신의 로컬 배치에 대해 BN 통계량을 계산하지만, DDP는 이를 효율적으로 동기화하여 전체 배치에 대한 통계량과 동일한 효과를 냅니다.
    *   **더 빠른 통신**: `all-reduce`와 같은 효율적인 통신 프리미티브를 사용하여 기울기 동기화 오버헤드가 적습니다.

*   **고려사항**: 
    *   **복잡한 설정**: DP보다 초기 설정(분산 환경 초기화, 프로세스 관리)이 더 복잡합니다.
    *   **명시적 데이터 분할**: `DistributedSampler`를 사용하여 각 프로세스가 고유한 데이터를 받도록 해야 합니다.

## 5. DP vs. DDP 비교

### 5.1. 주요 차이점 요약

| 특징           | `DataParallel` (DP)                               | `DistributedDataParallel` (DDP)                               |
| :------------- | :------------------------------------------------ | :------------------------------------------------------------ |
| **구현 복잡성**| 매우 낮음 (모델 래핑만)                           | 높음 (분산 환경 초기화, 프로세스 관리, `DistributedSampler`)  |
| **프로세스 수**| 1개 (단일 Python 프로세스)                        | GPU 수만큼 (각 GPU당 별도 Python 프로세스)                  |
| **GIL 영향**   | 있음 (CPU 바운드 작업에서 병목)                   | 없음 (각 프로세스가 독립적)                                   |
| **GPU 활용률** | 불균형 (GPU 0에 부하 집중)                        | 균형 (각 GPU가 동일한 작업량)                                 |
| **확장성**     | 단일 머신 내 다중 GPU만 가능                      | 단일 머신 및 다중 머신 환경 모두 가능                         |
| **Batch Norm** | 로컬 배치 통계 사용 (문제 발생 가능)              | 전역 배치 통계와 동일한 효과 (문제 해결)                      |
| **통신 방식**  | 메인 GPU로 기울기 수집 후 브로드캐스트            | `all-reduce`를 통한 효율적인 분산 통신                        |

### 5.2. 언제 무엇을 사용할까?

*   **`DataParallel` (DP)**: 
    *   **간단한 테스트**: 코드를 빠르게 실행하여 여러 GPU에서 동작하는지 확인하고 싶을 때.
    *   **매우 작은 모델**: 모델이 작고 GPU 0의 병목 현상이 크게 문제가 되지 않을 때.
    *   **빠른 프로토타이핑**: 복잡한 설정 없이 빠르게 다중 GPU를 사용하고 싶을 때.

*   **`DistributedDataParallel` (DDP)**: 
    *   **대규모 모델 학습**: 모델이 크거나 배치 크기가 커서 단일 GPU에 들어가지 않을 때.
    *   **생산 환경**: 안정적이고 효율적인 학습이 필요한 실제 서비스 환경.
    *   **다중 머신 학습**: 여러 대의 서버에 걸쳐 분산 학습을 수행해야 할 때.
    *   **최적의 성능**: GPU 활용률을 극대화하고 학습 시간을 최소화하고 싶을 때.

## 6. 분산 학습을 위한 데이터 로딩

`DistributedDataParallel`을 사용할 때는 각 프로세스가 데이터셋의 고유하고 겹치지 않는 부분집합을 로드하도록 해야 합니다. 이를 위해 PyTorch는 `torch.utils.data.distributed.DistributedSampler`를 제공합니다.

`DistributedSampler`는 `Dataset`을 래핑하고, 각 프로세스에 데이터셋의 특정 인덱스 범위를 할당하여 데이터 중복 없이 효율적으로 데이터를 분할합니다. `DataLoader`와 함께 사용되며, `shuffle=False`로 설정해야 합니다 (샘플러가 셔플링을 담당).

```python
# DistributedSampler 사용 예시 (개념적)
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data import DataLoader, TensorDataset

# dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,))) # 1000개 샘플

# # world_size: 총 프로세스 수, rank: 현재 프로세스의 랭크
# sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)

# # 각 프로세스는 dataloader를 통해 자신의 고유한 데이터 부분집합을 받습니다.
# for inputs, labels in dataloader:
#     # ... 학습 코드
print("DistributedSampler for DDP data loading (conceptual).")
```

## 7. 결론

PyTorch에서 분산 학습은 대규모 딥러닝 모델을 효율적으로 학습시키는 데 필수적인 기술입니다. `DataParallel`은 간단한 사용법으로 빠른 테스트에 적합하지만, `DistributedDataParallel`은 더 복잡한 설정에도 불구하고 진정한 병렬화, 균형 잡힌 GPU 활용, 뛰어난 확장성, 그리고 Batch Normalization 문제 해결 등 여러 장점을 제공하여 대규모 및 생산 환경 학습에 권장되는 방법입니다. `DistributedSampler`와 같은 도구와 함께 DDP를 효과적으로 활용한다면, 제한된 자원에서도 최첨단 딥러닝 모델을 성공적으로 학습시킬 수 있을 것입니다.
