# 09_순환_신경망_RNN_LSTM_GRU.md: 시퀀스 데이터 처리를 위한 딥러닝 모델

## 문서 목표
이 문서는 시퀀스 데이터(텍스트, 음성, 시계열 등) 처리에 특화된 순환 신경망(RNN)의 기본 구조와 장기 의존성 문제를 해결하는 LSTM 및 GRU의 원리를 학습하여 시퀀스 데이터 관련 딥러닝 문제 해결 능력을 향상시키는 데 도움이 되기를 바랍니다.

---

## 목차

- [1. 순환 신경망 (Recurrent Neural Network, RNN) 개요](#1-순환-신경망-recurrent-neural-network-rnn-개요)
  - [1.1. 시퀀스 데이터와 RNN의 필요성](#11-시퀀스-데이터와-rnn의-필요성)
  - [1.2. RNN의 구조와 동작 원리](#12-rnn의-구조와-동작-원리)
  - [1.3. RNN의 한계: 장기 의존성 문제](#13-rnn의-한계-장기-의존성-문제)
- [2. 장단기 메모리 (Long Short-Term Memory, LSTM)](#2-장단기-메모리-long-short-term-memory-lstm)
  - [2.1. LSTM의 등장 배경](#21-lstm의-등장-배경)
  - [2.2. LSTM의 구조와 게이트 (Gate) 메커니즘](#22-lstm의-구조와-게이트-gate-메커니즘)
    - [2.2.1. 망각 게이트 (Forget Gate)](#221-망각-게이트-forget-gate)
    - [2.2.2. 입력 게이트 (Input Gate)](#222-입력-게이트-input-gate)
    - [2.2.3. 셀 상태 (Cell State) 업데이트](#223-셀-상태-cell-state-업데이트)
    - [2.2.4. 출력 게이트 (Output Gate)](#224-출력-게이트-output-gate)
  - [2.3. LSTM의 장점](#23-lstm의-장점)
- [3. 게이트 순환 유닛 (Gated Recurrent Unit, GRU)](#3-게이트-순환-유닛-gated-recurrent-unit-gru)
  - [3.1. GRU의 등장 배경](#31-gru의-등장-배경)
  - [3.2. GRU의 구조와 게이트 메커니즘](#32-gru의-구조와-게이트-메커니즘)
    - [3.2.1. 업데이트 게이트 (Update Gate)](#321-업데이트-게이트-update-gate)
    - [3.2.2. 리셋 게이트 (Reset Gate)](#322-리셋-게이트-reset-gate)
  - [3.3. GRU의 특징](#33-gru의-특징)
- [4. RNN, LSTM, GRU 비교](#4-rnn-lstm-gru-비교)

---

## 1. 순환 신경망 (Recurrent Neural Network, RNN) 개요

### 1.1. 시퀀스 데이터와 RNN의 필요성

기존의 다층 퍼셉트론(MLP)이나 합성곱 신경망(CNN)은 독립적인 입력 데이터를 처리하는 데 효과적입니다. 하지만 텍스트, 음성, 비디오, 시계열 데이터와 같이 **시간적 또는 순서적 의존성을 가지는 시퀀스(Sequence) 데이터**를 처리하는 데는 한계가 있습니다. 예를 들어, 문장의 다음 단어를 예측하려면 이전 단어들의 정보를 기억하고 있어야 합니다.

**순환 신경망(Recurrent Neural Network, RNN)**은 이러한 시퀀스 데이터의 특성을 반영하여, 이전 단계의 정보를 현재 단계의 계산에 활용하는 '기억' 능력을 가진 신경망입니다.

### 1.2. RNN의 구조와 동작 원리

RNN은 은닉층의 뉴런이 자신에게 다시 연결되는 **순환(Recurrent) 연결**을 가집니다. 이 순환 연결을 통해 이전 시점의 은닉 상태(Hidden State)가 현재 시점의 입력과 함께 다음 은닉 상태를 계산하는 데 사용됩니다. 이는 RNN이 시퀀스 데이터를 처리하면서 정보를 '기억'하고 '전달'하는 메커니즘을 제공합니다.

```mermaid
graph TD
    Input[입력 x(t)] --> Hidden[은닉층 h(t)]
    Hidden --> Output[출력 y(t)]
    Hidden -- 순환 연결 --> Hidden
```

RNN은 시퀀스 길이에 따라 네트워크를 펼쳐서(Unroll) 표현할 수 있습니다. 각 시점(t)에서 입력 `x(t)`와 이전 시점의 은닉 상태 `h(t-1)`를 받아 현재 시점의 은닉 상태 `h(t)`와 출력 `y(t)`를 계산합니다.

### 1.3. RNN의 한계: 장기 의존성 문제

RNN은 이론적으로 장기적인 의존성을 학습할 수 있지만, 실제로는 **장기 의존성(Long-Term Dependencies) 문제**에 직면합니다. 시퀀스 길이가 길어질수록 역전파 과정에서 기울기가 점차 작아지거나(기울기 소실, Vanishing Gradient) 너무 커지는(기울기 폭주, Exploding Gradient) 문제가 발생하여, 먼 과거의 정보가 현재 시점까지 제대로 전달되지 못하거나 학습이 불안정해집니다.

## 2. 장단기 메모리 (Long Short-Term Memory, LSTM)

### 2.1. LSTM의 등장 배경

장기 의존성 문제를 해결하기 위해 1997년 호크라이터(Hochreiter)와 슈미트후버(Schmidhuber)가 제안한 것이 **장단기 메모리(Long Short-Term Memory, LSTM)**입니다. LSTM은 RNN의 순환 구조 내부에 '게이트(Gate)'라는 특별한 메커니즘을 추가하여 정보를 선택적으로 기억하거나 잊어버리도록 제어합니다.

### 2.2. LSTM의 구조와 게이트 (Gate) 메커니즘

LSTM은 셀 상태(Cell State)라는 별도의 메모리 셀을 유지하며, 세 가지 주요 게이트를 통해 이 셀 상태에 정보를 추가하거나 제거합니다.

*   **셀 상태 (Cell State, C)**: 정보를 장기적으로 저장하는 '컨베이어 벨트' 역할을 합니다. 게이트에 의해 정보가 추가되거나 제거될 수 있습니다.

#### 2.2.1. 망각 게이트 (Forget Gate, f)

*   **역할**: 이전 셀 상태 `C(t-1)`에서 어떤 정보를 '잊어버릴지' 결정합니다.
*   **동작**: 시그모이드 함수를 통해 0과 1 사이의 값을 출력하며, 0에 가까우면 정보를 잊고, 1에 가까우면 정보를 유지합니다.

#### 2.2.2. 입력 게이트 (Input Gate, i)

*   **역할**: 현재 입력 `x(t)`와 이전 은닉 상태 `h(t-1)`로부터 어떤 새로운 정보를 '기억할지' 결정합니다.
*   **동작**: 시그모이드 함수로 어떤 정보를 업데이트할지 결정하고, tanh 함수로 새로운 후보 셀 상태 `C_tilde(t)`를 생성합니다. 이 두 값을 조합하여 셀 상태에 추가할 정보를 결정합니다.

#### 2.2.3. 셀 상태 (Cell State) 업데이트

*   망각 게이트의 결과와 이전 셀 상태 `C(t-1)`를 곱하고, 입력 게이트의 결과와 새로운 후보 셀 상태 `C_tilde(t)`를 곱한 후 더하여 현재 셀 상태 `C(t)`를 업데이트합니다.

#### 2.2.4. 출력 게이트 (Output Gate, o)

*   **역할**: 현재 셀 상태 `C(t)`로부터 어떤 정보를 '출력할지' 결정합니다.
*   **동작**: 시그모이드 함수로 어떤 부분을 출력할지 결정하고, tanh 함수로 셀 상태를 변환한 후 이 두 값을 곱하여 현재 은닉 상태 `h(t)`를 생성합니다. 이 `h(t)`가 다음 시점으로 전달되고 출력으로 사용됩니다.

```mermaid
graph TD
    subgraph LSTM Cell
        Input_x[x(t)]
        Hidden_prev[h(t-1)]
        Cell_prev[C(t-1)]

        Input_x --> ForgetGate
        Hidden_prev --> ForgetGate
        ForgetGate[망각 게이트 f] --> Cell_update

        Input_x --> InputGate
        Hidden_prev --> InputGate
        InputGate[입력 게이트 i] --> Cell_update
        Input_x --> CandidateCell
        Hidden_prev --> CandidateCell
        CandidateCell[후보 셀 상태 C_tilde] --> Cell_update

        Cell_prev --> Cell_update
        Cell_update[셀 상태 업데이트] --> Cell_curr[C(t)]

        Input_x --> OutputGate
        Hidden_prev --> OutputGate
        OutputGate[출력 게이트 o] --> Hidden_curr
        Cell_curr --> Tanh_Cell
        Tanh_Cell[tanh(C(t))] --> Hidden_curr
        Hidden_curr[h(t)]
    end

    Hidden_curr --> Next_Hidden[h(t+1)]
    Cell_curr --> Next_Cell[C(t+1)]
    Hidden_curr --> Output_y[y(t)]
```

### 2.3. LSTM의 장점

*   **장기 의존성 문제 해결**: 게이트 메커니즘을 통해 기울기 소실 문제를 완화하고, 먼 과거의 정보를 효과적으로 기억하고 전달할 수 있습니다.
*   **다양한 시퀀스 데이터에 적용**: 자연어 처리, 음성 인식, 시계열 예측 등 다양한 시퀀스 데이터 문제에서 뛰어난 성능을 보입니다.

## 3. 게이트 순환 유닛 (Gated Recurrent Unit, GRU)

### 3.1. GRU의 등장 배경

2014년 조경현 교수팀이 제안한 **게이트 순환 유닛(Gated Recurrent Unit, GRU)**은 LSTM의 장기 의존성 해결 능력을 유지하면서도 구조를 단순화한 모델입니다. LSTM보다 파라미터 수가 적어 학습 속도가 빠르다는 장점이 있습니다.

### 3.2. GRU의 구조와 게이트 메커니즘

GRU는 LSTM의 셀 상태와 은닉 상태를 통합하고, 두 개의 게이트(업데이트 게이트, 리셋 게이트)만을 사용합니다.

#### 3.2.1. 업데이트 게이트 (Update Gate, z)

*   **역할**: 이전 은닉 상태 `h(t-1)`의 정보를 얼마나 현재 은닉 상태 `h(t)`로 가져올지 결정합니다. LSTM의 망각 게이트와 입력 게이트를 결합한 역할을 합니다.

#### 3.2.2. 리셋 게이트 (Reset Gate, r)

*   **역할**: 이전 은닉 상태 `h(t-1)`의 정보를 얼마나 '잊어버릴지' 결정합니다. 리셋 게이트의 값이 0에 가까우면 이전 은닉 상태의 정보를 무시하고 현재 입력에만 집중합니다.

```mermaid
graph TD
    subgraph GRU Cell
        Input_x[x(t)]
        Hidden_prev[h(t-1)]

        Input_x --> UpdateGate
        Hidden_prev --> UpdateGate
        UpdateGate[업데이트 게이트 z]

        Input_x --> ResetGate
        Hidden_prev --> ResetGate
        ResetGate[리셋 게이트 r]

        ResetGate --> CandidateHidden
        Hidden_prev --> CandidateHidden
        Input_x --> CandidateHidden
        CandidateHidden[후보 은닉 상태 h_tilde]

        UpdateGate --> Hidden_curr
        Hidden_prev --> Hidden_curr
        CandidateHidden --> Hidden_curr[h(t)]
    end

    Hidden_curr --> Next_Hidden[h(t+1)]
    Hidden_curr --> Output_y[y(t)]
```

### 3.3. GRU의 특징

*   **단순한 구조**: LSTM보다 게이트 수가 적어 모델이 더 가볍고 학습 속도가 빠릅니다.
*   **LSTM과 유사한 성능**: 많은 경우 LSTM과 비슷한 성능을 보이며, 데이터셋의 특성이나 문제에 따라 GRU가 더 좋은 성능을 보이기도 합니다.

## 4. RNN, LSTM, GRU 비교

| 특징         | RNN (바닐라 RNN)       | LSTM                     | GRU                      |
| :----------- | :--------------------- | :----------------------- | :----------------------- |
| **구조**     | 단일 순환 연결         | 셀 상태 + 3개 게이트     | 은닉 상태 통합 + 2개 게이트 |
| **복잡도**   | 가장 낮음              | 가장 높음                | 중간 (LSTM보다 낮음)     |
| **파라미터 수** | 가장 적음              | 가장 많음                | 중간 (LSTM보다 적음)     |
| **장기 의존성** | 취약 (기울기 소실/폭주) | 우수 (게이트로 제어)     | 우수 (게이트로 제어)     |
| **학습 속도** | 빠름 (불안정)          | 느림                     | 빠름 (LSTM보다 빠름)     |

일반적으로 LSTM과 GRU는 RNN의 장기 의존성 문제를 효과적으로 해결하여 시퀀스 데이터 처리에서 뛰어난 성능을 보입니다. 문제의 복잡도, 데이터셋의 크기, 계산 자원 등을 고려하여 적절한 모델을 선택할 수 있습니다.

다음 장에서는 최신 자연어 처리의 핵심인 **어텐션 메커니즘과 트랜스포머**에 대해 자세히 알아보겠습니다.