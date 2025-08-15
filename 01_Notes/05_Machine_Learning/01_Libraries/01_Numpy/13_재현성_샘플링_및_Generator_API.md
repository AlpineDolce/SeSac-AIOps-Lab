<h2>NumPy 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
이 문서는 재현 가능하고 안정적인 과학 실험을 위한 난수 생성 기법, 최신 난수 생성 API(`np.random.Generator`)의 사용법, 그리고 데이터를 무작위로 섞거나 샘플링하는 방법을 학습합니다.

<h2>목차</h2>

- [1. 재현 가능한 난수 생성](#1-재현-가능한-난수-생성)
  - [1.1. `np.random.seed`와 `RandomState`](#11-nprandomseed와-randomstate)
  - [1.2. 최신 난수 생성 API: `np.random.Generator`](#12-최신-난수-생성-api-nprandomgenerator)
    - [1.2.1. 왜 새로운 API를 사용해야 하는가?](#121-왜-새로운-api를-사용해야-하는가)
    - [1.2.2. `Generator` 생성 및 시드(Seed) 설정](#122-generator-생성-및-시드seed-설정)
    - [1.2.3. 주요 난수 생성 메서드](#123-주요-난수-생성-메서드)
      - [1.2.3.1. `rng.random()`](#1231-rngrandom)
      - [1.2.3.2. `rng.integers()`](#1232-rngintegers)
      - [1.2.3.3. `rng.standard_normal()`](#1233-rngstandard_normal)
      - [1.2.3.4. `rng.uniform()`](#1234-rnguniform)
      - [1.2.3.5. `rng.normal()`](#1235-rngnormal)
  - [1.3. 재현성 모범 사례 (Reproducibility Best Practices)](#13-재현성-모범-사례-reproducibility-best-practices)
- [2. 샘플링 및 순열](#2-샘플링-및-순열)
  - [2.1. 전통적인 API (`np.random.shuffle`, `np.random.permutation`, `np.random.choice`)](#21-전통적인-api-nprandomshuffle-nprandompermutation-nprandomchoice)
    - [2.1.1. `np.random.shuffle()`](#211-nprandomshuffle)
    - [2.1.2. `np.random.permutation()`](#212-nprandompermutation)
    - [2.1.3. `np.random.choice()`](#213-nprandomchoice)
  - [2.2. Generator API (`rng.shuffle`, `rng.permutation`, `rng.choice`)](#22-generator-api-rngshuffle-rngpermutation-rngchoice)
    - [2.2.1. `rng.shuffle()`](#221-rngshuffle)
    - [2.2.2. `rng.permutation()`](#222-rngpermutation)
    - [2.2.3. `rng.choice()`](#223-rngchoice)
- [3. 전통적인(Legacy) 난수 생성 API의 문제점](#3-전통적인legacy-난수-생성-api의-문제점)

## 1. 재현 가능한 난수 생성

난수 생성은 본질적으로 무작위적이지만, 과학 연구, 데이터 분석, 머신러닝 모델 개발 및 디버깅 시에는 **동일한 난수 시퀀스를 반복해서 생성**해야 할 필요가 있습니다. 이를 **재현성(Reproducibility)**이라고 하며, 실험 결과를 검증하고 공유하며, 모델의 성능을 일관되게 평가하는 데 필수적입니다. NumPy는 이러한 재현성을 위해 **시드(seed)**를 설정하는 두 가지 주요 방법을 제공합니다.

### 1.1. `np.random.seed`와 `RandomState`

**1. `np.random.seed(seed)`: 전역 난수 생성기 시드 설정 (레거시 방식)**

*   **정의**: `np.random.seed(seed)` 함수는 NumPy의 **전역(global) 의사 난수 생성기**의 초기 상태를 설정합니다. 한 번 시드를 설정하면, 이후에 `np.random` 모듈의 어떤 함수(`np.random.rand()`, `np.random.randn()`, `np.random.randint()` 등)를 호출하더라도 동일한 시퀀스의 난수가 생성됩니다.
*   **작동 방식**: 시드 값은 난수 생성 알고리즘의 시작점을 결정합니다. 동일한 시드 값은 항상 동일한 난수 시퀀스를 생성합니다.
*   **장점**: 간단한 스크립트나 단일 파일에서 빠르게 재현성을 확보할 때 편리합니다.
*   **단점**: 
    *   **전역 상태 오염**: 전역 상태를 변경하므로, 코드의 다른 부분이나 다른 모듈에서 `np.random` 함수를 호출할 경우 예상치 못한 방식으로 난수 시퀀스가 변경될 수 있습니다. 이는 복잡한 애플리케이션에서 디버깅을 어렵게 만듭니다.
    *   **스레드 안전성 부족**: 멀티스레드 환경에서 여러 스레드가 동시에 전역 난수 생성기를 사용하면 예측 불가능한 결과가 발생할 수 있습니다.

**2. `np.random.RandomState(seed)`: 독립적인 난수 생성기 객체 생성 (권장 방식)**

*   **정의**: `np.random.RandomState(seed)`는 **독립적인 의사 난수 생성기 객체**를 생성합니다. 이 객체는 자신만의 내부 상태를 가지며, 이 객체의 메서드를 통해서만 난수를 생성합니다.
*   **작동 방식**: 각 `RandomState` 객체는 고유한 시드에 의해 초기화되며, 다른 `RandomState` 객체나 전역 난수 생성기와 독립적으로 작동합니다.
*   **장점**: 
    *   **격리(Isolation)**: 코드의 여러 부분에서 독립적인 난수 스트림이 필요할 때 유용합니다. 예를 들어, 데이터 증강과 모델 초기화에 각각 다른, 그러나 재현 가능한 난수 스트림을 사용할 수 있습니다.
    *   **제어력**: 특정 난수 생성 작업에 대한 완벽한 제어력을 제공하여, 다른 코드 변경이 난수 시퀀스에 영향을 미치지 않도록 합니다.
    *   **스레드 안전성**: 멀티스레드 환경에서 각 스레드가 독립적인 `RandomState` 객체를 사용하면 안전하게 난수를 생성할 수 있습니다.
*   **단점**: `np.random.seed()`보다 약간 더 많은 코드를 작성해야 합니다.

**비교 및 권장 사항:**
복잡한 프로젝트, 라이브러리 개발, 또는 멀티스레드 환경에서는 `np.random.seed()`와 같은 전역 상태를 변경하는 방식보다는 `np.random.RandomState()` (또는 더 최신인 `np.random.default_rng()`)를 사용하여 독립적인 난수 생성기 객체를 관리하는 것이 **강력히 권장**됩니다. 이는 코드의 모듈성을 높이고, 예측 불가능한 버그를 방지하며, 실험의 재현성을 더욱 견고하게 보장합니다.

**예시:**

```python
import numpy as np

print("--- np.random.seed()를 사용한 재현성 ---")
# np.random.seed를 사용한 재현성
np.random.seed(42)
rand_a = np.random.rand(3)
print(f"시드 42 설정 후 난수 A: {rand_a}")

np.random.seed(42) # 다시 시드를 설정해야 동일한 시퀀스 시작
rand_b = np.random.rand(3)
print(f"시드 42 재설정 후 난수 B: {rand_b}") # rand_a와 동일한 결과

# 시드를 설정하지 않으면 매번 다른 결과
rand_c = np.random.rand(3)
print(f"시드 미설정 후 난수 C: {rand_c}")

# 시드를 다시 설정하면 이전 시퀀스와는 다른 새로운 시퀀스 시작
np.random.seed(100)
rand_d = np.random.rand(3)
print(f"시드 100 설정 후 난수 D: {rand_d}")

print("\n--- np.random.RandomState()를 사용한 재현성 ---")
# RandomState 객체를 사용한 재현성 (권장) 
# 각 RandomState 객체는 독립적인 난수 스트림을 가집니다.
rs1 = np.random.RandomState(123)
rs2 = np.random.RandomState(456)
rs3 = np.random.RandomState(123) # rs1과 동일한 시퀀스

print(f"RandomState 1 (시드 123, 첫 번째 호출): {rs1.rand(3)}")
print(f"RandomState 2 (시드 456, 첫 번째 호출): {rs2.rand(3)}")
print(f"RandomState 1 (시드 123, 두 번째 호출): {rs1.rand(3)}") # rs1의 다음 난수
print(f"RandomState 3 (시드 123, 첫 번째 호출): {rs3.rand(3)}") # rs1의 첫 번째 호출과 동일

# 전역 난수 생성기는 RandomState 객체에 영향을 받지 않음
print(f"\n전역 난수 (RandomState와 독립): {np.random.rand(3)}")
```
### 1.2. 최신 난수 생성 API: `np.random.Generator`

NumPy 1.17 버전부터는 새로운 난수 생성 API 사용이 강력하게 권장됩니다. 이는 기존의 전역 상태 기반(`np.random.seed()`) 또는 `RandomState` 객체 기반 방식에서 한 단계 발전한, **생성기(`Generator`) 객체**를 먼저 만든 후 그 객체의 메서드를 호출하는 방식입니다. 이 새로운 API는 난수 생성의 재현성, 통계적 품질, 성능 및 확장성을 크게 향상시킵니다.

**핵심 개념:**
새로운 API의 핵심은 **비트 생성기(BitGenerator)**와 **생성기(Generator)**의 분리입니다.
*   **비트 생성기**: 실제 무작위 비트 스트림을 생성하는 알고리즘(예: PCG64). 이는 난수의 통계적 품질과 속도를 담당합니다.
*   **생성기**: 비트 생성기에서 생성된 비트를 사용하여 다양한 확률 분포(정규, 균일, 이항 등)의 난수를 생성하는 인터페이스를 제공합니다.

`np.random.default_rng()` 함수는 기본적으로 **PCG64** 비트 생성기를 사용하는 `Generator` 객체를 생성합니다. PCG64는 기존의 MT19937(Mersenne Twister)보다 통계적 품질이 우수하고, 병렬 처리 및 시드 관리에 더 효율적입니다.

#### 1.2.1. 왜 새로운 API를 사용해야 하는가?

새로운 `Generator` API는 다음과 같은 중요한 이점을 제공합니다.
*   **향상된 재현성(Reproducibility)**: `np.random.seed()`와 같은 전역(global) 상태에 의존하지 않고, 각 `Generator` 객체는 독립적인 난수 스트림을 가집니다. 이는 코드의 다른 부분에 영향을 주지 않아 복잡한 프로그램이나 라이브러리에서 실험의 재현성을 보장하기 훨씬 용이합니다. 여러 개의 독립적인 난수 스트림이 필요할 때 충돌 없이 관리할 수 있습니다.
*   **우수한 통계적 품질**: 기본 비트 생성기인 PCG64는 기존의 MT19937보다 통계적 테스트에서 더 좋은 성능을 보이며, 더 예측 불가능한(진정한 의미의 무작위성에 가까운) 난수를 생성합니다.
*   **성능 향상**: PCG64는 특정 시나리오에서 MT19937보다 더 빠른 난수 생성을 제공합니다.
*   **확장성 및 유연성**: 비트 생성기와 생성기 인터페이스가 분리되어 있어, 향후 새로운 비트 생성 알고리즘을 쉽게 통합하거나 사용자가 특정 요구사항에 맞는 비트 생성기를 선택할 수 있습니다.
*   **스레드 안전성**: `Generator` 객체는 스레드 안전하게 설계되어 멀티스레드 환경에서 난수를 생성할 때 발생할 수 있는 문제를 줄여줍니다.
*   **명확한 API**: 모든 난수 생성 메서드가 `Generator` 객체에 속하므로, 코드를 읽고 이해하기가 더 직관적입니다.

**기본 사용 패턴:**
새로운 API의 사용은 매우 간단합니다. 먼저 `np.random.default_rng()`를 통해 `Generator` 객체를 생성하고, 그 객체의 메서드를 호출하여 난수를 생성합니다.

```python
import numpy as np

# 1. Generator 객체 생성 (시드 설정 가능)
rng = np.random.default_rng(seed=42)

# 2. 생성기 객체의 메서드를 사용하여 난수 생성
# 균일 분포 실수
random_floats = rng.random(size=(2, 3))
print(f"랜덤 실수 배열:\n{random_floats}")

# 정수
random_integers = rng.integers(0, 10, size=(2, 3))
print(f"랜덤 정수 배열:\n{random_integers}")

# 정규 분포 실수
random_normal = rng.normal(loc=0, scale=1, size=5)
print(f"정규 분포 난수: {random_normal}")
```

#### 1.2.2. `Generator` 생성 및 시드(Seed) 설정

`np.random.Generator` 객체를 생성하는 가장 일반적인 방법은 `np.random.default_rng()` 함수를 사용하는 것입니다. 이 함수에 정수 시드 값을 전달하여 생성기를 초기화할 수 있으며, 이는 난수 생성의 재현성을 보장하는 핵심적인 단계입니다.

**`np.random.default_rng(seed=None)`:**
*   **`seed`**: 선택 사항. `Generator` 객체의 초기 상태를 설정하는 데 사용되는 시드 값입니다. 
    *   `None` (기본값): 시스템의 엔트로피(예: 운영체제의 무작위성 소스)를 사용하여 시드를 설정합니다. 이 경우 매번 다른 난수 시퀀스가 생성됩니다.
    *   정수 값 (예: `42`): 특정 정수 값을 시드로 사용하면, 해당 시드로 초기화된 `Generator`는 항상 동일한 순서의 난수를 반환합니다. 이는 실험의 재현성을 위해 필수적입니다.
    *   `SeedSequence` 객체: 더 복잡한 시드 관리(예: 여러 하위 생성기에 시드 분배)를 위해 `np.random.SeedSequence` 객체를 전달할 수도 있습니다.

**재현성 보장:**
동일한 시드로 생성된 `Generator` 객체는 항상 동일한 난수 시퀀스를 생성합니다. 이는 연구 결과를 공유하거나, 버그를 재현하거나, 모델의 성능 변화를 추적할 때 매우 중요합니다. `np.random.seed()`와 달리, `Generator` 객체는 독립적인 난수 스트림을 관리하므로, 다른 `Generator` 객체나 전역 난수 생성기에 의해 영향을 받지 않습니다.

**예시:**

```python
import numpy as np

# 시드를 설정하여 난수 생성기 생성
rng = np.random.default_rng(seed=42)

# 생성기의 메서드를 사용하여 난수 생성
print(f"랜덤 실수 배열 (rng):\n{rng.random(size=(2, 3))}")
print(f"랜덤 정수 배열 (rng):\n{rng.integers(0, 10, size=(2, 3))}")

# 동일한 시드로 다시 생성하면 완전히 동일한 결과가 나옴
rng_same_seed = np.random.default_rng(seed=42)
print(f"\n동일 시드로 다시 생성한 배열 (rng_same_seed):\n{rng_same_seed.random(size=(2, 3))}")

# 시드를 설정하지 않으면 매번 다른 결과가 나옴
rng_no_seed = np.random.default_rng()
print(f"\n시드 없이 생성한 배열 (rng_no_seed, 매번 다름):\n{rng_no_seed.random(size=(2, 3))}")

# 다른 시드로 생성하면 다른 결과가 나옴
rng_other_seed = np.random.default_rng(seed=100)
print(f"\n다른 시드로 생성한 배열 (rng_other_seed):\n{rng_other_seed.random(size=(2, 3))}")
```

#### 1.2.3. 주요 난수 생성 메서드

`Generator` 객체는 다양한 분포의 난수를 생성하는 메서드를 제공합니다.

##### 1.2.3.1. `rng.random()`

`rng.random(size=None)` 메서드는 `0.0` (포함) 이상 `1.0` (미만) 사이의 **균일 분포(uniform distribution)**에서 난수를 생성합니다. 이는 `np.random.rand()`나 `np.random.random()`의 최신 `Generator` API 버전입니다.

**주요 특징:**
*   **분포**: `[0.0, 1.0)` 범위의 균일 분포에서 난수를 추출합니다. 모든 값이 동일한 확률로 나타납니다.
*   **인수**: `size` 키워드 인자를 통해 생성할 배열의 형태를 튜플로 지정합니다. `None`이면 단일 스칼라 값을 반환합니다.
*   **반환 값**: `size`가 `None`이면 단일 부동 소수점(float) 값을, 그렇지 않으면 지정된 형태의 `numpy.ndarray`를 반환합니다.
*   **Generator API**: `np.random.Generator` 객체의 메서드이므로, 독립적인 난수 스트림을 관리하고 PCG64 비트 생성기의 이점을 활용합니다.

**`np.random.rand()` 및 `np.random.random()`과의 관계:**
`rng.random()`은 기존의 `np.random.rand()` 및 `np.random.random()`과 동일한 기능을 수행하지만, `Generator` 객체에 속해 있어 더 나은 재현성 관리와 통계적 품질을 제공합니다. 따라서 새로운 프로젝트에서는 이 메서드를 사용하는 것이 권장됩니다.

**예시:**

```python
import numpy as np
rng = np.random.default_rng(seed=42)

# 단일 난수 생성 (size=None)
random_float_single = rng.random()
print(f"단일 균일 분포 실수 (rng.random): {random_float_single}")

# 0.0에서 1.0 사이의 균일 분포 실수 배열 생성 (1차원)
random_floats_1d = rng.random(size=5)
print(f"\n1차원 균일 분포 실수 (rng.random): {random_floats_1d}")

# 2x3 형태의 균일 분포 실수 행렬 생성
random_floats_2d = rng.random(size=(2, 3))
print(f"\n2x3 균일 분포 실수 (rng.random):\n{random_floats_2d}")

# 2x2x2 형태의 3차원 배열 생성
random_floats_3d = rng.random(size=(2, 2, 2))
print(f"\n2x2x2 균일 분포 실수 (rng.random):\n{random_floats_3d}")
```


##### 1.2.3.2. `rng.integers()`

`rng.integers(low, high=None, size=None, dtype=np.int64, endpoint=False)` 메서드는 지정된 범위 내에서 **정수 난수**를 생성합니다. 이 함수는 `np.random.randint()`의 최신 `Generator` API 버전으로, 특히 `endpoint` 인자를 통해 상한 값의 포함 여부를 명시적으로 제어할 수 있다는 장점이 있습니다.

**주요 특징:**
*   **범위**: 
    *   `low`: 생성될 가장 작은 정수 (포함).
    *   `high`: 생성될 가장 큰 정수. `endpoint=False` (기본값)이면 `high`는 미포함, `endpoint=True`이면 `high`는 포함됩니다.
    *   `high`가 `None`이면, `[0, low)` 또는 `[0, low]` (endpoint에 따라) 범위에서 난수를 생성합니다.
*   **인수**: 
    *   `size`: 선택 사항. 생성할 배열의 형태를 지정하는 튜플 또는 정수. `None`이면 단일 정수를 반환합니다.
    *   `dtype`: 선택 사항. 반환될 정수의 데이터 타입. 기본값은 `np.int64`입니다.
    *   `endpoint`: 불리언. `True`로 설정하면 `high` 값을 포함합니다. `False` (기본값)이면 `high` 값을 포함하지 않습니다.
*   **반환 값**: `size`가 `None`이면 단일 정수를, 그렇지 않으면 지정된 형태의 `numpy.ndarray`를 반환합니다.
*   **분포**: 이산 균일 분포(discrete uniform distribution)에서 난수를 추출합니다.
*   **Generator API**: `np.random.Generator` 객체의 메서드이므로, 독립적인 난수 스트림을 관리하고 PCG64 비트 생성기의 이점을 활용합니다.

**`np.random.randint()`와의 차이점:**
`rng.integers()`의 가장 큰 개선점은 `endpoint` 인자입니다. `np.random.randint()`는 항상 `high` 값을 미포함으로 처리하여 사용자가 혼동할 수 있었지만, `rng.integers()`는 이를 명시적으로 제어할 수 있게 하여 코드의 가독성과 정확성을 높였습니다.

**예시:**

```python
import numpy as np
rng = np.random.default_rng(seed=42)

# 1부터 10까지의 정수 난수 1개 생성 (10 미포함, 즉 1~9)
random_int_single = rng.integers(1, 10)
print(f"단일 정수 난수 (1~9): {random_int_single}")

# 1부터 10까지의 정수 난수 5개 생성 (10 미포함, 즉 1~9)
random_integers_1d = rng.integers(1, 10, size=5)
print(f"\n1차원 정수 난수 (1~9): {random_integers_1d}")

# 0부터 100까지의 정수 난수 3x3 행렬 생성 (100 포함, endpoint=True)
random_integers_matrix = rng.integers(0, 100, size=(3, 3), endpoint=True)
print(f"\n3x3 정수 난수 행렬 (0~100 포함):\n{random_integers_matrix}")

# high 인자 생략 시 (0부터 low-1까지, endpoint=False)
random_integers_no_high = rng.integers(5, size=3) # 0부터 4까지의 정수 3개
print(f"\nhigh 생략 시 (0~4): {random_integers_no_high}")

# high 인자 생략 시 (0부터 low까지, endpoint=True)
random_integers_no_high_inclusive = rng.integers(5, size=3, endpoint=True) # 0부터 5까지의 정수 3개
print(f"\nhigh 생략 시 (0~5 포함): {random_integers_no_high_inclusive}")
```


##### 1.2.3.3. `rng.standard_normal()`

`rng.standard_normal(size=None)` 메서드는 **표준 정규 분포(standard normal distribution)**에서 난수를 생성합니다. 표준 정규 분포는 평균(mean)이 0이고 표준편차(standard deviation)가 1인 정규 분포를 의미합니다. 이 함수는 기존의 `np.random.randn()`에 해당하는 `Generator` API의 메서드입니다.

**주요 특징:**
*   **분포**: 평균 0, 표준편차 1인 표준 정규 분포(가우시안 분포)에서 난수를 추출합니다. 종 모양의 대칭적인 분포를 가집니다.
*   **인수**: `size` 키워드 인자를 통해 생성할 배열의 형태를 튜플로 지정합니다. `None`이면 단일 스칼라 값을 반환합니다.
*   **반환 값**: `size`가 `None`이면 단일 부동 소수점(float) 값을, 그렇지 않으면 지정된 형태의 `numpy.ndarray`를 반환합니다.
*   **Generator API**: `np.random.Generator` 객체의 메서드이므로, 독립적인 난수 스트림을 관리하고 PCG64 비트 생성기의 이점을 활용합니다.

**`np.random.randn()`과의 관계:**
`rng.standard_normal()`은 `np.random.randn()`과 동일한 기능을 수행하지만, `Generator` 객체에 속해 있어 더 나은 재현성 관리와 통계적 품질을 제공합니다. 따라서 새로운 프로젝트에서는 이 메서드를 사용하는 것이 권장됩니다.

**`rng.normal()`과의 차이점:**
`rng.normal(loc, scale, size)` 메서드는 사용자가 직접 평균(`loc`)과 표준편차(`scale`)를 지정하여 정규 분포 난수를 생성할 수 있습니다. 반면 `rng.standard_normal()`은 평균 0, 표준편차 1로 고정된 표준 정규 분포만을 생성합니다. 특정 평균과 표준편차를 가진 정규 분포가 필요하다면 `rng.normal()`을 사용해야 합니다.

**예시:**

```python
import numpy as np
rng = np.random.default_rng(seed=42)

# 단일 표준 정규 분포 난수 생성
standard_normal_single = rng.standard_normal()
print(f"단일 표준 정규 분포 난수 (rng.standard_normal): {standard_normal_single}")

# 표준 정규 분포에서 5개의 난수 생성 (1차원)
standard_normal_1d = rng.standard_normal(size=5)
print(f"\n1차원 표준 정규 분포 난수 (rng.standard_normal): {standard_normal_1d}")

# 표준 정규 분포에서 2x3 크기의 배열 생성
normal_dist_rng_2d = rng.standard_normal(size=(2, 3))
print(f"\n2x3 표준 정규 분포 (rng.standard_normal):\n{normal_dist_rng_2d}")

# 표준 정규 분포에서 2x2x2 크기의 3차원 배열 생성
normal_dist_rng_3d = rng.standard_normal(size=(2, 2, 2))
print(f"\n2x2x2 표준 정규 분포 (rng.standard_normal):\n{normal_dist_rng_3d}")
```


##### 1.2.3.4. `rng.uniform()`

`rng.uniform(low=0.0, high=1.0, size=None)` 메서드는 지정된 범위 `[low, high)`에서 **균일 분포(uniform distribution)**의 실수를 생성합니다. 이 함수는 `np.random.default_rng().random()`의 일반화된 형태로, 사용자가 직접 난수가 생성될 구간의 최소값과 최대값을 설정할 수 있습니다.

**주요 특징:**
*   **분포**: `[low, high)` 범위의 균일 분포에서 난수를 추출합니다. 모든 값이 동일한 확률로 나타납니다.
*   **인수**: 
    *   `low`: 선택 사항. 생성될 난수의 하한(inclusive). 기본값은 `0.0`입니다.
    *   `high`: 선택 사항. 생성될 난수의 상한(exclusive). 기본값은 `1.0`입니다.
    *   `size`: 선택 사항. 생성할 배열의 형태를 지정하는 튜플 또는 정수. `None`이면 단일 스칼라 값을 반환합니다.
*   **반환 값**: `size`가 `None`이면 단일 부동 소수점(float) 값을, 그렇지 않으면 지정된 형태의 `numpy.ndarray`를 반환합니다.
*   **Generator API**: `np.random.Generator` 객체의 메서드이므로, 독립적인 난수 스트림을 관리하고 PCG64 비트 생성기의 이점을 활용합니다.

**`rng.random()`과의 관계:**
`rng.random()`은 `rng.uniform(low=0.0, high=1.0, size=None)`과 동일한 기능을 수행하는 특수한 경우입니다. 즉, `rng.uniform()`은 `rng.random()`보다 더 유연하게 난수 생성 범위를 제어할 수 있습니다.

**활용:**
*   특정 범위 내에서 무작위 값을 샘플링해야 할 때 (예: 시뮬레이션, 하이퍼파라미터 튜닝).
*   데이터 증강 시 이미지의 밝기, 대비 등을 무작위로 조절할 때.

**예시:**

```python
import numpy as np
rng = np.random.default_rng(seed=42)

# 0.0에서 1.0 사이의 균일 분포 실수 배열 생성 (기본값)
uniform_default = rng.uniform(size=5)
print(f"기본 균일 분포 실수 (0.0~1.0): {uniform_default}")

# -1부터 1까지의 균일 분포 실수 배열 생성
uniform_dist_custom = rng.uniform(low=-1, high=1, size=5)
print(f"\n균일 분포 실수 (-1~1): {uniform_dist_custom}")

# 10부터 20까지의 균일 분포 실수 2x2 행렬 생성
uniform_dist_matrix = rng.uniform(low=10, high=20, size=(2, 2))
print(f"\n균일 분포 실수 (10~20, 2x2 행렬):\n{uniform_dist_matrix}")

# 단일 실수 생성
uniform_single = rng.uniform(low=100, high=200)
print(f"\n단일 균일 분포 실수 (100~200): {uniform_single}")
```


##### 1.2.3.5. `rng.normal()`

`rng.normal(loc=0.0, scale=1.0, size=None)` 메서드는 **정규 분포(Normal Distribution)**에서 난수를 생성합니다. 이 함수는 사용자가 직접 평균(`loc`)과 표준편차(`scale`)를 지정할 수 있어, 다양한 형태의 정규 분포를 모델링할 수 있습니다. 이는 `np.random.default_rng().standard_normal()`의 일반화된 형태입니다.

**주요 특징:**
*   **분포**: 지정된 `loc` (평균)과 `scale` (표준편차)를 가지는 정규 분포에서 난수를 추출합니다. 종 모양의 대칭적인 분포를 가집니다.
*   **인수**: 
    *   `loc`: 선택 사항. 분포의 평균. 기본값은 `0.0`입니다.
    *   `scale`: 선택 사항. 분포의 표준편차. 기본값은 `1.0`입니다.
    *   `size`: 선택 사항. 생성할 배열의 형태를 지정하는 튜플 또는 정수. `None`이면 단일 스칼라 값을 반환합니다.
*   **반환 값**: `size`가 `None`이면 단일 부동 소수점(float) 값을, 그렇지 않으면 지정된 형태의 `numpy.ndarray`를 반환합니다.
*   **Generator API**: `np.random.Generator` 객체의 메서드이므로, 독립적인 난수 스트림을 관리하고 PCG64 비트 생성기의 이점을 활용합니다.

**`rng.standard_normal()`과의 관계:**
`rng.normal()`은 `rng.standard_normal()`보다 더 일반적인 함수입니다. `rng.standard_normal()`은 `rng.normal(loc=0.0, scale=1.0, size=None)`과 동일한 기능을 수행합니다. 따라서 표준 정규 분포가 아닌 다른 평균이나 표준편차를 가진 정규 분포가 필요할 때는 `rng.normal()`을 사용해야 합니다.

**활용:**
*   실제 데이터의 정규 분포 특성을 모방한 데이터 생성.
*   특정 평균과 분산을 가진 노이즈를 데이터에 추가.
*   통계적 모델링 및 시뮬레이션.
*   신경망 가중치 초기화 (예: He 초기화, Glorot 초기화).

**예시:**

```python
import numpy as np
rng = np.random.default_rng(seed=42)

# 평균 5, 표준편차 1.5인 정규 분포 난수 5개 생성
normal_custom_dist_1d = rng.normal(loc=5, scale=1.5, size=5)
print(f"정규 분포 난수 (평균 5, 표준편차 1.5): {normal_custom_dist_1d}")

# 평균 0, 표준편차 0.1인 정규 분포 난수 3x3 배열 생성 (작은 노이즈)
normal_small_noise = rng.normal(loc=0, scale=0.1, size=(3, 3))
print(f"\n정규 분포 난수 (작은 노이즈, 평균 0, 표준편차 0.1):\n{normal_small_noise}")

# 평균 100, 표준편차 10인 정규 분포 난수 2x2 배열 생성
normal_large_scale = rng.normal(loc=100, scale=10, size=(2, 2))
print(f"\n정규 분포 난수 (평균 100, 표준편차 10, 2x2 행렬):\n{normal_large_scale}")
```

### 1.3. 재현성 모범 사례 (Reproducibility Best Practices)

머신러닝 및 딥러닝 워크플로우에서 난수 생성의 **재현성(Reproducibility)**은 실험 결과를 신뢰하고 공유하며, 모델의 성능을 일관되게 평가하는 데 매우 중요합니다. 단순히 시드를 한 번 설정하는 것을 넘어, 전체 워크플로우에서 난수 생성을 일관되게 관리하는 모범 사례를 따르는 것이 중요합니다.

**1. 모든 난수 생성 지점에 시드 설정**: 
코드 내에서 난수를 생성하는 모든 부분(데이터 분할, 모델 초기화, 데이터 증강 등)에 명시적으로 시드를 설정해야 합니다. NumPy뿐만 아니라 Python의 기본 `random` 모듈, 그리고 PyTorch, TensorFlow와 같은 딥러닝 프레임워크에서도 각자의 시드 설정 메커니즘을 사용해야 합니다. 이는 각 라이브러리가 독립적인 의사 난수 생성기를 가질 수 있기 때문입니다.

```python
import numpy as np
import random
import torch # PyTorch 예시 (설치 필요)

def set_all_seeds(seed):
    """모든 주요 라이브러리의 난수 시드를 설정합니다."""
    np.random.seed(seed)       # NumPy 시드
    random.seed(seed)          # Python 기본 random 모듈 시드
    
    # PyTorch 시드 설정
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 모든 GPU에 시드 설정
    
    # 딥러닝 프레임워크의 결정론적 동작 설정 (성능 저하 가능성 있음)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

my_seed = 42
set_all_seeds(my_seed)

print(f"NumPy 난수: {np.random.rand(3)}")
print(f"Python random 난수: {random.random()}")
# PyTorch가 설치되어 있다면 다음 줄도 실행됩니다.
# print(f"PyTorch 난수: {torch.rand(3)}")
```

**2. `np.random.Generator` 사용**: 
전역 상태를 공유하는 `np.random.seed()` 대신, `np.random.default_rng(seed)`를 사용하여 독립적인 `Generator` 객체를 생성하고 관리하는 것이 좋습니다. 이는 복잡한 코드베이스에서 난수 스트림 간의 간섭을 줄여주고, 각 난수 생성 작업의 독립성을 보장합니다.

```python
import numpy as np

# 데이터 분할을 위한 독립적인 생성기
rng_data_split = np.random.default_rng(seed=100)
# 모델 초기화를 위한 독립적인 생성기
rng_model_init = np.random.default_rng(seed=200)
# 데이터 증강을 위한 독립적인 생성기
rng_augmentation = np.random.default_rng(seed=300)

print(f"데이터 분할 난수: {rng_data_split.random(3)}")
print(f"모델 초기화 난수: {rng_model_init.random(3)}")
print(f"데이터 증강 난수: {rng_augmentation.random(3)}")

# 각 생성기는 독립적으로 난수를 생성하며, 서로의 시퀀스에 영향을 주지 않습니다.
```

**3. 환경 관리 및 버전 제어**: 
*   **환경 관리**: `conda`, `pipenv`, `virtualenv`와 같은 도구를 사용하여 프로젝트의 모든 종속성(라이브러리 버전)을 명확하게 정의하고 고정해야 합니다. 이는 다른 환경에서 코드를 실행할 때 동일한 라이브러리 버전이 사용되도록 보장합니다.
*   **버전 제어**: Git과 같은 버전 제어 시스템을 사용하여 코드 변경 사항을 추적하고 관리합니다. 이는 특정 실험에 사용된 정확한 코드 버전을 쉽게 되돌리거나 확인할 수 있게 합니다.

**4. 데이터 버전 관리**: 
모델 학습에 사용된 데이터셋도 버전 관리가 필요합니다. DVC(Data Version Control)와 같은 도구를 사용하여 데이터셋의 변경 사항을 추적하고, 특정 실험에 사용된 데이터셋 버전을 명확히 할 수 있습니다.

**5. 실험 관리 시스템 활용**: 
MLflow, Weights & Biases, Comet ML, TensorBoard와 같은 실험 관리 시스템을 사용하여 실험의 모든 메타데이터(사용된 시드 값, 하이퍼파라미터, 모델 아키텍처, 성능 지표, 학습 로그 등)를 기록하고 추적하는 것이 좋습니다. 이는 대규모 실험에서 재현성을 보장하고, 효율적인 실험 비교 및 분석을 가능하게 합니다.

**6. 결정론적(Deterministic) 연산 보장**: 
특히 딥러닝에서는 GPU 연산이나 특정 알고리즘이 기본적으로 비결정론적일 수 있습니다. PyTorch의 `torch.backends.cudnn.deterministic = True`와 같은 설정을 통해 이러한 연산의 결정론적 동작을 강제할 수 있습니다. (단, 이는 성능 저하를 초래할 수 있습니다.)

이러한 모범 사례를 따르면, 당신의 데이터 과학 및 머신러닝 실험은 더욱 견고하고 신뢰할 수 있으며, 미래에도 쉽게 재현될 수 있을 것입니다.


## 2. 샘플링 및 순열 (Sampling & Permutation)

데이터를 무작위로 섞거나(순열) 특정 조건에 따라 추출하는(샘플링) 작업은 데이터 과학 및 머신러닝에서 매우 중요합니다. 이는 데이터의 통계적 특성을 유지하면서 다양한 분석 및 모델 학습 시나리오를 가능하게 합니다. 예를 들어, **교차 검증(Cross-validation)**, **부트스트래핑(Bootstrapping)**, **데이터 증강(Data Augmentation)**, 그리고 **훈련/검증/테스트 세트 분할** 등 다양한 통계적 검증 및 모델 학습 과정에서 필수적으로 사용됩니다. NumPy는 이러한 작업을 효율적으로 수행할 수 있는 함수들을 제공합니다.

**샘플링(Sampling):**
전체 데이터셋(모집단)에서 일부 데이터를 무작위로 선택하여 표본(sample)을 추출하는 과정입니다. 샘플링은 크게 두 가지 방식으로 나뉩니다.
*   **복원 추출(Sampling with Replacement)**: 한 번 선택된 데이터가 다시 선택될 수 있습니다. 부트스트래핑에 주로 사용됩니다.
*   **비복원 추출(Sampling without Replacement)**: 한 번 선택된 데이터는 다시 선택될 수 없습니다. 훈련/테스트 세트 분할에 주로 사용됩니다.

**순열(Permutation):**
데이터의 순서를 무작위로 재배열하는 과정입니다. 이는 데이터의 순서가 모델 학습에 영향을 미치지 않도록 하거나, 특정 순서 의존성을 제거할 때 사용됩니다.

NumPy의 `np.random` 모듈은 이러한 샘플링 및 순열 작업을 위한 다양한 함수를 제공합니다. 이 섹션에서는 전통적인 API(`np.random.shuffle()`, `np.random.permutation()`, `np.random.choice()`)를 다루며, 다음 섹션에서는 최신 `Generator` API를 통한 샘플링 및 순열 방법을 설명합니다.

### 2.1. `np.random.shuffle()`

`np.random.shuffle(x)` 함수는 배열 `x`의 순서를 **제자리에서(in-place)** 무작위로 섞습니다. 즉, 원본 배열 자체가 변경되며, 함수는 아무것도 반환하지 않습니다(`None`을 반환). 다차원 배열의 경우, 첫 번째 축(axis=0, 즉 행)을 따라서만 섞습니다.

**주요 특징:**
*   **In-place**: 원본 배열의 내용을 직접 수정합니다. 따라서 함수 호출 후 원본 변수를 다시 할당할 필요가 없습니다.
*   **반환값 없음**: 함수는 `None`을 반환합니다. `y = np.random.shuffle(x)`와 같이 사용하면 `y`는 `None`이 됩니다.
*   **첫 번째 축 섞기**: 1차원 배열은 모든 요소가 섞입니다. 2차원 배열의 경우 행(row) 단위로 섞이며, 각 행 내부의 요소 순서는 변경되지 않습니다. 즉, 행 전체가 무작위 위치로 이동합니다.
*   **레거시 API**: 이 함수는 `np.random` 모듈의 전통적인(legacy) API에 속합니다. 최신 코드에서는 `np.random.default_rng().shuffle()` 사용이 권장됩니다.

**활용 분야:**
*   **훈련 데이터 무작위화**: 머신러닝 모델 학습 전에 훈련 데이터셋의 순서를 무작위로 섞어 데이터의 순서에 의한 편향을 방지하고, 모델이 데이터의 특정 순서에 과적합되는 것을 막습니다.
*   **카드 덱 섞기**: 게임 시뮬레이션 등에서 카드 덱의 순서를 무작위로 섞을 때 사용될 수 있습니다.

**`np.random.permutation()`과의 차이점:**
`np.random.shuffle()`는 원본 배열을 제자리에서 수정하는 반면, `np.random.permutation()`은 원본 배열의 복사본을 섞어서 반환하므로 원본 배열은 변경되지 않습니다. 원본 데이터를 보존해야 할 때는 `permutation`을 사용하는 것이 좋습니다.

**예시:**

```python
import numpy as np

# 1차원 배열 섞기
arr_1d_shuffle = np.arange(10)
print(f"원본 1차원 배열: {arr_1d_shuffle}")
np.random.shuffle(arr_1d_shuffle)
print(f"shuffle 후 1차원 배열 (원본 변경됨): {arr_1d_shuffle}")

# 2차원 배열 섞기 (행 단위로 섞임)
arr_2d_shuffle = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
print(f"\n원본 2차원 배열:\n{arr_2d_shuffle}")
np.random.shuffle(arr_2d_shuffle)
print(f"shuffle 후 2차원 배열 (행 섞임, 원본 변경됨):\n{arr_2d_shuffle}")

# shuffle은 None을 반환하므로 변수에 할당해도 의미 없음
result = np.random.shuffle(arr_1d_shuffle)
print(f"\nshuffle 함수의 반환값: {result}") # None
```

### 2.2. `np.random.permutation()`

`np.random.permutation(x)` 함수는 배열 `x`의 순서를 무작위로 섞은 **새로운 배열(복사본)**을 반환합니다. 원본 배열은 변경되지 않습니다. `x`가 정수(`n`)로 주어지면 `np.arange(n)`의 순열을 반환합니다.

**주요 특징:**
*   **복사본 반환**: `np.random.shuffle()`와 달리, `permutation()`은 원본 배열을 수정하지 않고, 무작위로 섞인 새로운 배열을 반환합니다. 원본 데이터를 보존해야 할 때 유용합니다.
*   **정수 인자 지원**: `x`에 정수 `n`을 전달하면, `0`부터 `n-1`까지의 정수 배열(`np.arange(n)`)의 무작위 순열을 생성합니다.
*   **다차원 배열**: `shuffle`과 마찬가지로, 다차원 배열의 경우 첫 번째 축(axis=0, 즉 행)을 따라서만 섞습니다. 각 행 내부의 요소 순서는 유지됩니다.
*   **레거시 API**: 이 함수는 `np.random` 모듈의 전통적인(legacy) API에 속합니다. 최신 코드에서는 `np.random.default_rng().permutation()` 사용이 권장됩니다.

**활용 분야:**
*   **훈련/테스트 데이터 분할**: 데이터셋의 인덱스를 무작위로 섞은 후, 이를 사용하여 훈련 세트와 테스트 세트를 나눌 때 유용합니다.
*   **데이터 증강**: 이미지나 텍스트 데이터의 순서를 무작위로 변경하여 새로운 훈련 샘플을 생성할 때 사용될 수 있습니다.
*   **몬테카를로 시뮬레이션**: 특정 순서가 중요한 시뮬레이션에서 무작위 순서를 생성할 때.

**`np.random.shuffle()`와의 차이점:**
가장 큰 차이점은 `shuffle()`은 원본 배열을 제자리에서 수정하고 `None`을 반환하는 반면, `permutation()`은 원본을 변경하지 않고 새로운 배열을 반환한다는 점입니다.

**예시:**

```python
import numpy as np

# 정수 인자를 사용하여 0부터 9까지의 순열 생성
perm_int = np.random.permutation(10)
print(f"permutation 결과 (정수 인자, 0~9): {perm_int}")

# 1차원 배열의 순열 생성
arr_1d_perm = np.array([10, 20, 30, 40, 50])
print(f"\n원본 1차원 배열: {arr_1d_perm}")
perm_array_1d = np.random.permutation(arr_1d_perm)
print(f"permutation 결과 (배열 인자): {perm_array_1d}")
print(f"원본 1차원 배열 (변경 없음): {arr_1d_perm}") # 원본은 변경되지 않음

# 2차원 배열의 순열 생성 (행 단위로 섞인 복사본)
arr_2d_perm = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
print(f"\n원본 2차원 배열:\n{arr_2d_perm}")
perm_array_2d = np.random.permutation(arr_2d_perm)
print(f"permutation 결과 (2차원 배열, 행 섞임):\n{perm_array_2d}")
print(f"원본 2차원 배열 (변경 없음):\n{arr_2d_perm}") # 원본은 변경되지 않음
```


### 2.3. `np.random.choice()`

`np.random.choice(a, size=None, replace=True, p=None)` 함수는 주어진 배열 `a`에서 `size`만큼 무작위로 샘플을 추출합니다. 이 함수는 매우 유연하며 다양한 샘플링 시나리오에 활용될 수 있습니다.

**주요 파라미터:**
*   `a`: 샘플링할 1차원 배열 또는 정수. 정수 `n`이 주어지면 `np.arange(n)`에서 샘플링합니다.
*   `size`: 선택 사항. 추출할 샘플의 개수 또는 형태. 단일 샘플을 추출하려면 `None` (기본값) 또는 정수 1을 사용합니다. 여러 샘플을 배열 형태로 추출하려면 튜플을 사용하여 형태를 지정합니다 (예: `size=(2, 3)`).
*   `replace`: 불리언. `True` (복원 추출, 중복 허용) 또는 `False` (비복원 추출, 중복 없음). 기본값은 `True`입니다. 비복원 추출 시 `size`는 `a`의 크기보다 클 수 없습니다.
*   `p`: 선택 사항. `a`의 각 요소가 선택될 확률을 지정하는 1차원 배열. `a`와 같은 크기여야 하며, 모든 확률의 합은 1이어야 합니다.

**주요 특징:**
*   **유연한 샘플링**: 복원 추출, 비복원 추출, 가중치 기반 샘플링 등 다양한 샘플링 방식을 지원합니다.
*   **단일 요소 또는 배열 반환**: `size` 인자에 따라 단일 스칼라 값 또는 배열을 반환합니다.
*   **레거시 API**: 이 함수는 `np.random` 모듈의 전통적인(legacy) API에 속합니다. 최신 코드에서는 `np.random.default_rng().choice()` 사용이 권장됩니다.

**활용 분야:**
*   **훈련/테스트 데이터 분할**: 데이터셋에서 무작위로 샘플을 추출하여 훈련 세트와 테스트 세트를 나눌 때 사용됩니다.
*   **부트스트래핑(Bootstrapping)**: 통계적 추정에서 표본을 반복적으로 추출하여 통계량의 분포를 추정할 때 (복원 추출, `replace=True`).
*   **가중치 기반 샘플링**: 특정 항목이 다른 항목보다 더 자주 선택되도록 할 때 (예: 불균형 데이터셋에서 소수 클래스 오버샘플링).
*   **몬테카를로 시뮬레이션**: 확률적 모델에서 무작위 샘플을 생성하여 시스템의 동작을 시뮬레이션할 때.

**예시:**

```python
import numpy as np

# 0부터 9까지의 정수 중에서 5개를 복원 추출 (기본값: replace=True)
choice_basic = np.random.choice(10, size=5)
print(f"기본 복원 추출 (0-9에서 5개): {choice_basic}")

# 0부터 4까지의 정수 중에서 5개를 비복원 추출 (순열과 유사)
choice_no_replace = np.random.choice(5, size=5, replace=False)
print(f"비복원 추출 (0-4에서 5개, 순열과 유사): {choice_no_replace}")

# 특정 데이터에서 3개를 복원 추출
data_items = ['apple', 'banana', 'cherry', 'date', 'elderberry']
choice_from_list = np.random.choice(data_items, size=3)
print(f"\n리스트에서 복원 추출 (3개): {choice_from_list}")

# 확률을 지정하여 샘플 추출 (가중치 샘플링)
data_weighted = ['low', 'medium', 'high']
probabilities = [0.7, 0.2, 0.1] # 'low'가 선택될 확률이 가장 높음
weighted_choice = np.random.choice(data_weighted, size=10, p=probabilities)
print(f"\n가중치 기반 복원 추출 (10개): {weighted_choice}")

# 2차원 배열의 행을 무작위로 샘플링 (인덱스를 샘플링 후 원본 배열에서 선택)
data_2d = np.array([[1, 10, 100],
                    [2, 20, 200],
                    [3, 30, 300],
                    [4, 40, 400],
                    [5, 50, 500]])
# 행 인덱스를 비복원 샘플링
sampled_indices = np.random.choice(data_2d.shape[0], size=2, replace=False)
sampled_rows = data_2d[sampled_indices]
print(f"\n2차원 배열에서 행 샘플링 (2개 행):\n{sampled_rows}")

# 2차원 배열의 행을 가중치 기반으로 샘플링 (예: 첫 번째 행이 더 자주 선택되도록)
row_probabilities = [0.5, 0.2, 0.1, 0.1, 0.1]
weighted_sampled_indices = np.random.choice(data_2d.shape[0], size=5, replace=True, p=row_probabilities)
weighted_sampled_rows = data_2d[weighted_sampled_indices]
print(f"\n2차원 배열에서 가중치 기반 행 샘플링 (5개 행):\n{weighted_sampled_rows}")
```

## 3. 데이터 샘플링 및 순열 (Generator API)

이 섹션에서는 NumPy의 최신 **`np.random.Generator` 객체**를 사용하여 데이터를 샘플링하고 순열을 생성하는 방법을 자세히 다룹니다. `Generator` 객체는 전통적인 `np.random` 모듈의 함수들(`np.random.shuffle()`, `np.random.permutation()`, `np.random.choice()`)과 유사한 기능을 제공하지만, 난수 생성의 재현성을 더욱 효과적으로 관리하고 통계적으로 우수한 난수를 생성한다는 장점이 있습니다.

**`Generator` API를 통한 샘플링 및 순열의 이점:**
*   **향상된 재현성**: 각 `Generator` 객체는 독립적인 난수 스트림을 가지므로, 코드의 다른 부분이나 다른 모듈의 난수 생성에 영향을 주지 않습니다. 이는 복잡한 데이터 파이프라인이나 모델 학습 과정에서 일관된 결과를 얻는 데 필수적입니다.
*   **더 나은 통계적 품질**: `Generator`는 기본적으로 PCG64와 같은 최신 비트 생성기를 사용하며, 이는 기존의 Mersenne Twister보다 통계적 특성이 우수합니다.
*   **명확한 코드**: 모든 난수 관련 작업이 `Generator` 객체의 메서드를 통해 이루어지므로, 코드가 더 읽기 쉽고 유지보수하기 용이합니다.

**기본 사용 패턴:**
데이터 샘플링 및 순열을 위해 `Generator` 객체를 사용하는 일반적인 패턴은 다음과 같습니다.
1.  `np.random.default_rng(seed)`를 사용하여 `Generator` 객체를 생성합니다. 재현성을 위해 시드를 설정하는 것이 좋습니다.
2.  생성된 `Generator` 객체의 메서드(`rng.shuffle()`, `rng.permutation()`, `rng.choice()`)를 호출하여 원하는 샘플링 또는 순열 작업을 수행합니다.

이러한 메서드들은 전통적인 `np.random` 함수들과 유사한 인자와 동작 방식을 가지지만, `Generator` 객체에 종속되어 있다는 점이 다릅니다.

#### 3.1.1. `rng.shuffle()`

`rng.shuffle(x, axis=0)` 메서드는 배열 `x`의 순서를 **제자리에서(in-place)** 무작위로 섞습니다. 이 작업은 원본 배열 자체를 직접 수정하며, 아무것도 반환하지 않습니다(`None`). 이 방식은 메모리 사용 측면에서 효율적이지만, 원본 데이터가 손실된다는 점에 유의해야 합니다.

**주요 특징 및 매개변수:**

*   **`x` (ndarray)**: 섞을 배열입니다.
*   **`axis` (int, 선택 사항)**: 섞기를 수행할 축을 지정합니다. 기본값은 `0`이며, 이는 첫 번째 축(일반적으로 행)을 따라 섞는 것을 의미합니다. 예를 들어, 2차원 배열에서 `axis=0`은 행의 순서를 섞고, `axis=1`은 각 행 내부의 열 순서를 섞습니다.
*   **In-place (제자리 수정)**: 이 메서드는 새로운 배열을 생성하여 반환하는 대신, 입력된 배열 `x`의 내용을 직접 변경합니다. 따라서 `shuffled_array = rng.shuffle(my_array)`와 같이 코드를 작성하면 `shuffled_array`에는 `None`이 할당됩니다.
*   **반환값 없음**: 함수는 `None`을 반환합니다.

**동작 방식:**

*   **1차원 배열**: 배열의 모든 요소가 무작위로 섞입니다.
*   **다차원 배열**: 지정된 `axis`를 따라 슬라이스를 섞습니다. 예를 들어, 2차원 배열에서 `axis=0`으로 설정하면 행 전체가 하나의 단위로 취급되어 행의 순서가 무작위로 바뀝니다. 각 행 내부의 요소 순서는 그대로 유지됩니다. `axis=1`로 설정하면 각 행 내에서 열의 순서가 독립적으로 섞입니다.

**활용 사례:**

*   **머신러닝 데이터셋 섞기**: 모델 학습 전에 훈련 데이터와 레이블을 함께 섞어 데이터의 순서에 따른 편향을 방지하고, 모델이 데이터의 특정 순서에 과적합되는 것을 막습니다. 이 경우, 데이터와 레이블을 동일한 순서로 섞는 것이 매우 중요합니다.
*   **교차 검증(Cross-validation)**: 데이터를 여러 폴드(fold)로 나누기 전에 전체 데이터셋을 무작위로 섞어 각 폴드가 데이터 전체를 잘 대표하도록 합니다.
*   **시뮬레이션**: 카드 덱을 섞는 것과 같이 순서가 중요한 시뮬레이션에서 무작위성을 부여할 때 사용됩니다.

**`rng.permutation()`과의 차이점:**

`rng.shuffle()`은 원본 배열을 직접 수정하는 반면, `rng.permutation()`은 원본 배열을 변경하지 않고 섞인 **복사본**을 반환합니다. 원본 데이터를 보존해야 하는 경우에는 `rng.permutation()`을 사용해야 합니다.

**예시 코드:**

```python
import numpy as np
rng = np.random.default_rng(seed=42)

# 1. 1차원 배열 섞기
arr_1d = np.arange(10)
print(f"원본 1차원 배열: {arr_1d}")
rng.shuffle(arr_1d)  # 제자리에서 섞임
print(f"shuffle 후 1차원 배열: {arr_1d}")

# 2. 2차원 배열 섞기 (행 단위, axis=0)
arr_2d = np.arange(12).reshape((4, 3))
print(f"\n원본 2차원 배열:\n{arr_2d}")
rng.shuffle(arr_2d, axis=0)  # 행 순서를 섞음
print(f"shuffle 후 2차원 배열 (행 섞임):\n{arr_2d}")

# 3. 2차원 배열 섞기 (열 단위, axis=1)
arr_2d_cols = np.arange(12).reshape((4, 3))
print(f"\n원본 2차원 배열 (열 섞기 전):\n{arr_2d_cols}")
rng.shuffle(arr_2d_cols, axis=1) # 각 행 내에서 열 순서를 섞음
print(f"shuffle 후 2차원 배열 (열 섞임):\n{arr_2d_cols}")


# 4. 머신러닝 데이터와 레이블 함께 섞기
# 데이터와 레이블을 함께 섞기 위해서는 인덱스를 섞은 후, 그 인덱스를 사용해 재정렬해야 합니다.
# shuffle은 다중 배열을 동기화하여 섞는 기능을 직접 제공하지 않기 때문입니다.
# 이 경우에는 permutation을 사용하는 것이 더 직관적입니다.
# 하지만 굳이 shuffle을 사용한다면 다음과 같이 할 수 있습니다.

features = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
labels = np.array(['A', 'B', 'C', 'D'])

# 인덱스를 생성하고 섞습니다.
indices = np.arange(features.shape[0])
print(f"\n원본 인덱스: {indices}")
rng.shuffle(indices)
print(f"섞인 인덱스: {indices}")

# 섞인 인덱스를 사용하여 데이터와 레이블을 재정렬합니다.
shuffled_features = features[indices]
shuffled_labels = labels[indices]

print(f"\n원본 특성:\n{features}")
print(f"섞인 특성:\n{shuffled_features}")
print(f"\n원본 레이블: {labels}")
print(f"섞인 레이블: {shuffled_labels}")
```

#### 3.1.2. `rng.permutation()`

`rng.permutation(x, axis=0)` 메서드는 주어진 배열 `x`의 순서를 무작위로 섞은 **새로운 배열(복사본)**을 반환합니다. 이 메서드의 가장 중요한 특징은 **원본 배열을 변경하지 않는다**는 점입니다. 따라서 원본 데이터를 보존하면서 무작위 순열이 필요할 때 매우 유용합니다.

**주요 특징 및 매개변수:**

*   **`x` (int 또는 ndarray)**: 순열을 생성할 대상입니다.
    *   **정수(int)**: `x`가 정수 `n`이면, `0`부터 `n-1`까지의 정수(`np.arange(n)`)를 무작위로 섞은 배열을 생성하여 반환합니다. 이는 무작위 인덱스를 생성하는 데 매우 편리합니다.
    *   **배열(ndarray)**: `x`가 배열이면, 해당 배열의 복사본을 만들어 섞은 후 반환합니다.
*   **`axis` (int, 선택 사항)**: 섞기를 수행할 축을 지정합니다. 기본값은 `0`이며, 이는 첫 번째 축(일반적으로 행)을 따라 섞는 것을 의미합니다. 
*   **복사본 반환**: 원본 배열 `x`는 전혀 수정되지 않고, 무작위로 섞인 새로운 배열이 반환됩니다.
*   **반환값**: 섞인 배열의 복사본을 반환합니다.

**동작 방식:**

*   **정수 입력**: `rng.permutation(10)`은 `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`를 무작위로 섞은 새로운 배열을 반환합니다.
*   **배열 입력**: 
    *   **1차원 배열**: 배열의 복사본을 만들어 요소들을 무작위로 섞어 반환합니다.
    *   **다차원 배열**: `axis=0` (기본값)일 경우, 행의 순서가 무작위로 섞인 새로운 배열을 반환합니다. 각 행 내부의 요소 순서는 유지됩니다. `axis=1`일 경우, 각 행 내부의 열 순서가 섞인 새로운 배열을 반환합니다.

**활용 사례:**

*   **훈련/검증/테스트 데이터 분할**: 데이터셋의 인덱스를 무작위로 섞은 후, 이를 사용하여 데이터를 분할합니다. 원본 데이터 순서를 유지할 수 있어 안전합니다.
*   **데이터 증강**: 원본 데이터를 보존하면서 순서를 바꾼 새로운 데이터를 생성할 때 사용됩니다.
*   **교차 검증**: `K-Fold` 교차 검증을 위해 인덱스를 무작위로 섞어 각 폴드에 데이터가 편향되지 않게 분배할 때 유용합니다.
*   **재현 가능한 샘플링**: `rng.choice`의 `replace=False`와 유사한 비복원 샘플링을 구현할 때, 순열을 생성한 후 앞에서부터 필요한 개수만큼 잘라내어 사용할 수 있습니다.

**`rng.shuffle()`과의 차이점:**

가장 큰 차이점은 **수정 대상**입니다. `rng.shuffle()`은 **원본 배열을 직접 수정**하고 `None`을 반환하는 반면, `rng.permutation()`은 **원본을 그대로 두고** 섞인 **새로운 배열(복사본)을 반환**합니다.

**예시 코드:**

```python
import numpy as np
rng = np.random.default_rng(seed=42)

# 1. 정수 인자를 사용하여 순열 생성
perm_indices = rng.permutation(10)
print(f"permutation 결과 (정수 10 입력): {perm_indices}")

# 2. 1차원 배열의 순열 생성
arr_1d = np.array(['A', 'B', 'C', 'D', 'E'])
perm_arr_1d = rng.permutation(arr_1d)
print(f"\n원본 1차원 배열: {arr_1d}")
print(f"permutation 후 1차원 배열: {perm_arr_1d}")
print(f"-> 원본 배열은 변경되지 않았습니다.")

# 3. 2차원 배열의 순열 생성 (행 단위, axis=0)
arr_2d = np.array([[1, 10, 100],
                   [2, 20, 200],
                   [3, 30, 300]])
perm_arr_2d = rng.permutation(arr_2d)
print(f"\n원본 2차원 배열:\n{arr_2d}")
print(f"permutation 후 2차원 배열 (행 섞임):\n{perm_arr_2d}")
print(f"-> 원본 2차원 배열은 변경되지 않았습니다.")

# 4. 머신러닝 데이터와 레이블 함께 섞기 (권장 방식)
features = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
labels = np.array(['A', 'B', 'C', 'D'])

# permutation을 사용하여 섞인 인덱스 생성
shuffled_indices = rng.permutation(features.shape[0])
print(f"\n생성된 섞인 인덱스: {shuffled_indices}")

# 섞인 인덱스를 사용하여 데이터와 레이블 재정렬
shuffled_features = features[shuffled_indices]
shuffled_labels = labels[shuffled_indices]

print(f"\n원본 특성:\n{features}")
print(f"섞인 특성:\n{shuffled_features}")
print(f"\n원본 레이블: {labels}")
print(f"섞인 레이블: {shuffled_labels}")
```

#### 3.2. 무작위 샘플링 (`rng.choice`)

`rng.choice(a, size=None, replace=True, p=None, axis=0, shuffle=True)` 메서드는 주어진 배열 `a`에서 `size`만큼의 무작위 샘플을 추출하는 매우 유연하고 강력한 함수입니다. 복원/비복원 추출, 가중치 기반 샘플링 등 다양한 샘플링 시나리오를 지원하여 데이터 과학 및 머신러닝 워크플로우에서 핵심적인 역할을 합니다.

**주요 매개변수:**

*   **`a` (int 또는 1-D array_like)**: 샘플링할 모집단입니다.
    *   **정수(int)**: `a`가 정수 `n`이면, `np.arange(n)` 배열에서 샘플링합니다.
    *   **배열(array_like)**: 1차원 배열 또는 리스트에서 직접 샘플링합니다.
*   **`size` (int 또는 tuple, 선택 사항)**: 추출할 샘플의 크기(개수 또는 형태)를 지정합니다. `None`이면 단일 값을 반환합니다.
*   **`replace` (bool, 선택 사항)**: 복원 추출 여부를 결정합니다.
    *   `True` (기본값): **복원 추출**을 의미하며, 한 번 선택된 요소를 다시 선택할 수 있습니다. `size`는 `a`의 크기보다 클 수 있습니다.
    *   `False`: **비복원 추출**을 의미하며, 한 번 선택된 요소는 다시 선택될 수 없습니다. 이 경우 `size`는 `a`의 크기보다 클 수 없습니다.
*   **`p` (1-D array_like, 선택 사항)**: 가중치 기반 샘플링을 위한 확률 배열입니다. `a`의 각 요소가 선택될 확률을 지정하며, `a`와 같은 크기여야 하고 모든 확률의 합은 1이어야 합니다.
*   **`axis` (int, 선택 사항)**: 샘플링을 수행할 축을 지정합니다. 기본값은 `0`입니다. 이 매개변수는 `a`가 다차원 배열일 때 사용되며, 지정된 축을 따라 인덱싱하여 샘플을 추출합니다. (참고: `choice`는 기본적으로 1차원 배열을 가정하지만, `axis`를 통해 다차원 배열의 슬라이스를 샘플링할 수 있습니다.)
*   **`shuffle` (bool, 선택 사항)**: 샘플을 섞을지 여부를 결정합니다. 기본값은 `True`로, 생성된 샘플의 순서를 무작위로 섞습니다. `False`로 설정하면 샘플링은 되지만 순서는 섞이지 않습니다 (예: 가중치 기반 샘플링에서 확률이 높은 순서대로 나타날 경향이 있음).

**활용 사례:**

*   **부트스트래핑 (Bootstrapping)**: 데이터셋에서 중복을 허용하여(`replace=True`) 원본과 동일한 크기의 표본을 여러 번 추출하여 통계량의 신뢰 구간을 추정하거나 모델의 안정성을 평가합니다.
*   **미니배치(Mini-batch) 생성**: 대규모 데이터셋에서 학습을 위한 작은 데이터 묶음(미니배치)을 무작위로 추출할 때 사용됩니다 (`replace=False`).
*   **불균형 데이터셋 처리**: 
    *   **오버샘플링(Oversampling)**: 소수 클래스 데이터의 선택 확률(`p`)을 높여 더 많이 샘플링함으로써 클래스 불균형을 완화합니다.
    *   **언더샘플링(Undersampling)**: 다수 클래스 데이터에서 무작위로 일부만 샘플링하여 데이터 양을 줄입니다.
*   **설문조사 시뮬레이션**: 특정 응답 비율(`p`)에 따라 설문 응답 결과를 시뮬레이션합니다.

**예시 코드:**

```python
import numpy as np
rng = np.random.default_rng(seed=42)

# 1. 기본 샘플링
population = np.arange(10)
print(f"모집단: {population}")

# 복원 추출 (중복 가능)
sample_with_replace = rng.choice(population, size=10, replace=True)
print(f"복원 추출 (size=10): {sample_with_replace}")

# 비복원 추출 (중복 없음, size는 모집단 크기 이하)
sample_without_replace = rng.choice(population, size=5, replace=False)
print(f"비복원 추출 (size=5): {sample_without_replace}")


# 2. 가중치 기반 샘플링 (p 인자 사용)
items = ['A', 'B', 'C', 'D']
probabilities = [0.1, 0.1, 0.7, 0.1]  # 'C'가 나올 확률이 70%
weighted_sample = rng.choice(items, size=10, p=probabilities)
print(f"\n가중치 기반 샘플링: {weighted_sample}")


# 3. 다차원 배열에서 행 샘플링 (axis 인자)
# choice는 직접 다차원 배열을 받아 행을 샘플링하지 않으므로, 인덱스를 샘플링하는 것이 표준적인 방법입니다.
features = np.array([[1,10], [2,20], [3,30], [4,40], [5,50]])
print(f"\n원본 특성 데이터:\n{features}")

# 인덱스를 비복원 샘플링하여 미니배치 생성
batch_indices = rng.choice(features.shape[0], size=3, replace=False)
mini_batch = features[batch_indices]
print(f"샘플링된 인덱스: {batch_indices}")
print(f"생성된 미니배치:\n{mini_batch}")


# 4. shuffle 매개변수 효과
# shuffle=False로 설정하면 샘플링된 결과의 순서가 예측 가능해질 수 있습니다.
# (내부 구현에 따라 정렬된 순서와 유사하게 나타날 수 있음)
elements = ['a', 'b', 'c']
probs = [0.6, 0.3, 0.1]
shuffled_sample = rng.choice(elements, size=10, p=probs, shuffle=True)
unshuffled_sample = rng.choice(elements, size=10, p=probs, shuffle=False)

print(f"\nShuffle=True 샘플: {shuffled_sample}")
print(f"Shuffle=False 샘플: {unshuffled_sample}")
```

## 4. 전통적인(Legacy) 난수 생성 API

NumPy의 최신 `Generator` API가 도입되기 전에 사용되던 `np.random.seed()`, `np.random.rand()`, `np.random.randn()` 등의 함수들을 **전통적인(Legacy) API**라고 부릅니다. 이 API는 사용하기 간편해 보이지만, 현대적인 데이터 과학 및 소프트웨어 개발 관점에서 몇 가지 중요한 단점을 가지고 있어 **새로운 코드에서는 사용을 지양**하는 것이 좋습니다.

### 왜 Legacy API 사용을 피해야 하는가?

가장 큰 이유는 이 API가 **전역 상태(Global State)**에 의존하기 때문입니다. `np.random.seed()`를 호출하면, 이는 보이지 않는 전역 난수 생성기의 상태를 변경합니다. 그 후 `np.random.rand()`와 같은 함수를 호출하면 이 전역 생성기에서 난수를 가져옵니다.

이러한 방식은 다음과 같은 심각한 문제를 야기할 수 있습니다.

1.  **재현성 부족 (Lack of Reproducibility)**:
    *   스크립트의 어느 곳에서든 `np.random` 함수가 호출되면 전역 난수 생성기의 상태가 변합니다.
    *   만약 내가 작성한 함수와 내가 호출하는 라이브러리 함수 양쪽에서 `np.random`을 사용한다면, 라이브러리의 내부 구현이 바뀌거나 호출 순서가 달라지는 것만으로도 내 함수의 난수 생성 결과가 완전히 달라질 수 있습니다.
    *   이로 인해 실험 결과를 재현하거나 버그를 디버깅하는 것이 매우 어려워집니다.

2.  **스레드 안전성 부족 (Not Thread-safe)**:
    *   여러 스레드에서 동시에 `np.random` 함수를 호출하면, 각 스레드가 전역 난수 생성기의 상태를 예측 불가능하게 변경하여 경쟁 상태(race condition)를 일으킬 수 있습니다. 이는 데이터 손상이나 잘못된 결과를 초래할 수 있습니다.

3.  **캡슐화 및 모듈성 저해**: 
    *   함수나 클래스가 내부적으로 전역 상태에 의존하게 되면, 해당 코드의 동작을 이해하고 테스트하기가 더 어려워집니다. 좋은 코드는 가급적 외부 상태에 의존하지 않고 독립적으로(self-contained) 작동해야 합니다.

### 전역 상태의 문제점 예시

아래 코드는 전역 상태로 인해 어떻게 예상치 못한 결과가 발생할 수 있는지 보여줍니다.

```python
import numpy as np

# 분석 함수 A: 데이터에서 3개의 샘플을 뽑아 분석
def analyze_data_A():
    print(f"  [A] 분석 시작 - 현재 난수 상태: {np.random.get_state()[1][0]}")
    samples = np.random.rand(3)
    print(f"  [A] 추출된 샘플: {samples}")
    print(f"  [A] 분석 종료 - 현재 난수 상태: {np.random.get_state()[1][0]}")

# 분석 함수 B: 중간에 다른 라이브러리(처럼 보이는) 함수를 호출
def analyze_data_B():
    print(f"  [B] 분석 시작 - 현재 난수 상태: {np.random.get_state()[1][0]}")
    
    # 예상치 못하게 np.random을 사용하는 다른 함수 호출
    print("  [B] 외부 라이브러리 호출...")
    np.random.rand(5) # 이 함수가 전역 난수 생성기를 소모함
    print("  [B] 외부 라이브러리 호출 완료.")
    
    samples = np.random.rand(3)
    print(f"  [B] 추출된 샘플: {samples}")
    print(f"  [B] 분석 종료 - 현재 난수 상태: {np.random.get_state()[1][0]}")

# 시나리오 1: 함수 A만 호출
print("시나리오 1: analyze_data_A 호출")
np.random.seed(42)
analyze_data_A()

print("-" * 30)

# 시나리오 2: 함수 B 호출 (내부에서 난수 생성기가 추가로 소모됨)
print("시나리오 2: analyze_data_B 호출")
np.random.seed(42) # 동일한 시드로 초기화
analyze_data_B()

print("-" * 30)

# 시나리오 3: 다시 함수 A를 호출하여 B의 영향 확인
print("시나리오 3: analyze_data_A 다시 호출 (B의 영향 없음)")
np.random.seed(42) # 동일한 시드로 초기화
analyze_data_A()
```

**결과 분석:**
*   `analyze_data_A`와 `analyze_data_B`는 동일하게 `np.random.seed(42)`로 시작했지만, `analyze_data_B` 내부에서 예상치 못하게 `np.random.rand(5)`가 호출되면서 전역 난수 생성기의 상태가 앞으로 당겨졌습니다.
*   그 결과, `analyze_data_B`가 최종적으로 추출한 3개의 샘플은 `analyze_data_A`의 결과와 완전히 다릅니다.
*   이것이 바로 Legacy API의 가장 큰 문제입니다. 코드의 다른 부분에서 일어난 일이 내 코드의 결과에 영향을 미치는 것을 막을 수 없습니다.

### 권장 사항

이러한 문제들 때문에 NumPy 개발팀은 `np.random.Generator`를 사용하는 새로운 API를 강력히 권장합니다. `Generator` 객체는 독립적인 난수 스트림을 가지므로, 다른 코드의 난수 생성에 영향을 받지 않아 재현 가능하고 안정적인 코드를 작성할 수 있습니다.

**항상 새로운 API를 사용하세요:**
```python
import numpy as np

# Generator를 생성하여 함수에 전달하거나, 함수 내부에서 생성
rng = np.random.default_rng(seed=42)

# 이제 rng 객체를 통해서만 난수를 생성하므로 전역 상태와 무관하게 항상 동일한 결과를 보장합니다.
print(f"Generator API 결과: {rng.random(3)}")
print(f"Generator API 결과: {rng.random(3)}")
```