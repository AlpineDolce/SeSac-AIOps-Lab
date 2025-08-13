<h2>NumPy 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
이 문서는 안정적이고 정확한 NumPy 코드를 작성하기 위한 테스트 및 디버깅 기법을 다룹니다. 부동소수점 연산의 미세한 오차를 고려한 배열 비교 방법, `numpy.testing` 모듈의 활용법, 그리고 NumPy 코드에서 흔히 발생하는 오류를 효율적으로 찾아내고 해결하는 실용적인 디버깅 전략을 학습합니다.

<h2>목차</h2>

- [1. 왜 NumPy 코드 테스트가 중요한가?](#1-왜-numpy-코드-테스트가-중요한가)
  - [1.1. 부동소수점 오차의 영향](#11-부동소수점-오차의-영향)
  - [1.2. 암시적 동작과 잠재적 버그](#12-암시적-동작과-잠재적-버그)
- [2. `numpy.testing` 모듈 활용](#2-numpytesting-모듈-활용)
  - [2.1. `numpy.testing` 개요](#21-numpytesting-개요)
  - [2.2. 부동소수점 배열 비교: `assert_allclose`](#22-부동소수점-배열-비교-assert_allclose)
    - [2.2.1. `assert_allclose` 기본 사용법](#221-assert_allclose-기본-사용법)
    - [2.2.2. `rtol`과 `atol` 이해하기](#222-rtol과-atol-이해하기)
  - [2.3. 정확한 배열 비교: `assert_array_equal`](#23-정확한-배열-비교-assert_array_equal)
    - [2.3.1. 정수 및 문자열 배열 비교 (성공)](#231-정수-및-문자열-배열-비교-성공)
    - [2.3.2. 요소 불일치 (실패)](#232-요소-불일치-실패)
    - [2.3.3. 형태 불일치 (실패)](#233-형태-불일치-실패)
- [3. NumPy 코드 디버깅 전략](#3-numpy-코드-디버깅-전략)
  - [3.1. 디버깅의 중요성](#31-디버깅의-중요성)
  - [3.2. 가장 흔한 오류: Shape Mismatch](#32-가장-흔한-오류-shape-mismatch)
    - [3.2.1. `shape` 확인의 중요성](#321-shape-확인의-중요성)
    - [3.2.2. 행렬 곱셈 불일치 예시](#322-행렬-곱셈-불일치-예시)
    - [3.2.3. 올바른 행렬 곱셈 예시](#323-올바른-행렬-곱셈-예시)
  - [3.3. 데이터 타입(dtype) 확인](#33-데이터-타입dtype-확인)
    - [3.3.1. `dtype`의 중요성](#331-dtype의-중요성)
    - [3.3.2. 정수 나눗셈 문제](#332-정수-나눗셈-문제)
    - [3.3.3. 오버플로우 문제 및 해결](#333-오버플로우-문제-및-해결)
  - [3.4. 특수 값(NaN, Inf) 확인](#34-특수-값nan-inf-확인)
    - [3.4.1. `NaN` 생성 및 확인](#341-nan-생성-및-확인)
    - [3.4.2. `Inf` 생성 및 확인](#342-inf-생성-및-확인)
    - [3.4.3. `NaN`과 `Inf` 동시 확인](#343-nan과-inf-동시-확인)
  - [3.5. 작은 데이터로 문제 축소](#35-작은-데이터로-문제-축소)

--- 

## 1. 왜 NumPy 코드 테스트가 중요한가?

데이터 과학 및 머신러닝 코드는 본질적으로 실험적이지만, 프로덕션 환경에 적용되거나 연구 결과를 재현하기 위해서는 **정확성**과 **안정성**이 반드시 보장되어야 합니다. 특히 NumPy 코드는 다음과 같은 특징 때문에 테스트가 더욱 중요합니다.

### 1.1. 부동소수점 오차의 영향

컴퓨터의 이진수 표현 한계로 인해 부동소수점 연산은 미세한 오차를 포함할 수 있습니다. 이는 `0.1 + 0.2`가 정확히 `0.3`이 아닌 것처럼, 배열 간의 단순 비교(`==`)가 예상치 못한 `False`를 반환할 수 있습니다. 이러한 미세한 오차는 복잡한 계산 과정에서 누적되어 최종 결과에 큰 영향을 미 미칠 수 있으므로, 이를 고려한 테스트가 필수적입니다.

```python
import numpy as np

a = 0.1 + 0.2
b = 0.3
print(f"0.1 + 0.2 = {a}")
print(f"0.3 = {b}")
print(f"a == b: {a == b}") # False

arr1 = np.array([0.1, 0.2])
arr2 = np.array([0.3, 0.4])
print(f"\n(arr1 + arr1) == arr2: {(arr1 + arr1) == arr2}") # [False False]
```

### 1.2. 암시적 동작과 잠재적 버그

브로드캐스팅(Broadcasting)과 같은 NumPy의 강력한 기능은 코드를 간결하게 만들지만, 개발자의 의도와 다르게 배열의 `shape`이 변경되거나 연산이 수행될 수 있습니다. 이는 겉으로는 오류가 없어 보이지만, 잘못된 결과를 도출하는 잠재적인 버그의 원인이 되기도 합니다. 따라서 이러한 암시적 동작이 올바르게 작동하는지 확인하는 테스트가 중요합니다.

## 2. `numpy.testing` 모듈 활용

`numpy.testing` 모듈은 NumPy 배열을 효과적으로 테스트하기 위한 다양한 유틸리티 함수를 제공합니다. 이 모듈은 특히 부동소수점 연산의 특성을 고려한 비교 함수들을 포함하고 있어, 데이터 과학 및 머신러닝 코드의 정확성을 검증하는 데 필수적입니다. `pytest`와 같은 표준 테스트 프레임워크와 함께 사용하면 더욱 강력한 테스트 환경을 구축할 수 있습니다.

### 2.1. `numpy.testing` 개요

`numpy.testing`은 주로 두 배열이 '충분히 가까운지' 또는 '정확히 같은지'를 확인하는 어설션(assertion) 함수들을 제공합니다. 테스트 스크립트에서 이 함수들을 호출하여 예상 결과와 실제 결과가 일치하는지 검증합니다. 불일치할 경우 `AssertionError`를 발생시켜 테스트 실패를 알립니다.

### 2.2. 부동소수점 배열 비교: `assert_allclose`

`assert_allclose(actual, desired, rtol, atol)` 함수는 두 부동소수점 배열이 **허용 오차 내에서 거의 가까운지(all close)**를 확인하는 가장 표준적인 방법입니다. 부동소수점 오차로 인해 정확히 일치하지 않을 수 있는 경우에 사용됩니다. 테스트 통과 조건은 다음 공식을 따릅니다: `abs(actual - desired) <= atol + rtol * abs(desired)`

#### 2.2.1. `assert_allclose` 기본 사용법

`assert_allclose`의 기본적인 사용법을 통해 두 부동소수점 배열이 허용 오차 내에서 동일한지 확인하는 방법을 알아봅니다.

```python
import numpy as np
from numpy.testing import assert_allclose

# 성공 케이스: 허용 오차 내에서 동일한 경우
result1 = np.array([0.1 + 0.2, 1.0/3.0])
desired1 = np.array([0.3, 0.3333333333333333])

try:
    assert_allclose(result1, desired1)
    print("\n테스트 통과 (예시 1): 두 배열은 기본 허용 오차 내에서 동일합니다.")
except AssertionError as e:
    print(f"\n테스트 실패 (예시 1):\n{e}")

# 실패 케이스: 오차가 커서 실패하는 경우
result2 = np.array([0.300001])
desired2 = np.array([0.3])

try:
    assert_allclose(result2, desired2, rtol=1e-7, atol=1e-7) # 기본값보다 엄격한 오차
    print("\n테스트 통과 (예시 2): 두 배열은 지정된 허용 오차 내에서 동일합니다.")
except AssertionError as e:
    print(f"\n테스트 실패 (예시 2):\n{e}")
```

#### 2.2.2. `rtol`과 `atol` 이해하기

`assert_allclose` 함수에서 `rtol`과 `atol` 매개변수는 부동소수점 비교의 엄격도를 조절하는 핵심 요소입니다. 이 두 값의 의미와 활용법을 자세히 살펴봅니다.

-   `rtol` (relative tolerance): **상대 허용 오차**. `desired` 값의 크기에 비례하여 오차 범위를 설정합니다. 큰 값일수록 상대 오차의 중요성이 커집니다. 기본값은 `1e-7`입니다.
    -   예: `desired`가 100일 때 `rtol=1e-5`이면, 0.001까지의 오차를 허용합니다.
-   `atol` (absolute tolerance): **절대 허용 오차**. `desired` 값의 크기와 무관하게 최소한의 오차 범위를 보장합니다. `desired` 값이 0에 가까울 때 `rtol`만으로는 충분하지 않을 수 있으므로 `atol`이 중요합니다. 기본값은 `0`입니다.
    -   예: `atol=1e-8`이면, `desired` 값과 상관없이 0.00000001까지의 오차를 허용합니다.

두 매개변수를 적절히 조합하여 테스트의 엄격도를 조절할 수 있습니다. 일반적으로 `atol`은 작은 값에 대한 오차를, `rtol`은 큰 값에 대한 오차를 다룹니다.

```python
import numpy as np
from numpy.testing import assert_allclose

# atol의 중요성: desired 값이 0에 가까울 때
actual_small = np.array([1e-9])
desired_small = np.array([0.0])

# rtol만 사용 시: 0에 가까운 값의 오차를 잡기 어려움
try:
    assert_allclose(actual_small, desired_small, rtol=1e-5) # 실패할 수 있음
    print("\n테스트 통과 (rtol만 사용): rtol만으로 통과")
except AssertionError as e:
    print(f"\n테스트 실패 (rtol만 사용): rtol만으로는 부족\n{e}")

# atol 함께 사용 시: 작은 값의 오차를 명확히 허용
try:
    assert_allclose(actual_small, desired_small, rtol=1e-5, atol=1e-8)
    print("\n테스트 통과 (atol 함께 사용): atol 추가로 통과")
except AssertionError as e:
    print(f"\n테스트 실패 (atol 함께 사용):\n{e}")
```

### 2.3. 정확한 배열 비교: `assert_array_equal`

`assert_array_equal(x, y)` 함수는 두 배열의 형태(shape)와 모든 요소가 **정확히 일치하는지**를 확인합니다. 정수형 배열, 문자열 배열, 불리언 배열, 또는 객체 배열처럼 각 요소가 오차 없이 동일해야 하는 경우에 사용됩니다. 부동소수점 배열에는 `assert_allclose`를 사용하는 것이 더 적절합니다.

#### 2.3.1. 정수 및 문자열 배열 비교 (성공)

`assert_array_equal`은 정수, 문자열, 불리언 등 정확한 일치가 필요한 데이터 타입에 유용합니다. 다음은 성공적인 비교 예시입니다.

```python
from numpy.testing import assert_array_equal
import numpy as np

# 정수 배열 테스트 (성공)
result_int = np.array([1, 2, 3])
desired_int = np.array([1, 2, 3])
try:
    assert_array_equal(result_int, desired_int)
    print("\n정수 배열 테스트 통과.")
except AssertionError as e:
    print(f"\n정수 배열 테스트 실패:\n{e}")

# 문자열 배열 테스트 (성공)
result_str = np.array(['apple', 'banana'])
desired_str = np.array(['apple', 'banana'])
try:
    assert_array_equal(result_str, desired_str)
    print("문자열 배열 테스트 통과.")
except AssertionError as e:
    print(f"문자열 배열 테스트 실패:\n{e}")

# 불리언 배열 테스트 (성공)
result_bool = np.array([True, False])
desired_bool = np.array([True, False])
try:
    assert_array_equal(result_bool, desired_bool)
    print("불리언 배열 테스트 통과.")
except AssertionError as e:
    print(f"불리언 배열 테스트 실패:\n{e}")
```

#### 2.3.2. 요소 불일치 (실패)

두 배열의 형태는 같지만, 하나 이상의 요소 값이 일치하지 않을 경우 `AssertionError`가 발생합니다.

```python
from numpy.testing import assert_array_equal
import numpy as np

result_int = np.array([1, 2, 3])
# 실패 케이스 (요소 불일치)
fail_int = np.array([1, 2, 4])
try:
    assert_array_equal(result_int, fail_int)
except AssertionError as e:
    print(f"\n정수 배열 테스트 실패 (요소 불일치):\n{e}")
```

#### 2.3.3. 형태 불일치 (실패)

두 배열의 요소 값은 같을 수 있지만, 형태(shape)가 다를 경우에도 `AssertionError`가 발생합니다.

```python
from numpy.testing import assert_array_equal
import numpy as np

result_int = np.array([1, 2, 3])
# 실패 케이스 (형태 불일치)
fail_shape = np.array([[1, 2], [3, 4]])
try:
    assert_array_equal(result_int, fail_shape)
except AssertionError as e:
    print(f"\n정수 배열 테스트 실패 (형태 불일치):\n{e}")
```

## 3. NumPy 코드 디버깅 전략

NumPy 관련 버그는 종종 암시적으로 발생하거나 복잡한 배열 연산 과정에서 숨겨져 추적하기 어려울 수 있습니다. 효과적인 디버깅 전략은 문제 해결 시간을 단축하고 코드의 신뢰성을 높이는 데 필수적입니다.

### 3.1. 디버깅의 중요성

디버깅은 단순히 오류를 수정하는 것을 넘어, 코드의 동작 방식을 깊이 이해하고 잠재적인 문제를 미리 발견하는 과정입니다. 특히 NumPy와 같이 데이터의 형태와 타입이 중요한 라이브러리에서는 작은 실수가 큰 오류로 이어질 수 있으므로, 체계적인 디버깅 습관이 중요합니다.

### 3.2. 가장 흔한 오류: Shape Mismatch

NumPy 연산에서 발생하는 오류의 상당 부분은 배열의 `shape`이 맞지 않아 발생합니다. 특히 브로드캐스팅 규칙을 잘못 이해하거나, 행렬 곱셈(`@`)과 같은 연산에서 차원 호환성이 맞지 않을 때 흔히 발생합니다. 오류가 발생하면 가장 먼저 관련된 모든 배열의 `shape`을 출력하여 확인하는 습관을 들이는 것이 좋습니다.

#### 3.2.1. `shape` 확인의 중요성

`shape` 불일치는 NumPy 연산 오류의 가장 흔한 원인 중 하나입니다. 연산에 참여하는 배열들의 `shape`을 명시적으로 확인하는 것은 문제 발생 시 원인을 빠르게 파악하는 데 결정적인 역할을 합니다.

```python
import numpy as np

a = np.arange(6).reshape(2, 3) # 형태: (2, 3)
b = np.arange(6).reshape(3, 2) # 형태: (3, 2)
c_fail = np.arange(4).reshape(2, 2) # 형태: (2, 2)

print(f"a.shape: {a.shape}")
print(f"b.shape: {b.shape}")
print(f"c_fail.shape: {c_fail.shape}")
```

#### 3.2.2. 행렬 곱셈 불일치 예시

행렬 곱셈(`@` 연산자)은 내적을 수행하므로, 첫 번째 행렬의 열 수와 두 번째 행렬의 행 수가 일치해야 합니다. 이 조건이 충족되지 않으면 `ValueError`가 발생합니다.

```python
import numpy as np

a = np.arange(6).reshape(2, 3) # 형태: (2, 3)
c_fail = np.arange(4).reshape(2, 2) # 형태: (2, 2)

try:
    result_fail = a @ c_fail
    print(result_fail)
except ValueError as e:
    print(f"\n오류 발생 (a @ c_fail): {e}")
    print("a(2,3)와 c_fail(2,2)의 내적은 불가능함을 shape으로 바로 알 수 있습니다. (3 != 2)")
```

#### 3.2.3. 올바른 행렬 곱셈 예시

`shape` 규칙을 준수하여 올바르게 행렬 곱셈을 수행하는 예시입니다. 결과 배열의 `shape`도 함께 확인하여 연산이 의도대로 이루어졌는지 검증합니다.

```python
import numpy as np

a = np.arange(6).reshape(2, 3) # 형태: (2, 3)
b = np.arange(6).reshape(3, 2) # 형태: (3, 2)

try:
    result_success = a @ b
    print(f"\na @ b 결과:\n{result_success}")
    print(f"결과 shape: {result_success.shape}")
except ValueError as e:
    print(f"\n오류 발생 (a @ b): {e}")
```

### 3.3. 데이터 타입(dtype) 확인

의도치 않은 데이터 타입(`dtype`)은 연산 결과를 왜곡하거나 성능을 저하시킬 수 있습니다. 연산 중간중간 `arr.dtype`을 확인하여 의도한 타입이 맞는지 검사하는 것이 좋습니다.

#### 3.3.1. `dtype`의 중요성

-   **정확성 문제**: 정수 나눗셈, 오버플로우/언더플로우 등 `dtype`에 따라 연산 결과가 달라질 수 있습니다.
-   **성능 문제**: 불필요하게 큰 `dtype` 사용 시 메모리 사용량이 늘어나고 연산 효율이 저하될 수 있습니다.

```python
import numpy as np

a = np.array([5, 6])
b = np.array([2, 2])

print(f"a: {a}, dtype: {a.dtype}")
print(f"b: {b}, dtype: {b.dtype}")
```

#### 3.3.2. 정수 나눗셈 문제

Python 3의 일반 나눗셈(`a / b`)은 항상 부동소수점 결과를 반환하지만, NumPy 배열 간의 나눗셈은 피연산자의 `dtype`에 따라 결과 `dtype`이 결정될 수 있습니다. 특히 정수 `dtype` 배열 간의 나눗셈은 소수점 이하를 버리는 결과를 초래할 수 있습니다.

```python
import numpy as np

a = np.array([5, 6])
b = np.array([2, 2])

# 정수 나눗셈: 소수점 이하 버림 (결과 dtype이 float으로 자동 변환되지만, 값은 정수 나눗셈 결과)
result_int_div = a / b
print(f"\n정수 배열 나눗셈: {result_int_div} (dtype: {result_int_div.dtype})")

# 의도치 않은 타입 변환 방지: 명시적으로 float 타입으로 변환하여 정확한 실수 나눗셈 수행
a_float = a.astype(float)
print(f"a_float: {a_float}, dtype: {a_float.dtype}")
result_float_div = a_float / b
print(f"실수 배열 나눗셈: {result_float_div} (dtype: {result_float_div.dtype})")
```

#### 3.3.3. 오버플로우 문제 및 해결

정수형 `dtype`은 표현할 수 있는 값의 범위가 정해져 있습니다. 이 범위를 넘어서는 값을 계산하려 하면 오버플로우(overflow)가 발생하여 예상치 못한 결과가 나올 수 있습니다. 더 큰 `dtype`으로 명시적으로 변환하여 이를 방지할 수 있습니다.

```python
import numpy as np

# int32의 최대값 (2^31 - 1)
large_int = np.array([2**31 - 1], dtype=np.int32)
print(f"\nlarge_int: {large_int}, dtype: {large_int.dtype}")

# 오버플로우 발생: int32 범위를 넘어섬
overflow_int = large_int + 1
print(f"overflow_int (int32 + 1): {overflow_int}, dtype: {overflow_int.dtype}") # 음수로 변환될 수 있음

# 더 큰 타입(int64)으로 변환하여 오버플로우 방지
large_int_64 = large_int.astype(np.int64)
overflow_int_64 = large_int_64 + 1
print(f"overflow_int_64 (int64 + 1): {overflow_int_64}, dtype: {overflow_int_64.dtype}")
```

### 3.4. 특수 값(NaN, Inf) 확인

0으로 나누거나, 로그에 음수를 넣는 등 잘못된 수학 연산은 `np.nan`(Not a Number)이나 `np.inf`(Infinity)를 발생시킬 수 있습니다. 이 값들은 다른 모든 계산을 오염시키므로, 발생 즉시 찾아내고 원인을 해결하는 것이 중요합니다.

-   `np.isnan(arr)`: 배열의 각 요소가 `NaN`인지 여부를 불리언 배열로 반환합니다.
-   `np.isinf(arr)`: 배열의 각 요소가 `Inf`인지 여부를 불리언 배열로 반환합니다.
-   `.any()` 메서드를 함께 사용하여 배열에 `NaN`이나 `Inf`가 하나라도 있는지 빠르게 확인할 수 있습니다.

#### 3.4.1. `NaN` 생성 및 확인

`NaN`은 정의되지 않은 수학적 연산(예: 0/0, inf/inf)의 결과로 발생합니다. `np.isnan()` 함수를 사용하여 배열 내의 `NaN` 값을 식별할 수 있습니다.

```python
import numpy as np

# NaN 생성 예시: 0으로 나누기 (부동소수점 0.0으로 나누면 Inf, 0/0은 NaN)
arr_nan = np.array([1.0, 0.0, -1.0]) / np.array([0.0, 0.0, 0.0]) # 1.0/0.0 -> inf, 0.0/0.0 -> nan, -1.0/0.0 -> -inf
print(f"NaN 포함 배열: {arr_nan}")
print(f"NaN 여부 (요소별): {np.isnan(arr_nan)}")
if np.isnan(arr_nan).any():
    print("배열에 NaN 값이 하나라도 포함되어 있습니다.")

# 또 다른 NaN 생성 예시: log(음수)
# arr_nan_log = np.log([-1.0]) # RuntimeWarning: invalid value encountered in log
# print(f"log(음수)로 인한 NaN: {arr_nan_log}")
```

#### 3.4.2. `Inf` 생성 및 확인

`Inf`는 매우 큰 수를 표현하거나 0이 아닌 수를 0으로 나눌 때 발생합니다. `np.isinf()` 함수를 사용하여 배열 내의 `Inf` 값을 식별할 수 있습니다.

```python
import numpy as np

# Inf 생성 예시: 0이 아닌 수를 0으로 나누기
arr_inf = np.array([1.0, 2.0]) / np.array([0.0, 0.0])
print(f"\nInf 포함 배열: {arr_inf}")
print(f"Inf 여부 (요소별): {np.isinf(arr_inf)}")
if np.isinf(arr_inf).any():
    print("배열에 Inf 값이 하나라도 포함되어 있습니다.")

# 음의 Inf
arr_neg_inf = np.array([-1.0]) / 0.0
print(f"음의 Inf 포함 배열: {arr_neg_inf}")
print(f"음의 Inf 여부 (요소별): {np.isinf(arr_neg_inf)}")
```

#### 3.4.3. `NaN`과 `Inf` 동시 확인

하나의 배열에 `NaN`과 `Inf`가 모두 포함될 수 있습니다. 각 함수를 개별적으로 사용하여 두 가지 특수 값을 모두 확인할 수 있습니다.

```python
import numpy as np

arr_mixed = np.array([1, 2, np.nan, np.inf, 5, -np.inf])
print(f"\n혼합 배열: {arr_mixed}")
print(f"NaN 여부 (요소별): {np.isnan(arr_mixed)}")
print(f"Inf 여부 (요소별): {np.isinf(arr_mixed)}")

if np.isnan(arr_mixed).any():
    print("혼합 배열에 NaN이 있습니다.")
if np.isinf(arr_mixed).any():
    print("혼합 배열에 Inf가 있습니다.")
```

### 3.5. 작은 데이터로 문제 축소

대규모 배열에서 복잡한 연산 중 문제가 발생하면, 전체 데이터를 사용하여 디버깅하는 것은 매우 비효율적입니다. 이럴 때는 문제가 발생하는 로직을 재현할 수 있는 **아주 작은 크기(예: 3x3, 5x5)의 테스트용 배열**을 만들어 동일한 로직을 실행해보는 것이 매우 효과적입니다. 작은 배열은:

-   **눈으로 직접 값 추적**: 배열의 모든 요소를 쉽게 확인하고, 연산 과정에서의 변화를 추적할 수 있습니다.
-   **빠른 반복**: 코드를 수정하고 다시 실행하는 시간이 짧아 디버깅 주기를 단축할 수 있습니다.
-   **문제 격리**: 복잡한 전체 시스템에서 문제가 되는 특정 부분만 떼어내어 집중적으로 분석할 수 있습니다.

```python
import numpy as np

# 복잡한 연산 예시 (가상의 문제 상황)
def complex_operation(data_array, weights_matrix):
    # 실제 코드에서는 더 복잡한 여러 단계의 연산이 있을 수 있음
    intermediate_result = data_array @ weights_matrix
    final_result = np.sqrt(intermediate_result + 1e-6)
    return final_result

# 대규모 데이터 (디버깅이 어려움)
# large_data = np.random.rand(1000, 500)
# large_weights = np.random.rand(500, 100)
# result_large = complex_operation(large_data, large_weights)

# 작은 데이터로 문제 축소 (디버깅 용이)
small_data = np.array([[1.0, 2.0],
                       [3.0, 4.0]]) # (2, 2)
small_weights = np.array([[0.5, 0.1],
                          [0.2, 0.8]]) # (2, 2)

print(f"작은 데이터:\n{small_data}")
print(f"작은 가중치:\n{small_weights}")

# 작은 데이터로 연산 실행
result_small = complex_operation(small_data, small_weights)
print(f"\n작은 데이터 연산 결과:\n{result_small}")

# 만약 여기서 NaN이나 Inf, 혹은 예상치 못한 값이 나온다면
# small_data와 small_weights를 직접 보면서 중간 과정을 추적하기 용이함
# 예: intermediate_result = small_data @ small_weights 값을 직접 확인
```