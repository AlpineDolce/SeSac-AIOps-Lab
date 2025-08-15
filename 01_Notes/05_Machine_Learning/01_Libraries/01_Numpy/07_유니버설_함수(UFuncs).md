<h2>NumPy 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
이 문서는 NumPy의 핵심 성능 비결인 유니버설 함수(Universal Functions, ufuncs)와 벡터화(Vectorization)의 개념을 상세히 다룹니다. ufuncs가 어떻게 배열 연산을 효율적으로 수행하는지 이해하고, 실제 코드 예제를 통해 벡터화된 연산의 장점을 학습합니다. 다양한 ufuncs의 활용법과 브로드캐스팅 규칙, 그리고 성능 이점까지 포괄적으로 다룹니다.

<h2>목차</h2>

- [1. 유니버설 함수 (Universal Functions, ufuncs)와 벡터화](#1-유니버설-함수-universal-functions-ufuncs와-벡터화)
  - [1.1. ufuncs의 주요 특징](#11-ufuncs의-주요-특징)
    - [1.1.1. 성능 (Performance): C 구현 및 벡터화](#111-성능-performance-c-구현-및-벡터화)
    - [1.1.2. 다양성 (Diversity): 다양한 연산 지원](#112-다양성-diversity-다양한-연산-지원)
    - [1.1.3. 브로드캐스팅 지원 (Broadcasting Support)](#113-브로드캐스팅-지원-broadcasting-support)
    - [1.1.4. 메모리 효율성 (Memory Efficiency)](#114-메모리-효율성-memory-efficiency)
  - [1.2. 다양한 ufuncs 예시](#12-다양한-ufuncs-예시)
    - [1.2.1. 산술 연산 (Arithmetic Operations)](#121-산술-연산-arithmetic-operations)
    - [1.2.2. 삼각 및 지수/로그 함수 (Trigonometric, Exponential/Logarithmic Functions)](#122-삼각-및-지수로그-함수-trigonometric-exponentiellogarithmic-functions)
      - [1.2.2.1. 삼각 함수 (Trigonometric Functions)](#12221-삼각-함수-trigonometric-functions)
      - [1.2.2.2. 지수 및 로그 함수 (Exponential and Logarithmic Functions)](#12222-지수-및-로그-함수-exponential-and-logarithmic-functions)
    - [1.2.3. 비교 연산 (Comparison Operations)](#123-비교-연산-comparison-operations)
    - [1.2.4. 논리 연산 (Logical Operations)](#124-논리-연산-logical-operations)
    - [1.2.5. 비트 연산 (Bitwise Operations)](#125-비트-연산-bitwise-operations)
    - [1.2.6. 단항 ufuncs (Unary ufuncs)](#126-단항-ufuncs-unary-ufuncs)
    - [1.2.7. 이항 ufuncs (Binary ufuncs)](#127-이항-ufuncs-binary-ufuncs)
    - [1.2.8. 조건부 선택 (`np.where`)](#128-조건부-선택-npwhere)
    - [1.2.9. 배열 조작 및 집합 연산 (Array Manipulation and Set Operations)](#129-배열-조작-및-집합-연산-array-manipulation-and-set-operations)
    - [1.2.10. 사용자 정의 ufuncs 생성 (`np.frompyfunc`)](#1210-사용자-정의-ufuncs-생성-npfrompyfunc)
  - [1.3. 브로드캐스팅(Broadcasting)과 ufuncs](#13-브로드캐스팅broadcasting과-ufuncs)
  - [1.4. ufuncs의 성능 이점 (벡터화)](#14-ufuncs의-성능-이점-벡터화)
  - [1.5. ufuncs의 메서드 (Methods of ufuncs)](#15-ufuncs의-메서드-methods-of-ufuncs)
    - [1.5.1. `reduce` 메서드](#151-reduce-메서드)
    - [1.5.2. `accumulate` 메서드](#152-accumulate-메서드)
    - [1.5.3. `outer` 메서드](#153-outer-메서드)
    - [1.5.4. `at` 메서드](#154-at-메서드)

---

## 1. 유니버설 함수 (Universal Functions, ufuncs)와 벡터화

NumPy의 핵심 성능 비결 중 하나는 **유니버설 함수(Universal Functions, ufuncs)**입니다. ufunc는 `ndarray` 객체 내의 모든 요소에 대해 빠른 C 레벨의 연산을 수행하는 함수입니다. 사용자가 파이썬 `for` 루프를 작성할 필요 없이, 배열 전체에 대한 연산을 간결하게 표현하는 것을 **벡터화(Vectorization)**라고 하며, 이는 ufuncs를 통해 구현됩니다.

**ufuncs와 벡터화의 중요성:**

데이터 과학 및 수치 계산 분야에서 대규모 배열 데이터를 효율적으로 처리하는 것은 매우 중요합니다. 파이썬의 기본 `for` 루프는 유연하지만, 대량의 데이터에 대해 반복적인 연산을 수행할 때 성능 병목 현상을 일으킬 수 있습니다. NumPy는 이러한 문제를 해결하기 위해 ufuncs와 벡터화라는 강력한 개념을 도입했습니다.

*   **ufuncs**: C 언어로 구현된 고성능 함수로, 배열의 각 요소에 대해 빠르고 효율적인 연산을 수행합니다. 이는 파이썬의 느린 `for` 루프를 대체하여 연산 속도를 비약적으로 향상시킵니다.
*   **벡터화**: 명시적인 반복문 없이 배열 전체에 연산을 적용하는 프로그래밍 스타일입니다. ufuncs를 통해 벡터화된 코드를 작성하면, 코드가 더 간결하고 읽기 쉬워질 뿐만 아니라, NumPy 내부의 최적화된 C 구현 덕분에 훨씬 빠르게 실행됩니다.

이 문서는 ufuncs와 벡터화의 기본 개념부터 다양한 활용 예시, 그리고 성능 이점까지 포괄적으로 다루어, NumPy를 활용한 효율적인 데이터 처리에 대한 깊이 있는 이해를 돕고자 합니다.

### 1.1. ufuncs의 주요 특징

ufuncs는 NumPy 배열 연산의 핵심이며, 다음과 같은 주요 특징을 가집니다.

#### 1.1.1. 성능 (Performance): C 구현 및 벡터화

ufuncs의 가장 큰 장점은 뛰어난 성능입니다. 이는 주로 두 가지 요인에 기인합니다.

*   **C 구현**: NumPy의 ufuncs는 내부적으로 C 언어로 구현된 최적화된 루프를 사용합니다. 파이썬은 인터프리터 언어이기 때문에 `for` 루프와 같은 반복문에서 오버헤드가 발생하여 대규모 데이터 처리 시 속도가 느려질 수 있습니다. 하지만 ufuncs는 이러한 파이썬 인터프리터의 제약을 우회하고, 컴파일된 C 코드의 속도로 연산을 수행합니다.
    *   **GIL (Global Interpreter Lock) 우회**: 파이썬의 GIL은 한 번에 하나의 스레드만 파이썬 바이트코드를 실행하도록 제한합니다. 하지만 NumPy의 C로 구현된 연산은 GIL을 해제할 수 있어, 멀티코어 CPU에서 병렬 처리가 가능해지며, 이는 대규모 배열 연산의 속도를 더욱 향상시킵니다.
*   **벡터화**: ufuncs는 배열의 각 요소를 개별적으로 처리하는 대신, 배열 전체에 대한 연산을 한 번에 처리하는 **벡터화(Vectorization)**를 가능하게 합니다. 이는 명시적인 반복문(for 루프) 없이도 배열의 모든 요소에 대해 연산을 적용하므로 코드의 가독성을 높이고 실행 속도를 극대화합니다.
    *   **SIMD (Single Instruction, Multiple Data) 활용**: 현대 CPU는 SIMD 명령어를 지원하여 단일 명령어로 여러 데이터 요소를 동시에 처리할 수 있습니다. ufuncs는 이러한 SIMD 명령어를 효율적으로 활용하여 병렬 연산을 수행함으로써 성능을 극대화합니다.
    *   **캐시 효율성**: NumPy 배열은 메모리에 연속적으로 저장되는 경우가 많습니다. ufuncs는 이러한 연속적인 메모리 접근을 통해 CPU 캐시의 효율성을 높여 데이터 로딩 시간을 줄이고 연산 속도를 향상시킵니다.

#### 1.1.2. 다양성 (Diversity): 다양한 연산 지원

NumPy는 광범위한 수학적, 논리적, 비트 연산을 포함하는 수백 가지의 다양한 ufuncs를 미리 구현하여 제공합니다. 이는 사용자가 복잡한 연산을 직접 구현할 필요 없이, 내장된 고성능 함수를 활용할 수 있도록 합니다.

*   **산술 연산**: `np.add` (덧셈), `np.subtract` (뺄셈), `np.multiply` (곱셈), `np.divide` (나눗셈), `np.power` (거듭제곱), `np.mod` (나머지), `np.floor_divide` (몫) 등 기본적인 사칙연산부터 복잡한 연산까지 지원합니다.
*   **삼각 함수**: `np.sin`, `np.cos`, `np.tan`, `np.arcsin`, `np.arccos`, `np.arctan` 등 다양한 삼각 함수를 포함합니다.
*   **지수/로그 함수**: `np.exp` (자연 지수), `np.log` (자연로그), `np.log10` (상용로그), `np.log2` (이진로그) 등을 제공합니다.
*   **비교 연산**: `np.greater` (크다), `np.less` (작다), `np.equal` (같다), `np.not_equal` (같지 않다), `np.greater_equal` (크거나 같다), `np.less_equal` (작거나 같다) 등 배열 요소 간의 비교를 수행하여 불리언 배열을 반환합니다.
*   **논리 연산**: `np.logical_and`, `np.logical_or`, `np.logical_not`, `np.logical_xor` 등 불리언 배열에 대한 논리 연산을 수행합니다.
*   **비트 연산**: `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor`, `np.bitwise_not`, `np.left_shift`, `np.right_shift` 등 비트 단위 연산을 지원합니다.
*   **단항 ufuncs**: `np.abs` (절대값), `np.sqrt` (제곱근), `np.ceil` (올림), `np.floor` (내림), `np.round` (반올림), `np.sign` (부호) 등 단일 입력 배열에 대해 작동하는 함수들입니다.
*   **이항 ufuncs**: `np.add`, `np.multiply`, `np.maximum`, `np.minimum` 등 두 개의 입력 배열에 대해 작동하는 함수들입니다.

#### 1.1.3. 브로드캐스팅 지원 (Broadcasting Support)

ufuncs는 NumPy의 **브로드캐스팅(Broadcasting)** 규칙을 자동으로 지원합니다. 이는 형태(shape)가 다른 배열 간에도 연산을 유연하게 수행할 수 있도록 하는 강력한 메커니즘입니다.

*   **형태가 다른 배열 연산**: 브로드캐스팅 규칙에 따라, NumPy는 연산에 참여하는 배열들의 형태를 자동으로 확장하여 호환 가능하게 만듭니다. 예를 들어, 2차원 배열과 스칼라 값 또는 1차원 배열 간의 연산 시, NumPy는 작은 배열을 큰 배열의 형태에 맞춰 가상으로 확장하여 연산을 가능하게 합니다. 이 과정에서 실제 메모리 복사는 발생하지 않아 메모리 효율적입니다.
*   **코드 간결성**: 명시적으로 배열의 형태를 맞추거나 복잡한 반복문을 사용할 필요가 없어 코드가 훨씬 간결하고 읽기 쉬워집니다. 이는 개발 생산성을 크게 향상시킵니다.

#### 1.1.4. 메모리 효율성 (Memory Efficiency)

ufuncs는 불필요한 임시 배열 생성을 최소화하여 대규모 데이터셋 처리 시 효율성을 높입니다.

*   **불필요한 임시 배열 생성 최소화**: ufuncs는 연산 결과를 직접 출력 배열에 쓰거나, 인플레이스(in-place) 연산을 지원하여 불필요한 중간 배열 생성을 줄입니다. 이는 메모리 할당 및 해제 오버헤드를 감소시킵니다.
*   **인플레이스(in-place) 연산**: 많은 ufuncs는 `out` 인수를 통해 결과를 특정 배열에 직접 저장하거나, `+=`, `*=`와 같은 복합 할당 연산자를 통해 인플레이스 연산을 지원합니다. 예를 들어, `arr += 1`은 `arr = arr + 1`과 달리 새로운 배열을 생성하지 않고 `arr`의 내용을 직접 수정합니다.
    ```python
    import numpy as np

    arr = np.array([1, 2, 3])
    print(f"원본 arr: {arr}, id: {id(arr)}")

    arr += 1 # 인플레이스 연산
    print(f"arr += 1 후: {arr}, id: {id(arr)}") # id가 동일

    arr = arr + 1 # 새로운 배열 생성
    print(f"arr = arr + 1 후: {arr}, id: {id(arr)}") # id가 변경됨
    ```
*   **최적화된 메모리 접근**: NumPy 배열은 데이터를 연속된 메모리 블록에 저장하는 경우가 많습니다. ufuncs는 이러한 연속적인 메모리 레이아웃을 활용하여 CPU 캐시의 효율성을 극대화하고, 데이터 접근 속도를 향상시킵니다.

### 1.2. 다양한 ufuncs 예시

이 섹션에서는 NumPy에서 제공하는 다양한 유니버설 함수(ufuncs)의 구체적인 사용 예시를 살펴봅니다. 각 ufunc는 배열의 모든 요소에 대해 동일한 연산을 효율적으로 적용하며, 파이썬의 표준 연산자(`+`, `-`, `*` 등)는 내부적으로 해당 ufunc를 호출하는 문법적 설탕(syntactic sugar)입니다. 이러한 예시들을 통해 ufuncs의 강력함과 유연성을 이해하고, 실제 데이터 처리 작업에 효과적으로 적용하는 방법을 학습할 수 있습니다.

#### 1.2.1. 산술 연산 (Arithmetic Operations)

NumPy는 배열 간의 기본적인 산술 연산을 위한 다양한 ufunc를 제공합니다. 이러한 연산은 기본적으로 요소별(element-wise)로 수행되며, 파이썬의 표준 산술 연산자(`+`, `-`, `*`, `/`, `//`, `**`, `%`)를 통해 간편하게 사용할 수 있습니다. 이 연산자들은 내부적으로 해당 ufunc를 호출합니다.

**주요 산술 ufuncs:**
*   `np.add` 또는 `+`: 덧셈
*   `np.subtract` 또는 `-`: 뺄셈
*   `np.multiply` 또는 `*`: 곱셈
*   `np.divide` 또는 `/`: 나눗셈 (결과는 항상 부동소수점)
*   `np.floor_divide` 또는 `//`: 몫 (정수 나눗셈)
*   `np.power` 또는 `**`: 거듭제곱
*   `np.mod` 또는 `%`: 나머지 (모듈로 연산)
*   `np.remainder`: `np.mod`와 동일한 나머지 연산
*   `np.divmod`: 몫과 나머지를 동시에 반환

**예시 1: 요소별 덧셈**
```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")

# 연산자 사용
sum_arr_op = arr1 + arr2
print(f"\n요소별 덧셈 (연산자): {sum_arr_op}")

# ufunc 직접 호출
sum_arr_ufunc = np.add(arr1, arr2)
print(f"요소별 덧셈 (ufunc): {sum_arr_ufunc}")
```
**설명:**
NumPy 배열 간의 덧셈은 요소별로 수행됩니다. `arr1 + arr2`는 `np.add(arr1, arr2)`와 동일하며, 각 배열의 같은 위치에 있는 요소들을 더합니다.

**결과:**
```
arr1: [1 2 3]
arr2: [4 5 6]

요소별 덧셈 (연산자): [5 7 9]
요소별 덧셈 (ufunc): [5 7 9]
```

**예시 2: 요소별 뺄셈**
```python
# 연산자 사용
sub_arr_op = arr1 - arr2
print(f"\n요소별 뺄셈 (연산자): {sub_arr_op}")

# ufunc 직접 호출
sub_arr_ufunc = np.subtract(arr1, arr2)
print(f"요소별 뺄셈 (ufunc): {sub_arr_ufunc}")
```
**설명:**
뺄셈도 덧셈과 마찬가지로 요소별로 수행됩니다. `arr1 - arr2`는 `np.subtract(arr1, arr2)`와 동일합니다.

**결과:**
```
요소별 뺄셈 (연산자): [-3 -3 -3]
요소별 뺄셈 (ufunc): [-3 -3 -3]
```

**예시 3: 요소별 곱셈**
```python
# 연산자 사용
mul_arr_op = arr1 * arr2
print(f"\n요소별 곱셈 (연산자): {mul_arr_op}")

# ufunc 직접 호출
mul_arr_ufunc = np.multiply(arr1, arr2)
print(f"요소별 곱셈 (ufunc): {mul_arr_ufunc}")
```
**설명:**
곱셈 또한 요소별로 수행됩니다. `arr1 * arr2`는 `np.multiply(arr1, arr2)`와 동일합니다.

**결과:**
```
요소별 곱셈 (연산자): [ 4 10 18]
요소별 곱셈 (ufunc): [ 4 10 18]
```

**예시 4: 요소별 나눗셈**
```python
# 연산자 사용
div_arr_op = arr1 / arr2
print(f"\n요소별 나눗셈 (연산자): {div_arr_op}")

# ufunc 직접 호출
div_arr_ufunc = np.divide(arr1, arr2)
print(f"요소별 나눗셈 (ufunc): {div_arr_ufunc}")
```
**설명:**
나눗셈은 요소별로 수행되며, 결과는 항상 부동소수점 형태입니다. `arr1 / arr2`는 `np.divide(arr1, arr2)`와 동일합니다.

**결과:**
```
요소별 나눗셈 (연산자): [0.25 0.4  0.5 ]
요소별 나눗셈 (ufunc): [0.25 0.4  0.5 ]
```

**예시 5: 요소별 몫 (정수 나눗셈)**
```python
# 연산자 사용
floor_div_arr_op = arr2 // arr1 # arr2를 arr1으로 나눈 몫
print(f"\n요소별 몫 (연산자): {floor_div_arr_op}")

# ufunc 직접 호출
floor_div_arr_ufunc = np.floor_divide(arr2, arr1)
print(f"요소별 몫 (ufunc): {floor_div_arr_ufunc}")
```
**설명:**
몫 연산은 요소별로 수행되며, 결과는 정수 부분만 반환합니다. `arr2 // arr1`는 `np.floor_divide(arr2, arr1)`와 동일합니다.

**결과:**
```
요소별 몫 (연산자): [4 2 2]
요소별 몫 (ufunc): [4 2 2]
```

**예시 6: 요소별 거듭제곱**
```python
# 연산자 사용
pow_arr_op = arr1 ** 2 # arr1의 각 요소를 제곱
print(f"\n요소별 거듭제곱 (연산자): {pow_arr_op}")

# ufunc 직접 호출
pow_arr_ufunc = np.power(arr1, 2)
print(f"요소별 거듭제곱 (ufunc): {pow_arr_ufunc}")
```
**설명:**
거듭제곱 연산은 요소별로 수행됩니다. `arr1 ** 2`는 `np.power(arr1, 2)`와 동일합니다.

**결과:**
```
요소별 거듭제곱 (연산자): [1 4 9]
요소별 거듭제곱 (ufunc): [1 4 9]
```

**예시 7: 요소별 나머지 (모듈로 연산)**
```python
# 연산자 사용
mod_arr_op = arr2 % arr1 # arr2를 arr1으로 나눈 나머지
print(f"\n요소별 나머지 (연산자): {mod_arr_op}")

# ufunc 직접 호출
mod_arr_ufunc = np.mod(arr2, arr1)
print(f"요소별 나머지 (ufunc): {mod_arr_ufunc}")

# np.remainder는 np.mod와 동일합니다.
remainder_ufunc = np.remainder(arr2, arr1)
print(f"요소별 나머지 (np.remainder): {remainder_ufunc}")
```
**설명:**
나머지 연산은 요소별로 수행됩니다. `arr2 % arr1`는 `np.mod(arr2, arr1)`와 동일하며, `np.remainder`도 같은 기능을 합니다.

**결과:**
```
요소별 나머지 (연산자): [0 1 0]
요소별 나머지 (ufunc): [0 1 0]
요소별 나머지 (np.remainder): [0 1 0]
```

**예시 8: `np.divmod` (몫과 나머지 동시 반환)**
```python
import numpy as np

arr_divmod1 = np.array([10, 11, 12])
arr_divmod2 = np.array([3, 4, 5])

# np.divmod는 몫과 나머지를 튜플 형태로 반환합니다.
quotient, remainder = np.divmod(arr_divmod1, arr_divmod2)
print(f"\n원본 배열 1: {arr_divmod1}")
print(f"원본 배열 2: {arr_divmod2}")
print(f"몫 (np.divmod): {quotient}")
print(f"나머지 (np.divmod): {remainder}")
```
**설명:**
`np.divmod(x1, x2)`는 `x1 // x2`와 `x1 % x2`의 결과를 각각 튜플의 첫 번째와 두 번째 요소로 반환합니다.

**결과:**
```
원본 배열 1: [10 11 12]
원본 배열 2: [3 4 5]
몫 (np.divmod): [3 2 2]
나머지 (np.divmod): [1 3 2]
```

**나눗셈 시 특이 케이스 (0으로 나누기):**

정수형 배열에서 0으로 나누면 `RuntimeWarning`이 발생하고 결과는 `inf` (무한대) 또는 `nan` (Not a Number)이 될 수 있습니다. 부동소수점형 배열에서는 경고 없이 `inf` 또는 `nan`이 반환됩니다.

```python
import numpy as np

arr_zero_div = np.array([1, 0, -1])
arr_zero = np.array([0, 0, 0])

# 0으로 나누기 (정수형)
result_int_div_zero = arr_zero_div // arr_zero # RuntimeWarning 발생
print(f"\n정수형 0으로 나누기 (몫): {result_int_div_zero}")

# 0으로 나누기 (부동소수점형)
result_float_div_zero = arr_zero_div / arr_zero # RuntimeWarning 발생
print(f"부동소수점형 0으로 나누기: {result_float_div_zero}")
```
**결과:**
```
정수형 0으로 나누기 (몫): [9223372036854775807           0 -9223372036854775808] # 플랫폼에 따라 다를 수 있음
부동소수점형 0으로 나누기: [ inf  nan -inf]
```

#### 1.2.2. 삼각 및 지수/로그 함수 (Trigonometric, Exponential/Logarithmic Functions)

NumPy는 배열의 각 요소에 대해 삼각 함수, 지수 함수, 로그 함수를 적용하는 다양한 ufunc를 제공합니다. 이 함수들은 수학 및 과학 계산에서 매우 중요하게 활용됩니다.

##### 1.2.2.1. 삼각 함수 (Trigonometric Functions)

*   `np.sin(x)`: x의 사인 값
*   `np.cos(x)`: x의 코사인 값
*   `np.tan(x)`: x의 탄젠트 값
*   `np.arcsin(x)`: x의 아크사인 값 (역사인)
*   `np.arccos(x)`: x의 아크코사인 값 (역코사인)
*   `np.arctan(x)`: x의 아크탄젠트 값 (역탄젠트)
*   `np.degrees(x)` 또는 `np.rad2deg(x)`: 라디안 값을 도로 변환
*   `np.radians(x)` 또는 `np.deg2rad(x)`: 도 값을 라디안으로 변환
*   `np.sinh(x)`: x의 하이퍼볼릭 사인 값
*   `np.cosh(x)`: x의 하이퍼볼릭 코사인 값
*   `np.tanh(x)`: x의 하이퍼볼릭 탄젠트 값
*   `np.arcsinh(x)`: x의 역하이퍼볼릭 사인 값
*   `np.arccosh(x)`: x의 역하이퍼볼릭 코사인 값
*   `np.arctanh(x)`: x의 역하이퍼볼릭 탄젠트 값

**예시 1: 기본 삼각 함수 적용**
```python
import numpy as np

angles_rad = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi])
print(f"각도 (라디안): {angles_rad}")

sin_values = np.sin(angles_rad)
cos_values = np.cos(angles_rad)
tan_values = np.tan(angles_rad)

print(f"\n사인 값: {sin_values}")
print(f"코사인 값: {cos_values}")
print(f"탄젠트 값: {tan_values}")
```
**설명:**
`np.sin`, `np.cos`, `np.tan`은 입력 배열의 각 요소에 대해 해당 삼각 함수 값을 계산합니다. `np.pi`는 파이(π) 값을 나타냅니다.

**결과:**
```
각도 (라디안): [0.         0.52359878 0.78539816 1.04719755 1.57079633 3.14159265]

사인 값: [0.         0.5        0.70710678 0.8660254  1.         0.        ]
코사인 값: [ 1.00000000e+00  8.66025404e-01  7.07106781e-01  5.00000000e-01
  6.12323400e-17 -1.00000000e+00]
탄젠트 값: [ 0.00000000e+00  5.77350269e-01  1.00000000e-00  1.73205081e-00
  1.63312395e+16 -1.22464680e-16]
```

**예시 2: 역삼각 함수 및 각도 변환**
```python
import numpy as np

values = np.array([0, 0.5, 1])

arcsin_values = np.arcsin(values)
arccos_values = np.arccos(values)
arctan_values = np.arctan(values)

print(f"값: {values}")
print(f"\n아크사인 값 (라디안): {arcsin_values}")
print(f"아크코사인 값 (라디안): {arccos_values}")
print(f"아크탄젠트 값 (라디안): {arctan_values}")

# 라디안을 도로 변환 (np.degrees 또는 np.rad2deg)
degrees_values = np.degrees(arcsin_values)
rad2deg_values = np.rad2deg(arcsin_values)
print(f"아크사인 값 (도, np.degrees): {degrees_values}")
print(f"아크사인 값 (도, np.rad2deg): {rad2deg_values}")

# 도를 라디안으로 변환 (np.radians 또는 np.deg2rad)
radians_values = np.radians(np.array([0, 30, 45, 60, 90]))
deg2rad_values = np.deg2rad(np.array([0, 30, 45, 60, 90]))
print(f"도에서 라디안으로 변환 (np.radians): {radians_values}")
print(f"도에서 라디안으로 변환 (np.deg2rad): {deg2rad_values}")
```
**설명:**
역삼각 함수(`np.arcsin`, `np.arccos`, `np.arctan`)는 주어진 삼각 함수 값에 해당하는 각도(라디안)를 반환합니다. `np.degrees`와 `np.rad2deg`는 라디안을 도로, `np.radians`와 `np.deg2rad`는 도를 라디안으로 변환합니다.

**결과:**
```
값: [0.  0.5 1. ]

아크사인 값 (라디안): [0.         0.52359878 1.57079633]
아크코사인 값 (라디안): [1.57079633 1.04719755 0.        ]
아크탄젠트 값 (라디안): [0.         0.46364761 0.78539816]
아크사인 값 (도, np.degrees): [ 0. 30. 90.]
아크사인 값 (도, np.rad2deg): [ 0. 30. 90.]
도에서 라디안으로 변환 (np.radians): [0.         0.52359878 0.78539816 1.04719755 1.57079633]
도에서 라디안으로 변환 (np.deg2rad): [0.         0.52359878 0.78539816 1.04719755 1.57079633]
```

**예시 3: 하이퍼볼릭 삼각 함수**
```python
import numpy as np

x_hyper = np.array([0, 1, 2])

sinh_values = np.sinh(x_hyper)
cosh_values = np.cosh(x_hyper)
tanh_values = np.tanh(x_hyper)

print(f"x: {x_hyper}")
print(f"\n하이퍼볼릭 사인 (np.sinh): {sinh_values}")
print(f"하이퍼볼릭 코사인 (np.cosh): {cosh_values}")
print(f"하이퍼볼릭 탄젠트 (np.tanh): {tanh_values}")

# 역하이퍼볼릭 함수
arcsinh_values = np.arcsinh(sinh_values)
arccosh_values = np.arccosh(cosh_values)
arctanh_values = np.arctanh(tanh_values)

print(f"\n역하이퍼볼릭 사인 (np.arcsinh): {arcsinh_values}")
print(f"역하이퍼볼릭 코사인 (np.arccosh): {arccosh_values}")
print(f"역하이퍼볼릭 탄젠트 (np.arctanh): {arctanh_values}")
```
**설명:**
`np.sinh`, `np.cosh`, `np.tanh`는 각각 하이퍼볼릭 사인, 코사인, 탄젠트 함수를 계산합니다. `np.arcsinh`, `np.arccosh`, `np.arctanh`는 이들의 역함수입니다.

**결과:**
```
x: [0 1 2]

하이퍼볼릭 사인 (np.sinh): [0.         1.17520119 3.62686041]
하이퍼볼릭 코사인 (np.cosh): [1.         1.54308063 3.76219569]
하이퍼볼릭 탄젠트 (np.tanh): [0.         0.76159416 0.96402758]

역하이퍼볼릭 사인 (np.arcsinh): [0. 1. 2.]
역하이퍼볼릭 코사인 (np.arccosh): [0. 1. 2.]
역하이퍼볼릭 탄젠트 (np.arctanh): [0. 1. 2.]
```

##### 1.2.2.2. 지수 및 로그 함수 (Exponential and Logarithmic Functions)

*   `np.exp(x)`: 자연 상수 `e`를 밑으로 하는 `x`의 지수 함수 ($e^x$)
*   `np.expm1(x)`: $e^x - 1$ (작은 `x` 값에 대해 `np.exp(x) - 1`보다 더 정확함)
*   `np.log(x)`: 자연 로그 (밑이 `e`인 로그)
*   `np.log10(x)`: 상용 로그 (밑이 10인 로그)
*   `np.log2(x)`: 이진 로그 (밑이 2인 로그)
*   `np.log1p(x)`: $\ln(1+x)$ (작은 `x` 값에 대해 `np.log(1+x)`보다 더 정확함)

**예시 3: 지수 및 로그 함수 적용**
```python
import numpy as np

values_exp_log = np.array([1, 2, 3])

exp_values = np.exp(values_exp_log)
print(f"\n지수 함수 (np.exp) 적용: {exp_values}")

log_values = np.log(exp_values)
print(f"자연 로그 (np.log) 적용: {log_values}")

log10_values = np.log10(np.array([10, 100, 1000]))
print(f"상용 로그 (np.log10) 적용: {log10_values}")

log2_values = np.log2(np.array([2, 4, 8]))
print(f"이진 로그 (np.log2) 적용: {log2_values}")
```
**설명:**
`np.exp`는 자연 상수 `e`를 밑으로 하는 지수 함수를 계산합니다. `np.log`, `np.log10`, `np.log2`는 각각 자연 로그, 상용 로그, 이진 로그를 계산합니다. 이들은 서로 역함수 관계에 있습니다.

**결과:**
```
지수 함수 (np.exp) 적용: [ 2.71828183  7.3890561  20.08553692]
자연 로그 (np.log) 적용: [1. 2. 3.]
상용 로그 (np.log10) 적용: [1. 2. 3.]
이진 로그 (np.log2) 적용: [1. 2. 3.]
```

**예시 4: `np.expm1` 및 `np.log1p` (정확도 향상)**
```python
import numpy as np

x_small = np.array([1e-5, 1e-6])

# np.exp(x) - 1 vs np.expm1(x)
exp_minus_1 = np.exp(x_small) - 1
expm1_values = np.expm1(x_small)
print(f"\nx_small: {x_small}")
print(f"np.exp(x) - 1: {exp_minus_1}")
print(f"np.expm1(x): {expm1_values}")

# np.log(1 + x) vs np.log1p(x)
log_plus_1 = np.log(1 + x_small)
log1p_values = np.log1p(x_small)
print(f"\nnp.log(1 + x): {log_plus_1}")
print(f"np.log1p(x): {log1p_values}")
```
**설명:**
`np.expm1(x)`은 $e^x - 1$을, `np.log1p(x)`는 $\ln(1+x)$를 계산합니다. 이들은 `x`가 0에 매우 가까울 때 `np.exp(x) - 1`이나 `np.log(1 + x)`보다 더 높은 정밀도를 제공합니다. 부동소수점 연산의 한계로 인해 작은 값에서 정밀도 손실이 발생할 수 있는데, 이 함수들은 이를 방지합니다.

**결과:**
```
x_small: [1.e-05 1.e-06]
np.exp(x) - 1: [1.00000500e-05 1.00000050e-06]
np.expm1(x): [1.00000500e-05 1.00000050e-06]

np.log(1 + x): [1.00000000e-05 1.00000000e-06]
np.log1p(x): [1.00000000e-05 1.00000000e-06]
```

#### 1.2.3. 비교 연산 (Comparison Operations)

NumPy는 배열의 요소들을 비교하기 위한 다양한 유니버설 함수(ufunc)를 제공합니다. 이러한 비교 연산은 기본적으로 요소별(element-wise)로 수행되며, 결과로 불리언(Boolean) 배열을 반환합니다. 파이썬의 표준 비교 연산자(`>`, `<`, `==`, `!=`, `>=`, `<=`)를 통해 간편하게 사용할 수 있으며, 이 연산자들은 내부적으로 해당 ufunc를 호출합니다.

**주요 비교 ufuncs:**
*   `np.equal` 또는 `==`: 두 배열의 요소가 같은지 비교
*   `np.not_equal` 또는 `!=`: 두 배열의 요소가 다른지 비교
*   `np.greater` 또는 `>`: 첫 번째 배열의 요소가 두 번째 배열의 요소보다 큰지 비교
*   `np.greater_equal` 또는 `>=`: 첫 번째 배열의 요소가 두 번째 배열의 요소보다 크거나 같은지 비교
*   `np.less` 또는 `<`: 첫 번째 배열의 요소가 두 번째 배열의 요소보다 작은지 비교
*   `np.less_equal` 또는 `<=`: 첫 번째 배열의 요소가 두 번째 배열의 요소보다 작거나 같은지 비교

**예시 1: `np.equal` (같음 비교)**

```python
import numpy as np

arr_comp1 = np.array([1, 2, 3, 4, 5])
arr_comp2 = np.array([1, 2, 0, 4, 6])

print(f"arr_comp1: {arr_comp1}")
print(f"arr_comp2: {arr_comp2}")

# 연산자 사용
equal_op = arr_comp1 == arr_comp2
print(f"\n요소별 같음 (연산자): {equal_op}")

# ufunc 직접 호출
equal_ufunc = np.equal(arr_comp1, arr_comp2)
print(f"요소별 같음 (ufunc): {equal_ufunc}")
```
**설명:**
`np.equal` 또는 `==` 연산자는 두 배열의 각 요소가 서로 같은지 비교하여 불리언 배열을 반환합니다. 형태가 다른 배열 간에도 브로드캐스팅 규칙이 적용됩니다.

**결과:**
```
arr_comp1: [1 2 3 4 5]
arr_comp2: [1 2 0 4 6]

요소별 같음 (연산자): [ True  True False  True False]
요소별 같음 (ufunc): [ True  True False  True False]
```

**예시 2: `np.greater` (큼 비교)**

```python
# 연산자 사용
greater_op = arr_comp1 > arr_comp2
print(f"\n요소별 큼 (연산자): {greater_op}")

# ufunc 직접 호출
greater_ufunc = np.greater(arr_comp1, arr_comp2)
print(f"요소별 큼 (ufunc): {greater_ufunc}")
```
**설명:**
`np.greater` 또는 `>` 연산자는 첫 번째 배열의 요소가 두 번째 배열의 요소보다 큰지 비교하여 불리언 배열을 반환합니다.

**결과:**
```
요소별 큼 (연산자): [False False  True False False]
요소별 큼 (ufunc): [False False  True False False]
```

**예시 3: `np.less_equal` (작거나 같음 비교)**

```python
# 연산자 사용
less_equal_op = arr_comp1 <= arr_comp2
print(f"\n요소별 작거나 같음 (연산자): {less_equal_op}")

# ufunc 직접 호출
less_equal_ufunc = np.less_equal(arr_comp1, arr_comp2)
print(f"요소별 작거나 같음 (ufunc): {less_equal_ufunc}")
```
**설명:**
`np.less_equal` 또는 `<=` 연산자는 첫 번째 배열의 요소가 두 번째 배열의 요소보다 작거나 같은지 비교하여 불리언 배열을 반환합니다.

**결과:**
```
요소별 작거나 같음 (연산자): [ True  True False  True  True]
요소별 작거나 같음 (ufunc): [ True  True False  True  True]
```

**예시 4: 스칼라와의 비교 연산**

```python
import numpy as np

arr_scalar_comp = np.array([10, 20, 30, 40, 50])
scalar_val = 30

print(f"arr_scalar_comp: {arr_scalar_comp}")
print(f"scalar_val: {scalar_val}")

# 30보다 큰 요소
result_greater_than_30 = arr_scalar_comp > scalar_val
print(f"\n30보다 큰 요소: {result_greater_than_30}")

# 30과 같지 않은 요소
result_not_equal_30 = arr_scalar_comp != scalar_val
print(f"30과 같지 않은 요소: {result_not_equal_30}")
```
**설명:**
비교 연산도 브로드캐스팅을 지원합니다. 배열과 스칼라 값을 비교할 때, 스칼라 값은 배열의 모든 요소에 대해 브로드캐스팅되어 비교가 수행됩니다.

**결과:**
```
arr_scalar_comp: [10 20 30 40 50]
scalar_val: 30

30보다 큰 요소: [False False False  True  True]
30과 같지 않은 요소: [ True  True False  True  True]
```

#### 1.2.4. 논리 연산 (Logical Operations)

NumPy는 요소별 논리 연산을 위한 ufunc를 제공합니다. 이들은 주로 불리언 배열에 적용되어 새로운 불리언 배열을 생성합니다.

**파이썬 논리 연산자(`and`, `or`, `not`)와 NumPy 논리 연산자(`&`, `|`, `~`)의 차이:**

매우 중요하게 이해해야 할 점은 파이썬의 기본 논리 연산자(`and`, `or`, `not`)는 NumPy 배열에 대해 요소별로 작동하지 않는다는 것입니다. 이들은 배열 전체의 참/거짓 값(truthiness)을 평가하려고 시도하며, 여러 요소가 있는 배열에 직접 적용하면 `ValueError`를 발생시킵니다. 반면, NumPy는 요소별 논리 연산을 위해 비트wise 연산자(`&`, `|`, `~`)를 오버로드하여 사용하거나, `np.logical_` 접두사가 붙은 ufunc를 사용해야 합니다.

*   `&` (비트wise AND) 또는 `np.logical_and`: 요소별 논리 AND
*   `|` (비트wise OR) 또는 `np.logical_or`: 요소별 논리 OR
*   `~` (비트wise NOT) 또는 `np.logical_not`: 요소별 논리 NOT
*   `np.logical_xor`: 요소별 논리 XOR

**예시 1: `np.logical_and` (논리 AND)**

```python
import numpy as np

arr_bool1 = np.array([True, False, True])
arr_bool2 = np.array([True, True, False])

print(f"arr_bool1: {arr_bool1}")
print(f"arr_bool2: {arr_bool2}")

# 연산자 사용
result_and_op = arr_bool1 & arr_bool2
print(f"\n논리 AND 결과 (연산자): {result_and_op}")

# ufunc 직접 호출
result_and_ufunc = np.logical_and(arr_bool1, arr_bool2)
print(f"논리 AND 결과 (ufunc): {result_and_ufunc}")
# 결과: [ True False False]
```

**예시 2: `np.logical_or` (논리 OR)**

```python
# 연산자 사용
result_or_op = arr_bool1 | arr_bool2
print(f"\n논리 OR 결과 (연산자): {result_or_op}")

# ufunc 직접 호출
result_or_ufunc = np.logical_or(arr_bool1, arr_bool2)
print(f"논리 OR 결과 (ufunc): {result_or_ufunc}")
# 결과: [ True  True  True]
```

**예시 3: `np.logical_not` (논리 NOT)**

```python
# 연산자 사용
result_not_op = ~arr_bool1
print(f"\n논리 NOT 결과 (연산자): {result_not_op}")

# ufunc 직접 호출
result_not_ufunc = np.logical_not(arr_bool1)
print(f"논리 NOT 결과 (ufunc): {result_not_ufunc}")
# 결과: [False  True False]
```

**예시 4: `np.logical_xor` (논리 XOR)**

```python
# ufunc 직접 호출 (XOR 연산자는 없음)
result_xor_ufunc = np.logical_xor(arr_bool1, arr_bool2)
print(f"\n논리 XOR 결과 (ufunc): {result_xor_ufunc}")
# 결과: [False  True  True]
```

**예시 5: 파이썬 `and`, `or` 키워드 사용 시 오류**

```python
import numpy as np

arr_test = np.array([True, False])

try:
    # 파이썬 'and' 키워드는 요소별로 작동하지 않음
    result_python_and = arr_test and True
except ValueError as e:
    print(f"\n파이썬 'and' 키워드 사용 시 오류: {e}")

try:
    # 파이썬 'or' 키워드는 요소별로 작동하지 않음
    result_python_or = arr_test or False
except ValueError as e:
    print(f"파이썬 'or' 키워드 사용 시 오류: {e}")
```
**설명:**
이 예시는 파이썬의 `and` 및 `or` 키워드가 NumPy 배열에 대해 요소별로 작동하지 않고 `ValueError`를 발생시키는 것을 보여줍니다. NumPy 배열에 대한 요소별 논리 연산에는 반드시 `&`, `|`, `~` 연산자 또는 `np.logical_` ufunc를 사용해야 합니다.

**결과:**
```
파이썬 'and' 키워드 사용 시 오류: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
파이썬 'or' 키워드 사용 시 오류: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

#### 1.2.5. 비트 연산 (Bitwise Operations)

NumPy는 배열의 요소들에 대해 비트 단위 연산을 수행하는 ufunc를 제공합니다. 이 연산들은 정수형 데이터의 이진 표현에 직접 적용됩니다. 파이썬의 비트wise 연산자(`&`, `|`, `^`, `~`, `<<`, `>>`)를 통해 간편하게 사용할 수 있으며, 이 연산자들은 내부적으로 해당 ufunc를 호출합니다.

**주요 비트 연산 ufuncs:**
*   `np.bitwise_and` 또는 `&`: 비트wise AND
*   `np.bitwise_or` 또는 `|`: 비트wise OR
*   `np.bitwise_xor` 또는 `^`: 비트wise XOR
*   `np.bitwise_not` 또는 `~`: 비트wise NOT (1의 보수)
*   `np.left_shift` 또는 `<<`: 왼쪽 비트 시프트
*   `np.right_shift` 또는 `>>`: 오른쪽 비트 시프트

**예시 1: `np.bitwise_and` (비트wise AND)**

```python
import numpy as np

arr_bit1 = np.array([5, 6], dtype=np.uint8)  # 5: 0101, 6: 0110
arr_bit2 = np.array([3, 7], dtype=np.uint8)  # 3: 0011, 7: 0111

print(f"arr_bit1: {arr_bit1} (이진: {np.unpackbits(arr_bit1.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")
print(f"arr_bit2: {arr_bit2} (이진: {np.unpackbits(arr_bit2.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")

# 연산자 사용
result_and_op = arr_bit1 & arr_bit2
print(f"\n비트wise AND (연산자): {result_and_op} (이진: {np.unpackbits(result_and_op.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")

# ufunc 직접 호출
result_and_ufunc = np.bitwise_and(arr_bit1, arr_bit2)
print(f"비트wise AND (ufunc): {result_and_ufunc} (이진: {np.unpackbits(result_and_ufunc.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")
# 5 & 3 = 1 (0101 & 0011 = 0001)
# 6 & 7 = 6 (0110 & 0111 = 0110)
```

**예시 2: `np.bitwise_or` (비트wise OR)**

```python
# 연산자 사용
result_or_op = arr_bit1 | arr_bit2
print(f"\n비트wise OR (연산자): {result_or_op} (이진: {np.unpackbits(result_or_op.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")

# ufunc 직접 호출
result_or_ufunc = np.bitwise_or(arr_bit1, arr_bit2)
print(f"비트wise OR (ufunc): {result_or_ufunc} (이진: {np.unpackbits(result_or_ufunc.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")
# 5 | 3 = 7 (0101 | 0011 = 0111)
# 6 | 7 = 7 (0110 | 0111 = 0111)
```

**예시 3: `np.bitwise_xor` (비트wise XOR)**

```python
# 연산자 사용
result_xor_op = arr_bit1 ^ arr_bit2
print(f"\n비트wise XOR (연산자): {result_xor_op} (이진: {np.unpackbits(result_xor_op.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")

# ufunc 직접 호출
result_xor_ufunc = np.bitwise_xor(arr_bit1, arr_bit2)
print(f"비트wise XOR (ufunc): {result_xor_ufunc} (이진: {np.unpackbits(result_xor_ufunc.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")
# 5 ^ 3 = 6 (0101 ^ 0011 = 0110)
# 6 ^ 7 = 1 (0110 ^ 0111 = 0001)
```

**예시 4: `np.bitwise_not` (비트wise NOT)**

```python
# 연산자 사용
result_not_op = ~arr_bit1
print(f"\n비트wise NOT (연산자): {result_not_op} (이진: {np.unpackbits(result_not_op.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")

# ufunc 직접 호출
result_not_ufunc = np.bitwise_not(arr_bit1)
print(f"비트wise NOT (ufunc): {result_not_ufunc} (이진: {np.unpackbits(result_not_ufunc.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")
# ~5 = 250 (0101 -> 11111010, 8비트 부호 없는 정수 기준)
```

**예시 5: 비트 시프트 연산**

```python
# 왼쪽 비트 시프트 (<<)
result_lshift = arr_bit1 << 1
print(f"\n왼쪽 비트 시프트 (연산자): {result_lshift} (이진: {np.unpackbits(result_lshift.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")

# 오른쪽 비트 시프트 (>>)
result_rshift = arr_bit1 >> 1
print(f"오른쪽 비트 시프트 (연산자): {result_rshift} (이진: {np.unpackbits(result_rshift.view(np.uint8)).reshape(-1, 8)[:, 4:].astype(str).tolist()})")
```

#### 1.2.6. 단항 ufuncs (Unary ufuncs)

단항 ufunc는 하나의 입력 배열을 받아 요소별 연산을 수행하고, 동일한 형태의 출력 배열을 반환합니다. 이들은 수학적 변환이나 배열 요소의 속성을 확인하는 데 사용됩니다.

**주요 단항 ufuncs:**
*   `np.abs(x)` 또는 `np.absolute(x)`: x의 절대값
*   `np.sqrt(x)`: x의 제곱근
*   `np.ceil(x)`: x보다 크거나 같은 가장 작은 정수 (올림)
*   `np.floor(x)`: x보다 작거나 같은 가장 큰 정수 (내림)
*   `np.round(x)` 또는 `np.around(x)`: x를 가장 가까운 정수로 반올림
*   `np.trunc(x)`: x의 소수점 이하를 버리고 정수 부분만 반환 (0에 가까운 정수)
*   `np.fix(x)`: x의 소수점 이하를 버리고 정수 부분만 반환 (0에 가까운 정수, `trunc`와 유사)
*   `np.sign(x)`: x의 부호 (-1, 0, 1 반환)
*   `np.isnan(x)`: x가 NaN(Not a Number)인지 확인
*   `np.isinf(x)`: x가 무한대인지 확인
*   `np.sinh(x)`: x의 하이퍼볼릭 사인 값
*   `np.cosh(x)`: x의 하이퍼볼릭 코사인 값
*   `np.tanh(x)`: x의 하이퍼볼릭 탄젠트 값
*   `np.arcsinh(x)`: x의 역하이퍼볼릭 사인 값
*   `np.arccosh(x)`: x의 역하이퍼볼릭 코사인 값
*   `np.arctanh(x)`: x의 역하이퍼볼릭 탄젠트 값

**예시 1: `np.abs` 또는 `np.absolute` (절대값)**

```python
import numpy as np

arr_unary = np.array([-1.2, 0, 3.5, -4])

abs_values = np.abs(arr_unary)
absolute_values = np.absolute(arr_unary)
print(f"원본 배열: {arr_unary}")
print(f"\n절대값 (np.abs): {abs_values}")
print(f"절대값 (np.absolute): {absolute_values}")
```

**예시 2: `np.sqrt` (제곱근)**

```python
sqrt_values = np.sqrt(np.array([0, 4, 9, 16]))
print(f"\n제곱근 (np.sqrt): {sqrt_values}")
```

**예시 3: `np.ceil`, `np.floor`, `np.round` 또는 `np.around` (반올림/내림/올림)**

```python
float_arr = np.array([1.2, 2.7, 3.5, 4.8, -1.3, -2.7])

ceil_values = np.ceil(float_arr)
floor_values = np.floor(float_arr)
round_values = np.round(float_arr)
around_values = np.around(float_arr)

print(f"\n원본 부동소수점 배열: {float_arr}")
print(f"올림 (np.ceil): {ceil_values}")
print(f"내림 (np.floor): {floor_values}")
print(f"반올림 (np.round): {round_values}")
print(f"반올림 (np.around): {around_values}")
```

**예시 4: `np.trunc`와 `np.fix` (소수점 이하 버림)**

```python
float_arr_trunc = np.array([1.2, 2.7, -3.5, -4.8])

trunc_values = np.trunc(float_arr_trunc)
fix_values = np.fix(float_arr_trunc)

print(f"\n원본 배열: {float_arr_trunc}")
print(f"소수점 이하 버림 (np.trunc): {trunc_values}")
print(f"소수점 이하 버림 (np.fix): {fix_values}")
```
**설명:**
`np.trunc`와 `np.fix`는 모두 소수점 이하를 버리고 정수 부분만 반환합니다. 양수에서는 `floor`와 같고, 음수에서는 `ceil`과 같습니다 (즉, 0에 가까운 정수로 만듭니다).

**예시 5: `np.sign` (부호)**

```python
sign_values = np.sign(np.array([-5, 0, 5, -0.5]))
print(f"\n부호 (np.sign): {sign_values}")
```

**예시 6: 하이퍼볼릭 삼각 함수 (np.sinh, np.cosh, np.arcsinh, np.arctanh)**
(이 내용은 1.2.2.1 삼각 함수 섹션에 이미 포함되어 있으므로, 여기서는 간략히 언급하고 해당 섹션을 참조하도록 합니다.)
하이퍼볼릭 삼각 함수 및 역하이퍼볼릭 삼각 함수에 대한 자세한 내용은 [1.2.2.1. 삼각 함수 (Trigonometric Functions)](#12221-삼각-함수-trigonometric-functions) 섹션을 참조하십시오.

#### 1.2.7. 이항 ufuncs (Binary ufuncs)

이항 ufunc는 두 개의 입력 배열을 받아 요소별 연산을 수행하고, 동일한 형태의 출력 배열을 반환합니다. 산술 연산 ufunc(`np.add`, `np.multiply` 등)도 이항 ufunc에 속하지만, 여기서는 다른 일반적인 이항 ufunc를 다룹니다.

**주요 이항 ufuncs:**
*   `np.maximum(x1, x2)`: 두 배열의 요소 중 큰 값 반환
*   `np.minimum(x1, x2)`: 두 배열의 요소 중 작은 값 반환
*   `np.fmax(x1, x2)`: `np.maximum`과 유사하지만, NaN을 무시하고 유효한 값 반환
*   `np.fmin(x1, x2)`: `np.minimum`과 유사하지만, NaN을 무시하고 유효한 값 반환
*   `np.hypot(x1, x2)`: 직각삼각형의 빗변 길이 계산 (피타고라스 정리)

**예시 1: `np.maximum` (최대값)**

```python
import numpy as np

arr_bin1 = np.array([1, 5, 2, 8])
arr_bin2 = np.array([3, 2, 6, 4])

print(f"arr_bin1: {arr_bin1}")
print(f"arr_bin2: {arr_bin2}")

max_values = np.maximum(arr_bin1, arr_bin2)
print(f"\n요소별 최대값 (np.maximum): {max_values}")
# 결과: [3 5 6 8]
```

**예시 2: `np.minimum` (최소값)**

```python
min_values = np.minimum(arr_bin1, arr_bin2)
print(f"요소별 최소값 (np.minimum): {min_values}")
# 결과: [1 2 2 4]
```

**예시 3: `np.fmax`와 `np.fmin` (NaN 처리)**

```python
import numpy as np

arr_nan1 = np.array([1, np.nan, 3])
arr_nan2 = np.array([np.nan, 2, 4])

print(f"\narr_nan1: {arr_nan1}")
print(f"arr_nan2: {arr_nan2}")

# np.maximum은 NaN이 포함된 경우 NaN을 반환
max_nan = np.maximum(arr_nan1, arr_nan2)
print(f"np.maximum (NaN 포함): {max_nan}")

# np.fmax는 NaN을 무시하고 유효한 값 반환
fmax_nan = np.fmax(arr_nan1, arr_nan2)
print(f"np.fmax (NaN 무시): {fmax_nan}")
```

**예시 4: `np.hypot` (빗변 길이 계산)**
```python
import numpy as np

x_coords = np.array([3, 5, 8])
y_coords = np.array([4, 12, 15])

# np.hypot(x, y)는 sqrt(x**2 + y**2)를 계산합니다.
hypotenuses = np.hypot(x_coords, y_coords)
print(f"\nx 좌표: {x_coords}")
print(f"y 좌표: {y_coords}")
print(f"빗변 길이 (np.hypot): {hypotenuses}")
```
**설명:**
`np.hypot(x1, x2)`는 직각삼각형의 두 변 `x1`과 `x2`의 길이를 받아 빗변의 길이($\sqrt{x_1^2 + x_2^2}$)를 계산합니다.

**결과:**
```
x 좌표: [3 5 8]
y 좌표: [4 12 15]
빗변 길이 (np.hypot): [ 5. 13. 17.]
```

#### 1.2.8. 조건부 선택 (`np.where`)

`np.where`는 마스킹과 유사한 효과를 내는 매우 강력한 함수로, **`if-else` 논리를 배열 전체에 벡터화(vectorized)하여 적용**할 때 사용됩니다. 이는 `for` 루프를 사용하는 것보다 훨씬 빠르고 간결합니다.

`np.where(condition, x, y)`는 `condition` 마스크가 `True`인 위치에는 `x`의 값을, `False`인 위치에는 `y`의 값을 채운 **새로운 배열**을 반환합니다.

```python
arr = np.arange(10)
print(f"원본 배열: {arr}")

# 조건: 5보다 작은 요소는 제곱하고, 그렇지 않은 요소는 그대로 둠
result = np.where(arr < 5, arr**2, arr)
print(f"np.where 적용 후: {result}")

# 조건: 짝수는 -1로, 홀수는 1로 변경
result_even_odd = np.where(arr % 2 == 0, -1, 1)
print(f"짝/홀에 따라 값 변경: {result_even_odd}")
```
**결과:**
```
원본 배열: [0 1 2 3 4 5 6 7 8 9]
np.where 적용 후: [ 0  1  4  9 16  5  6  7  8  9]
짝/홀에 따라 값 변경: [-1  1 -1  1 -1  1 -1  1 -1  1]
```
`np.where`는 데이터 분석에서 파생 변수를 만들거나, 특정 조건에 따라 값을 일괄적으로 바꿀 때 매우 유용하게 사용됩니다.

#### 1.2.9. 배열 조작 및 집합 연산 (Array Manipulation and Set Operations)

NumPy는 ufunc는 아니지만, 배열을 효율적으로 조작하고 집합 연산을 수행하는 다양한 함수들을 제공합니다. 이 함수들은 데이터 전처리 및 분석 과정에서 매우 유용하게 사용됩니다.

**주요 함수:**
*   `np.concatenate((a1, a2, ...), axis=0)`: 여러 배열을 지정된 축을 따라 연결
*   `np.diff(a, n=1, axis=-1)`: N-th 차 이산 차분 계산
*   `np.union1d(ar1, ar2)`: 두 1차원 배열의 합집합 (정렬된 고유 값)
*   `np.intersect1d(ar1, ar2)`: 두 1차원 배열의 교집합 (정렬된 고유 값)
*   `np.setdiff1d(ar1, ar2)`: 첫 번째 배열에서 두 번째 배열에 없는 요소 (차집합)
*   `np.setxor1d(ar1, ar2)`: 두 배열의 대칭 차집합 (합집합 - 교집합)

**예시 1: `np.concatenate` (배열 연결)**
```python
import numpy as np

arr_concat1 = np.array([1, 2, 3])
arr_concat2 = np.array([4, 5, 6])
arr_concat3 = np.array([7, 8])

# 1차원 배열 연결
concatenated_1d = np.concatenate((arr_concat1, arr_concat2, arr_concat3))
print(f"1차원 배열 연결: {concatenated_1d}")

# 2차원 배열 연결
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6]])

# axis=0 (행 방향) 연결
concatenated_2d_row = np.concatenate((matrix1, matrix2), axis=0)
print(f"\n2차원 배열 행 방향 연결:\n{concatenated_2d_row}")

# axis=1 (열 방향) 연결 (shape이 맞아야 함)
matrix3 = np.array([[10], [20]])
concatenated_2d_col = np.concatenate((matrix1, matrix3), axis=1)
print(f"\n2차원 배열 열 방향 연결:\n{concatenated_2d_col}")
```
**설명:**
`np.concatenate`는 여러 배열을 튜플로 묶어 첫 번째 인자로 전달하고, `axis` 인자를 통해 연결할 축을 지정합니다.

**결과:**
```
1차원 배열 연결: [1 2 3 4 5 6 7 8]

2차원 배열 행 방향 연결:
[[1 2]
 [3 4]
 [5 6]]

2차원 배열 열 방향 연결:
[[ 1  2 10]
 [ 3  4 20]]
```

**예시 2: `np.diff` (이산 차분)**
```python
import numpy as np

arr_diff = np.array([1, 2, 4, 7, 0])

# 1차 차분 (n=1, 기본값)
diff_1st = np.diff(arr_diff)
print(f"원본 배열: {arr_diff}")
print(f"1차 차분: {diff_1st}") # [2-1, 4-2, 7-4, 0-7] = [1, 2, 3, -7]

# 2차 차분 (n=2)
diff_2nd = np.diff(np.diff(arr_diff))
print(f"2차 차분: {diff_2nd}") # np.diff(np.diff(arr_diff)) = np.diff([1, 2, 3, -7]) = [1, 1, -10]
```
**설명:**
`np.diff(a, n=1, axis=-1)`는 배열 `a`의 N-th 차 이산 차분을 계산합니다. 주로 시계열 데이터의 변화율을 분석할 때 사용됩니다.

**결과:**
```
원본 배열: [1 2 4 7 0]
1차 차분: [ 1  2  3 -7]
2차 차분: [  1  -1 -10]
```

**예시 3: 집합 연산 (`np.union1d`, `np.intersect1d`, `np.setdiff1d`, `np.setxor1d`)**
```python
import numpy as np

set1 = np.array([1, 3, 5, 7, 9])
set2 = np.array([1, 2, 3, 4, 5, 6])

print(f"집합 1: {set1}")
print(f"집합 2: {set2}")

# 합집합 (Union)
union_set = np.union1d(set1, set2)
print(f"\n합집합 (np.union1d): {union_set}")

# 교집합 (Intersection)
intersect_set = np.intersect1d(set1, set2)
print(f"교집합 (np.intersect1d): {intersect_set}")

# 차집합 (Set Difference: set1 - set2)
diff_set = np.setdiff1d(set1, set2)
print(f"차집합 (np.setdiff1d): {diff_set}")

# 대칭 차집합 (Symmetric Difference)
xor_set = np.setxor1d(set1, set2)
print(f"대칭 차집합 (np.setxor1d): {xor_set}")
```
**설명:**
이 함수들은 1차원 배열에 대한 집합 연산을 수행합니다. 결과는 항상 정렬된 고유 값으로 반환됩니다.

**결과:**
```
집합 1: [1 3 5 7 9]
집합 2: [1 2 3 4 5 6]

합집합 (np.union1d): [1 2 3 4 5 6 7 9]
교집합 (np.intersect1d): [1 3 5]
차집합 (np.setdiff1d): [7 9]
대칭 차집합 (np.setxor1d): [2 4 6 7 9]
```

#### 1.2.10. 사용자 정의 ufuncs 생성 (`np.frompyfunc`)

`np.frompyfunc(func, nin, nout)`는 파이썬 함수를 NumPy의 유니버설 함수(ufunc)로 변환하는 데 사용됩니다. 이를 통해 파이썬 함수를 NumPy 배열에 벡터화된 방식으로 적용할 수 있습니다.

**매개변수:**
*   `func`: ufunc로 변환할 파이썬 함수. 스칼라 입력만 처리해야 합니다.
*   `nin`: 입력 인수의 개수.
*   `nout`: 출력 인수의 개수.

**주의사항:**
`np.frompyfunc`로 생성된 ufunc는 C로 구현된 내장 ufunc만큼 빠르지 않습니다. 내부적으로 여전히 파이썬 루프를 사용하므로, 성능이 중요한 경우에는 `np.where`나 다른 벡터화된 NumPy 연산을 사용하는 것이 좋습니다. 이는 주로 복잡한 스칼라 함수를 배열에 적용할 때의 편의성을 위한 것입니다.

**예시 1: 간단한 사용자 정의 ufunc 생성**
```python
import numpy as np

# 스칼라 값을 처리하는 파이썬 함수
def my_custom_func(x, y):
    if x > y:
        return x + y
    else:
        return x * y

# np.frompyfunc를 사용하여 ufunc로 변환
# 입력 2개 (x, y), 출력 1개
vectorized_my_func = np.frompyfunc(my_custom_func, 2, 1)

arr_a = np.array([1, 5, 2])
arr_b = np.array([3, 4, 6])

result = vectorized_my_func(arr_a, arr_b)
print(f"arr_a: {arr_a}")
print(f"arr_b: {arr_b}")
print(f"사용자 정의 ufunc 결과: {result}")
```
**설명:**
`my_custom_func`는 두 개의 스칼라를 입력받아 하나의 스칼라를 반환하는 파이썬 함수입니다. `np.frompyfunc`를 통해 이 함수를 `vectorized_my_func`라는 ufunc로 만들었습니다. 이제 이 ufunc는 NumPy 배열에 요소별로 적용될 수 있습니다.

**결과:**
```
arr_a: [1 5 2]
arr_b: [3 4 6]
사용자 정의 ufunc 결과: [3 9 12]
```

**예시 2: 여러 개의 출력을 가지는 사용자 정의 ufunc**
```python
import numpy as np

def custom_divmod(x, y):
    """몫과 나머지를 반환하는 함수"""
    return x // y, x % y

# np.frompyfunc를 사용하여 ufunc로 변환
# 입력 2개 (x, y), 출력 2개 (몫, 나머지)
vectorized_divmod = np.frompyfunc(custom_divmod, 2, 2)

arr_x = np.array([10, 11, 12])
arr_y = np.array([3, 4, 5])

quotient, remainder = vectorized_divmod(arr_x, arr_y)
print(f"\narr_x: {arr_x}")
print(f"arr_y: {arr_y}")
print(f"몫: {quotient}")
print(f"나머지: {remainder}")
```
**설명:**
`custom_divmod` 함수는 두 개의 스칼라를 입력받아 두 개의 스칼라(몫과 나머지)를 반환합니다. `nout=2`로 지정하여 두 개의 출력을 처리할 수 있는 ufunc를 생성했습니다.

**결과:**
```
arr_x: [10 11 12]
arr_y: [3 4 5]
몫: [3 2 2]
나머지: [1 3 2]
```

### 1.3. 브로드캐스팅(Broadcasting)과 ufuncs

NumPy의 유니버설 함수(ufuncs)는 **브로드캐스팅(Broadcasting)** 규칙을 자동으로 적용하여 형태가 다른 배열 간에도 연산을 유연하게 수행할 수 있도록 합니다. 이는 명시적으로 배열의 형태를 맞추거나 복잡한 반복문을 사용할 필요 없이 코드를 간결하게 만들고, NumPy 내부의 최적화된 C 구현 덕분에 효율적인 연산을 가능하게 합니다.

브로드캐스팅의 개념, 규칙, 다양한 예시 및 주의점에 대한 자세한 내용은 [08_브로드캐스팅.md](08_브로드캐스팅.md) 문서를 참조해 주세요.

### 1.4. ufuncs의 성능 이점 (벡터화)

NumPy의 유니버설 함수(ufuncs)는 파이썬의 일반적인 반복문(for-loop)을 사용하는 것보다 훨씬 뛰어난 성능을 제공합니다. 이러한 성능 이점은 주로 **벡터화(Vectorization)**라는 개념에서 비롯됩니다.

**벡터화란?**
벡터화는 명시적인 파이썬 `for` 루프 없이 배열 연산을 구현하는 것을 의미합니다. 대신, NumPy는 C, C++, 포트란과 같은 저수준 언어로 구현된 최적화된 코드를 사용하여 배열의 모든 요소에 대해 연산을 한 번에 수행합니다. 이는 다음과 같은 이유로 성능 향상을 가져옵니다.

1.  **오버헤드 감소**: 파이썬 `for` 루프는 각 요소에 접근할 때마다 인터프리터 오버헤드가 발생합니다. 이는 파이썬이 동적 타입 언어이기 때문에 발생하는 비용입니다. NumPy ufuncs는 이러한 오버헤드를 제거하고, 컴파일된 코드가 직접 메모리 블록에 접근하여 연산을 수행합니다.
2.  **캐시 효율성**: NumPy 배열은 메모리에 연속적으로 저장됩니다. 이는 CPU 캐시의 효율성을 극대화합니다. 연속적인 메모리 접근은 캐시 미스(cache miss)를 줄여 데이터를 더 빠르게 가져올 수 있게 합니다. 파이썬 리스트는 메모리에 연속적으로 저장되지 않을 수 있어 캐시 효율성이 떨어집니다.
3.  **SIMD 활용**: 현대 CPU는 SIMD(Single Instruction, Multiple Data) 명령어를 지원합니다. 이는 하나의 명령어로 여러 데이터에 대한 연산을 동시에 수행할 수 있게 합니다. NumPy ufuncs는 내부적으로 이러한 SIMD 명령어를 활용하여 병렬 처리를 수행하므로, 대규모 배열 연산에서 엄청난 속도 향상을 가져옵니다.
4.  **메모리 접근 패턴 최적화**: NumPy는 배열의 데이터 타입과 구조를 미리 알고 있기 때문에, 메모리 접근 패턴을 최적화하여 데이터를 효율적으로 읽고 쓸 수 있습니다.

**성능 비교 예시**: 

간단한 배열 덧셈을 통해 벡터화의 성능 이점을 확인할 수 있습니다.

```python
import numpy as np
import time

# 큰 배열 생성
arr1 = np.random.rand(10**7)
arr2 = np.random.rand(10**7)

# 1. 파이썬 for 루프 사용
start_time = time.time()
result_list = []
for i in range(len(arr1)):
    result_list.append(arr1[i] + arr2[i])
end_time = time.time()
print(f"Python for loop time: {end_time - start_time:.4f} seconds")

# 2. NumPy ufunc (벡터화) 사용
start_time = time.time()
result_numpy = arr1 + arr2 # ufunc인 np.add가 내부적으로 호출됨
end_time = time.time()
print(f"NumPy ufunc time: {end_time - start_time:.4f} seconds")
```

위 예시를 실행해보면, NumPy ufunc를 사용한 연산이 파이썬 `for` 루프를 사용한 연산보다 수십 배에서 수백 배 더 빠르다는 것을 확인할 수 있습니다. 이는 데이터 과학 및 머신러닝 분야에서 대규모 데이터를 효율적으로 처리하는 데 NumPy가 필수적인 이유입니다.

### 1.5. ufuncs의 메서드 (Methods of ufuncs)

NumPy ufunc는 단순히 요소별 연산을 수행하는 것 외에도, `reduce`, `accumulate`, `outer`, `at`과 같은 특별한 메서드를 제공하여 더욱 강력하고 유연한 연산을 가능하게 합니다. 이 메서드들은 ufunc의 기능을 확장하여 배열 전체에 걸쳐 복합적인 연산을 효율적으로 수행할 수 있도록 돕습니다.

#### 1.5.1. `reduce` 메서드

`ufunc.reduce(array, axis=0, dtype=None, out=None, keepdims=False)`: 지정된 축을 따라 ufunc를 반복적으로 적용하여 배열의 차원을 하나 줄입니다. 예를 들어, `np.add.reduce`는 배열의 합계를 계산하고, `np.multiply.reduce`는 배열의 곱을 계산합니다. 이는 `np.sum`이나 `np.prod`와 유사한 기능을 제공하지만, 모든 ufunc에 대해 일반화된 방식으로 동작합니다.

**주요 `reduce` 관련 함수:**
*   `np.sum()`: 배열의 합계 (내부적으로 `np.add.reduce` 사용)
*   `np.prod()`: 배열의 곱 (내부적으로 `np.multiply.reduce` 사용)
*   `np.lcm.reduce()`: 배열 요소들의 최소 공배수 계산
*   `np.gcd.reduce()`: 배열 요소들의 최대 공약수 계산

**예시 1: `np.add.reduce` (합계 계산)**

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# np.add.reduce는 배열의 모든 요소를 더합니다 (np.sum과 유사)
sum_reduced = np.add.reduce(arr)
print(f"원본 배열: {arr}")
print(f"np.add.reduce 결과 (합계): {sum_reduced}")
# 결과: 15
```

**예시 2: `np.multiply.reduce` (곱 계산)**

```python
# np.multiply.reduce는 배열의 모든 요소를 곱합니다 (np.prod와 유사)
prod_reduced = np.multiply.reduce(arr)
print(f"np.multiply.reduce 결과 (곱): {prod_reduced}")
# 결과: 120 (1*2*3*4*5)
```

**예시 3: 2차원 배열에서 `axis` 지정**

```python
import numpy as np

arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"\n원본 2차원 배열:\n{arr_2d}")

# axis=0 (열 방향)으로 reduce: 각 열의 합계
sum_col_reduced = np.add.reduce(arr_2d, axis=0)
print(f"axis=0으로 reduce (각 열의 합계): {sum_col_reduced}")
# 결과: [12 15 18]

# axis=1 (행 방향)으로 reduce: 각 행의 합계
sum_row_reduced = np.add.reduce(arr_2d, axis=1)
print(f"axis=1으로 reduce (각 행의 합계): {sum_row_reduced}")
# 결과: [ 6 15 24]
```

**예시 4: `np.lcm.reduce` (최소 공배수)**
```python
import numpy as np

arr_lcm = np.array([2, 3, 4])
lcm_result = np.lcm.reduce(arr_lcm)
print(f"\n원본 배열: {arr_lcm}")
print(f"최소 공배수 (np.lcm.reduce): {lcm_result}") # lcm(2,3,4) = 12
```
**설명:**
`np.lcm.reduce()`는 배열의 모든 요소에 대한 최소 공배수를 계산합니다.

**결과:**
```
원본 배열: [2 3 4]
최소 공배수 (np.lcm.reduce): 12
```

**예시 5: `np.gcd.reduce` (최대 공약수)**
```python
import numpy as np

arr_gcd = np.array([12, 18, 24])
gcd_result = np.gcd.reduce(arr_gcd)
print(f"\n원본 배열: {arr_gcd}")
print(f"최대 공약수 (np.gcd.reduce): {gcd_result}") # gcd(12,18,24) = 6
```
**설명:**
`np.gcd.reduce()`는 배열의 모든 요소에 대한 최대 공약수를 계산합니다.

**결과:**
```
원본 배열: [12 18 24]
최대 공약수 (np.gcd.reduce): 6
```

#### 1.5.2. `accumulate` 메서드

`ufunc.accumulate(array, axis=0, dtype=None, out=None)`: 지정된 축을 따라 ufunc를 반복적으로 적용하되, 모든 중간 결과를 배열로 반환합니다. 이는 누적 합(cumulative sum)이나 누적 곱(cumulative product)과 같은 연산에 유용합니다.

**주요 `accumulate` 관련 함수:**
*   `np.cumsum()`: 누적 합계 (내부적으로 `np.add.accumulate` 사용)
*   `np.cumprod()`: 누적 곱 (내부적으로 `np.multiply.accumulate` 사용)

**예시 1: `np.add.accumulate` (누적 합계)**

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# np.add.accumulate는 누적 합계를 계산합니다 (np.cumsum과 유사)
sum_accumulated = np.add.accumulate(arr)
print(f"원본 배열: {arr}")
print(f"np.add.accumulate 결과 (누적 합계): {sum_accumulated}")
# 결과: [ 1  3  6 10 15]
```

**예시 2: `np.multiply.accumulate` (누적 곱)**

```python
# np.multiply.accumulate는 누적 곱을 계산합니다 (np.cumprod와 유사)
prod_accumulated = np.multiply.accumulate(arr)
print(f"np.multiply.accumulate 결과 (누적 곱): {prod_accumulated}")
# 결과: [  1   2   6  24 120]
```

**예시 3: 2차원 배열에서 `axis` 지정**

```python
import numpy as np

arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(f"\n원본 2차원 배열:\n{arr_2d}")

# axis=0 (열 방향)으로 accumulate: 각 열의 누적 합계
sum_col_accumulated = np.add.accumulate(arr_2d, axis=0)
print(f"axis=0으로 accumulate (각 열의 누적 합계):\n{sum_col_accumulated}")
# 결과:
# [[ 1  2  3]
#  [ 5  7  9]
#  [12 15 18]]

# axis=1 (행 방향)으로 accumulate: 각 행의 누적 합계
sum_row_accumulated = np.add.accumulate(arr_2d, axis=1)
print(f"axis=1으로 accumulate (각 행의 누적 합계):\n{sum_row_accumulated}")
# 결과:
# [[ 1  3  6]
#  [ 4  9 15]
#  [ 7 15 24]]
```

**예시 4: `np.cumsum` (누적 합계)**
```python
import numpy as np

arr_cumsum = np.array([1, 2, 3, 4, 5])
cumsum_result = np.cumsum(arr_cumsum)
print(f"\n원본 배열: {arr_cumsum}")
print(f"누적 합계 (np.cumsum): {cumsum_result}")
```
**설명:**
`np.cumsum()`은 배열의 누적 합계를 계산합니다. `np.add.accumulate()`와 동일한 기능을 합니다.

**결과:**
```
원본 배열: [1 2 3 4 5]
누적 합계 (np.cumsum): [ 1  3  6 10 15]
```

**예시 5: `np.cumprod` (누적 곱)**
```python
import numpy as np

arr_cumprod = np.array([1, 2, 3, 4, 5])
cumprod_result = np.cumprod(arr_cumprod)
print(f"\n원본 배열: {arr_cumprod}")
print(f"누적 곱 (np.cumprod): {cumprod_result}")
```
**설명:**
`np.cumprod()`은 배열의 누적 곱을 계산합니다. `np.multiply.accumulate()`와 동일한 기능을 합니다.

**결과:**
```
원본 배열: [1 2 3 4 5]
누적 곱 (np.cumprod): [  1   2   6  24 120]
```

#### 1.5.3. `outer` 메서드

`ufunc.outer(A, B, out=None)`: 두 배열 `A`와 `B`의 모든 요소 쌍에 대해 ufunc를 적용하여 외적(outer product)과 유사한 결과를 생성합니다. 결과 배열의 차원은 `A.ndim + B.ndim`이 됩니다.

**예시 1: `np.multiply.outer` (외적 곱)**

```python
import numpy as np

arr_a = np.array([1, 2, 3])
arr_b = np.array([10, 20])

# np.multiply.outer는 두 배열의 모든 요소 쌍에 대해 곱셈을 수행합니다.
outer_product = np.multiply.outer(arr_a, arr_b)
print(f"원본 arr_a: {arr_a}")
print(f"원본 arr_b: {arr_b}")
print(f"np.multiply.outer 결과:\n{outer_product}")
# 결과:
# [[10 20]
#  [20 40]
#  [30 60]]
# 설명: arr_a의 각 요소와 arr_b의 각 요소가 곱해져 새로운 2차원 배열을 형성합니다.
```

**예시 2: `np.add.outer` (외적 덧셈)**

```python
# np.add.outer는 두 배열의 모든 요소 쌍에 대해 덧셈을 수행합니다.
outer_sum = np.add.outer(arr_a, arr_b)
print(f"np.add.outer 결과:\n{outer_sum}")
# 결과:
# [[11 21]
#  [12 22]
#  [13 23]]
```

#### 1.5.4. `at` 메서드

`ufunc.at(a, indices, b=None)`: 지정된 `indices` 위치에 `ufunc` 연산을 인플레이스(in-place)로 적용합니다. 이는 `a[indices] = ufunc(a[indices], b)`와 유사하지만, `indices`에 중복된 값이 있을 경우 `at` 메서드는 모든 연산을 올바르게 수행합니다. 일반적인 인덱싱 할당은 중복된 인덱스에 대해 마지막 연산만 적용합니다.

**예시 1: `np.add.at` (중복 인덱스 처리)**

```python
import numpy as np

arr = np.zeros(5)
indices = np.array([0, 1, 1, 2, 0]) # 인덱스 0과 1이 중복됨
values = np.array([1, 1, 1, 1, 1])

print(f"원본 배열: {arr}")
print(f"인덱스: {indices}")
print(f"값: {values}")

# 일반적인 인덱싱 할당 (중복 인덱스에 대해 마지막 값만 적용)
# temp_arr = np.zeros(5)
# temp_arr[indices] += values
# print(f"\n일반 인덱싱 할당 결과: {temp_arr}") # 결과: [1. 1. 1. 1. 0.] (0번과 1번 인덱스에 마지막 1만 더해짐)

# np.add.at를 사용하면 모든 연산이 올바르게 적용됨
np.add.at(arr, indices, values)
print(f"\nnp.add.at 결과: {arr}")
# 결과: [2. 2. 1. 1. 0.]
# 설명: 인덱스 0에는 1+1=2, 인덱스 1에는 1+1=2가 올바르게 더해집니다.
```

**예시 2: `np.subtract.at`**

```python
import numpy as np

arr_sub = np.array([10, 20, 30])
indices_sub = np.array([0, 0, 1])
values_sub = np.array([1, 2, 3])

print(f"\n원본 배열: {arr_sub}")
print(f"인덱스: {indices_sub}")
print(f"값: {values_sub}")

np.subtract.at(arr_sub, indices_sub, values_sub)
print(f"np.subtract.at 결과: {arr_sub}")
# 결과: [ 7 17 30]
# 설명: 인덱스 0에는 10 - 1 - 2 = 7, 인덱스 1에는 20 - 3 = 17이 됩니다.
```