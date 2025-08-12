<h2>NumPy 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
이 문서는 NumPy의 핵심 성능 비결인 유니버설 함수(Universal Functions, ufuncs)와 벡터화(Vectorization)의 개념을 상세히 다룹니다. ufuncs가 어떻게 배열 연산을 효율적으로 수행하는지 이해하고, 실제 코드 예제를 통해 벡터화된 연산의 장점을 학습합니다. 다양한 ufuncs의 활용법과 브로드캐스팅 규칙, 그리고 성능 이점까지 포괄적으로 다룹니다.

<h2>목차</h2>

- [1. 유니버설 함수 (Universal Functions, ufuncs)와 벡터화](#1-유니버설-함수-universal-functions-ufuncs와-벡터화)
  - [1.1. ufuncs의 주요 특징](#11-ufuncs의-주요-특징)
  - [1.2. 다양한 ufuncs 예시](#12-다양한-ufuncs-예시)
    - [1.2.1. 산술 연산 (Arithmetic Operations)](#121-산술-연산-arithmetic-operations)
    - [1.2.2. 삼각 및 지수/로그 함수 (Trigonometric, Exponential/Logarithmic Functions)](#122-삼각-및-지수로그-함수-trigonometric-exponentiellogarithmic-functions)
    - [1.2.3. 비교 연산 (Comparison Operations)](#123-비교-연산-comparison-operations)
    - [1.2.4. 조건부 선택 (`np.where`)](#124-조건부-선택-npwhere)
  - [1.3. 브로드캐스팅(Broadcasting)과 ufuncs](#13-브로드캐스팅broadcasting과-ufuncs)
  - [1.4. ufuncs의 성능 이점 (벡터화)](#14-ufuncs의-성능-이점-벡터화)

---

## 1. 유니버설 함수 (Universal Functions, ufuncs)와 벡터화

NumPy의 핵심 성능 비결 중 하나는 **유니버설 함수(Universal Functions, ufuncs)**입니다. ufunc는 `ndarray` 객체 내의 모든 요소에 대해 빠른 C 레벨의 연산을 수행하는 함수입니다. 사용자가 파이썬 `for` 루프를 작성할 필요 없이, 배열 전체에 대한 연산을 간결하게 표현하는 것을 **벡터화(Vectorization)**라고 하며, 이는 ufuncs를 통해 구현됩니다.

### 1.1. ufuncs의 주요 특징

-   **성능**: 내부적으로 C로 구현된 루프를 사용하므로 순수 파이썬 코드보다 월등히 빠릅니다. 대규모 데이터셋 처리 시 필수적입니다.
-   **다양성**: 산술 연산(`np.add`, `np.multiply`), 삼각 함수(`np.sin`, `np.cos`), 통계 함수(`np.sum`, `np.mean`), 비교 연산(`np.greater`, `np.equal`) 등 수백 가지의 ufuncs가 미리 구현되어 있습니다.
-   **브로드캐스팅 지원**: ufuncs는 브로드캐스팅 규칙을 자동으로 지원하여 형태가 다른 배열 간의 연산을 가능하게 합니다.
-   **메모리 효율성**: 불필요한 임시 배열 생성을 줄여 메모리 사용을 최적화합니다.

### 1.2. 다양한 ufuncs 예시

`+`, `*`와 같은 연산자들은 내부적으로 해당 ufunc(`np.add`, `np.multiply`)를 호출합니다.

#### 1.2.1. 산술 연산 (Arithmetic Operations)

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")

# 요소별 덧셈 (내부적으로 np.add(arr1, arr2) 호출)
sum_arr = arr1 + arr2
print(f"\n요소별 덧셈: {sum_arr}") # [5 7 9]

# 요소별 곱셈 (내부적으로 np.multiply(arr1, arr2) 호출)
mul_arr = arr1 * arr2
print(f"요소별 곱셈: {mul_arr}") # [4 10 18]

# 스칼라 연산: 배열의 모든 요소에 스칼라 값 적용 (np.multiply(arr1, 10) 호출)
scalar_mul = arr1 * 10
print(f"스칼라 곱셈: {scalar_mul}") # [10 20 30]
```

#### 1.2.2. 삼각 및 지수/로그 함수 (Trigonometric, Exponential/Logarithmic Functions)

```python
import numpy as np

# 삼각 함수 ufuncs
angles = np.array([0, np.pi/2, np.pi])
sin_angles = np.sin(angles)
print(f"\n사인 함수 적용: {sin_angles}") # [0. 1. 0.] (근사치)

# 지수/로그 함수 ufuncs
exp_arr = np.exp(np.array([1, 2, 3]))
print(f"지수 함수 적용: {exp_arr}")
log_arr = np.log(exp_arr)
print(f"로그 함수 적용: {log_arr}") # 원본 배열과 동일한 값 (부동 소수점 오차)
```

#### 1.2.3. 비교 연산 (Comparison Operations)

```python
import numpy as np

compare_arr = np.array([10, 20, 30, 40, 50])
print(f"\ncompare_arr: {compare_arr}")
print(f"10보다 큰 요소: {np.greater(compare_arr, 10)}") # [False  True  True  True  True]
print(f"20과 같은 요소: {np.equal(compare_arr, 20)}") # [False  True False False False]
print(f"30보다 작거나 같은 요소: {np.less_equal(compare_arr, 30)}") # [ True  True  True False False]
```

#### 1.2.4. 조건부 선택 (`np.where`)

`np.where(condition, x, y)`: `condition`이 True인 위치에서는 `x`의 값을, False인 위치에서는 `y`의 값을 반환합니다.

```python
import numpy as np

compare_arr = np.array([10, 20, 30, 40, 50])
result_where = np.where(compare_arr > 30, compare_arr, 0)
print(f"\n30보다 큰 요소는 그대로, 아니면 0: {result_where}") # [ 0  0  0 40 50]

# 짝수/홀수 구분 예시
numbers = np.array([1, 2, 3, 4, 5, 6])
odd_even = np.where(numbers % 2 == 0, '짝수', '홀수')
print(f"숫자 짝수/홀수 구분: {odd_even}") # ['홀수' '짝수' '홀수' '짝수' '홀수' '짝수']
```

### 1.3. 브로드캐스팅(Broadcasting)과 ufuncs

ufuncs는 브로드캐스팅 규칙을 자동으로 적용하여 형태가 다른 배열 간의 연산을 가능하게 합니다. 이는 명시적으로 배열의 형태를 맞추지 않아도 되므로 코드를 간결하게 만듭니다.

```python
import numpy as np

arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]]) # shape: (2, 3)
scalar = 10 # shape: ()

# 스칼라와 배열 연산 (브로드캐스팅)
result_scalar = arr_2d + scalar
print(f"\n2D 배열 + 스칼라:
{result_scalar}")
# 결과:
# [[11 12 13]
#  [14 15 16]]

vector = np.array([100, 200, 300]) # shape: (3,)

# 2D 배열과 1D 배열 연산 (브로드캐스팅)
# (2, 3) + (3,) -> (2, 3) + (1, 3) -> (2, 3) + (2, 3)
result_vector = arr_2d + vector
print(f"\n2D 배열 + 1D 배열:
{result_vector}")
# 결과:
# [[101 202 303]
#  [104 205 306]]
```

### 1.4. ufuncs의 성능 이점 (벡터화)

ufuncs를 사용한 벡터화된 연산은 파이썬의 명시적인 `for` 루프를 사용하는 것보다 훨씬 빠릅니다. 이는 NumPy의 내부 구현이 C 언어로 되어 있어 오버헤드가 적기 때문입니다.

```python
import numpy as np
import time

size = 1_000_000
arr_a = np.random.rand(size)
arr_b = np.random.rand(size)

# 파이썬 for 루프를 사용한 덧셈
start_time = time.time()
result_python = []
for i in range(size):
    result_python.append(arr_a[i] + arr_b[i])
end_time = time.time()
print(f"\nPython for 루프 소요 시간: {end_time - start_time:.6f} 초")

# NumPy ufunc를 사용한 덧셈 (벡터화)
start_time = time.time()
result_numpy = arr_a + arr_b # np.add(arr_a, arr_b)와 동일
end_time = time.time()
print(f"NumPy ufunc 소요 시간: {end_time - start_time:.6f} 초")

# 일반적으로 NumPy ufunc가 Python for 루프보다 수십 배에서 수백 배 빠릅니다.
```