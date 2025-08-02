<h2>머신러닝/딥러닝을 위한 NumPy 라이브러리: 핵심 개념 및 활용 심화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-02

<h2>문서 목표</h2>
이 문서는 머신러닝(ML) 및 딥러닝(DL) 분야에서 필수적인 파이썬 라이브러리인 NumPy의 핵심 개념과 활용법을 상세히 다룹니다. NumPy의 다차원 배열(ndarray) 구조, 효율적인 벡터 연산, 브로드캐스팅, 그리고 ML/DL 모델 구현에 필요한 다양한 수학적 기능들을 이해하고 실제 코드 예제를 통해 적용하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. NumPy 소개](#1-numpy-소개)
  - [1.1. NumPy란?](#11-numpy란)
  - [1.2. ML/DL에서 NumPy의 중요성](#12-mldl에서-numpy의-중요성)
- [2. NumPy 핵심 데이터 구조: ndarray](#2-numpy-핵심-데이터-구조-ndarray)
  - [2.1. ndarray 생성](#21-ndarray-생성)
  - [2.2. ndarray의 속성](#22-ndarray의-속성)
- [3. ndarray 인덱싱 및 슬라이싱](#3-ndarray-인덱싱-및-슬라이싱)
  - [3.1. 단일 요소 접근](#31-단일-요소-접근)
  - [3.2. 슬라이싱](#32-슬라이싱)
  - [3.3. 불리언 인덱싱 (Boolean Indexing)](#33-불리언-인덱싱-boolean-indexing)
  - [3.4. 팬시 인덱싱 (Fancy Indexing)](#34-팬시-인덱싱-fancy-indexing)
- [4. NumPy 기본 연산](#4-numpy-기본-연산)
  - [4.1. 벡터화된 연산](#41-벡터화된-연산)
  - [4.2. 브로드캐스팅 (Broadcasting)](#42-브로드캐스팅-broadcasting)
  - [4.3. 행렬 곱셈 (Dot Product)](#43-행렬-곱셈-dot-product)
  - [4.4. 집계 함수 (Aggregation Functions)](#44-집계-함수-aggregation-functions)
  - [4.5. 전치 행렬 (Transpose)](#45-전치-행렬-transpose)
  - [4.6. 배열 형태 변경 (Reshaping)](#46-배열-형태-변경-reshaping)
  - [4.7. 배열 결합 및 분할 (Concatenation \& Splitting)](#47-배열-결합-및-분할-concatenation--splitting)
- [5. NumPy 파일 입출력](#5-numpy-파일-입출력)
  - [5.1. 단일 배열 저장/로드: `np.save()`, `np.load()`](#51-단일-배열-저장로드-npsave-npload)
  - [5.2. 여러 배열 저장/로드: `np.savez()`](#52-여러-배열-저장로드-npsavez)
- [6. 기타 유용한 기능](#6-기타-유용한-기능)
  - [6.1. 난수 생성](#61-난수-생성)
  - [6.2. 푸리에 변환 (Fourier Transform)](#62-푸리에-변환-fourier-transform)
- [7. 선형대수 (Linear Algebra)](#7-선형대수-linear-algebra)
  - [7.1. 역행렬 (Inverse Matrix)](#71-역행렬-inverse-matrix)
  - [7.2. 행렬식 (Determinant)](#72-행렬식-determinant)
  - [7.3. 고유값과 고유벡터 (Eigenvalues and Eigenvectors)](#73-고유값과-고유벡터-eigenvalues-and-eigenvectors)
  - [7.4. 선형 시스템 해법 (Solving Linear Systems)](#74-선형-시스템-해법-solving-linear-systems)
  - [7.5. 벡터 노름 (Vector Norms)](#75-벡터-노름-vector-norms)
  - [7.6. 특이값 분해 (Singular Value Decomposition, SVD)](#76-특이값-분해-singular-value-decomposition-svd)
- [8. 추가 고급 기능](#8-추가-고급-기능)
  - [8.1. 정렬 및 검색](#81-정렬-및-검색)
  - [8.2. 브로드캐스팅 규칙 심화](#82-브로드캐스팅-규칙-심화)
  - [8.3. 마스킹 (Masking)](#83-마스킹-masking)
  - [8.4. 구조화 배열 (Structured Arrays)](#84-구조화-배열-structured-arrays)
  - [8.5. 메모리 관리 및 뷰 vs. 복사](#85-메모리-관리-및-뷰-vs-복사)
  - [8.6. 성능 최적화 팁](#86-성능-최적화-팁)
  - [8.7. NumPy와 다른 라이브러리 연동](#87-numpy와-다른-라이브러리-연동)
  - [8.8. 실제 ML/DL 적용 사례](#88-실제-mldl-적용-사례)

---

## 1. NumPy 소개

### 1.1. NumPy란?
NumPy (Numerical Python)는 파이썬에서 과학 계산, 특히 다차원 배열(array)을 효율적으로 다루기 위한 핵심 라이브러리입니다. 파이썬의 기본 리스트(list)와 달리, NumPy의 `ndarray` 객체는 대규모 수치 데이터를 빠르고 효율적으로 처리할 수 있도록 최적화되어 있습니다.

### 1.2. ML/DL에서 NumPy의 중요성
머신러닝(ML) 및 딥러닝(DL) 모델은 대부분 행렬(matrix) 및 벡터(vector) 연산을 기반으로 합니다. NumPy는 이러한 연산을 파이썬에서 고성능으로 수행할 수 있게 하여 ML/DL 개발에 필수적인 도구로 자리매김했습니다.

1.  **빠르고 메모리 효율적인 다차원 배열 (`ndarray`)**:
    NumPy는 C, C++ 및 Fortran으로 구현된 내부 루틴을 사용하여 파이썬 리스트보다 훨씬 빠르고 메모리를 효율적으로 사용합니다. 이는 대규모 데이터셋을 다루는 ML/DL에서 성능 병목 현상을 줄이는 데 결정적인 역할을 합니다.

2.  **벡터화된 산술 연산 및 브로드캐스팅**:
    반복문(loop)을 명시적으로 작성할 필요 없이 전체 데이터 배열에 대해 빠른 연산을 제공하는 "벡터화(vectorization)" 기능을 지원합니다. 또한, 크기가 다른 배열 간의 연산을 가능하게 하는 "브로드캐스팅(broadcasting)" 기능을 제공하여 코드의 간결성과 효율성을 높입니다.

3.  **선형대수, 난수 생성, 푸리에 변환 등 다양한 수학 함수 제공**:
    ML/DL 알고리즘 구현에 필수적인 선형대수(행렬 곱셈, 역행렬 등), 통계(평균, 표준편차 등), 난수 생성, 푸리에 변환 등 광범위한 수학 함수를 내장하고 있습니다.

4.  **데이터 입출력 도구**:
    배열 데이터를 디스크에 저장하거나 읽을 수 있는 도구(`np.save`, `np.load`)를 제공하여 모델 학습에 필요한 대규모 데이터를 효율적으로 관리할 수 있습니다.


**파이썬 리스트와 NumPy 배열의 연산 비교 예시:**

```python
# 파이썬 리스트의 덧셈 (리스트 확장)
a_list = [1, 2, 3]
b_list = [4, 5, 6]
c_list = a_list + b_list
print(f"파이썬 리스트 덧셈 결과: {c_list}")
# 수행 결과: [1, 2, 3, 4, 5, 6] (두 리스트를 연결)

import numpy as np

# NumPy 배열의 덧셈 (요소별 연산)
a_np = np.array([1, 2, 3])
b_np = np.array([4, 5, 6])
c_np_sum = a_np + b_np
print(f"NumPy 배열 덧셈 결과: {c_np_sum}")
# 수행 결과: [5 7 9] (각 요소들의 합으로 벡터 연산 수행)

# NumPy 배열의 곱셈 (요소별 연산)
c_np_mul = a_np * b_np
print(f"NumPy 배열 곱셈 결과: {c_np_mul}")
# 수행 결과: [4 10 18] (각 요소들의 곱으로 벡터 연산 수행)

print(f"a_np의 타입: {type(a_np)}") # <class 'numpy.ndarray'>
print(f"c_np_sum의 첫 번째 요소: {c_np_sum[0]}") # 5
```

## 2. NumPy 핵심 데이터 구조: ndarray

NumPy의 핵심은 `ndarray` (N-dimensional array) 객체입니다. 이는 동일한 타입의 요소들로 구성된 다차원 배열입니다.

### 2.1. ndarray 생성

<h4>2.1.1. 파이썬 리스트로부터 생성</h4>
가장 일반적인 `ndarray` 생성 방법은 파이썬 리스트를 `np.array()` 함수에 전달하는 것입니다.

```python
import numpy as np

# 1차원 배열 생성
arr1d = np.array([1, 2, 3, 4, 5])
print(f"1차원 배열: {arr1d}")
print(f"1차원 배열의 차원: {arr1d.ndim}") # ndim: 배열의 차원 수
print(f"1차원 배열의 형태: {arr1d.shape}") # shape: 각 차원의 크기를 튜플로 반환

# 2차원 배열 (행렬) 생성
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2차원 배열:\n{arr2d}")
print(f"2차원 배열의 차원: {arr2d.ndim}")
print(f"2차원 배열의 형태: {arr2d.shape}") # (행의 수, 열의 수)

# 3차원 배열 생성 (예시)
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n3차원 배열:\n{arr3d}")
print(f"3차원 배열의 차원: {arr3d.ndim}")
print(f"3차원 배열의 형태: {arr3d.shape}") # (깊이, 행의 수, 열의 수)
```

<h4>2.1.2. `np.arange()`를 이용한 생성</h4>
`np.arange()` 함수는 파이썬의 `range()` 함수와 유사하게, 지정된 범위 내에서 일정한 간격으로 떨어진 값들로 배열을 생성합니다.

-   `np.arange(n)`: 0부터 `n-1`까지 1씩 증가하는 배열 생성.
-   `np.arange(start, end, step)`: `start`부터 `end-1`까지 `step`만큼 증가하는 배열 생성.

```python
import numpy as np

# 0부터 9까지의 배열 생성
arr_range1 = np.arange(10)
print(f"np.arange(10): {arr_range1}") # [0 1 2 3 4 5 6 7 8 9]

# 5부터 9까지의 배열 생성
arr_range2 = np.arange(5, 10)
print(f"np.arange(5, 10): {arr_range2}") # [5 6 7 8 9]

# 1부터 10까지 2씩 증가하는 배열 생성
arr_range3 = np.arange(1, 11, 2)
print(f"np.arange(1, 11, 2): {arr_range3}") # [1 3 5 7 9]

# 2부터 10까지 2씩 증가하는 배열 생성
arr_range4 = np.arange(2, 11, 2)
print(f"np.arange(2, 11, 2): {arr_range4}") # [2 4 6 8 10]
```

<h4>2.1.3. `np.linspace()`를 이용한 생성</h4>
`np.linspace()` 함수는 지정된 시작 값과 종료 값 사이를 균등한 간격으로 나눈 `num`개의 값으로 배열을 생성합니다. 그래프의 좌표 등을 만들 때 유용하게 사용됩니다.

-   `np.linspace(start, stop, num)`: `start`부터 `stop`까지 `num`개의 균등한 간격의 값 생성. `stop` 값을 포함합니다.

```python
import numpy as np

# 1부터 10까지 50개의 균등한 간격의 값으로 배열 생성
arr_linspace = np.linspace(1, 10, 50)
print(f"np.linspace(1, 10, 50) (첫 5개): {arr_linspace[:5]}")
print(f"np.linspace(1, 10, 50) (마지막 5개): {arr_linspace[-5:]}")
print(f"생성된 배열의 개수: {len(arr_linspace)}") # 50
```

<h4>2.1.4. 특수 배열 생성 함수</h4>
NumPy는 특정 패턴을 가진 배열을 빠르게 생성할 수 있는 다양한 함수를 제공합니다. 이들은 초기화, 테스트, 또는 특정 수학적 연산에 유용합니다.

-   `np.zeros(shape, dtype=float)`: 모든 요소가 0인 배열 생성.
-   `np.ones(shape, dtype=float)`: 모든 요소가 1인 배열 생성.
-   `np.full(shape, fill_value, dtype=None)`: 모든 요소가 지정된 `fill_value`인 배열 생성.
-   `np.empty(shape, dtype=float)`: 초기화되지 않은(임의의 값) 배열 생성. 매우 빠르지만, 사용 전에 값을 할당해야 합니다.
-   `np.eye(N, M=None, k=0, dtype=float)`: `N x M` 크기의 단위 행렬(identity matrix) 생성. 주대각선은 1, 나머지는 0. `k`는 대각선의 위치를 지정합니다.
-   `np.identity(n, dtype=None)`: `n x n` 크기의 정방 단위 행렬 생성.

```python
import numpy as np

# 모든 요소가 0인 2x3 배열 생성
zeros_arr = np.zeros((2, 3))
print(f"모든 요소가 0인 배열:\n{zeros_arr}")
# 결과:
# [[0. 0. 0.]
#  [0. 0. 0.]]

# 모든 요소가 1인 3x2 배열 생성
ones_arr = np.ones((3, 2))
print(f"\n모든 요소가 1인 배열:\n{ones_arr}")
# 결과:
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]]

# 모든 요소가 7인 2x2 배열 생성
full_arr = np.full((2, 2), 7)
print(f"\n모든 요소가 7인 배열:\n{full_arr}")
# 결과:
# [[7 7]
#  [7 7]]

# 초기화되지 않은 2x2 배열 생성 (임의의 값)
empty_arr = np.empty((2, 2))
print(f"\n초기화되지 않은 배열 (empty):\n{empty_arr}")
# 결과: (실행할 때마다 다른 임의의 값)

# 3x3 단위 행렬 생성
eye_arr = np.eye(3)
print(f"\n3x3 단위 행렬 (eye):\n{eye_arr}")
# 결과:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 4x4 단위 행렬 생성 (identity)
identity_arr = np.identity(4)
print(f"\n4x4 단위 행렬 (identity):\n{identity_arr}")
# 결과:
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
```

<h4>2.1.5. `_like` 함수를 이용한 생성</h4>

`_like` 접미사가 붙은 함수들은 기존 배열의 형태(shape)와 데이터 타입(dtype)을 그대로 사용하여 새로운 배열을 생성할 때 유용합니다. 이는 특히 딥러닝에서 텐서(tensor) 연산을 할 때 입력 텐서와 동일한 형태의 텐서를 생성해야 할 경우에 자주 사용됩니다.

-   `np.zeros_like(a, dtype=None)`: `a`와 동일한 형태와 데이터 타입을 가지는 0으로 채워진 배열 생성.
-   `np.ones_like(a, dtype=None)`: `a`와 동일한 형태와 데이터 타입을 가지는 1로 채워진 배열 생성.
-   `np.full_like(a, fill_value, dtype=None)`: `a`와 동일한 형태와 데이터 타입을 가지며, `fill_value`로 채워진 배열 생성.
-   `np.empty_like(a, dtype=None)`: `a`와 동일한 형태와 데이터 타입을 가지는 초기화되지 않은 배열 생성.

```python
import numpy as np

# 원본 배열
original_arr = np.array([[1, 2, 3],
                         [4, 5, 6]], dtype=np.float32)
print(f"원본 배열:\n{original_arr}")
print(f"원본 배열의 형태: {original_arr.shape}, 데이터 타입: {original_arr.dtype}")

# zeros_like
zeros_like_arr = np.zeros_like(original_arr)
print(f"\nzeros_like:\n{zeros_like_arr}")
print(f"형태: {zeros_like_arr.shape}, 데이터 타입: {zeros_like_arr.dtype}")

# ones_like
ones_like_arr = np.ones_like(original_arr, dtype=np.int32) # dtype 변경 가능
print(f"\nones_like:\n{ones_like_arr}")
print(f"형태: {ones_like_arr.shape}, 데이터 타입: {ones_like_arr.dtype}")

# full_like
full_like_arr = np.full_like(original_arr, 99)
print(f"\nfull_like (fill_value=99):\n{full_like_arr}")
print(f"형태: {full_like_arr.shape}, 데이터 타입: {full_like_arr.dtype}")

# empty_like
empty_like_arr = np.empty_like(original_arr)
print(f"\nempty_like:\n{empty_like_arr}")
print(f"형태: {empty_like_arr.shape}, 데이터 타입: {empty_like_arr.dtype}")
```

<h4>2.1.6. 데이터 타입 지정 및 변환</h4>

NumPy 배열의 요소들은 모두 동일한 데이터 타입(dtype)을 가집니다. 데이터 타입을 명시적으로 지정하거나, 생성된 배열의 데이터 타입을 변환할 수 있습니다. 이는 메모리 효율성, 연산 속도, 그리고 특정 라이브러리(예: 딥러닝 프레임워크)와의 호환성을 위해 중요합니다.

-   **생성 시 `dtype` 지정**: `np.array()` 또는 다른 배열 생성 함수에서 `dtype` 인자를 사용하여 데이터 타입을 지정할 수 있습니다.
-   **`astype()` 메서드를 이용한 변환**: 생성된 배열의 데이터 타입을 다른 타입으로 변환할 때 사용합니다. 새로운 배열을 반환합니다.

```python
import numpy as np

# 1. 생성 시 dtype 지정
int_arr = np.array([1, 2, 3], dtype=np.int32)
float_arr = np.array([1.0, 2.5, 3.7], dtype=np.float64)
complex_arr = np.array([1+2j, 3+4j], dtype=np.complex64)

print(f"int_arr: {int_arr}, dtype: {int_arr.dtype}")
print(f"float_arr: {float_arr}, dtype: {float_arr.dtype}")
print(f"complex_arr: {complex_arr}, dtype: {complex_arr.dtype}")

# 2. astype() 메서드를 이용한 변환
original_float_arr = np.array([1.1, 2.5, 3.9])
print(f"\n원본 float 배열: {original_float_arr}, dtype: {original_float_arr.dtype}")

# float -> int (소수점 이하 버림)
int_converted_arr = original_float_arr.astype(np.int32)
print(f"int로 변환: {int_converted_arr}, dtype: {int_converted_arr.dtype}")

# int -> float
int_to_float_arr = np.array([1, 2, 3]).astype(np.float32)
print(f"int -> float 변환: {int_to_float_arr}, dtype: {int_to_float_arr.dtype}")

# boolean 배열 생성 및 변환
bool_arr = np.array([0, 1, 10, 0], dtype=bool)
print(f"\nbool_arr: {bool_arr}, dtype: {bool_arr.dtype}")

# boolean -> int
bool_to_int_arr = bool_arr.astype(np.int8)
print(f"bool -> int 변환: {bool_to_int_arr}, dtype: {bool_to_int_arr.dtype}")
```

### 2.2. ndarray의 속성

`ndarray` 객체는 배열의 특성을 나타내는 여러 유용한 속성들을 가지고 있습니다.

-   `ndim`: 배열의 차원(dimension) 수.
-   `shape`: 배열의 각 차원별 크기를 나타내는 튜플. (예: `(행, 열)` 또는 `(깊이, 행, 열)`)
-   `dtype`: 배열 요소들의 데이터 타입. (예: `int32`, `float64`)
-   `size`: 배열의 전체 요소 개수.

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print(f"배열: \n{arr}")
print(f"차원 (ndim): {arr.ndim}") # 2
print(f"형태 (shape): {arr.shape}") # (2, 3)
print(f"데이터 타입 (dtype): {arr.dtype}") # int32 또는 int64 (시스템에 따라 다름)
print(f"요소 개수 (size): {arr.size}") # 6
```
## 3. ndarray 인덱싱 및 슬라이싱

NumPy `ndarray`는 파이썬 리스트와 유사하게 인덱싱(Indexing)과 슬라이싱(Slicing)을 지원하지만, 다차원 배열에 특화된 강력한 기능을 제공합니다.

### 3.1. 단일 요소 접근

다차원 배열의 특정 요소에 접근할 때는 각 차원의 인덱스를 쉼표(`,`)로 구분하여 사용합니다.

```python
import numpy as np

arr = np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]])

print(f"원본 배열:\n{arr}")

# 1행 1열 (인덱스 0부터 시작) 요소 접근
print(f"\narr[0, 0]: {arr[0, 0]}") # 결과: 1

# 3행 4열 요소 접근
print(f"arr[2, 3]: {arr[2, 3]}") # 결과: 14

# 음수 인덱싱: 파이썬 리스트와 동일하게 뒤에서부터 접근
print(f"arr[-1, -1]: {arr[-1, -1]}") # 결과: 25 (마지막 행, 마지막 열)
```

### 3.2. 슬라이싱

슬라이싱은 배열의 특정 부분집합을 추출하는 강력한 방법입니다. `[start:end:step]` 형식을 사용하며, `end` 인덱스는 포함되지 않습니다 (파이썬 리스트 슬라이싱과 동일).

```python
import numpy as np

arr = np.array([[1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25]])

print(f"원본 배열:\n{arr}")

# 첫 번째 행만 슬라이싱
print(f"\narr[0, :]: {arr[0, :]}") # 결과: [1 2 3 4 5]
print(f"arr[:1]:\n{arr[:1]}") # 결과: [[1 2 3 4 5]] (차원 유지)

# 첫 두 행 슬라이싱
print(f"\narr[:2]:\n{arr[:2]}")
# 결과:
# [[ 1  2  3  4  5]
#  [ 6  7  8  9 10]]

# 1, 3, 5번째 행 슬라이싱 (step 사용)
print(f"\narr[::2]:\n{arr[::2]}")
# 결과:
# [[ 1  2  3  4  5]
#  [11 12 13 14 15]
#  [21 22 23 24 25]]

# 특정 행과 열의 부분집합 슬라이싱
# 2행부터 3행까지 (인덱스 1, 2)의 3열부터 4열까지 (인덱스 2, 3) 추출
print(f"\narr[1:3, 2:4]:\n{arr[1:3, 2:4]}")
# 결과:
# [[ 8  9]
#  [13 14]]

# 모든 행의 특정 열만 선택
print(f"\narr[:, 0]: {arr[:, 0]}") # 결과: [ 1  6 11 16 21]
print(f"arr[:, -1]: {arr[:, -1]}") # 결과: [ 5 10 15 20 25] (마지막 열)
```

### 3.3. 불리언 인덱싱 (Boolean Indexing)

불리언 인덱싱은 배열의 요소들을 조건에 따라 선택하는 강력한 방법입니다. 조건식을 만족하는 요소(True)만 선택하고, 만족하지 않는 요소(False)는 제외합니다.

```python
import numpy as np

arr = np.array([1, 5, 10, 15, 20, 25])

# 배열의 요소가 10보다 큰 경우 선택
boolean_mask = arr > 10
print(f"원본 배열: {arr}")
print(f"불리언 마스크: {boolean_mask}")
print(f"10보다 큰 요소: {arr[boolean_mask]}")
# 결과:
# 원본 배열: [ 1  5 10 15 20 25]
# 불리언 마스크: [False False False  True  True  True]
# 10보다 큰 요소: [15 20 25]

# 조건식을 직접 인덱스로 사용
print(f"\n20보다 작거나 5로 나누어 떨어지는 요소: {arr[(arr < 20) | (arr % 5 == 0)]}")
# 결과: [ 1  5 10 15 20 25]

# 2차원 배열에서의 불리언 인덱싱
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(f"\n2차원 배열:\n{arr_2d}")
print(f"5보다 큰 요소:\n{arr_2d[arr_2d > 5]}")
# 결과:
# 5보다 큰 요소:
# [6 7 8 9]
```

### 3.4. 팬시 인덱싱 (Fancy Indexing)

팬시 인덱싱은 정수 배열을 사용하여 배열의 특정 요소나 행/열을 선택하는 방법입니다. 결과 배열은 원본 배열의 순서와 관계없이 인덱스 배열의 순서를 따릅니다.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

# 특정 인덱스의 요소 선택
indices = [0, 2, 5]
print(f"원본 배열: {arr}")
print(f"선택된 인덱스: {indices}")
print(f"팬시 인덱싱 결과: {arr[indices]}")
# 결과:
# 원본 배열: [10 20 30 40 50 60]
# 선택된 인덱스: [0, 2, 5]
# 팬시 인덱싱 결과: [10 30 60]

# 2차원 배열에서의 팬시 인덱싱
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(f"\n2차원 배열:\n{arr_2d}")

# 특정 행 선택
row_indices = [0, 2]
print(f"선택된 행:\n{arr_2d[row_indices]}")
# 결과:
# 선택된 행:
# [[1 2 3]
#  [7 8 9]]

# 특정 행과 열을 동시에 선택 (결과는 1차원 배열)
# (0,0), (1,1), (2,2) 요소 선택
print(f"\n특정 행과 열 동시 선택: {arr_2d[[0, 1, 2], [0, 1, 2]]}")
# 결과: [1 5 9]

# 특정 행과 열을 동시에 선택 (결과는 2차원 배열)
# (0,1), (0,2), (1,1), (1,2) 요소 선택
print(f"\n특정 행과 열 동시 선택 (2D 결과):\n{arr_2d[np.array([[0,0],[1,1]]), np.array([[1,2],[1,2]])]}")
# 결과:
# [[2 3]
#  [5 6]]
```

## 4. NumPy 기본 연산

NumPy는 배열 간의 빠르고 효율적인 연산을 위해 다양한 기능을 제공합니다. 이는 ML/DL에서 행렬 연산을 수행하는 데 필수적입니다.

### 4.1. 벡터화된 연산

NumPy 배열은 반복문 없이 배열의 모든 요소에 대해 연산을 수행하는 "벡터화된(vectorized)" 연산을 지원합니다. 이는 파이썬의 일반적인 반복문보다 훨씬 빠릅니다.

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 요소별 덧셈
sum_arr = arr1 + arr2
print(f"요소별 덧셈: {sum_arr}") # [5 7 9]

# 요소별 곱셈
mul_arr = arr1 * arr2
print(f"요소별 곱셈: {mul_arr}") # [4 10 18]

# 스칼라 연산: 배열의 모든 요소에 스칼라 값 적용
scalar_mul = arr1 * 10
print(f"스칼라 곱셈: {scalar_mul}") # [10 20 30]
```

### 4.2. 브로드캐스팅 (Broadcasting)

브로드캐스팅은 NumPy가 서로 다른 형태(shape)의 배열 간에 산술 연산을 수행할 수 있도록 하는 강력한 메커니즘입니다. 작은 배열이 큰 배열의 형태에 맞춰 자동으로 확장되어 연산이 가능해집니다.

```python
import numpy as np

arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]]) # 형태: (2, 3)
scalar = 10 # 스칼라 값

# 2D 배열과 스칼라 간의 덧셈 (스칼라가 배열의 모든 요소에 브로드캐스팅됨)
result_scalar_add = arr_2d + scalar
print(f"2D 배열 + 스칼라:\n{result_scalar_add}")
# 결과:
# [[11 12 13]
#  [14 15 16]]

vector_row = np.array([100, 200, 300]) # 형태: (3,)

# 2D 배열과 1D 행 벡터 간의 덧셈 (1D 벡터가 각 행에 브로드캐스팅됨)
result_vector_add = arr_2d + vector_row
print(f"\n2D 배열 + 1D 행 벡터:\n{result_vector_add}")
# 결과:
# [[101 202 303]
#  [104 205 306]]
```

### 4.3. 행렬 곱셈 (Dot Product)

NumPy는 선형대수 연산, 특히 행렬 곱셈을 효율적으로 수행하는 기능을 제공합니다. `np.dot()` 함수나 `@` 연산자를 사용합니다.

```python
import numpy as np

x = np.array([[1, 2],
              [3, 4]]) # 2x2 행렬
y = np.array([[5, 6],
              [7, 8]]) # 2x2 행렬

v = np.array([9, 10]) # 1x2 벡터
w = np.array([11, 12]) # 1x2 벡터

# 벡터 내적 (Dot Product) - 결과는 스칼라
print(f"벡터 v와 w의 내적 (v.dot(w)): {v.dot(w)}") # 9*11 + 10*12 = 99 + 120 = 219
print(f"벡터 v와 w의 내적 (np.dot(v, w)): {np.dot(v, w)}") # 219

# 행렬 곱셈 - 결과는 행렬
print(f"\n행렬 x와 y의 곱셈 (x.dot(y)):\n{x.dot(y)}")
print(f"행렬 x와 y의 곱셈 (np.dot(x, y)):\n{np.dot(x, y)}")
# 결과:
# [[1*5 + 2*7, 1*6 + 2*8],
#  [3*5 + 4*7, 3*6 + 4*8]]
# = [[ 5+14,  6+16],
#    [15+28, 18+32]]
# = [[19, 22],
#    [43, 50]]

# Python 3.5+ 부터는 @ 연산자로 행렬 곱셈 가능
print(f"\n행렬 x와 y의 곱셈 (x @ y):\n{x @ y}")
```

### 4.4. 집계 함수 (Aggregation Functions)

NumPy는 배열의 요소들에 대한 합계, 평균, 최댓값, 최솟값 등을 계산하는 다양한 집계 함수를 제공합니다. `axis` 파라미터를 사용하여 특정 축(행 또는 열)을 따라 연산을 수행할 수 있습니다.

-   `axis=None` (기본값): 배열의 모든 요소에 대해 연산 수행.
-   `axis=0`: 각 열(column)을 따라 연산 수행 (행 방향으로 합쳐짐).
-   `axis=1`: 각 행(row)을 따라 연산 수행 (열 방향으로 합쳐짐).

```python
import numpy as np

x = np.array([[1, 2],
              [3, 4]])

print(f"원본 배열 x:\n{x}")

# 모든 요소의 합계
print(f"\n모든 요소의 합계 (np.sum(x)): {np.sum(x)}") # 1 + 2 + 3 + 4 = 10

# 각 열의 합계 (axis=0)
print(f"각 열의 합계 (np.sum(x, axis=0)): {np.sum(x, axis=0)}") # [1+3, 2+4] = [4 6]

# 각 행의 합계 (axis=1)
print(f"각 행의 합계 (np.sum(x, axis=1)): {np.sum(x, axis=1)}") # [1+2, 3+4] = [3 7]

# 다른 집계 함수 예시
print(f"\n모든 요소의 평균 (np.mean(x)): {np.mean(x)}")
print(f"각 열의 최댓값 (np.max(x, axis=0)): {np.max(x, axis=0)}")
print(f"각 행의 최솟값 (np.min(x, axis=1)): {np.min(x, axis=1)}")
```

### 4.5. 전치 행렬 (Transpose)

전치 행렬은 행과 열을 바꾼 행렬입니다. NumPy에서는 `.T` 속성을 사용하여 쉽게 구할 수 있습니다.

```python
import numpy as np

x = np.array([[1, 2],
              [3, 4]])

print(f"원본 행렬 x:\n{x}")

# 전치 행렬
x_transpose = x.T
print(f"\n전치 행렬 x.T:\n{x_transpose}")
# 결과:
# [[1 3]
#  [2 4]]
```

### 4.6. 배열 형태 변경 (Reshaping)

배열의 형태(shape)를 변경하는 것은 머신러닝 및 딥러닝에서 데이터 전처리 시 매우 자주 사용되는 기능입니다. `reshape()` 메서드를 사용하며, 원본 배열의 요소 개수는 유지되어야 합니다.

-   `reshape(new_shape)`: 배열의 형태를 `new_shape`로 변경합니다. `-1`을 사용하면 해당 차원의 크기를 자동으로 계산합니다.

```python
import numpy as np

arr = np.arange(1, 13) # 1부터 12까지의 1차원 배열
print(f"원본 1차원 배열: {arr}")

# 1차원 배열을 3x4 2차원 배열로 변경
reshaped_arr = arr.reshape(3, 4)
print(f"\n3x4로 변경된 배열:\n{reshaped_arr}")
# 결과:
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]

# -1을 사용하여 차원 자동 계산
reshaped_arr_auto = arr.reshape(2, -1) # 2행, 열은 자동으로 계산 (6열)
print(f"\n2행으로 자동 계산된 배열:\n{reshaped_arr_auto}")
# 결과:
# [[ 1  2  3  4  5  6]
#  [ 7  8  9 10 11 12]]

# 1차원 배열을 2x2x3 3차원 배열로 변경
reshaped_3d_arr = arr.reshape(2, 2, 3)
print(f"\n2x2x3으로 변경된 3차원 배열:\n{reshaped_3d_arr}")
# 결과:
# [[[ 1  2  3]
#   [ 4  5  6]]
#
#  [[ 7  8  9]
#   [10 11 12]]]
```

### 4.7. 배열 결합 및 분할 (Concatenation & Splitting)

여러 배열을 하나로 합치거나, 하나의 배열을 여러 부분으로 나누는 기능은 데이터 전처리 과정에서 매우 중요합니다.

<h4>4.7.1. 배열 결합 (Concatenation)</h4>

-   `np.concatenate((arr1, arr2, ...), axis=0)`: 여러 배열을 지정된 축(axis)을 따라 결합합니다. 기본값은 `axis=0` (행 방향).
-   `np.vstack((arr1, arr2, ...))`: 수직으로(행 방향) 배열을 쌓습니다. `np.concatenate(..., axis=0)`과 유사합니다.
-   `np.hstack((arr1, arr2, ...))`: 수평으로(열 방향) 배열을 쌓습니다. `np.concatenate(..., axis=1)`과 유사합니다.

```python
import numpy as np

arr1 = np.array([[1, 2],
                 [3, 4]])
arr2 = np.array([[5, 6],
                 [7, 8]])

print(f"arr1:\n{arr1}")
print(f"\narr2:\n{arr2}")

# 행 방향으로 결합 (수직 스택)
concat_row = np.concatenate((arr1, arr2), axis=0)
print(f"\n행 방향 결합 (axis=0):\n{concat_row}")
# 결과:
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# 열 방향으로 결합 (수평 스택)
concat_col = np.concatenate((arr1, arr2), axis=1)
print(f"\n열 방향 결합 (axis=1):\n{concat_col}")
# 결과:
# [[1 2 5 6]
#  [3 4 7 8]]

# vstack 예시
vstack_arr = np.vstack((arr1, arr2))
print(f"\nvstack 결과:\n{vstack_arr}")

# hstack 예시
hstack_arr = np.hstack((arr1, arr2))
print(f"\nhstack 결과:\n{hstack_arr}")
```

<h4>4.7.2. 배열 분할 (Splitting)</h4>

-   `np.split(ary, indices_or_sections, axis=0)`: 배열을 여러 하위 배열로 분할합니다.
-   `np.vsplit(ary, indices_or_sections)`: 배열을 수직으로(행 방향) 분할합니다.
-   `np.hsplit(ary, indices_or_sections)`: 배열을 수평으로(열 방향) 분할합니다.

```python
import numpy as np

arr = np.arange(16).reshape(4, 4)
print(f"원본 배열:\n{arr}")

# 수직으로 2개로 분할
vsplit_arrs = np.vsplit(arr, 2)
print(f"\n수직 분할 결과 (2개):\n{vsplit_arrs[0]}\n\n{vsplit_arrs[1]}")

# 수평으로 4개로 분할
hsplit_arrs = np.hsplit(arr, 4)
print(f"\n수평 분할 결과 (4개):\n{hsplit_arrs[0]}\n\n{hsplit_arrs[1]}\n\n{hsplit_arrs[2]}\n\n{hsplit_arrs[3]}")
```

## 5. NumPy 파일 입출력


NumPy는 배열 데이터를 효율적으로 디스크에 저장하고 로드하는 기능을 제공합니다. 이는 대규모 데이터셋을 다루거나 모델의 가중치 등을 저장할 때 유용합니다.

### 5.1. 단일 배열 저장/로드: `np.save()`, `np.load()`

`np.save()`는 단일 NumPy 배열을 `.npy` 확장자를 가진 바이너리 파일로 저장하고, `np.load()`는 이 파일을 다시 로드합니다.

```python
import numpy as np

# 저장할 배열 생성
data_to_save = np.random.rand(5) # 0과 1 사이의 난수 5개로 구성된 1차원 배열
print(f"저장할 배열: {data_to_save}")

# 배열 저장
# 파일 확장자는 자동으로 .npy가 붙습니다.
np.save('datafile.npy', data_to_save)
print("\n'datafile.npy' 파일이 생성되었습니다.")

# 저장된 배열 로드
# 로드하기 전에 변수를 비워두어 제대로 로드되는지 확인
loaded_data = []
print(f"로드 전 변수: {loaded_data}")

loaded_data = np.load('datafile.npy')
print(f"로드된 배열: {loaded_data}")
# 로드된 배열은 저장했던 배열과 동일합니다.
```

### 5.2. 여러 배열 저장/로드: `np.savez()`

`np.savez()`는 여러 개의 NumPy 배열을 `.npz` 확장자를 가진 압축 파일 하나에 저장합니다. 각 배열은 키-값 쌍의 형태로 저장되며, 로드할 때는 딕셔너리처럼 접근할 수 있습니다.

```python
import numpy as np

# 저장할 여러 배열 생성
data1_save = np.arange(1, 11) # 1부터 10까지의 정수 배열
data2_save = np.random.rand(10) # 0과 1 사이의 난수 10개 배열
print(f"저장할 data1: {data1_save}")
print(f"저장할 data2: {data2_save}")

# 여러 배열을 'data.npz' 파일에 저장
# key1, key2는 배열에 접근할 때 사용할 이름입니다.
np.savez('data.npz', key1=data1_save, key2=data2_save)
print("\n'data.npz' 파일이 생성되었습니다.")

# 저장된 여러 배열 로드
outfile = np.load('data.npz')

# npz 파일에 저장된 배열들의 키(이름) 확인
print(f"\n저장된 배열 키: {outfile.files}") # ['key1', 'key2']

# 키를 사용하여 각 배열에 접근
loaded_data1 = outfile['key1']
loaded_data2 = outfile['key2']

print(f"로드된 data1: {loaded_data1}")
print(f"로드된 data2: {loaded_data2}")
```

## 6. 기타 유용한 기능

### 6.1. 난수 생성

NumPy의 `random` 모듈은 다양한 확률 분포로부터 난수를 생성하는 강력한 기능을 제공합니다. 이는 모델 초기화, 데이터 증강, 시뮬레이션 등에 활용됩니다.

```python
import numpy as np

# 0.0에서 1.0 사이의 균일 분포 난수 5개 생성
random_uniform = np.random.rand(5)
print(f"균일 분포 난수 (rand): {random_uniform}")

# 표준 정규 분포(평균 0, 표준편차 1)를 따르는 난수 5개 생성
random_normal = np.random.randn(5)
print(f"표준 정규 분포 난수 (randn): {random_normal}")

# 지정된 범위 [low, high)에서 정수 난수 생성
random_int = np.random.randint(low=1, high=10, size=5)
print(f"정수 난수 (randint): {random_int}")

# 재현 가능한 난수 생성을 위한 시드(seed) 설정
np.random.seed(42)
rand_a = np.random.rand(3)
np.random.seed(42)
rand_b = np.random.rand(3)
print(f"\n시드 설정 후 난수 A: {rand_a}")
print(f"시드 설정 후 난수 B: {rand_b}") # rand_a와 동일한 결과
```

### 6.2. 푸리에 변환 (Fourier Transform)

NumPy의 `fft` 모듈은 푸리에 변환을 수행하는 기능을 제공합니다. 푸리에 변환은 신호 처리, 이미지 처리, 스펙트럼 분석 등 다양한 분야에서 사용됩니다.

**개념**: 푸리에 변환(Fourier Transform, FT)은 시간에 대한 함수(혹은 신호)를 함수를 구성하고 있는 주파수 성분으로 분해하는 작업입니다. 예를 들어, 음악에서 악보에 코드를 나타낼 때 주파수(음높이)로 표현되는 것과 유사합니다. 복잡한 신호를 단순한 주파수 성분들의 합으로 나타낼 수 있게 해줍니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 간단한 신호 생성 (사인파)
sampling_rate = 100 # 1초에 100개의 샘플
t = np.linspace(0, 1, sampling_rate, endpoint=False) # 1초 동안의 시간 축
frequency1 = 5 # 5 Hz
frequency2 = 20 # 20 Hz
signal = 0.5 * np.sin(2 * np.pi * frequency1 * t) + 0.2 * np.sin(2 * np.pi * frequency2 * t)

# 푸리에 변환 수행
fft_result = np.fft.fft(signal)

# 주파수 축 생성
freq_axis = np.fft.fftfreq(sampling_rate, d=1/sampling_rate)

# 양의 주파수 성분만 추출 (대칭이므로)
positive_freq_idx = freq_axis >= 0
positive_freq_axis = freq_axis[positive_freq_idx]
positive_fft_magnitude = np.abs(fft_result[positive_freq_idx])

print("원본 신호 (첫 10개):", signal[:10])
print("\n푸리에 변환 결과 (크기, 첫 10개):", positive_fft_magnitude[:10])

# 시각화 (선택 사항: Matplotlib 설치 필요)
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(t, signal)
# plt.title('Time Domain Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')

# plt.subplot(1, 2, 2)
# plt.plot(positive_freq_axis, positive_fft_magnitude)
# plt.title('Frequency Domain (Magnitude Spectrum)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
```

## 7. 선형대수 (Linear Algebra)

선형대수는 머신러닝과 딥러닝의 핵심 기반이 되는 수학 분야입니다. NumPy는 `numpy.linalg` 모듈을 통해 강력하고 효율적인 선형대수 연산 기능을 제공합니다.

### 7.1. 역행렬 (Inverse Matrix)

역행렬은 어떤 행렬 A에 대해 곱했을 때 단위 행렬(Identity Matrix)이 되는 행렬을 의미합니다. 역행렬은 선형 시스템 해법, 최소 제곱법 등 다양한 분야에서 활용됩니다. 모든 정방 행렬이 역행렬을 가지는 것은 아니며, 행렬식이 0이 아닌 경우에만 존재합니다.

```python
import numpy as np

# 역행렬을 구할 2x2 행렬
A = np.array([[1, 2],
              [3, 4]])

print(f"원본 행렬 A:\n{A}")

# 역행렬 계산
try:
    A_inv = np.linalg.inv(A)
    print(f"\n행렬 A의 역행렬:\n{A_inv}")

    # 원본 행렬과 역행렬을 곱하면 단위 행렬이 되는지 확인
    identity_check = A @ A_inv
    print(f"\nA @ A_inv (단위 행렬 확인):\n{identity_check}")
    # 결과는 부동 소수점 오차로 인해 완벽한 단위 행렬이 아닐 수 있습니다.
    # np.allclose()를 사용하여 근사적으로 같은지 확인하는 것이 좋습니다.
    print(f"A @ A_inv가 단위 행렬과 근사적으로 같은가? {np.allclose(identity_check, np.eye(2))}")

except np.linalg.LinAlgError:
    print("\n이 행렬은 역행렬을 가지지 않습니다 (특이 행렬).")

# 역행렬이 존재하지 않는 특이 행렬 (Singular Matrix) 예시
B = np.array([[1, 2],
              [2, 4]]) # 두 번째 행이 첫 번째 행의 2배이므로 선형 종속

print(f"\n원본 행렬 B:\n{B}")
try:
    B_inv = np.linalg.inv(B)
    print(f"\n행렬 B의 역행렬:\n{B_inv}")
except np.linalg.LinAlgError:
    print("\n행렬 B는 역행렬을 가지지 않습니다 (특이 행렬).")
```

### 7.2. 행렬식 (Determinant)

행렬식(Determinant)은 정방 행렬에 스칼라 값을 할당하는 함수로, 행렬이 선형 변환을 나타낼 때 그 변환이 공간의 부피를 얼마나 확장하거나 축소하는지를 나타냅니다. 행렬식이 0이면 해당 행렬은 역행렬을 가지지 않으며, 선형 종속적인 관계를 가집니다.

```python
import numpy as np

# 행렬식 계산할 2x2 행렬
A = np.array([[1, 2],
              [3, 4]])

# 행렬식 계산
det_A = np.linalg.det(A)
print(f"행렬 A:\n{A}")
print(f"행렬 A의 행렬식: {det_A:.4f}") # 결과: -2.0000

# 행렬식 계산할 3x3 행렬
C = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

det_C = np.linalg.det(C)
print(f"\n행렬 C:\n{C}")
print(f"행렬 C의 행렬식: {det_C:.4f}") # 결과: 0.0000 (선형 종속이므로)
```

### 7.3. 고유값과 고유벡터 (Eigenvalues and Eigenvectors)

고유값(Eigenvalue)과 고유벡터(Eigenvector)는 선형 변환에서 특별한 관계를 가지는 벡터와 스칼라입니다. 행렬 A에 선형 변환을 가했을 때, 방향은 변하지 않고 크기만 변하는 벡터를 고유벡터라 하고, 이때 변하는 크기 비율을 고유값이라고 합니다. 주성분 분석(PCA), 스펙트럼 분석, 양자 역학 등 다양한 분야에서 활용됩니다.

```python
import numpy as np

# 고유값과 고유벡터를 구할 행렬
A = np.array([[0, 1],
              [1, 0]]) # 대칭 행렬 (예: 대칭 행렬은 항상 실수 고유값을 가짐)

print(f"원본 행렬 A:\n{A}")

# 고유값(eigenvalues)과 고유벡터(eigenvectors) 계산
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"\n고유값: {eigenvalues}")
print(f"고유벡터:\n{eigenvectors}")

# 고유값과 고유벡터의 관계 확인: A @ v = lambda * v
# 첫 번째 고유값과 고유벡터
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0]
print(f"\n첫 번째 고유값 ({lambda1:.2f})과 고유벡터 ({v1}):")
print(f"A @ v1:\n{A @ v1}")
print(f"lambda1 * v1:\n{lambda1 * v1}")
print(f"A @ v1 == lambda1 * v1? {np.allclose(A @ v1, lambda1 * v1)}")

# 두 번째 고유값과 고유벡터
lambda2 = eigenvalues[1]
v2 = eigenvectors[:, 1]
print(f"\n두 번째 고유값 ({lambda2:.2f})과 고유벡터 ({v2}):")
print(f"A @ v2:\n{A @ v2}")
print(f"lambda2 * v2:\n{lambda2 * v2}")
print(f"A @ v2 == lambda2 * v2? {np.allclose(A @ v2, lambda2 * v2)}")
```

### 7.4. 선형 시스템 해법 (Solving Linear Systems)

선형 시스템 $Ax = b$는 머신러닝에서 회귀 분석, 최적화 문제 등 다양한 형태로 나타납니다. NumPy의 `np.linalg.solve()` 함수는 이러한 선형 시스템의 해 $x$를 효율적으로 찾을 수 있도록 돕습니다.

```python
import numpy as np

# 선형 시스템 Ax = b 정의
# A: 계수 행렬
A = np.array([[3, 1],
              [1, 2]])

# b: 결과 벡터
b = np.array([9, 8])

print(f"계수 행렬 A:\n{A}")
print(f"결과 벡터 b: {b}")

# 선형 시스템 Ax = b의 해 x 계산
x = np.linalg.solve(A, b)
print(f"\n선형 시스템의 해 x: {x}")
# 결과: [2. 3.] (즉, x1=2, x2=3)

# 해가 올바른지 확인: A @ x = b
check_b = A @ x
print(f"\nA @ x (확인): {check_b}")
print(f"A @ x == b? {np.allclose(check_b, b)}")

```

### 7.5. 벡터 노름 (Vector Norms)

벡터 노름(Vector Norm)은 벡터의 크기 또는 길이를 측정하는 방법입니다. 머신러닝에서는 주로 모델의 가중치 크기를 제한하거나, 오차를 측정하는 손실 함수(Loss Function)의 일부로 사용됩니다.

-   **L1 노름 (맨해튼 노름, Manhattan Norm)**: 벡터 요소들의 절댓값의 합. 특성 선택(Feature Selection)에 유용하며, 희소성(Sparsity)을 유도합니다.
    $$ ||x||_1 = \sum_{i=1}^{n} |x_i| $$
-   **L2 노름 (유클리드 노름, Euclidean Norm)**: 벡터 요소들의 제곱의 합의 제곱근. 가장 흔히 사용되는 노름으로, 벡터의 기하학적 길이를 나타냅니다. 과적합(Overfitting) 방지에 사용되는 L2 정규화(Regularization)의 기반이 됩니다.
    $$ ||x||_2 = \sqrt{\sum_{i=1}^{n} x_i^2} $$
-   **무한대 노름 (Maximum Norm)**: 벡터 요소들 중 절댓값이 가장 큰 값. 주로 최악의 경우(worst-case) 시나리오를 고려할 때 사용됩니다.
    $$ ||x||_\infty = \max_{i} |x_i| $$

NumPy의 `np.linalg.norm()` 함수를 사용하여 다양한 노름을 계산할 수 있습니다.

```python
import numpy as np

v = np.array([1, -2, 3])
print(f"벡터 v: {v}")

# L1 노름 계산
l1_norm = np.linalg.norm(v, ord=1)
print(f"\nL1 노름: {l1_norm}") # |1| + |-2| + |3| = 1 + 2 + 3 = 6

# L2 노름 계산
l2_norm = np.linalg.norm(v, ord=2)
print(f"L2 노름: {l2_norm:.4f}") # sqrt(1^2 + (-2)^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14) approx 3.7417

# 무한대 노름 계산
inf_norm = np.linalg.norm(v, ord=np.inf)
print(f"무한대 노름: {inf_norm}") # max(|1|, |-2|, |3|) = 3

# 행렬 노름 (예시: Frobenius Norm)
M = np.array([[1, 2],
              [3, 4]])
print(f"\n행렬 M:\n{M}")

frobenius_norm = np.linalg.norm(M, ord='fro')
print(f"프로베니우스 노름: {frobenius_norm:.4f}") # sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(1+4+9+16) = sqrt(30) approx 5.4772
```

### 7.6. 특이값 분해 (Singular Value Decomposition, SVD)

특이값 분해(Singular Value Decomposition, SVD)는 임의의 행렬을 세 개의 특수한 행렬의 곱으로 분해하는 강력한 행렬 분해 기법입니다. 이는 차원 축소(Dimensionality Reduction), 추천 시스템, 이미지 압축, 잠재 의미 분석(Latent Semantic Analysis) 등 머신러닝의 다양한 분야에서 활용됩니다.

임의의 $m \times n$ 행렬 $A$는 다음과 같이 분해될 수 있습니다:
$$ A = U \Sigma V^T $$
여기서:
-   $U$: $m \times m$ 크기의 직교 행렬 (Orthogonal Matrix). $U U^T = I$.
-   $\Sigma$: $m \times n$ 크기의 직사각 대각 행렬(Rectangular Diagonal Matrix). 대각선에는 특이값(Singular Values)이 내림차순으로 정렬되어 있으며, 나머지 요소는 0입니다. 특이값은 행렬의 중요도나 정보량을 나타냅니다.
-   $V^T$: $n \times n$ 크기의 직교 행렬 $V$의 전치 행렬. $V V^T = I$.

NumPy의 `np.linalg.svd()` 함수를 사용하여 SVD를 수행할 수 있습니다.

```python
import numpy as np

# SVD를 수행할 행렬
A = np.array([[1, 1, 1, 0, 0],
              [3, 3, 3, 0, 0],
              [4, 4, 4, 0, 0],
              [5, 5, 5, 0, 0],
              [0, 0, 0, 4, 4],
              [0, 0, 0, 5, 5],
              [0, 0, 0, 2, 2]])

print(f"원본 행렬 A:\n{A}")

# SVD 수행
U, s, Vt = np.linalg.svd(A)

print(f"\nU 행렬 (좌측 특이 벡터):\n{U}")
print(f"\n특이값 (Singular Values): {s}") # 대각 행렬의 대각 요소들
print(f"\nVt 행렬 (우측 특이 벡터의 전치):\n{Vt}")

# 특이값 s를 대각 행렬로 변환
Sigma = np.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = np.diag(s)

# 원본 행렬 복원 확인 (U @ Sigma @ Vt)
reconstructed_A = U @ Sigma @ Vt
print(f"\n재구성된 행렬 (U @ Sigma @ Vt):\n{reconstructed_A}")

# 원본과 재구성된 행렬이 근사적으로 같은지 확인
print(f"\n원본과 재구성된 행렬이 근사적으로 같은가? {np.allclose(A, reconstructed_A)}")
```

## 8. 추가 고급 기능

NumPy는 앞서 다룬 기본 기능 외에도 데이터 분석 및 머신러닝/딥러닝에서 유용하게 활용될 수 있는 다양한 고급 기능들을 제공합니다.

### 8.1. 정렬 및 검색

배열의 요소를 정렬하거나 특정 값을 검색하는 기능은 데이터 분석에서 필수적입니다.

-   `np.sort(a, axis=-1, kind=None, order=None)`: 배열의 정렬된 복사본을 반환합니다. `axis`를 지정하여 특정 축을 따라 정렬할 수 있습니다.
-   `ndarray.sort(axis=-1, kind=None, order=None)`: 배열을 제자리에서(in-place) 정렬합니다 (원본 변경).
-   `np.argsort(a, axis=-1, kind=None, order=None)`: 정렬된 배열의 인덱스를 반환합니다.
-   `np.where(condition, [x, y])`: 조건에 따라 요소를 선택합니다. `x`와 `y`가 제공되면, `condition`이 True인 위치에는 `x`의 요소를, False인 위치에는 `y`의 요소를 반환합니다. `x`와 `y`가 없으면 True인 요소의 인덱스를 반환합니다.
-   `np.unique(ar, return_index=False, return_inverse=False, return_counts=False)`: 배열에서 고유한 요소들을 반환합니다.

```python
import numpy as np

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"원본 배열: {arr}")

# 배열 정렬 (복사본 반환)
sorted_arr = np.sort(arr)
print(f"정렬된 배열 (np.sort): {sorted_arr}")
print(f"원본 배열 (변화 없음): {arr}")

# 배열 제자리 정렬 (원본 변경)
arr.sort()
print(f"제자리 정렬된 배열 (arr.sort()): {arr}")

# 정렬된 인덱스 반환
idx = np.argsort(arr)
print(f"정렬된 인덱스: {idx}")
print(f"인덱스를 이용한 정렬 확인: {arr[idx]}")

# 2차원 배열 정렬
arr_2d = np.array([[3, 1, 2],
                   [6, 5, 4]])
print(f"\n원본 2차원 배열:\n{arr_2d}")

# 행 방향으로 정렬 (axis=1)
sorted_2d_row = np.sort(arr_2d, axis=1)
print(f"행 방향 정렬:\n{sorted_2d_row}")

# 열 방향으로 정렬 (axis=0)
sorted_2d_col = np.sort(arr_2d, axis=0)
print(f"열 방향 정렬:\n{sorted_2d_col}")

# np.where를 이용한 조건부 선택
scores = np.array([85, 92, 78, 65, 95])
pass_fail = np.where(scores >= 80, 'Pass', 'Fail')
print(f"\n점수: {scores}")
print(f"합격/불합격: {pass_fail}")

# 고유한 요소 찾기
duplicate_arr = np.array([1, 2, 2, 3, 1, 4, 5, 5])
unique_elements = np.unique(duplicate_arr)
print(f"\n고유한 요소: {unique_elements}")
```

### 8.2. 브로드캐스팅 규칙 심화

브로드캐스팅은 NumPy의 강력한 기능이지만, 그 규칙을 정확히 이해하는 것이 중요합니다. 두 배열의 형태가 호환되는지 여부는 다음 규칙에 따라 결정됩니다.

1.  **차원 수가 더 작은 배열의 형태가 1인 차원으로 확장됩니다.**
    -   예: `(3,)` 형태의 1차원 배열과 `(2, 3)` 형태의 2차원 배열이 있다면, 1차원 배열은 `(1, 3)`으로 확장됩니다.
2.  **두 배열의 차원 크기가 일치하거나, 둘 중 하나의 차원 크기가 1이어야 합니다.**
    -   차원 크기가 1인 경우, 해당 차원은 다른 배열의 크기에 맞춰 확장됩니다.
    -   이 규칙을 만족하지 않으면 `ValueError`가 발생합니다.

**예시:**

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]]) # 형태: (2, 3)
B = np.array([10, 20, 30]) # 형태: (3,)

# B는 (1, 3)으로 확장되어 A의 각 행에 브로드캐스팅됩니다.
C = A + B
print(f"A + B:\n{C}")
# 결과:
# [[11 22 33]
#  [14 25 36]]


D = np.array([[100],
              [200]]) # 형태: (2, 1)

# D는 (2, 3)으로 확장되어 A의 각 열에 브로드캐스팅됩니다.
E = A + D
print(f"\nA + D:\n{E}")
# 결과:
# [[101 102 103]
#  [204 205 206]]

# 호환되지 않는 형태의 예시
F = np.array([1, 2]) # 형태: (2,)

try:
    G = A + F
    print(G)
except ValueError as e:
    print(f"\n오류 발생: {e}")
    print("A의 형태 (2, 3)과 F의 형태 (2,)는 브로드캐스팅 규칙에 따라 호환되지 않습니다.")
    print("F는 (1, 2)로 확장되지만, A의 마지막 차원 (3)과 F의 마지막 차원 (2)이 일치하지 않습니다.")
```

### 8.3. 마스킹 (Masking)

마스킹(Masking)은 불리언 배열을 사용하여 원본 배열에서 특정 조건을 만족하는 요소들을 선택하거나 수정하는 기법입니다. 이는 데이터 필터링, 특정 값 대체, 이상치 처리 등 다양한 데이터 전처리 작업에 활용됩니다.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

# 1. 불리언 마스크 생성
mask = arr > 30
print(f"원본 배열: {arr}")
print(f"마스크 (arr > 30): {mask}")

# 2. 마스크를 이용한 요소 선택
selected_elements = arr[mask]
print(f"마스크를 이용한 선택: {selected_elements}")
# 결과: [40 50 60]

# 3. 마스크를 이용한 요소 수정
arr[arr % 2 == 0] = 0 # 짝수인 요소들을 0으로 변경
print(f"\n짝수를 0으로 변경한 배열: {arr}")
# 결과: [10  0 30  0 50  0]

# 4. 2차원 배열에서의 마스킹
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

mask_2d = arr_2d % 3 == 0 # 3의 배수인 요소
print(f"\n2차원 배열:\n{arr_2d}")
print(f"3의 배수 마스크:\n{mask_2d}")
print(f"3의 배수인 요소:\n{arr_2d[mask_2d]}")
# 결과:
# [3 6 9]

# 5. np.where를 이용한 마스킹 (조건에 따라 다른 값 할당)
modified_arr_2d = np.where(arr_2d > 5, arr_2d * 10, arr_2d)
print(f"\n5보다 큰 요소는 10배, 아니면 그대로:\n{modified_arr_2d}")
# 결과:
# [[ 1  2  3]
#  [ 4  5 60]
#  [70 80 90]]

```

### 8.4. 구조화 배열 (Structured Arrays)

구조화 배열(Structured Arrays)은 NumPy 배열 내에 다른 데이터 타입의 요소들을 저장할 수 있게 해주는 강력한 기능입니다. 이는 데이터베이스의 테이블이나 CSV 파일처럼 이질적인 데이터를 다룰 때 유용합니다. 각 요소는 이름(필드)과 데이터 타입을 가지는 구조체처럼 동작합니다.

```python
import numpy as np

# 구조화 배열의 데이터 타입 정의
dtype = [('name', 'S10'), ('age', 'i4'), ('height', 'f8')]
# 'name': 문자열 (최대 10바이트)
# 'age': 4바이트 정수
# 'height': 8바이트 부동 소수점

# 구조화 배열 생성
people = np.array([('Alice', 25, 165.5),
                   ('Bob', 30, 178.0),
                   ('Charlie', 22, 170.2)],
                  dtype=dtype)

print(f"구조화 배열:\n{people}")
print(f"데이터 타입: {people.dtype}")

# 필드(열) 접근
print(f"\n이름: {people['name']}")
print(f"나이: {people['age']}")
print(f"키: {people['height']}")

# 조건부 선택 (불리언 인덱싱 활용)
older_than_25 = people[people['age'] > 25]
print(f"\n25세 초과:\n{older_than_25}")

# 특정 필드만 선택하여 새로운 배열 생성
names_and_ages = people[['name', 'age']]
print(f"\n이름과 나이:\n{names_and_ages}")

# 데이터 수정
people['age'][0] = 26
print(f"\n나이 수정 후:\n{people}")
```

### 8.5. 메모리 관리 및 뷰 vs. 복사

NumPy는 대규모 데이터를 효율적으로 처리하기 위해 메모리 관리에 최적화되어 있습니다. 배열 연산 시 '뷰(view)'를 반환하는 경우와 '복사(copy)'를 반환하는 경우를 이해하는 것이 중요합니다.

-   **뷰 (View)**: 원본 배열의 데이터를 공유합니다. 뷰를 수정하면 원본 배열도 변경됩니다. 슬라이싱, `reshape()`, `transpose()` 등은 일반적으로 뷰를 반환합니다.
-   **복사 (Copy)**: 원본 배열의 데이터를 완전히 복사하여 새로운 배열을 생성합니다. 복사본을 수정해도 원본 배열은 변경되지 않습니다. `copy()` 메서드를 명시적으로 사용하거나, 일부 연산(예: 불리언 인덱싱 결과)은 복사본을 반환합니다.

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(f"원본 배열: {arr}")

# 1. 슬라이싱 (뷰)
view_arr = arr[1:4]
print(f"슬라이싱 (뷰): {view_arr}")

view_arr[0] = 99 # 뷰 수정
print(f"뷰 수정 후 원본 배열: {arr}") # 원본도 변경됨

# 2. copy() 메서드 (복사)
copy_arr = arr.copy()
print(f"\n복사본: {copy_arr}")

copy_arr[0] = 100 # 복사본 수정
print(f"복사본 수정 후 원본 배열: {arr}") # 원본은 변경되지 않음

# 3. 불리언 인덱싱 (복사)
filtered_arr = arr[arr > 50]
print(f"\n필터링된 배열 (복사): {filtered_arr}")

# filtered_arr[0] = 1000 # 이 라인은 오류를 발생시킬 수 있습니다. (배열이 비어있을 경우)
# filtered_arr가 비어있지 않다면, 수정해도 원본 arr에는 영향을 주지 않습니다.

# 4. reshape (뷰)
reshaped_arr = arr.reshape(1, -1)
print(f"\nreshape (뷰): {reshaped_arr}")

reshaped_arr[0, 0] = 1000 # reshape된 뷰 수정
print(f"reshape 뷰 수정 후 원본 배열: {arr}") # 원본도 변경됨
```

### 8.6. 성능 최적화 팁

NumPy는 기본적으로 매우 효율적이지만, 대규모 데이터셋이나 복잡한 연산에서는 성능 최적화 기법을 적용하여 속도를 더욱 향상시킬 수 있습니다.

1.  **벡터화된 연산 활용**: 파이썬 `for` 루프 대신 NumPy의 내장 함수나 연산자를 사용하여 배열 전체에 대한 연산을 수행합니다. 이는 C로 구현된 내부 루틴을 활용하여 훨씬 빠릅니다.

    ```python
    import numpy as np
    import time

    size = 1000000
    a = np.random.rand(size)
    b = np.random.rand(size)

    # 파이썬 리스트와 루프
    start_time = time.time()
    c_list = [a[i] + b[i] for i in range(size)]
    end_time = time.time()
    print(f"파이썬 루프 시간: {end_time - start_time:.4f} 초")

    # NumPy 벡터화된 연산
    start_time = time.time()
    c_np = a + b
    end_time = time.time()
    print(f"NumPy 벡터화 시간: {end_time - start_time:.4f} 초")
    ```

2.  **메모리 연속성 (Contiguity)**: NumPy 배열은 메모리에 연속적으로 저장될 때 가장 효율적입니다. `C-order` (행 우선)와 `F-order` (열 우선)를 이해하고, 데이터 접근 패턴에 맞춰 배열을 생성하거나 재정렬하면 캐시 효율성을 높일 수 있습니다.

    ```python
    import numpy as np

    # C-order (기본값)
    arr_c = np.random.rand(1000, 1000)
    print(f"C-order 배열은 C-contiguous? {arr_c.flags['C_CONTIGUOUS']}")

    # F-order로 변환
    arr_f = np.asfortranarray(arr_c)
    print(f"F-order 배열은 F-contiguous? {arr_f.flags['F_CONTIGUOUS']}")
    ```

3.  **적절한 데이터 타입 (dtype) 사용**: 필요한 최소한의 데이터 타입을 사용하여 메모리 사용량을 줄이고 연산 속도를 높입니다. 예를 들어, 정수만 저장하는 배열에 `float64`를 사용할 필요는 없습니다.

    ```python
    import numpy as np

    arr_int = np.arange(1000, dtype=np.int8)
    arr_float = np.arange(1000, dtype=np.float64)

    print(f"int8 배열 메모리: {arr_int.nbytes} 바이트")
    print(f"float64 배열 메모리: {arr_float.nbytes} 바이트")
    ```

4.  **불필요한 복사 피하기**: `copy()` 메서드를 명시적으로 호출하거나, 뷰 대신 복사본을 반환하는 연산을 사용할 때 주의합니다. 대규모 배열에서 불필요한 복사는 메모리 사용량과 연산 시간을 크게 늘릴 수 있습니다.

5.  **`einsum` 활용**: 복잡한 텐서 연산(예: 다차원 배열의 곱셈, 합계)을 간결하고 효율적으로 표현할 수 있습니다. 특히 딥러닝에서 가중치 업데이트나 활성화 함수 계산 등에 유용합니다.

    ```python
    import numpy as np

    A = np.random.rand(3, 4)
    B = np.random.rand(4, 5)

    # 행렬 곱셈 (np.dot 또는 @)
    C_dot = A @ B

    # einsum을 이용한 행렬 곱셈
    C_einsum = np.einsum('ij,jk->ik', A, B)

    print(f"np.dot 결과와 einsum 결과가 동일한가? {np.allclose(C_dot, C_einsum)}")
    

### 8.7. NumPy와 다른 라이브러리 연동

NumPy는 파이썬의 과학 계산 생태계의 핵심이며, 다른 주요 라이브러리들과 긴밀하게 연동됩니다. 이는 데이터 분석, 머신러닝, 딥러닝 워크플로우를 구축하는 데 필수적입니다.

1.  **Pandas**: 데이터 분석 라이브러리인 Pandas의 `DataFrame`과 `Series` 객체는 내부적으로 NumPy 배열을 기반으로 합니다. 따라서 NumPy 배열과 Pandas 객체 간의 변환이 매우 효율적입니다.

    ```python
    import numpy as np
    import pandas as pd

    # NumPy 배열을 Pandas Series로 변환
    np_array = np.array([10, 20, 30, 40, 50])
    pd_series = pd.Series(np_array)
    print(f"NumPy 배열 -> Pandas Series:\n{pd_series}")

    # NumPy 배열을 Pandas DataFrame으로 변환
    np_2d_array = np.array([[1, 2, 3], [4, 5, 6]])
    pd_dataframe = pd.DataFrame(np_2d_array, columns=['col1', 'col2', 'col3'])
    print(f"\nNumPy 배열 -> Pandas DataFrame:\n{pd_dataframe}")

    # Pandas Series/DataFrame을 NumPy 배열로 변환
    back_to_np_array = pd_series.to_numpy()
    print(f"\nPandas Series -> NumPy 배열: {back_to_np_array}")
    ```

2.  **Matplotlib**: 파이썬의 대표적인 시각화 라이브러리인 Matplotlib은 NumPy 배열을 입력으로 받아 그래프를 그리는 데 최적화되어 있습니다. 대부분의 플로팅 함수는 NumPy 배열을 직접 처리할 수 있습니다.

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    # x, y 좌표를 NumPy 배열로 생성
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # NumPy 배열을 사용하여 그래프 그리기
    # plt.plot(x, y)
    # plt.title('Sine Wave')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.grid(True)
    # plt.show()
    ```

3.  **Scikit-learn**: 머신러닝 라이브러리인 Scikit-learn의 대부분의 알고리즘은 입력 데이터로 NumPy 배열을 기대합니다. 이는 데이터 전처리부터 모델 학습까지 일관된 데이터 형식을 유지할 수 있게 합니다.

    ```python
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # 특성(X)과 타겟(y) 데이터를 NumPy 배열로 준비
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 5, 4, 5])

    # 선형 회귀 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X, y)

    # 새로운 데이터 예측
    new_X = np.array([[6]])
    prediction = model.predict(new_X)
    print(f"새로운 데이터 {new_X.flatten()}에 대한 예측: {prediction.flatten()}")
    ```

4.  **TensorFlow/PyTorch (딥러닝 프레임워크)**: 현대 딥러닝 프레임워크인 TensorFlow와 PyTorch는 자체적인 텐서(Tensor) 객체를 가지고 있지만, 이들 텐서는 NumPy 배열과 상호 운용성이 매우 높습니다. NumPy 배열을 텐서로 쉽게 변환하거나 그 반대로 변환할 수 있어 데이터 로딩 및 전처리 단계에서 NumPy의 강점을 활용할 수 있습니다.

    ```python
    import numpy as np
    # import tensorflow as tf # TensorFlow가 설치되어 있어야 합니다.
    # import torch # PyTorch가 설치되어 있어야 합니다.

    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)

    # NumPy 배열 -> TensorFlow Tensor
    # tf_tensor = tf.convert_to_tensor(np_array)
    # print(f"\nNumPy 배열 -> TensorFlow Tensor:\n{tf_tensor}")

    # NumPy 배열 -> PyTorch Tensor
    # torch_tensor = torch.from_numpy(np_array)
    # print(f"\nNumPy 배열 -> PyTorch Tensor:\n{torch_tensor}")

    # TensorFlow Tensor -> NumPy 배열
    # back_to_np_from_tf = tf_tensor.numpy()
    # print(f"\nTensorFlow Tensor -> NumPy 배열:\n{back_to_np_from_tf}")

    # PyTorch Tensor -> NumPy 배열
    # back_to_np_from_torch = torch_tensor.numpy()
    # print(f"\nPyTorch Tensor -> NumPy 배열:\n{back_to_np_from_torch}")
    ```

### 8.8. 실제 ML/DL 적용 사례

NumPy는 머신러닝 및 딥러닝 알고리즘의 구현과 데이터 처리 과정에서 핵심적인 역할을 합니다. 다음은 몇 가지 대표적인 적용 사례입니다.

1.  **선형 회귀 (Linear Regression) 구현**: 가장 기본적인 머신러닝 알고리즘 중 하나인 선형 회귀는 NumPy를 사용하여 행렬 연산으로 효율적으로 구현할 수 있습니다.

    ```python
    import numpy as np

    # 데이터 생성 (특성 X, 타겟 y)
    X = 2 * np.random.rand(100, 1) # 100개의 샘플, 1개의 특성
    y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + 노이즈

    # X에 편향(bias) 항을 추가 (모든 값이 1인 열)
    X_b = np.c_[np.ones((100, 1)), X]

    # 정규 방정식(Normal Equation)을 사용한 선형 회귀 해법
    # theta_best = (X_b^T * X_b)^(-1) * X_b^T * y
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    print(f"최적의 파라미터 (theta_best):\n{theta_best}")
    # 결과는 [4.xxx, 3.xxx]와 유사하게 나와야 합니다.

    # 예측
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b @ theta_best
    print(f"\n새로운 X에 대한 예측:\n{y_predict}")
    ```

2.  **신경망의 순전파 (Forward Propagation) 구현**: 딥러닝의 기본 구성 요소인 신경망의 순전파 과정은 행렬 곱셈과 활성화 함수 적용으로 이루어지며, NumPy로 쉽게 구현할 수 있습니다.

    ```python
    import numpy as np

    # 입력 데이터 (배치 크기 1, 특성 3개)
    X = np.array([[1.0, 0.5, -1.0]])

    # 첫 번째 레이어의 가중치와 편향
    W1 = np.array([[0.1, 0.2],
                   [0.3, 0.4],
                   [0.5, 0.6]])
    b1 = np.array([[0.1, 0.2]])

    # 두 번째 레이어의 가중치와 편향
    W2 = np.array([[0.7],
                   [0.8]])
    b2 = np.array([[0.3]])

    # 활성화 함수 (ReLU)
    def relu(x):
        return np.maximum(0, x)

    # 순전파
    # 첫 번째 레이어
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    print(f"첫 번째 레이어 출력 (활성화 후):\n{A1}")

    # 두 번째 레이어 (출력 레이어)
    Z2 = A1 @ W2 + b2
    output = Z2
    print(f"\n최종 출력:\n{output}")
    ```

3.  **이미지 처리**: 이미지는 픽셀 값의 2D 또는 3D 배열로 표현될 수 있으며, NumPy는 이미지 데이터를 로드, 조작, 저장하는 데 사용됩니다. 예를 들어, 이미지의 밝기 조절, 크기 변경, 필터 적용 등에 활용됩니다.

    ```python
    import numpy as np
    # from PIL import Image # Pillow 라이브러리가 설치되어 있어야 합니다.

    # 예시: 10x10 흑백 이미지 (0-255)
    # image_data = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    # print(f"원본 이미지 데이터 (일부):\n{image_data[:3, :3]}")

    # 이미지 밝기 20 증가 (클리핑 적용)
    # bright_image = np.clip(image_data + 20, 0, 255)
    # print(f"\n밝기 조절된 이미지 데이터 (일부):\n{bright_image[:3, :3]}")

    # 이미지 저장 (Pillow 사용 예시)
    # img = Image.fromarray(image_data)
    # img.save('random_image.png')
    # print("\n'random_image.png' 저장됨")
    ```

4.  **데이터 정규화 (Normalization)**: 머신러닝 모델의 성능 향상을 위해 데이터를 정규화하는 것은 일반적인 전처리 단계입니다. NumPy를 사용하여 평균 0, 표준편차 1로 데이터를 스케일링할 수 있습니다.

    ```python
    import numpy as np

    data = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    print(f"원본 데이터: {data}")

    # 평균과 표준편차 계산
    mean = np.mean(data)
    std = np.std(data)

    # Z-score 정규화: (x - mean) / std
    normalized_data = (data - mean) / std

    print(f"\n정규화된 데이터: {normalized_data}")
    print(f"정규화된 데이터의 평균: {np.mean(normalized_data):.4f}")
    print(f"정규화된 데이터의 표준편차: {np.std(normalized_data):.4f}")
    ```

