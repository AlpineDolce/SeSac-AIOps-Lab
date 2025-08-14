<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 Pandas의 1차원 데이터 구조인 `Series`의 개념, 특징, 다양한 생성 방법, 그리고 데이터 접근 및 수정 방법을 상세히 다룹니다. 특히 `loc`, `iloc`와 같은 명시적인 인덱싱 방법과 실무에서 자주 사용되는 데이터 처리 기법을 포함하여 `Series`를 효율적으로 다루는 역량을 강화하는 데 중점을 둡니다.

<h2>목차</h2>

- [1. Series (1차원 데이터)](#1-series-1차원-데이터)
  - [1.1. Series의 개념 및 특징](#11-series의-개념-및-특징)
  - [1.2. Series 생성 방법](#12-series-생성-방법)
    - [1.2.1. 파이썬 리스트(list)로 Series 만들기](#121-파이썬-리스트list로-series-만들기)
    - [1.2.2. 파이썬 딕셔너리(dict)로 Series 만들기](#122-파이썬-딕셔너리dict로-series-만들기)
    - [1.2.3. NumPy 배열(ndarray)로 Series 만들기](#123-numpy-배열ndarray로-series-만들기)
    - [1.2.4. 스칼라 값으로 Series 만들기](#124-스칼라-값으로-series-만들기)
    - [1.2.5. 인덱스를 명시하여 Series 만들기](#125-인덱스를-명시하여-series-만들기)
  - [1.3. Series 데이터 접근 및 수정](#13-series-데이터-접근-및-수정)
    - [1.3.1. 기본 접근 `[]`의 모호성 및 권장사항](#131-기본-접근-의-모호성-및-권장사항)
    - [1.3.2. `loc`를 이용한 레이블 기반 접근](#132-loc를-이용한-레이블-기반-접근)
    - [1.3.3. `iloc`를 이용한 정수 위치 기반 접근](#133-iloc를-이용한-정수-위치-기반-접근)
    - [1.3.4. `get()` 메서드를 이용한 안전한 접근](#134-get-메서드를-이용한-안전한-접근)
    - [1.3.5. Series의 벡터화 연산](#135-series의-벡터화-연산)
    - [1.3.6. 데이터 타입 변경 (astype)](#136-데이터-타입-변경-astype)

--- 

## 1. Series (1차원 데이터)

### 1.1. Series의 개념 및 특징

`Series`는 Pandas의 1차원 데이터 구조로, **인덱스(index)와 값(value)의 쌍**으로 구성됩니다. 파이썬의 리스트나 NumPy의 1차원 배열과 유사하지만, 각 값에 접근할 수 있는 **레이블(label) 또는 인덱스**를 가질 수 있다는 점에서 강력한 기능을 제공합니다.

**주요 구성 요소:**
*   **값 (Values)**: Series에 저장되는 실제 데이터입니다. 모든 값은 동일한 데이터 타입(`dtype`)을 가집니다. 만약 다른 데이터 타입이 혼합될 경우, Pandas는 자동으로 모든 값을 포괄할 수 있는 상위 데이터 타입(예: `int`와 `float` -> `float`, `int`와 `str` -> `object`)으로 **업캐스팅(upcasting)**합니다.
*   **인덱스 (Index)**: 각 데이터 항목을 고유하게 식별하는 레이블입니다. 명시적으로 지정하지 않으면 0부터 시작하는 정수 인덱스가 자동으로 부여됩니다.
*   **이름 (Name)**: `Series` 객체 자체와 `Index` 객체는 `name` 속성을 가질 수 있습니다. 이 속성은 `DataFrame`에 `Series`를 결합할 때 컬럼명이나 인덱스명으로 사용되어 매우 유용합니다.

**Series의 주요 속성:**
`Series` 객체는 `values`, `index`, `dtype`, `name` 등 다양한 속성을 통해 내부 데이터와 메타데이터에 접근할 수 있습니다.

- **Series 생성 예시**
    ```python
    import pandas as pd
    import numpy as np

    # Series 생성 예시
    s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'], name='My_Series')
    s.index.name = 'My_Index'

    print(f"Series 객체:\n{s}")
    ```

- **주요 속성 확인**
    ```python
    # Series의 값 (values) - NumPy 배열 형태로 반환
    print(f"\n.values: {s.values} (타입: {type(s.values)})")

    # Series의 인덱스 (index)
    print(f".index: {s.index} (타입: {type(s.index)})")

    # Series의 데이터 타입 (dtype)
    print(f".dtype: {s.dtype}")

    # Series의 이름 (name)
    print(f".name: {s.name}")

    # Series 인덱스의 이름 (index.name)
    print(f".index.name: {s.index.name}")

    # Series의 차원 (ndim) - 항상 1
    print(f".ndim: {s.ndim}")

    # Series의 모양 (shape) - (행,) 형태의 튜플
    print(f".shape: {s.shape}")

    # Series의 요소 개수 (size)
    print(f".size: {s.size}")

    # Series 내 결측치(NaN) 존재 여부 (hasnans)
    print(f".hasnans: {s.hasnans}")
    ```

### 1.2. Series 생성 방법

다양한 파이썬 객체로부터 `Series`를 생성할 수 있습니다.

#### 1.2.1. 파이썬 리스트(list)로 Series 만들기

가장 기본적인 방법으로, 리스트의 요소들이 `Series`의 값이 되고, 0부터 시작하는 기본 정수 인덱스가 부여됩니다. `dtype` 파라미터를 통해 데이터 타입을 명시적으로 지정할 수 있습니다.

```python
import pandas as pd

# 파이썬 list를 사용하여 Series 생성
data = [10, 20, 30, 40, 50]
s_from_list = pd.Series(data, dtype='float32') # dtype 명시적 지정
print("--- 리스트로 생성된 Series ---")
print(s_from_list)
```

#### 1.2.2. 파이썬 딕셔너리(dict)로 Series 만들기

딕셔너리의 키(key)가 `Series`의 인덱스가 되고, 값(value)이 `Series`의 값이 됩니다. 만약 `index`를 별도로 지정하면, 해당 인덱스에 맞게 데이터가 재정렬되며, 일치하는 키가 없는 인덱스에는 `NaN` (Not a Number)이 할당됩니다.

```python
import pandas as pd

# 딕셔너리를 사용하여 Series 생성
data_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
s_from_dict = pd.Series(data_dict)
print("\n--- 딕셔너리로 생성된 Series ---")
print(s_from_dict)

# 딕셔너리와 함께 인덱스를 지정하는 경우 (데이터 재정렬 및 NaN 발생)
# 'a', 'c', 'b'는 딕셔너리에서 값을 가져오고, 'e'는 없으므로 NaN이 됩니다.
custom_index = ['a', 'c', 'e', 'b']
s_dict_custom_index = pd.Series(data_dict, index=custom_index)
print("\n--- 딕셔너리에 사용자 정의 인덱스 적용 ---")
print(s_dict_custom_index)
```

#### 1.2.3. NumPy 배열(ndarray)로 Series 만들기

NumPy 배열은 Pandas의 내부 연산에 효율적으로 사용되므로, NumPy 배열로부터 `Series`를 생성하는 것은 대규모 수치 데이터를 다룰 때 일반적입니다.

```python
import numpy as np
import pandas as pd

# NumPy 배열로 Series 생성
np_array = np.array([100, 200, 300])
s_from_np = pd.Series(np_array)
print("\n--- NumPy 배열로 생성된 Series ---")
print(s_from_np)
```

#### 1.2.4. 스칼라 값으로 Series 만들기

단일 스칼라 값과 인덱스를 제공하여 `Series`를 생성할 수 있습니다. 이 경우, 스칼라 값이 모든 인덱스에 대해 반복됩니다.

```python
import pandas as pd

# 스칼라 값으로 Series 생성
s_from_scalar = pd.Series(5, index=['x', 'y', 'z'])
print("\n--- 스칼라 값으로 생성된 Series ---")
print(s_from_scalar)
```

#### 1.2.5. 인덱스를 명시하여 Series 만들기

리스트나 NumPy 배열로부터 `Series`를 생성할 때, `index` 매개변수를 사용하여 사용자 정의 인덱스를 명시적으로 지정할 수 있습니다. 인덱스의 길이는 데이터의 길이와 일치해야 합니다.

```python
import pandas as pd

data = [10, 20, 30]
custom_index = ['first', 'second', 'third']
s_custom_index = pd.Series(data, index=custom_index)
print("\n--- 사용자 정의 인덱스로 생성된 Series ---")
print(s_custom_index)
```

### 1.3. Series 데이터 접근 및 수정

`Series`의 데이터는 정수 인덱스(위치 기반) 또는 레이블 인덱스(이름 기반)를 사용하여 접근하고 수정할 수 있습니다. Pandas는 명시적인 접근을 위해 `loc`와 `iloc` 접근자를 제공하며, 이는 코드의 가독성을 높이고 잠재적인 오류를 방지하는 데 매우 중요합니다.

#### 1.3.1. 기본 접근 `[]`의 모호성 및 권장사항

대괄호 `[]`를 사용한 기본 인덱싱은 편리하지만, 인덱스의 종류에 따라 혼란을 유발할 수 있습니다. 특히, **정수형 레이블 인덱스**를 사용할 경우 위치 기반인지 레이블 기반인지 모호해집니다.

- **⚠️ 주의:** 코드의 명확성과 재현성을 위해 데이터 접근 시에는 `[]` 대신 `loc`와 `iloc`를 사용하는 것을 강력히 권장합니다.

```python
import pandas as pd

# 정수 레이블 인덱스를 가진 Series
s_int_label = pd.Series([10, 20, 30], index=[2, 4, 6])
print("--- 정수 레이블 Series ---")
print(s_int_label)

# s_int_label[2]는 위치(iloc[2])일까, 레이블(loc[2])일까?
# Pandas는 레이블을 우선하므로 loc[2]로 해석된다.
print(f"\ns_int_label[2] -> 레이블 '2'의 값: {s_int_label[2]}")

# 위치 기반으로 접근하려면 iloc를 명시적으로 사용해야 한다.
print(f"s_int_label.iloc[2] -> 2번 위치의 값: {s_int_label.iloc[2]}")
```

#### 1.3.2. `loc`를 이용한 레이블 기반 접근

`loc` 접근자는 **레이블(이름) 기반**으로 데이터에 접근하거나 수정할 때 사용합니다. 슬라이싱 시 끝 인덱스 레이블을 **포함**합니다.

- **기본 문법**: `s.loc[레이블]`
    ```python
    import pandas as pd
    s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
    print("\n--- Series loc 접근 (레이블 기반) ---")
    print("원본 Series:\n", s)
    ```

- **1. 단일 레이블 선택**
    ```python
    print(f"1. 레이블 'c' 데이터: {s.loc['c']}")
    ```

- **2. 여러 레이블 선택 (팬시 인덱싱)**
    ```python
    print(f"2. 레이블 'a', 'c', 'e' 데이터:\n{s.loc[['a', 'c', 'e']]}")
    ```

- **3. 레이블 슬라이싱 (끝 레이블 포함)**
    ```python
    print(f"3. 레이블 'b'부터 'd'까지:\n{s.loc['b':'d']}")
    ```

- **4. 조건식을 이용한 선택**
    ```python
    print(f"4. 값이 30 이상인 데이터:\n{s.loc[s >= 30]}")
    ```

- **5. 데이터 수정**
    ```python
    s_copy = s.copy()
    s_copy.loc['c'] = 300
    print(f"5. 수정 후 Series:\n{s_copy}")
    ```

#### 1.3.3. `iloc`를 이용한 정수 위치 기반 접근

`iloc` 접근자는 **정수 위치(position) 기반**으로 데이터에 접근하거나 수정할 때 사용합니다. 파이썬 리스트의 인덱싱과 유사하며, 슬라이싱 시 끝 인덱스를 **포함하지 않습니다**.

- **기본 문법**: `s.iloc[위치]`
    ```python
    import pandas as pd
    s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
    print("\n--- Series iloc 접근 (정수 위치 기반) ---")
    print("원본 Series:\n", s)
    ```

- **1. 단일 위치 선택**
    ```python
    print(f"1. 0번째 위치 데이터: {s.iloc[0]}")
    ```

- **2. 여러 위치 선택 (팬시 인덱싱)**
    ```python
    print(f"2. 0, 2, 4번째 위치 데이터:\n{s.iloc[[0, 2, 4]]}")
    ```

- **3. 위치 슬라이싱 (끝 위치 미포함)**
    ```python
    print(f"3. 1번째부터 3번째 위치까지 (1, 2):\n{s.iloc[1:3]}")
    ```

- **4. 데이터 수정**
    ```python
    s_copy = s.copy()
    s_copy.iloc[2] = 300
    print(f"4. 수정 후 Series:\n{s_copy}")
    ```

#### 1.3.4. `get()` 메서드를 이용한 안전한 접근
`get()` 메서드는 딕셔너리처럼 특정 레이블의 값을 안전하게 가져올 때 사용합니다. 해당 레이블이 존재하지 않으면 `KeyError`를 발생시키는 대신 `None`이나 지정된 기본값을 반환합니다.

```python
import pandas as pd
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# 존재하는 키 접근
print(f"\ns.get('a'): {s.get('a')}")

# 존재하지 않는 키 접근 (기본값: None)
print(f"s.get('x'): {s.get('x')}")

# 존재하지 않는 키 접근 (기본값 지정)
print(f"s.get('x', default=-1): {s.get('x', default=-1)}")
```

#### 1.3.5. Series의 벡터화 연산

`Series` 객체는 NumPy 배열과 유사하게 벡터화된(element-wise) 연산을 지원합니다. `+`, `-`, `*`, `/`, `**`, `//`, `%` 등 대부분의 표준 산술 연산자가 지원됩니다. `Series` 간의 연산은 동일한 인덱스를 기준으로 수행되며, 인덱스가 한쪽에만 존재할 경우 결과는 `NaN`이 됩니다.

- **1. `Series` 간의 연산**
    ```python
    import pandas as pd
    s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([10, 20, 30, 40], index=['a', 'c', 'd', 'e'])
    print("\n--- Series 연산 ---")
    print(f"s1:\n{s1}\n\ns2:\n{s2}")

    # 덧셈 연산 (인덱스 정렬 후 연산, 불일치 인덱스는 NaN)
    s_sum = s1 + s2
    print(f"\n1. s1 + s2:\n{s_sum}")
    ```

- **2. 연산 메서드와 `fill_value` 사용**
    ```python
    # 결측치를 0으로 채우고 연산
    s_sum_filled = s1.add(s2, fill_value=0)
    print(f"\n2. s1.add(s2, fill_value=0):\n{s_sum_filled}")
    ```

- **3. 스칼라 연산**
    ```python
    # 모든 요소에 동일 연산 적용
    s_scalar_mul = s1 * 10
    print(f"\n3. s1 * 10:\n{s_scalar_mul}")
    ```

#### 1.3.6. 데이터 타입 변경 (astype)

`astype()` 메서드는 `Series`의 데이터 타입을 변경하는 가장 일반적이고 효율적인 방법입니다. 메모리 사용량을 줄이거나, 연산을 위해 데이터 타입을 통일할 때 필수적입니다.

```python
import pandas as pd
s_types = pd.Series([1, 2, 3.0, 4.5])
print("\n--- astype을 이용한 타입 변경 ---")
print("원본 Series의 dtype:", s_types.dtype)

# float64 -> int64로 변경 (소수점 이하 버림)
s_int = s_types.astype(int)
print("\nint로 변경 후:\n", s_int)
print("변경된 dtype:", s_int.dtype)

# object(string) 타입으로 변경
s_str = s_types.astype(str)
print("\nstr로 변경 후:\n", s_str)
print("변경된 dtype:", s_str.dtype)
```