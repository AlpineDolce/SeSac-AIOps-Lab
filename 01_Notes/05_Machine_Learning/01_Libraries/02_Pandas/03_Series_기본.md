<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Pandas의 1차원 데이터 구조인 `Series`의 개념, 특징, 그리고 다양한 생성 방법을 상세히 다룹니다. 또한, `Series` 데이터에 정수 인덱스 및 레이블 인덱스를 사용하여 접근하고 수정하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. Series (1차원 데이터)](#1-series-1차원-데이터)
  - [1.1. Series의 개념 및 특징](#11-series의-개념-및-특징)
  - [1.2. Series 생성 방법](#12-series-생성-방법)
  - [1.3. Series 데이터 접근 및 수정](#13-series-데이터-접근-및-수정)

---

## 1. Series (1차원 데이터)

### 1.1. Series의 개념 및 특징

1.  **정의**: Series는 Pandas의 1차원 데이터 구조로, **인덱스(index)와 값(value)의 쌍**으로 구성됩니다. 파이썬의 리스트나 NumPy의 1차원 배열과 유사하지만, 각 값에 접근할 수 있는 레이블(label) 또는 인덱스를 가질 수 있다는 점에서 차이가 있습니다.
2.  **구조**: 딕셔너리(dict) 타입과 유사하게 키-값 쌍의 형태로 데이터를 저장하고 관리합니다.
    *   **인덱스**: 각 데이터 항목을 고유하게 식별하는 레이블입니다. 명시적으로 지정하지 않으면 0부터 시작하는 정수 인덱스가 자동으로 부여됩니다.
    *   **값**: Series에 저장되는 실제 데이터입니다. 모든 값은 동일한 데이터 타입을 가질 수 있지만, 다른 데이터 타입이 혼합될 경우 Pandas가 자동으로 적절한 상위 데이터 타입(예: `object` 타입)으로 변환합니다.
3.  **활용**: 인덱스를 통해 데이터를 빠르고 효율적으로 검색, 정렬, 선택, 결합할 수 있어 데이터의 특정 부분을 빠르게 참조하거나 조작하는 데 용이합니다.

### 1.2. Series 생성 방법

다양한 파이썬 객체로부터 Series를 생성할 수 있습니다.

1.  **파이썬 리스트(list)로 Series 만들기**
    가장 기본적인 방법으로, 리스트의 요소들이 Series의 값이 되고, 0부터 시작하는 기본 정수 인덱스가 부여됩니다.

    ```python
    import pandas as pd

    # 파이썬 list를 사용하여 Series 생성
    data = [10, 20, 30, 40, 50]
    series = pd.Series(data)
    print("--- 리스트로 생성된 Series ---")
    print(type(series))
    print(series)
    # 출력:
    # --- 리스트로 생성된 Series ---
    # <class 'pandas.core.series.Series'>
    # 0    10
    # 1    20
    # 2    30
    # 3    40
    # 4    50
    # dtype: int64
    ```

2.  **파이썬 딕셔너리(dict)로 Series 만들기**
    딕셔너리의 키(key)가 Series의 인덱스가 되고, 값(value)이 Series의 값이 됩니다. 이를 통해 사용자 정의 레이블 인덱스를 쉽게 부여할 수 있습니다.

    ```python
    import pandas as pd

    # 딕셔너리를 사용하여 Series 생성 (레이블 인덱스 부여)
    data2 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    series2 = pd.Series(data2)
    print("\n--- 딕셔너리로 생성된 Series (레이블 인덱스) ---")
    print(series2)
    # 출력:
    # --- 딕셔너리로 생성된 Series (레이블 인덱스) ---
    # a    1
    # b    2
    # c    3
    # d    4
    # e    5
    # dtype: int64
    ```

3.  **NumPy 배열(ndarray)로 Series 만들기**
    NumPy 배열은 Pandas의 내부 연산에 효율적으로 사용되므로, NumPy 배열로부터 Series를 생성하는 것도 일반적입니다.

    ```python
    import numpy as np
    import pandas as pd

    # NumPy 배열로 Series 생성
    np_array = np.array([100, 200, 300])
    series_from_np = pd.Series(np_array)
    print("\n--- NumPy 배열로 생성된 Series ---")
    print(series_from_np)
    # 출력:
    # --- NumPy 배열로 생성된 Series ---
    # 0    100
    # 1    200
    # 2    300
    # dtype: int32 (또는 int64)
    ```

### 1.3. Series 데이터 접근 및 수정

Series의 데이터는 정수 인덱스(위치 기반) 또는 레이블 인덱스(이름 기반)를 사용하여 접근하고 수정할 수 있습니다.

1.  **기본 접근 (정수 인덱스)**
    리스트와 유사하게 대괄호 `[]` 안에 정수 인덱스를 사용하여 특정 위치의 데이터에 접근합니다.

    ```python
    import pandas as pd

    data = [10, 20, 30, 40, 50, 60]
s = pd.Series(data)

    print("\n--- Series 기본 접근 ---")
    print(f"0번째 데이터: {s[0]}") # 0번째 데이터 출력
    # 출력: 0번째 데이터: 10

    s[1] = 200 # 1번째 데이터 수정
    print(f"수정 후 1번째 데이터: {s[1]}")
    # 출력: 수정 후 1번째 데이터: 200
    ```

2.  **슬라이싱 (Slicing)**
    Series의 특정 범위의 데이터를 추출할 때 사용합니다. 정수 인덱스 또는 레이블 인덱스 모두에 적용 가능합니다. **주의**: 정수 인덱스 슬라이싱은 끝 인덱스를 포함하지 않지만, 레이블 인덱스 슬라이싱은 끝 인덱스를 포함합니다.

    ```python
    print("\n--- Series 슬라이싱 (정수 인덱스) ---")
    print("처음부터 4번째까지:", s[:5]) # 인덱스 0부터 4까지 (인덱스 5 미포함)
    print("2번째부터 4번째까지:", s[2:5]) # 인덱스 2부터 4까지 (인덱스 5 미포함)
    print("3번째부터 끝까지:", s[3:]) # 인덱스 3부터 끝까지
    # 출력 예시:
    # 처음부터 4번째까지:
    # 0     10
    # 1    200
    # 2     30
    # 3     40
    # 4     50
    # dtype: int64
    ```

3.  **레이블 인덱스 접근**
    딕셔너리처럼 레이블 인덱스를 사용하여 데이터에 접근합니다. 슬라이싱 시 레이블 인덱스는 끝 인덱스를 포함합니다.

    ```python
    # 레이블 인덱스 사용
    data_labeled = {'one': '일', 'two': '이', 'three': '삼', 'four': '사', 'five': '오'}
    series_labeled = pd.Series(data_labeled)

    print("\n--- Series 레이블 인덱스 접근 ---")
    print(f"레이블 'one' 데이터: {series_labeled['one']}")
    # 출력: 레이블 'one' 데이터: 일

    print("\n--- Series 레이블 인덱스 슬라이싱 ---")
    print("레이블 'one'부터 'three'까지:\n", series_labeled['one']:'three'])
    # 출력:
    # 레이블 'one'부터 'three'까지:
    # one      일
    # two      이
    # three    삼
    # dtype: object
    ```

4.  **조건식을 이용한 필터링 (Boolean Indexing)**
    특정 조건을 만족하는 데이터만 선택할 때 유용합니다. 조건식의 결과로 True/False Series가 생성되며, 이를 사용하여 원본 Series에서 True에 해당하는 값만 추출합니다.

    ```python
    import pandas as pd

    s = pd.Series([10, 20, 30, 40, 50, 60])

    print("\n--- Series 조건식 필터링 ---")
    print("값이 30보다 큰 데이터:\n", s[s > 30])
    # 출력:
    # 값이 30보다 큰 데이터:
    # 3    40
    # 4    50
    # 5    60
    # dtype: int64

    print("값이 20 이상 50 이하인 데이터:\n", s[(s >= 20) & (s <= 50)])
    # 출력:
    # 값이 20 이상 50 이하인 데이터:
    # 1    20
    # 2    30
    # 3    40
    # 4    50
    # dtype: int64
    ```
