<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
이 문서는 Pandas의 핵심 2차원 데이터 구조인 `DataFrame`의 개념, 특징, 다양한 생성 방법, 그리고 데이터 접근 및 조작 방법을 상세히 다룹니다. `DataFrame`을 효율적으로 다루는 실무 역량을 강화하는 데 중점을 둡니다.

<h2>목차</h2>

- [1. DataFrame (2차원 데이터)](#1-dataframe-2차원-데이터)
  - [1.1. DataFrame의 개념 및 특징](#11-dataframe의-개념-및-특징)
  - [1.2. DataFrame 생성 방법](#12-dataframe-생성-방법)
    - [1.2.1. 딕셔너리(dict)로 DataFrame 만들기](#121-딕셔너리dict로-dataframe-만들기)
    - [1.2.2. 리스트(list)의 리스트로 DataFrame 만들기](#122-리스트list의-리스트로-dataframe-만들기)
    - [1.2.3. NumPy 배열(ndarray)로 DataFrame 만들기](#123-numpy-배열ndarray로-dataframe-만들기)
    - [1.2.4. 리스트의 딕셔너리로 DataFrame 만들기](#124-리스트의-딕셔너리로-dataframe-만들기)
    - [1.2.5. Series 딕셔너리로 DataFrame 만들기](#125-series-딕셔너리로-dataframe-만들기)
    - [1.2.6. 빈 DataFrame 만들기](#126-빈-dataframe-만들기)
  - [1.3. DataFrame 데이터 접근 및 조작](#13-dataframe-데이터-접근-및-조작)
    - [1.3.1. 컬럼(열) 선택](#131-컬럼열-선택)
    - [1.3.2. `head()`, `tail()`, `info()`, `describe()` 등 기본 정보 확인](#132-head-tail-info-describe-등-기본-정보-확인)
    - [1.3.3. `loc`를 이용한 레이블 기반 접근](#133-loc를-이용한-레이블-기반-접근)
    - [1.3.4. `iloc`를 이용한 정수 위치 기반 접근](#134-iloc를-이용한-정수-위치-기반-접근)
    - [1.3.5. 조건식을 이용한 필터링 (Boolean Indexing)](#135-조건식을-이용한-필터링-boolean-indexing)
    - [1.3.6. 컬럼 추가 및 수정](#136-컬럼-추가-및-수정)
    - [1.3.7. 컬럼 삭제](#137-컬럼-삭제)
    - [1.3.8. 행 추가 및 삭제](#138-행-추가-및-삭제)
    - [1.3.9. 결측치 처리 (간략)](#139-결측치-처리-간략)

---

## 1. DataFrame (2차원 데이터)

### 1.1. DataFrame의 개념 및 특징

`DataFrame`은 Pandas의 핵심 2차원 데이터 구조로, **행(row)과 열(column)으로 이루어진 테이블 형태**를 가집니다. 관계형 데이터베이스의 테이블, Excel 스프레드시트, 또는 CSV 파일과 매우 유사합니다. 각 열은 `Series` 객체로 볼 수 있으며, 서로 다른 데이터 타입을 가질 수 있습니다. 이는 각 컬럼이 독립적인 데이터 유형을 가질 수 있음을 의미합니다 (예: 이름은 문자열, 나이는 정수, 도시는 문자열).

**DataFrame의 주요 구성 요소:**
*   **값 (Values)**: DataFrame에 저장되는 실제 데이터입니다. 각 컬럼은 동일한 `dtype`을 가지지만, 컬럼 간에는 다른 `dtype`을 가질 수 있습니다.
*   **인덱스 (Index)**: 행을 고유하게 식별하는 레이블입니다. `Series`의 인덱스와 동일합니다.
*   **컬럼 (Columns)**: 열을 고유하게 식별하는 레이블입니다.

**DataFrame의 주요 속성:**
`DataFrame` 객체는 `shape`, `dtypes`, `index`, `columns`, `values` 등 다양한 속성을 통해 내부 데이터와 메타데이터에 접근할 수 있습니다.

- **DataFrame 생성 예시**
    ```python
    import pandas as pd
    import numpy as np

    # DataFrame 생성 예시
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'London', 'Paris', 'Tokyo'],
        'Score': [85.5, 90.0, 78.2, 92.1]
    }
    df = pd.DataFrame(data)

    print(f"DataFrame 객체:\n{df}")
    ```

- **주요 속성 확인**
    ```python
    # DataFrame의 형태 (shape): (행 수, 열 수)
    print(f"\nDataFrame의 형태 (shape): {df.shape}")

    # DataFrame의 데이터 타입 (dtypes): 각 컬럼의 데이터 타입
    print(f"\nDataFrame의 데이터 타입 (dtypes):\n{df.dtypes}")

    # DataFrame의 행 인덱스 (index)
    print(f"\nDataFrame의 행 인덱스 (index): {df.index}")

    # DataFrame의 컬럼 이름 (columns)
    print(f"\nDataFrame의 컬럼 이름 (columns): {df.columns}")

    # DataFrame의 값 (values): NumPy 배열로 반환
    print(f"\nDataFrame의 값 (values):\n{df.values}")
    ```

### 1.2. DataFrame 생성 방법

`DataFrame`은 다양한 파이썬 객체로부터 생성할 수 있습니다. 데이터의 초기 형태에 따라 적절한 방법을 선택하는 것이 중요합니다.

#### 1.2.1. 딕셔너리(dict)로 DataFrame 만들기

가장 일반적인 방법으로, 딕셔너리의 키(key)가 컬럼 이름이 되고, 값(value)은 리스트나 Series 형태의 데이터가 됩니다. 각 리스트의 길이는 동일해야 합니다.

```python
import pandas as pd

data_dict = {
    'name': ['홍길동', '임꺽정', '장길산'],
    'kor': [90, 80, 70],
    'eng': [99, 98, 97],
    'mat': [90, 70, 70],
}
df_from_dict = pd.DataFrame(data_dict)
print("--- 딕셔너리로 생성된 DataFrame ---")
print(df_from_dict)
```

#### 1.2.2. 리스트(list)의 리스트로 DataFrame 만들기

각 내부 리스트가 한 행을 나타내고, `columns` 인자를 사용하여 컬럼 이름을 지정할 수 있습니다. `columns`를 지정하지 않으면 0부터 시작하는 정수 컬럼 인덱스가 부여됩니다.

```python
import pandas as pd

data_list_of_lists = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'London'],
    ['Charlie', 35, 'Paris']
]
df_from_list_of_lists = pd.DataFrame(data_list_of_lists, columns=['Name', 'Age', 'City'])
print("\n--- 리스트의 리스트로 생성된 DataFrame ---")
print(df_from_list_of_lists)
```

#### 1.2.3. NumPy 배열(ndarray)로 DataFrame 만들기

NumPy 2차원 배열을 사용하여 `DataFrame`을 생성할 수 있습니다. 이는 대규모 수치 데이터를 `DataFrame`으로 변환할 때 효율적이며, `columns` 인자를 통해 컬럼 이름을 지정하는 것이 일반적입니다.

```python
import numpy as np
import pandas as pd

np_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df_from_np = pd.DataFrame(np_data, columns=['Col1', 'Col2', 'Col3'])
print("\n--- NumPy 배열로 생성된 DataFrame ---")
print(df_from_np)
```

#### 1.2.4. 리스트의 딕셔너리로 DataFrame 만들기

각 딕셔너리가 한 행을 나타내며, 딕셔너리의 키가 컬럼 이름이 됩니다. 이 방법은 JSON 데이터와 같이 구조화된 데이터를 `DataFrame`으로 변환할 때 유용합니다.

```python
import pandas as pd

data_list_of_dicts = [
    {'Name': 'Alice', 'Age': 25},
    {'Name': 'Bob', 'Age': 30, 'City': 'London'}, # City 컬럼이 없는 행도 가능
    {'Name': 'Charlie', 'Age': 35}
]
df_from_list_of_dicts = pd.DataFrame(data_list_of_dicts)
print("\n--- 리스트의 딕셔너리로 생성된 DataFrame ---")
print(df_from_list_of_dicts)
```

#### 1.2.5. Series 딕셔너리로 DataFrame 만들기

각 `Series`가 `DataFrame`의 한 컬럼이 됩니다. `Series`의 인덱스가 `DataFrame`의 행 인덱스가 됩니다.

```python
import pandas as pd

s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
s3 = pd.Series([100, 200, 300], index=['b', 'c', 'd']) # 인덱스 불일치 예시

df_from_series_dict = pd.DataFrame({'Col1': s1, 'Col2': s2, 'Col3': s3})
print("\n--- Series 딕셔너리로 생성된 DataFrame ---")
print(df_from_series_dict) # 인덱스 불일치 시 NaN 발생
```

#### 1.2.6. 빈 DataFrame 만들기

컬럼 이름만 지정하여 빈 `DataFrame`을 만들 수 있습니다. 나중에 데이터를 추가할 때 유용합니다.

```python
import pandas as pd

empty_df = pd.DataFrame(columns=['Name', 'Age', 'City'])
print("\n--- 빈 DataFrame ---")
print(empty_df)
print(f"빈 DataFrame의 형태: {empty_df.shape}")
```

### 1.3. DataFrame 데이터 접근 및 조작

`DataFrame`의 데이터는 다양한 방법으로 접근하고 조작할 수 있습니다. `Series`와 마찬가지로 `loc`와 `iloc` 접근자를 사용하여 명시적인 접근을 하는 것이 권장됩니다.

#### 1.3.1. 컬럼(열) 선택

단일 컬럼은 `Series` 형태로, 여러 컬럼은 `DataFrame` 형태로 반환됩니다. 컬럼 이름은 대소문자를 구분합니다.

```python
import pandas as pd

data = {
    'name': ['홍길동', '임꺽정', '장길산', '홍경래'],
    'kor': [90, 80, 70, 70],
    'eng': [99, 98, 97, 46],
    'mat': [90, 70, 70, 60],
}
df = pd.DataFrame(data)

print("--- DataFrame 컬럼 선택 ---")
print("원본 DataFrame:\n", df)
```

- **1. 단일 컬럼 선택 (대괄호 `[]` 사용)**

    컬럼 이름을 문자열로 전달하면 해당 컬럼이 `Series` 형태로 반환됩니다.

    ```python
    # 'name' 컬럼 선택 (Series 형태로 반환)
    name_series = df['name']
    print("\n1. 특정 열 ('name')만 출력 (Series 반환):\n", name_series)
    print(f"   반환된 객체의 타입: {type(name_series)}")
    ```

- **2. 여러 컬럼 선택 (대괄호 `[[]]` 사용)**

    컬럼 이름의 리스트를 전달하면 해당 컬럼들이 새로운 `DataFrame` 형태로 반환됩니다.

    ```python
    # 'kor', 'eng' 컬럼 선택 (DataFrame 형태로 반환)
    subset_df = df[['kor', 'eng']]
    print("\n2. 여러 열 ('kor', 'eng') 출력 (DataFrame 반환):\n", subset_df)
    print(f"   반환된 객체의 타입: {type(subset_df)}")
    ```

- **3. 점(`.`) 표기법으로 컬럼 접근**

    컬럼 이름이 파이썬 변수명 규칙에 맞고(공백이나 특수문자 없음), 기존 DataFrame의 메서드 이름과 겹치지 않을 때 사용할 수 있는 간편한 방법입니다.

    ```python
    # 점 표기법으로 'name' 컬럼 접근
    name_series_dot = df.name
    print("\n3. 점 표기법으로 'name' 컬럼 접근:\n", name_series_dot)
    ```

- **4. 모든 컬럼 이름 확인**

    `df.columns` 속성은 모든 컬럼 이름을 담고 있는 Index 객체를 반환합니다.

    ```python
    print("\n4. 모든 컬럼 이름 출력:", df.columns)
    ```

#### 1.3.2. `head()`, `tail()`, `info()`, `describe()` 등 기본 정보 확인

`DataFrame`의 크기가 클 때 전체를 출력하는 대신, 데이터의 구조와 통계적 요약을 빠르게 파악하는 데 유용합니다. 이 메서드들은 탐색적 데이터 분석(EDA)의 첫 단계에서 데이터셋의 전반적인 특성을 이해하는 데 필수적입니다.

-   **`df.head(n=5)`**:
    -   **목적**: DataFrame의 **상위 `n`개 행**을 반환하여 데이터의 첫 부분을 빠르게 확인합니다. 기본값은 5입니다.
    -   **활용**: 데이터가 올바르게 로드되었는지, 컬럼 이름이 예상대로인지, 데이터의 첫 몇 행이 어떤 패턴을 보이는지 등을 파악할 때 유용합니다.
    ```python
    import pandas as pd

    data = {
        'name': ['홍길동', '임꺽정', '장길산', '홍경래', '이상민', '김수경', '박영희', '최철수', '이민지', '정우성'],
        'kor': [90, 80, 70, 70, 60, 70, 85, 92, 78, 65],
        'eng': [99, 98, 97, 46, 77, 56, 88, 91, 75, 68],
        'mat': [90, 70, 70, 60, 88, 99, 80, 95, 72, 63],
    }
    df = pd.DataFrame(data)

    print("\n--- DataFrame 기본 정보 확인 ---")
    print("1. `df.head()`: 앞의 5행만 출력 (기본값)\n", df.head())
    print("\n2. `df.head(3)`: 앞의 3행만 출력\n", df.head(3))
    ```

-   **`df.tail(n=5)`**:
    -   **목적**: DataFrame의 **하위 `n`개 행**을 반환하여 데이터의 끝 부분을 빠르게 확인합니다. 기본값은 5입니다.
    -   **활용**: 데이터가 끝까지 올바르게 로드되었는지, 마지막 데이터 포인트에 특이한 점은 없는지 등을 확인할 때 유용합니다. 특히 로그 데이터나 시간 순서 데이터에서 최신 데이터를 확인할 때 유용합니다.
        ```python
        print("\n3. `df.tail()`: 뒤의 5행만 출력 (기본값)\n", df.tail())
        print("\n4. `df.tail(2)`: 뒤의 2행만 출력\n", df.tail(2))
        ```

-   **`df.info(verbose=True, memory_usage=True)`**:
    -   **목적**: DataFrame의 **간결한 요약 정보**를 출력합니다. 각 컬럼의 데이터 타입, Non-null 값의 개수, 메모리 사용량 등을 포함합니다.
    -   **활용**:
        -   **결측치 확인**: `Non-null` 값의 개수를 통해 각 컬럼의 결측치(Missing Value) 여부를 빠르게 파악할 수 있습니다.
        -   **데이터 타입 확인**: 각 컬럼의 데이터 타입(`dtype`)이 예상과 일치하는지 확인하여 데이터 전처리 방향을 설정합니다.
        -   **메모리 사용량**: DataFrame이 차지하는 메모리 양을 확인하여 대규모 데이터셋 처리 시 메모리 최적화 필요성을 판단합니다.
    -   **주요 파라미터**:
        -   `verbose`: `True` (기본값)이면 모든 컬럼 정보를 표시하고, `False`이면 요약된 정보만 표시합니다.
        -   `memory_usage`: `True` (기본값)이면 DataFrame의 총 메모리 사용량을 계산하여 표시합니다. `'deep'`으로 설정하면 객체(object) 타입 컬럼의 실제 메모리 사용량까지 정확히 계산합니다.
        ```python
        print("\n5. `df.info()`: DataFrame 정보 요약")
        df.info()
        print("\n6. `df.info(verbose=False)`: 간결한 정보 요약")
        df.info(verbose=False)
        print("\n7. `df.info(memory_usage='deep')`: 상세 메모리 사용량 포함")
        df.info(memory_usage='deep')
        ```

-   **`df.describe(include=None, exclude=None)`**:
    -   **목적**: DataFrame의 **수치형 컬럼에 대한 기술 통계(Descriptive Statistics) 요약**을 제공합니다. 개수(count), 평균(mean), 표준편차(std), 최소값(min), 25/50/75 백분위수(25%/50%/75%), 최대값(max) 등을 포함합니다.
    -   **활용**:
        -   **데이터 분포 파악**: 각 수치형 컬럼의 중심 경향(평균, 중앙값), 퍼짐 정도(표준편차), 범위(최소/최대값) 등을 빠르게 파악합니다.
        -   **이상치(Outlier) 탐지**: 최소값이나 최대값이 다른 값들과 현저히 차이 나는 경우 이상치 존재 가능성을 시사합니다.
        -   **데이터 스케일 확인**: 컬럼 간 값의 범위 차이를 확인하여 스케일링 필요성을 판단합니다.
    -   **주요 파라미터**:
        -   `include`: 특정 데이터 타입(예: `'object'`, `'number'`, `'all'`)의 컬럼만 포함하여 통계를 계산합니다.
        -   `exclude`: 특정 데이터 타입의 컬럼을 제외하고 통계를 계산합니다.
    ```python
    print("\n8. `df.describe()`: DataFrame 통계 요약 (수치형 컬럼만)")
    df.describe()

    print("\n9. `df.describe(include='object')`: 문자열(object) 컬럼 통계 요약")
    df.describe(include='object') # 문자열 컬럼에 대한 통계 (unique, top, freq)

    print("\n10. `df.describe(include='all')`: 모든 컬럼 통계 요약")
    df.describe(include='all') # 모든 컬럼에 대한 통계 (수치형+문자열)
    ```

-   **`df.value_counts()` (Series 메서드)**:
    -   **목적**: Series 내의 **고유한 값들의 빈도수**를 계산하여 반환합니다.
    -   **활용**: 범주형(Categorical) 데이터의 분포를 파악하거나, 특정 값의 출현 빈도를 확인할 때 매우 유용합니다.
    ```python
    print("\n11. `df['name'].value_counts()`: 'name' 컬럼의 고유 값 빈도수")
    df['name'].value_counts()
    ```

#### 1.3.3. `loc`를 이용한 레이블 기반 접근

`loc` 접근자는 Pandas DataFrame에서 **행의 레이블(이름)과 열의 레이블(컬럼명)을 기반**으로 데이터를 선택하거나 수정할 때 사용하는 강력한 인덱서입니다. `loc`는 항상 **명시적인 레이블**을 사용하며, 슬라이싱 시 끝 레이블을 **포함**하는 특징이 있습니다.

-   **기본 문법**: `df.loc[행_레이블, 열_레이블]`
    -   `행_레이블`: 단일 레이블, 레이블 리스트, 레이블 슬라이스, 불리언 배열, 또는 호출 가능한(callable) 함수.
    -   `열_레이블`: 단일 레이블, 레이블 리스트, 레이블 슬라이스, 불리언 배열, 또는 호출 가능한(callable) 함수.

    ```python
    import pandas as pd

    data = {
        'name': ['홍길동', '임꺽정', '장길산', '홍경래', '이상민', '김수경'],
        'kor': [90, 80, 70, 70, 60, 70],
        'eng': [99, 98, 97, 46, 77, 56],
        'mat': [90, 70, 70, 60, 88, 99],
    }
    df = pd.DataFrame(data, index=['a', 'b', 'c', 'd', 'e', 'f']) # 사용자 정의 인덱스

    print("--- DataFrame loc 접근 (레이블 기반) ---")
    print("원본 DataFrame:\n", df)
    ```

- **1. 단일 요소 선택**: `df.loc[행_레이블, 열_레이블]`
    ```python
    # 행 레이블 'a', 컬럼 'name' 데이터 선택
    print(f"1. 행 레이블 'a', 컬럼 'name' 데이터: {df.loc['a', 'name']}")
    ```
    ```python
    # 행 레이블 'c', 컬럼 'eng' 데이터 선택
    print(f"2. 행 레이블 'c', 컬럼 'eng' 데이터: {df.loc['c', 'eng']}")
    ```

- **2. 행 선택**
    ```python
    # 단일 행 선택: df.loc[행_레이블] (Series 반환)
    print(f"3. 행 레이블 'b' 선택:\n{df.loc['b']}")
    ```
    ```python
    # 여러 행 선택: df.loc[[행_레이블1, 행_레이블2, ...]] (DataFrame 반환)
    print(f"4. 행 레이블 'a', 'c', 'e' 선택:\n{df.loc[['a', 'c', 'e']]}")
    ```

- **3. 컬럼 선택**
    ```python
    # 단일 컬럼 선택: df.loc[:, 열_레이블] (Series 반환)
    print(f"5. 컬럼 'kor' 선택:\n{df.loc[:, 'kor']}")
    ```
    ```python
    # 여러 컬럼 선택: df.loc[:, [열_레이블1, 열_레이블2, ...]] (DataFrame 반환)
    print(f"6. 컬럼 'kor', 'mat' 선택:\n{df.loc[:, ['kor', 'mat']]}")
    ```

- **4. 행과 열 동시 선택**: `df.loc[[행_레이블], [열_레이블]]`
    ```python
    # 행 'a', 'c'의 컬럼 'kor', 'eng' 선택
    print(f"7. 행 'a', 'c'의 컬럼 'kor', 'eng' 선택:\n{df.loc[['a', 'c'], ['kor', 'eng']]}")
    ```

- **5. 레이블 슬라이싱**: `df.loc[시작_레이블:끝_레이블, 시작_레이블:끝_레이블]`
    - **중요**: 슬라이싱 시 끝 레이블을 **포함**합니다.
    ```python
    # 행 'a'부터 'c'까지, 컬럼 'kor'부터 'mat'까지
    print(f"8. 행 'a'부터 'c'까지, 컬럼 'kor'부터 'mat'까지:\n{df.loc['a':'c', 'kor':'mat']}")
    ```
    ```python
    # 행 'd'부터 끝까지, 모든 컬럼
    print(f"9. 행 'd'부터 끝까지, 모든 컬럼:\n{df.loc['d':, :]}")
    ```

- **6. 불리언 조건식을 이용한 선택**: `df.loc[조건식, 열_레이블]`
    - 조건식은 행의 개수와 동일한 길이의 불리언 Series여야 합니다.
    ```python
    # 국어 점수가 80점 이상인 학생의 'name'과 'kor' 점수
    print("10. 국어 점수가 80점 이상인 학생의 'name'과 'kor' 점수:")
    print(df.loc[df['kor'] >= 80, ['name', 'kor']])
    ```
    ```python
    # 여러 조건 결합
    print("11. 영어 점수가 90점 이상이고 수학 점수가 80점 이상인 학생의 모든 정보:")
    print(df.loc[(df['eng'] >= 90) & (df['mat'] >= 80)])
    ```

- **7. 데이터 수정**: `df.loc[행_레이블, 열_레이블] = 새_값`
    ```python
    df_copy = df.copy() # 원본 유지를 위해 복사본 사용
    # 행 'b', 컬럼 'eng' 데이터 수정
    df_copy.loc['b', 'eng'] = 100
    print(f"12. 수정 후 DataFrame (행 'b', 컬럼 'eng' 수정):\n{df_copy}")
    ```
    ```python
    df_copy2 = df.copy() # 원본 유지를 위해 복사본 사용
    # 조건식을 이용한 데이터 수정: 국어 점수가 70점 미만인 학생의 국어 점수를 70으로 수정
    df_copy2.loc[df_copy2['kor'] < 70, 'kor'] = 70
    print(f"13. 국어 점수 70점 미만 학생의 점수 수정 후:\n{df_copy2}")
    ```

#### 1.3.4. `iloc`를 이용한 정수 위치 기반 접근

`iloc` 접근자는 **행과 열의 정수 위치(position)를 기반**으로 데이터에 접근하거나 수정할 때 사용합니다. 파이썬 리스트의 인덱싱과 유사하며, 슬라이싱 시 끝 인덱스를 **포함하지 않습니다**.

-   **기본 문법**: `df.iloc[행_위치, 열_위치]`
    -   `행_위치`: 단일 정수, 정수 리스트, 정수 슬라이스, 불리언 배열, 또는 호출 가능한(callable) 함수.
    -   `열_위치`: 단일 정수, 정수 리스트, 정수 슬라이스, 불리언 배열, 또는 호출 가능한(callable) 함수.

    ```python
    import pandas as pd

    data = {
        'name': ['홍길동', '임꺽정', '장길산', '홍경래'],
        'kor': [90, 80, 70, 70],
        'eng': [99, 98, 97, 46],
        'mat': [90, 70, 70, 60],
    }
    df = pd.DataFrame(data) # 기본 정수 인덱스

    print("--- DataFrame iloc 접근 (정수 위치 기반) ---")
    print("원본 DataFrame:\n", df)
    ```

-   **1. 단일 요소 선택**: `df.iloc[행_위치, 열_위치]`
    ```python
    # 0행 0열 데이터 선택
    print(f"\n1. 0행 0열 데이터: {df.iloc[0, 0]}")
    # 1행 2열 데이터 선택
    print(f"   1행 2열 데이터: {df.iloc[1, 2]}")
    ```

-   **2. 행 선택**
    ```python
    # 특정 행 선택 (Series 반환)
    print(f"\n2. 1번째 행 선택:\n{df.iloc[1]}")

    # 여러 행 선택 (DataFrame 반환)
    print(f"\n3. 0, 2번째 행 선택:\n{df.iloc[[0, 2]]}")
    ```

-   **3. 열 선택**
    ```python
    # 특정 컬럼 선택 (Series 반환)
    print(f"\n4. 2번째 컬럼 ('eng') 선택:\n{df.iloc[:, 2]}")

    # 여러 컬럼 선택 (DataFrame 반환)
    print(f"\n5. 1, 3번째 컬럼 선택:\n{df.iloc[:, [1, 3]]}")
    ```

-   **4. 행과 열 동시 선택**
    ```python
    # 0, 2번째 행의 1, 3번째 컬럼 선택
    print(f"\n6. 0, 2번째 행의 1, 3번째 컬럼 선택:\n{df.iloc[[0, 2], [1, 3]]}")
    ```

-   **5. 위치 슬라이싱 (끝 위치 미포함)**
    ```python
    # 1번째부터 3번째 행까지 (1, 2, 3행), 0번째부터 2번째 컬럼까지 (0, 1, 2열)
    print(f"\n7. 1번째부터 3번째 행까지, 0번째부터 2번째 컬럼까지:\n{df.iloc[1:4, 0:3]}")
    ```

-   **6. 데이터 수정**
    ```python
    # 0행 1열 (kor)의 값을 95로 수정
    df_copy = df.copy() # 원본 유지를 위해 복사본 사용
    df_copy.iloc[0, 1] = 95
    print(f"\n8. 수정 후 DataFrame (0행 1열 수정):\n{df_copy}")
    ```


#### 1.3.5. 조건식을 이용한 필터링 (Boolean Indexing)

`DataFrame`에서도 `Series`와 유사하게 조건식을 사용하여 특정 조건을 만족하는 행을 선택할 수 있습니다. 여러 조건을 결합할 때는 `&` (AND), `|` (OR) 연산자를 사용하고 각 조건은 괄호로 묶어야 합니다. 조건에 맞는 특정 컬럼만 선택하거나, 조건에 맞는 데이터를 수정할 수도 있습니다.

```python
import pandas as pd

data = {
    'name': ['홍길동', '임꺽정', '장길산', '홍경래', '이상민', '김수경'],
    'kor': [90, 80, 70, 70, 60, 70],
    'eng': [99, 98, 97, 46, 77, 56],
    'mat': [90, 70, 70, 60, 88, 99],
}
df = pd.DataFrame(data)

print("\n--- DataFrame 조건식 필터링 ---")
print("원본 DataFrame:\n", df)
```

- **1. 단일 조건 필터링**
    ```python
    # 국어 점수가 80점 이상인 학생
    print("\n1. 국어 점수가 80점 이상인 학생:\n", df[df['kor'] >= 80])
    ```

- **2. 여러 조건 결합 (`&`, `|`)**
    ```python
    # 영어 점수가 90점 이상이고(AND) 수학 점수가 80점 이상인 학생
    print("\n2. 영어 점수가 90점 이상이고 수학 점수가 80점 이상인 학생:\n", df[(df['eng'] >= 90) & (df['mat'] >= 80)])

    # 국어 점수가 70점이거나(OR) 영어 점수가 60점 미만인 학생
    print("\n3. 국어 점수가 70점이거나 영어 점수가 60점 미만인 학생:\n", df[(df['kor'] == 70) | (df['eng'] < 60)])
    ```

- **3. 조건에 맞는 특정 컬럼 선택**
    ```python
    # 국어 점수가 70점 미만인 학생의 'name'과 'eng' 점수
    # .loc를 사용하면 조건 필터링과 컬럼 선택을 동시에 할 수 있어 권장됩니다.
    print("\n4. 국어 점수가 70점 미만인 학생의 이름과 영어 점수:\n", df.loc[df['kor'] < 70, ['name', 'eng']])
    ```

- **4. `loc`를 이용한 조건부 데이터 수정**
    ```python
    df_copy = df.copy() # 원본 유지를 위해 복사본 사용
    # 국어 점수가 70점 미만인 학생의 국어 점수를 70으로 수정
    df_copy.loc[df_copy['kor'] < 70, 'kor'] = 70
    print(f"\n5. 국어 점수 70점 미만 학생의 점수 수정 후:\n{df_copy}")
    ```


#### 1.3.6. 컬럼 추가 및 수정

새로운 컬럼을 추가하거나 기존 컬럼의 값을 수정하는 것은 매우 간단합니다. 새로운 컬럼은 기존 컬럼들의 연산 결과로 생성될 수 있으며, 여러 컬럼을 한 번에 추가하거나 컬럼 이름을 변경할 수도 있습니다.

```python
import pandas as pd

data = {
    'name': ['홍길동', '임꺽정', '장길산', '홍경래'],
    'kor': [90, 80, 70, 70],
    'eng': [99, 98, 97, 46],
    'mat': [90, 70, 70, 60],
}
df = pd.DataFrame(data)

print("\n--- DataFrame 컬럼 추가 및 수정 ---")
print("원본 DataFrame:\n", df)
```

- **1. 새로운 컬럼 추가**
    ```python
    df_copy = df.copy() # 원본 유지를 위해 복사본 사용
    # 'total' 컬럼 추가 (기존 컬럼들의 합)
    df_copy['total'] = df_copy['kor'] + df_copy['eng'] + df_copy['mat']
    print("\n1. total 컬럼 추가 후:\n", df_copy)

    # 'avg' 컬럼 추가 (total 컬럼의 평균)
    df_copy['avg'] = df_copy['total'] / 3
    print("\n2. avg 컬럼 추가 후:\n", df_copy)
    ```

- **2. 기존 컬럼 값 수정**
    ```python
    df_copy = df.copy() # 독립적인 예시를 위해 다시 복사
    # 'kor' 점수를 10점씩 올리기
    df_copy['kor'] = df_copy['kor'] + 10
    print("\n3. 'kor' 점수 10점씩 올린 후:\n", df_copy)
    ```

- **3. 여러 컬럼 한 번에 추가**
    ```python
    df_copy = df.copy() # 독립적인 예시를 위해 다시 복사
    # 모든 점수를 5점씩 올린 새 컬럼 추가
    df_copy[['kor_plus5', 'eng_plus5']] = df_copy[['kor', 'eng']] + 5
    print("\n4. 여러 컬럼 한 번에 추가 후:\n", df_copy)
    ```

- **4. 컬럼 이름 변경 (`rename`)**
    ```python
    # 'kor' -> 'Korean', 'eng' -> 'English'로 변경
    # rename()은 기본적으로 원본을 수정하지 않고 새 DataFrame을 반환합니다.
    df_renamed = df.rename(columns={'kor': 'Korean', 'eng': 'English'})
    print("\n5. 컬럼 이름 변경 후:\n", df_renamed)
    print("\n   원본 DataFrame은 변경되지 않음:\n", df)
    ```


#### 1.3.7. 컬럼 삭제

`drop()` 메서드를 사용하여 컬럼을 삭제할 수 있습니다. `axis=1`은 컬럼(열)을 의미합니다. `inplace=True`를 사용하면 원본 `DataFrame`을 직접 수정하고, 그렇지 않으면 수정된 새 `DataFrame`을 반환합니다. 원본 유지를 위해 `inplace=False` (기본값)를 사용하거나, 반환값을 새로운 변수에 할당하는 것이 좋습니다.

```python
import pandas as pd

data = {
    'name': ['홍길동', '임꺽정', '장길산', '홍경래'],
    'kor': [90, 80, 70, 70],
    'eng': [99, 98, 97, 46],
    'mat': [90, 70, 70, 60],
}
df = pd.DataFrame(data)

print("\n--- DataFrame 컬럼 삭제 ---")
print("원본 DataFrame:\n", df)
```

- **1. 단일 컬럼 삭제**
    ```python
    # 'kor' 컬럼 삭제 (drop은 기본적으로 원본을 수정하지 않고 새 DataFrame을 반환)
    df_dropped_kor = df.drop('kor', axis=1)
    print("\n1. 'kor' 컬럼 삭제 후:\n", df_dropped_kor)
    print("\n   원본 DataFrame은 변경되지 않음:\n", df)
    ```

- **2. 여러 컬럼 삭제**
    ```python
    # 'eng', 'mat' 컬럼 삭제
    df_dropped_multiple = df.drop(['eng', 'mat'], axis=1)
    print("\n2. 'eng', 'mat' 컬럼 삭제 후:\n", df_dropped_multiple)
    ```

- **3. 원본 DataFrame 직접 수정 (`inplace=True`)**
    ```python
    df_copy = df.copy() # 원본 유지를 위해 복사본 사용
    # 'mat' 컬럼을 원본에서 직접 삭제
    return_value = df_copy.drop('mat', axis=1, inplace=True)
    print("\n3. 'mat' 컬럼 원본에서 직접 삭제 후:\n", df_copy)
    print(f"   inplace=True일 때 반환값: {return_value}")
    ```

#### 1.3.8. 행 추가 및 삭제

행을 추가할 때는 `pd.concat()` 함수를 사용하는 것이 권장됩니다. 기존 `append()` 메서드는 Pandas 2.0부터 Deprecated(사용 중단)되었습니다. 행을 삭제할 때는 `drop()` 메서드를 사용하며, `axis=0`은 행을 의미합니다.

```python
import pandas as pd

data_fruits = {
    'fruits': ['망고', '딸기', '수박', '파인애플'],
    'price': [2500, 5000, 10000, 7000],
    'count': [5, 2, 2, 4],
}
df_fruits = pd.DataFrame(data_fruits)
print("--- DataFrame 행 추가 및 삭제 ---")
print("원본 DataFrame:\n", df_fruits)
```

- **1. 행 추가 (`pd.concat`)**

    `pd.concat` 함수는 여러 DataFrame이나 Series를 리스트 형태로 받아 연결합니다. 새로운 행을 추가하려면, 추가할 행을 별도의 DataFrame으로 만든 후 기존 DataFrame과 연결합니다. `ignore_index=True` 옵션은 기존 인덱스를 무시하고 0부터 시작하는 새 인덱스를 부여합니다.

    ```python
    # 새로운 행 추가 (pd.concat을 이용한 권장 방식)
    new_row = pd.DataFrame([{'fruits': '사과', 'price': 3500, 'count': 10}])
    df_added_row = pd.concat([df_fruits, new_row], ignore_index=True)
    print("\n1. 새로운 행 추가 후 (pd.concat 사용):\n", df_added_row)

    # 여러 행 추가
    new_rows_multiple = pd.DataFrame([
        {'fruits': '포도', 'price': 6000, 'count': 3},
        {'fruits': '오렌지', 'price': 4000, 'count': 7}
    ])
    df_added_rows = pd.concat([df_added_row, new_rows_multiple], ignore_index=True)
    print("\n2. 여러 행 추가 후:\n", df_added_rows)
    ```

- **2. 행 삭제 (`drop`)**

    `drop` 메서드에 삭제할 행의 인덱스 레이블을 전달하여 행을 삭제합니다. `axis=0`은 행 삭제를 의미하며, 기본값이므로 생략 가능합니다.

    ```python
    # 특정 행 삭제 (axis=0: 행)
    # drop()은 기본적으로 원본을 수정하지 않고 새 DataFrame을 반환합니다.
    df_dropped_row = df_added_rows.drop(0, axis=0) # 0번 인덱스 행 삭제
    print("\n3. 0번 인덱스 행 삭제 후:\n", df_dropped_row)

    # 여러 행 삭제
    df_dropped_multiple_rows = df_added_rows.drop([1, 3], axis=0) # 1번, 3번 인덱스 행 삭제
    print("\n4. 1번, 3번 인덱스 행 삭제 후:\n", df_dropped_multiple_rows)
    ```

#### 1.3.9. 결측치 처리 (간략)

`DataFrame`은 결측치(Missing Values)를 `NaN`으로 표현합니다. `dropna()`와 `fillna()` 메서드를 사용하여 결측치를 처리할 수 있습니다.

```python
import pandas as pd
import numpy as np

data_with_nan = {
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, 12]
}
df_nan = pd.DataFrame(data_with_nan)
print("--- DataFrame 결측치 처리 ---")
print("원본 DataFrame (결측치 포함):\n", df_nan)
```

- **1. 결측치 제거 (`dropna`)**

    `dropna()` 메서드는 결측치가 포함된 행 또는 열을 제거합니다. `axis=0`은 행(기본값), `axis=1`은 열을 의미합니다.

    ```python
    # 결측치가 하나라도 있는 행 삭제
    df_dropna_row = df_nan.dropna()
    print("\n1. 결측치가 있는 행 삭제 후:\n", df_dropna_row)

    # 결측치가 하나라도 있는 컬럼 삭제
    df_dropna_col = df_nan.dropna(axis=1)
    print("\n2. 결측치가 있는 컬럼 삭제 후:\n", df_dropna_col)
    ```

- **2. 결측치 채우기 (`fillna`)**

    `fillna()` 메서드는 결측치를 특정 값이나 계산된 값으로 대체합니다.

    ```python
    # 결측치를 특정 값(0)으로 채우기
    df_fillna_zero = df_nan.fillna(0)
    print("\n3. 결측치를 0으로 채운 후:\n", df_fillna_zero)

    # 결측치를 각 컬럼의 평균값으로 채우기
    # df_nan.mean()은 각 컬럼의 평균을 Series로 반환합니다.
    df_fillna_mean = df_nan.fillna(df_nan.mean())
    print("\n4. 결측치를 컬럼 평균으로 채운 후:\n", df_fillna_mean)
    ```
