<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
이 문서는 Pandas의 핵심 2차원 데이터 구조인 `DataFrame`의 개념, 특징, 다양한 생성 방법, 그리고 데이터 접근 및 조작 방법을 상세히 다룹니다. `DataFrame`을 효율적으로 다루는 실무 역량을 강화하는 데 중점을 둡니다.

<h2>목차</h2>

- [1. DataFrame의 개념 및 특징](#11-dataframe의-개념-및-특징)
- [2. DataFrame 생성 방법](#12-dataframe-생성-방법)
  - [2.1. 딕셔너리(dict)로 DataFrame 만들기](#121-딕셔너리dict로-dataframe-만들기)
  - [2.2. 리스트(list)의 리스트로 DataFrame 만들기](#122-리스트list의-리스트로-dataframe-만들기)
  - [2.3. NumPy 배열(ndarray)로 DataFrame 만들기](#123-numpy-배열ndarray로-dataframe-만들기)
  - [2.4. 리스트의 딕셔너리로 DataFrame 만들기](#124-리스트의-딕셔너리로-dataframe-만들기)
  - [2.5. Series 딕셔너리로 DataFrame 만들기](#125-series-딕셔너리로-dataframe-만들기)
  - [2.6. 빈 DataFrame 만들기](#126-빈-dataframe-만들기)
- [3. DataFrame 데이터 접근 및 조작](#13-dataframe-데이터-접근-및-조작)
  - [3.1. 컬럼(열) 선택](#131-컬럼열-선택)
  - [3.2. `loc`를 이용한 레이블 기반 접근](#132-loc를-이용한-레이블-기반-접근)
  - [3.3. `iloc`를 이용한 정수 위치 기반 접근](#133-iloc를-이용한-정수-위치-기반-접근)

---


### 1. DataFrame의 개념 및 특징

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

### 2. DataFrame 생성 방법

`DataFrame`은 다양한 파이썬 객체로부터 생성할 수 있습니다. 데이터의 초기 형태에 따라 적절한 방법을 선택하는 것이 중요합니다.

#### 2.1. 딕셔너리(dict)로 DataFrame 만들기

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

#### 2.2. 리스트(list)의 리스트로 DataFrame 만들기

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

#### 2.3. NumPy 배열(ndarray)로 DataFrame 만들기

NumPy 2차원 배열을 사용하여 `DataFrame`을 생성할 수 있습니다. 이는 대규모 수치 데이터를 `DataFrame`으로 변환할 때 효율적이며, `columns` 인자를 통해 컬럼 이름을 지정하는 것이 일반적입니다.

```python
import numpy as np
import pandas as pd

# NumPy 배열로 Series 생성
np_array = np.array([100, 200, 300])
s_from_np = pd.Series(np_array)
print("\n--- NumPy 배열로 생성된 DataFrame ---")
print(s_from_np)
```

#### 2.4. 리스트의 딕셔너리로 DataFrame 만들기

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

#### 2.5. Series 딕셔너리로 DataFrame 만들기

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

#### 2.6. 빈 DataFrame 만들기

컬럼 이름만 지정하여 빈 `DataFrame`을 만들 수 있습니다. 나중에 데이터를 추가할 때 유용합니다.

```python
import pandas as pd

empty_df = pd.DataFrame(columns=['Name', 'Age', 'City'])
print("\n--- 빈 DataFrame ---\
", empty_df)
print(f"빈 DataFrame의 형태: {empty_df.shape}")
```

### 3. DataFrame 데이터 접근 및 조작

`DataFrame`의 데이터는 다양한 방법으로 접근하고 조작할 수 있습니다. `Series`와 마찬가지로 `loc`와 `iloc` 접근자를 사용하여 명시적인 접근을 하는 것이 권장됩니다.

#### 3.1. 컬럼(열) 선택

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

print("--- DataFrame 컬럼 선택 ---\
", df)
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

#### 3.2. `loc`를 이용한 레이블 기반 접근

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

    print("--- DataFrame loc 접근 (레이블 기반) ---\
", df)
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

- **2. 단일 행 선택**: `df.loc[행_레이블]` (Series 반환)
    ```python
    print(f"3. 행 레이블 'b' 선택:\n{df.loc['b']}")
    ```

- **3. 단일 컬럼 선택**: `df.loc[:, 열_레이블]` (Series 반환)
    ```python
    print(f"4. 컬럼 'kor' 선택:\n{df.loc[:, 'kor']}")
    ```

- **4. 데이터 수정**: `df.loc[행_레이블, 열_레이블] = 새_값`
    ```python
    df_copy = df.copy() # 원본 유지를 위해 복사본 사용
    # 행 'b', 컬럼 'eng' 데이터 수정
    df_copy.loc['b', 'eng'] = 100
    print(f"5. 수정 후 DataFrame (행 'b', 컬럼 'eng' 수정):\n{df_copy}")
    ```
> **참고**: `loc`를 이용한 고급 슬라이싱, 여러 행/열 선택, 불리언 조건식 필터링 등은 [**12_고급_인덱싱_및_선택.md**](./12_고급_인덱싱_및_선택.md) 및 [**09_조건부_데이터_검색.md**](./09_조건부_데이터_검색.md)에서 더 자세히 다룹니다.

#### 3.3. `iloc`를 이용한 정수 위치 기반 접근

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

    print("--- DataFrame iloc 접근 (정수 위치 기반) ---", df)
    ```

-   **1. 단일 요소 선택**: `df.iloc[행_위치, 열_위치]`
    ```python
    # 0행 0열 데이터 선택
    print(f"\n1. 0행 0열 데이터: {df.iloc[0, 0]}")
    # 1행 2열 데이터 선택
    print(f"   1행 2열 데이터: {df.iloc[1, 2]}")
    ```

-   **2. 단일 행 선택**: `df.iloc[행_위치]` (Series 반환)
    ```python
    print(f"\n2. 1번째 행 선택:\n{df.iloc[1]}")
    ```

-   **3. 단일 컬럼 선택**: `df.iloc[:, 열_위치]` (Series 반환)
    ```python
    print(f"\n3. 2번째 컬럼 ('eng') 선택:\n{df.iloc[:, 2]}")
    ```

-   **4. 데이터 수정**: `df.iloc[행_위치, 열_위치] = 새_값`
    ```python
    # 0행 1열 (kor)의 값을 95로 수정
    df_copy = df.copy() # 원본 유지를 위해 복사본 사용
    df_copy.iloc[0, 1] = 95
    print(f"\n4. 수정 후 DataFrame (0행 1열 수정):\n{df_copy}")
    ```
- **참고**: `iloc`를 이용한 고급 슬라이싱, 여러 행/열 선택 등은 [**12_고급_인덱싱_및_선택.md**](./12_고급_인덱싱_및_선택.md)에서 더 자세히 다룹니다.