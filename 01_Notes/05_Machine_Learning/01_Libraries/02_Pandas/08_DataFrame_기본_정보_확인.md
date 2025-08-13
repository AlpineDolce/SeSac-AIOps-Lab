<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 Pandas `DataFrame`의 구조를 파악하고 기본적인 통계 정보를 얻는 데 유용한 다양한 API(메서드)를 다룹니다. `head()`, `tail()`, `shape`, `info()`, `describe()`와 같은 함수들을 사용하여 데이터 탐색(EDA)의 첫 단계에서 데이터의 전체적인 모습과 특성을 빠르게 파악하는 방법을 학습합니다.

<h2>목차</h2>

- [1. DataFrame API 활용](#1-dataframe-api-활용)
  - [1.1. 기본 정보 확인 API](#11-기본-정보-확인-api)
    - [1.1.1. `head()` / `tail()`: 데이터 미리보기](#111-head--tail-데이터-미리보기)
    - [1.1.2. `shape`, `ndim`, `size`: DataFrame의 차원 정보](#112-shape-ndim-size-dataframe의-차원-정보)
    - [1.1.3. `info()`: DataFrame의 간략한 정보 요약](#113-info-dataframe의-간략한-정보-요약)
    - [1.1.4. `describe()`: 숫자형 컬럼 기술 통계](#114-describe-숫자형-컬럼-기술-통계)
    - [1.1.5. `value_counts()`: 고유 값 빈도 계산](#115-value_counts-고유-값-빈도-계산)
    - [1.1.6. `unique()` / `nunique()`: 고유 값 및 개수 확인](#116-unique--nunique-고유-값-및-개수-확인)
    - [1.1.7. `isnull()` / `notnull()`: 결측치 확인](#117-isnull--notnull-결측치-확인)
    - [1.1.8. `dtypes` 속성: 컬럼별 데이터 타입 확인](#118-dtypes-속성-컬럼별-데이터-타입-확인)
    - [1.1.9. `index` 및 `columns` 속성: 인덱스/컬럼 레이블 확인](#119-index-및-columns-속성-인덱스컬럼-레이블-확인)

---

## 1. DataFrame API 활용

Pandas `DataFrame`은 데이터의 구조를 파악하고 기본적인 통계 정보를 얻는 데 유용한 다양한 API(메서드)를 제공합니다. 이는 데이터 탐색(EDA)의 첫 단계에서 매우 중요합니다.

### 1.1. 기본 정보 확인 API

#### 1.1.1. `head()` / `tail()`: 데이터 미리보기

`DataFrame`의 상위 또는 하위 n개의 행을 출력하여 데이터의 전체적인 모습을 빠르게 파악할 수 있습니다. 데이터 로드 후 가장 먼저 수행하는 작업 중 하나로, 데이터가 올바르게 불러와졌는지, 어떤 형태인지 등을 빠르게 확인하는 데 유용합니다. 기본값은 5개 행입니다.

```python
import pandas as pd
import numpy as np

# 예시 데이터셋 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi'],
    'Age': [25, 30, 35, 40, 28, 32, 29, 31],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul', 'London', 'New York', 'Paris'],
    'Score': [85, 92, 78, 95, 88, 75, 91, 80]
}
df = pd.DataFrame(data, index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])

print("--- DataFrame 미리보기 ---")
print("앞에서부터 5개 미리 보기 (df.head()):\n", df.head())

print("\n뒤에서부터 3개 미리 보기 (df.tail(3)):\n", df.tail(3))

print("\n앞에서부터 6개 미리 보기 (df.head(6)):\n", df.head(6))
```

#### 1.1.2. `shape`, `ndim`, `size`: DataFrame의 차원 정보

`DataFrame`의 전체적인 크기와 차원 정보를 파악하는 속성들입니다.

*   `shape`: `DataFrame`의 차원(dimensions)을 튜플 형태로 반환합니다. `(행의 개수, 열의 개수)`로 구성됩니다.
*   `ndim`: `DataFrame`의 차원 수(항상 2)를 반환합니다.
*   `size`: `DataFrame`의 총 요소(셀) 개수를 반환합니다. (`행의 개수 * 열의 개수`)

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
}
df = pd.DataFrame(data)

print("--- DataFrame의 차원 정보 ---")
print(f"DataFrame의 형태 (df.shape): {df.shape}")
print(f"DataFrame의 차원 수 (df.ndim): {df.ndim}")
print(f"DataFrame의 총 요소 개수 (df.size): {df.size}")

# 행과 열의 개수를 개별 변수에 할당
rows, cols = df.shape
print(f"행의 개수: {rows}")
print(f"열의 개수: {cols}")
```

#### 1.1.3. `info()`: DataFrame의 간략한 정보 요약

`DataFrame`의 간략한 정보를 출력합니다. 각 컬럼의 데이터 타입, Non-null 값의 개수, 메모리 사용량, 인덱스 정보 등을 포함하여 데이터의 누락 여부와 타입을 빠르게 확인할 수 있어 데이터 전처리 계획을 세우는 데 매우 유용합니다.

**`info()` 함수 제공 정보 요약**:
*   **RangeIndex**: `DataFrame`의 행 인덱스 범위와 스텝을 보여줍니다.
*   **Data columns (total N columns)**: 전체 컬럼의 개수를 나타냅니다.
*   **Column**: 컬럼 이름.
*   **Non-Null Count**: 각 컬럼에 결측치(`NaN`)가 아닌 유효한 데이터가 몇 개 있는지 보여줍니다. 이를 통해 결측치 여부를 쉽게 파악할 수 있습니다.
*   **Dtype**: 각 컬럼이 어떤 데이터 타입(int64, float64, object, category 등)을 가지는지 보여줍니다.
*   **Memory Usage**: `DataFrame`이 사용하는 총 메모리 양을 나타냅니다.

```python
import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, np.nan, 40, 28], # 결측치 포함
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul'],
    'Score': [85.5, 90.0, 78.2, np.nan, 92.1], # 결측치 포함
    'Gender': pd.Categorical(['M', 'F', 'M', 'F', 'M']) # 범주형 컬럼
}
df = pd.DataFrame(data)

print("--- DataFrame의 기본 구조 (df.info()) ---")
df.info()
```

#### 1.1.4. `describe()`: 숫자형 컬럼 기술 통계

`DataFrame`의 숫자형 컬럼에 대한 기술 통계(Descriptive Statistics)를 계산하여 출력합니다. 데이터의 분포와 중심 경향성, 퍼짐 정도 등을 파악하는 데 유용합니다.

**`describe()` 함수 제공 정보 요약**:
*   `count`: 해당 컬럼의 유효한(Non-null) 데이터 개수.
*   `mean`: 평균값.
*   `std`: 표준편차 (Standard Deviation).
*   `min`: 최솟값.
*   `25%` (1사분위수): 데이터를 오름차순으로 정렬했을 때 하위 25% 지점의 값.
*   `50%` (중앙값/2사분위수): 데이터를 오름차순으로 정렬했을 때 중간 지점의 값. 중앙값은 극단적인 값(이상치)에 덜 민감하여 데이터의 중심을 나타내는 데 평균보다 더 견고할 수 있습니다.
*   `75%` (3사분위수): 데이터를 오름차순으로 정렬했을 때 상위 25% 지점의 값.
*   `max`: 최댓값.

```python
import pandas as pd
import numpy as np

data = {
    'Age': [25, 30, 35, 40, 28, 32, 29, 31],
    'Score': [85, 92, 78, 95, 88, 75, 91, 80],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul', 'London', 'New York', 'Paris'],
    'Grade': pd.Categorical(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'])
}
df = pd.DataFrame(data)

print("--- 숫자형 컬럼의 요약 통계 정보 (df.describe()) ---")
print(df.describe())

# 비숫자형 컬럼 포함 (object, categorical)
print("\n--- 모든 컬럼의 요약 통계 정보 (df.describe(include='all')) ---")
print(df.describe(include='all'))

# 특정 데이터 타입만 포함
print("\n--- object 타입 컬럼의 요약 통계 정보 (df.describe(include=['object'])) ---")
print(df.describe(include=['object']))

print("\n--- categorical 타입 컬럼의 요약 통계 정보 (df.describe(include=['category'])) ---")
print(df.describe(include=['category']))
```

#### 1.1.5. `value_counts()`: 고유 값 빈도 계산

`Series`의 고유한 값들과 각 값의 등장 횟수를 계산하여 내림차순으로 반환합니다. 범주형 데이터나 문자열 데이터의 분포를 파악하는 데 매우 유용합니다.

```python
import pandas as pd

data = {
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul', 'London', 'New York', 'Paris'],
    'Grade': pd.Categorical(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'])
}
df = pd.DataFrame(data)

print("--- 'City' 컬럼의 고유 값 빈도 ---")
print(df['City'].value_counts())

print("\n--- 'Grade' 컬럼의 고유 값 빈도 ---")
print(df['Grade'].value_counts())

# 정규화된 빈도 (비율)
print("\n--- 'City' 컬럼의 정규화된 빈도 (비율) ---")
print(df['City'].value_counts(normalize=True))
```

#### 1.1.6. `unique()` / `nunique()`: 고유 값 및 개수 확인

*   `unique()`: `Series`의 모든 고유한 값들을 NumPy 배열 형태로 반환합니다.
*   `nunique()`: `Series`의 고유한 값들의 개수를 반환합니다. 결측치(`NaN`)는 기본적으로 포함하지 않습니다.

```python
import pandas as pd
import numpy as np

s = pd.Series(['A', 'B', 'A', 'C', np.nan, 'B'])

print("--- 고유 값 및 개수 확인 ---")
print(f"원본 Series:\n{s}")

print(f"\n고유 값 (s.unique()): {s.unique()}")
print(f"고유 값 개수 (s.nunique()): {s.nunique()}") # NaN은 기본적으로 제외

print(f"고유 값 개수 (s.nunique(dropna=False)): {s.nunique(dropna=False)}") # NaN 포함
```

#### 1.1.7. `isnull()` / `notnull()`: 결측치 확인

`DataFrame` 또는 `Series`의 각 요소가 결측치(`NaN`)인지 여부를 불리언 `DataFrame` 또는 `Series`로 반환합니다. `isnull()`은 `NaN`이면 `True`, `notnull()`은 `NaN`이 아니면 `True`를 반환합니다.

```python
import pandas as pd
import numpy as np

data = {
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, 12]
}
df = pd.DataFrame(data)

print("--- 결측치 확인 ---")
print("원본 DataFrame:\n", df)

print("\n결측치 여부 (df.isnull()):\n", df.isnull())
print("\n결측치 아님 여부 (df.notnull()):\n", df.notnull())

# 컬럼별 결측치 개수
print("\n컬럼별 결측치 개수:\n", df.isnull().sum())

# 전체 결측치 개수
print(f"\n전체 결측치 개수: {df.isnull().sum().sum()}")
```

#### 1.1.8. `dtypes` 속성: 컬럼별 데이터 타입 확인

`DataFrame`의 각 컬럼에 대한 데이터 타입(`dtype`)을 `Series` 형태로 반환합니다. 데이터 전처리 시 컬럼의 타입을 확인하고 필요에 따라 변환하는 데 중요합니다.

```python
import pandas as pd
import numpy as np

data = {
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30],
    'Score': [85.5, 90.0],
    'Is_Student': [True, False],
    'City': pd.Categorical(['New York', 'London'])
}
df = pd.DataFrame(data)

print("--- 컬럼별 데이터 타입 확인 (df.dtypes) ---")
print(df.dtypes)
```

#### 1.1.9. `index` 및 `columns` 속성: 인덱스/컬럼 레이블 확인

`DataFrame`의 행 인덱스(`index`)와 컬럼 레이블(`columns`)에 직접 접근할 수 있습니다. 이들은 `Index` 객체로 반환됩니다.

```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob'],
    'Age': [25, 30]
}
df = pd.DataFrame(data, index=['row1', 'row2'])

print("--- 인덱스 및 컬럼 레이블 확인 ---")
print(f"DataFrame의 행 인덱스 (df.index): {df.index}")
print(f"DataFrame의 컬럼 레이블 (df.columns): {df.columns}")

# 인덱스/컬럼을 리스트로 변환
print(f"\n행 인덱스 리스트: {df.index.tolist()}")
print(f"컬럼 레이블 리스트: {df.columns.tolist()}")
```