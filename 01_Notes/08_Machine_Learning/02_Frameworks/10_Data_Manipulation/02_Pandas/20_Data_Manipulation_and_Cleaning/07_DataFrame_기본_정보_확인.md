<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 Pandas `DataFrame`의 구조를 파악하고 기본적인 통계 정보를 얻는 데 유용한 다양한 API(메서드)를 다룹니다. `head()`, `tail()`, `shape`, `info()`, `describe()`와 같은 함수들을 사용하여 데이터 탐색(EDA)의 첫 단계에서 데이터의 전체적인 모습과 특성을 빠르게 파악하는 방법을 학습합니다.

<h2>목차</h2>

- [1. `head()` / `tail()`: 데이터 미리보기 (Quick Data Inspection)](#1-head--tail-데이터-미리보기-quick-data-inspection)
- [2. `shape`, `ndim`, `size`: DataFrame의 차원 정보 (Dimensionality and Size)](#2-shape-ndim-size-dataframe의-차원-정보-dimensionality-and-size)
- [3. `info()`: DataFrame의 간략한 정보 요약 (Concise DataFrame Summary)](#3-info-dataframe의-간략한-정보-요약-concise-dataframe-summary)
- [4. `describe()`: 숫자형 컬럼 기술 통계 (Descriptive Statistics for Numerical Columns)](#4-describe-숫자형-컬럼-기술-통계-descriptive-statistics-for-numerical-columns)
- [5. `value_counts()`: 고유 값 빈도 계산](#5-value_counts-고유-값-빈도-계산)
- [6. `unique()` / `nunique()`: 고유 값 및 개수 확인](#6-unique--nunique-고유-값-및-개수-확인)
  - [7. `isnull()` / `notnull()`: 결측치 확인](#7-isnull--notnull-결측치-확인)
- [8. `dtypes` 속성: 컬럼별 데이터 타입 확인](#8-dtypes-속성-컬럼별-데이터-타입-확인)
- [9. `index` 및 `columns` 속성: 인덱스/컬럼 레이블 확인](#9-index-및-columns-속성-인덱스컬럼-레이블-확인)

---

## 1. `head()` / `tail()`: 데이터 미리보기 (Quick Data Inspection)

`DataFrame`의 `head()`와 `tail()` 메서드는 데이터셋의 상위 또는 하위 `n`개의 행을 출력하여 데이터의 구조와 내용을 빠르게 탐색하는 데 사용됩니다. 이는 데이터 로드 후 가장 먼저 수행하는 작업 중 하나로, 데이터 탐색(EDA, Exploratory Data Analysis)의 필수적인 첫 단계입니다. 이 메서드들을 통해 데이터가 올바르게 불러와졌는지, 컬럼 이름과 데이터 형식은 어떤지, 그리고 데이터의 전반적인 패턴을 신속하게 파악할 수 있습니다.

**주요 기능 및 활용 목적**:
*   **데이터 유효성 검사**: 데이터 로드 직후 데이터가 예상대로 로드되었는지, 누락되거나 손상된 부분이 없는지 육안으로 빠르게 확인합니다.
*   **컬럼 및 데이터 타입 확인**: 컬럼 이름이 올바른지, 각 컬럼의 데이터가 예상하는 타입(숫자, 문자열, 날짜 등)으로 보이는지 초기 검토합니다.
*   **데이터 패턴 파악**: 데이터의 첫 부분과 끝 부분을 통해 전반적인 데이터 분포나 순서(예: 시간 순서)를 대략적으로 파악합니다.

**메서드 상세**:
*   **`head(n=5)`**: `DataFrame`의 **상위 `n`개 행**을 반환합니다. `n`의 기본값은 5입니다. `n`을 지정하지 않으면 기본적으로 5개의 행을 보여줍니다.
*   **`tail(n=5)`**: `DataFrame`의 **하위 `n`개 행**을 반환합니다. `n`의 기본값은 5입니다. `tail()`은 특히 시계열 데이터나 로그 데이터처럼 최신 데이터가 중요한 경우 유용합니다.

**예시 코드 및 출력**:

```python
import pandas as pd
import numpy as np

# 예시 데이터셋 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Heidi', 'Ivy', 'Jack'],
    'Age': [25, 30, 35, 40, 28, 32, 29, 31, 26, 33],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul', 'London', 'New York', 'Paris', 'Tokyo', 'Seoul'],
    'Score': [85, 92, 78, 95, 88, 75, 91, 80, 89, 77],
    'Enroll_Date': pd.to_datetime(['2023-0', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-01',
                                   '2023-06-12', '2023-07-01', '2023-08-18', '2023-09-25', '2023-10-30'])
}
df = pd.DataFrame(data, index=[f'idx_{i}' for i in range(1, 11)])

print("--- DataFrame 미리보기 ---")

print("
1. df.head() (기본값 n=5): DataFrame의 상위 5개 행")
print(df.head())

print("
2. df.head(3): DataFrame의 상위 3개 행")
print(df.head(3))

print("
3. df.tail() (기본값 n=5): DataFrame의 하위 5개 행")
print(df.tail())

print("
4. df.tail(2): DataFrame의 하위 2개 행")
print(df.tail(2))

# 특정 컬럼의 head/tail 확인 (Series에도 적용 가능)
print("
5. 'Score' 컬럼의 상위 3개 값")
print(df['Score'].head(3))
```

**출력 예시**:
```
--- DataFrame 미리보기 ---

1. df.head() (기본값 n=5): DataFrame의 상위 5개 행
       Name  Age      City  Score Enroll_Date
idx_1   Alice   25  New York     85  2023-0
idx_2     Bob   30    London     92  2023-02-20
idx_3 Charlie   35     Paris     78  2023-03-10
idx_4   David   40     Tokyo     95  2023-04-05
idx_5     Eve   28     Seoul     88  2023-05-01

2. df.head(3): DataFrame의 상위 3개 행
       Name  Age      City  Score Enroll_Date
idx_1   Alice   25  New York     85  2023-0
idx_2     Bob   30    London     92  2023-02-20
idx_3 Charlie   35     Paris     78  2023-03-10

3. df.tail() (기본값 n=5): DataFrame의 하위 5개 행
      Name  Age    City  Score Enroll_Date
idx_6  Frank   32  London     75  2023-06-12
idx_7  Grace   29  New York     91  2023-07-01
idx_8  Heidi   31     Paris     80  2023-08-18
idx_9    Ivy   26     Tokyo     89  2023-09-25
idx_10  Jack   33     Seoul     77  2023-10-30

4. df.tail(2): DataFrame의 하위 2개 행
      Name  Age   City  Score Enroll_Date
idx_9    Ivy   26  Tokyo     89  2023-09-25
idx_10  Jack   33  Seoul     77  2023-10-30

5. 'Score' 컬럼의 상위 3개 값
idx_1    85
idx_2    92
idx_3    78
Name: Score, dtype: int64
```

**모범 사례 및 활용 팁**:
*   **초기 데이터 검증**: 데이터를 처음 로드했을 때 `df.head()`를 사용하여 데이터가 예상대로 로드되었는지, 컬럼 이름이 올바른지, 데이터가 깨지지 않았는지 등을 빠르게 확인하는 습관을 들이세요. 이는 데이터 전처리 과정에서 발생할 수 있는 오류를 조기에 발견하는 데 큰 도움이 됩니다.
*   **시계열/로그 데이터**: 시간 순서가 중요한 데이터(예: 주식 가격, 센서 로그)의 경우, `df.tail()`을 사용하여 가장 최근의 데이터가 올바르게 추가되었는지, 또는 특정 기간의 마지막 데이터가 예상대로인지 확인할 수 있습니다.
*   **데이터 정렬 확인**: 데이터를 특정 기준으로 정렬한 후 `head()`나 `tail()`을 사용하여 정렬이 제대로 적용되었는지 검증할 수 있습니다. 예를 들어, `df.sort_values('Score', ascending=False).head()`를 통해 가장 높은 점수를 가진 데이터를 확인할 수 있습니다.
*   **메모리 효율성**: 대용량 데이터셋의 경우, 전체 데이터를 출력하는 대신 `head()`나 `tail()`을 사용하여 메모리 부담 없이 데이터의 일부만 빠르게 확인하는 것이 효율적입니다.

**핵심 요약 (Key Takeaways)**:
*   `head()`와 `tail()`은 데이터 탐색의 **가장 기본적인 시작점**입니다.
*   `n` 파라미터를 통해 원하는 개수의 행을 유연하게 확인할 수 있습니다.
*   데이터의 **유효성, 구조, 초기 패턴**을 빠르게 파악하는 데 필수적입니다.
*   `DataFrame`뿐만 아니라 `Series` 객체에도 동일하게 적용 가능합니다.

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

print("---"" DataFrame 미리보기 ---")
print("앞에서부터 5개 미리 보기 (df.head()):
", df.head())

print("\n뒤에서부터 3개 미리 보기 (df.tail(3)):
", df.tail(3))

print("\n앞에서부터 6개 미리 보기 (df.head(6)):
", df.head(6))
```

**모범 사례 및 활용 팁**:
*   데이터를 처음 로드했을 때 항상 `df.head()`를 사용하여 데이터가 예상대로 로드되었는지, 컬럼 이름이 올바른지, 데이터가 깨지지 않았는지 등을 빠르게 확인하세요.
*   시계열 데이터와 같이 순서가 중요한 데이터의 경우, `df.tail()`을 사용하여 가장 최근의 데이터가 올바르게 추가되었는지 확인할 수 있습니다.

## 2. `shape`, `ndim`, `size`: DataFrame의 차원 정보 (Dimensionality and Size)

`DataFrame`의 `shape`, `ndim`, `size` 속성들은 데이터셋의 전체적인 크기와 차원 정보를 파악하는 데 사용됩니다. 이들은 데이터 탐색(EDA) 초기 단계에서 데이터의 규모를 이해하고, 메모리 사용량 및 연산 복잡성을 예측하는 데 필수적인 정보를 제공합니다.

**각 속성의 상세 설명**:

*   **`shape`**: `DataFrame`의 **차원(dimensions)**을 튜플 형태로 반환합니다. 반환되는 튜플은 `(행의 개수, 열의 개수)`로 구성됩니다.
    *   **활용**: 데이터셋의 전체적인 크기를 한눈에 파악할 수 있으며, 특히 대규모 데이터셋을 다룰 때 메모리 사용량이나 연산 시간을 예측하는 데 매우 중요합니다. 데이터프레임의 구조가 예상과 일치하는지 확인하는 데도 사용됩니다.
    *   **예시**: `(1000, 10)`은 1000개의 행과 10개의 열을 가진 데이터프레임을 의미합니다.

*   **`ndim`**: `DataFrame`의 **차원 수(number of dimensions)**를 반환합니다. `DataFrame`은 본질적으로 2차원(행과 열) 구조를 가지므로, 이 속성은 항상 `2`를 반환합니다.
    *   **활용**: `Series` (1차원)와 `DataFrame` (2차원)을 구분하거나, 데이터 구조의 차원적 특성을 명시적으로 확인할 때 사용될 수 있습니다.

*   **`size`**: `DataFrame`의 **총 요소(셀) 개수**를 반환합니다. 이는 `행의 개수 * 열의 개수`와 같습니다.
    *   **활용**: 데이터프레임 내의 전체 데이터 포인트 수를 파악할 때 유용합니다. 이는 데이터 처리량이나 저장 공간 요구 사항을 추정하는 데 도움이 됩니다.

**예시 코드 및 출력**:

```python
import pandas as pd
import numpy as np

# 예시 데이터셋 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 28],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul'],
    'Score': [85, 92, 78, 95, 88]
}
df = pd.DataFrame(data)

print("--- DataFrame의 차원 정보 ---")
print(f"DataFrame의 형태 (df.shape): {df.shape}")
print(f"DataFrame의 차원 수 (df.ndim): {df.ndim}")
print(f"DataFrame의 총 요소 개수 (df.size): {df.size}")

# df.shape를 활용하여 행과 열의 개수를 개별 변수에 할당
rows, cols = df.shape
print(f"\n행의 개수: {rows}")
print(f"열의 개수: {cols}")

# 조건문에서 df.shape 활용 예시
if df.shape[0] > 1000:
    print("\n이 DataFrame은 1000개 이상의 행을 가지고 있습니다.")
else:
    print("\n이 DataFrame은 1000개 이하의 행을 가지고 있습니다.")

# Series의 ndim 비교
s = df['Age']
print(f"\nSeries의 차원 수 (s.ndim): {s.ndim}")
```

**출력 예시**:
```
--- DataFrame의 차원 정보 ---
DataFrame의 형태 (df.shape): (5, 4)
DataFrame의 차원 수 (df.ndim): 2
DataFrame의 총 요소 개수 (df.size): 20

행의 개수: 5
열의 개수: 4

이 DataFrame은 1000개 이하의 행을 가지고 있습니다.

Series의 차원 수 (s.ndim): 1
```

**모범 사례 및 활용 팁**:
*   **데이터 규모 파악**: `df.shape`는 데이터셋의 크기를 빠르게 파악하여 메모리 사용량이나 처리 시간을 예측하는 데 가장 유용합니다. 특히 대용량 데이터를 다룰 때 필수적입니다.
*   **데이터 유효성 검사**: 특정 연산을 수행하기 전에 `df.shape`를 사용하여 데이터프레임이 예상하는 최소한의 행이나 열을 가지고 있는지 확인할 수 있습니다. 예를 들어, 특정 컬럼이 존재해야 하는 경우 `df.shape[1]` (열의 개수)를 확인하여 오류를 방지할 수 있습니다.
*   **반복문 및 조건문 활용**: `rows, cols = df.shape`와 같이 행과 열의 개수를 개별 변수에 할당하여 반복문이나 조건문 등에서 유연하게 활용할 수 있습니다. 이는 코드의 가독성을 높이고 동적인 처리를 가능하게 합니다.
*   **데이터 일관성 유지**: 여러 데이터프레임을 병합하거나 연결할 때, `shape`를 비교하여 예상치 못한 크기 불일치나 데이터 손실이 없는지 검증하는 데 활용할 수 있습니다.

**핵심 요약 (Key Takeaways)**:
*   `shape`, `ndim`, `size`는 `DataFrame`의 **기본적인 구조와 규모**를 파악하는 데 필수적인 속성입니다.
*   `shape`는 `(행, 열)` 튜플을 반환하여 데이터의 크기를 명확히 보여줍니다.
*   `ndim`은 `DataFrame`이 2차원임을 나타내며, `size`는 전체 요소의 개수를 제공합니다.
*   이 속성들은 데이터 유효성 검사, 메모리 예측, 그리고 동적인 데이터 처리에 광범위하게 활용됩니다.

## 3. `info()`: DataFrame의 간략한 정보 요약 (Concise DataFrame Summary)

`DataFrame`의 `info()` 메서드는 데이터셋의 간략하지만 핵심적인 정보를 출력합니다. 이 메서드는 각 컬럼의 데이터 타입(`Dtype`), Non-null 값의 개수(`Non-Null Count`), 메모리 사용량(`Memory Usage`), 그리고 인덱스 정보 등을 포함하여 데이터의 누락 여부와 타입을 빠르게 확인할 수 있도록 돕습니다. `head()`와 함께 데이터 탐색(EDA)의 핵심적인 첫 단계로, 데이터 전처리 계획을 세우는 데 매우 유용합니다.

**`info()` 함수가 제공하는 핵심 정보**:

1.  **`RangeIndex` 또는 Custom Index 정보**:
    *   `DataFrame`의 행 인덱스 범위와 스텝을 보여줍니다. 기본 정수형 인덱스(`RangeIndex`)인 경우 `RangeIndex: 0 to N-1` 형태로 표시되며, 사용자 정의 인덱스(예: 날짜, 문자열)인 경우 해당 인덱스의 타입과 개수를 표시합니다.
    *   **활용**: 데이터의 행 개수를 빠르게 파악하고, 인덱스가 예상대로 설정되었는지 확인합니다.

2.  **`Data columns (total N columns)`**:
    *   `DataFrame`에 포함된 전체 컬럼의 개수를 나타냅니다.
    *   **활용**: 데이터셋의 폭(컬럼 수)을 한눈에 파악합니다.

3.  **`Column`**:
    *   각 컬럼의 이름을 나열합니다.

4.  **`Non-Null Count`**:
    *   각 컬럼에 결측치(`NaN`, `None`, `NaT` 등)가 아닌 **유효한 데이터가 몇 개 있는지**를 보여줍니다.
    *   **활용**: 이 정보는 데이터셋 내의 **결측치 존재 여부와 그 비율**을 빠르게 파악하는 데 가장 중요합니다. 전체 행의 개수(`RangeIndex`의 `N`)와 `Non-Null Count`를 비교하여 결측치의 양을 직관적으로 알 수 있습니다. 결측치 처리 전략(제거, 대체 등)을 수립하는 데 필수적인 정보입니다.

5.  **`Dtype`**:
    *   각 컬럼이 어떤 **데이터 타입**을 가지는지 보여줍니다 (예: `int64`, `float64`, `object`, `category`, `datetime64[ns]`).
    *   **활용**: 올바른 데이터 타입은 메모리 효율성, 연산 성능, 그리고 특정 함수(예: 날짜/시간 함수)의 올바른 작동에 큰 영향을 미칩니다. 예를 들어, 숫자로 인식되어야 할 컬럼이 `object` 타입으로 되어 있다면, 이는 데이터에 문자열이나 특수 문자가 포함되어 있거나 Pandas가 타입을 제대로 추론하지 못한 경우이므로, 데이터 클리닝 및 타입 변환이 필요함을 시사합니다.

6.  **`Memory Usage`**:
    *   `DataFrame`이 사용하는 총 메모리 양을 나타냅니다.
    *   **활용**: 대용량 데이터셋을 다룰 때 메모리 부족 문제를 예측하고 해결하는 데 도움을 줍니다. `memory_usage='deep'` 옵션을 사용하면 `object` 타입(주로 문자열) 컬럼의 실제 메모리 사용량까지 정확하게 계산하여 더 신뢰할 수 있는 정보를 제공합니다. 이는 문자열 데이터가 많은 경우 실제 메모리 사용량이 훨씬 클 수 있음을 보여줍니다.

**예시 코드 및 출력**:

```python
import pandas as pd
import numpy as np

# 예시 데이터셋 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, np.nan, 40, 28], # 결측치 포함
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul'],
    'Score': [85.5, 90.0, 78.2, np.nan, 92.1], # 결측치 포함
    'Gender': pd.Categorical(['M', 'F', 'M', 'F', 'M']), # 범주형 컬럼
    'Enroll_Date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'])
}
df = pd.DataFrame(data)

print("---" DataFrame의 기본 구조 (df.info())) ---")
df.info()

print("\n---" DataFrame의 기본 구조 (df.info(memory_usage='deep')) ---")
df.info(memory_usage='deep')

# 추가 예시: object 타입에 숫자와 문자열이 섞인 경우
df_mixed = pd.DataFrame({'Mixed_Col': ['1', '2', '3', 'A', '5']})
print("\n---" 혼합된 타입 컬럼의 info() ---")
df_mixed.info()
```

**출력 예시**:
```
--- DataFrame의 기본 구조 (df.info()) ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype         
---  ------       --------------  -----
 0   Name         5 non-null      object        
 1   Age          4 non-null      float64       
 2   City         5 non-null      object        
 3   Score        4 non-null      float64       
 4   Gender       5 non-null      category      
 5   Enroll_Date  5 non-null      datetime64[ns]
dtypes: category(1), datetime64[ns](1), float64(2), object(2)
memory usage: 705.0 bytes

--- DataFrame의 기본 구조 (df.info(memory_usage='deep')) ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype         
---  ------       --------------  -----
 0   Name         5 non-null      object        
 1   Age          4 non-null      float64       
 2   City         5 non-null      object        
 3   Score        4 non-null      float64       
 4   Gender       5 non-null      category      
 5   Enroll_Date  5 non-null      datetime64[ns]
dtypes: category(1), datetime64[ns](1), float64(2), object(2)
memory usage: 1.2 KB

--- 혼합된 타입 컬럼의 info() ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
Data columns (total 1 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  -----
 0   Mixed_Col  5 non-null      object
dtypes: object(1)
memory usage: 133.0+ bytes
```

**모범 사례 및 활용 팁**:
*   **결측치 식별**: `info()`를 통해 각 컬럼의 `Non-Null Count`를 확인하여 전체 행 수와 비교함으로써 결측치가 있는 컬럼을 빠르게 식별하고, 결측치 처리 전략(예: `fillna()`, `dropna()`)을 수립하는 데 활용하세요.
*   **데이터 타입 검증 및 변환**:
    *   `Dtype`을 확인하여 숫자로 인식되어야 할 컬럼이 `object` 타입으로 되어있지는 않은지 확인하세요. 이는 데이터 로드 시 흔히 발생하는 문제이며, `pd.to_numeric()`, `astype()`, 또는 `pd.to_datetime()` 등으로 적절히 변환해야 합니다.
    *   `object` 타입 컬럼 중 고유한 값의 개수가 적고 반복되는 문자열 데이터(예: '성별', '지역')가 많다면, `category` 타입으로 변환하여 메모리를 최적화하고 연산 속도를 향상시킬 수 있습니다.
*   **메모리 최적화**: 대용량 데이터셋의 경우 `memory_usage='deep'` 옵션을 사용하여 정확한 메모리 사용량을 확인하고, 필요에 따라 데이터 타입을 최적화(예: `int64`를 `int32`나 `int16`으로, `float64`를 `float32`로, `object`를 `category`로)하여 메모리 효율성을 높일 수 있습니다.

**핵심 요약 (Key Takeaways)**:
*   `info()`는 `DataFrame`의 **구조, 데이터 타입, 결측치 현황, 메모리 사용량**을 한눈에 파악할 수 있는 강력한 도구입니다.
*   데이터 전처리 및 클리닝 전략 수립의 **가장 중요한 첫 단계** 중 하나입니다.
*   `Non-Null Count`와 `Dtype`을 통해 데이터의 품질과 적합성을 빠르게 진단할 수 있습니다.

## 4. `describe()`: 숫자형 컬럼 기술 통계 (Descriptive Statistics for Numerical Columns)

`DataFrame`의 `describe()` 메서드는 숫자형 컬럼에 대한 **기술 통계(Descriptive Statistics)**를 계산하여 출력합니다. 이 메서드는 데이터의 분포, 중심 경향성(평균, 중앙값), 퍼짐 정도(표준편차, 사분위수) 등을 파악하는 데 매우 유용하며, 데이터 탐색(EDA) 과정에서 이상치(outlier)나 데이터 입력 오류를 감지하는 데 큰 도움을 줍니다.

**`describe()` 함수가 제공하는 핵심 통계 정보**:

*   **`count`**: 해당 컬럼의 **유효한(Non-null) 데이터 개수**입니다. `info()`의 `Non-Null Count`와 동일하며, 결측치 여부를 다시 한번 확인할 수 있습니다.
*   **`mean`**: **평균값**입니다. 데이터의 중심 위치를 나타내는 가장 일반적인 척도입니다.
*   **`std`**: **표준편차(Standard Deviation)**입니다. 데이터가 평균으로부터 얼마나 퍼져있는지, 즉 데이터의 변동성(산포도)을 나타냅니다. 값이 클수록 데이터의 변동성이 큽니다.
*   **`min`**: **최솟값**입니다. 데이터 범위의 하한을 나타냅니다.
*   **`25%` (1사분위수, Q1)**: 데이터를 오름차순으로 정렬했을 때 하위 25% 지점의 값입니다. 데이터의 4분의 1 지점입니다.
*   **`50%` (중앙값, Median, 2사분위수, Q2)**: 데이터를 오름차순으로 정렬했을 때 중간 지점의 값입니다. 중앙값은 극단적인 값(이상치)에 덜 민감하여 데이터의 중심을 나타내는 데 평균보다 더 견고할 수 있습니다.
*   **`75%` (3사분위수, Q3)**: 데이터를 오름차순으로 정렬했을 때 상위 25% 지점의 값입니다. 데이터의 4분의 3 지점입니다.
*   **`max`**: **최댓값**입니다. 데이터 범위의 상한을 나타냅니다.

**예시 코드 및 출력**:

```python
import pandas as pd
import numpy as np

# 예시 데이터셋 생성
data = {
    'Age': [25, 30, 35, 40, 28, 32, 29, 31, 150], # 이상치 포함
    'Score': [85, 92, 78, 95, 88, 75, 91, 80, 10], # 이상치 포함
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul', 'London', 'New York', 'Paris', 'Busan'],
    'Grade': pd.Categorical(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'D'])
}
df = pd.DataFrame(data)

print("--- 숫자형 컬럼의 요약 통계 정보 (df.describe())" ---)
print(df.describe())

print("\n--- 모든 컬럼의 요약 통계 정보 (df.describe(include='all'))" ---)
print(df.describe(include='all'))

print("\n--- object 타입 컬럼의 요약 통계 정보 (df.describe(include=['object']))" ---)
print(df.describe(include=['object']))

print("\n--- categorical 타입 컬럼의 요약 통계 정보 (df.describe(include=['category']))" ---)
print(df.describe(include=['category']))

print("\n--- 숫자형 컬럼 제외한 요약 통계 정보 (df.describe(exclude=[np.number]))" ---)
print(df.describe(exclude=[np.number]))
```

**출력 예시**:
```
--- 숫자형 컬럼의 요약 통계 정보 (df.describe()) ---
           Age      Score
count   9.000000   9.000000
mean   49.000000  79.333333
std    39.874807  25.000000
min    25.000000  10.000000
25%    29.000000  78.000000
50%    31.000000  85.000000
75%    35.000000  91.000000
max   150.000000  95.000000

--- 모든 컬럼의 요약 통계 정보 (df.describe(include='all')) ---
             Age      Score      City Grade
count   9.000000   9.000000         9     9
unique       NaN        NaN         5     4
top          NaN        NaN  New York     A
freq         NaN        NaN         2     3
mean   49.000000  79.333333       NaN   NaN
std    39.874807  25.000000       NaN   NaN
min    25.000000  10.000000       NaN   NaN
25%    29.000000  78.000000       NaN   NaN
50%    31.000000  85.000000       NaN   NaN
75%    35.000000  91.000000       NaN   NaN
max   150.000000  95.000000       NaN   NaN

--- object 타입 컬럼의 요약 통계 정보 (df.describe(include=['object'])) ---
           City
count         9
unique        5
top    New York
freq          2

--- categorical 타입 컬럼의 요약 통계 정보 (df.describe(include=['category'])) ---
      Grade
count     9
unique    4
top       A
freq      3

--- 숫자형 컬럼 제외한 요약 통계 정보 (df.describe(exclude=[np.number])) ---
           City Grade
count         9     9
unique        5     4
top    New York     A
freq          2     3
```

**모범 사례 및 활용 팁**:
*   **데이터 분포 및 왜도 파악**: `mean`과 `50%`(중앙값)을 비교하여 데이터의 왜도(skewness)를 대략적으로 파악할 수 있습니다.
    *   `평균 ≈ 중앙값`: 데이터가 대칭적인 분포를 가질 가능성이 높습니다.
    *   `평균 > 중앙값`: 데이터가 오른쪽으로 긴 꼬리(양의 왜도)를 가질 가능성이 높습니다 (예: 소득 분포).
    *   `평균 < 중앙값`: 데이터가 왼쪽으로 긴 꼬리(음의 왜도)를 가질 가능성이 높습니다.
*   **이상치(Outlier) 감지**: `min`과 `max` 값을 확인하여 데이터의 범위를 파악하고, 예상 범위를 벗어나는 극단적인 값(이상치)이나 데이터 입력 오류가 있는지 검토하세요. 사분위수(25%, 75%)와 함께 사용하여 IQR(Interquartile Range)을 계산하고 이상치 기준을 설정할 수도 있습니다.
*   **데이터 변동성 확인**: `std` (표준편차) 값이 0에 가깝다면 해당 컬럼의 값이 거의 동일하다는 의미이므로, 해당 컬럼이 분석에 유의미한지 다시 한번 고려해 볼 수 있습니다. 값이 매우 크다면 데이터의 산포도가 넓다는 의미입니다.
*   **모든 컬럼에 적용**: `df.describe(include='all')`을 사용하여 숫자형이 아닌 컬럼(object, category 등)에 대한 통계 정보(`count`, `unique`, `top`, `freq`)도 함께 확인하여 데이터의 전반적인 특성을 파악하세요.
    *   `unique`: 고유 값의 개수.
    *   `top`: 가장 자주 나타나는 값.
    *   `freq`: `top` 값의 빈도.
*   **특정 타입만 선택**: `include=[list_of_dtypes]`나 `exclude=[list_of_dtypes]`를 사용하여 특정 데이터 타입의 컬럼에 대해서만 기술 통계를 확인할 수 있습니다. 이는 분석 목적에 따라 유연하게 활용됩니다.

**핵심 요약 (Key Takeaways)**:
*   `describe()`는 숫자형 컬럼의 **핵심 통계량**을 제공하여 데이터의 **중심 경향성, 퍼짐 정도, 범위**를 빠르게 이해할 수 있도록 돕습니다.
*   `include='all'` 옵션을 통해 **비숫자형 컬럼의 요약 정보**도 함께 확인할 수 있어 데이터의 전반적인 특성을 파악하는 데 유용합니다.
*   이상치, 데이터 입력 오류, 데이터 분포의 왜곡 등을 **초기에 감지**하는 데 필수적인 도구입니다.

## 5. `value_counts()`: 고유 값 빈도 계산

`Series`의 고유한 값들과 각 값의 등장 횟수를 계산하여 내림차순으로 반환합니다. 범주형 데이터나 문자열 데이터의 분포를 파악하는 데 매우 유용하며, 데이터의 불균형을 확인하거나 가장 빈번한 항목을 식별할 때 사용됩니다.

**주요 기능 및 활용 목적**:
*   **데이터 분포 파악**: 범주형 또는 이산형 데이터의 각 고유 값이 얼마나 자주 나타나는지 확인하여 데이터의 전반적인 분포를 이해합니다.
*   **데이터 불균형 확인**: 특정 범주의 데이터가 다른 범주에 비해 현저히 많거나 적은 경우(클래스 불균형)를 식별하여 모델 학습 시 발생할 수 있는 문제를 예측하고 대응합니다.
*   **이상치 및 오타 감지**: 예상치 못한 고유 값이나 오타(예: 'New York'과 'new york', 'Male'과 'male')를 발견하여 데이터 정제 필요성을 파시합니다.
*   **가장 빈번한 항목 식별**: 특정 컬럼에서 가장 많이 등장하는 값(최빈값)을 쉽게 찾아낼 수 있습니다.

**반환 값**:
`value_counts()`는 `Series` 객체를 반환합니다. 이 `Series`의 인덱스는 원본 `Series`의 고유한 값들이고, 값은 각 고유 값의 등장 횟수(빈도)입니다. 기본적으로 빈도에 따라 내림차순으로 정렬됩니다.

**메서드 상세 및 파라미터**:

*   **`normalize`** (`bool`, 기본값 `False`):
    *   `True`로 설정하면 각 고유 값의 빈도 대신 **정규화된 빈도(비율)**를 반환합니다. 즉, 각 고유 값의 개수를 전체 유효한 값의 개수로 나눈 값을 반환합니다. 이는 전체 데이터에서 해당 범주가 차지하는 비중을 파악하는 데 유용합니다.
*   **`sort`** (`bool`, 기본값 `True`):
    *   `True`로 설정하면 결과 `Series`를 빈도(값)에 따라 정렬합니다.
    *   `False`로 설정하면 빈도 순으로 정렬하지 않고, 원본 `Series`에서 고유 값이 나타난 순서(또는 내부 해시 순서)에 따라 정렬됩니다.
*   **`ascending`** (`bool`, 기본값 `False`):
    *   `True`로 설정하면 결과 `Series`를 오름차순으로 정렬합니다. `sort=True`일 때만 유효합니다.
*   **`dropna`** (`bool`, 기본값 `True`):
    *   `True`로 설정하면 결측치(`NaN`, `None`, `NaT`)를 결과에서 제외합니다.
    *   `False`로 설정하면 결측치도 하나의 고유 값으로 간주하여 빈도에 포함합니다.
*   **`bins`** (`int`, 선택 사항):
    *   숫자형 데이터에만 적용됩니다. 이 파라미터에 정수 값을 제공하면, `value_counts()`는 연속적인 숫자 데이터를 지정된 개수(`bins`)의 이산적인 구간(bin)으로 나누고 각 구간에 속하는 값들의 빈도를 계산합니다. 결과는 `IntervalIndex`를 인덱스로 갖는 `Series`가 됩니다.
*   **`include_lowest`** (`bool`, 기본값 `False`):
    *   `bins` 파라미터와 함께 사용될 때만 유효합니다. `True`로 설정하면 가장 낮은 구간(bin)의 왼쪽 경계값도 해당 구간에 포함시킵니다.

**예시 코드 및 출력**:

```python
import pandas as pd
import numpy as np

# 예시 데이터셋 생성
data = {
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul', 'London', 'New York', 'Paris', 'New York', 'Seoul', 'London'],
    'Grade': pd.Categorical(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'A', 'C', 'B']),
    'Age_Group': ['Adult', 'Youth', 'Adult', 'Adult', 'Youth', 'Adult', 'Youth', 'Adult', 'Adult', 'Youth', 'Adult'],
    'Score': [85, 92, 78, 95, 88, 75, 91, 80, 89, 77, 60] # 숫자형 데이터
}
df = pd.DataFrame(data)

print("--- 'City' 컬럼의 고유 값 빈도 (기본 내림차순) ---")
print(df['City'].value_counts())

print("
--- 'Grade' 컬럼의 고유 값 빈도 ---")
print(df['Grade'].value_counts())

# normalize=True: 정규화된 빈도 (비율) 계산
print("
--- 'City' 컬럼의 정규화된 빈도 (비율) ---")
print(df['City'].value_counts(normalize=True))

# sort=False: 빈도 순으로 정렬하지 않음 (원본 순서 또는 해시 순서)
print("
--- 'City' 컬럼의 정렬하지 않은 빈도 ---")
print(df['City'].value_counts(sort=False))

# ascending=True: 오름차순으로 정렬
print("
--- 'City' 컬럼의 오름차순 빈도 ---")
print(df['City'].value_counts(ascending=True))

# dropna=False: 결측치도 빈도에 포함
df_nan = pd.DataFrame({'col': ['A', 'B', 'A', np.nan, 'B', 'C', np.nan]})
print("
--- 결측치 포함 컬럼의 빈도 (dropna=False) ---")
print(df_nan['col'].value_counts(dropna=False))

# bins 파라미터 활용 (숫자형 데이터)
print("
--- 'Score' 컬럼을 3개의 구간으로 나눈 빈도 ---")
print(df['Score'].value_counts(bins=3))

# bins 파라미터와 include_lowest=True 활용
print("
--- 'Score' 컬럼을 3개의 구간으로 나누고 최저값 포함 ---")
print(df['Score'].value_counts(bins=3, include_lowest=True))
```

**모범 사례 및 활용 팁**:
*   **데이터 분포 시각화 전 단계**: `value_counts()`의 결과를 막대 그래프(bar plot)나 파이 차트(pie chart)로 시각화하기 전에 데이터의 분포를 빠르게 확인하는 데 사용합니다.
*   **데이터 불균형 처리**: 분류 모델 학습 시 클래스 불균형이 심한 경우, `value_counts()`로 이를 확인하고 오버샘플링(oversampling)이나 언더샘플링(undersampling)과 같은 기법을 적용할지 결정하는 데 활용합니다.
*   **데이터 정제 및 표준화**: `value_counts()`를 통해 'Male', 'male', 'M'과 같이 동일한 의미를 가지지만 다르게 입력된 값들을 찾아내고, 이를 하나의 표준화된 값으로 통일하는 데이터 클리닝 작업에 활용합니다.
*   **범주형 변수 분석**: 설문조사 응답, 제품 카테고리, 지역 코드 등 범주형 변수의 특성을 이해하는 데 필수적입니다.

**핵심 요약 (Key Takeaways)**:
*   `value_counts()`는 `Series`의 고유 값 빈도를 계산하여 데이터의 **분포, 불균형, 이상치/오타**를 빠르게 파악하는 데 최적화된 메서드입니다.
*   `normalize`, `sort`, `ascending`, `dropna`, `bins`, `include_lowest` 등 다양한 파라미터를 통해 **유연하게 빈도 계산 및 정렬 방식**을 제어할 수 있습니다.
*   특히 **범주형 데이터의 탐색적 데이터 분석(EDA)**에서 핵심적인 역할을 수행하며, 데이터 전처리 및 모델링 전략 수립에 중요한 통찰을 제공합니다.

## 6. `unique()` / `nunique()`: 고유 값 및 개수 확인

`Series` 내의 고유한 값들을 확인하거나 그 개수를 세는 데 사용되는 메서드입니다. `value_counts()`와 유사하지만, 빈도 정보 없이 고유 값 자체나 그 개수만 필요할 때 더 간결하게 사용할 수 있습니다.

`Series` 내의 고유한 값들을 확인하거나 그 개수를 세는 데 사용되는 메서드입니다. `value_counts()`와 유사하지만, 빈도 정보 없이 고유 값 자체나 그 개수만 필요할 때 더 간결하게 사용할 수 있습니다.

**주요 기능 및 활용 목적**:
*   **고유 값 목록 확인**: 특정 컬럼에 어떤 종류의 고유한 값들이 존재하는지 빠르게 파악합니다. 이는 데이터의 범주를 이해하거나, 예상치 못한 값이 포함되어 있는지 확인하는 데 유용합니다.
*   **카디널리티(Cardinality) 파악**: 컬럼의 고유 값 개수를 통해 해당 컬럼의 다양성 정도를 측정합니다. 카디널리티가 너무 높으면 범주형 변수로서의 활용도가 낮아질 수 있고, 너무 낮으면 분석적 가치가 없을 수 있습니다.
*   **데이터 일관성 검사**: `unique()`를 통해 데이터 입력 오류나 오타(예: 'Male', 'male', 'M' 등)를 발견하고 데이터 정제 계획을 수립하는 데 활용합니다.

**메서드 상세**:

*   **`unique()`**:
    *   `Series`의 모든 고유한 값들을 **NumPy 배열 형태**로 반환합니다.
    *   반환되는 배열에는 결측치(`NaN`, `None`, `NaT`)도 하나의 고유한 값으로 포함됩니다.
    *   **활용**: 컬럼에 존재하는 모든 고유한 범주를 직접 확인하고자 할 때 사용합니다.

*   **`nunique()`**:
    *   `Series`의 고유한 값들의 **개수**를 반환합니다.
    *   기본적으로 결측치(`NaN`, `None`, `NaT`)는 고유 값의 개수에 포함하지 않습니다.
    *   **활용**: 컬럼의 카디널리티를 빠르게 파악하고자 할 때 사용합니다.

**`nunique()` 파라미터**:

*   **`dropna`** (`bool`, 기본값 `True`):
    *   `True`로 설정하면 결측치(`NaN`)를 고유 값 개수 계산에서 제외합니다.
    *   `False`로 설정하면 결측치도 하나의 고유한 값으로 간주하여 개수에 포함합니다.

**예시 코드 및 출력**:

```python
import pandas as pd
import numpy as np

# 예시 Series 생성
s = pd.Series(['A', 'B', 'A', 'C', np.nan, 'B', 'A', 'D', 'C'])
s_num = pd.Series([1, 2, 1, 3, np.nan, 2, 1, 4, 3])

print("---" 고유 값 및 개수 확인 ---")
print(f"원본 문자열 Series:
{s}")
print(f"원본 숫자형 Series:
{s_num}")

print(f"
문자열 Series의 고유 값 (s.unique()): {s.unique()}")
print(f"문자열 Series의 고유 값 개수 (s.nunique()): {s.nunique()} # NaN은 기본적으로 제외")
print(f"문자열 Series의 고유 값 개수 (s.nunique(dropna=False)): {s.nunique(dropna=False)} # NaN 포함")

print(f"
숫자형 Series의 고유 값 (s_num.unique()): {s_num.unique()}")
print(f"숫자형 Series의 고유 값 개수 (s_num.nunique()): {s_num.nunique()} # NaN은 기본적으로 제외")
print(f"숫자형 Series의 고유 값 개수 (s_num.nunique(dropna=False)): {s_num.nunique(dropna=False)} # NaN 포함")

# DataFrame의 특정 컬럼에 적용
df_unique = pd.DataFrame({
    'Product': ['Apple', 'Banana', 'Apple', 'Orange', 'Banana', 'Grape', 'Apple'],
    'Region': ['East', 'West', 'East', 'North', 'West', 'South', np.nan],
    'Price': [10, 20, 10, 30, 20, 40, 10]
})
print(f"
--- DataFrame의 'Product' 컬럼 고유 값 ---")
print(df_unique['Product'].unique())
print(f"--- DataFrame의 'Product' 컬럼 고유 값 개수 ---")
print(df_unique['Product'].nunique())

print(f"
--- DataFrame의 'Region' 컬럼 고유 값 ---")
print(df_unique['Region'].unique())
print(f"--- DataFrame의 'Region' 컬럼 고유 값 개수 (NaN 제외) ---")
print(df_unique['Region'].nunique())
print(f"--- DataFrame의 'Region' 컬럼 고유 값 개수 (NaN 포함) ---")
print(df_unique['Region'].nunique(dropna=False))

print(f"
--- DataFrame의 'Price' 컬럼 고유 값 ---")
print(df_unique['Price'].unique())
print(f"--- DataFrame의 'Price' 컬럼 고유 값 개수 ---")
print(df_unique['Price'].nunique())
```

**모범 사례 및 활용 팁**:
*   **데이터 탐색 초기 단계**: `unique()`를 사용하여 범주형 컬럼의 모든 가능한 값을 빠르게 확인하고, 예상치 못한 값이나 오타가 있는지 검사합니다.
*   **카디널리티 분석**: `nunique()`를 사용하여 컬럼의 카디널리티를 파악합니다.
    *   **낮은 카디널리티**: 고유 값의 개수가 적은 경우 (예: 성별, 지역). 범주형 변수로 적합하며, 원-핫 인코딩(One-Hot Encoding)이나 레이블 인코딩(Label Encoding)을 고려할 수 있습니다.
    *   **높은 카디널리티**: 고유 값의 개수가 많은 경우 (예: 사용자 ID, 이메일 주소). 범주형 변수로서 직접 사용하기 어려울 수 있으며, 다른 처리 방법(예: 해싱, 임베딩)을 고려하거나 해당 컬럼의 분석적 가치를 재평가해야 합니다.
*   **결측치 처리 전후 확인**: `nunique(dropna=False)`를 사용하여 결측치가 고유 값으로 포함되는지 확인하고, 결측치 처리 전후의 고유 값 개수 변화를 추적할 수 있습니다.
*   **데이터 타입 변환 결정**: `unique()`를 통해 컬럼의 내용이 모두 숫자로 구성되어 있지만 `object` 타입으로 되어 있는 경우를 발견하고, `pd.to_numeric()` 등으로 타입 변환을 결정하는 데 도움을 줍니다.

**핵심 요약 (Key Takeaways)**:
*   `unique()`는 `Series`의 **모든 고유한 값들을 NumPy 배열로 반환**하며, 결측치도 포함합니다.
*   `nunique()`는 `Series`의 **고유한 값들의 개수를 반환**하며, 기본적으로 결측치를 제외합니다 (`dropna=False`로 포함 가능).
*   이 두 메서드는 데이터의 **범주 파악, 카디널리티 분석, 데이터 일관성 검사** 등 탐색적 데이터 분석(EDA)의 중요한 단계에서 활용됩니다.
*   `value_counts()`가 각 고유 값의 빈도까지 제공하는 반면, `unique()`와 `nunique()`는 **고유 값 자체 또는 그 개수**에 집중하여 더 간결한 정보를 제공합니다.


```python
import pandas as pd
import numpy as np

s = pd.Series(['A', 'B', 'A', 'C', np.nan, 'B', 'A'])

print("---"" 고유 값 및 개수 확인 ---")
print(f"원본 Series:
{s}")

print(f"\n고유 값 (s.unique()): {s.unique()}")
print(f"고유 값 개수 (s.nunique()): {s.nunique()} # NaN은 기본적으로 제외")

print(f"고유 값 개수 (s.nunique(dropna=False)): {s.nunique(dropna=False)} # NaN 포함")

# DataFrame의 특정 컬럼에 적용
df_unique = pd.DataFrame({
    'Product': ['Apple', 'Banana', 'Apple', 'Orange', 'Banana'],
    'Region': ['East', 'West', 'East', 'North', 'West']
})
print(f"\n--- DataFrame의 'Product' 컬럼 고유 값 ---
{df_unique['Product'].unique()}")
print(f"--- DataFrame의 'Region' 컬럼 고유 값 개수 ---
{df_unique['Region'].nunique()}")
```

**모범 사례 및 활용 팁**:
*   `unique()`는 컬럼에 예상치 못한 값이 있는지 빠르게 확인하여 데이터 정제에 활용할 수 있습니다.
*   `nunique()`는 컬럼의 카디널리티(cardinality, 고유 값의 개수)를 파악하는 데 유용합니다. 카디널리티가 너무 높으면 범주형으로 처리하기 어렵고, 너무 낮으면 분석에 유의미하지 않을 수 있습니다.

### 7. `isnull()` / `notnull()`: 결측치 확인

(자세한 내용은 [13_결측치_처리.md - 결측치 탐지 섹션](./13_결측치_처리.md#2-결측치-탐지)을 참조하세요.)

`DataFrame` 또는 `Series`의 각 요소가 결측치(`NaN`, `None`, `NaT` 등)인지 여부를 불리언 `DataFrame` 또는 `Series`로 반환합니다. `isnull()`은 결측치이면 `True`, 아니면 `False`를 반환하며, `notnull()`은 그 반대입니다. 이들은 데이터 전처리 과정에서 결측치를 식별하고 처리하는 데 필수적인 첫 단계로 사용됩니다.

**주요 기능 및 활용 목적**:
*   **결측치 식별**: 데이터셋 내에 어떤 값이 누락되었는지 정확하게 파악합니다.
*   **결측치 위치 확인**: 특정 행이나 열에 결측치가 존재하는지 시각적으로 또는 프로그래밍 방식으로 확인합니다.
*   **결측치 개수 파악**: 각 컬럼별 또는 전체 데이터셋의 결측치 총 개수를 계산하여 결측치 처리 전략 수립의 기초 자료로 활용합니다.
*   **데이터 필터링**: 결측치가 있는 행이나 열을 제거하거나, 반대로 결측치가 없는 유효한 데이터만 추출하는 데 사용됩니다.

**메서드 상세**:
*   **`isnull()`**: `DataFrame` 또는 `Series`의 각 요소가 결측치(missing value)이면 `True`, 아니면 `False`를 반환합니다.
*   **`notnull()`**: `DataFrame` 또는 `Series`의 각 요소가 결측치가 아니면 `True`, 결측치이면 `False`를 반환합니다. `isnull()`의 역(inverse) 연산입니다.

**예시 코드 및 출력**:

```python
import pandas as pd
import numpy as np

# 예시 DataFrame 생성
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [6, np.nan, 8, 9, np.nan],
    'C': [11, 12, 13, 14, 15],
    'D': [np.nan, np.nan, np.nan, np.nan, np.nan] # 모든 값이 결측치인 컬럼
}
df = pd.DataFrame(data)

print("--- 원본 DataFrame ---")
print(df)

print("
--- df.isnull(): 각 요소의 결측치 여부 확인 (True: 결측치) ---")
print(df.isnull())

print("
--- df.notnull(): 각 요소의 결측치 아님 여부 확인 (True: 결측치 아님) ---")
print(df.notnull())

# --------------------------------------------------------------------
# 결측치 개수 확인 (가장 흔하게 사용되는 패턴)
# --------------------------------------------------------------------

# 컬럼별 결측치 개수: True는 1, False는 0으로 간주되므로 sum()을 사용하면 개수를 얻을 수 있습니다.
print("
--- df.isnull().sum(): 컬럼별 결측치 개수 ---")
print(df.isnull().sum())

# 전체 결측치 개수: sum()을 두 번 사용하여 DataFrame 전체의 결측치 개수를 얻습니다.
print(f"
--- df.isnull().sum().sum(): DataFrame 전체 결측치 개수 ---")
print(df.isnull().sum().sum())

# --------------------------------------------------------------------
# 결측치가 있는 행/열 필터링
# --------------------------------------------------------------------

# df.isnull().any(axis=1): 각 행에 결측치가 하나라도 있는지 확인 (Series 반환)
print("
--- df.isnull().any(axis=1): 각 행에 결측치가 하나라도 있는지 여부 ---")
print(df.isnull().any(axis=1))

# df[df.isnull().any(axis=1)]: 결측치가 하나라도 있는 행만 필터링
print("
--- df[df.isnull().any(axis=1)]: 결측치가 하나라도 있는 행 ---")
print(df[df.isnull().any(axis=1)])

# df.isnull().all(axis=1): 각 행의 모든 값이 결측치인지 확인 (Series 반환)
print("
--- df.isnull().all(axis=1): 각 행의 모든 값이 결측치인지 여부 ---")
print(df.isnull().all(axis=1))

# df.isnull().any(axis=0): 각 컬럼에 결측치가 하나라도 있는지 확인 (Series 반환)
print("
--- df.isnull().any(axis=0): 각 컬럼에 결측치가 하나라도 있는지 여부 ---")
print(df.isnull().any(axis=0))

# df.loc[:, df.isnull().any(axis=0)]: 결측치가 하나라도 있는 컬럼만 필터링
print("
--- df.loc[:, df.isnull().any(axis=0)]: 결측치가 하나라도 있는 컬럼 ---")
print(df.loc[:, df.isnull().any(axis=0)])

# df.isnull().all(axis=0): 각 컬럼의 모든 값이 결측치인지 확인 (Series 반환)
print("
--- df.isnull().all(axis=0): 각 컬럼의 모든 값이 결측치인지 여부 ---")
print(df.isnull().all(axis=0))

# df.loc[:, df.isnull().all(axis=0)]: 모든 값이 결측치인 컬럼만 필터링
print("
--- df.loc[:, df.isnull().all(axis=0)]: 모든 값이 결측치인 컬럼 ---")
print(df.loc[:, df.isnull().all(axis=0)])
```

**출력 예시 (위 코드 실행 시 예상되는 출력)**:
```
--- 원본 DataFrame ---
     A    B   C   D
0  1.0  6.0  11 NaN
1  2.0  NaN  12 NaN
2  NaN  8.0  13 NaN
3  4.0  9.0  14 NaN
4  5.0  NaN  15 NaN

--- df.isnull(): 각 요소의 결측치 여부 확인 (True: 결측치) ---
       A      B      C     D
0  False  False  False  True
1  False   True  False  True
2   True  False  False  True
3  False  False  False  True
4  False   True  False  True

--- df.notnull(): 각 요소의 결측치 아님 여부 확인 (True: 결측치 아님) ---
      A      B     C      D
0  True   True  True  False
1  True  False  True  False
2  False   True  True  False
3  True   True  True  False
4  True  False  True  False

--- df.isnull().sum(): 컬럼별 결측치 개수 ---
A    1
B    2
C    0
D    5
dtype: int64

--- df.isnull().sum().sum(): DataFrame 전체 결측치 개수 ---
8

--- df.isnull().any(axis=1): 각 행에 결측치가 하나라도 있는지 여부 ---
0     True
1     True
2     True
3     True
4     True
dtype: bool

--- df[df.isnull().any(axis=1)]: 결측치가 하나라도 있는 행 ---
     A    B   C   D
0  1.0  6.0  11 NaN
1  2.0  NaN  12 NaN
2  NaN  8.0  13 NaN
3  4.0  9.0  14 NaN
4  5.0  NaN  15 NaN

--- df.isnull().all(axis=1): 각 행의 모든 값이 결측치인지 여부 ---
0    False
1    False
2    False
3    False
4    False
dtype: bool

--- df.isnull().any(axis=0): 각 컬럼에 결측치가 하나라도 있는지 여부 ---
A     True
B     True
C    False
D     True
dtype: bool

--- df.loc[:, df.isnull().any(axis=0)]: 결측치가 하나라도 있는 컬럼 ---
     A    B   D
0  1.0  6.0 NaN
1  2.0  NaN NaN
2  NaN  8.0 NaN
3  4.0  9.0 NaN
4  5.0  NaN NaN

--- df.isnull().all(axis=0): 각 컬럼의 모든 값이 결측치인지 여부 ---
A    False
B    False
C    False
D     True
dtype: bool

--- df.loc[:, df.isnull().all(axis=0)]: 모든 값이 결측치인 컬럼 ---
     D
0  NaN
1  NaN
2  NaN
3  NaN
4  NaN
```

**모범 사례 및 활용 팁**:
*   **초기 결측치 진단**: 데이터 로드 후 `df.isnull().sum()`을 사용하여 각 컬럼의 결측치 개수를 빠르게 파악하는 것이 가장 일반적이고 효과적인 초기 진단 방법입니다. 이를 통해 어떤 컬럼에 결측치가 많고, 어떤 컬럼은 없는지 한눈에 알 수 있습니다.
*   **결측치 비율 계산**: `df.isnull().sum() / len(df)`를 통해 각 컬럼의 결측치 비율을 계산하여, 결측치 처리의 우선순위를 정하거나 특정 컬럼을 제거할지 결정하는 데 활용할 수 있습니다.
*   **조건부 필터링**: `df[df['컬럼명'].isnull()]`과 같이 사용하여 특정 컬럼에 결측치가 있는 행만 추출하거나, `df.dropna()` 메서드를 사용하기 전에 어떤 데이터가 제거될지 미리 확인하는 데 유용합니다.
*   **시각화와 연계**: `isnull()`의 결과를 히트맵(heatmap) 등으로 시각화하여 데이터셋 전체의 결측치 패턴을 파악하는 데 활용할 수 있습니다. (예: `import seaborn as sns; sns.heatmap(df.isnull())`)

**핵심 요약 (Key Takeaways)**:
*   `isnull()`과 `notnull()`은 `DataFrame` 또는 `Series`의 각 요소가 **결측치인지 아닌지**를 불리언 값으로 반환하는 기본적인 메서드입니다.
*   `sum()`, `any()`, `all()` 등의 집계 함수와 함께 사용하여 **결측치의 개수를 파악**하거나 **결측치가 포함된 행/열을 필터링**하는 데 광범위하게 활용됩니다.
*   데이터 전처리 과정에서 **결측치 처리 전략(제거, 대체 등)을 수립**하는 데 필수적인 정보를 제공합니다.

## 8. `dtypes` 속성: 컬럼별 데이터 타입 확인

`DataFrame`의 `dtypes` 속성은 각 컬럼에 대한 데이터 타입(`dtype`)을 `Series` 형태로 반환합니다. 이 속성은 데이터 전처리 과정에서 컬럼의 타입을 확인하고 필요에 따라 적절히 변환하는 데 매우 중요합니다. 올바른 데이터 타입은 메모리 효율성, 연산 성능, 그리고 특정 함수(예: 날짜/시간 함수)의 올바른 작동에 필수적입니다.

**`dtypes` 속성이 중요한 이유**:
*   **메모리 효율성**: 데이터 타입에 따라 메모리 사용량이 크게 달라집니다. 예를 들어, 작은 정수만 포함하는 컬럼에 `int64` 대신 `int8`이나 `int16`을 사용하면 메모리를 절약할 수 있습니다. 문자열이 적은 범주형 데이터는 `object` 대신 `category` 타입으로 변환하여 메모리를 크게 줄일 수 있습니다.
*   **연산 성능**: 올바른 데이터 타입은 연산 속도를 향상시킵니다. 예를 들어, 숫자형 연산은 `object` 타입의 문자열 숫자보다 `int`나 `float` 타입에서 훨씬 빠르게 수행됩니다.
*   **데이터 무결성 및 정확성**: 데이터 타입은 컬럼에 저장될 수 있는 값의 종류를 정의합니다. 예를 들어, 숫자형 컬럼에 문자열이 포함되면 오류가 발생하거나 예상치 못한 결과가 나올 수 있습니다.
*   **함수 호환성**: 특정 Pandas 함수나 라이브러리는 특정 데이터 타입에서만 올바르게 작동합니다. 예를 들어, 날짜/시간 연산은 `datetime64[ns]` 타입에서만 가능합니다.

**주요 `dtype` 종류 및 특징**:
*   **`object`**:
    *   가장 유연한 타입으로, 주로 문자열(string) 데이터를 나타냅니다. 하지만 숫자, 불리언, 리스트 등 다양한 Python 객체가 혼합되어 있을 수도 있습니다.
    *   **주의**: 숫자로만 구성된 컬럼이라도 따옴표로 묶여있거나(예: `'123'`), 결측치(`NaN`)가 포함된 숫자형 컬럼이 `int` 타입으로 로드될 수 없을 때 `object` 타입으로 로드되는 경우가 많습니다. 이 경우 명시적인 타입 변환이 필요합니다.
*   **`int64`, `int32`, `int16`, `int8`**:
    *   정수형 데이터. 숫자의 크기에 따라 비트 수가 달라지며, 더 작은 비트 수를 사용하면 메모리를 절약할 수 있습니다. (예: `int8`은 -128에서 127까지의 정수 표현)
*   **`float64`, `float32`**:
    *   부동소수점(실수) 데이터. `float32`는 `float64`보다 메모리를 덜 사용하지만 정밀도가 낮습니다.
*   **`bool`**:
    *   불리언(True/False) 데이터.
*   **`datetime64[ns]`**:
    *   날짜 및 시간 데이터. `pd.to_datetime()` 함수를 통해 문자열이나 숫자형 데이터를 이 타입으로 변환할 수 있습니다. 시계열 분석에 필수적입니다.
*   **`category`**:
    *   범주형 데이터. 고유한 값의 수가 적고 반복되는 문자열 데이터(예: 성별, 지역, 등급)에 대해 메모리 효율적입니다. Pandas 내부적으로 정수형 코드로 매핑하여 저장하므로, `object` 타입보다 메모리 사용량이 훨씬 적고 연산 속도도 빠릅니다.

**예시 코드 및 출력**:

```python
import pandas as pd
import numpy as np

# 예시 DataFrame 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 28],
    'Score': [85.5, 90.0, 78.2, 95.1, 88.7],
    'Is_Student': [True, False, True, False, True],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul'],
    'Enroll_Date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
    'Grade': ['A', 'B', 'A', 'C', 'B'] # 범주형으로 변환할 컬럼
}
df = pd.DataFrame(data)

print("--- 초기 DataFrame의 컬럼별 데이터 타입 (df.dtypes) ---")
print(df.dtypes)

# 데이터 타입 변환 예시
# 'Enroll_Date' 컬럼을 datetime 타입으로 변환
df['Enroll_Date'] = pd.to_datetime(df['Enroll_Date'])

# 'Age' 컬럼을 더 작은 정수형 타입으로 변환 (메모리 최적화)
df['Age'] = df['Age'].astype('int8')

# 'Grade' 컬럼을 category 타입으로 변환 (메모리 최적화 및 범주형 연산 용이)
df['Grade'] = df['Grade'].astype('category')

# 'Score' 컬럼을 float32 타입으로 변환 (정밀도 유지하며 메모리 최적화)
df['Score'] = df['Score'].astype('float32')

print("
--- 데이터 타입 변환 후 DataFrame의 컬럼별 데이터 타입 ---")
print(df.dtypes)

# object 타입 컬럼에 숫자가 문자열로 섞여있는 경우
df_mixed_num_str = pd.DataFrame({'ID': [', '102', '103A', '104']})
print("
--- 문자열 숫자가 섞인 object 타입 컬럼의 초기 dtypes ---")
print(df_mixed_num_str.dtypes)

# pd.to_numeric을 사용하여 숫자형으로 변환 (errors='coerce'로 변환 불가능한 값은 NaN으로 처리)
df_mixed_num_str['ID_numeric'] = pd.to_numeric(df_mixed_num_str['ID'], errors='coerce')
print("
--- pd.to_numeric 변환 후 dtypes ---")
print(df_mixed_num_str.dtypes)
print(df_mixed_num_str)
```

**출력 예시 (위 코드 실행 시 예상되는 출력)**:
```
--- 초기 DataFrame의 컬럼별 데이터 타입 (df.dtypes) ---
Name           object
Age             int64
Score         float64
Is_Student       bool
City           object
Enroll_Date    object
Grade          object
dtype: object

--- 데이터 타입 변환 후 DataFrame의 컬럼별 데이터 타입 ---
Name                   object
Age                      int8
Score                 float32
Is_Student               bool
City                   object
Enroll_Date    datetime64[ns]
Grade                category
dtype: object

--- 문자열 숫자가 섞인 object 타입 컬럼의 초기 dtypes ---
ID    object
dtype: object

--- pd.to_numeric 변환 후 dtypes ---
ID             object
ID_numeric    float64
dtype: object
     ID  ID_numeric
0         0
1   102       102.0
2  103A         NaN
3   104       104.0
```

**모범 사례 및 활용 팁**:
*   **데이터 로드 직후 확인**: 데이터를 로드한 후 가장 먼저 `df.dtypes`를 확인하여 각 컬럼의 데이터 타입이 예상과 일치하는지 확인하는 습관을 들이세요. `df.info()`와 함께 사용하면 더욱 효과적입니다.
*   **`object` 타입 주의**: 특히 `object` 타입 컬럼에 주의를 기울이세요.
    *   **숫자형으로 변환 필요**: 숫자로 인식되어야 할 컬럼이 `object` 타입으로 되어 있다면, 이는 데이터에 문자열이나 특수 문자가 포함되어 있거나, Pandas가 타입을 제대로 추론하지 못한 경우입니다. 이 경우 `astype('int')`, `astype('float')` 또는 `pd.to_numeric()` (특히 `errors='coerce'` 옵션과 함께) 등으로 변환해야 합니다.
    *   **날짜/시간으로 변환 필요**: 날짜/시간 정보가 `object` 타입으로 되어 있다면 `pd.to_datetime()`을 사용하여 `datetime64[ns]` 타입으로 변환해야 날짜/시간 연산을 수행할 수 있습니다.
    *   **범주형으로 변환 고려**: `object` 타입 컬럼 중 고유한 값의 개수가 적고 반복되는 문자열 데이터(저카디널리티)가 많다면, `astype('category')`로 변환하여 메모리를 최적화하고 연산 속도를 향상시킬 수 있습니다.
*   **메모리 최적화**: 대용량 데이터셋을 다룰 때는 `int64`를 `int32`나 `int16`으로, `float64`를 `float32`로, `object`를 `category`로 변환하는 등 데이터 타입을 최적화하여 메모리 사용량을 줄이는 것을 적극적으로 고려하세요. 이는 `df.info(memory_usage='deep')`을 통해 메모리 사용량 변화를 확인하며 진행할 수 있습니다.

**핵심 요약 (Key Takeaways)**:
*   `dtypes` 속성은 `DataFrame`의 각 컬럼에 대한 **데이터 타입 정보를 `Series` 형태로 제공**합니다.
*   데이터 타입은 **메모리 효율성, 연산 성능, 데이터 무결성, 함수 호환성**에 직접적인 영향을 미치므로, 데이터 전처리 과정에서 반드시 확인하고 필요에 따라 적절히 변환해야 합니다.
*   특히 `object` 타입 컬럼은 실제 데이터 내용에 따라 **숫자, 날짜/시간, 범주형** 등으로 명시적으로 변환해야 하는 경우가 많습니다.
*   `astype()`, `pd.to_numeric()`, `pd.to_datetime()` 등의 함수를 사용하여 데이터 타입을 변환할 수 있습니다.

## 9. `index` 및 `columns` 속성: 인덱스/컬럼 레이블 확인

`DataFrame`의 `index` 및 `columns` 속성은 각각 **행 인덱스(row index)**와 **컬럼(열) 레이블(column labels)**에 직접 접근할 수 있도록 해줍니다. 이 두 속성은 Pandas의 핵심 객체인 `Index` 타입으로 반환되며, 데이터의 구조를 이해하고 특정 행/열을 선택하거나 이름을 변경하는 데 매우 중요합니다.

**`Index` 객체의 특징**:
*   **유일성(Uniqueness)**: `Index` 객체 내의 레이블은 일반적으로 유일해야 하지만, Pandas는 중복된 레이블도 허용합니다. 그러나 유일한 레이블을 사용하는 것이 데이터 접근 및 관리에 더 효율적입니다.
*   **불변성(Immutability)**: `Index` 객체는 생성된 후에는 개별 요소를 직접 변경할 수 없습니다. 이는 데이터의 무결성을 보장하고, 해시 기반 연산의 안정성을 높이는 데 기여합니다. 만약 인덱스나 컬럼 레이블을 변경하려면 새로운 `Index` 객체를 생성하여 할당하거나, `rename()`과 같은 메서드를 사용해야 합니다.
*   **다양한 데이터 타입 지원**: 숫자, 문자열, 날짜/시간 등 다양한 데이터 타입을 인덱스 레이블로 사용할 수 있습니다.
*   **효율적인 데이터 접근**: `Index` 객체는 내부적으로 해시 테이블과 유사한 구조를 사용하여 빠른 데이터 검색 및 정렬을 가능하게 합니다.

**각 속성의 상세 설명**:

*   **`df.index`**: `DataFrame`의 **행 인덱스 객체**를 반환합니다. 이는 각 행을 고유하게 식별하는 레이블들의 집합입니다. 기본적으로 `RangeIndex` (0부터 시작하는 정수 인덱스)가 사용되지만, 데이터를 로드하거나 생성할 때 사용자 정의 인덱스(예: 날짜, ID, 이름)를 지정할 수 있습니다.
    *   **활용**: 특정 행을 선택하거나, 데이터프레임을 병합할 때 기준이 되는 키로 사용됩니다.

*   **`df.columns`**: `DataFrame`의 **컬럼(열) 레이블 객체**를 반환합니다. 이는 각 컬럼을 고유하게 식별하는 이름들의 집합입니다.
    *   **활용**: 특정 컬럼을 선택하거나, 컬럼 이름을 변경하거나, 데이터프레임의 구조를 파악하는 데 사용됩니다.

**예시 코드 및 출력**:

```python
import pandas as pd
import numpy as np

# 예시 DataFrame 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 28],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Seoul'],
    'Score': [85, 92, 78, 95, 88]
}
# 사용자 정의 인덱스를 사용하여 DataFrame 생성
df = pd.DataFrame(data, index=[f'ID_{i}' for i in range(1, 6)])

print("--- DataFrame의 인덱스 및 컬럼 레이블 확인 ---")

print("\n1. DataFrame의 행 인덱스 (df.index):")
print(df.index)
print(f"  - 타입: {type(df.index)}")
print(f"  - 이름: {df.index.name}") # 인덱스 이름 확인

print("\n2. DataFrame의 컬럼 레이블 (df.columns):")
print(df.columns)
print(f"  - 타입: {type(df.columns)}")
print(f"  - 이름: {df.columns.name}") # 컬럼 이름 확인

print("\n3. 인덱스/컬럼을 리스트로 변환하여 활용:")
print(f"  - 행 인덱스 리스트: {df.index.tolist()}")
print(f"  - 컬럼 레이블 리스트: {df.columns.tolist()}")

print("\n4. 특정 인덱스/컬럼 레이블에 접근:")
print(f"  - 첫 번째 행 인덱스: {df.index[0]}")
print(f"  - 첫 번째 컬럼 레이블: {df.columns[0]}")

print("\n5. 특정 레이블이 인덱스/컬럼에 존재하는지 확인:")
print(f"  - 'ID_3'이 인덱스에 있는가? {'ID_3' in df.index}")
print(f"  - 'Score'가 컬럼에 있는가? {'Score' in df.columns}")
print(f"  - 'Gender'가 컬럼에 있는가? {'Gender' in df.columns}")

print("\n6. 인덱스/컬럼 이름 설정:")
df.index.name = 'Student_ID'
df.columns.name = 'Student_Info'
print(f"  - 인덱스 이름 설정 후: {df.index.name}")
print(f"  - 컬럼 이름 설정 후: {df.columns.name}")
print("\n  - 이름 설정 후 DataFrame:")
print(df)

print("\n7. 인덱스/컬럼 레이블 변경 (df.rename() 사용 권장):")
# df.rename()을 사용하여 'Age' 컬럼 이름을 'Student_Age'로 변경
df_renamed = df.rename(columns={'Age': 'Student_Age'})
print("\n  - 'Age' 컬럼 이름을 'Student_Age'로 변경 후 (df.rename):")
print(df_renamed.columns)

# 인덱스 레이블 변경 예시
df_renamed_index = df.rename(index={'ID_1': 'First_Student'})
print("\n  - 'ID_1' 인덱스 이름을 'First_Student'로 변경 후 (df.rename):")
print(df_renamed_index.index)

print("\n8. 인덱스 객체의 불변성 확인 (오류 발생 예시):")
try:
    # df.index[0] = 'New_ID_1' # Index 객체는 불변이므로 이 코드는 오류를 발생시킵니다.
    print("  - df.index[0] = 'New_ID_1' 시도 (오류 예상):")
    df.index[0] = 'New_ID_1'
except TypeError as e:
    print(f"  - TypeError 발생: {e}")
    print("    (Index 객체는 불변(immutable)이므로 개별 요소를 직접 변경할 수 없습니다.)")

# 전체 컬럼 레이블을 한 번에 변경하는 예시 (길이가 일치해야 함)
original_columns = df.columns.tolist()
new_columns = ['Full_Name', 'Years_Old', 'Location', 'Exam_Score', 'Extra_Col'] # 예시로 컬럼 개수를 맞춤
if len(new_columns) == len(original_columns):
    df.columns = new_columns
    print("\n9. 전체 컬럼 레이블 직접 할당 후:")
    print(df.columns)
else:
    print("\n9. 전체 컬럼 레이블 직접 할당 실패: 새로운 리스트의 길이가 기존 컬럼 개수와 일치하지 않습니다.")

```

**출력 예시**:
```
--- DataFrame의 인덱스 및 컬럼 레이블 확인 ---

1. DataFrame의 행 인덱스 (df.index):
Index(['ID_1', 'ID_2', 'ID_3', 'ID_4', 'ID_5'], dtype='object')
  - 타입: <class 'pandas.core.indexes.base.Index'>
  - 이름: None

2. DataFrame의 컬럼 레이블 (df.columns):
Index(['Name', 'Age', 'City', 'Score'], dtype='object')
  - 타입: <class 'pandas.core.indexes.base.Index'>
  - 이름: None

3. 인덱스/컬럼을 리스트로 변환하여 활용:
  - 행 인덱스 리스트: ['ID_1', 'ID_2', 'ID_3', 'ID_4', 'ID_5']
  - 컬럼 레이블 리스트: ['Name', 'Age', 'City', 'Score']

4. 특정 인덱스/컬럼 레이블에 접근:
  - 첫 번째 행 인덱스: ID_1
  - 첫 번째 컬럼 레이블: Name

5. 특정 레이블이 인덱스/컬럼에 존재하는지 확인:
  - 'ID_3'이 인덱스에 있는가? True
  - 'Score'가 컬럼에 있는가? True
  - 'Gender'가 컬럼에 있는가? False

6. 인덱스/컬럼 이름 설정:
  - 인덱스 이름 설정 후: Student_ID
  - 컬럼 이름 설정 후: Student_Info

  - 이름 설정 후 DataFrame:
Student_Info     Name  Age      City  Score
Student_ID                                
ID_1            Alice   25  New York     85
ID_2              Bob   30    London     92
ID_3          Charlie   35     Paris     78
ID_4            David   40     Tokyo     95
ID_5              Eve   28     Seoul     88

7. 인덱스/컬럼 레이블 변경 (df.rename() 사용 권장):

  - 'Age' 컬럼 이름을 'Student_Age'로 변경 후 (df.rename):
Index(['Name', 'Student_Age', 'City', 'Score'], dtype='object', name='Student_Info')

  - 'ID_1' 인덱스 이름을 'First_Student'로 변경 후 (df.rename):
Index(['First_Student', 'ID_2', 'ID_3', 'ID_4', 'ID_5'], dtype='object', name='Student_ID')

8. 인덱스 객체의 불변성 확인 (오류 발생 예시):
  - TypeError 발생: 'Index' object does not support item assignment
    (Index 객체는 불변(immutable)이므로 개별 요소를 직접 변경할 수 없습니다.)

9. 전체 컬럼 레이블 직접 할당 후:
Index(['Full_Name', 'Years_Old', 'Location', 'Exam_Score'], dtype='object', name='Student_Info')
```

**모범 사례 및 활용 팁**:
*   **데이터 구조 파악**: `df.index`와 `df.columns`를 사용하여 데이터의 행과 열의 레이블을 빠르게 확인하고, 예상치 못한 인덱스나 컬럼 이름이 있는지 검사합니다. 이는 데이터 로드 후 가장 먼저 수행해야 할 작업 중 하나입니다.
*   **레이블 존재 여부 확인**: `label in df.index` 또는 `label in df.columns`와 같은 구문을 사용하여 특정 레이블이 데이터프레임에 존재하는지 효율적으로 확인할 수 있습니다.
*   **레이블 변경**:
    *   **부분 변경**: 특정 인덱스나 컬럼 레이블만 변경해야 할 경우, `df.rename()` 메서드를 사용하는 것이 가장 안전하고 권장되는 방법입니다. `rename()`은 원본 `DataFrame`을 변경하지 않고 새로운 `DataFrame`을 반환하므로, `inplace=True` 옵션을 사용하지 않는 한 원본 데이터의 무결성을 유지합니다.
    *   **전체 변경**: 모든 컬럼 레이블을 한 번에 새로운 리스트로 교체해야 할 경우에만 `df.columns = [...]`와 같이 직접 할당하는 방법을 사용합니다. 이때 할당하는 리스트의 길이가 기존 컬럼의 개수와 정확히 일치해야 합니다. 인덱스도 `df.index = [...]`와 같이 전체를 교체할 수 있습니다.
*   **인덱스/컬럼 이름 설정**: `df.index.name`이나 `df.columns.name` 속성을 사용하여 인덱스나 컬럼 축에 의미 있는 이름을 부여할 수 있습니다. 이는 특히 MultiIndex와 같은 복잡한 인덱스 구조에서 데이터의 가독성을 높이는 데 유용합니다.
*   **데이터 정렬 및 재색인**: `Index` 객체는 `sort_values()`, `reindex()` 등 데이터 정렬 및 재색인 작업의 기반이 됩니다.

**핵심 요약 (Key Takeaways)**:
*   `df.index`와 `df.columns`는 `DataFrame`의 **행 인덱스와 컬럼 레이블에 접근**하는 데 사용되는 핵심 속성입니다.
*   이들은 **불변(immutable)인 `Index` 객체**를 반환하며, 데이터의 무결성을 보장합니다.
*   데이터 구조 파악, 특정 레이블 존재 여부 확인, 그리고 레이블 변경(주로 `df.rename()` 사용)에 광범위하게 활용됩니다.
*   올바른 인덱스와 컬럼 관리는 **데이터 접근성, 가독성, 그리고 데이터프레임 연산의 효율성**을 높이는 데 필수적입니다.
