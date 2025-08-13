<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 `DataFrame`의 컬럼을 추가, 수정, 삭제, 이름 및 타입 변경 등 다양하게 조작하는 방법과, 다른 `DataFrame` 또는 `Series`와의 산술 연산을 수행하는 실용적인 방법을 다룹니다. 실제 코드 예제를 통해 데이터 전처리 및 분석 과정에서 필요한 데이터 조작 및 연산 능력을 강화하는 것을 목표로 합니다.

<h2>목차</h2>

- [1. DataFrame 컬럼 조작](#1-dataframe-컬럼-조작)
  - [1.1. 컬럼 추가 및 수정](#11-컬럼-추가-및-수정)
    - [1.1.1. 새 컬럼 추가](#111-새-컬럼-추가)
    - [1.1.2. 기존 컬럼 값 수정](#112-기존-컬럼-값-수정)
    - [1.1.3. `assign()` 메서드를 이용한 컬럼 추가](#113-assign-메서드를-이용한-컬럼-추가)
  - [1.2. 컬럼 삭제](#12-컬럼-삭제)
  - [1.3. 컬럼 이름 변경](#13-컬럼-이름-변경)
  - [1.4. 컬럼의 데이터 타입 변경](#14-컬럼의-데이터-타입-변경)
- [2. DataFrame 산술 연산](#2-dataframe-산술-연산)
  - [2.1. DataFrame과 스칼라의 연산](#21-dataframe과-스칼라의-연산)
  - [2.2. DataFrame 간의 연산](#22-dataframe-간의-연산)
  - [2.3. 비교 및 논리 연산](#23-비교-및-논리-연산)

--- 

## 1. DataFrame 컬럼 조작

`DataFrame`의 각 컬럼은 `Series` 객체입니다. 이 특성을 이해하면 컬럼 간의 연산을 통해 새로운 파생 컬럼을 만들거나 기존 데이터를 조작하는 작업을 매우 유연하게 수행할 수 있습니다.

### 1.1. 컬럼 추가 및 수정

#### 1.1.1. 새 컬럼 추가

- **기존 컬럼 연산을 통한 추가**

    가장 일반적인 방법으로, 기존 `DataFrame`의 하나 이상의 컬럼에 산술 연산을 적용하여 새로운 컬럼을 생성합니다.

    ```python
    import pandas as pd

    data = {
        'name': ['홍길동', '임꺽정', '장길산', '홍경래'],
        'kor': [90, 80, 70, 70],
        'eng': [99, 98, 97, 46],
        'mat': [90, 70, 70, 60],
    }
    df = pd.DataFrame(data)
    print("--- 원본 DataFrame ---")
    print(df)

    # 'total' 컬럼 추가: 'kor', 'eng', 'mat' 컬럼의 합
    df['total'] = df['kor'] + df['eng'] + df['mat']
    print("\n--- 'total' 컬럼 추가 후 ---")
    print(df)

    # 'avg' 컬럼 추가: 'total' 컬럼을 3으로 나눈 값
    df['avg'] = df['total'] / 3
    print("\n--- 'avg' 컬럼 추가 후 ---")
    print(df)
    ```

- **스칼라 값 또는 리스트/배열로 추가**

    새로운 컬럼을 모든 행에 동일한 스칼라 값으로 채우거나, `DataFrame`의 행 수와 동일한 길이의 리스트 또는 NumPy 배열로 채울 수 있습니다.

    - **스칼라 값으로 새 컬럼 추가**
    - **리스트/배열로 새 컬럼 추가** (길이가 DataFrame 행 수와 일치해야 함)
    - **람다 함수와 `apply()`를 이용한 컬럼 추가** (복잡한 로직)
    - **다른 Series 또는 DataFrame에서 컬럼 추가** (인덱스 정렬)

    ```python
    df['class'] = 'A'
    print("--- 'class' 컬럼 추가 후 (스칼라 값) ---")
    print(df)

    df['rank'] = [1, 2, 3, 3] # 4명의 학생
    print("--- 'rank' 컬럼 추가 후 (리스트) ---")
    print(df)

    # 참고: df.loc을 이용한 컬럼 추가 (명시적이고 안전한 방법)
    # df.loc[:, 'new_col_loc'] = [10, 20, 30, 40]
    # print("--- 'new_col_loc' 컬럼 추가 후 (df.loc) ---")
    # print(df)

    df['pass_fail'] = df.apply(lambda row: 'Pass' if row['avg'] >= 70 else 'Fail', axis=1) # 'avg'가 70점 이상이면 'Pass', 아니면 'Fail'
    print("--- 'pass_fail' 컬럼 추가 후 (apply) ---")
    print(df)

    new_series_data = pd.Series([True, False, True, False], index=[0, 1, 2, 3])
    df['is_scholarship'] = new_series_data
    print("--- 'is_scholarship' 컬럼 추가 후 (Series) ---")
    print(df)
    ```

#### 1.1.2. 기존 컬럼 값 수정

기존 컬럼의 값을 수정하는 것은 해당 컬럼을 선택한 후 새로운 값을 할당하는 방식으로 이루어집니다.

- **일괄 수정**

    모든 행에 대해 동일한 연산을 적용하여 컬럼 값을 일괄적으로 수정합니다.
    ```python
    df['kor'] = df['kor'] + 10
    print("--- 'kor' 점수 10점씩 올린 후 ---")
    print(df)
    ```

- **조건부 수정**

    `.loc` 인덱서를 사용하여 특정 조건을 만족하는 행의 컬럼 값만 선택적으로 수정할 수 있습니다.

    - **기본 조건부 수정**
        - `avg`가 80점 이상이면 `pass`, 아니면 `fail`을 부여
    - **`apply()`를 이용한 복잡한 조건부 수정**
        - `total` 점수에 따라 등급 부여 (A, B, C)
    - **`map()`을 이용한 값 매핑** (주로 범주형 데이터)
        - `class` 컬럼의 `A`를 `Excellent`로, `B`를 `Good`으로 매핑 (이 예제에서는 `class` 컬럼이 모두 `A`이므로 `Excellent`로만 변경됨)

    ```python
    df.loc[df['avg'] >= 80, 'result'] = 'pass' # 조건이 참인 행에 'result' 컬럼 생성 및 'pass' 할당
    df.loc[df['avg'] < 80, 'result'] = 'fail' # 조건이 거짓인 행에 'result' 컬럼 생성 및 'fail' 할당
    print("\n--- 조건부 'result' 컬럼 추가 후 ---")
    print(df)

    def assign_grade(total_score):
        if total_score >= 270:
            return 'A'
        elif total_score >= 200:
            return 'B'
        else:
            return 'C'

    df['grade'] = df['total'].apply(assign_grade)
    print("\n--- apply()로 'grade' 컬럼 수정 후 ---")
    print(df)

    # df['class'] = df['class'].map({'A': 'Excellent', 'B': 'Good'})
    # print("\n--- map()으로 'class' 컬럼 수정 후 ---")
    # print(df)
    ```

#### 1.1.3. `assign()` 메서드를 이용한 컬럼 추가

`assign()` 메서드는 새로운 컬럼을 추가한 `DataFrame`의 **복사본을 반환**합니다. 원본 `DataFrame`을 변경하지 않고, 여러 컬럼을 연달아(chaining) 추가할 때 매우 유용합니다.

- **기본 `assign()` 사용법**
    - `lambda` 함수를 사용하면 `assign()` 내에서 다른 컬럼을 참조할 수 있습니다.
- **여러 컬럼을 연달아(chaining) 추가하는 예제**
    - 이전 `assign()`에서 생성된 컬럼을 다음 `assign()`에서 참조할 수 있습니다.

```python
import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}
df_assign = pd.DataFrame(data)
print("---\u200b---\u200b원본 DataFrame ---\u200b---\u200b")
print(df_assign)

df_new = df_assign.assign(
    C = df_assign['A'] + df_assign['B'],
    D = lambda x: x['A'] * 2
)
print("\n--- assign()으로 컬럼 추가 후 ---\n")
print(df_new)

df_chained = df_assign.assign(
    C = lambda x: x['A'] + x['B'],
    D = lambda x: x['A'] * 2
).assign(
    E = lambda x: x['C'] + x['D'] # 이전 assign에서 생성된 'C'와 'D'를 참조
)
print("\n--- assign()으로 여러 컬럼 체이닝 추가 후 ---\n")
print(df_chained)

print("\n--- 원본 DataFrame (변경 없음) ---\n")
print(df_assign)

```

### 1.2. 컬럼 삭제

`DataFrame`에서 불필요한 컬럼을 삭제할 때는 `drop()` 메서드를 사용합니다. `axis=1` 또는 `columns=` 인자를 사용하여 컬럼을 삭제함을 명시해야 합니다.

- **`drop()` 메서드 사용법**
    - `labels`: 삭제할 컬럼 이름 (단일 이름은 문자열, 여러 개는 리스트)
    - `axis`: `1` 또는 `'columns'` (컬럼 삭제를 의미)
    - `inplace`: `True`로 설정 시 원본 `DataFrame`을 직접 수정 (기본값은 `False`)

- **예제**

    - **단일 컬럼 삭제**
    - **여러 컬럼 삭제** (`columns` 인자 사용)
    - **원본 `DataFrame` 직접 수정**
        - `inplace=True` 사용 시 원본이 변경됩니다.
        - `inplace=True`는 권장되지 않으며, 새로운 `DataFrame`을 할당하는 방식이 더 파이썬스럽고 예측 가능합니다.
    - **행(row) 삭제 예제** (`axis=0` 또는 `'index'` 사용)
    ```python
    df_dropped_kor = df.drop('kor', axis=1)
    print("\n--- 'kor' 컬럼 삭제 후 ---")
    print(df_dropped_kor)

    df_dropped_multiple = df.drop(columns=['eng', 'mat'])
    print("\n--- 'eng', 'mat' 컬럼 삭제 후 ---")
    print(df_dropped_multiple)

    df_inplace_copy = df.copy() # 원본 보호를 위해 복사본 사용
    df_inplace_copy.drop('name', axis=1, inplace=True)
    print("\n--- 'name' 컬럼 원본에서 삭제 후 (inplace=True) ---")
    print(df_inplace_copy)

    # 행(row) 삭제 예제 (axis=0 또는 'index' 사용)
    # df_row_dropped = df.drop([0, 2], axis=0) # 인덱스 0과 2에 해당하는 행 삭제
    # print("\n--- 인덱스 0, 2 행 삭제 후 ---")
    # print(df_row_dropped)
    ```

### 1.3. 컬럼 이름 변경

`rename()` 메서드나 `df.columns` 속성을 이용하여 컬럼 이름을 변경할 수 있습니다.

- **`rename()` 메서드 사용**

    딕셔너리를 사용하여 특정 컬럼의 이름을 선택적으로 변경할 때 유용합니다.

    - **여러 컬럼 이름 변경**
    - **함수를 이용한 컬럼 이름 변경** (예: 모든 컬럼 이름을 대문자로)

    ```python
    print("\n--- rename() 전 원본 DataFrame ---")
    print(df)

    df_renamed = df.rename(columns={'kor': '국어', 'eng': '영어', 'mat': '수학'})
    print("\n--- rename()으로 컬럼 이름 변경 후 ---")
    print(df_renamed)

    df_upper_cols = df.rename(columns=str.upper)
    print("\n--- 함수를 이용한 컬럼 이름 변경 (대문자) ---")
    print(df_upper_cols)
    ```

- **`df.columns`에 리스트 할당**

    모든 컬럼의 이름을 한 번에 변경할 때 사용합니다. 할당하는 리스트의 길이는 기존 컬럼의 수와 정확히 일치해야 합니다.

    ```python
    df_cols_copy = df.copy()
    df_cols_copy.columns = ['학생명', '국어점수', '영어점수', '수학점수', '총점', '평균', '반', '등수', '결과']
    print("\n--- df.columns로 전체 컬럼 이름 변경 후 ---")
    print(df_cols_copy)
    ```

### 1.4. 컬럼의 데이터 타입 변경

`astype()` 메서드는 컬럼의 데이터 타입을 변경하는 가장 일반적이고 효율적인 방법입니다. 메모리 사용량을 줄이거나, 연산을 위해 데이터 타입을 통일할 때 필수적입니다.

- **단일 컬럼 타입 변경**
- **여러 컬럼 타입 변경**
- **변환 불가능한 값이 있을 경우 처리**
    - `pd.to_numeric` 함수와 `errors='coerce'` 옵션을 사용하면 변환할 수 없는 값을 강제로 `NaN`(결측치)으로 만듭니다.
- **다른 일반적인 타입 변경 예시**
    - 문자열 숫자를 `float`으로
    - `0`/`1`을 `boolean`으로
    - 문자열 날짜를 `datetime`으로

```python
import pandas as pd

data_types = {
    'A': [1, 2, 3],
    'B': [4.0, 5.0, 6.0],
    'C': ['7', '8', '9'] # 문자열 형태의 숫자
}
df_types = pd.DataFrame(data_types)
print("--- 원본 DataFrame ---")
print(df_types)
print(f"\n원본 dtypes:\n{df_types.dtypes}")

# 단일 컬럼 타입 변경
df_types['B'] = df_types['B'].astype(int)
print("\n--- 'B' 컬럼을 int로 변경 후 ---")
print(df_types.dtypes)

# 여러 컬럼 타입 변경
df_types = df_types.astype({'A': float, 'C': int})
print("\n--- 'A'는 float, 'C'는 int로 변경 후 ---")
print(df_types.dtypes)

# 변환 불가능한 값이 있을 경우
df_error = pd.DataFrame({'col': ['1', '2', 'abc']})
# df_error['col'] = df_error['col'].astype(int) # 이 코드는 ValueError 발생

# to_numeric 함수와 errors='coerce' 옵션 사용
# 변환할 수 없는 값을 강제로 NaN(결측치)으로 만듦
df_error['col_numeric'] = pd.to_numeric(df_error['col'], errors='coerce')
print("\n--- to_numeric(errors='coerce') 적용 후 ---")
print(df_error)

# 다른 일반적인 타입 변경 예시
df_misc_types = pd.DataFrame({
    'num_str': ['1', '2', '3'],
    'bool_int': [0, 1, 0],
    'date_str': ['2023-01-01', '2023-01-02', '2023-01-03']
})
print(f"\n--- 기타 타입 변경 전 ---\n{df_misc_types.dtypes}")

df_misc_types['num_str'] = df_misc_types['num_str'].astype(float) # 문자열 숫자를 float으로
df_misc_types['bool_int'] = df_misc_types['bool_int'].astype(bool) # 0/1을 boolean으로
df_misc_types['date_str'] = pd.to_datetime(df_misc_types['date_str']) # 문자열 날짜를 datetime으로

print(f"\n--- 기타 타입 변경 후 ---\n{df_misc_types.dtypes}")
print(df_misc_types)

```

## 2. DataFrame 산술 연산

`DataFrame` 연산은 `06_데이터_연산_기본.md`에서 다룬 인덱스 정렬 및 브로드캐스팅 원리를 따릅니다. 여기서는 실제 연산자 및 메서드 사용법에 집중합니다.

### 2.1. DataFrame과 스칼라의 연산

DataFrame의 모든 요소에 스칼라(단일) 값이 브로드캐스팅되어 연산됩니다.

```python
import numpy as np

df_op = pd.DataFrame(np.arange(9).reshape(3, 3), columns=['A', 'B', 'C'])
print(f"원본 DataFrame:\n{df_op}")
print(f"DataFrame * 10:{df_op * 10}")
print(f"DataFrame + 5:{df_op + 5}")
print(f"DataFrame / 2:{df_op / 2}")
```

### 2.2. DataFrame 간의 연산

두 `DataFrame` 간의 산술 연산은 행 인덱스와 열 인덱스가 모두 일치하는 요소끼리 수행됩니다. 한쪽에만 존재하는 인덱스/컬럼의 값은 `NaN`이 됩니다.

- **기본 산술 연산**

    공통된 인덱스와 컬럼에 대해서만 연산이 수행되며, 한쪽에만 존재하는 인덱스/컬럼의 값은 `NaN`이 됩니다.
    ```python
    df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list('abc'))
    df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bce'))

    print(f"df1:\n{df1}")
    print(f"\ndf2:\n{df2}")

    print(f"\ndf1 + df2 (NaN 발생):\n{df1 + df2}")
    ```

- **`fill_value`를 이용한 결측치 처리**

    연산 메서드(`add`, `sub`, `mul`, `div` 등)를 사용하면 `fill_value`를 통해 한쪽 `DataFrame`에만 존재하는 값을 대체하여 연산할 수 있습니다. 이는 연산자 오버로딩과 동일한 결과를 반환하지만, `fill_value`와 같은 추가 옵션을 제공합니다.

    연산 메서드(`add`, `sub` 등)를 사용하면 `fill_value`를 통해 한쪽 DataFrame에만 존재하는 값을 대체하여 연산할 수 있습니다.

    ```python
    print(f"\ndf1.add(df2, fill_value=0):\n{df1.add(df2, fill_value=0)}")

    print(f"\ndf1.mul(df2, fill_value=1):\n{df1.mul(df2, fill_value=1)}")
    ```

### 2.3. 비교 및 논리 연산

`DataFrame`에 비교 연산자를 사용하면, 각 요소에 대해 비교를 수행하여 불리언 값으로 이루어진 `DataFrame`을 반환합니다.

```python
df_compare = pd.DataFrame({
    'A': [1, 5, 10],
    'B': [20, 15, 8]
})
print(f"비교 연산용 DataFrame:\n{df_compare}")

# 스칼라와 비교
print(f"DataFrame > 10:{df_compare > 10}")

print(f"\nDataFrame > df1:\n{df_compare > df1}")

df_logic = pd.DataFrame({
    'X': [True, False, True],
    'Y': [False, True, True]
})
print(f"\n논리 연산용 DataFrame:\n{df_logic}")

print(f"\n(df_logic['X'] & df_logic['Y']):")
print(df_logic['X'] & df_logic['Y']) # X AND Y
print(f"\n(df_logic['X'] | df_logic['Y']):")
print(df_logic['X'] | df_logic['Y']) # X OR Y
print(f"\n(~df_logic['X']):")
print(~df_logic['X']) # NOT X
```