<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 Pandas의 고급 데이터 조작 및 분석 기능 중 결측치(`NaN`) 처리 심화, MultiIndex(계층적 인덱싱) 활용, 메서드 체이닝(Method Chaining), 그리고 `.pipe()` 메서드를 이용한 재사용 가능한 워크플로우 구축 방법을 다룹니다. 이 기능들은 복잡한 데이터 전처리 및 분석 파이프라인을 효율적이고 가독성 높게 구축하는 데 필수적입니다.

<h2>목차</h2>

- [Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵](#pandas-학습-가이드-데이터-과학자를-위한-실무-중심-로드맵)
- [문서 목표](#문서-목표)
- [목차](#목차)
- [1. 고급 데이터 조작 및 분석](#1-고급-데이터-조작-및-분석)
  - [1.1. 결측치(NaN) 처리 심화](#11-결측치nan-처리-심화)
    - [1.1.1. 결측치 탐지](#111-결측치-탐지)
    - [1.1.2. 결측치 제거 (`dropna`)](#112-결측치-제거-dropna)
    - [1.1.3. 결측치 채우기 (`fillna`)](#113-결측치-채우기-fillna)
    - [1.1.4. 결측치 보간 (`interpolate`)](#114-결측치-보간-interpolate)
  - [1.2. MultiIndex (계층적 인덱싱) 활용](#12-multiindex-계층적-인덱싱-활용)
    - [1.2.1. MultiIndex 생성](#121-multiindex-생성)
    - [1.2.2. MultiIndex 데이터 접근 및 슬라이싱](#122-multiindex-데이터-접근-및-슬라이싱)
    - [1.2.3. MultiIndex 조작 (`stack`, `unstack`, `reset_index`)](#123-multiindex-조작-stack-unstack-reset_index)
  - [1.3. 메서드 체이닝 (Method Chaining)](#13-메서드-체이닝-method-chaining)
  - [1.4. 재사용 가능한 워크플로우: `.pipe()` 활용](#14-재사용-가능한-워크플로우-pipe-활용)

---

## 1. 고급 데이터 조작 및 분석

### 1.1. 결측치(NaN) 처리 심화

실제 데이터에서 결측치(`NaN`, Not a Number)를 다루는 것은 데이터 전처리에서 가장 중요한 단계 중 하나입니다. Pandas는 결측치를 탐지하고, 제거하고, 다른 값으로 대체하는 다양한 방법을 제공합니다.

#### 1.1.1. 결측치 탐지

*   `isnull()` 또는 `isna()`: 각 요소가 결측치인지 여부를 불리언 `DataFrame`으로 반환합니다.
*   `notnull()` 또는 `notna()`: `isnull()`의 반대입니다.
*   `isnull().sum()`: 컬럼별 결측치의 총 개수를 확인하는 데 가장 일반적으로 사용됩니다.
*   `isnull().any(axis=1)`: 행별로 결측치가 하나라도 있는지 확인합니다.

```python
import pandas as pd
import numpy as np

df_nan = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 10, 20, np.nan, 30],
    'C': [100, 200, 300, 400, 500],
    'D': [np.nan, np.nan, np.nan, np.nan, np.nan]
})

print("원본 DataFrame:\n", df_nan)

# 컬럼별 결측치 개수
print("\n컬럼별 결측치 개수:\n", df_nan.isnull().sum())

# 결측치가 있는 행 확인
print("\n결측치가 있는 행 (불리언):\n", df_nan.isnull().any(axis=1))
```

#### 1.1.2. 결측치 제거 (`dropna`)

`dropna()`는 결측치가 있는 행 또는 열을 제거합니다.

*   `axis`: `0` (행, 기본값) 또는 `1` (열).
*   `how`: `'any'` (하나라도 결측치면 제거, 기본값) 또는 `'all'` (모든 값이 결측치일 때만 제거).
*   `thresh`: `NaN`이 아닌 값이 최소한 `thresh`개 이상 있어야 행/열을 유지합니다.

```python
# 결측치가 하나라도 있는 행 제거
print("\n결측치 있는 행 제거 (how='any'):\n", df_nan.dropna(how='any'))

# 모든 값이 결측치인 행/열 제거
print("\n모든 값이 결측치인 열 제거 (how='all', axis=1):\n", df_nan.dropna(how='all', axis=1))

# NaN이 아닌 값이 4개 이상인 행만 유지
print("\nNaN이 아닌 값이 4개 이상인 행만 유지 (thresh=4):\n", df_nan.dropna(thresh=4))
```

#### 1.1.3. 결측치 채우기 (`fillna`)

`fillna()`는 결측치를 특정 값으로 대체합니다.

*   `value`: 결측치를 채울 값 (스칼라, 딕셔너리, `Series`).
*   `method`: `'ffill'` (Forward fill, 이전 값으로 채우기), `'bfill'` (Backward fill, 이후 값으로 채우기).
*   `limit`: `method`와 함께 사용할 때, 연속된 결측치 중 몇 개까지 채울지 지정.

```python
# 결측치를 0으로 채우기
print("\n결측치를 0으로 채우기:\n", df_nan.fillna(0))

# 컬럼별 다른 값으로 채우기 (A는 평균, B는 99)
fill_values = {'A': df_nan['A'].mean(), 'B': 99}
print("\n컬럼별 다른 값으로 채우기:\n", df_nan.fillna(value=fill_values))

# Forward Fill (ffill): 이전 값으로 채우기
df_ffill = pd.DataFrame({'Data': [10, np.nan, np.nan, 20, np.nan, 30]})
print("\nForward Fill (ffill) 예시:\n", df_ffill.fillna(method='ffill'))

# Backward Fill (bfill) 및 limit: 이후 값으로 채우고 최대 1개만 채움
df_bfill = pd.DataFrame({'Data': [np.nan, 10, np.nan, np.nan, 20, np.nan]})
print("\nBackward Fill (bfill) 및 limit 예시:\n", df_bfill.fillna(method='bfill', limit=1))
```

#### 1.1.4. 결측치 보간 (`interpolate`)

`interpolate()`는 결측치를 선형 보간법 등 다양한 보간 방법에 따라 채웁니다. 데이터가 일정한 트렌드를 가질 때 유용합니다.

*   `method`: `'linear'` (선형, 기본값), `'polynomial'`, `'spline'` 등.
*   `limit_direction`: `'forward'`, `'backward'`, `'both'`.

```python
df_inter = pd.DataFrame({'Data': [10, np.nan, np.nan, 40, np.nan, 70]})
print("\n보간법 (interpolate) 예시:\n", df_inter.interpolate(method='linear'))

# 방향 지정 (backward)
print("\n보간법 (interpolate, backward):\n", df_inter.interpolate(method='linear', limit_direction='backward'))
```

### 1.2. MultiIndex (계층적 인덱싱) 활용

MultiIndex(또는 계층적 인덱싱)는 `DataFrame`의 인덱스를 여러 레벨로 구성하여 고차원 데이터를 2차원 형식으로 표현할 수 있게 해주는 강력한 기능입니다. `groupby()`나 `pivot_table()`의 결과로 자연스럽게 생성되며, 데이터를 더 세분화하여 분석하고 조작하는 데 사용됩니다.

#### 1.2.1. MultiIndex 생성

*   `set_index()`: 하나 이상의 컬럼을 인덱스로 설정하여 MultiIndex를 생성합니다.
*   `pd.MultiIndex.from_product()`: 여러 리스트의 데카르트 곱(Cartesian product)으로 MultiIndex를 직접 생성합니다.

```python
# 예시 데이터
data_multi = {
    'Region': ['East', 'East', 'West', 'West', 'East', 'West'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Year': [2022, 2022, 2022, 2022, 2023, 2023],
    'Sales': [100, 150, 200, 250, 110, 260]
}
df_multi = pd.DataFrame(data_multi)

# Region과 Product를 인덱스로 설정하여 MultiIndex 생성
df_multi_indexed = df_multi.set_index(['Region', 'Product'])
print("MultiIndex DataFrame (set_index):\n", df_multi_indexed)

# pd.MultiIndex.from_product로 MultiIndex 직접 생성
index_from_prod = pd.MultiIndex.from_product([['A', 'B'], [1, 2]], names=['Level1', 'Level2'])
s_multi = pd.Series(np.random.rand(4), index=index_from_prod)
print("\nMultiIndex Series (from_product):\n", s_multi)
```

#### 1.2.2. MultiIndex 데이터 접근 및 슬라이싱

MultiIndex `DataFrame`에서 데이터를 선택할 때는 튜플을 사용하거나 `pd.IndexSlice`를 활용합니다.

```python
# MultiIndex DataFrame (Year, Region, Product를 인덱스로)
df_sales_multi = df_multi.set_index(['Year', 'Region', 'Product'])
print("MultiIndex DataFrame:\n", df_sales_multi)

# 특정 레벨의 단일 값 접근 (튜플 사용)
print("\n2022년 East 지역의 A 제품 판매:\n", df_sales_multi.loc[(2022, 'East', 'A')])

# 특정 레벨의 모든 값 접근 (슬라이싱)
print("\n2022년 East 지역 모든 제품 판매:\n", df_sales_multi.loc[(2022, 'East'), :])

# xs(): 특정 레벨의 데이터를 교차 선택
print("\nRegion이 East인 모든 데이터 (xs):\n", df_sales_multi.xs('East', level='Region'))

# pd.IndexSlice를 이용한 고급 슬라이싱
idx = pd.IndexSlice
# 모든 Year에 대해 Region이 'East'이고 Product가 'A'인 데이터
print("\nIndexSlice를 이용한 고급 슬라이싱:\n", df_sales_multi.loc[idx[:, 'East', 'A'], :])
```

#### 1.2.3. MultiIndex 조작 (`stack`, `unstack`, `reset_index`)

*   `stack()`: 컬럼 레벨을 인덱스 레벨로 변환합니다 (데이터를 '긴' 형식으로 만듦).
*   `unstack()`: 인덱스 레벨을 컬럼 레벨로 변환합니다 (데이터를 '넓은' 형식으로 만듦).
*   `reset_index()`: 인덱스의 일부 또는 전체를 컬럼으로 되돌립니다.

```python
# stack(): 컬럼을 인덱스로 변환
df_stacked = df_sales_multi.stack()
print("\nStacked DataFrame (긴 형식):\n", df_stacked)

# unstack(): 인덱스를 컬럼으로 변환 (가장 안쪽 인덱스 레벨을 컬럼으로)
print("\nUnstacked DataFrame (가장 안쪽 인덱스):\n", df_stacked.unstack())

# reset_index(): 인덱스를 컬럼으로 되돌리기
print("\n인덱스를 컬럼으로 되돌리기 (전체):\n", df_sales_multi.reset_index())
print("\n인덱스를 컬럼으로 되돌리기 (특정 레벨만):\n", df_sales_multi.reset_index(level='Product'))
```

### 1.3. 메서드 체이닝 (Method Chaining)

메서드 체이닝은 여러 데이터 처리 단계를 하나의 연속된 라인으로 연결하여 코드의 가독성과 흐름을 개선하는 코딩 스타일입니다. 각 메서드가 `DataFrame`을 반환하기 때문에, 그 결과에 다시 점(`.`)을 찍어 다음 메서드를 호출하는 방식입니다. 중간 과정에서 불필요한 변수 생성을 피할 수 있어 코드가 깔끔해집니다.

**메서드 체이닝의 장점:**
*   **가독성 향상**: 데이터의 흐름이 위에서 아래로 자연스럽게 이어져 코드를 읽기 편합니다.
*   **코드 간결성**: 중간 변수 선언 없이 여러 단계를 한 번에 표현할 수 있습니다.
*   **디버깅 용이**: 각 단계를 주석 처리하며 중간 결과를 쉽게 확인할 수 있습니다.

**가독성을 위해 각 메서드 호출을 괄호`()`로 감싸고 줄바꿈을 하는 것이 일반적입니다.**

```python
# 예시 데이터
df_chain = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A', 'C', 'A'],
    'value1': [10, 20, 30, 40, 50, 60, np.nan],
    'value2': [5, 15, 25, 35, 45, 55, 10],
    'status': ['active', 'inactive', 'active', 'active', 'inactive', 'active', 'active']
})

# 메서드 체이닝을 사용한 경우
final_df_chained = (
    df_chain
    .dropna(subset=['value1']) # value1에 NaN이 있는 행 제거
    .query("category != 'C'")  # category가 'C'가 아닌 행만 필터링
    .assign(total = lambda df: df.value1 + df.value2) # value1과 value2를 더한 total 컬럼 추가
    .groupby('category') # category로 그룹화
    .agg(avg_total=('total', 'mean'), max_value1=('value1', 'max')) # 그룹별 total 평균과 value1 최대값 계산
    .sort_values(by='avg_total', ascending=False) # avg_total 기준으로 내림차순 정렬
    .reset_index() # category를 컬럼으로 되돌림
)

print("메서드 체이닝 사용 결과:\n", final_df_chained)
```

### 1.4. 재사용 가능한 워크플로우: `.pipe()` 활용

메서드 체이닝은 훌륭하지만, 여러 단계의 복잡한 전처리 함수를 적용할 때는 `.pipe()` 메서드가 더 깔끔하고 재사용 가능한 코드를 만들어 줍니다. `pipe`는 `DataFrame` (또는 `Series`)을 사용자 정의 함수의 첫 번째 인자로 전달하여, 체인 중간에 사용자 정의 함수를 삽입할 수 있게 합니다. 이는 데이터 처리 로직을 함수 단위로 모듈화하고 재사용성을 높이는 데 매우 효과적입니다.

**`pipe()`의 장점:**
*   **모듈성**: 복잡한 전처리 단계를 작은, 재사용 가능한 함수로 분리하여 코드를 깔끔하게 유지합니다.
*   **재사용성**: 한 번 정의된 함수는 다른 `DataFrame`이나 다른 체인에서도 쉽게 재사용할 수 있습니다.
*   **가독성**: 체인의 각 단계가 명확한 함수 이름으로 표현되어 코드의 흐름을 이해하기 쉽습니다.

```python
# 예시 데이터
df_pipe = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Score1': [85, 90, np.nan, 75, 92],
    'Score2': [70, np.nan, 88, 95, 80],
    'City': ['Seoul', 'Busan', 'Seoul', 'Jeju', 'Busan']
})
print("원본 DataFrame:\n", df_pipe)

# 재사용 가능한 전처리 함수 정의
def fill_missing_scores(df_in, fill_value=0):
    """점수 컬럼의 결측치를 채우는 함수"""
    print(f"\n-> 결측치 채우기 (값: {fill_value})")
    return df_in.fillna({'Score1': fill_value, 'Score2': fill_value})

def convert_city_to_onehot(df_in, column_name='City'):
    """도시 컬럼을 One-Hot Encoding하는 함수"""
    print(f"-> {column_name} 컬럼 One-Hot Encoding")
    return pd.get_dummies(df_in, columns=[column_name], prefix=column_name)

def calculate_total_score(df_in):
    """총점을 계산하는 함수"""
    print("-> 총점 계산")
    return df_in.assign(Total_Score = df_in['Score1'] + df_in['Score2'])

# .pipe()를 이용한 워크플로우 구축
processed_df = (
    df_pipe
    .pipe(fill_missing_scores, fill_value=50) # Score1, Score2의 NaN을 50으로 채움
    .pipe(convert_city_to_onehot, column_name='City') # City 컬럼 One-Hot Encoding
    .pipe(calculate_total_score) # 총점 계산
    .drop(columns=['Score1', 'Score2']) # 원본 점수 컬럼 제거
)

print("\n최종 처리된 DataFrame:\n", processed_df)
```

이처럼 `.pipe()`를 활용하면 복잡한 데이터 전처리 파이프라인을 명확하고 모듈화된 형태로 구축할 수 있어, 코드의 유지보수성과 재사용성을 크게 향상시킬 수 있습니다.

```