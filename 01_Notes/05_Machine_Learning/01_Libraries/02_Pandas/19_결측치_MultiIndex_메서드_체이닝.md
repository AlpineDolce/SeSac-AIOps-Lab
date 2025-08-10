<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Pandas의 고급 데이터 조작 및 분석 기능 중 결측치(`NaN`) 처리 심화, MultiIndex(계층적 인덱싱) 활용, 메서드 체이닝(Method Chaining), 그리고 `.pipe()` 메서드를 이용한 재사용 가능한 워크플로우 구축 방법을 다룹니다. 이 기능들은 복잡한 데이터 전처리 및 분석 파이프라인을 효율적이고 가독성 높게 구축하는 데 필수적입니다.

<h2>목차</h2>

- [1. 고급 데이터 조작 및 분석](#1-고급-데이터-조작-및-분석)
  - [1.1. 결측치(NaN) 처리 심화](#11-결측치nan-처리-심화)
  - [1.2. MultiIndex (계층적 인덱싱) 활용](#12-multiindex-계층적-인덱싱-활용)
  - [1.3. 메서드 체이닝 (Method Chaining)](#13-메서드-체이닝-method-chaining)
  - [1.4. 재사용 가능한 워크플로우: `.pipe()` 활용](#14-재사용-가능한-워크플로우-pipe-활용)

---

## 1. 고급 데이터 조작 및 분석

### 1.1. 결측치(NaN) 처리 심화

실제 데이터에서 결측치(`NaN`)를 다루는 것은 데이터 전처리에서 가장 중요한 단계 중 하나입니다. Pandas는 결측치를 탐지하고, 제거하고, 다른 값으로 대체하는 다양한 방법을 제공합니다.

**1. 결측치 탐지**
- `isnull()` 또는 `isna()`: 각 요소가 결측치인지 여부를 불리언 DataFrame으로 반환합니다.
- `notnull()` 또는 `notna()`: `isnull()`의 반대입니다.
- `isnull().sum()`: 컬럼별 결측치의 총 개수를 확인하는 데 가장 일반적으로 사용됩니다.

**2. 결측치 제거**
- `dropna(axis=0, how='any')`: 결측치가 하나라도 있는 행(`axis=0`)을 제거합니다. `how='all'`은 모든 값이 결측치인 행만 제거합니다.

**3. 결측치 채우기 (Imputation)**
- `fillna(value)`: 결측치를 특정 스칼라 값이나 딕셔너리 형태로 각 컬럼별 다른 값으로 채웁니다.
- `fillna(method='ffill')`: Forward fill. 바로 앞의 유효한 값으로 결측치를 채웁니다. 시계열 데이터에 유용합니다.
- `fillna(method='bfill')`: Backward fill. 바로 뒤의 유효한 값으로 결측치를 채웁니다.
- `interpolate()`: 결측치를 선형 보간법 등 다양한 보간 방법에 따라 채웁니다. 데이터가 일정한 트렌드를 가질 때 유용합니다.

```python
import pandas as pd
import numpy as np

df_nan = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 10, 20, np.nan, 30],
    'C': [100, 200, 300, 400, 500]
})

print("원본 DataFrame:\n", df_nan)

# 결측치 탐지
print("\n컬럼별 결측치 개수:\n", df_nan.isnull().sum())

# 결측치 제거
print("\n결측치 있는 행 제거:\n", df_nan.dropna())

# 결측치 채우기
print("\n결측치를 0으로 채우기:\n", df_nan.fillna(0))

# 컬럼별 다른 값으로 채우기
fill_values = {'A': df_nan['A'].mean(), 'B': 99}
print("\n컬럼별 다른 값으로 채우기:\n", df_nan.fillna(value=fill_values))

# Forward Fill (ffill)
df_ffill = pd.DataFrame({'Data': [10, np.nan, np.nan, 20, np.nan, 30]})
print("\nForward Fill (ffill) 예시:")
print("원본:", df_ffill.T)
print("ffill 적용 후:", df_ffill.fillna(method='ffill').T)

# 보간법 (Interpolation)
df_inter = pd.DataFrame({'Data': [10, np.nan, np.nan, 40]})
print("\n보간법 (interpolate) 예시:")
print("원본:", df_inter.T)
print("interpolate 적용 후:", df_inter.interpolate().T)
```

### 1.2. MultiIndex (계층적 인덱싱) 활용

MultiIndex(또는 계층적 인덱싱)는 DataFrame의 인덱스를 여러 레벨로 구성하여 고차원 데이터를 2차원 형식으로 표현할 수 있게 해주는 강력한 기능입니다. `groupby()`나 `pivot_table()`의 결과로 자연스럽게 생성되며, 데이터를 더 세분화하여 분석하고 조작하는 데 사용됩니다.

**주요 기능:**
- `set_index()`: 하나 이상의 컬럼을 인덱스로 설정하여 MultiIndex를 생성합니다.
- `reset_index()`: 인덱스의 일부 또는 전체를 컬럼으로 되돌립니다.
- `stack()`: 컬럼 레벨을 인덱스 레벨로 변환합니다 (데이터를 '긴' 형식으로 만듦).
- `unstack()`: 인덱스 레벨을 컬럼 레벨로 변환합니다 (데이터를 '넓은' 형식으로 만듦).

```python
import pandas as pd

# 예시 데이터
data = {
    'Region': ['East', 'East', 'West', 'West'],
    'Product': ['A', 'B', 'A', 'B'],
    '2022_Sales': [100, 150, 200, 250],
    '2023_Sales': [110, 140, 220, 240]
}
df_multi = pd.DataFrame(data)

# Region과 Product를 인덱스로 설정하여 MultiIndex 생성
df_multi = df_multi.set_index(['Region', 'Product'])
print("MultiIndex DataFrame:\n", df_multi)

# MultiIndex를 이용한 데이터 접근
print("\nEast 지역의 A 제품 데이터:\n", df_multi.loc[('East', 'A')])

# stack(): 컬럼을 인덱스로 변환
df_stacked = df_multi.stack()
print("\nStacked DataFrame (긴 형식):\n", df_stacked)

# unstack(): 인덱스를 컬럼으로 변환
# df_stacked.unstack()을 하면 원래의 df_multi와 유사한 형태로 돌아감
print("\nUnstacked DataFrame (넓은 형식):\n", df_stacked.unstack())

# reset_index(): 인덱스를 컬럼으로 되돌리기
print("\n인덱스를 컬럼으로 되돌리기:\n", df_multi.reset_index())
```

**MultiIndex 고급 슬라이싱 (`pd.IndexSlice`)**

MultiIndex(계층적 인덱스)를 다룰 때 가장 어려운 부분은 특정 레벨의 데이터를 깔끔하게 슬라이싱하는 것입니다. `.loc`에 튜플을 사용하는 방식은 복잡해지기 쉽습니다. 이때 `pd.IndexSlice` 객체를 사용하면 여러 인덱스 레벨에 대해 간결하고 가독성 높은 슬라이싱이 가능합니다.

- **예시**: `df.loc[pd.IndexSlice[:, 'A'], :]` (모든 첫 번째 레벨 인덱스에 대해 두 번째 레벨 인덱스가 'A'인 행 선택)

### 1.3. 메서드 체이닝 (Method Chaining)

메서드 체이닝은 여러 데이터 처리 단계를 하나의 연속된 라인으로 연결하여 코드의 가독성과 흐름을 개선하는 코딩 스타일입니다. 각 메서드가 DataFrame을 반환하기 때문에, 그 결과에 다시 점(`.`)을 찍어 다음 메서드를 호출하는 방식입니다. 중간 과정에서 불필요한 변수 생성을 피할 수 있어 코드가 깔끔해집니다.

**메서드 체이닝의 장점:**
- **가독성 향상**: 데이터의 흐름이 위에서 아래로 자연스럽게 이어져 코드를 읽기 편합니다.
- **코드 간결성**: 중간 변수 선언 없이 여러 단계를 한 번에 표현할 수 있습니다.
- **디버깅 용이**: 각 단계를 주석 처리하며 중간 결과를 쉽게 확인할 수 있습니다.

**가독성을 위해 각 메서드 호출을 괄호`()`로 감싸고 줄바꿈을 하는 것이 일반적입니다.**

```python
import pandas as pd

# 예시 데이터
df_chain = pd.DataFrame({
    'category': ['A', 'B', 'A', 'B', 'A', 'C'],
    'value1': [10, 20, 30, 40, 50, 60],
    'value2': [5, 15, 25, 35, 45, 55]
})

# 메서드 체이닝을 사용하지 않은 경우
temp_df1 = df_chain[df_chain['category'] != 'C']
temp_df2 = temp_df1.copy() # SettingWithCopyWarning 방지
temp_df2['total'] = temp_df2['value1'] + temp_df2['value2']
final_df = temp_df2.groupby('category')['total'].mean().reset_index()
print("체이닝 미사용 결과:\n", final_df)

# 메서드 체이닝을 사용한 경우
final_df_chained = (
    df_chain
    .query("category != 'C'")  # 조건부 필터링
    .assign(total = lambda df: df.value1 + df.value2) # 새로운 컬럼 추가
    .groupby('category')['total']
    .mean()
    .reset_index()
)

print("\n메서드 체이닝 사용 결과:\n", final_df_chained)
```

### 1.4. 재사용 가능한 워크플로우: `.pipe()` 활용

메서드 체이닝은 훌륭하지만, 여러 단계의 복잡한 전처리 함수를 적용할 때는 `.pipe()` 메서드가 더 깔끔하고 재사용 가능한 코드를 만들어 줍니다. `pipe`는 사용자 정의 함수를 체인 중간에 삽입할 수 있게 하여, 데이터 처리 로직을 함수 단위로 모듈화할 수 있게 돕습니다.

**실무적 중요성**
여러 프로젝트에서 공통적으로 사용되는 전처리 로직(예: 특정 로그 포맷 정제)을 함수로 만들어두고, `.pipe()`를 통해 여러 분석 스크립트에서 재사용하는 것은 매우 효율적인 개발 방식입니다.

```