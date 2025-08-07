<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 `DataFrame` 간의 연산 방법과 컬럼을 추가, 수정, 삭제하는 방법을 상세히 다룹니다. `DataFrame`의 각 컬럼이 `Series` 객체임을 이해하고, 이를 활용하여 새로운 파생 컬럼을 생성하거나 기존 데이터를 효율적으로 조작하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. DataFrame 간 연산 및 컬럼 추가/삭제](#1-dataframe-간-연산-및-컬럼-추가삭제)
  - [1.1. 기본 연산과 새 필드 추가](#11-기본-연산과-새-필드-추가)
  - [1.2. DataFrame 간의 연산](#12-dataframe-간의-연산)

---

## 1. DataFrame 간 연산 및 컬럼 추가/삭제

DataFrame 간의 연산도 Series와 마찬가지로 행 인덱스와 열 인덱스 모두를 기준으로 정렬된 후 수행됩니다.

### 1.1. 기본 연산과 새 필드 추가

DataFrame의 각 컬럼은 Series 객체이므로, 컬럼 간의 연산은 Series 연산과 동일하게 작동합니다. 이를 활용하여 기존 컬럼의 데이터를 기반으로 새로운 파생 컬럼을 쉽게 생성할 수 있습니다.

```python
import pandas as pd

# Series를 결합하여 DataFrame 만들기 (append는 Deprecated, pd.concat 권장)
# 기존 코드:
# data = pd.DataFrame()
# data = data.append(data1, ignore_index=True)
# data = data.append(data2, ignore_index=True)
# data = data.append(data3, ignore_index=True)

# 권장 방식: 딕셔너리 리스트로 DataFrame 생성
data_for_df = [
    {'kor': 90, 'eng': 70, 'mat': 80},
    {'kor': 90, 'eng': 70, 'mat': 80},
    {'kor': 90, 'eng': 70, 'mat': 80}
]
df_scores = pd.DataFrame(data_for_df)
print("--- DataFrame 기본 연산 및 새 필드 추가 ---")
print("원본 DataFrame:\n", df_scores)

# 새 필드 'total' 추가: 'kor', 'eng', 'mat' 컬럼의 합
df_scores['total'] = df_scores.kor + df_scores.eng + df_scores.mat
print("\n'total' 컬럼 추가 후:\n", df_scores)

# 새 필드 'avg' 추가: 'total' 컬럼을 3으로 나눈 값
df_scores['avg'] = df_scores.total / 3
print("\navg 컬럼 추가 후:\n", df_scores)
# 출력 예시:
# 'avg' 컬럼 추가 후:
#    kor  eng  mat  total        avg
# 0   90   70   80    240  80.000000
# 1   90   70   80    240  80.000000
# 2   90   70   80    240  80.000000
```

### 1.2. DataFrame 간의 연산

두 DataFrame 간의 연산은 행 인덱스와 열 인덱스 모두를 기준으로 정렬된 후 수행됩니다. 공통되지 않은 인덱스/컬럼에 대해서는 `NaN`이 발생합니다.

```python
import pandas as pd

df_a = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}, index=['x', 'y', 'z'])

df_b = pd.DataFrame({
    'B': [10, 20, 30],
    'C': [40, 50, 60]
}, index=['y', 'z', 'w'])

print("\n--- DataFrame 간 연산 ---")
print("df_a:\n", df_a)
print("\ndf_b:\n", df_b)

# df_a와 df_b 더하기
# 공통된 인덱스('y', 'z')와 컬럼('B')에 대해서만 연산 수행
# 나머지는 NaN 처리
df_sum = df_a + df_b
print("\ndf_a + df_b 결과:\n", df_sum)
# 출력:
# df_a + df_b 결과:
#      A     B     C
# x  NaN   NaN   NaN
# y  NaN  15.0   NaN
# z  NaN  26.0   NaN
```
    위 결과에서 `A` 컬럼은 `df_b`에 없고, `C` 컬럼은 `df_a`에 없으므로 해당 컬럼들은 `NaN`이 됩니다. 또한, `x` 행은 `df_b`에 없고, `w` 행은 `df_a`에 없으므로 해당 행들도 `NaN`이 됩니다. 오직 공통된 인덱스와 컬럼(`y`, `z` 행의 `B` 컬럼)에 대해서만 연산이 수행됩니다.

