<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
 Pandas의 `pd.set_option()`을 사용하여 DataFrame의 표시 방식을 사용자 정의하고, `.plot()` 접근자를 활용한 내장 시각화 기능을 통해 데이터를 빠르게 탐색하고 시각적으로 분석하는 실무 역량을 강화합니다.

<h2>목차</h2>

- [1. Pandas 표시 옵션 설정 (`pd.set_option`)](#1-pandas-표시-옵션-설정-pdset_option)
- [2. Pandas 내장 시각화 (`.plot()` 접근자)](#2-pandas-내장-시각화-plot-접근자)
- [3. 실무 활용 시나리오](#3-실무-활용-시나리오)
- [4. 요약 및 모범 사례](#4-요약-및-모범-사례)

---

### 1. Pandas 표시 옵션 설정 (`pd.set_option`)

Jupyter Notebook이나 터미널에서 대용량 `DataFrame`을 다룰 때, Pandas는 기본적으로 출력의 일부를 `...`으로 생략하여 보여줍니다. `pd.set_option()` 함수는 이러한 기본 동작을 변경하여 데이터 탐색 및 디버깅을 더 효율적으로 만들어주는 강력한 기능입니다.

-   **기본 사용법**: `pd.set_option('option.name', value)`
-   **현재 설정 확인**: `pd.get_option('option.name')`
-   **기본값으로 리셋**: `pd.reset_option('option.name')`

---
**주요 옵션 상세 설명 및 예제**

```python
import pandas as pd
import numpy as np

# 예시 DataFrame 생성
df = pd.DataFrame(np.random.rand(20, 8), columns=[f'col_{i}' for i in range(8)])
df['long_text'] = ['This is a very long text string that might be truncated.' for _ in range(20)]
```

**1. 행/열 표시 개수 제어**

-   `display.max_rows`: 한 번에 표시할 최대 행의 개수. `None`으로 설정하면 모든 행을 표시합니다.
-   `display.max_columns`: 한 번에 표시할 최대 열의 개수. `None`으로 설정하면 모든 열을 표시합니다.

```python
print("기본 출력 (일부 행/열 생략):")
print(df)

pd.set_option('display.max_rows', 10) # 최대 10행까지 표시
pd.set_option('display.max_columns', 5) # 최대 5열까지 표시
print("\n최대 10행, 5열까지 표시:")
print(df)

pd.reset_option('display.max_rows') # 옵션을 기본값으로 복원
pd.reset_option('display.max_columns')
```

**2. 표시 너비 및 컬럼 내용 제어**

-   `display.width`: 한 줄에 표시될 최대 너비(문자 수). `None`으로 설정하면 터미널 너비에 맞춰 자동 조절됩니다.
-   `display.max_colwidth`: 각 컬럼에 표시될 내용의 최대 너비. 긴 텍스트가 잘리는 것을 방지할 때 유용합니다.

```python
pd.set_option('display.max_colwidth', 20) # 각 컬럼 내용을 최대 20자까지만 표시
print("\n최대 컬럼 너비 20으로 제한:")
print(df[['long_text']])

pd.set_option('display.max_colwidth', None) # 컬럼 내용 전체 표시
print("\n컬럼 너비 제한 해제:")
print(df[['long_text']])
```

**3. 숫자 형식 제어**

-   `display.precision`: 부동 소수점(float)의 소수점 표시 자릿수를 제어합니다.
-   `display.float_format`: `lambda`나 포맷 문자열을 사용하여 부동 소수점의 출력 형식을 직접 지정합니다. 과학적 표기법(scientific notation)을 억제하거나, 특정 형식(예: 통화, 퍼센트)으로 표현할 때 매우 유용합니다.

```python
df_float = pd.DataFrame(np.random.randn(5, 4) * 1e-6, columns=['A', 'B', 'C', 'D'])
print("\n기본 float 출력 (과학적 표기법):")
print(df_float)

# 소수점 8자리까지 표시 (과학적 표기법 억제 효과)
pd.set_option('display.float_format', '{:.8f}'.format)
print("\n소수점 8자리까지 표시:")
print(df_float)

# 퍼센트(%) 형식으로 표시
pd.set_option('display.float_format', '{:.2%}'.format)
print("\n퍼센트 형식으로 표시:")
print(df_float)

pd.reset_option('display.float_format')
```

**4. 임시 설정 변경 (`pd.option_context`)**

`with pd.option_context(...)`를 사용하면, `with` 블록 내에서만 일시적으로 옵션을 변경하고, 블록이 끝나면 자동으로 이전 설정으로 복원됩니다. 전역 설정을 건드리지 않으므로 특정 분석에만 다른 옵션을 적용하고 싶을 때 매우 안전하고 유용한 방법입니다.

```python
print("--- 컨텍스트 매니저 외부 (기본 설정) ---")
print(df.head(3))

with pd.option_context('display.max_rows', 3, 'display.precision', 2):
    print("\n--- 컨텍스트 매니저 내부 (임시 설정) ---")
    print(df)

print("\n--- 컨텍스트 매니저 외부 (다시 기본 설정으로 복원) ---")
print(df.head(3))
```
이러한 표시 옵션들을 잘 활용하면 데이터 분석 과정에서 불필요한 출력으로 인한 혼란을 줄이고, 원하는 정보를 명확하게 확인하여 작업 효율을 크게 높일 수 있습니다.

### 2. Pandas 내장 시각화 (`.plot()` 접근자)

Pandas는 `matplotlib` 라이브러리를 내부적으로 사용하여, DataFrame이나 Series 객체에서 바로 호출할 수 있는 간편한 시각화 기능(`.plot()`)을 제공합니다. 이는 데이터 탐색 초기 단계에서 복잡한 코드 없이 데이터의 패턴을 빠르게 확인하는 데 매우 유용합니다.

**`.plot()` 사용법**
- **기본 호출**: `df.plot()`는 가장 기본적인 선 그래프를 생성합니다.
- **`kind` 파라미터**: `df.plot(kind='bar')`와 같이 `kind` 파라미터로 그래프 종류를 지정할 수 있습니다.
- **접근자 스타일**: `df.plot.bar()`, `df.plot.hist()`처럼 `.plot` 다음에 원하는 그래프 종류를 직접 메서드로 호출하는 방식이 더 명시적이고 권장됩니다.

---
**주요 그래프 종류 및 예제**

```python
import matplotlib.pyplot as plt

# 시각화를 위한 데이터 생성
np.random.seed(42)
df_plot = pd.DataFrame({
    'A': np.random.randn(100).cumsum(),
    'B': np.random.randn(100).cumsum() + 50,
    'C': np.random.rand(100) * 50
}, index=pd.date_range('2023-01-01', periods=100))
```

**1. 선 그래프 (Line Plot)**
- **목적**: 시간의 흐름에 따른 데이터의 추세나 연속적인 변화를 보여주는 데 적합합니다.
- **특징**: `plot()`의 기본 형태로, 인덱스가 x축, 각 컬럼의 값이 y축이 됩니다.

```python
# A와 B 컬럼의 시계열 추세 확인
df_plot[['A', 'B']].plot(
    figsize=(10, 6),
    title='Time Series Line Plot',
    xlabel='Date',
    ylabel='Value',
    grid=True
)
plt.show()
```

**2. 막대 그래프 (Bar Plot)**
- **목적**: 범주형 데이터의 크기를 비교하는 데 사용됩니다.
- **특징**: `.plot.bar()`는 수직 막대, `.plot.barh()`는 수평 막대 그래프를 그립니다.

```python
df_bar = pd.DataFrame({'Count': [10, 20, 15, 25]}, index=['Cat A', 'Cat B', 'Cat C', 'Cat D'])
df_bar.plot.bar(figsize=(8, 5), title='Bar Plot of Counts', rot=0) # rot=0: x축 라벨 회전 안 함
plt.ylabel('Count')
plt.show()
```

**3. 히스토그램 (Histogram)**
- **목적**: 단일 수치형 변수의 데이터 분포(frequency distribution)를 확인합니다.
- **특징**: `.plot.hist()`를 사용하며, `bins` 파라미터로 계급(구간)의 개수를 조절합니다.

```python
# C 컬럼의 데이터 분포 확인
df_plot['C'].plot.hist(bins=20, figsize=(8, 5), title='Histogram of C')
plt.xlabel('Value')
plt.show()
```

**4. 박스 플롯 (Box Plot)**
- **목적**: 데이터의 사분위수, 중앙값, 이상치 등 통계적 분포를 시각화합니다. 여러 그룹 간의 분포를 비교하는 데 매우 유용합니다.
- **특징**: `.plot.box()`를 사용합니다.

```python
df_plot[['A', 'B', 'C']].plot.box(figsize=(8, 5), title='Box Plot of A, B, C')
plt.ylabel('Value')
plt.show()
```

**5. 산점도 (Scatter Plot)**
- **목적**: 두 수치형 변수 간의 관계(상관관계)를 파악하는 데 사용됩니다.
- **특징**: `.plot.scatter()`를 사용하며, `x`와 `y`축에 해당하는 컬럼을 반드시 지정해야 합니다.

```python
# A와 B 컬럼 간의 상관관계 확인
df_plot.plot.scatter(x='A', y='B', figsize=(8, 5), title='Scatter Plot of A vs B')
plt.show()
```

**6. 커스터마이징**
Pandas의 `.plot()` 메서드는 `matplotlib.axes.Axes` 객체를 반환하므로, `matplotlib`의 함수들을 연달아 사용하여 그래프를 세밀하게 꾸밀 수 있습니다.

```python
# .plot()으로 기본 그래프를 그리고, plt로 추가 설정
ax = df_plot['A'].plot(title='Customized Plot')
ax.set_xlabel("Date Axis")
ax.set_ylabel("Value Axis")
ax.legend(["Series A"])
plt.show()
```
이처럼 Pandas의 내장 시각화는 데이터 분석 과정에서 빠르고 간편하게 인사이트를 얻는 데 효과적인 도구입니다.

### 3. 실무 활용 시나리오

Pandas의 표시 옵션과 내장 시각화 기능은 데이터 분석 워크플로우의 여러 단계에서 실용적으로 활용됩니다.

**1. 탐색적 데이터 분석 (EDA: Exploratory Data Analysis)**

-   **목적**: 데이터에 대한 초기 이해를 높이고, 숨겨진 패턴, 이상치, 변수 간 관계 등을 발견하여 다음 분석 단계(전처리, 모델링)의 방향을 설정합니다.
-   **활용**:
    -   **표시 옵션**: `pd.set_option('display.max_rows', None)` 등으로 전체 데이터를 확인하며 이상한 값이나 패턴을 육안으로 빠르게 스캔합니다.
    -   **내장 시각화**: `df.plot.hist()`, `df.plot.box()`, `df.plot.scatter()` 등을 사용하여 각 변수의 분포, 변수 간 관계, 이상치 여부를 빠르게 시각적으로 확인하여 직관을 얻습니다. 예를 들어, `df['price'].plot.hist()`로 가격 분포를 보고, `df.plot.scatter(x='size', y='price')`로 크기와 가격의 관계를 즉시 파악할 수 있습니다.

**2. 디버깅 및 데이터 검토**

-   **목적**: 데이터 처리 과정에서 예상치 못한 결과가 나왔을 때, 중간 단계의 데이터를 자세히 검토하여 문제의 원인을 찾습니다.
-   **활용**:
    -   **표시 옵션**: `pd.set_option('display.float_format', '{:.4f}'.format)`로 소수점 정밀도를 높여 미세한 값의 변화를 확인하거나, `pd.set_option('display.max_colwidth', None)`으로 긴 텍스트 컬럼의 내용이 잘리지 않도록 하여 데이터 오류를 찾습니다.
    -   **`pd.option_context`**: 특정 코드 블록에서만 임시로 표시 옵션을 변경하여, 디버깅에 필요한 정보만 집중적으로 확인하고 블록 종료 후에는 자동으로 원래 설정으로 돌아가도록 합니다.

**3. 빠른 보고서 및 커뮤니케이션**

-   **목적**: 분석 결과를 동료나 비전문가에게 빠르고 명확하게 전달합니다.
-   **활용**:
    -   **내장 시각화**: 복잡한 시각화 도구를 사용하지 않고도 `df.plot.bar()`, `df.plot.pie()` 등으로 핵심적인 통계나 비율을 직관적인 그래프로 빠르게 생성하여 보고서에 첨부하거나 발표 자료로 활용합니다. 예를 들어, `df.groupby('category')['sales'].sum().plot.bar()`로 카테고리별 총 판매량을 즉시 시각화할 수 있습니다.

### 4. 요약 및 모범 사례

Pandas의 표시 옵션과 내장 시각화 기능은 데이터 분석가의 생산성을 크게 향상시키는 도구입니다. 이들을 효과적으로 활용하기 위한 핵심 요약과 모범 사례는 다음과 같습니다.

**1. 핵심 요약**

-   **Pandas 표시 옵션 (`pd.set_option`)**:
    -   **목적**: DataFrame 출력 방식을 제어하여 데이터 탐색 및 디버깅 효율성을 높입니다.
    -   **주요 옵션**: `display.max_rows`, `display.max_columns` (표시 개수), `display.width`, `display.max_colwidth` (너비), `display.precision`, `display.float_format` (숫자 형식).
    -   **활용**: 대용량 데이터의 전체 구조 파악, 특정 컬럼 내용 상세 검토, 숫자 값의 정밀한 확인.

-   **Pandas 내장 시각화 (`.plot()` 접근자)**:
    -   **목적**: Matplotlib 기반으로 데이터를 빠르고 간편하게 시각화하여 패턴을 파악합니다.
    -   **주요 그래프**: 선, 막대, 히스토그램, 박스, 산점도 등 다양한 기본 그래프를 지원합니다.
    -   **활용**: 탐색적 데이터 분석(EDA) 초기 단계에서 데이터 분포, 추세, 변수 간 관계를 시각적으로 확인.

**2. 모범 사례 (Best Practices)**

-   **옵션 설정의 일관성 유지**: 프로젝트나 분석 세션 초기에 자주 사용하는 `set_option`들을 한 번에 설정하여 일관된 작업 환경을 만듭니다.
    ```python
    # 예시: 분석 시작 시 공통 설정
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.3f}'.format)
    ```

-   **`pd.option_context`를 활용한 임시 설정**: 전역 설정을 변경하는 대신, 특정 코드 블록에서만 임시로 옵션을 변경해야 할 때는 `with pd.option_context(...)`를 사용합니다. 이는 다른 코드나 분석에 영향을 주지 않아 안전합니다.

-   **시각화 목적에 따른 도구 선택**:
    -   **빠른 탐색 및 초기 인사이트**: Pandas 내장 `.plot()`이 가장 효율적입니다.
    -   **심층 분석 및 보고서용 그래프**: `Seaborn` (통계 그래프), `Matplotlib` (세밀한 커스터마이징)을 직접 사용하는 것이 좋습니다.
    -   **대화형(Interactive) 시각화**: `Plotly`, `Bokeh` 등 전문 라이브러리를 고려합니다.

-   **코드의 재사용성 고려**: 자주 사용하는 시각화 패턴이나 표시 옵션 조합은 함수로 만들어 재사용성을 높입니다.

-   **주석 및 문서화**: 복잡한 표시 옵션 설정이나 시각화 코드에는 주석을 달아 의도를 명확히 하고, 분석 결과는 적절한 시각화와 함께 문서화하여 공유합니다.

이러한 모범 사례들을 따르면 Pandas를 활용한 데이터 분석 및 시각화 작업을 더욱 효율적이고 체계적으로 수행할 수 있습니다.


### 5. Pandas 코드 디버깅 팁 (Debugging Pandas Code Tips)

Pandas를 이용한 데이터 분석 및 전처리 과정에서는 다양한 오류와 예상치 못한 결과에 직면할 수 있습니다. 효과적인 디버깅은 문제의 원인을 빠르게 파악하고 해결하는 데 필수적입니다.

**1. `SettingWithCopyWarning` 이해 및 해결**

`SettingWithCopyWarning`은 Pandas에서 흔히 발생하는 경고 중 하나로, 원본 `DataFrame`의 '뷰(view)'에 대해 작업을 수행할 때 발생합니다. 이는 사용자가 복사본을 수정하고 있다고 착각할 수 있음을 알려주며, 실제로는 원본 `DataFrame`이 변경되지 않거나 예상치 못한 방식으로 변경될 수 있음을 경고합니다.

*   **원인**: `DataFrame`의 일부를 선택(슬라이싱, 불리언 인덱싱 등)한 후, 그 결과에 다시 값을 할당할 때 발생합니다. Pandas가 반환한 것이 원본의 '뷰'인지 '복사본'인지 모호할 때 나타납니다.
*   **해결책**:
    *   **`.loc` 사용**: 가장 권장되는 방법입니다. 명시적으로 `.loc`를 사용하여 행과 열을 동시에 선택하고 값을 할당하면, Pandas는 이것이 원본 `DataFrame`에 대한 직접적인 할당임을 인식합니다.
    ```python
    import pandas as pd
    df_warn = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # 경고 발생 가능성 있는 코드
    # df_warn[df_warn['A'] > 1]['B'] = 99 # SettingWithCopyWarning 발생 가능

    # 올바른 해결책: .loc 사용
    df_warn.loc[df_warn['A'] > 1, 'B'] = 99
    print("SettingWithCopyWarning 해결 후:\n", df_warn)
    ```
    *   **`.copy()` 사용**: 명시적으로 복사본을 만들어 작업합니다. 원본 `DataFrame`을 변경하고 싶지 않을 때 유용합니다.
    ```python
    df_copy_example = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    df_subset = df_copy_example[df_copy_example['A'] > 1].copy() # .copy() 추가
    df_subset['B'] = 99
    print("\n.copy() 사용 후 (원본은 변경되지 않음):\n", df_copy_example)
    print(

