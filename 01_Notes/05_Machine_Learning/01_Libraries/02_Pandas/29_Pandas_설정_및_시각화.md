<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
 Pandas의 `pd.set_option()`을 사용하여 DataFrame의 표시 방식을 사용자 정의하고, `.plot()` 접근자를 활용한 내장 시각화 기능을 통해 데이터를 빠르게 탐색하고 시각적으로 분석하는 실무 역량을 강화합니다.

<h2>목차</h2>

---

### 1. Pandas 표시 옵션 설정 (`pd.set_option`)

`pd.set_option()` 함수를 사용하면 Pandas의 다양한 전역(global) 표시 옵션을 설정할 수 있습니다. 이는 DataFrame이 출력될 때의 행/열 개수, 소수점 정밀도, 컬럼 너비 등을 제어합니다.

-   **기본 사용법**: `pd.set_option('option_name', value)`
-   **현재 설정 확인**: `pd.get_option('option_name')`
-   **기본값으로 리셋**: `pd.reset_option('option_name')` 또는 `pd.reset_option('all')`

```python
import pandas as pd
import numpy as np

# 예시 DataFrame 생성
df = pd.DataFrame(np.random.rand(10, 5), columns=[f'col_{i}' for i in range(5)])
df['long_text_column'] = ['This is a very long text string that might be truncated in display.' for _ in range(10)]
print("Original DataFrame (default display):\n", df)

# 1. 최대 표시 행/열 설정
# 모든 행을 표시 (기본값은 10개 정도)
pd.set_option('display.max_rows', None)
# 모든 열을 표시 (기본값은 20개 정도)
pd.set_option('display.max_columns', None)
print("\nDataFrame with max_rows/columns set to None:\n", df)

# 2. 출력 너비 설정 (컬럼이 잘리지 않도록)
pd.set_option('display.width', 1000) # 터미널/콘솔 너비에 맞춰 조절
print("\nDataFrame with increased display.width:\n", df)

# 3. 소수점 정밀도 설정
pd.set_option('display.precision', 4) # 소수점 4자리까지 표시
print("\nDataFrame with precision set to 4:\n", df)

# 4. 부동 소수점 형식 지정
pd.set_option('display.float_format', '{:.2f}'.format) # 소수점 2자리까지 표시
print("\nDataFrame with float_format set to 2 decimal places:\n", df)

# 5. 컬럼 헤더 정렬
pd.set_option('display.colheader_justify', 'left') # 컬럼 헤더 왼쪽 정렬
print("\nDataFrame with left-justified column headers:\n", df)

# 모든 옵션 리셋
pd.reset_option('all')
print("\nDataFrame after resetting all options:\n", df)
```

### 2. Pandas 내장 시각화 (`.plot()` 접근자)

Pandas DataFrame과 Series는 Matplotlib을 기반으로 하는 `.plot()` 접근자를 제공하여 데이터를 빠르고 쉽게 시각화할 수 있습니다. 이는 복잡한 시각화 코드를 작성할 필요 없이 데이터의 패턴을 즉시 파악하는 데 유용합니다.

```python
import matplotlib.pyplot as plt

# 시각화를 위한 데이터 생성
np.random.seed(42)
df_plot = pd.DataFrame({
    'A': np.random.randn(100).cumsum(),
    'B': np.random.randn(100).cumsum() + 10,
    'C': np.random.randn(100).cumsum() - 10
}, index=pd.date_range('2023-01-01', periods=100))

# 1. 선 그래프 (Line Plot)
df_plot[['A', 'B']].plot(figsize=(10, 6), title='Line Plot of A and B')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# 2. 막대 그래프 (Bar Plot)
df_bar = pd.DataFrame({'Count': [10, 20, 15, 25]}, index=['Cat A', 'Cat B', 'Cat C', 'Cat D'])
df_bar.plot.bar(figsize=(8, 5), title='Bar Plot of Counts')
plt.ylabel('Count')
plt.show()

# 3. 히스토그램 (Histogram)
df_plot['A'].plot.hist(bins=20, figsize=(8, 5), title='Histogram of A')
plt.xlabel('Value')
plt.show()

# 4. 산점도 (Scatter Plot)
df_scatter = pd.DataFrame(np.random.rand(50, 2), columns=['X', 'Y'])
df_scatter.plot.scatter(x='X', y='Y', figsize=(8, 5), title='Scatter Plot of X vs Y')
plt.show()

# 5. 박스 플롯 (Box Plot)
df_box = pd.DataFrame(np.random.rand(50, 3), columns=['Group1', 'Group2', 'Group3'])
df_box.plot.box(figsize=(8, 5), title='Box Plot of Groups')
plt.ylabel('Value')
plt.show()

# 6. 파이 차트 (Pie Chart)
df_pie = pd.Series([30, 20, 50], index=['Apple', 'Banana', 'Cherry'])
df_pie.plot.pie(figsize=(7, 7), autopct='%1.1f%%', title='Fruit Distribution')
plt.ylabel('') # y축 라벨 제거
plt.show()
```

### 3. 실무 활용 시나리오

-   **탐색적 데이터 분석 (EDA)**: 데이터의 초기 단계에서 분포, 추세, 이상치 등을 빠르게 시각적으로 확인하여 데이터에 대한 직관을 얻습니다.
-   **디버깅 및 데이터 검토**: DataFrame의 전체 내용을 확인하거나 특정 컬럼의 값을 자세히 검토할 때 표시 옵션을 활용합니다.
-   **빠른 보고서 작성**: 내부 보고서나 동료와의 공유를 위해 데이터의 핵심 내용을 빠르게 시각화하여 전달합니다.

### 4. 요약 및 모범 사례

-   **표시 옵션**: `pd.set_option()`을 통해 DataFrame 출력 방식을 제어하여 데이터 탐색 및 디버깅 효율성을 높일 수 있습니다. 특히 `display.max_rows`, `display.max_columns`, `display.float_format`은 자주 사용됩니다.
-   **내장 시각화**: `.plot()` 접근자는 Matplotlib을 직접 사용하는 것보다 훨씬 간결한 코드로 다양한 그래프를 생성할 수 있게 합니다. 이는 EDA 단계에서 데이터의 시각적 패턴을 빠르게 파악하는 데 매우 유용합니다.
-   **한계**: Pandas 내장 시각화는 빠른 탐색에 적합하지만, 고도로 커스터마이징된 복잡한 그래프나 대화형 시각화가 필요할 때는 Matplotlib, Seaborn, Plotly 등 전문 시각화 라이브러리를 직접 사용하는 것이 좋습니다.

