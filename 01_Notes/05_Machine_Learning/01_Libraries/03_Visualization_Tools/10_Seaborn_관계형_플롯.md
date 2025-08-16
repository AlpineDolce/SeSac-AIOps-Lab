<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 관계형 플롯(`relplot()`, `scatterplot()`, `lineplot()`)을 사용하여 두 개 이상의 변수 간의 통계적 관계를 시각화하는 방법을 다룹니다. 산점도와 라인 플롯을 통해 데이터의 분포, 추세, 그리고 추가적인 변수(`hue`, `size`, `style`)를 반영하여 관계를 심층적으로 분석하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. Seaborn 관계형 플롯 개요](#1-seaborn-관계형-플롯-개요)
  - [1.1. `relplot()`: 관계형 플롯의 일반적인 인터페이스](#11-relplot-관계형-플롯의-일반적인-인터페이스)
    - [1.1.1. `kind` 매개변수를 이용한 산점도 및 라인 플롯](#111-kind-매개변수를-이용한-산점도-및-라인-플롯)
    - [1.1.2. `col`, `row` 매개변수를 이용한 서브플롯 생성](#112-col-row-매개변수를-이용한-서브플롯-생성)
  - [1.2. `scatterplot()`: 산점도](#12-scatterplot-산점도)
    - [1.2.1. 기본 산점도](#121-기본-산점도)
    - [1.2.2. `hue`, `size`, `style` 매개변수를 이용한 다변수 시각화](#122-hue-size-style-매개변수를-이용한-다변수-시각화)
  - [1.3. `lineplot()`: 라인 플롯](#13-lineplot-라인-플롯)
    - [1.3.1. 기본 라인 플롯 및 추세선](#131-기본-라인-플롯-및-추세선)
    - [1.3.2. `hue`, `style`, `units` 매개변수를 이용한 그룹별 추세 시각화](#132-hue-style-units-매개변수를-이용한-그룹별-추세-시각화)
    - [1.3.3. 신뢰 구간(Confidence Interval) 설정](#133-신뢰-구간confidence-interval-설정)

<h2>Seaborn 플롯의 기본 개념: Figure-level vs. Axes-level 함수</h2>

Seaborn은 Matplotlib을 기반으로 구축되었으며, 플롯을 그리는 방식에 따라 크게 두 가지 유형의 함수를 제공합니다: **Figure-level 함수**와 **Axes-level 함수**. 이 두 가지 개념을 이해하는 것은 Seaborn을 효과적으로 사용하고 Matplotlib과 연동하는 데 매우 중요합니다.

### Figure-level 함수
*   **특징:** `FacetGrid` 또는 `PairGrid`와 같은 내부 객체를 사용하여 전체 그림(Figure)을 관리하고, 여러 개의 서브플롯(Axes)을 자동으로 생성합니다. 사용자가 직접 `plt.figure()`나 `plt.subplot()`을 호출할 필요가 없습니다.
*   **장점:** 복잡한 다변량 관계나 조건부 시각화를 쉽게 구현할 수 있습니다. `col`, `row`, `hue`와 같은 매개변수를 통해 데이터를 분할하여 여러 서브플롯에 동일한 유형의 플롯을 그릴 수 있습니다.
*   **예시:** `relplot()`, `displot()`, `catplot()`, `lmplot()`, `pairplot()`, `jointplot()` (Seaborn 0.11.0 이전 버전의 `jointplot()`은 Axes-level이었으나, 현재는 Figure-level로 동작).
*   **주의사항:** Figure-level 함수는 자체적으로 Figure와 Axes를 생성하므로, 기존 Matplotlib Axes에 직접 플롯을 추가하는 데는 적합하지 않습니다. `plt.show()`를 호출하기 전에 `plt.suptitle()`을 사용하여 전체 Figure의 제목을 설정할 수 있습니다.

### Axes-level 함수
*   **특징:** 특정 Matplotlib Axes 객체에 플롯을 그립니다. 사용자가 직접 `plt.figure()`와 `plt.subplot()`을 사용하여 Figure와 Axes를 생성하고, 해당 Axes 객체를 `ax` 매개변수를 통해 함수에 전달해야 합니다.
*   **장점:** 기존 Matplotlib 플롯에 Seaborn의 스타일과 기능을 추가하거나, 여러 종류의 플롯을 하나의 Figure 내에서 조합할 때 유용합니다. Matplotlib의 유연한 서브플롯 레이아웃을 그대로 활용할 수 있습니다.
*   **예시:** `scatterplot()`, `lineplot()`, `histplot()`, `kdeplot()`, `boxplot()`, `violinplot()`, `stripplot()`, `swarmplot()`, `barplot()`, `countplot()`, `regplot()`, `heatmap()`, `clustermap()`.
*   **주의사항:** Axes-level 함수는 `col`, `row`와 같은 `FacetGrid` 기반의 조건부 플로팅 기능을 직접 제공하지 않습니다. 이러한 기능이 필요할 경우 `FacetGrid`나 `PairGrid` 객체의 `map()` 메서드를 사용해야 합니다.

**언제 어떤 함수를 사용할까?**
*   **Figure-level 함수:** 데이터셋의 복잡한 관계를 여러 서브플롯에 걸쳐 탐색하거나, 특정 범주형 변수에 따라 데이터를 분할하여 비교할 때 유용합니다. 빠른 탐색적 데이터 분석에 적합합니다.
*   **Axes-level 함수:** 기존 Matplotlib 플롯에 Seaborn의 시각화 기능을 추가하거나, 하나의 Figure 내에서 여러 플롯을 정교하게 조합하여 맞춤형 시각화를 만들 때 유용합니다.

---

## 1. Seaborn 관계형 플롯 개요

Seaborn의 관계형 플롯(Relational Plots)은 두 개 이상의 변수 간의 통계적 관계를 시각화하는 데 특화되어 있습니다. 데이터의 분포, 추세, 그리고 추가적인 변수(`hue`, `size`, `style` 등)를 반영하여 관계를 심층적으로 분석할 수 있도록 돕습니다. 주요 함수로는 `relplot()`, `scatterplot()`, `lineplot()`이 있습니다.

### 1.1. `relplot()`: 관계형 플롯의 일반적인 인터페이스

`relplot()`은 Seaborn의 관계형 플롯을 위한 상위 레벨(figure-level) 인터페이스입니다. `scatterplot()`과 `lineplot()`의 기능을 모두 포함하며, `FacetGrid`를 사용하여 여러 서브플롯을 쉽게 생성하고 비교할 수 있게 해줍니다.

#### 1.1.1. `kind` 매개변수를 이용한 산점도 및 라인 플롯

`relplot()`의 `kind` 매개변수를 `'scatter'` 또는 `'line'`로 설정하여 산점도 또는 라인 플롯을 그릴 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips") # 팁 데이터셋 로드

# kind='scatter' (기본값)으로 산점도 그리기
sns.relplot(x="total_bill", y="tip", data=tips, kind="scatter")
plt.suptitle("relplot (kind='scatter'): Total Bill vs. Tip", y=1.02) # 전체 figure 제목
plt.savefig('relplot_scatter_basic.png') # 그래프를 이미지 파일로 저장
plt.show()
plt.clf() # 현재 figure를 닫아 메모리 해제

fmri = sns.load_dataset("fmri") # fmri 데이터셋 로드

# kind='line'으로 라인 플롯 그리기
sns.relplot(x="timepoint", y="signal", data=fmri, kind="line")
plt.suptitle("relplot (kind='line'): Signal Change Over Time", y=1.02)
plt.savefig('relplot_line_basic.png') # 그래프를 이미지 파일로 저장
plt.show()
plt.clf() # 현재 figure를 닫아 메모리 해제
```

#### 1.1.2. `col`, `row` 매개변수를 이용한 서브플롯 생성

`relplot()`의 강력한 기능 중 하나는 `col` 또는 `row` 매개변수를 사용하여 특정 범주형 변수에 따라 데이터를 분할하고, 각 서브플롯에 해당 범주의 데이터를 시각화하는 것입니다. 이는 여러 그룹 간의 관계를 쉽게 비교할 수 있게 해줍니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# 흡연 여부(smoker)에 따라 열(col)을 나누어 산점도 그리기
sns.relplot(x="total_bill", y="tip", hue="smoker", col="smoker", data=tips)
plt.suptitle("relplot: Total Bill vs. Tip by Smoker (col)", y=1.02)
plt.show()

# 요일(day)에 따라 행(row)을 나누어 라인 플롯 그리기
sns.relplot(x="timepoint", y="signal", hue="event", row="region", data=fmri, kind="line")
plt.suptitle("relplot: Signal Change by Region (row)", y=1.02)
plt.show()
```

### 1.2. `scatterplot()`: 산점도

`scatterplot()`은 두 연속형 변수 간의 관계를 점으로 표현하는 함수입니다. `relplot(kind='scatter')`와 동일한 기능을 제공하지만, `scatterplot()`은 Axes-level 함수이므로 기존 Matplotlib Axes에 직접 플롯을 추가할 때 유용합니다.

#### 1.2.1. 기본 산점도

두 변수 간의 기본적인 관계를 파악하는 데 사용됩니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# 총 계산액(total_bill)과 팁(tip)의 관계를 산점도로 표현
plt.figure(figsize=(7, 5))
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.title("scatterplot: Total Bill vs. Tip (Basic)")
plt.savefig('scatterplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()
plt.clf() # 현재 figure를 닫아 메모리 해제
```

#### 1.2.2. `hue`, `size`, `style` 매개변수를 이용한 다변수 시각화

`scatterplot()`은 `hue`, `size`, `style` 매개변수를 사용하여 세 개 이상의 변수를 동시에 시각화할 수 있습니다.

*   `hue`: 범주형 변수에 따라 점의 색상을 다르게 합니다.
*   `size`: 연속형 또는 범주형 변수에 따라 점의 크기를 다르게 합니다.
*   `style`: 범주형 변수에 따라 점의 마커 스타일을 다르게 합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(10, 7))

# 성별(sex)에 따라 색상을 다르게 표현
sns.scatterplot(x="total_bill", y="tip", hue="sex", data=tips)
plt.title("scatterplot: Total Bill vs. Tip by Sex (hue)")
plt.show()

plt.figure(figsize=(10, 7))
# 요일(day)에 따라 점의 크기(size)를, 시간(time)에 따라 점의 스타일(style)을 다르게 표현
sns.scatterplot(x="total_bill", y="tip", hue="day", size="size", style="time", data=tips, sizes=(20, 400))
plt.title("scatterplot: Total Bill vs. Tip by Day, Size, and Time (hue, size, style)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 범례 위치 조정
plt.tight_layout()
plt.show()
```
#### 1.2.3. 밀집 데이터 시각화: `x_bins`, `y_bins` 활용
산점도에서 데이터 포인트가 너무 많아 서로 겹쳐 보이는 오버플로팅(overplotting) 문제가 발생할 경우, `scatterplot()` 함수의 `x_bins` 또는 `y_bins` 매개변수를 사용하여 데이터를 집계하고 밀도를 시각화할 수 있습니다. 이는 2D 히스토그램과 유사하게 작동하여 데이터의 밀집된 영역을 파악하는 데 도움을 줍니다.

*   `x_bins`: x축을 따라 데이터를 나눌 구간(bin)의 개수 또는 구간 경계를 지정합니다.
*   `y_bins`: y축을 따라 데이터를 나눌 구간(bin)의 개수 또는 구간 경계를 지정합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 대규모 데이터 생성 (오버플로팅 예시)
np.random.seed(0)
x_dense = np.random.randn(5000)
y_dense = np.random.randn(5000) + x_dense * 0.5

plt.figure(figsize=(8, 6))
# x_bins와 y_bins를 사용하여 밀집된 산점도 시각화
sns.scatterplot(x=x_dense, y=y_dense, s=10, alpha=0.5, x_bins=50, y_bins=50, cmap='viridis') # s: 점의 크기, alpha: 투명도
plt.title('Scatter Plot with Bins for Dense Data')
plt.xlabel('X-value')
plt.ylabel('Y-value')
plt.colorbar(label='Count in bin')
plt.show()
```

### 1.3. `lineplot()`: 라인 플롯

`lineplot()`은 주로 시계열 데이터나 연속적인 데이터의 추세를 보여줄 때 사용됩니다. 기본적으로 각 x 값에 대한 y 값의 평균과 신뢰 구간을 표시합니다.

#### 1.3.1. 기본 라인 플롯 및 추세선

시간에 따른 변화나 연속적인 데이터의 추세를 시각화합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

fmri = sns.load_dataset("fmri")

# 시간(timepoint)에 따른 신호(signal) 변화를 라인 플롯으로 표현
plt.figure(figsize=(10, 6))
sns.lineplot(x="timepoint", y="signal", data=fmri)
plt.title("lineplot: Signal Change Over Time (Basic)")
plt.savefig('lineplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()
plt.clf() # 현재 figure를 닫아 메모리 해제
```

#### 1.3.2. `hue`, `style`, `units` 매개변수를 이용한 그룹별 추세 시각화

`lineplot()`도 `hue`, `style`, `units` 매개변수를 사용하여 여러 그룹의 추세를 동시에 비교할 수 있습니다.

*   `hue`: 범주형 변수에 따라 라인의 색상을 다르게 합니다.
*   `style`: 범주형 변수에 따라 라인의 스타일(점선, 실선 등)이나 마커를 다르게 합니다.
*   `units`: 개별 관측치를 나타내는 변수를 지정하여, 각 단위별로 라인을 그릴 수 있습니다. 신뢰 구간 계산에 영향을 줍니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

fmri = sns.load_dataset("fmri")

plt.figure(figsize=(10, 6))
# 이벤트(event)와 지역(region)에 따라 라인 분리
sns.lineplot(x="timepoint", y="signal", hue="event", style="region", data=fmri)
plt.title("lineplot: Signal Change by Event and Region (hue, style)")
plt.show()

# subject별 개별 라인 그리기 (units 사용)
plt.figure(figsize=(10, 6))
sns.lineplot(x="timepoint", y="signal", hue="event", units="subject", estimator=None, lw=1, data=fmri.loc[fmri['region'] == 'frontal']) # estimator=None: 집계하지 않고 개별 라인 표시, lw: 라인 두께
plt.title("lineplot: Individual Subject Lines (units)")
plt.show()
```

#### 1.3.3. 신뢰 구간(Confidence Interval) 설정

`lineplot()`은 기본적으로 신뢰 구간(Confidence Interval, CI)을 음영 처리된 영역으로 표시합니다. `ci` 매개변수를 사용하여 신뢰 구간의 크기를 조절하거나 표시하지 않을 수 있습니다.

*   `ci`: 신뢰 구간의 크기를 지정합니다 (예: `95` for 95% CI). `None`으로 설정하면 신뢰 구간을 표시하지 않습니다.
*   `estimator`: 각 x 값에 대한 y 값의 통계량을 지정합니다 (기본값: `mean`).

```python
import seaborn as sns
import matplotlib.pyplot as plt

fmri = sns.load_dataset("fmri")

plt.figure(figsize=(10, 6))
# 신뢰 구간 표시 안 함
sns.lineplot(x="timepoint", y="signal", data=fmri, ci=None)
plt.title("lineplot: Signal Change (No Confidence Interval)")
plt.show()

plt.figure(figsize=(10, 6))
# 68% 신뢰 구간 표시
sns.lineplot(x="timepoint", y="signal", data=fmri, ci=68)
plt.title("lineplot: Signal Change (68% Confidence Interval)")
plt.show()
```

#### 1.3.4. `errorbar` 매개변수를 이용한 오차 표현
Seaborn 0.12.0 버전부터 `lineplot()`의 `ci` 매개변수 대신 `errorbar` 매개변수가 도입되어 오차 표현에 대한 더 유연하고 명시적인 제어를 제공합니다. `errorbar`는 오차를 계산하고 시각화하는 방식을 지정합니다.

*   **`errorbar`**:
    *   `"sd"`: 표준 편차(standard deviation)를 오차로 표시합니다.
    *   `"se"`: 표준 오차(standard error)를 오차로 표시합니다.
    *   `("pi", 95)`: 95% 예측 구간(prediction interval)을 표시합니다.
    *   `("ci", 95)`: 95% 신뢰 구간(confidence interval)을 표시합니다 (기존 `ci`와 동일).
    *   튜플 `(estimator, error_measure)` 형태로 사용자 정의 함수를 전달할 수도 있습니다.
    *   `None`: 오차 막대를 표시하지 않습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

fmri = sns.load_dataset("fmri")

plt.figure(figsize=(10, 6))
# 표준 편차를 오차로 표시
sns.lineplot(x="timepoint", y="signal", data=fmri, errorbar="sd")
plt.title("lineplot: Signal Change (Standard Deviation Errorbar)")
plt.show()

plt.figure(figsize=(10, 6))
# 표준 오차를 오차로 표시
sns.lineplot(x="timepoint", y="signal", data=fmri, errorbar="se")
plt.title("lineplot: Signal Change (Standard Error Errorbar)")
plt.show()

plt.figure(figsize=(10, 6))
# 95% 예측 구간 표시
sns.lineplot(x="timepoint", y="signal", data=fmri, errorbar=("pi", 95))
plt.title("lineplot: Signal Change (95% Prediction Interval)")
plt.show()
```

