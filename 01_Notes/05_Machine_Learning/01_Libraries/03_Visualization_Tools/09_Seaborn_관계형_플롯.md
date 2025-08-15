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
plt.show()

fmri = sns.load_dataset("fmri") # fmri 데이터셋 로드

# kind='line'으로 라인 플롯 그리기
sns.relplot(x="timepoint", y="signal", data=fmri, kind="line")
plt.suptitle("relplot (kind='line'): Signal Change Over Time", y=1.02)
plt.show()
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
plt.show()
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
plt.show()
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
sns.lineplot(x="timepoint", y="signal", hue="event", units="subject", estimator=None, lw=1, data=fmri.loc[fmri['region'] == 'frontal'])
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
