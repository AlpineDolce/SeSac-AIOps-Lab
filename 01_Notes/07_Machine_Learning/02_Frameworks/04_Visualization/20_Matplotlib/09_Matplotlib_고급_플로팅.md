<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 고급 플로팅 기능인 `fill_between()`, `stackplot()`, `stem()`을 다룹니다. 두 곡선 사이의 영역을 채우거나, 누적 영역 차트를 생성하거나, 이산 데이터를 스템 플롯으로 표현하는 등 특정 시나리오에서 데이터를 효과적으로 시각화하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

* [1. Matplotlib 고급 플로팅](#1-matplotlib-고급-플로팅)
  * [1.1. `fill_between()`: 두 곡선 사이 영역 채우기](#11-fill_between-두-곡선-사이-영역-채우기)
  * [1.2. `stackplot()`: 누적 영역형 차트](#12-stackplot-누적-영역형-차트)
  * [1.3. `stem()`: 스템 플롯](#13-stem-스템-플롯)

---

## 1. Matplotlib 고급 플로팅

### 1.1. `fill_between()`: 두 곡선 사이 영역 채우기
`fill_between()` 함수는 두 개의 수평 곡선 사이의 영역을 색상으로 채우는 데 사용됩니다. 이는 두 데이터 시리즈 간의 차이를 시각적으로 강조하거나, 신뢰 구간이나 오차 범위를 표현하는 데 매우 유용합니다.

**주요 특징:**
*   **범위 시각화:** `y1`과 `y2` 두 배열을 받아 `x`축에 대해 두 곡선 사이의 공간을 채웁니다.
*   **조건부 채우기:** `where` 파라미터를 사용하여 특정 조건을 만족하는 영역만 선택적으로 채울 수 있습니다.
*   **신뢰 구간 표현:** 평균을 나타내는 곡선을 그리고, 그 위아래로 표준편차나 신뢰 구간에 해당하는 영역을 `fill_between`으로 채워서 데이터의 불확실성을 효과적으로 보여줄 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.sin(x) + 0.5

plt.figure(figsize=(10, 6))

# 기본 라인 플롯
plt.plot(x, y1, label='Sine Wave')
plt.plot(x, y2, label='Shifted Sine Wave')

# 두 곡선 사이 영역 채우기
plt.fill_between(x, y1, y2, color='skyblue', alpha=0.4, label='Difference')

plt.title('fill_between() Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

# 신뢰 구간 시각화 예시
mean = np.sin(x)
std_dev = 0.2

plt.figure(figsize=(10, 6))
plt.plot(x, mean, label='Mean')
plt.fill_between(x, mean - std_dev, mean + std_dev, color='lightgreen', alpha=0.5, label='Confidence Interval (±1 std dev)')
plt.title('Confidence Interval Visualization')
plt.legend()
plt.show()
```

### 1.2. `stackplot()`: 누적 영역형 차트
`stackplot`은 여러 데이터 시리즈의 값을 누적하여 그리는 영역형 차트입니다. 시간의 흐름이나 연속적인 축에 따라 전체 합계와 그 안에서 각 부분이 차지하는 비중의 변화를 함께 보여줄 때 매우 효과적입니다.

**주요 특징:**
*   **부분-전체 관계:** 각 시점에서 전체 합계가 어떻게 구성되는지 쉽게 파악할 수 있습니다.
*   **추세 비교:** 각 데이터 시리즈의 변화 추세와 함께 전체적인 추세를 비교 분석할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.arange(0, 10)
y1 = np.array([1, 3, 4, 5, 7, 6, 8, 9, 10, 11])
y2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y3 = np.array([2, 4, 3, 5, 6, 7, 8, 9, 10, 11])

labels = ["Series 1", "Series 2", "Series 3"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

plt.figure(figsize=(10, 6))

# 누적 영역형 차트 생성
plt.stackplot(x, y1, y2, y3, labels=labels, colors=colors, alpha=0.8)

plt.title('Stack Plot Example')
plt.xlabel('Time')
plt.ylabel('Total Value')
plt.legend(loc='upper left')
plt.show()
```

### 1.3. `stem()`: 스템 플롯
스템 플롯(Stem Plot)은 이산적인 데이터 시퀀스를 시각화하는 데 사용됩니다. 각 데이터 포인트는 기준선(baseline)에서 수직선(stem)으로 연결되고, 데이터 값의 위치에 마커가 표시됩니다. 롤리팝 차트(Lollipop Chart)와 유사한 형태를 가집니다.

**주요 특징:**
*   **이산 데이터 시각화:** 연속적이지 않고 개별적으로 존재하는 데이터 값을 표현하는 데 적합합니다.
*   **크기 비교:** 각 데이터 포인트의 크기를 기준선으로부터의 거리로 명확하게 보여주어 비교하기 용이합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.arange(0, 10)
y = np.sin(x) + 1.5

plt.figure(figsize=(10, 6))

# 스템 플롯 생성
markerline, stemlines, baseline = plt.stem(x, y, linefmt='grey', markerfmt='o', bottom=0.5, label='Discrete Data')

# 스타일 변경
plt.setp(markerline, 'markerfacecolor', '#1f77b4')
plt.setp(baseline, 'color', 'red', 'linewidth', 2)

plt.title('Stem Plot Example')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()
```
