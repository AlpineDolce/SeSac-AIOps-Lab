<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 관계형 플롯(`relplot()`, `scatterplot()`, `lineplot()`)을 사용하여 두 개 이상의 변수 간의 통계적 관계를 시각화하는 방법을 다룹니다. 산점도와 라인 플롯을 통해 데이터의 분포, 추세, 그리고 추가적인 변수(`hue`, `size`, `style`)를 반영하여 관계를 심층적으로 분석하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 관계형 플롯 (Relational Plots)](#1-관계형-플롯-relational-plots)
  - [1.1. `scatterplot()`](#11-scatterplot)
  - [1.2. `lineplot()`](#12-lineplot)

---

## 1. 관계형 플롯 (Relational Plots)
두 개 이상의 변수 간의 통계적 관계를 시각화하는 데 사용됩니다.

### 1.1. `scatterplot()`
산점도는 두 연속형 변수 간의 관계를 점으로 표현합니다. `hue`, `size`, `style` 등의 파라미터를 사용하여 추가적인 변수를 시각화에 반영할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips") # 팁 데이터셋 로드

# 총 계산액(total_bill)과 팁(tip)의 관계를 산점도로 표현
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.title("Total Bill vs. Tip")
plt.show()

# 성별(sex)에 따라 색상을 다르게 표현
sns.scatterplot(x="total_bill", y="tip", hue="sex", data=tips)
plt.title("Total Bill vs. Tip by Sex")
plt.show()

# 요일(day)에 따라 점의 크기(size)를, 시간(time)에 따라 점의 스타일(style)을 다르게 표현
sns.scatterplot(x="total_bill", y="tip", hue="day", size="size", style="time", data=tips)
plt.title("Total Bill vs. Tip by Day, Size, and Time")
plt.show()
```

### 1.2. `lineplot()`
라인 플롯은 주로 시계열 데이터나 연속적인 데이터의 추세를 보여줄 때 사용됩니다. 여러 관측치에 대한 신뢰 구간을 함께 표시할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

fmri = sns.load_dataset("fmri") # fmri 데이터셋 로드

# 시간(timepoint)에 따른 신호(signal) 변화를 라인 플롯으로 표현
sns.lineplot(x="timepoint", y="signal", data=fmri)
plt.title("Signal Change Over Time")
plt.show()

# 이벤트(event)와 지역(region)에 따라 라인 분리
sns.lineplot(x="timepoint", y="signal", hue="event", style="region", data=fmri)
plt.title("Signal Change by Event and Region")
plt.show()
```
