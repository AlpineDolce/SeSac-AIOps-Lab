<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 범주형 플롯(`boxplot()`, `violinplot()`, `stripplot()`, `swarmplot()`, `barplot()`, `countplot()`)을 사용하여 범주형 변수와 하나 이상의 연속형 변수 간의 관계를 시각화하는 방법을 다룹니다. 각 플롯의 특징과 용도를 이해하고 실제 코드 예제를 통해 데이터의 분포, 밀도, 개별 관측치, 평균 및 빈도 등을 파악하는 방법을 학습합니다.

<h2>목차</h2>

- [1. 범주형 플롯 (Categorical Plots)](#1-범주형-플롯-categorical-plots)
  - [1.1. `boxplot()`](#11-boxplot)
  - [1.2. `violinplot()`](#12-violinplot)
  - [1.3. `stripplot()`](#13-stripplot)
  - [1.4. `swarmplot()`](#14-swarmplot)
  - [1.5. `barplot()`](#15-barplot)
  - [1.6. `countplot()`](#16-countplot)

---

## 1. 범주형 플롯 (Categorical Plots)
범주형 변수와 하나 이상의 연속형 변수 간의 관계를 시각화하는 데 사용됩니다.

### 1.1. `boxplot()`
상자 그림은 범주별로 데이터의 분포(중앙값, 사분위수, 이상치)를 보여줍니다. 데이터의 중심 경향성과 퍼짐 정도, 이상치를 파악하는 데 유용합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("Total Bill by Day (Boxplot)")
plt.show()
```

### 1.2. `violinplot()`
바이올린 플롯은 상자 그림과 커널 밀도 추정(KDE)을 결합하여 데이터의 분포를 더 상세하게 보여줍니다. 데이터의 밀도와 분포 형태를 동시에 파악할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.violinplot(x="day", y="total_bill", data=tips)
plt.title("Total Bill by Day (Violinplot)")
plt.show()
```

### 1.3. `stripplot()`
스트립 플롯은 범주형 변수에 대한 개별 데이터 포인트들을 점으로 표시합니다. 데이터의 실제 분포를 보여주며, 데이터가 겹치는 것을 방지하기 위해 'jitter'를 추가할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
plt.title("Total Bill by Day (Stripplot)")
plt.show()
```

### 1.4. `swarmplot()`
스웜 플롯은 스트립 플롯과 유사하지만, 데이터 포인트들이 겹치지 않도록 자동으로 조정하여 각 데이터 포인트의 밀도를 더 잘 보여줍니다. 데이터의 분포와 개별 관측치를 동시에 파악하는 데 유용합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.swarmplot(x="day", y="total_bill", data=tips)
plt.title("Total Bill by Day (Swarmplot)")
plt.show()
```

### 1.5. `barplot()`
막대 그래프는 범주형 변수별로 연속형 변수의 평균(기본값)이나 다른 집계 값을 막대로 표현합니다. 신뢰 구간을 함께 표시할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.barplot(x="day", y="total_bill", data=tips, ci="sd") # ci="sd"로 표준편차 표시
plt.title("Average Total Bill by Day (Barplot)")
plt.show()
```

### 1.6. `countplot()`
카운트 플롯은 범주형 변수의 각 범주에 속하는 관측치의 개수를 막대로 표현합니다. 단일 범주형 변수의 빈도 분포를 시각화하는 데 사용됩니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.countplot(x="day", data=tips)
plt.title("Number of Observations by Day (Countplot)")
plt.show()
```
