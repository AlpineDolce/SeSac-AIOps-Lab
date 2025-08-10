<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 분포 플롯(`histplot()`, `kdeplot()`, `displot()`)을 사용하여 단일 변수 또는 여러 변수의 분포를 시각화하는 방법을 다룹니다. 히스토그램, 커널 밀도 추정(KDE) 플롯, 그리고 고수준 인터페이스인 `displot()`을 통해 데이터의 밀도와 분포 형태를 파악하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 분포 플롯 (Distribution Plots)](#1-분포-플롯-distribution-plots)
  - [1.1. `histplot()`](#11-histplot)
  - [1.2. `kdeplot()`](#12-kdeplot)
  - [1.3. `displot()`](#13-displot)

---

## 1. 분포 플롯 (Distribution Plots)
단일 변수 또는 여러 변수의 분포를 시각화하는 데 사용됩니다.

### 1.1. `histplot()`
히스토그램은 단일 변수의 분포를 막대로 표현합니다. `bins` 파라미터로 구간의 개수를 조절할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.histplot(x="total_bill", data=tips, bins=15, kde=True) # kde=True로 커널 밀도 추정 곡선 추가
plt.title("Distribution of Total Bill")
plt.show()
```

### 1.2. `kdeplot()`
커널 밀도 추정(Kernel Density Estimate, KDE) 플롯은 데이터의 분포를 부드러운 곡선으로 표현합니다. 히스토그램보다 데이터의 밀도를 더 명확하게 보여줄 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.kdeplot(x="total_bill", data=tips, fill=True) # fill=True로 영역 채우기
plt.title("KDE of Total Bill")
plt.show()

# 두 변수의 결합 분포 (2D KDE)
sns.kdeplot(x="total_bill", y="tip", data=tips, fill=True)
plt.title("2D KDE of Total Bill and Tip")
plt.show()
```

### 1.3. `displot()`
단일 변수의 분포를 시각화하는 고수준 인터페이스입니다. `kind` 파라미터를 통해 히스토그램, KDE, ECDF(Empirical Cumulative Distribution Function) 등 다양한 유형의 분포 플롯을 그릴 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.displot(x="total_bill", data=tips, kind="hist", bins=15) # 히스토그램
plt.title("Displot (Histogram) of Total Bill")
plt.show()

sns.displot(x="total_bill", data=tips, kind="kde", fill=True) # KDE 플롯
plt.title("Displot (KDE) of Total Bill")
plt.show()
```
