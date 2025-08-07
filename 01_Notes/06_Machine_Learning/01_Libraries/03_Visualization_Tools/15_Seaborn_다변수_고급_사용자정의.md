<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 다변수 분석 플롯(`pairplot()`, `jointplot()`)을 사용하여 데이터셋의 여러 변수 간 관계를 한 번에 파악하는 방법을 다룹니다. 또한, `FacetGrid`를 이용한 고급 플롯 제어, 그리고 Seaborn의 사용자 정의 및 테마 설정을 통해 플롯의 미적 품질을 향상시키는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 다변수 분석 플롯 (Multivariate Analysis Plots)](#1-다변수-분석-플롯-multivariate-analysis-plots)
  - [1.1. `pairplot()`](#11-pairplot)
  - [1.2. `jointplot()`](#12-jointplot)
- [2. 고급 플롯 제어: FacetGrid](#2-고급-플롯-제어-facetgrid)
- [3. 사용자 정의 및 테마](#3-사용자-정의-및-테마)

---

## 1. 다변수 분석 플롯 (Multivariate Analysis Plots)
Seaborn의 가장 강력한 기능 중 하나로, 데이터셋의 여러 변수 간 관계를 한 번에 파악할 수 있는 고수준 플롯을 제공합니다. 탐색적 데이터 분석(EDA) 과정에서 매우 유용합니다.

### 1.1. `pairplot()`
Pair Plot은 데이터프레임의 모든 숫자형 변수 쌍에 대한 산점도(scatterplot)를 그리고, 대각선에는 각 변수의 분포(히스토그램 또는 KDE)를 보여줍니다. 데이터셋의 전반적인 관계와 분포를 빠르게 파악하는 데 최고의 도구입니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

sns.pairplot(iris, hue="species") # 종(species)에 따라 색상 구분
plt.suptitle("Pair Plot of Iris Dataset", y=1.02) # 전체 제목 추가
plt.show()
```

### 1.2. `jointplot()`
Joint Plot은 두 변수 간의 관계와 각 변수의 분포를 동시에 시각화합니다. 중앙에는 산점도나 헥스빈 플롯이, 위쪽과 오른쪽에는 각 변수의 히스토그램이나 KDE 플롯이 배치됩니다. 두 변수를 깊이 있게 분석할 때 매우 효과적입니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# 산점도와 히스토그램을 함께 표시
sns.jointplot(x="total_bill", y="tip", data=tips, kind="scatter") # kind: scatter, hex, kde, reg
plt.suptitle("Joint Plot of Total Bill and Tip", y=1.02)
plt.show()

# 헥스빈(hexbin)과 KDE를 함께 표시
sns.jointplot(x="total_bill", y="tip", data=tips, kind="hex", cmap="hot")
plt.suptitle("Joint Plot (Hexbin) of Total Bill and Tip", y=1.02)
plt.show()
```

## 2. 고급 플롯 제어: FacetGrid
`FacetGrid`는 데이터의 하위 집합(subset)에 따라 여러 개의 서브플롯(facet)을 만들어 동일한 종류의 그래프를 그리는 강력한 기능입니다. `col`, `row`, `hue` 등의 변수를 기준으로 데이터를 나누어 시각화함으로써, 복잡한 데이터의 패턴을 다각도에서 비교 분석할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# FacetGrid 객체 생성
g = sns.FacetGrid(tips, col="time", row="sex")

# 각 서브플롯에 그래프 매핑
g.map(sns.scatterplot, "total_bill", "tip", alpha=.7)
g.add_legend()

plt.show()
```

## 3. 사용자 정의 및 테마
Seaborn은 Matplotlib의 기능을 상속받으므로, Matplotlib의 함수들을 사용하여 플롯을 추가적으로 커스터마이징할 수 있습니다. 또한, Seaborn은 자체적으로 다양한 테마와 스타일을 제공하여 플롯의 미적 품질을 쉽게 변경할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# Seaborn 테마 설정
sns.set_theme(style="whitegrid") # 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'

sns.scatterplot(x="total_bill", y="tip", hue="time", size="size", data=tips)
plt.title("Customized Scatter Plot with Seaborn Theme")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
plt.show()

# Matplotlib 함수와 함께 사용
sns.set_theme(style="darkgrid")
plt.figure(figsize=(8, 6))
sns.histplot(x="total_bill", data=tips, kde=True, color="skyblue")
plt.title("Histogram with Matplotlib Customization")
plt.suptitle("Using Matplotlib and Seaborn Together") # 전체 그림 제목
plt.show()
```

**통계적 유의성 주석(Annotation) 추가**

Seaborn으로 그린 Box plot이나 Bar plot에서 두 그룹 간의 차이가 통계적으로 유의미한지 여부를 시각적으로 보여주는 것은 분석의 신뢰도를 높입니다. `statannot`과 같은 보조 라이브러리를 사용하면, 두 그룹 사이에 p-value나 유의수준을 나타내는 별표(`*`, `**`, `***`)를 쉽게 추가할 수 있습니다.
