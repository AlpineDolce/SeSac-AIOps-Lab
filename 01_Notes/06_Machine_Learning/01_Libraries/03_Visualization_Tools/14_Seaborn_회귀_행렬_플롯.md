<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 회귀 플롯(`regplot()`, `lmplot()`)을 사용하여 두 변수 간의 선형 관계를 시각화하고 회귀선을 함께 표시하는 방법을 다룹니다. 또한, 행렬 플롯(`heatmap()`, `clustermap()`)을 통해 데이터 행렬의 관계를 색상으로 인코딩하여 시각화하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 회귀 플롯 (Regression Plots)](#1-회귀-플롯-regression-plots)
  - [1.1. `regplot()`](#11-regplot)
  - [1.2. `lmplot()`](#12-lmplot)
- [2. 행렬 플롯 (Matrix Plots)](#2-행렬-플롯-matrix-plots)
  - [2.1. `heatmap()`](#21-heatmap)
  - [2.2. `clustermap()`](#22-clustermap)

---

## 1. 회귀 플롯 (Regression Plots)
두 변수 간의 선형 관계를 시각화하고 회귀선을 함께 표시합니다.

### 1.1. `regplot()`
산점도와 함께 선형 회귀선을 그리고, 회귀선의 신뢰 구간을 함께 표시합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.regplot(x="total_bill", y="tip", data=tips)
plt.title("Regression Plot of Total Bill and Tip")
plt.show()
```

### 1.2. `lmplot()`
`regplot()`과 유사하지만, `col`, `row`, `hue` 등의 파라미터를 사용하여 여러 서브플롯에 걸쳐 회귀 관계를 시각화할 수 있는 고수준 인터페이스입니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips) # 흡연 여부에 따라 회귀선 분리
plt.title("Regression Plot of Total Bill and Tip by Smoker Status")
plt.show()
```

## 2. 행렬 플롯 (Matrix Plots)
데이터 행렬의 관계를 색상으로 인코딩하여 시각화합니다. 주로 상관 행렬이나 데이터의 유사성을 보여줄 때 사용됩니다.

### 2.1. `heatmap()`
히트맵은 행렬 데이터를 색상 강도로 표현하여 데이터의 패턴이나 관계를 한눈에 파악할 수 있게 합니다. 주로 상관 행렬을 시각화하는 데 사용됩니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris") # Iris 데이터셋 로드

# 숫자형 컬럼들의 상관 행렬 계산
corr = iris.select_dtypes(include=['float64', 'int64']).corr()

sns.heatmap(corr, annot=True, cmap='coolwarm') # annot=True로 값 표시, cmap으로 색상 맵 설정
plt.title("Correlation Heatmap of Iris Features")
plt.show()
```

### 2.2. `clustermap()`
클러스터맵은 히트맵과 계층적 클러스터링(Hierarchical Clustering)을 결합한 플롯입니다. 행과 열을 유사성에 따라 재정렬하여 데이터 내의 군집 패턴을 시각적으로 보여줍니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

# 숫자형 컬럼만 선택
iris_numeric = iris.select_dtypes(include=['float64', 'int64'])

sns.clustermap(iris_numeric, cmap='viridis', standard_scale=1) # standard_scale=1로 컬럼별 정규화
plt.title("Clustermap of Iris Features")
plt.show()
```
