<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 회귀 플롯(`regplot()`, `lmplot()`)을 사용하여 두 변수 간의 선형 관계를 시각화하고 회귀선을 함께 표시하는 방법을 다룹니다. 또한, 행렬 플롯(`heatmap()`, `clustermap()`)을 통해 데이터 행렬의 관계를 색상으로 인코딩하여 시각화하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

* [1. Seaborn 회귀 및 행렬 플롯 개요](#1-seaborn-회귀-및-행렬-플롯-개요)
  * [1.1. 회귀 플롯 (Regression Plots)](#11-회귀-플롯-regression-plots)
    * [1.1.1. `regplot()`: 회귀 플롯](#111-regplot-회귀-플롯)
    * [1.1.2. `lmplot()`: 고수준 회귀 플롯](#112-lmplot-고수준-회귀-플롯)
    * [1.1.3. `residplot()`: 잔차도](#113-residplot-잔차도)
  * [1.2. 행렬 플롯 (Matrix Plots)](#12-행렬-플롯-matrix-plots)
    * [1.2.1. `heatmap()`: 히트맵](#121-heatmap-히트맵)
    * [1.2.2. `clustermap()`: 클러스터맵](#122-clustermap-클러스터맵)

---

## 1. Seaborn 회귀 및 행렬 플롯 개요
Seaborn의 회귀 플롯(`regplot()`, `lmplot()`)은 두 변수 간의 선형 관계를 시각화하고 회귀선을 함께 표시하는 데 사용됩니다. 행렬 플롯(`heatmap()`, `clustermap()`)은 데이터 행렬의 관계를 색상으로 인코딩하여 시각화합니다. 이 문서에서는 각 플롯의 특징과 용도를 이해하고 실제 코드 예제를 통해 데이터의 관계와 패턴을 파악하는 방법을 학습합니다.

### 1.1. 회귀 플롯 (Regression Plots)
두 변수 간의 선형 관계를 시각화하고 회귀선을 함께 표시합니다.

#### 1.1.1. `regplot()`: 회귀 플롯
`regplot()`은 두 연속형 변수 간의 선형 관계를 시각화하는 데 사용되는 Seaborn의 함수입니다. 산점도(scatter plot) 위에 선형 회귀선과 그 회귀선의 신뢰 구간을 함께 표시하여 변수 간의 추세와 관계의 불확실성을 한눈에 파악할 수 있게 해줍니다.

##### 1.1.1.1. 주요 특징:
*   **산점도와 회귀선:** `x`와 `y`로 지정된 두 변수의 개별 데이터 포인트들을 산점도로 나타내고, 그 위에 이 데이터에 가장 잘 맞는 선형 회귀선을 그립니다.
*   **신뢰 구간 (Confidence Interval):** 회귀선 주변의 음영 처리된 영역은 회귀 추정치의 신뢰 구간을 나타냅니다. 기본적으로 95% 신뢰 구간을 표시하며, 이는 동일한 데이터를 여러 번 샘플링했을 때 회귀선이 이 영역 안에 들어올 확률이 95%임을 의미합니다. `ci=None`으로 설정하여 신뢰 구간을 표시하지 않을 수도 있습니다.
*   **다양한 옵션:** `x_estimator`를 사용하여 `x`축 변수의 각 값에 대한 `y`값의 평균을 계산하여 표시할 수 있으며, `logx=True`를 통해 `x`축을 로그 스케일로 변환하여 회귀 분석을 수행할 수도 있습니다.

##### 1.1.1.2. 사용 시기:
*   두 연속형 변수 간의 선형 관계를 탐색하고 싶을 때.
*   회귀 모델의 적합성을 시각적으로 평가하고 싶을 때.
*   데이터의 추세와 함께 그 추정치의 불확실성을 함께 보여주고자 할 때.

##### 1.1.1.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# regplot 그리기 (기본: 산점도, 회귀선, 95% 신뢰 구간)
sns.regplot(x="total_bill", y="tip", data=tips)
plt.title("총 계산액과 팁의 회귀 플롯")
plt.xlabel("총 계산액")
plt.ylabel("팁")
plt.savefig('regplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: 신뢰 구간 없이, x_estimator 사용
# 참고: Seaborn 0.12.0부터 `ci` 매개변수 대신 `errorbar` 매개변수 사용이 권장됩니다.
plt.figure(figsize=(8, 6))
sns.regplot(x="size", y="tip", data=tips, x_estimator=sum, errorbar=None, color="red")
plt.title("파티 규모별 팁 합계의 회귀 플롯 (신뢰 구간 없음)")
plt.xlabel("파티 규모")
plt.ylabel("팁 합계")
plt.show()
```

#### 1.1.2. `lmplot()`: 고수준 회귀 플롯
`lmplot()`은 `regplot()`과 마찬가지로 두 변수 간의 선형 회귀 관계를 시각화하지만, Seaborn의 "figure-level" 함수로서 더 복잡한 조건부 관계를 여러 서브플롯에 걸쳐 시각화할 수 있는 고수준 인터페이스를 제공합니다. `FacetGrid` 객체를 기반으로 작동하여 데이터셋의 다른 범주형 변수에 따라 회귀 플롯을 분리하여 그릴 수 있습니다.

##### 1.1.2.1. 주요 특징:
*   **조건부 시각화:** `col`, `row`, `hue`와 같은 파라미터를 사용하여 데이터셋의 다른 범주형 변수에 따라 플롯을 분리하거나 색상으로 구분하여 그릴 수 있습니다.
    *   `hue`: 지정된 범주형 변수의 각 레벨에 따라 다른 색상으로 데이터를 표시하고 별도의 회귀선을 그립니다.
    *   `col`: 지정된 범주형 변수의 각 레벨에 따라 열(column) 방향으로 서브플롯을 생성합니다.
    *   `row`: 지정된 범주형 변수의 각 레벨에 따라 행(row) 방향으로 서브플롯을 생성합니다.
*   **`regplot()`과의 차이:** `regplot()`은 단일 축(axes-level)에 플롯을 그리는 반면, `lmplot()`은 `FacetGrid`를 사용하여 여러 축에 걸쳐 플롯을 그립니다. 따라서 `lmplot()`은 더 복잡한 다변량 관계를 탐색하는 데 적합합니다.
*   **유연성:** `kind` 파라미터를 사용하여 `scatter` (기본값), `reg` 외에도 `resid` (잔차 플롯) 등 다양한 종류의 플롯을 그릴 수 있습니다.

##### 1.1.2.2. 사용 시기:
*   두 변수 간의 선형 관계가 다른 범주형 변수에 따라 어떻게 달라지는지 비교하고 싶을 때.
*   데이터셋 내의 복잡한 다변량 관계를 체계적으로 탐색하고 시각화하고 싶을 때.
*   여러 그룹에 대한 회귀 분석 결과를 한 번에 보여주고자 할 때.

##### 1.1.2.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# lmplot 그리기 (hue 사용: 흡연 여부에 따라 색상 및 회귀선 분리)
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)
plt.suptitle("총 계산액과 팁의 회귀 플롯 (흡연 여부별)", y=1.02) # 전체 제목 설정
plt.savefig('lmplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: col 사용 (요일별로 서브플롯 분리)
sns.lmplot(x="total_bill", y="tip", col="day", data=tips, col_wrap=2, height=4)
plt.suptitle("총 계산액과 팁의 회귀 플롯 (요일별)", y=1.02)
plt.show()

# 추가 예시: row와 hue 함께 사용
sns.lmplot(x="total_bill", y="tip", row="time", hue="sex", data=tips, height=3, aspect=1.5)
plt.suptitle("총 계산액과 팁의 회귀 플롯 (시간 및 성별)", y=1.02)
plt.show()
```

#### 1.1.3. `residplot()`: 잔차도
`residplot()`은 회귀 모델의 잔차(residuals)를 시각화하여 모델의 적합성을 진단하는 데 사용되는 중요한 도구입니다. 잔차는 실제 관측값과 회귀 모델의 예측값 간의 차이를 의미하며, 잔차도를 통해 모델이 데이터의 패턴을 잘 설명하는지, 특정 가정을 만족하는지 등을 확인할 수 있습니다.

##### 1.1.3.1. 주요 특징:
*   **모델 진단:** 잔차도를 통해 회귀 모델의 여러 문제점을 시각적으로 진단할 수 있습니다.
    *   **비선형성(Non-linearity):** 잔차들이 y=0 축을 기준으로 뚜렷한 패턴(예: 곡선 형태)을 보이면, 데이터에 선형 모델이 적합하지 않음을 의미합니다.
    *   **이분산성(Heteroscedasticity):** x값이 증가함에 따라 잔차의 퍼짐 정도가 달라지면(예: 깔때기 모양), 오차의 분산이 일정하다는 가정을 위배한 것입니다.
    *   **이상치(Outliers):** 다른 잔차들로부터 멀리 떨어진 점들은 모델의 성능에 큰 영향을 미치는 이상치일 수 있습니다.
*   **이상적인 잔차도:** 잘 적합된 모델의 잔차도는 y=0 축을 중심으로 특별한 패턴 없이 무작위로 흩어져 있어야 합니다.

##### 1.1.3.2. 사용 시기:
*   선형 회귀 모델을 학습시킨 후, 모델의 적합성을 시각적으로 평가하고 싶을 때.
*   모델의 예측 성능을 저해하는 잠재적인 문제점(비선형성, 이분산성 등)을 진단하고자 할 때.

##### 1.1.3.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# 잔차도 그리기
plt.figure(figsize=(8, 6))
sns.residplot(x="total_bill", y="tip", data=tips, lowess=True, 
              scatter_kws={"alpha": 0.5}, line_kws={"color": "red", "lw": 2})
# lowess=True: 잔차의 추세를 보기 위한 평활 곡선 추가
plt.title("총 계산액과 팁의 잔차도")
plt.xlabel("총 계산액")
plt.ylabel("잔차")
plt.show()
```

### 1.2. 행렬 플롯 (Matrix Plots)
데이터 행렬의 관계를 색상으로 인코딩하여 시각화합니다. 주로 상관 행렬이나 데이터의 유사성을 보여줄 때 사용됩니다.

#### 1.2.1. `heatmap()`: 히트맵
히트맵(Heatmap)은 행렬 데이터를 색상 강도로 표현하여 데이터의 패턴이나 관계를 한눈에 파악할 수 있게 하는 강력한 시각화 도구입니다. 주로 상관 행렬(correlation matrix)을 시각화하여 변수 간의 관계 강도와 방향을 보여주는 데 사용되지만, 다른 형태의 행렬 데이터(예: 혼동 행렬, 유전자 발현 데이터)를 시각화하는 데도 활용됩니다.

##### 1.2.1.1. 주요 특징:
*   **색상 인코딩:** 행렬의 각 셀에 해당하는 값을 색상의 농도나 색조로 매핑하여 시각적으로 표현합니다. 이를 통해 데이터 내의 고점, 저점, 패턴 등을 쉽게 식별할 수 있습니다.
*   **상관 행렬 시각화:** 데이터프레임의 `corr()` 메서드로 계산된 상관 행렬을 `heatmap()`에 전달하여 변수 간의 상관 관계를 직관적으로 보여줄 수 있습니다.
*   **다양한 커스터마이징 옵션:**
    *   `annot=True`: 각 셀에 실제 데이터 값을 숫자로 표시합니다.
    *   `fmt`: `annot=True`일 때 숫자의 포맷을 지정합니다 (예: `.2f`로 소수점 두 자리까지 표시).
    *   `cmap`: 색상 맵을 지정하여 데이터 값에 따른 색상 변화를 조절합니다 (예: `coolwarm`, `viridis`, `Blues`).
    *   `linewidths`, `linecolor`: 셀 간의 경계선을 추가하여 시각적 구분을 명확히 할 수 있습니다.
    *   `cbar`: 색상 막대(colorbar)의 표시 여부를 제어합니다.

##### 1.2.1.2. 사용 시기:
*   변수 간의 상관 관계를 시각적으로 탐색하고 싶을 때.
*   대규모 행렬 데이터에서 패턴이나 이상치를 빠르게 식별해야 할 때.
*   혼동 행렬(Confusion Matrix)과 같이 분류 모델의 성능을 시각적으로 평가할 때.

##### 1.2.1.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 예시 데이터 로드
iris = sns.load_dataset("iris")

# 숫자형 컬럼들의 상관 행렬 계산
corr = iris.select_dtypes(include=['float64', 'int64']).corr()

# heatmap 그리기 (기본)
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5) # annot=True로 값 표시, fmt로 소수점 포맷, linewidths로 셀 경계선
plt.title("Iris Features 상관 행렬 (Heatmap)")
plt.savefig('heatmap_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: 다른 데이터 (랜덤 데이터)
data = np.random.rand(10, 12)
plt.figure(figsize=(10, 8))
sns.heatmap(data, cmap="viridis", cbar=True) # cbar=True로 색상 막대 표시
plt.title("랜덤 데이터 히트맵")
plt.show()
```

#### 1.2.2. `clustermap()`: 클러스터맵
`clustermap()`은 히트맵과 계층적 클러스터링(Hierarchical Clustering)을 결합한 강력한 시각화 도구입니다. 데이터 행렬을 히트맵으로 표현하는 동시에, 행과 열을 유사성(거리 측정)에 따라 재정렬하고 그 과정을 덴드로그램(dendrogram)으로 시각화하여 데이터 내의 숨겨진 군집 패턴을 탐색하는 데 매우 유용합니다.

##### 1.2.2.1. 주요 특징:
*   **히트맵 + 덴드로그램:** 데이터의 각 셀 값은 색상으로 표현되며, 행과 열의 가장자리에는 계층적 클러스터링 결과를 보여주는 덴드로그램이 함께 그려집니다. 덴드로그램은 데이터 포인트(또는 변수)들이 어떻게 그룹화되는지 나무 구조로 보여줍니다.
*   **데이터 재정렬:** 클러스터링 결과에 따라 행과 열이 재정렬되므로, 유사한 특성을 가진 데이터 포인트나 변수들이 서로 가깝게 배치되어 군집 패턴을 시각적으로 쉽게 파악할 수 있습니다.
*   **정규화 옵션 (`standard_scale`):** `standard_scale` 파라미터를 사용하여 데이터를 정규화할 수 있습니다.
    *   `standard_scale=0`: 행(row)별로 정규화합니다.
    *   `standard_scale=1`: 열(column)별로 정규화합니다.
    *   `standard_scale=None` (기본값): 정규화하지 않습니다.
*   **거리 측정 및 연결 방법:** `metric` (거리 측정 방법, 예: `euclidean`, `correlation`)과 `method` (클러스터링 연결 방법, 예: `average`, `single`, `complete`) 파라미터를 통해 클러스터링 방식을 세밀하게 제어할 수 있습니다.

##### 1.2.2.2. 사용 시기:
*   대규모 데이터셋에서 데이터 포인트(샘플)나 변수(특성) 간의 유사성 및 군집 구조를 탐색하고 싶을 때.
*   유전자 발현 데이터, 고객 세분화 등 복잡한 데이터에서 자연스러운 그룹을 찾고자 할 때.
*   히트맵만으로는 파악하기 어려운 데이터 내의 계층적 관계를 시각적으로 이해하고자 할 때.

##### 1.2.2.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
iris = sns.load_dataset("iris")

# 숫자형 컬럼만 선택
iris_numeric = iris.select_dtypes(include=['float64', 'int64'])

# clustermap 그리기 (기본)
sns.clustermap(iris_numeric, cmap='viridis', standard_scale=1) # standard_scale=1로 컬럼별 정규화
plt.suptitle("Iris Features 클러스터맵", y=1.02) # 전체 제목 설정
plt.savefig('clustermap_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: 다른 데이터셋, row_colors, metric, method 지정
flights = sns.load_dataset("flights")
flights_pivot = flights.pivot_table(index="month", columns="year", values="passengers")

plt.figure(figsize=(10, 8))
sns.clustermap(flights_pivot, cmap="YlGnBu", metric="correlation", method="average",
               row_cluster=True, col_cluster=True,
               cbar_kws={"label": "승객 수"})
plt.suptitle("월별/연도별 승객 수 클러스터맵", y=1.02)
plt.show()
```