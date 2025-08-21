<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 범주형 플롯(`boxplot()`, `violinplot()`, `stripplot()`, `swarmplot()`, `barplot()`, `countplot()`)을 사용하여 범주형 변수와 하나 이상의 연속형 변수 간의 관계를 시각화하는 방법을 다룹니다. 각 플롯의 특징과 용도를 이해하고 실제 코드 예제를 통해 데이터의 분포, 밀도, 개별 관측치, 평균 및 빈도 등을 파악하는 방법을 학습합니다.

<h2>목차</h2>

* [1. Seaborn 범주형 플롯 개요](#1-seaborn-범주형-플롯-개요)
  * [1.1. `boxplot()`: 상자 그림](#11-boxplot-상자-그림)
  * [1.2. `violinplot()`: 바이올린 플롯](#12-violinplot-바이올린-플롯)
  * [1.3. `boxenplot()`: 박슨 플롯](#13-boxenplot-박슨-플롯)
  * [1.4. `stripplot()`: 스트립 플롯](#14-stripplot-스트립-플롯)
  * [1.5. `swarmplot()`: 스웜 플롯](#15-swarmplot-스웜-플롯)
  * [1.6. `barplot()`: 막대 그래프](#16-barplot-막대-그래프)
  * [1.7. `countplot()`: 카운트 플롯](#17-countplot-카운트-플롯)
  * [1.8. `catplot()`: 고수준 범주형 플롯 인터페이스](#18-catplot-고수준-범주형-플롯-인터페이스)

---

## 1. Seaborn 범주형 플롯 개요
Seaborn의 범주형 플롯(Categorical Plots)은 범주형 변수와 하나 이상의 연속형 변수 간의 관계를 시각화하는 데 사용됩니다. 각 플롯은 데이터의 분포, 밀도, 개별 관측치, 평균 및 빈도 등을 파악하여 데이터셋의 특성을 이해하는 데 도움을 줍니다. 주요 함수로는 `boxplot()`, `violinplot()`, `stripplot()`, `swarmplot()`, `barplot()`, `countplot()`이 있습니다.

### 1.1. `boxplot()`
상자 그림(Boxplot)은 범주형 변수에 따른 연속형 변수의 분포를 시각화하는 데 사용됩니다. 데이터의 중앙값, 사분위수(Q1, Q3), 이상치(outliers)를 한눈에 파악할 수 있어 데이터의 중심 경향성, 퍼짐 정도, 그리고 비대칭성을 이해하는 데 매우 유용합니다.

#### 1.1.1. 상자 그림의 구성 요소:
*   **중앙값 (Median):** 상자 내부의 선으로 표시되며, 데이터를 절반으로 나누는 값입니다 (50번째 백분위수).
*   **상자 (Box):** 상자의 하단은 1사분위수(Q1, 25번째 백분위수), 상단은 3사분위수(Q3, 75번째 백분위수)를 나타냅니다. 상자의 길이는 사분위 범위(IQR = Q3 - Q1)를 의미하며, 데이터의 중간 50%가 분포하는 구간입니다.
*   **수염 (Whiskers):** 상자 밖으로 뻗어 나가는 선으로, 일반적으로 IQR의 1.5배 이내에 있는 데이터의 범위를 나타냅니다. 수염의 끝은 해당 범위 내의 최댓값과 최솟값입니다.
*   **이상치 (Outliers):** 수염 밖의 개별 점으로 표시되며, IQR의 1.5배를 초과하는 값들입니다. 이는 데이터에서 특이하거나 비정상적인 관측치를 나타낼 수 있습니다.

#### 1.1.2. 사용 시기:
*   여러 그룹 간의 데이터 분포를 비교할 때.
*   데이터의 왜도(skewness)와 이상치를 빠르게 식별할 때.

#### 1.1.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# boxplot 그리기
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("요일별 총 계산액 분포 (Boxplot)")
plt.xlabel("요일")
plt.ylabel("총 계산액")
plt.savefig('boxplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: hue 파라미터 사용 (성별에 따른 요일별 총 계산액)
plt.figure(figsize=(10, 6))
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="viridis")
plt.title("요일 및 성별에 따른 총 계산액 분포 (Boxplot with Hue)")
plt.xlabel("요일")
plt.ylabel("총 계산액")
plt.legend(title="성별")
plt.show()
```

### 1.2. `violinplot()`
바이올린 플롯(Violinplot)은 상자 그림(boxplot)과 커널 밀도 추정(Kernel Density Estimate, KDE)을 결합하여 데이터의 분포를 더 상세하게 보여주는 시각화 도구입니다. 상자 그림이 중앙값, 사분위수, 이상치 등 요약 통계에 집중하는 반면, 바이올린 플롯은 데이터의 밀도와 분포 형태를 시각적으로 명확하게 파악할 수 있게 해줍니다.

#### 1.2.1. 바이올린 플롯의 구성 요소:
*   **중앙 상자 및 선:** 상자 그림과 유사하게 중앙값(흰색 점), 사분위수(두꺼운 검은색 막대), 그리고 95% 신뢰 구간(얇은 검은색 선)을 표시할 수 있습니다. (기본적으로는 중앙값과 IQR만 표시)
*   **바이올린 모양:** 각 범주에 대한 데이터의 분포 밀도를 나타냅니다. 바이올린의 폭이 넓을수록 해당 값 주변에 데이터 포인트가 밀집해 있음을 의미합니다. 이는 데이터의 다봉성(multimodality)이나 왜도(skewness)를 시각적으로 쉽게 파악할 수 있게 합니다.

#### 1.2.2. 사용 시기:
*   여러 그룹 간의 데이터 분포 형태를 비교할 때.
*   데이터의 밀도와 분포의 세부적인 특징(예: 이봉 분포)을 파악하고자 할 때.
*   상자 그림보다 더 풍부한 분포 정보를 제공하고자 할 때.

#### 1.2.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# violinplot 그리기
sns.violinplot(x="day", y="total_bill", data=tips)
plt.title("요일별 총 계산액 분포 (Violinplot)")
plt.xlabel("요일")
plt.ylabel("총 계산액")
plt.savefig('violinplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: hue 파라미터 사용 및 split=True (성별에 따른 요일별 총 계산액)
plt.figure(figsize=(10, 6))
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, palette="pastel", split=True)
plt.title("요일 및 성별에 따른 총 계산액 분포 (Violinplot with Hue and Split)")
plt.xlabel("요일")
plt.ylabel("총 계산액")
plt.legend(title="성별")
plt.show()
```

### 1.3. `boxenplot()`: 박슨 플롯
박슨 플롯(Boxenplot)은 기존의 상자 그림(`boxplot`)을 대규모 데이터셋에 더 적합하도록 개선한 플롯입니다. "Letter-value plot"이라고도 불리며, 데이터의 분포, 특히 꼬리 부분의 정보를 더 상세하게 표현합니다.

#### 1.3.1. 주요 특징:
*   **상세한 분위수 정보:** `boxplot`이 중앙값과 25/75 분위수만 보여주는 반면, `boxenplot`은 더 많은 분위수(quartiles, octiles 등)를 상자의 너비를 다르게 하여 시각적으로 표현합니다. 이를 통해 데이터 분포의 꼬리 부분에 대한 더 정밀한 정보를 얻을 수 있습니다.
*   **대규모 데이터에 적합:** 데이터 포인트가 많아질수록 `boxplot`의 이상치 표시가 너무 많아져 분포 파악이 어려운 문제를 해결합니다. `boxenplot`은 이상치를 더 안정적으로 정의하고 분포의 형태를 더 잘 나타냅니다.

#### 1.3.2. 사용 시기:
*   데이터셋의 크기가 클 때 `boxplot`을 대체하여 사용합니다.
*   데이터 분포의 꼬리 부분에 대한 상세한 정보를 시각화하고 싶을 때.

#### 1.3.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# boxenplot 그리기
plt.figure(figsize=(10, 6))
sns.boxenplot(x="day", y="total_bill", data=tips)
plt.title("요일별 총 계산액 분포 (Boxenplot)")
plt.xlabel("요일")
plt.ylabel("총 계산액")
plt.show()
```

### 1.4. `stripplot()`
스트립 플롯(Stripplot)은 범주형 변수에 대한 개별 데이터 포인트들을 점으로 표시하여 데이터의 실제 분포를 보여주는 플롯입니다. 특히 데이터 포인트의 수가 많지 않을 때 각 관측치의 위치를 명확하게 확인할 수 있다는 장점이 있습니다.

#### 1.4.1. 주요 특징:
*   **개별 관측치 표시:** 모든 데이터 포인트를 시각화하여 데이터의 밀집 영역과 희소 영역을 직관적으로 파악할 수 있습니다.
*   **겹침 방지 (`jitter`):** 데이터 포인트가 겹쳐서 보이지 않는 '과밀(overplotting)' 현상을 방지하기 위해 `jitter=True` 옵션을 사용하여 점들을 약간씩 무작위로 분산시킬 수 있습니다. 이는 데이터의 분포를 더 정확하게 보여줍니다.
*   **다른 플롯과의 조합:** 상자 그림(`boxplot()`)이나 바이올린 플롯(`violinplot()`)과 함께 사용하여 요약 통계와 개별 데이터 포인트를 동시에 보여주는 데 유용합니다.

#### 1.4.2. 사용 시기:
*   데이터 포인트의 정확한 위치를 확인하고 싶을 때.
*   데이터의 수가 너무 많지 않아 개별 점을 모두 표시해도 시각적으로 복잡하지 않을 때.
*   분포 요약(상자 그림, 바이올린 플롯)과 함께 개별 데이터의 분포를 함께 보고 싶을 때.

#### 1.4.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# stripplot 그리기
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
plt.title("요일별 총 계산액 (Stripplot)")
plt.xlabel("요일")
plt.ylabel("총 계산액")
plt.savefig('stripplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: boxplot과 stripplot 함께 사용
plt.figure(figsize=(10, 6))
sns.boxplot(x="day", y="total_bill", data=tips, color=".8") # 배경으로 boxplot
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True, color="black", size=4) # 그 위에 stripplot
plt.title("요일별 총 계산액 분포 (Boxplot with Stripplot)")
plt.xlabel("요일")
plt.ylabel("총 계산액")
plt.legend(title="성별")
plt.show()
```

### 1.5. `swarmplot()`
스웜 플롯(Swarmplot)은 스트립 플롯(`stripplot()`)과 유사하게 개별 데이터 포인트들을 표시하지만, 데이터 포인트들이 서로 겹치지 않도록 자동으로 조정하여 각 데이터 포인트의 밀도를 더 잘 보여주는 플롯입니다. 이는 데이터의 분포와 개별 관측치를 동시에 파악하는 데 매우 유용합니다.

#### 1.5.1. 주요 특징:
*   **겹침 없는 개별 점 표시:** `stripplot()`의 `jitter`와 달리, `swarmplot()`은 데이터 포인트가 겹치지 않도록 범주 축을 따라 점들을 "벌집"처럼 배치합니다. 이를 통해 데이터의 실제 밀도 분포를 시각적으로 정확하게 표현합니다.
*   **밀도와 개별 관측치 동시 파악:** 데이터가 밀집된 영역은 점들이 조밀하게 모여 있고, 데이터가 희소한 영역은 점들이 드문드문 나타나므로, 분포의 형태와 각 데이터 포인트의 위치를 동시에 이해할 수 있습니다.
*   **데이터 양에 따른 한계:** 데이터 포인트의 수가 매우 많아지면 계산 비용이 커지고 시각적으로 복잡해질 수 있습니다. 이 경우 `stripplot()`이나 다른 분포 플롯을 고려하는 것이 좋습니다.

#### 1.5.2. 사용 시기:
*   데이터 포인트의 정확한 위치와 함께 분포의 밀도를 명확하게 보고 싶을 때.
*   데이터의 수가 너무 많지 않아 모든 개별 점을 표시해도 시각적으로 혼란스럽지 않을 때.
*   상자 그림이나 바이올린 플롯과 함께 사용하여 요약 통계와 실제 데이터 분포를 함께 보여줄 때.

#### 1.5.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# swarmplot 그리기
sns.swarmplot(x="day", y="total_bill", data=tips)
plt.title("요일별 총 계산액 분포 (Swarmplot)")
plt.xlabel("요일")
plt.ylabel("총 계산액")
plt.savefig('swarmplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: violinplot과 swarmplot 함께 사용
plt.figure(figsize=(10, 6))
sns.violinplot(x="day", y="total_bill", data=tips, inner=None, color=".8") # 배경으로 violinplot
sns.swarmplot(x="day", y="total_bill", data=tips, color="black", size=4) # 그 위에 swarmplot
plt.title("요일별 총 계산액 분포 (Violinplot with Swarmplot)")
plt.xlabel("요일")
plt.ylabel("총 계산액")
plt.legend(title="성별")
plt.show()
```

### 1.6. `barplot()`
막대 그래프(Barplot)는 범주형 변수별로 연속형 변수의 **평균(기본값)**이나 다른 집계 값(예: 중앙값, 합계 등)을 막대의 높이로 표현하는 플롯입니다. 각 막대 위에 표시되는 오차 막대(error bar)는 해당 집계 값의 **신뢰 구간(confidence interval)**을 나타내어 추정치의 불확실성을 시각적으로 보여줍니다.

#### 1.6.1. 주요 특징:
*   **중심 경향성 시각화:** 각 범주에 대한 연속형 변수의 대표값(기본적으로 평균)을 쉽게 비교할 수 있습니다.
*   **신뢰 구간:** `ci` (confidence interval) 파라미터를 통해 신뢰 구간의 종류를 지정할 수 있습니다. 기본값은 95% 신뢰 구간이며, `sd`로 설정하면 표준편차를 표시할 수 있습니다. 신뢰 구간이 겹치지 않으면 두 그룹 간의 평균에 통계적으로 유의미한 차이가 있을 가능성이 높다고 해석할 수 있습니다.
*   **`countplot()`과의 차이:** `countplot()`은 범주형 변수의 각 범주에 속하는 관측치의 '개수'를 보여주는 반면, `barplot()`은 범주형 변수와 연속형 변수 간의 관계에서 연속형 변수의 '집계 값'을 보여줍니다.

#### 1.6.2. 사용 시기:
*   범주형 그룹 간의 연속형 변수 평균을 비교하고 싶을 때.
*   추정치의 불확실성(신뢰 구간)을 함께 보여주고자 할 때.
*   데이터의 요약된 통계적 특성을 명확하게 전달하고자 할 때.

#### 1.6.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# barplot 그리기 (기본: 평균과 95% 신뢰 구간)
sns.barplot(x="day", y="total_bill", data=tips)
plt.title("요일별 총 계산액 평균 (Barplot)")
plt.xlabel("요일")
plt.ylabel("총 계산액 평균")
plt.savefig('barplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: errorbar="sd"로 표준편차 표시, hue 파라미터 사용
# 참고: Seaborn 0.12.0부터 `ci` 매개변수 대신 `errorbar` 매개변수 사용이 권장됩니다.
plt.figure(figsize=(10, 6))
sns.barplot(x="day", y="total_bill", hue="sex", data=tips, errorbar="sd", palette="coolwarm")
plt.title("요일 및 성별에 따른 총 계산액 평균 (Barplot with SD and Hue)")
plt.xlabel("요일")
plt.ylabel("총 계산액 평균")
plt.legend(title="성별")
plt.show()
```

### 1.7. `countplot()`
카운트 플롯(Countplot)은 **단일 범주형 변수**의 각 범주에 속하는 관측치의 개수(빈도)를 막대로 표현하는 플롯입니다. 이는 범주형 데이터의 분포를 시각적으로 빠르게 파악하는 데 매우 효과적입니다.

#### 1.7.1. 주요 특징:
*   **빈도 분포 시각화:** `countplot()`은 `y`축을 지정할 필요 없이 `x`축(또는 `y`축)에 범주형 변수만 지정하면 자동으로 각 범주의 빈도를 계산하여 막대 그래프로 보여줍니다.
*   **`barplot()`과의 차이:** `barplot()`이 범주형 변수와 연속형 변수 간의 관계에서 연속형 변수의 집계 값(예: 평균)을 보여주는 반면, `countplot()`은 오직 범주형 변수의 '개수'만을 시각화합니다. 즉, `countplot()`은 `barplot()`에서 `y`값을 `count`로 설정하고 `estimator`를 `len`으로 설정한 것과 유사합니다.
*   **데이터 탐색의 시작점:** 범주형 데이터셋을 처음 탐색할 때 각 범주의 상대적인 크기를 이해하는 데 유용합니다.

#### 1.7.2. 사용 시기:
*   단일 범주형 변수의 빈도 분포를 확인하고 싶을 때.
*   각 범주에 속하는 데이터의 양을 비교하고자 할 때.
*   데이터셋의 범주형 변수 구성을 빠르게 파악하고자 할 때.

#### 1.7.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# countplot 그리기
sns.countplot(x="day", data=tips)
plt.title("요일별 관측치 수 (Countplot)")
plt.xlabel("요일")
plt.ylabel("관측치 수")
plt.savefig('countplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# 추가 예시: hue 파라미터 사용 (성별에 따른 요일별 관측치 수)
plt.figure(figsize=(10, 6))
sns.countplot(x="day", hue="sex", data=tips, palette="viridis")
plt.title("요일 및 성별에 따른 관측치 수 (Countplot with Hue)")
plt.xlabel("요일")
plt.ylabel("관측치 수")
plt.legend(title="성별")
plt.show()
```
### 1.8. `catplot()`: 고수준 범주형 플롯 인터페이스
`catplot()`은 Seaborn의 범주형 플롯을 위한 상위 레벨(figure-level) 인터페이스입니다. `boxplot()`, `violinplot()`, `stripplot()`, `swarmplot()`, `barplot()`, `countplot()` 등 다양한 범주형 플롯의 기능을 모두 포함하며, `FacetGrid`를 사용하여 여러 서브플롯을 쉽게 생성하고 비교할 수 있게 해줍니다. `relplot()`이나 `displot()`과 유사하게, `kind` 매개변수를 통해 원하는 플롯 유형을 지정할 수 있습니다.

*   **`kind`**: 그릴 범주형 플롯의 종류를 지정합니다 (예: `'box'`, `'violin'`, `'strip'`, `'swarm'`, `'bar'`, `'count'`).
*   **`col`, `row`**: 특정 범주형 변수에 따라 데이터를 분할하고, 각 서브플롯에 해당 범주의 데이터를 시각화합니다.
*   **`hue`**: 범주형 변수에 따라 색상을 다르게 합니다.

**`catplot()`과 `pointplot()`/`factorplot()`**
*   **`pointplot()`**: `catplot(kind='point')`와 유사하게 범주형 데이터에 대한 연속형 변수의 중심 경향(예: 평균)과 신뢰 구간을 점과 선으로 표시합니다. 이는 범주 간의 변화 추이를 보여줄 때 유용합니다.
*   **`factorplot()`**: `catplot()`의 이전 이름으로, 현재는 `catplot()`으로 대체되었습니다. `factorplot()`은 더 이상 사용되지 않으므로 `catplot()`을 사용하는 것이 권장됩니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# kind='box' (boxplot과 동일)
sns.catplot(x="day", y="total_bill", data=tips, kind="box", height=4, aspect=1.5)
plt.suptitle("catplot (kind='box'): Total Bill by Day", y=1.02)
plt.show()

# kind='violin' (violinplot과 동일)
sns.catplot(x="day", y="total_bill", data=tips, kind="violin", hue="sex", split=True, height=4, aspect=1.5)
plt.suptitle("catplot (kind='violin'): Total Bill by Day and Sex", y=1.02)
plt.show()

# kind='bar' (barplot과 동일)
sns.catplot(x="day", y="total_bill", data=tips, kind="bar", ci="sd", height=4, aspect=1.5)
plt.suptitle("catplot (kind='bar'): Total Bill Mean by Day", y=1.02)
plt.show()

# kind='point' (pointplot과 동일)
sns.catplot(x="day", y="tip", data=tips, kind="point", hue="smoker", height=4, aspect=1.5)
plt.suptitle("catplot (kind='point'): Tip by Day and Smoker", y=1.02)
plt.show()
```