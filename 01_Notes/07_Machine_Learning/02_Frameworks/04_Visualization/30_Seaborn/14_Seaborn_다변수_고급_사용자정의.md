<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 다변수 분석 플롯(`pairplot()`, `jointplot()`)을 사용하여 데이터셋의 여러 변수 간 관계를 한 번에 파악하는 방법을 다룹니다. 또한, `FacetGrid`를 이용한 고급 플롯 제어, 그리고 Seaborn의 사용자 정의 및 테마 설정을 통해 플롯의 미적 품질을 향상시키는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

* [1. Seaborn 다변수 및 고급 사용자 정의 플롯 개요](#1-seaborn-다변수-및-고급-사용자-정의-플롯-개요)
  * [1.1. 다변수 분석 플롯 (Multivariate Analysis Plots)](#11-다변수-분석-플롯-multivariate-analysis-plots)
    * [1.1.1. `pairplot()`: 쌍 플롯](#111-pairplot-쌍-플롯)
    * [1.1.2. `PairGrid`: 쌍 플롯의 고급 제어](#112-pairgrid-쌍-플롯의-고급-제어)
    * [1.1.3. `jointplot()`: 조인트 플롯](#113-jointplot-조인트-플롯)
  * [1.2. 고급 플롯 제어: FacetGrid](#12-고급-플롯-제어-facetgrid)
  * [1.3. 사용자 정의 및 테마](#13-사용자-정의-및-테마)

---

## 1. Seaborn 다변수 및 고급 사용자 정의 플롯 개요
이 섹션에서는 Seaborn의 다변수 분석 플롯(`pairplot()`, `jointplot()`)을 사용하여 데이터셋의 여러 변수 간 관계를 한 번에 파악하는 방법을 다룹니다. 또한, `FacetGrid`를 이용한 고급 플롯 제어, 그리고 Seaborn의 사용자 정의 및 테마 설정을 통해 플롯의 미적 품질을 향상시키는 방법을 실제 코드 예제를 통해 학습합니다.

### 1.1. 다변수 분석 플롯 (Multivariate Analysis Plots)
Seaborn의 가장 강력한 기능 중 하나로, 데이터셋의 여러 변수 간 관계를 한 번에 파악할 수 있는 고수준 플롯을 제공합니다. 탐색적 데이터 분석(EDA) 과정에서 매우 유용합니다.

#### 1.1.1. `pairplot()`: 쌍 플롯
`pairplot()`은 데이터프레임 내의 모든 숫자형 변수 쌍에 대한 관계를 한눈에 파악할 수 있도록 도와주는 강력한 다변수 분석 플롯입니다. 데이터셋의 전반적인 구조, 변수 간의 상관 관계, 그리고 각 변수의 분포를 빠르게 탐색하는 데 매우 유용합니다.

##### 1.1.1.1. 주요 특징:
*   **산점도 행렬:** 플롯의 비대각선(off-diagonal) 부분에는 모든 숫자형 변수 쌍에 대한 산점도(scatterplot)가 그려집니다. 이를 통해 두 변수 간의 선형 또는 비선형 관계, 군집 등을 시각적으로 확인할 수 있습니다.
*   **대각선 분포:** 대각선(diagonal) 부분에는 각 변수의 단변수 분포를 보여주는 그래프가 그려집니다. 기본적으로 히스토그램이 사용되지만, `diag_kind='kde'`를 설정하여 커널 밀도 추정(KDE) 플롯으로 변경할 수 있습니다.
*   **`hue` 파라미터:** `hue` 파라미터를 사용하여 범주형 변수를 지정하면, 해당 범주에 따라 데이터 포인트의 색상을 다르게 표시하여 그룹별 관계 및 분포의 차이를 쉽게 비교할 수 있습니다.
*   **`kind` 파라미터:** 산점도 대신 다른 종류의 플롯을 지정할 수 있습니다 (예: `kind='reg'`로 회귀선 추가).
*   **`vars` 파라미터:** 모든 숫자형 변수가 아닌, 특정 변수들만 선택하여 플롯을 그릴 수 있습니다.

##### 1.1.1.2. 사용 시기:
*   데이터셋을 처음 탐색할 때 변수 간의 전반적인 관계를 빠르게 파악하고자 할 때.
*   특정 그룹(예: `hue`로 구분된)에 따라 변수 간의 관계나 분포가 어떻게 달라지는지 비교하고자 할 때.
*   이상치나 특이한 패턴을 시각적으로 식별하고자 할 때.

##### 1.1.1.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
iris = sns.load_dataset("iris")

# pairplot 그리기 (기본: hue 사용)
sns.pairplot(iris, hue="species") # 종(species)에 따라 색상 구분
plt.suptitle("Iris Dataset Pair Plot (Species Hue)", y=1.02) # 전체 제목 추가
plt.show()

# 추가 예시: diag_kind='kde', kind='reg'
plt.figure(figsize=(10, 8))
sns.pairplot(iris, diag_kind='kde', kind='reg', hue="species", palette="viridis")
plt.suptitle("Iris Dataset Pair Plot (KDE Diagonal, Regression Kind)", y=1.02)
plt.show()
```

#### 1.1.2. `PairGrid`: 쌍 플롯의 고급 제어

`PairGrid`는 `pairplot()`의 기반이 되는 저수준(low-level) 인터페이스로, `pairplot()`보다 더 유연하고 세밀한 사용자 정의가 가능합니다. `FacetGrid`와 유사하게, `PairGrid` 객체를 생성한 후 `map_diag()`, `map_upper()`, `map_lower()` 메서드를 사용하여 대각선, 상단 삼각형, 하단 삼각형에 각각 다른 플롯 함수를 매핑할 수 있습니다.

##### 1.1.2.1. 주요 특징:
*   **세밀한 제어:** `pairplot()`이 제공하는 기본 옵션 외에, 각 서브플롯 영역(대각선, 상단, 하단)에 원하는 Matplotlib 또는 Seaborn 플롯 함수를 자유롭게 적용할 수 있습니다.
*   **유연한 매핑:** `map_diag()` (대각선), `map_upper()` (상단 삼각형), `map_lower()` (하단 삼각형) 메서드를 통해 각 영역에 독립적인 시각화를 구성할 수 있습니다.
*   **`hue` 지원:** `hue` 파라미터를 사용하여 그룹별로 플롯을 구분할 수 있습니다.

##### 1.1.2.2. 사용 시기:
*   `pairplot()`의 기본 기능으로는 부족하여 각 서브플롯 영역에 특정 플롯 유형이나 커스터마이징을 적용해야 할 때.
*   데이터의 특정 관계에 초점을 맞춰 시각화를 정교하게 제어하고자 할 때.

##### 1.1.2.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 예시 데이터 로드
iris = sns.load_dataset("iris")

# PairGrid 객체 생성
g = sns.PairGrid(iris, hue="species")

# 대각선에 히스토그램 매핑
g.map_diag(sns.histplot, kde=True)

# 상단 삼각형에 산점도 매핑
g.map_upper(sns.scatterplot)

# 하단 삼각형에 KDE 플롯 매핑
g.map_lower(sns.kdeplot)

# 범례 추가
g.add_legend()

plt.suptitle("Iris Dataset PairGrid Example", y=1.02)
plt.show()

# 추가 예시: 다른 플롯 함수 매핑
g_reg = sns.PairGrid(iris, hue="species")
g_reg.map_diag(sns.histplot, kde=True)
g_reg.map_upper(sns.regplot, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}) # 상단에 회귀선 포함 산점도
g_reg.map_lower(sns.kdeplot, fill=True)
g_reg.add_legend()
plt.suptitle("Iris Dataset PairGrid with Regression", y=1.02)
plt.show()
```

#### 1.1.3. `jointplot()`: 조인트 플롯
`jointplot()`은 두 변수 간의 관계(중앙 플롯)와 각 변수의 개별 분포(주변 플롯)를 동시에 시각화하는 강력한 도구입니다. 이를 통해 두 변수의 상관 관계뿐만 아니라 각 변수의 분포 특성까지 한 번에 파악할 수 있어, 데이터의 심층적인 탐색에 매우 효과적입니다.

##### 1.1.3.1. 주요 특징:
*   **중앙 플롯:** `kind` 파라미터에 따라 다양한 유형의 플롯을 중앙에 그릴 수 있습니다.
    *   `"scatter"` (기본값): 산점도.
    *   `"kde"`: 커널 밀도 추정 플롯.
    *   `"hex"`: 헥스빈 플롯 (데이터 밀도를 육각형으로 표시).
    *   `"reg"`: 선형 회귀 플롯.
*   **주변 플롯 (Marginal Plots):** 중앙 플롯의 위쪽과 오른쪽에 각 변수의 단변수 분포를 보여주는 플롯이 배치됩니다. 기본적으로 히스토그램이 사용되지만, `marginal_kws`를 통해 KDE 플롯 등으로 변경할 수 있습니다.
*   **데이터 관계와 분포 동시 파악:** 두 변수 간의 관계(예: 선형성, 군집)와 각 변수의 분포 형태(예: 정규성, 왜도)를 동시에 시각적으로 분석할 수 있습니다.

##### 1.1.3.2. 사용 시기:
*   두 연속형 변수 간의 관계와 각 변수의 분포를 상세하게 분석하고 싶을 때.
*   데이터의 이상치나 특정 패턴이 두 변수의 결합 분포에서 어떻게 나타나는지 확인하고자 할 때.
*   다양한 `kind` 옵션을 통해 데이터의 특성에 맞는 최적의 시각화를 선택하고자 할 때.

##### 1.1.3.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# 산점도와 히스토그램을 함께 표시 (기본)
sns.jointplot(x="total_bill", y="tip", data=tips, kind="scatter")
plt.suptitle("총 계산액과 팁의 Joint Plot (산점도)", y=1.02)
plt.show()

# 헥스빈(hexbin)과 KDE를 함께 표시
sns.jointplot(x="total_bill", y="tip", data=tips, kind="hex", cmap="hot")
plt.suptitle("총 계산액과 팁의 Joint Plot (헥스빈)", y=1.02)
plt.show()

# 추가 예시: KDE와 주변 KDE 플롯
sns.jointplot(x="total_bill", y="tip", data=tips, kind="kde", fill=True, cmap="Blues",
              marginal_kws=dict(bins=20, kde=True))
plt.suptitle("총 계산액과 팁의 Joint Plot (KDE)", y=1.02)
plt.show()
```

### 1.2. 고급 플롯 제어: FacetGrid
`FacetGrid`는 Seaborn에서 제공하는 강력한 "figure-level" 함수로, 데이터의 하위 집합(subset)에 따라 여러 개의 서브플롯(facet)을 만들어 동일한 종류의 그래프를 그리는 데 사용됩니다. `col`, `row`, `hue` 등의 변수를 기준으로 데이터를 나누어 시각화함으로써, 복잡한 데이터의 패턴을 다각도에서 비교 분석하고 그룹 간의 차이를 명확하게 드러낼 수 있습니다.

#### 1.2.1. 주요 특징:
*   **다중 서브플롯 생성:** `FacetGrid` 객체를 생성할 때 `col`, `row`, `hue` 파라미터를 사용하여 서브플롯의 레이아웃과 색상 구분을 정의합니다.
    *   `col`: 지정된 변수의 각 고유 값에 따라 열(column) 방향으로 서브플롯을 생성합니다.
    *   `row`: 지정된 변수의 각 고유 값에 따라 행(row) 방향으로 서브플롯을 생성합니다.
    *   `hue`: 지정된 변수의 각 고유 값에 따라 플롯 내의 데이터 포인트 색상을 다르게 합니다.
*   **`map()` 메서드:** `FacetGrid` 객체에 `map()` 메서드를 사용하여 각 서브플롯에 그릴 플롯 함수(예: `sns.scatterplot`, `plt.hist`)와 해당 함수의 인자들을 전달합니다. `map()`은 각 서브플롯에 동일한 플롯을 적용하여 일관된 비교를 가능하게 합니다.
*   **유연한 레이아웃:** `col_wrap` 파라미터를 사용하여 `col`로 생성된 서브플롯의 열 개수를 제한하여 플롯의 가독성을 높일 수 있습니다. `height`와 `aspect`를 통해 각 서브플롯의 크기를 조절할 수 있습니다.
*   **`add_legend()`:** `hue` 파라미터를 사용했을 경우, `g.add_legend()`를 호출하여 범례를 추가할 수 있습니다.

#### 1.2.2. 사용 시기:
*   데이터셋 내의 특정 범주형 변수에 따라 다른 변수 간의 관계가 어떻게 변화하는지 비교하고 싶을 때.
*   복잡한 다변량 데이터를 여러 관점에서 체계적으로 탐색하고 싶을 때.
*   동일한 유형의 플롯을 여러 그룹에 대해 반복적으로 그려야 할 때.

#### 1.2.3. 기본 사용법 및 커스터마이징
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 예시 데이터 로드
tips = sns.load_dataset("tips")

# FacetGrid 객체 생성 및 매핑 (시간과 성별에 따른 총 계산액과 팁의 관계)
g = sns.FacetGrid(tips, col="time", row="sex", hue="smoker",
                  height=3, aspect=1.2, col_wrap=2) # col_wrap으로 열 개수 제한
g.map(sns.scatterplot, "total_bill", "tip", alpha=.7)
g.add_legend(title="흡연 여부") # 범례 추가 및 제목 설정

plt.suptitle("시간 및 성별에 따른 총 계산액과 팁의 관계 (흡연 여부별)", y=1.02)
plt.show()

# 추가 예시: 다른 플롯 함수 매핑 (히스토그램)
g_hist = sns.FacetGrid(tips, col="day", col_wrap=2, height=3, aspect=1.2)
g_hist.map(sns.histplot, "total_bill", kde=True, bins=15)
plt.suptitle("요일별 총 계산액 분포 (히스토그램)", y=1.02)
plt.show()
```

### 1.3. 사용자 정의 및 테마
Seaborn은 Matplotlib을 기반으로 구축되었기 때문에, Matplotlib의 강력한 커스터마이징 기능을 그대로 활용할 수 있습니다. 동시에 Seaborn은 자체적으로 플롯의 미적 품질을 향상시키고 일관된 스타일을 적용할 수 있는 다양한 테마와 스타일링 도구를 제공합니다. 이를 통해 적은 노력으로도 전문적이고 시각적으로 매력적인 플롯을 생성할 수 있습니다.

#### 1.3.1. 주요 커스터마이징 기능:
*   **테마 설정 (`sns.set_theme()` 또는 `sns.set_style()`):**
    *   `sns.set_theme()`: 플롯의 전반적인 스타일(배경, 격자, 폰트 등)을 설정합니다. `style` 파라미터로 `'darkgrid'`, `'whitegrid'`, `'dark'`, `'white'`, `'ticks'` 등을 지정할 수 있습니다.
    *   `sns.set_style()`: `set_theme()`과 유사하게 스타일을 설정하지만, `set_theme()`은 더 많은 전역 설정을 포함합니다.
*   **색상 팔레트 (`sns.color_palette()`, `sns.set_palette()`):**
    *   `sns.color_palette()`: 특정 색상 팔레트를 생성합니다.
    *   `sns.set_palette()`: 모든 후속 플롯에 적용될 기본 색상 팔레트를 설정합니다. `'deep'`, `'muted'`, `'pastel'`, `'bright'`, `'dark'`, `'colorblind'` 등 다양한 내장 팔레트가 있습니다.
*   **플롯 요소 커스터마이징 (Matplotlib 활용):**
    *   `plt.figure(figsize=(width, height))`: 플롯의 전체 크기를 조절합니다.
    *   `plt.title()`, `plt.xlabel()`, `plt.ylabel()`: 제목과 축 라벨을 설정합니다.
    *   `plt.legend()`: 범례를 추가하거나 커스터마이징합니다.
    *   `plt.suptitle()`: 전체 그림(figure)에 대한 제목을 설정합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# Seaborn 테마 및 팔레트 설정
sns.set_theme(style="whitegrid", palette="pastel")

# 사용자 정의 산점도
plt.figure(figsize=(10, 7))
sns.scatterplot(x="total_bill", y="tip", hue="time", size="size", data=tips,
                sizes=(20, 400), alpha=0.7, edgecolor="w")
plt.title("사용자 정의 산점도 (Seaborn 테마 및 팔레트)", fontsize=16)
plt.xlabel("총 계산액 ($)", fontsize=12)
plt.ylabel("팁 ($)", fontsize=12)
plt.legend(title="식사 시간", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Matplotlib 함수와 함께 사용 예시 (히스토그램)
sns.set_theme(style="darkgrid", palette="deep")
plt.figure(figsize=(8, 6))
sns.histplot(x="total_bill", data=tips, kde=True, color="skyblue", bins=20)
plt.title("Matplotlib 커스터마이징을 포함한 히스토그램", fontsize=14)
plt.suptitle("Matplotlib과 Seaborn 함께 사용", fontsize=18, y=1.03)
plt.xlabel("총 계산액", fontsize=12)
plt.ylabel("빈도", fontsize=12)
plt.show()
```

#### 1.3.2. 통계적 유의성 주석(Annotation) 추가
Seaborn으로 그린 Box plot이나 Bar plot에서 두 그룹 간의 차이가 통계적으로 유의미한지 여부를 시각적으로 보여주는 것은 분석의 신뢰도를 높입니다. `statannot`과 같은 보조 라이브러리(별도 설치 필요: `pip install statannot`)를 사용하면, 두 그룹 사이에 p-value나 유의수준을 나타내는 별표(`*`, `**`, `***`)를 쉽게 추가할 수 있습니다. 이는 시각화에 통계적 검정 결과를 직접 통합하여 해석을 돕습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 이 예시를 실행하려면 'pip install statannot'이 필요합니다.
try:
    from statannot.statannot import add_stat_annotation
    
    tips = sns.load_dataset("tips")
    
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(x="day", y="total_bill", data=tips, palette="viridis")
    
    # 통계적 유의성 주석 추가
    add_stat_annotation(ax, data=tips, x="day", y="total_bill",
                        box_pairs=[("Thur", "Fri"), ("Fri", "Sun"), ("Sat", "Sun")],
                        test='t-test_ind', text_format='star', loc='inside', verbose=2)
                        
    plt.title("요일별 총 계산액 분포 (통계적 유의성 주석 포함)")
    plt.show()

except ImportError:
    print("statannot 라이브러리가 설치되어 있지 않습니다. 'pip install statannot'으로 설치해주세요.")
    # 대체 코드: statannot 없이 boxplot만 그리기
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="day", y="total_bill", data=tips, palette="viridis")
    plt.title("요일별 총 계산액 분포")
    plt.show()
```
