<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Seaborn의 분포 플롯(`histplot()`, `kdeplot()`, `displot()`)을 사용하여 단일 변수 또는 여러 변수의 분포를 시각화하는 방법을 다룹니다. 히스토그램, 커널 밀도 추정(KDE) 플롯, 그리고 고수준 인터페이스인 `displot()`을 통해 데이터의 밀도와 분포 형태를 파악하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. Seaborn 분포 플롯 개요](#1-seaborn-분포-플롯-개요)
  - [1.1. `histplot()`: 히스토그램](#11-histplot-히스토그램)
    - [1.1.1. 단일 변수 히스토그램 기본 사용법](#111-단일-변수-히스토그램-기본-사용법)
    - [1.1.2. `bins`, `kde`, `hue` 매개변수를 이용한 커스터마이징](#112-bins-kde-hue-매개변수를-이용한-커스터마이징)
    - [1.1.3. 누적 히스토그램 및 통계량 표시](#113-누적-히스토그램-및-통계량-표시)
  - [1.2. `kdeplot()`: 커널 밀도 추정 플롯](#12-kdeplot-커널-밀도-추정-플롯)
    - [1.2.1. 단일 변수 KDE 플롯](#121-단일-변수-kde-플롯)
    - [1.2.2. 두 변수의 결합 분포 (2D KDE)](#122-두-변수의-결합-분포-2d-kde)
    - [1.2.3. `fill`, `hue`, `levels` 매개변수를 이용한 커스터마이징](#123-fill-hue-levels-매개변수를-이용한-커스터마이징)
  - [1.3. `rugplot()`: 데이터 포인트 위치 시각화](#13-rugplot-데이터-포인트-위치-시각화)
  - [1.4. `displot()`: 고수준 분포 플롯 인터페이스](#14-displot-고수준-분포-플롯-인터페이스)
    - [1.4.1. `kind` 매개변수를 이용한 히스토그램, KDE, ECDF](#141-kind-매개변수를-이용한-히스토그램-kde-ecdf)
    - [1.4.2. `col`, `row` 매개변수를 이용한 서브플롯 생성](#142-col-row-매개변수를-이용한-서브플롯-생성)

---

## 1. Seaborn 분포 플롯 개요

Seaborn의 분포 플롯(Distribution Plots)은 단일 변수 또는 여러 변수의 분포를 시각화하는 데 사용됩니다. 데이터의 밀도, 빈도, 형태 등을 파악하여 데이터셋의 특성을 이해하는 데 도움을 줍니다. 주요 함수로는 `histplot()`, `kdeplot()`, `displot()`이 있습니다.

### 1.1. `histplot()`: 히스토그램

`histplot()`은 단일 변수의 분포를 막대(bins)로 표현하는 히스토그램을 그립니다. 데이터의 빈도 분포를 직관적으로 파악할 수 있습니다.

#### 1.1.1. 단일 변수 히스토그램 기본 사용법

`histplot()`에 `x` 또는 `y` 매개변수로 분포를 보고자 하는 변수를 지정합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# 총 계산액(total_bill)의 분포를 히스토그램으로 표현
plt.figure(figsize=(8, 5))
sns.histplot(x="total_bill", data=tips)
plt.title("histplot: Distribution of Total Bill (Basic)")
plt.savefig('histplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()
```

#### 1.1.2. `bins`, `kde`, `hue` 매개변수를 이용한 커스터마이징

*   `bins`: 히스토그램의 막대(bin) 개수 또는 경계를 지정합니다.
*   `kde`: `True`로 설정하면 커널 밀도 추정(KDE) 곡선을 함께 표시합니다.
*   `hue`: 범주형 변수에 따라 분포를 분리하여 색상으로 구분합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(12, 5))

# bins=15, kde=True로 커널 밀도 추정 곡선 추가
plt.subplot(1, 2, 1)
sns.histplot(x="total_bill", data=tips, bins=15, kde=True)
plt.title("histplot: Total Bill with KDE")

# 성별(sex)에 따라 분포를 분리하여 히스토그램 그리기
plt.subplot(1, 2, 2)
sns.histplot(x="total_bill", data=tips, hue="sex", bins=20, kde=True)
plt.title("histplot: Total Bill by Sex")

plt.tight_layout()
plt.show()
```

#### 1.1.3. 누적 히스토그램 및 통계량 표시

*   `cumulative`: `True`로 설정하면 누적 히스토그램을 그립니다.
*   `stat`: 막대의 높이를 어떤 통계량으로 나타낼지 지정합니다. (예: `'count'` (기본값), `'frequency'`, `'density'`, `'probability'`)

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(12, 5))

# 누적 히스토그램
plt.subplot(1, 2, 1)
sns.histplot(x="total_bill", data=tips, cumulative=True, stat="density")
plt.title("histplot: Cumulative Density of Total Bill")

# 막대 높이를 확률로 표시
plt.subplot(1, 2, 2)
sns.histplot(x="total_bill", data=tips, stat="probability")
plt.title("histplot: Probability of Total Bill")

plt.tight_layout()
plt.show()
```

### 1.2. `kdeplot()`: 커널 밀도 추정 플롯

`kdeplot()`은 데이터의 분포를 부드러운 곡선으로 표현하는 커널 밀도 추정(Kernel Density Estimate, KDE) 플롯을 그립니다. 히스토그램보다 데이터의 밀도 변화를 더 명확하게 보여줄 수 있습니다.

#### 1.2.1. 단일 변수 KDE 플롯

단일 변수의 확률 밀도 함수를 추정하여 시각화합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# 총 계산액(total_bill)의 KDE 플롯
plt.figure(figsize=(8, 5))
sns.kdeplot(x="total_bill", data=tips)
plt.title("kdeplot: KDE of Total Bill (Basic)")
plt.savefig('kdeplot_basic.png') # 그래프를 이미지 파일로 저장
plt.show()
```

#### 1.2.2. 두 변수의 결합 분포 (2D KDE)

`x`와 `y` 매개변수를 모두 사용하여 두 변수의 결합 분포를 2D KDE 플롯으로 시각화할 수 있습니다. 등고선(contour) 형태로 밀도를 표현합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# 총 계산액(total_bill)과 팁(tip)의 2D KDE 플롯
plt.figure(figsize=(8, 7))
sns.kdeplot(x="total_bill", y="tip", data=tips)
plt.title("kdeplot: 2D KDE of Total Bill and Tip")
plt.savefig('kdeplot_2d_basic.png') # 그래프를 이미지 파일로 저장
plt.show()
```

#### 1.2.3. `fill`, `hue`, `levels` 매개변수를 이용한 커스터마이징

*   `fill`: `True`로 설정하면 KDE 곡선 아래 영역을 채웁니다.
*   `hue`: 범주형 변수에 따라 KDE 플롯을 분리하여 색상으로 구분합니다.
*   `levels`: 2D KDE 플롯에서 등고선의 개수를 지정합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(12, 5))

# fill=True로 영역 채우기
plt.subplot(1, 2, 1)
sns.kdeplot(x="total_bill", data=tips, fill=True)
plt.title("kdeplot: Total Bill (Filled)")

# 성별(sex)에 따라 KDE 플롯 분리
plt.subplot(1, 2, 2)
sns.kdeplot(x="total_bill", data=tips, hue="sex", fill=True)
plt.title("kdeplot: Total Bill by Sex (Filled)")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 7))
# 2D KDE 플롯에서 등고선 레벨 조절
sns.kdeplot(x="total_bill", y="tip", data=tips, hue="sex", levels=5, fill=True, cmap="Reds")
plt.title("kdeplot: 2D KDE by Sex (Levels & Fill)")
plt.show()

```

### 1.3. `rugplot()`: 데이터 포인트 위치 시각화
`rugplot()`은 축을 따라 작은 선분(rug)을 그려 각 데이터 포인트의 정확한 위치를 보여주는 플롯입니다. 단독으로 사용되기보다는 `histplot`이나 `kdeplot`과 같은 다른 분포 플롯 위에 추가하여, 전체적인 분포 형태와 함께 개별 데이터의 위치를 동시에 파악하는 데 매우 유용합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plt.figure(figsize=(8, 5))
sns.kdeplot(x="total_bill", data=tips)
sns.rugplot(x="total_bill", data=tips, color="purple", height=0.05) # height로 선분 길이 조절
plt.title("KDE Plot with Rug Plot: Distribution of Total Bill")
plt.show()
```

### 1.4. `displot()`: 고수준 분포 플롯 인터페이스

`displot()`은 Seaborn의 분포 플롯을 위한 상위 레벨(figure-level) 인터페이스입니다. `histplot()`, `kdeplot()`, `ecdfplot()`의 기능을 모두 포함하며, `FacetGrid`를 사용하여 여러 서브플롯을 쉽게 생성하고 비교할 수 있게 해줍니다.

#### 1.4.1. `kind` 매개변수를 이용한 히스토그램, KDE, ECDF

`displot()`의 `kind` 매개변수를 `'hist'`, `'kde'`, 또는 `'ecdf'`로 설정하여 원하는 유형의 분포 플롯을 그릴 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# kind='hist' (히스토그램)
sns.displot(x="total_bill", data=tips, kind="hist", bins=15, aspect=1.5)
plt.suptitle("displot (kind='hist'): Total Bill", y=1.02)
plt.savefig('displot_hist_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# kind='kde' (KDE 플롯)
sns.displot(x="total_bill", data=tips, kind="kde", fill=True, aspect=1.5)
plt.suptitle("displot (kind='kde'): Total Bill", y=1.02)
plt.savefig('displot_kde_basic.png') # 그래프를 이미지 파일로 저장
plt.show()

# kind='ecdf' (경험적 누적 분포 함수)
sns.displot(x="total_bill", data=tips, kind="ecdf", aspect=1.5)
plt.suptitle("displot (kind='ecdf'): Total Bill", y=1.02)
plt.savefig('displot_ecdf_basic.png') # 그래프를 이미지 파일로 저장
plt.show()
```

#### 1.4.2. `col`, `row` 매개변수를 이용한 서브플롯 생성

`displot()`의 강력한 기능 중 하나는 `col` 또는 `row` 매개변수를 사용하여 특정 범주형 변수에 따라 데이터를 분할하고, 각 서브플롯에 해당 범주의 데이터를 시각화하는 것입니다. 이는 여러 그룹 간의 분포를 쉽게 비교할 수 있게 해줍니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# 성별(sex)에 따라 열(col)을 나누어 히스토그램 그리기
sns.displot(x="total_bill", data=tips, kind="hist", col="sex", bins=15)
plt.suptitle("displot: Total Bill by Sex (col)", y=1.02)
plt.show()

# 요일(day)에 따라 행(row)을 나누어 KDE 플롯 그리기
sns.displot(x="total_bill", data=tips, kind="kde", row="day", fill=True)
plt.suptitle("displot: Total Bill by Day (row)", y=1.02)
plt.show()
```
