<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 `pyplot` 모듈을 사용하여 에러 바(Error Bars), 2D 히스토그램(2D Histogram), 그리고 헥사곤 비닝(Hexbin) 플롯을 그리는 방법을 다룹니다. 각 플롯의 용도와 함께 데이터 생성부터 그래프 출력까지의 과정을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. Matplotlib 기본 플로팅: 에러 바, 2D 히스토그램, 헥사곤 비닝](#1-matplotlib-기본-플로팅-에러-바-2d-히스토그램-헥사곤-비닝)
  - [1.1. 에러 바 (Error Bars)](#11-에러-바-error-bars)
    - [1.1.1. 에러 바의 개념 및 용도](#111-에러-바의-개념-및-용도)
    - [1.1.2. `plt.errorbar()` 함수를 이용한 에러 바 플롯](#112-plterrorbar-함수를-이용한-에러-바-플롯)
    - [1.1.3. 막대 그래프에 에러 바 추가 (`yerr` 파라미터)](#113-막대-그래프에-에러-바-추가-yerr-파라미터)
  - [1.2. 2D 히스토그램 및 헥사곤 비닝 (2D Histogram & Hexbin)](#12-2d-히스토그램-및-헥사곤-비닝-2d-histogram--hexbin)
    - [1.2.1. 2D 히스토그램 (`plt.hist2d()`)](#121-2d-히스토그램-plthist2d)
    - [1.2.2. 헥사곤 비닝 플롯 (`plt.hexbin()`)](#122-헥사곤-비닝-플롯-plthexbin)
    - [1.2.3. 2D 밀도 플롯 커스터마이징 (bins, gridsize, cmap, colorbar)](#123-2d-밀도-플롯-커스터마이징-bins-gridsize-cmap-colorbar)

---

## 1. Matplotlib 기본 플로팅: 에러 바, 2D 히스토그램, 헥사곤 비닝

### 1.1. 에러 바 (Error Bars)
에러 바(Error Bars)는 데이터 포인트의 측정값에 대한 불확실성이나 오차 범위를 시각적으로 표현하는 데 사용됩니다. 주로 평균값과 함께 표준 편차, 표준 오차, 신뢰 구간 등을 나타내는 데 사용되며, 데이터의 신뢰도를 평가하고 데이터 간의 유의미한 차이를 판단하는 데 중요한 정보를 제공합니다.

#### 1.1.1. 에러 바의 개념 및 용도
*   **개념**: 에러 바는 그래프의 각 데이터 포인트 주변에 그려지는 선으로, 해당 측정값의 변동성 또는 불확실성의 정도를 나타냅니다.
*   **용도**:
    *   **신뢰도 표현**: 측정값의 정밀도나 신뢰 구간을 시각적으로 전달합니다. 에러 바가 짧을수록 측정값의 신뢰도가 높다고 해석할 수 있습니다.
    *   **비교 분석**: 여러 그룹이나 조건 간의 평균값을 비교할 때, 에러 바가 겹치는지 여부를 통해 통계적으로 유의미한 차이가 있는지 직관적으로 판단하는 데 도움을 줍니다.
    *   **데이터 변동성**: 데이터의 분산이나 퍼짐 정도를 보여줍니다.

#### 1.1.2. `plt.errorbar()` 함수를 이용한 에러 바 플롯
`plt.errorbar()` 함수는 라인 플롯이나 산점도에 직접 에러 바를 추가할 때 사용됩니다.

**주요 파라미터**:
*   `x`, `y`: 데이터 포인트의 x, y 좌표.
*   `yerr`: y축 방향의 오차 크기. 단일 값, 배열, 또는 상한/하한을 나타내는 2xN 배열이 될 수 있습니다.
*   `xerr`: x축 방향의 오차 크기 (선택 사항).
*   `fmt`: 라인 플롯의 스타일을 지정합니다 (예: `'o-'`는 원형 마커와 선).
*   `ecolor`: 에러 바의 색상.
*   `capsize`: 에러 바 끝에 있는 캡(cap)의 크기.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 5)
y = np.sin(x)
y_error = 0.1 + 0.2 * np.random.rand(len(x)) # 각 점의 랜덤 오차

# 에러 바를 포함한 라인 플롯 그리기
plt.figure(figsize=(8, 6))
plt.errorbar(x, y, yerr=y_error, fmt='-o', ecolor='red', capsize=5,
             label='Data with Error Bars')
plt.title('Line Plot with Error Bars')
plt.xlabel('X-value')
plt.ylabel('Y-value')
plt.legend()
plt.grid(True)
plt.savefig('line_plot_with_error_bars.png') # 그래프를 이미지 파일로 저장
plt.show()
```

#### 1.1.3. 막대 그래프에 에러 바 추가 (`yerr` 파라미터)
`plt.bar()` 함수나 `plt.barh()` 함수를 사용하여 막대 그래프를 그릴 때, `yerr` (수직 막대) 또는 `xerr` (수평 막대) 파라미터를 통해 에러 바를 쉽게 추가할 수 있습니다.

**코드 설명**:
1.  **데이터 생성**:
    *   `x = np.arange(5)`: 0부터 4까지의 정수 배열을 x축 위치로 사용합니다.
    *   `y = [20, 35, 30, 25, 40]`: 각 막대의 높이(측정값)를 나타냅니다.
    *   `y_error = [2, 3, 4, 2, 5]`: 각 막대에 대한 y축 방향의 오차 크기를 나타냅니다.

2.  **막대 그래프에 에러 바 추가**:
    *   `plt.bar(x, y, yerr=y_error, capsize=5, color='skyblue', ecolor='darkred')`:
        *   `yerr=y_error`: `y_error` 배열에 지정된 값만큼 y축 방향으로 에러 바를 그립니다.
        *   `capsize=5`: 에러 바 끝에 캡을 추가하여 시각적으로 명확하게 합니다.
        *   `color='skyblue'`: 막대의 색상을 하늘색으로 설정합니다.
        *   `ecolor='darkred'`: 에러 바의 색상을 어두운 빨간색으로 설정합니다.

3.  **축 레이블 및 제목 설정**:
    *   `plt.xticks(x, ['G1', 'G2', 'G3', 'G4', 'G5'])`: x축의 눈금 위치(`x`)에 사용자 정의 레이블(`['G1', ...]`)을 설정합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.arange(5)
y = [20, 35, 30, 25, 40]
# 각 데이터 포인트의 오차 (예: 표준 편차)
y_error = [2, 3, 4, 2, 5]

# 에러 바를 포함한 막대 그래프 그리기
plt.figure(figsize=(8, 6))
plt.bar(x, y, yerr=y_error, capsize=5, color='skyblue', ecolor='darkred')
plt.title('Bar Plot with Error Bars')
plt.xlabel('Group')
plt.ylabel('Measurement')
plt.xticks(x, ['G1', 'G2', 'G3', 'G4', 'G5'])
plt.savefig('bar_plot_with_error_bars.png') # 그래프를 이미지 파일로 저장
plt.show()
```

### 1.2. 2D 히스토그램 및 헥사곤 비닝 (2D Histogram & Hexbin)
데이터 포인트가 매우 많아 산점도에서 점들이 서로 겹쳐 분포를 파악하기 어려운 과밀 플롯(overplotting) 문제가 발생할 때, 2D 히스토그램이나 헥사곤 비닝을 사용하면 유용합니다. 이들은 2차원 공간을 사각형 또는 육각형으로 나누고 각 영역에 포함된 데이터 포인트의 개수를 색상으로 표현하여 데이터의 밀도를 효과적으로 시각화합니다.

#### 1.2.1. 2D 히스토그램 (`plt.hist2d()`)
2D 히스토그램은 2차원 평면을 사각형 격자(grid)로 나누고, 각 사각형 셀에 포함된 데이터 포인트의 개수(빈도)를 색상의 강도로 표현합니다. 이는 두 연속형 변수의 결합 분포(joint distribution)를 시각화하는 데 적합합니다.

**코드 설명**:
1.  **데이터 생성**:
    *   `mean = [0, 0]`, `cov = [[1, 1], [1, 2]]`: 평균이 `[0, 0]`이고 공분산 행렬이 `[[1, 1], [1, 2]]`인 2차원 정규 분포를 정의합니다.
    *   `x, y = np.random.multivariate_normal(mean, cov, 10000).T`: 위에서 정의한 분포를 따르는 10,000개의 2차원 데이터 포인트를 생성합니다. `.T`를 사용하여 x와 y 좌표를 각각의 배열로 분리합니다.

2.  **2D 히스토그램 그리기**:
    *   `plt.figure(figsize=(12, 5))`: 두 개의 서브플롯을 나란히 배치하기 위해 전체 그림의 크기를 설정합니다.
    *   `plt.subplot(1, 2, 1)`: 1행 2열의 서브플롯 중 첫 번째(왼쪽) 서브플롯을 선택합니다.
    *   `plt.hist2d(x, y, bins=30, cmap='Blues')`: `x`와 `y` 데이터를 사용하여 2D 히스토그램을 그립니다.
        *   `bins=30`: x축과 y축을 각각 30개의 구간으로 나누어 총 30x30개의 사각형 셀을 만듭니다.
        *   `cmap='Blues'`: 'Blues' 컬러맵을 사용하여 빈도수가 높을수록 더 진한 파란색으로 표시합니다.
    *   `plt.colorbar(label='Count in bin')`: 각 색상이 나타내는 빈도수를 설명하는 컬러바를 추가합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (다변량 정규 분포)
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T

# 2D 히스토그램
plt.figure(figsize=(7, 6)) # 단일 플롯을 위해 크기 조정
plt.hist2d(x, y, bins=30, cmap='Blues')
plt.colorbar(label='Count in bin')
plt.title('2D Histogram')
plt.xlabel('X-variable')
plt.ylabel('Y-variable')
plt.savefig('2d_histogram.png') # 그래프를 이미지 파일로 저장
plt.show()
```

#### 1.2.2. 헥사곤 비닝 플롯 (`plt.hexbin()`)
헥사곤 비닝 플롯은 2차원 평면을 육각형(hexagon) 격자로 나누고, 각 육각형 셀에 포함된 데이터 포인트의 개수를 색상의 강도로 표현합니다. 육각형은 사각형보다 시각적으로 덜 왜곡되고 인접한 셀과의 관계를 더 잘 보여주는 경향이 있어 데이터 밀도 시각화에 선호되기도 합니다.

**코드 설명**:
*   `plt.subplot(1, 2, 2)`: 1행 2열의 서브플롯 중 두 번째(오른쪽) 서브플롯을 선택합니다.
*   `plt.hexbin(x, y, gridsize=30, cmap='inferno')`: `x`와 `y` 데이터를 사용하여 헥사곤 비닝 플롯을 그립니다.
    *   `gridsize=30`: 육각형 격자의 밀도를 지정합니다. 값이 클수록 더 많은 육각형 셀이 생성됩니다.
    *   `cmap='inferno'`: 'inferno' 컬러맵을 사용하여 밀도를 색상으로 표현합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (이전과 동일)
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T

# 헥사곤 비닝 플롯
plt.figure(figsize=(7, 6)) # 단일 플롯을 위해 크기 조정
plt.hexbin(x, y, gridsize=30, cmap='inferno')
plt.colorbar(label='Count in bin')
plt.title('Hexbin Plot')
plt.xlabel('X-variable')
plt.ylabel('Y-variable')
plt.savefig('hexbin_plot.png') # 그래프를 이미지 파일로 저장
plt.show()
```

#### 1.2.3. 2D 밀도 플롯 커스터마이징 (bins, gridsize, cmap, colorbar)
`plt.hist2d()`와 `plt.hexbin()` 함수는 모두 데이터 밀도 시각화를 위한 다양한 커스터마이징 옵션을 제공합니다.

*   **`bins` (hist2d)** / **`gridsize` (hexbin)**: 격자의 해상도를 조절합니다. 값이 클수록 더 세밀한 밀도 표현이 가능하지만, 데이터가 적을 경우 빈 셀이 많아질 수 있습니다.
*   **`cmap`**: 밀도를 나타내는 색상 스펙트럼을 지정합니다. 데이터의 연속적인 변화를 잘 보여주는 순차적(sequential) 컬러맵(예: 'Blues', 'Greens', 'viridis', 'plasma', 'inferno', 'magma')을 사용하는 것이 좋습니다.
*   **`cmin`**: 최소 빈도수를 설정하여 그 이하의 빈도를 가진 셀은 표시하지 않도록 합니다. 노이즈를 줄이는 데 유용합니다.
*   **`norm`**: 색상 매핑에 사용할 정규화(normalization)를 지정합니다. 예를 들어, `LogNorm`을 사용하여 빈도수가 넓은 범위에 걸쳐 있을 때 색상 구분을 더 명확하게 할 수 있습니다.
*   **`plt.colorbar()`**: 밀도 값을 색상으로 매핑한 기준을 보여주는 컬러바를 추가합니다.

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm # LogNorm 임포트

# 데이터 생성 (이전과 동일)
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T

plt.figure(figsize=(14, 6))

# 커스터마이징된 2D 히스토그램
plt.subplot(1, 2, 1)
hist = plt.hist2d(x, y, bins=50, cmap='Greens', cmin=1, norm=LogNorm()) # bins 증가, cmin 설정, LogNorm 적용
plt.colorbar(hist[3], label='Log(Count in bin)') # LogNorm 적용 시 컬러바 레이블 변경
plt.title('Custom 2D Histogram (Log Scale)', fontsize=16)
plt.xlabel('X-variable')
plt.ylabel('Y-variable')

# 커스터마이징된 헥사곤 비닝 플롯
plt.subplot(1, 2, 2)
hexb = plt.hexbin(x, y, gridsize=40, cmap='magma', mincnt=1, norm=LogNorm()) # gridsize 증가, mincnt 설정, LogNorm 적용
plt.colorbar(hexb, label='Log(Count in bin)')
plt.title('Custom Hexbin Plot (Log Scale)', fontsize=16)
plt.xlabel('X-variable')
plt.ylabel('Y-variable')

plt.tight_layout()
plt.show()
```
