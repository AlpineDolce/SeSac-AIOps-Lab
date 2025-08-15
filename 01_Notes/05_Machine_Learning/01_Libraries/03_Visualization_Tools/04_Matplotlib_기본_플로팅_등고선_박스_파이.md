<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 `pyplot` 모듈을 사용하여 등고선 플롯(Contour Plot), 박스 플롯(Box Plot), 파이 차트(Pie Chart)를 그리는 방법을 다룹니다. 각 플롯의 용도와 함께 데이터 생성부터 그래프 출력까지의 과정을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. Matplotlib 기본 플로팅: 등고선, 박스, 파이 차트](#1-matplotlib-기본-플로팅-등고선-박스-파이-차트)
  - [1.1. 등고선 플롯 (Contour Plot)](#11-등고선-플롯-contour-plot)
    - [1.1.1. 등고선 (선) 플롯 그리기 (`plt.contour()`)](#111-등고선-선-플롯-그리기-pltcontour)
    - [1.1.2. 채워진 등고선 플롯 그리기 (`plt.contourf()`)](#112-채워진-등고선-플롯-그리기-pltcontourf)
    - [1.1.3. 등고선 플롯 커스터마이징 (레벨, 색상, 컬러맵, 레이블)](#113-등고선-플롯-커스터마이징-레벨-색상-컬러맵-레이블)
  - [1.2. 박스 플롯 (Box Plot)](#12-박스-플롯-box-plot)
    - [1.2.1. 기본 박스 플롯 그리기](#121-기본-박스-플롯-그리기)
    - [1.2.2. 여러 그룹의 박스 플롯 비교](#122-여러-그룹의-박스-플롯-비교)
    - [1.2.3. 박스 플롯 커스터마이징 (색상, 이상치, 방향)](#123-박스-플롯-커스터마이징-색상-이상치-방향)
  - [1.3. 파이 차트 (Pie Chart)](#13-파이-차트-pie-chart)
    - [1.3.1. 기본 파이 차트 그리기](#131-기본-파이-차트-그리기)
    - [1.3.2. 파이 차트 커스터마이징 (비율 표시, 조각 분리, 그림자)](#132-파이-차트-커스터마이징-비율-표시-조각-분리-그림자)

---

## 1. Matplotlib 기본 플로팅: 등고선, 박스, 파이 차트

### 1.1. 등고선 플롯 (Contour Plot)
등고선 플롯은 3차원 데이터를 2차원 평면에 표현하는 방법입니다. 동일한 값을 갖는 지점들을 선으로 연결하여 지형도처럼 표현하며, 두 변수에 따른 제3의 변수(높이, 밀도 등)의 변화를 시각화하는 데 매우 유용합니다. 데이터 과학에서는 두 변수의 확률 밀도 함수나 머신러닝 모델의 결정 경계(Decision Boundary)를 시각화하는 데 자주 사용됩니다.

등고선 플롯을 그리기 위해서는 3차원 데이터가 필요하며, 이는 일반적으로 `X`, `Y`, `Z` 세 개의 2차원 배열로 표현됩니다. `X`와 `Y`는 각각 x축과 y축의 좌표를 나타내고, `Z`는 해당 (x, y) 좌표에서의 값을 나타냅니다. `numpy.meshgrid` 함수를 사용하여 1차원 배열로부터 2차원 그리드 배열을 쉽게 생성할 수 있습니다.

#### 1.1.1. 등고선 (선) 플롯 그리기 (`plt.contour()`)
`plt.contour()` 함수는 동일한 `Z` 값을 갖는 지점들을 연결하는 선(등고선)을 그립니다.

**코드 설명**:
1.  **데이터 생성**:
    *   `x = np.linspace(-3.0, 3.0, 100)`: -3.0부터 3.0까지 100개의 등간격 x 좌표를 생성합니다.
    *   `y = np.linspace(-3.0, 3.0, 100)`: -3.0부터 3.0까지 100개의 등간격 y 좌표를 생성합니다.
    *   `X, Y = np.meshgrid(x, y)`: `x`와 `y` 배열을 이용하여 2차원 그리드 배열 `X`와 `Y`를 생성합니다. `X`는 모든 행이 `x`와 동일하고, `Y`는 모든 열이 `y`와 동일한 형태를 가집니다.
    *   `Z = np.exp(-(X**2 + Y**2)) * np.sin(X) * np.cos(Y)`: 각 (X, Y) 좌표에서의 Z 값을 계산합니다. 이는 예시를 위한 3차원 함수입니다.

2.  **등고선 플롯 그리기**:
    *   `plt.figure(figsize=(12, 5))`: 두 개의 서브플롯을 나란히 배치하기 위해 전체 그림의 크기를 설정합니다.
    *   `plt.subplot(1, 2, 1)`: 1행 2열의 서브플롯 중 첫 번째(왼쪽) 서브플롯을 선택합니다.
    *   `contour = plt.contour(X, Y, Z, 10, colors='black')`: `X`, `Y`, `Z` 데이터를 사용하여 등고선 플롯을 그립니다. `10`은 등고선의 레벨(개수)을 지정하며, `colors='black'`은 모든 등고선을 검은색으로 그립니다.
    *   `plt.clabel(contour, inline=True, fontsize=8)`: 그려진 등고선 위에 해당 등고선의 `Z` 값을 텍스트로 표시합니다. `inline=True`는 선을 따라 텍스트가 배치되도록 합니다.
    *   `plt.title()`, `plt.xlabel()`, `plt.ylabel()`: 제목과 축 레이블을 설정합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)) * np.sin(X) * np.cos(Y)

# 등고선 (선) 플롯 그리기
plt.figure(figsize=(6, 5)) # 단일 플롯을 위해 크기 조정
contour = plt.contour(X, Y, Z, 10, colors='black') # 10개의 레벨로 등고선
plt.clabel(contour, inline=True, fontsize=8) # 등고선에 값 표시
plt.title('Contour Plot (Lines)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

#### 1.1.2. 채워진 등고선 플롯 그리기 (`plt.contourf()`)
`plt.contourf()` 함수는 등고선 사이의 영역을 색으로 채워 `Z` 값의 변화를 색상의 변화로 시각화합니다. `f`는 'filled'를 의미합니다.

**코드 설명**:
*   `plt.subplot(1, 2, 2)`: 1행 2열의 서브플롯 중 두 번째(오른쪽) 서브플롯을 선택합니다.
*   `contourf = plt.contourf(X, Y, Z, 20, cmap='viridis')`: `X`, `Y`, `Z` 데이터를 사용하여 채워진 등고선 플롯을 그립니다. `20`은 색상 레벨의 개수를 지정하며, `cmap='viridis'`는 'viridis' 컬러맵을 사용하여 색상을 매핑합니다.
*   `plt.colorbar(label='Value')`: 채워진 등고선 플롯의 색상이 어떤 `Z` 값을 나타내는지 설명하는 컬러바를 추가합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (이전과 동일)
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)) * np.sin(X) * np.cos(Y)

# 채워진 등고선 플롯 그리기
plt.figure(figsize=(7, 5)) # 단일 플롯을 위해 크기 조정
contourf = plt.contourf(X, Y, Z, 20, cmap='viridis') # 20개의 레벨, viridis 컬러맵
plt.colorbar(label='Value') # 컬러바 추가
plt.title('Filled Contour Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

#### 1.1.3. 등고선 플롯 커스터마이징 (레벨, 색상, 컬러맵, 레이블)
등고선 플롯은 `levels`, `colors`, `cmap`, `clabel` 등 다양한 파라미터를 통해 세밀하게 커스터마이징할 수 있습니다.

*   **`levels`**: 등고선 또는 색상 영역을 나눌 `Z` 값의 개수 또는 특정 `Z` 값의 리스트를 지정합니다.
*   **`colors`**: `plt.contour()`에서 등고선 선의 색상을 지정합니다. 단일 색상 또는 레벨별 색상 리스트를 사용할 수 있습니다.
*   **`cmap`**: `plt.contourf()`에서 색상 영역을 채울 때 사용할 컬러맵을 지정합니다.
*   **`plt.clabel()`**: 등고선 위에 `Z` 값을 표시하여 특정 등고선이 나타내는 값을 명확히 합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (이전과 동일)
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)) * np.sin(X) * np.cos(Y)

# 등고선 플롯 커스터마이징 예시
plt.figure(figsize=(12, 6))

# 커스터마이징된 등고선 (선)
plt.subplot(1, 2, 1)
levels = np.arange(-0.5, 0.5, 0.1) # 특정 Z 값 레벨 지정
contour = plt.contour(X, Y, Z, levels=levels, colors='blue', linestyles='--')
plt.clabel(contour, inline=True, fontsize=9, fmt='%1.2f') # 소수점 두 자리까지 표시
plt.title('Custom Contour Plot (Lines)')
plt.xlabel('X')
plt.ylabel('Y')

# 커스터마이징된 채워진 등고선
plt.subplot(1, 2, 2)
contourf = plt.contourf(X, Y, Z, levels=20, cmap='RdBu', alpha=0.8) # 레벨 개수, 다른 컬러맵, 투명도
plt.colorbar(contourf, label='Function Value')
plt.title('Custom Filled Contour Plot')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
```

### 1.2. 박스 플롯 (Box Plot)
박스 플롯(Box Plot) 또는 상자 수염 그림(Box-and-Whisker Plot)은 데이터의 분포를 사분위수(Quartiles)를 이용하여 시각화하는 데 매우 효과적인 도구입니다. 데이터의 중앙값(median), 25% 지점(Q1), 75% 지점(Q3), 그리고 이상치(outlier)를 한눈에 보여주어 여러 그룹 간의 데이터 분포를 비교하는 데 널리 사용됩니다.

**박스 플롯의 구성 요소**:
*   **상자(Box)**: 데이터의 25번째 백분위수(Q1)부터 75번째 백분위수(Q3)까지의 범위를 나타냅니다. 상자의 길이는 IQR(Interquartile Range = Q3 - Q1)을 의미하며, 데이터의 중간 50%가 분포하는 범위를 보여줍니다.
*   **중앙선(Median Line)**: 상자 내의 선은 데이터의 중앙값(50번째 백분위수)을 나타냅니다.
*   **수염(Whiskers)**: 상자 밖으로 뻗어 나가는 선으로, 일반적으로 Q1 - 1.5 * IQR과 Q3 + 1.5 * IQR 범위 내의 데이터 중 가장 극단적인 값을 나타냅니다.
*   **이상치(Outliers)**: 수염 바깥에 위치하는 개별 점들은 이상치로 간주됩니다.

#### 1.2.1. 기본 박스 플롯 그리기
`plt.boxplot(data)` 함수는 데이터 배열 또는 데이터 배열의 리스트를 받아 박스 플롯을 그립니다.

**코드 설명**:
1.  **라이브러리 임포트**:
    *   `import matplotlib.pyplot as plt`: Matplotlib의 `pyplot` 모듈을 임포트합니다.
    *   `import numpy as np`: `numpy` 라이브러리를 임포트하여 데이터 생성에 활용합니다.

2.  **데이터 생성**:
    *   `data1 = np.random.normal(0, 1, 100)`: 평균 0, 표준편차 1인 정규 분포에서 100개의 난수를 생성합니다.
    *   `data = [data1]`: 단일 그룹의 박스 플롯을 그리기 위해 리스트에 하나의 데이터셋을 담습니다.

3.  **기본 박스 플롯 그리기**:
    *   `plt.boxplot(data)`: `data` 리스트에 담긴 데이터셋에 대한 박스 플롯을 그립니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (단일 그룹의 데이터)
data1 = np.random.normal(0, 1, 100)
data = [data1]

# 기본 박스 플롯 그리기
plt.figure(figsize=(6, 5))
plt.boxplot(data)
plt.title('Basic Box Plot')
plt.xlabel('Group')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```

#### 1.2.2. 여러 그룹의 박스 플롯 비교
여러 그룹의 데이터 분포를 비교할 때는 `plt.boxplot()` 함수에 데이터셋 리스트를 전달하고, `labels` 파라미터를 사용하여 각 박스 플롯에 이름을 부여할 수 있습니다.

**코드 설명**:
*   `data = [data1, data2, data3]`: 세 개의 다른 데이터셋을 리스트로 묶어 `plt.boxplot()`에 전달합니다.
*   `labels=['Group 1', 'Group 2', 'Group 3']`: 각 박스 플롯에 해당하는 레이블을 지정하여 어떤 그룹의 데이터인지 명확히 합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (세 그룹의 데이터)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(1, 1.5, 100)
data3 = np.random.normal(-1, 0.5, 100)
data = [data1, data2, data3]

# 여러 그룹의 박스 플롯 그리기
plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'])
plt.title('Box Plot of Three Groups')
plt.xlabel('Group')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```

#### 1.2.3. 박스 플롯 커스터마이징 (색상, 이상치, 방향)
`plt.boxplot()` 함수는 박스 플롯의 시각적 요소를 세밀하게 제어할 수 있는 다양한 파라미터를 제공합니다.

**주요 커스터마이징 파라미터**:

*   **`vert`**: `True` (기본값)로 설정하면 수직 박스 플롯을, `False`로 설정하면 수평 박스 플롯을 그립니다.
*   **`patch_artist`**: `True`로 설정하면 상자를 색으로 채울 수 있습니다. 이 경우 `boxprops` 파라미터를 사용하여 상자의 속성을 조절합니다.
*   **`boxprops`**: 상자의 속성(색상, 테두리 등)을 딕셔너리 형태로 지정합니다.
*   **`medianprops`**: 중앙값 선의 속성을 지정합니다.
*   **`whiskerprops`**: 수염 선의 속성을 지정합니다.
*   **`capprops`**: 수염 끝의 캡(cap) 속성을 지정합니다.
*   **`flierprops`**: 이상치(outlier) 점의 속성을 지정합니다.
*   **`showfliers`**: `False`로 설정하면 이상치를 표시하지 않습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (세 그룹의 데이터)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(1, 1.5, 100)
data3 = np.random.normal(-1, 0.5, 100)
data = [data1, data2, data3]

# 박스 플롯 커스터마이징
plt.figure(figsize=(10, 7))
bp = plt.boxplot(data, labels=['Group A', 'Group B', 'Group C'],
                 patch_artist=True, # 상자를 색으로 채울 수 있도록 설정
                 vert=True,         # 수직 박스 플롯 (기본값)
                 showfliers=True)   # 이상치 표시 (기본값)

# 상자 색상 설정
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# 중앙값 선 스타일 설정
for median in bp['medians']:
    median.set(color='red', linewidth=2)

# 이상치 점 스타일 설정
for flier in bp['fliers']:
    flier.set(marker='o', color='gray', alpha=0.5)

plt.title('Customized Box Plot', fontsize=16)
plt.xlabel('Data Group', fontsize=12)
plt.ylabel('Value Distribution', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

### 1.3. 파이 차트 (Pie Chart)
파이 차트(Pie Chart)는 전체에 대한 각 부분의 비율을 부채꼴의 면적으로 나타내는 그래프입니다. 데이터의 구성 비율을 직관적으로 보여줄 때 유용하지만, 항목이 너무 많거나 비율 차이가 미미할 경우 해석이 어려울 수 있어 사용에 주의가 필요합니다. Matplotlib에서는 `plt.pie()` 함수를 사용하여 파이 차트를 그립니다.

#### 1.3.1. 기본 파이 차트 그리기
`plt.pie(x)` 함수는 각 부분의 크기를 나타내는 숫자 배열을 받아 파이 차트를 그립니다.

**코드 설명**:
1.  **라이브러리 임포트**:
    *   `import matplotlib.pyplot as plt`: Matplotlib의 `pyplot` 모듈을 임포트합니다.

2.  **데이터 생성**:
    *   `sizes = [15, 30, 45, 10]`: 각 항목의 비율을 나타내는 숫자 리스트입니다. 이 값들의 합계가 전체(100%)를 구성합니다.
    *   `labels = ['Category A', 'Category B', 'Category C', 'Category D']`: 각 파이 조각에 대한 레이블을 정의합니다.

3.  **기본 파이 차트 그리기**:
    *   `plt.figure(figsize=(7, 7))`: 그래프의 크기를 설정하여 원형이 잘 보이도록 합니다.
    *   `plt.pie(sizes, labels=labels)`: `sizes`를 사용하여 파이 차트를 그리고, `labels`를 각 조각에 연결합니다.
    *   `plt.title('Basic Pie Chart')`: 그래프의 제목을 설정합니다.
    *   `plt.axis('equal')`: 파이 차트가 타원이 아닌 완벽한 원형으로 보이도록 종횡비를 동일하게 설정합니다.

```python
import matplotlib.pyplot as plt

# 데이터 생성
labels = ['Category A', 'Category B', 'Category C', 'Category D']
sizes = [15, 30, 45, 10] # 각 항목의 비율

# 기본 파이 차트 그리기
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels)
plt.title('Basic Pie Chart')
plt.axis('equal')  # 파이 차트를 원형으로 유지
plt.show()
```

#### 1.3.2. 파이 차트 커스터마이징 (비율 표시, 조각 분리, 그림자)
`plt.pie()` 함수는 파이 차트의 시각적 요소를 다양하게 커스터마이징할 수 있는 파라미터를 제공합니다.

**주요 커스터마이징 파라미터**:

*   **`autopct`**: 각 조각의 비율을 자동으로 계산하여 텍스트로 표시합니다. `%1.1f%%`와 같은 형식 문자열을 사용하여 소수점 자릿수를 조절할 수 있습니다.
*   **`explode`**: 각 조각을 원의 중심에서 얼마나 떨어뜨릴지 지정하는 배열입니다. 특정 조각을 강조할 때 유용합니다.
*   **`shadow`**: `True`로 설정하면 파이 차트에 그림자 효과를 추가합니다.
*   **`startangle`**: 첫 번째 조각이 시작하는 각도를 지정합니다 (기본값은 0도, x축 양의 방향).
*   **`colors`**: 각 조각의 색상을 지정하는 색상 리스트입니다.

```python
import matplotlib.pyplot as plt

# 데이터 생성
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10] # 각 항목의 비율
explode = (0, 0.1, 0, 0)  # 'B' 항목을 약간 돋보이게 함 (두 번째 조각)
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue'] # 각 조각의 색상

# 파이 차트 그리기 및 커스터마이징
plt.figure(figsize=(8, 8))
plt.pie(sizes,
        explode=explode,    # 조각 분리
        labels=labels,      # 레이블
        colors=colors,      # 색상
        autopct='%1.1f%%',  # 비율 표시 (소수점 첫째 자리까지)
        shadow=True,        # 그림자 효과
        startangle=140)     # 시작 각도 설정

plt.title('Customized Pie Chart of Proportions', fontsize=16)
plt.axis('equal')  # 파이 차트를 원형으로 유지
plt.show()
```
