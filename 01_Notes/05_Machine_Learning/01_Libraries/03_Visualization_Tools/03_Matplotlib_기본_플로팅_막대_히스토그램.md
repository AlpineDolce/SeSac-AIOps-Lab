<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 `pyplot` 모듈을 사용하여 막대 그래프(Bar Plot)와 히스토그램(Histogram)을 그리는 방법을 다룹니다. 각 플롯의 용도와 함께 데이터 생성부터 그래프 출력까지의 과정을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. Matplotlib 기본 플로팅: 막대 그래프와 히스토그램](#1-matplotlib-기본-플로팅-막대-그래프와-히스토그램)
  - [1.1. 막대 그래프 (Bar Plot)](#11-막대-그래프-bar-plot)
    - [1.1.1. 기본 막대 그래프 그리기](#111-기본-막대-그래프-그리기)
    - [1.1.2. 막대 그래프 커스터마이징 (색상, 제목, 축 레이블)](#112-막대-그래프-커스터마이징-색상-제목-축-레이블)
  - [1.2. 히스토그램 (Histogram)](#12-히스토그램-histogram)
    - [1.2.1. 기본 히스토그램 그리기](#121-기본-히스토그램-그리기)
    - [1.2.2. 밀도 히스토그램 그리기](#122-밀도-히스토그램-그리기)
    - [1.2.3. 히스토그램 커스터마이징 (bins, 색상, 투명도, 제목, 축 레이블)](#123-히스토그램-커스터마이징-bins-색상-투명도-제목-축-레이블)

---

## 1. Matplotlib 기본 플로팅: 막대 그래프와 히스토그램

### 1.1. 막대 그래프 (Bar Plot)
막대 그래프(Bar Plot)는 범주형 데이터의 빈도, 합계, 평균 등 특정 값을 막대의 길이로 표현하여 여러 범주 간의 비교를 시각적으로 용이하게 하는 데 사용됩니다. Matplotlib에서는 `plt.bar()` 함수를 사용하여 막대 그래프를 그립니다.

#### 1.1.1. 기본 막대 그래프 그리기
`plt.bar(x, height)` 함수는 x축에 해당하는 범주와 y축에 해당하는 값을 받아 막대 그래프를 그립니다.

**코드 설명**:
1.  **라이브러리 임포트**:
    *   `import matplotlib.pyplot as plt`: Matplotlib의 `pyplot` 모듈을 임포트합니다.

2.  **데이터 생성**:
    *   `categories = ['A', 'B', 'C', 'D']`: x축에 표시될 범주형 데이터(예: 제품 종류, 지역 등)를 리스트로 정의합니다.
    *   `values = [20, 35, 30, 25]`: 각 범주에 해당하는 수치형 데이터(예: 판매량, 빈도 등)를 리스트로 정의합니다. `categories` 리스트와 길이가 같아야 합니다.

3.  **기본 막대 그래프 그리기**:
    *   `plt.bar(categories, values)`: `categories`를 x축으로, `values`를 막대의 높이로 사용하여 막대 그래프를 그립니다.

4.  **그래프 표시**:
    *   `plt.show()`: 설정된 막대 그래프를 화면에 출력합니다.

```python
import matplotlib.pyplot as plt

# 데이터 생성
categories = ['A', 'B', 'C', 'D']
values = [20, 35, 30, 25]

# 기본 막대 그래프 그리기
plt.bar(categories, values)
plt.title("Basic Bar Plot")
plt.xlabel("Category")
plt.ylabel("Value")
plt.show()
```

#### 1.1.2. 막대 그래프 커스터마이징 (색상, 제목, 축 레이블)
`plt.bar()` 함수는 막대의 색상, 너비 등을 조절할 수 있는 다양한 파라미터를 제공하며, `pyplot`의 일반적인 커스터마이징 함수들을 사용하여 제목과 축 레이블을 설정할 수 있습니다.

**주요 커스터마이징 파라미터**:

*   **`color`**: 막대의 색상을 지정합니다. 단일 색상 문자열(예: 'skyblue')을 사용할 수도 있고, 각 막대에 다른 색상을 적용하려면 색상 문자열 리스트를 전달할 수 있습니다.
*   **`width`**: 막대의 너비를 0에서 1 사이의 값으로 지정합니다 (기본값은 0.8).

**코드 설명**:
*   `color=['skyblue', 'lightcoral', 'lightgreen', 'gold']`: 각 막대에 다른 색상을 적용하기 위해 색상 리스트를 `color` 파라미터로 전달합니다.
*   `plt.title()`, `plt.xlabel()`, `plt.ylabel()`: 그래프의 제목과 축 레이블을 설정하여 그래프의 의미를 명확히 전달합니다.

```python
import matplotlib.pyplot as plt

# 데이터 생성
categories = ['A', 'B', 'C', 'D']
values = [20, 35, 30, 25]

# 막대 그래프 그리기 및 커스터마이징
plt.bar(categories, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'], width=0.7)
plt.title("Customized Bar Plot of Categories", fontsize=15, color='darkblue')
plt.xlabel("Category", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7) # y축 그리드 추가
plt.show()
```

### 1.2. 히스토그램 (Histogram)
히스토그램은 단일 연속형 변수의 데이터 분포를 시각적으로 파악하는 데 사용되는 플롯입니다. 데이터를 여러 구간(bin)으로 나누고, 각 구간에 속하는 데이터 포인트의 개수(빈도)를 막대의 높이로 표현합니다. Matplotlib에서는 `plt.hist()` 함수를 사용하여 히스토그램을 그립니다.

#### 1.2.1. 기본 히스토그램 그리기
`plt.hist(data, bins=None)` 함수는 데이터 배열을 받아 히스토그램을 그립니다. `bins` 파라미터를 지정하지 않으면 Matplotlib이 자동으로 적절한 bin의 개수를 결정합니다.

**코드 설명**:
1.  **라이브러리 임포트**:
    *   `import matplotlib.pyplot as plt`: Matplotlib의 `pyplot` 모듈을 임포트합니다.
    *   `import numpy as np`: `numpy` 라이브러리를 임포트하여 데이터 생성에 활용합니다.

2.  **데이터 생성**:
    *   `data = np.random.randn(1000)`: 표준 정규 분포(평균 0, 표준편차 1)를 따르는 1000개의 난수를 생성합니다. 이 데이터는 연속형 변수의 분포를 시각화하기에 적합합니다.

3.  **기본 히스토그램 그리기**:
    *   `plt.hist(data)`: `data` 배열을 사용하여 히스토그램을 그립니다. 기본적으로 Matplotlib이 bin의 개수를 자동으로 설정하고, 각 bin에 속하는 데이터의 빈도를 y축에 표시합니다.

4.  **그래프 표시**:
    *   `plt.show()`: 설정된 히스토그램을 화면에 출력합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (정규 분포를 따르는 1000개의 난수)
data = np.random.randn(1000)

# 기본 히스토그램 그리기
plt.hist(data)
plt.title("Basic Histogram of Random Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

#### 1.2.2. 밀도 히스토그램 그리기
히스토그램에서 `density=True` 파라미터를 설정하면 막대의 높이가 빈도 대신 확률 밀도(probability density)를 나타냅니다. 이 경우 모든 막대의 면적을 합하면 1이 됩니다. 이는 서로 다른 크기의 데이터셋을 비교할 때 유용합니다.

**코드 설명**:
*   `plt.hist(data, bins=30, density=True)`: `density=True`를 추가하여 y축이 빈도 대신 밀도를 나타내도록 합니다. `bins=30`은 데이터를 30개의 구간으로 나눕니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (정규 분포를 따르는 1000개의 난수)
data = np.random.randn(1000)

# 밀도 히스토그램 그리기
plt.hist(data, bins=30, color='green', alpha=0.7, density=True)
plt.title("Density Histogram of Random Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```

#### 1.2.3. 히스토그램 커스터마이징 (bins, 색상, 투명도, 제목, 축 레이블)
`plt.hist()` 함수는 히스토그램의 모양과 스타일을 조절할 수 있는 다양한 파라미터를 제공합니다.

**주요 커스터마이징 파라미터**:

*   **`bins`**: 데이터를 나눌 구간(bin)의 개수 또는 구간 경계를 지정합니다. `bins`의 개수를 조절하여 데이터 분포의 세부적인 모습을 다르게 볼 수 있습니다.
*   **`color`**: 막대의 색상을 지정합니다.
*   **`alpha`**: 막대의 투명도를 0.0(완전 투명)에서 1.0(완전 불투명) 사이의 값으로 지정합니다. 여러 히스토그램을 겹쳐 그릴 때 유용합니다.
*   **`edgecolor`**: 막대 테두리의 색상을 지정합니다.
*   **`histtype`**: 히스토그램의 종류를 지정합니다 (예: 'bar' (기본), 'barstacked', 'step', 'stepfilled').

**코드 설명**:
*   `bins=30`: 데이터를 30개의 구간으로 나눕니다.
*   `color='purple'`: 막대의 색상을 보라색으로 설정합니다.
*   `alpha=0.7`: 막대의 투명도를 0.7로 설정합니다.
*   `plt.title()`, `plt.xlabel()`, `plt.ylabel()`: 그래프의 제목과 축 레이블을 설정합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (정규 분포를 따르는 1000개의 난수)
data = np.random.randn(1000)

# 히스토그램 커스터마이징
plt.hist(data, bins=40, color='skyblue', alpha=0.8, edgecolor='black', histtype='bar')
plt.title("Customized Histogram of Random Data", fontsize=16)
plt.xlabel("Value Range", fontsize=12)
plt.ylabel("Frequency Count", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7) # y축 그리드 추가
plt.show()
```
