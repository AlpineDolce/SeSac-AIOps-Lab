## 데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-02

## 문서 목표
이 문서는 파이썬의 주요 데이터 시각화 라이브러리인 Matplotlib, Seaborn, Plotly의 핵심 개념과 활용법을 상세히 다룹니다. 각 라이브러리의 특징, 설치 방법, 기본적인 플로팅부터 고급 시각화 기법까지 다양한 예제를 통해 설명하며, 머신러닝 및 딥러닝 프로젝트에서 데이터를 효과적으로 탐색하고 결과를 시각화하는 데 필요한 지식을 제공합니다.

## 목차

- [1. 데이터 시각화 소개](#1-데이터-시각화-소개)
  - [1.1. 데이터 시각화의 중요성](#11-데이터-시각화의-중요성)
  - [1.2. 파이썬 시각화 생태계](#12-파이썬-시각화-생태계)

- [2. Matplotlib](#2-matplotlib)
  - [2.1. Matplotlib 소개](#21-matplotlib-소개)
  - [2.2. 설치](#22-설치)
  - [2.3. 기본 플로팅](#23-기본-플로팅)
    - [2.3.1. 라인 플롯 (Line Plot)](#231-라인-플롯-line-plot)
    - [2.3.2. 산점도 (Scatter Plot)](#232-산점도-scatter-plot)
    - [2.3.3. 막대 그래프 (Bar Plot)](#233-막대-그래프-bar-plot)
    - [2.3.4. 히스토그램 (Histogram)](#234-히스토그램-histogram)
  - [2.4. 플롯 사용자 정의](#24-플롯-사용자-정의)
    - [2.4.1. 제목, 축 레이블, 범례](#241-제목-축-레이블-범례)
    - [2.4.2. 색상, 마커, 선 스타일](#242-색상-마커-선-스타일)
    - [2.4.3. 축 범위 및 눈금 설정](#243-축-범위-및-눈금-설정)
  - [2.5. 서브플롯 (Subplots)](#25-서브플롯-subplots)
  - [2.6. 플롯 저장](#26-플롯-저장)

- [3. Seaborn](#3-seaborn)
  - [3.1. Seaborn 소개](#31-seaborn-소개)
  - [3.2. 설치](#32-설치)
  - [3.3. 관계형 플롯 (Relational Plots)](#33-관계형-플롯-relational-plots)
    - [3.3.1. `scatterplot()`](#331-scatterplot)
    - [3.3.2. `lineplot()`](#332-lineplot)
  - [3.4. 분포 플롯 (Distribution Plots)](#34-분포-플롯-distribution-plots)
    - [3.4.1. `histplot()`](#341-histplot)
    - [3.4.2. `kdeplot()`](#342-kdeplot)
    - [3.4.3. `displot()`](#343-displot)
  - [3.5. 범주형 플롯 (Categorical Plots)](#35-범주형-플롯-categorical-plots)
    - [3.5.1. `boxplot()`](#351-boxplot)
    - [3.5.2. `violinplot()`](#352-violinplot)
    - [3.5.3. `stripplot()`](#353-stripplot)
    - [3.5.4. `swarmplot()`](#354-swarmplot)
    - [3.5.5. `barplot()`](#355-barplot)
    - [3.5.6. `countplot()`](#356-countplot)
  - [3.6. 회귀 플롯 (Regression Plots)](#36-회귀-플롯-regression-plots)
    - [3.6.1. `regplot()`](#361-regplot)
    - [3.6.2. `lmplot()`](#362-lmplot)
  - [3.7. 행렬 플롯 (Matrix Plots)](#37-행렬-플롯-matrix-plots)
    - [3.7.1. `heatmap()`](#371-heatmap)
    - [3.7.2. `clustermap()`](#372-clustermap)
  - [3.8. 사용자 정의 및 테마](#38-사용자-정의-및-테마)

- [4. Plotly](#4-plotly)
  - [4.1. Plotly 소개](#41-plotly-소개)
  - [4.2. 설치](#42-설치)
  - [4.3. 기본 플로팅 (Plotly Express)](#43-기본-플로팅-plotly-express)
    - [4.3.1. 산점도 (`px.scatter()`)](#431-산점도-pxscatter)
    - [4.3.2. 라인 플롯 (`px.line()`)](#432-라인-플롯-pxline)
    - [4.3.3. 막대 그래프 (`px.bar()`)](#433-막대-그래프-pxbar)
    - [4.3.4. 파이 차트 (`px.pie()`)](#434-파이-차트-pxpie)
  - [4.4. 인터랙티브 플롯](#44-인터랙티브-플롯)
  - [4.5. 서브플롯](#45-서브플롯)
  - [4.6. 플롯 저장](#46-플롯-저장)
  - [4.7. Dash와의 통합](#47-dash와의-통합)

- [5. 라이브러리 비교 및 사용 가이드](#5-라이브러리-비교-및-사용-가이드)
  - [5.1. Matplotlib vs. Seaborn vs. Plotly](#51-matplotlib-vs-seaborn-vs-plotly)
  - [5.2. 최적의 라이브러리 선택 가이드](#52-최적의-라이브러리-선택-가이드)
  - [5.3. 데이터 시각화 모범 사례](#53-데이터-시각화-모범-사례)


---

## 1. 데이터 시각화 소개

### 1.1. 데이터 시각화의 중요성
데이터 시각화는 복잡한 데이터를 이해하기 쉽고 직관적인 형태로 표현하는 과정입니다. 이는 데이터 분석의 핵심 단계 중 하나로, 올바른 차트 유형을 선택하여 데이터의 특징과 메시지를 효과적으로 전달하는 데 중요한 역할을 수행합니다.

1.  **패턴 및 추세 발견**: 방대한 양의 숫자 데이터 속에서 숨겨진 패턴, 추세, 이상치 등을 시각적으로 빠르게 식별할 수 있습니다.
2.  **인사이트 도출**: 데이터를 시각화함으로써 새로운 가설을 세우거나, 문제의 원인을 파악하고, 비즈니스 의사결정에 필요한 핵심 인사이트를 도출할 수 있습니다.
3.  **효과적인 커뮤니케이션**: 분석 결과를 비전문가에게도 명확하고 설득력 있게 전달하는 데 가장 효과적인 방법입니다. 복잡한 통계 수치나 모델 성능 지표를 그래프 하나로 쉽게 설명할 수 있습니다.
4.  **탐색적 데이터 분석 (EDA)**: 데이터의 분포, 변수 간 관계, 결측치 여부 등을 시각적으로 확인하여 데이터 전처리 및 모델링 방향을 설정하는 데 도움을 줍니다.
5.  **모델 성능 평가**: 머신러닝 모델의 예측 결과와 실제 값의 비교, 분류 모델의 혼동 행렬(Confusion Matrix) 시각화 등을 통해 모델의 성능을 직관적으로 평가할 수 있습니다.

### 1.2. 파이썬 시각화 생태계
파이썬은 데이터 과학 분야에서 가장 인기 있는 언어 중 하나이며, 강력하고 다양한 시각화 라이브러리를 제공합니다. 이들 라이브러리는 각각의 장단점과 특성을 가지고 있어, 분석 목적과 데이터의 종류에 따라 적절한 도구를 선택하는 것이 중요합니다.

1.  **Matplotlib**: 파이썬 시각화의 **기반이 되는 라이브러리**입니다. 저수준(low-level) API를 제공하여 플롯의 모든 요소를 세밀하게 제어할 수 있는 강력한 기능을 가집니다. 복잡하고 커스터마이징된 그래프를 만들 때 유용하며, 다른 많은 시각화 라이브러리들이 Matplotlib을 기반으로 합니다.
2.  **Seaborn**: Matplotlib을 기반으로 구축된 고수준(high-level) 시각화 라이브러리입니다. 통계 그래프를 쉽게 그릴 수 있도록 설계되었으며, Matplotlib보다 적은 코드로 미려하고 정보가 풍부한 그래프를 생성할 수 있습니다. 특히 데이터 분포, 변수 간 관계, 범주형 데이터 분석에 특화되어 있습니다.
3.  **Plotly**: 웹 기반의 인터랙티브(interactive) 시각화를 제공하는 라이브러리입니다. 정적 이미지뿐만 아니라 확대/축소, 데이터 포인트 정보 확인 등 사용자와 상호작용할 수 있는 동적인 그래프를 만들 수 있습니다. 대시보드 구축이나 웹 애플리케이션에 시각화를 포함할 때 매우 유용합니다.

이 외에도 `Bokeh`, `Altair`, `Folium` (지리 정보 시각화) 등 다양한 시각화 라이브러리들이 존재하며, 각자의 특성에 따라 특정 목적에 더 적합하게 사용될 수 있습니다.


## 2. Matplotlib

### 2.1. Matplotlib 소개
Matplotlib은 파이썬에서 정적, 애니메이션, 인터랙티브 시각화를 생성하기 위한 포괄적인 라이브러리입니다. 100년이 넘는 시각화 역사를 기반으로 하며, 파이썬 스크립트, Jupyter Notebook, 웹 애플리케이션 서버 등 다양한 환경에서 사용할 수 있습니다. Matplotlib은 플롯의 모든 요소를 세밀하게 제어할 수 있는 '저수준(low-level)' API를 제공하여, 사용자가 원하는 대로 그래프를 커스터마이징할 수 있는 강력한 유연성을 제공합니다. 특히, Matplotlib의 핵심 구성 요소는 `Figure` (전체 그림)와 `Axes` (개별 플롯) 객체이며, 이를 통해 객체 지향적인 방식으로 플롯을 생성하고 조작할 수 있습니다. 다른 많은 파이썬 시각화 라이브러리(예: Seaborn)가 Matplotlib을 기반으로 구축되어 있습니다.

### 2.2. 설치
Matplotlib은 `pip`를 사용하여 쉽게 설치할 수 있습니다. Jupyter Notebook이나 Anaconda 환경에서는 이미 설치되어 있을 가능성이 높습니다.

```bash
pip install matplotlib
```

설치 후에는 일반적으로 `pyplot` 모듈을 `plt`라는 별칭으로 임포트하여 사용합니다.

```python
import matplotlib.pyplot as plt
```

### 2.3. 기본 플로팅
Matplotlib의 `pyplot` 모듈은 다양한 종류의 그래프를 그릴 수 있는 함수를 제공합니다.

#### 2.3.1. 라인 플롯 (Line Plot)
가장 기본적인 플롯으로, 데이터 포인트들을 선으로 연결하여 시계열 데이터나 연속적인 데이터의 변화 추이를 보여줄 때 사용합니다. `plt.plot()` 함수를 사용합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100) # 0부터 10까지 100개의 등간격 숫자
y = np.sin(x) # x에 대한 사인 값
y2 = np.cos(x) # x에 대한 코사인 값

# 단일 라인 플롯 그리기
plt.plot(x, y)
plt.title("Simple Line Plot") # 그래프 제목
plt.xlabel("X-axis") # x축 레이블
plt.ylabel("Y-axis") # y축 레이블
plt.grid(True) # 그리드 표시
plt.show() # 그래프 보여주기

# 여러 라인 플롯 그리기
plt.plot(x, y, label='Sine')
plt.plot(x, y2, label='Cosine')
plt.title("Multiple Line Plots")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend() # 범례 표시
plt.grid(True)
plt.show()
```

#### 2.3.2. 산점도 (Scatter Plot)
두 변수 간의 관계를 점으로 표현하는 플롯입니다. 데이터 포인트들의 분포나 군집을 파악하는 데 유용합니다. `plt.scatter()` 함수를 사용합니다. `c` 파라미터는 점의 색상을, `s` 파라미터는 점의 크기를 지정합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.random.rand(50) * 10 # 0-10 사이의 랜덤 x 값 50개
y = np.random.rand(50) * 10 # 0-10 사이의 랜덤 y 값 50개
colors = np.random.rand(50) # 각 점의 색상
size = np.random.rand(50) * 100 # 각 점의 크기

# 산점도 그리기
plt.scatter(x, y, c=colors, s=size, alpha=0.7) # c: 색상, s: 크기, alpha: 투명도
plt.title("Simple Scatter Plot")
plt.xlabel("X-value")
plt.ylabel("Y-value")
plt.colorbar(label="Color Intensity") # 색상 바 추가
plt.show()
```

#### 2.3.3. 막대 그래프 (Bar Plot)
범주형 데이터의 빈도나 값을 막대의 길이로 표현하는 플롯입니다. 여러 범주 간의 비교에 적합합니다. `plt.bar()` 함수를 사용합니다.

```python
import matplotlib.pyplot as plt

# 데이터 생성
categories = ['A', 'B', 'C', 'D']
values = [20, 35, 30, 25]

# 막대 그래프 그리기
plt.bar(categories, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.title("Bar Plot of Categories")
plt.xlabel("Category")
plt.ylabel("Value")
plt.show()
```

#### 2.3.4. 히스토그램 (Histogram)
단일 변수의 데이터 분포를 보여주는 플롯입니다. 데이터를 여러 구간(bin)으로 나누고 각 구간에 속하는 데이터의 개수를 막대로 표현합니다. `plt.hist()` 함수를 사용합니다. `density=True`로 설정하면 막대의 높이가 빈도 대신 확률 밀도를 나타내어 전체 면적이 1이 됩니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (정규 분포를 따르는 1000개의 난수)
data = np.random.randn(1000)

# 히스토그램 그리기
plt.hist(data, bins=30, color='purple', alpha=0.7)
plt.title("Histogram of Random Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# 밀도 히스토그램 그리기
plt.hist(data, bins=30, color='green', alpha=0.7, density=True)
plt.title("Density Histogram of Random Data")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
```

### 2.4. 플롯 사용자 정의
Matplotlib은 플롯의 다양한 요소를 커스터마이징할 수 있는 기능을 제공하여 시각화의 가독성과 미적 품질을 향상시킬 수 있습니다.

#### 2.4.1. 제목, 축 레이블, 범례
플롯의 의미를 명확히 전달하기 위해 제목, 축 레이블, 그리고 여러 데이터 시리즈를 구분하는 범례를 추가할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='Sine Wave', color='blue', linestyle='-')
plt.plot(x, y2, label='Cosine Wave', color='red', linestyle='--')

plt.title("Sine and Cosine Waves") # 그래프 제목
plt.xlabel("Time (s)") # x축 레이블
plt.ylabel("Amplitude") # y축 레이블
plt.legend() # 범례 표시
plt.grid(True) # 그리드 표시
plt.show()
```

#### 2.4.2. 색상, 마커, 선 스타일
데이터 시리즈를 시각적으로 구분하고 강조하기 위해 색상, 마커(데이터 포인트 모양), 선 스타일(실선, 점선 등)을 변경할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10)
y1 = x
y2 = x**2
y3 = x**3

plt.plot(x, y1, color='green', linestyle='-', marker='o', label='Linear')
plt.plot(x, y2, color='red', linestyle='--', marker='s', label='Quadratic')
plt.plot(x, y3, color='blue', linestyle=':', marker='^', label='Cubic')

plt.title("Different Plot Styles")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
```

#### 2.4.3. 축 범위 및 눈금 설정
데이터의 특정 부분을 강조하거나, 불필요한 여백을 제거하여 플롯의 가독성을 높이기 위해 축의 범위(`xlim()`, `ylim()`)와 눈금(`xticks()`, `yticks()`)을 수동으로 설정할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave with Custom Axis Limits and Ticks")
plt.xlabel("Angle (radians)")
plt.ylabel("Sine Value")

# x축 범위 설정
plt.xlim(0, 2 * np.pi)
# y축 범위 설정
plt.ylim(-1.1, 1.1)

# x축 눈금 설정 (0, pi/2, pi, 3pi/2, 2pi)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], 
           ['0', '$\pi/2$', '$\pi$', '$\frac{3\pi}{2}$', '2$\pi$'])

# y축 눈금 설정
plt.yticks([-1, 0, 1])

plt.grid(True)
plt.show()
```

### 2.5. 서브플롯 (Subplots)
여러 개의 플롯을 하나의 그림(Figure) 안에 배치하여 데이터를 다양한 관점에서 비교하거나 관련성 있는 정보를 함께 보여줄 때 사용합니다. `plt.subplot()` 또는 객체 지향 방식의 `plt.subplots()`를 사용합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = x
y4 = x**2

# 2x2 그리드에 4개의 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # 2행 2열, 그림 크기 설정

# 첫 번째 서브플롯 (좌상단)
axes[0, 0].plot(x, y1, color='blue')
axes[0, 0].set_title("Sine Wave")
axes[0, 0].set_xlabel("X")
axes[0, 0].set_ylabel("Y")

# 두 번째 서브플롯 (우상단)
axes[0, 1].plot(x, y2, color='red')
axes[0, 1].set_title("Cosine Wave")
axes[0, 1].set_xlabel("X")
axes[0, 1].set_ylabel("Y")

# 세 번째 서브플롯 (좌하단)
axes[1, 0].plot(x, y3, color='green')
axes[1, 0].set_title("Linear")
axes[1, 0].set_xlabel("X")
axes[1, 0].set_ylabel("Y")

# 네 번째 서브플롯 (우하단)
axes[1, 1].plot(x, y4, color='purple')
axes[1, 1].set_title("Quadratic")
axes[1, 1].set_xlabel("X")
axes[1, 1].set_ylabel("Y")

plt.tight_layout() # 서브플롯 간의 간격 자동 조절
plt.show()
```

### 2.6. 플롯 저장
생성된 플롯을 이미지 파일로 저장하여 문서나 프레젠테이션에 활용할 수 있습니다. `plt.savefig()` 함수를 사용합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Plot to Save")
plt.xlabel("X")
plt.ylabel("Y")

# PNG 파일로 저장
plt.savefig("sine_wave.png")
print("'sine_wave.png' 파일이 저장되었습니다.")

# PDF 파일로 저장 (고품질 벡터 이미지)
plt.savefig("sine_wave.pdf")
print("'sine_wave.pdf' 파일이 저장되었습니다.")

# 해상도(dpi) 설정하여 저장
plt.savefig("sine_wave_high_res.png", dpi=300)
print("'sine_wave_high_res.png' 파일이 고해상도로 저장되었습니다.")

plt.show()
```


## 3. Seaborn

### 3.1. Seaborn 소개
Seaborn은 Matplotlib을 기반으로 하는 파이썬 데이터 시각화 라이브러리입니다. 통계 그래프를 그리는 데 특화되어 있으며, Matplotlib보다 적은 코드로도 미려하고 정보가 풍부한 그래프를 쉽게 생성할 수 있도록 고수준(high-level) API를 제공합니다. Seaborn은 데이터프레임과 같은 Pandas 데이터 구조와 잘 통합되어 있어, 복잡한 데이터셋의 관계와 분포를 탐색하는 데 매우 유용합니다.

**주요 특징**:
*   **통계적 시각화**: 데이터셋의 통계적 관계를 시각화하는 데 중점을 둡니다.
*   **아름다운 기본 스타일**: Matplotlib보다 더 세련되고 전문적인 기본 플롯 스타일을 제공합니다.
*   **Pandas DataFrame 통합**: Pandas DataFrame을 직접 입력으로 받아 처리하기 용이합니다.
*   **복잡한 플롯 유형**: 분포, 관계, 범주형 데이터, 회귀 분석 등을 위한 다양한 고급 플롯 유형을 제공합니다.

### 3.2. 설치
Seaborn은 `pip`를 사용하여 설치할 수 있습니다. Matplotlib이 필요하므로, Seaborn을 설치하면 Matplotlib도 함께 설치되거나 이미 설치되어 있어야 합니다.

```bash
pip install seaborn
```

설치 후에는 일반적으로 `sns`라는 별칭으로 임포트하여 사용합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt # Seaborn은 Matplotlib 기반이므로 함께 사용
```

Seaborn은 자체적으로 몇 가지 내장 데이터셋을 제공하여 예제를 쉽게 실행해 볼 수 있습니다. 예를 들어, `sns.load_dataset('tips')`를 사용하여 팁 데이터셋을 로드할 수 있습니다.

### 3.3. 관계형 플롯 (Relational Plots)
두 개 이상의 변수 간의 통계적 관계를 시각화하는 데 사용됩니다.

#### 3.3.1. `scatterplot()`
산점도는 두 연속형 변수 간의 관계를 점으로 표현합니다. `hue`, `size`, `style` 등의 파라미터를 사용하여 추가적인 변수를 시각화에 반영할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips") # 팁 데이터셋 로드

# 총 계산액(total_bill)과 팁(tip)의 관계를 산점도로 표현
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.title("Total Bill vs. Tip")
plt.show()

# 성별(sex)에 따라 색상을 다르게 표현
sns.scatterplot(x="total_bill", y="tip", hue="sex", data=tips)
plt.title("Total Bill vs. Tip by Sex")
plt.show()

# 요일(day)에 따라 점의 크기(size)를, 시간(time)에 따라 점의 스타일(style)을 다르게 표현
sns.scatterplot(x="total_bill", y="tip", hue="day", size="size", style="time", data=tips)
plt.title("Total Bill vs. Tip by Day, Size, and Time")
plt.show()
```

#### 3.3.2. `lineplot()`
라인 플롯은 주로 시계열 데이터나 연속적인 데이터의 추세를 보여줄 때 사용됩니다. 여러 관측치에 대한 신뢰 구간을 함께 표시할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

fmri = sns.load_dataset("fmri") # fmri 데이터셋 로드

# 시간(timepoint)에 따른 신호(signal) 변화를 라인 플롯으로 표현
sns.lineplot(x="timepoint", y="signal", data=fmri)
plt.title("Signal Change Over Time")
plt.show()

# 이벤트(event)와 지역(region)에 따라 라인 분리
sns.lineplot(x="timepoint", y="signal", hue="event", style="region", data=fmri)
plt.title("Signal Change by Event and Region")
plt.show()
```

### 3.4. 분포 플롯 (Distribution Plots)
단일 변수 또는 여러 변수의 분포를 시각화하는 데 사용됩니다.

#### 3.4.1. `histplot()`
히스토그램은 단일 변수의 분포를 막대로 표현합니다. `bins` 파라미터로 구간의 개수를 조절할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.histplot(x="total_bill", data=tips, bins=15, kde=True) # kde=True로 커널 밀도 추정 곡선 추가
plt.title("Distribution of Total Bill")
plt.show()
```

#### 3.4.2. `kdeplot()`
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

#### 3.4.3. `displot()`
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

### 3.5. 범주형 플롯 (Categorical Plots)
범주형 변수와 하나 이상의 연속형 변수 간의 관계를 시각화하는 데 사용됩니다.

#### 3.5.1. `boxplot()`
상자 그림은 범주별로 데이터의 분포(중앙값, 사분위수, 이상치)를 보여줍니다. 데이터의 중심 경향성과 퍼짐 정도, 이상치를 파악하는 데 유용합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.boxplot(x="day", y="total_bill", data=tips)
plt.title("Total Bill by Day (Boxplot)")
plt.show()
```

#### 3.5.2. `violinplot()`
바이올린 플롯은 상자 그림과 커널 밀도 추정(KDE)을 결합하여 데이터의 분포를 더 상세하게 보여줍니다. 데이터의 밀도와 분포 형태를 동시에 파악할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.violinplot(x="day", y="total_bill", data=tips)
plt.title("Total Bill by Day (Violinplot)")
plt.show()
```

#### 3.5.3. `stripplot()`
스트립 플롯은 범주형 변수에 대한 개별 데이터 포인트들을 점으로 표시합니다. 데이터의 실제 분포를 보여주며, 데이터가 겹치는 것을 방지하기 위해 'jitter'를 추가할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
plt.title("Total Bill by Day (Stripplot)")
plt.show()
```

#### 3.5.4. `swarmplot()`
스웜 플롯은 스트립 플롯과 유사하지만, 데이터 포인트들이 겹치지 않도록 자동으로 조정하여 각 데이터 포인트의 밀도를 더 잘 보여줍니다. 데이터의 분포와 개별 관측치를 동시에 파악하는 데 유용합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.swarmplot(x="day", y="total_bill", data=tips)
plt.title("Total Bill by Day (Swarmplot)")
plt.show()
```

#### 3.5.5. `barplot()`
막대 그래프는 범주형 변수별로 연속형 변수의 평균(기본값)이나 다른 집계 값을 막대로 표현합니다. 신뢰 구간을 함께 표시할 수 있습니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.barplot(x="day", y="total_bill", data=tips, ci="sd") # ci="sd"로 표준편차 표시
plt.title("Average Total Bill by Day (Barplot)")
plt.show()
```

#### 3.5.6. `countplot()`
카운트 플롯은 범주형 변수의 각 범주에 속하는 관측치의 개수를 막대로 표현합니다. 단일 범주형 변수의 빈도 분포를 시각화하는 데 사용됩니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.countplot(x="day", data=tips)
plt.title("Number of Observations by Day (Countplot)")
plt.show()
```

### 3.6. 회귀 플롯 (Regression Plots)
두 변수 간의 선형 관계를 시각화하고 회귀선을 함께 표시합니다.

#### 3.6.1. `regplot()`
산점도와 함께 선형 회귀선을 그리고, 회귀선의 신뢰 구간을 함께 표시합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.regplot(x="total_bill", y="tip", data=tips)
plt.title("Regression Plot of Total Bill and Tip")
plt.show()
```

#### 3.6.2. `lmplot()`
`regplot()`과 유사하지만, `col`, `row`, `hue` 등의 파라미터를 사용하여 여러 서브플롯에 걸쳐 회귀 관계를 시각화할 수 있는 고수준 인터페이스입니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips) # 흡연 여부에 따라 회귀선 분리
plt.title("Regression Plot of Total Bill and Tip by Smoker Status")
plt.show()
```

### 3.7. 행렬 플롯 (Matrix Plots)
데이터 행렬의 관계를 색상으로 인코딩하여 시각화합니다. 주로 상관 행렬이나 데이터의 유사성을 보여줄 때 사용됩니다.

#### 3.7.1. `heatmap()`
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

#### 3.7.2. `clustermap()`
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

### 3.8. 사용자 정의 및 테마
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

## 4. Plotly

### 4.1. Plotly 소개
Plotly는 웹 기반의 인터랙티브(interactive) 시각화를 생성하는 강력한 오픈 소스 라이브러리입니다. Matplotlib이나 Seaborn과 달리, Plotly로 생성된 그래프는 웹 브라우저에서 직접 렌더링되며, 확대/축소, 팬(pan), 데이터 포인트 정보 확인(hover), 애니메이션 등 다양한 상호작용 기능을 제공합니다. 이는 데이터 탐색, 대시보드 구축, 웹 애플리케이션에 동적인 시각화를 포함할 때 매우 유용합니다.

**주요 특징**:
*   **인터랙티브**: 사용자와 상호작용할 수 있는 동적인 그래프를 생성합니다.
*   **웹 기반**: HTML, JavaScript, CSS를 사용하여 웹 브라우저에서 렌더링됩니다.
*   **다양한 언어 지원**: 파이썬 외에도 R, MATLAB, JavaScript, Julia 등 다양한 프로그래밍 언어를 지원합니다.
*   **Plotly Express**: 고수준 API인 Plotly Express를 통해 적은 코드로 복잡한 그래프를 쉽게 생성할 수 있습니다.
*   **Dash 통합**: Plotly 기반의 웹 애플리케이션 프레임워크인 Dash와 함께 사용하여 인터랙티브 대시보드를 구축할 수 있습니다.

### 4.2. 설치
Plotly는 `pip`를 사용하여 설치할 수 있습니다. Jupyter 환경에서 인터랙티브 플롯을 사용하려면 `jupyterlab` 또는 `notebook`도 설치되어 있어야 합니다.

```bash
pip install plotly
pip install "jupyterlab>=3"
# 또는 pip install "notebook>=5.3"
```

Plotly는 두 가지 주요 인터페이스를 제공합니다:
*   **Plotly Express (`px`)**: 고수준 API로, Pandas DataFrame을 입력으로 받아 빠르게 그래프를 생성할 수 있습니다. 대부분의 일반적인 시각화 요구사항을 충족합니다.
*   **Plotly Graph Objects (`go`)**: 저수준 API로, 그래프의 모든 요소를 세밀하게 제어할 수 있는 유연성을 제공합니다. 복잡하거나 사용자 정의된 그래프를 만들 때 사용됩니다.

일반적으로 Plotly Express를 먼저 사용하고, 더 세밀한 제어가 필요할 때 Graph Objects를 활용하는 것이 좋습니다.

```python
import plotly.express as px
import plotly.graph_objects as go
```

### 4.3. 기본 플로팅 (Plotly Express)
Plotly Express는 Pandas DataFrame을 기반으로 다양한 유형의 인터랙티브 그래프를 쉽게 생성할 수 있습니다.

#### 4.3.1. 산점도 (`px.scatter()`)
두 변수 간의 관계를 점으로 표현하며, 마우스 오버 시 데이터 정보를 보여주는 등 인터랙티브 기능을 제공합니다.

```python
import plotly.express as px

# 내장 데이터셋 사용
iris = px.data.iris()

# 꽃잎 길이(petal_length)와 너비(petal_width)의 산점도
fig = px.scatter(iris, x="petal_length", y="petal_width", title="Iris Petal Length vs. Width")
fig.show()

# 종(species)에 따라 색상 구분 및 마우스 오버 정보 추가
fig = px.scatter(iris, x="petal_length", y="petal_width", color="species",
                 hover_data=['sepal_length', 'sepal_width'], title="Iris Petal Length vs. Width by Species")
fig.show()

# 시간에 따른 변화를 애니메이션으로 표현 (Gapminder 데이터셋 활용)
gapminder = px.data.gapminder()
fig = px.scatter(gapminder, x="gdpPercap", y="lifeExp", animation_frame="year",
                 animation_group="country", size="pop", color="continent", hover_name="country",
                 log_x=True, size_max=55, title="GDP per Capita vs. Life Expectancy Over Time")
fig.show()
```

#### 4.3.2. 라인 플롯 (`px.line()`)
시계열 데이터나 연속적인 데이터의 추세를 보여줄 때 사용하며, 인터랙티브 기능을 통해 특정 구간을 확대하거나 데이터 포인트를 확인할 수 있습니다.

```python
import plotly.express as px

# 내장 데이터셋 사용 (Gapminder: 시간에 따른 국가별 인구, 기대 수명, GDP)
gapminder = px.data.gapminder()

# 시간에 따른 아프가니스탄의 기대 수명 변화
fig = px.line(gapminder.query("country=='Afghanistan'"), x="year", y="lifeExp", title="Life Expectancy in Afghanistan Over Time")
fig.show()

# 대륙별 기대 수명 추이 (색상 구분)
fig = px.line(gapminder.query("continent=='Asia'"), x="year", y="lifeExp", color="country", title="Life Expectancy in Asia by Country")
fig.show()
```

#### 4.3.3. 막대 그래프 (`px.bar()`)
범주형 데이터의 빈도나 값을 막대로 표현하며, 인터랙티브 기능을 통해 각 막대의 상세 정보를 확인할 수 있습니다.

```python
import plotly.express as px

# 내장 데이터셋 사용 (팁 데이터)
tips = px.data.tips()

# 요일별 총 계산액 평균
fig = px.bar(tips, x="day", y="total_bill", title="Average Total Bill by Day")
fig.show()

# 성별에 따른 요일별 팁 합계 (스택 막대 그래프)
fig = px.bar(tips, x="day", y="tip", color="sex", title="Total Tip by Day and Sex", barmode='group') # barmode='group'으로 그룹화
fig.show()
```

#### 4.3.4. 파이 차트 (`px.pie()`)
전체에 대한 각 부분의 비율을 보여줄 때 사용합니다.

```python
import plotly.express as px

# 내장 데이터셋 사용 (팁 데이터)
tips = px.data.tips()

# 흡연 여부(smoker) 비율
fig = px.pie(tips, names='smoker', title="Proportion of Smokers and Non-Smokers")
fig.show()

# 요일별 팁 비율 (각 요일 내 흡연 여부 비율)
fig = px.pie(tips, values='tip', names='day', title="Tip Proportion by Day", hole=0.3) # hole로 도넛 차트 생성
fig.show()
```

### 4.4. 인터랙티브 플롯
Plotly의 가장 큰 장점은 생성된 그래프가 기본적으로 인터랙티브하다는 점입니다. 마우스로 그래프를 조작하여 데이터를 더 깊이 탐색할 수 있습니다.

**주요 인터랙티브 기능**:
*   **확대/축소 (Zoom)**: 마우스 휠 또는 드래그하여 특정 영역 확대/축소.
*   **팬 (Pan)**: 드래그하여 그래프 이동.
*   **데이터 포인트 정보 (Hover)**: 마우스 커서를 데이터 포인트 위에 올리면 상세 정보 표시.
*   **선택 (Select)**: 특정 영역을 선택하여 데이터 필터링.
*   **툴바 (Toolbar)**: 플롯 상단에 나타나는 툴바를 통해 다양한 기능(다운로드, 리셋 등) 사용.

이러한 기능들은 `fig.show()`를 통해 그래프를 렌더링할 때 자동으로 활성화됩니다. 추가적인 설정 없이도 풍부한 사용자 경험을 제공합니다.

### 4.5. 서브플롯
Plotly Graph Objects를 사용하면 여러 개의 플롯을 하나의 Figure에 배치하는 서브플롯을 생성할 수 있습니다. `make_subplots` 함수를 사용합니다.

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 서브플롯 레이아웃 정의 (1행 2열)
fig = make_subplots(rows=1, cols=2, subplot_titles=('Sine Wave', 'Cosine Wave'))

x = np.linspace(0, 2 * np.pi, 100)

# 첫 번째 서브플롯에 라인 플롯 추가
fig.add_trace(go.Scatter(x=x, y=np.sin(x), mode='lines', name='Sine'), row=1, col=1)

# 두 번째 서브플롯에 라인 플롯 추가
fig.add_trace(go.Scatter(x=x, y=np.cos(x), mode='lines', name='Cosine'), row=1, col=2)

fig.update_layout(title_text="Sine and Cosine Waves in Subplots")
fig.show()
```

### 4.6. 플롯 저장
Plotly로 생성된 인터랙티브 플롯은 HTML 파일로 저장하여 웹 브라우저에서 그대로 열어볼 수 있습니다. 또한, 정적 이미지(PNG, JPEG, SVG, PDF)로도 저장할 수 있습니다.

```python
import plotly.express as px

iris = px.data.iris()
fig = px.scatter(iris, x="petal_length", y="petal_width", color="species")

# HTML 파일로 저장 (인터랙티브 기능 유지)
fig.write_html("iris_scatter.html")
print("'iris_scatter.html' 파일이 저장되었습니다.")

# PNG 이미지로 저장 (정적 이미지)
# Kaleido 라이브러리가 설치되어 있어야 합니다: pip install kaleido
try:
    fig.write_image("iris_scatter.png")
    print("'iris_scatter.png' 파일이 저장되었습니다.")
except ValueError:
    print("Kaleido 라이브러리가 설치되어 있지 않아 PNG 이미지 저장을 건너뜁니다. 'pip install kaleido'를 실행하세요.")

# PDF 이미지로 저장
try:
    fig.write_image("iris_scatter.pdf")
    print("'iris_scatter.pdf' 파일이 저장되었습니다.")
except ValueError:
    print("Kaleido 라이브러리가 설치되어 있지 않아 PDF 이미지 저장을 건너뜁니다. 'pip install kaleido'를 실행하세요.")
```

### 4.7. Dash와의 통합
Dash는 Plotly를 기반으로 하는 파이썬 웹 애플리케이션 프레임워크입니다. Dash를 사용하면 복잡한 웹 개발 지식 없이도 인터랙티브한 대시보드와 데이터 시각화 애플리케이션을 구축할 수 있습니다. Plotly 그래프는 Dash 앱에 쉽게 통합될 수 있으며, 사용자 입력에 따라 동적으로 업데이트되는 대시보드를 만들 수 있습니다.

**간단한 Dash 앱 예시 (설치 및 실행 필요)**:

```python
# pip install dash
# pip install dash-core-components
# pip install dash-html-components

# import dash
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# import plotly.express as px

# app = dash.Dash(__name__)

# df = px.data.gapminder()

# app.layout = html.Div([
#     dcc.Graph(id='life-exp-vs-gdp'),
#     dcc.Slider(
#         df['year'].min(),
#         df['year'].max(),
#         step=None,
#         value=df['year'].min(),
#         marks={str(year): str(year) for year in df['year'].unique()},
#         id='year-slider'
#     )
# ])


# @app.callback(
#     Output('life-exp-vs-gdp', 'figure'),
#     Input('year-slider', 'value'))
# def update_figure(selected_year):
#     filtered_df = df[df.year == selected_year]

#     fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
#                      size="pop", color="continent", hover_name="country",
#                      log_x=True, size_max=55)

#     fig.update_layout(transition_duration=500)

#     return fig


# if __name__ == '__main__':
#     app.run_server(debug=True)
```

## 5. 라이브러리 비교 및 사용 가이드

### 5.1. Matplotlib vs. Seaborn vs. Plotly

각 라이브러리는 고유한 강점과 사용 사례를 가지고 있습니다. 어떤 라이브러리를 선택할지는 프로젝트의 요구사항, 필요한 시각화 유형, 그리고 상호작용성 여부에 따라 달라집니다.

| 특징/라이브러리 | Matplotlib | Seaborn | Plotly |
| :-------------- | :--------- | :------ | :----- |
| **기반**        | 독립적 (파이썬) | Matplotlib | D3.js (JavaScript) | HTML/CSS/JavaScript |
| **API 수준**    | 저수준 (Low-level) | 고수준 (High-level) | 고수준 (Plotly Express) / 저수준 (Graph Objects) |
| **주요 용도**   | 범용적인 정적 플롯, 세밀한 커스터마이징, 과학적 플롯 | 통계적 시각화, 아름다운 기본 스타일, 데이터 분포 및 관계 탐색 | 인터랙티브 웹 플롯, 대시보드, 3D 플롯, 지리 공간 데이터 | 
| **상호작용성**  | 제한적 (정적 이미지) | 제한적 (정적 이미지) | 매우 높음 (확대/축소, 팬, 호버 등) |
| **코드 복잡성** | 높음 (세밀한 제어) | 낮음 (간결한 코드) | 중간 (Plotly Express는 낮음, Graph Objects는 높음) |
| **미적 품질**   | 기본값은 투박, 커스터마이징 필요 | 기본적으로 미려하고 전문적 | 기본적으로 매우 미려하고 현대적 |
| **데이터 타입** | NumPy 배열, Python 리스트 | Pandas DataFrame, NumPy 배열 | Pandas DataFrame, NumPy 배열, Python 리스트 |
| **출력 형식**   | PNG, JPG, PDF, SVG 등 정적 이미지 | PNG, JPG, PDF, SVG 등 정적 이미지 | HTML (인터랙티브), PNG, JPG, PDF, SVG 등 정적 이미지 |

### 5.2. 최적의 라이브러리 선택 가이드

*   **Matplotlib**: 
    *   **세밀한 제어**: 그래프의 모든 요소를 완벽하게 제어하고 싶을 때.
    *   **과학적/공학적 플롯**: 복잡한 수학 함수나 물리 현상을 시각화할 때.
    *   **다른 라이브러리 기반**: Seaborn이나 다른 라이브러리에서 제공하지 않는 특정 유형의 플롯을 만들거나, 기존 플롯을 더욱 세밀하게 조정해야 할 때.

*   **Seaborn**: 
    *   **통계적 분석**: 데이터의 분포, 변수 간 관계, 범주형 데이터의 특성 등을 탐색하고 시각화할 때.
    *   **빠른 탐색적 데이터 분석 (EDA)**: 적은 코드로 빠르게 데이터의 패턴을 파악하고 싶을 때.
    *   **미려한 기본 스타일**: 별도의 커스터마이징 없이도 전문적인 품질의 그래프를 얻고 싶을 때.

*   **Plotly**: 
    *   **인터랙티브 대시보드**: 웹 기반의 동적인 대시보드를 구축하거나 웹 애플리케이션에 시각화를 포함할 때.
    *   **데이터 탐색**: 사용자가 직접 데이터를 확대/축소하고 상세 정보를 확인할 수 있도록 하여 데이터에 대한 깊은 이해를 돕고 싶을 때.
    *   **3D 시각화**: 3차원 데이터나 지리 공간 데이터를 시각화할 때.
    *   **공유 및 협업**: 인터랙티브 그래프를 HTML 파일로 쉽게 공유하고 싶을 때.

**결론적으로, 대부분의 데이터 분석 작업에서는 Seaborn으로 빠르게 탐색적 시각화를 수행하고, 필요에 따라 Matplotlib으로 세밀하게 조정하거나 Plotly로 인터랙티브한 대시보드를 구축하는 방식으로 조합하여 사용하는 것이 가장 효과적입니다.**

### 5.3. 데이터 시각화 모범 사례

효과적인 데이터 시각화를 위해서는 단순히 그래프를 그리는 것을 넘어, 시각화의 목적과 대상 독자를 고려한 디자인 원칙을 따르는 것이 중요합니다.

1.  **명확한 목적 설정**: 어떤 질문에 답하고 싶은지, 어떤 메시지를 전달하고 싶은지 명확히 정의합니다.
2.  **올바른 차트 유형 선택**: 데이터의 종류(범주형, 연속형, 시계열 등)와 전달하고자 하는 메시지에 가장 적합한 차트 유형을 선택합니다. (예: 추세는 라인 플롯, 비교는 막대 그래프, 분포는 히스토그램/KDE).
3.  **간결하고 명확한 디자인**: 불필요한 요소(과도한 색상, 복잡한 배경, 3D 효과 등)를 제거하고, 데이터 자체에 집중할 수 있도록 합니다.
4.  **적절한 레이블링**: 제목, 축 레이블, 범례, 데이터 레이블 등을 명확하고 간결하게 작성하여 그래프의 의미를 쉽게 이해할 수 있도록 합니다.
5.  **색상 사용**: 색상은 데이터를 구분하거나 강조하는 데 효과적이지만, 과도하거나 부적절한 색상 사용은 혼란을 줄 수 있습니다. 색맹을 고려하고, 일관된 색상 팔레트를 사용합니다.
6.  **데이터 정규화/스케일링**: 여러 변수를 비교할 때는 스케일이 다른 변수들을 정규화하거나 스케일링하여 올바른 비교가 가능하도록 합니다.
7.  **상호작용성 활용 (필요시)**: Plotly와 같은 인터랙티브 도구를 사용하여 사용자가 데이터를 직접 탐색하고 더 깊은 통찰력을 얻을 수 있도록 합니다.
8.  **스토리텔링**: 시각화를 통해 데이터가 전달하는 스토리를 효과적으로 구성합니다. 여러 그래프를 조합하여 논리적인 흐름을 만들고, 핵심 메시지를 강조합니다.