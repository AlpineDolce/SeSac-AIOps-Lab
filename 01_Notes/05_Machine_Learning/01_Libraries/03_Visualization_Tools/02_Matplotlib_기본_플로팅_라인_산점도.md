<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 `pyplot` 모듈을 사용하여 가장 기본적인 라인 플롯(Line Plot)과 산점도(Scatter Plot)를 그리는 방법을 다룹니다. 각 플롯의 용도와 함께 데이터 생성부터 그래프 출력까지의 과정을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. Matplotlib 기본 플로팅](#1-matplotlib-기본-플로팅)
  - [1.1. 라인 플롯 (Line Plot)](#11-라인-플롯-line-plot)
    - [1.1.1. 단일 라인 플롯 그리기](#111-단일-라인-플롯-그리기)
    - [1.1.2. 여러 라인 플롯 함께 그리기](#112-여러-라인-플롯-함께-그리기)
    - [1.1.3. 라인 플롯 커스터마이징 (제목, 축 레이블, 그리드, 범례)](#113-라인-플롯-커스터마이징-제목-축-레이블-그리드-범례)
  - [1.2. 산점도 (Scatter Plot)](#12-산점도-scatter-plot)
    - [1.2.1. 기본 산점도 그리기](#121-기본-산점도-그리기)
    - [1.2.2. 산점도 커스터마이징 (색상, 크기, 투명도, 컬러바)](#122-산점도-커스터마이징-색상-크기-투명도-컬러바)

---

## 1. Matplotlib 기본 플로팅
Matplotlib의 `pyplot` 모듈은 다양한 종류의 그래프를 그릴 수 있는 함수를 제공합니다.

### 1.1. 라인 플롯 (Line Plot)
가장 기본적인 플롯으로, 데이터 포인트들을 선으로 연결하여 시계열 데이터나 연속적인 데이터의 변화 추이를 보여줄 때 사용합니다. `plt.plot()` 함수를 사용합니다.

#### 1.1.1. 단일 라인 플롯 그리기
Matplotlib에서 가장 기본적인 시각화는 `plt.plot()` 함수를 사용하여 라인 플롯을 그리는 것입니다. 이 함수는 주로 연속적인 데이터의 변화 추이를 시각화할 때 사용됩니다.

**`plt.plot(x, y)` 함수의 기본 사용법**:
`plt.plot()` 함수는 최소한 두 개의 인자, 즉 x축에 해당하는 데이터와 y축에 해당하는 데이터를 필요로 합니다. 이 두 데이터 배열의 길이가 같아야 합니다.

**코드 설명**:
1.  **라이브러리 임포트**:
    *   `import matplotlib.pyplot as plt`: Matplotlib의 `pyplot` 모듈을 `plt`라는 별칭으로 임포트합니다. `pyplot`은 그래프를 생성하고 조작하는 데 필요한 다양한 함수들을 제공합니다.
    *   `import numpy as np`: 수치 계산을 위한 `numpy` 라이브러리를 `np`라는 별칭으로 임포트합니다. 데이터 생성에 주로 사용됩니다.

2.  **데이터 생성**:
    *   `x = np.linspace(0, 10, 100)`: `numpy.linspace` 함수는 지정된 시작 값(0)과 끝 값(10) 사이를 균등한 간격으로 나눈 숫자들을 생성합니다. 여기서는 100개의 숫자를 생성하여 x축 데이터로 사용합니다. 이는 연속적인 변화를 보여주는 데 적합한 데이터를 만듭니다.
    *   `y = np.sin(x)`: 생성된 `x` 값들에 대한 사인(sine) 함수 값을 계산하여 y축 데이터로 사용합니다.

3.  **라인 플롯 그리기 및 커스터마이징**:
    *   `plt.plot(x, y)`: `x`와 `y` 데이터를 사용하여 라인 플롯을 그립니다. Matplotlib은 내부적으로 Figure(전체 그림 영역)와 Axes(실제 플롯이 그려지는 영역) 객체를 생성하고, 이 Axes 위에 라인을 그립니다.
    *   `plt.title("Simple Line Plot")`: 그래프의 상단에 표시될 제목을 설정합니다.
    *   `plt.xlabel("X-axis")`: x축에 대한 레이블을 설정합니다.
    *   `plt.ylabel("Y-axis")`: y축에 대한 레이블을 설정합니다.
    *   `plt.grid(True)`: 그래프 배경에 격자(그리드)를 표시하여 데이터 포인트를 더 쉽게 읽을 수 있도록 돕습니다. `True` 대신 `False`를 사용하면 그리드를 숨길 수 있습니다.

4.  **그래프 표시**:
    *   `plt.show()`: 지금까지 설정한 모든 플롯과 커스터마이징을 화면에 렌더링하고 표시합니다. 이 함수를 호출해야 실제로 그래프 창이 나타나거나 Jupyter Notebook/Lab 환경에서 인라인으로 그래프가 출력됩니다. `plt.show()`가 호출되기 전까지는 그래프가 메모리에만 존재합니다.

이 과정을 통해 간단하지만 정보 전달력이 높은 라인 플롯을 생성할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100) # 0부터 10까지 100개의 등간격 숫자
y = np.sin(x) # x에 대한 사인 값

# 단일 라인 플롯 그리기
plt.plot(x, y)
plt.title("Simple Line Plot") # 그래프 제목
plt.xlabel("X-axis") # x축 레이블
plt.ylabel("Y-axis") # y축 레이블
plt.grid(True) # 그리드 표시
plt.show() # 그래프 보여주기
```

#### 1.1.2. 여러 라인 플롯 함께 그리기
하나의 Matplotlib Axes(플롯 영역)에 여러 개의 라인 플롯을 함께 그릴 수 있습니다. 이는 서로 다른 데이터셋의 추이를 비교하거나, 동일한 데이터셋 내에서 여러 변수의 변화를 한눈에 파악할 때 매우 유용합니다.

**여러 라인 플롯을 그리는 방법**:
동일한 Axes에 여러 라인 플롯을 추가하려면 `plt.plot()` 함수를 원하는 라인의 수만큼 연속적으로 호출하면 됩니다. Matplotlib은 자동으로 각 라인에 다른 색상을 할당하여 구별하기 쉽게 합니다.

**코드 설명**:
1.  **데이터 생성**:
    *   `x = np.linspace(0, 10, 100)`: x축 데이터는 이전 예제와 동일하게 0부터 10까지의 100개 등간격 숫자를 사용합니다.
    *   `y = np.sin(x)`: 첫 번째 라인 플롯을 위한 사인(sine) 함수 값을 생성합니다.
    *   `y2 = np.cos(x)`: 두 번째 라인 플롯을 위한 코사인(cosine) 함수 값을 생성합니다.

2.  **여러 라인 플롯 그리기**:
    *   `plt.plot(x, y, label='Sine')`: 첫 번째 라인 플롯을 그립니다. 여기서 `label='Sine'`은 이 라인을 식별할 수 있는 이름을 부여합니다. 이 `label`은 나중에 범례(legend)를 표시할 때 사용됩니다.
    *   `plt.plot(x, y2, label='Cosine')`: 두 번째 라인 플롯을 그립니다. 마찬가지로 `label='Cosine'`을 통해 이름을 부여합니다.

3.  **그래프 커스터마이징 및 범례 추가**:
    *   `plt.title("Multiple Line Plots")`, `plt.xlabel("X-axis")`, `plt.ylabel("Y-axis")`, `plt.grid(True)`: 이전 예제와 동일하게 그래프 제목, 축 레이블, 그리드를 설정합니다.
    *   `plt.legend()`: 이 함수를 호출하면 `plt.plot()` 함수에서 `label` 파라미터로 지정했던 이름들이 그래프 내에 범례로 표시됩니다. 범례는 각 라인이 어떤 데이터를 나타내는지 명확하게 알려주어 그래프의 가독성을 크게 높여줍니다.

4.  **그래프 표시**:
    *   `plt.show()`: 설정된 모든 라인 플롯과 커스터마이징을 화면에 출력합니다.

이 방법을 통해 복잡한 데이터 관계를 하나의 시각화로 효과적으로 비교하고 분석할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100) # 0부터 10까지 100개의 등간격 숫자
y = np.sin(x) # x에 대한 사인 값
y2 = np.cos(x) # x에 대한 코사인 값

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

#### 1.1.3. 라인 플롯 커스터마이징 (제목, 축 레이블, 그리드, 범례)
Matplotlib은 그래프의 가독성과 정보 전달력을 높이기 위해 다양한 요소를 커스터마이징할 수 있는 풍부한 기능을 제공합니다. 여기서는 라인 플롯에서 자주 사용되는 제목, 축 레이블, 그리드, 범례 설정 방법에 대해 자세히 설명합니다.

**주요 커스터마이징 함수**:

1.  **`plt.title(title_string, **kwargs)` - 그래프 제목 설정**:
    *   그래프의 전체 제목을 설정합니다.
    *   `title_string`: 그래프에 표시할 제목 문자열입니다.
    *   `**kwargs`: 폰트 크기(`fontsize`), 색상(`color`), 폰트 스타일(`fontweight`) 등 다양한 텍스트 속성을 추가로 지정할 수 있습니다.
    *   예시: `plt.title("My Awesome Plot", fontsize=16, color='blue')`

2.  **`plt.xlabel(label_string, **kwargs)` / `plt.ylabel(label_string, **kwargs)` - 축 레이블 설정**:
    *   각 축의 의미를 설명하는 레이블을 설정합니다.
    *   `label_string`: 축에 표시할 레이블 문자열입니다.
    *   `**kwargs`: `fontsize`, `color` 등 텍스트 속성을 지정할 수 있습니다.
    *   예시: `plt.xlabel("Time (seconds)", fontsize=12)`

3.  **`plt.grid(b=None, which='major', axis='both', **kwargs)` - 그리드 표시**:
    *   그래프 배경에 격자(그리드)를 표시하여 데이터 포인트를 더 쉽게 읽고 비교할 수 있도록 돕습니다.
    *   `b`: `True` 또는 `False`로 그리드 표시 여부를 설정합니다. (최신 버전에서는 `True`만 사용해도 됨)
    *   `which`: 'major' (주요 눈금), 'minor' (보조 눈금), 'both' (모두) 중 어떤 그리드를 표시할지 지정합니다.
    *   `axis`: 'x', 'y', 'both' 중 어떤 축에 그리드를 표시할지 지정합니다.
    *   `**kwargs`: 그리드 선의 색상(`color`), 스타일(`linestyle`), 두께(`linewidth`) 등을 조절할 수 있습니다.
    *   예시: `plt.grid(True, linestyle='--', alpha=0.7)`

4.  **`plt.legend(**kwargs)` - 범례 표시**:
    *   여러 라인 플롯을 그렸을 때, 각 라인이 어떤 데이터를 나타내는지 설명하는 범례를 표시합니다. `plt.plot()` 함수 호출 시 `label` 파라미터로 각 라인의 이름을 지정해야 합니다.
    *   `loc`: 범례의 위치를 지정합니다 (예: 'upper right', 'lower left', 'best' 등).
    *   `fontsize`: 범례 텍스트의 폰트 크기를 설정합니다.
    *   예시: `plt.legend(loc='upper right', fontsize='small')`

이러한 커스터마이징 함수들을 적절히 활용하면, 복잡한 데이터도 명확하고 효과적으로 전달하는 시각화를 만들 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 라인 플롯 그리기
plt.plot(x, y_sin, label='Sine Wave', color='red', linestyle='-')
plt.plot(x, y_cos, label='Cosine Wave', color='blue', linestyle='--')

# 그래프 커스터마이징
plt.title("Sine and Cosine Waves", fontsize=18, color='darkgreen') # 제목 설정
plt.xlabel("Angle (radians)", fontsize=14, color='gray') # x축 레이블 설정
plt.ylabel("Amplitude", fontsize=14, color='gray') # y축 레이블 설정

plt.grid(True, linestyle=':', alpha=0.6) # 그리드 표시 및 스타일 설정

plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True) # 범례 표시 및 위치, 스타일 설정

# 축 범위 설정 (선택 사항)
plt.xlim(0, 10)
plt.ylim(-1.2, 1.2)

plt.show()
```

### 1.2. 산점도 (Scatter Plot)
두 변수 간의 관계를 점으로 표현하는 플롯입니다. 데이터 포인트들의 분포나 군집을 파악하는 데 유용합니다. `plt.scatter()` 함수를 사용합니다.

#### 1.2.1. 기본 산점도 그리기
산점도(Scatter Plot)는 두 변수 간의 관계를 시각적으로 탐색하는 데 사용되는 강력한 도구입니다. 각 데이터 포인트는 x축과 y축의 값에 따라 평면상의 한 점으로 표현됩니다. Matplotlib에서는 `plt.scatter()` 함수를 사용하여 산점도를 그립니다.

**`plt.scatter(x, y)` 함수의 기본 사용법**:
`plt.scatter()` 함수는 `plt.plot()`과 유사하게 x축과 y축에 해당하는 데이터 배열을 인자로 받습니다. 하지만 `plt.plot()`이 데이터 포인트를 선으로 연결하는 반면, `plt.scatter()`는 각 데이터 포인트를 독립적인 점으로 표시합니다. 이는 데이터의 분포, 군집, 이상치(outliers) 등을 파악하는 데 특히 유용합니다.

**코드 설명**:
1.  **라이브러리 임포트**:
    *   `import matplotlib.pyplot as plt`: Matplotlib의 `pyplot` 모듈을 임포트합니다.
    *   `import numpy as np`: `numpy` 라이브러리를 임포트하여 데이터 생성에 활용합니다.

2.  **데이터 생성**:
    *   `x = np.random.rand(50) * 10`: `numpy.random.rand(50)`는 0과 1 사이의 균일 분포에서 50개의 난수를 생성합니다. 여기에 10을 곱하여 0부터 10 사이의 랜덤한 x 값을 50개 생성합니다.
    *   `y = np.random.rand(50) * 10`: x와 마찬가지로 0부터 10 사이의 랜덤한 y 값을 50개 생성합니다. 이처럼 무작위로 생성된 데이터는 두 변수 사이에 명확한 관계가 없는 경우의 산점도를 보여줍니다.

3.  **기본 산점도 그리기**:
    *   `plt.scatter(x, y)`: 생성된 `x`와 `y` 데이터를 사용하여 산점도를 그립니다. 각 (x, y) 쌍이 하나의 점으로 플롯됩니다.
    *   `plt.title("Basic Scatter Plot")`: 그래프의 제목을 설정합니다.
    *   `plt.xlabel("X-value")`: x축의 레이블을 설정합니다.
    *   `plt.ylabel("Y-value")`: y축의 레이블을 설정합니다.

4.  **그래프 표시**:
    *   `plt.show()`: 설정된 산점도를 화면에 출력합니다.

이 예제를 통해 두 변수 간의 기본적인 관계를 점의 분포로 시각화하는 방법을 이해할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.random.rand(50) * 10 # 0-10 사이의 랜덤 x 값 50개
y = np.random.rand(50) * 10 # 0-10 사이의 랜덤 y 값 50개

# 기본 산점도 그리기
plt.scatter(x, y)
plt.title("Basic Scatter Plot")
plt.xlabel("X-value")
plt.ylabel("Y-value")
plt.show()
```

#### 1.2.2. 산점도 커스터마이징 (색상, 크기, 투명도, 컬러바)
산점도는 단순히 두 변수 간의 관계를 보여주는 것을 넘어, 점의 색상, 크기, 투명도 등을 조절하여 세 번째, 네 번째 변수까지 시각화할 수 있는 강력한 기능을 제공합니다. 이를 통해 다차원 데이터의 패턴을 한눈에 파악할 수 있습니다.

**주요 커스터마이징 파라미터**:

1.  **`c` (color) - 점의 색상**:
    *   각 점의 색상을 지정합니다.
    *   **단일 색상**: 'red', 'blue', '#FF5733' 등 색상 이름이나 헥스 코드를 문자열로 전달할 수 있습니다.
    *   **변수에 따른 색상**: 각 점에 해당하는 숫자 배열을 전달하면, Matplotlib은 이 숫자 값의 범위에 따라 점의 색상을 다르게 표현합니다. 이는 세 번째 변수를 색상으로 인코딩하는 데 사용됩니다.
    *   **`cmap` (colormap) - 컬러맵**: `c` 파라미터에 숫자 배열을 전달했을 때, 해당 숫자 값을 어떤 색상 스펙트럼으로 매핑할지 결정합니다. 'viridis', 'plasma', 'coolwarm' 등 다양한 내장 컬러맵이 있습니다.

2.  **`s` (size) - 점의 크기**:
    *   각 점의 크기를 지정합니다.
    *   **단일 크기**: 모든 점에 동일한 크기를 적용하려면 숫자를 전달합니다 (예: `s=50`).
    *   **변수에 따른 크기**: 각 점에 해당하는 숫자 배열을 전달하면, 해당 숫자 값에 비례하여 점의 크기가 달라집니다. 이는 네 번째 변수를 크기로 인코딩하는 데 사용됩니다.

3.  **`alpha` (투명도) - 점의 투명도**:
    *   점의 투명도를 0.0(완전 투명)에서 1.0(완전 불투명) 사이의 값으로 지정합니다.
    *   데이터 포인트가 많아 서로 겹쳐 보일 때 `alpha` 값을 1.0보다 작게 설정하면, 겹치는 영역의 밀도를 시각적으로 표현하여 데이터의 밀집도를 파악하는 데 유용합니다.

4.  **`plt.colorbar(label=None, **kwargs)` - 컬러바 추가**:
    *   `c` 파라미터에 숫자 배열을 사용하여 점의 색상을 지정했을 때, 해당 색상이 어떤 값의 범위를 나타내는지 설명하는 컬러바를 그래프 옆에 추가합니다.
    *   `label`: 컬러바의 제목을 설정합니다.
    *   `**kwargs`: 컬러바의 위치, 크기, 폰트 등 다양한 속성을 조절할 수 있습니다.

이러한 커스터마이징 옵션들을 활용하면, 2차원 산점도에 추가적인 정보를 효과적으로 담아내어 데이터 분석의 깊이를 더할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
np.random.seed(42) # 재현성을 위해 시드 설정
num_points = 100
x = np.random.rand(num_points) * 10
y = np.random.rand(num_points) * 10
# 세 번째 변수 (색상으로 표현)
colors = np.random.rand(num_points) * 100 # 0-100 사이의 랜덤 값
# 네 번째 변수 (크기로 표현)
sizes = np.random.rand(num_points) * 500 + 50 # 50-550 사이의 랜덤 값 (최소 크기 50)

# 산점도 커스터마이징
plt.figure(figsize=(10, 7)) # 그래프 크기 설정
scatter = plt.scatter(x, y,
                      c=colors,       # 색상: colors 배열의 값에 따라
                      s=sizes,        # 크기: sizes 배열의 값에 따라
                      alpha=0.6,      # 투명도: 0.6
                      cmap='viridis', # 컬러맵: 'viridis'
                      edgecolors='w', # 점 테두리 색상: 흰색
                      linewidth=0.5)  # 점 테두리 두께

# 그래프 제목 및 축 레이블 설정
plt.title("Customized Scatter Plot: Encoding Multiple Variables", fontsize=16)
plt.xlabel("X-value (Variable 1)", fontsize=12)
plt.ylabel("Y-value (Variable 2)", fontsize=12)

# 컬러바 추가
plt.colorbar(scatter, label="Color Intensity (Variable 3)")

plt.grid(True, linestyle=':', alpha=0.7) # 그리드 추가
plt.show()
```
