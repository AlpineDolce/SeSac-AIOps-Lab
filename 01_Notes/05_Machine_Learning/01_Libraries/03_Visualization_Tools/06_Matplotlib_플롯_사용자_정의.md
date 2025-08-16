<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib 플롯의 다양한 요소를 커스터마이징하는 방법을 다룹니다. 제목, 축 레이블, 범례 설정부터 색상, 마커, 선 스타일 변경, 축 범위 및 눈금 설정, 그리고 고급 커스터마이징 기법까지 학습하여 시각화의 가독성과 미적 품질을 향상시키는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Matplotlib 플롯 사용자 정의](#1-matplotlib-플롯-사용자-정의)
  - [1.1. 제목, 축 레이블, 범례 설정](#11-제목-축-레이블-범례-설정)
    - [1.1.1. `plt.title()`: 그래프 제목 설정](#111-plt-title-그래프-제목-설정)
    - [1.1.2. `plt.xlabel()`, `plt.ylabel()`: 축 레이블 설정](#112-plt-xlabel-plt-ylabel-축-레이블-설정)
    - [1.1.3. `plt.legend()`: 범례 표시](#113-plt-legend-범례-표시)
    - [1.1.4. `plt.grid()`: 그리드 표시](#114-plt-grid-그리드-표시)
    - [1.1.5. 그림 크기 및 해상도 설정](#115-그림-크기-및-해상도-설정)
    - [1.1.6. 고급 범례 설정 (ncol, bbox_to_anchor)](#116-고급-범례-설정-ncol-bbox_to_anchor)
  - [1.2. 색상, 마커, 선 스타일 변경](#12-색상-마커-선-스타일-변경)
    - [1.2.1. `color`: 선/마커 색상](#121-color-선-마커-색상)
    - [1.2.2. `marker`: 데이터 포인트 마커](#122-marker-데이터-포인트-마커)
    - [1.2.3. `linestyle`: 선 스타일](#123-linestyle-선-스타일)
    - [1.2.4. `linewidth`, `markersize`: 선 두께 및 마커 크기](#124-linewidth-markersize-선-두께-및-마커-크기)
  - [1.3. 축 범위 및 눈금 설정](#13-축-범위-및-눈금-설정)
    - [1.3.1. `plt.xlim()`, `plt.ylim()`: 축 범위 설정](#131-plt-xlim-plt-ylim-축-범위-설정)
    - [1.3.2. `plt.xticks()`, `plt.yticks()`: 눈금 위치 및 레이블 설정](#132-plt-xticks-plt-yticks-눈금-위치-및-레이블-설정)
    - [1.3.3. 서브플롯 간격 조정 (`plt.tight_layout()`)](#133-서브플롯-간격-조정-plttight_layout)
    - [1.3.4. 주/보조 눈금 설정 (Major/Minor Ticks)](#134-주보조-눈금-설정-majorminor-ticks)
  - [1.4. 고급 커스터마이징 (Advanced Customization)](#14-고급-커스터마이징-advanced-customization)
    - [1.4.1. 스타일 시트 사용 (`plt.style.use()`)](#141-스타일-시트-사용-plt-style-use)
    - [1.4.2. `rcParams`를 이용한 전역 설정](#142-rcParams를-이용한-전역-설정)
    - [1.4.3. 객체 지향 API를 통한 세밀한 제어](#143-객체-지향-API를-통한-세밀한-제어)
    - [1.4.4. 텍스트 및 주석 추가](#144-텍스트-및-주석-추가)
    - [1.4.5. 축 테두리 (Spines) 커스터마이징](#145-축-테두리-spines-커스터마이징)
    - [1.4.6. 축에 수직/수평선 추가 (axvline/axhline)](#146-축에-수직수평선-추가-axvlineaxhline)

---

## 1. Matplotlib 플롯 사용자 정의
Matplotlib은 플롯의 다양한 요소를 커스터마이징할 수 있는 기능을 제공하여 시각화의 가독성과 미적 품질을 향상시킬 수 있습니다.

### 1.1. 제목, 축 레이블, 범례 설정
Matplotlib에서 그래프의 의미를 명확하게 전달하고 가독성을 높이는 가장 기본적인 방법은 제목, 축 레이블, 그리고 범례를 적절히 사용하는 것입니다.

#### 1.1.1. `plt.title()`: 그래프 제목 설정
`plt.title()` 함수는 그래프의 전체 제목을 설정합니다. 그래프의 내용을 한눈에 파악할 수 있도록 간결하고 설명적인 제목을 사용하는 것이 중요합니다.

**주요 파라미터**:
*   `title_string`: 그래프에 표시할 제목 문자열.
*   `fontsize`: 제목의 폰트 크기.
*   `color`: 제목의 색상.
*   `fontweight`: 제목의 폰트 두께 (예: 'bold').

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)

plt.plot(x, y1)
plt.title("Sine Wave Example", fontsize=16, color='darkblue', fontweight='bold')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

#### 1.1.2. `plt.xlabel()`, `plt.ylabel()`: 축 레이블 설정
`plt.xlabel()`과 `plt.ylabel()` 함수는 각각 x축과 y축의 레이블을 설정합니다. 각 축이 나타내는 데이터의 단위나 의미를 명확히 설명하여 그래프를 이해하는 데 도움을 줍니다.

**주요 파라미터**:
*   `label_string`: 축에 표시할 레이블 문자열.
*   `fontsize`: 레이블의 폰트 크기.
*   `color`: 레이블의 색상.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)

plt.plot(x, y1)
plt.title("Sine Wave with Axis Labels")
plt.xlabel("Time (seconds)", fontsize=12, color='gray')
plt.ylabel("Amplitude (units)", fontsize=12, color='gray')
plt.show()
```

#### 1.1.3. `plt.legend()`: 범례 표시
여러 개의 데이터 시리즈(예: 여러 라인, 막대)를 하나의 그래프에 그릴 때, 각 시리즈가 무엇을 의미하는지 구분하기 위해 범례(legend)를 사용합니다. `plt.plot()` 등의 함수에서 `label` 파라미터를 사용하여 각 시리즈의 이름을 지정한 후, `plt.legend()`를 호출하면 해당 레이블들이 범례로 표시됩니다.

**주요 파라미터**:
*   `loc`: 범례의 위치를 지정합니다 (예: 'upper right', 'lower left', 'best').
*   `fontsize`: 범례 텍스트의 폰트 크기.
*   `frameon`: 범례 주위에 프레임을 그릴지 여부 (True/False).
*   `shadow`: 범례에 그림자 효과를 추가할지 여부 (True/False).

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='Sine Wave', color='blue', linestyle='-')
plt.plot(x, y2, label='Cosine Wave', color='red', linestyle='--')

plt.title("Sine and Cosine Waves with Legend")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True) # 범례 표시 및 커스터마이징
plt.show()
```

#### 1.1.4. `plt.grid()`: 그리드 표시
`plt.grid()` 함수는 그래프 배경에 격자(그리드)를 표시하여 데이터 포인트를 더 쉽게 읽고 비교할 수 있도록 돕습니다.

**주요 파라미터**:
*   `b`: 그리드 표시 여부 (True/False). (최신 버전에서는 `plt.grid(True)`처럼 직접 `True`를 전달하는 것이 일반적)
*   `which`: 'major' (주요 눈금), 'minor' (보조 눈금), 'both' (모두) 중 어떤 그리드를 표시할지 지정.
*   `axis`: 'x', 'y', 'both' 중 어떤 축에 그리드를 표시할지 지정.
*   `linestyle`: 그리드 선의 스타일 (예: `'-'`, `'--'`, `':'`).
*   `alpha`: 그리드 선의 투명도.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)

plt.plot(x, y1)
plt.title("Sine Wave with Grid")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True, linestyle='--', alpha=0.7, color='gray') # 그리드 표시 및 커스터마이징
plt.show()
```
#### 1.1.5. 그림 크기 및 해상도 설정
그래프의 전체적인 크기(`figsize`)와 해상도(`dpi`)는 시각화의 가독성과 출력 품질에 직접적인 영향을 미칩니다. `plt.figure()` 함수를 사용하여 이들을 설정할 수 있습니다.

*   **`figsize`**: 그림의 크기를 인치 단위의 튜플 `(width, height)`로 지정합니다. (예: `(10, 6)`은 가로 10인치, 세로 6인치)
*   **`dpi`**: 그림의 해상도를 인치당 도트(dots per inch)로 지정합니다. 값이 높을수록 더 선명하고 고품질의 이미지가 생성됩니다. 특히 인쇄용이나 고해상도 디스플레이를 위한 이미지를 만들 때 중요합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

# 그림 크기를 10x6 인치로, 해상도를 100 dpi로 설정
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(x, y)
plt.title("Plot with Custom Figure Size and DPI")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()

# 참고: plt.savefig()에서도 dpi를 설정할 수 있습니다.
# plt.savefig('custom_size_dpi_plot.png', dpi=300)
```

#### 1.1.6. 고급 범례 설정 (ncol, bbox_to_anchor)
범례 항목이 많거나 플롯과 겹칠 때, `ncol`과 `bbox_to_anchor` 파라미터를 사용하여 범례를 더 효과적으로 배치할 수 있습니다.

*   **`ncol`**: 범례에 표시될 열(column)의 개수를 지정합니다.
*   **`bbox_to_anchor`**: 범례의 위치를 정교하게 조정합니다. 플롯 영역을 기준으로 상대적인 좌표를 사용하며, `loc` 파라미터와 함께 사용하여 범례를 플롯 영역 밖으로 배치할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x*2)
y4 = np.cos(x*2)

plt.plot(x, y1, label='Sine')
plt.plot(x, y2, label='Cosine')
plt.plot(x, y3, label='Sine x2')
plt.plot(x, y4, label='Cosine x2')

plt.title("Advanced Legend Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 범례를 2열로, 플롯의 오른쪽 바깥에 배치
plt.legend(ncol=2, loc='upper left', bbox_to_anchor=(1, 1))

# tight_layout은 bbox_to_anchor로 인해 플롯 영역이 변경된 후 호출되어야 할 수 있습니다.
# 하지만, 때로는 범례가 잘릴 수 있으므로, fig.tight_layout(pad=...)나 subplots_adjust로 조정이 필요할 수 있습니다.
plt.show()
```

### 1.2. 색상, 마커, 선 스타일 변경
Matplotlib에서는 `plt.plot()` 함수를 사용할 때 `color`, `marker`, `linestyle` 등의 파라미터를 통해 라인과 데이터 포인트의 시각적 속성을 세밀하게 제어할 수 있습니다. 이를 통해 여러 데이터 시리즈를 명확하게 구분하고, 그래프의 미적 품질을 향상시킬 수 있습니다.

#### 1.2.1. `color`: 선/마커 색상
`color` 파라미터는 라인이나 마커의 색상을 지정합니다. 다양한 방법으로 색상을 지정할 수 있습니다.

*   **색상 이름**: `'red'`, `'blue'`, `'green'`, `'black'`, `'cyan'`, `'magenta'`, `'yellow'`, `'white'` 등.
*   **약어**: `'r'`, `'g'`, `'b'`, `'k'`, `'c'`, `'m'`, `'y'`, `'w'` 등.
*   **Hex 코드**: `'#FF0000'` (빨강), `'#00FF00'` (초록) 등.
*   **RGB 튜플**: `(0.1, 0.2, 0.5)` (0에서 1 사이의 값).

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, color='purple', label='Sine')
plt.plot(x, y2, color='#FF8C00', label='Cosine') # DarkOrange Hex code

plt.title("Plot with Custom Colors")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
```

#### 1.2.2. `marker`: 데이터 포인트 마커
`marker` 파라미터는 각 데이터 포인트에 표시될 마커의 모양을 지정합니다. 데이터 포인트의 위치를 명확히 하거나, 특정 데이터 포인트를 강조할 때 유용합니다.

**주요 마커 스타일**:
*   `'o'`: 원
*   `'s'`: 사각형
*   `'^'`: 위쪽 삼각형
*   `'v'`: 아래쪽 삼각형
*   `'D'`: 다이아몬드
*   `'x'`: X
*   `'+'`: 플러스
*   `'*'`: 별표
*   `'.'`: 점

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10)
y = x**2

plt.plot(x, y, marker='s', markersize=8, color='blue', linestyle='-', label='Squared')

plt.title("Plot with Square Markers")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
```

#### 1.2.3. `linestyle`: 선 스타일
`linestyle` 파라미터는 라인의 스타일(실선, 점선 등)을 지정합니다. 여러 라인을 구분하는 데 색상과 마커 외에 추가적인 시각적 단서를 제공합니다.

**주요 선 스타일**:
*   `'-'` 또는 `'solid'`: 실선 (기본값)
*   `'--'` 또는 `'dashed'`: 점선
*   `'-.'` 또는 `'dashdot'`: 점-선
*   `':'` 또는 `'dotted'`: 점선

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, linestyle='--', color='green', label='Sine (Dashed)')
plt.plot(x, y2, linestyle=':', color='orange', label='Cosine (Dotted)')

plt.title("Plot with Custom Line Styles")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
```

#### 1.2.4. `linewidth`, `markersize`: 선 두께 및 마커 크기
`linewidth` 파라미터는 라인의 두께를, `markersize` 파라미터는 마커의 크기를 조절합니다. 이들을 통해 특정 라인이나 데이터 포인트를 강조할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10)
y1 = x
y2 = x**2
y3 = x**3

plt.plot(x, y1, color='green', linestyle='-', marker='o', linewidth=2, markersize=8, label='Linear')
plt.plot(x, y2, color='red', linestyle='--', marker='s', linewidth=1.5, markersize=6, label='Quadratic')
plt.plot(x, y3, color='blue', linestyle=':', marker='^', linewidth=1, markersize=4, label='Cubic')

plt.title("Different Plot Styles")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
```

### 1.3. 축 범위 및 눈금 설정
축의 범위와 눈금(ticks)을 적절히 설정하는 것은 그래프의 특정 부분을 강조하거나, 불필요한 여백을 제거하여 데이터의 패턴을 더 명확하게 보여주는 데 중요합니다.

#### 1.3.1. `plt.xlim()`, `plt.ylim()`: 축 범위 설정
`plt.xlim(xmin, xmax)`와 `plt.ylim(ymin, ymax)` 함수는 각각 x축과 y축의 표시 범위를 수동으로 설정합니다. 이를 통해 데이터의 특정 구간을 확대하여 보거나, 이상치(outlier)로 인해 그래프가 왜곡되는 것을 방지할 수 있습니다.

**주요 파라미터**:
*   `xmin`, `xmax`: x축의 최소 및 최대 값.
*   `ymin`, `ymax`: y축의 최소 및 최대 값.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.exp(x) # 지수 함수 데이터

plt.plot(x, y)
plt.title("Exponential Growth with Custom Y-axis Limit")
plt.xlabel("X-value")
plt.ylabel("Y-value")

# y축 범위를 0에서 1000으로 제한
plt.ylim(0, 1000)

plt.grid(True)
plt.show()
```

#### 1.3.2. `plt.xticks()`, `plt.yticks()`: 눈금 위치 및 레이블 설정
`plt.xticks(ticks, labels)`와 `plt.yticks(ticks, labels)` 함수는 축의 눈금(tick)이 표시될 위치와 해당 눈금에 표시될 레이블을 수동으로 제어합니다. 이는 특히 비선형적인 데이터나 특정 의미를 가진 눈금 레이블이 필요할 때 유용합니다.

**주요 파라미터**:
*   `ticks`: 눈금이 표시될 위치를 나타내는 숫자 리스트 또는 배열.
*   `labels`: 각 눈금 위치에 표시될 문자열 레이블 리스트 (선택 사항). `labels`를 지정하지 않으면 `ticks` 값이 그대로 레이블로 사용됩니다.

**`plt.tick_params()`: 눈금의 세부 속성 제어**
`plt.tick_params()` 함수는 눈금의 방향, 길이, 색상, 레이블의 폰트 크기 등 눈금과 관련된 다양한 세부 속성을 제어할 수 있습니다.
*   `axis`: 'x', 'y', 'both' 중 어떤 축의 눈금을 제어할지 지정합니다.
*   `direction`: 눈금의 방향 ('in', 'out', 'inout').
*   `length`: 눈금의 길이.
*   `width`: 눈금의 두께.
*   `colors`: 눈금의 색상.
*   `labelsize`: 눈금 레이블의 폰트 크기.
*   `bottom`, `top`, `left`, `right`: 해당 위치의 눈금 표시 여부 (True/False).

예시: `plt.tick_params(axis='x', direction='inout', length=6, colors='r', labelsize=10)`

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
           ['0', '$\pi/2, '$\pi, '3$\pi/2, '2$\pi])
```
#### 1.3.3. 서브플롯 간격 조정 (`plt.tight_layout()`)
여러 개의 서브플롯을 한 Figure에 그릴 때, 때로는 제목이나 축 레이블이 겹치거나 플롯 간의 간격이 너무 좁아 가독성이 떨어지는 문제가 발생할 수 있습니다. `plt.tight_layout()` 함수는 이러한 문제를 자동으로 해결하여 서브플롯과 그 주변 요소들이 겹치지 않도록 최적의 간격을 조정해 줍니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 4)) # 전체 그림 크기 설정

plt.subplot(1, 2, 1) # 첫 번째 서브플롯
plt.plot(x, y1)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2) # 두 번째 서브플롯
plt.plot(x, y2, color='red')
plt.title('Cosine Wave')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout() # 서브플롯 간격 자동 조정
plt.show()
```

#### 1.3.4. 주/보조 눈금 설정 (Major/Minor Ticks)
데이터의 세밀한 변화를 더 잘 표현하기 위해 주 눈금(Major Ticks) 사이에 보조 눈금(Minor Ticks)을 추가할 수 있습니다. `matplotlib.ticker` 모듈과 함께 사용하면 눈금 간격을 정교하게 제어할 수 있습니다.

*   **`ax.minorticks_on()`**: 보조 눈금을 자동으로 활성화합니다.
*   **`ax.tick_params(which='minor', ...)`**: 보조 눈금의 스타일(방향, 길이 등)을 지정합니다.
*   **`ax.xaxis.set_minor_locator()`**: 보조 눈금의 위치를 지정하는 로케이터를 설정합니다. `MultipleLocator`를 사용하면 특정 간격으로 보조 눈금을 배치할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

x = np.linspace(0, 10, 1000)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)

# 주 눈금 간격 설정
ax.xaxis.set_major_locator(MultipleLocator(2)) # 2 단위로 주 눈금 설정
ax.yaxis.set_major_locator(MultipleLocator(0.5)) # 0.5 단위로 주 눈금 설정

# 보조 눈금 활성화 및 간격 설정
ax.minorticks_on()
ax.xaxis.set_minor_locator(MultipleLocator(0.5)) # 0.5 단위로 보조 눈금 설정
ax.yaxis.set_minor_locator(MultipleLocator(0.1)) # 0.1 단위로 보조 눈금 설정

# 눈금 스타일 지정
ax.tick_params(axis='x', which='major', length=10, width=1.5, direction='in')
ax.tick_params(axis='x', which='minor', length=5, width=1, direction='in')

ax.set_title("Major and Minor Ticks Example")
ax.grid(which='major', linestyle='-')
ax.grid(which='minor', linestyle=':')
plt.show()
```

### 1.4. 고급 커스터마이징 (Advanced Customization)
Matplotlib은 기본적인 커스터마이징 외에도 출판용(publication-quality) 그래프를 만들거나, 복잡한 시각화를 구현하기 위한 고급 커스터마이징 기능을 제공합니다.

#### 1.4.1. 스타일 시트 사용 (`plt.style.use()`)
Matplotlib은 미리 정의된 스타일 시트(stylesheet)를 제공하여 몇 줄의 코드로 그래프의 전체적인 모양을 변경할 수 있습니다. 이는 일관된 디자인을 유지하거나 특정 목적(예: 발표 자료, 논문)에 맞는 스타일을 적용할 때 유용합니다.

**주요 스타일 시트**:
*   `'default'`: Matplotlib의 기본 스타일.
*   `'ggplot'`: R의 ggplot2와 유사한 스타일.
*   `'seaborn'`, `'seaborn-v0_8'`: Seaborn 라이브러리의 기본 스타일.
*   `'dark_background'`: 어두운 배경 스타일.
*   `'bmh'`: Bayesian Methods for Hackers 스타일.
*   `'fivethirtyeight'`: FiveThirtyEight 웹사이트의 스타일.

```python
import matplotlib.pyplot as plt
import numpy as np

# 'ggplot' 스타일 시트 적용
plt.style.use('ggplot')

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label='Sine Wave')
plt.title("Sine Wave with 'ggplot' Style")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

# 스타일 초기화 (다음 플롯에 영향을 주지 않기 위해)
plt.style.use('default')
```

#### 1.4.2. `rcParams`를 이용한 전역 설정
`matplotlib.rcParams`는 Matplotlib의 모든 기본 설정을 담고 있는 딕셔너리입니다. 이 딕셔너리의 값을 변경함으로써 모든 플롯에 적용되는 전역 설정을 변경할 수 있습니다. 폰트, 글자 크기, 선 두께, 그림 크기, 해상도(DPI) 등을 일괄적으로 제어할 때 유용합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 전역 폰트 크기 및 라인 두께 설정
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

x = np.linspace(0, 10, 100)
y = np.cos(x)

plt.plot(x, y, label='Cosine Wave')
plt.title("Cosine Wave with Global rcParams Settings")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

# 설정 초기화 (선택 사항)
plt.rcParams.update(plt.rcParamsDefault)
```

#### 1.4.3. 객체 지향 API를 통한 세밀한 제어
`matplotlib.pyplot`은 편리한 인터페이스를 제공하지만, 복잡하거나 여러 개의 플롯을 다룰 때는 Matplotlib의 객체 지향(Object-Oriented) API를 직접 사용하는 것이 더 유연하고 강력합니다. `Figure`와 `Axes` 객체에 직접 접근하여 플롯의 모든 요소를 세밀하게 제어할 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Figure와 Axes 객체 생성
fig, ax = plt.subplots(figsize=(8, 6))

# Axes 객체를 사용하여 플롯 그리기
ax.plot(x, y1, label='Sine Wave', color='blue')
ax.plot(x, y2, label='Cosine Wave', color='red', linestyle='--')

# Axes 객체를 사용하여 제목, 레이블, 범례 등 설정
ax.set_title("Sine and Cosine Waves (Object-Oriented API)", fontsize=16)
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Amplitude", fontsize=12)
ax.legend(loc='upper right')
ax.grid(True)

plt.show()
```

#### 1.4.4. 텍스트 및 주석 추가
그래프에 특정 지점을 강조하거나 추가 정보를 제공하기 위해 텍스트나 주석(annotation)을 추가할 수 있습니다.

*   **`plt.text(x, y, s, **kwargs)`**: 특정 (x, y) 좌표에 텍스트를 추가합니다.
*   **`plt.annotate(s, xy, xytext, arrowprops, **kwargs)`**: 특정 데이터 포인트에 화살표와 함께 주석을 추가합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave with Annotation")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)

# 특정 지점에 텍스트 추가
plt.text(3, 0.5, 'Peak Value', fontsize=12, color='green')

# 특정 지점에 주석 추가 (화살표 포함)
plt.annotate('Local Max', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, color='red')

plt.show()
```

#### 1.4.5. 축 테두리 (Spines) 커스터마이징
Matplotlib에서 플롯의 'spines'는 데이터 영역을 둘러싸는 선(축 테두리)을 의미합니다. 기본적으로 상하좌우 네 개의 스파인이 표시되지만, 이를 제거하거나 위치를 변경하여 플롯의 미적 품질을 높이고 데이터에 더 집중하게 만들 수 있습니다. 주로 객체 지향 API를 통해 `Axes` 객체의 `spines` 속성에 접근하여 제어합니다.

*   **스파인 제거:** 불필요한 상단(`top`)과 오른쪽(`right`) 스파인을 제거하여 플롯을 더 깔끔하게 만들 수 있습니다.
*   **스파인 위치 이동:** x축과 y축 스파인을 데이터 영역의 중앙이나 특정 위치로 이동시킬 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
c, s = np.cos(x), np.sin(x)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, c, color="blue", linewidth=2.5, linestyle="-", label="Cosine")
ax.plot(x, s, color="red", linewidth=2.5, linestyle="-", label="Sine")

ax.set_xlim(x.min() * 1.1, x.max() * 1.1)
ax.set_ylim(c.min() * 1.1, c.max() * 1.1)

# 상단과 오른쪽 스파인 제거
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 하단과 왼쪽 스파인 위치 이동 (데이터 영역의 0 지점으로)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# 눈금 레이블 위치 조정
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.title("Customized Spines Example", fontsize=16)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
```

#### 1.4.6. 축에 수직/수평선 추가 (axvline/axhline)
그래프에 특정 값이나 임계점을 나타내는 수직선 또는 수평선을 추가하여 특정 데이터 포인트를 강조하거나 기준을 명시할 수 있습니다.

*   **`plt.axvline()`**: 수직선을 추가합니다.
*   **`plt.axhline()`**: 수평선을 추가합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("axvline and axhline Example")

# x=5 위치에 빨간색 점선 수직선 추가
plt.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Vertical Line at x=5')

# y=0.5 위치에 초록색 실선 수평선 추가
plt.axhline(y=0.5, color='green', linestyle='-', linewidth=2, label='Horizontal Line at y=0.5')

plt.legend()
plt.show()
```
