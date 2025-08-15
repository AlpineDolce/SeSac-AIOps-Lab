<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib에서 여러 개의 플롯을 하나의 그림 안에 배치하는 서브플롯(Subplots) 생성 방법, 생성된 플롯을 이미지 파일로 저장하는 방법, 그리고 실무에서 권장되는 객체 지향(Object-Oriented) API 사용법을 다룹니다. 이를 통해 복잡한 시각화 작업을 효율적으로 관리하고 재사용성을 높이는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Matplotlib 서브플롯 (Subplots)](#1-matplotlib-서브플롯-subplots)
  - [1.1. 서브플롯의 개념 및 필요성](#11-서브플롯의-개념-및-필요성)
  - [1.2. `plt.subplots()`를 이용한 서브플롯 생성](#12-plt-subplots-를-이용한-서브플롯-생성)
  - [1.3. 서브플롯에 개별 플롯 그리기 및 커스터마이징](#13-서브플롯에-개별-플롯-그리기-및-커스터마이징)
  - [1.4. `plt.tight_layout()`: 서브플롯 간 간격 조절](#14-plt-tight-layout-서브플롯-간-간격-조절)
- [2. 플롯 저장 (`plt.savefig()`)](#2-플롯-저장-plt-savefig)
  - [2.1. 기본 저장 방법](#21-기본-저장-방법)
  - [2.2. 다양한 파일 형식으로 저장](#22-다양한-파일-형식으로-저장)
  - [2.3. 해상도(DPI) 및 여백 설정](#23-해상도-dpi-및-여백-설정)
- [3. 객체 지향 API (Object-Oriented API) 활용](#3-객체-지향-api-object-oriented-api-활용)
  - [3.1. Pyplot 인터페이스 vs. 객체 지향 인터페이스](#31-pyplot-인터페이스-vs-객체-지향-인터페이스)
  - [3.2. Figure와 Axes 객체 이해](#32-figure와-axes-객체-이해)
  - [3.3. 객체 지향 방식으로 플롯 그리기 및 커스터마이징](#33-객체-지향-방식으로-플롯-그리기-및-커스터마이징)
  - [3.4. 왜 객체 지향 API를 사용해야 하는가?](#34-왜-객체-지향-api를-사용해야-하는가)

---

## 1. Matplotlib 서브플롯 (Subplots)
서브플롯(Subplots)은 Matplotlib에서 여러 개의 플롯을 하나의 그림(Figure) 안에 배치하는 기능입니다. 이를 통해 데이터를 다양한 관점에서 비교하거나, 관련성 있는 정보를 함께 보여주어 시각화의 효율성과 정보 전달력을 높일 수 있습니다.

### 1.1. 서브플롯의 개념 및 필요성
*   **개념**: 하나의 Figure(전체 그림 영역) 내에 여러 개의 Axes(개별 플롯 영역)를 생성하여 각각의 Axes에 독립적인 그래프를 그리는 것을 의미합니다.
*   **필요성**:
    *   **비교 분석**: 여러 변수 간의 관계나 동일 변수의 다른 조건에서의 변화를 한눈에 비교할 수 있습니다.
    *   **공간 효율성**: 여러 그래프를 하나의 이미지로 묶어 공간을 효율적으로 사용하고, 보고서나 프레젠테이션에 포함하기 용이합니다.
    *   **스토리텔링**: 데이터 분석의 흐름이나 결론을 여러 단계의 시각화를 통해 논리적으로 전달할 수 있습니다.

### 1.2. `plt.subplots()`를 이용한 서브플롯 생성
Matplotlib에서 서브플롯을 생성하는 가장 권장되는 방법은 `plt.subplots()` 함수를 사용하는 것입니다. 이 함수는 Figure 객체와 Axes 객체(또는 Axes 객체들의 배열)를 동시에 반환합니다.

**`plt.subplots(nrows, ncols, figsize=(width, height))`**:
*   `nrows`: 생성할 서브플롯의 행(row) 개수.
*   `ncols`: 생성할 서브플롯의 열(column) 개수.
*   `figsize`: 전체 Figure의 크기를 인치 단위로 지정합니다 (선택 사항).

**반환 값**:
*   `fig`: Figure 객체. 전체 그림 영역을 나타냅니다.
*   `axes`: Axes 객체 또는 Axes 객체들의 NumPy 배열. 각 서브플롯 영역을 나타냅니다. `nrows`나 `ncols`가 1인 경우 단일 Axes 객체가 반환될 수 있습니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 2x2 그리드에 4개의 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # 2행 2열, 그림 크기 설정

# axes는 2x2 NumPy 배열 형태의 Axes 객체들을 담고 있습니다.
print(type(fig))
print(type(axes))
print(axes.shape) # (2, 2)
```

### 1.3. 서브플롯에 개별 플롯 그리기 및 커스터마이징
`plt.subplots()`를 통해 얻은 `axes` 객체를 사용하여 각 서브플롯에 독립적인 그래프를 그리고 커스터마이징할 수 있습니다. 각 `axes` 객체는 `plt.plot()`, `plt.title()`, `plt.xlabel()`, `plt.ylabel()` 등 `pyplot` 함수의 객체 지향 버전 메서드를 가집니다 (예: `ax.plot()`, `ax.set_title()`).

**코드 설명**:
*   `axes[0, 0]`: 2x2 그리드의 첫 번째 행, 첫 번째 열(좌상단)에 해당하는 Axes 객체입니다.
*   `axes[0, 0].plot(x, y1, color='blue')`: 해당 Axes 객체에 라인 플롯을 그립니다.
*   `axes[0, 0].set_title("Sine Wave")`: 해당 Axes 객체의 제목을 설정합니다. `plt.title()` 대신 `ax.set_title()`를 사용합니다.
*   `axes[0, 0].set_xlabel("X")`, `axes[0, 0].set_ylabel("Y")`: 해당 Axes 객체의 축 레이블을 설정합니다. `plt.xlabel()` 대신 `ax.set_xlabel()`를 사용합니다.

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
```

### 1.4. `plt.tight_layout()`: 서브플롯 간 간격 조절
`plt.tight_layout()` 함수는 서브플롯들이 서로 겹치지 않도록 자동으로 간격을 조절해줍니다. 제목, 축 레이블, 눈금 레이블 등이 잘리지 않고 깔끔하게 보이도록 할 때 유용합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (이전과 동일)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = x
y4 = x**2

# 2x2 그리드에 4개의 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 각 서브플롯에 플롯 그리기 (이전과 동일)
axes[0, 0].plot(x, y1, color='blue')
axes[0, 0].set_title("Sine Wave")
axes[0, 0].set_xlabel("X")
axes[0, 0].set_ylabel("Y")

axes[0, 1].plot(x, y2, color='red')
axes[0, 1].set_title("Cosine Wave")
axes[0, 1].set_xlabel("X")
axes[0, 1].set_ylabel("Y")

axes[1, 0].plot(x, y3, color='green')
axes[1, 0].set_title("Linear")
axes[1, 0].set_xlabel("X")
axes[1, 0].set_ylabel("Y")

axes[1, 1].plot(x, y4, color='purple')
axes[1, 1].set_title("Quadratic")
axes[1, 1].set_xlabel("X")
axes[1, 1].set_ylabel("Y")

plt.tight_layout() # 서브플롯 간의 간격 자동 조절
plt.show()
```

## 2. 플롯 저장 (plt.savefig())

`plt.savefig()` 함수는 Matplotlib에서 생성된 플롯(Figure)을 다양한 이미지 파일 형식으로 저장하는 데 사용됩니다. 현재 활성화된 Figure를 저장하며, 객체 지향 방식에서는 특정 Figure 객체의 `savefig` 메서드(`fig.savefig()`)를 사용하는 것이 더 명확하고 권장됩니다.

### `plt.savefig()` 주요 매개변수

`plt.savefig(fname, dpi=None, format=None, bbox_inches=None, pad_inches=0.1, transparent=False, facecolor='auto', edgecolor='auto', **kwargs)`

---

### 2.1. 기본 저장 방법

가장 기본적인 사용법은 저장할 파일의 이름과 확장자를 지정하는 것입니다. Matplotlib은 파일 확장자를 기반으로 저장 형식을 자동으로 결정합니다.

*   `fname`: 저장할 파일의 이름 또는 경로입니다. (예: `'my_plot.png'`, `'path/to/plot.pdf'`).

```python
import matplotlib.pyplot as plt
import numpy as np

# 기본 저장 예시
plt.figure(figsize=(6, 4))
plt.plot([0, 1, 2], [0, 1, 4])
plt.title("Basic Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.savefig('basic_plot.png')
print("basic_plot.png 저장 완료")

# 추가 기본 저장 예시 (이전 내용에서 가져옴)
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(6, 4)) # 새로운 figure 생성
plt.plot(x, y)
plt.title("Plot to Save")
plt.xlabel("X")
plt.ylabel("Y")

# PNG 파일로 저장
plt.savefig("sine_wave.png")
print('\'sine_wave.png\' 파일이 저장되었습니다.')
```

---

### 2.2. 다양한 파일 형식으로 저장

Matplotlib은 PNG, PDF, SVG, JPG 등 다양한 이미지 파일 형식을 지원합니다. `format` 매개변수를 사용하여 명시적으로 파일 형식을 지정할 수 있습니다. `fname`의 확장자와 `format` 값이 다를 경우 `format`이 우선합니다.

*   `format`: 저장할 파일의 형식을 명시적으로 지정합니다 (예: `'png'`, `'pdf'`, `'svg'`, `'jpg'`).

```python
import matplotlib.pyplot as plt
import numpy as np

# PDF 형식으로 저장 예시 (객체 지향 방식)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([0, 1, 2], [2, 1, 0], color='red')
ax.set_title("Object-Oriented Plot (PDF)")
fig.savefig('oo_plot.pdf', format='pdf') # PDF 형식으로 저장
print("oo_plot.pdf 저장 완료")

# 추가 PDF 저장 예시 (이전 내용에서 가져옴)
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(6, 4)) # 새로운 figure 생성
plt.plot(x, y)
plt.title("Another Plot to Save")
plt.xlabel("X")
plt.ylabel("Y")

# PDF 파일로 저장 (고품질 벡터 이미지)
plt.savefig("sine_wave.pdf")
print('\'sine_wave.pdf\' 파일이 저장되었습니다.')
```

---

### 2.3. 해상도(DPI) 및 여백 설정

저장되는 이미지의 품질과 여백을 조절할 수 있습니다.

*   `dpi`: 해상도(dots per inch)를 설정합니다. 값이 높을수록 고화질의 이미지가 생성됩니다. 인쇄나 출판용으로는 300 이상의 DPI를 권장합니다.
*   `bbox_inches`: 저장할 그림의 경계 상자를 지정합니다. `'tight'`로 설정하면 플롯 주변의 불필요한 여백을 자동으로 제거하여 그림이 잘리지 않고 깔끔하게 저장됩니다.
*   `pad_inches`: `bbox_inches='tight'`를 사용할 때 플롯 주변에 추가할 여백(인치 단위)을 설정합니다. 기본값은 0.1입니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 고해상도 및 여백 제거 저장 예시
plt.figure(figsize=(6, 4))
plt.scatter(np.random.rand(50), np.random.rand(50))
plt.title("Scatter Plot (High DPI, Tight Layout)")
plt.savefig('scatter_plot_high_res_tight.png', dpi=300, bbox_inches='tight')
print("scatter_plot_high_res_tight.png 저장 완료")

# 추가 해상도(dpi) 설정 저장 예시 (이전 내용에서 가져옴)
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(6, 4)) # 새로운 figure 생성
plt.plot(x, y)
plt.title("High Res Plot")
plt.xlabel("X")
plt.ylabel("Y")

# 해상도(dpi) 설정하여 저장
plt.savefig("sine_wave_high_res.png", dpi=300)
print('\'sine_wave_high_res.png\' 파일이 고해상도로 저장되었습니다.')
```

---

### 기타 고급 설정

*   `transparent`: 배경을 투명하게 저장할지 여부를 설정합니다 (`True` 또는 `False`). PNG와 같은 형식에서 유용합니다.
*   `facecolor`, `edgecolor`: 그림의 배경색과 테두리색을 설정합니다. 기본값은 `'auto'`입니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 투명 배경 저장 예시
plt.figure(figsize=(6, 4))
plt.bar(['A', 'B', 'C'], [10, 20, 15])
plt.title("Bar Plot (Transparent Background)")
plt.savefig('bar_plot_transparent.png', transparent=True)
print("bar_plot_transparent.png 저장 완료")

plt.close('all') # 모든 플롯 창 닫기
```


## 3. 객체 지향 API (Object-Oriented API) 활용
Matplotlib에는 두 가지 주요 사용 방식이 있습니다. 하나는 `pyplot` 모듈을 통해 제공되는 **상태 머신(state-machine) 인터페이스**이고, 다른 하나는 **객체 지향(Object-Oriented) 인터페이스**입니다. 실무에서는 복잡한 시각화나 여러 개의 플롯을 다룰 때 객체 지향 API 사용이 강력히 권장됩니다.

### 3.1. Pyplot 인터페이스 vs. 객체 지향 인터페이스
*   **Pyplot 인터페이스**: `plt.plot()`, `plt.title()`와 같이 `matplotlib.pyplot` 모듈의 함수들을 직접 호출하는 방식입니다. 현재 활성화된 Figure나 Axes에 암묵적으로 작동합니다. 코드가 간결하여 간단한 플롯을 빠르게 그릴 때 편리합니다.
*   **객체 지향 인터페이스**: `fig, ax = plt.subplots()`와 같이 Figure(그림 전체)와 Axes(개별 플롯) 객체를 명시적으로 생성하고, 이 객체들의 메서드(예: `ax.plot()`, `ax.set_title()`)를 호출하여 제어하는 방식입니다.

### 3.2. Figure와 Axes 객체 이해
Matplotlib의 객체 지향 API를 이해하는 핵심은 `Figure`와 `Axes` 객체의 개념을 파악하는 것입니다.
*   **`Figure` 객체**: 전체 그림 영역을 나타냅니다. 종이 한 장이라고 생각할 수 있습니다. 이 `Figure` 안에 하나 이상의 `Axes` 객체가 포함될 수 있습니다. `plt.figure()` 함수로 생성하거나, `plt.subplots()` 함수를 통해 `Axes` 객체와 함께 생성됩니다.
*   **`Axes` 객체**: 실제 데이터가 그려지는 개별 플롯 영역을 나타냅니다. `Figure` 안에 여러 개의 `Axes`가 있을 수 있으며, 각 `Axes`는 독립적인 x축, y축, 제목, 범례 등을 가집니다. `ax.plot()`, `ax.set_title()` 등 대부분의 플로팅 및 커스터마이징 메서드는 이 `Axes` 객체에 속합니다.

### 3.3. 객체 지향 방식으로 플롯 그리기 및 커스터마이징
`plt.subplots()` 함수를 사용하여 `Figure`와 `Axes` 객체를 생성한 후, `Axes` 객체의 메서드를 호출하여 플롯을 그립니다.

**코드 설명**:
1.  **`fig, ax = plt.subplots(figsize=(8, 5))`**: `plt.subplots()`를 호출하여 하나의 Figure(`fig`)와 하나의 Axes(`ax`) 객체를 생성합니다. `figsize`는 Figure의 크기를 설정합니다.
2.  **`ax.plot(x, y, label='Sine Wave', color='coral')`**: `ax` 객체의 `plot()` 메서드를 사용하여 라인 플롯을 그립니다. `pyplot` 방식의 `plt.plot()`과 동일한 파라미터를 사용합니다.
3.  **`ax.set_title('Object-Oriented Plotting')`**: `ax` 객체의 `set_title()` 메서드를 사용하여 플롯의 제목을 설정합니다. `pyplot` 방식의 `plt.title()` 대신 `set_title()`을 사용합니다.
4.  **`ax.set_xlabel('X-axis')`, `ax.set_ylabel('Y-axis')`**: `ax` 객체의 `set_xlabel()`과 `set_ylabel()` 메서드를 사용하여 축 레이블을 설정합니다.
5.  **`ax.legend()`**: `ax` 객체의 `legend()` 메서드를 사용하여 범례를 표시합니다.
6.  **`ax.grid(True)`**: `ax` 객체의 `grid()` 메서드를 사용하여 그리드를 표시합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

# 객체 지향 방식으로 플롯 그리기
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, y, label='Sine Wave', color='coral')
ax.set_title('Object-Oriented Plotting')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()
ax.grid(True)

plt.show()
```

### 3.4. 왜 객체 지향 API를 사용해야 하는가?
*   **명확성 및 제어**: 어떤 플롯(Axes)에 어떤 요소를 추가하는지 명시적으로 지정하므로 코드가 더 명확해지고, 복잡한 Figure에서 특정 Axes를 정확히 제어할 수 있습니다.
*   **유연성**: 여러 개의 서브플롯을 다루거나, Figure와 Axes의 속성을 개별적으로 조작할 때 훨씬 더 유연합니다.
*   **재사용성**: Figure와 Axes 객체를 함수나 클래스에 전달하여 재사용 가능한 플로팅 함수를 만들 수 있습니다.
*   **복잡한 레이아웃**: `GridSpec`이나 `add_axes()`와 같은 고급 기능을 사용하여 복잡한 서브플롯 레이아웃을 만들 때 객체 지향 API가 필수적입니다.
*   **상태 관리**: `pyplot` 인터페이스는 내부적으로 "현재 Figure"와 "현재 Axes"라는 상태를 관리하므로, 여러 플롯을 연속적으로 그릴 때 의도치 않은 결과가 발생할 수 있습니다. 객체 지향 API는 이러한 상태 관리에 대한 의존성을 줄여줍니다.

따라서 간단한 일회성 플롯이 아니라면, Matplotlib을 사용할 때는 객체 지향 API를 사용하는 것을 습관화하는 것이 좋습니다.
