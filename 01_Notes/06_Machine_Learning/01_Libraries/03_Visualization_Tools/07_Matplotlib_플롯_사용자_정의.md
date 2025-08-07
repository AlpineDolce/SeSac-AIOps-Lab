<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib 플롯의 다양한 요소를 커스터마이징하는 방법을 다룹니다. 제목, 축 레이블, 범례 설정부터 색상, 마커, 선 스타일 변경, 축 범위 및 눈금 설정, 그리고 고급 커스터마이징 기법까지 학습하여 시각화의 가독성과 미적 품질을 향상시키는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. 플롯 사용자 정의](#1-플롯-사용자-정의)
  - [1.1. 제목, 축 레이블, 범례](#11-제목-축-레이블-범례)
  - [1.2. 색상, 마커, 선 스타일](#12-색상-마커-선-스타일)
  - [1.3. 축 범위 및 눈금 설정](#13-축-범위-및-눈금-설정)
  - [1.4. 고급 커스터마이징 (Advanced Customization)](#14-고급-커스터마이징-advanced-customization)

---

## 1. 플롯 사용자 정의
Matplotlib은 플롯의 다양한 요소를 커스터마이징할 수 있는 기능을 제공하여 시각화의 가독성과 미적 품질을 향상시킬 수 있습니다.

### 1.1. 제목, 축 레이블, 범례
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

### 1.2. 색상, 마커, 선 스타일
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

### 1.3. 축 범위 및 눈금 설정
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

### 1.4. 고급 커스터마이징 (Advanced Customization)

논문이나 공식 보고서에 사용될 출판용(publication-quality) 그래프를 만들기 위해서는 세밀한 제어가 필요합니다. `plt.style.use('seaborn-paper')`와 같은 스타일 시트를 사용하거나, `plt.rcParams` 딕셔너리를 직접 수정하여 전역 폰트, 글자 크기, 해상도(DPI) 등을 설정할 수 있습니다. 또한, 플롯을 구성하는 개별 Artist 객체(예: `Line2D`, `Patch`, `Text`)에 직접 접근하여 속성을 변경하는 방법을 다루면 좋습니다.
