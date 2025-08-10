<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 `pyplot` 모듈을 사용하여 등고선 플롯(Contour Plot), 박스 플롯(Box Plot), 파이 차트(Pie Chart)를 그리는 방법을 다룹니다. 각 플롯의 용도와 함께 데이터 생성부터 그래프 출력까지의 과정을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 기본 플로팅](#1-기본-플로팅)
  - [1.1. 등고선 플롯 (Contour Plot)](#11-등고선-플롯-contour-plot)
  - [1.2. 박스 플롯 (Box Plot)](#12-박스-플롯-box-plot)
  - [1.3. 파이 차트 (Pie Chart)](#13-파이-차트-pie-chart)

---

## 1. 기본 플로팅

### 1.1. 등고선 플롯 (Contour Plot)
등고선 플롯은 3차원 데이터를 2차원 평면에 표현하는 방법입니다. 동일한 값을 갖는 지점들을 선으로 연결하여 지형도처럼 표현하며, 두 변수에 따른 제3의 변수(높이)의 변화를 시각화하는 데 매우 유용합니다. 데이터 과학에서는 두 변수의 확률 밀도 함수나 머신러닝 모델의 결정 경계(Decision Boundary)를 시각화하는 데 자주 사용됩니다.

- **`plt.contour()`**: 등고선(선)을 그립니다.
- **`plt.contourf()`**: 등고선 사이의 영역을 색으로 채웁니다. (`f`는 'filled'를 의미)

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.linspace(-3.0, 3.0, 100)
y = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.exp(-(X**2 + Y**2)) * np.sin(X) * np.cos(Y)

# 등고선 플롯 그리기
plt.figure(figsize=(12, 5))

# 등고선 (선)
plt.subplot(1, 2, 1)
contour = plt.contour(X, Y, Z, 10, colors='black') # 10개의 레벨로 등고선
plt.clabel(contour, inline=True, fontsize=8) # 등고선에 값 표시
plt.title('Contour Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 채워진 등고선
plt.subplot(1, 2, 2)
contourf = plt.contourf(X, Y, Z, 20, cmap='viridis') # 20개의 레벨, viridis 컬러맵
plt.colorbar(label='Value') # 컬러바 추가
plt.title('Filled Contour Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.tight_layout()
plt.show()
```

### 1.2. 박스 플롯 (Box Plot)
박스 플롯(상자 수염 그림)은 데이터의 분포를 사분위수를 이용하여 시각화하는 데 매우 효과적인 도구입니다. 데이터의 중앙값(median), 25% 지점(Q1), 75% 지점(Q3), 그리고 이상치(outlier)를 한눈에 보여주어 여러 그룹 간의 데이터 분포를 비교하는 데 널리 사용됩니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (세 그룹의 데이터)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(1, 1.5, 100)
data3 = np.random.normal(-1, 0.5, 100)
data = [data1, data2, data3]

# 박스 플롯 그리기
plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'])
plt.title('Box Plot of Three Groups')
plt.xlabel('Group')
plt.ylabel('Value')
plt.grid(True)
plt.show()
```

### 1.3. 파이 차트 (Pie Chart)
파이 차트는 전체에 대한 각 부분의 비율을 부채꼴의 면적으로 나타내는 그래프입니다. 데이터의 구성 비율을 직관적으로 보여줄 때 유용하지만, 항목이 너무 많거나 비율 차이가 미미할 경우 해석이 어려울 수 있어 사용에 주의가 필요합니다.

```python
import matplotlib.pyplot as plt

# 데이터 생성
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10] # 각 항목의 비율
explode = (0, 0.1, 0, 0)  # 'B' 항목을 약간 돋보이게 함

# 파이 차트 그리기
plt.figure(figsize=(7, 7))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Pie Chart of Proportions')
plt.axis('equal')  # 파이 차트를 원형으로 유지
plt.show()
```
