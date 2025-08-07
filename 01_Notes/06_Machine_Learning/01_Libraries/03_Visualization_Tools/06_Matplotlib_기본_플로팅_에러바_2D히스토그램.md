<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 `pyplot` 모듈을 사용하여 에러 바(Error Bars), 2D 히스토그램(2D Histogram), 그리고 헥사곤 비닝(Hexbin) 플롯을 그리는 방법을 다룹니다. 각 플롯의 용도와 함께 데이터 생성부터 그래프 출력까지의 과정을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 기본 플로팅](#1-기본-플로팅)
  - [1.1. 에러 바 (Error Bars)](#11-에러-바-error-bars)
  - [1.2. 2D 히스토그램 및 헥사곤 비닝 (2D Histogram & Hexbin)](#12-2d-히스토그램-및-헥사곤-비닝-2d-histogram--hexbin)

---

## 1. 기본 플로팅

### 1.1. 에러 바 (Error Bars)
에러 바는 데이터 포인트의 측정값에 대한 불확실성이나 오차 범위를 시각적으로 표현합니다. 주로 평균값과 함께 표준 편차, 표준 오차, 신뢰 구간 등을 나타내는 데 사용되며, 데이터의 신뢰도를 평가하는 데 중요한 정보를 제공합니다.

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
plt.show()
```

### 1.2. 2D 히스토그램 및 헥사곤 비닝 (2D Histogram & Hexbin)
데이터 포인트가 매우 많아 산점도에서 점들이 서로 겹쳐 분포를 파악하기 어려운 과밀 플롯(overplotting) 문제가 발생할 때, 2D 히스토그램이나 헥사곤 비닝을 사용하면 유용합니다. 이들은 2차원 공간을 사각형 또는 육각형으로 나누고 각 영역에 포함된 데이터 포인트의 개수를 색상으로 표현하여 데이터의 밀도를 효과적으로 시각화합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성 (다변량 정규 분포)
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T

plt.figure(figsize=(12, 5))

# 2D 히스토그램
plt.subplot(1, 2, 1)
plt.hist2d(x, y, bins=30, cmap='Blues')
plt.colorbar(label='Count in bin')
plt.title('2D Histogram')
plt.xlabel('X-variable')
plt.ylabel('Y-variable')

# 헥사곤 비닝 플롯
plt.subplot(1, 2, 2)
plt.hexbin(x, y, gridsize=30, cmap='inferno')
plt.colorbar(label='Count in bin')
plt.title('Hexbin Plot')
plt.xlabel('X-variable')
plt.ylabel('Y-variable')

plt.tight_layout()
plt.show()
```
