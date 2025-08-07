<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 `pyplot` 모듈을 사용하여 가장 기본적인 라인 플롯(Line Plot)과 산점도(Scatter Plot)를 그리는 방법을 다룹니다. 각 플롯의 용도와 함께 데이터 생성부터 그래프 출력까지의 과정을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 기본 플로팅](#1-기본-플로팅)
  - [1.1. 라인 플롯 (Line Plot)](#11-라인-플롯-line-plot)
  - [1.2. 산점도 (Scatter Plot)](#12-산점도-scatter-plot)

---

## 1. 기본 플로팅
Matplotlib의 `pyplot` 모듈은 다양한 종류의 그래프를 그릴 수 있는 함수를 제공합니다.

### 1.1. 라인 플롯 (Line Plot)
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

### 1.2. 산점도 (Scatter Plot)
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
