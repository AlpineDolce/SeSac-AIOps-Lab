<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 `pyplot` 모듈을 사용하여 막대 그래프(Bar Plot)와 히스토그램(Histogram)을 그리는 방법을 다룹니다. 각 플롯의 용도와 함께 데이터 생성부터 그래프 출력까지의 과정을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 기본 플로팅](#1-기본-플로팅)
  - [1.1. 막대 그래프 (Bar Plot)](#11-막대-그래프-bar-plot)
  - [1.2. 히스토그램 (Histogram)](#12-히스토그램-histogram)

---

## 1. 기본 플로팅

### 1.1. 막대 그래프 (Bar Plot)
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

### 1.2. 히스토그램 (Histogram)
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
