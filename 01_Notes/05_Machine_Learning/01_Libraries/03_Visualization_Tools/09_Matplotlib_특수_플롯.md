<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 특수 플롯 기능인 이미지 시각화(`imshow()`)와 3D 시각화(`mplot3d`)를 다룹니다. 2D 배열 데이터를 이미지 형태로 표현하거나, 3차원 공간에 데이터를 시각화하여 다차원 데이터의 구조를 입체적으로 파악하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 특수 플롯 (Specialized Plots)](#1-특수-플롯-specialized-plots)
  - [1.1. 이미지 시각화 (imshow)](#11-이미지-시각화-imshow)
  - [1.2. 3D 시각화 (mplot3d)](#12-3d-시각화-mplot3d)

---

## 1. 특수 플롯 (Specialized Plots)

### 1.1. 이미지 시각화 (imshow)
`imshow()` 함수는 2D 배열이나 행렬 데이터를 이미지 형태로 시각화합니다. 머신러닝에서는 모델의 **혼동 행렬(Confusion Matrix)**이나 이미지 데이터 자체를 시각화하는 데 매우 유용하게 사용됩니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 5x5 랜덤 행렬 데이터 생성
matrix = np.random.rand(5, 5)

plt.figure(figsize=(6, 5))
plt.imshow(matrix, cmap='viridis') # viridis 컬러맵 사용
plt.colorbar(label='Value')
plt.title('Image Show (imshow) of a Matrix')

# 각 셀에 값 표시
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white')

plt.show()
```

### 1.2. 3D 시각화 (mplot3d)
Matplotlib의 `mplot3d` 툴킷을 사용하면 3차원 공간에 데이터를 시각화할 수 있습니다. 3D 산점도, 곡면, 와이어프레임 등 다양한 3D 플롯을 지원하여 다차원 데이터의 구조를 입체적으로 파악하는 데 도움을 줍니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 3D Axes 객체 생성
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 3D 산점도 데이터 생성
z = np.linspace(0, 1, 100)
x = z * np.sin(20 * z)
y = z * np.cos(20 * z)

# 3D 산점도 그리기
ax.scatter3D(x, y, z, c=z, cmap='Blues')

ax.set_title('3D Scatter Plot')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()
```
