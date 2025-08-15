<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib의 특수 플롯 기능인 이미지 시각화(`imshow()`)와 3D 시각화(`mplot3d`)를 다룹니다. 2D 배열 데이터를 이미지 형태로 표현하거나, 3차원 공간에 데이터를 시각화하여 다차원 데이터의 구조를 입체적으로 파악하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. Matplotlib 특수 플롯 개요](#1-matplotlib-특수-플롯-개요)
  - [1.1. 2D 배열 데이터 시각화: `imshow()`](#11-2d-배열-데이터-시각화-imshow)
    - [1.1.1. `imshow()` 기본 사용법](#111-imshow-기본-사용법)
    - [1.1.2. 주요 매개변수 (cmap, interpolation 등)](#112-주요-매개변수-cmap-interpolation-등)
    - [1.1.3. 활용 예시 (혼동 행렬, 이미지 데이터)](#113-활용-예시-혼동-행렬-이미지-데이터)
  - [1.2. 3차원 데이터 시각화: `mplot3d`](#12-3차원-데이터-시각화-mplot3d)
    - [1.2.1. 3D Axes 생성](#121-3d-axes-생성)
    - [1.2.2. 3D 산점도 (`scatter3D`)](#122-3d-산점도-scatter3d)
    - [1.2.3. 3D 곡면/와이어프레임 플롯 (`plot_surface`, `plot_wireframe`)](#123-3d-곡면-와이어프레임-플롯-plot_surface-plot_wireframe)
    - [1.2.4. 3D 막대 플롯 (`plot_2d_from_3d_data`)](#124-3d-막대-플롯-plot_2d_from_3d_data)

---

## 1. Matplotlib 특수 플롯 개요

Matplotlib은 기본적인 2D 플롯 외에도 이미지 데이터 시각화, 3차원 데이터 시각화 등 다양한 특수 플롯 기능을 제공합니다. 이러한 기능들은 특정 형태의 데이터를 효과적으로 탐색하고 분석하는 데 필수적입니다.

### 1.1. 2D 배열 데이터 시각화: `imshow()`

`imshow()` 함수는 2D 배열이나 행렬 데이터를 이미지 형태로 시각화하는 데 사용됩니다. 이는 이미지 처리, 머신러닝 모델의 혼동 행렬 시각화, 히트맵 등 다양한 분야에서 활용됩니다.

#### 1.1.1. `imshow()` 기본 사용법

`imshow()`는 입력으로 2D 배열을 받아 각 셀의 값을 색상으로 매핑하여 이미지를 생성합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 5x5 랜덤 행렬 데이터 생성 (기존 예시)
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

# 간단한 흑백 이미지 시각화
gray_image = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 0, 1, 0],
                       [0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0]])
plt.figure(figsize=(4, 4))
plt.imshow(gray_image, cmap='gray')
plt.title('Simple Grayscale Image')
plt.show()
```

#### 1.1.2. 주요 매개변수 (cmap, interpolation 등)

`imshow()` 함수는 다양한 매개변수를 통해 시각화의 세부적인 부분을 제어할 수 있습니다.

*   `cmap`: 컬러맵(Colormap)을 지정합니다. 데이터 값을 색상으로 매핑하는 방식을 결정합니다. (예: `'viridis'`, `'gray'`, `'hot'`, `'Blues'`)
*   `interpolation`: 픽셀을 표시하는 보간(interpolation) 방법을 지정합니다. 이미지를 확대/축소할 때 픽셀 간의 색상을 어떻게 채울지 결정합니다. (예: `'nearest'`, `'bilinear'`, `'bicubic'`)
*   `aspect`: 이미지의 가로세로 비율을 설정합니다. (예: `'auto'`, `'equal'`)
*   `vmin`, `vmax`: 컬러맵에 매핑될 데이터 값의 최소 및 최대 범위를 지정합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
data = np.random.rand(10, 10) * 100

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar(label='Value')
plt.title('Colormap: hot, Interpolation: nearest')

plt.subplot(1, 2, 2)
plt.imshow(data, cmap='Blues', interpolation='bilinear', vmin=20, vmax=80)
plt.colorbar(label='Value')
plt.title('Colormap: Blues, Interpolation: bilinear (vmin=20, vmax=80)')

plt.tight_layout()
plt.show()
```

#### 1.1.3. 활용 예시 (혼동 행렬, 이미지 데이터)

`imshow()`는 머신러닝에서 모델의 성능을 평가하는 혼동 행렬(Confusion Matrix)을 시각화하거나, 실제 이미지 데이터를 표시하는 데 매우 유용합니다.

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # 혼동 행렬 시각화에 자주 사용

# 혼동 행렬 예시 데이터 (가상의 분류 결과)
confusion_matrix = np.array([[90, 5, 2],
                             [3, 85, 7],
                             [1, 8, 95]])
class_names = ['Class A', 'Class B', 'Class C']

plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Example')
plt.show()

# 실제 이미지 데이터 시각화 (가상의 흑백 이미지)
from PIL import Image # 이미지 로딩 라이브러리 (설치 필요: pip install Pillow)

# 가상의 이미지 데이터 생성 (실제로는 파일에서 로드)
# 예를 들어, 28x28 픽셀의 MNIST 숫자 이미지
mnist_like_image = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
mnist_like_image[5:10, 5:10] = 255 # 일부 영역을 밝게

plt.figure(figsize=(5, 5))
plt.imshow(mnist_like_image, cmap='gray')
plt.title('MNIST-like Image Visualization')
plt.axis('off') # 축 제거
plt.show()
```

### 1.2. 3차원 데이터 시각화: `mplot3d`

Matplotlib의 `mplot3d` 툴킷을 사용하면 3차원 공간에 데이터를 시각화할 수 있습니다. 3D 산점도, 곡면, 와이어프레임 등 다양한 3D 플롯을 지원하여 다차원 데이터의 구조를 입체적으로 파악하는 데 도움을 줍니다.

#### 1.2.1. 3D Axes 생성

3D 플롯을 그리기 위해서는 먼저 3차원 Axes 객체를 생성해야 합니다. `fig.add_subplot(projection='3d')` 또는 `plt.figure().add_subplot(projection='3d')`를 사용합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 3D Axes 객체 생성 방법 1: fig.add_subplot
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(121, projection='3d') # 1행 2열 중 첫 번째
ax1.set_title('3D Axes (Method 1)')

# 3D Axes 객체 생성 방법 2: plt.figure().add_subplot
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d') # 단일 3D Axes
ax2.set_title('3D Axes (Method 2)')

plt.show()
```

#### 1.2.2. 3D 산점도 (`scatter3D`)

`scatter3D()` 함수는 3차원 공간에 점들을 표시하여 데이터 분포를 시각화합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

# 3D Axes 객체 생성
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 3D 산점도 데이터 생성 (기존 예시)
z = np.linspace(0, 1, 100)
x = z * np.sin(20 * z)
y = z * np.cos(20 * z)

# 3D 산점도 그리기
ax.scatter3D(x, y, z, c=z, cmap='Blues') # c 매개변수로 색상 지정 가능

ax.set_title('3D Scatter Plot')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.show()

# 또 다른 3D 산점도 예시
from mpl_toolkits.mplot3d import Axes3D # 명시적 임포트

np.random.seed(42)
n_points = 50
xs = np.random.rand(n_points)
ys = np.random.rand(n_points)
zs = np.random.rand(n_points)
colors = np.random.rand(n_points) # 색상으로 사용할 값

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c=colors, cmap='viridis', s=50) # s로 마커 크기 조절

ax.set_title('Another 3D Scatter Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
```

#### 1.2.3. 3D 곡면/와이어프레임 플롯 (`plot_surface`, `plot_wireframe`)

`plot_surface()`는 3차원 곡면을, `plot_wireframe()`은 와이어프레임 형태의 곡면을 그립니다. 주로 함수 `f(x, y) = z` 형태의 데이터를 시각화할 때 사용됩니다.

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 데이터 생성 (x, y 그리드 및 z 값)
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2)) # 예시 함수: sin(sqrt(x^2 + y^2))

fig = plt.figure(figsize=(14, 7))

# 3D 곡면 플롯
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('3D Surface Plot')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 3D 와이어프레임 플롯
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_wireframe(X, Y, Z, color='blue')
ax2.set_title('3D Wireframe Plot')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

plt.tight_layout()
plt.show()
```

#### 1.2.4. 3D 막대 플롯 (`plot_2d_from_3d_data`)

`mplot3d`는 3차원 막대 플롯을 직접적으로 제공하지는 않지만, `bar3d` 함수를 통해 구현할 수 있습니다. 이는 2D 데이터의 높이를 3차원으로 표현할 때 유용합니다.

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 3D 막대 플롯 데이터 생성
xpos = np.arange(2)
ypos = np.arange(3)
xpos, ypos = np.meshgrid(xpos, ypos)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

dx = np.ones_like(xpos) * 0.5
dy = np.ones_like(ypos) * 0.5
dz = np.random.rand(len(xpos)) * 5 # 막대 높이

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='skyblue', zsort='average')

ax.set_title('3D Bar Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
```
