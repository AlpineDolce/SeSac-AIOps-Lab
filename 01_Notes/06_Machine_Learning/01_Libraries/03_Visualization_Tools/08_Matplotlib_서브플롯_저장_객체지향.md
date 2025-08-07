<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib에서 여러 개의 플롯을 하나의 그림 안에 배치하는 서브플롯(Subplots) 생성 방법, 생성된 플롯을 이미지 파일로 저장하는 방법, 그리고 실무에서 권장되는 객체 지향(Object-Oriented) API 사용법을 다룹니다. 이를 통해 복잡한 시각화 작업을 효율적으로 관리하고 재사용성을 높이는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. 서브플롯 (Subplots)](#1-서브플롯-subplots)
- [2. 플롯 저장](#2-플롯-저장)
- [3. 객체 지향 API (Object-Oriented API)](#3-객체-지향-api-object-oriented-api)

---

## 1. 서브플롯 (Subplots)
여러 개의 플롯을 하나의 그림(Figure) 안에 배치하여 데이터를 다양한 관점에서 비교하거나 관련성 있는 정보를 함께 보여줄 때 사용합니다. `plt.subplot()` 또는 객체 지향 방식의 `plt.subplots()`를 사용합니다.

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

plt.tight_layout() # 서브플롯 간의 간격 자동 조절
plt.show()
```

## 2. 플롯 저장
생성된 플롯을 이미지 파일로 저장하여 문서나 프레젠테이션에 활용할 수 있습니다. `plt.savefig()` 함수를 사용합니다.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Plot to Save")
plt.xlabel("X")
plt.ylabel("Y")

# PNG 파일로 저장
plt.savefig("sine_wave.png")
print(''''sine_wave.png''' 파일이 저장되었습니다.')

# PDF 파일로 저장 (고품질 벡터 이미지)
plt.savefig("sine_wave.pdf")
print(''''sine_wave.pdf''' 파일이 저장되었습니다.')

# 해상도(dpi) 설정하여 저장
plt.savefig("sine_wave_high_res.png", dpi=300)
print(''''sine_wave_high_res.png''' 파일이 고해상도로 저장되었습니다.')

plt.show()
```

## 3. 객체 지향 API (Object-Oriented API)
Matplotlib에는 두 가지 사용 방식이 있습니다. 하나는 지금까지 주로 사용한 `pyplot` 상태 머신(state-machine) 인터페이스이고, 다른 하나는 실무에서 더 권장되는 **객체 지향(Object-Oriented)** 인터페이스입니다.

- **Pyplot 인터페이스**: `plt.plot()`, `plt.title()`처럼 현재 활성화된 Figure나 Axes에 암묵적으로 작동합니다. 코드가 간결하여 간단한 플롯에 적합합니다.
- **객체 지향 인터페이스**: `fig, ax = plt.subplots()`로 Figure(그림 전체)와 Axes(개별 플롯) 객체를 명시적으로 생성하고, `ax.plot()`, `ax.set_title()`처럼 각 객체의 메서드를 호출하여 제어합니다. 복잡한 플롯이나 여러 개의 서브플롯을 다룰 때 훨씬 더 유연하고 코드의 가독성과 재사용성이 높습니다.

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
