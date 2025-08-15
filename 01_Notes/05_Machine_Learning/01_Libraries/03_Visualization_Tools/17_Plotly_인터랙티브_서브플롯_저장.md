<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly의 핵심 강점인 인터랙티브 플롯 기능과 이를 활용한 데이터 탐색 방법을 다룹니다. 또한, Plotly Graph Objects를 사용하여 여러 개의 플롯을 하나의 Figure에 배치하는 서브플롯 생성 방법과 생성된 플롯을 HTML 또는 정적 이미지 파일로 저장하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 인터랙티브 플롯](#1-인터랙티브-플롯)
- [2. 서브플롯](#2-서브플롯)
- [3. 플롯 저장](#3-플롯-저장)

---

## 1. 인터랙티브 플롯
Plotly의 가장 큰 장점은 생성된 그래프가 기본적으로 인터랙티브하다는 점입니다. 마우스로 그래프를 조작하여 데이터를 더 깊이 탐색할 수 있습니다.

**주요 인터랙티브 기능**:
*   **확대/축소 (Zoom)**: 마우스 휠 또는 드래그하여 특정 영역 확대/축소.
*   **팬 (Pan)**: 드래그하여 그래프 이동.
*   **데이터 포인트 정보 (Hover)**: 마우스 커서를 데이터 포인트 위에 올리면 상세 정보 표시.
*   **선택 (Select)**: 특정 영역을 선택하여 데이터 필터링.
*   **툴바 (Toolbar)**: 플롯 상단에 나타나는 툴바를 통해 다양한 기능(다운로드, 리셋 등) 사용.

이러한 기능들은 `fig.show()`를 통해 그래프를 렌더링할 때 자동으로 활성화됩니다. 추가적인 설정 없이도 풍부한 사용자 경험을 제공합니다.

## 2. 서브플롯
Plotly Graph Objects를 사용하면 여러 개의 플롯을 하나의 Figure에 배치하는 서브플롯을 생성할 수 있습니다. `make_subplots` 함수를 사용합니다.

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 서브플롯 레이아웃 정의 (1행 2열)
fig = make_subplots(rows=1, cols=2, subplot_titles=('Sine Wave', 'Cosine Wave'))

x = np.linspace(0, 2 * np.pi, 100)

# 첫 번째 서브플롯에 라인 플롯 추가
fig.add_trace(go.Scatter(x=x, y=np.sin(x), mode='lines', name='Sine'), row=1, col=1)

# 두 번째 서브플롯에 라인 플롯 추가
fig.add_trace(go.Scatter(x=x, y=np.cos(x), mode='lines', name='Cosine'), row=1, col=2)

fig.update_layout(title_text="Sine and Cosine Waves in Subplots")
fig.show()
```

## 3. 플롯 저장
Plotly로 생성된 인터랙티브 플롯은 HTML 파일로 저장하여 웹 브라우저에서 그대로 열어볼 수 있습니다. 또한, 정적 이미지(PNG, JPEG, SVG, PDF)로도 저장할 수 있습니다.

```python
import plotly.express as px

iris = px.data.iris()
fig = px.scatter(iris, x="petal_length", y="petal_width", color="species")

# HTML 파일로 저장 (인터랙티브 기능 유지)
fig.write_html("iris_scatter.html")
print(''''iris_scatter.html''' 파일이 저장되었습니다.')

# PNG 이미지로 저장 (정적 이미지)
# Kaleido 라이브러리가 설치되어 있어야 합니다: pip install kaleido
try:
    fig.write_image("iris_scatter.png")
    print(''''iris_scatter.png''' 파일이 저장되었습니다.')
except ValueError:
    print("Kaleido 라이브러리가 설치되어 있지 않아 PNG 이미지 저장을 건너뜁니다. 'pip install kaleido'를 실행하세요.")

# PDF 이미지로 저장
try:
    fig.write_image("iris_scatter.pdf")
    print(''''iris_scatter.pdf''' 파일이 저장되었습니다.')
except ValueError:
    print("Kaleido 라이브러리가 설치되어 있지 않아 PDF 이미지 저장을 건너뜁니다. 'pip install kaleido'를 실행하세요.")
```
