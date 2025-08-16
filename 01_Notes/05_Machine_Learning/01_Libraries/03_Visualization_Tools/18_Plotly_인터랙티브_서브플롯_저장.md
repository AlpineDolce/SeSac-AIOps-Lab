<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly의 핵심 강점인 인터랙티브 플롯 기능과 이를 활용한 데이터 탐색 방법을 다룹니다. 또한, Plotly Graph Objects를 사용하여 여러 개의 플롯을 하나의 Figure에 배치하는 서브플롯 생성 방법과 생성된 플롯을 HTML 또는 정적 이미지 파일로 저장하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

* [1. Plotly 인터랙티브 플롯 개요](#1-plotly-인터랙티브-플롯-개요)
  * [1.1. 인터랙티브 플롯](#11-인터랙티브-플롯)
    * [1.1.1. 주요 특징](#111-주요-특징)
    * [1.1.2. 기본 사용법 및 커스터마이징](#112-기본-사용법-및-커스터마이징)
  * [1.2. 서브플롯](#12-서브플롯)
    * [1.2.1. 주요 특징](#121-주요-특징)
    * [1.2.2. 사용 시기](#122-사용-시기)
    * [1.2.3. 기본 사용법 및 커스터마이징](#123-기본-사용법-및-커스터마이징)
  * [1.3. 플롯 저장](#13-플롯-저장)
    * [1.3.1. 주요 저장 방법](#131-주요-저장-방법)
    * [1.3.2. 기본 사용법 및 커스터마이징](#132-기본-사용법-및-커스터마이징)

---

## 1. Plotly 인터랙티브 플롯 개요

### 1.1. 인터랙티브 플롯
Plotly는 정적인 이미지 그래프를 넘어, 사용자가 직접 데이터를 탐색하고 상호작용할 수 있는 **인터랙티브 플롯**을 생성하는 데 특화된 라이브러리입니다. Plotly로 생성된 모든 그래프는 기본적으로 웹 기반의 인터랙티브 기능을 내장하고 있어, 데이터 분석의 깊이와 효율성을 크게 향상시킵니다. 이러한 인터랙티브 기능은 Plotly.js라는 JavaScript 라이브러리에 의해 구동됩니다.

#### 1.1.1. 주요 특징
Plotly의 주요 인터랙티브 기능:
*   **확대/축소 (Zoom):** 마우스 휠을 사용하거나 그래프 영역을 드래그하여 특정 데이터 구간을 확대하거나 축소할 수 있습니다. 이를 통해 데이터의 미세한 패턴이나 특정 영역의 상세 분포를 쉽게 확인할 수 있습니다.
*   **팬 (Pan):** 확대된 그래프를 마우스 드래그하여 좌우 또는 상하로 이동시킬 수 있습니다.
*   **데이터 포인트 정보 (Hover):** 마우스 커서를 데이터 포인트 위에 올리면 해당 포인트의 정확한 값, 범주, 추가 정보 등 상세한 데이터 툴팁이 자동으로 표시됩니다. 이는 개별 데이터의 특성을 파악하는 데 매우 유용합니다.
*   **선택 (Select) 및 올가미 (Lasso Select):** 그래프에서 특정 영역을 드래그하여 데이터 포인트들을 선택할 수 있습니다. 선택된 데이터는 강조되거나, 다른 플롯에서 필터링되어 표시될 수 있습니다. 올가미 선택은 불규칙한 모양의 영역을 선택할 때 유용합니다.
*   **툴바 (Modebar):** 플롯의 상단 또는 하단에 자동으로 나타나는 툴바를 통해 다양한 기능을 사용할 수 있습니다.
    *   **다운로드:** 플롯을 PNG, SVG, JPEG, PDF 등 다양한 정적 이미지 파일로 저장할 수 있습니다.
    *   **자동 스케일 (Autoscale):** 플롯의 축 범위를 데이터에 맞게 자동으로 재설정합니다.
    *   **홈 (Home):** 플롯을 초기 상태로 되돌립니다.
    *   **비교 (Compare data on hover):** 여러 라인이나 그룹의 데이터를 동시에 비교하는 툴팁을 활성화합니다.
    *   **스파이크 라인 (Spikelines):** 마우스 커서 위치에 따라 축에 수직/수평선을 표시하여 값 읽기를 돕습니다.

이러한 인터랙티브 기능들은 `fig.show()`를 통해 그래프를 렌더링할 때 자동으로 활성화되며, 추가적인 설정 없이도 풍부한 사용자 경험을 제공하여 데이터 탐색, 이상치 발견, 패턴 인식 등을 더욱 효율적으로 수행할 수 있게 합니다.

#### 1.1.2. 기본 사용법 및 커스터마이징
```python
import plotly.express as px
import pandas as pd

# 예시 데이터 생성
data = {
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'y': [10, 12, 8, 15, 11, 13, 9, 16, 14, 17],
    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'value': [100, 120, 80, 150, 110, 130, 90, 160, 140, 170]
}
df = pd.DataFrame(data)

# 인터랙티브 산점도 예시
fig = px.scatter(df, x='x', y='y', color='category', size='value',
                 hover_data=['x', 'y', 'category', 'value'],
                 title='인터랙티브 산점도 예시')
fig.show()

# 라인 플롯에서의 인터랙티브 기능 예시
fig_line = px.line(df, x='x', y='y', color='category', title='인터랙티브 라인 플롯 예시')
fig_line.show()
```
#### 1.1.3. 고급 인터랙티브 기능 및 플롯 연동
Plotly는 기본적인 상호작용 외에도, 사용자가 데이터를 더욱 깊이 탐색할 수 있도록 돕는 고급 인터랙티브 기능을 제공합니다. 이러한 기능들은 특히 여러 플롯을 함께 분석할 때 강력한 시너지를 발휘합니다.

*   **커스텀 컨트롤 및 위젯 (Custom Controls & Widgets):** Plotly 자체는 복잡한 UI 위젯을 직접 제공하지 않지만, `plotly.graph_objects`의 `update_layout` 메서드를 통해 버튼(`updatemenus`)이나 슬라이더(`sliders`)와 같은 간단한 컨트롤을 추가하여 플롯의 특정 속성(예: 데이터 필터링, 축 범위 변경)을 동적으로 제어할 수 있습니다. 더 복잡한 대시보드 형태의 위젯은 Dash나 Streamlit과 같은 프레임워크에서 주로 구현됩니다.

*   **이벤트 핸들링 (Event Handling):** Plotly 그래프는 사용자의 상호작용(클릭, 선택, 호버 등)에 대한 이벤트를 발생시킵니다. 이러한 이벤트는 Dash와 같은 웹 프레임워크에서 콜백(callback) 함수를 통해 감지하고 처리하여, 사용자의 입력에 따라 다른 플롯이나 UI 요소를 동적으로 업데이트하는 데 활용됩니다. 예를 들어, 한 그래프에서 특정 데이터 포인트를 클릭하면 다른 그래프에 해당 데이터의 상세 정보가 표시되도록 구현할 수 있습니다.

*   **플롯 간 연동 (Linked Views):** 여러 개의 플롯을 동시에 표시할 때, 한 플롯에서의 상호작용(예: 데이터 선택)이 다른 플롯에 영향을 미치도록 연결하는 기능입니다. Plotly 자체에서는 직접적인 '링크' 기능을 제공하지 않지만, Dash와 같은 프레임워크의 콜백 시스템을 활용하여 쉽게 구현할 수 있습니다. 예를 들어, 산점도에서 특정 그룹의 데이터를 선택하면, 해당 그룹의 통계 분포를 보여주는 히스토그램이 업데이트되도록 만들 수 있습니다. 이는 탐색적 데이터 분석(EDA)에서 데이터의 다양한 측면을 동시에 비교하고 분석하는 데 매우 유용합니다.

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd

# 예시 데이터: Iris 데이터셋
iris = px.data.iris()

# 두 개의 서브플롯 생성
fig = make_subplots(rows=1, cols=2, subplot_titles=('Petal Length vs. Width', 'Sepal Length vs. Width'))

# 첫 번째 서브플롯: 꽃잎 길이 vs 너비 산점도
fig.add_trace(go.Scatter(
    x=iris['petal_length'],
    y=iris['petal_width'],
    mode='markers',
    marker=dict(color=iris['species_id'], colorscale='Viridis', showscale=True),
    name='Iris Petal',
    customdata=iris[['species', 'sepal_length', 'sepal_width']], # 추가 정보 저장
    hovertemplate='<b>Petal Length</b>: %{x}<br><b>Petal Width</b>: %{y}<br><b>Species</b>: %{customdata[0]}<extra></extra>'
), row=1, col=1)

# 두 번째 서브플롯: 꽃받침 길이 vs 너비 산점도
fig.add_trace(go.Scatter(
    x=iris['sepal_length'],
    y=iris['sepal_width'],
    mode='markers',
    marker=dict(color=iris['species_id'], colorscale='Viridis', showscale=True),
    name='Iris Sepal',
    customdata=iris[['species', 'petal_length', 'petal_width']], # 추가 정보 저장
    hovertemplate='<b>Sepal Length</b>: %{x}<br><b>Sepal Width</b>: %{y}<br><b>Species</b>: %{customdata[0]}<extra></extra>'
), row=1, col=2)

fig.update_layout(title_text="Iris Dataset: Linked Views (Conceptual Example)", showlegend=False)

# 이 예시는 Plotly 자체에서 직접적인 '플롯 간 연동'을 보여주지는 않습니다.
# 실제 플롯 간 연동은 Dash와 같은 웹 프레임워크의 콜백 기능을 통해 구현됩니다.
# 예를 들어, Dash 앱에서는 첫 번째 플롯에서 선택된 데이터 포인트를 기반으로
# 두 번째 플롯의 데이터를 필터링하거나 강조할 수 있습니다.

fig.show()
```
### 1.1.4. 플롯 레이아웃 및 트레이스 커스터마이징 (`update_layout()`, `update_traces()`)
Plotly Express(`px`)는 고수준 API로 편리하지만, 플롯의 제목, 축 라벨, 범례, 마커 스타일, 선 속성 등 세부적인 요소를 정교하게 제어하려면 `fig.update_layout()`과 `fig.update_traces()` 메서드를 사용하는 것이 일반적입니다. 이 메서드들은 `plotly.graph_objects` 기반의 Figure 객체에 직접 접근하여 플롯의 시각적 속성을 변경합니다.

#### 1.1.4.1. `fig.update_layout()`: 전체 플롯 레이아웃 변경
`update_layout()`은 Figure의 전반적인 레이아웃 속성(제목, 축 설정, 여백, 배경색, 범례 위치 등)을 변경할 때 사용합니다.

```python
import plotly.express as px

# 예시 플롯 생성
iris = px.data.iris()
fig = px.scatter(iris, x="petal_length", y="petal_width", color="species",
                 title="Iris Petal Length vs. Width")

# 레이아웃 커스터마이징
fig.update_layout(
    title_text="아이리스 꽃잎 길이와 너비 (커스터마이징)", # 제목 변경
    xaxis_title="꽃잎 길이 (cm)", # x축 라벨 변경
    yaxis_title="꽃잎 너비 (cm)", # y축 라벨 변경
    font=dict(family="Arial", size=12, color="RebeccaPurple"), # 폰트 설정
    plot_bgcolor="lightgray", # 플롯 배경색
    paper_bgcolor="lightyellow", # 종이 배경색
    margin=dict(l=40, r=40, t=80, b=40), # 여백 설정
    hovermode="closest" # 호버 모드 설정
)
fig.show()
```

#### 1.1.4.2. `fig.update_traces()`: 개별 트레이스 속성 변경
`update_traces()`는 플롯 내의 하나 이상의 트레이스(예: 산점도의 점, 라인 플롯의 선)의 시각적 속성을 변경할 때 사용합니다. `selector` 매개변수를 사용하여 특정 트레이스만 선택적으로 변경할 수 있습니다.

```python
import plotly.express as px

# 예시 플롯 생성
tips = px.data.tips()
fig = px.scatter(tips, x="total_bill", y="tip", color="day",
                 title="Total Bill vs. Tip by Day")

# 모든 트레이스의 마커 크기 변경
fig.update_traces(marker_size=10, selector=dict(mode='markers')) # mode가 'markers'인 트레이스 선택

# 특정 트레이스(예: 'Thur' 요일)의 색상 변경
# fig.data는 트레이스 리스트입니다. 각 트레이스의 name 속성을 확인하여 선택
for trace in fig.data:
    if trace.name == "Thur":
        trace.marker.color = "red"
    elif trace.name == "Fri":
        trace.marker.color = "blue"

fig.show()

# 라인 플롯에서 선 스타일 변경 예시
df_stocks = px.data.stocks()
fig_line = px.line(df_stocks, x="date", y="GOOG", title="GOOG Stock Price")
fig_line.update_traces(line=dict(width=4, dash='dot'), marker=dict(symbol='star', size=8))
fig_line.show()
```
```

### 1.2. 서브플롯
데이터 분석 과정에서 여러 개의 관련 플롯을 나란히 배치하여 비교하거나, 데이터의 다양한 측면을 동시에 보여줘야 할 필요가 있습니다. Plotly는 `plotly.subplots` 모듈의 `make_subplots` 함수를 통해 이러한 서브플롯(Subplots) 기능을 강력하게 지원합니다. 이를 통해 하나의 Figure 내에 여러 개의 개별 플롯을 효율적으로 구성할 수 있습니다.

#### 1.2.1. 주요 특징
*   **`make_subplots` 함수:** 서브플롯의 레이아웃(행과 열의 개수)을 정의하는 핵심 함수입니다.
    *   `rows`, `cols`: 서브플롯 그리드의 행과 열의 개수를 지정합니다.
    *   `subplot_titles`: 각 서브플롯에 대한 제목을 지정할 수 있습니다.
    *   `specs`: 각 서브플롯의 유형(예: 2D, 3D, 극좌표)이나 크기를 세밀하게 제어할 수 있습니다.
    *   `shared_xaxes`, `shared_yaxes`: 여러 서브플롯 간에 x축 또는 y축을 공유할지 여부를 설정하여 축 범위를 일관되게 유지할 수 있습니다.
*   **`add_trace` 메서드:** `make_subplots`로 생성된 Figure 객체에 `add_trace` 메서드를 사용하여 각 서브플롯에 개별 트레이스(플롯 요소)를 추가합니다. 이때 `row`와 `col` 파라미터를 사용하여 트레이스가 그려질 서브플롯의 위치를 지정합니다.
*   **`update_layout`:** 전체 Figure의 제목, 여백, 배경색 등 전반적인 레이아웃을 설정하거나 업데이트할 수 있습니다.

#### 1.2.2. 사용 시기
*   서로 다른 변수 간의 관계를 여러 플롯으로 비교하고 싶을 때.
*   동일한 변수를 다른 시각화 방식으로 동시에 보여주고 싶을 때 (예: 히스토그램과 KDE 플롯).
*   데이터의 하위 그룹별로 동일한 유형의 분석 결과를 나란히 제시하여 비교 분석의 효율성을 높일 때.

#### 1.2.3. 기본 사용법 및 커스터마이징
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px

# 서브플롯 레이아웃 정의 (1행 2열)
fig = make_subplots(rows=1, cols=2, subplot_titles=('Sine Wave', 'Cosine Wave'))

x = np.linspace(0, 2 * np.pi, 100)

# 첫 번째 서브플롯에 라인 플롯 추가
fig.add_trace(go.Scatter(x=x, y=np.sin(x), mode='lines', name='Sine'), row=1, col=1)

# 두 번째 서브플롯에 라인 플롯 추가
fig.add_trace(go.Scatter(x=x, y=np.cos(x), mode='lines', name='Cosine'), row=1, col=2)

fig.update_layout(title_text="Sine and Cosine Waves in Subplots")
fig.show()

# 추가 예시: Plotly Express와 make_subplots 함께 사용
# make_subplots로 Figure 객체 생성
fig_px = make_subplots(rows=1, cols=2, subplot_titles=('Iris Petal Length vs. Width', 'Iris Sepal Length vs. Width'))

iris = px.data.iris()

# px.scatter로 생성된 그래프의 data 속성을 add_trace로 추가
fig_px.add_trace(px.scatter(iris, x="petal_length", y="petal_width", color="species").data[0], row=1, col=1)
fig_px.add_trace(px.scatter(iris, x="sepal_length", y="sepal_width", color="species").data[0], row=1, col=2)

# Plotly Express의 trace는 기본적으로 여러 개일 수 있으므로, 모든 trace를 추가하려면 반복문 사용
# for trace in px.scatter(iris, x="petal_length", y="petal_width", color="species").data:
#     fig_px.add_trace(trace, row=1, col=1)
# for trace in px.scatter(iris, x="sepal_length", y="sepal_width", color="species").data:
#     fig_px.add_trace(trace, row=1, col=2)

fig_px.update_layout(title_text="Iris Dataset Subplots with Plotly Express")
fig_px.show()
```

### 1.3. 플롯 저장
Plotly로 생성된 인터랙티브 플롯은 분석 결과를 공유하거나 보고서에 포함할 때 매우 유용합니다. Plotly는 플롯을 웹 브라우저에서 그대로 열어볼 수 있는 HTML 파일로 저장하는 기능과, 인쇄나 문서에 적합한 다양한 정적 이미지 형식으로 저장하는 기능을 모두 제공합니다.

#### 1.3.1. 주요 저장 방법
*   **HTML 파일로 저장 (`fig.write_html()`):**
    *   **장점:** Plotly의 모든 인터랙티브 기능(확대/축소, 팬, 호버 등)이 그대로 유지됩니다. 웹 브라우저만 있으면 누구나 플롯을 조작하며 데이터를 탐색할 수 있습니다.
    *   **사용 시기:** 웹 기반 보고서, 대시보드, 또는 인터랙티브한 데이터 탐색 기능을 공유하고자 할 때 가장 적합합니다.
*   **정적 이미지 파일로 저장 (`fig.write_image()`):**
    *   **종류:** PNG (웹 및 일반 문서), JPEG (사진), SVG (확대해도 깨지지 않는 벡터 그래픽), PDF (인쇄 및 문서) 등 다양한 형식으로 저장할 수 있습니다.
    *   **필수 라이브러리:** `fig.write_image()`를 사용하여 정적 이미지를 저장하려면 `kaleido` 라이브러리가 반드시 설치되어 있어야 합니다 (`pip install kaleido`). 이 라이브러리는 Plotly Figure를 다양한 이미지 형식으로 변환하는 데 사용됩니다.
    *   **사용 시기:** 인쇄용 보고서, 프레젠테이션 슬라이드, 또는 웹사이트에 정적인 이미지를 삽입해야 할 때 사용합니다.
*   **Figure 데이터 저장 (`fig.to_json()`):**
    *   플롯의 모든 데이터와 레이아웃 정보를 JSON 형식으로 저장할 수 있습니다. 이는 플롯을 나중에 다시 로드하여 수정하거나 다른 Plotly 환경에서 재사용할 때 유용합니다.

#### 1.3.2. 기본 사용법 및 커스터마이징
```python
import plotly.express as px
import plotly.graph_objects as go
import json

iris = px.data.iris()
fig = px.scatter(iris, x="petal_length", y="petal_width", color="species",
                 title="Iris Petal Length vs. Width by Species")

# HTML 파일로 저장 (인터랙티브 기능 유지)
html_file_path = "iris_scatter.html"
fig.write_html(html_file_path)
print(f"'{html_file_path}' 파일이 저장되었습니다.")

# PNG 이미지로 저장 (정적 이미지)
# Kaleido 라이브러리가 설치되어 있어야 합니다: pip install kaleido
png_file_path = "iris_scatter.png"
try:
    fig.write_image(png_file_path)
    print(f"'{png_file_path}' 파일이 저장되었습니다.")
except ValueError:
    print(f"Kaleido 라이브러리가 설치되어 있지 않아 '{png_file_path}' 이미지 저장을 건너뜁니다. 'pip install kaleido'를 실행하세요.")

# PDF 이미지로 저장
pdf_file_path = "iris_scatter.pdf"
try:
    fig.write_image(pdf_file_path)
    print(f"'{pdf_file_path}' 파일이 저장되었습니다.")
except ValueError:
    print(f"Kaleido 라이브러리가 설치되어 있지 않아 '{pdf_file_path}' 이미지 저장을 건너뜁니다. 'pip install kaleido'를 실행하세요.")

# Figure 데이터를 JSON으로 저장
json_file_path = "iris_scatter_data.json"
with open(json_file_path, "w") as f:
    json.dump(fig.to_json(), f)
print(f"'{json_file_path}' 파일이 저장되었습니다.")

# 저장된 JSON 파일에서 Figure 로드 예시
# with open(json_file_path, "r") as f:
#     loaded_fig_json = json.load(f)
# loaded_fig = go.Figure(json.loads(loaded_fig_json))
# loaded_fig.show()
```
