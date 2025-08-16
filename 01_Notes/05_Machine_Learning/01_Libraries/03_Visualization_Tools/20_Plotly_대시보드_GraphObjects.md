<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly를 기반으로 하는 웹 애플리케이션 프레임워크인 Dash와 Streamlit을 사용하여 인터랙티브 대시보드를 구축하는 방법을 다룹니다. 또한, Plotly Graph Objects(`go`)를 이용한 저수준 API를 통해 복잡하고 사용자 정의된 인터랙티브 시각화를 구현하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

* [1. Plotly 대시보드 및 Graph Objects 개요](#1-plotly-대시보드-및-graph-objects-개요)
  * [1.1. 인터랙티브 대시보드 구축](#11-인터랙티브-대시보드-구축)
    * [1.1.1. Dash (Plotly Dash)](#111-dash-plotly-dash)
    * [1.1.2. Streamlit](#112-streamlit)
    * [1.1.3. Dash vs Streamlit 비교](#113-dash-vs-streamlit-비교)
    * [1.1.4. 대시보드에서의 고급 인터랙티브 기능 및 플롯 연동](#114-대시보드에서의-고급-인터랙티브-기능-및-플롯-연동)
  * [1.2. Plotly Graph Objects 심화](#12-plotly-graph-objects-심화)
    * [1.2.1. 핵심 개념](#121-핵심-개념)
    * [1.2.2. 플롯 생성 과정](#122-플롯-생성-과정)
    * [1.2.3. 사용 시기](#123-사용-시기)
    * [1.2.4. 기본 사용법 및 커스터마이징](#124-기본-사용법-및-커스터마이징)
    * [1.2.5. 인터랙티브 컨트롤 추가 (버튼, 슬라이더)](#125-인터랙티브-컨트롤-추가-버튼-슬라이더)

---

## 1. Plotly 대시보드 및 Graph Objects 개요

### 1.1. 인터랙티브 대시보드 구축
데이터 분석 결과를 효과적으로 공유하고 사용자와 상호작용할 수 있는 웹 기반 대시보드는 현대 데이터 과학에서 필수적인 요소입니다. Plotly는 그 자체로 인터랙티브한 그래프를 생성하지만, 이를 웹 애플리케이션 형태로 제공하기 위해서는 Dash나 Streamlit과 같은 대시보드 프레임워크가 필요합니다. 이 두 프레임워크는 파이썬 기반으로, 복잡한 웹 개발 지식 없이도 데이터 애플리케이션을 구축할 수 있게 해줍니다.

#### 1.1.1. Dash (Plotly Dash)
Dash는 Plotly에서 개발한 파이썬 웹 애플리케이션 프레임워크입니다. React.js, Plotly.js, Flask를 기반으로 하며, 파이썬 코드만으로 인터랙티브한 대시보드를 만들 수 있게 해줍니다.

##### 1.1.1.1. 주요 특징
*   **컴포넌트 기반:** HTML, CSS, JavaScript 요소를 파이썬 클래스로 추상화한 `dash_html_components`와 `dash_core_components`를 사용하여 레이아웃을 구성합니다.
*   **콜백(Callback) 시스템:** `@app.callback` 데코레이터를 사용하여 사용자 입력(예: 슬라이더 조작, 드롭다운 선택)에 따라 그래프나 다른 컴포넌트를 동적으로 업데이트하는 로직을 구현합니다.
*   **강력한 커스터마이징:** 복잡하고 맞춤화된 기업용 대시보드 및 분석 애플리케이션 구축에 적합하며, 세밀한 제어가 가능합니다.
*   **확장성:** 대규모 애플리케이션 개발 및 배포에 유리합니다.

##### 1.1.1.2. 간단한 Dash 앱 예시
```python
# pip install dash
# pip install dash-core-components
# pip install dash-html-components

# import dash
# from dash import dcc
# from dash import html
# from dash.dependencies import Input, Output
# import plotly.express as px

# app = dash.Dash(__name__)

# df = px.data.gapminder()

# app.layout = html.Div([
#     dcc.Graph(id='life-exp-vs-gdp'),
#     dcc.Slider(
# #        df['year'].min(), # 주석 처리된 부분 수정
# #        df['year'].max(), # 주석 처리된 부분 수정
#         min=df['year'].min(),
#         max=df['year'].max(),
#         step=None,
#         value=df['year'].min(),
#         marks={str(year): str(year) for year in df['year'].unique()},
#         id='year-slider'
#     )
# ])


# @app.callback(
#     Output('life-exp-vs-gdp', 'figure'),
#     Input('year-slider', 'value'))
# def update_figure(selected_year):
#     filtered_df = df[df.year == selected_year]

#     fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
#                      size="pop", color="continent", hover_name="country",
#                      log_x=True, size_max=55)

#     fig.update_layout(transition_duration=500)

#     return fig


# if __name__ == '__main__':
#     app.run_server(debug=True)
```

#### 1.1.2. Streamlit
Streamlit은 데이터 과학자와 머신러닝 엔지니어가 파이썬 스크립트를 몇 줄의 코드만으로 인터랙티브 웹 앱으로 빠르게 변환할 수 있도록 설계된 오픈소스 프레임워크입니다.

##### 1.1.2.1. 주요 특징
*   **간단한 사용법:** 파이썬 스크립트 상단에 `import streamlit as st`를 추가하고, `st.write()`, `st.sidebar()`, `st.slider()` 등 직관적인 API를 사용하여 UI 요소를 추가합니다.
*   **빠른 프로토타이핑:** 매우 적은 코드로 빠르게 웹 앱을 개발하고 배포할 수 있어 아이디어 검증이나 초기 단계의 대시보드 구축에 매우 적합합니다.
*   **데이터 중심:** 데이터 시각화 및 머신러닝 모델 배포에 최적화되어 있습니다.
*   **자동 업데이트:** 코드 변경 시 앱이 자동으로 업데이트되어 개발 속도가 빠릅니다.

##### 1.1.2.2. 간단한 Streamlit 앱 예시
```python
# pip install streamlit
# pip install plotly

# import streamlit as st
# import plotly.express as px
# import pandas as pd

# st.title('Streamlit과 Plotly 예시 대시보드')

# # 데이터 로드
# df = px.data.gapminder()

# # 연도 선택 슬라이더
# year = st.slider('연도 선택', min_value=int(df['year'].min()), max_value=int(df['year'].max()), step=1, value=int(df['year'].min()))

# # 선택된 연도에 따른 데이터 필터링
# filtered_df = df[df['year'] == year]

# # Plotly Express 산점도 생성
# fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
#                  size="pop", color="continent", hover_name="country",
#                  log_x=True, size_max=55, title=f'{year}년 GDP 대비 기대 수명')

# # Plotly 그래프를 Streamlit에 표시
# st.plotly_chart(fig, use_container_width=True)

# # 사이드바에 추가 정보 표시
# st.sidebar.header('데이터 정보')
# st.sidebar.write(f'선택된 연도: {year}')
# st.sidebar.write(f'총 국가 수: {len(filtered_df)}')
```

#### 1.1.3. Dash vs Streamlit 비교
| 특징         | Dash                                     | Streamlit                                |
| :----------- | :--------------------------------------- | :--------------------------------------- |
| **개발 철학**  | 웹 애플리케이션 프레임워크 (React 기반) | 데이터 스크립트를 웹 앱으로 변환         |
| **복잡성**     | 더 복잡하고 세밀한 제어 가능             | 매우 간단하고 직관적                     |
| **유스케이스** | 복잡하고 맞춤화된 기업용 대시보드, 프로덕션 앱 | 빠른 프로토타이핑, 간단한 데이터 앱, ML 모델 데모 |
| **커스터마이징** | 높은 수준의 UI/UX 커스터마이징 가능      | 제한적이지만, 빠르게 개발 가능           |
| **학습 곡선**  | 더 가파름 (웹 개발 개념 필요)            | 매우 완만함 (파이썬 스크립트 작성하듯)   |
| **배포**       | 더 복잡할 수 있음 (Flask 기반)           | Streamlit Cloud 등 간편한 배포 옵션 제공 |

**결론:**
*   **Dash:** 복잡한 레이아웃, 세밀한 UI/UX 제어, 대규모 프로덕션 환경에 적합한 기업용 대시보드를 구축할 때 유리합니다.
*   **Streamlit:** 데이터 분석 스크립트를 빠르게 웹 앱으로 만들고 싶을 때, 또는 간단한 대시보드나 머신러닝 모델 데모를 만들 때 압도적으로 효율적입니다.

#### 1.1.4. 대시보드에서의 고급 인터랙티브 기능 및 플롯 연동
Dash와 Streamlit과 같은 대시보드 프레임워크는 Plotly의 강력한 인터랙티브 기능을 활용하여, 단순한 그래프 표시를 넘어선 복잡하고 동적인 데이터 애플리케이션을 구축할 수 있게 합니다. 특히, 여러 플롯 간의 상호작용과 사용자 정의 컨트롤을 통해 데이터 탐색의 깊이를 더할 수 있습니다.

*   **커스텀 컨트롤 및 위젯 (Custom Controls & Widgets):**
    *   **Dash:** `dash_core_components`와 `dash_html_components`를 사용하여 슬라이더, 드롭다운, 버튼, 체크박스 등 다양한 UI 컴포넌트를 대시보드에 추가할 수 있습니다. 이 컴포넌트들은 사용자의 입력을 받아 Plotly 그래프를 동적으로 업데이트하는 데 활용됩니다. 예를 들어, 사용자가 드롭다운에서 특정 범주를 선택하면 해당 범주에 해당하는 데이터만 그래프에 표시되도록 할 수 있습니다.
    *   **Streamlit:** `st.slider()`, `st.selectbox()`, `st.button()` 등 직관적인 API를 통해 손쉽게 위젯을 추가하고, 이 위젯의 값 변화에 따라 Plotly 그래프를 포함한 앱의 모든 요소를 실시간으로 업데이트할 수 있습니다.

*   **이벤트 핸들링 및 플롯 간 연동 (Event Handling & Linked Views):**
    *   **Dash의 콜백 시스템:** Dash의 핵심은 `@app.callback` 데코레이터를 이용한 콜백 시스템입니다. 이를 통해 특정 컴포넌트의 속성 변화(예: 슬라이더 값 변경, 그래프에서 데이터 선택)를 입력(Input)으로 받아, 다른 컴포넌트의 속성(예: 다른 그래프의 데이터나 레이아웃)을 출력(Output)으로 업데이트할 수 있습니다.
        *   **플롯 간 연동:** 이 콜백 시스템을 활용하면 여러 Plotly 그래프 간의 강력한 연동을 구현할 수 있습니다. 예를 들어, 한 산점도에서 특정 데이터 포인트들을 선택(Lasso Select 또는 Box Select)하면, 이 선택된 데이터에 해당하는 상세 정보가 다른 테이블이나 히스토그램에 표시되도록 만들 수 있습니다. 이는 데이터의 특정 부분에 대한 심층 분석을 가능하게 하여 탐색적 데이터 분석(EDA)의 효율성을 극대화합니다.
        *   **사용자 정의 상호작용:** 그래프의 클릭 이벤트, 호버 이벤트 등을 감지하여 특정 정보를 팝업으로 표시하거나, 관련 데이터를 동적으로 로드하는 등 복잡한 사용자 정의 상호작용을 구현할 수 있습니다.

*   **예시 (개념적 설명):**
    *   사용자가 대시보드의 드롭다운 메뉴에서 특정 지역을 선택하면, 해당 지역의 시계열 데이터가 Plotly 라인 그래프로 표시됩니다.
    *   이 라인 그래프에서 사용자가 특정 기간을 확대(zoom)하면, 그 기간 동안의 상세 이벤트가 아래의 테이블에 자동으로 필터링되어 나타납니다.
    *   또는, 한 그래프에서 이상치(outlier)를 클릭하면, 해당 이상치에 대한 추가적인 분석 정보가 다른 Plotly 그래프(예: 분포 플롯)로 시각화되어 나타나는 방식입니다.

이러한 고급 인터랙티브 기능들은 데이터 분석가가 데이터를 더욱 직관적이고 효율적으로 탐색하고, 발견한 인사이트를 청중에게 동적으로 시연하며 설득력을 높이는 데 결정적인 역할을 합니다.

### 1.2. Plotly Graph Objects 심화

Plotly Express(`px`)는 고수준 API로, 적은 코드로 빠르게 플롯을 생성할 수 있다는 장점이 있습니다. 하지만, 플롯의 세부적인 요소를 제어하거나 여러 종류의 플롯을 하나의 Figure에 결합하는 등 복잡하고 사용자 정의된 인터랙티브 시각화를 구현해야 할 때는 Plotly의 저수준 API인 **Plotly Graph Objects(`go`)**를 사용해야 합니다. `go` 모듈은 Plotly 그래프의 모든 구성 요소를 객체 지향적으로 다룰 수 있게 해줍니다.

#### 1.2.1. 핵심 개념
*   **Figure 객체 (`go.Figure`):** Plotly 그래프의 최상위 컨테이너입니다. 모든 트레이스(데이터 시각화 요소)와 레이아웃(축, 제목, 범례 등)을 포함합니다.
*   **Trace 객체 (`go.Scatter`, `go.Bar`, `go.Heatmap` 등):** 실제 데이터를 시각화하는 요소입니다. 각 트레이스는 데이터의 종류(산점도, 막대, 라인 등)와 시각적 속성(색상, 마커 모양, 선 스타일 등)을 정의합니다. `go.Scatter`는 산점도뿐만 아니라 라인 플롯도 그릴 수 있습니다.
*   **Layout 객체 (`go.Layout`):** Figure의 전반적인 시각적 속성(제목, 축 라벨, 범례 위치, 배경색 등)을 제어합니다. `fig.update_layout()` 메서드를 통해 레이아웃을 업데이트할 수 있습니다.

#### 1.2.2. 플롯 생성 과정
1.  `go.Figure()`를 사용하여 빈 Figure 객체를 생성합니다.
2.  `go.Scatter()`, `go.Bar()` 등 적절한 Trace 객체를 생성하고, `fig.add_trace()` 메서드를 사용하여 Figure에 추가합니다.
3.  `fig.update_layout()` 메서드를 사용하여 Figure의 레이아웃을 커스터마이징합니다.

#### 1.2.3. 사용 시기
*   Plotly Express로는 구현하기 어려운 복잡하고 사용자 정의된 플롯을 만들 때.
*   하나의 Figure에 여러 종류의 플롯(예: 산점도와 라인 플롯, 막대 그래프와 라인 플롯)을 결합해야 할 때.
*   버튼, 슬라이더 등 커스텀 컨트롤을 추가하여 플롯의 동작을 세밀하게 제어해야 할 때.
*   플롯의 모든 시각적 속성을 저수준에서 완벽하게 제어하고자 할 때.

#### 1.2.4. 기본 사용법 및 커스터마이징
```python
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# 예시 데이터 생성
x_data = np.linspace(0, 10, 100)
y_data_sin = np.sin(x_data)
y_data_cos = np.cos(x_data)

# 1. 빈 Figure 객체 생성
fig = go.Figure()

# 2. Trace 객체 추가 (사인파 라인 플롯)
fig.add_trace(go.Scatter(x=x_data, y=y_data_sin, mode='lines', name='Sine Wave',
                         line=dict(color='royalblue', width=2)))

# 3. 다른 Trace 객체 추가 (코사인파 라인 플롯)
fig.add_trace(go.Scatter(x=x_data, y=y_data_cos, mode='lines+markers', name='Cosine Wave',
                         marker=dict(symbol='circle', size=5),
                         line=dict(color='firebrick', width=2, dash='dot')))

# 4. 레이아웃 커스터마이징
fig.update_layout(
    title={
        'text': "사인파와 코사인파 비교 (Graph Objects)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="X 값",
    yaxis_title="Y 값",
    hovermode="x unified", # 마우스 오버 시 x축을 기준으로 모든 트레이스 정보 표시
    template="plotly_white" # 배경 템플릿 설정
)

fig.show()

# 추가 예시: 막대 그래프와 라인 플롯 결합
df_sales = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
    'Sales': [100, 120, 150, 130, 180],
    'Target': [110, 110, 140, 140, 170]
})

fig_combined = go.Figure()

# 막대 그래프 트레이스 추가
fig_combined.add_trace(go.Bar(
    x=df_sales['Month'],
    y=df_sales['Sales'],
    name='실제 판매량',
    marker_color='lightseagreen'
))

# 라인 플롯 트레이스 추가
fig_combined.add_trace(go.Scatter(
    x=df_sales['Month'],
    y=df_sales['Target'],
    mode='lines+markers',
    name='목표 판매량',
    line=dict(color='red', width=3, dash='dash'),
    marker=dict(size=8, symbol='star')
))

fig_combined.update_layout(
    title='월별 판매량 및 목표 비교',
    xaxis_title='월',
    yaxis_title='판매량',
    barmode='group' # 막대 그래프가 겹치지 않도록 설정
)

fig_combined.show()
```

#### 1.2.5. 인터랙티브 컨트롤 추가 (버튼, 슬라이더)
`go.Figure`의 `update_layout` 메서드를 사용하면 버튼(`updatemenus`)이나 슬라이더(`sliders`)를 추가하여 사용자가 플롯과 상호작용할 수 있도록 만들 수 있습니다. 이는 Dash와 같은 별도의 프레임워크 없이도 동적인 데이터 탐색 기능을 제공합니다.

```python
import plotly.graph_objects as go
import pandas as pd

# 데이터 로드
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = go.Figure()

# 초기 라인 플롯 (AAPL.High)
fig.add_trace(go.Scatter(x=list(df.Date), y=list(df['AAPL.High']), name="High"))

# 보이지 않는 다른 트레이스들 추가
fig.add_trace(go.Scatter(x=list(df.Date), y=list(df['AAPL.Low']), name="Low", visible=False))
fig.add_trace(go.Scatter(x=list(df.Date), y=list(df['AAPL.Close']), name="Close", visible=False))

# 버튼 생성
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="High",
                     method="update",
                     args=[{"visible": [True, False, False]},
                           {"title": "Apple High Prices Over Time"}]),
                dict(label="Low",
                     method="update",
                     args=[{"visible": [False, True, False]},
                           {"title": "Apple Low Prices Over Time"}]),
                dict(label="Close",
                     method="update",
                     args=[{"visible": [False, False, True]},
                           {"title": "Apple Close Prices Over Time"}]),
            ]),
        )
    ])

fig.update_layout(title_text="Apple Stock Prices Over Time")
fig.show()
```
