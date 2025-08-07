<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly를 기반으로 하는 웹 애플리케이션 프레임워크인 Dash와 Streamlit을 사용하여 인터랙티브 대시보드를 구축하는 방법을 다룹니다. 또한, Plotly Graph Objects(`go`)를 이용한 저수준 API를 통해 복잡하고 사용자 정의된 인터랙티브 시각화를 구현하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 인터랙티브 대시보드 구축: Dash와 Streamlit](#1-인터랙티브-대시보드-구축-dash와-streamlit)
- [2. Plotly Graph Objects 심화](#2-plotly-graph-objects-심화)

---

## 1. 인터랙티브 대시보드 구축: Dash와 Streamlit
Dash는 Plotly를 기반으로 하는 파이썬 웹 애플리케이션 프레임워크입니다. Dash를 사용하면 복잡한 웹 개발 지식 없이도 인터랙티브한 대시보드와 데이터 시각화 애플리케이션을 구축할 수 있습니다. Plotly 그래프는 Dash 앱에 쉽게 통합될 수 있으며, 사용자 입력에 따라 동적으로 업데이트되는 대시보드를 만들 수 있습니다.

**간단한 Dash 앱 예시 (설치 및 실행 필요)**:

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
#         df['year'].min(),
#         df['year'].max(),
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

Dash가 잘 소개되었지만, 실무에서는 목적에 따라 다른 대시보드 도구도 활발히 사용됩니다. 특히 **Streamlit**은 매우 적은 코드로 데이터 분석 스크립트를 인터랙티브 웹 앱으로 빠르게 변환할 수 있어 프로토타이핑에 압도적인 인기를 얻고 있습니다. Dash와 Streamlit의 장단점(Dash: 복잡하고 맞춤화된 기업용 앱에 적합, Streamlit: 빠르고 간단한 데이터 앱에 적합)을 비교 설명하면 사용자가 상황에 맞는 도구를 선택하는 데 도움이 됩니다.

## 2. Plotly Graph Objects 심화

Plotly Express(`px`)는 빠르고 편리하지만, 복잡한 인터랙티브 플롯을 만드는 데는 한계가 있습니다. 저수준 API인 Plotly Graph Objects(`go`)는 `go.Figure` 객체를 생성하고, `go.Scatter`, `go.Bar`와 같은 'Trace' 객체를 추가하며, `fig.update_layout()`으로 레이아웃을 세밀하게 제어하는 방식을 사용합니다. 이를 통해 여러 종류의 그래프를 하나의 Figure에 결합하거나, 버튼, 슬라이더와 같은 커스텀 컨트롤을 추가하는 등 완전히 사용자 정의된 인터랙티브 시각화를 구현할 수 있습니다.
