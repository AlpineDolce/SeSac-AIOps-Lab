<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly Express(`px`)를 사용하여 산점도(`px.scatter()`)와 라인 플롯(`px.line()`)을 그리는 방법을 다룹니다. Plotly의 인터랙티브 기능과 애니메이션 프레임을 활용하여 데이터의 관계와 시간 변화에 따른 추세를 동적으로 시각화하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 기본 플로팅 (Plotly Express)](#1-기본-플로팅-plotly-express)
  - [1.1. 산점도 (`px.scatter()`)](#11-산점도-pxscatter)
  - [1.2. 라인 플롯 (`px.line()`)](#12-라인-플롯-pxline)

---

## 1. 기본 플로팅 (Plotly Express)
Plotly Express는 Pandas DataFrame을 기반으로 다양한 유형의 인터랙티브 그래프를 쉽게 생성할 수 있습니다.

### 1.1. 산점도 (`px.scatter()`)
두 변수 간의 관계를 점으로 표현하며, 마우스 오버 시 데이터 정보를 보여주는 등 인터랙티브 기능을 제공합니다.

```python
import plotly.express as px

# 내장 데이터셋 사용
iris = px.data.iris()

# 꽃잎 길이(petal_length)와 너비(petal_width)의 산점도
fig = px.scatter(iris, x="petal_length", y="petal_width", title="Iris Petal Length vs. Width")
fig.show()

# 종(species)에 따라 색상 구분 및 마우스 오버 정보 추가
fig = px.scatter(iris, x="petal_length", y="petal_width", color="species",
                 hover_data=['sepal_length', 'sepal_width'], title="Iris Petal Length vs. Width by Species")
fig.show()

# 시간에 따른 변화를 애니메이션으로 표현 (Gapminder 데이터셋 활용)
gapminder = px.data.gapminder()
fig = px.scatter(gapminder, x="gdpPercap", y="lifeExp", animation_frame="year",
                 animation_group="country", size="pop", color="continent", hover_name="country",
                 log_x=True, size_max=55, title="GDP per Capita vs. Life Expectancy Over Time")
fig.show()
```

### 1.2. 라인 플롯 (`px.line()`)
시계열 데이터나 연속적인 데이터의 추세를 보여줄 때 사용하며, 인터랙티브 기능을 통해 특정 구간을 확대하거나 데이터 포인트를 확인할 수 있습니다.

```python
import plotly.express as px

# 내장 데이터셋 사용 (Gapminder: 시간에 따른 국가별 인구, 기대 수명, GDP)
gapminder = px.data.gapminder()

# 시간에 따른 아프가니스탄의 기대 수명 변화
fig = px.line(gapminder.query("country=='Afghanistan'"), x="year", y="lifeExp", title="Life Expectancy in Afghanistan Over Time")
fig.show()

# 대륙별 기대 수명 추이 (색상 구분)
fig = px.line(gapminder.query("continent=='Asia'"), x="year", y="lifeExp", color="country", title="Life Expectancy in Asia by Country")
fig.show()
```
