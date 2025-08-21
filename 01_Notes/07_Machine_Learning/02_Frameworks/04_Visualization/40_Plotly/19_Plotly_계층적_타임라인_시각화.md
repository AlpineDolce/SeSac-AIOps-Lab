<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly Express(`px`)를 사용하여 계층적 데이터(`px.treemap`, `px.sunburst`)와 타임라인(`px.timeline`)을 시각화하는 방법을 다룹니다. 복잡한 계층 구조와 시간 기반 이벤트를 효과적으로 탐색하고 표현하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

* [1. Plotly 계층적 및 타임라인 플롯](#1-plotly-계층적-및-타임라인-플롯)
  * [1.1. 계층적 데이터 시각화](#11-계층적-데이터-시각화)
    * [1.1.1. `px.treemap()`: 트리맵](#111-pxtreemap-트리맵)
    * [1.1.2. `px.sunburst()`: 선버스트 차트](#112-pxsunburst-선버스트-차트)
  * [1.2. `px.timeline()`: 타임라인 (간트 차트)](#12-pxtimeline-타임라인-간트-차트)

---

## 1. Plotly 계층적 및 타임라인 플롯

### 1.1. 계층적 데이터 시각화
계층적 데이터는 부모-자식 관계를 가지는 데이터를 의미합니다. Plotly는 이러한 데이터 구조를 효과적으로 시각화하는 `treemap`과 `sunburst` 차트를 제공합니다.

#### 1.1.1. `px.treemap()`: 트리맵
트리맵은 중첩된 사각형을 사용하여 계층 구조를 표현합니다. 각 사각형의 크기는 특정 값(예: 인구, 매출)에 비례하며, 색상은 다른 변수를 나타내는 데 사용될 수 있습니다. 전체 구조 내에서 각 항목의 비율을 한눈에 파악하는 데 매우 유용합니다.

**주요 특징:**
*   **공간 효율성:** 제한된 공간 내에서 많은 계층적 데이터를 효율적으로 표현합니다.
*   **비율 시각화:** 각 사각형의 크기가 값에 비례하므로, 전체에서 각 부분이 차지하는 비중을 직관적으로 비교할 수 있습니다.
*   **인터랙티브 탐색:** 특정 사각형을 클릭하면 해당 계층으로 드릴다운(drill down)하여 더 상세한 내용을 탐색할 수 있습니다.

**주요 파라미터:**
*   `path`: 계층 구조를 정의하는 컬럼들의 리스트입니다. (예: `['continent', 'country']`)
*   `values`: 사각형의 크기를 결정하는 숫자형 컬럼입니다.
*   `color`: 사각형의 색상을 결정하는 컬럼입니다.

```python
import plotly.express as px

# 예시 데이터 로드 (Gapminder)
gapminder = px.data.gapminder().query("year == 2007")

# 대륙 > 국가 계층 구조로 트리맵 생성
fig = px.treemap(gapminder, path=[px.Constant("world"), 'continent', 'country'], 
                values='pop', color='lifeExp',
                hover_data=['iso_alpha'],
                color_continuous_scale='RdBu',
                title='2007년 대륙 및 국가별 인구와 기대 수명 (Treemap)')

fig.show()
```

#### 1.1.2. `px.sunburst()`: 선버스트 차트
선버스트 차트는 방사형 트리맵으로, 계층 구조를 동심원으로 표현합니다. 중앙에서 바깥쪽으로 갈수록 더 깊은 계층을 나타냅니다. 계층 간의 관계와 각 계층의 구성을 시각적으로 탐색하는 데 효과적입니다.

**주요 특징:**
*   **계층 구조 강조:** 동심원 구조가 데이터의 계층적 관계를 명확하게 보여줍니다.
*   **인터랙티브 드릴다운:** 트리맵과 마찬가지로, 특정 부채꼴을 클릭하여 하위 계층으로 드릴다운할 수 있습니다.

**주요 파라미터:**
*   `path`: 트리맵과 동일하게 계층 구조를 정의합니다.
*   `values`: 부채꼴의 크기를 결정합니다.
*   `color`: 부채꼴의 색상을 결정합니다.

```python
import plotly.express as px

# 예시 데이터 로드 (Tips)
tips = px.data.tips()

# 요일 > 시간 > 성별 계층 구조로 선버스트 차트 생성
fig = px.sunburst(tips, path=['day', 'time', 'sex'], values='total_bill',
                  color='tip', hover_data=['tip'],
                  color_continuous_scale='Blues',
                  title='요일, 시간, 성별에 따른 총 계산액과 팁 (Sunburst)')
fig.show()
```

### 1.2. `px.timeline()`: 타임라인 (간트 차트)
타임라인 플롯은 일반적으로 간트 차트(Gantt Chart)로 알려져 있으며, 시간 축에 따라 각 작업이나 이벤트의 시작과 끝을 시각화합니다. 프로젝트 관리, 자원 할당, 이벤트 스케줄링 등 다양한 분야에서 활용됩니다.

**주요 특징:**
*   **시간 기반 스케줄링:** 각 막대는 특정 작업이나 이벤트를 나타내며, 시간 축 위의 위치와 길이를 통해 해당 작업의 기간과 시점을 명확하게 보여줍니다.
*   **그룹화:** `y`축과 `color` 파라미터를 사용하여 작업들을 특정 그룹(예: 담당자, 프로젝트)별로 묶어서 시각화할 수 있습니다.

**주요 파라미터:**
*   `df`: 타임라인 정보를 담은 데이터프레임.
*   `x_start`: 각 작업의 시작 시간을 나타내는 컬럼.
*   `x_end`: 각 작업의 종료 시간을 나타내는 컬럼.
*   `y`: 각 작업을 구분하는 범주형 컬럼 (y축에 표시됨).
*   `color`: 작업을 그룹화하여 색상으로 구분하는 컬럼.

```python
import plotly.express as px
import pandas as pd

# 예시 데이터 생성 (프로젝트 작업)
df = pd.DataFrame([
    dict(Task="프로젝트 A", Start='2023-01-01', Finish='2023-02-28', Resource="개발팀"),
    dict(Task="프로젝트 B", Start='2023-03-05', Finish='2023-04-15', Resource="디자인팀"),
    dict(Task="프로젝트 C", Start='2023-02-20', Finish='2023-05-30', Resource="개발팀"),
    dict(Task="프로젝트 D", Start='2023-04-10', Finish='2023-06-20', Resource="기획팀")
])

# 타임라인(간트 차트) 생성
fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Resource",
                  title="프로젝트 타임라인 (Gantt Chart)")
fig.update_yaxes(autorange="reversed") # y축 순서 뒤집기
fig.show()
```
