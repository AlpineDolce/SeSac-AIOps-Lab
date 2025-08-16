<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly Express(`px`)를 사용하여 산점도(`px.scatter()`)와 라인 플롯(`px.line()`)을 그리는 방법을 다룹니다. Plotly의 인터랙티브 기능과 애니메이션 프레임을 활용하여 데이터의 관계와 시간 변화에 따른 추세를 동적으로 시각화하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

* [1. Plotly 기본 플로팅 개요](#1-plotly-기본-플로팅-개요)
  * [1.1. 산점도 (`px.scatter()`)](#11-산점도-pxscatter)
    * [1.1.1. 주요 특징](#111-주요-특징)
    * [1.1.2. 사용 시기](#112-사용-시기)
    * [1.1.3. 기본 사용법 및 커스터마이징](#113-기본-사용법-및-커스터마이징)
  * [1.2. 라인 플롯 (`px.line()`)](#12-라인-플롯-pxline)
    * [1.2.1. 주요 특징](#121-주요-특징)
    * [1.2.2. 사용 시기](#122-사용-시기)
    * [1.2.3. 기본 사용법 및 커스터마이징](#123-기본-사용법-및-커스터마이징)

---

## 1. Plotly 기본 플로팅 개요
Plotly Express(`px`)는 Pandas DataFrame을 기반으로 다양한 유형의 인터랙티브 그래프를 쉽게 생성할 수 있는 고수준 인터페이스입니다. 이 섹션에서는 `px.scatter()`를 이용한 산점도와 `px.line()`를 이용한 라인 플롯의 기본 사용법과 고급 기능을 다룹니다. Plotly의 강력한 인터랙티브 기능과 애니메이션 프레임을 활용하여 데이터의 관계와 시간 변화에 따른 추세를 동적으로 시각화하는 방법을 학습합니다.

### 1.1. 산점도 (`px.scatter()`)
산점도(Scatter Plot)는 두 연속형 변수 간의 관계를 점으로 표현하여 시각화하는 가장 기본적인 플롯 중 하나입니다. Plotly Express의 `px.scatter()`는 이러한 산점도를 쉽게 생성할 수 있게 해주며, Plotly의 강력한 인터랙티브 기능과 애니메이션 기능을 기본적으로 제공하여 데이터 탐색을 더욱 풍부하게 만듭니다.

#### 1.1.1. 주요 특징:
*   **인터랙티브 기능:** 생성된 플롯은 확대/축소(zoom), 이동(pan), 데이터 포인트 정보 확인(hover), 선택(select) 등 다양한 인터랙티브 기능을 지원합니다.
*   **다변수 인코딩:**
    *   `x`, `y`: 산점도의 기본 축을 정의합니다.
    *   `color`: 범주형 또는 연속형 변수에 따라 점의 색상을 다르게 하여 추가적인 차원을 표현합니다.
    *   `size`: 점의 크기를 다른 연속형 변수의 값에 비례하게 설정하여 또 다른 차원을 시각화합니다.
    *   `hover_data`, `hover_name`: 마우스 오버 시 표시될 추가 데이터 정보나 이름을 지정합니다.
*   **애니메이션 프레임 (`animation_frame`, `animation_group`):** 시간에 따른 데이터의 변화를 동적으로 보여주는 애니메이션을 생성할 수 있습니다.
    *   `animation_frame`: 애니메이션의 각 프레임을 정의하는 시간 또는 순서 변수입니다.
    *   `animation_group`: 애니메이션 프레임이 변경될 때 동일한 객체로 유지되어야 하는 그룹을 정의합니다.
*   **패싯 플롯 (`facet_col`, `facet_row`):** `col` 또는 `row` 파라미터를 사용하여 범주형 변수에 따라 여러 개의 서브플롯(패싯)을 생성할 수 있습니다.

#### 1.1.2. 사용 시기:
*   두 연속형 변수 간의 관계(상관 관계, 군집 등)를 탐색하고 싶을 때.
*   데이터의 패턴이나 이상치를 시각적으로 식별하고자 할 때.
*   시간이나 다른 범주형 변수에 따른 데이터의 동적인 변화를 보여주고자 할 때.

#### 1.1.3. 기본 사용법 및 커스터마이징
```python
import plotly.express as px
import pandas as pd # pandas import 추가

# 예시 데이터 로드 (CSV 파일에서 로드하는 것을 가정)
# 실제 사용 시에는 pd.read_csv('iris.csv')와 같이 사용합니다.
iris = px.data.iris() # 내장 데이터셋 사용
# iris = pd.read_csv('iris.csv') # 실제 CSV 파일에서 로드하는 예시 (주석 처리)

# 꽃잎 길이(petal_length)와 너비(petal_width)의 산점도 (기본)
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

# 추가 예시: 패싯 플롯 (대륙별 GDP와 기대 수명)
fig = px.scatter(gapminder, x="gdpPercap", y="lifeExp", color="continent",
                 facet_col="continent", facet_col_wrap=3,
                 title="GDP per Capita vs. Life Expectancy by Continent")
fig.show()
```

### 1.2. 라인 플롯 (`px.line()`)
라인 플롯(Line Plot)은 시계열 데이터나 연속적인 데이터의 추세와 변화를 시각화하는 데 가장 적합한 플롯입니다. Plotly Express의 `px.line()`는 시간의 흐름에 따른 데이터의 변화, 여러 그룹 간의 추세 비교 등을 인터랙티브하게 보여줄 수 있어 동적인 데이터 분석에 매우 유용합니다.

#### 1.2.1. 주요 특징:
*   **추세 시각화:** `x`축에 시간이나 순서가 있는 변수를, `y`축에 측정값을 배치하여 데이터의 변화 추이를 명확하게 보여줍니다.
*   **인터랙티브 기능:** 산점도와 마찬가지로 확대/축소, 이동, 마우스 오버 시 정보 확인 등 Plotly의 기본 인터랙티브 기능을 제공합니다.
*   **다중 라인 및 그룹화:**
    *   `color`: 범주형 변수를 `color` 파라미터에 지정하여 각 범주별로 다른 색상의 라인을 그릴 수 있습니다.
    *   `line_group`: `x`축에 동일한 값을 가지는 여러 라인이 있을 때, 각 라인을 고유하게 식별하고 그룹화하여 올바른 라인 연결을 보장합니다.
*   **애니메이션 지원:** `animation_frame`을 사용하여 시간에 따른 라인 플롯의 변화를 애니메이션으로 표현할 수도 있습니다.

#### 1.2.2. 사용 시기:
*   주식 가격, 기온 변화, 인구 성장률 등 시간의 흐름에 따른 데이터의 추세를 분석하고 싶을 때.
*   여러 그룹(예: 국가, 제품) 간의 특정 지표 변화 추이를 비교하고자 할 때.
*   연속적인 데이터 포인트 간의 관계나 패턴을 시각적으로 탐색하고자 할 때.

#### 1.2.3. 기본 사용법 및 커스터마이징
```python
import plotly.express as px

# 내장 데이터셋 사용 (Gapminder: 시간에 따른 국가별 인구, 기대 수명, GDP)
gapminder = px.data.gapminder()

# 시간에 따른 아프가니스탄의 기대 수명 변화 (기본)
fig = px.line(gapminder.query("country=='Afghanistan'"), x="year", y="lifeExp", title="Life Expectancy in Afghanistan Over Time")
fig.show()

# 대륙별 기대 수명 추이 (색상 구분)
fig = px.line(gapminder.query("continent=='Asia'"), x="year", y="lifeExp", color="country", title="Life Expectancy in Asia by Country")
fig.show()

# 추가 예시: 시간에 따른 대륙별 GDP 변화 (line_group 사용)
fig = px.line(gapminder, x="year", y="gdpPercap", color="continent", line_group="country",
              title="GDP per Capita by Continent Over Time")
fig.show()

# 추가 예시: 시간에 따른 국가별 인구 변화 (애니메이션)
fig = px.line(gapminder, x="year", y="pop", color="country", animation_frame="year",
              animation_group="country", title="Population by Country Over Time")
fig.show()
```
### 플롯 저장하기 (HTML 및 이미지)
Plotly로 생성된 인터랙티브 플롯은 HTML 파일로 저장하여 웹 브라우저에서 인터랙티브하게 공유할 수 있으며, 정적 이미지 파일(PNG, JPEG 등)로도 저장할 수 있습니다.

```python
import plotly.express as px

# 예시 플롯 생성
iris = px.data.iris()
fig_save_example = px.scatter(iris, x="petal_length", y="petal_width", color="species",
                              title="저장 예시 플롯")

# HTML 파일로 저장 (인터랙티브 기능 유지)
html_file_path = "plotly_scatter_example.html"
fig_save_example.write_html(html_file_path)
print(f"'{html_file_path}' 파일이 저장되었습니다.")

# PNG 이미지로 저장 (정적 이미지)
# 참고: 정적 이미지 저장을 위해서는 `kaleido` 라이브러리 설치가 필요합니다 (`pip install kaleido`).
png_file_path = "plotly_scatter_example.png"
try:
    fig_save_example.write_image(png_file_path)
    print(f"'{png_file_path}' 파일이 저장되었습니다.")
except ValueError:
    print(f"Kaleido 라이브러리가 설치되어 있지 않아 이미지 저장을 건너뜁니다. 'pip install kaleido'를 실행하세요.")
```

더 자세한 플롯 저장 및 고급 기능에 대한 내용은 [17_Plotly_인터랙티브_서브플롯_저장.md](17_Plotly_인터랙티브_서브플롯_저장.md) 문서를 참고하십시오.
