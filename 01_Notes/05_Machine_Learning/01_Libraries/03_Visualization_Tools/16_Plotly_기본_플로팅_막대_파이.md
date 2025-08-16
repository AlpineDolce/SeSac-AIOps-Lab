<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly Express(`px`)를 사용하여 막대 그래프(`px.bar()`), 파이 차트(`px.pie()`), 그리고 히트맵(`px.imshow`)을 그리는 방법을 다룹니다. Plotly의 인터랙티브 기능을 활용하여 범주형 데이터, 비율 데이터, 행렬 데이터를 동적으로 시각화하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

* [1. Plotly 기본 플로팅 개요](#1-plotly-기본-플로팅-개요)
  * [1.1. 막대 그래프 (`px.bar()`)](#11-막대-그래프-pxbar)
    * [1.1.1. 주요 특징](#111-주요-특징)
    * [1.1.2. 사용 시기](#112-사용-시기)
    * [1.1.3. 기본 사용법 및 커스터마이징](#113-기본-사용법-및-커스터마이징)
  * [1.2. 파이 차트 (`px.pie()`)](#12-파이-차트-pxpie)
    * [1.2.1. 주요 특징](#121-주요-특징)
    * [1.2.2. 사용 시기](#122-사용-시기)
    * [1.2.3. 기본 사용법 및 커스터마이징](#123-기본-사용법-및-커스터마이징)
  * [1.3. 히트맵 및 밀도 플롯 (`px.imshow`, `px.density_heatmap`)](#13-히트맵-및-밀도-플롯-pximshow-pxdensity_heatmap)
    * [1.3.1. 주요 특징](#131-주요-특징)
    * [1.3.2. 사용 시기](#132-사용-시기)
    * [1.3.3. 기본 사용법 및 커스터마이징](#133-기본-사용법-및-커스터마이징)

---

## 1. Plotly 기본 플로팅 개요
Plotly Express(`px`)는 Pandas DataFrame을 기반으로 다양한 유형의 인터랙티브 그래프를 쉽게 생성할 수 있는 고수준 인터페이스입니다. 이 섹션에서는 `px.bar()`를 이용한 막대 그래프와 `px.pie()`를 이용한 파이 차트의 기본 사용법과 고급 기능을 다룹니다. Plotly의 인터랙티브 기능을 활용하여 범주형 데이터의 빈도, 값, 그리고 전체에 대한 각 부분의 비율을 동적으로 시각화하는 방법을 학습합니다.

### 1.1. 막대 그래프 (`px.bar()`)
막대 그래프(Bar Chart)는 범주형 데이터의 빈도, 합계, 평균 등 특정 값을 막대의 길이로 표현하여 여러 범주 간의 비교를 용이하게 하는 플롯입니다. Plotly Express의 `px.bar()`는 인터랙티브한 막대 그래프를 쉽게 생성할 수 있게 해주며, 다양한 옵션을 통해 데이터를 다각도로 분석할 수 있습니다.

#### 1.1.1. 주요 특징:
*   **범주형 데이터 비교:** 각 막대는 특정 범주를 나타내며, 막대의 길이는 해당 범주의 값을 의미합니다. 이를 통해 범주 간의 상대적인 크기를 직관적으로 비교할 수 있습니다.
*   **인터랙티브 기능:** 마우스 오버 시 상세 정보 표시, 확대/축소, 이동 등 Plotly의 기본 인터랙티브 기능을 지원합니다.
*   **다변수 인코딩:**
    *   `x`, `y`: 막대 그래프의 축을 정의합니다. `x`에 범주형, `y`에 연속형 변수를 주로 사용합니다.
    *   `color`: 다른 범주형 변수에 따라 막대의 색상을 다르게 하여 그룹 내 비교를 가능하게 합니다.
    *   `barmode`: `color` 파라미터와 함께 사용될 때 막대의 표시 방식을 결정합니다.
        *   `'group'` (기본값): 그룹별로 막대를 나란히 표시합니다.
        *   `'stack'`: 그룹별로 막대를 쌓아서 표시합니다 (전체 합계 파악에 용이).
        *   `'overlay'`: 막대를 겹쳐서 표시합니다 (투명도 조절 필요).
    *   `text`: 막대 위에 직접 값을 표시할 수 있습니다.
    *   `hover_data`: 마우스 오버 시 툴팁에 표시될 추가 정보를 지정합니다.

#### 1.1.2. 사용 시기:
*   여러 범주 간의 수량, 빈도, 평균 등을 비교하고 싶을 때.
*   시간에 따른 범주형 데이터의 변화를 보여줄 때 (시계열 막대 그래프).
*   전체에 대한 각 부분의 기여도를 스택 막대 그래프로 보여줄 때.

#### 1.1.3. 기본 사용법 및 커스터마이징
```python
import plotly.express as px
import pandas as pd # pandas import 추가

# 예시 데이터 로드 (CSV 파일에서 로드하는 것을 가정)
# 실제 사용 시에는 pd.read_csv('tips.csv')와 같이 사용합니다.
tips = px.data.tips() # 내장 데이터셋 사용
# tips = pd.read_csv('tips.csv') # 실제 CSV 파일에서 로드하는 예시 (주석 처리)

# 요일별 총 계산액 평균 (기본)
fig = px.bar(tips, x="day", y="total_bill", title="요일별 총 계산액 평균")
fig.show()

# 성별에 따른 요일별 팁 합계 (그룹 막대 그래프)
fig = px.bar(tips, x="day", y="tip", color="sex", title="요일 및 성별에 따른 팁 합계", barmode='group')
fig.show()

# 추가 예시: 스택 막대 그래프 (흡연 여부에 따른 요일별 총 계산액)
fig = px.bar(tips, x="day", y="total_bill", color="smoker", title="요일 및 흡연 여부에 따른 총 계산액", barmode='stack')
fig.show()

# 추가 예시: 텍스트 표시 및 정렬
df = px.data.gapminder().query("year == 2007").sort_values("pop", ascending=False)[:10]
fig = px.bar(df, x="pop", y="country", orientation='h', text='pop', title="2007년 인구 상위 10개국")
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside') # 텍스트 포맷 및 위치
fig.show()
```

### 1.2. 파이 차트 (`px.pie()`)
파이 차트(Pie Chart)는 전체에 대한 각 부분의 비율을 시각적으로 보여줄 때 사용되는 플롯입니다. 각 '조각(slice)'의 크기는 해당 범주가 전체에서 차지하는 비율을 나타냅니다. Plotly Express의 `px.pie()`는 인터랙티브한 파이 차트를 쉽게 생성하며, 도넛 차트 형태로도 표현할 수 있습니다.

#### 1.2.1. 주요 특징:
*   **비율 시각화:** 범주형 데이터의 각 항목이 전체에서 차지하는 상대적인 비율을 직관적으로 비교할 수 있습니다.
*   **인터랙티브 기능:** 마우스 오버 시 각 조각의 이름, 값, 비율 등을 상세하게 확인할 수 있습니다.
*   **도넛 차트 (`hole`):** `hole` 파라미터를 사용하여 파이 차트의 중앙을 비워 도넛 차트 형태로 만들 수 있습니다. 이는 시각적인 다양성을 제공하며, 중앙에 추가 정보를 표시할 공간을 마련할 수 있습니다.
*   **다변수 인코딩:**
    *   `names`: 각 조각의 이름을 정의하는 범주형 변수입니다.
    *   `values`: 각 조각의 크기를 결정하는 연속형 변수입니다. 이 값들의 합계가 전체를 구성합니다.
    *   `color`: 다른 범주형 변수에 따라 조각의 색상을 다르게 할 수 있습니다.
    *   `title`: 차트의 제목을 설정합니다.

#### 1.2.2. 사용 시기:
*   전체에 대한 각 부분의 비율을 명확하게 보여주고 싶을 때.
*   범주의 수가 적고, 각 범주의 비율 차이가 클 때 효과적입니다.
*   (주의) 범주의 수가 너무 많거나 각 범주의 비율이 비슷할 때는 파이 차트 대신 막대 그래프나 다른 시각화 방법을 고려하는 것이 좋습니다.

#### 1.2.3. 기본 사용법 및 커스터마이징
```python
import plotly.express as px

# 내장 데이터셋 사용 (팁 데이터)
tips = px.data.tips()

# 흡연 여부(smoker) 비율 (기본 파이 차트)
fig = px.pie(tips, names='smoker', title="흡연 여부 비율")
fig.show()

# 요일별 팁 비율 (도넛 차트)
fig = px.pie(tips, values='tip', names='day', title="요일별 팁 비율", hole=0.3) # hole로 도넛 차트 생성
fig.show()

# 추가 예시: 대륙별 인구 비율 (Gapminder 데이터셋)
gapminder = px.data.gapminder().query("year == 2007")
fig = px.pie(gapminder, values='pop', names='continent', title="2007년 대륙별 인구 비율",
             color_discrete_sequence=px.colors.sequential.RdBu) # 색상 팔레트 지정
fig.show()
```

### 1.3. 히트맵 및 밀도 플롯 (`px.imshow`, `px.density_heatmap`)
히트맵은 행렬 데이터를 색상으로 표현하여 데이터의 패턴이나 관계를 한눈에 파악할 수 있게 하는 시각화 도구입니다. Plotly Express는 `px.imshow`와 `px.density_heatmap`을 통해 인터랙티브한 히트맵을 쉽게 생성할 수 있습니다.

#### 1.3.1. 주요 특징:
*   **`px.imshow`**: 2D 배열(행렬) 데이터를 이미지 형태로 시각화합니다. 각 셀의 값은 특정 색상에 매핑됩니다. 상관 행렬, 혼동 행렬, 또는 실제 이미지 데이터를 시각화하는 데 매우 유용합니다.
*   **`px.density_heatmap`**: 두 연속형 변수의 분포를 2D 히스토그램 또는 밀도 플롯으로 표현합니다. 산점도에서 데이터 포인트가 너무 많아 발생하는 오버플로팅 문제를 해결하는 데 효과적입니다.
*   **인터랙티브 기능**: 마우스 오버 시 각 셀의 값(x, y, z)을 확인할 수 있으며, 색상 막대(colorbar)를 통해 값의 범위를 쉽게 파악할 수 있습니다.

#### 1.3.2. 사용 시기:
*   **`px.imshow`**: 변수 간의 상관 관계, 모델의 혼동 행렬, 또는 2차원 배열 형태의 모든 데이터를 시각화할 때.
*   **`px.density_heatmap`**: 대규모 데이터셋의 산점도를 그려야 할 때, 점들이 겹쳐 분포를 파악하기 어려운 경우.

#### 1.3.3. 기본 사용법 및 커스터마이징
```python
import plotly.express as px
import numpy as np

# px.imshow 예시 (상관 행렬)
flights = px.data.flights()
flights_pivot = flights.pivot_table(index='month', columns='year', values='passengers')
fig = px.imshow(flights_pivot, text_auto=True, aspect="auto",
                title="월별/연도별 승객 수 (imshow)")
fig.show()

# px.density_heatmap 예시
tips = px.data.tips()
fig = px.density_heatmap(tips, x="total_bill", y="tip", marginal_x="rug", marginal_y="histogram",
                         title="총 계산액과 팁의 밀도 히트맵")
fig.show()
```

### 플롯 저장하기 (HTML 및 이미지)
Plotly로 생성된 인터랙티브 플롯은 HTML 파일로 저장하여 웹 브라우저에서 인터랙티브하게 공유할 수 있으며, 정적 이미지 파일(PNG, JPEG 등)으로도 저장할 수 있습니다.

```python
import plotly.express as px

# 예시 플롯 생성
tips = px.data.tips()
fig_save_example = px.bar(tips, x="day", y="total_bill", title="저장 예시 플롯")

# HTML 파일로 저장 (인터랙티브 기능 유지)
html_file_path = "plotly_bar_example.html"
fig_save_example.write_html(html_file_path)
print(f"'{html_file_path}' 파일이 저장되었습니다.")

# PNG 이미지로 저장 (정적 이미지)
# 참고: 정적 이미지 저장을 위해서는 `kaleido` 라이브러리 설치가 필요합니다 (`pip install kaleido`).
png_file_path = "plotly_bar_example.png"
try:
    fig_save_example.write_image(png_file_path)
    print(f"'{png_file_path}' 파일이 저장되었습니다.")
except ValueError:
    print(f"Kaleido 라이브러리가 설치되어 있지 않아 이미지 저장을 건너뜁니다. 'pip install kaleido'를 실행하세요.")
```

더 자세한 플롯 저장 및 고급 기능에 대한 내용은 [17_Plotly_인터랙티브_서브플롯_저장.md](17_Plotly_인터랙티브_서브플롯_저장.md) 문서를 참고하십시오.