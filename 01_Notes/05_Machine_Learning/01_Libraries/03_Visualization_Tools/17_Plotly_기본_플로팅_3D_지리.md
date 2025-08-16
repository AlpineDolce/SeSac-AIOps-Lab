<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly Express(`px`)를 사용하여 3D 산점도(`px.scatter_3d()`)와 지리 정보 시각화(`px.scatter_mapbox()`)를 그리는 방법을 다룹니다. Plotly의 강력한 인터랙티브 3D 기능과 지도 시각화 기능을 활용하여 다차원 데이터 및 위치 기반 데이터를 동적으로 탐색하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

* [1. Plotly 3D 및 지리 플로팅 개요](#1-plotly-3d-및-지리-플로팅-개요)
  * [1.1. 3D 산점도 (`px.scatter_3d()`)](#11-3d-산점도-pxscatter_3d)
    * [1.1.1. 주요 특징](#111-주요-특징)
    * [1.1.2. 사용 시기](#112-사용-시기)
    * [1.1.3. 기본 사용법 및 커스터마이징](#113-기본-사용법-및-커스터마이징)
  * [1.2. 지리 정보 시각화 (`px.scatter_mapbox()`)](#12-지리-정보-시각화-pxscatter_mapbox)
    * [1.2.1. 주요 특징](#121-주요-특징)
    * [1.2.2. 사용 시기](#122-사용-시기)
    * [1.2.3. Plotly 외 지리 정보 시각화 생태계](#123-plotly-외-지리-정보-시각화-생태계)
    * [1.2.4. 실무적 중요성](#124-실무적-중요성)
    * [1.2.5. 기본 사용법 및 커스터마이징](#125-기본-사용법-및-커스터마이징)

---

## 1. Plotly 3D 및 지리 플로팅 개요
이 문서는 Plotly Express(`px`)를 사용하여 3D 산점도(`px.scatter_3d()`)와 지리 정보 시각화(`px.scatter_mapbox()`)를 그리는 방법을 다룹니다. Plotly의 강력한 인터랙티브 3D 기능과 지도 시각화 기능을 활용하여 다차원 데이터 및 위치 기반 데이터를 동적으로 탐색하는 방법을 실제 코드 예제를 통해 학습합니다.

### 1.1. 3D 산점도 (`px.scatter_3d()`)
3D 산점도(3D Scatter Plot)는 세 개 이상의 변수 간의 관계를 3차원 공간에 시각화하여 보여주는 플롯입니다. Plotly Express의 `px.scatter_3d()`는 이러한 3D 산점도를 쉽게 생성할 수 있게 해주며, Plotly의 강력한 인터랙티브 3D 기능을 통해 사용자가 직접 플롯을 회전, 확대/축소, 이동하며 데이터를 다각도에서 탐색할 수 있도록 지원합니다.

#### 1.1.1. 주요 특징:
*   **3차원 관계 시각화:** `x`, `y`, `z` 세 축에 각각 다른 연속형 변수를 매핑하여 데이터 포인트의 3차원 공간 내 위치를 표현합니다.
*   **인터랙티브 3D 탐색:** 생성된 3D 플롯은 마우스 드래그를 통해 자유롭게 회전할 수 있으며, 스크롤을 통해 확대/축소, 클릭을 통해 데이터 포인트 정보 확인(hover)이 가능합니다. 이는 복잡한 다차원 데이터의 패턴을 직관적으로 이해하는 데 매우 효과적입니다.
*   **다변수 인코딩:**
    *   `color`: 범주형 또는 연속형 변수에 따라 점의 색상을 다르게 하여 네 번째 차원을 표현합니다.
    *   `size`: 점의 크기를 다른 연속형 변수의 값에 비례하게 설정하여 다섯 번째 차원을 시각화합니다.
    *   `symbol`: 범주형 변수에 따라 점의 모양(심볼)을 다르게 하여 추가적인 차원을 표현합니다.
    *   `hover_data`: 마우스 오버 시 툴팁에 표시될 추가 정보를 지정합니다.

#### 1.1.2. 사용 시기:
*   세 개 이상의 연속형 변수 간의 복잡한 관계를 탐색하고 싶을 때.
*   데이터 내의 군집(cluster)이나 이상치(outlier)가 3차원 공간에서 어떻게 분포하는지 확인하고자 할 때.
*   데이터의 다차원적인 패턴을 직관적이고 동적으로 보여주고자 할 때.

#### 1.1.3. 기본 사용법 및 커스터마이징
```python
import plotly.express as px
import pandas as pd # pandas import 추가

# 예시 데이터 로드 (CSV 파일에서 로드하는 것을 가정)
# 실제 사용 시에는 pd.read_csv('iris.csv')와 같이 사용합니다.
iris = px.data.iris() # 내장 데이터셋 사용
# iris = pd.read_csv('iris.csv') # 실제 CSV 파일에서 로드하는 예시 (주석 처리)

# 3D 산점도 그리기 (기본: 색상으로 종 구분)
fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',
                    color='species', title="Iris Dataset 3D 산점도")
fig.show()

# 추가 예시: 크기와 심볼로 추가 차원 표현
tips = px.data.tips()
fig = px.scatter_3d(tips, x='total_bill', y='tip', z='size',
                    color='day', size='total_bill', symbol='smoker',
                    title="팁 데이터 3D 산점도 (요일, 총 계산액, 흡연 여부)")
fig.show()
```

### 1.2. 지리 정보 시각화 (`px.scatter_mapbox()`)
`px.scatter_mapbox()`는 위도(latitude)와 경도(longitude) 데이터를 지도 위에 점으로 표현하여 지리 정보 데이터를 시각화하는 데 사용됩니다. Plotly의 강력한 지도 시각화 기능을 활용하여 위치 기반 데이터를 인터랙티브하게 탐색하고, 데이터의 공간적 분포나 패턴을 파악하는 데 매우 유용합니다.

#### 1.2.1. 주요 특징:
*   **인터랙티브 지도:** 생성된 지도는 확대/축소, 이동, 회전 등 다양한 인터랙티브 기능을 지원하여 사용자가 원하는 지역을 상세하게 탐색할 수 있습니다.
*   **다양한 지도 스타일:** `mapbox_style` 파라미터를 통해 다양한 지도 배경 스타일을 선택할 수 있습니다. "open-street-map"은 별도의 Mapbox API 토큰 없이 사용할 수 있는 편리한 옵션입니다.
*   **다변수 인코딩:**
    *   `lat`, `lon`: 데이터 포인트의 위도와 경도를 지정합니다.
    *   `color`: 범주형 또는 연속형 변수에 따라 점의 색상을 다르게 하여 추가적인 차원을 표현합니다.
    *   `size`: 점의 크기를 다른 연속형 변수의 값에 비례하게 설정하여 데이터의 중요도나 규모를 시각화합니다.
    *   `hover_data`, `hover_name`: 마우스 오버 시 툴팁에 표시될 추가 정보를 지정합니다.
    *   `zoom`: 지도의 초기 확대 레벨을 설정합니다.

#### 1.2.2. 사용 시기:
*   위치 기반 데이터(예: 매장 위치, 사건 발생 지점, 센서 데이터)의 공간적 분포를 시각화하고 싶을 때.
*   지리적 영역 내에서 데이터의 밀집도나 패턴을 탐색하고자 할 때.
*   지도 위에 특정 지표의 값을 색상이나 크기로 표현하여 공간적 관계를 분석하고자 할 때.

#### 1.2.3. Plotly 외 지리 정보 시각화 생태계:
`px.scatter_mapbox()`는 빠르고 쉽게 지리 정보를 시각화하는 데 훌륭하지만, 본격적인 지리 정보 분석 및 시각화를 위해서는 파이썬의 다른 강력한 라이브러리들을 함께 고려하는 것이 좋습니다.
*   **GeoPandas:** Pandas DataFrame과 유사한 `GeoDataFrame` 객체를 제공하여 공간 데이터를 효율적으로 다룰 수 있게 합니다. 공간 데이터의 읽기/쓰기, 조작, 분석 등 GIS(Geographic Information System)의 핵심 기능을 파이썬 환경에서 수행할 수 있습니다.
*   **Folium:** JavaScript 라이브러리인 Leaflet.js를 기반으로 인터랙티브 지도를 생성하는 데 특화되어 있습니다. 지도 위에 마커, 원, 다각형, Choropleth map(지역별 통계값 색상 표시) 등을 쉽게 추가할 수 있으며, 웹 환경에서 공유하기 용이합니다.

#### 1.2.4. 실무적 중요성:
부동산, 물류, 환경 데이터, 도시 계획, 역학 조사 등 위치 정보를 다루는 다양한 분야의 분석에서는 GeoPandas와 Folium이 데이터 처리부터 고급 시각화까지 거의 표준처럼 사용됩니다. Plotly는 인터랙티브한 웹 기반 시각화에 강점을 가지며, 이들 라이브러리와 상호 보완적으로 활용될 수 있습니다.

#### 1.2.5. 기본 사용법 및 커스터마이징
```python
import plotly.express as px

# 내장 데이터셋 사용 (캐나다 카셰어링 위치 데이터)
carshare = px.data.carshare()

# 지리 정보 산점도 그리기 (기본)
fig = px.scatter_mapbox(carshare, lat="centroid_lat", lon="centroid_lon", color="peak_hour",
                        size="car_hours", size_max=15, zoom=10,
                        mapbox_style="open-street-map",
                        title="몬트리올 카셰어링 위치 (시간대별 사용량)")
fig.show()

# 추가 예시: Choropleth Map (국가별 GDP) - px.choropleth 사용
gapminder = px.data.gapminder().query("year==2007")
fig = px.choropleth(gapminder, locations="iso_alpha", color="gdpPercap",
                    hover_name="country", color_continuous_scale=px.colors.sequential.Plasma,
                    title="2007년 국가별 1인당 GDP")
fig.show()
```
### 플롯 저장하기 (HTML 및 이미지)
Plotly로 생성된 인터랙티브 플롯은 HTML 파일로 저장하여 웹 브라우저에서 인터랙티브하게 공유할 수 있으며, 정적 이미지 파일(PNG, JPEG 등)로도 저장할 수 있습니다.

```python
import plotly.express as px

# 예시 플롯 생성
iris = px.data.iris()
fig_save_example = px.scatter_3d(iris, x="sepal_length", y="sepal_width", z="petal_width",
                                 color="species", title="저장 예시 플롯 (3D)")

# HTML 파일로 저장 (인터랙티브 기능 유지)
html_file_path = "plotly_3d_scatter_example.html"
fig_save_example.write_html(html_file_path)
print(f"'{html_file_path}' 파일이 저장되었습니다.")

# PNG 이미지로 저장 (정적 이미지)
# 참고: 정적 이미지 저장을 위해서는 `kaleido` 라이브러리 설치가 필요합니다 (`pip install kaleido`).
png_file_path = "plotly_3d_scatter_example.png"
try:
    fig_save_example.write_image(png_file_path)
    print(f"'{png_file_path}' 파일이 저장되었습니다.")
except ValueError:
    print(f"Kaleido 라이브러리가 설치되어 있지 않아 이미지 저장을 건너뜁니다. 'pip install kaleido'를 실행하세요.")
```

더 자세한 플롯 저장 및 고급 기능에 대한 내용은 [17_Plotly_인터랙티브_서브플롯_저장.md](17_Plotly_인터랙티브_서브플롯_저장.md) 문서를 참고하십시오.

### 다른 3D 및 지리 플롯 유형 (참고)
이 문서에서는 `px.scatter_3d()`, `px.scatter_mapbox()`, `px.choropleth()`와 같은 기본적인 3D 및 지리 플롯을 다루었습니다. Plotly Express는 이 외에도 다양한 유형의 3D 및 지리 플롯을 제공합니다.

*   **다른 3D 플롯:** `px.line_3d()` (3D 라인 플롯), `px.surface()` (3D 표면 플롯), `px.mesh()` (3D 메시 플롯) 등이 있습니다.
*   **다른 지리 플롯:** `px.density_mapbox()` (지도 위 밀도 플롯), `px.line_mapbox()` (지도 위 라인 플롯), `px.scatter_geo()` (국가 경계선 기반 산점도), `px.line_geo()` (국가 경계선 기반 라인 플롯), `px.choropleth_mapbox()` (Mapbox 기반 코로플레스 맵) 등이 있습니다.

이러한 플롯들은 특정 데이터 유형이나 분석 목적에 따라 유용하게 활용될 수 있습니다. 더 자세한 정보는 Plotly 공식 문서([https://plotly.com/python/](https://plotly.com/python/))를 참조하시기 바랍니다.
