<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly Express(`px`)를 사용하여 3D 산점도(`px.scatter_3d()`)와 지리 정보 시각화(`px.scatter_mapbox()`)를 그리는 방법을 다룹니다. Plotly의 강력한 인터랙티브 3D 기능과 지도 시각화 기능을 활용하여 다차원 데이터 및 위치 기반 데이터를 동적으로 탐색하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 기본 플로팅 (Plotly Express)](#1-기본-플로팅-plotly-express)
  - [1.1. 3D 산점도 (`px.scatter_3d()`)](#11-3d-산점도-pxscatter_3d)
  - [1.2. 지리 정보 시각화 (`px.scatter_mapbox()`)](#12-지리-정보-시각화-pxscatter_mapbox)

---

## 1. 기본 플로팅 (Plotly Express)

### 1.1. 3D 산점도 (`px.scatter_3d()`)
Plotly는 인터랙티브한 3D 시각화에 매우 강력합니다. `px.scatter_3d`를 사용하면 3차원 공간에 데이터를 표현하고, 마우스로 회전, 확대/축소하며 다각도에서 데이터를 탐색할 수 있습니다.

```python
import plotly.express as px

iris = px.data.iris()

fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',
                    color='species', title="3D Scatter Plot of Iris Dataset")
fig.show()
```

### 1.2. 지리 정보 시각화 (`px.scatter_mapbox()`)
Plotly는 지도 위에 데이터를 시각화하는 강력한 기능을 제공합니다. `px.scatter_mapbox`를 사용하면 위도, 경도 데이터를 지도 위의 점으로 표현할 수 있습니다. `mapbox_style`을 "open-street-map"으로 설정하면 별도의 API 토큰 없이 사용할 수 있습니다.

```python
import plotly.express as px

# 내장 데이터셋 사용 (캐나다 카셰어링 위치 데이터)
carshare = px.data.carshare()

fig = px.scatter_mapbox(carshare, lat="centroid_lat", lon="centroid_lon", color="peak_hour",
                        size="car_hours", size_max=15, zoom=10,
                        mapbox_style="open-street-map",
                        title="Carshare Locations in Montreal")
fig.show()
```

`px.scatter_mapbox` 예제는 좋지만, 본격적인 지리 정보 분석 및 시각화를 위해서는 관련 생태계를 함께 언급하는 것이 좋습니다. **GeoPandas**는 Pandas DataFrame과 유사한 GeoDataFrame 객체로 공간 데이터를 다루는 핵심 라이브러리이며, **Folium**은 leaflet.js 기반의 인터랙티브 지도를 만드는 데 특화되어 있습니다. (예: 지도 위에 Choropleth map, 마커 클러스터링 등 구현)

**실무적 중요성**
부동산, 물류, 환경 데이터 등 위치 정보를 다루는 분석에서는 GeoPandas와 Folium이 거의 표준처럼 사용됩니다.
