# 데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드

이 가이드는 파이썬의 주요 데이터 시각화 라이브러리인 Matplotlib, Seaborn, Plotly의 핵심 개념과 활용법을 상세히 다룹니다. 각 라이브러리의 특징, 설치 방법, 기본적인 플로팅부터 고급 시각화 기법까지 다양한 예제를 통해 설명하며, 머신러닝 및 딥러닝 프로젝트에서 데이터를 효과적으로 탐색하고 결과를 시각화하는 데 필요한 지식을 제공합니다.

---

### Part 1: 데이터 시각화 소개

-   [**01_데이터_시각화_소개.md**](./01_데이터_시각화_소개.md): 데이터 시각화의 중요성과 파이썬 시각화 생태계를 이해합니다.

### Part 2: Matplotlib - The Foundation

-   [**02_Matplotlib_소개_및_설치.md**](./02_Matplotlib_소개_및_설치.md): Matplotlib의 기본 개념과 설치 방법을 학습합니다.
-   [**03_Matplotlib_기본_플로팅_라인_산점도.md**](./03_Matplotlib_기본_플로팅_라인_산점도.md): 라인 플롯과 산점도의 기본 사용법을 익힙니다.
-   [**04_Matplotlib_기본_플로팅_막대_히스토그램.md**](./04_Matplotlib_기본_플로팅_막대_히스토그램.md): 막대 그래프와 히스토그램의 기본 사용법을 익힙니다.
-   [**05_Matplotlib_기본_플로팅_등고선_박스_파이.md**](./05_Matplotlib_기본_플로팅_등고선_박스_파이.md): 등고선 플롯, 박스 플롯, 파이 차트의 기본 사용법을 익힙니다.
-   [**06_Matplotlib_기본_플로팅_에러바_2D히스토그램.md**](./06_Matplotlib_기본_플로팅_에러바_2D히스토그램.md): 에러 바, 2D 히스토그램 및 헥사곤 비닝의 기본 사용법을 익힙니다.
-   [**07_Matplotlib_플롯_사용자_정의.md**](./07_Matplotlib_플롯_사용자_정의.md): 플롯의 제목, 축 레이블, 범례, 색상, 마커, 선 스타일, 축 범위 및 눈금 설정 등 사용자 정의 방법을 학습합니다.
-   [**08_Matplotlib_서브플롯_저장_객체지향.md**](./08_Matplotlib_서브플롯_저장_객체지향.md): 서브플롯 생성, 플롯 저장, 객체 지향 API 사용법을 학습합니다.
-   [**09_Matplotlib_특수_플롯.md**](./09_Matplotlib_특수_플롯.md): 이미지 시각화(`imshow`) 및 3D 시각화(`mplot3d`) 등 특수 플롯을 다룹니다.

### Part 3: Seaborn - Statistical Visualization

-   [**10_Seaborn_소개_및_설치.md**](./10_Seaborn_소개_및_설치.md): Seaborn의 기본 개념과 설치 방법을 학습합니다.
-   [**11_Seaborn_관계형_플롯.md**](./11_Seaborn_관계형_플롯.md): `scatterplot()` 및 `lineplot()`을 이용한 관계형 플롯을 다룹니다.
-   [**12_Seaborn_분포_플롯.md**](./12_Seaborn_분포_플롯.md): `histplot()`, `kdeplot()`, `displot()`을 이용한 분포 플롯을 다룹니다.
-   [**13_Seaborn_범주형_플롯.md**](./13_Seaborn_범주형_플롯.md): `boxplot()`, `violinplot()`, `stripplot()`, `swarmplot()`, `barplot()`, `countplot()`을 이용한 범주형 플롯을 다룹니다.
-   [**14_Seaborn_회귀_행렬_플롯.md**](./14_Seaborn_회귀_행렬_플롯.md): `regplot()`, `lmplot()`을 이용한 회귀 플롯과 `heatmap()`, `clustermap()`을 이용한 행렬 플롯을 다룹니다.
-   [**15_Seaborn_다변수_고급_사용자정의.md**](./15_Seaborn_다변수_고급_사용자정의.md): `pairplot()`, `jointplot()`을 이용한 다변수 분석 플롯과 `FacetGrid`를 이용한 고급 플롯 제어, 사용자 정의 및 테마 설정을 다룹니다.

### Part 4: Plotly - Interactive Visualization

-   [**16_Plotly_소개_및_설치.md**](./16_Plotly_소개_및_설치.md): Plotly의 기본 개념과 설치 방법을 학습합니다.
-   [**17_Plotly_기본_플로팅_산점도_라인.md**](./17_Plotly_기본_플로팅_산점도_라인.md): `px.scatter()` 및 `px.line()`을 이용한 기본 플로팅을 다룹니다.
-   [**18_Plotly_기본_플로팅_막대_파이.md**](./18_Plotly_기본_플로팅_막대_파이.md): `px.bar()` 및 `px.pie()`을 이용한 기본 플로팅을 다룹니다.
-   [**19_Plotly_기본_플로팅_3D_지리.md**](./19_Plotly_기본_플로팅_3D_지리.md): `px.scatter_3d()` 및 `px.scatter_mapbox()`을 이용한 3D 및 지리 정보 시각화를 다룹니다.
-   [**20_Plotly_인터랙티브_서브플롯_저장.md**](./20_Plotly_인터랙티브_서브플롯_저장.md): 인터랙티브 플롯 기능, 서브플롯 생성, 플롯 저장 방법을 학습합니다.
-   [**21_Plotly_대시보드_GraphObjects.md**](./21_Plotly_대시보드_GraphObjects.md): Dash 및 Streamlit을 이용한 인터랙티브 대시보드 구축과 Plotly Graph Objects 심화를 다룹니다.

### Part 5: 라이브러리 비교 및 사용 가이드

-   [**22_라이브러리_비교_및_사용_가이드.md**](./22_라이브러리_비교_및_사용_가이드.md): Matplotlib, Seaborn, Plotly의 비교, 최적의 라이브러리 선택 가이드, 데이터 시각화 모범 사례를 다룹니다.
