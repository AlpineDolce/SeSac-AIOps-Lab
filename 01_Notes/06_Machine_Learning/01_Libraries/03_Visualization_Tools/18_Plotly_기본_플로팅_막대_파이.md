<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Plotly Express(`px`)를 사용하여 막대 그래프(`px.bar()`)와 파이 차트(`px.pie()`)를 그리는 방법을 다룹니다. Plotly의 인터랙티브 기능을 활용하여 범주형 데이터의 빈도, 값, 그리고 전체에 대한 각 부분의 비율을 동적으로 시각화하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 기본 플로팅 (Plotly Express)](#1-기본-플로팅-plotly-express)
  - [1.1. 막대 그래프 (`px.bar()`)](#11-막대-그래프-pxbar)
  - [1.2. 파이 차트 (`px.pie()`)](#12-파이-차트-pxpie)

---

## 1. 기본 플로팅 (Plotly Express)

### 1.1. 막대 그래프 (`px.bar()`)
범주형 데이터의 빈도나 값을 막대로 표현하며, 인터랙티브 기능을 통해 각 막대의 상세 정보를 확인할 수 있습니다.

```python
import plotly.express as px

# 내장 데이터셋 사용 (팁 데이터)
tips = px.data.tips()

# 요일별 총 계산액 평균
fig = px.bar(tips, x="day", y="total_bill", title="Average Total Bill by Day")
fig.show()

# 성별에 따른 요일별 팁 합계 (스택 막대 그래프)
fig = px.bar(tips, x="day", y="tip", color="sex", title="Total Tip by Day and Sex", barmode='group') # barmode='group'으로 그룹화
fig.show()
```

### 1.2. 파이 차트 (`px.pie()`)
전체에 대한 각 부분의 비율을 보여줄 때 사용합니다.

```python
import plotly.express as px

# 내장 데이터셋 사용 (팁 데이터)
tips = px.data.tips()

# 흡연 여부(smoker) 비율
fig = px.pie(tips, names='smoker', title="Proportion of Smokers and Non-Smokers")
fig.show()

# 요일별 팁 비율 (각 요일 내 흡연 여부 비율)
fig = px.pie(tips, values='tip', names='day', title="Tip Proportion by Day", hole=0.3) # hole로 도넛 차트 생성
fig.show()
```
