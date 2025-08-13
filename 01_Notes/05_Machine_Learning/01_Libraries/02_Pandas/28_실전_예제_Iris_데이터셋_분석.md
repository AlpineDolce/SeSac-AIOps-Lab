<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 머신러닝의 "Hello, World!"와 같은 Iris 데이터셋을 활용하여, Pandas를 사용한 실전 데이터 분석 워크플로우를 학습합니다. 데이터 로딩부터 탐색, 필터링, 그룹화, 그리고 시각화를 통한 인사이트 도출까지, 데이터 분석의 전체 과정을 단계별로 따라가며 Pandas 활용 능력을 강화하는 것을 목표로 합니다.

<h2>목차</h2>

- [목차](#목차)
- [1. 실전 예제: Iris 데이터셋 분석](#1-실전-예제-iris-데이터셋-분석)
  - [1.1. 분석 개요: Iris 데이터셋이란?](#11-분석-개요-iris-데이터셋이란)
  - [1.2. 1단계: 데이터 로딩 및 기본 탐색](#12-1단계-데이터-로딩-및-기본-탐색)
  - [1.3. 2단계: 데이터 필터링 및 조건부 분석](#13-2단계-데이터-필터링-및-조건부-분석)
  - [1.4. 3단계: 그룹화 및 집계 분석](#14-3단계-그룹화-및-집계-분석)
  - [1.5. 4단계: 데이터 시각화를 통한 인사이트 발견](#15-4단계-데이터-시각화를-통한-인사이트-발견)
  - [1.6. 분석 결론](#16-분석-결론)

---

## 1. 실전 예제: Iris 데이터셋 분석

### 1.1. 분석 개요: Iris 데이터셋이란?

Iris(붓꽃) 데이터셋은 통계학과 머신러닝에서 분류(Classification) 문제의 예시로 가장 널리 사용되는 데이터셋 중 하나입니다. 붓꽃의 세 가지 품종(Setosa, Versicolor, Virginica)에 대한 꽃잎(petal)과 꽃받침(sepal)의 길이 및 너비 데이터를 담고 있으며, 각 품종이 뚜렷한 특징을 가지고 있어 데이터 분석 및 모델링 입문용으로 매우 적합합니다.

**분석 목표**: Pandas를 활용하여 Iris 데이터셋의 특징을 파악하고, 각 품종이 어떤 데이터 특성을 보이는지 탐색합니다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 예제 데이터 (실제 iris.csv 파일이라고 가정)
csv_data = """
sepal.length,sepal.width,petal.length,petal.width,variety
5.1,3.5,1.4,0.2,Setosa
4.9,3.0,1.4,0.2,Setosa
4.7,3.2,1.3,0.2,Setosa
7.0,3.2,4.7,1.4,Versicolor
6.4,3.2,4.5,1.5,Versicolor
6.9,3.1,4.9,1.5,Versicolor
6.3,3.3,6.0,2.5,Virginica
5.8,2.7,5.1,1.9,Virginica
7.1,3.0,5.9,2.1,Virginica
"""
# 실제 분석에서는 pd.read_csv('./data/iris.csv')와 같이 파일 경로를 사용합니다.
data = pd.read_csv(io.StringIO(csv_data))
```

### 1.2. 1단계: 데이터 로딩 및 기본 탐색

데이터 분석의 첫걸음은 데이터를 불러와 그 구조와 기본 정보를 파악하는 것입니다.

**1. 필드 정보 확인**: `data.info()`는 데이터셋의 전체적인 개요를 보여줍니다. 컬럼명, 각 컬럼의 데이터 개수(Non-null Count), 그리고 데이터 타입(Dtype)을 확인할 수 있어 결측치 유무를 빠르게 파악하는 데 유용합니다.
```python
print("=== 1. 데이터셋 필드 정보 ===")
data.info()
```
**결과 분석**: 총 9개의 데이터(entries)가 있으며, 5개의 컬럼 모두 결측치가 없음을 확인했습니다. 4개의 수치형 특성(float64)과 1개의 범주형 품종 정보(object)로 구성되어 있습니다.

**2. 데이터 미리보기**: `data.head()`는 데이터의 처음 몇 줄을 보여주어 실제 데이터가 어떻게 생겼는지 직관적으로 이해하게 돕습니다.
```python
print("\n=== 2. 데이터 미리보기 (상위 5개) ===")
print(data.head())
```

**3. 기술 통계 요약**: `data.describe()`는 수치형 데이터에 대한 핵심 통계량(개수, 평균, 표준편차, 최소/최대값, 사분위수)을 요약해 보여줍니다. 데이터의 분포와 스케일을 파악하는 데 필수적입니다.
```python
print("\n=== 3. 기술 통계 요약 ===")
print(data.describe())
```
**결과 분석**: 꽃받침 길이(`sepal.length`)의 평균은 약 6.1cm이고, 꽃잎 너비(`petal.width`)는 최소 0.2cm에서 최대 2.5cm까지 분포하는 등 각 특성의 대략적인 범위를 알 수 있습니다.

### 1.3. 2단계: 데이터 필터링 및 조건부 분석

전체 데이터가 아닌, 특정 조건을 만족하는 데이터만 추출하여 분석하면 더 깊은 인사이트를 얻을 수 있습니다.

**1. 특정 품종(`Setosa`) 데이터 분석**: `variety` 컬럼 값이 'Setosa'인 데이터만 필터링하여 통계량을 확인합니다.
```python
print("\n=== 4. 'Setosa' 품종 데이터 통계량 ===")
setosa_data = data[data["variety"] == 'Setosa']
print(setosa_data.describe())
```
**결과 분석**: Setosa 품종은 전체 평균에 비해 꽃잎(petal)의 길이와 너비가 현저히 작고, 꽃받침(sepal)의 너비는 오히려 약간 더 넓은 경향을 보입니다. 이처럼 특정 그룹의 데이터는 전체와 다른 특성을 가질 수 있습니다.

**2. 복합 조건 필터링**: 'Setosa' 품종이면서, 꽃받침 길이가 5cm 이상인 데이터를 추출합니다. Pandas에서는 `&` (AND), `|` (OR) 연산자를 사용하여 여러 조건을 조합할 수 있습니다.
```python
print("\n=== 5. 'Setosa'이면서 sepal.length >= 5인 데이터 ===")
condition_data = data[(data["variety"] == 'Setosa') & (data["sepal.length"] >= 5)]
print(f"조건을 만족하는 데이터 개수: {len(condition_data)}개")
print(condition_data)
```

### 1.4. 3단계: 그룹화 및 집계 분석

`groupby()`는 데이터를 특정 기준으로 그룹화하여 그룹별 통계량을 계산하는 강력한 기능입니다. 품종별 특성을 비교하는 데 매우 효과적입니다.

**1. 품종별 평균 계산**: 각 품종(`variety`)별로 모든 수치 특성의 평균을 계산합니다.
```python
print("\n=== 6. 각 품종별 평균 ===")
print(data.groupby('variety').mean())
```
**결과 분석**: 한눈에 봐도 Setosa는 다른 두 품종에 비해 모든 측정치가 작고, Virginica가 가장 큰 경향을 보입니다. 특히 `petal.length`와 `petal.width`에서 품종 간 차이가 두드러집니다.

**2. 특정 컬럼에 여러 집계 함수 적용**: `agg()` 함수를 사용하면 각 그룹에 대해 여러 통계량(예: 평균, 표준편차)을 한 번에 계산할 수 있습니다.
```python
print("\n=== 7. 품종별 sepal/petal length의 평균과 표준편차 ===")
agg_result = data.groupby('variety').agg({
    'sepal.length': ['mean', 'std'],
    'petal.length': ['mean', 'std']
})
print(agg_result)
```
**결과 분석**: 각 품종의 평균적인 크기뿐만 아니라, 데이터가 얼마나 퍼져 있는지도(표준편차) 비교할 수 있어 더욱 풍부한 분석이 가능합니다.

### 1.5. 4단계: 데이터 시각화를 통한 인사이트 발견

숫자만으로는 파악하기 어려운 데이터의 패턴과 관계를 시각화를 통해 명확하게 확인할 수 있습니다.

**1. Pair Plot으로 변수 간 관계 전체보기**: `seaborn.pairplot`은 데이터프레임의 모든 수치형 변수 쌍에 대한 산점도와 각 변수의 분포를 한 번에 그려줍니다. 품종(`hue='variety'`)에 따라 색을 다르게 표시하면 품종별 데이터 분포를 명확히 비교할 수 있습니다.
```python
print("\n=== 8. Pair Plot을 통한 시각적 탐색 ===")
sns.pairplot(data, hue='variety', palette='bright')
plt.suptitle('Iris Data Pair Plot', y=1.02)
plt.show()
```
**결과 분석**: Setosa(파란색 점)는 다른 두 품종과 명확히 구분됩니다. 반면 Versicolor와 Virginica는 일부 특성에서 겹치는 영역이 존재합니다. 특히 `petal.length`와 `petal.width`가 품종을 구분하는 데 매우 중요한 특성임을 시각적으로 확인할 수 있습니다.

**2. Box Plot으로 품종별 특성 분포 비교**: Box plot은 각 품종별 데이터의 분포(중앙값, 사분위수, 이상치)를 비교하는 데 효과적입니다.
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='variety', y='petal.length')
plt.title('Petal Length Distribution by Variety')
plt.show()
```
**결과 분석**: Box plot을 통해 Setosa의 꽃잎 길이는 매우 좁은 범위에 분포하는 반면, 다른 두 품종은 더 넓게 퍼져 있으며 값의 차이가 큼을 명확히 알 수 있습니다.

### 1.6. 분석 결론

이 실전 예제를 통해 Pandas를 활용하여 데이터를 불러오고, 탐색, 정제, 그룹화, 시각화하는 데이터 분석의 전 과정을 수행했습니다. 분석 결과, Iris의 세 품종은 꽃잎과 꽃받침의 측정치에서 뚜렷한 차이를 보이며, 특히 **꽃잎의 길이와 너비(`petal.length`, `petal.width`)가 품종을 구별하는 가장 핵심적인 특징**임을 확인했습니다. 이러한 탐색적 데이터 분석(EDA) 과정은 향후 머신러닝 모델을 구축할 때 어떤 특성을 중요하게 다룰지 결정하는 데 중요한 근거가 됩니다.

```