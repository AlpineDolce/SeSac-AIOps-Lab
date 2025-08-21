<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 머신러닝의 "Hello, World!"와 같은 Iris 데이터셋을 활용하여, Pandas를 사용한 실전 데이터 분석 워크플로우를 학습합니다. 데이터 로딩부터 탐색, 필터링, 그룹화, 그리고 시각화를 통한 인사이트 도출까지, 데이터 분석의 전체 과정을 단계별로 따라가며 Pandas 활용 능력을 강화하는 것을 목표로 합니다.

<h2>목차</h2>

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

Iris(붓꽃) 데이터셋은 통계학자 로널드 피셔(Ronald Fisher)가 1936년에 소개한, 데이터 과학 및 머신러닝 분야에서 가장 유명하고 널리 사용되는 데이터셋 중 하나입니다. 데이터 분석 및 분류(Classification) 모델링을 처음 배울 때 "Hello, World!"처럼 거쳐가는 예제이기도 합니다.

이 데이터셋은 붓꽃의 세 가지 품종(Setosa, Versicolor, Virginica)을 구분하기 위해, 각 품종별로 꽃잎(petal)과 꽃받침(sepal)의 길이와 너비를 측정한 데이터를 담고 있습니다. 각 품종이 뚜렷한 측정치 특성을 가지고 있어, 데이터 분석을 통해 품종 간의 차이점을 발견하고, 특정 측정치를 바탕으로 품종을 예측하는 모델을 만드는 데 이상적입니다.

**컬럼 설명**

| 컬럼명 | 데이터 타입 | 설명 |
| :--- | :--- | :--- |
| `sepal.length` | `float` | 꽃받침의 길이 (cm) |
| `sepal.width` | `float` | 꽃받침의 너비 (cm) |
| `petal.length` | `float` | 꽃잎의 길이 (cm) |
| `petal.width` | `float` | 꽃잎의 너비 (cm) |
| `variety` | `object` | 붓꽃의 품종. 우리가 예측하고자 하는 **타겟 변수(Target)** 입니다. |

**분석 목표**

이 실전 예제의 목표는 Pandas의 핵심 기능들을 활용하여 Iris 데이터셋을 다각도로 분석하고, 다음과 같은 질문에 답하는 것입니다.

1.  각 붓꽃 특성(꽃잎/꽃받침의 길이/너비)은 어떤 분포를 보이는가?
2.  세 가지 품종은 외형적 특성에서 어떤 차이를 보이는가?
3.  어떤 특성이 품종을 구별하는 데 가장 결정적인 역할을 하는가?

이러한 탐색적 데이터 분석(EDA) 과정은 향후 머신러닝 모델을 구축할 때 어떤 특성을 중요하게 다룰지 결정하는 데 중요한 근거가 됩니다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 예제 데이터 (실제 iris.csv 파일이라고 가정)
# 실제 분석에서는 pd.read_csv('./data/iris.csv')와 같이 파일 경로를 사용합니다.
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
data = pd.read_csv(io.StringIO(csv_data))
```

### 1.2. 1단계: 데이터 로딩 및 기본 탐색

데이터 분석의 첫걸음은 데이터를 불러와 데이터셋의 '건강 상태'를 확인하고 기본적인 특징을 파악하는 것입니다. 이 과정을 통해 데이터에 대한 첫인상을 얻고, 다음 분석 단계의 방향을 설정합니다.

**1. 데이터 크기 및 컬럼 확인**
가장 먼저 데이터가 몇 개의 행과 열로 이루어져 있는지, 컬럼 이름은 무엇인지 확인합니다.

```python
print(f"데이터 형태 (행, 열): {data.shape}")
print(f"컬럼 목록: {data.columns.tolist()}")
```

**2. 데이터 타입 및 결측치 확인 (`.info()`)
`data.info()`는 데이터셋의 핵심 요약 정보를 제공합니다. 각 컬럼의 데이터 타입과 non-null 데이터의 개수를 통해 결측치 유무를 한 번에 파악할 수 있습니다.

```python
print("=== 1. 데이터셋 필드 정보 ===")
data.info()
```
**결과 분석**:
- **Entries & Columns**: 총 9개의 행과 5개의 열로 구성된 작은 데이터셋입니다.
- **Non-Null Count**: 모든 컬럼이 9 non-null로, 데이터에 **결측치가 없음**을 의미합니다. 이는 데이터 정제 과정을 매우 간단하게 만들어 줍니다.
- **Dtype**: 4개의 수치형 특성(`float64`)과 1개의 범주형 타겟 변수(`object`)로 구성되어 있음을 확인했습니다.

**3. 기술 통계 요약 (`.describe()`)
`data.describe()`는 수치형 데이터에 대한 핵심 통계량(개수, 평균, 표준편차, 최소/최대값, 사분위수)을 요약하여 데이터의 분포와 스케일을 파악하는 데 필수적입니다.

```python
print("=== 2. 기술 통계 요약 ===")
print(data.describe())
```
**결과 분석**:
- **평균(mean) vs 중앙값(50%)**: `sepal.length`의 평균(6.13)과 중앙값(6.4)이 비슷한 반면, `petal.length`는 평균(4.2)과 중앙값(4.7)의 차이가 조금 있습니다. 이를 통해 각 특성의 분포가 대칭적인지, 한쪽으로 치우쳐 있는지 대략적으로 짐작할 수 있습니다.
- **표준편차(std)**: `sepal.width`(0.25)의 표준편차가 다른 특성들에 비해 매우 작은 것을 볼 수 있습니다. 이는 꽃받침 너비는 품종에 관계없이 비교적 일정한 크기를 가질 수 있음을 시사합니다.
- **최소(min)/최대(max)값**: 각 특성의 값 범위를 파악할 수 있습니다. 예를 들어, 꽃잎 너비(`petal.width`)는 0.2cm에서 2.5cm까지 분포합니다.

**4. 타겟 변수 분포 확인 (`.value_counts()`)
분류 문제에서는 타겟 변수의 클래스별 데이터 개수를 확인하는 것이 매우 중요합니다. 데이터가 각 클래스에 균등하게 분포하는지, 아니면 특정 클래스에 치우쳐 있는지(불균형 데이터) 파악해야 합니다.

```python
print("=== 3. 품종(타겟 변수) 분포 확인 ===")
print(data['variety'].value_counts())
```
**결과 분석**: 이 예제 데이터에서는 각 품종(Setosa, Versicolor, Virginica)이 3개씩 동일하게 분포하고 있습니다. 이는 데이터가 **균형(balanced)**을 이루고 있음을 의미하며, 모델을 학습하고 평가하기에 이상적인 조건입니다.


### 1.3. 2단계: 데이터 필터링 및 조건부 분석

전체 데이터를 대상으로 하는 분석을 넘어, 특정 조건을 만족하는 데이터 부분집합을 추출하여 분석하면 더 깊고 구체적인 질문에 답할 수 있습니다. Pandas는 다양한 방식으로 데이터를 필터링하는 강력한 기능을 제공합니다.

**1. 단일 조건 필터링 (Boolean Indexing)**

가장 기본적인 필터링 방식으로, 특정 조건을 만족하는 행을 추출합니다. `data['variety'] == 'Setosa'`는 각 행이 조건에 맞는지 여부를 `True`/`False`로 담은 불리언(boolean) Series를 반환하고, 이를 다시 `data[]`에 넣어 `True`인 행만 선택합니다.

```python
# 'Setosa' 품종 데이터만 추출하여 통계량 확인
print("\n=== 4. 'Setosa' 품종 데이터 통계량 ===")
setosa_data = data[data["variety"] == 'Setosa']
print(setosa_data.describe())
```
**결과 분석**: Setosa 품종은 전체 평균에 비해 꽃잎(petal)의 길이와 너비가 현저히 작고, 꽃받침(sepal)의 너비는 오히려 약간 더 넓은 경향을 보입니다. 이처럼 특정 그룹의 데이터는 전체와 다른 특성을 가질 수 있습니다.

**2. 복합 조건 필터링 (`&`, `|`)**

여러 조건을 조합할 때는 `&`(AND), `|`(OR) 연산자를 사용합니다. 각 조건은 반드시 소괄호 `()`로 묶어주어야 합니다.

```python
# 'Setosa' 품종이면서, 꽃받침 길이가 5cm 이상인 데이터 추출
print("\n=== 5. 'Setosa'이면서 sepal.length >= 5인 데이터 ===")
condition_data = data[(data["variety"] == 'Setosa') & (data["sepal.length"] >= 5)]
print(condition_data)
```

**3. isin() 메서드를 활용한 필터링**

여러 값 중 하나에 해당하는 데이터를 찾을 때 `|` 연산자를 여러 번 쓰는 대신 `.isin()` 메서드를 사용하면 코드가 훨씬 간결해집니다.

```python
# 'Setosa' 또는 'Virginica' 품종 데이터만 추출
print("\n=== 6. 'Setosa' 또는 'Virginica' 품종 데이터 ===")
subset_isin = data[data['variety'].isin(['Setosa', 'Virginica'])]
print(subset_isin.head())
```

**4. query() 메서드를 활용한 가독성 높은 필터링**

SQL 쿼리문과 유사한 문자열을 사용하여 조건을 표현할 수 있어, 복잡한 조건문을 더 읽기 쉽게 만들 수 있습니다.

```python
# 'petal.length'가 5.0보다 크고 'petal.width'가 2.0보다 큰 데이터
# 컬럼명에 '.'이 포함되어 있으므로 백틱(`)으로 감싸줍니다.
subset_query = data.query("`petal.length` > 5.0 and `petal.width` > 2.0")
print("\n=== 7. query를 이용한 필터링 ===")
print(subset_query)
```
이러한 필터링 기법들은 가설을 검증하거나, 특정 데이터 그룹의 세부적인 특징을 심도 있게 분석하는 데 필수적입니다.


### 1.4. 3단계: 그룹화 및 집계 분석

`groupby()`는 데이터를 특정 기준으로 그룹화하여 그룹별 통계량을 계산함으로써, **그룹 간의 차이점을 체계적으로 비교 분석**하는 가장 강력한 방법입니다. 품종별 특성을 비교하는 데 매우 효과적입니다.

**1. 품종별 평균 계산**

가장 기본적인 그룹화 분석으로, 각 품종(`variety`)별로 모든 수치 특성의 평균을 계산하여 품종별 전반적인 경향을 파악합니다.

```python
print("\n=== 8. 각 품종별 평균 ===")
# 소수점 둘째 자리까지만 표시하도록 설정
pd.set_option('display.precision', 2)
print(data.groupby('variety').mean())
```
**결과 분석**: 한눈에 봐도 Setosa는 다른 두 품종에 비해 꽃잎(petal)과 꽃받침(sepal)의 길이/너비가 모두 작은 경향을 보입니다. 반면 Virginica가 전반적으로 가장 큰 크기를 가집니다. 특히 `petal.length`와 `petal.width`에서 품종 간 평균값 차이가 매우 두드러지게 나타납니다.

**2. `agg()`를 이용한 다중 집계**

`agg()` 함수를 사용하면 각 그룹에 대해 여러 통계량(예: 평균, 표준편차, 최소/최대값)을 한 번에 계산하여 더 풍부한 정보를 얻을 수 있습니다.

```python
print("\n=== 9. 품종별 petal.length의 상세 통계량 ===")
# Named Aggregation을 사용하여 결과 컬럼명을 직접 지정
petal_length_agg = data.groupby('variety').agg(
    avg_petal_length=('petal.length', 'mean'),
    std_petal_length=('petal.length', 'std'),
    max_petal_length=('petal.length', 'max'),
    min_petal_length=('petal.length', 'min')
)
print(petal_length_agg)
```
**결과 분석**:
- **평균(avg)**: Setosa의 평균 꽃잎 길이는 1.37cm로 다른 두 종(Versicolor 4.63cm, Virginica 5.83cm)과 확연히 구분됩니다.
- **표준편차(std)**: Setosa의 표준편차는 0.15로 매우 작습니다. 이는 Setosa 품종의 꽃잎 길이는 개체 간 차이가 거의 없이 매우 균일하다는 의미입니다.
- **최소/최대(min/max)**: Setosa의 최대 꽃잎 길이(1.4cm)가 Versicolor의 최소 꽃잎 길이(4.5cm)보다도 작습니다. 이 사실만으로도 꽃잎 길이는 두 품종을 구분하는 매우 결정적인 특성임을 알 수 있습니다.

**3. 사용자 정의 함수와 `lambda` 활용**

`agg()` 안에서 직접 정의한 함수나 `lambda`를 사용하여 원하는 통계량을 계산할 수 있습니다.

```python
print("\n=== 10. 품종별 꽃잎 길이의 범위(range) 계산 ===")
# 최대값과 최소값의 차이를 계산하는 lambda 함수 적용
petal_range = data.groupby('variety').agg(
    petal_length_range=('petal.length', lambda x: x.max() - x.min()),
    petal_width_range=('petal.width', lambda x: x.max() - x.min())
)
print(petal_range)
```
**결과 분석**: `petal_length_range`를 보면, Setosa의 꽃잎 길이 변화량은 0.1cm에 불과하지만, 다른 두 종은 각각 0.4cm, 0.9cm로 변화의 폭이 더 큼을 알 수 있습니다.


### 1.5. 4단계: 데이터 시각화를 통한 인사이트 발견

숫자로 요약된 통계는 데이터의 전체적인 그림을 보여주지만, 복잡한 패턴이나 변수 간의 관계를 직관적으로 파악하기는 어렵습니다. 데이터 시각화는 이러한 숫자들을 명확한 그림으로 변환하여, 분석가가 더 깊은 인사이트를 얻고 결과를 효과적으로 전달할 수 있게 돕습니다. Pandas DataFrame은 Seaborn, Matplotlib과 같은 라이브러리와 완벽하게 호환됩니다.

**1. Pair Plot으로 변수 간 관계 전체보기**

`seaborn.pairplot`은 데이터프레임의 모든 수치형 변수 쌍에 대한 관계를 한 번에 시각화하는 강력한 도구입니다.
- **대각선 (Diagonal)**: 각 변수 자체의 분포를 보여주는 히스토그램 또는 KDE 플롯.
- **비대각선 (Off-diagonal)**: 두 변수 간의 관계를 보여주는 산점도(scatter plot).

`hue` 파라미터에 타겟 변수(`variety`)를 지정하면, 품종별로 색상을 다르게 표시하여 그룹 간의 분포 차이를 명확하게 비교할 수 있습니다.

```python
print("\n=== 11. Pair Plot을 통한 시각적 탐색 ===")
sns.pairplot(data, hue='variety', palette='bright', height=2.5)
plt.suptitle('Iris Data Pair Plot', y=1.02)
plt.show()
```
**결과 분석**:
- **Setosa의 분리**: 파란색 점으로 표시된 Setosa는 모든 산점도에서 다른 두 품종과 명확하게 분리되어 있습니다. 이는 Setosa가 매우 독특한 특성을 가짐을 의미합니다.
- **핵심 구분자**: `petal.length`와 `petal.width`의 산점도를 보면, 세 품종이 거의 겹치지 않고 잘 군집을 이루고 있습니다. 이는 이 두 특성이 품종을 구분하는 데 매우 중요한, 즉 **예측력이 높은 특성**임을 강력하게 시사합니다.
- **Versicolor와 Virginica**: 주황색(Versicolor)과 초록색(Virginica) 점들은 일부 영역에서 겹치지만, 전반적으로 Virginica가 Versicolor보다 꽃잎과 꽃받침이 더 큰 경향을 보입니다.

**2. Box Plot으로 품종별 특성 분포 비교**

Box plot은 각 품종별로 특정 수치 데이터의 분포를 비교하는 데 매우 효과적입니다. 데이터의 중앙값, 사분위수 범위(IQR), 이상치(outlier) 등을 한눈에 보여줍니다.

```python
# 4개의 특성에 대해 품종별 Box Plot을 한 번에 그리기
plt.figure(figsize=(14, 10))
# 컬럼명에 `.`이 있어 반복문에서 사용하기 편하도록 임시 변경
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'variety']
for i, col in enumerate(data.columns[:-1]): # 마지막 'variety' 컬럼 제외
    plt.subplot(2, 2, i+1) # 2x2 그리드에 순서대로 그리기
    sns.boxplot(data=data, x='variety', y=col)
    plt.title(f'{col} Distribution by Variety')
plt.tight_layout()
plt.show()
```
**결과 분석**:
- `petal.length`와 `petal.width`의 Box plot을 보면, 세 품종의 박스가 거의 겹치지 않습니다. 이는 이 두 특성만으로도 품종을 상당 부분 예측할 수 있음을 의미합니다.
- `sepal.width`의 경우, 세 품종의 박스가 많이 겹쳐 있어, 이 특성 하나만으로는 품종을 구분하기 어렵다는 것을 알 수 있습니다.

**3. Heatmap으로 상관관계 시각화**

1단계에서 계산한 상관관계 행렬을 히트맵(Heatmap)으로 시각화하면, 변수 간의 관계를 색상으로 직관적으로 파악할 수 있습니다.

```python
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Iris Features')
plt.show()
```
**결과 분석**:
- `petal.length`와 `petal.width`는 0.96이라는 매우 강한 양의 상관관계를 가집니다 (꽃잎이 길수록 너비도 넓다).
- `sepal.length`도 꽃잎의 길이/너비와 강한 양의 상관관계를 보입니다.
- `sepal.width`는 다른 특성들과 뚜렷한 상관관계를 보이지 않습니다.

이러한 시각적 탐색은 숫자만으로는 발견하기 어려운 데이터의 구조적 특징과 패턴을 명확히 드러내어, 데이터에 대한 깊은 이해를 가능하게 합니다.


### 1.6. 분석 결론

이 실전 예제를 통해, 우리는 Pandas의 핵심 기능들을 활용하여 Iris 데이터셋을 로딩하고, 데이터의 구조와 통계적 특성을 탐색했으며, 필터링, 그룹화, 시각화 등 다양한 분석을 체계적으로 수행했습니다. 이 탐색적 데이터 분석(EDA) 과정을 통해 얻은 핵심적인 결론은 다음과 같습니다.

**1. 품종별 특성은 뚜렷하게 구분된다.**

-   **Setosa**: 다른 두 품종과 비교했을 때, 꽃잎(petal)의 길이와 너비가 월등히 작고, 그 값의 변화 폭(표준편차)도 매우 작아 매우 동질적인 특성을 보입니다. 반면, 꽃받침(sepal)의 너비는 상대적으로 넓은 편입니다. 이러한 독보적인 특성 덕분에 다른 품종과 매우 쉽게 구별됩니다.
-   **Versicolor & Virginica**: 두 품종은 Setosa와 명확히 구분되지만, 서로 간에는 일부 특성이 겹치는 경향을 보입니다. 전반적으로 Virginica가 Versicolor보다 꽃잎과 꽃받침의 크기가 더 큽니다.

**2. 꽃잎(Petal)의 크기가 품종을 구별하는 핵심 지표이다.**

-   그룹화 분석과 모든 시각화 결과에서 **`petal.length`와 `petal.width`가 세 품종을 구별하는 가장 결정적인 특성**임을 일관되게 확인할 수 있었습니다. 특히 Setosa는 이 두 가지 특성만으로도 거의 완벽하게 분류가 가능합니다.
-   반면, `sepal.width`는 품종 간 분포가 많이 겹쳐, 품종을 구별하는 능력(변별력)이 상대적으로 가장 낮은 특성으로 파악되었습니다.

**3. 향후 머신러닝 모델링 방향**

-   **특성 선택 (Feature Selection)**: 만약 더 단순하고 해석하기 쉬운 모델을 만든다면, `petal.length`와 `petal.width` 두 가지 특성만 사용해도 매우 높은 분류 성능을 얻을 수 있을 것으로 기대됩니다.
-   **모델 선택 (Model Selection)**: 데이터의 특성들이 품종별로 잘 구분되기 때문에, 로지스틱 회귀, 서포트 벡터 머신(SVM), 결정 트리(Decision Tree) 등 비교적 간단한 전통적인 분류 모델로도 충분히 높은 정확도를 달성할 수 있을 것으로 예상됩니다.

결론적으로, Pandas를 활용한 체계적인 EDA는 단순히 데이터를 요약하는 것을 넘어, 데이터에 숨겨진 패턴을 발견하고, 어떤 특성이 중요한지 파악하며, 향후 진행될 머신러닝 모델링의 전략을 수립하는 데 필수적인 과정임을 이 예제를 통해 확인할 수 있습니다.