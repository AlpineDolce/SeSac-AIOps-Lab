<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 머신러닝 프로젝트의 각 단계에서 Pandas가 어떻게 핵심적인 역할을 수행하는지 상세히 다룹니다. 데이터 로딩 및 탐색, 데이터 전처리, 특성 공학(Feature Engineering), 데이터 분할 및 모델 입력, 시각화 연동 등 머신러닝 워크플로우 전반에 걸쳐 Pandas의 활용법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. Pandas와 머신러닝 워크플로우](#1-pandas와-머신러닝-워크플로우)
  - [1.1. 데이터 로딩 및 탐색](#11-데이터-로딩-및-탐색)
  - [1.2. 데이터 전처리](#12-데이터-전처리)
  - [1.3. 특성 공학 (Feature Engineering)](#13-특성-공학-feature-engineering)
  - [1.4. 데이터 분할 및 모델 입력](#14-데이터-분할-및-모델-입력)
  - [1.5. 시각화 연동](#15-시각화-연동)

---

## 1. Pandas와 머신러닝 워크플로우

Pandas는 머신러닝 프로젝트의 거의 모든 단계, 특히 데이터 준비 및 탐색 과정에서 핵심적인 역할을 수행합니다. 이 문서에서는 **타이타닉 생존자 예측**이라는 가상 시나리오를 통해 머신러닝 전체 워크플로우에서 Pandas가 어떻게 활용되는지 단계별로 살펴보겠습니다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# 예제 데이터 (타이타닉 데이터셋의 일부)
csv_data = """
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",female,27,0,2,347742,11.1333,,S
10,1,2,"Nasser, Mrs. Nicholas (Adele Achem)",female,14,1,0,237736,30.0708,,C
"""

df = pd.read_csv(io.StringIO(csv_data))
```

### 1.1. 데이터 로딩 및 탐색

머신러닝 프로젝트의 가장 첫 단계는 데이터를 불러와 그 속을 들여다보는 것입니다. 이 탐색적 데이터 분석(EDA, Exploratory Data Analysis) 과정을 통해 데이터의 구조, 결측치, 통계적 특성, 변수 간의 관계 등을 파악하고, 앞으로 진행할 전처리 및 특성 공학의 방향을 설정합니다.

**1. 데이터 구조 확인 (`.head()`, `.info()`)**

가장 먼저 데이터의 전반적인 형태와 각 컬럼의 정보(데이터 타입, 결측치 유무)를 확인합니다.

```python
print("---\n--- 데이터 상위 5개 ---")
print(df.head())

print("\n--- 데이터 정보 ---")
df.info()
```

**결과 분석**: 
- **전체 구조**: 10개의 행(Entries)과 12개의 열(Columns)으로 구성되어 있습니다.
- **데이터 타입**: `Age`, `Fare` 등은 숫자형(float/int), `Name`, `Sex` 등은 문자열(object)입니다. 머신러닝 모델은 숫자형 입력을 받으므로, 문자열 컬럼들은 추후 숫자형으로 변환(인코딩)해야 합니다.
- **결측치 확인**: `Non-Null Count`를 보면 `Age`(8 non-null), `Cabin`(3 non-null) 컬럼에 결측치가 존재함을 알 수 있습니다. `Cabin`은 결측치가 매우 많아 사용하기 어려워 보이며, `Age`는 다른 값으로 채우는 전략을 고려해야 합니다.

**2. 기술 통계 확인 (`.describe()`)**

숫자형(numerical) 데이터 컬럼들의 핵심 통계량을 요약하여 데이터의 분포와 스케일을 파악합니다.

```python
print("\n--- 기술 통계 ---")
print(df.describe())
```

**결과 분석**:
- **`Survived`**: 평균이 0.5이므로, 이 샘플 데이터에서는 생존자와 사망자 비율이 50:50입니다. (실제 전체 데이터셋과는 다를 수 있습니다.)
- **`Pclass`**: 평균이 2.2로, 3등실 승객이 많음을 짐작할 수 있습니다.
- **`Age`**: 평균 나이는 약 29.8세이며, 최소 2세부터 최대 54세까지 분포합니다.
- **`Fare`**: `std`(표준편차)가 35.8로 매우 크고, `mean`(평균) 30.5에 비해 `max`(최대) 71.28이 상대적으로 큽니다. 이는 요금 분포가 오른쪽으로 긴 꼬리를 가진, 즉 일부 매우 비싼 요금이 존재할 수 있음을 시사합니다. 이러한 분포는 모델 성능에 영향을 줄 수 있어 로그 변환(log transformation) 등을 고려할 수 있습니다.

**3. 범주형 데이터 탐색 (`.value_counts()`)**

문자열과 같이 범주를 나누는 데이터(categorical)의 고유값별 개수를 확인하여 데이터의 편중 여부를 파악합니다.

```python
print("\n--- 성별 분포 ---")
print(df['Sex'].value_counts())

print("\n--- 선실 등급 분포 ---")
print(df['Pclass'].value_counts())
```

**결과 분석**: 남성(male)이 6명, 여성(female)이 4명이며, 3등실 승객이 5명으로 가장 많습니다. 이러한 분포는 모델이 특정 범주에 편향되어 학습하는 원인이 될 수 있으므로 확인이 필요합니다.

**4. 그룹별 탐색 (`.groupby()`)**

`groupby`를 사용하면 특정 범주에 따른 타겟 변수(여기서는 `Survived`)의 평균을 계산하여, 각 범주가 생존에 미치는 영향을 직관적으로 파악할 수 있습니다.

```python
print("\n--- 성별에 따른 생존율 ---")
print(df.groupby('Sex')['Survived'].mean())

print("\n--- 선실 등급에 따른 생존율 ---")
print(df.groupby('Pclass')['Survived'].mean())
```

**결과 분석**: 여자의 생존율은 75%, 남자는 약 16.7%로 성별이 생존에 큰 영향을 미치는 변수임을 알 수 있습니다. 또한, 1등실의 생존율은 100%인 반면 3등실은 40%로, 선실 등급 역시 중요한 변수임을 강력하게 시사합니다.

**5. 상관관계 분석 (`.corr()`)**

숫자형 변수들 간의 선형적인 관계를 파악합니다. 상관계수는 -1에서 1 사이의 값을 가지며, 1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계를 의미합니다.

```python
print("\n--- 상관관계 행렬 ---")
print(df.corr(numeric_only=True))
```

**결과 분석**: `Survived`와 `Pclass`는 -0.4의 음의 상관관계를 가집니다. 즉, 등급(숫자)이 낮을수록(1등실) 생존율이 높은 경향이 있음을 수치적으로 확인할 수 있습니다. `Fare`와 `Survived`는 양의 상관관계를 보여, 요금을 많이 낼수록 생존율이 높은 경향을 보입니다. 이러한 관계들은 모델링 시 중요한 특성이 될 수 있습니다.

### 1.2. 데이터 전처리

데이터 탐색 단계에서 파악한 문제점들을 해결하고, 머신러닝 모델이 학습할 수 있는 깨끗하고 정제된 데이터 형태로 가공하는 과정입니다.

**1. 결측치 처리**

결측치는 모델 학습을 방해하는 주요 요인이므로 반드시 처리해야 합니다. 처리 전략은 데이터의 특성과 결측치의 양에 따라 달라집니다.

-   **`Age` 컬럼 처리**: `Age`는 생존에 중요한 영향을 미칠 수 있는 숫자형 특성이지만 일부 값이 누락되었습니다. 데이터 손실을 최소화하기 위해, 단순히 행을 제거하는 대신 다른 값으로 대체하는 것이 합리적입니다.
    -   **전략**: 전체 승객의 **평균(mean) 나이**로 결측치를 채웁니다. 중앙값(median)을 사용하는 것도 좋은 전략이며, 특히 이상치(outlier)가 많을 때 더 안정적인 선택이 될 수 있습니다.
    ```python
    # Age의 결측치를 전체 승객의 나이 평균으로 채웁니다.
    # inplace=True는 원본 DataFrame을 직접 수정하는 옵션입니다.
    age_mean = df['Age'].mean()
    df['Age'].fillna(age_mean, inplace=True)
    print(f"--- 'Age' 결측치를 평균값({age_mean:.2f})으로 대체 ---")
    ```

-   **`Cabin` 컬럼 처리**: `Cabin`(객실 번호)은 결측치가 대부분을 차지합니다.
    -   **전략**: 유용한 정보를 추출하기 어려울 정도로 데이터가 부족하므로, 이 컬럼은 **제거(drop)**하는 것이 가장 간단하고 합리적인 방법입니다.
    ```python
    # Cabin 컬럼은 결측치가 너무 많으므로 제거합니다.
    df.drop('Cabin', axis=1, inplace=True)
    print("--- 'Cabin' 컬럼 제거 ---")
    ```

-   **`Embarked` 컬럼 처리**: `Embarked`(탑승 항구)는 범주형 데이터입니다.
    -   **전략**: 숫자형 데이터와 달리 평균이나 중앙값을 사용할 수 없으므로, 가장 많이 나타나는 값, 즉 **최빈값(mode)**으로 결측치를 채우는 것이 일반적입니다.
    ```python
    # (예제 데이터에는 없지만, 실제 데이터에는 존재한다고 가정)
    if 'Embarked' in df.columns and df['Embarked'].isnull().any():
        embarked_mode = df['Embarked'].mode()[0]
        df['Embarked'].fillna(embarked_mode, inplace=True)
        print(f"--- 'Embarked' 결측치를 최빈값({embarked_mode})으로 대체 ---")

    print("\n--- 결측치 처리 후 정보 ---")
    df.info()
    ```

**2. 불필요한 컬럼 제거**

모델의 예측 성능에 도움이 되지 않거나, 노이즈로 작용할 수 있는 컬럼들을 제거합니다.

-   **`PassengerId`**: 승객의 고유 ID로, 생존 여부와는 아무런 관련이 없는 식별자입니다.
-   **`Name`**: 이름 자체는 예측에 도움이 되지 않습니다. (물론, 이름에서 'Mr.', 'Miss.' 등 호칭을 추출하여 새로운 특성을 만드는 고급 기법도 가능합니다.)
-   **`Ticket`**: 티켓 번호는 고유한 값이 너무 많고 패턴을 찾기 어려워, 일반적인 모델에서는 특성으로 사용하기 어렵습니다.

```python
# 모델링에 불필요한 컬럼들 제거
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
print("\n--- 불필요한 컬럼 제거 후 데이터 ---")
print(df.head())
```
이러한 전처리 과정을 통해 데이터셋은 결측치가 없고, 모델 학습에 더 적합한 특성들만 남은 상태가 됩니다.


### 1.3. 특성 공학 (Feature Engineering)

특성 공학은 **모델의 성능을 결정하는 가장 중요한 단계** 중 하나로, 기존 데이터를 가공하여 모델이 더 잘 학습할 수 있는 새로운 '특성(feature)'을 만들어내는 과정입니다. 도메인 지식과 창의성이 요구되는 단계이며, Pandas는 이러한 작업을 수행하는 데 매우 강력한 도구를 제공합니다.

**1. 범주형 데이터 인코딩 (Categorical Data Encoding)**

머신러닝 모델은 대부분 숫자형 데이터만 처리할 수 있으므로, 'Sex'나 'Embarked'와 같은 문자열 범주형 데이터를 숫자형으로 변환해야 합니다.

-   **One-Hot Encoding**: 가장 일반적으로 사용되는 방법으로, 각 범주를 새로운 컬럼으로 만들고 해당 여부를 0 또는 1로 표시합니다. `pd.get_dummies()` 함수로 손쉽게 구현할 수 있습니다.
    -   `drop_first=True`: 이 옵션은 다중공선성(multicollinearity) 문제를 방지하기 위해 사용됩니다. 예를 들어 `Sex` 컬럼을 원-핫 인코딩하면 `Sex_male`과 `Sex_female` 두 컬럼이 생기는데, 하나가 1이면 다른 하나는 반드시 0이므로 두 변수는 완벽한 선형 관계를 가집니다. `drop_first=True`는 이 중 하나(예: `Sex_female`)를 제거하여, `Sex_male`이 1이면 남성, 0이면 여성으로 표현하게 해줍니다. 이는 일부 선형 모델에서 중요한 전처리 과정입니다.

```python
# One-Hot Encoding 수행
# Embarked는 결측치가 없다고 가정하고 진행
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print("--- One-Hot Encoding 후 데이터 ---")
print(df_encoded.head())
```

**2. 새로운 특성 생성**

기존 컬럼들을 조합하거나 가공하여 새로운 의미를 갖는 특성을 만듭니다.

-   **`FamilySize` (가족 크기)**: `SibSp`(형제/배우자 수)와 `Parch`(부모/자녀 수)는 개별적으로보다 합쳐서 '총 가족 수'로 보는 것이 더 의미 있을 수 있습니다.
    ```python
    # 가족 크기 특성 생성 (본인 포함)
    df_encoded['FamilySize'] = df_encoded['SibSp'] + df_encoded['Parch'] + 1
    print("\n--- 'FamilySize' 특성 추가 ---")
    print(df_encoded[['SibSp', 'Parch', 'FamilySize']].head())
    ```

-   **`IsAlone` (혼자 탑승 여부)**: 가족 크기 정보를 바탕으로, '혼자 왔는지' 여부를 나타내는 이진 특성을 만들 수 있습니다.
    ```python
    # FamilySize가 1이면 혼자, 아니면 동반 탑승
    df_encoded['IsAlone'] = (df_encoded['FamilySize'] == 1).astype(int)
    print("\n--- 'IsAlone' 특성 추가 ---")
    print(df_encoded[['FamilySize', 'IsAlone']].head())
    ```

-   **`Age_Group` (나이대)**: 연속형 변수인 `Age`를 'Child', 'Adult' 등과 같은 범주형 변수로 만들면, 모델이 특정 나이대의 패턴을 더 쉽게 학습할 수 있습니다. `pd.cut()` 함수를 사용합니다.
    ```python
    # 나이를 기준으로 범주 나누기
    bins = [0, 12, 18, 60, 100] # 0-12: Child, 13-18: Teen, 19-60: Adult, 61-100: Senior
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df_encoded['Age_Group'] = pd.cut(df_encoded['Age'], bins=bins, labels=labels, right=False)
    
    # Age_Group도 One-Hot Encoding 필요
    df_encoded = pd.get_dummies(df_encoded, columns=['Age_Group'], drop_first=True)

    print("\n--- 'Age_Group' 특성 추가 및 인코딩 ---")
    print(df_encoded.head())
    ```
이처럼 Pandas를 활용하면 기존 데이터에 대한 이해를 바탕으로 모델의 예측력을 높일 수 있는 새로운 특성들을 유연하게 생성할 수 있습니다.

### 1.4. 데이터 분할 및 모델 입력

모든 준비가 끝난 데이터를 머신러닝 모델에 학습시키기 전, 반드시 거쳐야 하는 중요한 단계입니다. 모델의 성능을 객관적으로 평가하기 위해 데이터를 **학습용(train set)**과 **테스트용(test set)**으로 분리합니다.

-   **학습용 데이터 (Train Set)**: 모델이 데이터의 패턴을 학습하는 데 사용됩니다. (문제집)
-   **테스트용 데이터 (Test Set)**: 학습이 완료된 모델이 얼마나 잘 작동하는지 평가하는 데 사용됩니다. 이 데이터는 모델이 학습 과정에서 **전혀 보지 못한** 새로운 데이터여야 합니다. (실전 시험)

**1. 특성(X)과 타겟(y) 분리**

먼저, 우리가 예측하고자 하는 목표 변수(target)와 이를 예측하는 데 사용할 설명 변수(features)를 분리합니다.

-   **`y` (Target)**: 예측의 대상이 되는 컬럼입니다. (예: `Survived`)
-   **`X` (Features)**: `y`를 제외한 나머지 모든 특성 컬럼들입니다.

```python
# df_final은 이전 단계에서 모든 전처리와 특성 공학이 완료된 DataFrame이라고 가정
# 예시를 위해 이전 단계의 최종 결과인 df_encoded를 df_final로 간주
# FamilySize로 역할이 대체된 SibSp, Parch 컬럼은 최종 모델 입력에서 제외
df_final = df_encoded.drop(columns=['SibSp', 'Parch'])

X = df_final.drop('Survived', axis=1)
y = df_final['Survived']

print("--- 특성(X) 데이터 ---")
print(X.head())
print("\n--- 타겟(y) 데이터 ---")
print(y.head())
```

**2. 학습/테스트 데이터 분할 (`train_test_split`)**

`scikit-learn` 라이브러리의 `train_test_split` 함수는 이 분할 작업을 매우 편리하게 수행해 줍니다.

-   `test_size`: 전체 데이터 중 테스트 세트가 차지할 비율을 지정합니다. `0.2`는 20%를 의미합니다.
-   `random_state`: 데이터를 무작위로 섞을 때 사용되는 시드(seed) 값입니다. 이 값을 특정 숫자로 고정하면, 코드를 다시 실행해도 항상 **동일한 방식**으로 데이터가 분할됩니다. 이는 분석 결과를 재현하고 다른 사람과 공유하는 데 필수적입니다.
-   `stratify`: **매우 중요한 파라미터**. 분류 문제에서 타겟 변수(`y`)의 클래스 비율을 학습 데이터와 테스트 데이터에서 동일하게 유지해 줍니다. 예를 들어, 원본 데이터의 생존/사망 비율이 4:6이었다면, `stratify=y` 옵션은 학습 데이터와 테스트 데이터 모두 생존/사망 비율을 4:6으로 맞춰줍니다. 이는 모델이 편향되지 않고 안정적으로 평가받기 위해 꼭 필요합니다.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,       # 30%를 테스트 데이터로 사용
    random_state=42,     # 재현 가능성을 위한 시드 고정
    stratify=y           # 타겟 변수의 클래스 비율을 유지
)

print("--- 데이터 분할 결과 ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\n학습 데이터의 생존율:", y_train.mean())
print("테스트 데이터의 생존율:", y_test.mean())
```
이제 `X_train`과 `y_train`을 사용하여 머신러닝 모델(예: Logistic Regression, Random Forest 등)을 학습시키고, 학습된 모델을 `X_test`에 적용하여 예측을 수행한 뒤, 그 예측 결과를 실제 값인 `y_test`와 비교하여 모델의 성능(정확도, 정밀도 등)을 객관적으로 평가할 수 있습니다.

### 1.5. 시각화 연동

숫자로만 데이터를 파악하는 데는 한계가 있습니다. Pandas는 **Matplotlib**과 **Seaborn** 같은 파이썬의 대표적인 시각화 라이브러리와 완벽하게 호환되어, 데이터에 숨겨진 패턴과 인사이트를 시각적으로 탐색하는 데 핵심적인 역할을 합니다. Pandas DataFrame은 이들 라이브러리가 가장 선호하는 데이터 입력 형태입니다.

**시각화의 목적**
- **탐색적 데이터 분석(EDA)**: 변수의 분포를 확인하고, 변수 간의 관계를 파악하며, 이상치를 탐지하고, 특성 공학에 대한 아이디어를 얻습니다.
- **결과 전달**: 분석 결과나 모델의 예측을 다른 사람들에게 효과적으로 전달합니다.

---
**주요 시각화 유형별 예제**

*데이터는 전처리 이전의 원본 `df`를 다시 사용하여 탐색적 분석을 수행합니다.*
```python
# 시각화를 위해 원본 데이터를 다시 로드합니다.
original_df = pd.read_csv(io.StringIO(csv_data))
# 결측치 처리는 시각화에 영향을 줄 수 있으므로, 여기서는 나이 결측치만 채웁니다.
original_df['Age'].fillna(original_df['Age'].median(), inplace=True)
```

**1. 단변수 분석 (Univariate Analysis): 개별 특성의 분포 확인**

- **범주형 데이터 (`countplot`)**: 각 카테고리에 속한 데이터의 개수를 막대그래프로 보여줍니다.
    ```python
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(data=original_df, x='Sex', hue='Survived')
    plt.title('Survival Count by Sex')

    plt.subplot(1, 2, 2)
    sns.countplot(data=original_df, x='Pclass', hue='Survived')
    plt.title('Survival Count by Pclass')
    plt.tight_layout()
    plt.show()
    ```
    **분석**: 성별과 객실 등급에 따라 생존자/사망자 수가 확연히 차이 나는 것을 시각적으로 확인할 수 있습니다. 이는 두 변수가 매우 중요한 예측 변수임을 의미합니다.

- **수치형 데이터 (`histplot`, `kdeplot`)**: 데이터가 어떤 값에 집중되어 있고 어떻게 퍼져 있는지(분포)를 보여줍니다.
    ```python
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=original_df, x='Age', bins=20, kde=True)
    plt.title('Age Distribution')

    plt.subplot(1, 2, 2)
    sns.histplot(data=original_df, x='Fare', bins=30)
    plt.title('Fare Distribution')
    plt.tight_layout()
    plt.show()
    ```
    **분석**: 나이는 대체로 정규분포에 가깝지만, 요금(Fare)은 매우 낮은 가격대에 데이터가 극도로 편중되어 있습니다. 이는 EDA 단계에서 `describe()`로 확인했던 사실과 일치하며, 로그 변환 등의 필요성을 다시 한번 시사합니다.

**2. 이변수 분석 (Bivariate Analysis): 두 변수 간의 관계 확인**

- **수치형 vs 범주형 (`boxplot`, `violinplot`)**: 범주별로 수치 데이터의 분포를 비교할 때 매우 효과적입니다.
    ```python
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=original_df, x='Pclass', y='Age', hue='Survived')
    plt.title('Age Distribution by Pclass and Survival')

    plt.subplot(1, 2, 2)
    sns.violinplot(data=original_df, x='Sex', y='Fare', hue='Survived', split=True)
    plt.title('Fare Distribution by Sex and Survival')
    plt.tight_layout()
    plt.show()
    ```
    **분석**: 1등실의 나이대가 다른 등실보다 높고, 생존자들의 나이대가 조금 더 다양한 분포를 보입니다. 또한, 여성(female)이면서 생존한 사람들의 요금(Fare) 분포가 더 넓게 퍼져 있음을 알 수 있습니다.

**3. 다변수 분석 (Multivariate Analysis): 여러 변수 간의 관계 동시 확인**

- **상관관계 히트맵 (`heatmap`)**: 모든 숫자형 변수 간의 상관관계를 색상으로 표현하여, 어떤 변수들이 서로 강한 관계를 맺고 있는지 한눈에 파악할 수 있습니다.
    ```python
    # 숫자형 데이터만 선택하여 상관관계 계산
    numeric_df = original_df.select_dtypes(include=np.number)
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
    ```
    **분석**: `Pclass`와 `Fare`는 강한 음의 상관관계(-0.55)를 보입니다 (등급이 높을수록 요금은 비싸짐). `Survived`는 `Pclass`와 음의 관계, `Fare`와 양의 관계를 가짐을 다시 한번 확인할 수 있습니다.

이처럼 Pandas로 가공된 데이터를 시각화 라이브러리에 전달하여 그래프를 그려보는 과정은, 데이터에 대한 깊은 이해를 얻고 더 나은 머신러닝 모델을 만들기 위한 필수적인 단계입니다.