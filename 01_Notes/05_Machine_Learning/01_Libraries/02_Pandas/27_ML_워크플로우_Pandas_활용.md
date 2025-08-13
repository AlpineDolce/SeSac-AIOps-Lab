<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 머신러닝 프로젝트의 각 단계에서 Pandas가 어떻게 핵심적인 역할을 수행하는지 상세히 다룹니다. 데이터 로딩 및 탐색, 데이터 전처리, 특성 공학(Feature Engineering), 데이터 분할 및 모델 입력, 시각화 연동 등 머신러닝 워크플로우 전반에 걸쳐 Pandas의 활용법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [목차](#목차)
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

가장 먼저 데이터를 불러오고, 데이터의 기본적인 구조와 특성을 파악합니다.

1.  **데이터 구조 확인**: `head()`와 `info()`를 사용하여 데이터의 형태와 각 컬럼의 정보(데이터 타입, 결측치 등)를 확인합니다.
    ```python
    print("---\n--- 데이터 상위 5개 ---")
    print(df.head())
    
    print("\n--- 데이터 정보 ---")
    print(df.info())
    ```
    **결과 분석**: `Age`와 `Cabin` 컬럼에 결측치(Non-null 값이 전체 개수보다 적음)가 있음을 확인했습니다. `Cabin`은 결측치가 너무 많아 사용하기 어려워 보입니다.

2.  **기술 통계 확인**: `describe()`를 사용하여 숫자형 데이터의 분포(평균, 표준편차, 사분위수 등)를 파악합니다.
    ```python
    print("\n--- 기술 통계 ---")
    print(df.describe())
    ```
    **결과 분석**: `Age`의 평균은 약 29.8세이며, `Fare`(요금)의 편차가 매우 크다는 것을 알 수 있습니다.

3.  **범주형 데이터 확인**: `value_counts()`를 사용하여 범주형 데이터의 고유값과 빈도를 확인합니다.
    ```python
    print("\n--- 성별 분포 ---")
    print(df['Sex'].value_counts())
    
    print("\n--- 선실 등급 분포 ---")
    print(df['Pclass'].value_counts())
    ```
    **결과 분석**: 남성 승객이 여성보다 많고, 3등실 승객이 가장 많다는 사실을 파악했습니다.

### 1.2. 데이터 전처리

탐색 단계에서 발견한 문제들을 해결하고, 모델이 학습하기 좋은 형태로 데이터를 가공합니다.

1.  **결측치 처리**: 결측치를 적절한 값으로 채우거나, 결측치가 포함된 행/열을 제거합니다.
    ```python
    # Age의 결측치는 전체 승객의 나이 평균으로 채웁니다.
    age_mean = df['Age'].mean()
    df['Age'].fillna(age_mean, inplace=True)
    
    # Cabin 컬럼은 결측치가 너무 많으므로 제거합니다.
    df.drop('Cabin', axis=1, inplace=True)
    
    # Embarked의 결측치는 최빈값으로 채웁니다.
    # (예제 데이터에는 없지만, 실제 데이터에는 존재)
    if 'Embarked' in df.columns:
        embarked_mode = df['Embarked'].mode()[0]
        df['Embarked'].fillna(embarked_mode, inplace=True)

    print("--- 결측치 처리 후 정보 ---")
    print(df.info())
    ```
    **결과 분석**: `Age`와 `Cabin`의 결측치 문제가 해결되었습니다.

2.  **불필요한 컬럼 제거**: 모델링에 직접적으로 사용하기 어려운 `PassengerId`, `Name`, `Ticket` 같은 식별자 성격의 컬럼을 제거합니다.
    ```python
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    print("\n--- 불필요한 컬럼 제거 후 데이터 ---")
    print(df.head())
    ```

### 1.3. 특성 공학 (Feature Engineering)

기존 특성을 변환하거나 조합하여 모델의 성능을 높일 수 있는 새로운 특성을 만듭니다.

1.  **범주형 데이터 인코딩**: 머신러닝 모델은 문자열 값을 직접 처리하지 못하므로, `Sex`, `Embarked` 같은 범주형 데이터를 숫자형으로 변환합니다. **One-Hot Encoding**이 가장 대표적인 방법입니다.
    ```python
    # One-Hot Encoding 수행
    df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    print("--- One-Hot Encoding 후 데이터 ---")
    print(df_encoded.head())
    ```
    **결과 분석**: `Sex_male`, `Embarked_Q`, `Embarked_S` 와 같이 새로운 이진(0/1) 컬럼이 생성되었습니다. `drop_first=True`는 다중공선성 문제를 방지하기 위해 각 변수의 첫 번째 카테고리를 제거하는 옵션입니다.

### 1.4. 데이터 분할 및 모델 입력

전처리와 특성 공학이 완료된 데이터를 모델 학습을 위한 학습용(train)과 테스트용(test) 데이터로 분할합니다.

1.  **특성(X)과 타겟(y) 분리**
    ```python
    X = df_encoded.drop('Survived', axis=1)
    y = df_encoded['Survived']
    ```

2.  **학습/테스트 데이터 분할**: `sklearn.model_selection.train_test_split`을 사용합니다.
    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("--- 데이터 분할 결과 ---")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    ```
    이제 `X_train`과 `y_train`을 사용하여 머신러닝 모델을 학습시키고, `X_test`와 `y_test`로 모델의 성능을 평가할 수 있습니다.

### 1.5. 시각화 연동

Pandas는 Matplotlib, Seaborn과 같은 시각화 라이브러리와 긴밀하게 연동하여 데이터에 대한 깊은 통찰력을 제공합니다.

**시각화 예제**: 생존 여부에 따른 주요 특성 분포 확인
```python
# 원본 데이터프레임으로 시각화를 진행합니다.
original_df = pd.read_csv(io.StringIO(csv_data))

plt.figure(figsize=(12, 5))

# 1. 성별에 따른 생존율
plt.subplot(1, 2, 1)
sns.countplot(data=original_df, x='Sex', hue='Survived')
plt.title('Survival Count by Sex')

# 2. 선실 등급에 따른 생존율
plt.subplot(1, 2, 2)
sns.countplot(data=original_df, x='Pclass', hue='Survived')
plt.title('Survival Count by Pclass')

plt.tight_layout()
plt.show()

# 3. 나이 분포에 따른 생존 여부
plt.figure(figsize=(10, 6))
sns.histplot(data=original_df, x='Age', hue='Survived', kde=True, multiple="stack")
plt.title('Age Distribution by Survival')
plt.show()
```
**결과 분석**:
*   여성의 생존자 수가 남성보다 많습니다.
*   1등실 승객의 생존 비율이 다른 등급보다 높고, 3등실 승객의 사망 비율이 높습니다.
*   어린 아이들의 생존율이 상대적으로 높은 경향을 보입니다.

이처럼 Pandas는 머신러닝 프로젝트의 A to Z, 즉 데이터 로딩부터 최종 모델 입력 준비까지 전 과정에 걸쳐 데이터를 자유자재로 다룰 수 있는 필수적인 도구입니다.

```