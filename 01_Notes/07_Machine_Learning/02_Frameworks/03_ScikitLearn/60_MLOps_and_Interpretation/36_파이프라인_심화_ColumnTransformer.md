# 파이프라인 심화: ColumnTransformer를 이용한 혼합 데이터 전처리
작성자: Alpine_Dolce | 날짜: 2025-07-01

## 문서 목표
이 문서는 Scikit-learn의 `Pipeline` 기능을 심화 학습하고, `ColumnTransformer`를 이용하여 수치형 및 범주형 특성이 혼합된 데이터에 서로 다른 전처리 단계를 적용하는 방법을 상세히 다룹니다. `make_column_transformer`를 이용한 간결한 문법과 `FeatureUnion`과의 차이점을 이해하며, 파이프라인 내에서 복잡한 전처리 로직을 구현하는 실무 역량을 강화합니다.

## 목차

* [1. 파이프라인(Pipeline) 복습](#1-파이프라인pipeline-복습)
  * [1.1. 파이프라인의 개념 및 장점](#11-파이프라인의-개념-및-장점)
  * [1.2. `Pipeline` 기본 사용 예시](#12-pipeline-기본-사용-예시)
* [2. `ColumnTransformer`: 혼합 데이터 타입 전처리](#2-columntransformer-혼합-데이터-타입-전처리)
  * [2.1. `ColumnTransformer`의 필요성](#21-columntransformer의-필요성)
  * [2.2. `ColumnTransformer` 기본 사용법](#22-columntransformer-기본-사용법)
  * [2.3. `make_column_transformer`를 이용한 간결한 문법](#23-make_column_transformer를-이용한-간결한-문법)
  * [2.4. `FeatureUnion` vs. `ColumnTransformer`](#24-featureunion-vs-columntransformer)
* [3. 실습: 혼합 데이터 타입 파이프라인 구축](#3-실습-혼합-데이터-타입-파이프라인-구축)
  * [3.1. 데이터 준비](#31-데이터-준비)
  * [3.2. `ColumnTransformer`를 포함한 파이프라인 구축 및 학습](#32-columntransformer를-포함한-파이프라인-구축-및-학습)
* [4. 파이프라인 심화 활용 팁](#4-파이프라인-심화-활용-팁)

---

## 1. 파이프라인(Pipeline) 복습

### 1.1. 파이프라인의 개념 및 장점
머신러닝 워크플로우는 데이터 전처리, 특성 추출, 모델 학습 등 여러 단계로 구성됩니다. Scikit-learn의 `Pipeline`은 이러한 연속적인 변환 및 학습 과정을 하나의 객체로 묶어주는 강력한 도구입니다. 이는 코드의 가독성을 높이고, 재사용성을 향상시키며, 데이터 누수(data leakage)를 방지하는 데 도움을 줍니다.

**장점**:
1.  **코드 간결성 및 가독성**: 여러 전처리 단계와 모델 학습을 한 줄의 코드로 표현할 수 있어 코드가 간결해지고 이해하기 쉬워집니다.
2.  **데이터 누수 방지**: 교차 검증 시 훈련 데이터에만 `fit()`을 적용하고 테스트 데이터에는 `transform()`만 적용하도록 강제하여, 테스트 데이터의 정보가 훈련 과정에 유출되는 데이터 누수를 효과적으로 방지합니다.
3.  **재사용성**: 한 번 정의된 파이프라인은 다른 데이터셋이나 프로젝트에서도 쉽게 재사용할 수 있습니다.
4.  **하이퍼파라미터 튜닝 용이**: `GridSearchCV`나 `RandomizedSearchCV`와 같은 하이퍼파라미터 튜닝 도구를 사용하여 파이프라인 내의 모든 단계(전처리 및 모델)의 하이퍼파라미터를 한 번에 최적화할 수 있습니다.

### 1.2. `Pipeline` 기본 사용 예시
`Pipeline`은 `make_pipeline` 함수를 사용하여 더 간결하게 생성할 수도 있습니다.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 파이프라인 정의: StandardScaler -> LogisticRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()), # 첫 번째 단계: 데이터 스케일링
    ('logreg', LogisticRegression(max_iter=200, random_state=42)) # 두 번째 단계: 로지스틱 회귀 모델
])

# 파이프라인 학습
pipeline.fit(X_train, y_train)

# 예측 및 평가
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"파이프라인을 사용한 모델 정확도: {accuracy:.4f}")
```

## 2. `ColumnTransformer`: 혼합 데이터 타입 전처리

### 2.1. `ColumnTransformer`의 필요성
실제 데이터셋은 수치형 특성(Numerical Features)과 범주형 특성(Categorical Features)이 혼합된 경우가 많습니다. 예를 들어, '나이', '수입'과 같은 수치형 특성에는 스케일링이 필요하고, '도시', '성별'과 같은 범주형 특성에는 원-핫 인코딩이 필요합니다. `Pipeline`만으로는 모든 특성에 동일한 변환을 적용하므로, 이러한 혼합 데이터 타입에 서로 다른 전처리 방식을 적용하기 어렵습니다.

`ColumnTransformer`는 이러한 문제를 해결하기 위해 도입되었습니다. 각기 다른 컬럼 그룹에 서로 다른 전처리 단계를 적용할 수 있도록 하여, 파이프라인 내에서 복잡한 전처리 로직을 구현하는 핵심 도구입니다.

### 2.2. `ColumnTransformer` 기본 사용법
`ColumnTransformer`는 `(이름, 변환기, 적용할 컬럼 리스트)` 형태의 튜플 리스트를 `transformers` 파라미터로 받습니다.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

# 가상 데이터 생성
data = {
    'Age': [25, 45, 35, 50, 23, np.nan],
    'City': ['New York', 'London', 'Paris', 'New York', 'Paris', 'London'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Salary': [50000, 80000, 65000, 90000, 45000, 70000],
    'Purchased': [0, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

X = df.drop('purchased', axis=1)
y = df['purchased']

# 특성별 컬럼 이름 정의
numerical_features = ['Age', 'Salary']
categorical_features_ohe = ['City'] # One-Hot Encoding 적용할 컬럼
categorical_features_ordinal = ['Gender'] # Ordinal Encoding 적용할 컬럼

# ColumnTransformer 정의
# 'remainder='passthrough''는 변환기에 지정되지 않은 컬럼을 그대로 통과시킵니다.
# 'remainder='drop''은 지정되지 않은 컬럼을 제거합니다.
preprocessor = ColumnTransformer(
    transformers=[
        ('num_imputer', SimpleImputer(strategy='mean'), numerical_features), # 숫자형 결측치 처리
        ('num_scaler', StandardScaler(), numerical_features), # 숫자형 스케일링
        ('cat_ohe', OneHotEncoder(handle_unknown='ignore'), categorical_features_ohe), # 범주형 원-핫 인코딩
        ('cat_ordinal', OrdinalEncoder(), categorical_features_ordinal) # 범주형 순서형 인코딩
    ],
    remainder='passthrough' # 지정되지 않은 컬럼은 그대로 유지
)

# ColumnTransformer 적용
X_processed = preprocessor.fit_transform(df.drop('purchased', axis=1))
print(f"ColumnTransformer 적용 후 데이터 형태: {X_processed.shape}")
print(f"ColumnTransformer 적용 후 데이터 (일부):\n{X_processed[:5]}")
```

### 2.3. `make_column_transformer`를 이용한 간결한 문법
`make_column_transformer` 함수는 `ColumnTransformer`를 더 간결하게 생성할 수 있도록 돕습니다. 변환기의 이름을 자동으로 생성해주므로, 코드가 더 깔끔해집니다.

```python
from sklearn.compose import make_column_transformer

# make_column_transformer 사용 예시
preprocessor_concise = make_column_transformer(
    (SimpleImputer(strategy='mean'), numerical_features),
    (StandardScaler(), numerical_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features_ohe),
    (OrdinalEncoder(), categorical_features_ordinal),
    remainder='passthrough'
)

X_processed_concise = preprocessor_concise.fit_transform(df.drop('purchased', axis=1))
print(f"\nmake_column_transformer 적용 후 데이터 형태: {X_processed_concise.shape}")
```

### 2.4. `FeatureUnion` vs. `ColumnTransformer`
`FeatureUnion`은 여러 변환기의 출력을 단순히 결합(concatenate)하는 데 사용됩니다. 하지만 `FeatureUnion`은 특정 컬럼에만 변환을 적용하는 기능을 직접 제공하지 않으며, 모든 변환기가 모든 입력 특성에 적용된다고 가정합니다.

반면 `ColumnTransformer`는 특정 컬럼에만 특정 변환을 적용하고, 나머지 컬럼은 그대로 두거나 제거하는 등 **컬럼별로 다른 전처리 파이프라인을 구성**하는 데 특화되어 있습니다. 따라서 혼합 데이터 타입 전처리에는 `ColumnTransformer`가 훨씬 강력하고 유연한 도구입니다.

## 3. 실습: 혼합 데이터 타입 파이프라인 구축

### 3.1. 데이터 준비
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 가상 데이터 생성 (수치형 + 범주형 + 결측치)
data = {
    'Age': [25, 45, 35, 50, 23, np.nan, 30, 40, 55, 28],
    'City': ['New York', 'London', 'Paris', 'New York', 'Paris', 'London', 'Berlin', 'New York', 'Paris', 'London'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Salary': [50000, 80000, 65000, 90000, 45000, 70000, 60000, 85000, 95000, 55000],
    'Purchased': [0, 1, 1, 1, 0, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df.drop('purchased', axis=1)
y = df['purchased']

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 특성별 컬럼 이름 정의
numerical_features = ['Age', 'Salary']
categorical_features = ['City', 'Gender']
```

### 3.2. `ColumnTransformer`를 포함한 파이프라인 구축 및 학습
```python
# 숫자형 특성 전처리 파이프라인
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 범주형 특성 전처리 파이프라인
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 범주형 결측치 처리
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer 정의: 각 특성 그룹에 다른 전처리 적용
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 최종 파이프라인 구축: 전처리 -> 모델
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

# 파이프라인 학습
full_pipeline.fit(X_train, y_train)

# 예측 및 평가
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"ColumnTransformer를 사용한 파이프라인 정확도: {accuracy:.4f}")
```

## 4. 파이프라인 심화 활용 팁
*   **하이퍼파라미터 튜닝:** `ColumnTransformer` 내의 변환기 파라미터도 `GridSearchCV`나 `RandomizedSearchCV`를 통해 튜닝할 수 있습니다. 예를 들어, `full_pipeline.named_steps['preprocessor'].named_transformers['num'].named_steps['imputer'].strategy`와 같이 접근하여 `preprocessor__num__imputer__strategy`와 같은 파라미터 이름을 사용합니다.
*   **커스텀 변환기:** `FunctionTransformer`나 `BaseEstimator`, `TransformerMixin`을 상속받아 자신만의 커스텀 변환기를 만들어 파이프라인에 통합할 수 있습니다.
*   **모델 스택킹/앙상블:** 파이프라인의 마지막 단계에 여러 모델을 결합하는 `VotingClassifier`나 `StackingClassifier` 등을 사용하여 앙상블 모델을 구축할 수 있습니다.