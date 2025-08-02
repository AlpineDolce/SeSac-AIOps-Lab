<h2>머신러닝을 위한 Scikit-Learn 라이브러리: 핵심 개념 및 활용 심화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Scikit-learn 라이브러리의 핵심 이론과 개념을 총체적으로 정리한 자료입니다. 데이터 전처리, 지도 학습(분류 및 회귀), 비지도 학습(클러스터링 및 차원 축소), 모델 선택 및 평가, 파이프라인 구축, 모델 영속성, 그리고 다른 파이썬 라이브러리(Pandas, NumPy, Matplotlib/Seaborn)와의 연동을 포함한 머신러닝 워크플로우 전반을 상세히 다룹니다. 본 문서를 통해 Scikit-learn에 대한 깊이 있는 이해를 돕고, 실제 데이터 분석 및 Machine Learning(ML) 문제 해결에 효과적으로 적용하는 역량을 강화하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Scikit-learn (사이킷런): 머신러닝 핵심 라이브러리](#1-scikit-learn-사이킷런-머신러닝-핵심-라이브러리)
  - [1.1. Scikit-learn 소개](#11-scikit-learn-소개)
  - [1.2. Scikit-learn 라이브러리란?](#12-scikit-learn-라이브러리란)
  - [1.3. Scikit-learn의 주요 구성요소](#13-scikit-learn의-주요-구성요소)
  - [1.4. Scikit-learn 설치 및 환경 설정](#14-scikit-learn-설치-및-환경-설정)
  - [1.5. Scikit-learn의 머신러닝 워크플로우](#15-scikit-learn의-머신러닝-워크플로우)
- [2. 데이터 전처리 (Data Preprocessing)](#2-데이터-전처리-data-preprocessing)
  - [2.1. 결측치 처리 (Handling Missing Values)](#21-결측치-처리-handling-missing-values)
  - [2.2. 데이터 스케일링 (Data Scaling)](#22-데이터-스케일링-data-scaling)
  - [2.3. 범주형 데이터 인코딩 (Categorical Data Encoding)](#23-범주형-데이터-인코딩-categorical-data-encoding)
  - [2.4. 특성 공학 (Feature Engineering)](#24-특성-공학-feature-engineering)
- [3. 지도 학습 (Supervised Learning)](#3-지도-학습-supervised-learning)
  - [3.1. 분류 (Classification)](#31-분류-classification)
  - [3.2. 회귀 (Regression)](#32-회귀-regression)
- [4. 비지도 학습 (Unsupervised Learning)](#4-비지도-학습-unsupervised-learning)
  - [4.4.1. 클러스터링 (Clustering)](#441-클러스터링-clustering)
  - [4.4.2. 차원 축소 (Dimensionality Reduction)](#442-차원-축소-dimensionality-reduction)
- [5. 모델 선택 및 평가 (Model Selection \& Evaluation)](#5-모델-선택-및-평가-model-selection--evaluation)
  - [5.1. 데이터 분할 (Data Splitting)](#51-데이터-분할-data-splitting)
  - [5.2. 교차 검증 (Cross-Validation)](#52-교차-검증-cross-validation)
  - [5.3. 하이퍼파라미터 튜닝 (Hyperparameter Tuning)](#53-하이퍼파라미터-튜닝-hyperparameter-tuning)
  - [5.4. 성능 평가 지표 (Evaluation Metrics)](#54-성능-평가-지표-evaluation-metrics)
- [6. 파이프라인 (Pipeline)](#6-파이프라인-pipeline)
  - [6.1. 파이프라인의 개념 및 장점](#61-파이프라인의-개념-및-장점)
  - [6.2. `Pipeline` 사용 예시](#62-pipeline-사용-예시)
- [7. 모델 영속성 (Model Persistence)](#7-모델-영속성-model-persistence)
  - [7.1. 모델 저장 및 로드](#71-모델-저장-및-로드)
- [8. Scikit-learn과 다른 라이브러리 연동](#8-scikit-learn과-다른-라이브러리-연동)
  - [8.1. Pandas와의 연동](#81-pandas와의-연동)
  - [8.2. NumPy와의 연동](#82-numpy와의-연동)
  - [8.3. Matplotlib/Seaborn과의 연동](#83-matplotlibseaborn과의-연동)
- [9. 실제 ML/DL 적용 사례 (Scikit-learn 중심)](#9-실제-mldl-적용-사례-scikit-learn-중심)
  - [9.1. 분류 문제: 붓꽃(Iris) 데이터셋 분류](#91-분류-문제-붓꽃iris-데이터셋-분류)
  - [9.2. 회귀 문제: 보스턴 주택 가격 예측](#92-회귀-문제-보스턴-주택-가격-예측)
  - [9.3. 클러스터링 문제: 고객 세분화](#93-클러스터링-문제-고객-세분화)


## 1. Scikit-learn (사이킷런): 머신러닝 핵심 라이브러리

### 1.1. Scikit-learn 소개

Scikit-learn (사이킷런)은 파이썬으로 머신러닝을 수행하는 데 가장 널리 사용되는 오픈소스 라이브러리입니다. 2007년 David Cournapeau에 의해 처음 개발되었으며, 이후 활발한 커뮤니티 기여를 통해 지속적으로 발전해왔습니다.

### 1.2. Scikit-learn 라이브러리란?

Scikit-learn은 다양한 머신러닝 알고리즘과 유틸리티를 효율적으로 사용할 수 있도록 설계된 파이썬 라이브러리입니다. 다음과 같은 핵심적인 특징을 가집니다:

1.  **일관된 API**: 모든 모델과 변환기가 `fit()`, `transform()`, `predict()`와 같은 일관된 메서드 이름을 사용하여 직관적이고 사용하기 쉽습니다. 이는 다른 모델이나 전처리 기법으로 쉽게 교체하며 실험할 수 있게 합니다.
2.  **다양한 알고리즘 지원**: 분류(Classification), 회귀(Regression), 클러스터링(Clustering), 차원 축소(Dimensionality Reduction) 등 지도 학습 및 비지도 학습의 광범위한 알고리즘을 제공합니다.
3.  **데이터 전처리 및 특성 공학 도구**: 데이터 스케일링, 인코딩, 결측치 처리, 특성 선택 등 머신러닝 워크플로우의 핵심인 데이터 전처리 및 특성 공학 기능을 강력하게 지원합니다.
4.  **모델 선택 및 평가 도구**: 교차 검증(Cross-validation), 하이퍼파라미터 튜닝(GridSearchCV, RandomizedSearchCV), 다양한 성능 평가 지표 등을 제공하여 모델의 성능을 객관적으로 검증하고 최적화할 수 있습니다.
5.  **활발한 커뮤니티 및 문서**: 풍부한 예제와 잘 정리된 문서를 통해 학습 및 문제 해결에 용이하며, 지속적인 업데이트와 개선이 이루어지고 있습니다.

### 1.3. Scikit-learn의 주요 구성요소

Scikit-learn은 기능별로 다양한 모듈로 구성되어 있으며, 주요 모듈은 다음과 같습니다:

*   **`sklearn.base`**: 모든 Scikit-learn 추정기(Estimator)의 기본 클래스를 정의합니다. `fit`, `transform`, `predict` 등의 메서드가 여기서 정의됩니다.
*   **`sklearn.preprocessing`**: 데이터 스케일링(MinMaxScaler, StandardScaler, RobustScaler), 범주형 인코딩(OneHotEncoder, LabelEncoder), 다항 특성 생성(PolynomialFeatures) 등 다양한 전처리 기능을 제공합니다.
*   **`sklearn.impute`**: 결측치 처리(SimpleImputer, KNNImputer) 기능을 제공합니다.
*   **`sklearn.feature_selection`**: 특성 선택(SelectKBest, RFE) 기능을 제공하여 모델 학습에 가장 중요한 특성을 선별할 수 있게 합니다.
*   **`sklearn.feature_selection`**: 특성 선택(SelectKBest, RFE) 기능을 제공하여 모델 학습에 가장 중요한 특성을 선별할 수 있게 합니다.
*   **`sklearn.model_selection`**: 데이터셋 분할(train_test_split), 교차 검증(KFold, StratifiedKFold), 하이퍼파라미터 튜닝(GridSearchCV, RandomizedSearchCV) 등 모델 선택 및 평가에 필요한 도구를 제공합니다.
*   **`sklearn.metrics`**: 분류, 회귀, 클러스터링 등 다양한 문제 유형에 대한 성능 평가 지표(accuracy_score, precision_score, recall_score, mean_squared_error, r2_score 등)를 제공합니다.
*   **`sklearn.linear_model`**: 선형 회귀, 로지스틱 회귀, Ridge, Lasso 등 선형 모델을 포함합니다.
*   **`sklearn.tree`**: 의사결정트리 분류기 및 회귀기를 제공합니다.
*   **`sklearn.ensemble`**: 랜덤 포레스트, Gradient Boosting 등 앙상블 모델을 제공합니다.
*   **`sklearn.svm`**: 서포트 벡터 머신(SVM) 분류기 및 회귀기를 제공합니다.
*   **`sklearn.neighbors`**: K-최근접 이웃(KNN) 분류기 및 회귀기를 제공합니다.
*   **`sklearn.cluster`**: K-Means, DBSCAN 등 클러스터링 알고리즘을 제공합니다.
*   **`sklearn.decomposition`**: PCA, NMF 등 차원 축소 알고리즘을 제공합니다.

### 1.4. Scikit-learn 설치 및 환경 설정

Scikit-learn은 파이썬 패키지 관리자인 `pip`를 사용하여 쉽게 설치할 수 있습니다. 설치 전에 `numpy`와 `scipy`가 미리 설치되어 있어야 합니다.

```bash
pip install numpy scipy scikit-learn
```

아나콘다(Anaconda) 환경을 사용한다면, 다음 명령어를 통해 설치할 수 있습니다.

```bash
conda install scikit-learn
```

설치 후에는 파이썬 스크립트나 Jupyter Notebook에서 `import sklearn`을 통해 라이브러리를 사용할 수 있습니다.

### 1.5. Scikit-learn의 머신러닝 워크플로우

Scikit-learn은 머신러닝 프로젝트의 일반적인 워크플로우를 효율적으로 지원합니다.

1.  **데이터 로딩 및 탐색**: Pandas와 NumPy를 사용하여 데이터를 불러오고 기본적인 통계 및 시각화를 통해 데이터를 이해합니다.
2.  **데이터 전처리**: `sklearn.preprocessing`, `sklearn.impute` 모듈의 변환기(Transformer)를 사용하여 결측치 처리, 스케일링, 인코딩 등을 수행합니다.
    *   **변환기(Transformer)의 `fit()` 및 `transform()` 메서드**:
        *   `fit()`: 훈련 데이터로부터 변환에 필요한 파라미터(예: MinMaxScaler의 최솟값/최댓값, StandardScaler의 평균/표준편차)를 학습합니다.
        *   `transform()`: 학습된 파라미터를 사용하여 데이터를 변환합니다.
        *   `fit_transform()`: `fit()`과 `transform()`을 한 번에 수행합니다.
3.  **데이터 분할**: `sklearn.model_selection.train_test_split`을 사용하여 데이터를 훈련 세트와 테스트 세트로 나눕니다.
4.  **모델 선택 및 학습**: `sklearn`의 다양한 알고리즘(Estimator) 중 문제 유형에 맞는 모델을 선택하고, 훈련 세트에 대해 `fit()` 메서드를 호출하여 모델을 학습시킵니다.
5.  **예측**: 학습된 모델의 `predict()` 메서드를 사용하여 새로운 데이터에 대한 예측을 수행합니다. 분류 모델의 경우 `predict_proba()`를 통해 확률을 얻을 수도 있습니다.
6.  **모델 평가**: `sklearn.metrics` 모듈의 다양한 평가 지표를 사용하여 모델의 성능을 객관적으로 측정합니다.
7.  **하이퍼파라미터 튜닝**: `sklearn.model_selection`의 `GridSearchCV`나 `RandomizedSearchCV`를 사용하여 모델의 성능을 최적화합니다.

이러한 일관된 워크플로우는 머신러닝 모델 개발 과정을 체계적이고 효율적으로 만들어 줍니다.

## 2. 데이터 전처리 (Data Preprocessing)

데이터 전처리는 머신러닝 모델의 성능에 결정적인 영향을 미치는 중요한 단계입니다. Scikit-learn은 다양한 전처리 도구를 제공하여 원시 데이터를 모델 학습에 적합한 형태로 변환할 수 있도록 돕습니다.

### 2.1. 결측치 처리 (Handling Missing Values)
결측치(Missing Values)는 데이터셋에 값이 누락된 부분을 의미합니다. 결측치를 적절히 처리하지 않으면 모델 학습에 오류가 발생하거나 성능이 저하될 수 있습니다. Scikit-learn의 `sklearn.impute` 모듈은 결측치를 채우는(imputation) 다양한 전략을 제공합니다.

<h4>2.1.1. `SimpleImputer`</h4>
`SimpleImputer`는 가장 기본적인 결측치 처리 도구로, 평균(mean), 중앙값(median), 최빈값(most_frequent), 또는 상수(constant)와 같은 간단한 통계량으로 결측치를 대체합니다.

```python
import numpy as np
from sklearn.impute import SimpleImputer

# 결측치가 포함된 데이터
data = np.array([
    [1, 2, np.nan, 4],
    [5, np.nan, 7, 8],
    [9, 10, 11, np.nan],
    [13, 14, 15, 16]
])

print("원본 데이터:\n", data)

# 평균으로 결측치 대체
imputer_mean = SimpleImputer(strategy='mean')
data_imputed_mean = imputer_mean.fit_transform(data)
print("\n평균으로 대체된 데이터:\n", data_imputed_mean)

# 중앙값으로 결측치 대체
imputer_median = SimpleImputer(strategy='median')
data_imputed_median = imputer_median.fit_transform(data)
print("\n중앙값으로 대체된 데이터:\n", data_imputed_median)

# 최빈값으로 결측치 대체
imputer_most_frequent = SimpleImputer(strategy='most_frequent')
data_imputed_most_frequent = imputer_most_frequent.fit_transform(data)
print("\n최빈값으로 대체된 데이터:\n", data_imputed_most_frequent)

# 특정 상수(0)로 결측치 대체
imputer_constant = SimpleImputer(strategy='constant', fill_value=0)
data_imputed_constant = imputer_constant.fit_transform(data)
print("\n상수(0)로 대체된 데이터:\n", data_imputed_constant)
```

<h4>2.1.2. `KNNImputer`</h4>
`KNNImputer`는 K-최근접 이웃(K-Nearest Neighbors) 알고리즘을 사용하여 결측치를 대체합니다. 이는 결측치가 있는 샘플과 가장 가까운 K개의 이웃 샘플들의 값을 기반으로 결측치를 추정하므로, `SimpleImputer`보다 더 정교한 대체가 가능합니다.

```python
import numpy as np
from sklearn.impute import KNNImputer

# 결측치가 포함된 데이터
data = np.array([
    [1, 2, np.nan, 4],
    [5, np.nan, 7, 8],
    [9, 10, 11, np.nan],
    [13, 14, 15, 16]
])

print("원본 데이터:\n", data)

# KNNImputer를 사용하여 결측치 대체 (n_neighbors=2)
imputer_knn = KNNImputer(n_neighbors=2)
data_imputed_knn = imputer_knn.fit_transform(data)
print("\nKNNImputer로 대체된 데이터 (n_neighbors=2):\n", data_imputed_knn)
```

### 2.2. 데이터 스케일링 (Data Scaling)
데이터 스케일링은 특성(Feature)들의 값 범위를 조정하여 모델 학습 시 특정 특성이 다른 특성보다 더 큰 영향을 미치는 것을 방지하고, 모델의 수렴 속도 및 성능을 향상시키는 데 도움을 줍니다. 특히 거리 기반 알고리즘(KNN, SVM 등)에서 중요합니다.

<h4>2.2.1. `StandardScaler`</h4>
`StandardScaler`는 각 특성의 평균을 0, 표준편차를 1로 조정하여 정규 분포와 유사하게 만듭니다. 이상치에 민감하게 반응할 수 있습니다.

$$ X_{scaled} = \frac{X - \mu}{\sigma} $$

여기서 $\mu$는 특성의 평균, $\sigma$는 특성의 표준편차입니다.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

# 스케일링할 데이터
data = np.array([
    [1, 100],
    [2, 200],
    [3, 300],
    [4, 400]
])

print("원본 데이터:\n", data)

# StandardScaler 적용
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
print("\nStandardScaler 적용 후:\n", data_scaled)
print("평균:\n", scaler.mean_)
print("표준편차:\n", scaler.scale_)
```

<h4>2.2.2. `MinMaxScaler`</h4>
`MinMaxScaler`는 각 특성의 값을 0과 1 사이(또는 지정된 범위)로 조정합니다. 데이터의 분포를 유지하면서 최솟값과 최댓값을 기준으로 스케일링합니다. 이상치에 매우 민감합니다.

$$ X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}} $$

여기서 $X_{min}$은 특성의 최솟값, $X_{max}$는 특성의 최댓값입니다.

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 스케일링할 데이터
data = np.array([
    [1, 100],
    [2, 200],
    [3, 300],
    [4, 400]
])

print("원본 데이터:\n", data)

# MinMaxScaler 적용
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
print("\nMinMaxScaler 적용 후:\n", data_scaled)
print("최솟값:\n", scaler.data_min_)
print("최댓값:\n", scaler.data_max_)
```

<h4>2.2.3. `RobustScaler`</h4>
`RobustScaler`는 중앙값(median)과 사분위 범위(IQR: Interquartile Range)를 사용하여 특성을 스케일링합니다. 이는 이상치(Outlier)의 영향을 최소화하는 데 강점이 있습니다.

$$ X_{scaled} = \frac{X - Q_2}{Q_3 - Q_1} $$

여기서 $Q_1$은 1사분위수(25th percentile), $Q_2$는 중앙값(50th percentile), $Q_3$는 3사분위수(75th percentile)입니다.

```python
import numpy as np
from sklearn.preprocessing import RobustScaler

# 스케일링할 데이터 (이상치 포함)
data = np.array([
    [1, 100],
    [2, 200],
    [3, 300],
    [4, 400],
    [100, 10000] # 이상치
])

print("원본 데이터:\n", data)

# RobustScaler 적용
scaler = RobustScaler()
data_scaled = scaler.fit_transform(data)
print("\nRobustScaler 적용 후:\n", data_scaled)
print("중앙값:\n", scaler.center_)
print("스케일 (IQR):\n", scaler.scale_)
```

###  2.3. 범주형 데이터 인코딩 (Categorical Data Encoding)
머신러닝 모델은 대부분 숫자형 데이터를 입력으로 받기 때문에, '남', '여' 또는 '서울', '부산'과 같은 범주형 데이터를 숫자형으로 변환하는 과정이 필요합니다. Scikit-learn은 `sklearn.preprocessing` 모듈을 통해 다양한 인코딩 방법을 제공합니다.

<h4>2.3.1. `LabelEncoder`</h4>
`LabelEncoder`는 범주형 변수의 각 고유한 값에 0부터 N-1까지의 정수 레이블을 할당합니다. 주로 타겟 변수(종속 변수)와 같이 순서가 없는 범주형 데이터를 인코딩할 때 사용됩니다. 특성(독립 변수)에 적용할 경우, 모델이 잘못된 순서 관계를 학습할 수 있으므로 주의해야 합니다.

```python
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 범주형 데이터
data = np.array(['red', 'green', 'blue', 'red', 'green'])
print("원본 데이터:", data)

# LabelEncoder 적용
encoder = LabelEncoder()
data_encoded = encoder.fit_transform(data)
print("\n인코딩된 데이터:", data_encoded)
print("클래스 (원본 레이블):", encoder.classes_)

# 인코딩된 데이터를 다시 원본으로 디코딩
data_decoded = encoder.inverse_transform(data_encoded)
print("\n디코딩된 데이터:", data_decoded)
```

<h4>2.3.2. `OneHotEncoder`</h4>
`OneHotEncoder`는 범주형 변수를 '원-핫(One-Hot)' 벡터로 변환합니다. 각 범주를 독립적인 이진(0 또는 1) 특성으로 표현하여, 모델이 범주 간의 잘못된 순서 관계를 학습하는 것을 방지합니다. 주로 특성 변수(독립 변수)에 사용됩니다.

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# 범주형 데이터 (2차원 배열 형태여야 함)
data = np.array([['red'], ['green'], ['blue'], ['red'], ['green']])
print("원본 데이터:\n", data)

# OneHotEncoder 적용
# sparse_output=False로 설정하여 NumPy 배열로 반환 (기본값은 희소 행렬)
encoder = OneHotEncoder(sparse_output=False)
data_encoded = encoder.fit_transform(data)
print("\n원-핫 인코딩된 데이터:\n", data_encoded)
print("카테고리 (원본 레이블):", encoder.categories_)

# 새로운 데이터에 적용
new_data = np.array([['blue'], ['red']])
new_data_encoded = encoder.transform(new_data)
print("\n새로운 데이터에 적용된 원-핫 인코딩:\n", new_data_encoded)
```

###  2.4. 특성 공학 (Feature Engineering)
특성 공학은 기존 특성을 사용하여 새로운 특성을 만들거나, 특성을 변환하여 모델의 성능을 향상시키는 과정입니다. Scikit-learn은 몇 가지 유용한 특성 공학 도구를 제공합니다.

<h4>2.4.1. `PolynomialFeatures`</h4>
`PolynomialFeatures`는 기존 특성들의 다항식 조합을 생성하여 새로운 특성을 만듭니다. 이는 선형 모델이 비선형 관계를 학습할 수 있도록 돕습니다.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# 원본 특성 데이터
X = np.array([[1, 2],
              [3, 4]])
print("원본 특성:\n", X)

# 2차 다항 특성 생성 (degree=2)
# include_bias=False로 설정하여 절편(bias) 특성(모두 1인 컬럼)은 생성하지 않음
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print("\n2차 다항 특성 생성 후:\n", X_poly)
print("생성된 특성 이름:\n", poly.get_feature_names_out(['feature1', 'feature2']))
# 결과 해석:
# [1, 2] -> [1, 2, 1^2, 1*2, 2^2] = [1, 2, 1, 2, 4]
# [3, 4] -> [3, 4, 3^2, 3*4, 4^2] = [3, 4, 9, 12, 16]
# include_bias=False이므로 첫 번째 1은 제외됨.
# feature1, feature2, feature1^2, feature1*feature2, feature2^2
```

<h4>2.4.2. 특성 선택 (Feature Selection)</h4>
특성 선택은 모델 학습에 가장 관련성이 높거나 중요한 특성들만 선택하여 모델의 복잡성을 줄이고, 과적합을 방지하며, 학습 시간을 단축시키는 과정입니다. Scikit-learn의 `sklearn.feature_selection` 모듈은 다양한 특성 선택 방법을 제공합니다.

**1. 단변량 통계 기반 선택 (`SelectKBest`, `SelectPercentile`)**
각 특성과 타겟 변수 간의 통계적 관계를 평가하여 상위 K개의 특성 또는 상위 N%의 특성을 선택합니다. 분류 문제에서는 카이제곱($\chi^2$) 통계량, 회귀 문제에서는 F-값(F-value) 등을 사용할 수 있습니다.

```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

print("원본 특성 데이터 형태:", X.shape)

# 상위 2개의 특성 선택 (분류 문제이므로 f_classif 사용)
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

print("\n선택된 특성 데이터 형태:", X_selected.shape)
print("선택된 특성 인덱스:", selector.get_support(indices=True)) # 선택된 특성의 원본 인덱스
print("각 특성의 점수:", selector.scores_) # 각 특성의 통계 점수
```

**2. 모델 기반 특성 선택 (`SelectFromModel`)**
특성 중요도(Feature Importance)를 제공하는 모델(예: 트리 기반 모델, 선형 모델)을 사용하여 특성을 선택합니다. 모델이 학습된 후, 각 특성의 중요도 점수를 기준으로 임계값 이상의 특성만 선택합니다.

```python
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

print("원본 특성 데이터 형태:", X.shape)

# RandomForestClassifier를 기반으로 특성 선택
# threshold='median'은 특성 중요도 중앙값 이상인 특성 선택
selector = SelectFromModel(estimator=RandomForestClassifier(random_state=42), threshold='median')
X_selected = selector.fit_transform(X, y)

print("\n선택된 특성 데이터 형태:", X_selected.shape)
print("선택된 특성 인덱스:", selector.get_support(indices=True))
print("각 특성의 중요도:", selector.estimator_.feature_importances_)
```

**3. 재귀적 특성 제거 (`RFE`: Recursive Feature Elimination)**
모델을 반복적으로 학습시키면서 가장 중요도가 낮은 특성을 하나씩 제거하는 방식입니다. 원하는 특성 개수가 남을 때까지 이 과정을 반복합니다. `estimator`와 `n_features_to_select`를 지정합니다.

```python
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

print("원본 특성 데이터 형태:", X.shape)

# LogisticRegression을 기반으로 RFE 수행, 2개의 특성 선택
selector = RFE(estimator=LogisticRegression(max_iter=1000, random_state=42), n_features_to_select=2)
X_selected = selector.fit_transform(X, y)

print("\n선택된 특성 데이터 형태:", X_selected.shape)
print("선택된 특성 인덱스 (True: 선택됨):", selector.support_)
print("특성 랭킹 (1이 가장 중요):", selector.ranking_)
```

## 3. 지도 학습 (Supervised Learning)

지도 학습은 가장 일반적인 머신러닝 패러다임으로, 레이블(정답)이 있는 훈련 데이터를 사용하여 모델을 학습시킵니다. Scikit-learn은 다양한 지도 학습 알고리즘을 제공하며, 크게 분류(Classification)와 회귀(Regression) 문제로 나눌 수 있습니다.

### 3.1. 분류 (Classification)
분류는 입력 데이터를 미리 정의된 여러 클래스(범주) 중 하나로 할당하는 문제입니다. 예를 들어, 이메일이 스팸인지 아닌지, 환자가 특정 질병에 걸렸는지 아닌지 등을 예측하는 것이 분류 문제에 해당합니다.

<h4>3.1.1. 로지스틱 회귀 (Logistic Regression)</h4>
로지스틱 회귀는 이름에 '회귀'가 들어가지만, 실제로는 이진 분류(Binary Classification)에 주로 사용되는 선형 모델입니다. 입력 특성의 선형 조합을 사용하여 특정 클래스에 속할 확률을 예측하고, 이 확률을 기반으로 클래스를 분류합니다. Scikit-learn의 `LogisticRegression`은 다중 클래스 분류도 지원합니다.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드 (분류 예제)
iris = load_iris()
X, y = iris.data, iris.target

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 로지스틱 회귀 모델 생성 및 학습
# max_iter를 충분히 크게 설정하여 수렴 보장
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"로지스틱 회귀 정확도: {accuracy:.4f}")

# 새로운 데이터 예측
new_data = np.array([[5.1, 3.5, 1.4, 0.2]]) # Setosa 품종의 특성
predicted_class = model.predict(new_data)
print(f"새로운 데이터 예측 클래스: {iris.target_names[predicted_class][0]}")
```

<h4>3.1.2. 의사결정 트리 (Decision Tree)</h4>
의사결정 트리는 데이터를 특정 기준에 따라 분할하여 예측을 수행하는 트리 형태의 모델입니다. 직관적이고 해석하기 쉽다는 장점이 있으며, 분류와 회귀 문제 모두에 사용될 수 있습니다. `DecisionTreeClassifier`는 분류에 사용됩니다.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 의사결정 트리 모델 생성 및 학습
# max_depth로 트리의 최대 깊이 제한 (과적합 방지)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"의사결정 트리 정확도: {accuracy:.4f}")

# 특성 중요도 확인
print(f"특성 중요도: {model.feature_importances_}")
```

<h4>3.1.3. 서포트 벡터 머신 (Support Vector Machine, SVM)</h4>
SVM은 분류, 회귀, 이상치 탐지 등에 사용되는 강력한 지도 학습 모델입니다. 데이터를 고차원 공간으로 매핑하여 클래스 간의 최적의 결정 경계(Decision Boundary)를 찾는 것을 목표로 합니다. `SVC` (Support Vector Classifier)는 분류에 사용됩니다.

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SVM 모델 생성 및 학습 (선형 커널)
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"선형 SVM 정확도: {accuracy:.4f}")

# 비선형 SVM 모델 생성 및 학습 (RBF 커널)
model_rbf = SVC(kernel='rbf', random_state=42)
model_rbf.fit(X_train, y_train)

y_pred_rbf = model_rbf.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"RBF SVM 정확도: {accuracy_rbf:.4f}")
```

<h4>3.1.4. K-최근접 이웃 (K-Nearest Neighbors, KNN)</h4>
KNN은 매우 간단하고 직관적인 비모수 분류 알고리즘입니다. 새로운 데이터 포인트가 주어졌을 때, 훈련 데이터에서 가장 가까운 K개의 이웃을 찾고, 이 이웃들의 클래스 중 가장 많은 클래스로 새로운 데이터 포인트를 분류합니다. `KNeighborsClassifier`는 분류에 사용됩니다.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN 모델 생성 및 학습 (K=3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN (K=3) 정확도: {accuracy:.4f}")

# K 값에 따른 성능 변화 확인 (예시)
accuracies = []
for k in range(1, 11):
    model_k = KNeighborsClassifier(n_neighbors=k)
    model_k.fit(X_train, y_train)
    y_pred_k = model_k.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_k))

print(f"\nK=1부터 10까지의 정확도: {accuracies}")
```

<h4>3.1.5. 앙상블 모델 (Ensemble Models)</h4>
앙상블 학습은 여러 개의 개별 모델(약한 학습기)을 조합하여 하나의 강력한 모델을 만드는 기법입니다. 개별 모델의 단점을 보완하고, 예측 성능을 향상시키며, 과적합을 줄이는 효과가 있습니다. Scikit-learn은 다양한 앙상블 모델을 제공합니다.

<h5>랜덤 포레스트 (Random Forest)</h5>
랜덤 포레스트는 여러 개의 의사결정 트리를 무작위로 생성하고, 각 트리의 예측을 종합하여 최종 예측을 수행하는 앙상블 모델입니다. 배깅(Bagging) 기법을 사용하며, 과적합에 강하고 안정적인 성능을 보입니다.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 랜덤 포레스트 모델 생성 및 학습
# n_estimators: 트리의 개수, random_state: 재현성을 위한 시드
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"랜덤 포레스트 정확도: {accuracy:.4f}")

# 특성 중요도 확인
print(f"특성 중요도: {model.feature_importances_}")
```

<h5>그레디언트 부스팅 (Gradient Boosting)</h5>
그레디언트 부스팅은 이전 모델의 예측 오차(잔차)를 보정하는 방향으로 새로운 모델을 순차적으로 추가하는 앙상블 기법입니다. 부스팅(Boosting) 계열의 대표적인 알고리즘으로, 매우 높은 예측 성능을 자랑합니다. `GradientBoostingClassifier`는 분류에 사용됩니다.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 그레디언트 부스팅 모델 생성 및 학습
# n_estimators: 트리의 개수, learning_rate: 학습률
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"그레디언트 부스팅 정확도: {accuracy:.4f}")
```

### 3.2. 회귀 (Regression)
회귀는 입력 특성을 기반으로 연속적인 숫자 값을 예측하는 문제입니다. 예를 들어, 주택 가격 예측, 주식 가격 예측, 연비 예측 등이 회귀 문제에 해당합니다.

<h4>3.2.1. 선형 회귀 (Linear Regression)</h4>
선형 회귀는 가장 간단하고 널리 사용되는 회귀 모델입니다. 입력 특성과 타겟 변수 간의 선형 관계를 모델링하여 예측을 수행합니다. `LinearRegression`은 최소 제곱법을 사용하여 최적의 회귀 계수를 찾습니다.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# 회귀용 가상 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"선형 회귀 MSE: {mse:.2f}")
print(f"선형 회귀 R2: {r2:.2f}")

# 회귀 계수 및 절편 확인
print(f"계수 (Coefficient): {model.coef_[0]:.2f}")
print(f"절편 (Intercept): {model.intercept_:.2f}")
```

<h4>3.2.2. 릿지 회귀 (Ridge Regression)</h4>
릿지 회귀는 선형 회귀에 L2 정규화(Regularization)를 추가한 모델입니다. 과적합을 방지하고 모델의 일반화 성능을 향상시키는 데 사용됩니다. L2 정규화는 회귀 계수의 크기를 제한하여 모델의 복잡성을 줄입니다. `alpha` 파라미터로 정규화 강도를 조절합니다.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# 회귀용 가상 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 릿지 회귀 모델 생성 및 학습 (alpha=1.0)
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"릿지 회귀 MSE: {mse:.2f}")
print(f"릿지 회귀 R2: {r2:.2f}")

# 회귀 계수 확인
print(f"계수 (Coefficient): {model.coef_[0]:.2f}")
```

<h4>3.2.3. 라쏘 회귀 (Lasso Regression)</h4>
라쏘 회귀는 선형 회귀에 L1 정규화(Regularization)를 추가한 모델입니다. 릿지 회귀와 마찬가지로 과적합을 방지하지만, L1 정규화는 특성 선택(Feature Selection) 효과를 가집니다. 즉, 중요하지 않은 특성의 회귀 계수를 0으로 만들어 해당 특성을 모델에서 제외시킵니다. `alpha` 파라미터로 정규화 강도를 조절합니다.

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# 회귀용 가상 데이터 생성 (특성 10개)
X, y = make_regression(n_samples=100, n_features=10, noise=20, random_state=42)

# 훈련 세트와 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 라쏘 회귀 모델 생성 및 학습 (alpha=0.1)
model = Lasso(alpha=0.1, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"라쏘 회귀 MSE: {mse:.2f}")
print(f"라쏘 회귀 R2: {r2:.2f}")

# 회귀 계수 확인 (일부 계수가 0이 될 수 있음)
print(f"계수 (Coefficients): {model.coef_}")
```

## 4. 비지도 학습 (Unsupervised Learning)

비지도 학습은 레이블(정답)이 없는 데이터를 사용하여 데이터의 숨겨진 구조나 패턴을 발견하는 머신러닝 패러다임입니다. Scikit-learn은 클러스터링(Clustering)과 차원 축소(Dimensionality Reduction)와 같은 비지도 학습 알고리즘을 제공합니다.

### 4.4.1. 클러스터링 (Clustering)
클러스터링은 데이터를 유사한 특성을 가진 그룹(클러스터)으로 묶는 비지도 학습 기법입니다. 고객 세분화, 이미지 분할, 문서 분류 등에 활용됩니다.

<h4>K-평균 (K-Means)</h4>
K-평균은 가장 널리 사용되는 파티셔닝 기반 클러스터링 알고리즘입니다. 데이터를 K개의 클러스터로 나누고, 각 클러스터는 해당 클러스터의 중심(centroid)에 가장 가까운 데이터 포인트들로 구성됩니다. `KMeans` 클래스를 사용합니다.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 가상 클러스터링 데이터 생성
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

print("원본 데이터 형태:", X.shape)

# K-Means 모델 생성 및 학습 (클러스터 개수 K=4)
model = KMeans(n_clusters=4, random_state=42, n_init=10) # n_init: 다른 centroid 시드 값으로 여러 번 실행
model.fit(X)

# 각 데이터 포인트의 클러스터 레이블
labels = model.labels_
print("\n클러스터 레이블 (일부):", labels[:10])

# 클러스터 중심
centroids = model.cluster_centers_
print("클러스터 중심:\n", centroids)

# 시각화 (선택 사항)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.7, marker='X')
# plt.title("K-Means Clustering")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()
```

<h4>DBSCAN</h4>
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)은 밀도 기반 클러스터링 알고리즘입니다. 밀집된 데이터 포인트들을 클러스터로 묶고, 밀도가 낮은 영역에 있는 포인트들을 노이즈(noise)로 분류합니다. 클러스터 개수를 미리 지정할 필요가 없으며, 다양한 형태의 클러스터를 찾을 수 있다는 장점이 있습니다. `eps` (이웃 탐색 반경)와 `min_samples` (클러스터를 형성하는 데 필요한 최소 샘플 수) 파라미터가 중요합니다.

```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 가상 클러스터링 데이터 생성 (초승달 모양)
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

print("원본 데이터 형태:", X.shape)

# DBSCAN 모델 생성 및 학습
# eps: 0.3, min_samples: 5
model = DBSCAN(eps=0.3, min_samples=5)
model.fit(X)

# 각 데이터 포인트의 클러스터 레이블 (-1은 노이즈)
labels = model.labels_
print("\n클러스터 레이블 (일부):", labels[:10])
print("고유한 클러스터 레이블:", np.unique(labels))

# 시각화 (선택 사항)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# plt.title("DBSCAN Clustering")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()
```

### 4.4.2. 차원 축소 (Dimensionality Reduction)
차원 축소는 데이터의 특성(차원) 개수를 줄이는 기법입니다. 데이터의 본질적인 정보를 최대한 유지하면서 불필요하거나 중복되는 특성을 제거하여, 모델의 복잡성을 줄이고, 과적합을 방지하며, 시각화를 용이하게 합니다.

<h4>주성분 분석 (Principal Component Analysis, PCA)</h4>
PCA는 가장 널리 사용되는 선형 차원 축소 기법입니다. 데이터의 분산을 가장 잘 설명하는 새로운 직교 좌표계(주성분)를 찾아 데이터를 이 주성분 공간에 투영합니다. `PCA` 클래스를 사용합니다.

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

print("원본 데이터 형태:", X.shape)

# PCA 모델 생성 (2개의 주성분으로 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("\nPCA 적용 후 데이터 형태:", X_pca.shape)
print("설명된 분산 비율:", pca.explained_variance_ratio_)
print("누적 설명된 분산 비율:", np.sum(pca.explained_variance_ratio_))

# 시각화 (선택 사항)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA of Iris Dataset")
# plt.colorbar(label='Species')
# plt.show()
```

<h4>비음수 행렬 분해 (Non-negative Matrix Factorization, NMF)</h4>
NMF는 비음수 값을 가지는 행렬을 두 개의 비음수 행렬의 곱으로 분해하는 차원 축소 기법입니다. 텍스트 마이닝(토픽 모델링), 이미지 처리(얼굴 인식), 추천 시스템 등에 활용됩니다. `NMF` 클래스를 사용합니다.

```python
import numpy as np
from sklearn.decomposition import NMF

# 비음수 값을 가지는 데이터 행렬 (예: 문서-단어 행렬)
X = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1]
])

print("원본 데이터 형태:\n", X)

# NMF 모델 생성 (2개의 잠재 요인으로 분해)
nmf = NMF(n_components=2, random_state=42, max_iter=1000)
X_nmf = nmf.fit_transform(X)

print("\nNMF 적용 후 변환된 데이터 (W):\n", X_nmf)
print("NMF 적용 후 컴포넌트 (H):\n", nmf.components_)

# 원본 행렬 복원 확인 (W @ H)
reconstructed_X = X_nmf @ nmf.components_
print("\n재구성된 행렬:\n", reconstructed_X)
print("원본과 재구성된 행렬이 근사적으로 같은가?", np.allclose(X, reconstructed_X, atol=1e-5))
```

## 5. 모델 선택 및 평가 (Model Selection & Evaluation)

머신러닝 모델을 개발하는 과정에서 모델의 성능을 객관적으로 평가하고 최적의 모델을 선택하는 것은 매우 중요합니다. Scikit-learn은 이러한 작업을 위한 다양한 도구와 지표를 제공합니다.

### 5.1. 데이터 분할 (Data Splitting)
모델의 일반화 성능을 평가하기 위해 데이터를 훈련 세트(training set)와 테스트 세트(test set)로 나누는 것은 필수적입니다. 훈련 세트로 모델을 학습시키고, 테스트 세트로 모델의 성능을 평가하여 모델이 새로운, 보지 못한 데이터에 얼마나 잘 작동하는지 확인합니다.

<h4>5.1.1. `train_test_split`</h4>
`train_test_split` 함수는 데이터를 훈련 세트와 테스트 세트로 무작위로 분할하는 가장 기본적인 방법입니다. 분류 문제의 경우 `stratify` 옵션을 사용하여 클래스 비율을 유지할 수 있습니다.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

print("원본 데이터 형태:", X.shape, y.shape)

# 데이터를 훈련 세트(70%)와 테스트 세트(30%)로 분할
# random_state: 재현성을 위한 시드
# stratify: 분류 문제에서 클래스 비율을 훈련/테스트 세트에서 동일하게 유지
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\n훈련 세트 형태:", X_train.shape, y_train.shape)
print("테스트 세트 형태:", X_test.shape, y_test.shape)

print("\n원본 y의 클래스별 개수:\n", np.bincount(y))
print("훈련 y의 클래스별 개수:\n", np.bincount(y_train))
print("테스트 y의 클래스별 개수:\n", np.bincount(y_test))
```

### 5.2. 교차 검증 (Cross-Validation)
교차 검증은 데이터 분할 방식의 한계를 극복하고 모델의 일반화 성능을 더 신뢰성 있게 평가하기 위한 기법입니다. 데이터를 여러 개의 폴드(fold)로 나누어 각 폴드를 한 번씩 테스트 세트로 사용하고 나머지를 훈련 세트로 사용하여 여러 번 모델을 학습하고 평가합니다.

<h4>5.2.1. `KFold`</h4>
`KFold`는 데이터를 K개의 동일한 크기의 폴드로 나누고, 각 반복마다 하나의 폴드를 테스트 세트로, 나머지 K-1개의 폴드를 훈련 세트로 사용합니다. 회귀 문제에 주로 사용됩니다.

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# 가상 회귀 데이터 생성
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# KFold 교차 검증 설정 (5개의 폴드)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = []

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    print(f"Fold {fold+1} MSE: {mse:.2f}")

print(f"\n평균 MSE: {np.mean(mse_scores):.2f}")
print(f"MSE 표준편차: {np.std(mse_scores):.2f}")
```

<h4>5.2.2. `StratifiedKFold`</h4>
`StratifiedKFold`는 `KFold`와 유사하지만, 각 폴드에 원본 데이터셋의 클래스 비율이 유지되도록 데이터를 분할합니다. 이는 분류 문제, 특히 클래스 불균형이 있는 데이터셋에서는 모델의 성능을 더 정확하게 평가하는 데 중요합니다.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# StratifiedKFold 교차 검증 설정 (5개의 폴드)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

print(f"\n평균 정확도: {np.mean(accuracy_scores):.4f}")
print(f"정확도 표준편차: {np.std(accuracy_scores):.4f}")
```

### 5.3. 하이퍼파라미터 튜닝 (Hyperparameter Tuning)
하이퍼파라미터는 모델 학습 전에 사용자가 직접 설정하는 값으로, 모델의 성능에 큰 영향을 미칩니다. 최적의 하이퍼파라미터 조합을 찾는 과정을 하이퍼파라미터 튜닝이라고 합니다. Scikit-learn은 `GridSearchCV`와 `RandomizedSearchCV`를 통해 체계적인 튜닝 방법을 제공합니다.

<h4>5.3.1. `GridSearchCV`</h4>
`GridSearchCV`는 사용자가 지정한 하이퍼파라미터의 모든 가능한 조합에 대해 모델을 학습하고 교차 검증을 수행하여 최적의 조합을 찾습니다. 모든 조합을 탐색하므로 시간이 오래 걸릴 수 있지만, 최적의 조합을 찾을 가능성이 높습니다.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 탐색할 하이퍼파라미터 그리드 정의
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# GridSearchCV 객체 생성
# estimator: 튜닝할 모델
# param_grid: 탐색할 하이퍼파라미터 그리드
# cv: 교차 검증 폴드 수
# scoring: 평가 지표
# n_jobs: 사용할 CPU 코어 수 (-1은 모든 코어 사용)
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# 그리드 서치 수행
grid_search.fit(X, y)

# 최적의 하이퍼파라미터 조합
print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")

# 최적의 모델 성능
print(f"최고 교차 검증 정확도: {grid_search.best_score_:.4f}")

# 최적의 모델
best_model = grid_search.best_estimator_
print(f"최적의 모델: {best_model}")
```

<h4>5.3.2. `RandomizedSearchCV`</h4>
`RandomizedSearchCV`는 `GridSearchCV`와 달리, 사용자가 지정한 하이퍼파라미터 공간에서 무작위로 샘플링된 조합에 대해서만 모델을 학습하고 교차 검증을 수행합니다. 모든 조합을 탐색하지 않으므로 `GridSearchCV`보다 빠르지만, 최적의 조합을 찾지 못할 수도 있습니다. 탐색할 조합의 개수(`n_iter`)를 지정합니다.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from scipy.stats import randint

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 탐색할 하이퍼파라미터 분포 정의
param_dist = {
    'n_estimators': randint(10, 200), # 10에서 200 사이의 정수
    'max_depth': randint(1, 10) # 1에서 10 사이의 정수
}

# RandomizedSearchCV 객체 생성
# n_iter: 무작위로 샘플링할 조합의 개수
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)

# 랜덤 서치 수행
random_search.fit(X, y)

# 최적의 하이퍼파라미터 조합
print(f"최적의 하이퍼파라미터: {random_search.best_params_}")

# 최적의 모델 성능
print(f"최고 교차 검증 정확도: {random_search.best_score_:.4f}")

# 최적의 모델
best_model = random_search.best_estimator_
print(f"최적의 모델: {best_model}")
```

### 5.4. 성능 평가 지표 (Evaluation Metrics)
모델의 성능을 객관적으로 측정하기 위해 다양한 평가 지표가 사용됩니다. 문제 유형(분류, 회귀)에 따라 적절한 지표를 선택하는 것이 중요합니다. Scikit-learn의 `sklearn.metrics` 모듈은 이러한 지표들을 제공합니다.

<h4>5.4.1. 분류 모델 평가 지표</h4>
분류 모델의 성능을 평가하는 데 사용되는 주요 지표들입니다.

<h5>정확도 (Accuracy)</h5>
전체 예측 중 올바르게 예측한 비율입니다. 가장 직관적인 지표이지만, 클래스 불균형이 심한 데이터셋에서는 오해의 소지가 있을 수 있습니다.

$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$

*   TP (True Positive): 실제 True를 True로 올바르게 예측
*   TN (True Negative): 실제 False를 False로 올바르게 예측
*   FP (False Positive): 실제 False를 True로 잘못 예측 (1종 오류)
*   FN (False Negative): 실제 True를 False로 잘못 예측 (2종 오류)

```python
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy:.4f}")
```

<h5>정밀도 (Precision)</h5>
양성(Positive)으로 예측한 것 중에서 실제로 양성인 비율입니다. FP를 줄이는 것이 중요할 때 사용됩니다 (예: 스팸 메일 분류, 암 진단).

$$ Precision = \frac{TP}{TP + FP} $$

```python
from sklearn.metrics import precision_score

# 이진 분류 예시 (y_true, y_pred는 0 또는 1)
y_true = [0, 1, 0, 1, 0, 0, 1, 1]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1]

precision = precision_score(y_true, y_pred)
print(f"정밀도: {precision:.4f}")

# 다중 클래스 분류의 경우 average 파라미터 사용
# precision_score(y_true_multi, y_pred_multi, average='macro')
```

<h5>재현율 (Recall)</h5>
실제 양성인 것 중에서 모델이 올바르게 양성으로 예측한 비율입니다. FN을 줄이는 것이 중요할 때 사용됩니다 (예: 놓치면 안 되는 질병 진단, 침입 탐지).

$$ Recall = \frac{TP}{TP + FN} $$

```python
from sklearn.metrics import recall_score

y_true = [0, 1, 0, 1, 0, 0, 1, 1]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1]

recall = recall_score(y_true, y_pred)
print(f"재현율: {recall:.4f}")
```

<h5>F1-점수 (F1-Score)</h5>
정밀도와 재현율의 조화 평균입니다. 정밀도와 재현율이 모두 중요할 때 사용되는 균형 지표입니다.

$$ F1-Score = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

```python
from sklearn.metrics import f1_score

y_true = [0, 1, 0, 1, 0, 0, 1, 1]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1]

f1 = f1_score(y_true, y_pred)
print(f"F1-점수: {f1:.4f}")
```

<h5>ROC 곡선 및 AUC (ROC Curve & AUC)</h5>
ROC (Receiver Operating Characteristic) 곡선은 분류 모델의 임계값(threshold) 변화에 따른 TPR (True Positive Rate, 재현율)과 FPR (False Positive Rate)의 관계를 시각화한 것입니다. AUC (Area Under the Curve)는 ROC 곡선 아래 면적으로, 1에 가까울수록 좋은 모델입니다. 이진 분류 모델의 성능을 종합적으로 평가하는 데 유용합니다.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 이진 분류 가상 데이터 생성
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 양성 클래스에 대한 예측 확률
y_prob = model.predict_proba(X_test)[:, 1]

# ROC 곡선 계산
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc:.4f}")

# ROC 곡선 시각화 (선택 사항)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
```

<h5>혼동 행렬 (Confusion Matrix)</h5>
혼동 행렬은 분류 모델의 예측 결과를 표 형태로 요약한 것입니다. 실제 클래스와 예측 클래스 간의 관계를 보여주며, TP, TN, FP, FN 값을 직접 확인할 수 있어 모델의 어떤 유형의 오류를 범하는지 파악하는 데 유용합니다.

```python
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(f"혼동 행렬:\n{cm}")

# 혼동 행렬 시각화 (선택 사항)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=iris.target_names, yticklabels=iris.target_names)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()
```

<h4>5.4.2. 회귀 모델 평가 지표</h4>
회귀 모델의 성능을 평가하는 데 사용되는 주요 지표들입니다.

<h5>평균 제곱 오차 (Mean Squared Error, MSE)</h5>
예측값과 실제값의 차이(오차)를 제곱하여 평균한 값입니다. 오차의 크기를 나타내며, 값이 작을수록 모델의 예측 성능이 좋습니다. 이상치에 민감하게 반응합니다.

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

*   $y_i$: 실제값
*   $\hat{y}_i$: 예측값
*   $n$: 데이터 포인트 수

```python
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")
```

<h5>R-제곱 (R-squared)</h5>
결정 계수(Coefficient of Determination)라고도 불리며, 모델이 분산을 얼마나 잘 설명하는지를 나타냅니다. 0과 1 사이의 값을 가지며, 1에 가까울수록 모델이 데이터를 잘 설명합니다. 음수 값도 가질 수 있으며, 이는 모델이 평균보다도 성능이 나쁘다는 것을 의미합니다.

$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} $$

*   $SS_{res}$: 잔차 제곱합 (Sum of Squares of Residuals) = $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
*   $SS_{tot}$: 총 제곱합 (Total Sum of Squares) = $\sum_{i=1}^{n} (y_i - \bar{y})^2$

```python
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"R-제곱: {r2:.2f}")
```

## 6. 파이프라인 (Pipeline)

머신러닝 워크플로우는 데이터 전처리, 특성 추출, 모델 학습 등 여러 단계로 구성됩니다. Scikit-learn의 `Pipeline`은 이러한 연속적인 변환 및 학습 과정을 하나의 객체로 묶어주는 강력한 도구입니다. 이는 코드의 가독성을 높이고, 재사용성을 향상시키며, 데이터 누수(data leakage)를 방지하는 데 도움을 줍니다.

### 6.1. 파이프라인의 개념 및 장점
파이프라인은 여러 변환기(Transformer)와 마지막 추정기(Estimator)를 순차적으로 연결한 것입니다. 데이터는 파이프라인의 각 단계를 거치면서 변환되고, 최종적으로 모델 학습에 사용됩니다.

**장점**:
1.  **코드 간결성 및 가독성**: 여러 전처리 단계와 모델 학습을 한 줄의 코드로 표현할 수 있어 코드가 간결해지고 이해하기 쉬워집니다.
2.  **데이터 누수 방지**: 교차 검증 시 훈련 데이터에만 `fit()`을 적용하고 테스트 데이터에는 `transform()`만 적용하도록 강제하여, 테스트 데이터의 정보가 훈련 과정에 유출되는 데이터 누수를 효과적으로 방지합니다.
3.  **재사용성**: 한 번 정의된 파이프라인은 다른 데이터셋이나 프로젝트에서도 쉽게 재사용할 수 있습니다.
4.  **하이퍼파라미터 튜닝 용이**: `GridSearchCV`나 `RandomizedSearchCV`와 같은 하이퍼파라미터 튜닝 도구를 사용하여 파이프라인 내의 모든 단계(전처리 및 모델)의 하이퍼파라미터를 한 번에 최적화할 수 있습니다.

### 6.2. `Pipeline` 사용 예시
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
# 단계는 (이름, 변환기/모델) 튜플의 리스트로 구성됩니다.
pipeline = Pipeline([
    ('scaler', StandardScaler()), # 첫 번째 단계: 데이터 스케일링
    ('logreg', LogisticRegression(max_iter=200, random_state=42)) # 두 번째 단계: 로지스틱 회귀 모델
])

# 파이프라인 학습 (훈련 데이터에 fit_transform, 테스트 데이터에 transform이 자동으로 적용됨)
pipeline.fit(X_train, y_train)

# 예측
y_pred = pipeline.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"파이프라인을 사용한 모델 정확도: {accuracy:.4f}")

# 파이프라인 내의 개별 단계에 접근
print(f"\n파이프라인 내 스케일러의 평균: {pipeline.named_steps['scaler'].mean_}")
```

**하이퍼파라미터 튜닝과 파이프라인**: `GridSearchCV`와 함께 파이프라인을 사용하면, 전처리 단계의 하이퍼파라미터까지 한 번에 최적화할 수 있습니다. 이때 하이퍼파라미터 이름은 `단계이름__하이퍼파라미터` 형식으로 지정합니다.

```python
from sklearn.model_selection import GridSearchCV

# 탐색할 하이퍼파라미터 그리드 정의
param_grid = {
    'scaler__with_mean': [True, False], # StandardScaler의 with_mean 파라미터
    'logreg__C': [0.1, 1, 10] # LogisticRegression의 C 파라미터
}

# GridSearchCV 객체 생성 및 수행
grid_search_pipeline = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_pipeline.fit(X, y) # 전체 데이터에 대해 교차 검증 수행

print(f"\n최적의 파이프라인 하이퍼파라미터: {grid_search_pipeline.best_params_}")
print(f"최고 교차 검증 정확도: {grid_search_pipeline.best_score_:.4f}")

# 최적의 모델
best_model = grid_search_pipeline.best_estimator_
print(f"최적의 모델: {best_model}")
```


## 7. 모델 영속성 (Model Persistence)

모델 영속성(Model Persistence)은 훈련된 머신러닝 모델을 디스크에 저장하고, 필요할 때 다시 로드하여 재훈련 없이 사용할 수 있도록 하는 과정입니다. 이는 모델을 배포하거나, 장시간이 소요되는 훈련 과정을 반복하지 않기 위해 필수적입니다. Scikit-learn은 파이썬의 표준 직렬화 라이브러리인 `pickle`과 더 효율적인 `joblib` 라이브러리를 사용하여 모델을 저장하고 로드하는 기능을 제공합니다.

### 7.1. 모델 저장 및 로드

<h4>7.1.1. `joblib` 사용</h4>
`joblib` 라이브러리는 NumPy 배열과 같이 큰 데이터를 효율적으로 직렬화하고 역직렬화하는 데 최적화되어 있습니다. Scikit-learn 모델을 저장하고 로드하는 데 권장되는 방법입니다.

```python
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 모델 훈련
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X, y)

# 모델 저장
filename = 'logistic_regression_model.joblib'
joblib.dump(model, filename)
print(f"모델이 '{filename}'으로 저장되었습니다.")

# 모델 로드
loaded_model = joblib.load(filename)
print(f"모델이 '{filename}'에서 성공적으로 로드되었습니다.")

# 로드된 모델로 예측 수행
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = loaded_model.predict(new_data)
print(f"로드된 모델로 예측: {iris.target_names[prediction][0]}")
```

<h4>7.1.2. `pickle` 사용</h4>
`pickle`은 파이썬 객체를 직렬화하는 표준 라이브러리입니다. `joblib`만큼 대용량 데이터에 최적화되어 있지는 않지만, 간단한 모델이나 파이썬 객체를 저장하는 데 사용될 수 있습니다.

```python
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# 모델 훈련
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X, y)

# 모델 저장
filename = 'logistic_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print(f"모델이 '{filename}'으로 저장되었습니다.")

# 모델 로드
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)
print(f"모델이 '{filename}'에서 성공적으로 로드되었습니다.")

# 로드된 모델로 예측 수행
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = loaded_model.predict(new_data)
print(f"로드된 모델로 예측: {iris.target_names[prediction][0]}")
```


## 8. Scikit-learn과 다른 라이브러리 연동

Scikit-learn은 파이썬의 다른 주요 과학 계산 및 데이터 분석 라이브러리들과 긴밀하게 연동되어 머신러닝 워크플로우를 원활하게 구축할 수 있도록 합니다.

### 8.1. Pandas와의 연동
Pandas DataFrame은 Scikit-learn 모델의 입력 데이터로 가장 일반적으로 사용됩니다. Scikit-learn의 대부분의 함수와 클래스는 NumPy 배열을 입력으로 기대하지만, Pandas DataFrame을 직접 전달해도 내부적으로 NumPy 배열로 변환하여 처리합니다.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Iris 데이터셋을 Pandas DataFrame으로 로드
iris = load_iris(as_frame=True) # as_frame=True로 설정하여 DataFrame으로 로드
df = iris.frame

# 특성(X)과 타겟(y) 분리
X = df[iris.feature_names]
y = df['target']

print("Pandas DataFrame (X) head:\n", X.head())
print("Pandas Series (y) head:\n", y.head())

# Scikit-learn 모델에 Pandas DataFrame/Series 직접 사용
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

print("\n모델 학습 완료 (Pandas DataFrame 사용).")
```

### 8.2. NumPy와의 연동
Scikit-learn은 내부적으로 NumPy 배열을 기반으로 작동합니다. 따라서 데이터를 NumPy 배열 형태로 준비하면 Scikit-learn의 모든 기능과 효율적으로 연동될 수 있습니다.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# NumPy 배열로 가상 데이터 생성
X_np, y_np = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

print("NumPy 배열 (X) 형태:", X_np.shape)
print("NumPy 배열 (y) 형태:", y_np.shape)

# Scikit-learn 모델에 NumPy 배열 직접 사용
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("\n모델 학습 완료 (NumPy 배열 사용).")
```

### 8.3. Matplotlib/Seaborn과의 연동
Scikit-learn 모델의 결과를 시각화하거나, 데이터 전처리 전후의 분포를 확인하는 데 Matplotlib과 Seaborn이 활용됩니다. 특히 모델의 결정 경계, 클러스터링 결과, 특성 중요도 등을 시각적으로 표현할 때 유용합니다.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Iris 데이터셋 로드
iris = load_iris()
X, y = iris.data, iris.target

# PCA를 사용하여 2차원으로 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-Means 클러스터링 (시각화를 위해)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# PCA 결과 시각화 (Matplotlib)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset (True Labels)")
plt.colorbar(label='Species')
plt.show()

# 클러스터링 결과 시각화 (Seaborn)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=50, edgecolor='k')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering on PCA-reduced Iris Dataset")
plt.legend(title='Cluster')
plt.show()
```

## 9. 실제 ML/DL 적용 사례 (Scikit-learn 중심)

Scikit-learn은 다양한 실제 머신러닝 문제 해결에 활용될 수 있습니다. 다음은 Scikit-learn을 중심으로 한 몇 가지 대표적인 적용 사례입니다.

### 9.1. 분류 문제: 붓꽃(Iris) 데이터셋 분류
붓꽃 데이터셋은 머신러닝 분류 문제의 'Hello World'와 같은 예제입니다. 꽃잎과 꽃받침의 길이/너비 특성을 사용하여 붓꽃의 세 가지 종(Setosa, Versicolor, Virginica) 중 하나로 분류하는 문제입니다.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 파이프라인 구축 (스케일링 -> 모델 학습)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=200, random_state=42))
])

# 4. 모델 학습
pipeline.fit(X_train, y_train)

# 5. 예측
y_pred = pipeline.predict(X_test)

# 6. 모델 평가
print("\n--- 붓꽃 데이터셋 분류 결과 ---")
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n분류 보고서:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('예측된 클래스')
plt.ylabel('실제 클래스')
plt.title('혼동 행렬')
plt.show()
```

### 9.2. 회귀 문제: 보스턴 주택 가격 예측
보스턴 주택 가격 데이터셋은 회귀 문제의 고전적인 예제입니다. 주택과 관련된 다양한 특성(범죄율, 공기 오염도, 방 개수 등)을 사용하여 주택 가격을 예측하는 문제입니다.

```python
from sklearn.datasets import load_boston # load_boston은 scikit-learn 1.2부터 deprecated됨
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt

# 경고 무시 (load_boston deprecated 관련)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 1. 데이터 로드 (대안 데이터셋 사용 또는 직접 데이터 로드)
# 여기서는 load_boston 대신 make_regression을 사용하거나, 실제 데이터를 로드하는 방식으로 대체합니다.
# 실제 load_boston을 사용하려면 scikit-learn 버전 1.2 미만을 사용하거나, 다른 데이터셋을 찾아야 합니다.
# 예시를 위해 make_regression으로 가상 데이터 생성
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=500, n_features=13, noise=10, random_state=42)

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 파이프라인 구축 (스케일링 -> 모델 학습)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# 4. 모델 학습
pipeline.fit(X_train, y_train)

# 5. 예측
y_pred = pipeline.predict(X_test)

# 6. 모델 평가
print("\n--- 가상 주택 가격 예측 결과 ---")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 예측 결과 시각화 (일부 데이터만)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2) # y=x 라인
plt.xlabel('실제 가격')
plt.ylabel('예측 가격')
plt.title('실제 가격 vs. 예측 가격')
plt.grid(True)
plt.show()
```

### 9.3. 클러스터링 문제: 고객 세분화
고객 세분화는 고객 데이터를 기반으로 유사한 특성을 가진 고객 그룹을 식별하는 비지도 학습 문제입니다. 이를 통해 마케팅 전략을 맞춤화하거나 고객 경험을 개선할 수 있습니다.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 가상 고객 데이터 생성 (클러스터링 예시)
# n_features: 고객 특성 (예: 구매 빈도, 구매 금액, 방문 횟수 등)
# centers: 실제 고객 그룹 수 (알고 있다고 가정)
X, y_true = make_blobs(n_samples=300, n_features=2, centers=4, cluster_std=1.0, random_state=42)

# 2. 데이터 스케일링 (K-Means는 거리 기반이므로 스케일링이 중요)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-Means 모델 학습 (클러스터 개수 K=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) # n_init: 여러 번 초기화 시도
clusters = kmeans.fit_predict(X_scaled)

# 4. 클러스터링 결과 시각화
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='viridis', s=100, alpha=0.7, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering for Customer Segmentation')
plt.xlabel('Scaled Feature 1')
plt.ylabel('Scaled Feature 2')
plt.legend()
plt.grid(True)
plt.show()

print("\n클러스터링 완료. 고객이 4개의 세그먼트로 분류되었습니다.")
print("각 고객의 클러스터 레이블 (일부):", clusters[:20])
```