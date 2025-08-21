# 분류: 선형 모델과 서포트 벡터 머신
작성자: Alpine_Dolce | 날짜: 2025-07-01

## 문서 목표
이 문서는 분류 문제 해결을 위한 대표적인 두 가지 알고리즘, 로지스틱 회귀(Logistic Regression)와 서포트 벡터 머신(Support Vector Machine, SVM)의 핵심 개념과 Scikit-learn을 이용한 구현 방법을 학습합니다. 각 모델의 특징과 주요 하이퍼파라미터를 이해하고, 어떤 상황에서 각 모델을 사용하는 것이 효과적인지에 대한 가이드를 제공합니다. 또한, 대규모 데이터셋에 적합한 `SGDClassifier`를 소개합니다.

## 목차

* [1. 로지스틱 회귀 (Logistic Regression)](#1-로지스틱-회귀-logistic-regression)
  * [1.1. 핵심 개념](#11-핵심-개념)
  * [1.2. 주요 하이퍼파라미터](#12-주요-하이퍼파라미터)
  * [1.3. 실습: 유방암 데이터 분류](#13-실습-유방암-데이터-분류)
* [2. 서포트 벡터 머신 (Support Vector Machine, SVM)](#2-서포트-벡터-머신-support-vector-machine-svm)
  * [2.1. 핵심 개념](#21-핵심-개념)
  * [2.2. 주요 하이퍼파라미터](#22-주요-하이퍼파라미터)
  * [2.3. 실습: 붓꽃 데이터 분류](#23-실습-붓꽃-데이터-분류)
* [3. 확률적 경사 하강법 분류기 (SGDClassifier)](#3-확률적-경사-하강법-분류기-sgdclassifier)
  * [3.1. 핵심 개념](#31-핵심-개념)
  * [3.2. 주요 하이퍼파라미터](#32-주요-하이퍼파라미터)
  * [3.3. 실습: 대규모 데이터셋 분류](#33-실습-대규모-데이터셋-분류)
* [4. 선형 모델과 SVM의 선택 가이드](#4-선형-모델과-svm의-선택-가이드)

---

## 1. 로지스틱 회귀 (Logistic Regression)

### 1.1. 핵심 개념
로지스틱 회귀는 이름에 '회귀'가 포함되어 있지만, 실제로는 **분류(Classification)** 알고리즘입니다. 특히 이진 분류(Binary Classification) 문제에 널리 사용됩니다. 선형 회귀와 유사하게 입력 특성($x$)에 대한 선형 방정식($w^T x + b$)을 기반으로 하지만, 그 결과를 **시그모이드(Sigmoid) 함수**에 통과시켜 0과 1 사이의 확률 값으로 변환합니다. 이 확률 값을 기준으로 특정 임계값(보통 0.5)을 넘어 서면 클래스 1로, 그렇지 않으면 클래스 0으로 분류합니다.

*   **시그모이드 함수 (Sigmoid Function):**
    *   $ \sigma(z) = \frac{1}{1 + e^{-z}} $
    *   선형 모델의 출력($z = w^T x + b$)을 0과 1 사이의 값으로 압축하여 확률로 해석할 수 있도록 합니다. $z$가 커질수록 1에 가까워지고, $z$가 작아질수록 0에 가까워집니다.

*   **분류 과정:**
    1.  입력 특성($x$)과 학습된 가중치($w$) 및 편향($b$)을 사용하여 선형 결합($z = w^T x + b$)을 계산합니다.
    2.  계산된 $z$ 값을 시그모이드 함수에 통과시켜 0과 1 사이의 확률($p$)을 얻습니다. 이 확률은 해당 샘플이 양성 클래스(클래스 1)에 속할 확률을 나타냅니다.
    3.  미리 정의된 임계값(threshold, 기본값 0.5)과 비교하여 최종 클래스를 결정합니다. $p \ge 0.5$이면 클래스 1, $p < 0.5$이면 클래스 0으로 분류합니다.

*   **장점:**
    *   **학습 속도:** 비교적 빠르고 구현이 간단하여 대규모 데이터셋에도 적용하기 용이합니다.
    *   **해석 용이성:** 각 특성에 할당된 계수(coefficient)의 크기와 부호를 통해 해당 특성이 타겟 클래스에 미치는 영향의 방향과 중요도를 파악할 수 있어 모델 해석이 용이합니다.
    *   **확률 예측:** 단순히 클래스 레이블뿐만 아니라, 각 클래스에 속할 확률을 제공합니다.
    *   **널리 사용:** 다양한 분야에서 널리 사용되며, 성능이 준수하여 좋은 베이스라인 모델이 됩니다.
*   **단점:**
    *   **선형 결정 경계:** 선형적인 결정 경계를 가지므로, 데이터가 비선형적으로 복잡하게 분포되어 있을 경우 성능이 떨어질 수 있습니다.
    *   **이상치 민감성:** 이상치에 민감하게 반응하여 모델의 안정성을 해칠 수 있습니다.

### 1.2. 주요 하이퍼파라미터
로지스틱 회귀 모델의 성능은 하이퍼파라미터 설정에 따라 크게 달라질 수 있습니다. 주요 하이퍼파라미터는 다음과 같습니다.

*   **`penalty` (규제):**
    *   **설명:** 모델의 복잡도를 제어하고 과적합(Overfitting)을 방지하기 위해 사용되는 규제(Regularization)의 종류를 지정합니다. 규제는 모델의 가중치(계수)가 너무 커지는 것을 제한하여 모델의 일반화 성능을 향상시킵니다.
    *   `'l2'` (릿지 규제): 기본값이며, 모든 계수를 0에 가깝게 만듭니다. 계수들의 제곱합에 비례하는 페널티를 부여합니다. 모든 특성을 사용하지만, 중요하지 않은 특성의 계수를 작게 만듭니다.
    *   `'l1'` (라쏘 규제): 중요하지 않은 특성의 계수를 0으로 만들어 특성 선택(Feature Selection) 효과를 가집니다. 계수들의 절대값 합에 비례하는 페널티를 부여합니다.
    *   `'elasticnet'`: L1과 L2 규제를 모두 사용합니다. `l1_ratio` 파라미터를 통해 L1과 L2 규제의 혼합 비율을 조절할 수 있습니다.
    *   `'none'`: 규제를 적용하지 않습니다. 과적합 위험이 있습니다.

*   **`C` (규제 강도):**
    *   **설명:** 규제의 강도를 조절하는 파라미터입니다. `C`는 규제 강도 파라미터인 `alpha`의 역수(`1 / alpha`)와 같습니다.
    *   **`C` 값이 작을수록:** 규제가 **강해집니다**. 모델의 복잡도가 줄어들어 과소적합(Underfitting)이 발생할 가능성이 증가합니다. 모델이 훈련 데이터에 덜 민감하게 반응합니다.
    *   **`C` 값이 클수록:** 규제가 **약해집니다**. 모델의 복잡도가 증가하여 과적합이 발생할 가능성이 증가합니다. 모델이 훈련 데이터에 더 정확하게 맞추려고 합니다.
    *   **선택:** `C` 값은 일반적으로 0.001, 0.01, 0.1, 1, 10, 100, 1000과 같은 로그 스케일로 탐색하는 것이 일반적입니다.

*   **`solver` (최적화 알고리즘):**
    *   **설명:** 로지스틱 회귀 모델의 손실 함수를 최소화하는 데 사용할 최적화 알고리즘을 지정합니다. 데이터셋의 크기, `penalty` 종류, 그리고 다중 클래스 분류 방식에 따라 적절한 solver를 선택해야 합니다.
    *   `'liblinear'`: 작은 데이터셋에 적합하며, L1, L2 규제를 모두 지원합니다. 이진 분류에 주로 사용됩니다.
    *   `'lbfgs'`: 기본값이며, L2 규제만 지원합니다. 작은 데이터셋과 중간 크기 데이터셋에 적합합니다. 다중 클래스 분류에 'multinomial' 옵션을 지원합니다.
    *   `'newton-cg'`: L2 규제만 지원하며, 'lbfgs'와 유사하게 작동하지만 더 큰 데이터셋에 적합할 수 있습니다.
    *   `'sag'` (Stochastic Average Gradient): 대용량 데이터셋에 적합하며, L2 규제만 지원합니다. 확률적 경사 하강법의 변형으로, 빠른 수렴 속도를 보입니다.
    *   `'saga'`: `sag`의 변형으로, L1, L2, Elastic-Net 규제를 모두 지원하며, 대용량 데이터셋에 가장 적합합니다.

*   **`max_iter` (최대 반복 횟수):**
    *   **설명:** 최적화 알고리즘이 수렴하기 위한 최대 반복 횟수를 지정합니다. `max_iter`가 너무 작으면 모델이 충분히 학습되지 않아 수렴하지 못할 수 있습니다.
    *   **고려사항:** `ConvergenceWarning`이 발생하면 `max_iter` 값을 늘려야 합니다.

*   **`random_state`:**
    *   **설명:** 모델의 초기화나 데이터 분할 등 무작위성이 포함된 작업의 재현성을 보장하기 위한 난수 시드입니다. 특정 정수 값을 지정하면 동일한 결과를 항상 얻을 수 있습니다.

**하이퍼파라미터 튜닝:**
최적의 하이퍼파라미터 조합을 찾기 위해서는 `GridSearchCV`나 `RandomizedSearchCV`와 같은 교차 검증 기반의 하이퍼파라미터 튜닝 도구를 사용하는 것이 일반적입니다.

### 1.3. 실습: 유방암 데이터 분류
로지스틱 회귀 모델을 사용하여 유방암 데이터셋을 분류하는 실습 예제입니다. 이 데이터셋은 이진 분류 문제에 적합하며, 특성 스케일링이 모델 성능에 영향을 미칠 수 있으므로 `StandardScaler`와 `Pipeline`을 함께 사용합니다.

```python
from sklearn.datasets import load_breast_cancer # 유방암 데이터셋 로드
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.preprocessing import StandardScaler # 특성 스케일링
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델
from sklearn.pipeline import Pipeline # 파이프라인 구축
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # 모델 평가 지표
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 데이터 로드
# load_breast_cancer() 함수는 유방암 데이터셋을 로드합니다.
# X는 특성 데이터, y는 타겟 레이블(양성/음성)입니다.
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"데이터셋 형태 (X): {X.shape}")
print(f"타겟 레이블 분포 (y): {np.bincount(y)}") # 클래스 0(악성), 클래스 1(양성)의 개수

# 2. 데이터 분할
# 훈련 세트와 테스트 세트로 데이터를 분할합니다.
# test_size=0.3: 전체 데이터의 30%를 테스트 세트로 사용합니다.
# random_state=42: 재현성을 위한 난수 시드입니다.
# stratify=y: 타겟 변수(y)의 클래스 비율을 훈련 세트와 테스트 세트에서 동일하게 유지합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n훈련 세트 형태: {X_train.shape}, {y_train.shape}")
print(f"테스트 세트 형태: {X_test.shape}, {y_test.shape}")

# 3. 파이프라인 설정 (스케일링 -> 로지스틱 회귀)
# Pipeline을 사용하여 전처리(StandardScaler)와 모델(LogisticRegression)을 연결합니다.
# StandardScaler: 특성들의 스케일을 표준화하여 로지스틱 회귀 모델의 학습을 안정화합니다.
# LogisticRegression: C=1 (규제 강도), penalty='l2' (L2 규제), random_state=42 (재현성)
# solver='liblinear': 작은 데이터셋에 적합한 solver를 명시합니다.
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(C=1, penalty='l2', random_state=42, solver='liblinear'))
])

# 4. 모델 학습 및 평가
# pipe.fit(X_train, y_train)을 호출하면 파이프라인 내의 StandardScaler가 X_train에 fit_transform되고,
# 변환된 데이터로 LogisticRegression 모델이 학습됩니다.
pipe.fit(X_train, y_train)
# 테스트 세트(X_test)에 대한 예측을 수행합니다. StandardScaler가 X_test에 transform된 후 모델이 예측합니다.
y_pred = pipe.predict(X_test)

# 정확도(Accuracy) 계산
accuracy = accuracy_score(y_test, y_pred)

print(f"\n로지스틱 회귀 모델 정확도: {accuracy:.4f}")

# 분류 보고서 및 혼동 행렬을 통한 상세 평가
print("\n분류 보고서:", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.title('로지스틱 회귀 혼동 행렬')
plt.show()
```

**실습 결과 해석:**
*   **정확도:** 모델이 테스트 데이터에서 얼마나 정확하게 분류했는지 보여줍니다.
*   **분류 보고서:** 각 클래스(양성/음성)에 대한 정밀도, 재현율, F1-점수를 제공합니다. 특히 유방암 진단과 같은 문제에서는 '양성'(암)을 놓치지 않는 재현율(Recall)이 매우 중요할 수 있습니다.
*   **혼동 행렬:** 모델이 어떤 유형의 오류(False Positive, False Negative)를 범하는지 시각적으로 보여줍니다. 이를 통해 모델의 강점과 약점을 파악할 수 있습니다.

이 실습은 로지스틱 회귀가 간단하면서도 효과적인 분류 모델임을 보여줍니다. 특히 데이터가 선형적으로 분리 가능하거나, 모델의 해석 가능성이 중요할 때 좋은 선택이 될 수 있습니다.


## 2. 서포트 벡터 머신 (Support Vector Machine, SVM)

### 2.1. 핵심 개념
**서포트 벡터 머신(Support Vector Machine, SVM)**은 두 클래스 간의 **마진(Margin)을 최대화**하는 최적의 결정 경계(Decision Boundary)를 찾는 것을 목표로 하는 강력한 분류 알고리즘입니다. 이 결정 경계는 두 클래스의 데이터 포인트들을 가장 잘 분리하는 '초평면(Hyperplane)'입니다.

*   **마진 최대화:**
    *   **마진:** 결정 경계와 가장 가까운 각 클래스의 데이터 포인트(이를 **서포트 벡터(Support Vector)**라고 함) 사이의 거리를 의미합니다.
    *   SVM은 이 마진을 최대화하는 초평면을 찾습니다. 마진이 클수록 모델의 일반화 성능이 좋다고 알려져 있습니다.

*   **서포트 벡터 (Support Vectors):**
    *   결정 경계를 정의하는 데 사용되는 훈련 데이터 포인트들입니다. 이들은 마진 경계선 위에 위치하거나 그 안에 있는 데이터 포인트들입니다.
    *   SVM 모델은 서포트 벡터에 의해서만 결정되므로, 나머지 데이터 포인트들을 제거해도 결정 경계에는 영향을 미치지 않습니다.

*   **커널 트릭 (Kernel Trick):**
    *   SVM의 가장 큰 특징 중 하나는 '커널 트릭'을 사용하여 선형적으로 분리할 수 없는 데이터를 고차원 공간으로 매핑하여 선형 분리가 가능하도록 만드는 것입니다.
    *   실제로 데이터를 고차원 공간으로 변환하지 않고, 고차원 공간에서의 내적(dot product)을 계산하는 커널 함수를 사용하여 계산 비용을 줄입니다. 이를 통해 비선형적인 데이터에 대해서도 높은 성능을 보입니다.

*   **장점:**
    *   **고차원 공간에서 효과적:** 특성의 수가 샘플 수보다 많을 때도 효과적으로 작동합니다.
    *   **비선형 분류:** 커널 트릭을 통해 복잡한 비선형 분류 문제를 해결할 수 있습니다.
    *   **과적합에 강함:** 마진을 최대화하는 원리 덕분에 과적합에 비교적 강한 경향을 보입니다.
    *   **메모리 효율적:** 결정 경계를 정의하는 데 서포트 벡터만 사용하므로, 모든 훈련 데이터를 저장할 필요가 없어 메모리 효율적입니다.
*   **단점:**
    *   **대규모 데이터셋에서 학습 속도:** 샘플 수가 매우 많은 대규모 데이터셋에서는 학습 속도가 느릴 수 있습니다.
    *   **모델 해석의 어려움:** 커널 트릭을 사용하면 모델의 결정 과정을 직관적으로 이해하기 어렵습니다.
    *   **하이퍼파라미터 민감성:** `C`, `gamma`, `kernel` 등 하이퍼파라미터 설정에 따라 성능이 크게 달라지므로, 최적의 조합을 찾는 것이 중요합니다.

### 2.2. 주요 하이퍼파라미터
SVM 모델의 성능은 하이퍼파라미터 설정에 따라 크게 달라질 수 있습니다. 특히 `C`, `kernel`, `gamma`는 SVM의 복잡도와 일반화 성능에 결정적인 영향을 미칩니다.

*   **`C` (규제 파라미터):**
    *   **설명:** 로지스틱 회귀의 `C`와 유사하게 규제의 강도를 조절합니다. `C`는 마진의 폭과 분류 오류 사이의 트레이드오프를 결정합니다. 즉, 모델이 훈련 데이터의 오류를 얼마나 허용할 것인지를 제어합니다.
    *   **`C` 값이 클수록:** 규제가 **약해집니다**. 모델은 훈련 데이터의 오류를 최소화하려고 노력하며, 마진이 좁아지고 훈련 데이터에 더 정확하게 맞추려고 합니다. 이는 과적합(Overfitting) 가능성을 증가시킬 수 있습니다.
    *   **`C` 값이 작을수록:** 규제가 **강해집니다**. 모델은 더 넓은 마진을 선호하며, 훈련 데이터의 일부 오류를 허용합니다. 이는 과소적합(Underfitting) 가능성을 증가시킬 수 있지만, 일반화 성능을 높일 수 있습니다.
    *   **선택:** `C` 값은 일반적으로 0.001, 0.01, 0.1, 1, 10, 100, 1000과 같은 로그 스케일로 탐색하는 것이 일반적입니다.

*   **`kernel` (커널 함수):**
    *   **설명:** 데이터를 고차원 공간으로 매핑하는 데 사용될 커널 함수를 지정합니다. 커널 함수는 SVM이 비선형 분류 문제를 해결할 수 있도록 하는 핵심 요소입니다.
    *   `'linear'`: 선형 커널. 데이터가 선형적으로 분리 가능할 때 사용합니다. 가장 간단하고 빠릅니다.
    *   `'poly'`: 다항식 커널. `degree` 파라미터를 통해 다항식의 차수를 조절합니다.
    *   `'rbf'` (Radial Basis Function, 방사형 기저 함수): 가우시안 커널이라고도 불립니다. 비선형 데이터에 널리 사용되는 기본값이며, `gamma` 파라미터와 함께 사용됩니다.
    *   `'sigmoid'`: 시그모이드 커널.
    *   **선택:** 데이터의 특성과 분포에 따라 적절한 커널을 선택해야 합니다. 일반적으로 `linear` 커널로 시작하여 성능이 좋지 않으면 `rbf` 커널을 시도하는 것이 좋습니다.

*   **`gamma` (커널 계수):**
    *   **설명:** `rbf`, `poly`, `sigmoid` 커널 등에서 사용되는 파라미터로, 하나의 훈련 데이터 샘플이 미치는 영향의 범위를 결정합니다. 즉, 결정 경계의 '곡률'을 제어합니다.
    *   **`gamma` 값이 클수록:** 영향의 범위가 좁아져 결정 경계가 훈련 데이터의 각 샘플에 더 민감하게 반응합니다. 이는 결정 경계가 더 복잡해지고 과적합(Overfitting) 가능성을 증가시킬 수 있습니다.
    *   **`gamma` 값이 작을수록:** 영향의 범위가 넓어져 결정 경계가 더 부드러워지고 일반화됩니다. 이는 과소적합(Underfitting) 가능성을 증가시킬 수 있습니다.
    *   **선택:** `gamma` 값도 `C`와 마찬가지로 로그 스케일로 탐색하는 것이 일반적입니다. `'scale'` (기본값) 또는 `'auto'`로 설정하면 데이터의 특성 수에 따라 자동으로 조정됩니다.

*   **`degree` (다항식 차수):**
    *   **설명:** `kernel='poly'`일 때 사용되는 파라미터로, 다항식 커널의 차수를 지정합니다. 차수가 높을수록 모델의 복잡도가 증가합니다.

*   **`random_state`:**
    *   **설명:** 모델의 초기화나 내부적인 무작위성이 포함된 작업의 재현성을 보장하기 위한 난수 시드입니다. 특정 정수 값을 지정하면 동일한 결과를 항상 얻을 수 있습니다.

**하이퍼파라미터 튜닝:**
SVM은 하이퍼파라미터에 민감하므로, `GridSearchCV`나 `RandomizedSearchCV`와 같은 교차 검증 기반의 하이퍼파라미터 튜닝 도구를 사용하여 최적의 조합을 찾는 것이 매우 중요합니다.

### 2.3. 실습: 붓꽃 데이터 분류
SVM 모델을 사용하여 붓꽃(Iris) 데이터셋을 분류하는 실습 예제입니다. 붓꽃 데이터셋은 다중 클래스 분류 문제에 적합하며, SVM은 특성 스케일에 민감하므로 `StandardScaler`와 `Pipeline`을 함께 사용합니다.

```python
from sklearn.datasets import load_iris # 붓꽃 데이터셋 로드
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.preprocessing import StandardScaler # 특성 스케일링
from sklearn.svm import SVC # 서포트 벡터 머신 분류기
from sklearn.pipeline import Pipeline # 파이프라인 구축
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # 모델 평가 지표
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 데이터 로드
# load_iris() 함수는 붓꽃 데이터셋을 로드합니다.
# X는 특성 데이터, y는 타겟 레이블(3가지 붓꽃 품종)입니다.
iris = load_iris()
X, y = iris.data, iris.target

print(f"데이터셋 형태 (X): {X.shape}")
print(f"타겟 레이블 분포 (y): {np.bincount(y)}") # 클래스 0, 1, 2의 개수

# 2. 데이터 분할
# 훈련 세트와 테스트 세트로 데이터를 분할합니다.
# test_size=0.3: 전체 데이터의 30%를 테스트 세트로 사용합니다.
# random_state=42: 재현성을 위한 난수 시드입니다.
# stratify=y: 타겟 변수(y)의 클래스 비율을 훈련 세트와 테스트 세트에서 동일하게 유지합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\n훈련 세트 형태: {X_train.shape}, {y_train.shape}")
print(f"테스트 세트 형태: {X_test.shape}, {y_test.shape}")

# 3. 파이프라인 설정 (스케일링 -> SVM)
# Pipeline을 사용하여 전처리(StandardScaler)와 모델(SVC)을 연결합니다.
# StandardScaler: 특성들의 스케일을 표준화하여 SVM 모델의 학습을 안정화합니다.
# SVC: kernel='rbf' (가우시안 커널), C=10 (규제 강도), gamma=0.1 (커널 계수), random_state=42 (재현성)
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma=0.1, random_state=42))
])

# 4. 모델 학습 및 평가
# pipe_svm.fit(X_train, y_train)을 호출하면 파이프라인 내의 StandardScaler가 X_train에 fit_transform되고,
# 변환된 데이터로 SVC 모델이 학습됩니다.
pipe_svm.fit(X_train, y_train)
# 테스트 세트(X_test)에 대한 예측을 수행합니다. StandardScaler가 X_test에 transform된 후 모델이 예측합니다.
y_pred_svm = pipe_svm.predict(X_test)

# 정확도(Accuracy) 계산
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"\nSVM 모델 정확도: {accuracy_svm:.4f}")

# 분류 보고서 및 혼동 행렬을 통한 상세 평가
print("\n분류 보고서:", classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.title('SVM 혼동 행렬')
plt.show()
```

**실습 결과 해석:**
*   **정확도:** 모델이 테스트 데이터에서 얼마나 정확하게 분류했는지 보여줍니다.
*   **분류 보고서:** 각 클래스(붓꽃 품종)에 대한 정밀도, 재현율, F1-점수를 제공합니다. 다중 클래스 분류에서는 각 클래스별 성능을 확인하는 것이 중요합니다.
*   **혼동 행렬:** 모델이 어떤 품종들을 서로 혼동하는지 시각적으로 보여줍니다. 예를 들어, 'versicolor'를 'virginica'로 잘못 예측하는 패턴 등을 파악할 수 있습니다.

이 실습은 SVM이 비선형적인 데이터에서도 강력한 분류 성능을 보일 수 있음을 보여줍니다. 특히 데이터가 고차원이거나 복잡한 결정 경계가 필요할 때 좋은 선택이 될 수 있습니다.

## 3. 확률적 경사 하강법 분류기 (SGDClassifier)

### 3.1. 핵심 개념
`SGDClassifier`는 확률적 경사 하강법(Stochastic Gradient Descent, SGD)을 사용하여 선형 분류 모델(예: 선형 SVM, 로지스틱 회귀)을 학습시키는 알고리즘입니다. SGD는 전체 훈련 데이터셋 대신 **각 훈련 샘플(또는 작은 미니배치) 하나씩**을 사용하여 모델 파라미터(가중치)를 업데이트하는 반복적인 최적화 알고리즘입니다.

*   **확률적 경사 하강법 (Stochastic Gradient Descent, SGD):**
    *   **설명:** 일반적인 경사 하강법(Batch Gradient Descent)이 전체 훈련 데이터셋의 모든 샘플을 사용하여 한 번의 파라미터 업데이트를 수행하는 반면, SGD는 각 반복마다 무작위로 선택된 하나의 샘플(또는 작은 미니배치)의 그레디언트(기울기)를 계산하여 파라미터를 업데이트합니다.
    *   **장점:**
        *   **대규모 데이터셋에 매우 효율적:** 전체 데이터셋을 메모리에 로드할 필요가 없으므로, 데이터셋의 크기에 선형적으로 비례하는 학습 시간을 가집니다. 이는 수십만, 수백만 개의 샘플을 가진 대규모 데이터셋에 특히 유리합니다.
        *   **온라인 학습(Online Learning) 가능:** 새로운 데이터가 계속 들어올 때마다 모델을 점진적으로 업데이트할 수 있습니다. 이는 실시간으로 데이터가 생성되는 환경에서 모델을 최신 상태로 유지하는 데 유용합니다.
        *   **지역 최적점 탈출:** 그레디언트 계산에 무작위성이 포함되므로, 손실 함수의 지역 최적점(local optima)에 갇히지 않고 전역 최적점(global optima)을 찾을 가능성이 높아집니다.
    *   **단점:**
        *   **불안정한 수렴:** 파라미터 업데이트 과정에 무작위성이 있어 손실 함수가 불안정하게 진동하며 수렴할 수 있습니다.
        *   **하이퍼파라미터 민감성:** 학습률(Learning Rate) 등 하이퍼파라미터 설정에 매우 민감하며, 최적의 값을 찾기 어렵습니다.

*   **`SGDClassifier`의 적용:**
    *   `SGDClassifier`는 `loss` 파라미터를 통해 다양한 선형 분류 모델을 구현할 수 있습니다. 예를 들어, `loss='log_loss'`로 설정하면 로지스틱 회귀와 동일하게 동작하고, `loss='hinge'`로 설정하면 선형 SVM과 동일하게 동작합니다.

### 3.2. 주요 하이퍼파라미터
`SGDClassifier`는 다양한 선형 모델을 구현할 수 있으므로, 하이퍼파라미터도 해당 모델의 특성과 SGD 최적화 과정에 맞춰 다양하게 제공됩니다.

*   **`loss` (손실 함수):**
    *   **설명:** 최적화할 손실 함수를 지정합니다. 이 파라미터에 따라 `SGDClassifier`가 어떤 선형 분류 모델처럼 동작할지 결정됩니다.
    *   `'hinge'`: 선형 SVM (기본값). 마진을 최대화하는 손실 함수입니다.
    *   `'log_loss'`: 로지스틱 회귀. 예측 확률과 실제 레이블 간의 차이를 최소화하는 손실 함수입니다.
    *   `'perceptron'`: 퍼셉트론. 오류가 발생한 경우에만 가중치를 업데이트하는 손실 함수입니다.
    *   그 외에도 `'modified_huber'`, `'squared_hinge'` 등 다양한 손실 함수를 지원합니다.

*   **`penalty` (규제):**
    *   **설명:** 모델의 복잡도를 제어하고 과적합을 방지하기 위한 규제의 종류를 지정합니다.
    *   `'l2'`: L2 규제 (릿지). 모든 계수를 0에 가깝게 만듭니다.
    *   `'l1'`: L1 규제 (라쏘). 중요하지 않은 특성의 계수를 0으로 만들어 특성 선택 효과를 가집니다.
    *   `'elasticnet'`: L1과 L2 규제를 모두 사용합니다. `l1_ratio` 파라미터를 통해 L1과 L2 규제의 혼합 비율을 조절할 수 있습니다.
    *   `'none'`: 규제를 적용하지 않습니다.

*   **`alpha` (규제 강도):**
    *   **설명:** 규제의 강도를 조절하는 상수입니다. 값이 클수록 규제가 강해져 모델의 복잡도가 줄어들고, 과소적합 가능성이 증가합니다.
    *   `alpha`는 `C` (로지스틱 회귀, SVM)와는 반대 개념으로, `alpha`가 작을수록 규제가 약해집니다.

*   **`learning_rate` (학습률 스케줄):**
    *   **설명:** 학습률(Learning Rate)은 각 반복에서 모델 파라미터를 얼마나 크게 업데이트할지를 결정합니다. `SGDClassifier`는 학습률 스케줄을 지정할 수 있습니다.
    *   `'constant'`: 고정된 학습률을 사용합니다. `eta0` 파라미터로 초기 학습률을 지정합니다.
    *   `'optimal'`: 초기 학습률 `eta0`을 기반으로 최적의 학습률을 자동으로 계산합니다.
    *   `'invscaling'`: `eta = eta0 / pow(t, power_t)` 공식에 따라 학습률이 점진적으로 감소합니다.
    *   `'adaptive'`: 훈련 오류가 감소하지 않을 때 학습률을 줄입니다.

*   **`eta0` (초기 학습률):**
    *   **설명:** `learning_rate`가 `'constant'`, `'invscaling'`, `'adaptive'`일 때 사용되는 초기 학습률입니다.

*   **`max_iter` (최대 반복 횟수):**
    *   **설명:** 훈련 데이터셋을 반복할 최대 횟수(에포크, epoch)를 지정합니다. `max_iter`가 너무 작으면 모델이 충분히 학습되지 않아 수렴하지 못할 수 있습니다.
    *   **고려사항:** `ConvergenceWarning`이 발생하면 `max_iter` 값을 늘려야 합니다.

*   **`tol` (수렴 허용 오차):**
    *   **설명:** 손실이 `tol`보다 작아지면 학습을 조기 종료합니다. `max_iter`와 함께 수렴 조건을 제어합니다.

*   **`random_state`:**
    *   **설명:** 모델의 초기화나 내부적인 무작위성이 포함된 작업의 재현성을 보장하기 위한 난수 시드입니다. 특정 정수 값을 지정하면 동일한 결과를 항상 얻을 수 있습니다.

**하이퍼파라미터 튜닝:**
`SGDClassifier`는 하이퍼파라미터에 매우 민감하므로, `GridSearchCV`나 `RandomizedSearchCV`와 같은 교차 검증 기반의 하이퍼파라미터 튜닝 도구를 사용하여 최적의 조합을 찾는 것이 매우 중요합니다. 특히 `loss`, `penalty`, `alpha`, `learning_rate` 및 `eta0`의 조합을 신중하게 탐색해야 합니다.

### 3.3. 실습: 대규모 데이터셋 분류
`SGDClassifier`는 대규모 데이터셋에 매우 효율적이므로, 여기서는 `sklearn.datasets.make_classification`을 사용하여 대규모 가상 데이터를 생성하여 실습합니다. 이 실습은 `SGDClassifier`가 많은 양의 데이터에서도 빠르게 학습하고 예측할 수 있음을 보여줍니다.

```python
from sklearn.linear_model import SGDClassifier # SGDClassifier 임포트
from sklearn.datasets import make_classification # 가상 분류 데이터셋 생성
from sklearn.model_selection import train_test_split # 데이터 분할
from sklearn.preprocessing import StandardScaler # 특성 스케일링
from sklearn.pipeline import Pipeline # 파이프라인 구축
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # 모델 평가 지표
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 대규모 가상 데이터 생성
# n_samples=100000: 10만 개의 샘플을 생성합니다.
# n_features=20: 20개의 특성을 가집니다.
# n_informative=10: 10개의 특성이 타겟 변수와 관련이 있습니다.
# n_redundant=5: 5개의 특성이 다른 특성들과 중복됩니다.
# random_state=42: 재현성을 위한 난수 시드입니다.
X_large, y_large = make_classification(n_samples=100000, n_features=20, n_informative=10, n_redundant=5, random_state=42)

print(f"대규모 데이터셋 형태 (X): {X_large.shape}")
print(f"타겟 레이블 분포 (y): {np.bincount(y_large)}")

# 2. 데이터 분할
# 훈련 세트와 테스트 세트로 데이터를 분할합니다.
# test_size=0.2: 전체 데이터의 20%를 테스트 세트로 사용합니다.
# stratify=y_large: 타겟 변수(y_large)의 클래스 비율을 훈련 세트와 테스트 세트에서 동일하게 유지합니다.
X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(X_large, y_large, test_size=0.2, random_state=42, stratify=y_large)

print(f"\n훈련 세트 형태: {X_train_large.shape}, {y_train_large.shape}")
print(f"테스트 세트 형태: {X_test_large.shape}, {y_test_large.shape}")

# 3. 파이프라인 설정 (스케일링 -> SGDClassifier)
# SGDClassifier는 특성 스케일에 민감하므로 StandardScaler를 먼저 적용합니다.
# loss='log_loss': 로지스틱 회귀와 유사하게 동작하도록 손실 함수를 설정합니다.
# penalty='l2': L2 규제를 적용합니다.
# alpha=0.0001: 규제의 강도를 조절합니다.
# max_iter=1000: 최대 반복 횟수를 설정합니다.
pipe_sgd = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd_classifier', SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000, random_state=42))
])

# 4. 모델 학습 및 평가
# pipe_sgd.fit(X_train_large, y_train_large)을 호출하면 파이프라인 내의 StandardScaler가 X_train_large에 fit_transform되고,
# 변환된 데이터로 SGDClassifier 모델이 학습됩니다.
pipe_sgd.fit(X_train_large, y_train_large)
# 테스트 세트(X_test_large)에 대한 예측을 수행합니다.
y_pred_sgd = pipe_sgd.predict(X_test_large)

# 정확도(Accuracy) 계산
accuracy_sgd = accuracy_score(y_test_large, y_pred_sgd)

print(f"\nSGDClassifier 모델 정확도: {accuracy_sgd:.4f}")

# 분류 보고서 및 혼동 행렬을 통한 상세 평가
print("\n분류 보고서:", classification_report(y_test_large, y_pred_sgd))

cm_sgd = confusion_matrix(y_test_large, y_pred_sgd)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_sgd, annot=True, fmt='d', cmap='Blues')
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.title('SGDClassifier 혼동 행렬')
plt.show()
```

**실습 결과 해석:**
*   **정확도:** 대규모 데이터셋에서도 `SGDClassifier`가 효율적으로 학습하여 높은 정확도를 달성할 수 있음을 보여줍니다.
*   **분류 보고서 및 혼동 행렬:** 모델의 정밀도, 재현율, F1-점수, 그리고 어떤 클래스에서 오분류가 발생하는지 상세히 파악할 수 있습니다.
*   **학습 시간:** `SGDClassifier`는 대규모 데이터셋에서도 비교적 빠른 학습 시간을 보입니다. 이는 배치 경사 하강법(Batch Gradient Descent) 기반의 모델이 전체 데이터셋을 한 번에 처리하는 것과 대조됩니다.

이 실습은 `SGDClassifier`가 대규모 데이터셋이나 온라인 학습 환경에서 선형 분류 문제를 해결하는 데 매우 유용한 도구임을 보여줍니다.


## 4. 선형 모델과 SVM의 선택 가이드

로지스틱 회귀, SVM, 그리고 `SGDClassifier`는 모두 분류 문제에 사용될 수 있는 강력한 알고리즘이지만, 각각의 특징과 장단점이 명확하므로 데이터와 문제의 특성에 따라 적절한 모델을 선택하는 것이 중요합니다. 다음 표는 각 모델의 주요 특징을 비교하고, 어떤 상황에서 각 모델을 사용하는 것이 효과적인지에 대한 가이드를 제공합니다.

| 특징 | 로지스틱 회귀 (Logistic Regression) | 서포트 벡터 머신 (SVM) | SGDClassifier |
| :--- | :--- | :--- | :--- |
| **결정 경계** | 선형. 입력 특성들의 선형 결합으로 결정 경계를 형성합니다. | 선형 또는 비선형 (커널 트릭). 커널 함수를 통해 고차원 공간에서 비선형 결정 경계를 만들 수 있습니다. | 선형. `loss` 파라미터에 따라 로지스틱 회귀나 선형 SVM처럼 동작합니다. |
| **해석 용이성** | **높음.** 각 특성의 계수(coefficient)를 통해 해당 특성이 타겟 클래스에 미치는 영향의 방향과 중요도를 쉽게 파악할 수 있습니다. | **낮음.** 특히 비선형 커널을 사용할 경우 결정 경계가 복잡해져 모델의 작동 방식을 직관적으로 이해하기 어렵습니다. | **높음.** 선형 모델의 경우 계수를 통해 특성 영향력 파악이 가능합니다. |
| **학습 속도** | 빠름. 비교적 빠르게 수렴합니다. | 데이터셋의 크기(샘플 수)가 클 경우 학습 속도가 느릴 수 있습니다. 특히 비선형 커널 사용 시 더욱 그렇습니다. | **매우 빠름.** 대규모 데이터셋에 대해 확률적 경사 하강법을 사용하므로 학습 시간이 데이터셋 크기에 선형적으로 비례하여 효율적입니다. |
| **데이터 스케일** | 필요. 최적화 알고리즘(solver)에 따라 스케일링이 필요하거나 권장됩니다. | **필수.** 거리 기반 알고리즘이므로 특성 스케일링이 반드시 필요합니다. | **필수.** 경사 하강법 기반이므로 특성 스케일링이 반드시 필요합니다. |
| **성능** | 데이터가 선형적으로 분리될 때 좋은 성능을 보입니다. 간단한 베이스라인 모델로 적합합니다. | 비선형 데이터, 고차원 데이터에서 강력한 성능을 발휘합니다. 마진 최대화 원리로 과적합에 강합니다. | 대규모 데이터셋에서 효율적이고 강력한 성능을 보입니다. 다양한 선형 모델을 유연하게 구현할 수 있습니다. |
| **사용 추천** | **빠른 베이스라인 모델**을 구축할 때, **모델의 해석**이 중요할 때, 데이터가 비교적 선형적으로 분리될 수 있을 때, 예측 확률이 필요할 때. | 데이터의 패턴을 잘 모를 때, **비선형적인 관계**가 예상될 때, **최고의 성능**을 목표로 할 때, 특성 수가 샘플 수보다 많을 때. | **대규모 데이터셋**을 다룰 때, **온라인 학습**이 필요할 때, 자원 제약이 있는 환경에서 빠른 학습이 필요할 때. |

**모델 선택 시 고려사항:**
*   **데이터의 크기:** 데이터셋이 매우 크다면 `SGDClassifier`와 같이 대규모 데이터에 최적화된 모델을 우선적으로 고려해야 합니다.
*   **데이터의 선형성/비선형성:** 데이터가 선형적으로 분리될 수 있다면 로지스틱 회귀나 선형 SVM이 좋은 선택입니다. 비선형적인 관계가 있다면 커널 SVM을 고려해야 합니다.
*   **모델 해석 가능성:** 모델의 예측 결과를 설명해야 하는 비즈니스 요구사항이 있다면 로지스틱 회귀나 `SGDClassifier` (선형 손실 함수 사용 시)가 더 유리합니다.
*   **성능 목표:** 최고의 예측 성능이 최우선 목표라면 SVM (특히 커널 SVM)이나 앙상블 모델(다음 문서에서 다룸)을 고려할 수 있습니다.
*   **훈련 시간 및 자원:** 모델의 훈련 시간과 필요한 컴퓨팅 자원도 중요한 고려사항입니다.

이러한 가이드를 바탕으로 데이터와 문제의 특성에 가장 적합한 분류 모델을 선택하고, 하이퍼파라미터 튜닝을 통해 모델의 성능을 최적화해야 합니다.