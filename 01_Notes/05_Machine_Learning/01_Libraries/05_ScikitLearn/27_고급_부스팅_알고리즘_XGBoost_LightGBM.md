# 고급 부스팅 알고리즘: XGBoost, LightGBM, CatBoost
작성자: Alpine_Dolce | 날짜: 2025-07-01

## 문서 목표
이 문서는 Scikit-learn의 기본 `GradientBoostingClassifier`를 넘어, 캐글(Kaggle)과 같은 데이터 경진대회 및 실무에서 압도적인 성능으로 널리 사용되는 3대 고급 부스팅 알고리즘 - **XGBoost, LightGBM, CatBoost** - 를 학습합니다. 각 라이브러리의 핵심 개념, 장점, 주요 하이퍼파라미터를 이해하고, Scikit-learn 래퍼(Wrapper)를 이용한 기본 구현 방법을 익힙니다.

## 목차

* [1. 왜 고급 부스팅 알고리즘을 사용하는가?](#1-왜-고급-부스팅-알고리즘을-사용하는가)
* [2. XGBoost (eXtreme Gradient Boosting)](#2-xgboost-extreme-gradient-boosting)
  * [2.1. 핵심 개념 및 장점](#21-핵심-개념-및-장점)
  * [2.2. 주요 하이퍼파라미터](#22-주요-하이퍼파라미터)
  * [2.3. 실습](#23-실습)
* [3. LightGBM (Light Gradient Boosting Machine)](#3-lightgbm-light-gradient-boosting-machine)
  * [3.1. 핵심 개념 및 장점](#31-핵심-개념-및-장점)
  * [3.2. 주요 하이퍼파라미터](#32-주요-하이퍼파라미터)
  * [3.3. 실습](#33-실습)
* [4. CatBoost (Categorical Boosting)](#4-catboost-categorical-boosting)
  * [4.1. 핵심 개념 및 장점](#41-핵심-개념-및-장점)
  * [4.2. 주요 하이퍼파라미터](#42-주요-하이퍼파라미터)
  * [4.3. 실습](#43-실습)
* [5. 고급 부스팅 라이브러리 비교 요약](#5-고급-부스팅-라이브러리-비교-요약)

---

## 1. 왜 고급 부스팅 알고리즘을 사용하는가?
Scikit-learn의 `GradientBoostingClassifier`와 `GradientBoostingRegressor`는 그레디언트 부스팅의 원리를 이해하기에 훌륭한 구현체입니다. 하지만 실무에서는 대용량 데이터를 더 빠르고 효율적으로 처리하면서 더 높은 예측 성능을 내는 것이 중요합니다.

XGBoost, LightGBM, CatBoost는 기존 그레디언트 부스팅 알고리즘을 다음과 같이 개선하여 이러한 요구사항을 충족시킵니다.
*   **성능:** 병렬 처리, 캐시 최적화, 효율적인 트리 분할 알고리즘 등을 통해 학습 속도가 훨씬 빠릅니다.
*   **예측력:** 규제(Regularization) 기능이 내장되어 과적합을 효과적으로 제어하고, 더 정교한 모델 학습을 통해 예측 성능이 더 높습니다.
*   **편의 기능:** 결측치를 자체적으로 처리하거나, 범주형 변수를 별도의 인코딩 없이 처리하는 등 사용 편의성을 높이는 다양한 기능을 제공합니다.

이 라이브러리들은 Scikit-learn과 호환되는 **래퍼(Wrapper) 클래스**를 제공하므로, `Pipeline`이나 `GridSearchCV` 등 Scikit-learn의 생태계와 완벽하게 통합하여 사용할 수 있습니다.

## 2. XGBoost (eXtreme Gradient Boosting)

### 2.1. 핵심 개념 및 장점
XGBoost는 그레디언트 부스팅을 병렬 학습이 가능하도록 구현하여 속도와 성능을 크게 향상시킨 라이브러리입니다. 캐글과 같은 데이터 경진대회에서 압도적인 성과를 보이며 유명해졌습니다.

*   **장점:**
    *   **병렬 처리:** CPU 코어를 효율적으로 사용하여 학습 속도가 빠릅니다.
    *   **규제 기능:** L1, L2 규제를 포함하여 과적합 방지에 효과적입니다.
    *   **결측치 자체 처리:** 결측치를 별도로 처리하지 않아도 모델이 알아서 학습합니다.
    *   **교차 검증 내장:** 모델 학습 시 교차 검증을 함께 수행할 수 있습니다.

### 2.2. 주요 하이퍼파라미터
*   `n_estimators`: 부스팅 라운드 수 (생성할 트리의 개수).
*   `learning_rate`: 학습률.
*   `max_depth`: 각 트리의 최대 깊이.
*   `subsample`: 각 트리를 학습할 때 사용할 훈련 데이터의 샘플 비율.
*   `colsample_bytree`: 각 트리를 학습할 때 사용할 특성의 샘플 비율.

### 2.3. 실습
```python
# pip install xgboost
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 로드
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

# XGBoost 모델 생성 및 학습
# Scikit-learn 래퍼 사용
xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)

# 예측 및 평가
y_pred = xgb_clf.predict(X_test)
print(f"XGBoost 정확도: {accuracy_score(y_test, y_pred):.4f}")
```

## 3. LightGBM (Light Gradient Boosting Machine)

### 3.1. 핵심 개념 및 장점
LightGBM은 Microsoft에서 개발한 라이브러리로, XGBoost보다 더 빠른 학습 속도와 더 적은 메모리 사용량을 목표로 합니다. **리프 중심 트리 분할(Leaf-wise tree growth)** 방식을 사용하여 성능을 극대화합니다.

*   **리프 중심 분할:** 기존 대부분의 부스팅 모델이 사용하는 **레벨 중심(Level-wise)** 분할 방식과 달리, LightGBM은 손실 변화가 가장 큰 리프 노드를 지속적으로 분할합니다. 이를 통해 더 적은 트리를 생성하면서도 높은 정확도를 달성할 수 있습니다.
*   **장점:**
    *   **매우 빠른 학습 속도:** 대용량 데이터셋에서 XGBoost보다 월등히 빠른 속도를 보입니다.
    *   **적은 메모리 사용량:** 효율적인 데이터 구조를 사용합니다.
    *   **범주형 특성 지원:** 정수 인코딩된 범주형 특성을 명시적으로 처리하여 성능을 높일 수 있습니다.

### 3.2. 주요 하이퍼파라미터
*   `n_estimators`, `learning_rate`, `max_depth`, `subsample`
*   `num_leaves`: 개별 트리가 가질 수 있는 최대 리프 노드의 수. `max_depth`와 함께 모델 복잡도를 제어하는 핵심 파라미터입니다.
*   `colsample_bytree`: 특성 샘플링 비율.

### 3.3. 실습
```python
# pip install lightgbm
from lightgbm import LGBMClassifier

# 데이터는 위와 동일
# LightGBM 모델 생성 및 학습
lgbm_clf = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lgbm_clf.fit(X_train, y_train)

# 예측 및 평가
y_pred = lgbm_clf.predict(X_test)
print(f"LightGBM 정확도: {accuracy_score(y_test, y_pred):.4f}")
```

## 4. CatBoost (Categorical Boosting)

### 4.1. 핵심 개념 및 장점
CatBoost는 Yandex에서 개발한 라이브러리로, 이름에서 알 수 있듯이 **범주형 특성(Categorical Features)**을 처리하는 데 매우 뛰어난 성능을 보입니다.

*   **Ordered Boosting & Ordered TS:** 기존 부스팅 알고리즘의 타겟 누수(Target Leakage) 문제를 해결하기 위해 Ordered Boosting과 Ordered Target Statistics(TS)라는 독자적인 기법을 사용합니다.
*   **장점:**
    *   **뛰어난 범주형 특성 처리:** 원-핫 인코딩 등 별도의 전처리 없이 범주형 특성을 매우 효과적으로 처리합니다.
    *   **높은 안정성과 예측력:** 적은 하이퍼파라미터 튜닝으로도 안정적이고 높은 성능을 보입니다.
    *   **시각화 도구 내장:** 학습 과정을 시각화하는 유용한 도구를 제공합니다.

### 4.2. 주요 하이퍼파라미터
*   `iterations`: `n_estimators`와 동일.
*   `learning_rate`, `depth` (`max_depth`와 동일)
*   `cat_features`: 범주형 특성의 인덱스를 지정하는 리스트.
*   `verbose`: 학습 과정을 출력할지 여부.

### 4.3. 실습
```python
# pip install catboost
from catboost import CatBoostClassifier

# 데이터는 위와 동일
# CatBoost 모델 생성 및 학습
cat_clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=3, random_state=42, verbose=0)
cat_clf.fit(X_train, y_train)

# 예측 및 평가
y_pred = cat_clf.predict(X_test)
print(f"CatBoost 정확도: {accuracy_score(y_test, y_pred):.4f}")
```

## 5. 고급 부스팅 라이브러리 비교 요약

| 특징 | Scikit-learn GBDT | XGBoost | LightGBM | CatBoost |
| :--- | :--- | :--- | :--- | :--- |
| **학습 속도** | 느림 | 빠름 | **매우 빠름** | 빠름 |
| **메모리 사용량** | 높음 | 높음 | **낮음** | 중간 |
| **예측 성능** | 좋음 | 매우 좋음 | 매우 좋음 | **매우 좋음** |
| **범주형 특성 처리** | 원-핫 인코딩 등 수동 처리 필요 | 수동 처리 필요 | 일부 지원 | **자동 및 최적화 처리** |
| **결측치 처리** | 수동 처리 필요 | **자체 처리** | 자체 처리 | 자체 처리 |
| **사용 편의성** | 쉬움 | 중간 (파라미터 많음) | 중간 | **쉬움** (튜닝 덜 필요) |
| **추천 상황** | 기본 원리 학습 | 성능과 속도의 균형 | **대용량 데이터**, 속도가 중요할 때 | **범주형 특성이 많을 때**, 안정적인 고성능이 필요할 때 |
