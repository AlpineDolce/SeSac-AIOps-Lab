<h2>Scikit-learn 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 지도 학습(Supervised Learning)의 분류(Classification) 문제 중 K-최근접 이웃(K-Nearest Neighbors, KNN)과 앙상블 모델(랜덤 포레스트, 그레디언트 부스팅)을 상세히 다룹니다. 각 모델의 개념과 Scikit-learn의 `KNeighborsClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier` 클래스를 이용한 모델 생성, 학습, 예측, 그리고 정확도 평가 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 지도 학습 (Supervised Learning)](#1-지도-학습-supervised-learning)
  - [1.1. 분류 (Classification)](#11-분류-classification)
    - [1.1.1. K-최근접 이웃 (K-Nearest Neighbors, KNN)](#111-k-최근접-이웃-k-nearest-neighbors-knn)
    - [1.1.2. 앙상블 모델 (Ensemble Models)](#112-앙상블-모델-ensemble-models)
      - [랜덤 포레스트 (Random Forest)](#랜덤-포레스트-random-forest)
      - [그레디언트 부스팅 (Gradient Boosting)](#그레디언트-부스팅-gradient-boosting)

---

## 1. 지도 학습 (Supervised Learning)

지도 학습은 가장 일반적인 머신러닝 패러다임으로, 레이블(정답)이 있는 훈련 데이터를 사용하여 모델을 학습시킵니다. Scikit-learn은 다양한 지도 학습 알고리즘을 제공하며, 크게 분류(Classification)와 회귀(Regression) 문제로 나눌 수 있습니다.

### 1.1. 분류 (Classification)
분류는 입력 데이터를 미리 정의된 여러 클래스(범주) 중 하나로 할당하는 문제입니다. 예를 들어, 이메일이 스팸인지 아닌지, 환자가 특정 질병에 걸렸는지 아닌지 등을 예측하는 것이 분류 문제에 해당합니다.

#### 1.1.1. K-최근접 이웃 (K-Nearest Neighbors, KNN)
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

#### 1.1.2. 앙상블 모델 (Ensemble Models)
앙상블 학습은 여러 개의 개별 모델(약한 학습기)을 조합하여 하나의 강력한 모델을 만드는 기법입니다. 개별 모델의 단점을 보완하고, 예측 성능을 향상시키며, 과적합을 줄이는 효과가 있습니다. Scikit-learn은 다양한 앙상블 모델을 제공합니다.

##### 랜덤 포레스트 (Random Forest)
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

##### 그레디언트 부스팅 (Gradient Boosting)
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
