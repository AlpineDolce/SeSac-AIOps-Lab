<h2>Scikit-learn 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 지도 학습(Supervised Learning)의 분류(Classification) 문제 중 의사결정 트리(Decision Tree)와 서포트 벡터 머신(Support Vector Machine, SVM) 모델을 상세히 다룹니다. 각 모델의 개념과 Scikit-learn의 `DecisionTreeClassifier` 및 `SVC` 클래스를 이용한 모델 생성, 학습, 예측, 그리고 정확도 평가 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 지도 학습 (Supervised Learning)](#1-지도-학습-supervised-learning)
  - [1.1. 분류 (Classification)](#11-분류-classification)
    - [1.1.1. 의사결정 트리 (Decision Tree)](#111-의사결정-트리-decision-tree)
    - [1.1.2. 서포트 벡터 머신 (Support Vector Machine, SVM)](#112-서포트-벡터-머신-support-vector-machine-svm)

---

## 1. 지도 학습 (Supervised Learning)

지도 학습은 가장 일반적인 머신러닝 패러다임으로, 레이블(정답)이 있는 훈련 데이터를 사용하여 모델을 학습시킵니다. Scikit-learn은 다양한 지도 학습 알고리즘을 제공하며, 크게 분류(Classification)와 회귀(Regression) 문제로 나눌 수 있습니다.

### 1.1. 분류 (Classification)
분류는 입력 데이터를 미리 정의된 여러 클래스(범주) 중 하나로 할당하는 문제입니다. 예를 들어, 이메일이 스팸인지 아닌지, 환자가 특정 질병에 걸렸는지 아닌지 등을 예측하는 것이 분류 문제에 해당합니다.

#### 1.1.1. 의사결정 트리 (Decision Tree)
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

#### 1.1.2. 서포트 벡터 머신 (Support Vector Machine, SVM)
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
