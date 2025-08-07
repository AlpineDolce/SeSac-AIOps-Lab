<h2>Scikit-learn 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Scikit-learn을 활용한 실제 머신러닝 적용 사례 중 분류 문제(Classification)를 다룹니다. 붓꽃(Iris) 데이터셋을 사용하여 데이터 로드, 분할, 파이프라인 구축, 모델 학습, 예측, 그리고 정확도, 분류 보고서, 혼동 행렬을 이용한 모델 평가 과정을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 실제 ML/DL 적용 사례 (Scikit-learn 중심)](#1-실제-mldl-적용-사례-scikit-learn-중심)
  - [1.1. 분류 문제: 붓꽃(Iris) 데이터셋 분류](#11-분류-문제-붓꽃iris-데이터셋-분류)

---

## 1. 실제 ML/DL 적용 사례 (Scikit-learn 중심)

Scikit-learn은 다양한 실제 머신러닝 문제 해결에 활용될 수 있습니다. 다음은 Scikit-learn을 중심으로 한 몇 가지 대표적인 적용 사례입니다.

### 1.1. 분류 문제: 붓꽃(Iris) 데이터셋 분류
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

```