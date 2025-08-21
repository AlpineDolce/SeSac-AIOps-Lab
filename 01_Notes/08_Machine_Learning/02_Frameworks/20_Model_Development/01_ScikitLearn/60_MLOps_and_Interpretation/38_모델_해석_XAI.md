# 모델 운영: 모델 해석 가능성 (XAI)
작성자: Alpine_Dolce | 날짜: 2025-07-01

## 문서 목표
이 문서는 머신러닝 모델의 예측 성능만큼이나 중요한 "왜 모델이 그렇게 예측했는가?"를 설명하는 **모델 해석 가능성(Model Interpretability)** 기법을 학습합니다. 블랙박스 모델의 의사결정 과정을 이해하고 설명하는 **설명 가능한 AI (XAI)**의 중요성을 이해하며, 전역적(Global) 및 지역적(Local) 해석 방법론과 SHAP, LIME, Permutation Importance, Partial Dependence Plots 등 주요 해석 도구들의 개념과 실무적 중요성을 학습합니다.

## 목차

* [1. 모델 해석 가능성(XAI)이란?](#1-모델-해석-가능성xai이란)
  * [1.1. 왜 모델 해석이 필요한가?](#11-왜-모델-해석이-필요한가)
  * [1.2. 전역적(Global) 해석 vs. 지역적(Local) 해석](#12-전역적global-해석-vs-지역적local-해석)
* [2. 전역적(Global) 모델 해석](#2-전역적global-모델-해석)
  * [2.1. 특성 중요도 (Feature Importance)](#21-특성-중요도-feature-importance)
  * [2.2. Permutation Importance](#22-permutation-importance)
  * [2.3. 부분 의존성 플롯 (Partial Dependence Plots, PDP)](#23-부분-의존성-플롯-partial-dependence-plots-pdp)
  * [2.4. 특성 상호작용 플롯 (Feature Interaction Plots)](#24-특성-상호작용-플롯-feature-interaction-plots)
* [3. 지역적(Local) 모델 해석](#3-지역적local-모델-해석)
  * [3.1. LIME (Local Interpretable Model-agnostic Explanations)](#31-lime-local-interpretable-model-agnostic-explanations)
  * [3.2. SHAP (SHapley Additive exPlanations)](#32-shap-shapley-additive-explanations)
    * [3.2.1. SHAP 값의 의미](#321-shap-값의-의미)
    * [3.2.2. SHAP 요약 플롯 (Summary Plot)](#322-shap-요약-플롯-summary-plot)
    * [3.2.3. SHAP 의존성 플롯 (Dependence Plot)](#323-shap-의존성-플롯-dependence-plot)
    * [3.2.4. SHAP Force Plot](#324-shap-force-plot)
* [4. 모델 해석 시각화 도구](#4-모델-해석-시각화-도구)
* [5. 해석 가능성 시각화의 고려사항](#5-해석-가능성-시각화의-고려사항)

---

## 1. 모델 해석 가능성(XAI)이란?

### 1.1. 왜 모델 해석이 필요한가?
머신러닝 모델, 특히 딥러닝과 같은 복잡한 모델은 뛰어난 예측 성능을 보이지만, 그 내부 동작 방식이 "블랙박스"처럼 불투명하다는 비판을 받아왔습니다. **모델 해석 가능성(Model Interpretability)** 또는 **설명 가능한 AI (Explainable AI, XAI)**는 이러한 블랙박스 모델의 의사결정 과정을 인간이 이해할 수 있는 형태로 설명하는 것을 목표로 합니다.

*   **신뢰 구축:** 모델이 왜 특정 예측을 내렸는지 이해하면 사용자와 이해관계자의 모델에 대한 신뢰를 높일 수 있습니다. (특히 의료, 금융 등 고위험 분야)
*   **디버깅 및 개선:** 모델이 잘못된 예측을 하거나 예상치 못한 동작을 보일 때, 문제의 원인을 파악하고 모델을 개선하는 데 도움을 줍니다.
*   **공정성 및 편향 감지:** 모델이 특정 그룹에 대해 편향된 예측을 하는지 확인하고, 편향의 원인이 되는 특성을 식별하여 모델의 공정성을 확보합니다.
*   **규제 준수:** GDPR(유럽 일반 개인정보 보호법)의 "설명할 권리"와 같이, 모델의 의사결정 과정에 대한 설명을 요구하는 규제가 증가하고 있습니다.
*   **과학적 발견:** 모델이 데이터에서 새로운 패턴이나 관계를 학습했을 때, 이를 해석하여 도메인 지식을 확장하거나 새로운 가설을 수립할 수 있습니다.

### 1.2. 전역적(Global) 해석 vs. 지역적(Local) 해석
모델 해석 기법은 크게 두 가지 관점으로 나눌 수 있습니다.
*   **전역적(Global) 해석:** 모델 전체의 동작 방식이나 모든 예측에 걸쳐 어떤 특성이 중요한 영향을 미치는지 이해하는 데 중점을 둡니다. "모델이 전반적으로 어떻게 작동하는가?"에 답합니다.
*   **지역적(Local) 해석:** 특정 개별 예측(예: 한 환자의 질병 진단 결과)이 왜 그렇게 나왔는지 이해하는 데 중점을 둡니다. "이 특정 예측은 왜 이렇게 나왔는가?"에 답합니다.

## 2. 전역적(Global) 모델 해석

### 2.1. 특성 중요도 (Feature Importance)
모델의 예측에 각 특성이 얼마나 기여하는지 수치화하고 시각화하는 가장 기본적인 방법입니다. 트리 기반 모델(랜덤 포레스트, 그레디언트 부스팅)은 자체적으로 특성 중요도를 제공합니다.

### 2.2. Permutation Importance
모델 불가지론적(model-agnostic) 특성 중요도 측정 방법입니다. 특정 특성의 값을 무작위로 섞었을 때(permutation) 모델의 예측 성능(예: 정확도, F1-점수)이 얼마나 감소하는지를 측정하여 해당 특성의 중요도를 평가합니다. 모델의 종류에 관계없이 적용할 수 있다는 장점이 있습니다.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드 및 모델 학습
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Permutation Importance 계산
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = result.importances_mean.argsort()

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=cancer.feature_names[sorted_idx])
ax.set_title("Permutation Importance")
fig.tight_layout()
plt.show()
```

### 2.3. 부분 의존성 플롯 (Partial Dependence Plots, PDP)
특정 특성(또는 특성 조합)의 값이 변할 때 모델의 예측(평균)이 어떻게 변하는지 보여줍니다. 다른 모든 특성들의 영향은 평균화됩니다. 모델이 특정 특성과 예측 간에 어떤 관계를 학습했는지 전역적으로 이해하는 데 유용합니다.

*   **주의사항:** 특성 간의 강한 상호작용이 있는 경우, PDP는 오해의 소지가 있는 정보를 제공할 수 있습니다.

```python
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import GradientBoostingClassifier

# 데이터 및 모델 (위와 동일)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# PDP 시각화
features = ['mean radius', 'mean texture', 'mean perimeter']
plot_partial_dependence(gb_model, X_train, features, feature_names=cancer.feature_names, target_names=cancer.target_names)
fig = plt.gcf()
fig.set_size_inches(10, 8)
fig.tight_layout()
plt.show()
```

### 2.4. 특성 상호작용 플롯 (Feature Interaction Plots)
두 개 이상의 특성이 모델의 예측에 미치는 상호작용 효과를 시각화합니다. 즉, 한 특성의 효과가 다른 특성의 값에 따라 어떻게 달라지는지를 보여줍니다. Scikit-learn의 `PartialDependenceDisplay`는 두 특성 간의 상호작용을 2D PDP로 시각화할 수 있습니다.

```python
from sklearn.inspection import PartialDependenceDisplay

# PDP 시각화 (두 특성 조합)
features_interaction = [('mean radius', 'mean texture')]
PartialDependenceDisplay.from_estimator(gb_model, X_train, features_interaction, feature_names=cancer.feature_names, target_names=cancer.target_names)
fig = plt.gcf()
fig.set_size_inches(10, 5)
fig.tight_layout()
plt.show()
```

## 3. 지역적(Local) 모델 해석

### 3.1. LIME (Local Interpretable Model-agnostic Explanations)
LIME은 특정 예측을 설명하기 위해, 해당 예측 주변의 데이터 포인트를 샘플링하고, 이 샘플링된 데이터에 대해 간단하고 해석 가능한 모델(예: 선형 모델, 의사결정 트리)을 학습시켜 원래 모델의 동작을 근사합니다. 이 간단한 모델의 특성 가중치를 통해 원래 모델이 해당 예측을 내린 이유를 설명합니다.

*   **장점:** 모델 불가지론적(어떤 모델에도 적용 가능), 지역적 해석 제공.
*   **단점:** 샘플링 방식에 따라 결과가 불안정할 수 있음, 고차원 데이터에 대한 해석이 어려울 수 있음.

(LIME 라이브러리 설치: `pip install lime`)
```python
# LIME 실습은 외부 라이브러리 설치 및 복잡성으로 인해 코드 예시를 생략합니다.
# 개념적 이해를 돕기 위한 설명입니다.
# 실제 사용 시에는 lime.lime_tabular.LimeTabularExplainer 등을 활용합니다.
```

### 3.2. SHAP (SHapley Additive exPlanations)
SHAP은 게임 이론의 Shapley 값을 기반으로 각 특성이 모델의 예측에 얼마나 기여했는지 공정하게 분배합니다. SHAP 값은 각 특성이 기준 예측(예: 데이터셋의 평균 예측)에서 실제 예측으로 변화하는 데 기여한 정도를 나타냅니다. 모델 불가지론적이며, 지역적 설명과 전역적 설명을 모두 제공할 수 있습니다.

*   **장점:** 이론적 기반이 탄탄함, 일관성 있는 특성 기여도 제공, 지역적/전역적 해석 모두 가능.
*   **단점:** 계산 비용이 높을 수 있음, 특성 간 종속성이 강할 때 해석에 주의 필요.

(SHAP 라이브러리 설치: `pip install shap`)

#### 3.2.1. SHAP 값의 의미
*   각 특성의 SHAP 값은 해당 특성이 예측에 미치는 영향의 크기와 방향(긍정적/부정적)을 나타냅니다. 양수 SHAP 값은 예측을 높이는 방향으로, 음수 SHAP 값은 예측을 낮추는 방향으로 기여했음을 의미합니다.

#### 3.2.2. SHAP 요약 플롯 (Summary Plot)
각 특성의 SHAP 값 분포를 보여주어 전역적인 특성 중요도와 각 특성이 예측에 미치는 영향의 방향을 한눈에 파악할 수 있습니다.

```python
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 데이터 로드 및 모델 학습
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP Explainer 생성 (트리 모델용)
explainer = shap.TreeExplainer(model)

# 테스트 세트의 SHAP 값 계산
shap_values = explainer.shap_values(X_test)

# SHAP 요약 플롯 (클래스 1에 대한 SHAP 값)
# shap.summary_plot(shap_values[1], X_test, feature_names=cancer.feature_names)
# 위 코드는 Jupyter 환경에서만 잘 작동할 수 있습니다.
# 일반 파이썬 스크립트에서는 plt.show()가 필요할 수 있습니다.
# 또는 shap.plots.beeswarm(shap_values[1], X_test, feature_names=cancer.feature_names)
shap.summary_plot(shap_values[1], X_test, feature_names=cancer.feature_names, show=False)
plt.title("SHAP Summary Plot (Class 1)")
plt.show()
```

#### 3.2.3. SHAP 의존성 플롯 (Dependence Plot)
특정 특성의 SHAP 값이 해당 특성 값에 따라 어떻게 변하는지 보여줍니다. 다른 특성과의 상호작용을 색상으로 인코딩하여 보여줄 수 있습니다.

```python
# SHAP 의존성 플롯
# 'mean radius' 특성에 대한 의존성 플롯, 'mean texture' 특성과의 상호작용을 색상으로 표시
shap.dependence_plot("mean radius", shap_values[1], X_test, feature_names=cancer.feature_names, interaction_index="mean texture", show=False)
plt.title("SHAP Dependence Plot: Mean Radius vs. Mean Texture Interaction")
plt.show()
```

#### 3.2.4. SHAP Force Plot
개별 예측에 대한 특성들의 기여도를 시각적으로 보여줍니다. 기준 예측(base value)에서 시작하여 각 특성이 예측을 어떻게 밀어 올리거나(양수 기여) 밀어 내리는지(음수 기여)를 보여줍니다.

```python
# SHAP Force Plot (첫 번째 테스트 샘플에 대한 예측 설명)
# shap.initjs() # Jupyter 환경에서 필요
# shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test[0,:], feature_names=cancer.feature_names)
# 위 코드는 Jupyter 환경에서만 잘 작동할 수 있습니다.
# 일반 파이썬 스크립트에서는 HTML로 저장하여 확인하는 것이 일반적입니다.
# 예를 들어, shap.save_html("force_plot_example.html", shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test[0,:], feature_names=cancer.feature_names))
```

## 4. 모델 해석 시각화 도구
Scikit-learn 자체는 기본적인 해석 도구(Permutation Importance, PDP)를 제공하지만, 더 풍부하고 다양한 해석 기능을 위해서는 외부 라이브러리를 활용하는 것이 일반적입니다.
*   **`ELI5`:** 모델의 특성 중요도, 개별 예측에 대한 특성 기여도 등을 시각화하는 데 사용됩니다. 특히 텍스트 데이터에 대한 해석에 강점이 있습니다.
*   **`Skater`:** 다양한 모델 불가지론적 해석 기법(특성 중요도, PDP, ICE, LIME 등)을 제공하며, 시각화 기능도 포함하고 있습니다.
*   **`InterpretML`:** Microsoft에서 개발한 라이브러리로, 해석 가능한 모델(Explainable Boosting Machines)과 블랙박스 모델 해석 기법(SHAP, LIME 등)을 통합하여 제공합니다. 인터랙티브한 대시보드 형태의 시각화가 강점입니다.
*   **`Yellowbrick`:** Scikit-learn API와 통합되어 모델 평가 및 진단을 위한 시각화 도구를 제공합니다. 특성 중요도, 잔차 플롯 등 다양한 시각화를 통해 모델의 성능과 동작을 이해하는 데 도움을 줍니다.

## 5. 해석 가능성 시각화의 고려사항

모델 해석 가능성 기법을 활용하여 시각화를 수행할 때는 단순히 결과를 생성하는 것을 넘어, 그 결과가 어떻게 해석되고 전달되어야 하는지에 대한 깊은 이해가 필요합니다. 잘못된 해석이나 부적절한 시각화는 오히려 혼란을 가중시키거나 잘못된 의사결정으로 이어질 수 있습니다.

*   **복잡성과 정확성 간의 균형 (Trade-off between Complexity and Accuracy):**
    *   **설명:** 일반적으로 모델의 복잡도가 높아질수록 예측 성능은 향상되지만, 모델의 내부 작동 방식을 이해하고 설명하기는 더 어려워집니다. 반대로, 해석 가능한 모델(예: 선형 회귀, 의사결정 트리)은 설명하기 쉽지만, 복잡한 데이터 패턴을 학습하는 데 한계가 있어 예측 성능이 낮을 수 있습니다.
    *   **고려사항:** 해석 가능성 기법(XAI)은 이러한 블랙박스 모델의 의사결정 과정을 설명하는 데 도움을 주지만, 완벽하게 모든 것을 설명할 수는 없습니다. 따라서 모델의 예측 성능과 해석 가능성 사이에서 적절한 균형점을 찾아야 합니다. 비즈니스 문제의 중요도와 규제 요구사항에 따라 이 균형점은 달라질 수 있습니다.

*   **오해의 소지 방지 (Avoiding Misinterpretation):**
    *   **설명:** 모델 해석 결과가 항상 인과관계를 의미하는 것은 아닙니다. 해석 기법은 특성과 예측 간의 상관관계나 기여도를 보여줄 뿐, 한 특성의 변화가 다른 특성이나 타겟에 직접적인 원인이 된다는 것을 의미하지는 않습니다.
    *   **고려사항:** 해석 결과를 제시할 때 상관관계와 인과관계를 명확히 구분하여 설명해야 합니다. 또한, 시각화가 특정 특성의 중요도를 과장하거나 축소하지 않도록 디자인 원칙을 준수해야 합니다. 예를 들어, 축의 범위, 색상 스케일, 데이터 포인트의 밀도 등을 신중하게 선택해야 합니다.

*   **대상 청중 고려 (Audience Consideration):**
    *   **설명:** 모델 해석 결과를 누구에게 설명할 것인지(데이터 과학자, 도메인 전문가, 비전문가, 경영진)에 따라 시각화의 복잡성, 사용되는 용어, 설명의 깊이를 조절해야 합니다.
    *   **고려사항:**
        *   **데이터 과학자/ML 엔지니어:** 모델의 내부 메커니즘, 알고리즘적 세부 사항, 통계적 유의미성 등을 포함한 상세한 기술적 설명을 선호합니다.
        *   **도메인 전문가:** 자신의 도메인 지식과 연결될 수 있는, 비즈니스 맥락에서의 특성 중요도나 예측의 이유를 이해하기 쉽게 설명해야 합니다.
        *   **비전문가/경영진:** 복잡한 기술적 세부 사항보다는 모델의 핵심적인 작동 원리, 비즈니스 의사결정에 미치는 영향, 그리고 신뢰성에 대한 직관적이고 단순한 시각화가 더 효과적입니다. 복잡한 그래프보다는 요약된 인사이트와 실행 가능한 권장 사항에 집중해야 합니다.

*   **모델 불가지론적(Model-agnostic) vs. 모델 특정적(Model-specific) 기법:**
    *   **설명:** 모델 불가지론적 기법(예: LIME, SHAP, Permutation Importance)은 특정 모델에 종속되지 않고 다양한 모델에 적용할 수 있습니다. 모델 특정적 기법(예: 선형 모델의 계수, 트리 모델의 특성 중요도)은 해당 모델의 구조를 활용하여 해석을 제공합니다.
    *   **고려사항:** 블랙박스 모델을 해석할 때는 모델 불가지론적 기법이 유용하지만, 모델 특정적 기법이 더 정확하거나 깊이 있는 통찰을 제공할 수도 있습니다. 두 가지 유형의 기법을 조합하여 사용하는 것이 가장 효과적일 수 있습니다.

*   **데이터의 품질과 대표성:**
    *   해석 결과는 사용된 데이터의 품질과 대표성에 크게 의존합니다. 훈련 데이터가 실제 운영 환경의 데이터를 제대로 반영하지 못한다면, 해석 결과 또한 오해의 소지가 있을 수 있습니다.

모델 해석 가능성 시각화는 단순한 그래프 생성을 넘어, 모델의 동작을 정확하고 효과적으로 전달하여 신뢰를 구축하고 데이터 기반의 의사결정을 돕는 중요한 과정입니다.