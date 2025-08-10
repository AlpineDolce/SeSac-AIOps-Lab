<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

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

Pandas는 머신러닝 프로젝트의 거의 모든 단계에서 핵심적인 역할을 수행합니다. 특히 데이터 준비 및 탐색 단계에서 그 중요성이 두드러집니다. 부트캠프에서 다루는 머신러닝 과정에서 Pandas가 어떻게 활용되는지 살펴보겠습니다.

### 1.1. 데이터 로딩 및 탐색

1.  **다양한 데이터 소스 로딩**: Pandas는 CSV, Excel, SQL 데이터베이스, JSON 등 다양한 형식의 원시 데이터를 `DataFrame`으로 쉽게 불러올 수 있습니다. 예를 들어, `pd.read_csv('data.csv')`와 같이 간단한 명령으로 데이터를 메모리에 로드할 수 있습니다.
2.  **초기 데이터 탐색**: 로드된 데이터의 구조, 통계적 특성, 결측치 여부 등을 빠르게 파악하는 데 사용됩니다.
    *   `df.head()`, `df.tail()`: 데이터의 상위/하위 일부를 확인하여 데이터의 형태를 빠르게 파악합니다.
    *   `df.info()`: 각 컬럼의 데이터 타입, Non-null 값의 개수, 메모리 사용량 등 전반적인 정보를 제공하여 결측치 여부를 확인하는 데 도움을 줍니다.
    *   `df.describe()`: 숫자형 컬럼에 대한 기술 통계(평균, 표준편차, 최소/최대값, 사분위수 등)를 제공하여 데이터의 분포를 이해하는 데 활용됩니다.
    *   `df.value_counts()`: 특정 범주형 컬럼의 고유 값과 그 빈도를 확인합니다.
    *   `df.corr()`: 컬럼 간의 상관관계를 계산하여 특성 간의 선형 관계를 파악합니다.

### 1.2. 데이터 전처리

머신러닝 모델은 깨끗하고 정돈된 데이터를 필요로 합니다. Pandas는 데이터 전처리를 위한 강력한 도구들을 제공합니다.

1.  **결측치 처리**: 누락된 데이터(`NaN`)를 처리하는 것은 전처리 과정에서 매우 중요합니다.
    *   `df.isnull().sum()`: 각 컬럼별 결측치의 개수를 확인합니다.
    *   `df.fillna(value)`: 결측치를 특정 값(평균, 중앙값, 최빈값 등)으로 채웁니다.
    *   `df.dropna()`: 결측치가 있는 행 또는 열을 제거합니다.
2.  **데이터 타입 변환**: 컬럼의 데이터 타입을 변경하여 메모리 효율성을 높이거나 특정 연산을 가능하게 합니다 (예: `df['column'].astype('int')`).
3.  **중복 데이터 처리**: `df.duplicated().sum()`으로 중복된 행을 확인하고, `df.drop_duplicates()`로 중복을 제거합니다.
4.  **이상치 탐지 및 처리**: Pandas의 통계 함수와 필터링 기능을 활용하여 이상치(Outlier)를 식별하고 제거하거나 변환합니다. 예를 들어, IQR(Interquartile Range) 방법을 사용하여 이상치를 정의하고 제거할 수 있습니다.

### 1.3. 특성 공학 (Feature Engineering)

기존 데이터를 기반으로 새로운 유의미한 특성을 생성하는 과정입니다. Pandas의 강력한 데이터 조작 기능이 여기서 빛을 발합니다.

1.  **새로운 특성 생성**: 기존 컬럼들을 조합하거나 변환하여 새로운 특성을 만듭니다. 예를 들어, `df['total_score'] = df['math'] + df['science']`와 같이 새로운 점수 합계 컬럼을 만들 수 있습니다.
2.  **범주형 데이터 인코딩**: 머신러닝 모델은 숫자형 데이터를 선호하므로, 범주형 특성을 숫자형으로 변환해야 합니다.
    *   **One-Hot Encoding**: `pd.get_dummies(df['categorical_column'])`를 사용하여 범주형 변수를 여러 개의 이진(0 또는 1) 컬럼으로 변환합니다. 이는 순서가 없는 범주형 데이터에 적합합니다.
    *   **Label Encoding**: `sklearn.preprocessing.LabelEncoder`를 사용하여 각 범주에 고유한 정수 값을 할당합니다. 이는 순서가 있는 범주형 데이터에 적합할 수 있습니다.
3.  **날짜/시간 특성 추출**: 날짜/시간 컬럼에서 연도, 월, 일, 요일, 시간 등 다양한 정보를 추출하여 새로운 특성으로 활용할 수 있습니다 (예: `df['date_column'].dt.year`).

### 1.4. 데이터 분할 및 모델 입력

1.  **데이터 분할**: Pandas DataFrame은 `sklearn.model_selection.train_test_split`과 같은 함수를 사용하여 학습(training), 검증(validation), 테스트(test) 세트로 데이터를 분할하는 데 사용됩니다. 분할된 DataFrame은 NumPy 배열로 변환되어 Scikit-learn과 같은 머신러닝 라이브러리의 모델 입력으로 사용됩니다.
    ```python
    from sklearn.model_selection import train_test_split
    # X: 특성 데이터 (DataFrame), y: 타겟 데이터 (Series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
2.  **모델 입력**: Pandas DataFrame이나 Series는 `.values` 속성을 통해 NumPy 배열로 쉽게 변환될 수 있으며, 이는 대부분의 머신러닝 모델이 요구하는 입력 형식입니다.

### 1.5. 시각화 연동

Pandas는 Matplotlib, Seaborn과 같은 파이썬 시각화 라이브러리와 긴밀하게 연동됩니다. DataFrame의 데이터를 직접 시각화 함수에 전달하여 데이터의 분포, 관계, 패턴 등을 그래프로 표현할 수 있습니다. 이는 탐색적 데이터 분석(EDA) 단계에서 데이터에 대한 깊은 통찰력을 얻는 데 필수적입니다.

*   **Matplotlib**: `df.plot()` 메서드를 통해 기본적인 플롯(선, 막대, 히스토그램 등)을 쉽고 빠르게 그릴 수 있습니다.
*   **Seaborn**: 통계적 시각화에 특화된 라이브러리로, Pandas DataFrame을 입력으로 받아 더욱 풍부하고 미려한 그래프를 생성합니다.

**시각화 예제**:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Iris 데이터셋 로드
df_iris = pd.read_csv("./data/iris.csv")

# 1. Pandas 내장 plot() 사용 (히스토그램)
df_iris['sepal.length'].plot(kind='hist', title='Sepal Length Histogram')
plt.xlabel("Sepal Length")
plt.show()

# 2. Pandas 내장 plot() 사용 (산점도)
df_iris.plot(kind='scatter', x='sepal.length', y='sepal.width', 
             title='Sepal Length vs Width')
plt.show()

# 3. Seaborn을 이용한 Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_iris, x='variety', y='petal.length')
plt.title('Petal Length by Variety')
plt.show()

# 4. Seaborn을 이용한 Pair Plot (변수 간 모든 관계 시각화)
sns.pairplot(df_iris, hue='variety')
plt.show()
```
