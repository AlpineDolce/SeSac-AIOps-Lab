<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Pandas `DataFrame`의 구조를 파악하고 기본적인 통계 정보를 얻는 데 유용한 다양한 API(메서드)를 다룹니다. `head()`, `tail()`, `shape`, `info()`, `describe()`와 같은 함수들을 사용하여 데이터 탐색(EDA)의 첫 단계에서 데이터의 전체적인 모습과 특성을 빠르게 파악하는 방법을 학습합니다.

<h2>목차</h2>

- [1. DataFrame API 활용](#1-dataframe-api-활용)
  - [1.1. 기본 정보 확인 API](#11-기본-정보-확인-api)

---

## 1. DataFrame API 활용

Pandas DataFrame은 데이터의 구조를 파악하고 기본적인 통계 정보를 얻는 데 유용한 다양한 API(메서드)를 제공합니다. 이는 데이터 탐색(EDA)의 첫 단계에서 매우 중요합니다.

### 1.1. 기본 정보 확인 API

1.  **`head()` / `tail()`**: DataFrame의 상위 또는 하위 n개의 행을 출력하여 데이터의 전체적인 모습을 빠르게 파악할 수 있습니다. 기본값은 5개 행입니다.

    ```python
    import pandas as pd

    # auto-mpg.csv 파일 로드 (예시 데이터셋)
    data = pd.read_csv("./data/auto-mpg.csv")

    print("--- 앞에서부터 5개 미리 보기 (data.head())")
    print(data.head())

    print("\n--- 뒤에서부터 5개 미리 보기 (data.tail())")
    print(data.tail())

    print("\n--- 앞에서부터 10개 미리 보기 (data.head(10))")
    print(data.head(10))
    ```

2.  **`shape`**: DataFrame의 차원(dimensions)을 튜플 형태로 반환합니다. `(행의 개수, 열의 개수)`로 구성됩니다.

    ```python
    # data DataFrame이 로드되어 있다고 가정
    print("--- DataFrame의 차원 (shape)")
    print(data.shape)  # 예: (398, 9) -> 398행, 9열

    # 행과 열의 개수를 개별 변수에 할당
    row, col = data.shape
    print(f"행의 개수: {row}")
    print(f"열의 개수: {col}")
    ```

3.  **`info()`**: DataFrame의 간략한 정보를 출력합니다. 각 컬럼의 데이터 타입, Non-null 값의 개수, 메모리 사용량, 인덱스 정보 등을 포함하여 데이터의 누락 여부와 타입을 빠르게 확인할 수 있습니다.

    ```python
    import pandas as pd

    data = pd.read_csv("./data/auto-mpg.csv")
    print("--- 데이터의 기본 구조 (data.info())")
    data.info()
    ```

    **`info()` 함수 제공 정보 요약**:
    *   **데이터 타입 (Dtype)**: 각 컬럼이 어떤 데이터 타입(int64, float64, object 등)을 가지는지 보여줍니다.
    *   **Non-Null Count**: 각 컬럼에 결측치(NaN)가 아닌 유효한 데이터가 몇 개 있는지 보여줍니다. 이를 통해 결측치 여부를 쉽게 파악할 수 있습니다.
    *   **메모리 사용량 (Memory Usage)**: DataFrame이 사용하는 총 메모리 양을 나타냅니다.
    *   **인덱스 정보 (RangeIndex)**: DataFrame의 행 인덱스 범위와 스텝을 보여줍니다.

4.  **`describe()`**: 숫자형 컬럼에 대한 기술 통계(Descriptive Statistics)를 계산하여 출력합니다. 데이터의 분포와 중심 경향성, 퍼짐 정도 등을 파악하는 데 유용합니다.

    ```python
    # data DataFrame이 로드되어 있다고 가정
    print("--- 데이터의 요약 통계 정보 (data.describe())")
    print(data.describe())
    ```

    **`describe()` 함수 제공 정보 요약**:
    *   `count`: 해당 컬럼의 유효한(Non-null) 데이터 개수.
    *   `mean`: 평균값.
    *   `std`: 표준편차 (Standard Deviation).
    *   `min`: 최솟값.
    *   `25%` (1사분위수): 데이터를 오름차순으로 정렬했을 때 하위 25% 지점의 값.
    *   `50%` (중앙값/2사분위수): 데이터를 오름차순으로 정렬했을 때 중간 지점의 값. 중앙값은 극단적인 값(이상치)에 덜 민감하여 데이터의 중심을 나타내는 데 평균보다 더 견고할 수 있습니다.
    *   `75%` (3사분위수): 데이터를 오름차순으로 정렬했을 때 상위 25% 지점의 값.
    *   `max`: 최댓값.
