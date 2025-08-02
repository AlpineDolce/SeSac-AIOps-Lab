<h2>머신러닝/딥러닝을 위한 Pandas 라이브러리: 핵심 개념 및 활용 심화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 부트캠프에서 학습한 Pandas 라이브러리의 핵심 이론과 개념을 총체적으로 정리한 자료입니다. Series와 DataFrame 같은 기본 데이터 구조부터 데이터 생성, 접근, 조작, 연산, 외부 파일 입출력, 그리고 머신러닝 워크플로우에서의 Pandas 활용법까지 상세히 다룹니다. 본 문서를 통해 Pandas에 대한 깊이 있는 이해를 돕고, 실제 데이터 분석 및 머신러닝(ML) 및 딥러닝(DL) 문제 해결에 효과적으로 적용하는 역량을 강화하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Pandas 소개](#1-pandas-소개)
  - [1.1. 데이터 분석과 Pandas의 역할](#11-데이터-분석과-pandas의-역할)
  - [1.2. Pandas 라이브러리란?](#12-pandas-라이브러리란)
  - [1.3. Pandas의 주요 구성 요소](#13-pandas의-주요-구성-요소)
- [2. Pandas 설치 및 환경 설정](#2-pandas-설치-및-환경-설정)
  - [2.1. Anaconda 사용 시](#21-anaconda-사용-시)
  - [2.2. pip를 이용한 설치](#22-pip를-이용한-설치)
  - [2.3. 설치 문제 해결 팁](#23-설치-문제-해결-팁)
  - [2.4. Pandas 불러오기](#24-pandas-불러오기)
- [3. Pandas 핵심 데이터 구조](#3-pandas-핵심-데이터-구조)
  - [3.1. Series (1차원 데이터)](#31-series-1차원-데이터)
    - [3.1.1. Series의 개념 및 특징](#311-series의-개념-및-특징)
    - [3.1.2. Series 생성 방법](#312-series-생성-방법)
    - [3.1.3. Series 데이터 접근 및 수정](#313-series-데이터-접근-및-수정)
  - [3.2. DataFrame (2차원 데이터)](#32-dataframe-2차원-데이터)
    - [3.2.1. DataFrame의 개념 및 특징](#321-dataframe의-개념-및-특징)
    - [3.2.2. DataFrame 생성 방법](#322-dataframe-생성-방법)
    - [3.2.3. DataFrame 데이터 접근 및 조작](#323-dataframe-데이터-접근-및-조작)
  - [3.3. Panel (3차원 데이터 - Deprecated)](#33-panel-3차원-데이터---deprecated)
- [4. Pandas 데이터 연산](#4-pandas-데이터-연산)
  - [4.1. 연산의 기본 원리](#41-연산의-기본-원리)
  - [4.2. Series 간 연산](#42-series-간-연산)
  - [4.3. DataFrame 간 연산 및 컬럼 추가/삭제](#43-dataframe-간-연산-및-컬럼-추가삭제)
- [5. 실전 예제: 데이터프레임 활용](#5-실전-예제-데이터프레임-활용)
  - [과제 1: 데이터프레임 만들기 및 연산](#과제-1-데이터프레임-만들기-및-연산)
- [6. Pandas와 머신러닝 워크플로우](#6-pandas와-머신러닝-워크플로우)
  - [6.1. 데이터 로딩 및 탐색](#61-데이터-로딩-및-탐색)
  - [6.2. 데이터 전처리](#62-데이터-전처리)
  - [6.3. 특성 공학 (Feature Engineering)](#63-특성-공학-feature-engineering)
  - [6.4. 데이터 분할 및 모델 입력](#64-데이터-분할-및-모델-입력)
  - [6.5. 시각화 연동](#65-시각화-연동)
- [7. DataFrame 심화](#7-dataframe-심화)
  - [7.1. DataFrame의 주요 특징](#71-dataframe의-주요-특징)
  - [7.2. Pandas가 지원하는 파일 형식](#72-pandas가-지원하는-파일-형식)
- [8. 외부 파일 읽기/쓰기](#8-외부-파일-읽기쓰기)
  - [8.1. 외부 파일 처리의 중요성](#81-외부-파일-처리의-중요성)
- [9. 파일 경로 처리](#9-파일-경로-처리)
  - [9.1. 경로 표현 방식](#91-경로-표현-방식)
  - [9.2. 파이썬에서의 경로 처리 주의사항](#92-파이썬에서의-경로-처리-주의사항)
- [10. CSV 파일 처리](#10-csv-파일-처리)
  - [10.1. CSV 파일의 특징](#101-csv-파일의-특징)
  - [10.2. CSV 파일 읽기](#102-csv-파일-읽기)
  - [10.3. CSV 파일 읽기 예제](#103-csv-파일-읽기-예제)
    - [1. 기본 CSV 파일 읽기](#1-기본-csv-파일-읽기)
    - [2. 제목 줄이 없는 CSV 파일 처리](#2-제목-줄이-없는-csv-파일-처리)
    - [3. 제목 줄이 특정 위치에 있는 경우](#3-제목-줄이-특정-위치에-있는-경우)
  - [10.4. CSV 파일 저장](#104-csv-파일-저장)
- [11. Excel 파일 처리](#11-excel-파일-처리)
  - [11.1. Excel 파일의 장점](#111-excel-파일의-장점)
  - [11.2. Excel 파일 읽기/쓰기 예제](#112-excel-파일-읽기쓰기-예제)
- [12. DataFrame API 활용](#12-dataframe-api-활용)
  - [12.1. 기본 정보 확인 API](#121-기본-정보-확인-api)
- [13. 조건부 데이터 검색](#13-조건부-데이터-검색)
  - [13.1. 기본 조건 검색](#131-기본-조건-검색)
  - [13.2. 복합 조건 검색](#132-복합-조건-검색)
- [14. 통계 함수 활용](#14-통계-함수-활용)
  - [14.1. DataFrame의 통계 함수들](#141-dataframe의-통계-함수들)
  - [14.2. 통계 함수 활용 예제](#142-통계-함수-활용-예제)
  - [14.3. 주요 통계 개념 설명](#143-주요-통계-개념-설명)
- [15. 실전 예제: Iris 데이터셋 분석](#15-실전-예제-iris-데이터셋-분석)
- [16. 고급 데이터 조작 및 분석](#16-고급-데이터-조작-및-분석)
  - [16.1. 데이터 그룹화 (Groupby)](#161-데이터-그룹화-groupby)
  - [16.2. 데이터 병합 (Merge \& Join)](#162-데이터-병합-merge--join)
  - [16.3. 피벗 테이블 (Pivot Table)](#163-피벗-테이블-pivot-table)
  - [16.4. 데이터 변형 (Melt)](#164-데이터-변형-melt)
  - [16.5. 시계열 데이터 처리 (Time Series)](#165-시계열-데이터-처리-time-series)
  - [16.6. 성능 최적화 팁](#166-성능-최적화-팁)
    - [1. 벡터화된 연산 활용](#1-벡터화된-연산-활용)
    - [2. `apply()` 대신 벡터화된 함수 사용](#2-apply-대신-벡터화된-함수-사용)
    - [3. 적절한 데이터 타입 (dtype) 사용](#3-적절한-데이터-타입-dtype-사용)
    - [4. `inplace=True` 사용 주의](#4-inplacetrue-사용-주의)
    - [5. `read_csv` 옵션 활용](#5-read_csv-옵션-활용)
    - [6. `Categorical` 타입 활용](#6-categorical-타입-활용)

---

## 1. Pandas 소개

### 1.1. 데이터 분석과 Pandas의 역할

1.  **데이터셋의 중요성**: 수집된 데이터들은 각자 동일하거나 다른 형태의 데이터들로 구성됩니다. 이러한 원시 데이터는 그대로 분석에 사용하기 어렵기 때문에, **데이터셋**은 다양한 형태의 데이터들을 하나의 구조로 만들어서 관리나 분석을 용이하게 하는 역할을 합니다.
2.  **파이썬에서의 데이터 관리**: 파이썬에서는 데이터셋을 효율적으로 관리하고 분석하기 위해 `pandas` 라이브러리를 제공합니다. `pandas`는 데이터 과학 분야에서 가장 널리 사용되는 라이브러리 중 하나로, 복잡한 데이터를 정돈된 형태로 만들어 분석을 용이하게 합니다.

### 1.2. Pandas 라이브러리란?

1.  **정의**: Pandas는 "**P**ython **Da**ta **N**alysis **L**ibrary"의 약자로, 데이터 조작 및 분석을 위한 강력하고 유연한 오픈 소스 라이브러리입니다. R의 데이터프레임과 유사한 기능을 파이썬에서 제공하며, 특히 **정형 데이터(테이블 형태의 데이터)**를 다루는 데 최적화되어 있습니다.
2.  **주요 특징**: Pandas는 다음과 같은 특징을 가집니다:
    *   **고성능 데이터 구조**: `Series` (1차원)와 `DataFrame` (2차원)과 같은 고성능 데이터 구조를 제공하여 대규모 데이터를 효율적으로 처리합니다.
    *   **다양한 데이터 소스 지원**: CSV, Excel, SQL 데이터베이스, JSON, HDF5 등 다양한 형식의 데이터를 쉽게 읽고 쓸 수 있는 함수를 제공합니다.
    *   **강력한 데이터 처리 기능**: 데이터 정렬, 필터링, 그룹화, 병합, 결측치 처리, 통계 분석 등 데이터 분석에 필요한 광범위한 기능을 내장하고 있습니다.
    *   **NumPy 기반**: 내부적으로 파이썬의 과학 계산 라이브러리인 NumPy를 활용하여 효율적인 배열 연산을 수행합니다. 이로 인해 머신러닝 및 딥러닝 라이브러리(예: Scikit-learn, TensorFlow, PyTorch)와 높은 호환성을 가집니다.
    *   **설치 용이성**: Anaconda 배포판에 기본 포함되어 있어 별도 설치 없이 바로 사용 가능하며, `pip`를 통해서도 쉽게 설치할 수 있습니다.

### 1.3. Pandas의 주요 구성 요소

1.  **클래스 및 내장 함수**: Pandas는 다양한 클래스와 내장 함수로 구성되어 있으며, 이를 통해 파이썬의 기본 데이터 타입(리스트, 딕셔너리 등)을 Pandas의 고유한 데이터 구조로 변환하여 데이터 분석에 적합하게 만듭니다.
2.  **핵심 데이터 타입**: Pandas가 제공하는 주요 데이터 타입은 다음과 같습니다:
    *   **Series (1차원)**: 인덱스와 값 쌍으로 구성된 1차원 배열과 같은 구조입니다. 단일 컬럼 데이터를 표현하는 데 적합합니다.
    *   **DataFrame (2차원)**: 행과 열로 이루어진 테이블 형태의 2차원 구조입니다. 관계형 데이터베이스의 테이블이나 Excel 스프레드시트와 유사하며, **가장 많이 사용되는 데이터 구조**입니다.
    *   **Panel (3차원 - Deprecated)**: 과거에는 3차원 데이터 구조인 Panel도 제공했으나, 현재는 사용이 권장되지 않으며, 대신 MultiIndex DataFrame이나 `xarray` 라이브러리 사용이 권장됩니다.

## 2. Pandas 설치 및 환경 설정

Pandas는 파이썬 데이터 과학 생태계의 핵심 라이브러리이므로, 대부분의 파이썬 배포판에 포함되어 있거나 쉽게 설치할 수 있습니다.

### 2.1. Anaconda 사용 시

1.  **기본 포함**: Anaconda는 데이터 과학을 위한 파이썬 배포판으로, Pandas를 포함한 대부분의 필수 라이브러리가 기본적으로 설치되어 있습니다. 따라서 Anaconda를 통해 파이썬을 설치했다면 별도의 설치 과정 없이 바로 Pandas를 사용할 수 있습니다.

### 2.2. pip를 이용한 설치

1.  **설치 명령**: Anaconda를 사용하지 않거나, 특정 환경에 Pandas를 설치해야 하는 경우 파이썬의 패키지 관리자인 `pip`를 사용하여 설치할 수 있습니다. **가상 환경(virtual environment)을 활성화한 후 설치하는 것을 권장**합니다.

    ```bash
    # 가상 환경 생성 (선택 사항이지만 권장)
    # python -m venv myenv
    # source myenv/bin/activate  # Linux/macOS
    # myenv\Scripts\activate     # Windows

    # Pandas 설치 명령
    pip install pandas
    ```

### 2.3. 설치 문제 해결 팁

1.  **`pip` 명령 작동 오류**: `pip install` 명령이 작동하지 않는 경우, 파이썬 및 `pip`가 시스템의 환경 변수(PATH)에 올바르게 등록되어 있는지 확인해야 합니다.
2.  **권한 부족 오류**: `pip install` 명령 실행 시 권한 관련 오류가 발생하면, **관리자 권한으로 명령 프롬프트(Windows) 또는 터미널(Linux/macOS)을 실행**하여 다시 시도합니다.
3.  **업그레이드**: 이미 Pandas가 설치되어 있지만 최신 버전으로 업데이트하고 싶다면 `pip install --upgrade pandas` 명령을 사용합니다.

### 2.4. Pandas 불러오기

1.  **`import` 문**: Pandas 라이브러리를 사용하기 위해서는 파이썬 스크립트나 Jupyter Notebook에서 `import` 문을 통해 불러와야 합니다. 관례적으로 `pd`라는 **별칭을 사용하여 코드를 간결하게 작성**합니다.

    ```python
    import pandas as pd

    # 이제 pd.Series, pd.DataFrame 등으로 Pandas 기능을 사용할 수 있습니다.
    ```

## 3. Pandas 핵심 데이터 구조

Pandas는 주로 `Series`와 `DataFrame`이라는 두 가지 강력한 데이터 구조를 제공하여 다양한 형태의 데이터를 효율적으로 처리할 수 있게 합니다.

### 3.1. Series (1차원 데이터)

#### 3.1.1. Series의 개념 및 특징

1.  **정의**: Series는 Pandas의 1차원 데이터 구조로, **인덱스(index)와 값(value)의 쌍**으로 구성됩니다. 파이썬의 리스트나 NumPy의 1차원 배열과 유사하지만, 각 값에 접근할 수 있는 레이블(label) 또는 인덱스를 가질 수 있다는 점에서 차이가 있습니다.
2.  **구조**: 딕셔너리(dict) 타입과 유사하게 키-값 쌍의 형태로 데이터를 저장하고 관리합니다.
    *   **인덱스**: 각 데이터 항목을 고유하게 식별하는 레이블입니다. 명시적으로 지정하지 않으면 0부터 시작하는 정수 인덱스가 자동으로 부여됩니다.
    *   **값**: Series에 저장되는 실제 데이터입니다. 모든 값은 동일한 데이터 타입을 가질 수 있지만, 다른 데이터 타입이 혼합될 경우 Pandas가 자동으로 적절한 상위 데이터 타입(예: `object` 타입)으로 변환합니다.
3.  **활용**: 인덱스를 통해 데이터를 빠르고 효율적으로 검색, 정렬, 선택, 결합할 수 있어 데이터의 특정 부분을 빠르게 참조하거나 조작하는 데 용이합니다.

#### 3.1.2. Series 생성 방법

다양한 파이썬 객체로부터 Series를 생성할 수 있습니다.

1.  **파이썬 리스트(list)로 Series 만들기**
    가장 기본적인 방법으로, 리스트의 요소들이 Series의 값이 되고, 0부터 시작하는 기본 정수 인덱스가 부여됩니다.

    ```python
    import pandas as pd

    # 파이썬 list를 사용하여 Series 생성
    data = [10, 20, 30, 40, 50]
    series = pd.Series(data)
    print("--- 리스트로 생성된 Series ---")
    print(type(series))
    print(series)
    # 출력:
    # --- 리스트로 생성된 Series ---
    # <class 'pandas.core.series.Series'>
    # 0    10
    # 1    20
    # 2    30
    # 3    40
    # 4    50
    # dtype: int64
    ```

2.  **파이썬 딕셔너리(dict)로 Series 만들기**
    딕셔너리의 키(key)가 Series의 인덱스가 되고, 값(value)이 Series의 값이 됩니다. 이를 통해 사용자 정의 레이블 인덱스를 쉽게 부여할 수 있습니다.

    ```python
    import pandas as pd

    # 딕셔너리를 사용하여 Series 생성 (레이블 인덱스 부여)
    data2 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    series2 = pd.Series(data2)
    print("\n--- 딕셔너리로 생성된 Series (레이블 인덱스) ---")
    print(series2)
    # 출력:
    # --- 딕셔너리로 생성된 Series (레이블 인덱스) ---
    # a    1
    # b    2
    # c    3
    # d    4
    # e    5
    # dtype: int64
    ```

3.  **NumPy 배열(ndarray)로 Series 만들기**
    NumPy 배열은 Pandas의 내부 연산에 효율적으로 사용되므로, NumPy 배열로부터 Series를 생성하는 것도 일반적입니다.

    ```python
    import numpy as np
    import pandas as pd

    # NumPy 배열로 Series 생성
    np_array = np.array([100, 200, 300])
    series_from_np = pd.Series(np_array)
    print("\n--- NumPy 배열로 생성된 Series ---")
    print(series_from_np)
    # 출력:
    # --- NumPy 배열로 생성된 Series ---
    # 0    100
    # 1    200
    # 2    300
    # dtype: int32 (또는 int64)
    ```

#### 3.1.3. Series 데이터 접근 및 수정

Series의 데이터는 정수 인덱스(위치 기반) 또는 레이블 인덱스(이름 기반)를 사용하여 접근하고 수정할 수 있습니다.

1.  **기본 접근 (정수 인덱스)**
    리스트와 유사하게 대괄호 `[]` 안에 정수 인덱스를 사용하여 특정 위치의 데이터에 접근합니다.

    ```python
    import pandas as pd

    data = [10, 20, 30, 40, 50, 60]
    s = pd.Series(data)

    print("\n--- Series 기본 접근 ---")
    print(f"0번째 데이터: {s[0]}") # 0번째 데이터 출력
    # 출력: 0번째 데이터: 10

    s[1] = 200 # 1번째 데이터 수정
    print(f"수정 후 1번째 데이터: {s[1]}")
    # 출력: 수정 후 1번째 데이터: 200
    ```

2.  **슬라이싱 (Slicing)**
    Series의 특정 범위의 데이터를 추출할 때 사용합니다. 정수 인덱스 또는 레이블 인덱스 모두에 적용 가능합니다. **주의**: 정수 인덱스 슬라이싱은 끝 인덱스를 포함하지 않지만, 레이블 인덱스 슬라이싱은 끝 인덱스를 포함합니다.

    ```python
    print("\n--- Series 슬라이싱 (정수 인덱스) ---")
    print("처음부터 4번째까지:", s[:5]) # 인덱스 0부터 4까지 (인덱스 5 미포함)
    print("2번째부터 4번째까지:", s[2:5]) # 인덱스 2부터 4까지 (인덱스 5 미포함)
    print("3번째부터 끝까지:", s[3:]) # 인덱스 3부터 끝까지
    # 출력 예시:
    # 처음부터 4번째까지:
    # 0     10
    # 1    200
    # 2     30
    # 3     40
    # 4     50
    # dtype: int64
    ```

3.  **레이블 인덱스 접근**
    딕셔너리처럼 레이블 인덱스를 사용하여 데이터에 접근합니다. 슬라이싱 시 레이블 인덱스는 끝 인덱스를 포함합니다.

    ```python
    # 레이블 인덱스 사용
    data_labeled = {'one': '일', 'two': '이', 'three': '삼', 'four': '사', 'five': '오'}
    series_labeled = pd.Series(data_labeled)

    print("\n--- Series 레이블 인덱스 접근 ---")
    print(f"레이블 'one' 데이터: {series_labeled['one']}")
    # 출력: 레이블 'one' 데이터: 일

    print("\n--- Series 레이블 인덱스 슬라이싱 ---")
    print("레이블 'one'부터 'three'까지:\n", series_labeled['one']:'three'])
    # 출력:
    # 레이블 'one'부터 'three'까지:
    # one      일
    # two      이
    # three    삼
    # dtype: object
    ```

4.  **조건식을 이용한 필터링 (Boolean Indexing)**
    특정 조건을 만족하는 데이터만 선택할 때 유용합니다. 조건식의 결과로 True/False Series가 생성되며, 이를 사용하여 원본 Series에서 True에 해당하는 값만 추출합니다.

    ```python
    import pandas as pd

    s = pd.Series([10, 20, 30, 40, 50, 60])

    print("\n--- Series 조건식 필터링 ---")
    print("값이 30보다 큰 데이터:\n", s[s > 30])
    # 출력:
    # 값이 30보다 큰 데이터:
    # 3    40
    # 4    50
    # 5    60
    # dtype: int64

    print("값이 20 이상 50 이하인 데이터:\n", s[(s >= 20) & (s <= 50)])
    # 출력:
    # 값이 20 이상 50 이하인 데이터:
    # 1    20
    # 2    30
    # 3    40
    # 4    50
    # dtype: int64
    ```

### 3.2. DataFrame (2차원 데이터)

#### 3.2.1. DataFrame의 개념 및 특징

1.  **정의**: DataFrame은 Pandas의 핵심 2차원 데이터 구조로, **행(row)과 열(column)로 이루어진 테이블 형태**를 가집니다. 관계형 데이터베이스의 테이블, Excel 스프레드시트, 또는 CSV 파일과 매우 유사합니다.
2.  **구조**: 각 열은 Series 객체로 볼 수 있으며, 서로 다른 데이터 타입을 가질 수 있습니다. 이는 각 컬럼이 독립적인 데이터 유형을 가질 수 있음을 의미합니다 (예: 이름은 문자열, 나이는 정수, 도시는 문자열).
3.  **활용**: 대부분의 정형 데이터 분석 작업은 DataFrame을 중심으로 이루어지며, 다양한 데이터 소스(CSV, Excel, SQL 등)의 데이터를 DataFrame으로 쉽게 불러오고 저장할 수 있습니다.

#### 3.2.2. DataFrame 생성 방법

다양한 방법으로 DataFrame을 생성할 수 있습니다.

1.  **딕셔너리(dict)로 DataFrame 만들기**
    가장 일반적인 방법으로, 딕셔너리의 키(key)가 컬럼 이름이 되고, 값(value)은 리스트나 Series 형태의 데이터가 됩니다. 각 리스트의 길이는 동일해야 합니다.

    ```python
    import pandas as pd

    data = {
        'name': ['홍길동', '임꺽정', '장길산', '홍경래'],
        'kor': [90, 80, 70, 70],
        'eng': [99, 98, 97, 46],
        'mat': [90, 70, 70, 60],
    }
    df = pd.DataFrame(data)
    print("--- 딕셔너리로 생성된 DataFrame ---")
    print("타입:", type(df))
    print(df)
    # 출력:
    # --- 딕셔너리로 생성된 DataFrame ---
    # 타입: <class 'pandas.core.frame.DataFrame'>
    #   name  kor  eng  mat
    # 0  홍길동   90   99   90
    # 1  임꺽정   80   98   70
    # 2  장길산   70   97   70
    # 3  홍경래   70   46   60
    ```

2.  **리스트(list)의 리스트로 DataFrame 만들기**
    각 내부 리스트가 한 행을 나타내고, `columns` 인자를 사용하여 컬럼 이름을 지정할 수 있습니다. `columns`를 지정하지 않으면 0부터 시작하는 정수 컬럼 인덱스가 부여됩니다.

    ```python
    import pandas as pd

    data_list = [
        ['Alice', 25, 'New York'],
        ['Bob', 30, 'London'],
        ['Charlie', 35, 'Paris']
    ]
    df_from_list = pd.DataFrame(data_list, columns=['Name', 'Age', 'City'])
    print("\n--- 리스트의 리스트로 생성된 DataFrame ---")
    print(df_from_list)
    # 출력:
    # --- 리스트의 리스트로 생성된 DataFrame ---
    #       Name  Age      City
    # 0    Alice   25  New York
    # 1      Bob   30    London
    # 2  Charlie   35     Paris
    ```

3.  **NumPy 배열(ndarray)로 DataFrame 만들기**
    NumPy 2차원 배열을 사용하여 DataFrame을 생성할 수 있습니다. 이 경우에도 `columns` 인자를 통해 컬럼 이름을 지정하는 것이 일반적입니다.

    ```python
    import numpy as np
    import pandas as pd

    np_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_from_np = pd.DataFrame(np_data, columns=['Col1', 'Col2', 'Col3'])
    print("\n--- NumPy 배열로 생성된 DataFrame ---")
    print(df_from_np)
    # 출력:
    # --- NumPy 배열로 생성된 DataFrame ---
    #    Col1  Col2  Col3
    # 0     1     2     3
    # 1     4     5     6
    # 2     7     8     9
    ```

#### 3.2.3. DataFrame 데이터 접근 및 조작

DataFrame의 데이터는 다양한 방법으로 접근하고 조작할 수 있습니다.

1.  **컬럼(열) 선택**
    단일 컬럼은 Series 형태로, 여러 컬럼은 DataFrame 형태로 반환됩니다. 컬럼 이름은 대소문자를 구분합니다.

    ```python
    import pandas as pd

    data = {
        'name': ['홍길동', '임꺽정', '장길산', '홍경래', '이상민', '김수경'],
        'kor': [90, 80, 70, 70, 60, 70],
        'eng': [99, 98, 97, 46, 77, 56],
        'mat': [90, 70, 70, 60, 88, 99],
    }
    df = pd.DataFrame(data)

    print("--- DataFrame 컬럼 선택 ---")
    print("특정 열 ('name')만 출력:\n", df['name']) # Series 반환
    print("\n여러 열 ('kor', 'eng') 출력:\n", df[['kor', 'eng']]) # DataFrame 반환
    print("\n모든 컬럼 이름 출력:", df.columns)
    # 출력 예시:
    # 특정 열 ('name')만 출력:
    # 0    홍길동
    # 1    임꺽정
    # 2    장길산
    # 3    홍경래
    # 4    이상민
    # 5    김수경
    # Name: name, dtype: object
    ```

2.  **`head()`, `tail()`, `info()`, `describe()` 등 기본 정보 확인**
    데이터프레임의 크기가 클 때 전체를 출력하는 대신, 데이터의 구조와 통계적 요약을 빠르게 파악하는 데 유용합니다.

    ```python
    print("\n--- DataFrame 기본 정보 확인 ---")
    print("앞의 다섯 행만 출력 (df.head()):\n", df.head()) # 기본값 5행, df.head(n)으로 n행 지정 가능
    print("\n뒤의 세 행만 출력 (df.tail(3)):\n", df.tail(3)) # df.tail(n)으로 n행 지정 가능
    print("\nDataFrame 정보 요약 (df.info()):")
    df.info() # 컬럼별 데이터 타입, Non-null 값 개수, 메모리 사용량 등
    print("\nDataFrame 통계 요약 (df.describe()):\n", df.describe()) # 숫자형 컬럼의 개수, 평균, 표준편차, 최소/최대값, 사분위수 등 기술 통계
    ```

3.  **`iloc` 함수 (위치 기반 인덱싱)**
    행과 열의 정수 인덱스 번호(위치)를 이용하여 데이터에 접근합니다. `iloc[행 인덱스, 열 인덱스]` 형식으로 사용합니다. 슬라이싱 시 끝 인덱스는 포함하지 않습니다 (파이썬 리스트 슬라이싱과 동일).

    ```python
    print("\n--- DataFrame iloc 함수 사용 (위치 기반) ---")
    print(f"df.iloc[0, 0]: {df.iloc[0, 0]}") # 0행 0열 데이터 (홍길동)
    print(f"df.iloc[3, 2]: {df.iloc[3, 2]}") # 3행 2열 데이터 (eng 컬럼의 홍경래 점수: 46)
    print("df.iloc[2:4, 2] (2~3행의 2열):\n", df.iloc[2:4, 2]) # 2행(인덱스 2)부터 3행(인덱스 3)까지의 2열(eng) 데이터
    print("df.iloc[2:4, 2:4] (2~3행의 2~3열):\n", df.iloc[2:4, 2:4]) # 2행부터 3행까지의 2열(eng)부터 3열(mat)까지 데이터
    # 출력 예시:
    # df.iloc[0, 0]: 홍길동
    # df.iloc[3, 2]: 46
    ```

4.  **`loc` 함수 (레이블 기반 인덱싱)**
    행의 레이블 인덱스(기본적으로 정수)와 열의 컬럼명(레이블)을 이용하여 데이터에 접근합니다. `loc[행 레이블, 열 레이블]` 형식으로 사용합니다. 슬라이싱 시 끝 레이블을 포함합니다 (Series 레이블 슬라이싱과 동일).

    ```python
    print("\n--- DataFrame loc 함수 사용 (레이블 기반) ---")
    print(f"df.loc[0, 'name']: {df.loc[0, 'name']}") # 0행 'name' 컬럼 데이터 (홍길동)
    print(f"df.loc[3, 'eng']: {df.loc[3, 'eng']}") # 3행 'eng' 컬럼 데이터 (46)
    print("df.loc[:, 'name':'eng'] (모든 행의 'name'부터 'eng'까지):\n", df.loc[:, 'name']:'eng'])
    # 출력 예시:
    # df.loc[0, 'name']: 홍길동
    # df.loc[3, 'eng']: 46
    ```

5.  **조건식을 이용한 필터링 (Boolean Indexing)**
    DataFrame에서도 Series와 유사하게 조건식을 사용하여 특정 조건을 만족하는 행을 선택할 수 있습니다. 여러 조건을 결합할 때는 `&` (AND), `|` (OR) 연산자를 사용하고 각 조건은 괄호로 묶어야 합니다.

    ```python
    print("\n--- DataFrame 조건식 필터링 ---")
    print("국어 점수가 80점 이상인 학생:\n", df[df['kor'] >= 80])
    # 출력:
    # 국어 점수가 80점 이상인 학생:
    #   name  kor  eng  mat
    # 0  홍길동   90   99   90
    # 1  임꺽정   80   98   70

    print("\n영어 점수가 90점 이상이고 수학 점수가 80점 이상인 학생:\n", df[(df['eng'] >= 90) & (df['mat'] >= 80)])
    # 출력:
    # 영어 점수가 90점 이상이고 수학 점수가 80점 이상인 학생:
    #   name  kor  eng  mat
    # 0  홍길동   90   99   90
    ```

6.  **컬럼 추가 및 수정**
    새로운 컬럼을 추가하거나 기존 컬럼의 값을 수정하는 것은 매우 간단합니다. 새로운 컬럼은 기존 컬럼들의 연산 결과로 생성될 수 있습니다.

    ```python
    print("\n--- DataFrame 컬럼 추가 및 수정 ---")
    # 'total' 컬럼 추가 (기존 컬럼들의 합)
    df['total'] = df['kor'] + df['eng'] + df['mat']
    print("total 컬럼 추가 후:\n", df)

    # 'avg' 컬럼 추가 (total 컬럼의 평균)
    df['avg'] = df['total'] / 3
    print("\navg 컬럼 추가 후:\n", df)

    # 기존 컬럼 값 수정 (예: 'kor' 점수를 10점씩 올리기)
    df['kor'] = df['kor'] + 10
    print("\n'kor' 점수 10점씩 올린 후:\n", df)
    ```

7.  **컬럼 삭제**
    `drop()` 메서드를 사용하여 컬럼을 삭제할 수 있습니다. `axis=1`은 컬럼(열)을 의미합니다. `inplace=True`를 사용하면 원본 DataFrame을 직접 수정하고, 그렇지 않으면 수정된 새 DataFrame을 반환합니다. 원본 유지를 위해 `inplace=False` (기본값)를 사용하거나, 반환값을 새로운 변수에 할당하는 것이 좋습니다.

    ```python
    print("\n--- DataFrame 컬럼 삭제 ---")
    df_dropped_total = df.drop('total', axis=1) # 'total' 컬럼 삭제 (원본 유지)
    print("total 컬럼 삭제 후 (원본 유지):\n", df_dropped_total)

    # 여러 컬럼 삭제
    df_dropped_multiple = df.drop(['avg', 'mat'], axis=1)
    print("\navg, mat 컬럼 삭제 후:\n", df_dropped_multiple)
    ```

8.  **행 추가 및 삭제**
    행을 추가할 때는 `pd.concat()` 함수를 사용하는 것이 권장됩니다. 기존 `append()` 메서드는 Pandas 2.0부터 Deprecated(사용 중단)되었습니다.

    ```python
    print("\n--- DataFrame 행 추가 및 삭제 ---")
    data_fruits = {
        'fruits': ['망고', '딸기', '수박', '파인애플'],
        'price': [2500, 5000, 10000, 7000],
        'count': [5, 2, 2, 4],
    }
    df_fruits = pd.DataFrame(data_fruits)
    print("원본 과일 DataFrame:\n", df_fruits)

    # 새로운 행 추가 (pd.concat을 이용한 권장 방식)
    # pd.concat은 여러 DataFrame이나 Series를 연결할 때 사용합니다.
    # 새로운 행을 DataFrame 형태로 만들어 기존 DataFrame과 연결합니다.
    new_row = pd.DataFrame([{'fruits': '사과', 'price': 3500, 'count': 10}])
    df_fruits = pd.concat([df_fruits, new_row], ignore_index=True)
    print("\n새로운 행 추가 후 (pd.concat 사용):\n", df_fruits)


    # 특정 행 삭제 (axis=0: 행)
    df_fruits_dropped_row = df_fruits.drop(0, axis=0) # 0번 인덱스 행 삭제
    print("\n0번 인덱스 행 삭제 후:\n", df_fruits_dropped_row)

    # 여러 행 삭제
    df_fruits_dropped_multiple_rows = df_fruits.drop([1, 3], axis=0) # 1번, 3번 인덱스 행 삭제
    print("\n1번, 3번 인덱스 행 삭제 후:\n", df_fruits_dropped_multiple_rows)
    ```

### 3.3. Panel (3차원 데이터 - Deprecated)

1.  **과거의 3차원 구조**: Panel은 과거 Pandas에서 3차원 데이터를 다루기 위해 제공했던 구조입니다. 3개의 축(Axis 0: items, Axis 1: major_axis, Axis 2: minor_axis)을 가졌으며, Axis 0은 2차원 DataFrame에 해당하고, Axis 1은 DataFrame의 행(row), Axis 2는 DataFrame의 열(column)에 해당했습니다.
2.  **사용 중단**: **중요**: Pandas 0.25.0 버전부터 Panel은 공식적으로 **Deprecated(사용 중단)** 되었으며, 향후 버전에서는 완전히 제거될 예정입니다. 이는 Panel의 복잡성과 사용성의 한계 때문입니다.
3.  **대안**: 3차원 이상의 데이터를 다룰 때는 다음과 같은 대안을 사용하는 것이 권장됩니다:
    *   **MultiIndex (계층적 인덱스) DataFrame**: 기존 DataFrame에 여러 레벨의 인덱스를 사용하여 3차원 이상의 데이터를 2차원 형태로 효율적으로 표현할 수 있습니다. 이는 시계열 데이터나 패널 데이터 분석에 유용합니다.
    *   **`xarray` 라이브러리**: 다차원 배열 데이터를 다루는 데 특화된 라이브러리로, Pandas와 유사한 인터페이스를 제공하며 기상학, 해양학 등 과학 데이터 분석에 널리 사용됩니다. NumPy 배열에 레이블을 붙여 다차원 데이터를 쉽게 관리할 수 있게 합니다.

따라서 Panel을 사용하는 대신 위 대안들을 고려해야 합니다. 기존 코드에 Panel이 있다면 MultiIndex DataFrame으로 전환하는 것을 권장합니다.

## 4. Pandas 데이터 연산

Pandas 객체(Series, DataFrame) 간의 산술 연산은 NumPy 배열과 유사하게 **벡터화된 방식**으로 수행됩니다. 이는 파이썬의 일반적인 반복문보다 훨씬 빠르고 효율적입니다.

### 4.1. 연산의 기본 원리

Pandas 객체 간의 연산은 다음 3단계를 거쳐 수행됩니다:

1.  **인덱스 정렬 (Alignment)**: 연산을 수행하기 전에 Pandas는 연산에 참여하는 모든 객체의 행(row) 및 열(column) 인덱스를 자동으로 정렬합니다. 즉, 동일한 레이블을 가진 요소끼리 연산이 이루어집니다.
2.  **일대일 대응**: 정렬된 인덱스를 기반으로 동일한 위치(인덱스/컬럼명)에 있는 원소끼리 일대일로 대응됩니다.
3.  **연산 처리**: 대응된 원소끼리 지정된 산술 연산(덧셈, 뺄셈, 곱셈, 나눗셈 등)을 수행합니다.
    *   **NaN (Not a Number) 처리**: 만약 연산 과정에서 한 객체에는 존재하지만 다른 객체에는 대응되는 인덱스/컬럼이 없는 경우, 해당 위치의 결과는 `NaN` (결측치)으로 처리됩니다. 이는 데이터의 불일치성을 명확히 보여주는 Pandas의 중요한 특징입니다. `NaN`은 "Not a Number"의 약자로, 유효하지 않거나 정의되지 않은 값을 나타냅니다.

### 4.2. Series 간 연산

Series 간의 연산은 인덱스를 기준으로 정렬된 후 수행됩니다.

1.  **기본 산술 연산**: 동일한 인덱스를 가진 Series끼리 연산이 수행됩니다. 인덱스 순서가 달라도 Pandas가 자동으로 정렬하여 올바른 연산을 수행합니다.

    ```python
    import pandas as pd

    data1 = {'kor': 90, 'eng': 70, 'mat': 80}
    data2 = {'kor': 90, 'eng': 70, 'mat': 80}
    data3 = {'kor': 90, 'eng': 70, 'mat': 80}
    data4 = {'eng': 90, 'mat': 70, 'kor': 80} # 인덱스 순서가 달라도 자동 정렬

    series1 = pd.Series(data1)
    series2 = pd.Series(data2)
    series3 = pd.Series(data3)
    series4 = pd.Series(data4)

    print("--- Series 기본 연산 ---")
    print("series1:\n", series1)
    print("series4 (인덱스 순서 다름):\n", series4)

    result_sum = series1 + series2 + series3 + series4
    result_avg = result_sum / 4

    print("\n총점 (result_sum):\n", result_sum)
    print("\n평균 (result_avg):\n", result_avg)
    # 출력 예시:
    # 총점 (result_sum):
    # eng    300.0
    # kor    350.0
    # mat    300.0
    # dtype: float64
    # 평균 (result_avg):
    # eng    75.0
    # kor    87.5
    # mat    75.0
    # dtype: float64
    ```
    위 예시에서 `series4`의 인덱스 순서가 다르지만, Pandas는 `eng`, `kor`, `mat` 인덱스를 기준으로 자동으로 정렬하여 올바른 연산을 수행합니다. 이는 Pandas의 강력한 인덱스 정렬 기능 덕분입니다.

2.  **NaN (결측치) 처리**: 연산 대상 Series 중 한쪽에만 존재하는 인덱스가 있을 경우, 해당 위치의 결과는 `NaN`이 됩니다. 이는 데이터의 불완전성을 나타냅니다.

    ```python
    import pandas as pd

    # 인덱스가 없을 경우 NaN 처리 예시
    data_s1 = {'kor': 90, 'mat': 80}
    data_s2 = {'kor': 90, 'eng': 70}
    data_s3 = {'kor': 90, 'eng': 70, 'mat': 80}

    series_nan1 = pd.Series(data_s1)
    series_nan2 = pd.Series(data_s2)
    series_nan3 = pd.Series(data_s3)

    print("\n--- Series NaN 처리 예시 ---")
    print("series_nan1:\n", series_nan1)
    print("series_nan2:\n", series_nan2)
    print("series_nan3:\n", series_nan3)

    result_nan = series_nan1 + series_nan2 + series_nan3
    print("\nNaN 포함 결과 (result_nan):\n", result_nan)
    # 출력:
    # NaN 포함 결과 (result_nan):
    # eng      NaN
    # kor    270.0
    # mat      NaN
    # dtype: float64
    ```
    `eng`의 경우 `series_fill1`에는 없지만 `series_fill2`와 `series_fill3`에 있으므로 `0 + 70 + 70 = 140`이 됩니다. `mat`의 경우 `series_fill2`에는 없지만 `series_fill1`과 `series_fill3`에 있으므로 `80 + 0 + 80 = 160`이 됩니다. 이 옵션은 결측치로 인해 연산이 중단되는 것을 방지하고, 특정 기본값을 가정하여 계산을 이어갈 때 유용합니다.

3.  **`fill_value` 옵션 사용**: `NaN`이 발생하는 것을 방지하고 싶을 때, `add()`, `sub()`, `mul()`, `div()`와 같은 메서드에 `fill_value` 인자를 사용하여 결측치를 특정 값으로 채운 후 연산을 수행할 수 있습니다. 이는 데이터에 결측치가 있더라도 연산을 강제로 수행하고 싶을 때 유용합니다.

    ```python
    import pandas as pd

    data_s1 = {'kor': 90, 'mat': 80}
    data_s2 = {'kor': 90, 'eng': 70}
    data_s3 = {'kor': 90, 'eng': 70, 'mat': 80}

    series_fill1 = pd.Series(data_s1)
    series_fill2 = pd.Series(data_s2)
    series_fill3 = pd.Series(data_s3)

    print("\n--- Series fill_value 옵션 사용 ---")
    # fill_value=0으로 NaN 대신 0 사용
    # series_fill1.add(series_fill2, fill_value=0)는 series_fill1과 series_fill2를 더하되,
    # 둘 중 하나에만 있는 인덱스의 값은 0으로 간주하여 더합니다.
    # 그 결과에 다시 series_fill3를 더할 때도 동일하게 fill_value=0을 적용합니다.
    result_fill = series_fill1.add(series_fill2, fill_value=0).add(series_fill3, fill_value=0)
    print("fill_value=0 적용 결과 (result_fill):\n", result_fill)
    # 출력:
    # fill_value=0 적용 결과 (result_fill):
    # eng    140.0
    # kor    270.0
    # mat    160.0
    # dtype: float64
    ```
    `eng`의 경우 `series_fill1`에는 없지만 `series_fill2`와 `series_fill3`에 있으므로 `0 + 70 + 70 = 140`이 됩니다. `mat`의 경우 `series_fill2`에는 없지만 `series_fill1`과 `series_fill3`에 있으므로 `80 + 0 + 80 = 160`이 됩니다. 이 옵션은 결측치로 인해 연산이 중단되는 것을 방지하고, 특정 기본값을 가정하여 계산을 이어갈 때 유용합니다.

### 4.3. DataFrame 간 연산 및 컬럼 추가/삭제

DataFrame 간의 연산도 Series와 마찬가지로 행 인덱스와 열 인덱스 모두를 기준으로 정렬된 후 수행됩니다.

1.  **기본 연산과 새 필드 추가**: DataFrame의 각 컬럼은 Series 객체이므로, 컬럼 간의 연산은 Series 연산과 동일하게 작동합니다. 이를 활용하여 기존 컬럼의 데이터를 기반으로 새로운 파생 컬럼을 쉽게 생성할 수 있습니다.

    ```python
    import pandas as pd

    # Series를 결합하여 DataFrame 만들기 (append는 Deprecated, pd.concat 권장)
    # 기존 코드:
    # data = pd.DataFrame()
    # data = data.append(data1, ignore_index=True)
    # data = data.append(data2, ignore_index=True)
    # data = data.append(data3, ignore_index=True)

    # 권장 방식: 딕셔너리 리스트로 DataFrame 생성
    data_for_df = [
        {'kor': 90, 'eng': 70, 'mat': 80},
        {'kor': 90, 'eng': 70, 'mat': 80},
        {'kor': 90, 'eng': 70, 'mat': 80}
    ]
    df_scores = pd.DataFrame(data_for_df)
    print("--- DataFrame 기본 연산 및 새 필드 추가 ---")
    print("원본 DataFrame:\n", df_scores)

    # 새 필드 'total' 추가: 'kor', 'eng', 'mat' 컬럼의 합
    df_scores['total'] = df_scores.kor + df_scores.eng + df_scores.mat
    print("\n'total' 컬럼 추가 후:\n", df_scores)

    # 새 필드 'avg' 추가: 'total' 컬럼을 3으로 나눈 값
    df_scores['avg'] = df_scores.total / 3
    print("\navg 컬럼 추가 후:\n", df_scores)
    # 출력 예시:
    # 'avg' 컬럼 추가 후:
    #    kor  eng  mat  total        avg
    # 0   90   70   80    240  80.000000
    # 1   90   70   80    240  80.000000
    # 2   90   70   80    240  80.000000
    ```

2.  **DataFrame 간의 연산**: 두 DataFrame 간의 연산은 행 인덱스와 열 인덱스 모두를 기준으로 정렬된 후 수행됩니다. 공통되지 않은 인덱스/컬럼에 대해서는 `NaN`이 발생합니다.

    ```python
    import pandas as pd

    df_a = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }, index=['x', 'y', 'z'])

    df_b = pd.DataFrame({
        'B': [10, 20, 30],
        'C': [40, 50, 60]
    }, index=['y', 'z', 'w'])

    print("\n--- DataFrame 간 연산 ---")
    print("df_a:\n", df_a)
    print("\ndf_b:\n", df_b)

    # df_a와 df_b 더하기
    # 공통된 인덱스('y', 'z')와 컬럼('B')에 대해서만 연산 수행
    # 나머지는 NaN 처리
    df_sum = df_a + df_b
    print("\ndf_a + df_b 결과:\n", df_sum)
    # 출력:
    # df_a + df_b 결과:
    #      A     B     C
    # x  NaN   NaN   NaN
    # y  NaN  15.0   NaN
    # z  NaN  26.0   NaN
    ```
    위 결과에서 `A` 컬럼은 `df_b`에 없고, `C` 컬럼은 `df_a`에 없으므로 해당 컬럼들은 `NaN`이 됩니다. 또한, `x` 행은 `df_b`에 없고, `w` 행은 `df_a`에 없으므로 해당 행들도 `NaN`이 됩니다. 오직 공통된 인덱스와 컬럼(`y`, `z` 행의 `B` 컬럼)에 대해서만 연산이 수행됩니다.

## 5. 실전 예제: 데이터프레임 활용

다음은 DataFrame을 생성하고 데이터를 추가한 후, 새로운 파생 컬럼을 계산하는 실습 예제입니다. 이 예제를 통해 Pandas의 기본적인 데이터 조작 능력을 익힐 수 있습니다.

### 과제 1: 데이터프레임 만들기 및 연산

다음 테이블 데이터를 DataFrame 객체로 만들고, 새로운 데이터 (10, 20, 30, 40)를 추가한 후 `total` 필드를 만들어 각 필드의 합을 구하세요.

| X1  | X2  | X3   | X4 |
| :-- | :-- | :--- | :- |
| 2.9 | 9.2 | 13.2 | 2  |
| 2.4 | 8.7 | 11.5 | 3  |
| 2   | 7.2 | 10.8 | 4  |
| 2.3 | 8.5 | 12.3 | 3  |
| 3.2 | 9.6 | 12.6 | 2  |

```python
import pandas as pd

data = {
    'X1': [2.9, 2.4, 2, 2.3, 3.2],
    'X2': [9.2, 8.7, 7.2, 8.5, 9.6],
    'X3': [13.2, 11.5, 10.8, 12.3, 12.6],
    'X4': [2, 3, 4, 3, 2]
}

df = pd.DataFrame(data)
print("--- 초기 DataFrame ---")
print(df)

# 새로운 행 추가 (pd.concat 사용 권장)
# pd.concat을 사용하여 기존 DataFrame에 새로운 행을 추가합니다.
# new_row_data는 단일 행을 가진 DataFrame으로 생성되어야 합니다.
new_row_data = pd.DataFrame([{'X1': 10, 'X2': 20, 'X3': 30, 'X4': 40}])
df = pd.concat([df, new_row_data], ignore_index=True)
print("\n--- 새로운 행 추가 후 DataFrame ---")
print(df)

# total 필드 추가: X1, X2, X3, X4 필드의 합
# DataFrame의 컬럼들을 직접 더하여 새로운 'total' 컬럼을 생성합니다.
df['total'] = df.X1 + df.X2 + df.X3 + df.X4
print("\n--- 'total' 필드 추가 후 DataFrame ---")
print(df)
# 출력 예시:
# --- 'total' 필드 추가 후 DataFrame ---
#      X1    X2    X3  X4  total
# 0   2.9   9.2  13.2   2   27.3
# 1   2.4   8.7  11.5   3   25.6
# 2   2.0   7.2  10.8   4   24.0
# 3   2.3   8.5  12.3   3   26.1
# 4   3.2   9.6  12.6   2   27.4
# 5  10.0  20.0  30.0  40  100.0
```

## 6. Pandas와 머신러닝 워크플로우

Pandas는 머신러닝 프로젝트의 거의 모든 단계에서 핵심적인 역할을 수행합니다. 특히 데이터 준비 및 탐색 단계에서 그 중요성이 두드러집니다. 부트캠프에서 다루는 머신러닝 과정에서 Pandas가 어떻게 활용되는지 살펴보겠습니다.

### 6.1. 데이터 로딩 및 탐색

1.  **다양한 데이터 소스 로딩**: Pandas는 CSV, Excel, SQL 데이터베이스, JSON 등 다양한 형식의 원시 데이터를 `DataFrame`으로 쉽게 불러올 수 있습니다. 예를 들어, `pd.read_csv('data.csv')`와 같이 간단한 명령으로 데이터를 메모리에 로드할 수 있습니다.
2.  **초기 데이터 탐색**: 로드된 데이터의 구조, 통계적 특성, 결측치 여부 등을 빠르게 파악하는 데 사용됩니다.
    *   `df.head()`, `df.tail()`: 데이터의 상위/하위 일부를 확인하여 데이터의 형태를 빠르게 파악합니다.
    *   `df.info()`: 각 컬럼의 데이터 타입, Non-null 값의 개수, 메모리 사용량 등 전반적인 정보를 제공하여 결측치 여부를 확인하는 데 도움을 줍니다.
    *   `df.describe()`: 숫자형 컬럼에 대한 기술 통계(평균, 표준편차, 최소/최대값, 사분위수 등)를 제공하여 데이터의 분포를 이해하는 데 활용됩니다.
    *   `df.value_counts()`: 특정 범주형 컬럼의 고유 값과 그 빈도를 확인합니다.
    *   `df.corr()`: 컬럼 간의 상관관계를 계산하여 특성 간의 선형 관계를 파악합니다.

### 6.2. 데이터 전처리

머신러닝 모델은 깨끗하고 정돈된 데이터를 필요로 합니다. Pandas는 데이터 전처리를 위한 강력한 도구들을 제공합니다.

1.  **결측치 처리**: 누락된 데이터(`NaN`)를 처리하는 것은 전처리 과정에서 매우 중요합니다.
    *   `df.isnull().sum()`: 각 컬럼별 결측치의 개수를 확인합니다.
    *   `df.fillna(value)`: 결측치를 특정 값(평균, 중앙값, 최빈값 등)으로 채웁니다.
    *   `df.dropna()`: 결측치가 있는 행 또는 열을 제거합니다.
2.  **데이터 타입 변환**: 컬럼의 데이터 타입을 변경하여 메모리 효율성을 높이거나 특정 연산을 가능하게 합니다 (예: `df['column'].astype('int')`).
3.  **중복 데이터 처리**: `df.duplicated().sum()`으로 중복된 행을 확인하고, `df.drop_duplicates()`로 중복을 제거합니다.
4.  **이상치 탐지 및 처리**: Pandas의 통계 함수와 필터링 기능을 활용하여 이상치(Outlier)를 식별하고 제거하거나 변환합니다. 예를 들어, IQR(Interquartile Range) 방법을 사용하여 이상치를 정의하고 제거할 수 있습니다.

### 6.3. 특성 공학 (Feature Engineering)

기존 데이터를 기반으로 새로운 유의미한 특성을 생성하는 과정입니다. Pandas의 강력한 데이터 조작 기능이 여기서 빛을 발합니다.

1.  **새로운 특성 생성**: 기존 컬럼들을 조합하거나 변환하여 새로운 특성을 만듭니다. 예를 들어, `df['total_score'] = df['math'] + df['science']`와 같이 새로운 점수 합계 컬럼을 만들 수 있습니다.
2.  **범주형 데이터 인코딩**: 머신러닝 모델은 숫자형 데이터를 선호하므로, 범주형 특성을 숫자형으로 변환해야 합니다.
    *   **One-Hot Encoding**: `pd.get_dummies(df['categorical_column'])`를 사용하여 범주형 변수를 여러 개의 이진(0 또는 1) 컬럼으로 변환합니다. 이는 순서가 없는 범주형 데이터에 적합합니다.
    *   **Label Encoding**: `sklearn.preprocessing.LabelEncoder`를 사용하여 각 범주에 고유한 정수 값을 할당합니다. 이는 순서가 있는 범주형 데이터에 적합할 수 있습니다.
3.  **날짜/시간 특성 추출**: 날짜/시간 컬럼에서 연도, 월, 일, 요일, 시간 등 다양한 정보를 추출하여 새로운 특성으로 활용할 수 있습니다 (예: `df['date_column'].dt.year`).

### 6.4. 데이터 분할 및 모델 입력

1.  **데이터 분할**: Pandas DataFrame은 `sklearn.model_selection.train_test_split`과 같은 함수를 사용하여 학습(training), 검증(validation), 테스트(test) 세트로 데이터를 분할하는 데 사용됩니다. 분할된 DataFrame은 NumPy 배열로 변환되어 Scikit-learn과 같은 머신러닝 라이브러리의 모델 입력으로 사용됩니다.
    ```python
    from sklearn.model_selection import train_test_split
    # X: 특성 데이터 (DataFrame), y: 타겟 데이터 (Series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
2.  **모델 입력**: Pandas DataFrame이나 Series는 `.values` 속성을 통해 NumPy 배열로 쉽게 변환될 수 있으며, 이는 대부분의 머신러닝 모델이 요구하는 입력 형식입니다.

### 6.5. 시각화 연동

Pandas는 Matplotlib, Seaborn과 같은 파이썬 시각화 라이브러리와 긴밀하게 연동됩니다. DataFrame의 데이터를 직접 시각화 함수에 전달하여 데이터의 분포, 관계, 패턴 등을 그래프로 표현할 수 있습니다. 이는 탐색적 데이터 분석(EDA) 단계에서 데이터에 대한 깊은 통찰력을 얻는 데 필수적입니다.

*   **Matplotlib**: `df.plot()` 메서드를 통해 기본적인 플롯(선, 막대, 히스토그램 등)을 그릴 수 있습니다.
*   **Seaborn**: 통계적 시각화에 특화된 라이브러리로, Pandas DataFrame을 입력으로 받아 더욱 풍부하고 미려한 그래프를 생성합니다.

## 7. DataFrame 심화

### 7.1. DataFrame의 주요 특징

1.  **2차원 배열 형태**: Pandas가 제공하는 2차원 배열 형태의 데이터 타입으로, 행(row)과 열(column)으로 구성된 테이블 구조를 가집니다.
2.  **다양한 데이터 타입 지원**: 하나의 타입으로만 구성되지 않고, 각 열(컬럼)이 각기 다른 데이터 타입(예: 정수, 실수, 문자열, 불리언 등)으로 구성될 수 있습니다. 이는 실제 데이터의 복잡성을 잘 반영합니다.
3.  **객체 지향적 데이터 처리**: 별도의 구조체나 클래스를 만들지 않고도 사용자 데이터를 취급하기 쉬운 객체입니다. 직관적인 메서드와 속성을 통해 데이터를 쉽게 조작할 수 있습니다.
4.  **풍부한 통계 함수 제공**: 데이터의 요약 통계, 분포 분석 등을 위한 다양한 통계 관련 함수들을 내장하고 있어 데이터 탐색 및 분석에 매우 유용합니다.
5.  **자유로운 접근**: 각 열과 행에 대해 인덱스(위치 기반) 또는 레이블(이름 기반)을 사용하여 자유롭게 접근하고 데이터를 추출하거나 수정할 수 있습니다.

### 7.2. Pandas가 지원하는 파일 형식

Pandas는 다양한 외부 파일 형식을 `DataFrame`으로 읽어오거나 `DataFrame`을 외부 파일로 저장하는 기능을 강력하게 지원합니다. 이는 데이터 분석 워크플로우에서 필수적인 요소입니다.

| 파일 형식 | 설명 | Pandas 함수 (읽기) | Pandas 함수 (쓰기) |
| :-------- | :---------------------------------------------------------------- | :----------------- | :----------------- |
| **CSV**   | 쉼표(Comma)로 구분된 값 (Comma-Separated Values)을 가진 텍스트 파일. 가장 널리 사용되는 데이터 교환 형식. | `pd.read_csv()`    | `df.to_csv()`      |
| **Excel** | Microsoft Excel 스프레드시트 파일 (.xlsx, .xls). 여러 시트 지원. | `pd.read_excel()`  | `df.to_excel()`    |
| **JSON**  | JavaScript Object Notation. 웹 기반 데이터 교환에 주로 사용.   | `pd.read_json()`   | `df.to_json()`     |
| **HTML**  | 웹 페이지의 테이블 데이터를 읽어올 때 사용.                      | `pd.read_html()`   | `df.to_html()`     |
| **SQL**   | 관계형 데이터베이스에서 데이터를 직접 읽어오거나 저장.           | `pd.read_sql()`    | `df.to_sql()`      |
| **XML**   | eXtensible Markup Language. 구조화된 데이터 표현.               | `pd.read_xml()`    | `df.to_xml()`      |
| **Parquet** | 컬럼 기반의 효율적인 이진 파일 형식. 빅데이터 환경에서 성능 우수. | `pd.read_parquet()`| `df.to_parquet()`  |
| **Pickle** | 파이썬 객체를 직렬화하여 저장하는 형식.                           | `pd.read_pickle()` | `df.to_pickle()`   |
| **기타**  | TSV (Tab-Separated Values), 고정폭 파일 등 다양한 형식 지원.   | `pd.read_fwf()`, `pd.read_table()` | - |

## 8. 외부 파일 읽기/쓰기

### 8.1. 외부 파일 처리의 중요성

1.  **데이터 활용**: 실제 데이터 분석 프로젝트에서는 대부분 외부에서 제공되는 데이터를 활용하게 됩니다. 이는 데이터베이스, 웹 스크래핑, API 연동, 또는 파일 형태로 제공될 수 있습니다.
2.  **효율적인 데이터 처리**: Pandas는 다양한 소스에서 수집된 데이터를 일관된 `DataFrame` 형태로 효율적으로 불러오고 처리할 수 있는 기능을 제공하여 데이터 전처리 과정을 간소화합니다.
3.  **분석 결과 공유**: 데이터 분석을 통해 얻은 인사이트나 가공된 데이터를 다른 시스템이나 사용자에게 전달하기 위해, 분석 결과를 다시 외부 파일(CSV, Excel 등)로 저장하는 기능이 필수적입니다.

## 9. 파일 경로 처리

데이터 파일을 읽거나 쓸 때 파일의 위치를 정확하게 지정하는 것은 매우 중요합니다. 파일 경로는 크게 절대 경로와 상대 경로로 나뉩니다.

### 9.1. 경로 표현 방식

1.  **절대 경로 (Absolute Path)**:
    *   파일 시스템의 루트(최상위) 디렉토리부터 파일의 전체 위치를 기술하는 방식입니다.
    *   운영체제에 따라 시작점이 다릅니다. (예: Windows는 `C:\`, Linux/macOS는 `/`)
    *   **예시**: `C:/pandas_workspace/uni_10/data/score.csv` 또는 `/home/user/data/score.csv`
    *   **장점**: 파일의 위치가 명확하여 혼동의 여지가 적습니다.
    *   **단점**: 파일이나 프로젝트의 위치가 변경되면 경로를 수정해야 하는 번거로움이 있습니다.

2.  **상대 경로 (Relative Path)**:
    *   현재 애플리케이션(스크립트)이 실행 중인 폴더를 기준으로 파일의 위치를 기술하는 방식입니다.
    *   `.` (도트): 현재 디렉토리를 의미합니다.
    *   `..` (도트 두 개): 현재 디렉토리의 상위(부모) 디렉토리를 의미합니다.
    *   **예시**: `./data/score.csv` (현재 폴더 아래 `data` 폴더의 `score.csv`)
    *   **장점**: 프로젝트의 위치가 변경되어도 경로를 수정할 필요가 없어 프로젝트 이식성이 높습니다.
    *   **단점**: 현재 작업 디렉토리를 정확히 알아야 합니다.

### 9.2. 파이썬에서의 경로 처리 주의사항

Windows 운영체제에서는 파일 경로를 나타낼 때 역슬래시(`\`)를 사용하지만, 파이썬 문자열에서 역슬래시는 이스케이프 문자(`\n`, `\t` 등)로 해석될 수 있어 문제가 발생할 수 있습니다. 이를 해결하는 방법은 다음과 같습니다.

1.  **이스케이프 문자 사용**: 역슬래시를 두 번 사용하여 이스케이프 처리합니다.
    *   **예시**: `"c:\\pandas_workspace\\data\\score.csv"`
2.  **원시 문자열 (Raw String) 사용**: 문자열 앞에 `r`을 붙여 해당 문자열을 있는 그대로 해석하도록 합니다.
    *   **예시**: `r"c:\pandas_workspace\data\score.csv"`
3.  **슬래시(`/`) 사용 (권장)**: Windows에서도 슬래시(`/`)를 경로 구분자로 사용할 수 있으며, 이는 운영체제에 독립적이므로 가장 권장되는 방법입니다.
    *   **예시**: `"c:/pandas_workspace/data/score.csv"`

**경로 선택 권장사항**: 일반적으로 **절대 경로보다는 상대 경로 사용을 권장**합니다. 이는 폴더 이동 시 경로 수정 필요가 없어 프로젝트의 이식성을 크게 향상시키기 때문입니다.

## 10. CSV 파일 처리

CSV (Comma-Separated Values) 파일은 데이터를 쉼표(`,`)로 구분하여 저장하는 텍스트 파일 형식입니다. 가장 보편적으로 사용되는 데이터 교환 형식 중 하나입니다.

### 10.1. CSV 파일의 특징

1.  **텍스트 기반**: 데이터를 쉼표(`,`)로 구분하는 일반 텍스트 파일입니다. 특정 프로그램 없이 메모장과 같은 일반 텍스트 에디터로도 내용을 확인하고 작성할 수 있습니다.
2.  **간편한 편집**: Excel과 같은 스프레드시트 프로그램에서도 쉽게 열고 편집할 수 있습니다.
3.  **널리 사용**: 빅데이터 환경에서 데이터를 저장하고 교환하는 데 가장 많이 사용되는 형태 중 하나입니다. 다양한 시스템 간의 데이터 연동에 용이합니다.

### 10.2. CSV 파일 읽기

Pandas의 `read_csv()` 함수를 사용하여 CSV 파일을 `DataFrame`으로 불러올 수 있습니다. 이 함수는 다양한 옵션을 제공하여 복잡한 CSV 파일도 유연하게 처리할 수 있습니다.

**기본 읽기 구문**:
```python
import pandas as pd

# 현재 스크립트가 있는 디렉토리의 data 폴더 안에 score.csv 파일이 있다고 가정
data = pd.read_csv("./data/score.csv")
```

**주요 `read_csv()` 옵션**:

*   `header`: 제목 줄(컬럼명)의 위치를 지정합니다. 기본값은 `0` (첫 번째 줄)입니다.
    *   `header=None`: 파일에 제목 줄이 없음을 나타냅니다. 이 경우 Pandas가 0부터 시작하는 정수 인덱스를 컬럼명으로 자동 부여합니다.
    *   `header=N`: N+1번째 줄을 제목 줄로 사용합니다 (0부터 시작하는 인덱스).
*   `encoding`: 파일의 문자 인코딩 방식을 지정합니다. (예: `'utf-8'`, `'cp949'`, `'euc-kr'`)
*   `sep` 또는 `delimiter`: 데이터를 구분하는 구분자(separator)를 지정합니다. 기본값은 쉼표(`,`)입니다. 탭으로 구분된 파일(`TSV`)의 경우 `sep='\t'`로 지정할 수 있습니다.
*   `index_col`: 특정 컬럼을 DataFrame의 인덱스로 사용할 때 지정합니다.
*   `names`: `header=None`일 때 사용할 컬럼명 리스트를 직접 지정합니다.

### 10.3. CSV 파일 읽기 예제

#### 1. 기본 CSV 파일 읽기

`score.csv` 파일이 다음과 같다고 가정합니다:
```csv
name,kor,eng,mat
홍길동,90,99,90
임꺽정,80,98,70
장길산,70,97,70
홍경래,70,46,60
```

```python
import pandas as pd

data = pd.read_csv("./data/score.csv")
print("--- 기본 CSV 파일 읽기 결과 ---")
print("컬럼명:", data.columns) # DataFrame의 컬럼명 출력
print("인덱스:", data.index)   # DataFrame의 인덱스 정보 출력

# 총점, 평균 구하기: 기존 컬럼을 활용하여 새로운 파생 컬럼 생성
data["total"] = data["kor"] + data["eng"] + data["mat"]
data["avg"] = data["total"] / 3
print("\n--- 총점 및 평균 추가 후 DataFrame ---")
print(data)
# 출력 예시:
# --- 기본 CSV 파일 읽기 결과 ---
# 컬럼명: Index(["name", "kor", "eng", "mat"], dtype="object")
# 인덱스: RangeIndex(start=0, stop=4, step=1)
#
# --- 총점 및 평균 추가 후 DataFrame ---
#   name  kor  eng  mat  total        avg
# 0  홍길동   90   99   90    279  93.000000
# 1  임꺽정   80   98   70    248  82.666667
# 2  장길산   70   97   70    237  79.000000
# 3  홍경래   70   46   60    176  58.666667
```

#### 2. 제목 줄이 없는 CSV 파일 처리

`score_noheader.csv` 파일이 다음과 같다고 가정합니다:
```csv
홍길동,90,99,90
임꺽정,80,98,70
장길산,70,97,70
홍경래,70,46,60
```

```python
import pandas as pd

# 제목 줄이 없을 경우 header=None 옵션 사용
data = pd.read_csv("./data/score_noheader.csv", header=None)
print("--- 제목 줄 없이 읽은 CSV 파일 (자동 컬럼명) ---")
print("컬럼명:", data.columns) # Pandas가 0부터 시작하는 정수 컬럼명을 자동 부여

# 직접 컬럼명 부여: read_csv 후 data.columns 속성을 통해 컬럼명 변경
data.columns = ["name", "kor", "eng", "mat"]
print("\n--- 컬럼명 부여 후 DataFrame ---")
print("컬럼 부여 후:", data.columns)
print(data)

# 총점, 평균 구하기
data["total"] = data["kor"] + data["eng"] + data["mat"]
data["avg"] = data["total"] / 3
print("\n--- 총점 및 평균 추가 후 DataFrame ---")
print(data)
# 출력 예시:
# --- 제목 줄 없이 읽은 CSV 파일 (자동 컬럼명) ---
# 컬럼명: Int64Index([0, 1, 2, 3], dtype="int64")
#
# --- 컬럼명 부여 후 DataFrame ---
# 컬럼 부여 후: Index(["name", "kor", "eng", "mat"], dtype="object")
#   name  kor  eng  mat
# 0  홍길동   90   99   90
# 1  임꺽정   80   98   70
# 2  장길산   70   97   70
# 3  홍경래   70   46   60
```

#### 3. 제목 줄이 특정 위치에 있는 경우

`score_header.csv` 파일이 다음과 같다고 가정합니다: 
```csv
# 이 파일은 학생들의 성적 데이터입니다.
# 데이터 출처: 2025년 1학기
# 컬럼 설명: name(이름), kor(국어), eng(영어), mat(수학)
name,kor,eng,mat
홍길동,90,99,90
임꺽정,80,98,70
```

```python
import pandas as pd

# header가 4번째 줄에 있음 (0부터 시작하는 인덱스로 3)
data = pd.read_csv("./data/score_header.csv", header=3)
print("--- 특정 위치에 제목 줄이 있는 CSV 파일 읽기 결과 ---")
print("컬럼명:", data.columns)
print("인덱스:", data.index)

# 총점, 평균 구하기
data["total"] = data["kor"] + data["eng"] + data["mat"]
data["avg"] = data["total"] / 3
print("\n--- 총점 및 평균 추가 후 DataFrame ---")
print(data)
# 출력 예시:
# --- 특정 위치에 제목 줄이 있는 CSV 파일 읽기 결과 ---
# 컬럼명: Index(["name", "kor", "eng", "mat"], dtype="object")
# 인덱스: RangeIndex(start=0, stop=2, step=1)
#
# --- 총점 및 평균 추가 후 DataFrame ---
#   name  kor  eng  mat  total        avg
# 0  홍길동   90   99   90    279  93.000000
# 1  임꺽정   80   98   70    248  82.666667
```

### 10.4. CSV 파일 저장

`DataFrame` 객체를 CSV 파일로 저장할 때는 `to_csv()` 메서드를 사용합니다. 저장 시 다양한 옵션을 통해 파일 형식을 제어할 수 있습니다.

**기본 저장 구문**:
```python
# DataFrame을 CSV 파일로 저장
data.to_csv("output_file.csv")
```

**주요 `to_csv()` 옵션**:

*   `path_or_buf`: 저장할 파일 경로 및 이름.
*   `sep`: 구분자. 기본값은 쉼표(`,`).
*   `na_rep`: `NaN` (결측치) 값을 대체할 문자열. 기본값은 빈 문자열.
*   `float_format`: 부동 소수점 숫자의 출력 형식 지정.
*   `columns`: 저장할 컬럼의 리스트. 지정하지 않으면 모든 컬럼 저장.
*   `header`: 컬럼명(헤더)을 파일에 쓸지 여부. `True` (기본값) 또는 `False`.
*   `index`: DataFrame의 인덱스를 파일에 쓸지 여부. `True` (기본값) 또는 `False`.
*   `mode`: 파일 쓰기 모드. `'w'` (쓰기, 기본값), `'a'` (추가).
*   `encoding`: 파일의 문자 인코딩 방식. (예: `'utf-8'`, `'cp949'`).
**Excel에서 CSV 파일을 열 때 한글 깨짐 현상이 발생한다면 `encoding="cp949"`를 시도해 볼 수 있습니다.**

**예시**: `score_result.csv` 파일로 저장 (인덱스 제외, cp949 인코딩)
```python
# CSV 파일로 저장
# Excel에서 열어보려면 cp949 인코딩 필요
# index=False로 인덱스 저장 안 함
data.to_csv("score_result.csv", mode='w', encoding="cp949", index=False)
```

## 11. Excel 파일 처리

Pandas는 Microsoft Excel 파일(.xlsx, .xls)을 직접 읽고 쓰는 기능을 제공합니다. 이는 Excel을 주로 사용하는 환경에서 데이터 분석 결과를 공유하거나 데이터를 불러올 때 매우 편리합니다.

### 11.1. Excel 파일의 장점

1.  **별도 라이브러리 불필요**: `openpyxl` (xlsx), `xlrd` (xls)와 같은 백엔드 엔진이 필요하지만, Pandas 설치 시 대부분 함께 설치되므로 사용자가 별도로 COM 라이브러리나 다른 복잡한 라이브러리를 설치할 필요가 없습니다.
2.  **Pandas 직접 지원**: Pandas 내부에 `read_excel()` 및 `to_excel()` 함수가 내장되어 있어 파이썬 코드 내에서 Excel 파일을 쉽게 다룰 수 있습니다.
3.  **복잡한 데이터 구조 처리**: 여러 시트(sheet)를 가진 Excel 파일이나 특정 범위의 데이터도 유연하게 처리할 수 있습니다.

### 11.2. Excel 파일 읽기/쓰기 예제

`score.xlsx` 파일이 다음과 같다고 가정합니다:

| name | kor | eng | mat |
| :--- | :-- | :-- | :-- |
| 홍길동 | 90  | 99  | 90  |
| 임꺽정 | 80  | 98  | 70  |

```python
import pandas as pd

# Excel 파일 읽기: score.xlsx 파일을 DataFrame으로 불러오기
data = pd.read_excel("./data/score.xlsx")

# 총점 및 평균 컬럼 추가
data["total"] = data["kor"] + data["eng"] + data["mat"]
data["avg"] = data["total"] / 3
print("--- Excel 파일 읽기 및 계산 결과 ---")
print(data)

# Excel 파일로 저장
# score_result1.xlsx: DataFrame의 인덱스도 함께 저장 (기본값)
data.to_excel("score_result1.xlsx")
print("\nscore_result1.xlsx 파일이 생성되었습니다 (인덱스 포함).")

# score_result2.xlsx: DataFrame의 인덱스 제외하고 저장
data.to_excel("score_result2.xlsx", index=False)
print("score_result2.xlsx 파일이 생성되었습니다 (인덱스 제외).")

# 출력 예시:
# --- Excel 파일 읽기 및 계산 결과 ---
#   name  kor  eng  mat  total        avg
# 0  홍길동   90   99   90    279  93.000000
# 1  임꺽정   80   98   70    248  82.666667
#
# score_result1.xlsx 파일이 생성되었습니다 (인덱스 포함).
# score_result2.xlsx 파일이 생성되었습니다 (인덱스 제외).
```

**`to_excel()` 저장 시 주요 옵션**:

*   `excel_writer`: 저장할 파일 경로 및 이름.
*   `sheet_name`: 저장할 시트의 이름. 기본값은 `'Sheet1'`.
*   `na_rep`: `NaN` (결측치) 값을 대체할 문자열.
*   `header`: 컬럼명(헤더)을 파일에 쓸지 여부. `True` (기본값) 또는 `False`.
*   `index`: DataFrame의 인덱스를 파일에 쓸지 여부. `True` (기본값) 또는 `False`.

## 12. DataFrame API 활용

Pandas DataFrame은 데이터의 구조를 파악하고 기본적인 통계 정보를 얻는 데 유용한 다양한 API(메서드)를 제공합니다. 이는 데이터 탐색(EDA)의 첫 단계에서 매우 중요합니다.

### 12.1. 기본 정보 확인 API

1.  **`head()` / `tail()`**: DataFrame의 상위 또는 하위 n개의 행을 출력하여 데이터의 전체적인 모습을 빠르게 파악할 수 있습니다. 기본값은 5개 행입니다.

    ```python
    import pandas as pd

    # auto-mpg.csv 파일 로드 (예시 데이터셋)
    data = pd.read_csv("./data/auto-mpg.csv")

    print("--- 앞에서부터 5개 미리 보기 (data.head()) ---")
    print(data.head())

    print("\n--- 뒤에서부터 5개 미리 보기 (data.tail()) ---")
    print(data.tail())

    print("\n--- 앞에서부터 10개 미리 보기 (data.head(10)) ---")
    print(data.head(10))
    ```

2.  **`shape`**: DataFrame의 차원(dimensions)을 튜플 형태로 반환합니다. `(행의 개수, 열의 개수)`로 구성됩니다.

    ```python
    # data DataFrame이 로드되어 있다고 가정
    print("--- DataFrame의 차원 (shape) ---")
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
    print("--- 데이터의 기본 구조 (data.info()) ---")
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
    print("--- 데이터의 요약 통계 정보 (data.describe()) ---")
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

## 13. 조건부 데이터 검색

DataFrame에서 특정 조건을 만족하는 데이터를 선택하는 것은 데이터 분석의 핵심 기능 중 하나입니다. Pandas는 불리언 인덱싱(Boolean Indexing)을 통해 강력한 조건부 검색 기능을 제공합니다.

### 13.1. 기본 조건 검색

단일 조건을 사용하여 DataFrame의 행을 필터링할 수 있습니다. 조건식은 각 행에 대해 `True` 또는 `False`를 반환하는 불리언 Series를 생성하며, 이 Series를 사용하여 `True`인 행만 선택합니다.

```python
import pandas as pd

data = pd.read_csv("./data/auto-mpg.csv")

print("--- 실린더 개수가 4개인 데이터 (단일 조건) ---")
# data.cylinders == 4는 각 행의 cylinders 값이 4인지 여부를 True/False로 반환하는 Series를 생성
print(data[data.cylinders == 4])

print("\n--- 연비(mpg)가 27 이상인 데이터 (단일 조건) ---")
print(data[data.mpg >= 27])

print("\n--- 모델 연도(model-year)가 70년인 데이터 (단일 조건) ---")
print(data[data["model-year"] == 70])
```

### 13.2. 복합 조건 검색

여러 조건을 동시에 만족하거나(AND) 둘 중 하나라도 만족하는(OR) 데이터를 검색할 때는 복합 조건식을 사용합니다. 이때 파이썬의 기본 논리 연산자(`and`, `or`) 대신 비트wise 논리 연산자(`&`, `|`)를 사용해야 하며, 각 조건식은 반드시 괄호로 묶어야 합니다.

**주의사항**:
*   파이썬의 `and`, `or` 연산자는 Series 전체에 대해 불리언 값을 평가할 수 없으므로 사용 불가합니다.
*   NumPy 기반의 Pandas에서는 요소별(element-wise) 논리 연산을 위해 비트wise 연산자(`&` for AND, `|` for OR, `~` for NOT)를 사용해야 합니다.
*   각 개별 조건식은 반드시 괄호 `()`로 묶어야 합니다. 이는 연산자 우선순위 문제로 인해 발생할 수 있는 오류를 방지합니다.

**NumPy 논리 함수 사용 (대안)**:
`numpy.logical_and()`, `numpy.logical_or()` 함수를 사용하여 복합 조건을 구성할 수도 있습니다. 이는 가독성을 높이는 데 도움이 될 수 있습니다.

```python
import numpy as np
import pandas as pd

data = pd.read_csv("./data/auto-mpg.csv")

print("--- 모델 연도가 70년이고 연비가 25 이상인 데이터 (AND 조건) ---")
# (data["model-year"] == 70) & (data["mpg"] >= 25) 와 동일
print(data[np.logical_and(data["model-year"] == 70, data["mpg"] >= 25)])

print("\n--- 모델 연도가 70년이거나 연비가 30 이상인 데이터 (OR 조건) ---")
# (data["model-year"] == 70) | (data["mpg"] >= 30) 와 동일
print(data[np.logical_or(data["model-year"] == 70, data["mpg"] >= 30)])
```

**오류 예시**: 파이썬의 `and`, `or` 연산자를 사용했을 때 발생하는 `ValueError`
```python
# 이렇게 사용하면 에러 발생 (ValueError: The truth value of a Series is ambiguous.)
# data["model-year"]==70 and data["mpg"]>=25]
```
이 오류는 Pandas Series가 단일 True/False 값으로 명확하게 평가될 수 없기 때문에 발생합니다. 각 요소를 개별적으로 평가하려면 비트wise 연산자를 사용해야 합니다.

## 14. 통계 함수 활용

Pandas DataFrame과 Series는 데이터의 특성을 이해하고 요약하는 데 필수적인 다양한 통계 함수를 제공합니다. 이 함수들은 데이터의 중심 경향성, 분산, 분포 등을 파악하는 데 사용됩니다.

### 14.1. DataFrame의 통계 함수들

DataFrame의 각 열은 Series 타입이므로, Series에 적용 가능한 모든 통계 함수를 개별 컬럼에 직접 적용할 수 있습니다. 또한, DataFrame 전체에 적용하여 각 컬럼별 통계량을 얻을 수도 있습니다.

**기본 통계 함수 목록**:

*   `count()`: 유효한(Non-null) 값의 개수
*   `sum()`: 값들의 합계
*   `mean()`: 산술 평균
*   `median()`: 중앙값 (데이터를 정렬했을 때 중간에 위치한 값)
*   `std()`: 표준편차 (Standard Deviation)
*   `var()`: 분산 (Variance)
*   `min()`: 최솟값
*   `max()`: 최댓값
*   `quantile(q)`: 사분위수 (q는 0과 1 사이의 값, 예: `0.25`는 1사분위수)
*   `mode()`: 최빈값 (가장 자주 나타나는 값)
*   `value_counts()`: Series에서 고유한 값들의 빈도수를 Series 형태로 반환
*   `corr()`: DataFrame의 컬럼 간 상관계수 행렬 계산
*   `cov()`: DataFrame의 컬럼 간 공분산 행렬 계산

### 14.2. 통계 함수 활용 예제

`auto-mpg.csv` 데이터셋을 사용하여 다양한 통계 함수를 적용하는 예제입니다.

```python
import pandas as pd

data = pd.read_csv("./data/auto-mpg.csv")

print("--- 모델 연도별 고유 개수 (value_counts()) ---")
# "model-year" 컬럼의 각 연도별 차량 개수 (빈도수)를 계산
print(data["model-year"].value_counts())

print("\n--- \"mpg\" (연비) 컬럼의 기본 통계량 ---")
print(f"연비 평균: {data["mpg"].mean():.2f}") # 소수점 둘째 자리까지 포맷팅
print(f"연비 최댓값: {data["mpg"].max():.2f}")
print(f"연비 최솟값: {data["mpg"].min():.2f}")
print(f"연비 중간값: {data["mpg"].median():.2f}")
print(f"연비 분산: {data["mpg"].var():.2f}")
print(f"연비 표준편차: {data["mpg"].std():.2f}")

print("\n--- \"mpg\" (연비) 컬럼의 사분위수 (quantile()) ---")
print(f"1사분위수 (Q1): {data["mpg"].quantile(0.25):.2f}")
print(f"2사분위수 (Q2, 중앙값): {data["mpg"].quantile(0.5):.2f}")
print(f"3사분위수 (Q3): {data["mpg"].quantile(0.75):.2f}")
```

### 14.3. 주요 통계 개념 설명

데이터 분석에서 자주 사용되는 통계 개념들을 이해하는 것은 Pandas 함수를 효과적으로 활용하는 데 도움이 됩니다.

1.  **표준편차 (Standard Deviation)**:
    *   데이터가 평균으로부터 얼마나 떨어져 분포하는지를 나타내는 척도입니다.
    *   값이 클수록 데이터가 평균에서 넓게 흩어져 분포하고, 값이 작을수록 데이터가 평균 주변에 밀집하여 모여 있음을 의미합니다.
    *   데이터의 변동성(variability)을 측정하는 데 사용됩니다.

2.  **분산 (Variance)**:
    *   데이터가 평균으로부터 얼마나 흩어져 있는지를 나타내는 척도로, 표준편차의 제곱 값입니다.
    *   표준편차와 유사하게 데이터의 퍼짐 정도를 나타내지만, 단위가 제곱이므로 해석이 직관적이지 않을 수 있습니다.

3.  **사분위수 (Quantile)**:
    *   데이터를 크기 순으로 정렬했을 때, 전체 데이터를 4등분하는 위치에 있는 값들을 의미합니다.
    *   **1사분위수 (Q1, 25%)**: 전체 데이터의 하위 25% 지점에 해당하는 값.
    *   **2사분위수 (Q2, 50%, 중앙값)**: 전체 데이터의 중간 지점에 해당하는 값. 중앙값은 극단적인 값(이상치)에 덜 민감하여 데이터의 중심을 나타내는 데 평균보다 더 견고할 수 있습니다.
    *   **3사분위수 (Q3, 75%)**: 전체 데이터의 상위 25% 지점에 해당하는 값.
    *   사분위수를 통해 데이터의 분포 형태와 이상치 여부를 파악하는 데 도움을 받을 수 있습니다. (예: IQR = Q3 - Q1)

4.  **중간값 (Median)**:
    *   데이터를 크기 순으로 정렬했을 때 정확히 중간에 위치한 값입니다.
    *   평균값과 달리 극단적인 값(이상치)의 영향을 덜 받으므로, 데이터 분포가 비대칭이거나 이상치가 존재할 때 데이터의 중심을 나타내는 데 더 적합할 수 있습니다.

## 15. 실전 예제: Iris 데이터셋 분석

Iris 데이터셋은 머신러닝 및 통계학에서 분류(Classification) 문제의 예시로 자주 사용되는 유명한 데이터셋입니다. 붓꽃의 세 가지 종(Setosa, Versicolor, Virginica)에 대한 꽃잎(petal)과 꽃받침(sepal)의 길이 및 너비 정보를 포함하고 있습니다. 이 예제를 통해 Pandas의 다양한 기능을 활용하여 실제 데이터셋을 탐색하고 분석하는 방법을 익혀보겠습니다.

**`data/iris.csv` 파일 구조 (예시)**:
```csv
sepal.length,sepal.width,petal.length,petal.width,variety
5.1,3.5,1.4,0.2,Setosa
4.9,3.0,1.4,0.2,Setosa
...
6.3,3.3,6.0,2.5,Virginica
```

**문제**: `data/iris.csv` 파일을 읽어서 다음 작업을 수행하세요.

1.  Iris 데이터셋의 필드(컬럼) 개수와 각 필드의 타입 확인
2.  맨 앞의 데이터 7개 출력
3.  Iris 데이터셋의 통계량 요약 정보 확인
4.  `variety`가 'Setosa'인 데이터의 통계량 출력
5.  각 `variety`별 `sepal.length` 값의 평균값 출력
6.  꽃의 종류가 'Setosa'이면서 `sepal.length`가 5cm 이상인 데이터 개수 출력

**해답**:

```python
import pandas as pd
import numpy as np

# 데이터 로드: data/iris.csv 파일을 DataFrame으로 불러오기
data = pd.read_csv("./data/iris.csv")

# 1) 필드 정보 확인: data.info()를 사용하여 컬럼 정보, Non-null 개수, 데이터 타입 확인
print("=== 1. 데이터셋 필드 정보 (data.info()) ===")
data.info()

# 2) 앞의 7개 데이터 출력: data.head(7)을 사용하여 상위 7개 행 출력
print("\n=== 2. 앞의 7개 데이터 (data.head(7)) ===")
print(data.head(7))

# 3) 통계량 요약 정보: data.describe()를 사용하여 숫자형 컬럼의 기술 통계 확인
print("\n=== 3. 통계량 요약 (data.describe()) ===")
print(data.describe())

# 4) variety가 'Setosa'인 데이터의 통계량 출력: 조건부 필터링 후 describe() 적용
print("\n=== 4. 'Setosa' 데이터 통계량 ===")
setosa_data = data[data["variety"] == 'Setosa']
print(setosa_data.describe())

# 5) 각 variety별 sepal.length 평균: groupby()와 mean()을 활용하여 그룹별 평균 계산
print("\n=== 5. 각 variety별 sepal.length 평균 ===")
# 방법 1: 각 종별로 필터링하여 평균 계산
setosa_avg = data[data["variety"] == 'Setosa']["sepal.length"].mean()
print(f"Setosa 평균 sepal.length: {setosa_avg:.2f}")

versicolor_avg = data[data["variety"] == 'Versicolor']["sepal.length"].mean()
print(f"Versicolor 평균 sepal.length: {versicolor_avg:.2f}")

virginica_avg = data[data["variety"] == 'Virginica']["sepal.length"].mean()
print(f"Virginica 평균 sepal.length: {virginica_avg:.2f}")

# 방법 2 (권장): groupby()를 사용하여 더 효율적으로 계산
print("\n--- groupby()를 이용한 각 variety별 sepal.length 평균 ---")
print(data.groupby('variety')["sepal.length"].mean())

# 6) Setosa이면서 sepal.length >= 5인 데이터 개수: 복합 조건 필터링 및 len() 사용
print("\n=== 6. 'Setosa'이면서 sepal.length >= 5인 데이터 ===")
condition_data = data[np.logical_and(data["variety"] == 'Setosa', 
                                   data["sepal.length"] >= 5)]
# 또는 비트wise 연산자 사용: data[(data["variety"] == 'Setosa') & (data["sepal.length"] >= 5)]
print(f"조건을 만족하는 데이터 개수: {len(condition_data)}개")
print(condition_data)
```
## 16. 고급 데이터 조작 및 분석

Pandas는 단순한 데이터 접근 및 수정 외에도 복잡한 데이터 분석 작업을 위한 강력한 고급 기능을 제공합니다. 이 기능들은 데이터를 다양한 관점에서 탐색하고, 의미 있는 패턴을 발견하며, 머신러닝 모델에 적합한 형태로 데이터를 가공하는 데 필수적입니다.

### 16.1. 데이터 그룹화 (Groupby)

`groupby()` 메서드는 SQL의 `GROUP BY` 절과 유사하게, 하나 이상의 컬럼을 기준으로 데이터를 그룹화하고 각 그룹에 대해 집계(aggregation), 변환(transformation), 필터링(filtration) 등의 연산을 수행할 수 있게 합니다. 이는 범주형 데이터에 대한 통계 분석이나 특정 그룹의 특성을 파악하는 데 매우 유용합니다.

**기본 사용법**: `df.groupby('컬럼명').집계함수()`

```python
import pandas as pd

# 예시 데이터프레임 생성
data = {
    'City': ['Seoul', 'Seoul', 'Busan', 'Busan', 'Seoul', 'Busan'],
    'Product': ['A', 'B', 'A', 'C', 'A', 'B'],
    'Sales': [100, 150, 200, 120, 180, 90]
}
df = pd.DataFrame(data)
print("원본 DataFrame:\n", df)

# 'City'별 'Sales'의 평균 계산
print("\n도시별 판매액 평균:\n", df.groupby('City')['Sales'].mean())

# 여러 컬럼으로 그룹화하고 여러 집계 함수 적용
print("\n도시 및 제품별 판매액 합계와 평균:\n", 
      df.groupby(['City', 'Product'])['Sales'].agg(['sum', 'mean']))

# 그룹화된 결과에 대한 추가 연산 (예: reset_index()로 DataFrame으로 변환)
grouped_df = df.groupby('City')['Sales'].sum().reset_index()
print("\n도시별 판매액 합계 (DataFrame 형태):\n", grouped_df)
```

### 16.2. 데이터 병합 (Merge & Join)

`merge()` 함수는 여러 DataFrame을 공통 컬럼(키)을 기준으로 결합하는 데 사용됩니다. SQL의 JOIN 연산과 유사하며, 서로 다른 데이터 소스에서 얻은 정보를 통합할 때 필수적입니다.

**주요 `merge()` 옵션**:
- `on`: 병합의 기준이 되는 컬럼명 (양쪽 DataFrame에 모두 존재해야 함)
- `left_on`, `right_on`: 왼쪽/오른쪽 DataFrame에서 병합의 기준이 되는 컬럼명
- `how`: 병합 방식 (SQL의 JOIN 타입과 유사)
  - `'inner'` (기본값): 양쪽 DataFrame에 모두 존재하는 키만 포함
  - `'left'`: 왼쪽 DataFrame의 모든 키를 포함하고, 오른쪽 DataFrame에서 일치하는 키가 없으면 `NaN`으로 채움
  - `'right'`: 오른쪽 DataFrame의 모든 키를 포함하고, 왼쪽 DataFrame에서 일치하는 키가 없으면 `NaN`으로 채움
  - `'outer'`: 양쪽 DataFrame의 모든 키를 포함하고, 일치하지 않는 부분은 `NaN`으로 채움

```python
import pandas as pd

# 예시 DataFrame 생성
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David']
})

df2 = pd.DataFrame({
    'ID': [1, 2, 5],
    'Score': [90, 85, 95]
})

print("df1:\n", df1)
print("\ndf2:\n", df2)

# Inner Join (기본값): ID가 일치하는 행만 병합
merged_inner = pd.merge(df1, df2, on='ID', how='inner')
print("\nInner Join 결과:\n", merged_inner)

# Left Join: df1의 모든 행을 유지하고 df2에서 일치하는 정보 병합
merged_left = pd.merge(df1, df2, on='ID', how='left')
print("\nLeft Join 결과:\n", merged_left)

# Outer Join: 양쪽 DataFrame의 모든 ID를 포함하여 병합
merged_outer = pd.merge(df1, df2, on='ID', how='outer')
print("\nOuter Join 결과:\n", merged_outer)
```

### 16.3. 피벗 테이블 (Pivot Table)

`pivot_table()` 함수는 데이터를 재구성하여 요약 통계를 생성하는 강력한 도구입니다. 스프레드시트의 피벗 테이블과 유사하게, 하나 이상의 컬럼을 인덱스(행), 다른 컬럼을 컬럼(열)으로 사용하여 데이터를 집계합니다. 이는 복잡한 데이터 요약 및 분석에 매우 효과적입니다.

**주요 `pivot_table()` 옵션**:
- `values`: 집계할 값 컬럼
- `index`: 행 인덱스로 사용할 컬럼
- `columns`: 열 컬럼으로 사용할 컬럼
- `aggfunc`: 집계 함수 (기본값은 `mean`). `sum`, `count`, `median`, `np.sum` 등 다양한 함수 사용 가능
- `fill_value`: `NaN` 값을 채울 값

```python
import pandas as pd
import numpy as np

# 예시 데이터프레임 생성
df = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-01'],
    'Region': ['East', 'West', 'East', 'West', 'East'],
    'Product': ['A', 'B', 'A', 'C', 'C'],
    'Sales': [100, 150, 200, 120, 180]
})

print("원본 DataFrame:\n", df)

# 날짜별, 지역별 판매액 합계 피벗 테이블
pivot_sales = df.pivot_table(values='Sales', index='Date', columns='Region', aggfunc='sum')
print("\n날짜별, 지역별 판매액 합계:\n", pivot_sales)

# 날짜별, 제품별 판매액 평균 (NaN은 0으로 채움)
pivot_avg_sales = df.pivot_table(values='Sales', index='Date', columns='Product', 
                                 aggfunc='mean', fill_value=0)
print("\n날짜별, 제품별 판매액 평균 (NaN 0으로 채움):\n", pivot_avg_sales)
```

### 16.4. 데이터 변형 (Melt)

`melt()` 함수는 DataFrame을 '넓은' 형식(wide format)에서 '긴' 형식(long format)으로 변형하는 데 사용됩니다. 이는 특히 시각화 라이브러리(예: Seaborn)에서 여러 변수를 하나의 컬럼으로 모아 플로팅할 때 유용합니다. 즉, 컬럼을 행으로 '녹이는' 작업입니다.

```python
import pandas as pd

# 예시 데이터프레임 (넓은 형식)
df_wide = pd.DataFrame({
    'ID': [1, 2],
    'Math': [90, 85],
    'Science': [70, 95],
    'English': [80, 75]
})

print("원본 (넓은 형식) DataFrame:\n", df_wide)

# 'ID'를 고정하고 'Math', 'Science', 'English'를 행으로 변형
df_long = df_wide.melt(id_vars=['ID'], var_name='Subject', value_name='Score')
print("\n변형된 (긴 형식) DataFrame:\n", df_long)
```

### 16.5. 시계열 데이터 처리 (Time Series)

Pandas는 시계열 데이터(시간 순서에 따라 기록된 데이터)를 다루는 데 매우 강력한 기능을 제공합니다. 날짜/시간 인덱싱, 리샘플링, 시간대 처리 등이 포함됩니다.

```python
import pandas as pd
import numpy as np

# 날짜 범위 생성 및 시계열 Series 생성
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)
print("원본 시계열 Series (일부):\n", ts.head())

# 특정 날짜 범위 선택
print("\n2023년 1월 데이터:\n", ts['2023-01'].head())

# 리샘플링: 일별 데이터를 주별 평균으로 변환
weekly_mean = ts.resample('W').mean()
print("\n주별 평균 (일부):\n", weekly_mean.head())

# 이동 평균 (Moving Average) 계산
moving_avg = ts.rolling(window=7).mean()
print("\n7일 이동 평균 (일부):\n", moving_avg.head(10))
```

### 16.6. 성능 최적화 팁

대규모 데이터를 다룰 때 Pandas의 성능을 최적화하는 것은 매우 중요합니다. 비효율적인 코드는 메모리 부족이나 긴 처리 시간으로 이어질 수 있습니다.

#### 1. 벡터화된 연산 활용
`for` 루프 대신 Pandas의 내장 함수나 연산자를 사용합니다. 이는 C/Cython으로 구현되어 있어 훨씬 빠릅니다.

```python
import pandas as pd
import numpy as np
import time

size = 10**6
df_perf = pd.DataFrame({
    'A': np.random.rand(size),
    'B': np.random.rand(size)
})

# 비효율적인 for 루프
start_time = time.time()
result_list = []
for i in range(len(df_perf)):
    result_list.append(df_perf['A'][i] + df_perf['B'][i])
end_time = time.time()
print(f"For 루프 시간: {end_time - start_time:.4f} 초")

# 벡터화된 Pandas 연산
start_time = time.time()
result_pandas = df_perf['A'] + df_perf['B']
end_time = time.time()
print(f"Pandas 벡터화 시간: {end_time - start_time:.4f} 초")
```

#### 2. `apply()` 대신 벡터화된 함수 사용
`apply()`는 행/열 단위로 파이썬 함수를 적용하므로 느릴 수 있습니다. 가능한 경우 NumPy나 Pandas의 벡터화된 함수를 사용합니다.

```python
# 비효율적인 apply()
# df_perf['C'] = df_perf.apply(lambda row: row['A'] * 2 + row['B'] * 3, axis=1)

# 효율적인 벡터화된 연산
df_perf['C'] = df_perf['A'] * 2 + df_perf['B'] * 3
```

#### 3. 적절한 데이터 타입 (dtype) 사용
메모리 사용량을 줄이고 연산 속도를 높입니다. 예를 들어, 작은 정수 범위의 컬럼에는 `int8`, `int16` 등을 사용합니다.

```python
df_perf['A'] = df_perf['A'].astype('float32')  # 메모리 절약
```

#### 4. `inplace=True` 사용 주의
`inplace=True`는 원본 DataFrame을 직접 수정하므로 메모리 복사를 피할 수 있지만, 체이닝(chaining)을 방해하고 예상치 못한 부작용을 일으킬 수 있습니다. 일반적으로 새로운 DataFrame을 반환받는 것이 더 안전하고 가독성이 좋습니다.

#### 5. `read_csv` 옵션 활용
대규모 CSV 파일을 읽을 때 `dtype`, `usecols`, `chunksize` 등의 옵션을 사용하여 메모리 사용량을 최적화하고 로딩 속도를 높일 수 있습니다.

```python
# data = pd.read_csv('large_data.csv', 
#                   dtype={'col1': 'int32', 'col2': 'float16'}, 
#                   usecols=['col1', 'col2'])
```

#### 6. `Categorical` 타입 활용
고유한 값이 적은 문자열 컬럼의 경우 `Categorical` 타입으로 변환하면 메모리 사용량을 크게 줄일 수 있습니다.

```python
# df['Product'] = df['Product'].astype('category')
```

---

[⏮️ 이전 문서](./0701_ML정리.md) | [다음 문서 ⏭️](./0703_ML정리.md)