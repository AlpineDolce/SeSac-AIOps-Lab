<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Pandas의 핵심 2차원 데이터 구조인 `DataFrame`의 개념, 특징, 그리고 다양한 생성 방법을 상세히 다룹니다. 또한, `DataFrame` 데이터에 컬럼 선택, `iloc`, `loc` 함수를 이용한 접근, 조건부 필터링, 컬럼 추가/수정/삭제, 행 추가/삭제 등 기본적인 데이터 조작 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. DataFrame (2차원 데이터)](#1-dataframe-2차원-데이터)
  - [1.1. DataFrame의 개념 및 특징](#11-dataframe의-개념-및-특징)
  - [1.2. DataFrame 생성 방법](#12-dataframe-생성-방법)
  - [1.3. DataFrame 데이터 접근 및 조작](#13-dataframe-데이터-접근-및-조작)

---

## 1. DataFrame (2차원 데이터)

### 1.1. DataFrame의 개념 및 특징

1.  **정의**: DataFrame은 Pandas의 핵심 2차원 데이터 구조로, **행(row)과 열(column)로 이루어진 테이블 형태**를 가집니다. 관계형 데이터베이스의 테이블, Excel 스프레드시트, 또는 CSV 파일과 매우 유사합니다.
2.  **구조**: 각 열은 Series 객체로 볼 수 있으며, 서로 다른 데이터 타입을 가질 수 있습니다. 이는 각 컬럼이 독립적인 데이터 유형을 가질 수 있음을 의미합니다 (예: 이름은 문자열, 나이는 정수, 도시는 문자열).
3.  **활용**: 대부분의 정형 데이터 분석 작업은 DataFrame을 중심으로 이루어지며, 다양한 데이터 소스(CSV, Excel, SQL 등)의 데이터를 DataFrame으로 쉽게 불러오고 저장할 수 있습니다.

### 1.2. DataFrame 생성 방법

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

### 1.3. DataFrame 데이터 접근 및 조작

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
    print("\nDataFrame 통계 요약 (df.describe()):\n", df.describe())
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
