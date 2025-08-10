<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 CSV(Comma-Separated Values) 파일의 특징을 이해하고, Pandas의 `read_csv()` 함수를 사용하여 CSV 파일을 `DataFrame`으로 불러오는 다양한 방법을 다룹니다. `header`, `encoding`, `sep`, `index_col`, `names` 등 `read_csv()`의 주요 옵션들을 학습하고, `to_csv()`를 이용한 `DataFrame` 저장 방법을 실제 코드 예제를 통해 익힙니다.

<h2>목차</h2>

- [1. CSV 파일 처리](#1-csv-파일-처리)
  - [1.1. CSV 파일의 특징](#11-csv-파일의-특징)
  - [1.2. CSV 파일 읽기](#12-csv-파일-읽기)
  - [1.3. CSV 파일 읽기 예제](#13-csv-파일-읽기-예제)
    - [1.3.1. 기본 CSV 파일 읽기](#131-기본-csv-파일-읽기)
    - [1.3.2. 제목 줄이 없는 CSV 파일 처리](#132-제목-줄이-없는-csv-파일-처리)
    - [1.3.3. 제목 줄이 특정 위치에 있는 경우](#133-제목-줄이-특정-위치에-있는-경우)
  - [1.4. CSV 파일 저장](#14-csv-파일-저장)

---

## 1. CSV 파일 처리

CSV (Comma-Separated Values) 파일은 데이터를 쉼표(`,`)로 구분하여 저장하는 텍스트 파일 형식입니다. 가장 보편적으로 사용되는 데이터 교환 형식 중 하나입니다.

### 1.1. CSV 파일의 특징

1.  **텍스트 기반**: 데이터를 쉼표(`,`)로 구분하는 일반 텍스트 파일입니다. 특정 프로그램 없이 메모장과 같은 일반 텍스트 에디터로도 내용을 확인하고 작성할 수 있습니다.
2.  **간편한 편집**: Excel과 같은 스프레드시트 프로그램에서도 쉽게 열고 편집할 수 있습니다.
3.  **널리 사용**: 빅데이터 환경에서 데이터를 저장하고 교환하는 데 가장 많이 사용되는 형태 중 하나입니다. 다양한 시스템 간의 데이터 연동에 용이합니다.

### 1.2. CSV 파일 읽기

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
*   `sep` 또는 `delimiter`: 데이터를 구분하는 구분자(separator)를 지정합니다. 기본값은 쉼표(`,`)입니다. 탭으로 구분된 파일(`TSV`)의 경우 `sep='	'`로 지정할 수 있습니다.
*   `index_col`: 특정 컬럼을 DataFrame의 인덱스로 사용할 때 지정합니다.
*   `names`: `header=None`일 때 사용할 컬럼명 리스트를 직접 지정합니다.

### 1.3. CSV 파일 읽기 예제

#### 1.3.1. 기본 CSV 파일 읽기

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

#### 1.3.2. 제목 줄이 없는 CSV 파일 처리

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

#### 1.3.3. 제목 줄이 특정 위치에 있는 경우

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

### 1.4. CSV 파일 저장

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

```