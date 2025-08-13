<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 CSV(Comma-Separated Values) 파일의 특징을 이해하고, Pandas의 `read_csv()` 함수를 사용하여 CSV 파일을 `DataFrame`으로 불러오는 다양한 방법을 다룹니다. `read_csv()`의 주요 옵션들을 학습하고, `to_csv()`를 이용한 `DataFrame` 저장 방법을 실제 코드 예제를 통해 익힙니다.

<h2>목차</h2>

- [1. CSV 파일 처리](#1-csv-파일-처리)
  - [1.1. CSV 파일의 특징](#11-csv-파일의-특징)
  - [1.2. CSV 파일 읽기](#12-csv-파일-읽기)
    - [1.2.1. `read_csv()` 함수 개요](#121-read_csv-함수-개요)
    - [1.2.2. 주요 `read_csv()` 옵션 상세](#122-주요-read_csv-옵션-상세)
  - [1.3. `read_csv()` 활용 예시](#13-read_csv-활용-예시)
    - [1.3.1. 기본 CSV 파일 읽기](#131-기본-csv-파일-읽기)
    - [1.3.2. 제목 줄이 없는 CSV 파일 처리 (`header=None`, `names`)](#132-제목-줄이-없는-csv-파일-처리-headernone-names)
    - [1.3.3. 제목 줄이 특정 위치에 있는 경우 (`header`)](#133-제목-줄이-특정-위치에-있는-경우-header)
    - [1.3.4. 특정 컬럼만 불러오기 (`usecols`)](#134-특정-컬럼만-불러오기-usecols)
    - [1.3.5. 데이터 타입 지정 (`dtype`)](#135-데이터-타입-지정-dtype)
    - [1.3.6. 날짜 컬럼 파싱 (`parse_dates`)](#136-날짜-컬럼-파싱-parse_dates)
    - [1.3.7. 인덱스 컬럼 지정 (`index_col`)](#137-인덱스-컬럼-지정-index_col)
    - [1.3.8. 구분자 변경 (`sep`)](#138-구분자-변경-sep)
    - [1.3.9. 특정 행 건너뛰기 (`skiprows`, `nrows`)](#139-특정-행-건너뛰기-skiprows-nrows)
    - [1.3.10. 인코딩 문제 해결 (`encoding`)](#1310-인코딩-문제-해결-encoding)
  - [1.4. CSV 파일 저장](#14-csv-파일-저장)
    - [1.4.1. `to_csv()` 함수 개요](#141-to_csv-함수-개요)
    - [1.4.2. 주요 `to_csv()` 옵션 상세](#142-주요-to_csv-옵션-상세)
    - [1.4.3. `to_csv()` 활용 예시](#143-to_csv-활용-예시)

---

## 1. CSV 파일 처리

CSV (Comma-Separated Values) 파일은 데이터를 쉼표(`,`)로 구분하여 저장하는 텍스트 파일 형식입니다. 가장 보편적으로 사용되는 데이터 교환 형식 중 하나입니다.

### 1.1. CSV 파일의 특징

CSV 파일은 그 단순함과 범용성 덕분에 데이터 교환의 사실상 표준으로 자리 잡았습니다.

*   **단순성 및 텍스트 기반**: 데이터를 쉼표(`,`)로 구분하는 일반 텍스트 파일입니다. 특정 프로그램 없이 메모장과 같은 일반 텍스트 에디터로도 내용을 확인하고 작성할 수 있어 사람이 읽고 이해하기 쉽습니다.
*   **간편한 편집**: Excel과 같은 스프레드시트 프로그램에서도 쉽게 열고 편집할 수 있습니다.
*   **범용 호환성**: 대부분의 데이터 분석 도구, 프로그래밍 언어, 데이터베이스 시스템에서 CSV 파일을 지원하므로, 다양한 시스템 간의 데이터 연동 및 교환에 매우 용이합니다.

**CSV 파일의 제한 사항:**
*   **데이터 타입 부재**: 모든 데이터가 텍스트로 저장되므로, 숫자, 날짜, 불리언 등 원래의 데이터 타입을 유지하지 못합니다. 읽어올 때 타입을 추론하거나 명시적으로 지정해야 합니다.
*   **복잡한 구조 표현 불가**: 계층적인 데이터나 중첩된 구조(예: JSON, XML)를 직접 표현할 수 없습니다.
*   **구분자 문제**: 데이터 내에 쉼표가 포함되어 있을 경우, 따옴표(`"`)로 필드를 감싸서 해결하지만, 데이터 자체에 따옴표가 포함되면 복잡성이 증가합니다.
*   **인코딩 표준 부재**: 공식적인 인코딩 표준이 없어, 파일을 생성한 시스템과 읽는 시스템의 기본 인코딩이 다를 경우(예: Windows의 `cp949`와 macOS/Linux의 `utf-8`) 글자가 깨지는 문제가 발생할 수 있습니다.

### 1.2. CSV 파일 읽기

#### 1.2.1. `read_csv()` 함수 개요
Pandas의 `read_csv()` 함수는 CSV 파일을 읽어 `DataFrame` 객체로 변환하는 핵심적인 기능을 제공합니다. 다양한 옵션을 통해 복잡한 형태의 CSV 파일도 유연하게 처리할 수 있습니다.

#### 1.2.2. 주요 `read_csv()` 옵션 상세

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `filepath_or_buffer` | 파일 경로, URL, 또는 파일과 유사한 객체(e.g., `StringIO`). | - |
| `sep` | 데이터 필드를 구분하는 구분자. | `,` |
| `header` | 제목(컬럼명)으로 사용할 행의 번호. `None`으로 지정 시 컬럼명이 없다고 간주. | `0` |
| `names` | `header=None`일 때 사용할 컬럼명 리스트. | `None` |
| `index_col` | 인덱스로 사용할 컬럼의 번호나 이름. | `None` |
| `usecols` | 불러올 컬럼만 리스트나 함수로 지정. 메모리 효율성을 높일 수 있음. | `None` |
| `dtype` | 컬럼별 데이터 타입을 딕셔너리 형태로 지정. | `None` |
| `parse_dates` | 날짜/시간으로 파싱할 컬럼의 리스트. `True`로 설정 시 날짜 형식 추론. | `False` |
| `skiprows` | 파일 시작 부분에서 건너뛸 행의 개수 또는 리스트. | `None` |
| `nrows` | 불러올 행의 개수. 대용량 파일의 일부만 확인할 때 유용. | `None` |
| `na_values` | `NaN`으로 처리할 값들의 리스트. | `['', '#N/A', ...]` |
| `encoding` | 파일 인코딩 형식 지정. 한글이 깨질 때 `cp949`나 `utf-8-sig` 시도. | `utf-8` |

### 1.3. `read_csv()` 활용 예시
```python
import pandas as pd
import io
```

#### 1.3.1. 기본 CSV 파일 읽기
가장 일반적인 형태로, 첫 번째 행에 헤더가 있고 쉼표로 구분된 CSV 파일을 읽습니다.

**샘플 데이터**
```csv
name,age,city
Alice,25,New York
Bob,30,London
Charlie,35,Paris
```

**코드**
```python
csv_data = """name,age,city
Alice,25,New York
Bob,30,London
Charlie,35,Paris
"""

df = pd.read_csv(io.StringIO(csv_data))
print(df)
```

**결과**
```
      name  age       city
0    Alice   25   New York
1      Bob   30     London
2  Charlie   35      Paris
```

#### 1.3.2. 제목 줄이 없는 CSV 파일 처리 (`header=None`, `names`)
파일에 컬럼명이 없는 경우, `header=None`으로 지정하면 Pandas가 자동으로 0부터 시작하는 정수형 컬럼명을 부여합니다. `names` 옵션으로 직접 컬럼명을 지정할 수 있습니다.

**샘플 데이터**
```csv
Alice,25,New York
Bob,30,London
Charlie,35,Paris
```

**코드**
```python
csv_data = """Alice,25,New York
Bob,30,London
Charlie,35,Paris
"""

df = pd.read_csv(io.StringIO(csv_data), header=None, names=['사용자명', '나이', '도시'])
print(df)
```

**결과**
```
    사용자명  나이        도시
0    Alice   25  New York
1      Bob   30    London
2  Charlie   35     Paris
```

#### 1.3.3. 제목 줄이 특정 위치에 있는 경우 (`header`)
파일 상단에 불필요한 정보가 있고, 실제 헤더가 다른 행에 위치할 때 `header` 옵션에 행 번호(0부터 시작)를 지정합니다.

**샘플 데이터**
```csv
# 데이터 정보: 사용자 리스트
# 생성일: 2025-07-01
name,age,city
Alice,25,New York
Bob,30,London
```

**코드**
```python
csv_data = """# 데이터 정보: 사용자 리스트
# 생성일: 2025-07-01
name,age,city
Alice,25,New York
Bob,30,London
"""

df = pd.read_csv(io.StringIO(csv_data), header=2)
print(df)
```

**결과**
```
    name  age       city
0  Alice   25   New York
1    Bob   30     London
```

#### 1.3.4. 특정 컬럼만 불러오기 (`usecols`)
메모리를 절약하거나 분석에 필요한 컬럼만 선택적으로 불러올 때 사용합니다.

**샘플 데이터**
```csv
id,name,age,city,country
1,Alice,25,New York,USA
2,Bob,30,London,UK
3,Charlie,35,Paris,France
```

**코드**
```python
csv_data = """id,name,age,city,country
1,Alice,25,New York,USA
2,Bob,30,London,UK
3,Charlie,35,Paris,France
"""

df = pd.read_csv(io.StringIO(csv_data), usecols=['name', 'city'])
print(df)
```

**결과**
```
      name       city
0    Alice   New York
1      Bob     London
2  Charlie      Paris
```

#### 1.3.5. 데이터 타입 지정 (`dtype`)
Pandas는 자동으로 데이터 타입을 추론하지만, 때로는 명시적으로 지정해야 합니다. 예를 들어, ID와 같이 숫자 형태이지만 문자열로 다루어야 하는 경우에 유용합니다.

**샘플 데이터**
```csv
id,value
001,10.5
002,20.3
003,30.8
```

**코드**
```python
csv_data = """id,value
001,10.5
002,20.3
003,30.8
"""

df = pd.read_csv(io.StringIO(csv_data), dtype={'id': str, 'value': float})
print(df.info())
```

**결과**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   id      3 non-null      object 
 1   value   3 non-null      float64
dtypes: float64(1), object(1)
memory usage: 176.0+ bytes
None
```

#### 1.3.6. 날짜 컬럼 파싱 (`parse_dates`)
문자열로 저장된 날짜 데이터를 `datetime` 객체로 변환하여 시간 관련 분석을 용이하게 합니다.

**샘플 데이터**
```csv
date,event
2025-01-01,New Year
2025-05-05,Children's Day
2025-12-25,Christmas
```

**코드**
```python
csv_data = """date,event
2025-01-01,New Year
2025-05-05,Children's Day
2025-12-25,Christmas
"""

df = pd.read_csv(io.StringIO(csv_data), parse_dates=['date'])
print(df.info())
```

**결과**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype         
---  ------  --------------  -----         
 0   date    3 non-null      datetime64[ns]
 1   event   3 non-null      object        
dtypes: datetime64[ns](1), object(1)
memory usage: 176.0+ bytes
None
```

#### 1.3.7. 인덱스 컬럼 지정 (`index_col`)
특정 컬럼을 `DataFrame`의 인덱스로 사용하고 싶을 때 지정합니다.

**샘플 데이터**
```csv
id,name,score
S01,Alice,95
S02,Bob,88
S03,Charlie,76
```

**코드**
```python
csv_data = """id,name,score
S01,Alice,95
S02,Bob,88
S03,Charlie,76
"""

df = pd.read_csv(io.StringIO(csv_data), index_col='id')
print(df)
```

**결과**
```
       name  score
id                
S01   Alice     95
S02     Bob     88
S03 Charlie     76
```

#### 1.3.8. 구분자 변경 (`sep`)
쉼표(`,`)가 아닌 다른 구분자(세미콜론 `;`, 탭 `	` 등)로 필드가 구분된 파일을 읽을 때 사용합니다.

**샘플 데이터**
```csv
name;age;city
Alice;25;New York
Bob;30;London
```

**코드**
```python
csv_data = """name;age;city
Alice;25;New York
Bob;30;London
"""

df = pd.read_csv(io.StringIO(csv_data), sep=';')
print(df)
```

**결과**
```
    name  age       city
0  Alice   25   New York
1    Bob   30     London
```

#### 1.3.9. 특정 행 건너뛰기 (`skiprows`, `nrows`)
`skiprows`는 파일 상단의 특정 행들을 건너뛸 때, `nrows`는 파일에서 읽어올 행의 수를 제한할 때 사용합니다. 대용량 파일의 구조를 파악하거나 샘플링할 때 유용합니다.

**샘플 데이터**
```csv
# 사용자 데이터 파일
# 최종 수정: 2025-07-01
name,age,city
Alice,25,New York
Bob,30,London
Charlie,35,Paris
David,40,Tokyo
```

**코드**
```python
csv_data = """# 사용자 데이터 파일
# 최종 수정: 2025-07-01
name,age,city
Alice,25,New York
Bob,30,London
Charlie,35,Paris
David,40,Tokyo
"""

# 상위 2줄은 건너뛰고, 데이터는 2개만 읽기
df = pd.read_csv(io.StringIO(csv_data), skiprows=2, nrows=2)
print(df)
```

**결과**
```
    name  age       city
0  Alice   25   New York
1    Bob   30     London
```

#### 1.3.10. 인코딩 문제 해결 (`encoding`)
한글이 포함된 CSV 파일을 Windows 환경에서 생성한 경우, `cp949` 인코딩을 사용하는 경우가 많습니다. 이 파일을 다른 시스템에서 `utf-8`로 읽으면 글자가 깨지므로, `encoding` 옵션을 명시해야 합니다.

**샘플 데이터 (cp949로 저장되었다고 가정)**
```csv
이름,나이,도시
홍길동,30,서울
이순신,45,부산
```

**코드**
```python
# 아래 코드는 실제 cp949 인코딩된 파일을 읽는 예시입니다.
# with open('user_cp949.csv', 'r', encoding='cp949') as f:
#     df = pd.read_csv(f)

# 여기서는 시뮬레이션을 위해 utf-8로 데이터를 만듭니다.
csv_data = """이름,나이,도시
홍길동,30,서울
이순신,45,부산
"""

# 만약 파일이 cp949로 저장되어 있다면 아래와 같이 읽어야 합니다.
# df = pd.read_csv('path/to/your/file.csv', encoding='cp949')

# utf-8 파일은 그냥 읽으면 됩니다.
df = pd.read_csv(io.StringIO(csv_data))
print(df)
```

**결과**
```
    이름  나이  도시
0  홍길동  30  서울
1  이순신  45  부산
```

### 1.4. CSV 파일 저장

#### 1.4.1. `to_csv()` 함수 개요
`to_csv()` 함수는 `DataFrame` 객체를 CSV 파일로 저장하는 기능을 합니다. 다양한 옵션을 통해 저장 형식을 제어할 수 있습니다.

#### 1.4.2. 주요 `to_csv()` 옵션 상세

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `path_or_buf` | 저장할 파일 경로 또는 버퍼. | - |
| `sep` | 사용할 필드 구분자. | `,` |
| `na_rep` | `NaN` 값을 대체할 문자열. | `''` (빈 문자열) |
| `columns` | 저장할 컬럼을 리스트로 지정. | `None` (모든 컬럼) |
| `header` | 컬럼명을 파일에 쓸지 여부. | `True` |
| `index` | 인덱스를 파일에 쓸지 여부. | `True` |
| `index_label` | 인덱스 컬럼의 라벨(이름) 지정. | `None` |
| `mode` | 파일 쓰기 모드. `w` (덮어쓰기), `a` (이어쓰기). | `w` |
| `encoding` | 파일 인코딩 형식 지정. 한글 포함 시 `utf-8-sig` 권장. | `utf-8` |

#### 1.4.3. `to_csv()` 활용 예시

**샘플 DataFrame 생성**
```python
df = pd.DataFrame({
    '이름': ['Alice', 'Bob', None],
    '나이': [25, 30, 22],
    '도시': ['서울', '부산', '대구']
}, index=['S01', 'S02', 'S03'])
df.index.name = 'ID'
```

**1. 기본 저장 (인덱스 포함)**
```python
# to_csv()는 파일 경로를 받지만, 여기서는 결과를 문자열로 보기 위해 버퍼를 사용합니다.
output_buffer = io.StringIO()
df.to_csv(output_buffer)
print(output_buffer.getvalue())
```
**결과 CSV**
```csv
ID,이름,나이,도시
S01,Alice,25,서울
S02,Bob,30,부산
S03,,22,대구
```

**2. 인덱스 제외하고 저장 (`index=False`)**
데이터 분석 시 인덱스가 불필요한 경우가 많습니다.
```python
output_buffer = io.StringIO()
df.to_csv(output_buffer, index=False)
print(output_buffer.getvalue())
```
**결과 CSV**
```csv
이름,나이,도시
Alice,25,서울
Bob,30,부산
,22,대구
```

**3. 결측값을 특정 문자로 대체 (`na_rep`) 및 한글 인코딩 지정**
결측값(None, NaN)을 'N/A'로 표시하고, Excel에서 바로 열어도 깨지지 않도록 `utf-8-sig`로 인코딩합니다.
```python
output_buffer = io.StringIO()
# 실제 파일 저장 시: df.to_csv('output.csv', index=False, na_rep='N/A', encoding='utf-8-sig')
df.to_csv(output_buffer, index=False, na_rep='N/A') 
print(output_buffer.getvalue())
```
**결과 CSV**
```csv
이름,나이,도시
Alice,25,서울
Bob,30,부산
N/A,22,대구
```

**4. 특정 컬럼만 저장하고 구분자 변경**
'이름'과 '도시' 컬럼만 세미콜론(;)으로 구분하여 저장합니다.
```python
output_buffer = io.StringIO()
df.to_csv(output_buffer, columns=['이름', '도시'], sep=';')
print(output_buffer.getvalue())
```
**결과 CSV**
```csv
ID;이름;도시
S01;Alice;서울
S02;Bob;부산
S03;;대구
```
