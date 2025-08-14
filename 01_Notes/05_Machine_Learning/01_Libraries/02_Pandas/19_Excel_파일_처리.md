<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 Pandas를 사용하여 Microsoft Excel 파일(.xlsx, .xls)을 직접 읽고 쓰는 방법을 다룹니다. Excel 파일 처리의 장점과 필수 라이브러리를 이해하고, `read_excel()` 함수를 이용한 파일 불러오기, `to_excel()` 메서드를 이용한 `DataFrame` 저장 방법을 실제 코드 예제를 통해 학습합니다. 특히 여러 시트를 다루거나 특정 옵션을 활용하는 실무 예제를 중점적으로 살펴봅니다.

<h2>목차</h2>

- [1. Excel 파일 처리](#1-excel-파일-처리)
  - [1.1. Excel 파일 처리의 장점 및 필수 라이브러리](#11-excel-파일-처리의-장점-및-필수-라이브러리)
    - [1.1.1. Excel 파일 처리의 주요 장점](#111-excel-파일-처리의-주요-장점)
    - [1.1.2. Excel 파일 처리를 위한 필수 라이브러리 (Engines)](#112-excel-파일-처리를-위한-필수-라이브러리-engines)
  - [1.2. Excel 파일 읽기 (`read_excel`)](#12-excel-파일-읽기-read_excel)
    - [1.2.1. 주요 `read_excel` 옵션 상세](#121-주요-read_excel-옵션-상세)
    - [1.2.2. `read_excel` 활용 예시](#122-read_excel-활용-예시)
  - [1.3. Excel 파일 저장 (`to_excel`)](#13-excel-파일-저장-to_excel)
    - [1.3.1. 주요 `to_excel` 옵션](#131-주요-to_excel-옵션)
    - [1.3.2. `to_excel` 활용 예시](#132-to_excel-활용-예시)

---

## 1. Excel 파일 처리

Pandas는 Microsoft Excel 파일(.xlsx, .xls)을 직접 읽고 쓰는 강력한 기능을 제공합니다. 이는 Excel을 주로 사용하는 현업 환경에서 데이터를 가져오거나 분석 결과를 공유할 때 매우 유용합니다.

### 1.1. Excel 파일 처리의 장점 및 필수 라이브러리

Pandas를 사용하여 Microsoft Excel 파일(.xlsx, .xls)을 처리하는 것은 데이터 과학 및 분석 워크플로우에서 매우 중요한 부분입니다. Excel은 비기술적인 사용자들에게도 친숙한 형식이며, 다양한 비즈니스 데이터가 Excel 파일 형태로 유통되기 때문입니다. Pandas는 이러한 Excel 파일을 Python 환경에서 효율적으로 다룰 수 있는 강력한 기능을 제공합니다.

#### 1.1.1. Excel 파일 처리의 주요 장점

*   **Pandas와의 직접적인 통합 및 사용 편의성**:
    *   Pandas는 `read_excel()` 함수와 `DataFrame.to_excel()` 메서드를 내장하고 있어, 별도의 복잡한 설정 없이 Python 코드 내에서 Excel 파일을 직접 읽고 쓸 수 있습니다. 이는 데이터 로딩 및 저장 과정을 간소화하고, Python 기반의 데이터 분석 파이프라인을 구축하는 데 있어 일관된 경험을 제공합니다.
    *   CSV 파일과 유사하게 직관적인 API를 제공하여, Pandas에 익숙한 사용자라면 쉽게 Excel 파일도 다룰 수 있습니다.

*   **복잡한 Excel 데이터 구조 처리 능력**:
    *   **다중 시트 지원**: Excel 파일은 여러 개의 워크시트(sheet)를 포함할 수 있습니다. Pandas는 `sheet_name` 옵션을 통해 특정 시트만 선택적으로 불러오거나, 모든 시트를 한 번에 불러와 딕셔너리 형태로 관리할 수 있도록 지원합니다. 이는 복잡하게 구성된 Excel 통합 문서에서 필요한 데이터만 추출하는 데 매우 유용합니다.
    *   **특정 범위 데이터 로드**: `usecols`, `skiprows`, `nrows` 등의 옵션을 활용하여 Excel 시트 내의 특정 셀 범위에 있는 데이터만 선택적으로 읽어올 수 있습니다. 이는 불필요한 데이터를 로드하지 않아 메모리 효율성을 높이고, 대용량 Excel 파일 처리 시 유연성을 제공합니다.
    *   **다양한 데이터 타입 처리**: Excel 파일은 숫자, 텍스트, 날짜, 시간 등 다양한 데이터 타입을 포함할 수 있습니다. Pandas는 이러한 데이터 타입을 자동으로 추론하여 `DataFrame`으로 변환하며, 필요에 따라 `dtype`이나 `parse_dates` 옵션을 통해 명시적으로 데이터 타입을 지정할 수도 있습니다.

*   **서식 및 메타데이터 유지/적용 가능성**:
    *   CSV 파일이 순수 텍스트 기반으로 서식 정보를 포함하지 않는 것과 달리, Excel 파일은 셀 서식(글꼴, 색상, 테두리), 셀 너비, 틀 고정(freeze panes) 등 다양한 시각적 서식 정보를 포함할 수 있습니다.
    *   Pandas 자체는 복잡한 Excel 스타일링 기능을 직접 제공하지 않지만, `ExcelWriter` 객체와 `openpyxl` 또는 `XlsxWriter`와 같은 백엔드 엔진을 함께 사용하면, 저장 시 이러한 서식 정보를 적용하거나 유지할 수 있습니다. 이는 분석 결과를 시각적으로 보기 좋게 보고서 형태로 만들거나, 현업에서 사용하는 특정 Excel 템플릿에 맞춰 데이터를 내보낼 때 큰 장점이 됩니다.

#### 1.1.2. Excel 파일 처리를 위한 필수 라이브러리 (Engines)

Pandas는 Excel 파일을 직접 파싱하고 생성하는 로직을 모두 내장하고 있지 않습니다. 대신, 파이썬 생태계의 다른 전문 라이브러리들을 "엔진(Engine)"으로 활용하여 Excel 파일과의 상호작용을 처리합니다. 따라서 Pandas를 통해 Excel 파일을 읽고 쓰려면, 사용하려는 Excel 파일 형식(구형 `.xls` 또는 신형 `.xlsx`)에 맞는 백엔드 라이브러리를 별도로 설치해야 합니다.

*   **`openpyxl`**:
    *   **주요 용도**: Microsoft Excel 2007 이후 버전에서 사용되는 `.xlsx` 파일(Open XML 형식)을 읽고 쓰는 데 사용되는 가장 권장되는 라이브러리입니다. 현재 Pandas의 `.xlsx` 파일 처리의 기본 엔진입니다.
    *   **특징**: 최신 Excel 파일 형식을 완벽하게 지원하며, 쓰기 기능이 안정적이고 다양한 서식 옵션과 고급 기능을 제공합니다.
    *   **설치**: `pip install openpyxl`

*   **`xlrd`**:
    *   **주요 용도**: Microsoft Excel 97-2003 버전에서 사용되던 구형 `.xls` 파일(Binary Interchange File Format)을 읽을 때 사용됩니다.
    *   **특징**: `.xls` 파일 읽기에 특화되어 있지만, **쓰기 기능은 더 이상 지원하지 않습니다.** (버전 2.0.1부터 쓰기 기능이 제거됨). 따라서 `.xls` 파일을 읽어야 할 때만 필요하며, `.xlsx` 파일 처리를 위해서는 `openpyxl`이 필수입니다.
    *   **설치**: `pip install xlrd`

*   **`XlsxWriter`**:
    *   **주요 용도**: `.xlsx` 파일을 **쓰는 데 특화된** 라이브러리입니다. `openpyxl`과 유사하게 `.xlsx` 파일을 생성하지만, 특히 복잡한 차트, 조건부 서식, 데이터 유효성 검사 등 고급 Excel 기능을 프로그래밍 방식으로 적용할 때 더 강력한 제어 기능을 제공합니다.
    *   **특징**: 읽기 기능은 없으며, 오직 쓰기 기능만 제공합니다. Pandas의 `to_excel()` 메서드에서 `engine='xlsxwriter'`로 지정하여 사용할 수 있습니다.
    *   **설치**: `pip install XlsxWriter`

**라이브러리 설치 방법:**
Pandas를 사용하여 Excel 파일을 원활하게 처리하려면, 일반적으로 `openpyxl`과 `xlrd`를 함께 설치하는 것이 좋습니다. `XlsxWriter`는 고급 쓰기 기능이 필요할 때 추가로 설치합니다.

```bash
pip install openpyxl xlrd XlsxWriter
```

이러한 백엔드 라이브러리들이 설치되어 있어야 Pandas의 `read_excel()` 및 `to_excel()` 함수가 정상적으로 작동하며, 다양한 Excel 파일 형식과 기능을 처리할 수 있습니다.

### 1.2. Excel 파일 읽기 (`read_excel`)

Pandas의 `read_excel()` 함수는 Microsoft Excel 파일(.xlsx, .xls)을 읽어 `DataFrame` 객체로 변환하는 핵심적인 기능을 제공합니다. 이 함수는 CSV 파일을 읽는 `read_csv()`와 유사하게 다양한 옵션을 통해 복잡한 형태의 Excel 파일도 유연하게 처리할 수 있습니다.

#### 1.2.1. 주요 `read_excel` 옵션 상세

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `io` | **필수**: 읽어올 Excel 파일의 경로(로컬 파일 시스템), 웹 URL(예: `http://example.com/data.xlsx`), 또는 파일과 유사한 객체(예: `io.BytesIO`로 열린 파일 객체)를 지정합니다. | - |
| `sheet_name` | **시트 지정**: 불러올 시트의 이름(문자열)이나 번호(정수, 0부터 시작)를 지정합니다. <br> - `0` (기본값): 첫 번째 시트를 불러옵니다.<br> - 시트 이름(예: `'Sheet1'`, `'성적'`): 해당 이름의 시트를 불러옵니다.<br> - 시트 번호(예: `1`): 두 번째 시트를 불러옵니다.<br> - `None`: Excel 파일 내의 모든 시트를 읽어와 시트 이름을 키(key)로, 해당 시트의 `DataFrame`을 값(value)으로 갖는 딕셔너리를 반환합니다.<br> - 시트 이름 또는 번호의 리스트(예: `['Sheet1', 'Sheet3']` 또는 `[0, 2]`): 지정된 시트들만 딕셔너리 형태로 불러옵니다. | `0` |
| `header` | **헤더 행 지정**: 컬럼명으로 사용할 행의 번호(0부터 시작)를 지정합니다. <br> - `0` (기본값): 첫 번째 행을 컬럼명으로 사용합니다.<br> - `None`: 파일에 컬럼명이 없다고 간주하고, Pandas가 0부터 시작하는 정수형 컬럼명(0, 1, 2, ...)을 자동으로 부여합니다.<br> - 특정 숫자: 해당 번호의 행을 컬럼명으로 사용하고, 그 이전 행들은 건너뜁니다.<br> - 리스트: 여러 행을 계층적(MultiIndex) 컬럼명으로 사용할 수 있습니다. | `0` |
| `names` | **컬럼명 지정**: `header=None`으로 설정하여 파일에 컬럼명이 없다고 지정했을 때, 사용할 컬럼명 리스트를 직접 지정합니다. 이 리스트의 길이가 데이터의 컬럼 수와 일치해야 합니다. | `None` |
| `index_col` | **인덱스 컬럼 지정**: `DataFrame`의 인덱스(행 라벨)로 사용할 컬럼의 번호(0부터 시작)나 이름을 지정합니다. <br> - `None` (기본값): Pandas가 0부터 시작하는 기본 정수 인덱스를 생성합니다.<br> - 특정 숫자/이름: 해당 컬럼을 인덱스로 설정합니다.<br> - 리스트: 여러 컬럼을 MultiIndex로 설정할 수 있습니다. | `None` |
| `usecols` | **선택적 컬럼 불러오기**: Excel 파일에서 특정 컬럼만 선택적으로 불러올 때 사용합니다. <br> - 컬럼 이름 리스트(예: `['name', 'age']`)<br> - 컬럼 번호 리스트(예: `[0, 2]`)<br> - Excel 스타일 범위 문자열(예: `'A:C'`는 A, B, C열을 의미, `'A,C,E:G'`는 A, C, E, F, G열을 의미)<br> - 함수: 각 컬럼 이름에 대해 True/False를 반환하는 함수를 전달하여 필터링할 수 있습니다. <br> 대용량 파일에서 필요한 데이터만 로드하여 메모리 사용량을 줄이고 처리 속도를 높이는 데 매우 효과적입니다. | `None` |
| `dtype` | **데이터 타입 명시**: 각 컬럼의 데이터 타입을 명시적으로 지정합니다. Pandas의 자동 타입 추론이 잘못되거나, 특정 컬럼을 원하는 타입으로 강제하고 싶을 때 유용합니다. 딕셔너리 형태로 `{컬럼명: 데이터타입}`을 지정합니다(예: `{'id': str, 'value': float}`). | `None` |
| `parse_dates` | **날짜/시간 컬럼 파싱**: 문자열로 저장된 날짜 및 시간 데이터를 Pandas의 `datetime` 객체로 변환합니다. <br> - `True`: Pandas가 모든 컬럼을 스캔하여 날짜/시간 형식으로 추론 가능한 컬럼을 자동으로 파싱합니다.<br> - 컬럼 이름 리스트(예: `['date_col', 'timestamp']`): 지정된 컬럼만 파싱을 시도합니다.<br> - 중첩 리스트(예: `[['year', 'month', 'day']]`): 여러 컬럼을 조합하여 하나의 날짜/시간 컬럼으로 파싱할 수 있습니다. | `False` |
| `skiprows` | **특정 행 건너뛰기**: 파일의 시작 부분에서 지정된 수의 행을 건너뛰거나, 특정 행 번호(0부터 시작)들을 건너뛸 때 사용합니다. <br> - 정수: 파일의 맨 위에서부터 해당 개수만큼의 행을 건너뜁니다.<br> - 리스트: 지정된 행 번호들만 건너뜠습니다.<br> - 함수: 각 행 번호에 대해 True/False를 반환하는 함수를 전달하여 조건부로 건너뛸 수 있습니다. | `None` |
| `nrows` | **불러올 행의 개수 제한**: 파일의 시작부터 지정된 개수만큼의 행만 불러옵니다. 대용량 파일의 전체를 로드하기 전에 파일 구조나 데이터 샘플을 빠르게 확인할 때 유용합니다. `skiprows`와 함께 사용하여 특정 범위의 데이터만 읽을 수도 있습니다. | `None` |
| `engine` | **백엔드 라이브러리 지정**: Excel 파일을 처리할 때 사용할 백엔드 라이브러리를 명시적으로 지정합니다. <br> - `'openpyxl'`: `.xlsx` 파일을 읽고 쓸 때 권장됩니다.<br> - `'xlrd'`: 구형 `.xls` 파일을 읽을 때 사용됩니다.<br> - `'odf'`: OpenDocument Spreadsheet(`.ods`) 파일을 읽을 때 사용됩니다.<br> - `'pyxlsb'`: `.xlsb` (Excel Binary Workbook) 파일을 읽을 때 사용됩니다.<br> 일반적으로 Pandas가 파일 확장자를 보고 적절한 엔진을 자동으로 선택하지만, 특정 엔진을 강제해야 할 때 사용합니다. | 자동 선택 |
| `converters` | **컬럼별 변환 함수 지정**: 특정 컬럼의 데이터를 읽어올 때 적용할 함수를 딕셔너리 형태로 지정합니다. `{컬럼명: 함수}`. 예를 들어, 특정 문자열을 숫자로 변환하거나, 특정 패턴을 추출하는 등의 전처리 작업을 로드 시점에 수행할 수 있습니다. | `None` |
| `decimal` | **소수점 구분자 지정**: 숫자의 소수점 구분자가 마침표(`.`)가 아닌 다른 문자(예: 쉼표 `,`)로 되어 있을 때 해당 문자를 지정합니다. 유럽권에서 자주 사용됩니다. | `.` |

#### 1.2.2. `read_excel` 활용 예시

다음 예시들을 통해 `read_excel()` 함수의 다양한 옵션들을 실제 코드와 함께 살펴보겠습니다. 예시를 위해 가상의 Excel 파일을 메모리 내에서 생성하여 사용합니다. 실제 사용 시에는 파일 경로를 `io.BytesIO` 객체 대신 직접 지정하면 됩니다.

```python
import pandas as pd
import io

# 예제용 가상 Excel 파일을 메모리에 생성합니다.
# 실제 사용 시에는 파일 경로를 io.BytesIO 대신 사용합니다. (예: 'data/score.xlsx')
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
    # 첫 번째 시트: 성적 정보
    pd.DataFrame({
        '이름': ['홍길동', '임꺽정', '장보고'],
        '국어': [90, 80, 95],
        '영어': [99, 98, 85],
        '수학': [90, 70, 100]
    }).to_excel(writer, sheet_name='성적', index=False)

    # 두 번째 시트: 도시 정보 (헤더가 1행 아래에 있고, 불필요한 컬럼 포함)
    pd.DataFrame({
        '도시명': ['서울', '부산', '제주'],
        '인구(만명)': [950, 340, 67],
        '면적(km2)': [605, 770, 1850],
        '특징': ['수도', '항구도시', '관광지']
    }).to_excel(writer, sheet_name='도시정보', index=False, startrow=1)

    # 세 번째 시트: 날짜 데이터 (날짜 컬럼이 문자열로 저장)
    pd.DataFrame({
        '날짜': ['2023-01-01', '2023-01-02', '2023-01-03'],
        '판매량': [100, 120, 150]
    }).to_excel(writer, sheet_name='판매데이터', index=False)

    # 네 번째 시트: 특정 범위 데이터 (A1:B3만 유효한 데이터)
    pd.DataFrame({
        '컬럼A': [1, 2, 3, 4, 5],
        '컬럼B': [10, 20, 30, 40, 50],
        '컬럼C': [100, 200, 300, 400, 500]
    }).to_excel(writer, sheet_name='부분데이터', index=False)

excel_buffer.seek(0) # 버퍼의 커서를 처음으로 이동시켜 read_excel이 읽을 수 있도록 준비
```

**1. 기본 읽기 (첫 번째 시트)**
`sheet_name` 옵션을 생략하면 `read_excel()`은 기본적으로 Excel 파일의 첫 번째 시트(인덱스 0)를 읽어옵니다. 첫 행을 헤더로 인식하고 데이터 타입을 자동으로 추론합니다.

**코드**
```python
df_score = pd.read_excel(excel_buffer) # sheet_name을 생략하면 첫 번째 시트인 '성적' 시트를 읽음
print("--- \n기본 읽기 (첫 번째 시트)\n---")
print(df_score)
```
**결과:**
```
--- 기본 읽기 (첫 번째 시트)
---
    이름  국어  영어   수학
0  홍길동   90   99   90
1  임꺽정   80   98   70
2  장보고   95   85  100
```
**결과 설명**
*   '성적' 시트의 데이터가 정확히 로드되었으며, '이름', '국어', '영어', '수학'이 컬럼명으로 인식되었습니다.

**2. 특정 시트 이름으로 읽기 (`sheet_name`)**
Excel 파일에 여러 시트가 있을 때, `sheet_name` 옵션에 원하는 시트의 이름을 문자열로 지정하여 해당 시트의 데이터만 불러올 수 있습니다.

**코드**
```python
excel_buffer.seek(0) # 버퍼 초기화
df_city = pd.read_excel(excel_buffer, sheet_name='도시정보')
print("\n--- '도시정보' 시트 읽기 (기본 헤더 인식)
---")
print(df_city)
```
**결과:**
```
--- '도시정보' 시트 읽기 (기본 헤더 인식)
---
  Unnamed: 0 Unnamed: 1 Unnamed: 2 Unnamed: 3
0         도시명    인구(만명)    면적(km2)         특징
1          서울        950        605        수도
2          부산        340        770     항구도시
3          제주         67       1850      관광지
```
**결과 설명**
*   '도시정보' 시트는 `startrow=1`로 인해 실제 헤더가 두 번째 행(인덱스 1)에 있습니다. `read_excel`의 기본 `header=0` 때문에 첫 번째 행이 헤더로 인식되어 'Unnamed' 컬럼들이 생성되었습니다. 다음 예시에서 이를 수정합니다.

**3. 특정 시트와 헤더 위치 지정 (`sheet_name`, `header`)**
실제 헤더가 첫 행이 아닌 특정 행에 위치할 경우, `header` 옵션에 해당 행의 번호(0부터 시작)를 지정하여 올바른 컬럼명을 인식하도록 합니다.

**코드**
```python
excel_buffer.seek(0) # 버퍼 초기화
df_city_correct = pd.read_excel(excel_buffer, sheet_name='도시정보', header=1)
print("\n--- '도시정보' 시트 읽기 (헤더 위치 지정)
---")
print(df_city_correct)
```
**결과:**
```
--- '도시정보' 시트 읽기 (헤더 위치 지정)
---
  도시명  인구(만명)  면적(km2)     특징
0   서울       950      605     수도
1   부산       340      770  항구도시
2   제주        67     1850   관광지
```
**결과 설명**
*   `header=1`을 지정함으로써 '도시정보' 시트의 두 번째 행이 올바르게 컬럼명으로 인식되어 '도시명', '인구(만명)', '면적(km2)', '특징' 컬럼이 생성되었습니다.

**4. 모든 시트 한 번에 읽기 (`sheet_name=None`)**
Excel 파일의 모든 시트 데이터를 한 번에 불러와서 처리해야 할 때 `sheet_name=None` 옵션을 사용합니다. 이 경우, `read_excel()`은 각 시트의 이름을 키(key)로 하고 해당 시트의 `DataFrame`을 값(value)으로 하는 딕셔너리를 반환합니다.

**코드**
```python
excel_buffer.seek(0) # 버퍼 초기화
all_sheets = pd.read_excel(excel_buffer, sheet_name=None)
print("\n--- 모든 시트 읽기 (sheet_name=None)
---")
for sheet_name, df in all_sheets.items():
    print(f"\nSheet: {sheet_name}")
    print(df.head())
```
**결과:**
```
--- 모든 시트 읽기 (sheet_name=None)
---

Sheet: 성적
    이름  국어  영어   수학
0  홍길동   90   99   90
1  임꺽정   80   98   70
2  장보고   95   85  100

Sheet: 도시정보
  Unnamed: 0 Unnamed: 1 Unnamed: 2 Unnamed: 3
0         도시명    인구(만명)    면적(km2)         특징
1          서울        950        605        수도
2          부산        340        770     항구도시
3          제주         67       1850      관광지

Sheet: 판매데이터
         날짜  판매량
0  2023-01-01  100
1  2023-01-02  120
2  2023-01-03  150

Sheet: 부분데이터
   컬럼A  컬럼B  컬럼C
0    1   10  100
1    2   20  200
2    3   30  300
3    4   40  400
4    5   50  500
```
**결과 설명**
*   `all_sheets`는 딕셔너리 형태로, 각 시트의 이름이 키로, 해당 시트의 데이터가 `DataFrame`으로 저장되어 있습니다. 이를 통해 여러 시트의 데이터를 쉽게 반복 처리하거나 접근할 수 있습니다.

**5. 특정 컬럼만 읽고 인덱스 지정 (`usecols`, `index_col`)**
분석에 필요한 컬럼만 선택적으로 불러오고, 특정 컬럼을 `DataFrame`의 인덱스로 지정하여 메모리 효율성을 높이고 데이터 접근을 용이하게 할 수 있습니다. `usecols`는 컬럼 이름 리스트나 Excel 스타일의 범위 문자열을 받을 수 있습니다.

**코드**
```python
excel_buffer.seek(0) # 버퍼 초기화
# '성적' 시트에서 '이름'과 '국어' 컬럼만 불러오고, '이름' 컬럼을 인덱스로 지정합니다.
df_partial = pd.read_excel(excel_buffer, sheet_name='성적', usecols=['이름', '국어'], index_col='이름')
print("\n--- 특정 컬럼 읽고 인덱스 지정 (usecols, index_col)
---")
print(df_partial)

excel_buffer.seek(0) # 버퍼 초기화
# '부분데이터' 시트에서 A열과 B열만 불러옵니다 (Excel 스타일 범위 지정).
df_range = pd.read_excel(excel_buffer, sheet_name='부분데이터', usecols='A:B')
print("\n--- 특정 범위 컬럼 읽기 (usecols='A:B')
---")
print(df_range)
```
**결과:**
```
--- 특정 컬럼 읽고 인덱스 지정 (usecols, index_col)
---
     국어
이름     
홍길동   90
임꺽정   80
장보고   95

--- 특정 범위 컬럼 읽기 (usecols='A:B')
---
   컬럼A  컬럼B
0    1   10
1    2   20
2    3   30
3    4   40
4    5   50
```
**결과 설명**
*   `usecols`와 `index_col`을 함께 사용하여 필요한 데이터만 효율적으로 로드하고, '이름' 컬럼을 인덱스로 설정했습니다.
*   `usecols='A:B'`를 통해 Excel의 A열과 B열에 해당하는 데이터만 정확히 불러왔습니다.

**6. 날짜 컬럼 파싱 (`parse_dates`)**
Excel 파일에 날짜 데이터가 텍스트 형태로 저장되어 있을 때, `parse_dates` 옵션을 사용하여 이를 Pandas의 `datetime` 객체로 변환할 수 있습니다. 이는 날짜/시간 기반의 분석을 수행하는 데 필수적입니다.

**코드**
```python
excel_buffer.seek(0) # 버퍼 초기화
# '판매데이터' 시트에서 '날짜' 컬럼을 datetime 객체로 파싱합니다.
df_sales = pd.read_excel(excel_buffer, sheet_name='판매데이터', parse_dates=['날짜'])
print("\n--- 날짜 컬럼 파싱 (parse_dates)
---")
print(df_sales.info())
print("\n데이터프레임 내용:
", df_sales)
```
**결과:**
```
--- 날짜 컬럼 파싱 (parse_dates)
---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype         
---  ------  --------------  -----         
 0   날짜      3 non-null      datetime64[ns]
 1   판매량     3 non-null      int64         
dtypes: datetime64[ns](1), int64(1)
memory usage: 176.0+ bytes
None

데이터프레임 내용:
          날짜  판매량
0 2023-01-01  100
1 2023-01-02  120
2 2023-01-03  150
```
**결과 설명**
*   `df_sales.info()`를 통해 '날짜' 컬럼의 `Dtype`이 `datetime64[ns]`로 변경된 것을 확인할 수 있습니다. 이는 날짜/시간 연산을 수행할 수 있는 형태로 변환되었음을 의미합니다.

**7. 특정 행 건너뛰기 및 행 개수 제한 (`skiprows`, `nrows`)**
Excel 파일의 특정 부분을 건너뛰고 싶거나, 파일의 일부만 빠르게 확인하고 싶을 때 `skiprows`와 `nrows` 옵션을 사용합니다. `skiprows`는 정수(시작부터 건너뛸 행 수)나 리스트(건너뛸 특정 행 번호들)를 받을 수 있습니다.

**코드**
```python
excel_buffer.seek(0) # 버퍼 초기화
# '부분데이터' 시트에서 첫 2행을 건너뛰고, 그 다음 2행만 읽어옵니다.
df_skip_nrows = pd.read_excel(excel_buffer, sheet_name='부분데이터', skiprows=2, nrows=2)
print("\n--- 특정 행 건너뛰기 및 행 개수 제한 (skiprows, nrows)
---")
print(df_skip_nrows)
```
**결과:**
```
--- 특정 행 건너뛰기 및 행 개수 제한 (skiprows, nrows)
---
   컬럼A  컬럼B  컬럼C
0    3   30  300
1    4   40  400
```
**결과 설명**
*   `skiprows=2`로 인해 첫 두 행(헤더와 첫 번째 데이터 행)이 무시되고, 세 번째 행부터 읽기가 시작됩니다.
*   `nrows=2`로 인해 세 번째 행부터 두 개의 행만 읽어와 `DataFrame`에 포함됩니다.

이처럼 `read_excel()` 함수는 다양한 옵션을 통해 Excel 파일의 복잡한 구조와 데이터를 유연하게 읽어올 수 있는 강력한 기능을 제공합니다. 실제 데이터 분석 환경에서 Excel 파일을 다룰 때 이 옵션들을 적절히 활용하는 것이 중요합니다.


`read_excel` 함수는 Excel 파일을 `DataFrame`으로 불러옵니다.

### 1.3. Excel 파일 저장 (`to_excel`)

데이터 분석 및 처리 과정에서 생성되거나 수정된 `DataFrame`을 Excel 파일로 저장하는 것은 분석 결과를 공유하거나, 다른 시스템으로 데이터를 전달할 때 매우 중요합니다. Pandas의 `DataFrame.to_excel()` 메서드는 이러한 작업을 효율적으로 수행할 수 있도록 다양한 옵션을 제공하며, 저장될 파일의 형식, 포함할 데이터, 서식 등을 세밀하게 제어할 수 있습니다.

#### 1.3.1. 주요 `to_excel` 옵션 상세

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `excel_writer` | **필수**: `DataFrame`을 저장할 파일 경로(문자열, 예: `'output.xlsx'`, `'data/processed_data.xlsx'`)를 지정하거나, `pd.ExcelWriter` 객체를 전달합니다. `ExcelWriter` 객체를 사용하면 한 Excel 파일에 여러 시트를 저장하거나, 고급 서식 기능을 적용할 수 있습니다. | - |
| `sheet_name` | **시트 이름 지정**: `DataFrame`이 저장될 Excel 시트의 이름을 지정합니다. 이 옵션을 생략하면 기본값인 `'Sheet1'`으로 저장됩니다. | `'Sheet1'` |
| `na_rep` | **결측치 표현**: `DataFrame` 내의 `NaN`(Not a Number) 또는 `None`과 같은 결측값들을 Excel 파일에 어떤 문자열로 표현할지 지정합니다. 기본적으로는 빈 문자열(`''`)로 저장됩니다. 예를 들어, `na_rep='N/A'`로 설정하면 결측치가 'N/A'로 저장됩니다. | `''` (빈 문자열) |
| `header` | **헤더(컬럼명) 포함 여부**: Excel 파일의 첫 행에 컬럼명(헤더)을 포함할지 여부를 `True` 또는 `False`로 지정합니다. <br> - `True` (기본값): 컬럼명을 파일에 씁니다.<br> - `False`: 컬럼명을 파일에 쓰지 않습니다. 이는 주로 여러 `DataFrame`을 병합하여 하나의 시트에 저장할 때 중복 헤더를 방지하는 데 유용합니다. | `True` |
| `index` | **인덱스 포함 여부**: `DataFrame`의 인덱스(행 라벨)를 Excel 파일의 첫 번째 컬럼으로 포함할지 여부를 `True` 또는 `False`로 지정합니다. <br> - `True` (기본값): 인덱스를 파일에 씁니다.<br> - `False`: 인덱스를 파일에 쓰지 않습니다. 데이터를 다른 시스템으로 옮기거나, 인덱스가 불필요한 보고서를 생성할 때 자주 사용됩니다. | `True` |
| `index_label` | **인덱스 컬럼 라벨 지정**: `index=True`로 인덱스를 파일에 포함할 때, 해당 인덱스 컬럼의 헤더(라벨)를 지정합니다. 기본값은 `None`이며, 이 경우 `DataFrame` 인덱스에 이름이 있다면 그 이름이 사용되고, 없다면 빈 문자열로 처리됩니다. | `None` |
| `startrow`, `startcol` | **데이터 쓰기 시작 위치**: Excel 시트 내에서 데이터를 쓰기 시작할 셀의 행 번호(`startrow`)와 열 번호(`startcol`)를 0부터 시작하는 인덱스로 지정합니다. 이 옵션은 보고서 상단에 제목이나 요약 정보를 추가할 공간을 확보하거나, 기존 시트의 특정 위치에 데이터를 삽입할 때 유용합니다. | `0` |
| `engine` | **백엔드 라이브러리 지정**: Excel 파일을 생성할 때 사용할 백엔드 라이브러리를 명시적으로 지정합니다. <br> - `'openpyxl'` (기본값): `.xlsx` 파일을 읽고 쓸 때 권장됩니다.<br> - `'xlsxwriter'`: `.xlsx` 파일을 쓰는 데 특화된 라이브러리로, 고급 서식 및 차트 기능을 적용할 때 더 강력한 제어 기능을 제공합니다. | `openpyxl` |
| `freeze_panes` | **틀 고정**: Excel에서 특정 행과 열을 기준으로 화면을 고정하는 '틀 고정' 기능을 적용합니다. `(행 번호, 열 번호)` 튜플 형태로 지정합니다. 예를 들어, `(1, 0)`은 첫 번째 행을 고정하고, `(1, 1)`은 첫 번째 행과 첫 번째 열을 모두 고정합니다. 데이터가 많아 스크롤해야 할 때 유용합니다. | `None` |
| `encoding` | **파일 인코딩 형식 지정**: 저장될 Excel 파일의 문자 인코딩 방식을 지정합니다. `.xlsx` 파일은 기본적으로 UTF-8 기반이므로 이 옵션이 CSV만큼 중요하지는 않지만, 특정 환경에서 문제가 발생할 경우 고려할 수 있습니다. | `utf-8` |

#### 1.3.2. `to_excel()` 활용 예시
다음 예시들을 통해 `to_excel()` 함수의 다양한 옵션들을 실제 코드와 함께 살펴보겠습니다. 예시를 위해 `DataFrame`을 생성하고, 생성된 Excel 파일은 실제 파일 시스템에 저장됩니다.

```python
import pandas as pd
import os # 파일 삭제를 위해 os 모듈 임포트

# 예제용 DataFrame 생성
df1 = pd.DataFrame({
    '제품': ['A', 'B', 'C'],
    '가격': [15000, 22000, 8000],
    '재고': [100, 50, 200]
})
df2 = pd.DataFrame({
    '직원': ['Kim', 'Lee', 'Park'],
    '부서': ['영업', '개발', '마케팅'],
    '입사일': ['2020-01-01', '2021-03-15', '2019-07-20']
})

print("원본 df1:\n", df1)
print("\n원본 df2:\n", df2)
```

**원본 DataFrame:**
```
원본 df1:
  제품    가격   재고
0    A  15000  100
1    B  22000   50
2    C   8000  200

원본 df2:
    직원   부서        입사일
0  Kim   영업  2020-01-01
1  Lee   개발  2021-03-15
2  Park  마케팅  2019-07-20
```

**1. 기본 저장 (인덱스 및 헤더 포함)**
가장 기본적인 형태로, `DataFrame`의 모든 컬럼과 인덱스, 그리고 헤더(컬럼명)를 포함하여 Excel 파일로 저장합니다. `sheet_name`을 지정하지 않으면 기본값인 `'Sheet1'`으로 시트가 생성됩니다.

**코드**
```python
output_file_basic = "basic_output.xlsx"
df1.to_excel(output_file_basic, sheet_name="제품정보")
print(f"\n1. '{output_file_basic}' 파일이 생성되었습니다 (인덱스 포함, 시트명: 제품정보).")

# 생성된 파일 확인 (선택 사항)
# df_read_basic = pd.read_excel(output_file_basic, sheet_name="제품정보")
# print("\n읽어온 데이터:\n", df_read_basic)

# 예시 파일 삭제
os.remove(output_file_basic)
print(f'\'{output_file_basic}\' 파일 삭제 완료.')
```
**결과 설명**
*   `basic_output.xlsx` 파일이 생성되고, '제품정보' 시트에 `df1`의 내용이 저장됩니다. `DataFrame`의 인덱스가 Excel 파일의 첫 번째 열로 포함됩니다.

**2. 인덱스 제외하고 저장 (`index=False`)**
데이터 분석 시 `DataFrame`의 인덱스가 단순히 행 번호 역할을 하거나, Excel 파일에 인덱스 정보가 불필요한 경우가 많습니다. 이럴 때는 `index=False` 옵션을 사용하여 인덱스를 Excel 파일에 포함하지 않을 수 있습니다.

**코드**
```python
output_file_no_index = "no_index_output.xlsx"
df1.to_excel(output_file_no_index, index=False)
print(f"\n2. '{output_file_no_index}' 파일이 생성되었습니다 (인덱스 제외).")

# 예시 파일 삭제
os.remove(output_file_no_index)
print(f'\'{output_file_no_index}\' 파일 삭제 완료.')
```
**결과 설명**
*   `no_index_output.xlsx` 파일이 생성되고, `df1`의 데이터 컬럼만 저장됩니다. 인덱스 열은 Excel 파일에 나타나지 않습니다.

**3. 결측값을 특정 문자로 대체 (`na_rep`)**
`DataFrame` 내의 결측값(`NaN`, `None`)을 Excel 파일에 저장할 때 빈 셀 대신 특정 문자열(예: 'N/A', '-')로 표시하고 싶을 때 `na_rep` 옵션을 사용합니다. 이는 데이터의 가독성을 높이거나, 특정 시스템에서 결측값을 특정 형식으로 요구할 때 유용합니다.

**코드**
```python
df_with_nan = pd.DataFrame({
    'A': [1, 2, None],
    'B': [10, None, 30]
})

output_file_na_rep = "na_rep_output.xlsx"
df_with_nan.to_excel(output_file_na_rep, index=False, na_rep='결측치')
print(f"\n3. '{output_file_na_rep}' 파일이 생성되었습니다 (결측치 '결측치'로 대체).")

# 예시 파일 삭제
os.remove(output_file_na_rep)
print(f'\'{output_file_na_rep}\' 파일 삭제 완료.')
```
**결과 설명**
*   `df_with_nan`의 `None` 값들이 Excel 파일에 '결측치'라는 문자열로 저장됩니다.

**4. 특정 컬럼만 저장하고 구분자 변경 (`columns`, `sep`)**
`DataFrame`의 모든 컬럼을 저장할 필요 없이 특정 컬럼만 저장하고 싶을 때 `columns` 옵션을 사용합니다. Excel 파일은 CSV와 달리 `sep` 옵션이 직접적으로 구분자를 변경하지는 않지만, `ExcelWriter`를 통해 다른 엔진을 사용하거나, CSV로 저장할 때와 유사한 개념으로 이해할 수 있습니다. (Excel 파일 자체는 셀 기반이므로 `sep`은 `to_excel`에서 직접적인 의미는 없습니다. 이 예시는 CSV와 비교를 위해 남겨둡니다.)

**코드**
```python
output_file_cols_sep = "cols_sep_output.xlsx"
# '제품'과 '가격' 컬럼만 저장합니다.
df1.to_excel(output_file_cols_sep, columns=['제품', '가격'], index=False)
print(f"\n4. '{output_file_cols_sep}' 파일이 생성되었습니다 (특정 컬럼만 저장).")

# 예시 파일 삭제
os.remove(output_file_cols_sep)
print(f'\'{output_file_cols_sep}\' 파일 삭제 완료.')
```
**결과 설명**
*   `cols_sep_output.xlsx` 파일에는 `df1`의 '제품'과 '가격' 컬럼만 저장되고, '재고' 컬럼은 제외됩니다.

**5. 여러 DataFrame을 다른 시트로 저장 (`pd.ExcelWriter`)**
`pd.ExcelWriter` 객체를 사용하면 하나의 Excel 통합 문서(`Workbook`) 내에 여러 개의 `DataFrame`을 각각 다른 시트(`Sheet`)로 저장할 수 있습니다. 이는 여러 분석 결과를 하나의 파일로 통합하여 관리하거나 공유할 때 매우 유용합니다.

**코드**
```python
output_file_multi_sheet = 'multi_sheet_output.xlsx'
with pd.ExcelWriter(output_file_multi_sheet, engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='제품정보', index=False)
    df2.to_excel(writer, sheet_name='직원정보', index=False)
print(f"\n5. '{output_file_multi_sheet}' 파일에 여러 시트가 저장되었습니다 ('제품정보', '직원정보').")

# 예시 파일 삭제
os.remove(output_file_multi_sheet)
print(f'\'{output_file_multi_sheet}\' 파일 삭제 완료.')
```
**결과 설명**
*   `multi_sheet_output.xlsx` 파일이 생성되고, 이 파일 안에 '제품정보'와 '직원정보'라는 두 개의 시트가 각각 `df1`과 `df2`의 내용을 담아 저장됩니다.

**6. 특정 위치부터 쓰고 틀 고정하기 (`startrow`, `startcol`, `freeze_panes`)**
보고서 형식의 Excel 파일을 만들 때, 데이터가 시작하는 위치를 조정하거나 특정 행/열을 고정하여 스크롤 시에도 헤더가 보이도록 하는 '틀 고정' 기능을 적용할 수 있습니다. 이는 데이터의 가독성을 크게 향상시킵니다.

**코드**
```python
output_file_report = 'report_output.xlsx'
with pd.ExcelWriter(output_file_report, engine='openpyxl') as writer:
    df1.to_excel(writer, 
                 sheet_name='월간 보고서', 
                 index=False, 
                 startrow=3, # 0부터 시작, 4번째 행(A4)부터 데이터 쓰기 시작
                 startcol=1, # 0부터 시작, B열(B1)부터 데이터 쓰기 시작
                 freeze_panes=(4, 2) # 4행(인덱스 3)과 2열(인덱스 1)을 기준으로 틀 고정
                )
print(f"\n6. '{output_file_report}' 파일이 보고서 형식으로 저장되었습니다 (시작 위치 및 틀 고정 적용).")

# 예시 파일 삭제
os.remove(output_file_report)
print(f'\'{output_file_report}\' 파일 삭제 완료.')
```
**결과 설명**
*   `report_output.xlsx` 파일의 '월간 보고서' 시트에서 `df1`의 데이터가 B4 셀부터 쓰여집니다. 이는 A1부터 A3, 그리고 A열에 제목이나 다른 정보를 추가할 수 있는 공간을 확보합니다.
*   `freeze_panes=(4, 2)`는 Excel에서 4번째 행(인덱스 3)과 B열(인덱스 1)을 기준으로 틀을 고정하여, 스크롤 시에도 이 영역이 항상 보이도록 합니다.

이처럼 `to_excel()` 함수는 다양한 옵션을 통해 `DataFrame`을 원하는 형식의 Excel 파일로 유연하게 저장할 수 있는 강력한 기능을 제공합니다. 데이터 내보내기 및 공유 시 이 옵션들을 적절히 활용하는 것이 중요합니다.
