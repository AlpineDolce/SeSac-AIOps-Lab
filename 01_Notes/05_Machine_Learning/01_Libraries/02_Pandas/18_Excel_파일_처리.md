<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 Pandas를 사용하여 Microsoft Excel 파일(.xlsx, .xls)을 직접 읽고 쓰는 방법을 다룹니다. Excel 파일 처리의 장점과 필수 라이브러리를 이해하고, `read_excel()` 함수를 이용한 파일 불러오기, `to_excel()` 메서드를 이용한 `DataFrame` 저장 방법을 실제 코드 예제를 통해 학습합니다. 특히 여러 시트를 다루거나 특정 옵션을 활용하는 실무 예제를 중점적으로 살펴봅니다.

<h2>목차</h2>

- [1. Excel 파일 처리](#1-excel-파일-처리)
  - [1.1. Excel 파일 처리의 장점 및 필수 라이브러리](#11-excel-파일-처리의-장점-및-필수-라이브러리)
  - [1.2. Excel 파일 읽기 (`read_excel`)](#12-excel-파일-읽기-read_excel)
    - [1.2.1. 주요 `read_excel` 옵션](#121-주요-read_excel-옵션)
    - [1.2.2. `read_excel` 활용 예시](#122-read_excel-활용-예시)
  - [1.3. Excel 파일 저장 (`to_excel`)](#13-excel-파일-저장-to_excel)
    - [1.3.1. 주요 `to_excel` 옵션](#131-주요-to_excel-옵션)
    - [1.3.2. `to_excel` 활용 예시](#132-to_excel-활용-예시)

---

## 1. Excel 파일 처리

Pandas는 Microsoft Excel 파일(.xlsx, .xls)을 직접 읽고 쓰는 강력한 기능을 제공합니다. 이는 Excel을 주로 사용하는 현업 환경에서 데이터를 가져오거나 분석 결과를 공유할 때 매우 유용합니다.

### 1.1. Excel 파일 처리의 장점 및 필수 라이브러리

*   **Pandas 직접 지원**: Pandas에 `read_excel()` 및 `to_excel()` 함수가 내장되어 있어 다른 라이브러리를 직접 호출할 필요 없이 일관된 방식으로 Excel 파일을 다룰 수 있습니다.
*   **복잡한 데이터 구조 처리**: 여러 시트(sheet)를 가진 Excel 파일이나 특정 범위의 데이터도 유연하게 처리할 수 있습니다.
*   **서식 유지**: CSV와 달리, Excel 파일로 저장 시 셀 서식, 너비 등 다양한 스타일을 적용하여 가독성 높은 보고서를 만들 수 있습니다. (스타일링은 `XlsxWriter` 같은 엔진과 함께 사용할 때 더욱 강력해집니다.)

**필수 라이브러리 (Engines):**
Pandas는 내부적으로 다른 라이브러리를 사용하여 Excel 파일을 처리합니다. 사용하려는 파일 형식에 맞는 라이브러리를 설치해야 합니다.

*   **`openpyxl`**: `.xlsx` 파일을 읽고 쓸 때 사용됩니다. (권장)
*   **`xlrd`**: 구형 `.xls` 파일을 읽을 때 사용됩니다. (쓰기 기능은 지원 중단)

라이브러리가 없다면 `pip`을 이용해 설치할 수 있습니다.
```bash
pip install openpyxl xlrd
```

### 1.2. Excel 파일 읽기 (`read_excel`)

`read_excel` 함수는 Excel 파일을 `DataFrame`으로 불러옵니다.

#### 1.2.1. 주요 `read_excel` 옵션

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `io` | 파일 경로, URL, 또는 `ExcelFile` 객체. | - |
| `sheet_name` | 불러올 시트의 이름(str)이나 번호(int, 0부터 시작). `None`으로 지정 시 모든 시트를 딕셔너리로 불러옴. | `0` |
| `header` | 컬럼명으로 사용할 행의 번호. | `0` |
| `names` | `header=None`일 때 사용할 컬럼명 리스트. | `None` |
| `index_col` | 인덱스로 사용할 컬럼의 번호나 이름. | `None` |
| `usecols` | 불러올 컬럼의 리스트나 범위(예: `'A:C'`). | `None` |
| `dtype` | 컬럼별 데이터 타입을 딕셔너리로 지정. | `None` |
| `skiprows` | 파일 상단에서 건너뛸 행의 리스트. | `None` |
| `nrows` | 불러올 행의 개수. | `None` |
| `engine` | 사용할 백엔드 라이브러리. `.xlsx`는 `openpyxl`, `.xls`는 `xlrd`. | 자동 선택 |

#### 1.2.2. `read_excel` 활용 예시

```python
import pandas as pd
import io

# 예제용 가상 Excel 파일을 만듭니다.
# 실제 사용 시에는 파일 경로를 io 대신 사용합니다. (예: 'score.xlsx')
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
    pd.DataFrame({
        'name': ['홍길동', '임꺽정'],
        'kor': [90, 80],
        'eng': [99, 98],
        'mat': [90, 70]
    }).to_excel(writer, sheet_name='성적', index=False)
    pd.DataFrame({
        'city': ['서울', '부산'],
        'population': [950, 340]
    }).to_excel(writer, sheet_name='도시정보', index=False)
excel_buffer.seek(0) # 버퍼의 커서를 처음으로 이동
```

**1. 기본 읽기 (첫 번째 시트)**
```python
df_score = pd.read_excel(excel_buffer) # sheet_name을 생략하면 첫 번째 시트를 읽음
print("---\n""기본 읽기 (첫 번째 시트)""---")
print(df_score)
```
**결과:**
```
--- 기본 읽기 (첫 번째 시트) ---
    name  kor  eng  mat
0  홍길동   90   99   90
1  임꺽정   80   98   70
```

**2. 특정 시트 이름으로 읽기 (`sheet_name`)**
```python
df_city = pd.read_excel(excel_buffer, sheet_name='도시정보')
print("\n--- '도시정보' 시트 읽기 ---")
print(df_city)
```
**결과:**
```
--- '도시정보' 시트 읽기 ---
  city  population
0   서울         950
1   부산         340
```

**3. 모든 시트 한 번에 읽기 (`sheet_name=None`)**
모든 시트를 읽어와 시트 이름을 key로, DataFrame을 value로 갖는 딕셔너리를 반환합니다.
```python
all_sheets = pd.read_excel(excel_buffer, sheet_name=None)
print("\n--- 모든 시트 읽기 ---")
for sheet_name, df in all_sheets.items():
    print(f"\nSheet: {sheet_name}")
    print(df)
```
**결과:**
```
--- 모든 시트 읽기 ---

Sheet: 성적
    name  kor  eng  mat
0  홍길동   90   99   90
1  임꺽정   80   98   70

Sheet: 도시정보
  city  population
0   서울         950
1   부산         340
```

**4. 특정 컬럼만 읽고 인덱스 지정 (`usecols`, `index_col`)**
```python
# 버퍼를 다시 초기화해야 합니다.
excel_buffer.seek(0)
df_partial = pd.read_excel(excel_buffer, usecols=['name', 'kor'], index_col='name')
print("\n--- 특정 컬럼 읽고 인덱스 지정 ---")
print(df_partial)
```
**결과:**
```
--- 특정 컬럼 읽고 인덱스 지정 ---
     kor
name    
홍길동   90
임꺽정   80
```

### 1.3. Excel 파일 저장 (`to_excel`)

`to_excel` 메서드는 `DataFrame`을 Excel 파일로 저장합니다.

#### 1.3.1. 주요 `to_excel` 옵션

| 옵션 | 설명 | 기본값 |
| --- | --- | --- |
| `excel_writer` | 저장할 파일 경로 또는 `ExcelWriter` 객체. | - |
| `sheet_name` | 저장할 시트의 이름. | `'Sheet1'` |
| `na_rep` | `NaN` (결측치) 값을 대체할 문자열. | `''` |
| `header` | 컬럼명을 파일에 쓸지 여부. | `True` |
| `index` | DataFrame의 인덱스를 파일에 쓸지 여부. | `True` |
| `index_label` | 인덱스 컬럼의 라벨(이름) 지정. | `None` |
| `startrow`, `startcol` | 데이터 쓰기를 시작할 셀의 행/열 번호 (0부터 시작). | `0` |
| `engine` | 사용할 백엔드 라이브러리. | `openpyxl` |
| `freeze_panes` | `(행, 열)` 튜플을 지정하여 틀 고정. | `None` |

#### 1.3.2. `to_excel` 활용 예시

**예제용 DataFrame 생성**
```python
df1 = pd.DataFrame({
    '제품': ['A', 'B', 'C'],
    '가격': [15000, 22000, 8000]
})
df2 = pd.DataFrame({
    '직원': ['Kim', 'Lee', 'Park'],
    '부서': ['영업', '개발', '마케팅']
})
```

**1. 기본 저장 (인덱스 포함)**
```python
# 실제 파일로 저장됩니다.
df1.to_excel("basic_output.xlsx", sheet_name="제품정보")
print("basic_output.xlsx 파일이 생성되었습니다 (인덱스 포함).")
```

**2. 인덱스 제외하고 저장**
`index=False`는 데이터를 다른 시스템으로 옮길 때 매우 자주 사용됩니다.
```python
df1.to_excel("no_index_output.xlsx", index=False)
print("no_index_output.xlsx 파일이 생성되었습니다 (인덱스 제외).")
```

**3. 여러 DataFrame을 다른 시트로 저장**
`pd.ExcelWriter`를 사용하면 한 Excel 파일에 여러 시트를 생성할 수 있습니다.
```python
with pd.ExcelWriter('multi_sheet_output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='제품', index=False)
    df2.to_excel(writer, sheet_name='직원', index=False)
print("multi_sheet_output.xlsx 파일에 여러 시트가 저장되었습니다.")
```

**4. 특정 위치부터 쓰고 틀 고정하기**
보고서 형식으로 만들 때 유용합니다.
```python
with pd.ExcelWriter('report_output.xlsx') as writer:
    df1.to_excel(writer, 
                 sheet_name='월간 보고서', 
                 index=False, 
                 startrow=3, 
                 startcol=1,
                 freeze_panes=(4, 2) # 4행과 2열을 기준으로 틀 고정
                )
print("report_output.xlsx 파일이 보고서 형식으로 저장되었습니다.")
```
위 코드는 '월간 보고서' 시트의 B4 셀부터 데이터를 쓰기 시작하며, 4번째 행과 B열을 기준으로 틀을 고정합니다. 보고서 상단에 제목이나 요약 정보를 추가할 공간을 확보할 수 있습니다.

```