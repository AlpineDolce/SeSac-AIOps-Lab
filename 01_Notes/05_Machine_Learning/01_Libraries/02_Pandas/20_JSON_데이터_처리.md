<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 Pandas를 사용하여 JSON(JavaScript Object Notation) 형식의 데이터를 효율적으로 처리하는 방법을 다룹니다. JSON 파일 또는 문자열을 DataFrame으로 불러오고, DataFrame을 JSON 형식으로 저장하는 방법을 학습합니다. 특히 중첩된 JSON 데이터를 평탄화(flattening)하는 기법을 실습합니다.

<h2>목차</h2>

- [목차](#목차)
- [1. JSON 데이터 개요](#1-json-데이터-개요)
- [2. JSON 파일/문자열 불러오기 (`read_json`)](#2-json-파일문자열-불러오기-read_json)
- [3. 중첩된 JSON 데이터 처리](#3-중첩된-json-데이터-처리)
- [4. DataFrame을 JSON으로 저장 (`to_json`)](#4-dataframe을-json으로-저장-to_json)

---

## 1. JSON 데이터 개요

JSON(JavaScript Object Notation)은 경량의 데이터 교환 형식으로, 사람이 읽고 쓰기 쉬우며 기계가 파싱하고 생성하기 용이합니다. 웹 애플리케이션에서 서버와 클라이언트 간의 데이터 전송, RESTful API의 응답, NoSQL 데이터베이스(예: MongoDB)의 저장 형식 등으로 널리 사용됩니다.

**JSON의 기본 구조:**
*   **객체 (Object)**: 중괄호 `{}`로 표현되며, 순서 없는 키(문자열)와 값의 쌍으로 구성됩니다. (Python의 딕셔너리와 유사)
    ```json
    {
      "name": "Alice",
      "age": 30,
      "isStudent": false
    }
    ```
*   **배열 (Array)**: 대괄호 `[]`로 표현되며, 순서 있는 값들의 목록입니다. (Python의 리스트와 유사)
    ```json
    [
      "apple",
      "banana",
      "cherry"
    ]
    ```
값으로는 문자열, 숫자, 불리언, null, 다른 객체, 다른 배열이 올 수 있습니다.

## 2. JSON 파일/문자열 불러오기 (`read_json`)

Pandas의 `read_json()` 함수는 JSON 형식의 데이터를 `DataFrame`으로 쉽게 불러올 수 있게 해줍니다. 파일 경로, URL, 또는 JSON 문자열을 직접 입력으로 받을 수 있습니다.

**주요 `read_json()` 옵션:**

| 옵션 |
| --- | --- |
| `path_or_buffer` | JSON 파일 경로, URL, 또는 JSON 문자열. |
| `orient` | JSON 데이터의 구조를 Pandas가 어떻게 해석할지 지정. (`'columns'`, `'index'`, `'records'`, `'split'`, `'values'`). `'records'`가 가장 일반적. |
| `typ` | 반환할 객체 타입. `'frame'` (DataFrame, 기본값) 또는 `'series'`. |
| `lines` | 각 라인이 별도의 JSON 객체인지 여부 (JSON Lines 형식). |
| `dtype` | 컬럼별 데이터 타입을 딕셔너리로 지정. |
| `convert_dates` | 날짜/시간 문자열을 `datetime` 객체로 변환할지 여부. |

```python
import pandas as pd
import io

# 예시 1: 간단한 JSON 문자열 (records orient)
json_data_records = '''
[
  {"id": 1, "name": "Alice", "age": 30},
  {"id": 2, "name": "Bob", "age": 25},
  {"id": 3, "name": "Charlie", "age": 35}
]
'''
df_records = pd.read_json(io.StringIO(json_data_records), orient='records')
print("---", "records orient로 읽은 DataFrame", "---")
print(df_records)

# 예시 2: JSON 문자열 (columns orient - 기본값)
# 키가 컬럼명, 값이 해당 컬럼의 값 리스트
json_data_columns = '''
{
  "id": [1, 2, 3],
  "name": ["Alice", "Bob", "Charlie"],
  "age": [30, 25, 35]
}
'''
df_columns = pd.read_json(io.StringIO(json_data_columns), orient='columns')
print("\n---", "columns orient로 읽은 DataFrame", "---")
print(df_columns)

# 예시 3: JSON Lines (JSONL) 형식 읽기 (lines=True)
json_data_lines = '''
{"id": 1, "name": "Alice"}
{"id": 2, "name": "Bob"}
{"id": 3, "name": "Charlie"}
'''
df_lines = pd.read_json(io.StringIO(json_data_lines), lines=True)
print("\n---", "JSON Lines 형식으로 읽은 DataFrame", "---")
print(df_lines)

# 파일에서 읽기 (예시, 실제 파일이 있어야 함)
# df_from_file = pd.read_json('data.json', orient='records')
```

## 3. 중첩된 JSON 데이터 처리

실제 JSON 데이터는 종종 중첩된 구조를 가집니다. Pandas `DataFrame`은 기본적으로 평탄한(flat) 구조를 기대하므로, 중첩된 데이터를 처리하기 위한 특별한 기법이 필요합니다. `json_normalize()` 함수는 이러한 중첩된 JSON 데이터를 평탄화하는 데 매우 유용합니다.

**`pd.json_normalize()` 함수:**
`json_normalize()`는 `pandas.json_normalize` 모듈에 포함되어 있으며, 중첩된 리스트나 딕셔너리를 `DataFrame`으로 변환하는 데 사용됩니다.

**주요 `json_normalize()` 옵션:**

| 옵션 |
| --- | --- |
| `data` | 평탄화할 JSON 데이터 (딕셔너리 또는 딕셔너리 리스트). |
| `record_path` | 중첩된 리스트가 있는 경로. 이 경로의 리스트를 행으로 확장합니다. |
| `meta` | `record_path`와 함께 유지할 상위 레벨의 메타데이터 컬럼. |
| `sep` | 중첩된 키를 평탄화할 때 컬럼명에 사용할 구분자. | `.` |

```python
# 예시 1: 간단한 중첩 JSON
json_nested_simple = '''
[
  {
    "id": 1,
    "name": "Alice",
    "info": {"age": 30, "city": "New York"}
  },
  {
    "id": 2,
    "name": "Bob",
    "info": {"age": 25, "city": "London"}
  }
]
'''
df_nested_simple = pd.read_json(io.StringIO(json_nested_simple))
print("---", "원본 중첩 DataFrame (info 컬럼이 딕셔너리)", "---")
print(df_nested_simple)

# json_normalize를 사용하여 'info' 컬럼 평탄화
from pandas import json_normalize

df_flattened_simple = json_normalize(df_nested_simple['info'])
print("\n---", "'info' 컬럼만 평탄화", "---")
print(df_flattened_simple)

# 원본 DataFrame과 병합 (id, name 컬럼을 유지하면서 info 평탄화)
df_combined_simple = pd.concat([df_nested_simple[['id', 'name']], df_flattened_simple], axis=1)
print("\n---", "원본과 평탄화된 info 병합", "---")
print(df_combined_simple)

# 예시 2: record_path와 meta를 사용한 중첩 리스트 평탄화
json_nested_list = '''
[
  {
    "user_id": "U1",
    "username": "Alice",
    "orders": [
      {"order_id": "O1", "item": "Laptop", "price": 1200},
      {"order_id": "O2", "item": "Mouse", "price": 25}
    ]
  },
  {
    "user_id": "U2",
    "username": "Bob",
    "orders": [
      {"order_id": "O3", "item": "Keyboard", "price": 75}
    ]
  }
]
'''

df_flattened_list = json_normalize(
    pd.read_json(io.StringIO(json_nested_list)),
    record_path='orders',
    meta=['user_id', 'username'],
    sep='_' # 컬럼명 구분자 지정
)
print("\n---", "record_path와 meta를 사용한 평탄화", "---")
print(df_flattened_list)
```

## 4. DataFrame을 JSON으로 저장 (`to_json`)

`DataFrame`을 JSON 형식의 문자열이나 파일로 저장할 때는 `to_json()` 메서드를 사용합니다. `orient` 옵션을 통해 다양한 JSON 출력 형식을 제어할 수 있습니다.

**주요 `to_json()` 옵션:**

| 옵션 |
| --- | --- |
| `path_or_buf` | 저장할 파일 경로 또는 버퍼. |
| `orient` | JSON 출력 형식. (`'columns'`, `'index'`, `'records'`, `'split'`, `'values'`, `'table'`). |
| `date_format` | 날짜/시간 데이터의 출력 형식. (`'iso'`, `'epoch'`). |
| `double_precision` | 부동 소수점 숫자의 정밀도. |
| `force_ascii` | ASCII가 아닌 문자를 이스케이프할지 여부. |
| `lines` | 각 행을 별도의 JSON 객체로 출력할지 여부 (JSON Lines 형식). |

```python
# 예시 DataFrame 생성
df_to_json = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Keyboard'],
    'price': [1200, 25, 75],
    'available': [True, True, False],
    'last_updated': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10'])
})
print("---", "원본 DataFrame", "---")
print(df_to_json)

# 1. 기본 저장 (orient='columns')
# 컬럼을 키로, 해당 컬럼의 값 리스트를 값으로 가짐
json_output_columns = df_to_json.to_json(orient='columns')
print("\n---", "orient='columns'", "---")
print(json_output_columns)

# 2. records orient (가장 일반적이고 API에서 많이 사용)
# 각 행이 하나의 JSON 객체로 표현된 리스트
json_output_records = df_to_json.to_json(orient='records')
print("\n---", "orient='records'", "---")
print(json_output_records)

# 3. index orient
# 인덱스를 키로, 각 행의 데이터를 객체로 가짐
json_output_index = df_to_json.to_json(orient='index')
print("\n---", "orient='index'", "---")
print(json_output_index)

# 4. JSON Lines 형식으로 저장 (lines=True)
# 각 행이 별도의 JSON 객체로 한 줄에 하나씩 저장
json_output_lines = df_to_json.to_json(orient='records', lines=True)
print("\n---", "lines=True (JSONL)", "---")
print(json_output_lines)

# 파일로 저장 (예시, 실제 파일이 생성됨)
# df_to_json.to_json('output.json', orient='records', indent=4) # indent로 가독성 높임
```
