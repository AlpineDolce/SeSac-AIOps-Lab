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

JSON(JavaScript Object Notation)은 웹 애플리케이션에서 서버와 클라이언트 간의 데이터 교환을 위해 고안된 경량(lightweight)의 데이터 교환 형식입니다. 그 단순함과 유연성 덕분에 현재는 웹 API의 표준 응답 형식, NoSQL 데이터베이스(예: MongoDB, Couchbase)의 저장 형식, 설정 파일, 로그 데이터 등 다양한 분야에서 광범위하게 사용되고 있습니다.

### 1.1. JSON의 특징 및 장점

*   **인간 친화적 (Human-readable)**: JSON은 텍스트 기반이며, 중괄호 `{}`와 대괄호 `[]`, 그리고 키-값 쌍의 명확한 구조를 사용하여 사람이 읽고 이해하기 매우 쉽습니다.
*   **기계 친화적 (Machine-parseable)**: 구조가 명확하고 단순하여 파싱(parsing) 및 생성 로직 구현이 용이하며, 대부분의 프로그래밍 언어에서 JSON을 처리하기 위한 내장 함수나 라이브러리를 제공합니다.
*   **경량 (Lightweight)**: XML과 같은 다른 데이터 교환 형식에 비해 문법이 간결하고 불필요한 태그가 없어 파일 크기가 작고 전송 속도가 빠릅니다.
*   **계층적 구조 표현 가능**: 객체와 배열을 중첩하여 복잡한 계층적 데이터를 유연하게 표현할 수 있습니다. 이는 관계형 데이터베이스의 평면적인 테이블 구조로는 표현하기 어려운 비정형 또는 반정형 데이터를 다룰 때 큰 장점이 됩니다.
*   **언어 독립적**: JSON은 JavaScript에서 파생되었지만, 특정 프로그래밍 언어에 종속되지 않습니다. 다양한 언어(Python, Java, C++, Ruby, PHP 등)에서 JSON 데이터를 생성하고 파싱할 수 있습니다.

### 1.2. JSON의 기본 구조

JSON 데이터는 크게 두 가지 기본 구조로 구성됩니다.

#### 1.2.1. 객체 (Object)

*   중괄호 `{}`로 표현됩니다.
*   순서가 없는 **키(Key)**와 **값(Value)**의 쌍으로 구성됩니다. 키는 반드시 문자열이어야 하며, 큰따옴표로 묶여야 합니다. 키와 값은 콜론(`:`)으로 구분되며, 각 키-값 쌍은 쉼표(`,`)로 구분됩니다.
*   Python의 딕셔너리(dictionary)와 매우 유사합니다.

**예시:**
```json
{
  "name": "Alice",
  "age": 30,
  "isStudent": false,
  "courses": ["Math", "Science"],
  "address": {
    "street": "123 Main St",
    "city": "New York"
  }
}
```

#### 1.2.2. 배열 (Array)

*   대괄호 `[]`로 표현됩니다.
*   순서가 있는 값들의 목록입니다. 각 값은 쉼표(`,`)로 구분됩니다.
*   Python의 리스트(list)와 매우 유사합니다.

**예시:**
```json
[
  "apple",
  "banana",
  "cherry",
  {
    "fruit_id": 1,
    "color": "red"
  }
]
```

### 1.3. JSON 값의 타입

JSON에서 값(Value)으로 올 수 있는 데이터 타입은 다음과 같습니다:

*   **문자열 (String)**: 큰따옴표 `""`로 묶인 유니코드 문자열 (예: `"Hello, World!"`)
*   **숫자 (Number)**: 정수 또는 부동 소수점 숫자 (예: `123`, `3.14`, `-5`)
*   **불리언 (Boolean)**: `true` 또는 `false`
*   **`null`**: 값이 없음을 나타내는 특별한 값
*   **객체 (Object)**: 중첩된 JSON 객체
*   **배열 (Array)**: 중첩된 JSON 배열

### 1.4. 데이터 과학에서의 JSON 활용

데이터 과학 분야에서 JSON은 다음과 같은 경우에 주로 활용됩니다:

*   **웹 API 연동**: 대부분의 웹 서비스 API는 JSON 형식으로 데이터를 주고받습니다. Pandas의 `read_json()`을 사용하여 API 응답을 `DataFrame`으로 쉽게 변환할 수 있습니다.
*   **로그 데이터 분석**: 웹 서버 로그, 애플리케이션 로그 등이 JSON Lines(JSONL) 형식으로 저장되는 경우가 많습니다. 각 줄이 하나의 JSON 객체인 이러한 파일은 Pandas로 효율적으로 처리할 수 있습니다.
*   **NoSQL 데이터베이스 연동**: MongoDB와 같은 문서 지향(Document-oriented) NoSQL 데이터베이스는 JSON과 유사한 BSON(Binary JSON) 형식으로 데이터를 저장합니다. 이 데이터를 Pandas로 불러와 분석할 수 있습니다.
*   **반정형 데이터 처리**: 정형화되지 않은 복잡한 데이터를 다룰 때 JSON의 유연한 구조가 유용합니다. Pandas의 `json_normalize()`와 같은 함수를 사용하여 중첩된 JSON을 평탄화하여 분석 가능한 형태로 만들 수 있습니다.
*   **설정 파일**: 복잡한 설정이나 파라미터를 JSON 형식으로 저장하고, 이를 Pandas로 불러와 관리할 수 있습니다.

Pandas는 이러한 JSON 데이터의 특성을 이해하고, 이를 `DataFrame`으로 효율적으로 변환하거나 `DataFrame`을 JSON으로 내보내는 강력한 기능을 제공하여 데이터 과학자의 생산성을 높여줍니다.

## 2. JSON 파일/문자열 불러오기 (`read_json`)

Pandas의 `read_json()` 함수는 JSON 형식의 데이터를 `DataFrame`으로 쉽게 불러올 수 있게 해주는 핵심적인 도구입니다. 이 함수는 로컬 파일 경로, 웹 URL, 또는 Python 문자열 형태의 JSON 데이터를 입력으로 받아 Pandas `DataFrame` 객체로 변환합니다. JSON 데이터의 다양한 구조를 유연하게 처리할 수 있도록 여러 옵션을 제공합니다.

### 2.1. `read_json()` 함수 개요

`read_json(path_or_buffer, orient=None, typ='frame', dtype=None, convert_dates=True, lines=False, encoding=None, **kwargs)`

*   **`path_or_buffer` (필수)**: 읽어올 JSON 데이터의 소스를 지정합니다. 다음 중 하나가 될 수 있습니다:
    *   JSON 파일의 경로(문자열, 예: `'data.json'`, `'http://example.com/data.json'`)
    *   파일과 유사한 객체(예: `io.StringIO` 또는 `io.BytesIO` 객체)
    *   JSON 문자열 자체
*   **`orient` (선택)**: JSON 데이터의 구조를 Pandas가 `DataFrame`으로 어떻게 해석할지 지정하는 가장 중요한 옵션입니다. JSON 데이터가 어떤 방식으로 구성되어 있는지에 따라 적절한 `orient` 값을 선택해야 합니다. 주요 옵션은 다음과 같습니다:
    *   `'columns'` (기본값): JSON 객체의 키가 `DataFrame`의 컬럼명이고, 해당 키의 값이 리스트 형태일 때 사용합니다. (예: `{"col1": [1, 2], "col2": [3, 4]}`)
    *   `'index'`: JSON 객체의 키가 `DataFrame`의 인덱스이고, 해당 키의 값이 또 다른 객체(행 데이터)일 때 사용합니다. (예: `{"idx1": {"col1": 1}, "idx2": {"col1": 2}}`)
    *   `'records'`: JSON 데이터가 딕셔너리들의 리스트 형태일 때 사용합니다. 각 딕셔너리가 `DataFrame`의 한 행이 됩니다. 웹 API 응답에서 가장 흔히 볼 수 있는 형식입니다. (예: `[{"col1": 1}, {"col1": 2}]`)
    *   `'split'`: JSON 데이터가 `{"index": [...], "columns": [...], "data": [[...]]}`와 같은 특정 구조를 가질 때 사용합니다.
    *   `'values'`: JSON 데이터가 값들의 리스트(배열) 형태일 때 사용합니다. (예: `[[1, 2], [3, 4]]`)
*   **`typ` (선택)**: 반환할 객체의 타입을 지정합니다. `'frame'` (DataFrame, 기본값) 또는 `'series'`를 지정할 수 있습니다.
*   **`lines` (선택)**: JSON 파일이 JSON Lines(JSONL) 형식인지 여부를 지정합니다. `True`로 설정하면 파일의 각 줄을 별도의 JSON 객체로 파싱합니다. 로그 파일 등에서 자주 사용됩니다.
*   **`dtype` (선택)**: 각 컬럼의 데이터 타입을 명시적으로 지정합니다. 딕셔너리 형태로 `{컬럼명: 데이터타입}`을 지정합니다.
*   **`convert_dates` (선택)**: 날짜/시간으로 보이는 문자열을 자동으로 Pandas `datetime` 객체로 변환할지 여부를 지정합니다. 기본값은 `True`입니다.
*   **`encoding` (선택)**: JSON 파일의 문자 인코딩 방식을 지정합니다. 기본값은 `'utf-8'`입니다.

### 2.2. `read_json()` 활용 예시

다음 예시들을 통해 `read_json()` 함수의 다양한 `orient` 옵션과 `lines` 옵션의 활용법을 자세히 살펴보겠습니다. `io.StringIO`를 사용하여 문자열 데이터를 파일처럼 읽는 방식을 사용합니다.

```python
import pandas as pd
import io
```

**1. `orient='records'` (가장 일반적인 JSON 형식)**
JSON 데이터가 딕셔너리(객체)들의 리스트 형태로 되어 있을 때 사용합니다. 각 딕셔너리가 `DataFrame`의 한 행이 됩니다. 웹 API 응답에서 가장 흔히 볼 수 있는 형식입니다.

**샘플 JSON (`data_records.json`)**
```json
[
  {"id": 1, "name": "Alice", "age": 30},
  {"id": 2, "name": "Bob", "age": 25},
  {"id": 3, "name": "Charlie", "age": 35}
]
```

**코드**
```python
json_data_records = '''
[
  {"id": 1, "name": "Alice", "age": 30},
  {"id": 2, "name": "Bob", "age": 25},
  {"id": 3, "name": "Charlie", "age": 35}
]
'''
df_records = pd.read_json(io.StringIO(json_data_records), orient='records')
print("---\\n1. records orient로 읽은 DataFrame\n---")
print(df_records)
```
**결과:**
```
---
1. records orient로 읽은 DataFrame
---
   id     name  age
0   1    Alice   30
1   2      Bob   25
2   3  Charlie   35
```
**결과 설명**
*   각 JSON 객체가 `DataFrame`의 개별 행으로 변환되었고, 객체의 키(`id`, `name`, `age`)가 컬럼명으로 사용되었습니다.

**2. `orient='columns'` (기본값)**
JSON 객체의 키가 `DataFrame`의 컬럼명이고, 해당 키의 값이 리스트 형태일 때 사용합니다. Pandas의 기본 `orient` 값입니다.

**샘플 JSON (`data_columns.json`)**
```json
{
  "id": [1, 2, 3],
  "name": ["Alice", "Bob", "Charlie"],
  "age": [30, 25, 35]
}
```

**코드**
```python
json_data_columns = '''
{
  "id": [1, 2, 3],
  "name": ["Alice", "Bob", "Charlie"],
  "age": [30, 25, 35]
}
'''
df_columns = pd.read_json(io.StringIO(json_data_columns), orient='columns')
print("\n--- 2. columns orient로 읽은 DataFrame\n---")
print(df_columns)
```
**결과:**
```
--- 2. columns orient로 읽은 DataFrame ---
   id     name  age
0   1    Alice   30
1   2      Bob   25
2   3  Charlie   35
```
**결과 설명**
*   JSON 객체의 키(`id`, `name`, `age`)가 컬럼명으로, 해당 키에 연결된 리스트 값들이 각 컬럼의 데이터로 변환되었습니다.

**3. `orient='index'`**
JSON 객체의 키가 `DataFrame`의 인덱스이고, 해당 키의 값이 또 다른 객체(행 데이터)일 때 사용합니다.

**샘플 JSON (`data_index.json`)**
```json
{
  "user_001": {"name": "Alice", "age": 30},
  "user_002": {"name": "Bob", "age": 25}
}
```

**코드**
```python
json_data_index = '''
{
  "user_001": {"name": "Alice", "age": 30},
  "user_002": {"name": "Bob", "age": 25}
}
'''
df_index = pd.read_json(io.StringIO(json_data_index), orient='index')
print("\n--- 3. index orient로 읽은 DataFrame\n---")
print(df_index)
```
**결과:**
```
--- 3. index orient로 읽은 DataFrame ---
          name  age
user_001  Alice   30
user_002    Bob   25
```
**결과 설명**
*   JSON 객체의 최상위 키(`user_001`, `user_002`)가 `DataFrame`의 인덱스로, 내부 객체의 키(`name`, `age`)가 컬럼명으로 사용되었습니다.

**4. `lines=True` (JSON Lines 형식 읽기)**
JSON Lines(JSONL) 형식은 각 줄이 독립적인 JSON 객체로 구성된 파일 형식입니다. 로그 데이터나 스트리밍 데이터에서 자주 사용됩니다. `lines=True` 옵션을 사용하여 각 줄을 별도의 레코드로 파싱합니다.

**샘플 JSONL (`data.jsonl`)**
```json
{"id": 1, "name": "Alice", "event": "login"}
{"id": 2, "name": "Bob", "event": "logout"}
{"id": 3, "name": "Charlie", "event": "purchase"}
```

**코드**
```python
json_data_lines = '''
{"id": 1, "name": "Alice", "event": "login"}
{"id": 2, "name": "Bob", "event": "logout"}
{"id": 3, "name": "Charlie", "event": "purchase"}
'''
df_lines = pd.read_json(io.StringIO(json_data_lines), lines=True)
print("\n--- 4. JSON Lines 형식으로 읽은 DataFrame (lines=True)\n---")
print(df_lines)
```
**결과:**
```
--- 4. JSON Lines 형식으로 읽은 DataFrame (lines=True) ---
   id     name     event
0   1    Alice     login
1   2      Bob    logout
2   3  Charlie  purchase
```
**결과 설명**
*   각 줄이 독립적인 JSON 객체로 파싱되어 `DataFrame`의 개별 행으로 변환되었습니다.

**5. `dtype` 및 `convert_dates` 옵션 활용**
데이터 타입을 명시적으로 지정하거나, 날짜/시간 문자열을 `datetime` 객체로 변환할 수 있습니다.

**샘플 JSON (`data_types_dates.json`)**
```json
[
  {"item_id": "001", "value": 10.5, "timestamp": "2023-01-01T10:00:00"},
  {"item_id": "002", "value": 20.3, "timestamp": "2023-01-02T11:30:00"}
]
```

**코드**
```python
json_data_types_dates = '''
[
  {"item_id": "001", "value": 10.5, "timestamp": "2023-01-01T10:00:00"},
  {"item_id": "002", "value": 20.3, "timestamp": "2023-01-02T11:30:00"}
]
'''

df_typed_dates = pd.read_json(
    io.StringIO(json_data_types_dates),
    dtype={'item_id': str, 'value': float},
    convert_dates=['timestamp'] # 또는 True로 설정하여 자동 추론
)
print("\n--- 5. dtype 및 convert_dates 옵션 활용 DataFrame\n---")
print(df_typed_dates.info())
print("\n데이터프레임 내용:\n", df_typed_dates)
```
**결과 설명**
*   `item_id`는 문자열로, `value`는 실수형으로 명시적으로 지정되었고, `timestamp` 컬럼은 `datetime64[ns]` 타입으로 성공적으로 변환되었습니다.

`read_json()` 함수는 JSON 데이터의 다양한 구조와 특성을 이해하고, 이를 Pandas `DataFrame`으로 효율적으로 변환하는 데 필수적인 도구입니다. `orient`와 `lines` 옵션을 적절히 활용하면 대부분의 JSON 데이터를 문제없이 처리할 수 있습니다.


## 3. 중첩된 JSON 데이터 처리

실제 세계의 JSON 데이터는 종종 단순한 키-값 쌍이나 평면적인 리스트 형태를 넘어, 객체 안에 또 다른 객체나 배열이 포함되는 **중첩된(Nested) 구조**를 가집니다. 이러한 중첩된 JSON 데이터를 Pandas `DataFrame`으로 직접 불러오면, 중첩된 부분은 딕셔너리나 리스트 형태의 단일 컬럼으로 로드되어 데이터 분석에 어려움이 있습니다. Pandas `DataFrame`은 기본적으로 평탄한(flat) 테이블 구조를 기대하기 때문입니다.

이러한 중첩된 데이터를 효과적으로 분석하기 위해서는 "평탄화(Flattening)" 과정이 필요합니다. Pandas는 이를 위해 `json_normalize()` 함수를 제공하며, 이 함수는 중첩된 JSON 데이터를 `DataFrame`의 개별 컬럼으로 확장하여 평탄한 형태로 만들어줍니다.

### 3.1. `json_normalize()` 함수 개요

`pandas.json_normalize(data, record_path=None, meta=None, meta_prefix=None, sep='.', max_level=None)`

*   **`data` (필수)**: 평탄화할 JSON 데이터입니다. 일반적으로 딕셔너리들의 리스트(JSON 배열) 또는 단일 딕셔너리(JSON 객체) 형태입니다. `read_json()`으로 불러온 `DataFrame`의 특정 컬럼(딕셔너리 또는 리스트를 포함하는)을 직접 전달할 수도 있습니다.
*   **`record_path` (선택)**: 중첩된 리스트(JSON 배열)가 있는 경로를 지정합니다. 이 경로에 있는 배열의 각 요소가 `DataFrame`의 새로운 행으로 확장됩니다. 예를 들어, "orders" 또는 `["user", "address"]`와 같이 경로를 지정할 수 있습니다. `record_path`가 지정되면, 해당 경로의 배열 내 각 객체가 새로운 행이 되고, 그 객체 내의 키-값 쌍이 컬럼으로 변환됩니다.
*   **`meta` (선택)**: `record_path`와 함께 사용될 때, `record_path`로 확장되는 데이터와 함께 유지할 상위 레벨의 메타데이터 컬럼을 지정합니다. 리스트 형태로 컬럼 이름을 지정합니다(예: `['user_id', 'username']`). `meta`에 지정된 컬럼들은 확장된 각 행에 복제되어 추가됩니다.
*   **`sep` (선택)**: 중첩된 키를 평탄화할 때 새로운 컬럼명에 사용할 구분자(delimiter)를 지정합니다. 예를 들어, "info": {"age": 30}이 평탄화될 때 `sep='_'`이면 `info_age` 컬럼이 생성됩니다. 기본값은 마침표(`.`)입니다.
*   **`meta_prefix` (선택)**: `meta` 컬럼에 접두사를 추가하여 컬럼명 충돌을 방지합니다.
*   **`max_level` (선택)**: 중첩된 객체를 평탄화할 최대 깊이를 지정합니다.

### 3.2. `json_normalize()` 활용 예시

다음 예시들을 통해 `json_normalize()` 함수를 사용하여 다양한 형태의 중첩된 JSON 데이터를 평탄화하는 방법을 살펴보겠습니다.

```python
import pandas as pd
from pandas import json_normalize # json_normalize는 pandas 모듈에서 직접 임포트하는 것이 일반적
import io
```

**1. 간단한 중첩 객체 평탄화 (단일 레벨 중첩)**
JSON 데이터 내에 딕셔너리(객체)가 중첩되어 있는 가장 기본적인 형태입니다. `record_path` 없이 `json_normalize()`를 호출하면 최상위 레벨의 객체와 중첩된 객체의 키들이 평탄화됩니다.

**샘플 JSON (`data_nested_simple.json`)**
```json
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
```

**코드**
```python
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

# 먼저 read_json으로 DataFrame으로 불러옵니다. 이때 'info' 컬럼은 딕셔너리 형태가 됩니다.
df_nested_simple = pd.read_json(io.StringIO(json_nested_simple))
print("---\\n1. 원본 중첩 DataFrame (info 컬럼이 딕셔너리)
---")
print(df_nested_simple)
print("info 컬럼의 타입:", type(df_nested_simple['info'][0]))

# json_normalize를 사용하여 전체 DataFrame을 평탄화합니다.
# record_path를 지정하지 않으면, 모든 중첩된 객체가 평탄화됩니다.
df_flattened_full = json_normalize(pd.read_json(io.StringIO(json_nested_simple)))
print("\n--- 1.1. json_normalize로 전체 평탄화 (기본 sep='.')\n---")
print(df_flattened_full)

# sep 옵션 변경 예시
df_flattened_sep = json_normalize(pd.read_json(io.StringIO(json_nested_simple)), sep='_')
print("\n--- 1.2. json_normalize로 전체 평탄화 (sep='_')\n---")
print(df_flattened_sep)
```
**결과 설명**
*   `df_nested_simple`에서는 `info` 컬럼이 딕셔너리 객체로 저장되어 있습니다.
*   `df_flattened_full`에서는 `info` 딕셔너리 내부의 `age`와 `city` 키가 `info.age`, `info.city`와 같은 새로운 컬럼으로 평탄화되었습니다.
*   `sep='_'`를 사용하면 컬럼명이 `info_age`, `info_city`와 같이 생성됩니다.

**2. `record_path`와 `meta`를 사용한 중첩 리스트 평탄화**
JSON 데이터 내에 "레코드(record)"의 리스트(배열)가 중첩되어 있고, 이 레코드들을 개별 행으로 확장하면서 상위 레벨의 메타데이터를 함께 유지하고 싶을 때 `record_path`와 `meta` 옵션을 사용합니다. 이는 "1 대 다(One-to-Many)" 관계의 데이터를 평탄화할 때 매우 유용합니다.

**샘플 JSON (`data_nested_list.json`)**
```json
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
```

**코드**
```python
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

# record_path='orders': 'orders' 배열 내의 각 객체를 새로운 행으로 확장합니다.
# meta=['user_id', 'username']: 각 주문 레코드에 해당 사용자의 user_id와 username을 복제하여 추가합니다.
# sep='_': 평탄화된 컬럼명에 언더스코어(_)를 구분자로 사용합니다.
df_flattened_list = json_normalize(
    pd.read_json(io.StringIO(json_nested_list)),
    record_path='orders',
    meta=['user_id', 'username'],
    sep='_'
)
print("\n--- 2. record_path와 meta를 사용한 중첩 리스트 평탄화\n---")
print(df_flattened_list)
```
**결과 설명**
*   `orders` 배열 내의 각 주문 객체가 개별 행으로 확장되었습니다.
*   각 주문 행에는 해당 주문을 한 사용자의 `user_id`와 `username`이 `meta` 옵션에 의해 복제되어 추가되었습니다.
*   `order_id`, `item`, `price`와 같은 주문 상세 정보가 새로운 컬럼으로 평탄화되었습니다.

**3. 다단계 중첩 객체 평탄화 (record_path의 리스트 사용)**
JSON 데이터가 여러 단계로 중첩된 객체와 리스트를 포함할 때, `record_path`에 리스트를 전달하여 특정 경로를 따라 들어가 평탄화할 수 있습니다.

**샘플 JSON (`data_multi_nested.json`)**
```json
[
  {
    "company_id": "C1",
    "company_name": "Alpha Corp",
    "departments": [
      {
        "dept_id": "D1",
        "dept_name": "Sales",
        "employees": [
          {"emp_id": "E1", "emp_name": "Alice"},
          {"emp_id": "E2", "emp_name": "Bob"}
        ]
      },
      {
        "dept_id": "D2",
        "dept_name": "Marketing",
        "employees": [
          {"emp_id": "E3", "emp_name": "Charlie"}
        ]
      }
    ]
  }
]
```

**코드**
```python
json_multi_nested = '''
[
  {
    "company_id": "C1",
    "company_name": "Alpha Corp",
    "departments": [
      {
        "dept_id": "D1",
        "dept_name": "Sales",
        "employees": [
          {"emp_id": "E1", "emp_name": "Alice"},
          {"emp_id": "E2", "emp_name": "Bob"}
        ]
      },
      {
        "dept_id": "D2",
        "dept_name": "Marketing",
        "employees": [
          {"emp_id": "E3", "emp_name": "Charlie"}
        ]
      }
    ]
  }
]
'''

# record_path=['departments', 'employees']: 'departments' 배열 안의 'employees' 배열을 확장합니다.
# meta=['company_id', 'company_name', ['departments', 'dept_id'], ['departments', 'dept_name']]:
# 상위 레벨의 company 정보와, 바로 위 부서 레벨의 dept_id, dept_name을 메타데이터로 유지합니다.
# 중첩된 meta 경로를 지정할 때는 리스트 안에 리스트로 표현합니다.
df_multi_flattened = json_normalize(
    pd.read_json(io.StringIO(json_multi_nested)),
    record_path=['departments', 'employees'],
    meta=['company_id', 'company_name', ['departments', 'dept_id'], ['departments', 'dept_name']],
    sep='_'
)
print("\n--- 3. 다단계 중첩 객체 평탄화 (record_path 리스트, 중첩 meta)\n---")
print(df_multi_flattened)
```
**결과 설명**
*   `departments` 배열 안의 `employees` 배열이 평탄화되어 각 직원이 개별 행이 되었습니다.
*   `company_id`, `company_name`과 함께 각 직원이 속한 부서의 `dept_id`와 `dept_name`이 메타데이터로 정확히 복제되어 추가되었습니다.

`json_normalize()` 함수는 복잡한 중첩 JSON 데이터를 Pandas `DataFrame`의 평탄한 구조로 변환하는 데 필수적인 도구입니다. `record_path`와 `meta` 옵션을 적절히 활용하면 다양한 형태의 중첩 데이터를 효과적으로 분석 가능한 형태로 만들 수 있습니다.


## 4. DataFrame을 JSON으로 저장 (`to_json`)

데이터 분석 및 처리 과정에서 생성되거나 수정된 Pandas `DataFrame`을 JSON 형식으로 내보내는 것은 웹 서비스 API의 응답을 생성하거나, NoSQL 데이터베이스에 데이터를 저장하거나, 다른 시스템과 데이터를 교환할 때 매우 중요합니다. `DataFrame` 객체의 `to_json()` 메서드는 이러한 작업을 효율적으로 수행할 수 있도록 다양한 옵션을 제공하며, JSON 출력 형식을 세밀하게 제어할 수 있습니다.

### 4.1. `to_json()` 메서드 개요

`DataFrame.to_json(path_or_buf=None, orient=None, date_format='iso', double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', indent=None, storage_options=None)`

*   **`path_or_buf` (선택)**: JSON 데이터를 저장할 파일 경로(문자열, 예: `'output.json'`)를 지정하거나, `io.StringIO`와 같은 파일과 유사한 객체를 전달하여 파일 시스템에 실제로 저장하지 않고 문자열로 결과를 받을 수 있습니다. 생략하면 JSON 문자열을 반환합니다.
*   **`orient` (선택)**: JSON 출력 형식을 지정하는 가장 중요한 옵션입니다. `DataFrame`의 구조를 JSON으로 어떻게 매핑할지 결정합니다. 주요 옵션은 다음과 같습니다:
    *   `'columns'` (기본값): JSON 객체의 키가 `DataFrame`의 컬럼명이고, 해당 키의 값이 리스트 형태일 때 사용합니다. (예: `{"col1": [1, 2], "col2": [3, 4]}`)
    *   `'index'`: JSON 객체의 키가 `DataFrame`의 인덱스이고, 해당 키의 값이 또 다른 객체(행 데이터)일 때 사용합니다. (예: `{"idx1": {"col1": 1}, "idx2": {"col1": 2}}`)
    *   `'records'`: JSON 데이터가 딕셔너리들의 리스트 형태일 때 사용합니다. 각 딕셔너리가 `DataFrame`의 한 행이 됩니다. 웹 API 응답에서 가장 흔히 사용되는 형식입니다. (예: `[{"col1": 1}, {"col1": 2}]`)
    *   `'split'`: JSON 데이터가 `{"index": [...], "columns": [...], "data": [[...]]}`와 같은 특정 구조를 가질 때 사용합니다. 인덱스, 컬럼, 데이터가 분리되어 표현됩니다.
    *   `'values'`: JSON 데이터가 값들의 리스트(배열) 형태일 때 사용합니다. 컬럼명이나 인덱스 정보 없이 순수 데이터 값만 포함합니다. (예: `[[1, 2], [3, 4]]`)
    *   `'table'`: JSON 데이터를 [Table Schema](https://specs.frictionlessdata.io/tabular-data-resource/#table-schema) 형식으로 저장합니다. 스키마 정보와 데이터가 함께 포함되어 데이터의 재해석이 용이합니다.
*   **`date_format` (선택)**: 날짜/시간(`datetime`) 데이터의 출력 형식을 지정합니다.
    *   `'iso'` (기본값): ISO 8601 형식의 문자열로 출력합니다 (예: `"2023-01-01T10:00:00.000Z"`).
    *   `'epoch'`: Unix epoch 시간(1970년 1월 1일 00:00:00 UTC부터의 초 또는 밀리초)으로 출력합니다. `date_unit`과 함께 사용됩니다.
*   **`double_precision` (선택)**: 부동 소수점 숫자의 정밀도를 지정합니다. 기본값은 10입니다.
*   **`force_ascii` (선택)**: ASCII가 아닌 문자를 `\uXXXX` 형태로 이스케이프할지 여부를 지정합니다. `True` (기본값)로 설정하면 한글 등이 이스케이프됩니다. `False`로 설정하면 원본 문자로 출력됩니다 (파일 저장 시 유용).
*   **`lines` (선택)**: `True`로 설정하면 각 행을 별도의 JSON 객체로 한 줄에 하나씩 출력합니다 (JSON Lines 형식). `orient='records'`와 함께 사용될 때 가장 일반적입니다.
*   **`indent` (선택)**: JSON 출력의 들여쓰기 수준을 지정합니다. 정수 값을 지정하면 해당 칸만큼 들여쓰기하여 JSON의 가독성을 높입니다. 파일 저장 시 유용합니다.

### 4.2. `to_json()` 활용 예시

다음 예시들을 통해 `to_json()` 메서드의 다양한 `orient` 옵션과 기타 유용한 옵션들의 활용법을 살펴보겠습니다. 예시를 위해 `DataFrame`을 생성하고, 결과를 문자열로 출력하거나 실제 파일로 저장합니다.

```python
import pandas as pd
import io
import os # 파일 삭제를 위해 os 모듈 임포트

# 예시 DataFrame 생성
df_to_json = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Keyboard'],
    'price': [1200, 25, 75],
    'available': [True, True, False],
    'last_updated': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10'])
})
print("---\n원본 DataFrame\n---")
print(df_to_json)
print("\n원본 DataFrame 정보:\n")
df_to_json.info()
```

**원본 DataFrame:**
```
--- 원본 DataFrame ---
    product  price  available last_updated
0    Laptop   1200       True   2023-01-01
1     Mouse     25       True   2023-01-05
2  Keyboard     75      False   2023-01-10

원본 DataFrame 정보:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype         
---  ------        --------------  -----         
 0   product       3 non-null      object        
 1   price         3 non-null      int64         
 2   available     3 non-null      bool          
 3   last_updated  3 non-null      datetime64[ns]
dtypes: bool(1), datetime64[ns](1), int64(1), object(1)
memory usage: 263.0+ bytes
```

**1. `orient='columns'` (기본값)**
`DataFrame`의 컬럼을 JSON 객체의 키로, 각 컬럼의 값 리스트를 해당 키의 값으로 매핑합니다. 이는 `DataFrame`을 컬럼 지향적으로 표현할 때 사용됩니다.

**코드**
```python
json_output_columns = df_to_json.to_json(orient='columns')
print("\n--- 1. orient='columns' (기본값) ---")
print(json_output_columns)
```
**결과:**
```json
--- 1. orient='columns' (기본값) ---
{"product":{"0":"Laptop","1":"Mouse","2":"Keyboard"},"price":{"0":1200,"1":25,"2":75},"available":{"0":true,"1":true,"2":false},"last_updated":{"0":1672531200000,"1":1672876800000,"2":1673222400000}}
```
**결과 설명**
*   각 컬럼명(`product`, `price` 등)이 최상위 키가 되고, 그 값으로는 `DataFrame`의 인덱스를 키로 하는 객체가 포함됩니다. 날짜는 기본적으로 epoch 밀리초로 변환됩니다.

**2. `orient='records'` (가장 일반적인 API 형식)**
각 `DataFrame` 행을 하나의 JSON 객체로 표현하고, 이 객체들을 리스트로 묶습니다. 웹 API 응답이나 문서 지향 데이터베이스에 데이터를 저장할 때 가장 널리 사용되는 형식입니다.

**코드**
```python
json_output_records = df_to_json.to_json(orient='records')
print("\n--- 2. orient='records' ---")
print(json_output_records)
```
**결과:**
```json
--- 2. orient='records' ---
[{"product":"Laptop","price":1200,"available":true,"last_updated":1672531200000},{"product":"Mouse","price":25,"available":true,"last_updated":1672876800000},{"product":"Keyboard","price":75,"available":false,"last_updated":1673222400000}]
```
**결과 설명**
*   `DataFrame`의 각 행이 독립적인 JSON 객체로 변환되어 리스트 안에 포함됩니다. 각 객체의 키는 컬럼명입니다.

**3. `orient='index'`**
`DataFrame`의 인덱스를 JSON 객체의 키로 사용하고, 각 인덱스에 해당하는 행 데이터를 또 다른 JSON 객체로 매핑합니다.

**코드**
```python
json_output_index = df_to_json.to_json(orient='index')
print("\n--- 3. orient='index' ---")
print(json_output_index)
```
**결과:**
```json
--- 3. orient='index' ---
{"0":{"product":"Laptop","price":1200,"available":true,"last_updated":1672531200000},"1":{"product":"Mouse","price":25,"available":true,"last_updated":1672876800000},"2":{"product":"Keyboard","price":75,"available":false,"last_updated":1673222400000}}
```
**결과 설명**
*   `DataFrame`의 인덱스(0, 1, 2)가 최상위 키가 되고, 각 키의 값으로는 해당 행의 컬럼명과 데이터가 포함된 객체가 됩니다.

**4. `orient='split'`**
JSON 데이터를 `{"index": [...], "columns": [...], "data": [[...]]}`와 같이 인덱스, 컬럼, 실제 데이터 값을 분리하여 표현합니다. 이는 데이터의 구조를 명확하게 전달할 때 유용합니다.

**코드**
```python
json_output_split = df_to_json.to_json(orient='split')
print("\n--- 4. orient='split' ---")
print(json_output_split)
```
**결과:**
```json
--- 4. orient='split' ---
{"index":[0,1,2],"columns":["product","price","available","last_updated"],"data":[["Laptop",1200,true,1672531200000],["Mouse",25,true,1672876800000],["Keyboard",75,false,1673222400000]]}
```
**결과 설명**
*   `index`, `columns`, `data`라는 세 개의 최상위 키로 데이터가 구조화됩니다.

**5. `orient='values'`**
`DataFrame`의 값들만 중첩된 리스트(배열) 형태로 출력합니다. 컬럼명이나 인덱스 정보는 포함되지 않습니다. 순수 데이터 값만 필요할 때 사용합니다.

**코드**
```python
json_output_values = df_to_json.to_json(orient='values')
print("\n--- 5. orient='values' ---")
print(json_output_values)
```
**결과:**
```json
--- 5. orient='values' ---
[["Laptop",1200,true,1672531200000],["Mouse",25,true,1672876800000],["Keyboard",75,false,1673222400000]]
```
**결과 설명**
*   `DataFrame`의 값들만 행 단위로 묶인 리스트의 리스트 형태로 출력됩니다.

**6. `lines=True` (JSON Lines 형식으로 저장)**
`lines=True` 옵션을 사용하면 각 `DataFrame` 행을 별도의 JSON 객체로 변환하여 한 줄에 하나씩 출력합니다. 이는 `orient='records'`와 함께 사용될 때 가장 일반적이며, 대용량 로그 파일이나 스트리밍 데이터를 처리할 때 유용합니다.

**코드**
```python
json_output_lines = df_to_json.to_json(orient='records', lines=True)
print("\n--- 6. lines=True (JSON Lines 형식) ---")
print(json_output_lines)
```
**결과:**
```json
--- 6. lines=True (JSON Lines 형식) ---
{"product":"Laptop","price":1200,"available":true,"last_updated":1672531200000}
{"product":"Mouse","price":25,"available":true,"last_updated":1672876800000}
{"product":"Keyboard","price":75,"available":false,"last_updated":1673222400000}
```
**결과 설명**
*   각 행이 독립적인 JSON 객체로 한 줄에 하나씩 출력되어, JSONL 파일로 저장하기에 적합한 형태가 됩니다.

**7. 파일로 저장 및 가독성 향상 (`indent`, `force_ascii=False`, `date_format='iso'`)**
`to_json()`을 사용하여 실제 파일로 저장할 때, `indent` 옵션으로 들여쓰기를 추가하여 JSON 파일의 가독성을 높일 수 있습니다. 또한, `force_ascii=False`를 사용하여 한글과 같은 비ASCII 문자가 이스케이프되지 않고 원본 그대로 저장되도록 할 수 있으며, `date_format='iso'`로 날짜 형식을 표준 ISO 8601로 지정할 수 있습니다.

**코드**
```python
output_file_json = 'output_pretty.json'
# indent=4: 4칸 들여쓰기
# force_ascii=False: 한글 등 비ASCII 문자 이스케이프 안 함
# date_format='iso': 날짜를 ISO 8601 문자열로 저장
df_to_json.to_json(output_file_json, orient='records', indent=4, force_ascii=False, date_format='iso')
print(f"\n7. '{output_file_json}' 파일이 생성되었습니다 (가독성 향상, 한글 포함).")

# 생성된 파일 내용 확인 (선택 사항)
# with open(output_file_json, 'r', encoding='utf-8') as f:
#     print("\n파일 내용:\n", f.read())

# 예시 파일 삭제
os.remove(output_file_json)
print(f"'{output_file_json}' 파일 삭제 완료.")
```
**결과 (output_pretty.json 파일 내용):**
```json
[
    {
        "product": "Laptop",
        "price": 1200,
        "available": true,
        "last_updated": "2023-01-01T00:00:00.000Z"
    },
    {
        "product": "Mouse",
        "price": 25,
        "available": true,
        "last_updated": "2023-01-05T00:00:00.000Z"
    },
    {
        "product": "Keyboard",
        "price": 75,
        "available": false,
        "last_updated": "2023-01-10T00:00:00.000Z"
    }
]
```
**결과 설명**
*   JSON 파일이 들여쓰기되어 가독성이 높아졌고, 날짜가 ISO 형식으로 저장되었습니다. 만약 `product` 컬럼에 한글이 있었다면 `force_ascii=False` 덕분에 깨지지 않고 저장되었을 것입니다.

`to_json()` 메서드는 Pandas `DataFrame`의 데이터를 다양한 JSON 형식으로 유연하게 내보낼 수 있는 강력한 기능을 제공합니다. 웹 서비스 연동, 데이터 교환, 설정 파일 생성 등 다양한 시나리오에서 이 옵션들을 적절히 활용하는 것이 중요합니다.
