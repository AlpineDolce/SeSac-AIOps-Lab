<h2>실무 데이터 처리: 파일, API, 환경변수</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-12

<h2>문서 목표</h2>
<p>이 문서는 Python을 활용하여 <strong>실무 데이터를 처리하는 다양한 기법</strong>에 대해 심도 있게 다룹니다. 파일 입출력(File I/O)을 통한 로컬 데이터 관리, CSV 및 JSON과 같은 표준 데이터 형식 처리, 외부 API 연동을 통한 웹 데이터 활용, 환경 변수를 이용한 설정 관리, 그리고 파일 시스템 제어 방법을 상세한 예제와 함께 설명합니다. 이를 통해 파이썬으로 실제 데이터를 효과적으로 수집, 저장, 처리하는 능력을 기르는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 파일 입출력 (File I/O)](#1-파일-입출력-file-io)
  - [1.1 `with open(...)`을 사용한 안전한 파일 처리](#11-with-open을-사용한-안전한-파일-처리)
  - [1.2 파일 열기 모드 (`'r'`, `'w'`, `'a'`, `'x'`, `'b'`, `'t'`, `'+'`)](#12-파일-열기-모드-r-w-a-x-b-t-)
  - [1.3 파일 읽기/쓰기 메서드](#13-파일-읽기쓰기-메서드)
- [2. 표준 데이터 형식 다루기](#2-표준-데이터-형식-다루기)
  - [2.1 CSV 파일 처리: `csv` 모듈 활용](#21-csv-파일-처리-csv-모듈-활용)
  - [2.2 JSON 데이터 처리: `json` 모듈 활용](#22-json-데이터-처리-json-모듈-활용)
  - [2.3 객체 직렬화: `pickle` 모듈](#23-객체-직렬화-pickle-모듈)
- [3. 외부 API 연동: `requests` 라이브러리](#3-외부-api-연동-requests-라이브러리)
  - [3.1 API Rate Limiting (요청 제한) 및 Pagination (페이지네이션)](#31-api-rate-limiting-요청-제한-및-pagination-페이지네이션)
- [4. 데이터 유효성 검사 (Data Validation)](#4-데이터-유효성-검사-data-validation)
  - [4.1 왜 데이터 유효성 검사가 중요한가?](#41-왜-데이터-유효성-검사가-중요한가)
  - [4.2 일반적인 유효성 검사 유형](#42-일반적인-유효성-검사-유형)
  - [4.3 데이터 유효성 검사 구현 예시](#43-데이터-유효성-검사-구현-예시)
  - [4.4 고급 유효성 검사 라이브러리](#44-고급-유효성-검사-라이브러리)
- [5. 설정 관리: 환경 변수와 `python-dotenv`](#5-설정-관리-환경-변수와-python-dotenv)
- [6. 파일 시스템 제어 (`os`, `pathlib`)](#6-파일-시스템-제어-os-pathlib)
  - [6.1 `os` 모듈: 운영체제와 상호작용](#61-os-모듈-운영체제와-상호작용)
  - [6.2 `pathlib`: 객체 지향적인 파일 경로 다루기](#62-pathlib-객체-지향적인-파일-경로-다루기)

--- 

## 1. 파일 입출력 (File I/O)

파이썬에서 파일은 데이터를 영구적으로 저장하고 읽어오는 데 사용됩니다. 텍스트 파일, 바이너리 파일 등 다양한 종류의 파일을 다룰 수 있습니다.

### 1.1 `with open(...)`을 사용한 안전한 파일 처리

파일을 열고 작업한 후에는 반드시 닫아주어야 합니다. `with` 문을 사용하면 파일이 자동으로 닫히므로, 리소스 누수를 방지하고 예외 발생 시에도 안전하게 파일을 처리할 수 있습니다. 또한, 파일 관련 예외 처리를 통해 프로그램의 안정성을 높일 수 있습니다.

- **`with` 문의 동작 원리:** `with` 문은 **컨텍스트 관리자(Context Manager)** 프로토콜을 따릅니다. `with` 블록에 진입할 때 객체의 `__enter__` 메서드가 호출되고, 블록을 벗어날 때(정상 종료든 예외 발생이든) `__exit__` 메서드가 호출됩니다. `__exit__` 메서드에서 파일 닫기 등의 정리 작업을 수행하므로, 개발자가 명시적으로 `close()`를 호출할 필요가 없어집니다. 파일 외에도 데이터베이스 연결, 락(lock) 획득/해제 등 리소스 관리가 필요한 다양한 상황에서 `with` 문을 활용할 수 있습니다.

```python
# 파일 쓰기 예시
file_path = "example.txt"
with open(file_path, 'w', encoding='utf-8') as f:
    f.write("Hello, Python File I/O!\n")
    f.write("이것은 두 번째 줄입니다.\n")

print(f"'{file_path}' 파일이 성공적으로 작성되었습니다.")

# 파일 읽기 예시 (예외 처리 포함)
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"\n'{file_path}' 파일 내용:\n{content}")
except FileNotFoundError:
    print(f"\n오류: 파일 '{file_path}'을(를) 찾을 수 없습니다.")
except IOError as e:
    print(f"\n오류: 파일 '{file_path}'을(를) 읽는 중 문제가 발생했습니다: {e}")
```


### 1.2 파일 열기 모드 (`'r'`, `'w'`, `'a'`, `'x'`, `'b'`, `'t'`, `'+'`) 

`open()` 함수는 두 번째 인자로 파일 열기 모드를 지정합니다.

| 모드 | 설명 |
| :--- | :--- |
| `'r'` | 읽기 모드 (기본값). 파일이 없으면 `FileNotFoundError` 발생. |
| `'w'` | 쓰기 모드. 파일이 있으면 내용을 덮어쓰고, 없으면 새로 생성. |
| `'a'` | 추가(append) 모드. 파일 끝에 내용을 추가. 파일이 없으면 새로 생성. |
| `'x'` | 독점 생성 모드. 파일이 없으면 새로 생성하고, 있으면 `FileExistsError` 발생. |
| `'b'` | 바이너리 모드. 텍스트가 아닌 바이너리 데이터(이미지, 실행 파일 등)를 다룰 때 사용. (예: `'rb'`, `'wb'`) |
| `'t'` | 텍스트 모드 (기본값). 텍스트 데이터를 다룰 때 사용. (예: `'rt'`, `'wt'`) |
| `'+'` | 읽기/쓰기 모드. 다른 모드와 함께 사용 (예: `'r+'`, `'w+'`, `'a+'`). |

### 1.3 파일 읽기/쓰기 메서드

- **`read()`:** 파일 전체 내용을 문자열로 읽어옵니다. 인자로 바이트 수를 지정하면 해당 바이트만큼 읽습니다.
- **`readline()`:** 파일에서 한 줄을 읽어옵니다.
- **`readlines()`:** 파일의 모든 줄을 리스트 형태로 읽어옵니다.
- **`write(string)`:** 문자열을 파일에 씁니다.
- **`writelines(list_of_strings)`:** 문자열 리스트를 파일에 씁니다. 각 문자열 끝에 줄 바꿈 문자를 직접 추가해야 합니다.

```python
# 한 줄씩 읽기
with open("example.txt", 'r', encoding='utf-8') as f:
    print("\n한 줄씩 읽기:")
    print(f.readline(), end='')
    print(f.readline(), end='')

# 모든 줄을 리스트로 읽기
with open("example.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print("\n모든 줄을 리스트로 읽기:")
    for line in lines:
        print(line, end='')

# 파일에 여러 줄 쓰기
new_lines = ["첫 번째 새 줄\n", "두 번째 새 줄\n", "세 번째 새 줄\n"]
with open("new_example.txt", 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("\n'new_example.txt' 파일이 성공적으로 작성되었습니다.")
```

## 2. 표준 데이터 형식 다루기

파이썬은 CSV, JSON, Pickle과 같은 다양한 표준 데이터 형식을 다루기 위한 내장 모듈을 제공합니다.

### 2.1 CSV 파일 처리: `csv` 모듈 활용

CSV (Comma Separated Values)는 데이터를 쉼표로 구분하여 저장하는 텍스트 파일 형식입니다. `csv` 모듈은 CSV 파일을 쉽게 읽고 쓸 수 있도록 도와줍니다.

```python
import csv

# CSV 파일 쓰기 (writerows)
data = [
    ['Name', 'Age', 'City'],
    ['Alice', 30, 'New York'],
    ['Bob', 24, 'London']
]

with open('people.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(data)
print("'people.csv' 파일이 성공적으로 작성되었습니다.")

# CSV 파일 쓰기 (DictWriter)
people_data = [
    {'Name': 'Charlie', 'Age': 35, 'City': 'Paris'},
    {'Name': 'David', 'Age': 28, 'City': 'Berlin'}
]

with open('more_people.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Name', 'Age', 'City']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader() # 헤더 쓰기
    writer.writerows(people_data)
print("'more_people.csv' 파일이 성공적으로 작성되었습니다.")

# CSV 파일 읽기
with open('people.csv', 'r', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    print("\n'people.csv' 파일 내용:")
    for row in csv_reader:
        print(row)

# 딕셔너리 형태로 읽기 (DictReader)
with open('people.csv', 'r', encoding='utf-8') as csvfile:
    dict_reader = csv.DictReader(csvfile)
    print("\n'people.csv' 파일 내용 (DictReader):")
    for row in dict_reader:
        print(row['Name'], row['Age'], row['City'])
```
- **Pandas를 이용한 CSV 처리 (데이터 과학 분야):**
  데이터 과학 분야에서는 `pandas` 라이브러리의 `read_csv()` 함수가 CSV 파일을 `DataFrame` 형태로 불러오는 데 가장 널리 사용됩니다. 이는 대규모 CSV 파일을 효율적으로 처리하고, 데이터 분석 및 조작을 위한 강력한 기능을 제공합니다. `csv` 모듈은 저수준의 행 단위 처리에 적합하며, `pandas`는 고수준의 데이터프레임 기반 처리에 적합합니다.
  ```python
  import pandas as pd
  # df = pd.read_csv('people.csv')
  # print(df)
  ```



### 2.2 JSON 데이터 처리: `json` 모듈 활용

JSON (JavaScript Object Notation)은 웹 애플리케이션에서 데이터를 교환할 때 널리 사용되는 경량 데이터 교환 형식입니다. 파이썬의 딕셔너리와 리스트는 JSON 데이터 구조와 거의 1:1로 매핑됩니다. (자세한 내용은 0428정리.md 참조)

```python
import json

# 파이썬 딕셔너리
python_data = {
    'name': 'John Doe',
    'age': 30,
    'isStudent': False,
    'courses': [{'title': 'Math', 'credits': 3}, {'title': 'History', 'credits': 2}]
}

# 파이썬 객체를 JSON 문자열로 변환 (직렬화)
json_string = json.dumps(python_data, indent=4, ensure_ascii=False)
print("\n--- Python 객체 -> JSON 문자열 ---")
print(json_string)

# JSON 문자열을 파이썬 객체로 변환 (역직렬화)
parsed_data = json.loads(json_string)
print("\n--- JSON 문자열 -> Python 객체 ---")
print(parsed_data)
print(f"첫 번째 과목 제목: {parsed_data['courses'][0]['title']}")
```

### 2.3 객체 직렬화: `pickle` 모듈

`pickle` 모듈은 파이썬 객체를 바이트 스트림으로 변환(직렬화, pickling)하고, 바이트 스트림을 다시 파이썬 객체로 복원(역직렬화, unpickling)하는 기능을 제공합니다. 파이썬 객체 구조를 그대로 저장하고 싶을 때 유용합니다.

```python
import pickle

# 직렬화할 파이썬 객체
my_complex_data = {
    'numbers': [1, 2, 3, 4, 5],
    'text': 'Hello Pickle',
    'is_active': True,
    'nested': {'a': 1, 'b': [10, 20]}
}

# 객체 직렬화 (파일에 쓰기)
with open('data.pickle', 'wb') as f:
    pickle.dump(my_complex_data, f)
print("\n'data.pickle' 파일에 객체가 직렬화되었습니다.")

# 객체 역직렬화 (파일에서 읽기)
with open('data.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
print(f"\n'data.pickle'에서 로드된 객체: {loaded_data}")
print(f"로드된 데이터의 타입: {type(loaded_data)}")
print(f"로드된 데이터의 'text': {loaded_data['text']}")
```

- **주의:** `pickle`은 파이썬에 특화된 형식이며, 보안에 취약할 수 있습니다. 신뢰할 수 없는 소스에서 온 `pickle` 파일은 실행하지 않는 것이 좋습니다. 다른 언어와의 데이터 교환에는 JSON이나 CSV를 사용하는 것이 일반적입니다.

## 3. 외부 API 연동: `requests` 라이브러리

`requests` 라이브러리는 파이썬에서 HTTP 요청을 보내는 가장 인기 있고 사용하기 쉬운 라이브러리입니다. 웹 API와 통신하거나 웹 페이지의 내용을 가져올 때 주로 사용됩니다.

```python
import requests

# GET 요청 보내기
url = "https://jsonplaceholder.typicode.com/todos/1"
response = requests.get(url, timeout=5) # 5초 타임아웃 설정

print(f"\nGET 요청 상태 코드: {response.status_code}")
print(f"GET 요청 응답 내용 (JSON): {response.json()}")

# POST 요청 보내기
post_url = "https://jsonplaceholder.typicode.com/posts"
new_post = {'title': 'foo', 'body': 'bar', 'userId': 1}
post_response = requests.post(post_url, json=new_post, timeout=5)

print(f"\nPOST 요청 상태 코드: {post_response.status_code}")
print(f"POST 요청 응답 내용 (JSON): {post_response.json()}")

# 에러 처리
try:
    response.raise_for_status() # HTTP 에러 발생 시 예외 발생
    print("요청 성공!")
except requests.exceptions.HTTPError as e:
    print(f"HTTP 에러 발생: {e}")
except requests.exceptions.ConnectionError as e:
    print(f"연결 에러 발생: {e}")
except requests.exceptions.Timeout as e:
    print(f"타임아웃 에러 발생: {e}")
except requests.exceptions.RequestException as e: # 모든 requests 관련 에러를 포괄적으로 처리
    print(f"요청 중 예상치 못한 에러 발생: {e}")
```
 - **Session 객체 사용 (여러 요청 시 효율적):**
  `requests.Session` 객체는 여러 요청에 걸쳐 동일한 TCP 연결을 재사용하고, 쿠키, 헤더 등의 상태를 유지할 수 있게 해줍니다. 이는 특히 동일한 호스트에 반복적으로 요청을 보내는 경우 **성능 향상(연결 풀링)**과 **상태 관리(인증, 세션 유지)**에 매우 중요합니다.
```python
import requests

# Session 객체 사용
with requests.Session() as session:
    session.auth = ('user', 'pass') # 인증 정보 설정
    session.headers.update({'x-test': 'true'}) # 헤더 설정

    response1 = session.get('https://httpbin.org/headers')
    print(f"\nSession GET 응답: {response1.json()}")

    response2 = session.get('https://httpbin.org/cookies')
    print(f"Session GET 응답 (쿠키): {response2.json()}")

# 3.2 API 호출 로깅 예시
import logging

# 로깅 설정 (basicConfig는 한 번만 설정하는 것이 좋음)
# 여기서는 예시를 위해 간단히 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_and_log(url):
    logging.info(f"API 요청 시작: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # HTTP 에러 발생 시 예외 발생
        logging.info(f"API 요청 성공: {url}, 상태 코드: {response.status_code}")
        return response.json()
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP 에러 발생: {url}, 상태 코드: {e.response.status_code}, 응답: {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        logging.critical(f"요청 중 치명적인 에러 발생: {url}, 에러: {e}")
        raise

print("\n--- API 호출 로깅 예시 ---")
try:
    data = fetch_and_log("https://jsonplaceholder.typicode.com/posts/1")
    print(f"가져온 데이터: {data['title']}")
    fetch_and_log("https://jsonplaceholder.typicode.com/nonexistent-url") # 에러 발생 예시
except (requests.exceptions.RequestException, requests.exceptions.HTTPError):
    print("API 호출 중 에러가 발생했습니다. 로그를 확인하세요.")
```

- **설치:** `requests`는 내장 라이브러리가 아니므로 `pip install requests`로 설치해야 합니다.
- **주요 메서드:** `requests.get()`, `requests.post()`, `requests.put()`, `requests.delete()` 등.
- **응답 객체:** `response.status_code`, `response.text`, `response.json()`, `response.headers` 등 다양한 속성을 제공합니다.
- **타임아웃 (Timeout):** `requests` 요청 시 `timeout` 매개변수를 사용하여 일정 시간 내에 응답이 없으면 예외를 발생시키도록 설정하는 것이 중요합니다. 이는 네트워크 지연 등으로 인해 프로그램이 무한정 대기하는 것을 방지합니다.
- **포괄적인 에러 처리:** `requests.exceptions.RequestException`은 `ConnectionError`, `HTTPError`, `Timeout` 등 `requests` 라이브러리에서 발생할 수 있는 모든 예외의 기본 클래스입니다. 따라서 `except requests.exceptions.RequestException as e:`와 같이 처리하면 `requests` 관련 모든 잠재적 오류를 포괄적으로 처리할 수 있어 안정적인 코드 작성에 도움이 됩니다.

### 3.1 API Rate Limiting (요청 제한) 및 Pagination (페이지네이션)

실제 웹 API를 사용할 때는 **요청 제한(Rate Limiting)**과 **페이지네이션(Pagination)**이라는 두 가지 중요한 개념을 이해하고 처리해야 합니다.

-   **Rate Limiting (요청 제한):**
    -   대부분의 공용 API는 서버 과부하를 방지하기 위해 일정 시간 동안 보낼 수 있는 요청의 수를 제한합니다. 이 제한을 초과하면 `429 Too Many Requests`와 같은 HTTP 상태 코드를 반환합니다.
    -   **처리 방법:** API 응답 헤더(예: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`)를 확인하여 남은 요청 수와 초기화 시간을 파악하고, 필요한 경우 `time.sleep()`을 사용하여 요청 사이에 지연 시간을 두거나, 지수 백오프(Exponential Backoff) 전략을 구현하여 재시도합니다.

-   **Pagination (페이지네이션):**
    -   대량의 데이터를 한 번의 요청으로 모두 반환하는 대신, 여러 "페이지"로 나누어 반환하는 방식입니다. 이는 서버의 부담을 줄이고 클라이언트가 필요한 데이터만 효율적으로 가져올 수 있게 합니다.
    -   **처리 방법:** API 문서에 따라 `page`, `per_page`, `offset`, `limit`, `next_page_url` 등과 같은 매개변수를 사용하여 다음 페이지의 데이터를 요청하고, 모든 데이터를 가져올 때까지 반복적으로 요청을 보냅니다.

**실무적 의미:**
-   API를 통해 대량의 데이터를 수집할 때는 Rate Limiting과 Pagination을 고려하지 않으면 데이터 수집이 실패하거나, API 제공자로부터 차단될 수 있습니다.
-   안정적인 데이터 수집 파이프라인을 구축하기 위해서는 이러한 API 정책을 준수하는 코드를 작성하는 것이 필수적입니다.

## 4. 데이터 유효성 검사 (Data Validation)

데이터를 처리하는 모든 과정에서 **데이터 유효성 검사(Data Validation)**는 매우 중요합니다. "Garbage In, Garbage Out"이라는 말처럼, 유효하지 않거나 예상치 못한 형식의 데이터는 프로그램의 오류를 유발하거나 잘못된 분석 결과를 초래할 수 있습니다. 특히 외부 소스(파일, API, 사용자 입력 등)로부터 데이터를 받을 때는 반드시 유효성 검사를 수행해야 합니다.

### 4.1 왜 데이터 유효성 검사가 중요한가?
-   **오류 방지:** 잘못된 데이터 타입, 범위, 형식 등으로 인한 런타임 오류를 예방합니다.
-   **데이터 무결성 유지:** 데이터베이스나 시스템에 저장되는 데이터의 품질과 일관성을 보장합니다.
-   **보안 강화:** 악의적인 입력(SQL Injection, XSS 등)으로부터 시스템을 보호합니다.
-   **신뢰성 있는 분석:** 정확하고 신뢰할 수 있는 데이터를 기반으로 분석 및 모델링을 수행할 수 있도록 합니다.

### 4.2 일반적인 유효성 검사 유형

-   **타입 검사 (Type Check):** 데이터가 예상한 타입(예: 정수, 문자열, 불리언)인지 확인합니다.
-   **범위 검사 (Range Check):** 숫자가 특정 범위 내에 있는지 확인합니다 (예: 나이는 0~150).
-   **형식 검사 (Format Check):** 데이터가 특정 형식(예: 이메일 주소, 전화번호, 날짜 형식)을 따르는지 확인합니다. 정규표현식이 유용하게 사용됩니다.
-   **필수 값 검사 (Presence Check):** 특정 필드가 누락되지 않았는지 확인합니다.
-   **값 목록 검사 (Value List Check):** 데이터가 미리 정의된 유효한 값 목록 중 하나인지 확인합니다.

### 4.3 데이터 유효성 검사 구현 예시

```python
def validate_user_data(user_data: dict) -> bool:
    """사용자 데이터의 유효성을 검사하는 함수"""
    # 1. 필수 필드 검사
    required_fields = ['name', 'age', 'email']
    for field in required_fields:
        if field not in user_data:
            print(f"오류: 필수 필드 '{field}'가 누락되었습니다.")
            return False

    # 2. 타입 및 범위 검사
    if not isinstance(user_data['name'], str) or not user_data['name']:
        print("오류: 이름은 비어있지 않은 문자열이어야 합니다.")
        return False

    if not isinstance(user_data['age'], int) or not (0 <= user_data['age'] <= 120):
        print("오류: 나이는 0에서 120 사이의 정수여야 합니다.")
        return False

    # 3. 이메일 형식 검사 (간단한 정규표현식 사용)
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
```

### 4.4 고급 유효성 검사 라이브러리

복잡한 데이터 구조나 API 스키마에 대한 유효성 검사가 필요한 경우, 다음과 같은 전문 라이브러리를 활용하면 개발 효율성을 높일 수 있습니다.

-   **`Pydantic`:** 파이썬 타입 힌트를 사용하여 데이터 유효성 검사 및 설정을 정의할 수 있는 라이브러리입니다. 데이터 모델을 정의하면 자동으로 유효성 검사를 수행하고, JSON 직렬화/역직렬화 기능을 제공하여 FastAPI와 같은 웹 프레임워크에서 API 요청/응답 데이터의 유효성 검사에 널리 사용됩니다.
-   **`Cerberus`:** 유연하고 확장 가능한 데이터 유효성 검사 라이브러리입니다. YAML이나 JSON 스키마를 사용하여 복잡한 규칙을 정의할 수 있습니다.

이러한 라이브러리들은 수동으로 유효성 검사 로직을 작성하는 것보다 훨씬 강력하고 유지보수하기 쉬운 코드를 작성할 수 있도록 돕습니다.

- **스키마 정의 언어 활용:**
  `Pydantic`이나 `Cerberus`와 같은 라이브러리 외에도, **JSON Schema**나 **OpenAPI/Swagger 스키마**와 같은 표준 스키마 정의 언어를 사용하여 데이터의 구조와 유효성 규칙을 정의할 수 있습니다. 이러한 스키마는 API 문서화, 코드 자동 생성, 그리고 런타임 유효성 검사에 활용되어 데이터 일관성을 보장하고 개발 프로세스를 자동화하는 데 큰 도움을 줍니다.

## 5. 설정 관리: 환경 변수와 `python-dotenv`

민감한 정보(API 키, 데이터베이스 비밀번호 등)나 환경에 따라 달라지는 설정 값은 코드 내에 직접 하드코딩하는 대신 환경 변수로 관리하는 것이 보안상 안전하고 유연합니다. `python-dotenv` 라이브러리는 `.env` 파일에서 환경 변수를 로드하는 것을 쉽게 해줍니다.

**`.env` 파일 예시:**

```
# .env

API_KEY=your_super_secret_api_key
DATABASE_URL=postgresql://user:password@host:port/dbname
DEBUG_MODE=True
```

**파이썬 코드에서 환경 변수 사용:**

```python
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수 접근
api_key = os.getenv('API_KEY')
db_url = os.getenv('DATABASE_URL')
debug_mode = os.getenv('DEBUG_MODE', 'False').lower() == 'true' # 기본값 설정 및 불리언 변환

print(f"\nAPI Key: {api_key}")
print(f"Database URL: {db_url}")
print(f"Debug Mode: {debug_mode}")

# 환경 변수가 설정되지 않았을 경우
non_existent_var = os.getenv('NON_EXISTENT_VAR', '기본값')
print(f"존재하지 않는 변수: {non_existent_var}")
```

- **설치:** `python-dotenv`는 내장 라이브러리가 아니므로 `pip install python-dotenv`로 설치해야 합니다.
- **보안:** `.env` 파일은 버전 관리 시스템(Git 등)에 포함되지 않도록 `.gitignore`에 추가하는 것이 중요합니다.

## 6. 파일 시스템 제어 (`os`, `pathlib`)

파이썬은 운영체제와 상호작용하여 파일 및 디렉토리를 생성, 삭제, 이동하거나 정보를 얻는 기능을 제공합니다. `os` 모듈은 전통적인 방식이며, `pathlib` 모듈은 객체 지향적인 방식으로 경로를 다룹니다.

### 6.1 `os` 모듈: 운영체제와 상호작용

`os` 모듈은 파일 및 디렉토리 조작, 환경 변수 접근 등 운영체제와 관련된 다양한 기능을 제공합니다.

```python
import os

# 현재 작업 디렉토리 확인
current_dir = os.getcwd()
print(f"\n현재 작업 디렉토리: {current_dir}")

# 디렉토리 생성
new_dir = "my_new_directory"
if not os.path.exists(new_dir):
    os.makedirs(new_dir) # 하위 디렉토리까지 생성
    print(f"'{new_dir}' 디렉토리 생성.")

# 파일 목록 확인
print(f"'{current_dir}'의 파일 및 디렉토리 목록: {os.listdir(current_dir)}")

# 파일/디렉토리 존재 여부 확인
print(f"'{new_dir}' 존재 여부: {os.path.exists(new_dir)}")
print(f"'example.txt' 파일인가? {os.path.isfile('example.txt')}")
print(f"'my_new_directory' 디렉토리인가? {os.path.isdir('my_new_directory')}")

# 파일 이름과 확장자 분리
file_name_with_ext = "document.pdf"
name, ext = os.path.splitext(file_name_with_ext)
print(f"파일 이름: {name}, 확장자: {ext}")

# 경로 결합 (운영체제 독립적)
combined_path = os.path.join('folder', 'subfolder', 'file.txt')
print(f"결합된 경로: {combined_path}")

# 디렉토리 삭제
# os.rmdir(new_dir) # 비어있는 디렉토리만 삭제 가능
# os.removedirs('parent/child') # 비어있는 하위 디렉토리까지 삭제
# import shutil
# shutil.rmtree(new_dir) # 비어있지 않아도 강제 삭제 (주의!)
# shutil 모듈은 파일 및 디렉토리 복사, 이동, 삭제 등 고수준 파일 작업을 제공합니다.
```

### 6.2 `pathlib`: 객체 지향적인 파일 경로 다루기

`pathlib` 모듈은 파이썬 3.4부터 표준 라이브러리에 포함되었으며, 파일 시스템 경로를 객체 지향적으로 다룰 수 있게 해줍니다. `os.path` 함수들을 대체하며, 더 직관적이고 파이썬스러운 코드를 작성할 수 있습니다.

```python
from pathlib import Path

# Path 객체 생성
current_path = Path.cwd()
print(f"\n현재 작업 디렉토리 (pathlib): {current_path}")

# 경로 결합
new_file_path = current_path / "data" / "report.txt"
print(f"새 파일 경로: {new_file_path}")

# 디렉토리 생성
new_dir_path = Path("pathlib_dir")
new_dir_path.mkdir(exist_ok=True) # 이미 존재해도 에러 발생 안 함
print(f"'{new_dir_path}' 디렉토리 생성.")

# 파일 쓰기
file_to_write = new_dir_path / "hello.txt"
file_to_write.write_text("Hello from pathlib!")
print(f"'{file_to_write}' 파일 작성.")

# 파일 읽기
read_content = file_to_write.read_text()
print(f"'{file_to_write}' 파일 내용: {read_content}")

# 파일/디렉토리 존재 여부 확인
print(f"'{new_dir_path}' 존재 여부: {new_dir_path.exists()}")
print(f"'{file_to_write}' 파일인가? {file_to_write.is_file()}")
print(f"'{new_dir_path}' 디렉토리인가? {new_dir_path.is_dir()}")

# 파일 이름, 확장자, 부모 디렉토리 등
print(f"파일 이름: {new_file_path.name}")
print(f"확장자: {new_file_path.suffix}")
print(f"부모 디렉토리: {new_file_path.parent}")

# 절대 경로 및 심볼릭 링크 해결
relative_path = Path("../0423정리.md") # 상대 경로
resolved_path = relative_path.resolve()
print(f"상대 경로: {relative_path}")
print(f"절대 경로 (resolve): {resolved_path}")

# 디렉토리 내 파일 순회 (glob)
print("\n현재 디렉토리의 .txt 파일 목록 (glob):")
for p in current_path.glob('*.txt'):
    print(p.name)

# 디렉토리 내 모든 항목 순회 (iterdir)
print("\n현재 디렉토리의 모든 항목 (iterdir):")
for item in current_path.iterdir():
    print(item.name)

# 파일/디렉토리 삭제
# file_to_write.unlink() # 파일 삭제
# new_dir_path.rmdir() # 비어있는 디렉토리 삭제
```

- **권장 사항:** 현대 파이썬 개발에서는 `pathlib` 모듈을 사용하는 것이 더 파이썬스럽고, 객체 지향적인 접근 방식으로 인해 코드의 가독성과 유지보수성이 향상됩니다.
```python
    if not re.match(email_pattern, user_data['email']):
        print("오류: 유효하지 않은 이메일 형식입니다.")
        return False

    print("데이터 유효성 검사 통과.")
    return True

# 테스트
valid_data = {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}
invalid_data_missing_field = {'name': 'Bob', 'age': 25}
invalid_data_age_type = {'name': 'Charlie', 'age': 'twenty', 'email': 'charlie@example.com'}

validate_user_data(valid_data)
validate_user_data(invalid_data_missing_field)
validate_user_data(invalid_data_age_type)
```





