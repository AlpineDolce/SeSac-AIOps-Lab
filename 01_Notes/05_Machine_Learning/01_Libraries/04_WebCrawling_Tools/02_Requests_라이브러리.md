<h2>데이터 크롤링 도구: Selenium, BeautifulSoup 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 파이썬에서 HTTP 요청을 보내는 데 사용되는 `Requests` 라이브러리의 기본 사용법을 다룹니다. 웹 페이지의 HTML 콘텐츠를 가져오는 방법과 GET/POST 요청, 헤더 설정 등 기본적인 웹 통신 방법을 학습하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Requests 라이브러리](#1-requests-라이브러리)
  - [1.1. Requests 설치](#11-requests-설치)
  - [1.2. GET 요청 보내기](#12-get-요청-보내기)
  - [1.3. POST 요청 보내기](#13-post-요청-보내기)
  - [1.4. 응답 객체 다루기](#14-응답-객체-다루기)
  - [1.5. 헤더 및 파라미터 설정](#15-헤더-및-파라미터-설정)

---

## 1. Requests 라이브러리

### 1.1. Requests 설치

`Requests` 라이브러리는 파이썬의 표준 라이브러리가 아니므로, 사용하기 전에 `pip`를 이용하여 설치해야 합니다. 웹 크롤링 프로젝트를 진행할 때는 파이썬 가상 환경(Virtual Environment)을 사용하는 것이 권장됩니다. 이는 프로젝트별로 필요한 라이브러리들을 독립적으로 관리하여 의존성 충돌을 방지하고, 개발 환경을 깔끔하게 유지하는 데 도움을 줍니다.

```bash
# 1. 가상 환경 생성 (선택 사항이지만 권장)
python -m venv venv_crawling

# 2. 가상 환경 활성화
# Windows:
.\venv_crawling\Scripts\activate
# macOS/Linux:
source venv_crawling/bin/activate

# 3. Requests 라이브러리 설치
pip install requests

# 설치 확인
pip show requests
```

> **[실무 노트] 가상 환경 사용의 중요성:**
> 데이터 과학 프로젝트에서는 다양한 라이브러리와 그 버전을 사용하게 됩니다. 예를 들어, 특정 프로젝트에서는 `requests` 2.20 버전을 필요로 하고, 다른 프로젝트에서는 2.28 버전을 필요로 할 수 있습니다. 이때 가상 환경을 사용하지 않으면 라이브러리 버전 충돌이 발생하여 프로젝트가 제대로 동작하지 않을 수 있습니다.
>
> 가상 환경은 각 프로젝트를 위한 독립적인 파이썬 실행 환경을 제공하므로, 이러한 문제를 효과적으로 해결할 수 있습니다. `venv` 외에도 `conda`와 같은 도구도 널리 사용됩니다. 실무에서는 프로젝트 시작 시 가상 환경을 먼저 설정하는 것을 습관화해야 합니다.

### 1.2. GET 요청 보내기

GET 요청은 웹 서버로부터 데이터를 조회할 때 사용되는 가장 일반적인 HTTP 메서드입니다. 웹 브라우저에서 URL을 입력하여 웹 페이지에 접속하는 것이 대표적인 GET 요청입니다. `Requests` 라이브러리에서는 `requests.get()` 함수를 사용하여 GET 요청을 보낼 수 있습니다.

```python
import requests

# 1. 기본적인 GET 요청
url = "https://www.example.com"
response = requests.get(url)

print(f"Status Code: {response.status_code}")
print(f"Content Type: {response.headers.get('Content-Type')}")
# print(response.text) # 웹 페이지의 HTML 내용 출력

# 2. 쿼리 파라미터(Query Parameters)와 함께 GET 요청 보내기
# URL에 ?key1=value1&key2=value2 형태로 추가되는 데이터
search_url = "https://www.google.com/search"
params = {
    'q': '웹 크롤링',
    'hl': 'ko' # 한국어 검색
}
response = requests.get(search_url, params=params)

print(f"\nSearch URL: {response.url}") # 실제 요청된 URL 확인
print(f"Status Code: {response.status_code}")
# print(response.text) # 검색 결과 페이지 HTML 출력
```

> **[실무 노트] GET 요청의 활용과 주의사항:**
> GET 요청은 주로 데이터를 조회하거나, 웹 페이지의 HTML 콘텐츠를 가져올 때 사용됩니다. URL에 쿼리 파라미터를 포함하여 서버에 추가 정보를 전달할 수 있습니다.
> 
> *   **언제 GET을 사용할까?**
>     *   웹 페이지의 HTML 소스를 가져와 파싱할 때.
>     *   검색 엔진에 쿼리를 보내 검색 결과를 얻을 때.
>     *   API에서 특정 리소스의 정보를 조회할 때 (예: `GET /api/products?category=electronics`).
>     *   데이터를 변경하지 않는 모든 조회 작업.
> *   **주의사항:**
>     *   **데이터 노출:** 쿼리 파라미터는 URL에 직접 노출되므로, 비밀번호나 민감한 개인 정보와 같은 데이터를 GET 요청으로 보내서는 안 됩니다. 이러한 데이터는 POST 요청으로 보내야 합니다.
>     *   **URL 길이 제한:** 대부분의 웹 서버와 브라우저는 URL 길이에 제한을 둡니다. 따라서 매우 많은 양의 데이터를 쿼리 파라미터로 전달하는 것은 적합하지 않습니다.
>     *   **캐싱:** GET 요청은 서버에 의해 캐싱될 수 있습니다. 이는 동일한 요청에 대해 더 빠른 응답을 받을 수 있다는 장점이 있지만, 항상 최신 데이터를 받아야 하는 경우에는 캐싱 정책을 고려해야 합니다.
> 
> `Requests` 라이브러리는 `params` 인자를 사용하여 쿼리 파라미터를 딕셔너리 형태로 전달하면, 자동으로 URL 인코딩을 처리해주므로 편리하고 안전하게 사용할 수 있습니다.


### 1.3. POST 요청 보내기

POST 요청은 서버에 데이터를 제출하거나, 새로운 리소스를 생성할 때 사용되는 HTTP 메서드입니다. 웹사이트에서 회원가입, 로그인, 게시물 작성 등 사용자가 데이터를 입력하여 서버로 보낼 때 주로 사용됩니다. `Requests` 라이브러리에서는 `requests.post()` 함수를 사용하여 POST 요청을 보낼 수 있습니다.

```python
import requests
import json

# 1. 폼 데이터(Form Data)와 함께 POST 요청 보내기
# HTML 폼을 통해 전송되는 데이터와 유사합니다.
login_url = "https://httpbin.org/post" # 테스트용 URL
form_data = {
    'username': 'testuser',
    'password': 'testpass',
    'remember_me': 'on'
}
response = requests.post(login_url, data=form_data)

print(f"\nForm Data POST Status Code: {response.status_code}")
print(f"Form Data POST Response JSON: {response.json()}")

# 2. JSON 데이터(JSON Data)와 함께 POST 요청 보내기
# RESTful API 통신에서 주로 사용됩니다.
api_url = "https://httpbin.org/post" # 테스트용 URL
json_data = {
    'name': 'Alice',
    'age': 30,
    'city': 'New York'
}
response = requests.post(api_url, json=json_data)

print(f"\nJSON Data POST Status Code: {response.status_code}")
print(f"JSON Data POST Response JSON: {response.json()}")
```

> **[실무 노트] POST 요청의 활용과 주의사항:**
> POST 요청은 서버에 데이터를 안전하게 전송하고, 새로운 리소스를 생성하거나 기존 리소스를 업데이트할 때 사용됩니다.
> 
> *   **언제 POST를 사용할까?**
>     *   로그인, 회원가입, 게시물 작성, 파일 업로드 등 사용자 입력 데이터를 서버로 전송할 때.
>     *   API를 통해 서버에 새로운 데이터를 생성하거나 기존 데이터를 변경할 때.
>     *   민감한 정보(비밀번호, 개인 정보)를 전송할 때 (URL에 노출되지 않음).
> *   **`data` vs `json` 인자:**
>     *   `data` 인자: 딕셔너리, 튜플 리스트, 바이트 문자열 등 다양한 형태로 폼 데이터를 전송할 때 사용합니다. `Content-Type` 헤더가 `application/x-www-form-urlencoded`로 자동 설정됩니다.
>     *   `json` 인자: 파이썬 딕셔너리를 JSON 형식으로 변환하여 전송할 때 사용합니다. `Content-Type` 헤더가 `application/json`으로 자동 설정됩니다. RESTful API와 통신할 때 매우 편리합니다.
> *   **주의사항:**
>     *   **멱등성(Idempotency) 없음:** POST 요청은 일반적으로 멱등성을 가지지 않습니다. 즉, 동일한 POST 요청을 여러 번 보내면 서버에 여러 개의 리소스가 생성되거나 데이터가 여러 번 변경될 수 있습니다. (예: 게시물 중복 작성)
>     *   **캐싱 안 됨:** POST 요청은 캐싱되지 않습니다. 항상 서버에 새로운 요청을 보냅니다.
> 
> `Requests` 라이브러리는 `data`와 `json` 인자를 통해 다양한 형태의 데이터 전송을 지원하므로, 서버의 API 명세에 맞춰 적절한 인자를 선택하여 사용해야 합니다.


### 1.4. 응답 객체 다루기

`Requests` 라이브러리로 HTTP 요청을 보내면 `Response` 객체를 반환합니다. 이 `Response` 객체에는 서버로부터 받은 응답에 대한 모든 정보가 담겨 있으며, 이를 통해 웹 페이지의 내용, 상태 코드, 헤더 등을 확인할 수 있습니다.

```python
import requests

url = "https://www.naver.com"
response = requests.get(url)

# 1. 상태 코드 (Status Code) 확인
# HTTP 요청의 성공/실패 여부를 나타냅니다. 200은 성공을 의미합니다.
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    print("Request successful!")
else:
    print(f"Request failed with status code: {response.status_code}")

# 2. 응답 내용 (Content) 가져오기
# response.text: 응답 내용을 텍스트(유니코드)로 반환합니다. 인코딩을 자동으로 감지합니다.
# response.content: 응답 내용을 바이트(bytes)로 반환합니다. 이미지, 비디오 등 바이너리 데이터에 적합합니다.
print(f"\nResponse Text (first 200 chars):\n{response.text[:200]}...")
# print(f"Response Content (first 20 bytes):\n{response.content[:20]}...")

# 3. JSON 응답 파싱
# 서버가 JSON 형식의 데이터를 반환할 경우, response.json() 메서드를 사용하여 파이썬 딕셔너리로 변환합니다.
json_url = "https://jsonplaceholder.typicode.com/todos/1" # 테스트용 JSON API
json_response = requests.get(json_url)
if json_response.status_code == 200:
    todo_item = json_response.json()
    print(f"\nJSON Response:\n{todo_item}")
    print(f"User ID: {todo_item['userId']}, Title: {todo_item['title']}")

# 4. 응답 헤더 (Headers) 확인
# 서버가 보낸 응답 헤더 정보를 딕셔너리 형태로 확인할 수 있습니다.
print(f"\nResponse Headers:\n{response.headers}")
print(f"Content-Type from Headers: {response.headers.get('Content-Type')}")

# 5. 응답 인코딩 (Encoding) 확인 및 설정
# Requests는 응답의 인코딩을 자동으로 추측하지만, 때로는 명시적으로 설정해야 할 수 있습니다.
print(f"\nDetected Encoding: {response.encoding}")
# response.encoding = 'utf-8' # 필요한 경우 수동으로 인코딩 설정
# print(f"Content after manual encoding: {response.text[:200]}...")

# 6. 요청 URL 확인
print(f"\nRequested URL: {response.url}")
```

> **[실무 노트] 응답 객체 활용 팁:**
> `Response` 객체를 효과적으로 다루는 것은 웹 크롤링의 핵심입니다. 특히 인코딩 문제와 상태 코드 처리는 실무에서 자주 마주치는 부분이므로 정확히 이해해야 합니다.
>
> *   **상태 코드 (Status Code) 처리:**
>     *   `200 OK`: 요청 성공. 가장 일반적인 성공 코드입니다.
>     *   `404 Not Found`: 요청한 페이지를 찾을 수 없음. URL 오류나 페이지 삭제 시 발생합니다.
>     *   `403 Forbidden`: 서버가 요청을 거부함. User-Agent 변경, 프록시 사용 등 우회 기술이 필요할 수 있습니다.
>     *   `500 Internal Server Error`: 서버 내부 오류. 웹사이트 자체의 문제일 가능성이 높습니다.
>     *   **`response.raise_for_status()`:** 이 메서드를 호출하면 상태 코드가 200번대가 아닐 경우 `HTTPError` 예외를 발생시킵니다. 이를 `try-except` 블록과 함께 사용하여 오류 처리를 간결하게 할 수 있습니다.
> *   **인코딩 문제 해결:**
>     *   Requests는 `response.encoding`을 통해 응답 헤더나 HTML 메타 태그에서 인코딩을 자동으로 감지합니다. 하지만 한글 페이지의 경우 `EUC-KR` 등으로 잘못 감지하여 글자가 깨지는 경우가 있습니다.
>     *   이때는 `response.encoding = response.apparent_encoding`을 사용하거나, `response.encoding = 'utf-8'`과 같이 명시적으로 설정한 후 `response.text`를 다시 읽어들이면 해결되는 경우가 많습니다.
> *   **바이너리 데이터 처리:** 이미지, 동영상, PDF 파일 등 바이너리 데이터를 다운로드할 때는 `response.content`를 사용하여 바이트 형태로 데이터를 가져와야 합니다.
> *   **리다이렉션 추적:** `response.history` 속성을 통해 요청이 리다이렉션된 경우, 이전 응답 객체들을 확인할 수 있습니다. `response.is_redirect`로 리다이렉션 여부를 확인할 수 있습니다.
>
> `Requests`의 `Response` 객체는 웹 크롤링 과정에서 발생하는 다양한 상황에 대응할 수 있는 풍부한 정보를 제공하므로, 각 속성과 메서드의 역할을 숙지하는 것이 중요합니다.


### 1.5. 헤더 및 파라미터 설정

HTTP 요청을 보낼 때 헤더(Headers)와 파라미터(Parameters)를 설정하는 것은 웹 서버와 통신하는 데 매우 중요합니다. 헤더는 요청에 대한 추가 정보(예: 클라이언트 정보, 허용하는 콘텐츠 타입)를 서버에 전달하고, 파라미터는 GET 요청 시 URL에 데이터를 포함하거나 POST 요청 시 본문에 데이터를 포함하는 데 사용됩니다.

```python
import requests

# 1. 헤더(Headers) 설정
# User-Agent는 웹 서버에 요청을 보내는 클라이언트(브라우저)의 정보를 알려줍니다.
# 웹 크롤러임을 숨기거나 특정 브라우저인 것처럼 위장할 때 사용됩니다.
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://www.google.com/' # 이전 페이지 정보
}

url = "https://www.naver.com"
response = requests.get(url, headers=headers)

print(f"Status Code with Custom Headers: {response.status_code}")
# print(response.text[:200])

# 2. 쿼리 파라미터(Parameters) 설정 (GET 요청 시)
# params 인자를 사용하여 딕셔너리 형태로 전달하면 Requests가 자동으로 URL 인코딩을 처리합니다.
search_url = "https://search.naver.com/search.naver"
params = {
    'query': '파이썬 웹 크롤링',
    'where': 'nexearch'
}
response = requests.get(search_url, params=params, headers=headers)

print(f"\nSearch URL with Parameters: {response.url}")
print(f"Status Code with Parameters: {response.status_code}")
# print(response.text[:200])

# 3. POST 요청 시 폼 데이터(data) 또는 JSON 데이터(json) 설정
# (이전 섹션에서 다루었으므로 예시는 생략합니다.)
# requests.post(url, data=form_data, headers=headers)
# requests.post(url, json=json_data, headers=headers)
```

> **[실무 노트] 헤더와 파라미터 활용 전략:**
> 웹 크롤링 시 헤더와 파라미터를 적절히 설정하는 것은 웹 서버의 봇 탐지를 우회하고, 원하는 데이터를 정확히 가져오는 데 매우 중요합니다.
> 
> *   **`User-Agent` 설정의 중요성:**
>     *   대부분의 웹 서버는 `User-Agent` 헤더를 통해 요청을 보내는 클라이언트가 웹 브라우저인지, 아니면 자동화된 봇인지를 판단합니다. 기본 `Requests`의 `User-Agent`는 'python-requests/X.X' 형태로, 이는 봇으로 쉽게 감지될 수 있습니다.
>     *   일반적인 웹 브라우저의 `User-Agent` 문자열(예: Chrome, Firefox)로 변경하여 서버가 요청을 정상적인 브라우저 요청으로 인식하도록 유도할 수 있습니다. 이는 봇 차단을 우회하는 가장 기본적인 방법 중 하나입니다.
> *   **`Referer` 헤더:**
>     *   이전 페이지의 URL을 서버에 알려줍니다. 일부 웹사이트는 `Referer` 헤더가 없거나 비정상적일 경우 요청을 차단하기도 합니다. 실제 브라우저에서 웹사이트를 탐색하는 것처럼 `Referer`를 설정하면 차단을 피할 수 있습니다.
> *   **`Accept-Language` 헤더:**
>     *   클라이언트가 선호하는 언어를 서버에 알려줍니다. 이를 통해 서버는 해당 언어에 맞는 콘텐츠를 제공할 수 있습니다. (예: `ko-KR,ko;q=0.9`는 한국어를 선호함을 의미)
> *   **`Cookie` 헤더:**
>     *   로그인 세션 유지, 사용자 맞춤형 콘텐츠 제공 등에 사용됩니다. `Requests`는 `Session` 객체를 통해 쿠키를 자동으로 관리해주지만, 필요에 따라 수동으로 설정할 수도 있습니다.
> *   **`params` 인자 vs URL 직접 구성:**
>     *   `params` 인자를 사용하는 것이 URL을 직접 문자열로 구성하는 것보다 훨씬 안전하고 편리합니다. `Requests`가 자동으로 URL 인코딩을 처리해주므로, 특수 문자나 한글이 포함된 쿼리 파라미터도 문제없이 전송할 수 있습니다.
> 
> 웹 크롤링 시 웹사이트의 요청 패턴을 분석하고, 필요한 헤더와 파라미터를 적절히 설정하는 것은 성공적인 데이터 수집을 위한 필수적인 기술입니다.

