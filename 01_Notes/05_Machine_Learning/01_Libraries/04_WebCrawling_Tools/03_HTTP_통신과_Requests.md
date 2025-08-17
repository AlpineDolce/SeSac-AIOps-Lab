<h2>[Part 2-1] HTTP 통신과 Requests</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-17

<h2>문서 목표</h2>
<p>이 문서는 웹 통신의 기초인 HTTP 프로토콜을 이해하고, `requests` 라이브러리를 사용하여 서버와 통신하는 방법을 학습합니다. 특히 `Session` 객체를 활용하여 로그인과 같이 상태를 유지해야 하는 크롤링 기법을 익히는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 웹의 기본, HTTP 이해하기](#1-웹의-기본-http-이해하기)
- [2. `requests` 기본 사용법](#2-requests-기본-사용법)
- [3. Session 객체: 똑똑한 크롤러의 필수품](#3-session-객체-똑똑한-크롤러의-필수품)

---

## 1. 웹의 기본, HTTP 이해하기

웹 크롤링은 결국 웹 브라우저가 하는 일을 코드로 흉내 내는 것입니다. 브라우저는 웹 서버와 **HTTP(HyperText Transfer Protocol)**라는 약속(프로토콜)을 통해 대화합니다. 이 대화는 **요청(Request)**과 **응답(Response)**으로 이루어집니다.

- **요청 (Request):** 클라이언트(브라우저, 내 코드)가 서버에게 보내는 메시지
    - **Method:** 원하는 행동 (e.g., `GET`: 정보 줘, `POST`: 이 정보 받아)
    - **Headers:** 요청의 세부 정보 (e.g., `User-Agent`: 나 이런 브라우저야, `Cookie`: 나 아까 로그인했던 사람이야)
    - **Body:** `POST` 요청 시 서버에 전달할 데이터 (e.g., 로그인 아이디/비밀번호)

- **응답 (Response):** 서버가 클라이언트에게 보내는 답변
    - **Status Code:** 요청 결과 (e.g., `200 OK`: 성공, `404 Not Found`: 없어, `403 Forbidden`: 권한 없어)
    - **Headers:** 응답의 세부 정보 (e.g., `Content-Type`: 이거 HTML 파일이야)
    - **Body:** 실제 데이터 (HTML, JSON, 이미지 등)

`requests` 라이브러리는 파이썬에서 이 HTTP 요청을 매우 쉽게 보낼 수 있도록 도와주는 도구입니다.

## 2. `requests` 기본 사용법

**1. GET 요청:** 서버로부터 정보를 가져올 때 사용합니다.
```python
import requests

# User-Agent를 지정하여 코드가 아닌 브라우저인 것처럼 위장
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

response = requests.get('http://example.com', headers=headers)

# 응답 정보 확인
print("상태 코드:", response.status_code) # 200
print("응답 내용 (HTML):", response.text[:100]) # 내용의 첫 100글자만 출력
```

**2. POST 요청:** 서버에 정보를 제출할 때 사용합니다. (e.g., 로그인, 검색)
```python
import requests

# 서버에 전달할 데이터
data = {'key1': 'value1', 'key2': 'value2'}

response = requests.post('http://httpbin.org/post', data=data)

# 서버가 받은 요청 정보를 JSON 형태로 반환
print(response.json())
```

## 3. Session 객체: 똑똑한 크롤러의 필수품

기본 `requests.get()`이나 `requests.post()`는 **상태가 없는(stateless)** 요청입니다. 즉, 매번 새로운 연결을 맺기 때문에 이전 요청의 상태(e.g., 로그인 성공 여부)를 기억하지 못합니다.

로그인 후의 페이지를 크롤링하려면, 서버가 "이 사용자가 로그인된 사용자"임을 알 수 있도록 **쿠키(Cookie)**를 매 요청마다 함께 보내야 합니다. 이 작업을 자동으로 해주는 것이 바로 `Session` 객체입니다.

**`Session`을 이용한 로그인 유지 흐름:**

1.  `requests.Session()` 객체를 생성합니다.
2.  생성된 세션 객체(`s`)를 통해 로그인 페이지에 `POST` 요청을 보냅니다.
3.  서버는 로그인 성공 시, 인증 정보가 담긴 쿠키를 응답 헤더에 담아 보내줍니다.
4.  세션 객체(`s`)는 이 쿠키를 **자동으로 저장**합니다.
5.  이후 세션 객체(`s`)를 통해 다른 페이지에 `GET` 요청을 보내면, 저장해 둔 쿠키를 **자동으로 요청 헤더에 포함**시켜 보냅니다.
6.  서버는 쿠키를 보고 "아, 로그인된 사용자구나!"라고 인지하고 해당 페이지의 내용을 보여줍니다.

**개념적 예시:**
```python
import requests

# 1. 세션 객체 생성
s = requests.Session()

# User-Agent 등 기본 헤더 설정도 가능
s.headers.update({'User-Agent': 'Mozilla/5.0 ...'})

# 2. 로그인 정보로 POST 요청 (세션 객체를 통해)
login_payload = {'username': 'my_id', 'password': 'my_password'}
login_url = 'https://example.com/login'
s.post(login_url, data=login_payload)

# 3. 로그인 후 접근 가능한 "마이페이지"에 GET 요청
# 세션이 쿠키를 기억하고 있으므로, 별도 처리 없이 바로 요청 가능
mypage_url = 'https://example.com/mypage'
response = s.get(mypage_url)

# response.text에는 로그인된 사용자의 마이페이지 HTML이 담겨 있음
print(response.text)
```

---

**핵심 요약:** **단순한 정보 조회는 `requests.get()`**으로 충분하지만, **로그인이 필요하거나 여러 페이지에 걸쳐 상태를 유지해야 하는 크롤링 작업에는 반드시 `requests.Session()` 객체**를 사용해야 합니다.