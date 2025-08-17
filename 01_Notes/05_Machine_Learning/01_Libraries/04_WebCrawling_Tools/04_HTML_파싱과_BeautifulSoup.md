<h2>[Part 2-2] HTML 파싱과 BeautifulSoup</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-17

<h2>문서 목표</h2>
<p>이 문서는 `requests`로 얻은 HTML 텍스트를 `BeautifulSoup` 라이브러리를 이용해 파싱하고, CSS 선택자를 활용하여 원하는 데이터를 정확히 찾고 추출하며, 불필요한 부분을 제거하는 정제(cleaning) 방법을 학습하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. HTML 파싱이란?](#1-html-파싱이란)
- [2. 기본 탐색: `find`와 `find_all`](#2-기본-탐색-find와-find_all)
- [3. 핵심 기술: CSS 선택자 (Selector)](#3-핵심-기술-css-선택자-selector)
- [4. 데이터 추출 및 정제 (Cleaning)](#4-데이터-추출-및-정제-cleaning)

---

## 1. HTML 파싱이란?

`requests`로 얻은 `response.text`는 단순한 문자열(string) 덩어리입니다. 이 문자열에서 우리가 원하는 정보(e.g., 특정 제목, 가격, 날짜)를 체계적으로 추출하려면, 컴퓨터가 이해할 수 있는 객체 구조로 변환해야 합니다. 이 과정을 **파싱(Parsing)**이라고 하며, 파이썬에서는 `BeautifulSoup` 라이브러리가 이 역할을 훌륭하게 수행합니다.

```python
from bs4 import BeautifulSoup
import requests

# 1. requests로 HTML 텍스트 가져오기
response = requests.get("http://example.com")
html_text = response.text

# 2. BeautifulSoup 객체로 파싱하기
# 'html.parser'는 파이썬에 내장된 기본 파서입니다.
soup = BeautifulSoup(html_text, 'html.parser')
```
이제 `soup` 객체를 통해 HTML 문서의 모든 요소에 접근할 수 있습니다.

## 2. 기본 탐색: `find`와 `find_all`

- **`soup.find(태그명, 속성)`:** 조건에 맞는 **첫 번째** 태그 하나를 찾아 반환합니다. 없으면 `None`을 반환합니다.
- **`soup.find_all(태그명, 속성)`:** 조건에 맞는 **모든** 태그를 찾아 리스트 형태로 반환합니다.

```python
# 가장 먼저 나오는 <h1> 태그 하나 찾기
title_tag = soup.find('h1')

# 모든 <p> 태그 찾기
all_p_tags = soup.find_all('p')

# id가 'content'인 <div> 태그 찾기
content_div = soup.find('div', {'id': 'content'})

# class가 'article'인 <p> 태그 찾기 (class는 파이썬 예약어이므로 class_ 사용)
article_p = soup.find('p', class_='article')
```

## 3. 핵심 기술: CSS 선택자 (Selector)

`find`와 `find_all`도 유용하지만, 복잡한 구조의 HTML에서 원하는 요소를 한 번에 정확히 찾아내려면 **CSS 선택자**를 사용하는 것이 훨씬 효율적입니다. 웹 브라우저의 개발자 도구(F12)에서 특정 요소에 대한 선택자를 쉽게 복사할 수 있어 편리합니다.

- **`soup.select_one(선택자)`:** CSS 선택자에 해당하는 **첫 번째** 요소를 반환합니다.
- **`soup.select(선택자)`:** CSS 선택자에 해당하는 **모든** 요소를 리스트로 반환합니다.

**자주 사용하는 CSS 선택자 문법:**

| 종류 | 문법 | 예시 | 설명 |
| --- | --- | --- | --- |
| 태그 | `태그명` | `p` | `<p>` 태그 |
| 클래스 | `.클래스명` | `.content` | `class="content"` 속성을 가진 태그 |
| ID | `#ID명` | `#main` | `id="main"` 속성을 가진 태그 |
| 자손 | `A B` | `div p` | `<div>` 태그의 모든 자손 `<p>` 태그 |
| 자식 | `A > B` | `ul > li` | `<ul>` 태그의 바로 아래 자식 `<li>` 태그 |
| 속성 | `[속성=값]`| `a[target="_blank"]` | `target="_blank"` 속성을 가진 `<a>` 태그 |

**CSS 선택자 활용 예시:**
```python
# id가 'news-list'인 <ul> 태그 안의 <li> 태그들만 모두 선택
news_items = soup.select('ul#news-list > li')

for item in news_items:
    # 각 <li> 태그 안에서 class가 'title'인 <a> 태그 하나를 선택
    title_tag = item.select_one('a.title')
    if title_tag:
        print(title_tag.get_text())
```

## 4. 데이터 추출 및 정제 (Cleaning)

원하는 태그를 찾았다면, 그 안에서 실제 텍스트나 속성 값을 꺼내야 합니다. 이 과정에서 불필요한 공백이나 줄바꿈 문자를 제거하는 **정제(Cleaning)** 작업이 필수적입니다.

- **텍스트 추출:** `.get_text()` 또는 `.text`
- **속성값 추출:** `태그['속성명']` (마치 딕셔너리처럼)

```python
html_doc = """
<div class='item'>
    <a href="/product/1001" class="name">  
        새우깡 
    </a>
    <span class="price">1,500원</span>
</div>
"""
soup = BeautifulSoup(html_doc, 'html.parser')

item_div = soup.find('div', class_='item')

# 텍스트 추출 및 정제
# .get_text()와 strip=True 옵션으로 양 끝의 공백/줄바꿈을 한번에 제거
name = item_div.select_one('a.name').get_text(strip=True)
# 결과: '새우깡' (불필요한 공백과 줄바꿈 사라짐)

# 속성값 추출
link = item_div.select_one('a.name')['href']
# 결과: '/product/1001'

# 숫자 데이터 정제
price_text = item_div.select_one('span.price').get_text()
# price_text는 '1,500원'
price = int(price_text.replace(',', '').replace('원', ''))
# 결과: 1500 (정수형)

print(f"상품명: {name}, 링크: {link}, 가격: {price}")
```

---

**핵심 요약:** `requests`로 가져온 HTML을 `BeautifulSoup` 객체로 만들고, **CSS 선택자(`select`, `select_one`)**로 원하는 태그를 정교하게 찾은 뒤, `.get_text(strip=True)`와 `['속성명']`을 이용해 데이터를 추출하고, `.strip()`, `.replace()` 등으로 깨끗하게 정제하는 것이 정적 페이지 크롤링의 핵심 워크플로우입니다.