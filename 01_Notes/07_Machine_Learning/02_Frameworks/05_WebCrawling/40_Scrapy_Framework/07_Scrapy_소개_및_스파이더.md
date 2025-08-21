<h2>[Part 4-1] Scrapy 소개 및 스파이더</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-17

<h2>문서 목표</h2>
<p>이 문서는 Scrapy 프레임워크의 구조와 핵심 컴포넌트를 이해하고, `startproject`, `genspider` 명령으로 프로젝트를 구성하며, 데이터 추출의 핵심인 Spider를 작성하고 실행하는 방법을 학습하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 왜 Scrapy를 사용하는가?](#1-왜-scrapy를-사용하는가)
- [2. Scrapy 프로젝트 시작하기](#2-scrapy-프로젝트-시작하기)
- [3. 데이터 추출의 핵심: 스파이더(Spider)](#3-데이터-추출의-핵심-스파이더spider)
- [4. 데이터 추출 및 실행](#4-데이터-추출-및-실행)

---

## 1. 왜 Scrapy를 사용하는가?

`requests`와 `BeautifulSoup`이 개별 부품(타이어, 핸들)이라면, Scrapy는 잘 만들어진 자동차(프레임워크)입니다. Scrapy는 비동기 네트워크 처리를 기반으로 설계되어 매우 빠르며, 대규모 크롤링 프로젝트를 위한 체계적인 구조를 제공합니다.

단순히 요청하고 파싱하는 것을 넘어, 데이터 수집, 처리, 저장의 전 과정을 관리하는 강력한 기능을 내장하고 있어, 개발자는 **"무엇을"** 가져올지에만 집중하면 됩니다.

## 2. Scrapy 프로젝트 시작하기

**1. 설치:**
```bash
pip install Scrapy
```

**2. 프로젝트 생성:**
```bash
# mycrawler라는 이름의 프로젝트 생성
scrapy startproject mycrawler
```
이 명령을 실행하면 다음과 같은 표준 디렉토리 구조가 생성됩니다.
```
mycrawler/
├── scrapy.cfg            # 프로젝트 배포 설정 파일
└── mycrawler/              # 프로젝트의 파이썬 모듈
    ├── __init__.py
    ├── items.py          # 데이터 구조(스키마) 정의
    ├── middlewares.py    # 요청/응답 중간 처리기
    ├── pipelines.py      # 데이터 후처리 및 저장
    ├── settings.py       # 크롤러 설정 파일
    └── spiders/          # 스파이더(실제 크롤링 로직) 폴더
        └── __init__.py
```

## 3. 데이터 추출의 핵심: 스파이더(Spider)

스파이더는 **어떤 웹사이트를 어떻게 크롤링할지 정의하는 클래스**입니다. 어디서부터 시작해서, 어떤 링크를 따라가고, 각 페이지에서 어떤 데이터를 추출할지 명시합니다.

**1. 스파이더 생성:**
`spiders` 디렉토리로 이동할 필요 없이, 프로젝트 최상위 폴더에서 다음 명령을 실행합니다.
```bash
# quotes라는 이름의 스파이더를 생성. 대상 도메인은 quotes.toscrape.com
scrapy genspider quotes quotes.toscrape.com
```
`mycrawler/spiders/quotes.py` 파일이 자동으로 생성됩니다.

**2. 스파이더의 기본 구조:**
```python
# mycrawler/spiders/quotes.py
import scrapy

class QuotesSpider(scrapy.Spider):
    # 1. 스파이더의 고유 이름 (필수)
    name = "quotes"
    
    # 2. 크롤링을 허용할 도메인 (선택 사항)
    allowed_domains = ["quotes.toscrape.com"]
    
    # 3. 크롤링을 시작할 URL 리스트 (필수)
    start_urls = ["http://quotes.toscrape.com/"]

    # 4. start_urls의 각 URL에 대한 응답을 처리하는 기본 콜백 메서드
    def parse(self, response):
        # 이 메서드 안에 데이터 추출 로직을 작성합니다.
        pass
```

## 4. 데이터 추출 및 실행

Scrapy의 `response` 객체는 자체적으로 CSS 및 XPath 선택자 기능을 내장하고 있어, `BeautifulSoup`을 따로 쓸 필요가 없습니다.

- **`response.css('선택자')`**: CSS 선택자로 요소들을 선택합니다.
- **`.get()`**: 선택된 요소 중 **첫 번째** 것의 내용을 가져옵니다.
- **`.getall()`**: 선택된 요소 **모든** 것의 내용을 리스트로 가져옵니다.
- **`::text`**: 태그 내부의 텍스트 콘텐츠를 의미합니다.
- **`::attr(속성명)`**: 태그의 속성 값을 의미합니다. (e.g., `::attr(href)`)

**데이터 추출 로직을 포함한 `parse` 메서드 예시:**
```python
# mycrawler/spiders/quotes.py

# ... (상단 코드는 동일)

    def parse(self, response):
        self.log(f'크롤링 시작: {response.url}')

        # 페이지의 모든 명언(div.quote)을 순회
        for quote in response.css('div.quote'):
            # yield 키워드는 추출된 데이터를 Scrapy 엔진으로 하나씩 전달하는 역할을 함
            yield {
                'text': quote.css('span.text::text').get(),
                'author': quote.css('small.author::text').get(),
                'tags': quote.css('div.tags a.tag::text').getall(),
            }
```

**스파이더 실행:**
프로젝트 최상위 폴더(`scrapy.cfg` 파일이 있는 곳)에서 다음 명령을 실행합니다.
```bash
# quotes 스파이더를 실행하고, 결과를 quotes.json 파일로 저장(-o)
scrapy crawl quotes -o quotes.json
```
이 명령 한 줄로 Scrapy는 `quotes` 스파이더를 찾아 실행하고, `yield`된 모든 데이터를 모아 `quotes.json` 파일로 깔끔하게 저장해줍니다.

---

**핵심 요약:** Scrapy는 **`startproject`로 프로젝트를 만들고, `genspider`로 스파이더를 생성**하는 것으로 시작합니다. 스파이더의 **`parse` 메서드 안에서 `response.css()`를 이용해 데이터를 추출**하고 `yield`로 반환하는 것이 핵심 로직입니다. 실행은 **`scrapy crawl`** 명령어로 합니다.