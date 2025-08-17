<h2>[Part 4-2] Scrapy 아이템 파이프라인</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-17

<h2>문서 목표</h2>
<p>이 문서는 Scrapy의 핵심 컴포넌트인 아이템 파이프라인의 역할과 중요성을 이해합니다. 스파이더와 후처리 로직을 분리하고, 파이프라인을 통해 데이터 정제, 유효성 검사, 데이터베이스 저장 등 연쇄적인 작업을 자동화하는 방법을 학습하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 스파이더와 저장 로직의 분리](#1-스파이더와-저장-로직의-분리)
- [2. 데이터의 뼈대: 아이템(Item) 정의하기](#2-데이터의-뼈대-아이템item-정의하기)
- [3. 파이프라인 작성하기](#3-파이프라인-작성하기)
- [4. 파이프라인 활성화하기](#4-파이프라인-활성화하기)

---

## 1. 스파이더와 저장 로직의 분리

스파이더의 핵심 책임은 웹 페이지에서 **데이터를 추출**하는 것입니다. 만약 스파이더의 `parse` 메서드 안에 데이터 정제, 유효성 검사, DB 저장 등 온갖 후처리 로직을 다 넣는다면 어떻게 될까요? 스파이더는 금방 거대하고 복잡해져 유지보수가 불가능한 상태가 될 것입니다.

**아이템 파이프라인(Item Pipeline)**은 스파이더로부터 분리된, 데이터 후처리를 위한 전용 컴포넌트입니다. 스파이더가 데이터를 `yield`하면, 이 데이터는 파이프라인으로 전달되어 여러 단계를 거치며 체계적으로 처리됩니다. 이는 마치 공장의 조립 라인과 같습니다.

**파이프라인의 역할:**
- 데이터 정제 (e.g., HTML 태그 제거, 공백 정리)
- 데이터 유효성 검사 (e.g., 필수 필드가 비어있는 아이템은 버리기)
- 중복 데이터 처리 (e.g., 이미 수집된 데이터인지 확인)
- 데이터베이스, 파일 등 다양한 저장소에 저장

## 2. 데이터의 뼈대: 아이템(Item) 정의하기

파이프라인을 사용하기 전에, 우리가 수집할 데이터의 구조(스키마)를 `items.py` 파일에 미리 정의하는 것이 좋습니다. 이는 코드의 가독성을 높이고 실수를 줄여줍니다.

```python
# mycrawler/items.py
import scrapy

class QuoteItem(scrapy.Item):
    # 우리가 수집할 데이터의 필드를 정의합니다.
    text = scrapy.Field()
    author = scrapy.Field()
    tags = scrapy.Field()
```

이제 스파이더에서는 딕셔너리 대신 이 `QuoteItem` 객체를 `yield` 합니다.

```python
# mycrawler/spiders/quotes.py
from mycrawler.items import QuoteItem

# ... (상단 코드 생략)

    def parse(self, response):
        for quote in response.css('div.quote'):
            item = QuoteItem()
            item['text'] = quote.css('span.text::text').get()
            item['author'] = quote.css('small.author::text').get()
            item['tags'] = quote.css('div.tags a.tag::text').getall()
            yield item
```

## 3. 파이프라인 작성하기

실제 후처리 로직은 `pipelines.py` 파일 안에 작성합니다. 파이프라인은 `process_item(self, item, spider)` 메서드를 가진 클래스입니다.

**예시 1: 데이터 정제 및 유효성 검사 파이프라인**
```python
# mycrawler/pipelines.py
from scrapy.exceptions import DropItem

class TextPipeline:
    def process_item(self, item, spider):
        # author 필드의 양쪽 공백 제거
        if 'author' in item:
            item['author'] = item['author'].strip()
        
        # text 필드가 없는 아이템은 버리기(DropItem 예외 발생)
        if not item.get('text'):
            raise DropItem(f"Missing text in {item}")
            
        # 처리가 끝난 아이템을 반드시 반환해야 함
        return item
```

**예시 2: SQLite 데이터베이스 저장 파이프라인**
```python
# mycrawler/pipelines.py
import sqlite3

class SQLitePipeline:
    def open_spider(self, spider):
        # 스파이더가 시작될 때 호출됨
        self.connection = sqlite3.connect("quotes.db")
        self.cursor = self.connection.cursor()
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS quotes(
            text TEXT,
            author TEXT,
            tags TEXT
        )
        """)
        self.connection.commit()

    def close_spider(self, spider):
        # 스파이더가 종료될 때 호출됨
        self.connection.close()

    def process_item(self, item, spider):
        self.cursor.execute("INSERT INTO quotes (text, author, tags) VALUES (?, ?, ?)", (
            item.get('text'),
            item.get('author'),
            ",".join(item.get('tags', [])) # 태그 리스트를 문자열로 변환
        ))
        self.connection.commit()
        return item
```

## 4. 파이프라인 활성화하기

파이프라인 클래스를 작성한 후, Scrapy 엔진이 이를 사용하도록 `settings.py` 파일에 등록해야 합니다.

```python
# mycrawler/settings.py

# ITEM_PIPELINES 설정의 주석을 해제하고 파이프라인을 등록합니다.
ITEM_PIPELINES = {
   # 숫자는 파이프라인의 실행 순서. 낮을수록 먼저 실행됩니다.
   'mycrawler.pipelines.TextPipeline': 300,
   'mycrawler.pipelines.SQLitePipeline': 400,
}
```
이제 스파이더를 실행하면, `yield`된 모든 아이템은 `TextPipeline`을 먼저 거쳐 정제된 후, `SQLitePipeline`을 통해 데이터베이스에 저장됩니다.

---

**핵심 요약:** 스파이더는 **추출**에만 집중하고, 데이터 **후처리(정제, 검증, 저장)는 파이프라인에 위임**하세요. 이는 코드의 **재사용성**과 **유지보수성**을 극대화하는 Scrapy의 핵심 설계 사상입니다.