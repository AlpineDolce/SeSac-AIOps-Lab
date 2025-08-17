<h2>[Part 4-3] Scrapy 미들웨어 및 설정</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-17

<h2>문서 목표</h2>
<p>이 문서는 Scrapy 크롤러의 동작을 전역적으로 제어하는 `settings.py`의 주요 설정과, 요청과 응답을 중간에서 가로채 동적인 처리를 추가하는 미들웨어의 개념을 이해합니다. 특히 다운로더 미들웨어를 활용하여 User-Agent와 Proxy를 동적으로 변경하는 실용적인 방법을 학습하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. `settings.py`: 크롤러의 두뇌](#1-settingspy-크롤러의-두뇌)
- [2. 미들웨어(Middleware)란?](#2-미들웨어middleware란)
- [3. 다운로더 미들웨어로 User-Agent 무작위 변경하기](#3-다운로더-미들웨어로-user-agent-무작위-변경하기)
- [4. 다운로더 미들웨어로 IP 차단 우회하기 (Proxy)](#4-다운로더-미들웨어로-ip-차단-우회하기-proxy)

---

## 1. `settings.py`: 크롤러의 두뇌

`settings.py` 파일은 Scrapy 프로젝트의 모든 동작을 제어하는 중앙 통제실입니다. 스파이더 코드에 하드코딩하는 대신, 설정을 이곳에 모아두면 관리가 용이합니다.

**주요 기본 설정:**

- **`USER_AGENT`**: 크롤러의 유저 에이전트 문자열. `requests`에서 헤더를 설정하듯, 여기에 설정하면 모든 요청에 기본으로 적용됩니다.
- **`ROBOTSTXT_OBEY`**: `robots.txt` 파일의 규칙을 준수할지 여부 (기본값: `True`). 사이트에 부담을 주지 않기 위해 `True`로 두는 것을 권장합니다.
- **`DOWNLOAD_DELAY`**: 각 요청 사이에 지연 시간을 설정 (단위: 초). 서버에 과도한 부하를 주는 것을 방지하는 가장 기본적인 방법입니다. `1` 또는 `2`로 설정하는 것이 일반적입니다.
- **`CONCURRENT_REQUESTS`**: 동시에 보낼 수 있는 최대 요청 수 (기본값: `16`). `DOWNLOAD_DELAY`와 함께 조절하여 크롤링 속도와 안정성 사이의 균형을 맞춥니다.

```python
# mycrawler/settings.py

# 기본 User-Agent 설정
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# robots.txt 규칙 준수
ROBOTSTXT_OBEY = True

# 2초의 다운로드 지연 설정
DOWNLOAD_DELAY = 2
```

## 2. 미들웨어(Middleware)란?

미들웨어는 Scrapy 엔진과 다른 컴포넌트(Downloader, Spider) 사이의 관문 역할을 하는 처리 계층입니다. 요청이 인터넷으로 나가기 전, 또는 응답이 스파이더로 전달되기 전에 가로채서 원하는 처리를 추가할 수 있습니다.

- **다운로더 미들웨어 (Downloader Middleware):** Engine ↔ Downloader 사이에 위치. **요청(Request)을 보내기 직전**이나 **응답(Response)을 받자마자** 특정 로직을 수행합니다. **프록시(Proxy)나 User-Agent를 동적으로 변경**하는 작업이 주로 여기서 이루어집니다.

## 3. 다운로더 미들웨어로 User-Agent 무작위 변경하기

매번 똑같은 User-Agent로 요청을 보내면 사이트에서 봇으로 인지하고 차단하기 쉽습니다. 여러 개의 실제 브라우저 User-Agent를 리스트로 만들어두고, 요청마다 무작위로 선택하여 보내는 것이 좋습니다.

**1. `middlewares.py`에 미들웨어 작성:**
```python
# mycrawler/middlewares.py
import random

class RandomUserAgentMiddleware:
    def __init__(self):
        # 실제 브라우저들의 User-Agent 리스트
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ...',
            'Mozilla/5.0 (X11; Linux x86_64) ...'
        ]

    def process_request(self, request, spider):
        # 요청을 보내기 직전에 이 메서드가 호출됩니다.
        # 무작위로 User-Agent를 선택하여 요청 헤더에 설정합니다.
        request.headers['User-Agent'] = random.choice(self.user_agents)
```

**2. `settings.py`에서 미들웨어 활성화:**
```python
# mycrawler/settings.py
DOWNLOADER_MIDDLEWARES = {
   # 숫자는 실행 순서. 낮을수록 먼저 실행됨
   'mycrawler.middlewares.RandomUserAgentMiddleware': 543,
}
```

## 4. 다운로더 미들웨어로 IP 차단 우회하기 (Proxy)

동일한 IP에서 단시간에 너무 많은 요청을 보내면 서버가 해당 IP를 차단할 수 있습니다. 유료 또는 무료 프록시 서버 목록을 확보했다면, 미들웨어를 통해 각 요청이 다른 IP를 통해 나가도록 설정할 수 있습니다.

**1. `middlewares.py`에 프록시 미들웨어 작성:**
```python
# mycrawler/middlewares.py
import random

class ProxyMiddleware:
    def __init__(self):
        # 실제로는 파일이나 API로부터 프록시 목록을 동적으로 가져와야 합니다.
        self.proxies = [
            'http://proxy_ip_1:port',
            'http://proxy_ip_2:port',
            'https://username:password@proxy_ip_3:port' # 인증이 필요한 경우
        ]

    def process_request(self, request, spider):
        # 요청의 meta 딕셔너리에 'proxy' 키로 프록시 주소를 설정합니다.
        proxy = random.choice(self.proxies)
        request.meta['proxy'] = proxy
        spider.log(f"Using proxy: {proxy}")
```

**2. `settings.py`에서 미들웨어 활성화:**
```python
# mycrawler/settings.py
DOWNLOADER_MIDDLEWARES = {
   'mycrawler.middlewares.RandomUserAgentMiddleware': 543,
   'mycrawler.middlewares.ProxyMiddleware': 600, # 프록시 미들웨어 추가
}
```

---

**핵심 요약:** **정적인 설정은 `settings.py`**에서, **동적인 처리는 미들웨어**에서 담당합니다. 특히 다운로더 미들웨어는 User-Agent, Proxy 등 차단 우회 전략을 구현하는 핵심적인 장소입니다. 이 둘을 잘 활용하면 매우 유연하고 강력한 크롤러를 만들 수 있습니다.