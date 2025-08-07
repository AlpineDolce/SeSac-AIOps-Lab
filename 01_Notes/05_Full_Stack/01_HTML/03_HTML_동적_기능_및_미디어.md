<h2>웹 페이지의 동적 기능 및 미디어 통합</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 HTML의 핵심 개념과 주요 태그 사용법을 체계적으로 정리하여, 웹 페이지의 구조를 이해하고 작성하는 능력을 기르는 것을 목표로 합니다. 각 태그의 의미와 사용 사례를 통해 시맨틱 웹의 중요성을 이해하고, 풀스택 개발 실무에 필요한 보안, 성능 최적화 관점까지 통합하여 실용적인 예제를 통해 학습 효과를 높이고자 합니다.</p>

<h2>목차</h2>

- [3. 웹 페이지의 동적 기능 및 미디어 통합](#3-웹-페이지의-동적-기능-및-미디어-통합)
  - [3.1. JavaScript 통합: 웹 페이지의 동적 기능 구현](#31-javascript-통합-웹-페이지의-동적-기능-구현)
  - [3.1.1. 웹 워커 (Web Workers) 및 서비스 워커 (Service Workers)](#311-웹-워커-web-workers-및-서비스-워커-service-workers)
  - [3.1.2. HTML 템플릿 (`<template>` 및 `<slot>`)](#312-html-템플릿-template-및-slot)
  - [3.2. `<canvas>`: 그래픽 및 애니메이션](#32-canvas-그래픽-및-애니메이션)
  - [3.3. `<audio>` 및 `<video>`: 미디어 삽입](#33-audio-및-video-미디어-삽입)
  - [3.4. `<iframe>`: 외부 콘텐츠 임베드](#34-iframe-외부-콘텐츠-임베드)

---

## 3. 웹 페이지의 동적 기능 및 미디어 통합

### 3.1. JavaScript 통합: 웹 페이지의 동적 기능 구현
JavaScript는 웹 페이지에 동적인 기능과 상호작용성을 부여하는 프로그래밍 언어입니다. HTML은 웹 페이지의 구조를 정의하고, CSS는 스타일을 입히며, JavaScript는 사용자의 행동에 반응하거나 데이터를 동적으로 변경하는 등의 기능을 담당합니다. 풀스택 개발에서 JavaScript는 프론트엔드(클라이언트 측)와 백엔드(서버 측, Node.js 사용 시) 모두에서 중요한 역할을 합니다.

**HTML에 JavaScript 포함하는 방법:**
1.  **인라인 스크립트**: `<script>` 태그 안에 JavaScript 코드를 직접 작성합니다. (간단한 스크립트에 적합)
2.  **외부 스크립트 파일**: 별도의 `.js` 파일에 JavaScript 코드를 작성하고, `<script src="경로/파일.js"></script>` 형태로 HTML에 연결합니다. (가장 권장되는 방법)

**주요 속성:**
*   `src`: 외부 JavaScript 파일의 경로를 지정합니다.
*   `defer`: HTML 파싱이 완료된 후 스크립트를 실행하도록 지시합니다. 스크립트가 문서의 구조(DOM)에 접근해야 할 때 유용하며, `<head>`에 배치해도 `<body>`가 로드된 후 실행됩니다.
*   `async`: 스크립트를 비동기적으로 로드하고 실행합니다. HTML 파싱과 동시에 스크립트 로드가 시작되며, 로드가 완료되는 즉시 실행됩니다. 스크립트 간의 의존성이 없거나, 페이지 로드 속도가 중요할 때 사용됩니다.

**JavaScript의 역할 (풀스택 관점):**
*   **DOM 조작**: HTML 요소의 내용을 변경하거나, 새로운 요소를 추가/삭제하고, 스타일을 동적으로 변경합니다.
*   **이벤트 처리**: 사용자 클릭, 키보드 입력, 폼 제출 등 다양한 이벤트에 반응하여 특정 동작을 수행합니다.
*   **비동기 통신 (AJAX/Fetch API)**: 페이지를 새로고침하지 않고 서버와 데이터를 주고받아 동적인 콘텐츠를 업데이트합니다. (예: 실시간 검색, 댓글 로딩)
*   **데이터 유효성 검사**: 폼 제출 전에 사용자 입력의 유효성을 클라이언트 측에서 미리 검사하여 서버 부하를 줄이고 사용자 경험을 향상시킵니다.
*   **애니메이션 및 시각 효과**: 복잡한 UI 애니메이션이나 시각 효과를 구현합니다.

**코드 사례 (JavaScript 통합):**
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>JavaScript 예제</title>
    <!-- 외부 JavaScript 파일 연결 (권장) -->
    <script src="script.js" defer></script> 
    <style>
        #myButton {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <h1 id="greeting">안녕하세요!</h1>
    <button id="myButton">클릭하세요</button>

    <script>
        // 인라인 스크립트 예시 (간단한 경우)
        document.getElementById('greeting').style.color = 'blue';

        // 버튼 클릭 이벤트 처리
        document.getElementById('myButton').addEventListener('click', function() {
            alert('버튼이 클릭되었습니다!');
        });
    </script>
</body>
</html>
```
**`script.js` 파일 내용 (외부 스크립트 예시):**
```javascript
// 이 코드는 HTML 파일의 <script src="script.js" defer></script>를 통해 로드됩니다.
console.log("외부 스크립트가 로드되었습니다.");

// 페이지 로드 후 실행될 코드
document.addEventListener('DOMContentLoaded', function() {
    // 여기에 DOM 요소에 접근하는 코드 작성
    // 예: document.getElementById('anotherElement').textContent = '새로운 내용';
});
```

### 3.1.1. 웹 워커 (Web Workers) 및 서비스 워커 (Service Workers)
웹 애플리케이션의 성능과 기능을 향상시키는 데 중요한 역할을 하는 JavaScript API입니다.

- **웹 워커 (Web Workers)**: 메인 스레드와 분리된 백그라운드 스레드에서 JavaScript 코드를 실행할 수 있게 해줍니다. 이를 통해 복잡한 계산이나 대용량 데이터 처리와 같은 시간이 오래 걸리는 작업을 메인 스레드를 블로킹하지 않고 수행하여, 웹 페이지의 응답성과 사용자 경험을 향상시킬 수 있습니다.

    **코드 사례 (메인 스크립트):
    ```javascript
    // main.js
    const worker = new Worker('worker.js');

    worker.postMessage({ command: 'start', data: 1000000000 });

    worker.onmessage = function(event) {
        console.log('메인 스레드에서 받은 메시지:', event.data);
    };

    console.log('메인 스레드 작업 계속...');
    ```

    **코드 사례 (워커 스크립트 - `worker.js`):
    ```javascript
    // worker.js
    onmessage = function(event) {
        const { command, data } = event.data;
        if (command === 'start') {
            let sum = 0;
            for (let i = 0; i < data; i++) {
                sum += i;
            }
            postMessage(sum); // 결과 반환
        }
    };
    ```

- **서비스 워커 (Service Workers)**: 브라우저와 네트워크 사이의 프록시 역할을 하는 JavaScript 파일입니다. 이를 통해 오프라인 지원, 푸시 알림, 리소스 캐싱 등 프로그레시브 웹 앱(PWA)의 핵심 기능을 구현할 수 있습니다. 웹 워커와 마찬가지로 메인 스레드와 독립적으로 작동합니다.
    - **캐싱 전략 (Caching Strategies)**: 서비스 워커는 네트워크 요청을 가로채어 리소스를 캐싱하고 관리하는 다양한 전략을 구현할 수 있습니다. 이는 오프라인 지원 및 성능 최적화에 필수적입니다.
        - **Cache-First**: 캐시에 리소스가 있으면 캐시된 버전을 즉시 반환하고, 없으면 네트워크에서 가져와 캐시합니다. (오프라인 우선)
        - **Network-First**: 항상 네트워크에서 리소스를 먼저 시도하고, 실패하면 캐시된 버전을 반환합니다. (최신 콘텐츠 우선)
        - **Stale-While-Revalidate**: 캐시된 버전을 즉시 반환하고, 동시에 네트워크에서 최신 버전을 가져와 캐시를 업데이트합니다. (빠른 응답 + 최신성 유지)
        - **Cache-Only**: 캐시된 리소스만 사용합니다. (정적 자산에 적합)
        - **Network-Only**: 항상 네트워크에서만 리소스를 가져옵니다. (캐싱하지 않는 동적 데이터에 적합)

    **코드 사례 (서비스 워커 등록 - `main.js`):
    ```javascript
    // main.js
    if ('serviceWorker' in navigator) {
        window.addEventListener('load', function() {
            navigator.serviceWorker.register('/service-worker.js').then(function(registration) {
                console.log('ServiceWorker registration successful with scope: ', registration.scope);
            }, function(err) {
                console.log('ServiceWorker registration failed: ', err);
            });
        });
    }
    ```

    **코드 사례 (서비스 워커 - `service-worker.js`):
    ```javascript
    // service-worker.js
    const CACHE_NAME = 'my-site-cache-v1';
    const urlsToCache = [
        '/',
        '/index.html',
        '/styles.css',
        '/script.js'
    ];

    self.addEventListener('install', function(event) {
        // Install event: 캐시할 파일들을 미리 캐싱
        event.waitUntil(
            caches.open(CACHE_NAME)
                .then(function(cache) {
                    console.log('Opened cache');
                    return cache.addAll(urlsToCache);
                })
        );
    });

    self.addEventListener('fetch', function(event) {
        // Fetch event: 네트워크 요청을 가로채 캐시에서 응답하거나 네트워크에서 가져옴
        event.respondWith(
            caches.match(event.request)
                .then(function(response) {
                    if (response) {
                        return response; // 캐시에 있으면 캐시된 응답 반환
                    }
                    return fetch(event.request); // 캐시에 없으면 네트워크 요청
                })
        );
    });
    ```

### 3.1.2. HTML 템플릿 (`<template>` 및 `<slot>`)
`<template>` 태그는 페이지 로드 시 즉시 렌더링되지 않는 HTML 콘텐츠를 정의합니다. 이 콘텐츠는 JavaScript를 통해 동적으로 복제되어 문서에 삽입될 수 있습니다. `<slot>` 태그는 웹 컴포넌트(Custom Elements) 내부에서 콘텐츠를 삽입할 위치를 지정하는 데 사용됩니다.

- **`<template>`**: 클라이언트 측에서 재사용 가능한 HTML 구조를 정의할 때 유용합니다. 브라우저는 `<template>` 내부의 콘텐츠를 파싱하지만 렌더링하지 않으며, 스크립트도 실행하지 않습니다.
- **`<slot>`**: 웹 컴포넌트의 섀도 DOM(Shadow DOM) 내에서 콘텐츠를 외부로부터 주입받을 수 있는 플레이스홀더 역할을 합니다. 이를 통해 컴포넌트의 유연성과 재사용성을 높일 수 있습니다.

**코드 사례 (`<template>` 및 `<slot>`):**
```html
<!-- HTML 템플릿 정의 -->
<template id="my-card-template">
  <style>
    .card {
      border: 1px solid #ccc;
      padding: 10px;
      margin: 10px;
    }
    h3 { color: blue; }
  </style>
  <div class="card">
    <h3><slot name="card-title">기본 제목</slot></h3>
    <p><slot name="card-content">기본 내용</slot></p>
  </div>
</template>

<!-- 커스텀 엘리먼트 정의 (JavaScript) -->
<script>
  class MyCard extends HTMLElement {
    constructor() {
      super();
      const template = document.getElementById('my-card-template').content;
      const shadowRoot = this.attachShadow({ mode: 'open' });
      shadowRoot.appendChild(template.cloneNode(true));
    }
  }
  customElements.define('my-card', MyCard);
</script>

<!-- HTML에서 커스텀 엘리먼트 사용 -->
<my-card>
  <span slot="card-title">특별한 카드</span>
  <p slot="card-content">이것은 슬롯을 통해 삽입된 내용입니다.</p>
</my-card>

<my-card></my-card> <!-- 기본 제목과 내용 사용 -->
```

### 3.2. `<canvas>`: 그래픽 및 애니메이션
JavaScript를 사용하여 2D 그래픽, 애니메이션, 게임 등을 동적으로 그릴 수 있는 영역을 만듭니다.

**코드 사례 (`캔버스1.html`)**:
```html
<canvas id="myCanvas" width="200" height="100" style="border:1px solid #000;"></canvas>
<script>
    var c = document.getElementById("myCanvas"); 
    var ctx = c.getContext("2d");
    ctx.fillStyle = "red";
    ctx.fillRect(20, 20, 150, 75); // x, y, width, height
</script>
```

### 3.3. `<audio>` 및 `<video>`: 미디어 삽입
웹 페이지에 오디오나 비디오 파일을 삽입하여 재생할 수 있게 합니다. **`controls`** 속성을 추가하면 재생 컨트롤러가 표시됩니다. `autoplay` (자동 재생), `loop` (반복 재생), `muted` (음소거) 등의 속성을 사용할 수 있습니다.

**코드 사례:**
```html
<video width="400" controls poster="poster.jpg">
  <source src="medias/flower.mp4" type="video/mp4">
  <source src="medias/flower.webm" type="video/webm">
  브라우저가 video 태그를 지원하지 않습니다.
</video>
```
* `<source>` 태그를 여러 개 사용하여 브라우저가 지원하는 첫 번째 비디오 형식을 재생하도록 할 수 있습니다.
* `poster` 속성은 비디오가 재생되기 전에 표시될 이미지의 URL을 지정합니다.

### 3.4. `<iframe>`: 외부 콘텐츠 임베드
현재 HTML 페이지 안에 다른 HTML 페이지를 삽입(임베드)할 때 사용합니다. 유튜브 비디오나 구글 지도를 페이지에 포함시키는 데 흔히 사용됩니다.

- **보안 관련 `sandbox` 속성 강조**: `<iframe>`은 외부 콘텐츠를 삽입하므로 보안 취약점이 될 수 있습니다. `sandbox` 속성을 사용하여 임베드된 콘텐츠의 권한을 제한하는 것이 매우 중요합니다.
    - `sandbox`: 모든 제한을 적용합니다.
    - `sandbox="allow-scripts allow-same-origin"`: 특정 권한만 허용합니다.
- **성능 최적화**: `<img>` 태그와 마찬가지로 `<iframe>` 태그에도 `loading="lazy"` 속성을 추가하여 뷰포트(viewport)에 들어올 때까지 로드를 지연시켜 초기 페이지 로딩 성능을 개선할 수 있습니다.

**코드 사례:**
```html
<iframe src="https://google.com/maps/embed?..." 
        width="600" 
        height="450" 
        style="border:0;" 
        allowfullscreen="" 
        loading="lazy">
</iframe>

<!-- 외부 신뢰할 수 없는 콘텐츠를 삽입할 때 sandbox 속성을 사용하여 보안 강화 -->
<iframe src="https://untrusted-content.com/some-page.html"
        width="600"
        height="400"
        sandbox="allow-scripts allow-forms"> <!-- 스크립트와 폼 제출만 허용 -->
</iframe>
```