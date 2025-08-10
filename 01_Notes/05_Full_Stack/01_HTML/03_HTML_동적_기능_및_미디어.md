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

1.  **개요 (Overview)**
    *   JavaScript는 웹 페이지에 동적인 기능과 상호작용성을 부여하는 프로그래밍 언어입니다. HTML은 웹 페이지의 구조를 정의하고, CSS는 스타일을 입히며, JavaScript는 사용자의 행동에 반응하거나 데이터를 동적으로 변경하는 등의 기능을 담당합니다. 풀스택 개발에서 JavaScript는 프론트엔드(클라이언트 측)와 백엔드(서버 측, Node.js 사용 시) 모두에서 중요한 역할을 합니다.

2.  **HTML에 JavaScript 포함하는 방법 (Methods to Include JavaScript in HTML)**
    *   **인라인 스크립트**: `<script>` 태그 안에 JavaScript 코드를 직접 작성합니다. (간단한 스크립트나 테스트에 적합)
        ```html
        <script>
            console.log("Hello from inline script!");
        </script>
        ```
    *   **외부 스크립트 파일**: 별도의 `.js` 파일에 JavaScript 코드를 작성하고, `<script src="경로/파일.js"></script>` 형태로 HTML에 연결합니다. (가장 권장되는 방법)
        *   **장점**: 코드 재사용성, 유지보수성, 캐싱을 통한 성능 향상.

3.  **스크립트 로딩 전략 (`<script>` 태그의 속성) (Script Loading Strategies)**
    *   **기본 동작 (Blocking)**: `async`나 `defer` 속성 없이 `<script>` 태그를 사용하면, 브라우저는 HTML 파싱을 중단하고 스크립트를 다운로드하여 즉시 실행합니다. 스크립트 실행이 완료될 때까지 HTML 파싱이 재개되지 않아 페이지 로딩이 지연될 수 있습니다.
    *   **`defer` 속성**:
        *   **동작**: HTML 파싱과 스크립트 다운로드가 동시에 진행됩니다. 스크립트 실행은 HTML 파싱이 완전히 완료된 후, `DOMContentLoaded` 이벤트가 발생하기 직전에 이루어집니다.
        *   **사용 시점**: 스크립트가 문서의 구조(DOM)에 접근하거나, 다른 스크립트와의 실행 순서가 중요한 경우에 유용합니다. `<head>`에 배치해도 `<body>`가 로드된 후 실행되므로, DOM 조작 코드를 안전하게 작성할 수 있습니다.
    *   **`async` 속성**:
        *   **동작**: HTML 파싱과 스크립트 다운로드가 동시에 진행됩니다. 스크립트 다운로드가 완료되는 즉시 HTML 파싱을 중단하고 스크립트를 실행합니다. 스크립트 실행이 완료되면 HTML 파싱이 재개됩니다.
        *   **사용 시점**: 스크립트 간의 의존성이 없거나, 페이지 로드 속도가 매우 중요하여 스크립트가 가능한 한 빨리 실행되어야 할 때 사용됩니다. (예: 분석 스크립트, 광고 스크립트)
    *   **모듈 (`type="module"`)**:
        *   **동작**: ES Modules 표준을 따르는 JavaScript 파일을 로드할 때 사용합니다. `defer`와 유사하게 동작하지만, 모듈 간의 의존성 관리가 가능하며, 기본적으로 엄격 모드(strict mode)로 실행됩니다.
        *   **장점**: 코드의 모듈화, 재사용성, 명확한 의존성 관리.
        *   **사용 시점**: 현대적인 웹 개발에서 코드 베이스를 구조화하고 관리하는 데 필수적입니다.

4.  **JavaScript의 핵심 역할 (Core Roles of JavaScript)**
    *   **DOM(Document Object Model) 조작**: HTML 요소의 내용을 변경하거나, 새로운 요소를 추가/삭제하고, 스타일을 동적으로 변경합니다. 웹 페이지의 동적인 UI를 구현하는 데 필수적입니다.
    *   **이벤트 처리 (Event Handling)**: 사용자 클릭, 키보드 입력, 폼 제출, 마우스 오버 등 다양한 이벤트에 반응하여 특정 동작을 수행합니다.
    *   **비동기 통신 (Asynchronous Communication - AJAX/Fetch API)**: 페이지를 새로고침하지 않고 서버와 데이터를 주고받아 동적인 콘텐츠를 업데이트합니다.
        *   **Fetch API**: 최신 웹 표준으로, 네트워크 요청을 보내고 응답을 처리하는 데 사용되는 강력하고 유연한 인터페이스입니다. Promise 기반으로 동작하여 비동기 코드 작성을 간결하게 만듭니다.
    *   **데이터 유효성 검사 (Data Validation)**: 폼 제출 전에 사용자 입력의 유효성을 클라이언트 측에서 미리 검사하여 서버 부하를 줄이고 사용자 경험을 향상시킵니다. (보안을 위해 서버 측 유효성 검사는 필수)
    *   **애니메이션 및 시각 효과 (Animations & Visual Effects)**: CSS 애니메이션만으로는 구현하기 어려운 복잡한 UI 애니메이션이나 시각 효과를 구현합니다.
    *   **데이터 처리 및 로직 (Data Processing & Logic)**: 클라이언트 측에서 데이터를 가공하거나 복잡한 비즈니스 로직을 수행합니다.

5.  **풀스택 관점에서의 JavaScript (JavaScript from a Full-Stack Perspective)**
    *   **프론트엔드 (클라이언트 측)**: 사용자 인터페이스(UI)의 동적인 동작, 사용자 경험(UX) 개선, 서버와의 비동기 통신을 담당합니다. React, Vue, Angular와 같은 프론트엔드 프레임워크는 JavaScript를 기반으로 복잡한 UI 개발을 효율적으로 돕습니다.
    *   **백엔드 (서버 측)**: Node.js 런타임을 사용하면 JavaScript로 서버 애플리케이션을 개발할 수 있습니다. Express.js, NestJS와 같은 프레임워크를 통해 RESTful API, 웹 소켓, 데이터베이스 연동 등을 구현하여 풀스택 JavaScript 개발이 가능해집니다.

6.  **보안 고려사항 (Security Considerations)**
    *   **XSS (Cross-Site Scripting) 방지**: 사용자로부터 입력받은 데이터를 HTML에 직접 삽입할 경우, 악의적인 스크립트가 실행될 수 있습니다. 항상 사용자 입력은 **살균(Sanitization)** 처리하거나 적절히 **이스케이프(Escaping)** 처리하여 XSS 공격을 방지해야 합니다.

7.  **성능 최적화 (Performance Optimization)**
    *   **코드 압축 및 번들링 (Minification & Bundling)**: 배포 시 JavaScript 파일을 압축하고 여러 파일을 하나로 묶어 파일 크기를 줄이고 네트워크 요청 수를 최소화합니다.
    *   **CDN (Content Delivery Network) 사용**: 스크립트를 CDN을 통해 제공하여 사용자에게 가장 가까운 서버에서 파일을 전송함으로써 로딩 속도를 향상시킵니다.
    *   **지연 로딩 (Lazy Loading)**: 당장 필요하지 않은 스크립트는 필요할 때까지 로드를 지연시켜 초기 페이지 로드 시간을 단축합니다.

**코드 사례 (JavaScript 통합 및 모듈 사용):**
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>JavaScript 통합 예제</title>
    <!-- 외부 JavaScript 파일 연결 (defer 속성 사용) -->
    <script src="script.js" defer></script>
    <!-- 모듈 스크립트 연결 -->
    <script type="module" src="module.js"></script>
    <style>
        #myButton {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        #myButton:hover {
            background-color: #0056b3;
        }
        #dataDisplay {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <h1 id="greeting">안녕하세요!</h1>
    <button id="myButton">데이터 불러오기</button>
    <div id="dataDisplay">여기에 데이터가 표시됩니다.</div>

    <script>
        // 인라인 스크립트 예시: 페이지 로드 시 인사말 색상 변경
        document.getElementById('greeting').style.color = '#28a745'; // 녹색으로 변경
    </script>
</body>
</html>
```
**`script.js` 파일 내용 (외부 스크립트 예시):**
```javascript
// 이 코드는 HTML 파일의 <script src="script.js" defer></script>를 통해 로드됩니다.
console.log("외부 스크립트: script.js가 로드되었습니다.");

// DOMContentLoaded 이벤트는 HTML 파싱이 완료되고 DOM 트리가 준비되었을 때 발생합니다.
document.addEventListener('DOMContentLoaded', function() {
    const myButton = document.getElementById('myButton');
    const dataDisplay = document.getElementById('dataDisplay');

    myButton.addEventListener('click', async function() {
        dataDisplay.textContent = '데이터를 불러오는 중...';
        try {
            // Fetch API를 사용하여 비동기적으로 데이터 불러오기
            const response = await fetch('https://jsonplaceholder.typicode.com/todos/1');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            dataDisplay.textContent = `불러온 데이터: ${JSON.stringify(data, null, 2)}`;
        } catch (error) {
            dataDisplay.textContent = `데이터 불러오기 실패: ${error.message}`;
            console.error('Fetch error:', error);
        }
    });
});
```
**`module.js` 파일 내용 (모듈 스크립트 예시):**
```javascript
// 이 파일은 <script type="module" src="module.js"></script>를 통해 로드됩니다.
// 모듈은 기본적으로 defer처럼 동작하며, 자체 스코프를 가집니다.

export function greet(name) {
    return `안녕하세요, ${name}님!`;
}

// 다른 모듈에서 import하여 사용할 수 있습니다.
// 예: import { greet } from './module.js';

console.log("모듈 스크립트: module.js가 로드되었습니다.");
```
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

1.  **개요 및 목적 (Overview and Purpose)**
    *   HTML `<canvas>` 요소는 JavaScript를 사용하여 2D 그래픽, 애니메이션, 게임, 데이터 시각화, 이미지 처리 등을 동적으로 그릴 수 있는 **비트맵 캔버스** 영역을 제공합니다.
    *   픽셀 기반으로 작동하며, 모든 그래픽 작업은 JavaScript API를 통해 직접 픽셀을 조작하여 이루어집니다.

2.  **기본 설정 및 렌더링 컨텍스트 (Basic Setup and Rendering Context)**
    *   `<canvas>` 태그 자체는 그리기 영역을 정의할 뿐, 실제 그리기는 JavaScript를 통해 이루어집니다.
    *   **렌더링 컨텍스트 가져오기**: `getContext()` 메서드를 사용하여 캔버스에 그림을 그릴 수 있는 렌더링 컨텍스트를 가져옵니다.
        *   `'2d'`: 2D 그래픽을 그리기 위한 표준 API (가장 흔하게 사용).
        *   `'webgl'` 또는 `'webgl2'`: 3D 그래픽을 그리기 위한 WebGL API.
    *   **크기 설정**: `width`와 `height` 속성을 사용하여 캔버스의 크기를 지정합니다. CSS로 크기를 조절할 수도 있지만, 이는 캔버스 내용을 늘리거나 줄일 뿐 실제 해상도를 변경하지 않으므로, 픽셀 단위의 정확한 크기는 HTML 속성으로 지정하는 것이 좋습니다.

3.  **기본 도형 그리기 (Drawing Basic Shapes)**
    *   **사각형 (Rectangles)**:
        *   `fillRect(x, y, width, height)`: 채워진 사각형을 그립니다.
        *   `strokeRect(x, y, width, height)`: 윤곽선만 있는 사각형을 그립니다.
        *   `clearRect(x, y, width, height)`: 지정된 사각형 영역을 투명하게 지웁니다.
    *   **경로 (Paths)**: 복잡한 도형이나 선을 그릴 때 사용합니다.
        *   `beginPath()`: 새로운 경로를 시작합니다.
        *   `moveTo(x, y)`: 경로의 시작점을 이동합니다.
        *   `lineTo(x, y)`: 현재 위치에서 지정된 점까지 선을 그립니다.
        *   `arc(x, y, radius, startAngle, endAngle, counterclockwise)`: 원 또는 호를 그립니다.
        *   `closePath()`: 현재 경로를 시작점으로 닫습니다.
        *   `stroke()`: 현재 경로의 윤곽선을 그립니다.
        *   `fill()`: 현재 경로를 채웁니다.

4.  **스타일 및 색상 (Styles and Colors)**
    *   `fillStyle`: 도형을 채울 색상, 그라디언트 또는 패턴을 설정합니다.
    *   `strokeStyle`: 도형의 윤곽선 색상, 그라디언트 또는 패턴을 설정합니다.
    *   `lineWidth`: 선의 두께를 설정합니다.
    *   `lineCap`: 선의 끝 모양을 설정합니다 (`butt`, `round`, `square`).
    *   `lineJoin`: 선이 만나는 모서리 모양을 설정합니다 (`round`, `bevel`, `miter`).
    *   **그라디언트 (Gradients)**:
        *   `createLinearGradient(x0, y0, x1, y1)`: 선형 그라디언트를 생성합니다.
        *   `createRadialGradient(x0, y0, r0, x1, y1, r1)`: 방사형 그라디언트를 생성합니다.
        *   `addColorStop(offset, color)`: 그라디언트에 색상 정지점을 추가합니다.

5.  **텍스트 및 이미지 (Text and Images)**
    *   **텍스트**:
        *   `font`: 텍스트의 글꼴 스타일을 설정합니다 (CSS `font` 속성과 유사).
        *   `fillText(text, x, y, maxWidth)`: 채워진 텍스트를 그립니다.
        *   `strokeText(text, x, y, maxWidth)`: 윤곽선만 있는 텍스트를 그립니다.
        *   `textAlign`, `textBaseline`: 텍스트 정렬 및 기준선을 설정합니다.
    *   **이미지**:
        *   `drawImage(image, dx, dy)`: 이미지를 캔버스에 그립니다.
        *   `drawImage(image, dx, dy, dWidth, dHeight)`: 이미지 크기를 조절하여 그립니다.
        *   `drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)`: 이미지의 특정 부분을 잘라내어 그립니다.

6.  **변환 (Transformations)**
    *   `save()`: 현재 캔버스 상태(변환, 스타일 등)를 스택에 저장합니다.
    *   `restore()`: 스택에서 가장 최근에 저장된 상태를 복원합니다.
    *   `translate(x, y)`: 캔버스의 원점(0,0)을 이동합니다.
    *   `rotate(angle)`: 캔버스를 회전합니다 (라디안 단위).
    *   `scale(x, y)`: 캔버스의 크기를 조절합니다.

7.  **애니메이션 (Animation)**
    *   `requestAnimationFrame(callback)`: 브라우저에게 다음 리페인트(repaint) 시점에 애니메이션 프레임을 업데이트하도록 요청합니다. 부드러운 애니메이션을 구현하는 데 최적화된 방법입니다.
    *   애니메이션 루프는 일반적으로 `clearRect`로 이전 프레임을 지우고, 새로운 위치에 요소를 다시 그린 후, `requestAnimationFrame`을 다시 호출하는 방식으로 구현됩니다.

8.  **상호작용 (Interactivity)**
    *   캔버스 요소에 마우스 및 키보드 이벤트 리스너를 추가하여 사용자 입력을 감지하고 반응할 수 있습니다.
    *   `event.clientX`, `event.clientY` 등을 사용하여 캔버스 내의 마우스 좌표를 얻을 수 있습니다.

9.  **성능 최적화 팁 (Performance Optimization Tips)**
    *   **불필요한 그리기 피하기**: 변경된 부분만 다시 그립니다.
    *   **캔버스 크기 관리**: 너무 큰 캔버스는 성능에 영향을 줄 수 있습니다. 필요한 만큼만 사용하거나, CSS로 크기를 조절하되 실제 캔버스 해상도는 낮게 유지하는 것을 고려합니다.
    *   **`save()`/`restore()` 사용**: 변환이나 스타일 변경이 잦을 때 `save()`와 `restore()`를 사용하여 상태를 효율적으로 관리합니다.
    *   **오프스크린 캔버스 (Offscreen Canvas)**: 복잡한 계산이나 이미지 처리를 메인 스레드를 블로킹하지 않고 백그라운드에서 수행할 수 있습니다 (웹 워커와 함께 사용).

**코드 사례 (종합 예제: 도형 그리기 및 간단한 애니메이션)**
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Canvas 그래픽 및 애니메이션</title>
    <style>
        body { display: flex; flex-direction: column; align-items: center; font-family: sans-serif; }
        canvas { border: 2px solid #333; background-color: #f0f0f0; margin-top: 20px; }
        .controls { margin-top: 10px; }
        button { padding: 8px 15px; margin: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Canvas 그래픽 및 애니메이션 예제</h1>
    <canvas id="myCanvas" width="600" height="400"></canvas>
    <div class="controls">
        <button id="startAnimation">애니메이션 시작</button>
        <button id="stopAnimation">애니메이션 중지</button>
    </div>

    <script>
        const canvas = document.getElementById('myCanvas');
        const ctx = canvas.getContext('2d');

        let animationFrameId;
        let rotationAngle = 0;
        let circleX = 50;
        let circleDirection = 1; // 1 for right, -1 for left

        // 캔버스 초기화 및 정적 도형 그리기
        function drawStaticShapes() {
            // 배경 지우기
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 사각형 그리기
            ctx.fillStyle = 'rgba(255, 99, 71, 0.8)'; // Tomato color with transparency
            ctx.fillRect(50, 50, 100, 75);

            // 선 그리기
            ctx.strokeStyle = 'blue';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(180, 50);
            ctx.lineTo(280, 125);
            ctx.stroke();

            // 원 그리기
            ctx.strokeStyle = 'green';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(350, 80, 40, 0, Math.PI * 2); // x, y, radius, startAngle, endAngle
            ctx.stroke();

            // 텍스트 그리기
            ctx.fillStyle = '#333';
            ctx.font = '24px Arial';
            ctx.fillText('Hello Canvas!', 400, 80);

            // 그라디언트 사각형
            const linearGradient = ctx.createLinearGradient(0, 0, 200, 0);
            linearGradient.addColorStop(0, 'purple');
            linearGradient.addColorStop(1, 'orange');
            ctx.fillStyle = linearGradient;
            ctx.fillRect(50, 150, 200, 50);
        }

        // 애니메이션 프레임 그리기
        function animate() {
            // 이전 프레임 지우기 (전체 캔버스 지우기)
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 정적 도형 다시 그리기
            drawStaticShapes();

            // 움직이는 원 그리기
            ctx.fillStyle = 'red';
            ctx.beginPath();
            ctx.arc(circleX, 250, 30, 0, Math.PI * 2);
            ctx.fill();

            // 원의 위치 업데이트
            circleX += circleDirection * 2; // 2픽셀씩 이동
            if (circleX + 30 > canvas.width || circleX - 30 < 0) {
                circleDirection *= -1; // 방향 반전
            }

            // 회전하는 사각형 그리기
            ctx.save(); // 현재 상태 저장
            ctx.translate(canvas.width / 2, canvas.height / 2); // 원점 이동
            ctx.rotate(rotationAngle); // 회전
            ctx.fillStyle = 'rgba(0, 123, 255, 0.7)'; // Blue with transparency
            ctx.fillRect(-75, -75, 150, 150); // 새로운 원점을 기준으로 사각형 그리기
            ctx.restore(); // 이전 상태 복원

            rotationAngle += 0.02; // 회전 각도 증가

            // 다음 프레임 요청
            animationFrameId = requestAnimationFrame(animate);
        }

        // 초기 그리기
        drawStaticShapes();

        // 이벤트 리스너
        document.getElementById('startAnimation').addEventListener('click', () => {
            if (!animationFrameId) { // 애니메이션이 이미 실행 중이 아니면 시작
                animate();
            }
        });

        document.getElementById('stopAnimation').addEventListener('click', () => {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null; // ID 초기화
            drawStaticShapes(); // 애니메이션 중지 후 정적 상태로 복원
        });
    </script>
</body>
</html>
```

### 3.3. `<audio>` 및 `<video>`: 미디어 삽입

1.  **개요 및 목적 (Overview and Purpose)**
    *   HTML5의 `<audio>` 및 `<video>` 태그는 웹 페이지에 오디오 및 비디오 콘텐츠를 직접 삽입하고 재생할 수 있는 표준적인 방법을 제공합니다.
    *   이전에는 플러그인(예: Adobe Flash)이 필요했지만, HTML5부터는 브라우저 자체에서 미디어를 지원하여 접근성, 성능, 보안이 크게 향상되었습니다.

2.  **주요 속성 (Key Attributes)**
    *   **`src`**: 미디어 파일의 URL을 지정합니다. `<source>` 태그를 사용할 경우 생략할 수 있습니다.
    *   **`controls`**: 브라우저의 기본 재생 컨트롤러(재생/일시정지, 볼륨, 진행 바, 전체 화면 등)를 표시합니다. 사용자에게 미디어 제어 기능을 제공하는 데 필수적입니다.
    *   **`autoplay`**: 페이지 로드 시 미디어를 자동으로 재생합니다.
        *   **주의**: 대부분의 최신 브라우저는 사용자 경험을 위해 `autoplay`를 제한합니다. 일반적으로 `muted` 속성과 함께 사용하거나, 사용자의 상호작용(클릭 등) 후에만 허용됩니다.
    *   **`loop`**: 미디어 재생이 끝나면 자동으로 처음부터 다시 재생합니다.
    *   **`muted`**: 미디어의 오디오를 기본적으로 음소거합니다. `autoplay`와 함께 사용될 때 유용합니다.
    *   **`preload`**: 브라우저가 미디어 파일을 어떻게 로드할지 힌트를 제공합니다.
        *   `none`: 미디어를 미리 로드하지 않습니다. (대역폭 절약)
        *   `metadata`: 미디어의 메타데이터(길이, 크기 등)만 로드합니다.
        *   `auto`: 전체 미디어 파일을 미리 로드할 수 있습니다. (기본값, 사용자 경험 향상)
    *   **`poster` (비디오 전용)**: 비디오가 재생되기 전이나 로드되는 동안 표시될 이미지의 URL을 지정합니다. 비디오의 미리보기 역할을 합니다.
    *   **`width`, `height` (비디오 전용)**: 비디오 플레이어의 너비와 높이를 픽셀 단위로 지정합니다. 레이아웃 안정성(CLS 방지)을 위해 권장됩니다.

3.  **`<source>` 태그: 다양한 미디어 형식 지원 (Supporting Multiple Media Formats)**
    *   모든 브라우저가 동일한 미디어 형식을 지원하는 것은 아니므로, `<source>` 태그를 사용하여 여러 미디어 파일 경로를 제공하는 것이 중요합니다.
    *   브라우저는 나열된 `<source>` 태그 중 자신이 지원하는 첫 번째 형식을 선택하여 재생합니다.
    *   **`type` 속성**: 미디어 파일의 MIME 타입을 지정하여 브라우저가 파일을 다운로드하기 전에 지원 여부를 확인할 수 있게 합니다.
    *   **일반적인 미디어 형식**: 
        *   **비디오**: `video/mp4` (H.264), `video/webm` (VP8/VP9), `video/ogg` (Theora)
        *   **오디오**: `audio/mpeg` (MP3), `audio/ogg` (Vorbis), `audio/wav`

4.  **접근성 및 대체 콘텐츠 (Accessibility and Fallback Content)**
    *   **`<track>` 태그**: 미디어 콘텐츠에 대한 텍스트 트랙(자막, 캡션, 설명 등)을 제공합니다.
        *   `kind`: 트랙의 종류를 지정합니다 (`subtitles`, `captions`, `descriptions`, `chapters`, `metadata`).
        *   `src`: 트랙 파일(WebVTT 형식)의 URL을 지정합니다.
        *   `srclang`: 트랙의 언어를 지정합니다.
        *   `label`: 트랙을 식별하는 사용자에게 표시될 텍스트 레이블입니다.
        *   `default`: 기본적으로 활성화될 트랙을 지정합니다.
    *   **대체 콘텐츠**: `<audio>` 또는 `<video>` 태그 내부에 브라우저가 해당 태그를 지원하지 않을 때 표시될 텍스트나 링크를 제공합니다.

5.  **JavaScript API를 통한 제어 (Control via JavaScript API)**
    *   JavaScript를 사용하여 미디어 재생을 프로그래밍 방식으로 제어할 수 있습니다.
    *   **메서드**: `play()`, `pause()`, `load()`.
    *   **속성**: `volume`, `currentTime`, `duration`, `paused`, `ended`, `muted`.
    *   **이벤트**: `play`, `pause`, `ended`, `timeupdate`, `volumechange`, `error`.

6.  **성능 최적화 팁 (Performance Optimization Tips)**
    *   **적절한 미디어 형식 및 압축**: 웹에 최적화된 형식(예: WebM, MP4)을 사용하고, 파일 크기를 최소화하기 위해 적절히 압축합니다.
    *   **`preload` 속성 활용**: 필요에 따라 `none` 또는 `metadata`를 사용하여 불필요한 대역폭 사용을 줄입니다.
    *   **CDN 사용**: 미디어 파일을 CDN(콘텐츠 전송 네트워크)을 통해 제공하여 사용자에게 더 빠르게 전달합니다.
    *   **반응형 미디어**: CSS를 사용하여 다양한 화면 크기에 맞게 미디어 플레이어의 크기를 조절합니다 (예: `max-width: 100%; height: auto;`).

**코드 사례 (종합 예제: 비디오 및 오디오 삽입)**
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>미디어 삽입 예제</title>
    <style>
        body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; }
        video, audio { margin: 20px 0; border: 1px solid #ccc; }
        .media-controls button { padding: 8px 15px; margin: 5px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>HTML5 미디어 삽입</h1>

    <h2>비디오 예제</h2>
    <video width="640" height="360" controls poster="https://via.placeholder.com/640x360?text=Video+Poster" preload="metadata">
        <source src="https://www.w3schools.com/html/mov_bbb.mp4" type="video/mp4">
        <source src="https://www.w3schools.com/html/mov_bbb.ogg" type="video/ogg">
        <track kind="subtitles" src="subtitles_ko.vtt" srclang="ko" label="한국어 자막" default>
        <track kind="captions" src="captions_en.vtt" srclang="en" label="English Captions">
        <p>브라우저가 비디오 태그를 지원하지 않습니다. <a href="https://www.w3schools.com/html/mov_bbb.mp4">비디오 다운로드</a></p>
    </video>
    <div class="media-controls">
        <button id="playVideo">재생</button>
        <button id="pauseVideo">일시정지</button>
        <button id="muteVideo">음소거/해제</button>
    </div>

    <h2>오디오 예제</h2>
    <audio controls loop preload="auto">
        <source src="https://www.w3schools.com/html/horse.mp3" type="audio/mpeg">
        <source src="https://www.w3schools.com/html/horse.ogg" type="audio/ogg">
        <p>브라우저가 오디오 태그를 지원하지 않습니다. <a href="https://www.w3schools.com/html/horse.mp3">오디오 다운로드</a></p>
    </audio>
    <div class="media-controls">
        <button id="playAudio">재생</button>
        <button id="pauseAudio">일시정지</button>
    </div>

    <script>
        const videoElement = document.querySelector('video');
        const audioElement = document.querySelector('audio');

        document.getElementById('playVideo').addEventListener('click', () => videoElement.play());
        document.getElementById('pauseVideo').addEventListener('click', () => videoElement.pause());
        document.getElementById('muteVideo').addEventListener('click', () => videoElement.muted = !videoElement.muted);

        document.getElementById('playAudio').addEventListener('click', () => audioElement.play());
        document.getElementById('pauseAudio').addEventListener('click', () => audioElement.pause());

        // 비디오 재생 종료 시 이벤트
        videoElement.addEventListener('ended', () => {
            console.log('비디오 재생이 종료되었습니다.');
            alert('비디오 재생이 끝났습니다!');
        });

        // 예시: 동적으로 자막 파일 생성 (실제 환경에서는 서버에서 제공)
        // WebVTT 형식의 자막 파일 내용
        const vttContentKo = `WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\n안녕하세요, 비디오 예제입니다.\n\n2\n00:00:04.000 --> 00:00:07.000\n이것은 HTML5 비디오 태그를 사용합니다.`;

        const vttContentEn = `WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nHello, this is a video example.\n\n2\n00:00:04.000 --> 00:00:07.000\nThis uses the HTML5 video tag.`;

        // Blob을 사용하여 가상 VTT 파일 URL 생성
        const createVTTBlob = (content) => {
            const blob = new Blob([content], { type: 'text/vtt' });
            return URL.createObjectURL(blob);
        };

        // 실제 환경에서는 <track> 태그의 src 속성에 직접 VTT 파일 경로를 지정합니다.
        // 여기서는 예시를 위해 동적으로 URL을 생성합니다.
        const trackKo = videoElement.querySelector('track[srclang="ko"]');
        if (trackKo) {
            trackKo.src = createVTTBlob(vttContentKo);
        }
        const trackEn = videoElement.querySelector('track[srclang="en"]');
        if (trackEn) {
            trackEn.src = createVTTBlob(vttContentEn);
        }
    </script>
</body>
</html>
```

### 3.4. `<iframe>`: 외부 콘텐츠 임베드

1.  **개요 및 목적 (Overview and Purpose)**
    *   `<iframe>` (Inline Frame) 태그는 현재 HTML 문서 안에 다른 HTML 문서를 삽입(임베드)할 때 사용됩니다. 이를 통해 외부 웹 페이지, 애플리케이션, 또는 미디어 콘텐츠를 현재 페이지의 일부처럼 표시할 수 있습니다.
    *   주로 유튜브 비디오, 구글 지도, 외부 광고, 결제 위젯, 소셜 미디어 피드 등을 페이지에 포함시키는 데 활용됩니다.

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