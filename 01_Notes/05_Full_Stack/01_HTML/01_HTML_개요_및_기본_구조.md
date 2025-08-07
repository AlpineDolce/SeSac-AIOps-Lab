<h2>HTML 개요 및 기본 구조</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 HTML의 핵심 개념과 주요 태그 사용법을 체계적으로 정리하여, 웹 페이지의 구조를 이해하고 작성하는 능력을 기르는 것을 목표로 합니다. 각 태그의 의미와 사용 사례를 통해 시맨틱 웹의 중요성을 이해하고, 풀스택 개발 실무에 필요한 보안, 성능 최적화 관점까지 통합하여 실용적인 예제를 통해 학습 효과를 높이고자 합니다.</p>

<h2>목차</h2>

- [1. HTML 개요 및 기본 구조](#1-html-개요-및-기본-구조)
  - [1.1. HTML이란?](#11-html이란)
  - [1.2. 문서 기본 구조 요소](#12-문서-기본-구조-요소)
    - [1.2.1. `<!DOCTYPE html>`](#121-doctype-html)
    - [1.2.2. `<html>` 태그](#122-html-태그)
    - [1.2.3. `<head>` 태그: 메타데이터 및 외부 리소스](#123-head-태그-메타데이터-및-외부-리소스)
      - [1.2.3.1. `<title>` 및 `<meta>` 태그의 활용](#1231-title-및-meta-태그의-활용)
      - [1.2.3.2. CSS 통합 및 스타일링](#1232-css-통합-및-스타일링)
    - [1.2.4. `<body>` 태그: 실제 콘텐츠 영역](#124-body-태그-실제-콘텐츠-영역)
  - [1.3. 풀스택 관점: HTML 템플릿 및 동적 생성](#13-풀스택-관점-html-템플릿-및-동적-생성)

---

## 1. HTML 개요 및 기본 구조

### 1.1. HTML이란?
**HTML (Hyper Text Markup Language)** 은 웹 페이지와 그 내용을 구조화하기 위한 표준 마크업 언어입니다.

모든 HTML 문서는 특정한 기본 구조를 따릅니다. 이 구조는 웹 브라우저가 문서를 올바르게 해석하고 렌더링하는 데 필수적입니다. `hello.html` 파일은 가장 기본적인 형태를 보여줍니다.

**코드 사례 (`hello.html`)**:
```html
<!DOCTYPE html>
<html>
    <head>
        <title>Hello web</title>
    </head>
    <body>
        이 부분이 화면에 보여질 부분이다 
    </body>
</html>
```

### 1.2. 문서 기본 구조 요소

#### 1.2.1. `<!DOCTYPE html>`
문서 형식 선언(DTD)입니다. 이 선언은 웹 브라우저에게 현재 문서가 **HTML5 표준**에 따라 작성되었음을 알립니다. 항상 HTML 문서의 가장 첫 줄에 위치해야 합니다.

#### 1.2.2. `<html>` 태그
전체 HTML 문서를 감싸는 **루트(root) 요소**입니다. `lang` 속성을 사용하여 문서의 주 언어를 지정할 수 있으며, 이는 검색 엔진 최적화(SEO)와 접근성에 도움을 줍니다。
    ```html
    <html lang="ko">
    ```

#### 1.2.3. `<head>` 태그: 메타데이터 및 외부 리소스
문서의 **메타데이터(metadata)**를 담는 컨테이너입니다. 메타데이터는 브라우저와 검색 엔진에게 문서에 대한 정보를 제공하지만, 페이지 본문에는 직접 표시되지 않습니다.

##### 1.2.3.1. `<title>` 및 `<meta>` 태그의 활용
- `<title>` 태그: 브라우저 탭이나 창의 제목 표시줄에 표시될 문서의 제목을 정의합니다. 검색 결과에서도 중요한 역할을 합니다.
- `<meta>` 태그: 문자 인코딩, 뷰포트 설정, 페이지 설명, 키워드 등 다양한 메타 정보를 정의합니다。
    - `charset="UTF-8"`: 문서의 문자 인코딩을 UTF-8로 설정하여 한글이나 다른 특수 문자가 깨지지 않도록 합니다。
    - `name="viewport"`: 모바일 기기에서 페이지가 어떻게 보일지 제어합니다. `width=device-width, initial-scale=1.0`은 페이지의 너비를 기기 화면 너비에 맞추고 초기 확대/축소 수준을 1로 설정합니다.
    - **SEO 관련 메타 태그**:
        - `name="description"`: 검색 엔진 결과 페이지(SERP)에 표시될 페이지의 간략한 설명을 제공합니다.
        - `name="keywords"`: 페이지의 콘텐츠와 관련된 키워드를 나열합니다. (최근에는 SEO 중요도가 낮아짐)
    - **Open Graph 프로토콜 (SNS 공유 최적화)**: 페이스북, 카카오톡 등 소셜 미디어에서 링크를 공유할 때 미리보기 정보를 제어합니다.
        - `property="og:title"`: 공유될 때 표시될 제목.
        - `property="og:description"`: 공유될 때 표시될 설명.
        - `property="og:image"`: 공유될 때 표시될 이미지 URL.
        - `property="og:url"`: 공유될 페이지의 정식 URL.
    - `name="author"`: 문서의 작성자를 명시합니다.
    - `http-equiv="refresh"`: 특정 시간 후 페이지를 새로고침하거나 다른 URL로 리다이렉트합니다. (사용에 주의 필요)
- **`<link rel="canonical">`**: 검색 엔진 최적화(SEO)를 위해 중요한 태그입니다. 동일하거나 매우 유사한 콘텐츠를 가진 여러 URL이 있을 때, 검색 엔진에 어떤 URL이 원본(정식) 페이지인지 알려주어 중복 콘텐츠 문제를 방지하고 검색 순위 분산을 막습니다. 풀스택 애플리케이션에서 동적 URL이나 여러 경로로 접근 가능한 페이지가 있을 때 특히 중요합니다.
- **`<base>` 태그**: 문서 내 모든 상대 URL(링크, 이미지, 스크립트 등)의 기준 URL을 지정합니다. `<head>` 내에 한 번만 사용될 수 있습니다. SPA(Single Page Application)나 복잡한 라우팅을 사용하는 풀스택 애플리케이션에서 경로 관리를 단순화하는 데 유용할 수 있습니다.
- **`<noscript>` 태그**: JavaScript가 비활성화되었거나 지원되지 않는 브라우저를 위한 대체 콘텐츠를 제공할 때 사용합니다. 풀스택 개발에서 클라이언트 사이드 로직에 크게 의존하는 경우, 사용자 경험을 고려하여 폴백(fallback)을 제공하는 데 중요합니다.

    **코드 사례 (확장된 `<head>`)**:
    ```html
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="description" content="HTML 기본 개념 정리 문서입니다.">
        <meta name="keywords" content="HTML, 웹 개발, 프론트엔드">
        <meta name="author" content="Your Name">
        <meta property="og:title" content="HTML 기본 개념 정리">
        <meta property="og:description" content="HTML의 핵심 개념과 주요 태그 사용법을 체계적으로 정리합니다.">
        <meta property="og:image" content="https://example.com/thumbnail.jpg">
        <meta property="og:url" content="https://example.com/html-guide.html">
        <link rel="canonical" href="https://example.com/html-guide.html">
        <base href="https://example.com/base/">
        <title>HTML 기본 개념 정리</title>
        
        <!-- 성능 최적화 관련 <link> 태그 -->
        <link rel="preload" href="/fonts/myfont.woff2" as="font" type="font/woff2" crossorigin>
        <link rel="preconnect" href="https://api.example.com">
        <link rel="dns-prefetch" href="https://static.example.com">
    </head>
    ```
    - **성능 최적화 관련 `<link>` 태그 상세 설명**:
        - **`rel="preload"`**: 현재 페이지에서 곧 사용될 리소스(예: 폰트, 스크립트, 이미지)를 미리 로드하도록 브라우저에 지시합니다. 렌더링 경로의 후반에 발견될 수 있는 중요한 리소스를 조기에 로드하여 렌더링 차단을 줄이고 페이지 로드 속도를 개선합니다.
            - **사용 사례**: CSS 파일 깊숙이 정의된 웹 폰트나, 페이지 하단에서 로드되는 중요한 스크립트를 미리 로드할 때 유용합니다.
        - **`rel="preconnect"`**: 브라우저가 특정 도메인에 대한 연결(DNS 조회, TCP 핸드셰이크, TLS 협상)을 미리 설정하도록 지시합니다. 실제 리소스 요청이 발생했을 때 연결 설정 시간을 절약하여 리소스를 더 빠르게 가져올 수 있습니다.
            - **사용 사례**: Google Fonts나 API 서버와 같이 중요한 리소스를 제공하는 외부 도메인에 미리 연결할 때 효과적입니다.
        - **`rel="dns-prefetch"`**: `preconnect`보다 가벼운 최적화로, 특정 도메인에 대한 DNS 조회만 미리 수행하도록 지시합니다.
            - **사용 사례**: 페이지에서 사용될 가능성이 있는 여러 외부 도메인(예: 소셜 미디어 위젯, 분석 도구)에 대한 DNS 조회 시간을 절약할 때 사용됩니다.
        
##### 1.2.3.2. CSS 통합 및 스타일링
```html
    <!-- 외부 스타일시트 연결: 가장 일반적인 방법으로, 별도의 .css 파일에 스타일을 정의하고 연결합니다. -->
    <link rel="stylesheet" href="styles.css">
    
    <!-- 내부 스타일시트: HTML 문서 내 <style> 태그 안에 직접 스타일을 정의합니다. -->
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #0056b3;
        }
    </style>
```

#### 1.2.4. `<body>` 태그: 실제 콘텐츠 영역
웹 페이지에 **실제로 표시되는 모든 콘텐츠**를 담는 영역입니다. 텍스트, 제목, 문단, 이미지, 링크, 테이블, 리스트 등 사용자가 보는 모든 요소가 이 태그 안에 위치합니다. JavaScript를 통해 페이지가 로드되거나 언로드될 때 특정 함수를 실행하는 `onload`, `onunload` 같은 이벤트 속성을 가질 수 있습니다.

- **`DOMContentLoaded` 이벤트**: `<body>` 태그의 `onload` 이벤트는 페이지의 모든 리소스(이미지, CSS, JavaScript 등)가 완전히 로드된 후에 발생합니다. 반면, `DOMContentLoaded` 이벤트는 HTML 문서가 완전히 로드되고 파싱되었을 때 발생하며, 외부 리소스의 로드를 기다리지 않습니다. 따라서 JavaScript로 DOM을 조작해야 할 경우 `DOMContentLoaded`를 사용하는 것이 더 효율적이고 권장됩니다.

```html
<body onload="initializePage()" onunload="cleanup()">
    <!-- 페이지의 모든 콘텐츠 -->
</body>
```

### 1.3. 풀스택 관점: HTML 템플릿 및 동적 생성
풀스택 개발에서 HTML은 단순히 정적인 파일로 제공되는 것을 넘어, 서버 측 또는 클라이언트 측에서 **동적으로 생성되거나 데이터와 결합되어 렌더링**되는 경우가 대부분입니다. 이는 사용자별 맞춤형 콘텐츠 제공, 데이터베이스 연동, 복잡한 UI 구현 등을 가능하게 합니다.

- **서버 사이드 렌더링 (SSR) / 템플릿 엔진**: 백엔드 프레임워크(예: Python의 Django/Flask, Node.js의 Express, Java의 Spring)는 **템플릿 엔진**을 사용하여 서버에서 데이터를 HTML 템플릿에 주입하고, 완성된 HTML 페이지를 클라이언트(브라우저)로 전송합니다. 사용자는 이미 완성된 페이지를 받으므로 초기 로딩 속도가 빠르고 SEO에 유리합니다.
    - **예시 템플릿 엔진**: Jinja2 (Python), EJS (Node.js), Thymeleaf (Java), Handlebars.js 등
    - **동작 방식**: 서버에서 데이터(예: 사용자 이름, 게시글 목록)를 가져와 HTML 파일 내의 특정 플레이스홀더(placeholder)를 해당 데이터로 채워 넣습니다.

    **코드 사례 (가상의 서버 사이드 템플릿):**
    ```html
    <!-- 예를 들어, Jinja2 템플릿 엔진 사용 시 -->
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <title>{{ page_title }}</title>
    </head>
    <body>
        <h1>환영합니다, {{ user_name }}님!</h1>
        <ul>
            {% for item in items %}
            <li>{{ item.name }} - {{ item.price }}원</li>
            {% endfor %}
        </ul>
    </body>
    </html>
    ```

- **클라이언트 사이드 렌더링 (CSR) / JavaScript 프레임워크**: 브라우저가 서버로부터 빈 HTML 파일과 JavaScript 코드를 받은 후, JavaScript가 데이터를 비동기적으로 가져와(AJAX/Fetch API) 브라우저에서 직접 HTML 요소를 생성하고 조작하여 페이지를 렌더링합니다. 초기 로딩은 느릴 수 있지만, 이후 페이지 전환이 빠르고 사용자 경험이 부드럽습니다.
    - **예시 프레임워크/라이브러리**: React, Vue.js, Angular
    - **동작 방식**: JavaScript 코드가 DOM을 직접 조작하여 HTML을 생성하거나 업데이트합니다. (예: `document.createElement`, `element.appendChild`)

    **코드 사례 (가상의 클라이언트 사이드 JavaScript):**
    ```javascript
    // JavaScript를 사용하여 동적으로 HTML 요소 생성
    const appDiv = document.getElementById('app');
    const data = { title: "동적 페이지", content: "이 내용은 JavaScript로 추가되었습니다." };

    const h1 = document.createElement('h1');
    h1.textContent = data.title;
    appDiv.appendChild(h1);

    const p = document.createElement('p');
    p.textContent = data.content;
    appDiv.appendChild(p);
    ```

이처럼 풀스택 개발에서 HTML은 정적인 문서로서의 역할뿐만 아니라, 동적으로 변화하는 웹 애플리케이션의 '뼈대'로서 중요한 역할을 수행합니다.