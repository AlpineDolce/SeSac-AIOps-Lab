<h2>DOM 조작 및 브라우저 API</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 JavaScript의 핵심 문법과 주요 개념을 체계적으로 정리하여, 동적인 웹 페이지를 개발하는 능력을 기르는 것을 목표로 합니다. 변수 선언, 함수, 객체, 비동기 처리 등 필수 개념들을 상세한 코드 예제와 함께 학습하여 실무 활용도를 높이고자 합니다.</p>

<h2>목차</h2>

- [4. DOM 조작 및 브라우저 API](#4-dom-조작-및-브라우저-api)
  - [4.1. DOM (Document Object Model) 조작](#41-dom-document-object-model-조작)
  - [4.2. 이벤트 처리 및 위임](#42-이벤트-처리-및-위임)
  - [4.3. 브라우저 저장소 (Web Storage)](#43-브라우저-저장소-web-storage)
  - [4.4. History API](#44-history-api)
  - [4.5. Intersection Observer / Mutation Observer](#45-intersection-observer--mutation-observer)
  - [4.6. 타이머 함수 (Timer Functions)](#46-타이머-함수-timer-functions)
  - [4.7. Web Components (Custom Elements, Shadow DOM, HTML Templates)](#47-web-components-custom-elements-shadow-dom-html-templates)

---

## 4. DOM 조작 및 브라우저 API

### 4.1. DOM (Document Object Model) 조작

DOM은 웹 페이지(HTML 문서)를 계층적인 트리 구조의 객체로 표현한 것입니다. JavaScript는 DOM API를 통해 이 구조에 접근하고, 문서의 내용, 구조, 스타일을 동적으로 변경할 수 있습니다.

- **요소 선택 (Selecting Elements)**: 특정 HTML 요소를 선택하여 제어할 수 있습니다.
    - `document.getElementById('id')`: 주어진 `id`를 가진 요소를 선택합니다.
    - `document.querySelector('selector')`: 주어진 CSS 선택자와 일치하는 **첫 번째 요소**를 선택합니다. (가장 많이 사용됨)
    - `document.querySelectorAll('selector')`: 주어진 CSS 선택자와 일치하는 **모든 요소**를 `NodeList`(유사 배열)로 반환합니다.

- **내용 및 속성 변경 (Manipulating Content & Attributes)**
    - `element.textContent`: 요소 안의 텍스트 내용만 가져오거나 설정합니다.
    - `element.innerHTML`: 요소 안의 HTML 전체를 가져오거나 설정합니다. (보안에 유의)
    - `element.setAttribute('name', 'value')`: 요소의 HTML 속성을 설정합니다.

- **스타일 변경 (Manipulating Styles)**
    - `element.style.property`: 요소의 인라인 스타일을 변경합니다. (CSS 속성은 카멜 케이스로 변환. 예: `background-color` -> `backgroundColor`)
    - `element.classList.add('class')`, `.remove('class')`, `.toggle('class')`

- **요소 생성 및 추가/삭제 (Creating & Adding/Removing Elements)**
    - `document.createElement('tag')`: 새로운 HTML 요소를 생성합니다.
    - `parentNode.appendChild(childNode)`: `childNode`를 `parentNode`의 마지막 자식으로 추가합니다.
    - `element.remove()`: 해당 요소를 DOM 트리에서 제거합니다.

- **이벤트 처리 (Event Handling)**: 사용자의 행동(클릭, 키보드 입력 등)에 반응하여 특정 코드를 실행합니다.
    - `element.addEventListener('event-type', function)`: 요소에 이벤트 리스너(핸들러)를 등록하는 가장 권장되는 방식입니다.

    **코드 사례:**
    ```html
    <button id="myButton">클릭하세요</button>
    <ul id="myList"></ul>
    <script>
        const btn = document.getElementById('myButton');
        const list = document.getElementById('myList');
        let count = 1;

        btn.addEventListener('click', function() {
            const newLi = document.createElement('li');
            newLi.textContent = `새 항목 ${count++}`;
            list.appendChild(newLi);
        });
    </script>
    ```

- **DOM 조작과 브라우저 렌더링 성능**
    - **렌더링 과정**: 브라우저는 HTML을 파싱하여 DOM 트리를, CSS를 파싱하여 CSSOM 트리를 만듭니다. 이 두 트리를 결합하여 렌더 트리를 생성한 후, 각 노드의 위치와 크기를 계산하는 **레이아웃(Layout/Reflow)** 과정을 거치고, 최종적으로 화면에 픽셀을 그리는 **페인트(Paint/Repaint)** 과정을 통해 페이지를 표시합니다.
    - **성능 저하**: JavaScript로 DOM을 조작하여 요소의 크기나 위치가 변경되면, 브라우저는 변경된 스타일을 다시 계산하기 위해 **리플로우(Reflow)**와 **리페인트(Repaint)**를 다시 실행해야 합니다. 이러한 과정은 비용이 많이 들며, 특히 빈번하게 발생할 경우 애니메이션이 끊기거나 UI 반응이 느려지는 등 성능 저하의 주요 원인이 됩니다.
    - **성능 최적화**:
        - **DOM 접근 최소화**: DOM에 접근하는 횟수를 줄이고, 변경할 내용을 변수에 저장했다가 한 번에 적용합니다.
        - **`DocumentFragment` 사용**: 여러 요소를 DOM에 추가할 때, `DocumentFragment`라는 메모리상의 컨테이너에 요소를 모두 추가한 뒤, 최종적으로 DOM에 한 번만 추가하여 리플로우를 최소화할 수 있습니다.
        - **가상 DOM (Virtual DOM)의 등장 배경**: React, Vue와 같은 현대 프론트엔드 라이브러리는 이러한 문제를 해결하기 위해 가상 DOM을 사용합니다. 가상 DOM은 실제 DOM의 복사본을 메모리에 저장하고, 데이터가 변경되면 실제 DOM을 직접 조작하는 대신 메모리상의 가상 DOM을 먼저 변경합니다. 그 후, 이전 가상 DOM과 변경된 가상 DOM을 비교하여 변경된 부분만 실제 DOM에 한 번에 적용(Batch Update)함으로써 DOM 조작을 최소화하고 성능을 최적화합니다.

### 4.2. 이벤트 처리 및 위임
- **이벤트 위임 (Event Delegation)**: 여러 자식 요소에 개별적으로 이벤트 리스너를 등록하는 대신, 공통의 부모 요소에 하나의 이벤트 리스너를 등록하여 자식 요소에서 발생한 이벤트를 처리하는 기법입니다. 동적으로 추가되는 요소에 대한 이벤트 처리나 성능 최적화에 매우 유용합니다.
    - **장점**: 메모리 사용량을 줄이고, 동적으로 추가되는 요소에 자동으로 이벤트가 적용됩니다.
    - **원리**: 이벤트 버블링(Event Bubbling)을 활용하여 자식 요소에서 발생한 이벤트가 부모 요소로 전파되는 것을 이용합니다. 부모 요소에서 `event.target`을 통해 실제 이벤트가 발생한 자식 요소를 식별합니다.

    **코드 사례:**
    ```html
    <ul id="parentList">
        <li>항목 1</li>
        <li>항목 2</li>
        <li>항목 3</li>
    </ul>
    <button id="addItem">항목 추가</button>

    <script>
        const parentList = document.getElementById('parentList');
        const addItemBtn = document.getElementById('addItem');
        let itemNum = 4;

        // 부모 요소에 이벤트 리스너 등록
        parentList.addEventListener('click', function(event) {
            // 클릭된 요소가 <li> 태그인지 확인
            if (event.target.tagName === 'LI') {
                console.log('클릭된 항목:', event.target.textContent);
                event.target.style.color = 'red';
            }
        });

        addItemBtn.addEventListener('click', function() {
            const newLi = document.createElement('li');
            newLi.textContent = `새 항목 ${itemNum++}`;
            parentList.appendChild(newLi);
        });
    </script>
    ```

### 4.3. 브라우저 저장소 (Web Storage)

웹 브라우저는 페이지를 새로고침하거나 브라우저를 닫았다 열어도 데이터를 유지할 수 있는 간단한 저장소 메커니즘을 제공합니다. `key-value` 형태로 데이터를 저장합니다.

- **`localStorage`**: 데이터를 **영구적으로** 저장합니다. 사용자가 직접 삭제하지 않는 한, 브라우저를 닫았다가 다시 열어도 데이터가 유지됩니다. 동일한 도메인의 다른 탭이나 창 간에 데이터가 공유됩니다.

- **`sessionStorage`**: **하나의 세션** 동안만 데이터를 저장합니다. 즉, 브라우저 탭이나 창을 닫으면 데이터가 사라집니다. 같은 탭 내에서 페이지를 새로고침해도 데이터는 유지되지만, 다른 탭과는 공유되지 않습니다.

- **주요 메서드**:
    - `setItem(key, value)`: 데이터를 저장합니다. (값은 반드시 문자열이어야 함)
    - `getItem(key)`: 데이터를 불러옵니다.
    - `removeItem(key)`: 특정 키의 데이터를 삭제합니다.
    - `clear()`: 모든 데이터를 삭제합니다.

    **코드 사례:**
    ```javascript
    // 객체를 저장하려면 JSON 문자열로 변환해야 합니다.
    const user = { name: "홍길동", theme: "dark" };

    // localStorage에 사용자 설정 저장
    localStorage.setItem('userSettings', JSON.stringify(user));

    // localStorage에서 데이터 불러오기 및 객체로 파싱
    const savedSettings = localStorage.getItem('userSettings');
    if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        console.log(parsedSettings.name); // "홍길동"
    }

    // sessionStorage에 임시 데이터 저장
    sessionStorage.setItem('tempData', '12345');

    // 데이터 삭제
    // localStorage.removeItem('userSettings');
    // localStorage.clear();
    ```

### 4.4. History API
History API는 브라우저의 세션 기록(Session History)을 조작할 수 있는 기능을 제공합니다. 이를 통해 페이지를 새로고침하지 않고도 URL을 변경하고, 뒤로가기/앞으로가기 버튼 동작을 제어하여 SPA(Single Page Application)에서 라우팅을 구현할 때 유용하게 사용됩니다.

- **`history.pushState(state, title, url)`**: 현재 세션 기록에 새로운 상태를 추가합니다. URL을 변경하지만 페이지를 새로고침하지 않습니다.
    - `state`: 새로운 기록 항목과 연결될 상태 객체. `popstate` 이벤트 발생 시 전달됩니다.
    - `title`: 브라우저의 기록 스택에 표시될 제목. 대부분의 브라우저에서 현재 무시됩니다.
    - `url`: 새로운 기록 항목의 URL. 선택 사항이며, 현재 URL과 동일한 출처여야 합니다.
- **`history.replaceState(state, title, url)`**: 현재 세션 기록 항목의 상태를 수정합니다. `pushState`와 달리 새로운 기록 항목을 추가하지 않습니다.
- **`window.onpopstate`**: 사용자가 브라우저의 뒤로가기/앞으로가기 버튼을 클릭하여 세션 기록 항목이 변경될 때 발생하는 이벤트입니다. 이 이벤트를 통해 변경된 상태를 감지하고 UI를 업데이트할 수 있습니다.

**코드 사례:**
```javascript
// 초기 상태 설정
history.replaceState({ page: 'home' }, 'Home', '/');

// 페이지 이동 시 URL 변경 및 상태 저장
document.getElementById('goToAbout').addEventListener('click', function() {
  history.pushState({ page: 'about' }, 'About Us', '/about');
  renderPage('about'); // UI 업데이트 함수 호출
});

// 뒤로가기/앞으로가기 버튼 처리
window.addEventListener('popstate', function(event) {
  if (event.state) {
    console.log('현재 상태:', event.state.page);
    renderPage(event.state.page); // UI 업데이트
  }
});

function renderPage(pageName) {
  console.log(`${pageName} 페이지를 렌더링합니다.`);
  // 실제 페이지 콘텐츠를 변경하는 로직
}
```

### 4.5. Intersection Observer / Mutation Observer
이 두 API는 웹 페이지의 성능 최적화 및 동적인 콘텐츠 관리에 매우 유용합니다.

- **Intersection Observer**: 타겟 요소가 뷰포트(viewport)나 다른 요소(루트 요소)와 교차(intersection)하는지 비동기적으로 관찰하는 방법을 제공합니다. 주로 이미지/비디오 지연 로딩(Lazy Loading), 무한 스크롤 구현, 광고 가시성 추적 등에 사용됩니다.

    **코드 사례 (Intersection Observer - 지연 로딩):**
    ```html
    <img data-src="image-to-load.jpg" alt="지연 로딩 이미지" class="lazy-image">

    <script>
    const lazyImages = document.querySelectorAll('.lazy-image');

    const observer = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) { // 요소가 뷰포트에 들어왔을 때
          const img = entry.target;
          img.src = img.dataset.src; // data-src의 이미지를 로드
          img.classList.remove('lazy-image');
          observer.unobserve(img); // 관찰 중단
        }
      });
    });

    lazyImages.forEach(image => {
      observer.observe(image); // 각 이미지를 관찰
    });
    </script>
    ```

- **Mutation Observer**: DOM 트리의 변경(노드 추가/제거, 속성 변경, 텍스트 내용 변경 등)을 비동기적으로 감지합니다. 주로 동적으로 로드되는 콘텐츠의 변경을 감지하거나, 서드파티 스크립트에 의해 DOM이 변경될 때 특정 작업을 수행하는 데 사용됩니다.

    **코드 사례 (Mutation Observer - DOM 변경 감지):**
    ```javascript
    const targetNode = document.getElementById('some-div');

    const config = { attributes: true, childList: true, subtree: true, characterData: true };

    const callback = function(mutationsList, observer) {
        for(const mutation of mutationsList) {
            if (mutation.type === 'childList') {
                console.log('자식 노드가 추가되거나 제거되었습니다.');
            } else if (mutation.type === 'attributes') {
                console.log(`${mutation.attributeName} 속성이 변경되었습니다.`);
            } else if (mutation.type === 'characterData') {
                console.log('텍스트 내용이 변경되었습니다.');
            }
        }
    };

    const observer = new MutationObserver(callback);
    observer.observe(targetNode, config);

    // 예시: DOM 변경 유발
    // targetNode.appendChild(document.createElement('p'));
    // targetNode.setAttribute('data-new', 'value');
    // targetNode.textContent = '새로운 텍스트';
    ```

### 4.6. 타이머 함수 (Timer Functions)
JavaScript는 특정 코드를 지정된 시간 후에 실행하거나, 일정한 시간 간격으로 반복 실행할 수 있는 내장 타이머 함수를 제공합니다. 비동기 작업의 한 형태로, 사용자 경험을 향상시키거나 특정 로직을 지연 실행할 때 유용합니다.

- **`setTimeout(function, delay)`**: `delay` 밀리초(ms) 후에 `function`을 **한 번만** 실행합니다.
    - 반환값: 타이머의 고유 ID. `clearTimeout()`으로 취소할 수 있습니다.

    **코드 사례:**
    ```javascript
    console.log("시작");
    setTimeout(() => {
        console.log("2초 후에 실행됩니다.");
    }, 2000); // 2000ms = 2초
    console.log("타이머 설정 완료");

    // 타이머 취소
    const timerId = setTimeout(() => {
        console.log("이 메시지는 출력되지 않습니다.");
    }, 3000);
    clearTimeout(timerId);
    ```

- **`setInterval(function, delay)`**: `delay` 밀리초(ms)마다 `function`을 **반복적으로** 실행합니다.
    - 반환값: 인터벌의 고유 ID. `clearInterval()`으로 취소할 수 있습니다.

    **코드 사례:**
    ```javascript
    let count = 0;
    const intervalId = setInterval(() => {
        console.log(`인터벌 실행: ${++count}회`);
        if (count === 3) {
            clearInterval(intervalId); // 3회 실행 후 인터벌 중지
            console.log("인터벌 중지");
        }
    }, 1000); // 1초마다 실행
    ```

- **`requestAnimationFrame(callback)`**: 브라우저에게 다음 리페인트(repaint)가 발생하기 전에 애니메이션을 업데이트할 함수를 호출하도록 요청합니다. 주로 부드러운 애니메이션을 구현할 때 사용되며, 브라우저의 렌더링 주기에 맞춰 실행되므로 `setTimeout`이나 `setInterval`보다 효율적입니다.
    - 반환값: 요청 ID. `cancelAnimationFrame()`으로 취소할 수 있습니다.

    **코드 사례 (간단한 애니메이션):
    ```html
    <div id="box" style="width:50px; height:50px; background-color:blue; position:relative; left:0px;"></div>
    <script>
        const box = document.getElementById('box');
        let position = 0;
        let direction = 1; // 1: 오른쪽, -1: 왼쪽

        function animate() {
            position += direction * 2; // 2px씩 이동
            box.style.left = position + 'px';

            if (position > 200 || position < 0) {
                direction *= -1; // 방향 전환
            }
            requestAnimationFrame(animate);
        }
        requestAnimationFrame(animate);
    </script>
    ```

    - 반환값: 요청 ID. `cancelAnimationFrame()`으로 취소할 수 있습니다.

    **코드 사례 (간단한 애니메이션):
    ```html
    <div id="box" style="width:50px; height:50px; background-color:blue; position:relative; left:0px;"></div>
    <script>
        const box = document.getElementById('box');
        let position = 0;
        let direction = 1; // 1: 오른쪽, -1: 왼쪽

        function animate() {
            position += direction * 2; // 2px씩 이동
            box.style.left = position + 'px';

            if (position > 200 || position < 0) {
                direction *= -1; // 방향 전환
            }
            requestAnimationFrame(animate);
        }
        requestAnimationFrame(animate);
    </script>
    ```

### 4.7. Web Components (Custom Elements, Shadow DOM, HTML Templates)
Web Components는 재사용 가능한 UI 컴포넌트를 만들고 웹 애플리케이션에 통합하는 데 사용되는 웹 표준 기술 집합입니다. 이 기술은 HTML, CSS, JavaScript를 캡슐화하여 다른 코드와의 충돌 없이 독립적으로 작동하는 컴포넌트를 만들 수 있게 합니다.

- **주요 기술 요소**:
    - **Custom Elements**: 개발자가 직접 정의한 HTML 태그(예: `<my-button>`, `<user-card>`)를 만들고, 그 동작을 JavaScript로 정의할 수 있게 합니다.
        - `customElements.define('tag-name', ClassName)`: 새로운 커스텀 엘리먼트를 등록합니다. 태그 이름에는 반드시 하이픈(-)이 포함되어야 합니다.
    - **Shadow DOM**: 웹 컴포넌트의 내부 구조(HTML, CSS, JavaScript)를 메인 문서의 DOM으로부터 격리(캡슐화)합니다. 이를 통해 컴포넌트 내부의 스타일이 외부에 영향을 주지 않고, 외부 스타일이 내부에 영향을 주지 않도록 합니다.
        - `element.attachShadow({ mode: 'open' | 'closed' })`: 요소에 Shadow DOM을 연결합니다. `open` 모드는 외부 JavaScript에서 Shadow DOM에 접근할 수 있게 하고, `closed` 모드는 접근을 제한합니다.
    - **HTML Templates (`<template>` 및 `<slot>`)**: `<template>` 태그는 페이지 로드 시 즉시 렌더링되지 않는 HTML 콘텐츠를 정의합니다. 이 콘텐츠는 JavaScript를 통해 동적으로 복제되어 문서에 삽입될 수 있습니다. `<slot>` 태그는 웹 컴포넌트 내부에서 콘텐츠를 삽입할 위치를 지정하는 데 사용됩니다. HTML 템플릿에 대한 더 자세한 내용은 [HTML 문서의 '3.1.2. HTML 템플릿 (`<template>` 및 `<slot>`)' 섹션](0623정리.md#312-html-템플릿-template-및-slot)을 참조하십시오.

**코드 사례 (간단한 커스텀 엘리먼트와 Shadow DOM):**
```html
<!-- HTML에서 커스텀 엘리먼트 사용 -->
<my-greeting name="World"></my-greeting>

<script>
// JavaScript에서 커스텀 엘리먼트 정의
class MyGreeting extends HTMLElement {
    constructor() {
        super();
        // Shadow DOM을 사용하여 내부를 캡슐화
        const shadow = this.attachShadow({ mode: 'open' });
        const span = document.createElement('span');
        span.textContent = `Hello, ${this.getAttribute('name')}!`;
        shadow.appendChild(span);

        // 스타일도 캡슐화
        const style = document.createElement('style');
        style.textContent = `
            span {
                color: purple;
                font-weight: bold;
            }
        `;
        shadow.appendChild(style);
    }
}

// 커스텀 엘리먼트 등록
customElements.define('my-greeting', MyGreeting);
</script>
```

**풀스택 관점**: 웹 컴포넌트는 프레임워크에 독립적인 재사용 가능한 UI 요소를 만들 수 있게 하여, 다양한 프론트엔드 기술 스택을 사용하는 프로젝트에서도 일관된 UI를 유지하는 데 도움을 줍니다. 백엔드 개발자는 프론트엔드 팀이 제공하는 웹 컴포넌트를 활용하여 UI를 빠르게 구성할 수 있습니다.