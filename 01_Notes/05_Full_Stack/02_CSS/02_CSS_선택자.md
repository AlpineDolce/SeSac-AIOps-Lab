<h2>CSS 선택자 (Selectors)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 CSS(Cascading Style Sheets)의 핵심 개념을 이해하고, 웹 페이지의 시각적 디자인과 레이아웃을 효과적으로 제어하는 방법을 학습하는 것을 목표로 합니다. CSS 적용 방법, 선택자, 박스 모델, 레이아웃 기법 및 부트스트랩 프레임워크 활용법까지 체계적으로 정리합니다.</p>

<h2>목차</h2>

- [2. CSS 선택자 (Selectors)](#2-css-선택자-selectors)
  - [2.1. 기본 선택자](#21-기본-선택자)
  - [2.2. 결합자 (Combinators)**:](#22-결합자-combinators)
  - [2.3. 가상 클래스(Pseudo-classes)](#23-가상-클래스pseudo-classes)
  - [2.4. 가상 요소(Pseudo-elements)](#24-가상-요소pseudo-elements)
  - [2.5. 최신 선택자](#25-최신-선택자)
  - [2.6. 고급 선택자 (where, is)](#26-고급-선택자-where-is)

---

## 2. CSS 선택자 (Selectors)
선택자는 스타일을 적용할 HTML 요소를 정확하게 특정하는 역할을 합니다. 다양한 선택자를 조합하여 원하는 요소를 정교하게 선택할 수 있습니다.

### 2.1. 기본 선택자
- **전체 선택자 (Universal Selector)**: **`*`** 기호를 사용하여 페이지의 **모든 요소**를 선택합니다. 주로 페이지 전체의 기본 여백, 글꼴, `box-sizing` 등을 초기화할 때 유용합니다.

    **코드 사례:**
    ```css
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    ```

- **타입 선택자 (Type Selector)**: **HTML 태그 이름**을 직접 사용하여 해당 태그를 가진 모든 요소를 선택합니다. (예: `h1`, `p`, `div`)

- **ID 선택자 (ID Selector)**: **`#`** 기호와 요소의 `id` 속성값을 사용하여 특정 `id`를 가진 요소를 선택합니다. `id`는 페이지 내에서 **반드시 고유해야 하므로**, 하나의 특정 요소에만 스타일을 적용할 때 사용됩니다.

    **코드 사례 (`css1.html`):
    ```css
    p#para1 {
        text-align: right;
        color: blueviolet;
    }
    ```

- **클래스 선택자 (Class Selector)**: **`.`** 기호와 요소의 `class` 속성값을 사용하여 특정 클래스를 가진 모든 요소를 선택합니다. 클래스는 **여러 요소에 중복해서 사용**할 수 있어, 공통된 스타일을 여러 요소에 적용할 때 매우 유용합니다.

    **코드 사례 (`css1.html`):
    ```css
    .para2 {
        background-color: darkgreen;
        color: yellow;
    }
    ```

- **속성 선택자 (Attribute Selector)**: 특정 속성이나 속성값을 가진 요소를 선택합니다.
    - `[target]`: `target` 속성을 가진 모든 `<a>` 요소.
    - `[type="password"]`: `type` 속성의 값이 `password`인 모든 `<input>` 요소.

### 2.2. 결합자 (Combinators)**:
    - **자손 결합자 (공백)**: `div p` - `<div>`의 모든 하위 `<p>` 요소를 선택합니다. 복잡한 후손 선택자(예: `div.container > ul li a`)는 브라우저가 요소를 찾는 데 더 많은 시간을 소모하여 렌더링 성능에 영향을 줄 수 있으므로 주의해야 합니다.
    - **자식 결합자 (`>`)**: `div > p` - `<div>`의 직계 자식인 `<p>` 요소만 선택합니다.
    - **인접 형제 결합자 (`+`)**: `h1 + p` - `<h1>` 바로 다음에 오는 형제 `<p>` 요소 하나만 선택합니다.
    - **일반 형제 결합자 (`~`)**: `h1 ~ p` - `<h1>` 뒤에 오는 모든 형제 `<p>` 요소를 선택합니다.
  
### 2.3. 가상 클래스(Pseudo-classes)
요소의 특정 상태나 조건에 따라 스타일을 적용합니다. (5번 항목에서 다루는 링크 관련 가상 클래스 외의 유용한 가상 클래스들입니다.)
- **`:first-child` / `:last-child`**: 형제 요소 중 첫 번째 또는 마지막 요소를 선택합니다.
- **`:nth-child(n)`**: 형제 요소 중 `n`번째 요소를 선택합니다. (예: `:nth-child(2n)`은 짝수 번째, `:nth-child(odd)`는 홀수 번째 요소를 선택)
- **`:not(selector)`**: 특정 선택자를 제외한 나머지 요소를 선택합니다. 여러 `:not()`을 중첩하여 더 복잡한 조건을 만들 수 있습니다 (예: `:not(:not(.class))`는 `.class`를 가진 요소를 선택).
- **`:focus-within`**: 요소 또는 그 자손 요소 중 하나라도 포커스를 받았을 때 선택합니다.
- **`:checked`**: 체크된 라디오 버튼, 체크박스, 또는 선택된 옵션 요소를 선택합니다.
- **`:disabled`**: 비활성화된 요소를 선택합니다.

**코드 사례:**
```css
/* 리스트의 짝수 번째 항목에 배경색 적용 */
li:nth-child(2n) {
  background-color: #f2f2f2;
}

/* "special" 클래스를 갖지 않는 모든 p 태그 */
p:not(.special) {
  color: gray;
}

/* 폼 그룹 내의 입력 필드가 포커스되었을 때 */
.form-group:focus-within {
  border-color: blue;
  box-shadow: 0 0 5px rgba(0, 0, 255, 0.5);
}
```

### 2.4. 가상 요소(Pseudo-elements)
선택한 요소의 특정 부분에 스타일을 적용하기 위해 사용됩니다. HTML을 추가하지 않고도 장식적인 콘텐츠를 추가할 수 있습니다.
- **`::before` / `::after`**: 요소의 내용 앞이나 뒤에 새로운 콘텐츠를 생성합니다. `content` 속성과 함께 사용해야 합니다.
- **`::first-letter` / `::first-line`**: 단락의 첫 글자 또는 첫 줄에만 스타일을 적용합니다.
- **`::selection`**: 사용자가 드래그하여 선택한 텍스트에 스타일을 적용합니다.
- **`::placeholder`**: `<input>` 또는 `<textarea>` 요소의 플레이스홀더 텍스트에 스타일을 적용합니다.
- **`::marker`**: 리스트 아이템(예: `<li>`)의 마커(점, 숫자 등)를 선택하여 스타일을 적용합니다.

**코드 사례:**
```css
/* 모든 h2 태그 앞에 장식용 아이콘 추가 */
h2::before {
  content: "🚀 ";
}

/* 단락의 첫 글자를 크게 강조 */
p::first-letter {
  font-size: 2em;
  font-weight: bold;
  color: navy;
}

/* 선택된 텍스트의 배경색 변경 */
::selection {
  background-color: yellow;
  color: black;
}

/* 플레이스홀더 텍스트 스타일 */
input::placeholder {
  color: #aaa;
  font-style: italic;
}
```

### 2.5. 최신 선택자
- **`:has(selector)`**: (일명 부모 선택자) 특정 자손 요소를 포함하는 부모 요소를 선택합니다. 매우 강력한 기능으로, 복잡한 조건부 스타일링에 활용될 수 있습니다.

**코드 사례:**
```css
/* .card 내부에 .image가 있는 경우 .card에 그림자 추가 */
.card:has(.image) {
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* .item 다음에 .active 클래스를 가진 형제 요소가 있는 경우 .item에 스타일 적용 */
.item:has(+ .active) {
  border-bottom: 2px solid blue;
}
```

### 2.6. 고급 선택자 (where, is)
- **`:where(selector-list)`**: 괄호 안의 선택자 목록 중 하나와 일치하는 요소를 선택합니다. `:is()`와 유사하지만, `:where()`는 **특수성(specificity)이 0**입니다. 이는 스타일을 오버라이딩하기 쉽게 만들어주므로, 라이브러리나 프레임워크에서 기본 스타일을 정의할 때 유용합니다.

- **`:is(selector-list)`**: 괄호 안의 선택자 목록 중 하나와 일치하는 요소를 선택합니다. `:where()`와 달리, `:is()`는 **가장 높은 특수성**을 가진 선택자의 특수성을 상속받습니다. 여러 선택자에 동일한 스타일을 적용할 때 코드를 간결하게 만들 수 있습니다.

**코드 사례:**
```css
/* 모든 h1, h2, h3 태그의 마진을 0으로 설정 (특수성 0) */
:where(h1, h2, h3) {
  margin: 0;
}

/* .container 내부의 article, section, aside 요소에 패딩 적용 (가장 높은 특수성 상속) */
.container :is(article, section, aside) {
  padding: 1em;
}
```