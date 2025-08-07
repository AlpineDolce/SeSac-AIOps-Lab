<h2>HTML 핵심 요소: 콘텐츠 구조화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 HTML의 핵심 개념과 주요 태그 사용법을 체계적으로 정리하여, 웹 페이지의 구조를 이해하고 작성하는 능력을 기르는 것을 목표로 합니다. 각 태그의 의미와 사용 사례를 통해 시맨틱 웹의 중요성을 이해하고, 풀스택 개발 실무에 필요한 보안, 성능 최적화 관점까지 통합하여 실용적인 예제를 통해 학습 효과를 높이고자 합니다.</p>

<h2>목차</h2>

- [2. 핵심 HTML 요소: 콘텐츠 구조화](#2-핵심-html-요소-콘텐츠-구조화)
  - [2.1. 텍스트 관련 태그](#21-텍스트-관련-태그)
  - [2.2. 목록 태그](#22-목록-태그)
  - [2.3. 링크 태그 (`<a>`)](#23-링크-태그-a)
  - [2.4. 이미지 태그 (`<img>`)](#24-이미지-태그-img)
  - [2.5. 테이블 태그](#25-테이블-태그)
  - [2.6. 폼 태그 (`<form>`)](#26-폼-태그-form)
  - [2.7. 그룹화 및 레이아웃 (`<div>`, `<span>`, 시맨틱 태그)](#27-그룹화-및-레이아웃-div-span-시맨틱-태그)
  - [2.8. 커스텀 데이터 속성 (`data-*` attributes)](#28-커스텀-데이터-속성-data--attributes)

---

## 2. 핵심 HTML 요소: 콘텐츠 구조화

### 2.1. 텍스트 관련 태그
HTML은 텍스트의 구조와 의미를 정의하는 다양한 태그를 제공합니다.

- **제목(Headings)**: **`<h1>`**부터 **`<h6>`**까지의 태그는 문서의 제목이나 섹션의 제목을 나타냅니다. `<h1>`이 가장 중요한 최상위 제목이며, `<h6>`으로 갈수록 중요도가 낮아지고 글자 크기도 작아집니다. 검색 엔진은 제목 태그를 사용하여 문서의 구조와 내용을 파악하므로, 문서의 계층 구조에 맞게 사용하는 것이 중요합니다.

    **코드 사례 (`sample1.html`):
    ```html
    <h1>가장 중요한 제목</h1>
    <h2>부제목</h2>
    <h3>하위 섹션 제목</h3>
    ```

- **문단(Paragraph)**: **`<p>`** 태그는 하나의 문단을 정의합니다. 브라우저는 `<p>` 태그 위아래에 약간의 여백을 자동으로 추가하여 문단을 구분합니다.

    **코드 사례 (`sample1.html`):
    ```html
    <p> 
    이것은 첫 번째 문단입니다. 문단은 관련된 문장들의 그룹입니다.
    </p>
    <p>
    이것은 두 번째 문단입니다.
    </p>
    ```

- **줄바꿈(Line Break)**: **`<br>`** 태그는 문단 내에서 강제로 줄을 바꿀 때 사용됩니다. 시(poem)나 주소처럼 줄바꿈 자체가 의미를 갖는 경우에 유용합니다. 닫는 태그가 없는 빈 태그(empty tag)입니다.

- **수평선(Horizontal Rule)**: **`<hr>`** 태그는 내용에서 주제가 변경되었음을 나타내기 위해 수평선을 삽입합니다. 이 역시 빈 태그입니다.

    **코드 사례 (`sample2.html`):
    ```html
    <p>첫 번째 주제에 대한 내용입니다.</p>
    <hr>
    <p>이제 주제가 바뀌어 두 번째 내용이 시작됩니다.</p>
    ```

- **강조**:
    - **`<strong>`**: 텍스트가 내용상 **매우 중요함**을 나타냅니다. 스크린 리더는 이 태그를 강조하여 읽어줍니다. 브라우저는 기본적으로 굵은 글씨체로 표시합니다.
    - **`<b>`**: 특별한 중요성은 없지만, 주변 텍스트와 구분하기 위해 **굵은 글씨체**로 표시할 때 사용합니다. (예: 제품명, 키워드)
    - **`<em>`** (Emphasis): 텍스트의 특정 부분을 **강조**하여 억양을 다르게 표현하고 싶을 때 사용합니다. 스크린 리더는 기울임 톤으로 읽어주며, 브라우저는 기울임꼴로 표시합니다.
    - **`<i>`** (Italic): 기술 용어, 외국어 구문, 생각 등 일반적인 산문과 다른 톤이나 분위기의 텍스트에 사용됩니다. 시각적으로는 `<em>`과 같지만 의미론적 중요도는 없습니다.
    - **`<blockquote>`**: 다른 출처에서 인용한 긴 텍스트 블록을 나타냅니다. `cite` 속성으로 출처 URL을 명시할 수 있습니다.
    - **`<cite>`**: 작품의 제목(책, 영화, 노래 등)이나 인용문의 출처를 나타냅니다.
    - **`<abbr>`**: 약어(abbreviation)를 나타내며, `title` 속성으로 전체 이름을 제공하여 접근성을 높일 수 있습니다.
    - **`<address>`**: 문서나 섹션의 작성자/소유자에 대한 연락처 정보를 제공합니다.
    - **`<code>`, `<pre>`**: 코드 스니펫을 표시할 때 유용합니다. 특히 `<pre>`는 미리 서식이 지정된 텍스트(공백, 줄바꿈 유지)를 나타냅니다. 기술 문서나 블로그에서 필수적입니다.

    **코드 사례 (`sample2.html`)**:
    ```html
    <p>이 프로젝트는 <strong>반드시</strong> 내일까지 완료해야 합니다.</p>
    <p>이것은 <b>HTML</b> 학습 문서입니다.</p>
    <p>이것은 <em>정말</em> 멋진 기능입니다!</p>
    <p>그는 생각했다, <i>이게 가능할까?</i></p>
    <p>이것은 <abbr title="HyperText Markup Language">HTML</abbr>에 대한 문서입니다.</p>
    <blockquote>
        <p>웹은 모든 사람을 위한 것입니다.</p>
        <cite>팀 버너스 리</cite>
    </blockquote>
    <pre><code>
function hello() {
    console.log("Hello, World!");
}
    </code></pre>
    ```

- **인라인 스타일링 컨테이너 (`<span>`)**: **`<span>`** 태그는 텍스트의 일부를 그룹화하여 CSS로 스타일을 적용하기 위한 **인라인(inline) 컨테이너**입니다. 그 자체로는 아무런 시각적 변화를 주지 않으며, 특정 부분에만 다른 색상이나 폰트를 적용하고 싶을 때 유용합니다.

    **코드 사례 (`sample1.html`):
    ```html
    <h1><span style="color:red;font-weight:bold;">Learn</span> HTML</h1>
    ```

### 2.2. 목록 태그
정보를 구조화하여 목록 형태로 보여주기 위한 태그들입니다.

- **순서 없는 목록 (Unordered List)**: **`<ul>`** 태그는 순서가 중요하지 않은 항목들의 목록을 만들 때 사용합니다. 각 목록 항목은 **`<li>`** (list item) 태그로 표시됩니다. `list-style-type` CSS 속성을 사용하여 목록 앞의 마커(bullet) 모양을 변경할 수 있습니다 (예: `none`, `disc`, `circle`, `square`).

    **코드 사례 (`리스트1.html`):
    ```html
    <ul style="list-style-type:circle;">
        <li>사과</li>
        <li>바나나</li>
        <li>오렌지</li>
    </ul>
    ```

- **순서 있는 목록 (Ordered List)**: **`<ol>`** 태그는 순서가 중요한 목록을 만들 때 사용합니다. 각 항목은 **`<li>`** 태그로 표시되며, 브라우저는 자동으로 번호를 매깁니다. `type` 속성을 사용하여 번호의 종류( `1`, `A`, `a`, `I`, `i` )를, `start` 속성을 사용하여 시작 번호를 지정할 수 있습니다.

    **코드 사례 (`리스트1.html` 보강):
    ```html
    <ol type="A" start="3">
        <li>세 번째 단계 (C)</li>
        <li>네 번째 단계 (D)</li>
        <li>다섯 번째 단계 (E)</li>
    </ol>
    ```

- **정의 목록 (Description List)**: **`<dl>`** 태그는 용어와 그에 대한 설명을 목록으로 만들 때 사용됩니다. **`<dt>`** (definition term) 태그로 용어를, **`<dd>`** (definition description) 태그로 설명을 나타냅니다. 사전처럼 용어를 정의하거나, 질문과 답변 형식의 내용을 구조화할 때 유용합니다.

    **코드 사례 (`리스트1.html`):
    ```html
    <dl>
        <dt>HTML</dt>
        <dd>Hyper Text Markup Language의 약자로, 웹 페이지의 구조를 정의합니다.</dd>
        <dt>CSS</dt>
        <dd>Cascading Style Sheets의 약자로, 웹 페이지의 디자인과 레이아웃을 담당합니다.</dd>
    </dl>
    ```

### 2.3. 링크 태그 (`<a>`)
**`<a>`** (anchor) 태그는 다른 웹 페이지, 파일, 이메일 주소, 또는 같은 페이지 내의 특정 위치로 연결되는 **하이퍼링크**를 만듭니다.

- **외부/다른 문서 링크**: **`href`** (hypertext reference) 속성에 이동할 페이지의 URL이나 파일 경로를 지정합니다. **`target="_blank"`** 속성을 추가하면 링크가 새 브라우저 탭에서 열려 사용자가 현재 페이지를 벗어나지 않게 할 수 있습니다. **보안을 위해 `target="_blank"` 사용 시 `rel="noopener noreferrer"` 속성을 함께 사용하는 것이 권장됩니다. 자세한 내용은 [4.3.2. 링크 보안](#432-링크-보안-relnoopener-noreferrer)을 참조하세요.**

    **코드 사례 (`링크.html`)**:
    ```html
    <a href="https://www.google.com" target="_blank">구글(새 탭)</a> <br/>
    <a href="./sample1.html">같은 폴더의 sample1.html로 이동</a>
    ```

- **페이지 내 특정 위치로 이동 (내부 링크)**: `href` 속성값에 `#` 기호와 함께 이동하고 싶은 요소의 `id` 속성값을 적습니다. 해당 링크를 클릭하면 그 `id`를 가진 요소의 위치로 부드럽게 스크롤됩니다. 페이지가 긴 경우 목차를 만드는 데 유용합니다.

    **코드 사례 (`링크2.html`)**:
    ```html
    <a href="#section3">3번 섹션으로 가기</a>
    ...
    <h2 id="section3">3번 섹션</h2>
    ```

- **이미지 링크**: `<a>` 태그 안에 `<img>` 태그를 중첩하여 이미지를 클릭 가능한 링크로 만들 수 있습니다.

    **코드 사례 (`링크.html`)**:
    ```html
    <a href="https://w3schools.com"><img src="./images/smiley.gif" alt="W3Schools로 이동"></a>
    ```

- **이메일 링크**: `href`에 **`mailto:`** 를 사용하면 사용자의 기본 이메일 클라이언트를 열어 바로 이메일을 보낼 수 있게 합니다.

    **코드 사례:**
    ```html
    <a href="mailto:contact@example.com">문의하기</a>
    ```

- **버튼을 이용한 링크**: JavaScript를 사용하여 버튼 클릭 시 특정 페이지로 이동하게 할 수 있습니다.
    **코드 사례 (`링크.html`)**:
    ```html
    <button onclick="document.location='./sample2.html'">버튼으로 이동</button>
    ```

### 2.4. 이미지 태그 (`<img>`)
**`<img>`** 태그는 웹 페이지에 이미지를 삽입할 때 사용합니다. 닫는 태그가 없는 빈 태그입니다.

- **`src` (source)**: **필수 속성**으로, 이미지 파일의 경로(URL)를 지정합니다.
- **`alt` (alternative text)**: **필수 속성**으로, 네트워크 오류나 경로 문제로 이미지를 표시할 수 없을 때 대신 표시될 텍스트를 지정합니다. 또한, 스크린 리더 사용자를 위해 이미지를 설명하는 역할을 하므로 접근성 측면에서 매우 중요합니다.
- **`width`, `height`**: 이미지의 너비와 높이를 지정합니다. CSS로도 지정할 수 있지만, HTML에 지정하면 이미지가 로드되기 전에 브라우저가 해당 공간을 미리 확보하여 레이아웃이 흔들리는 현상을 방지할 수 있습니다.

**지원하는 주요 형식**:
- **`jpg`/`jpeg`**: 사진과 같이 색상이 많은 이미지에 적합하며, 손실 압축을 사용합니다.
- **`png`**: 배경을 투명하게 처리할 수 있어 로고나 아이콘 등에 많이 사용됩니다. 비손실 압축을 사용합니다.
- **`gif`**: 여러 장의 이미지를 합쳐 움직이는 효과(애니메이션)를 만들 수 있으며, 제한된 색상만 지원합니다.
- **`svg`** (Scalable Vector Graphics): 벡터 기반 이미지로, 확대하거나 축소해도 품질이 저하되지 않아 아이콘이나 로고에 매우 적합합니다.
- **`webp`**: JPEG와 PNG의 장점을 결합한 차세대 이미지 포맷으로, 더 좋은 품질과 낮은 용량을 제공하지만 일부 구형 브라우저에서는 지원되지 않을 수 있습니다.

- **반응형 이미지 (`<picture>` 요소)**: `<img>` 태그의 `srcset`과 `sizes` 속성 외에도, `<picture>` 요소를 사용하면 뷰포트 크기나 장치 특성(예: 해상도, 이미지 형식 지원 여부)에 따라 다른 이미지 파일이나 아트 디렉션(art direction)을 적용할 수 있습니다. 이는 최적의 반응형 이미지 전략을 구현하는 데 중요합니다.

**코드 사례 (`<picture>` 요소):**
```html
<picture>
  <source srcset="images/hero-large.webp" type="image/webp" media="(min-width: 1200px)">
  <source srcset="images/hero-medium.webp" type="image/webp" media="(min-width: 768px)">
  <source srcset="images/hero-small.webp" type="image/webp">
  <img src="images/hero-fallback.jpg" alt="반응형 히어로 이미지">
</picture>
```

**코드 사례 (`이미지1.html`)**:
```html
<img src="./images/1.jpg" width="200" alt="붉은 오각형" loading="lazy">
```

> **보충: `<img>` 태그와 CSS `background-image`의 차이**
> - **`<img>` 태그**: 이미지가 **콘텐츠의 일부**일 때 사용합니다. (예: 상품 이미지, 기사 내 사진). 검색 엔진이 이미지를 인덱싱할 수 있고, `alt` 텍스트를 통해 의미를 전달할 수 있습니다.
> - **`background-image`**: 순전히 **장식적인 목적**으로 이미지를 사용할 때 적합합니다. (예: 페이지 배경, 버튼 아이콘). 콘텐츠의 일부가 아니므로 스크린 리더가 읽지 않습니다.

### 2.5. 테이블 태그
**`<table>`** 태그는 데이터를 **행(row)과 열(column)의 표 형식**으로 나타낼 때 사용합니다. 과거에는 레이아웃을 잡기 위해 사용되기도 했지만, 현재는 시맨틱 웹 표준에 따라 데이터 표시에만 사용하는 것이 권장됩니다.

- **`<table>`**: 테이블 전체를 감싸는 컨테이너입니다.
- **`<tr>`** (table row): 테이블의 **행**을 정의합니다.
- **`<td>`** (table data): 행 안의 각 **셀(칸)**을 정의합니다. 실제 데이터가 들어가는 부분입니다.
- **`<th>`** (table header): 테이블의 **제목(헤더) 셀**을 정의합니다. `scope` 속성(`col` 또는 `row`)을 사용하여 해당 헤더가 열의 제목인지 행의 제목인지 명시해주면 접근성을 향상시킬 수 있습니다.
- **`rowspan`, `colspan`**: 각각 **행과 열을 병합(merge)**하는 속성입니다. 값은 병합할 셀의 개수입니다.
- **`<colgroup>`, `<col>`**: 테이블의 특정 열 전체에 공통적인 스타일(예: 너비, 배경색)을 적용하고자 할 때 사용합니다.
- **`<thead>`, `<tbody>`, `<tfoot>`**: 테이블의 내용을 **머리글, 본문, 바닥글 그룹**으로 묶어 구조를 더 명확하게 만듭니다. 이는 접근성과 스타일링에 유용하며, 긴 테이블을 인쇄할 때 각 페이지마다 `<thead>`와 `<tfoot>`이 반복되도록 할 수 있습니다.
- **`<caption>`**: 테이블의 **제목이나 설명**을 제공합니다. 항상 `<table>` 태그 바로 다음에 위치해야 합니다.

**코드 사례: 종합 (`테이블1.html`, `테이블2.html` 보강):
```html
<table border="1">
    <caption>월별 판매 실적</caption>
    <colgroup>
        <col style="background-color: #f2f2f2;">
        <col span="2">
    </colgroup>
    <thead>
        <tr>
            <th scope="col">월</th>
            <th scope="col">품목</th>
            <th scope="col">판매량</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2">1월</td>
            <td>노트북</td>
            <td>100</td>
        </tr>
        <tr>
            <td>모니터</td>
            <td>150</td>
        </tr>
        <tr>
            <td>2월</td>
            <td>키보드</td>
            <td>200</td>
        </tr>
    </tbody>
    <tfoot>
        <tr>
            <th scope="row" colspan="2">총 합계</th>
            <td>450</td>
        </tr>
    </tfoot>
</table>
```

### 2.6. 폼 태그 (`<form>`)
**`<form>`** 태그는 사용자로부터 텍스트 입력, 항목 선택, 파일 첨부 등의 입력을 받아 서버로 전송하기 위한 영역을 정의합니다.

- **주요 속성**:
    - **`action`**: 폼 데이터가 전송될 서버 측 스크립트의 URL을 지정합니다.
    - **`method`**: HTTP 전송 방식을 지정합니다 (`GET` 또는 `POST`).
    - **`enctype`**: `method`가 `post`일 때, 폼 데이터가 서버로 전송될 때의 인코딩 유형을 지정합니다. 파일을 첨부할 때는 반드시 **`multipart/form-data`** 로 설정해야 합니다.

- **`<label>` 태그**: 입력 필드에 대한 설명을 나타냅니다. **`for`** 속성값을 입력 요소의 **`id`** 속성값과 일치시키면, 라벨을 클릭했을 때 해당 입력 요소에 포커스가 이동하여 사용자 편의성과 접근성이 향상됩니다.

- **`<fieldset>`과 `<legend>`**: 폼 안에서 관련된 여러 입력 요소들을 그룹화할 때 사용합니다. **`<fieldset>`**으로 그룹을 감싸고, **`<legend>`**로 그룹의 제목을 붙입니다.

- **입력 요소 (`<input>`, `<select>`, `<textarea>`, `<button>`)**:
    - **`<input>`**: `type` 속성에 따라 다양한 형태의 입력 필드를 만듭니다.
        - `text`, `password`, `radio`, `checkbox`, `submit`, `reset`, `button`, `number`, `range`
        - **HTML5 추가 타입**: `email`(이메일 형식 자동 검증), `url`, `tel`, `date`(날짜 선택), `color`(색상 선택), `file`(파일 첨부) 등
        - **HTML5 유효성 검사 속성**: `pattern`(정규 표현식), `min`(최소값), `max`(최대값), `minlength`(최소 길이), `maxlength`(최대 길이), `step`(증가/감소 단위) 등 다양한 속성을 사용하여 클라이언트 측에서 입력값의 유효성을 검사할 수 있습니다. 이는 사용자 경험을 향상시키고 불필요한 서버 요청을 줄이지만, **보안과 데이터 무결성을 위해 서버 측 유효성 검사는 필수적**입니다. 클라이언트 측 유효성 검사는 사용자가 쉽게 우회할 수 있으므로, 서버 측 검사를 절대 대체할 수 없습니다.

    - **`<select>`와 `<option>`**: 드롭다운 목록을 만듭니다. `multiple` 속성을 추가하면 다중 선택이 가능해집니다.
    - **`<textarea>`**: 여러 줄의 텍스트를 입력받습니다.
    - **`<button>`**: `<input type="submit">`이나 `<input type="button">`과 유사한 버튼을 만듭니다. `<img>`나 `<strong>` 같은 다른 태그를 내부에 포함할 수 있어 더 유연한 스타일링이 가능합니다.
    - **`<datalist>`**: `<input>` 태그와 함께 사용하여 사용자에게 입력 제안 목록을 제공합니다. 드롭다운 목록처럼 보이지만, 사용자가 직접 입력할 수도 있습니다.
    - **`<optgroup>`**: `<select>` 태그 내에서 관련 `<option>`들을 그룹화할 때 사용합니다.
    - `form` 속성: `<input>` 요소가 `<form>` 태그 외부에 있더라도 특정 폼과 연결할 수 있게 합니다.

- **기타 속성**:
    - **`name`**: 각 입력 요소의 이름을 지정하며, 서버로 데이터를 전송할 때 `key=value` 쌍에서 `key` 역할을 합니다.
    - **`value`**: 해당 입력 요소의 초기값 또는 선택되었을 때 서버로 전송될 값입니다.
    - **`placeholder`**: 입력 필드에 사용자가 어떤 값을 입력해야 하는지 알려주는 힌트 텍스트를 표시합니다.
    - **`required`**: 폼 전송 시 해당 필드가 반드시 채워져 있어야 함을 나타냅니다.
    - **`readonly`**: 값을 수정할 수 없지만, 폼 데이터는 서버로 전송됩니다.
    - **`disabled`**: 요소를 완전히 비활성화하여 사용 및 클릭이 불가능하게 만들고, 데이터도 서버로 전송되지 않습니다.

- **풀스택 관점: 폼 데이터 처리 및 유효성 검사**:
    - **서버 측 처리**: 폼 데이터는 `action` 속성에 지정된 URL로 전송되며, 백엔드 서버는 이 데이터를 파싱하여 비즈니스 로직을 수행합니다. 보안과 데이터 무결성을 위해 **반드시 서버 측에서도 유효성 검사를 수행**해야 합니다. 클라이언트 측 유효성 검사는 사용자 경험 향상을 위한 것이며, 보안을 보장하지 않습니다.
    - **AJAX 폼 제출**: 현대 웹 애플리케이션에서는 페이지 새로고침 없이 JavaScript(Fetch API, Axios 등)를 사용하여 폼 데이터를 비동기적으로 서버에 제출하고 응답을 처리하는 방식이 흔히 사용됩니다. 이는 사용자 경험을 향상시키고 SPA(Single Page Application) 개발에 필수적입니다.
    - **폼 데이터 객체 (FormData)**: `FormData` 객체를 사용하면 `<form>` 태그의 데이터를 쉽게 캡처하여 `Fetch API`나 `XMLHttpRequest`를 통해 서버로 전송할 수 있습니다. 파일 업로드와 같은 `multipart/form-data` 형식의 데이터 전송에 특히 유용합니다.

**코드 사례: 종합 (`form` 관련 파일 보강):
```html
<form action="/signup" method="post">
    <fieldset>
        <legend>개인 정보</legend>
        <p>
            <label for="username">사용자 이름:</label>
            <input type="text" id="username" name="username" required minlength="2" placeholder="이름을 입력하세요">
        </p>
        <p>
            <label for="email">이메일:</label>
            <input type="email" id="email" name="email" required>
        </p>
        <p>
            <label for="password">비밀번호:</label>
            <input type="password" id="password" name="password" required minlength="8" pattern="(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}" title="최소 8자 이상, 대문자, 소문자, 숫자를 포함해야 합니다.">
        </p>
    </fieldset>
    
    <fieldset>
        <legend>관심사</legend>
        <p>
            <label><input type="checkbox" name="interest" value="sports"> 운동</label>
            <label><input type="checkbox" name="interest" value="music"> 음악</label>
            <label><input type="checkbox" name="interest" value="movie"> 영화</label>
        </p>
        <p>
            <label for="job">직업:</label>
            <select id="job" name="job">
                <option value="">--선택--</option>
                <option value="developer">개발자</option>
                <option value="designer">디자이너</option>
            </select>
        </p>
    </fieldset>
    
    <button type="submit">가입하기</button>
</form>

<label for="browser">선호하는 브라우저:</label>
<input list="browsers" name="browser" id="browser">
<datalist id="browsers">
    <option value="Chrome">
    <option value="Firefox">
    <option value="Safari">
</datalist>

<select name="cars">
    <optgroup label="스웨덴 자동차">
        <option value="volvo">Volvo</option>
        <option value="saab">Saab</option>
    </optgroup>
    <optgroup label="독일 자동차">
        <option value="mercedes">Mercedes</option>
        <option value="audi">Audi</option>
    </optgroup>
</select>
```

### 2.7. 그룹화 및 레이아웃 (`<div>`, `<span>`, 시맨틱 태그)
콘텐츠를 구조적으로 묶고 웹 페이지의 레이아웃을 잡기 위해 사용되는 태그들입니다.

- **`<div>` (Division)**: 특별한 의미 없이 콘텐츠를 그룹화하는 데 사용되는 대표적인 **블록 레벨(block-level) 요소**입니다. 주로 CSS와 함께 사용되어 페이지의 특정 구역이나 레이아웃을 정의합니다.

- **`<span>`**: `<div>`와 비슷하게 콘텐츠를 그룹화하지만, **인라인 레벨(inline-level) 요소**라는 점이 다릅니다. 주로 문장 안의 특정 단어나 구절에 별도의 스타일을 적용할 때 사용됩니다.

- **HTML5 시맨틱(Semantic) 레이아웃 태그**: HTML5에서는 `<div>` 태그만으로 레이아웃을 구성하는 대신, 각 구역의 의미를 명확하게 나타내는 시맨틱 태그 사용을 권장합니다. 이는 코드의 가독성을 높이고 검색 엔진 최적화(SEO), 접근성 향상뿐만 아니라 **CSS 스타일링의 용이성**도 제공합니다. 의미 있는 태그를 사용하면 CSS 선택자를 더 명확하게 작성할 수 있어 유지보수성이 향상됩니다.
    - **`<header>`**: 페이지나 특정 섹션의 머리말.
    - **`<nav>`**: 네비게이션 링크들의 집합.
    - **`<main>`**: 문서의 핵심적인 주요 콘텐츠.
    - **`<section>`**: 문서 내에서 관련된 콘텐츠들의 구역.
    - **`<article>`**: 독립적으로 배포 가능한 콘텐츠.
    - **`<aside>`**: 주요 내용과 간접적으로 관련된 부분 (사이드바 등).
    - **`<footer>`**: 페이지나 섹션의 꼬리말.

**코드 사례: 시맨틱 레이아웃 구조**
```html
<body>
    <header>
        <h1>웹사이트 로고</h1>
        <nav>
            <ul>
                <li><a href="#">홈</a></li>
                <li><a href="#">소개</a></li>
                <li><a href="#">연락처</a></li>
            </ul>
        </nav>
    </header>

    <main>
        <article>
            <h2>블로그 글 제목</h2>
            <p>이것은 블로그 글의 내용입니다...</p>
        </article>
        <aside>
            <h3>관련 링크</h3>
            <ul>
                <li><a href="#">링크 1</a></li>
                <li><a href="#">링크 2</a></li>
            </ul>
        </aside>
    </main>

    <footer>
        <p>&copy; 2025 My Website. All rights reserved.</p>
    </footer>
</body>
```

### 2.8. 커스텀 데이터 속성 (`data-*` attributes)
`data-*` 속성은 HTML 요소에 사용자 정의 데이터를 저장하는 표준적인 방법을 제공합니다. 이 속성들은 페이지에 시각적인 영향을 주지 않으면서, JavaScript에서 쉽게 접근하고 조작할 수 있어 동적인 웹 애플리케이션 개발에서 매우 유용하게 활용됩니다.

- **활용 목적**: 주로 JavaScript와 연동하여 UI/UX를 제어하거나, 특정 요소에 대한 추가 정보를 저장할 때 사용됩니다. 예를 들어, 버튼에 특정 사용자 ID를 저장하거나, 모달 창을 열기 위한 데이터 등을 저장할 수 있습니다.
- **접근 방법**: JavaScript에서 `dataset` 속성을 통해 `data-*` 속성 값에 접근할 수 있습니다. `data-example-name`과 같은 속성은 `element.dataset.exampleName`으로 접근합니다.

**코드 사례 (`data-*` attributes):**
```html
<button data-user-id="123" data-action="delete" data-item-name="상품A">삭제</button>

<div id="product-info" data-product-id="P001" data-price="15000">
  상품 A에 대한 정보
</div>

<script>
  const deleteButton = document.querySelector('button[data-action="delete"]');
  deleteButton.addEventListener('click', function() {
    const userId = this.dataset.userId; // '123'
    const action = this.dataset.action; // 'delete'
    const itemName = this.dataset.itemName; // '상품A'
    console.log(`User ${userId} wants to ${action} ${itemName}`);
  });

  const productDiv = document.getElementById('product-info');
  console.log(`Product ID: ${productDiv.dataset.productId}, Price: ${productDiv.dataset.price}`);
</script>
```