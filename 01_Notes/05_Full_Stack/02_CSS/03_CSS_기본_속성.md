<h2>CSS 기본 속성: 스타일링의 기초</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 CSS(Cascading Style Sheets)의 핵심 개념을 이해하고, 웹 페이지의 시각적 디자인과 레이아웃을 효과적으로 제어하는 방법을 학습하는 것을 목표로 합니다. CSS 적용 방법, 선택자, 박스 모델, 레이아웃 기법 및 부트스트랩 프레임워크 활용법까지 체계적으로 정리합니다.</p>

<h2>목차</h2>

- [3. CSS 기본 속성: 스타일링의 기초](#3-css-기본-속성-스타일링의-기초)
  - [3.1. 박스 모델 (Box Model)](#31-박스-모델-box-model)
  - [3.2. 글꼴 및 텍스트 스타일](#32-글꼴-및-텍스트-스타일)
  - [3.3. 링크 스타일 (가상 클래스)](#33-링크-스타일-가상-클래스)
  - [3.4. CSS 단위 (CSS Units)](#34-css-단위-css-units)
  - [3.5. 배경 스타일](#35-배경-스타일)
  - [3.6. 테두리 스타일](#36-테두리-스타일)
  - [3.7. 오버플로우](#37-오버플로우)
  - [3.8. 그림자 효과](#38-그림자-효과)
  - [3.9. 가시성 및 상호작용](#39-가시성-및-상호작용)

---

## 3. CSS 기본 속성: 스타일링의 기초

### 3.1. 박스 모델 (Box Model)
모든 HTML 요소는 사각형의 박스로 표현되며, CSS **박스 모델**은 이 박스의 크기와 다른 요소와의 간격을 조절하는 방법을 정의합니다. 박스는 **내용(Content)**, **안쪽 여백(Padding)**, **테두리(Border)**, **바깥쪽 여백(Margin)**으로 구성됩니다.

- **`margin`**: 요소의 **바깥쪽 여백**입니다. 다른 요소와의 간격을 만들 때 사용합니다.
- **`padding`**: 요소의 테두리(border)와 내용(content) 사이의 **안쪽 여백**입니다. 콘텐츠와 테두리 사이에 공간을 확보할 때 사용합니다.
- **`border`**: 내용과 안쪽 여백을 감싸는 **테두리**입니다. `border: 1px solid black;`처럼 두께, 스타일, 색상을 함께 지정할 수 있습니다.

    **코드 사례 (`css3.html`):
    ```css
    h1 {
        border: 1px solid red; /* 테두리 */
        margin: 10px;          /* 바깥쪽 여백 */
        padding: 20px;         /* 안쪽 여백 */
    }
    ```

- **`box-sizing`**: 요소의 너비와 높이를 계산하는 방식을 결정하여 레이아웃 설계를 용이하게 합니다.
    - **`content-box`** (기본값): `width`와 `height` 속성이 **콘텐츠 영역**의 크기만을 의미합니다. 최종적인 박스 크기는 `width/height + padding + border`가 되어 예측하기 어렵습니다.
    - **`border-box`**: `width`와 `height` 속성이 **테두리까지 포함**한 크기를 의미합니다. `padding`과 `border`가 추가되어도 박스의 전체 크기는 변하지 않고, 콘텐츠 영역의 크기가 자동으로 조절됩니다. **레이아웃을 예측하기 쉬워 매우 유용하게 사용됩니다.**

- **여백 중앙 정렬 및 단축 속성**:
    - `margin: 0 auto;`: 블록 요소의 `width`가 지정된 상태에서, 위아래 여백은 0, 좌우 여백은 `auto`로 설정하여 요소를 **수평 중앙 정렬**할 수 있습니다.
    - **단축 속성 (Shorthand)**: `margin`과 `padding`은 상, 우, 하, 좌(시계방향) 순서로 값을 한 번에 지정할 수 있습니다.
        - `margin: 10px;` (모든 방향 10px)
        - `margin: 10px 20px;` (상하 10px, 좌우 20px)
        - `margin: 10px 20px 30px 40px;` (상 10px, 우 20px, 하 30px, 좌 40px)
  
### 3.2. 글꼴 및 텍스트 스타일
웹 페이지의 가독성과 시각적 매력을 높이는 데 중요한 역할을 하는 글꼴 및 텍스트 관련 속성들입니다.

- **`font-family`**: 글꼴을 지정합니다. 사용자의 시스템에 특정 글꼴이 없을 경우를 대비하여, **대체 글꼴**을 쉼표로 구분하여 여러 개 나열하는 것이 좋습니다. 목록의 마지막에는 `serif`(명조체 계열), `sans-serif`(고딕체 계열) 같은 **제네릭 글꼴(Generic Font)**을 지정하는 것이 일반적입니다.

- **`font-size`**: 글자 크기를 지정합니다. `px`(고정 크기), `%`(부모 요소 기준 백분율), `em`(부모 요소의 글꼴 크기 기준 배율), `rem`(루트 요소(`html`)의 글꼴 크기 기준 배율) 등 다양한 단위를 사용할 수 있습니다. 반응형 디자인에서는 `rem` 단위가 자주 사용됩니다.

- **`color`**: 글자 색상을 지정합니다. 색상 이름(`red`), 16진수 코드(`#ff0000`), RGB 값(`rgb(255, 0, 0)`) 등 다양한 방식으로 지정할 수 있습니다.

- **`text-align`**: 텍스트의 수평 정렬을 지정합니다. (`left`, `right`, `center`, `justify`(양쪽 정렬))

- **`text-decoration`**: 텍스트에 장식(밑줄, 윗줄, 취소선 등)을 추가하거나 제거합니다. `a` 태그의 기본 밑줄을 없앨 때 `text-decoration: none;`을 매우 흔하게 사용합니다.

- **`font-weight`**: 글자의 굵기를 지정합니다. (`normal`, `bold`, 또는 100에서 900까지의 숫자 값)

- **`line-height`**: 줄 간격을 지정하여 텍스트의 가독성을 높입니다.

- **`cursor`**: 마우스 포인터가 요소 위에 있을 때의 모양을 지정합니다. (`pointer`, `text`, `not-allowed`, `grab` 등)

- **웹 폰트**: 사용자의 컴퓨터에 설치되지 않은 폰트도 웹에서 동적으로 불러와 사용할 수 있게 해줍니다. **Google Fonts**나 **눈누(noonnu)** 같은 서비스를 통해 다양한 한글 및 영문 폰트를 쉽게 적용할 수 있습니다.

    **코드 사례 (`글꼴1.html`):
    ```css
    /* Google Fonts에서 웹 폰트 import */
    @import url('https://fonts.googleapis.com/css2?family=Nanum+Pen+Script&display=swap');

    .nanum-pen-script-regular {
      font-family: "Nanum Pen Script", cursive; /* 'cursive'는 대체 제네릭 글꼴 */
      font-weight: 400;
      font-style: normal;
    }
    ```

### 3.3. 링크 스타일 (가상 클래스)
**가상 클래스(Pseudo-classes)**는 요소의 특정 상태(예: 마우스 오버, 클릭 등)에 따라 스타일을 적용할 수 있게 해주는 특별한 선택자입니다. 특히 링크(`<a>` 태그)의 상태 변화를 시각적으로 표현하는 데 유용합니다.

- **`:link`**: 사용자가 아직 **방문하지 않은** 링크의 기본 상태.
- **`:visited`**: 사용자가 한 번 이상 **방문한** 링크. (보안상의 이유로 `color` 등 일부 속성만 변경 가능)
- **`:hover`**: **마우스 포인터**를 링크 위에 올렸을 때. 사용자에게 클릭 가능한 요소임을 시각적으로 알려주는 중요한 상호작용입니다.
- **`:active`**: 링크를 **클릭하는 순간**.
- **`:focus`**: 키보드의 `Tab` 키 등으로 요소가 **포커스**를 받았을 때. 웹 접근성 측면에서 매우 중요합니다.

> **LVHA 순서**: 가상 클래스를 정의할 때는 **L**ink → **V**isited → **H**over → **A**ctive 순서로 작성하는 것이 좋습니다. 우선순위 규칙에 따라 이 순서가 지켜지지 않으면 일부 스타일이 적용되지 않을 수 있습니다.

  **코드 사례 (`글꼴1.html`):
  ```css
  a:link { color: red; text-decoration: none; }       /* 방문 전, 밑줄 없음 */
  a:visited { color: green; }   /* 방문 후 */
  a:hover { color: hotpink; text-decoration: underline; }  /* 마우스 오버, 밑줄 표시 */
  a:active { color: blue; }     /* 클릭 시 */
  a:focus { outline: 2px solid blue; } /* 포커스 시 파란색 외곽선 */
  ```

### 3.4. CSS 단위 (CSS Units)
CSS에서 크기나 길이를 지정할 때 사용되는 단위는 크게 **절대 단위**와 **상대 단위**로 나뉩니다. 반응형 웹 디자인에서는 상대 단위의 활용이 매우 중요합니다.

- **절대 단위 (Absolute Units)**: 화면의 크기나 해상도에 관계없이 항상 고정된 물리적 크기를 가집니다. 인쇄물 등 고정된 레이아웃에 적합합니다.
    - `px` (픽셀): 가장 흔하게 사용되며, 1픽셀은 화면의 한 점을 나타냅니다. 기기 해상도에 따라 물리적 크기가 달라질 수 있습니다.
    - `pt` (포인트): 1pt는 1/72인치입니다. 인쇄물에 주로 사용됩니다.
    - `cm`, `mm`, `in` (센티미터, 밀리미터, 인치): 물리적 길이 단위.

- **상대 단위 (Relative Units)**: 다른 요소의 크기나 뷰포트(브라우저 화면)의 크기에 비례하여 크기가 결정됩니다. 반응형 디자인에 필수적입니다.
    - **`em`**: 부모 요소의 `font-size`를 기준으로 합니다. (예: 부모 `font-size: 16px;`일 때 `1.5em`은 `24px`)
        - **장점**: 부모 요소의 크기에 따라 비례적으로 조절되어 유연한 레이아웃 구성에 유리합니다.
        - **단점**: 중첩된 요소에서 계산이 복잡해질 수 있습니다 (복합적인 `em` 값).
    - **`rem`**: 루트 요소(`<html>`)의 `font-size`를 기준으로 합니다. (예: `html font-size: 16px;`일 때 `1.5rem`은 `24px`)
        - **장점**: `em`의 단점을 보완하여, 항상 루트 요소 기준으로 계산되므로 예측 가능하고 일관된 크기 조절이 가능합니다. 반응형 디자인에서 글꼴 크기나 간격 설정에 널리 사용됩니다.
    - **`vw` (viewport width)**: 뷰포트 너비의 1%를 기준으로 합니다. (예: 뷰포트 너비가 1000px일 때 `10vw`는 100px)
    - **`vh` (viewport height)**: 뷰포트 높이의 1%를 기준으로 합니다. (예: 뷰포트 높이가 800px일 때 `10vh`는 80px)
        - **장점**: 뷰포트 크기에 따라 요소의 크기가 동적으로 변하여 유동적인 레이아웃에 적합합니다.
    - `vmin` / `vmax`: 뷰포트 너비와 높이 중 작은/큰 값을 기준으로 합니다.
    - `%` (백분율): 부모 요소의 크기에 대한 백분율입니다.

**코드 사례:**
```css
html {
  font-size: 16px; /* 기본 폰트 크기 설정 (rem의 기준) */
}

body {
  font-size: 1rem; /* 16px */
}

h1 {
  font-size: 2.5rem; /* 40px */
  margin-bottom: 1.5rem; /* 24px */
}

.hero-section {
  height: 50vh; /* 뷰포트 높이의 50% */
  width: 80vw;  /* 뷰포트 너비의 80% */
}

.responsive-text {
  font-size: clamp(1rem, 2vw + 0.5rem, 2.5rem); /* 최소 1rem, 최대 2.5rem, 뷰포트 너비에 따라 유동적 */
}
```

### 3.5. 배경 스타일
요소의 배경을 꾸미는 데 사용되는 속성들입니다.
- **`background-color`**: 배경 색상을 지정합니다.
- **`background-image`**: 배경 이미지를 지정합니다. 여러 이미지를 쉼표로 구분하여 지정할 수 있습니다.
- **`background-repeat`**: 배경 이미지의 반복 방식을 지정합니다. (`no-repeat`, `repeat-x`, `repeat-y`, `repeat`)
- **`background-position`**: 배경 이미지의 위치를 지정합니다. (`center`, `top left`, `50% 50%`, `10px 20px`)
- **`background-size`**: 배경 이미지의 크기를 지정합니다. (`auto`, `cover`, `contain`, `100% 100%`, `50px 50px`)
- **`background-attachment`**: 배경 이미지가 스크롤될 때 고정될지 여부를 지정합니다. (`scroll`, `fixed`, `local`)
- **`background` (단축 속성)**: 위 속성들을 한 번에 지정할 수 있는 단축 속성입니다.

**코드 사례:**
```css
.hero-banner {
  background: url('hero-bg.jpg') no-repeat center / cover;
  height: 400px;
  color: white;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
}
```

### 3.6. 테두리 스타일
요소의 테두리를 꾸미는 데 사용되는 속성들입니다.
- **`border-width`**: 테두리 두께를 지정합니다.
- **`border-style`**: 테두리 스타일을 지정합니다. (`solid`, `dashed`, `dotted`, `double` 등)
- **`border-color`**: 테두리 색상을 지정합니다.
- **`border-radius`**: 요소의 모서리를 둥글게 만듭니다. 단일 값 또는 네 모서리 각각의 값을 지정할 수 있습니다.
- **`outline`**: 요소가 포커스를 받았을 때 그려지는 외곽선입니다. `outline: none;`으로 제거할 경우 웹 접근성을 위해 대체 스타일을 제공해야 합니다. `outline: width style color;`
- **`outline-offset`**: `outline`과 요소의 테두리 사이의 간격을 지정합니다.

**코드 사례:**
```css
.rounded-box {
  border: 2px solid #333;
  border-radius: 10px; /* 모든 모서리를 10px 둥글게 */
  padding: 20px;
  margin: 20px;
}

.circle-image {
  border-radius: 50%; /* 원형 이미지 */
  overflow: hidden;
}
```

### 3.7. 오버플로우
요소의 콘텐츠가 요소의 박스를 넘어설 때(오버플로우) 콘텐츠를 어떻게 처리할지 지정합니다.
- **`overflow: visible;`**: 기본값. 콘텐츠가 박스를 넘어서도 잘리지 않고 그대로 표시됩니다.
- **`overflow: hidden;`**: 콘텐츠가 박스를 넘어서면 넘어선 부분을 숨깁니다.
- **`overflow: scroll;`**: 콘텐츠가 박스를 넘어서면 스크롤바를 항상 표시합니다.
- **`overflow: auto;`**: 콘텐츠가 박스를 넘어서는 경우에만 스크롤바를 표시합니다.
- **`overflow-x` / `overflow-y`**: 가로 또는 세로 방향의 오버플로우만 개별적으로 제어합니다.

**코드 사례:**
```css
.scrollable-content {
  width: 300px;
  height: 150px;
  overflow: auto; /* 내용이 넘칠 때만 스크롤바 표시 */
  border: 1px solid #ccc;
  padding: 10px;
}
```

### 3.8. 그림자 효과
요소에 그림자 효과를 추가하여 입체감을 부여합니다.
- **`box-shadow`**: 요소의 박스에 그림자를 추가합니다. `box-shadow: h-offset v-offset blur spread color inset;` 여러 그림자를 쉼표(`,`)로 구분하여 적용할 수 있습니다.
    - `h-offset`: 가로 거리 (필수)
    - `v-offset`: 세로 거리 (필수)
    - `blur`: 그림자 흐림 정도 (선택)
    - `spread`: 그림자 확장 정도 (선택)
    - `color`: 그림자 색상 (선택)
    - `inset`: 내부 그림자 (선택)
- **`text-shadow`**: 텍스트에 그림자를 추가합니다. `text-shadow: h-offset v-offset blur color;` 여러 그림자를 쉼표(`,`)로 구분하여 적용할 수 있습니다.

**코드 사례:**
```css
.card-with-shadow {
  box-shadow: 3px 3px 5px rgba(0, 0, 0, 0.2); /* 가로 3px, 세로 3px, 흐림 5px, 투명도 0.2 검은색 */
  padding: 20px;
  margin: 20px;
  background-color: white;
}
```
```css
.shadowed-text {
  text-shadow: 2px 2px 4px #666; /* 가로 2px, 세로 2px, 흐림 4px, 회색 */
  font-size: 3em;
  color: #333;
}
```

### 3.9. 가시성 및 상호작용
요소의 가시성과 사용자 상호작용에 영향을 미치는 속성들입니다.

- **`opacity`**: 요소의 투명도를 지정합니다. `0`은 완전 투명, `1`은 완전 불투명입니다. `0`으로 설정해도 요소는 공간을 차지하며 이벤트(클릭 등)를 받을 수 있습니다.

- **`visibility`**: 요소를 보이거나 숨깁니다. `hidden`으로 설정하면 요소는 화면에서 사라지지만, 여전히 공간을 차지하며 레이아웃에 영향을 줍니다. `display: none;`과 달리 공간을 유지한다는 점이 중요합니다.

- **`pointer-events`**: 요소가 마우스 이벤트(클릭, 호버 등)에 어떻게 반응할지 지정합니다. `none`으로 설정하면 요소는 이벤트를 무시하고 그 아래에 있는 요소가 이벤트를 받게 됩니다. 오버레이나 비활성화된 UI 요소에 유용합니다.

**코드 사례:**
```css
.transparent-box {
  opacity: 0.5; /* 50% 투명 */
  background-color: blue;
  width: 100px;
  height: 100px;
}

.hidden-but-space {
  visibility: hidden; /* 숨겨지지만 공간은 차지 */
  width: 100px;
  height: 100px;
  border: 1px solid red;
}
```