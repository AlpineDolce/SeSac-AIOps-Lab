<h2>CSS 레이아웃 기법: 웹 페이지 구조화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 CSS(Cascading Style Sheets)의 핵심 개념을 이해하고, 웹 페이지의 시각적 디자인과 레이아웃을 효과적으로 제어하는 방법을 학습하는 것을 목표로 합니다. CSS 적용 방법, 선택자, 박스 모델, 레이아웃 기법 및 부트스트랩 프레임워크 활용법까지 체계적으로 정리합니다.</p>

<h2>목차</h2>

- [4. 레이아웃 기법: 웹 페이지 구조화](#4-레이아웃-기법-웹-페이지-구조화)
  - [4.1. `display` 속성](#41-display-속성)
  - [4.2. `float` 및 `position` 속성](#42-float-및-position-속성)
    - [Position 속성](#position-속성)
  - [4.3. Flexbox 레이아웃](#43-flexbox-레이아웃)
  - [4.4. Grid 레이아웃](#44-grid-레이아웃)
  - [4.5. 반응형 웹 디자인 (Responsive Web Design)](#45-반응형-웹-디자인-responsive-web-design)
  - [4.6. 논리적 속성 (Logical Properties)](#46-논리적-속성-logical-properties)
  - [4.7. 컨테이너 쿼리 (Container Queries)](#47-컨테이너-쿼리-container-queries)
  - [4.8. Subgrid](#48-subgrid)
  - [4.9. 미디어 요소 레이아웃](#49-미디어-요소-레이아웃)

---

## 4. 레이아웃 기법: 웹 페이지 구조화
웹 페이지의 전체적인 구조와 요소의 배치를 결정하는 핵심 속성들입니다.

### 4.1. `display` 속성
요소가 화면에 어떻게 표시될지(렌더링 방식)와 다른 요소와의 관계를 결정합니다.
  - **`block`**: 항상 **새 줄**에서 시작하고, 부모 요소의 **사용 가능한 전체 너비**를 차지합니다. `width`, `height`, `margin`, `padding` 속성을 모두 사용할 수 있습니다. (예: `<h1>`, `<p>`, `<div>`, `<section>`)
  - **`inline`**: **새 줄에서 시작하지 않고**, 다른 인라인 요소와 같은 줄에 배치됩니다. 콘텐츠의 너비만큼만 공간을 차지하며, **`width`와 `height` 속성을 적용할 수 없습니다**. 상하 `margin`과 `padding`도 의도대로 적용되지 않을 수 있습니다. (예: `<a>`, `<span>`, `<img>`, `<b>`)
  - **`inline-block`**: `inline`처럼 다른 요소와 같은 줄에 배치되면서도, `block`처럼 **`width`와 `height`, `margin`, `padding`을 모두 지정**할 수 있는 유용한 속성입니다. 가로 메뉴 등을 만들 때 자주 사용됩니다.
  - **`none`**: 요소를 화면에서 **완전히 사라지게** 합니다. 마치 처음부터 없었던 것처럼 공간도 차지하지 않습니다. (JavaScript와 함께 사용하여 특정 조건에서 요소를 숨기거나 보일 때 유용)
  - **`contents`**: 요소 자체는 렌더링되지 않고 그 자식 요소들만 렌더링됩니다. Flexbox나 Grid 레이아웃에서 HTML 구조를 변경하지 않고도 특정 요소의 자식들을 직접적인 Flex/Grid 아이템으로 만들 때 유용합니다.

### 4.2. `float` 및 `position` 속성
- **`float`**: 요소를 일반적인 문서 흐름에서 벗어나 띄워서 왼쪽이나 오른쪽으로 배치합니다. 다른 인라인 콘텐츠는 `float`된 요소의 주위를 흐르게 됩니다. 과거에는 전체 페이지 레이아웃을 잡기 위해 많이 사용되었으나, 현재는 이미지와 텍스트를 배치하는 등 제한적인 용도로 사용되며, 전체 레이아웃은 **Flexbox나 Grid를 사용하는 것이 더 현대적이고 효율적**입니다.
- **`clear`**: `float` 속성이 적용된 요소가 후속 요소에 미치는 영향을 차단합니다. `float`된 요소들 바로 다음에 오는 요소에 `clear: both;`를 적용하여 레이아웃이 깨지는 것을 방지하는 데 사용됩니다.

    **코드 사례 (`float.html`):
    ```css
    .image-wrapper {
      float: left; /* 이미지를 왼쪽으로 띄움 */
      margin-right: 15px; 
    }
    .text-content {
        /* 이 텍스트는 이미지 주위를 흐릅니다. */
    }
    .footer {
        clear: both; /* float 효과를 여기서 해제하여 푸터가 정상적으로 배치되도록 함 */
    }
    ```
#### Position 속성
요소의 위치를 지정하는 방식을 결정합니다. `top`, `right`, `bottom`, `left` 속성과 함께 사용됩니다.
- **`static`**: 기본값. 일반적인 문서 흐름에 따라 배치됩니다.
- **`relative`**: 일반적인 위치를 기준으로 상대적으로 이동합니다. `top`, `left` 등으로 위치를 옮겨도 원래 있던 공간은 그대로 차지합니다.
- **`absolute`**: 가장 가까운 `position: relative/absolute/fixed`인 조상 요소를 기준으로 위치가 결정됩니다. 만약 그런 조상이 없다면, 문서의 `<body>`를 기준으로 합니다. 일반적인 문서 흐름에서 완전히 벗어나므로, 원래 있던 공간을 차지하지 않습니다.
- **`fixed`**: 뷰포트(브라우저 창)를 기준으로 위치가 고정됩니다. 스크롤을 해도 항상 같은 자리에 보입니다.
- **`sticky`**: 평소에는 `static`처럼 동작하다가, 스크롤 위치가 특정 지점에 도달하면 `fixed`처럼 동작합니다. (예: 스크롤해도 상단에 고정되는 내비게이션 바)
- **`z-index`**: `position`이 `static`이 아닌 요소들의 쌓임 순서(stack order)를 결정합니다. 숫자가 클수록 위에 표시됩니다. `z-index`는 **쌓임 맥락(Stacking Context)** 내에서만 작동하며, 쌓임 맥락은 `position` 속성 외에도 `opacity`, `transform`, `filter`, `will-change` 등 특정 CSS 속성에 의해 생성될 수 있습니다. 새로운 쌓임 맥락이 생성되면, 그 안의 `z-index`는 외부 쌓임 맥락의 영향을 받지 않고 독립적으로 작동합니다.


### 4.3. Flexbox 레이아웃
**1차원 레이아웃**을 위한 강력하고 유연한 모델입니다. 아이템들을 행이나 열 방향으로 쉽게 정렬하고 배치할 수 있습니다.
- **컨테이너 속성 (부모 요소에 적용)**
    - `display: flex;`: 자식 요소(아이템)들을 Flexbox 레이아웃으로 만듭니다.
    - `flex-direction`: 아이템이 배치될 주축의 방향을 결정합니다. (`row`, `column` 등)
    - `flex-wrap`: 아이템들이 컨테이너를 넘칠 때 줄바꿈을 할지 결정합니다. (`nowrap`, `wrap`, `wrap-reverse`)
    - `justify-content`: **주축** 방향으로 아이템들을 정렬합니다. (`flex-start`, `center`, `space-between` 등)
    - `align-items`: **교차축** 방향으로 아이템들을 정렬합니다. (`flex-start`, `center`, `stretch` 등)
    - `align-content`: 여러 줄의 Flex 아이템들이 교차축 방향으로 정렬되는 방식을 결정합니다. (`flex-start`, `center`, `space-between` 등)
    - **`gap`**: Flexbox 아이템들 사이의 간격을 지정합니다. `row-gap`과 `column-gap`의 단축 속성입니다. (Grid와 동일하게 사용 가능)
- **아이템 속성 (자식 요소에 적용)**
    - `flex-grow`: 아이템이 컨테이너 내 여백을 차지하는 비율을 지정합니다.
    - `flex-shrink`: 아이템이 컨테이너 크기가 줄어들 때 축소되는 비율을 지정합니다.
    - `flex-basis`: 아이템의 기본 크기를 지정합니다.
    - `flex`: `flex-grow`, `flex-shrink`, `flex-basis`의 단축 속성입니다.
    - `order`: 아이템의 시각적 순서를 변경합니다. (기본값은 0)
    - `align-self`: 개별 아이템의 교차축 정렬 방식을 재정의합니다.
    - **`place-items`**: `align-items`와 `justify-items`의 단축 속성입니다. (Flexbox에서는 `align-items`만 적용)
    - **`place-content`**: `align-content`와 `justify-content`의 단축 속성입니다.
    - **`place-self`**: `align-self`와 `justify-self`의 단축 속성입니다. (Flexbox에서는 `align-self`만 적용)

**코드 사례:**
```css
.container {
  display: flex;
  flex-wrap: wrap; /* 아이템이 넘치면 다음 줄로 */
  justify-content: space-around; /* 주축(가로) 방향으로 균등한 간격으로 배치 */
  align-items: center;         /* 교차축(세로) 방향으로 중앙 정렬 */
  height: 200px;
  background-color: #eee;
}
.item:nth-child(1) {
  order: 2; /* 첫 번째 아이템의 순서를 2로 변경 */
}
.item:nth-child(2) {
  align-self: flex-end; /* 두 번째 아이템만 교차축 끝으로 정렬 */
}
```

### 4.4. Grid 레이아웃
**2차원(행과 열) 레이아웃**을 위한 모델로, 복잡한 레이아웃도 간단하게 구성할 수 있습니다.
- **컨테이너 속성 (부모 요소에 적용)**
    - `display: grid;`: 자식 요소들을 Grid 레이아웃으로 만듭니다.
    - `grid-template-columns` / `grid-template-rows`: 그리드의 열과 행의 크기와 개수를 정의합니다. (예: `grid-template-columns: 1fr 1fr 1fr;`는 동일한 비율의 3개 열을 생성)
    - `grid-template-areas`: 그리드 영역의 이름을 정의하여 레이아웃을 시각적으로 구성할 수 있습니다.
    - `gap` (또는 `grid-gap`): 그리드 아이템 사이의 간격을 지정합니다.
    - `grid-auto-rows` / `grid-auto-columns`: 명시적으로 정의되지 않은 암시적 그리드 트랙의 크기를 지정합니다.
- **아이템 속성 (자식 요소에 적용)**
    - `grid-row` / `grid-column`: 아이템이 차지할 행/열의 시작 및 끝 라인을 지정합니다.
    - `grid-area`: `grid-row`와 `grid-column`의 단축 속성이거나, `grid-template-areas`에서 정의된 영역 이름을 사용하여 아이템을 배치합니다.

**코드 사례:**
```css
.grid-container {
  display: grid;
  grid-template-columns: repeat(3, 1fr); /* 1fr(비율) 크기의 열 3개 */
  grid-template-rows: auto 100px; /* 첫 행은 자동, 두 번째 행은 100px */
  grid-template-areas:
    "header header header"
    "main   main   sidebar";
  gap: 10px; /* 아이템 사이의 간격은 10px */
}
.header { grid-area: header; background-color: lightblue; }
.main { grid-area: main; background-color: lightcoral; }
.sidebar { grid-area: sidebar; background-color: lightgreen; }
.item { /* 일반 그리드 아이템 */ }
.item-span-2 { 
  grid-column: span 2; /* 2개 열 차지 */
  grid-row: 1 / 3; /* 1행부터 3행까지 차지 */
}
```

### 4.5. 반응형 웹 디자인 (Responsive Web Design)
반응형 웹 디자인은 다양한 기기(데스크톱, 태블릿, 모바일)의 화면 크기에 따라 웹 페이지의 레이아웃과 요소들이 자동으로 최적화되어 표시되도록 하는 디자인 접근 방식입니다. 하나의 HTML 문서와 CSS로 모든 기기에 대응할 수 있어 유지보수 효율성이 높습니다.

- **미디어 쿼리 (`@media`)**: 특정 조건(화면 너비, 높이, 해상도 등)에 따라 다른 CSS 스타일을 적용할 수 있게 해줍니다. 반응형 웹 디자인의 핵심 기술입니다.
    - **`min-width`**: 지정된 너비 이상일 때 적용.
    - **`max-width`**: 지정된 너비 이하일 때 적용.
    - **`orientation`**: 뷰포트의 방향(가로 또는 세로)에 따라 적용.

    **코드 사례:**
    ```css
    /* 기본 스타일 (모바일 우선) */
    body {
        font-size: 16px;
    }

    /* 화면 너비가 768px 이상일 때 적용되는 스타일 (태블릿 및 데스크톱) */
    @media (min-width: 768px) {
        body {
            font-size: 18px;
        }
        .container {
            width: 750px;
        }
    }

    /* 화면 너비가 1200px 이상일 때 적용되는 스타일 (큰 데스크톱) */
    @media (min-width: 1200px) {
        body {
            font-size: 20px;
        }
        .container {
            width: 1170px;
        }
    }
    ```

- **모바일 우선 (Mobile-First) 디자인**: 작은 화면(모바일)을 기준으로 기본 스타일을 작성하고, 미디어 쿼리를 사용하여 점진적으로 큰 화면에 대한 스타일을 추가하는 방식입니다. 성능과 접근성 측면에서 유리하며, 현대 웹 디자인의 표준으로 자리 잡고 있습니다.

- **유동적인 그리드 및 이미지**: 고정된 `px` 값 대신 `%`, `vw`, `vh` 등 상대 단위를 사용하여 요소의 크기가 화면 크기에 비례하여 조절되도록 합니다. 이미지의 경우 `max-width: 100%; height: auto;`를 사용하여 부모 요소의 너비에 맞춰 자동으로 크기가 조절되도록 합니다.

    **코드 사례:**
    ```css
    img {
        max-width: 100%; /* 이미지가 부모 요소보다 커지지 않도록 */
        height: auto;    /* 가로 비율에 맞춰 세로 자동 조절 */
        display: block;  /* 이미지 하단 여백 제거 */
    }
    ```

- **`aspect-ratio`**: 요소의 가로세로 비율을 명시적으로 지정합니다. 이미지, 비디오, `iframe` 등 다양한 요소에 적용하여 반응형 레이아웃에서 콘텐츠의 비율을 유지하는 데 매우 유용합니다.

    **코드 사례:**
    ```css
    .video-container {
      width: 100%;
      aspect-ratio: 16 / 9; /* 16:9 비율 유지 */
    }

    .square-image {
      width: 100px;
      aspect-ratio: 1; /* 1:1 정사각형 비율 유지 */
    }
    ```

- **뷰포트 메타 태그**: HTML `<head>`에 `<meta name="viewport" content="width=device-width, initial-scale=1.0">`를 반드시 포함하여 브라우저가 페이지를 기기 너비에 맞춰 렌더링하도록 지시해야 합니다.

- **반응형 실습 예제: 카드 레이아웃**
    아래 예제는 일반적인 화면 분기점(Breakpoint)을 사용하여 화면 크기에 따라 카드 컴포넌트의 레이아웃이 변경되는 것을 보여줍니다.

    **HTML 구조:**
    ```html
    <div class="card-container">
        <div class="card">
            <h3>카드 1</h3>
            <p>이것은 카드 내용입니다.</p>
        </div>
        <div class="card">
            <h3>카드 2</h3>
            <p>이것은 카드 내용입니다.</p>
        </div>
        <div class="card">
            <h3>카드 3</h3>
            <p>이것은 카드 내용입니다.</p>
        </div>
    </div>
    ```

    **CSS 스타일:**
    ```css
    .card-container {
        display: grid;
        gap: 1rem;
    }

    .card {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
    }

    /* 모바일 (기본 스타일): 1열 레이아웃 */
    .card-container {
        grid-template-columns: 1fr;
    }

    /* 태블릿 (768px 이상): 2열 레이아웃 */
    @media (min-width: 768px) {
        .card-container {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    /* 데스크톱 (1024px 이상): 3열 레이아웃 */
    @media (min-width: 1024px) {
        .card-container {
            grid-template-columns: repeat(3, 1fr);
        }
    }
    ```

### 4.6. 논리적 속성 (Logical Properties)
CSS 논리적 속성은 물리적 방향(top, right, bottom, left) 대신 논리적 방향(start, end, block, inline)을 사용하여 스타일을 정의합니다. 이는 다국어 웹사이트에서 텍스트 방향(좌-우, 우-좌, 상-하 등)이 변경될 때 레이아웃이 자동으로 조정되도록 하여 국제화(i18n)에 매우 유용합니다.

- **`margin-inline-start` / `margin-inline-end`**: 인라인 방향(텍스트 흐름 방향)의 시작/끝 여백.
- **`margin-block-start` / `margin-block-end`**: 블록 방향(새 줄이 시작되는 방향)의 시작/끝 여백.
- **`padding-inline-start` / `padding-block-end`**: 패딩도 동일하게 적용됩니다.

**코드 사례:**
```css
.element {
  margin-inline-start: 20px; /* LTR에서는 left-margin, RTL에서는 right-margin */
  padding-block-end: 10px;   /* 수평 쓰기 모드에서는 bottom-padding */
}
```

### 4.7. 컨테이너 쿼리 (Container Queries)
미디어 쿼리가 뷰포트(브라우저 창)의 크기에 따라 스타일을 변경하는 반면, 컨테이너 쿼리는 **부모 컨테이너 요소의 크기**에 따라 자식 요소의 스타일을 변경할 수 있게 해주는 최신 CSS 기능입니다. 컴포넌트 기반 개발에서 재사용성과 유연성을 크게 향상시킵니다.

- **사용법**: `@container` 규칙을 사용하며, 컨테이너에 `container-type` 및 `container-name` 속성을 지정해야 합니다.

**코드 사례:**
```css
/* 컨테이너 정의 */
.card-container {
  container-type: inline-size; /* 인라인 방향(가로) 크기에 반응 */
  container-name: card-layout; /* 컨테이너 이름 지정 */
}

/* 컨테이너 크기에 따른 스타일 변경 */
@container card-layout (min-width: 400px) {
  .card-title {
    font-size: 1.5rem;
  }
  .card-image {
    display: block;
  }
}

@container card-layout (max-width: 300px) {
  .card-description {
    display: none; /* 작은 컨테이너에서는 설명 숨기기 */
  } 
}
```

### 4.8. Subgrid
CSS Grid의 강력한 기능 중 하나로, 중첩된 그리드 아이템이 부모 그리드의 트랙 정의를 상속받을 수 있게 해줍니다. 이를 통해 복잡한 그리드 레이아웃에서 아이템들을 부모 그리드 라인에 정확히 정렬할 수 있어, 더욱 정교하고 일관된 디자인을 구현할 수 있습니다.

**코드 사례:**
```css
.parent-grid {
  display: grid;
  grid-template-columns: 1fr 2fr 1fr;
  gap: 10px;
}

.nested-item {
  display: grid;
  grid-column: 1 / -1; /* 부모 그리드의 전체 너비 차지 */
  grid-template-columns: subgrid; /* 부모의 열 트랙을 상속 */
  gap: inherit; /* 부모의 갭 상속 */
}

.nested-item > div:nth-child(1) {
  grid-column: 1; /* 부모의 첫 번째 열에 정렬 */
}
```
```css
.nested-item > div:nth-child(2) {
  grid-column: 2 / span 2; /* 부모의 두 번째 열부터 두 칸 차지 */
}
```

### 4.9. 미디어 요소 레이아웃
이미지, 비디오 등 미디어 콘텐츠가 요소의 박스 내에서 어떻게 표시될지 제어하는 속성들입니다.

- **`object-fit`**: `<img>`나 `<video>`와 같은 대체(replaced) 요소의 콘텐츠가 요소의 박스에 어떻게 맞춰질지 지정합니다.
    - `fill`: 콘텐츠를 요소의 박스에 맞게 늘리거나 줄입니다. (기본값)
    - `contain`: 콘텐츠의 가로세로 비율을 유지하면서 요소의 박스 안에 완전히 포함되도록 크기를 조절합니다.
    - `cover`: 콘텐츠의 가로세로 비율을 유지하면서 요소의 박스를 완전히 덮도록 크기를 조절합니다. 일부 콘텐츠가 잘릴 수 있습니다.
    - `none`: 콘텐츠의 크기를 조절하지 않습니다.
    - `scale-down`: `none`과 `contain` 중 더 작은 크기를 선택합니다.

- **`object-position`**: `object-fit` 속성이 `contain` 또는 `cover`일 때, 요소의 박스 내에서 콘텐츠의 위치를 지정합니다. `background-position`과 유사하게 작동합니다.

**코드 사례:**
```css
.responsive-image {
  width: 100%;
  height: 200px;
  object-fit: cover; /* 이미지가 박스를 꽉 채우도록, 비율 유지 */
  object-position: center; /* 이미지 중앙 정렬 */
}

.video-thumbnail {
  width: 300px;
  height: 150px;
  object-fit: contain; /* 비디오가 박스 안에 완전히 보이도록, 비율 유지 */
  object-position: top; /* 비디오 상단 정렬 */
}
```