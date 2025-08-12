<h2>Django Backend: Template과 Form</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django의 Template 시스템과 Form 처리 방식을 학습하는 것을 목표로 합니다. 템플릿을 사용하여 동적인 웹 페이지를 생성하는 방법과, ModelForm을 활용하여 사용자 입력을 효율적으로 처리하고 데이터베이스와 상호작용하는 방법을 이해합니다.</p>

<h2>목차</h2> 

- [1. Django Template (템플릿 시스템): 실무 가이드](#1-django-template-템플릿-시스템-실무-가이드)
  - [1.1. 템플릿 설정 및 구조](#11-템플릿-설정-및-구조)
    - [1.1.1. `settings.py`의 `TEMPLATES` 설정](#111-settingspy의-templates-설정)
    - [1.1.2. 템플릿 파일의 위치와 이름 규칙](#112-템플릿-파일의-위치와-이름-규칙)
    - [1.1.3. 템플릿 로딩 순서](#113-템플릿-로딩-순서)
  - [1.2. 템플릿의 기본 구성 요소](#12-템플릿의-기본-구성-요소)
    - [1.2.1. 변수 (Variables): `{{ ... }}`](#121-변수-variables---)
    - [1.2.2. 태그 (Tags): `{% ... %}`](#122-태그-tags---)
      - [1.1.2.1. 제어 흐름 태그 (Control Flow Tags)](#1121-제어-흐름-태그-control-flow-tags)
      - [1.2.2.2. 데이터 출력 및 포함 태그 (Data Output \& Inclusion Tags)](#1222-데이터-출력-및-포함-태그-data-output--inclusion-tags)
      - [1.2.2.3. URL 및 정적 파일 태그 (URL \& Static File Tags)](#1223-url-및-정적-파일-태그-url--static-file-tags)
      - [1.2.2.4. 보안 태그 (Security Tags)](#1224-보안-태그-security-tags)
    - [1.2.3. 필터 (Filters): `|`](#123-필터-filters-)
  - [1.3. 템플릿 상속: 코드 재사용의 핵심](#13-템플릿-상속-코드-재사용의-핵심)
  - [1.4. 커스텀 템플릿 태그와 필터 만들기](#14-커스텀-템플릿-태그와-필터-만들기)
- [2. Django Form (폼 처리): 실무 가이드](#2-django-form-폼-처리-실무-가이드)
  - [2.1. HTTP 메서드 (GET vs POST)와 폼 처리](#21-http-메서드-get-vs-post와-폼-처리)
    - [2.1.1 GET 메서드](#211-get-메서드)
    - [2.1.2 POST 메서드](#212-post-메서드)
    - [2.1.3 Django에서의 폼 처리와 GET/POST](#213-django에서의-폼-처리와-getpost)
  - [2.2. Form vs ModelForm: 언제 무엇을 쓸까?](#22-form-vs-modelform-언제-무엇을-쓸까)
  - [2.3. ModelForm: 모델과 폼의 완벽한 결합](#23-modelform-모델과-폼의-완벽한-결합)
  - [2.4. 뷰에서의 폼 처리: 표준 워크플로우](#24-뷰에서의-폼-처리-표준-워크플로우)
  - [2.5. 폼 렌더링 커스터마이징](#25-폼-렌더링-커스터마이징)
  - [2.6. 폼 유효성 검사(Validation) 심화](#26-폼-유효성-검사validation-심화)
  - [2.7. 위젯 커스터마이징](#27-위젯-커스터마이징)

---

## 1. Django Template (템플릿 시스템): 실무 가이드

Django 템플릿 시스템의 핵심 철학은 **표현과 로직의 분리**입니다. 템플릿은 최종적으로 사용자에게 보여질 모습(HTML 구조, CSS 등)에 집중하고, 복잡한 비즈니스 로직은 뷰(View)에서 처리하여 서로의 역할과 책임을 명확히 구분합니다. 이를 통해 디자이너와 개발자의 협업이 원활해지고 코드의 유지보수성이 향상됩니다.

### 1.1. 템플릿 설정 및 구조

Django가 템플릿 파일을 찾고 렌더링하기 위해서는 `settings.py`에 적절한 설정이 필요하며, 파일 시스템 내에서 일정한 구조를 따르는 것이 중요합니다.

#### 1.1.1. `settings.py`의 `TEMPLATES` 설정

`settings.py` 파일에는 Django가 템플릿을 로드하는 방법을 정의하는 `TEMPLATES` 설정이 있습니다. 이는 리스트 형태이며, 각 항목은 하나의 템플릿 엔진 설정을 나타내는 딕셔너리입니다.

```python
# myproject/settings.py

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'], # 프로젝트 전역 템플릿 디렉토리
        'APP_DIRS': True, # 각 앱의 templates/ 디렉토리에서 템플릿을 찾을지 여부
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

*   **`DIRS`**:
    *   Django가 템플릿 파일을 찾을 디렉토리들의 리스트입니다. 주로 **프로젝트 전역에서 사용되는 템플릿** (예: `base.html`, `home.html`, `404.html` 등)을 저장하는 데 사용됩니다.
    *   일반적으로 `BASE_DIR / 'templates'`와 같이 프로젝트 루트에 `templates` 디렉토리를 만들고 이곳을 지정합니다.
*   **`APP_DIRS`**:
    *   `True`로 설정하면, Django는 `INSTALLED_APPS`에 등록된 각 앱 내부의 `templates/` 디렉토리에서도 템플릿을 찾습니다.
    *   이는 **앱별로 독립적인 템플릿**을 관리할 때 매우 유용하며, 앱의 재사용성을 높여줍니다.

#### 1.1.2. 템플릿 파일의 위치와 이름 규칙

Django는 템플릿을 찾을 때 정해진 순서와 규칙을 따릅니다.

*   **프로젝트 전역 템플릿:**
    *   `settings.py`의 `DIRS`에 지정된 디렉토리 (예: `myproject/templates/`) 내에 직접 템플릿 파일을 저장합니다.
    *   예시: `myproject/templates/base.html`, `myproject/templates/home.html`

*   **앱별 템플릿 (권장):**
    *   각 Django 앱 내부에 `templates/` 디렉토리를 생성하고, 그 안에 **앱 이름과 동일한 서브 디렉토리**를 한 번 더 만드는 것이 일반적인 관례이자 **강력히 권장되는 방법**입니다.
    *   예시: `blog/templates/blog/post_list.html`, `accounts/templates/accounts/login.html`
    *   **왜 앱 이름으로 서브 디렉토리를 만드는가?**
        *   **이름 충돌 방지:** 여러 앱에서 동일한 이름의 템플릿 파일(예: `list.html`)을 사용할 경우 발생할 수 있는 이름 충돌을 방지합니다.
        *   **명확한 출처:** 템플릿이 어떤 앱에 속하는지 명확하게 알 수 있어 코드의 가독성과 유지보수성이 향상됩니다.
        *   **재사용성:** 앱을 다른 프로젝트에 재사용할 때 템플릿 경로가 명확해집니다.

#### 1.1.3. 템플릿 로딩 순서

Django는 템플릿을 로드할 때 다음 순서로 디렉토리를 탐색합니다.

1.  `settings.py`의 `TEMPLATES` 설정에서 `DIRS`에 지정된 디렉토리들 (순서대로).
2.  `settings.py`의 `INSTALLED_APPS`에 등록된 앱들의 `templates/` 디렉토리들 (등록된 순서대로).

예를 들어, `{% include 'blog/post_card.html' %}`와 같이 템플릿을 로드할 때, Django는 먼저 `DIRS`에 지정된 `myproject/templates/blog/post_card.html`을 찾고, 없으면 `INSTALLED_APPS`에 등록된 앱들(예: `blog/templates/blog/post_card.html`)을 순서대로 탐색합니다.

---

### 1.2. 템플릿의 기본 구성 요소

Django 템플릿은 크게 세 가지 기본 구성 요소로 이루어져 있습니다.

#### 1.2.1. 변수 (Variables): `{{ ... }}`

뷰에서 템플릿으로 전달된 데이터를 표시할 때 사용합니다. 점(`.`)을 사용하여 객체의 속성, 딕셔너리의 키, 리스트의 인덱스, 심지어는 모델 객체의 메서드에도 접근할 수 있습니다.

-   **기본 사용법:**
    ```html
    <p>게시글 제목: {{ post.title }}</p>
    <p>작성자: {{ post.author.username }}</p>
    ```

-   **딕셔너리/리스트 접근:**
    ```html
    {# 뷰에서 context = {'user_info': {'name': 'Alice', 'age': 30}, 'skills': ['Python', 'Django']} #}
    <p>이름: {{ user_info.name }}</p>
    <p>첫 번째 스킬: {{ skills.0 }}</p>
    ```

-   **메서드 호출 (인자 없는 경우):**
    ```html
    {# Post 모델에 get_absolute_url 메서드가 정의되어 있을 때 #}
    <a href="{{ post.get_absolute_url }}">게시글 보기</a>
    ```

#### 1.2.2. 태그 (Tags): `{% ... %}`

템플릿의 제어 흐름을 담당하며, 반복, 조건 분기, 상속, 데이터 로드 등 다양한 기능을 수행합니다. 태그는 시작 태그와 종료 태그가 한 쌍을 이루는 경우가 많습니다.

##### 1.1.2.1. 제어 흐름 태그 (Control Flow Tags)

*   **조건문: `{% if %}`, `{% elif %}`, `{% else %}`**
    주어진 조건에 따라 다른 내용을 렌더링합니다. 파이썬의 `if/elif/else`와 유사합니다.
    ```html
    {% if user.is_authenticated %}
        <p>환영합니다, {{ user.username }}님!</p>
        <a href="{% url 'logout' %}">로그아웃</a>
    {% elif user.is_staff %}
        <p>관리자 페이지로 이동</p>
    {% else %}
        <a href="{% url 'login' %}">로그인</a>
        <a href="{% url 'signup' %}">회원가입</a>
    {% endif %}

    {# 변수가 비어있는지, False인지, None인지 확인 #}
    {% if post_list %}
        <p>게시글이 있습니다.</p>
    {% else %}
        <p>게시글이 없습니다.</p>
    {% endif %}

    {# not, and, or 연산자 사용 가능 #}
    {% if user.is_authenticated and user.has_perm 'blog.add_post' %}
        <button>새 게시글 작성</button>
    {% endif %}
    ```

*   **반복문: `{% for %}`, `{% empty %}`**
    리스트, 튜플, 쿼리셋 등 반복 가능한(iterable) 객체를 순회하며 내용을 반복 렌더링합니다. `{% empty %}` 태그는 반복할 항목이 없을 때 실행될 내용을 정의합니다.
    ```html
    <ul>
        {% for post in post_list %}
            <li>
                <h3>{{ post.title }}</h3>
                <p>작성자: {{ post.author.username }}</p>
                <p>{{ post.content|truncatewords:20 }}</p>
            </li>
        {% empty %}
            {# post_list가 비어있을 때 #}
            <li>아직 게시물이 없습니다.</li>
        {% endfor %}
    </ul>

    {# forloop 객체 활용 #}
    <ol>
        {% for item in items %}
            <li>{{ forloop.counter }}. {{ item }}</li>
            {% if forloop.last %}
                <p>마지막 항목입니다.</p>
            {% endif %}
        {% endfor %}
    </ol>
    ```

*   **변수 할당: `{% with %}`**
    복잡하거나 반복적인 연산 결과를 변수에 할당하여 템플릿의 가독성을 높이고 성능을 개선합니다. `{% with %}` 블록 내에서만 유효합니다.
    ```html
    {% with total_comments=post.comments.count %}
        <p>이 게시글에는 총 {{ total_comments }}개의 댓글이 있습니다.</p>
    {% endwith %}
    ```

*   **주석: `{% comment %}`**
    템플릿 코드 내에 주석을 추가할 때 사용합니다. 렌더링 시 HTML 출력에 포함되지 않습니다.
    ```html
    {% comment "관리자용 주석" %}
        이 부분은 사용자에게 보이지 않습니다.
        개발자나 디자이너가 템플릿에 대한 설명을 남길 때 유용합니다.
    {% endcomment %}
    ```

##### 1.2.2.2. 데이터 출력 및 포함 태그 (Data Output & Inclusion Tags)

*   **템플릿 포함: `{% include %}`**
    다른 템플릿 파일을 현재 위치에 삽입합니다. 재사용 가능한 작은 템플릿 조각(예: 내비게이션 바, 푸터, 광고 배너)을 만들 때 유용합니다.
    ```html
    {# blog/templates/blog/post_card.html #}
    <div class="post-card">
        <h3>{{ post.title }}</h3>
        <p>{{ post.author.username }}</p>
    </div>

    {# blog/templates/blog/post_list.html 에서 사용 #}
    {% for post in post_list %}
        {% include "blog/post_card.html" with post=post %}
    {% endfor %}
    ```

*   **내용 그대로 출력: `{% verbatim %}`**
    `{% verbatim %}` 블록 내의 모든 내용은 Django 템플릿 엔진에 의해 처리되지 않고, 있는 그대로 출력됩니다. JavaScript 코드나 다른 템플릿 엔진의 문법을 HTML에 포함할 때 유용합니다.
    ```html
    {% verbatim %}
        <script>
            // 이 안의 {{ variable }} 이나 {% tag %} 는 Django 템플릿 문법으로 해석되지 않습니다.
            var data = {{ some_js_variable }};
        </script>
    {% endverbatim %}
    ```

##### 1.2.2.3. URL 및 정적 파일 태그 (URL & Static File Tags)

*   **URL 생성: `{% url '...' %}`**
    URL을 하드코딩하는 대신, `urls.py`에 정의된 URL `name` (별칭)을 사용하여 동적으로 경로를 생성합니다. URL 패턴이 변경되어도 템플릿 코드를 수정할 필요가 없어 유지보수성이 크게 향상됩니다.

    #### `urls.py`에서 URL 정의하기

    템플릿에서 `{% url %}` 태그를 사용하려면, 먼저 `urls.py` 파일에서 해당 URL 패턴에 `name` 인자를 지정해야 합니다. 앱의 URL을 프로젝트의 `urls.py`에 포함(include)할 때는 `app_name`을 지정하여 네임스페이스를 설정할 수 있습니다.

    **프로젝트 `urls.py` (`myproject/urls.py`):**
    ```python
    # myproject/urls.py
    from django.contrib import admin
    from django.urls import path, include
    from . import views # 예시를 위한 임포트

    urlpatterns = [
        path('admin/', admin.site.urls),
        path('', views.home, name='home'), # 프로젝트 레벨 URL (별칭: 'home')
        path('blog/', include('blog.urls')), # blog 앱의 urls.py 포함 (네임스페이스: 'blog')
        path('accounts/', include('accounts.urls')), # accounts 앱의 urls.py 포함 (네임스페이스: 'accounts')
    ]
    ```

    **앱 `urls.py` (`blog/urls.py`):**
    ```python
    # blog/urls.py
    from django.urls import path
    from . import views

    app_name = 'blog' # URL 네임스페이스 설정

    urlpatterns = [
        path('', views.post_list, name='post_list'), # 별칭: 'post_list'
        path('<int:post_id>/', views.post_detail, name='post_detail'), # 별칭: 'post_detail'
        path('<int:year>/<int:month>/', views.post_archive, name='post_archive'), # 별칭: 'post_archive'
    ]
    ```

    #### 템플릿에서 `{% url %}` 태그 사용법

    `{% url %}` 태그는 `urls.py`에 정의된 URL 패턴의 `name`을 사용하여 해당 URL을 동적으로 생성합니다. 네임스페이스가 있는 경우 `네임스페이스:별칭` 형태로 사용합니다.

    *   **인자 없는 URL:**
        ```html
        <a href="{% url 'home' %}">홈으로 가기</a> {# myproject/urls.py의 'home' 별칭 사용 #}
        <a href="{% url 'blog:post_list' %}">블로그 목록</a> {# blog/urls.py의 'post_list' 별칭 사용 #}
        ```

    *   **위치 인자 (Positional Arguments) 전달:**
        URL 패턴에 `<int:post_id>`와 같이 위치 인자가 필요한 경우, `{% url %}` 태그 뒤에 해당 인자들을 순서대로 전달합니다.
        ```html
        {# blog/urls.py의 'post_detail' 패턴: path('<int:post_id>/', ...) #}
        <a href="{% url 'blog:post_detail' post.id %}">게시글 상세 보기</a>
        ```
        여러 개의 위치 인자가 필요한 경우:
        ```html
        {# blog/urls.py의 'post_archive' 패턴: path('<int:year>/<int:month>/', ...) #}
        <a href="{% url 'blog:post_archive' 2025 8 %}">2025년 8월 게시글</a>
        ```

    *   **키워드 인자 (Keyword Arguments) 전달:**
        인자를 `키=값` 형태로 명시적으로 전달할 수 있습니다. 이는 인자의 순서가 중요하지 않거나, 가독성을 높이고 싶을 때 유용합니다.
        ```html
        {# blog/urls.py의 'post_detail' 패턴: path('<int:post_id>/', ...) #}
        <a href="{% url 'blog:post_detail' post_id=post.id %}">게시글 상세 보기 (키워드 인자)</a>
        ```
        여러 개의 키워드 인자가 필요한 경우:
        ```html
        {# blog/urls.py의 'post_archive' 패턴: path('<int:year>/<int:month>/', ...) #}
        <a href="{% url 'blog:post_archive' year=2025 month=8 %}">2025년 8월 게시글 (키워드 인자)</a>
        ```

*   **정적 파일 경로: `{% static '...' %}`**
    정적 파일(CSS, JS, 이미지 등)의 URL을 생성합니다. `settings.py`의 `STATIC_URL` 설정을 기반으로 경로를 만듭니다. 사용하기 전, 템플릿 상단에 `{% load static %}`을 선언해야 합니다.
    ```html
    {% load static %}
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
    <img src="{% static 'images/logo.png' %}" alt="로고">
    ```

##### 1.2.2.4. 보안 태그 (Security Tags)

*   **CSRF 토큰: `{% csrf_token %}`**
    POST 방식의 폼에서 CSRF(Cross-Site Request Forgery) 공격을 방지하기 위해 반드시 포함해야 하는 보안 태그입니다. Django는 이 태그를 통해 숨겨진 입력 필드에 고유한 토큰을 삽입하고, 요청이 들어올 때 이 토큰을 검증합니다.
    ```html
    <form method="post">
        {% csrf_token %}
        <input type="text" name="title">
        <button type="submit">제출</button>
    </form>
    ```

#### 1.2.3. 필터 (Filters): `|`

변수의 표시 형식을 간편하게 변경합니다. 파이프(`|`) 기호를 사용하여 변수 뒤에 필터를 적용하며, 여러 필터를 연달아 사용할 수도 있습니다.

-   **날짜/시간 형식 지정:**
    ```html
    <p>작성일: {{ post.created_at|date:"Y년 m월 d일 H시 i분" }}</p>
    {# 출력 예: 작성일: 2025년 07월 04일 14시 30분 #}
    ```

-   **텍스트 자르기:**
    ```html
    <p>{{ post.content|truncatewords:30 }}</p> {# 내용을 30단어로 자르고 ...을 붙임 #}
    <p>{{ post.content|truncatechars:50 }}</p> {# 내용을 50글자로 자르고 ...을 붙임 #}
    ```

-   **줄바꿈 처리:**
    ```html
    <p>{{ post.content|linebreaks }}</p> {# 일반 텍스트의 줄바꿈을 <p>와 <br> 태그로 변환 #}
    <p>{{ post.content|linebreaksbr }}</p> {# 줄바꿈만 <br> 태그로 변환 #}
    ```

-   **기본값 설정:**
    ```html
    {# user.profile_image_url이 없거나 False일 경우 기본 이미지 사용 #}
    <img src="{{ user.profile_image_url|default:"/static/images/default_avatar.png" }}" alt="프로필 이미지">
    ```

-   **파일 크기 형식:**
    ```html
    <p>파일 크기: {{ file.size|filesizeformat }}</p>
    {# 출력 예: 파일 크기: 1.2 MB #}
    ```

-   **HTML 안전 처리 (주의!):**
    ```html
    <p>{{ user_input|safe }}</p>
    ```
    **`safe` 필터는 변수에 포함된 HTML을 이스케이프(escape)하지 않고 그대로 렌더링합니다.** 신뢰할 수 없는 사용자 입력에 사용하면 XSS(Cross-Site Scripting) 공격에 취약해지므로, 관리자가 입력한 위지윅 에디터 콘텐츠 등 **출처가 명확하고 신뢰할 수 있는 경우에만 제한적으로 사용해야 합니다.**

### 1.3. 템플릿 상속: 코드 재사용의 핵심

웹사이트의 모든 페이지는 공통된 상단 내비게이션 바, 푸터, CSS/JS 링크 등을 가집니다. 이 모든 것을 각 HTML 파일마다 복사/붙여넣기 하는 것은 비효율적이고 유지보수를 어렵게 만듭니다. **템플릿 상속**은 이런 문제를 해결하는 Django 템플릿 시스템의 가장 중요하고 강력한 기능입니다.

-   **`{% extends "base.html" %}`**: 다른 템플릿을 상속받겠다는 선언입니다. 반드시 템플릿 파일의 가장 첫 줄에 위치해야 합니다.
-   **`{% block block_name %}` ... `{% endblock %}`**: 부모 템플릿에 "이 영역은 자식 템플릿이 채워넣을 수 있다"고 표시하는 구멍을 만드는 것과 같습니다. 자식 템플릿에서 이 블록을 재정의하여 내용을 채워 넣습니다.

**실무 예시:**

1.  **`templates/base.html` (부모 템플릿) 작성**
    ```html
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>{% block title %}My Awesome Site{% endblock %}</title>
        <link rel="stylesheet" href="{% static 'css/base.css' %}">
        {% block extra_head %}{% endblock %}
    </head>
    <body>
        <nav>
            <a href="/">홈</a>
            <a href="{% url 'blog:post_list' %}">블로그</a>
        </nav>

        <main>
            {% block content %}
            <!-- 이 블록이 각 페이지의 주된 내용으로 채워집니다. -->
            {% endblock %}
        </main>

        <footer>
            &copy; 2025 My Awesome Site.
        </footer>
        <script src="{% static 'js/base.js' %}"></script>
        {% block extra_script %}{% endblock %}
    </body>
    </html>
    ```

2.  **`blog/templates/blog/post_list.html` (자식 템플릿) 작성**
    ```html
    {% extends "base.html" %}
    {% load static %}

    {% block title %}블로그 목록 - {{ block.super }}{% endblock %}

    {% block extra_head %}
        <link rel="stylesheet" href="{% static 'css/blog.css' %}">
    {% endblock %}

    {% block content %}
        <h1>블로그 목록</h1>
        <ul>
            {% for post in post_list %}
                <li><a href="{{ post.get_absolute_url }}">{{ post.title }}</a></li>
            {% empty %}
                <li>아직 게시물이 없습니다.</li>
            {% endfor %}
        </ul>
    {% endblock %}
    ```
    - `{{ block.super }}`: 부모 템플릿의 블록에 정의된 내용을 그대로 가져와서 이어붙일 때 사용합니다.

### 1.4. 커스텀 템플릿 태그와 필터 만들기

내장 기능만으로 부족할 때, 직접 파이썬 코드로 커스텀 필터나 태그를 만들어 템플릿의 기능을 확장할 수 있습니다.

1.  앱 내부에 `templatetags` 디렉토리를 생성하고, `__init__.py` 파일(빈 파일)을 만듭니다.
2.  `templatetags` 디렉토리 안에 태그 파일을 만듭니다. (예: `my_app_tags.py`)
3.  해당 파일에 필터나 태그를 정의합니다.

**커스텀 필터 예시: 숫자를 원화(₩) 형식으로 변환**
```python
# my_app/templatetags/my_app_tags.py
from django import template

register = template.Library()

@register.filter
def currency(value):
    try:
        return f"₩{int(value):,}"
    except (ValueError, TypeError):
        return value
```

**템플릿에서 사용법:**
```html
{# 템플릿 상단에 태그 파일을 로드 #}
{% load my_app_tags %}

<p>상품 가격: {{ product.price|currency }}</p>
{# 출력 예: 상품 가격: ₩15,000 #}
```

## 2. Django Form (폼 처리): 실무 가이드

Django의 폼 시스템은 단순히 HTML `<form>` 태그를 만드는 것을 넘어, 웹 애플리케이션의 핵심적인 처리 과정 세 가지를 책임지는 강력한 도구입니다.

1.  **데이터 렌더링**: 데이터를 HTML 폼 위젯으로 변환하여 사용자에게 보여줍니다.
2.  **유효성 검사 (Validation)**: 사용자가 제출한 데이터가 우리가 정한 규칙(예: 이메일 형식, 글자 수 제한)에 맞는지 검증합니다.
3.  **데이터 처리**: 유효성 검사를 통과한 데이터를 파이썬 자료형으로 변환(cleaning)하고, 데이터베이스에 저장하거나 다른 비즈니스 로직에 사용합니다.

### 2.1. HTTP 메서드 (GET vs POST)와 폼 처리

웹에서 클라이언트(브라우저)가 서버에 요청을 보낼 때 사용하는 가장 기본적인 HTTP 메서드는 GET과 POST입니다. 폼 데이터를 처리할 때 이 두 메서드의 특성을 이해하는 것은 매우 중요합니다.

#### 2.1.1 GET 메서드

GET 메서드는 주로 **서버로부터 데이터를 조회하거나 가져올 때** 사용됩니다.

*   **특징**:
    *   **데이터 전송 방식**: 폼 데이터가 URL의 쿼리 스트링(`?key1=value1&key2=value2` 형태로)에 추가되어 전송됩니다.
    *   **가시성**: URL에 데이터가 노출되므로, 민감한 정보(비밀번호 등)를 전송하는 데 부적합합니다.
    *   **캐싱**: 브라우저나 프록시 서버에 의해 캐싱될 수 있습니다.
    *   **북마크/공유**: URL에 데이터가 포함되므로 북마크하거나 공유하기 용이합니다.
    *   **멱등성(Idempotence)**: 같은 요청을 여러 번 보내도 서버의 상태를 변경하지 않거나, 변경하더라도 결과가 동일합니다. (예: 게시글 조회)

*   **주요 사용 사례**:
    *   검색 폼 (예: `google.com/search?q=django`)
    *   필터링 및 정렬 (예: `example.com/products?category=electronics&sort=price`)
    *   단순한 페이지 이동이나 데이터 조회.

#### 2.1.2 POST 메서드

POST 메서드는 주로 **서버에 데이터를 제출하여 생성, 수정, 삭제와 같이 서버의 상태를 변경할 때** 사용됩니다.

*   **특징**:
    *   **데이터 전송 방식**: 폼 데이터가 HTTP 요청 본문(body)에 포함되어 전송됩니다.
    *   **가시성**: URL에 데이터가 노출되지 않으므로, 민감한 정보(비밀번호, 개인 정보 등)를 전송하는 데 적합합니다.
    *   **캐싱**: 일반적으로 캐싱되지 않습니다.
    *   **북마크/공유**: URL에 데이터가 포함되지 않으므로 북마크하거나 공유하기 어렵습니다.
    *   **비멱등성(Non-Idempotence)**: 같은 요청을 여러 번 보내면 서버의 상태가 여러 번 변경될 수 있습니다. (예: 게시글 생성 버튼을 여러 번 누르면 게시글이 여러 개 생성될 수 있음)

*   **주요 사용 사례**:
    *   회원가입, 로그인 폼
    *   게시글 작성, 수정, 삭제
    *   파일 업로드
    *   주문 처리, 결제 등 서버의 상태를 변경하는 모든 작업.

#### 2.1.3 Django에서의 폼 처리와 GET/POST

Django 뷰에서는 `request.method` 속성을 사용하여 요청이 GET인지 POST인지 구분하고, 이에 따라 다른 로직을 수행하는 것이 일반적인 패턴입니다.

*   **GET 요청**: 폼을 처음 로드할 때 사용됩니다. 뷰에서는 비어있는 폼 인스턴스를 생성하여 템플릿으로 전달합니다.
    ```python
    # views.py
    if request.method == 'GET':
        form = MyForm() # 비어있는 폼 생성
    ```
*   **POST 요청**: 사용자가 폼을 작성하고 제출할 때 사용됩니다. 뷰에서는 `request.POST` (POST 데이터)와 `request.FILES` (파일 데이터)를 사용하여 폼 인스턴스를 생성하고, 유효성 검사를 수행한 후 데이터를 처리합니다.
    ```python
    # views.py
    if request.method == 'POST':
        form = MyForm(request.POST, request.FILES) # 제출된 데이터로 폼 생성
        if form.is_valid():
            # 데이터 처리 로직
            pass
    ```

이러한 GET과 POST의 특성을 이해하고 적절히 활용하는 것은 웹 애플리케이션의 보안, 성능, 그리고 사용자 경험을 최적화하는 데 필수적입니다.

### 2.2. Form vs ModelForm: 언제 무엇을 쓸까?

- **`forms.Form`**: 데이터베이스 모델과 직접적인 관련이 없는 폼을 만들 때 사용합니다. 예를 들어, 검색 폼, 문의하기 폼, 로그인 폼 등이 여기에 해당합니다. 필드를 하나하나 직접 정의해야 합니다.

- **`forms.ModelForm`**: **(가장 흔하게 사용)** 특정 모델(Model)과 직접적으로 연결된 폼을 만들 때 사용합니다. 모델에 정의된 필드를 기반으로 폼 필드를 자동으로 생성해주므로, 코드 중복을 크게 줄일 수 있습니다. 게시글 작성, 회원 정보 수정 등 대부분의 CRUD 기능에 사용됩니다.

### 2.3. ModelForm: 모델과 폼의 완벽한 결합

`ModelForm`을 사용하면 모델의 정보를 바탕으로 폼을 손쉽게 생성할 수 있습니다. 내부 `Meta` 클래스에 정보를 기술합니다.

**`forms.py` 예시:**
```python
from django import forms
from .models import Post

class PostForm(forms.ModelForm):
    class Meta:
        model = Post  # 이 폼이 어떤 모델과 연결되는지 지정
        fields = ['title', 'content', 'is_published'] # 폼에 표시할 필드 목록
        # exclude = ['author'] # 또는 제외할 필드 목록을 지정할 수도 있음

        # 각 필드의 라벨, 도움말, 에러 메시지 등을 세밀하게 제어
        labels = {
            'title': '게시글 제목',
            'content': '내용',
            'is_published': '공개 발행 여부',
        }
        help_texts = {
            'title': '255자 이내로 작성해주세요.',
        }
        error_messages = {
            'title': {
                'max_length': "제목이 너무 깁니다. 255자 이하로 작성해주세요.",
            },
        }
```

### 2.4. 뷰에서의 폼 처리: 표준 워크플로우

뷰에서 폼을 처리하는 로직은 매우 정형화된 패턴을 따릅니다. 이 패턴을 이해하는 것이 중요합니다.

**`views.py` - 게시글 생성(Create) 뷰 예시:**
```python
from django.shortcuts import render, redirect, get_object_or_404
from .forms import PostForm
from .models import Post

def post_create(request):
    # 1. POST 요청인 경우 (폼 데이터가 제출되었을 때)
    if request.method == 'POST':
        # 2. 제출된 데이터로 폼 인스턴스 생성
        form = PostForm(request.POST)
        # 3. 유효성 검사
        if form.is_valid():
            # 4. commit=False: DB 저장을 잠시 미루고 모델 인스턴스만 가져옴
            post = form.save(commit=False)
            # 5. 추가적인 데이터(작성자 등)를 할당
            post.author = request.user
            # 6. 최종적으로 DB에 저장
            post.save()
            # 7. 성공 후에는 다른 페이지로 리다이렉트 (새로고침 시 폼 중복 제출 방지)
            return redirect('blog:post_detail', pk=post.pk)
    # 8. GET 요청인 경우 (페이지에 처음 접속했을 때)
    else:
        # 9. 비어있는 폼 인스턴스 생성
        form = PostForm()
    
    # 10. 템플릿 렌더링 (GET 요청이거나, POST 요청에서 유효성 검사 실패 시)
    return render(request, 'blog/post_form.html', {'form': form})
```

### 2.5. 폼 렌더링 커스터마이징

`{{ form.as_p }}`는 편리하지만 디자인에 제약이 많습니다. Bootstrap 같은 CSS 프레임워크를 사용하려면 필드를 수동으로 렌더링해야 합니다.

- **수동 렌더링**: 템플릿에서 `form` 객체를 순회하며 각 `field` 객체의 속성들을 직접 사용합니다.
    - `{{ field.label_tag }}`: `<label>` 태그를 생성합니다.
    - `{{ field }}`: `<input>`, `<textarea>` 등 위젯 자체를 렌더링합니다.
    - `{{ field.help_text }}`: 도움말 텍스트를 표시합니다.
    - `{{ field.errors }}`: 해당 필드의 유효성 검사 에러 목록을 렌더링합니다.
    - `{{ form.non_field_errors }}`: 특정 필드가 아닌, 폼 전체에 대한 에러를 표시합니다.

**`post_form.html` 템플릿 예시 (Bootstrap 적용):**
```html
<form method="post" novalidate>
    {% csrf_token %}

    {% if form.non_field_errors %}
        <div class="alert alert-danger">
            {% for error in form.non_field_errors %}
                <p>{{ error }}</p>
            {% endfor %}
        </div>
    {% endif %}

    {% for field in form %}
        <div class="form-group mb-3">
            {{ field.label_tag }}
            {{ field }}
            {% if field.help_text %}
                <small class="form-text text-muted">{{ field.help_text }}</small>
            {% endif %}
            {% for error in field.errors %}
                <div class="invalid-feedback d-block">{{ error }}</div>
            {% endfor %}
        </div>
    {% endfor %}

    <button type="submit" class="btn btn-primary">저장하기</button>
</form>
```

### 2.6. 폼 유효성 검사(Validation) 심화

- **`clean_<fieldname>()`**: **특정 필드 하나**에 대한 유효성 검사 로직을 추가하고 싶을 때 사용합니다.
    ```python
    class PostForm(forms.ModelForm):
        # ...
        def clean_title(self):
            title = self.cleaned_data.get('title')
            if "비속어" in title:
                raise forms.ValidationError("제목에 비속어를 사용할 수 없습니다.")
            return title
    ```

- **`clean()`**: **두 개 이상의 필드에 걸친** 복잡한 유효성 검사가 필요할 때 사용합니다.
    ```python
    class EventForm(forms.Form):
        start_date = forms.DateField()
        end_date = forms.DateField()

        def clean(self):
            cleaned_data = super().clean() # 반드시 부모 클래스의 clean()을 먼저 호출
            start_date = cleaned_data.get("start_date")
            end_date = cleaned_data.get("end_date")

            if start_date and end_date and start_date > end_date:
                raise forms.ValidationError("종료일은 시작일보다 빠를 수 없습니다.")
            
            return cleaned_data # 반드시 정제된 데이터 전체를 반환
    ```

### 2.7. 위젯 커스터마이징

`Meta` 클래스의 `widgets` 속성을 사용하면 각 필드의 기본 HTML 위젯을 변경하거나 `class`, `placeholder` 같은 속성을 추가할 수 있습니다.

```python
class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'content']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control form-control-lg',
                'placeholder': '제목을 입력하세요'
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 10
            }),
        }
```
