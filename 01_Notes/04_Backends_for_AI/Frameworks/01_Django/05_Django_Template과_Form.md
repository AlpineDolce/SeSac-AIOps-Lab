<h2>Django Backend: Template과 Form</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django의 Template 시스템과 Form 처리 방식을 학습하는 것을 목표로 합니다. 템플릿을 사용하여 동적인 웹 페이지를 생성하는 방법과, ModelForm을 활용하여 사용자 입력을 효율적으로 처리하고 데이터베이스와 상호작용하는 방법을 이해합니다.</p>

<h2>목차</h2> 

- [1. Django Template (템플릿 시스템): 실무 가이드](#1-django-template-템플릿-시스템-실무-가이드)
- [1.1. 템플릿의 기본 구성 요소: 변수, 태그, 필터](#11-템플릿의-기본-구성-요소-변수-태그-필터)
- [1.2. 템플릿 상속: 코드 재사용의 핵심](#12-템플릿-상속-코드-재사용의-핵심)
- [1.3. 강력한 내장 태그 활용](#13-강력한-내장-태그-활용)
- [1.4. 커스텀 템플릿 태그와 필터 만들기](#14-커스텀-템플릿-태그와-필터-만들기)
- [2. Django Form (폼 처리): 실무 가이드](#2-django-form-폼-처리-실무-가이드)
  - [2.1. Form vs ModelForm: 언제 무엇을 쓸까?](#21-form-vs-modelform-언제-무엇을-쓸까)
  - [2.2. ModelForm: 모델과 폼의 완벽한 결합](#22-modelform-모델과-폼의-완벽한-결합)
  - [2.3. 뷰에서의 폼 처리: 표준 워크플로우](#23-뷰에서의-폼-처리-표준-워크플로우)
  - [2.4. 폼 렌더링 커스터마이징](#24-폼-렌더링-커스터마이징)
  - [2.5. 폼 유효성 검사(Validation) 심화](#25-폼-유효성-검사validation-심화)
  - [2.6. 위젯 커스터마이징](#26-위젯-커스터마이징)

---

## 1. Django Template (템플릿 시스템): 실무 가이드

Django 템플릿 시스템의 핵심 철학은 **표현과 로직의 분리**입니다. 템플릿은 최종적으로 사용자에게 보여질 모습(HTML 구조, CSS 등)에 집중하고, 복잡한 비즈니스 로직은 뷰(View)에서 처리하여 서로의 역할과 책임을 명확히 구분합니다. 이를 통해 디자이너와 개발자의 협업이 원활해지고 코드의 유지보수성이 향상됩니다.

## 1.1. 템플릿의 기본 구성 요소: 변수, 태그, 필터

- **변수 `{{ ... }}`**: 뷰에서 전달된 데이터를 템플릿에 표시합니다. 점(.)을 사용하여 객체의 속성, 딕셔너리의 키, 리스트의 인덱스, 심지어는 모델 객체의 메서드에도 접근할 수 있습니다.
    - `{{ post.title }}`: `post` 객체의 `title` 속성
    - `{{ my_dict.key }}`: `my_dict` 딕셔너리의 `key`에 해당하는 값
    - `{{ my_list.0 }}`: `my_list` 리스트의 첫 번째 항목

- **태그 `{% ... %}`**: 템플릿의 제어 흐름을 담당합니다. 반복, 조건 분기, 상속 등 다양한 기능을 수행합니다.
    - `{% for post in post_list %}` ... `{% endfor %}`: 리스트를 순회합니다. 항목이 없을 때를 대비한 `{% empty %}` 태그와 함께 사용하는 것이 좋습니다.
    - `{% if user.is_authenticated %}` ... `{% elif ... %}` ... `{% else %}` ... `{% endif %}`: 조건에 따라 다른 내용을 보여줍니다.
    - `{% with total=items.count %}`: 복잡하거나 반복적인 연산 결과를 변수에 할당하여 가독성을 높이고 성능을 개선합니다. `{{ total }}`로 사용할 수 있습니다.

- **필터 `|`**: 변수의 표시 형식을 간편하게 변경합니다. 여러 필터를 연달아 사용할 수도 있습니다.
    - `{{ post.created_at|date:"Y년 m월 d일" }}`: 날짜 형식을 지정합니다.
    - `{{ post.content|truncatewords:30 }}`: 내용을 30단어로 자르고 `...`을 붙입니다.
    - `{{ post.content|linebreaks }}`: 일반 텍스트의 줄바꿈을 `<p>`와 `<br>` 태그로 변환합니다.
    - `{{ user.profile_image_url|default:"/static/images/default_avatar.png" }}`: 변수 값이 없거나 `False`일 경우 지정된 기본값을 사용합니다.
    - `{{ value|filesizeformat }}`: 파일 크기를 사람이 읽기 좋은 형태(KB, MB 등)로 변환합니다.
    - `{{ user_input|safe }}`: **(주의!)** 변수에 포함된 HTML을 이스케이프(escape)하지 않고 그대로 렌더링합니다. 신뢰할 수 없는 사용자 입력에 사용하면 XSS 공격에 취약해지므로, 관리자가 입력한 위지윅 에디터 콘텐츠 등 **출처가 명확하고 신뢰할 수 있는 경우에만 제한적으로 사용해야 합니다.**

## 1.2. 템플릿 상속: 코드 재사용의 핵심

웹사이트의 모든 페이지는 공통된 상단 내비게이션 바, 푸터, CSS/JS 링크 등을 가집니다. 이 모든 것을 각 HTML 파일마다 복사/붙여넣기 하는 것은 비효율적이고 유지보수를 어렵게 만듭니다. **템플릿 상속**은 이런 문제를 해결하는 Django 템플릿 시스템의 가장 중요하고 강력한 기능입니다.

- **`{% extends "base.html" %}`**: 다른 템플릿을 상속받겠다는 선언입니다. 반드시 템플릿 파일의 가장 첫 줄에 위치해야 합니다.
- **`{% block block_name %}` ... `{% endblock %}`**: 부모 템플릿에 "이 영역은 자식 템플릿이 채워넣을 수 있다"고 표시하는 구멍을 만드는 것과 같습니다. 자식 템플릿에서 이 블록을 재정의하여 내용을 채워 넣습니다.

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

## 1.3. 강력한 내장 태그 활용

- **`{% url '...' %}`**: URL 하드코딩을 방지하기 위해 **반드시 사용해야 합니다.** `urls.py`에 정의된 URL `name`을 사용하여 동적으로 경로를 생성합니다.
    - `{% url 'blog:post_detail' post.id %}`

- **`{% static '...' %}`**: 정적 파일(CSS, JS, 이미지 등)의 URL을 생성합니다. `settings.py`의 `STATIC_URL` 설정을 기반으로 경로를 만듭니다. 사용하기 전, 템플릿 상단에 `{% load static %}`을 선언해야 합니다.
    - `<link rel="stylesheet" href="{% static 'css/style.css' %}">`

- **`{% csrf_token %}`**: POST 방식의 폼에서 CSRF(Cross-Site Request Forgery) 공격을 방지하기 위해 반드시 포함해야 하는 보안 태그입니다.

## 1.4. 커스텀 템플릿 태그와 필터 만들기

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

### 2.1. Form vs ModelForm: 언제 무엇을 쓸까?

- **`forms.Form`**: 데이터베이스 모델과 직접적인 관련이 없는 폼을 만들 때 사용합니다. 예를 들어, 검색 폼, 문의하기 폼, 로그인 폼 등이 여기에 해당합니다. 필드를 하나하나 직접 정의해야 합니다.

- **`forms.ModelForm`**: **(가장 흔하게 사용)** 특정 모델(Model)과 직접적으로 연결된 폼을 만들 때 사용합니다. 모델에 정의된 필드를 기반으로 폼 필드를 자동으로 생성해주므로, 코드 중복을 크게 줄일 수 있습니다. 게시글 작성, 회원 정보 수정 등 대부분의 CRUD 기능에 사용됩니다.

### 2.2. ModelForm: 모델과 폼의 완벽한 결합

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

### 2.3. 뷰에서의 폼 처리: 표준 워크플로우

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

### 2.4. 폼 렌더링 커스터마이징

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

### 2.5. 폼 유효성 검사(Validation) 심화

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

### 2.6. 위젯 커스터마이징

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
