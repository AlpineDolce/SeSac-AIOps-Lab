<h2>Django Backend: Template과 Form</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django의 Template 시스템과 Form 처리 방식을 학습하는 것을 목표로 합니다. 템플릿을 사용하여 동적인 웹 페이지를 생성하는 방법과, ModelForm을 활용하여 사용자 입력을 효율적으로 처리하고 데이터베이스와 상호작용하는 방법을 이해합니다.</p>

<h2>목차</h2> 

- [1. Django Template (템플릿 시스템)](#1-django-template-템플릿-시스템)
  - [1.1. 템플릿 변수 및 필터](#11-템플릿-변수-및-필터)
  - [1.2. 템플릿 태그 (for, if)](#12-템플릿-태그-for-if)
  - [1.3. URL 태그 (url)](#13-url-태그-url)
  - [1.4. CSRF 토큰 (csrf_token)](#14-csrf-토큰-csrf_token)
- [2. Django Form (폼 처리)](#2-django-form-폼-처리)
  - [2.1. ModelForm 사용](#21-modelform-사용)
  - [2.2. 폼 필드 렌더링 (as_p)](#22-폼-필드-렌더링-as_p)
  - [2.3. 폼 데이터 저장 (form.save())](#23-폼-데이터-저장-formsave)
  - [2.4. 폼(Form) 커스터마이징 및 고급 유효성 검사](#24-폼form-커스터마이징-및-고급-유효성-검사)
    - [2.4.1. 필드 위젯 커스터마이징](#241-필드-위젯-커스터마이징)
    - [2.4.2. 고급 유효성 검사 (`clean()` 메서드)](#242-고급-유효성-검사-clean-메서드)

---

## 1. Django Template (템플릿 시스템)

Django 템플릿 시스템은 HTML 코드에 파이썬 데이터를 삽입하여 동적인 웹 페이지를 생성할 수 있게 해줍니다. 템플릿은 로직과 표현을 분리하여 코드의 가독성과 유지보수성을 높입니다.

### 1.1. 템플릿 변수 및 필터

*   **변수**: 뷰에서 템플릿으로 전달된 데이터를 표시할 때 사용합니다. `{{ variable_name }}` 형식으로 사용합니다.
    *   **예시 (`myhome1/templates/blog/blog_list.html`):
        ```html
        <ul>
        {% for item in blogList %}
            <li>{{item.title}}</li>
        {%endfor%}    
        </ul>
        ```
    *   `blogList`는 뷰에서 전달된 리스트이며, `item.title`은 리스트 내 각 객체의 `title` 속성에 접근합니다.

*   **필터**: 변수의 표시 형식을 변경할 때 사용합니다. `{{ variable|filter_name:"argument" }}` 형식으로 사용합니다.
    *   **예시 (`mysite1/templates/score/score_list.html`):
        ```html
        <td>{{ score.wdate|date:"Y-m-d" }}</td> {# 예시: 날짜 포맷 #}
        ```
    *   `date:"Y-m-d"` 필터는 `score.wdate` (날짜/시간 객체)를 'YYYY-MM-DD' 형식의 문자열로 변환합니다.

### 1.2. 템플릿 태그 (for, if)

템플릿 태그는 템플릿 내에서 제어 흐름(반복, 조건 등)을 구현할 때 사용합니다. `{% tag_name %}` 형식으로 사용합니다.

*   **`{% for ... in ... %}`**: 리스트나 쿼리셋과 같은 반복 가능한 객체를 순회하며 내용을 반복 출력합니다.
    *   **예시 (`myhome1/templates/blog/blog_list.html`):
        ```html
        <ul>
        {% for item in blogList %}
            <li>{{item.title}}</li>
        {%endfor%}    
        </ul>
        ```
    *   `{% empty %}` 태그를 사용하여 반복할 항목이 없을 때 표시할 내용을 정의할 수 있습니다.
        *   **예시 (`mysite1/templates/score/score_list.html`):
            ```html
            {% for score in page_obj.object_list %}
                {# ... #}
            {% empty %} {# 쿼리셋이 비어있을 경우 #}
                <tr>
                    <td colspan="5">데이터가 없습니다.</td>
                </tr>
            {% endfor %}
            ```
    *   **예시 (`장고사이트구축방법.txt` - `qna_list1.html`):
        ```html
        <h2>Question List</h2>
        <ul>
          {% for question in questions %}
            <li>
              <strong>{{ question.subject }}</strong><br>
              {{ question.content }}<br>
              작성일: {{ question.create_date }}
            </li>
          {% endfor %}
        </ul>

        <h2>Answer List</h2>
        <ul>
          {% for answer in answers %}
            <li>
              <strong>질문: {{ answer.question.subject }}</strong><br>
              답변: {{ answer.content }}<br>
              작성일: {{ answer.create_date }}
            </li>
          {% endfor %}
        </ul>
        ```
    *   **예시 (`장고사이트구축방법.txt` - `qna_list2.html`):
        ```html
        <h2>질문과 답변 목록</h2>
        <ul>
          {% for q in questions %}
            <li>
              <strong>Q: {{ q.subject }}</strong><br>
              {{ q.content }}<br>
              작성일: {{ q.create_date }}

              <ul>
                {% for a in q.answer_set.all %}
                  <li>A: {{ a.content }} ({{ a.create_date }})</li>
                {% empty %}
                  <li>답변 없음</li>
                {% endfor %}
              </ul>
            </li>
          {% endfor %}
        </ul>
        ```

*   **`{% if ... %}`**: 조건에 따라 내용을 표시하거나 숨깁니다. `{% elif %}`, `{% else %}`와 함께 사용할 수 있습니다.
    *   **예시 (`mysite1/templates/score/score_list.html` - 페이지네이션):
        ```html
        {% if page_obj.has_previous %}
            <a href="?page={{ page_obj.previous_page_number }}">이전</a>
        {% else %}
            <span class="disabled">이전</span>
        {% endif %}
        ```

### 1.3. URL 태그 (url)

`{% url 'namespace:name' arg1 arg2 ... %}` 태그는 `urls.py`에 정의된 URL 패턴의 `name`을 사용하여 동적으로 URL을 생성합니다. 하드코딩된 URL 대신 이 태그를 사용하면 URL 구조가 변경되어도 템플릿을 수정할 필요가 없어 유지보수성이 높아집니다.

*   **예시 (`myhome1/templates/blog/blog_list.html`):
    ```html
    <li><a href="{%url 'blog:view' item.id%}">{{item.title}}</a></li>
    ```
    *   `blog:view`는 `blog` 앱의 `view`라는 이름의 URL 패턴을 참조하며, `item.id`는 해당 URL 패턴에 필요한 인자(`id`)로 전달됩니다.

### 1.4. CSRF 토큰 (csrf_token)

Django는 CSRF(Cross-Site Request Forgery) 공격을 방지하기 위해 `{% csrf_token %}` 템플릿 태그를 제공합니다. 이 태그는 폼 내부에 숨겨진 입력 필드를 생성하여, POST 요청 시 유효한 토큰이 함께 전송되도록 합니다. 모든 POST 폼에는 이 태그를 포함하는 것이 좋습니다.

*   **예시 (`myhome1/templates/blog/blog_write.html`):
    ```html
    <form name="form" action="/blog/save" method="post">
          {%csrf_token%}
          <!-- ... 폼 필드 ... -->
          <button>등록</button>
    </form>
    ```

## 2. Django Form (폼 처리)

Django의 폼 시스템은 HTML 폼을 생성하고, 사용자 입력을 유효성 검사하며, 데이터베이스에 저장하는 과정을 효율적으로 처리할 수 있도록 돕습니다.

### 2.1. ModelForm 사용

`ModelForm`은 모델(Model)을 기반으로 폼을 자동으로 생성해주는 클래스입니다. 모델의 필드와 동일한 필드를 가진 폼을 쉽게 만들 수 있습니다.

*   **`Meta` 클래스**: `ModelForm` 내부에 정의되며, 어떤 모델과 연결할지(`model`), 어떤 필드를 폼에 포함할지(`fields`), 필드의 라벨을 어떻게 표시할지(`labels`) 등을 설정합니다.
    *   **예시 (`myhome1/blog/forms.py`):
        ```python
        from django import forms 
        from blog.models import Blog 

        class BlogForms(forms.ModelForm):
            class Meta:
                model = Blog 
                fields = ['title', 'writer', 'contents']
                labels ={
                    'title':"제목",
                    'writer':"작성자",
                    "contents":"내용"
                }
        ```
    *   **예시 (`mysite1/score/forms.py`):
        ```python
        from django import forms 
        from .models import Score 

        class ScoreForm(forms.ModelForm):
            class Meta:
                model = Score 
                fields =['name', 'kor', 'eng', 'mat']
                labels = {
                    'name':"이름",
                    'kor':'국어',
                    'eng':'영어',
                    'mat':'수학',
                }
        ```

### 2.2. 폼 필드 렌더링 (as_p)

템플릿에서 `{{ form.as_p }}`와 같이 사용하면, `ModelForm`에 정의된 모든 필드를 `<p>` 태그로 감싸서 자동으로 HTML 폼 필드로 렌더링해줍니다. 이는 빠른 개발에 유용합니다.

*   **예시 (`mysite1/templates/score/score_write.html`):
    ```html
    <form name="myform" id="myform" >
        {%csrf_token%}
        {{form.as_p}}
    </form>
    ```

### 2.3. 폼 데이터 저장 (form.save())

`ModelForm` 객체의 `save()` 메서드를 호출하면, 폼에서 유효성 검사를 통과한 데이터를 연결된 모델 인스턴스에 저장하고 데이터베이스에 반영합니다.

*   **`form.save(commit=False)`**: 데이터베이스에 즉시 저장하지 않고, 모델 인스턴스만 반환합니다. 이를 통해 추가적인 로직(예: `wdate`, `hit`, `total`, `avg` 계산)을 수행한 후 수동으로 `save()`를 호출할 수 있습니다.
    *   **예시 (`myhome1/blog/views.py` - `save` 함수):
        ```python
        from .forms import BlogForms 
        from django.utils import timezone 

        def save(request): 
            form = BlogForms(request.POST)
            blog = form.save(commit=False) # 아직 DB에 저장하지 않고 객체만 반환
            blog.wdate = timezone.now() 
            blog.hit=0 
            blog.save() # 최종 저장
            return redirect("blog:list")
        ```
    *   **예시 (`mysite1/score/views.py` - `save` 함수):
        ```python
        from .forms import ScoreForm
        from django.utils import timezone

        def save(request):
            if request.method =="POST":
                scoreform = ScoreForm(request.POST)
                scoreModel = scoreform.save(commit=False)
                scoreModel.total = scoreModel.kor + scoreModel.eng + scoreModel.mat
                scoreModel.avg = scoreModel.total / 3 
                scoreModel.wdate = timezone.now() 
                scoreModel.save() 
            return redirect("score:score_list")
        ```

    ```python
    # myhome1/blog/views.py (save 함수 예시 보강)
    from .forms import BlogForms
    from django.utils import timezone
    from django.shortcuts import render, redirect # render 임포트 추가

    def save(request):
        if request.method == "POST":
            form = BlogForms(request.POST)
            if form.is_valid(): # 폼 유효성 검사 추가
                blog = form.save(commit=False) # 아직 DB에 저장하지 않고 객체만 반환
                blog.wdate = timezone.now()
                blog.hit = 0
                blog.save() # 최종 저장
                return redirect("blog:list")
            else:
                # 폼 유효성 검사 실패 시, 에러 메시지와 함께 폼 다시 렌더링
                return render(request, "blog/blog_write.html", {"form": form})
        else:
            # GET 요청 시 빈 폼을 렌더링
            form = BlogForms()
            return render(request, "blog/blog_write.html", {"form": form})
    ```
    *   `mysite1/score/views.py`의 `save` 함수도 유사하게 `else` (GET 요청) 처리와 `form.is_valid()` 검사를 명시적으로 추가하여 폼 처리의 일반적인 패턴을 보여주는 것이 좋습니다.


### 2.4. 폼(Form) 커스터마이징 및 고급 유효성 검사

Django의 폼 시스템은 강력하지만, 때로는 기본 제공 기능만으로는 부족할 수 있습니다. 필드의 HTML 속성을 변경하거나, 여러 필드에 걸친 복잡한 유효성 검사를 수행해야 할 때가 있습니다.

#### 2.4.1. 필드 위젯 커스터마이징

폼 필드의 HTML 위젯(예: `<input type="text">`, `<textarea>`)을 변경하거나, CSS 클래스, 플레이스홀더 등의 HTML 속성을 추가할 수 있습니다.

**예시 (`forms.py`):
```python
from django import forms
from .models import MyModel

class MyModelForm(forms.ModelForm):
    class Meta:
        model = MyModel
        fields = ['title', 'content', 'publish_date']
        widgets = {
            'title': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '제목을 입력하세요'}),
            'content': forms.Textarea(attrs={'rows': 5, 'class': 'form-control'}),
            'publish_date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        }
        labels = {
            'title': '게시글 제목',
            'content': '내용',
            'publish_date': '발행일',
        }
```

#### 2.4.2. 고급 유효성 검사 (`clean()` 메서드)

단일 필드의 유효성 검사는 필드 타입이나 `validators` 옵션으로 처리할 수 있지만, 여러 필드의 값을 조합하여 유효성을 검사해야 할 때는 폼 클래스의 `clean()` 메서드를 오버라이드합니다.

**예시 (`forms.py`):
```python
from django import forms
from django.core.exceptions import ValidationError

class EventForm(forms.Form):
    start_date = forms.DateField(label='시작일')
    end_date = forms.DateField(label='종료일')
    title = forms.CharField(max_length=100, label='이벤트 제목')

    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get('start_date')
        end_date = cleaned_data.get('end_date')

        if start_date and end_date:
            if start_date > end_date:
                # 특정 필드에 에러 추가
                self.add_error('end_date', '종료일은 시작일보다 빠를 수 없습니다.')
                # 또는 폼 전체에 에러 추가
                # raise ValidationError('종료일은 시작일보다 빠를 수 없습니다.')
        return cleaned_data
```
