<h2>Django Backend: View와 URL 심화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django의 View 계층을 깊이 있게 다룹니다. 함수 기반 뷰(FBV)와 클래스 기반 뷰(CBV)의 작성법, HTTP 요청 및 응답 처리, 페이지네이션, 에러 핸들링 등 View와 관련된 핵심 개념들을 학습하여 동적인 웹 페이지 로직을 효과적으로 구현하는 것을 목표로 합니다.</p>

<h2>목차</h2> 

- [1. 함수 기반 뷰 (Function-Based Views, FBV)](#1-함수-기반-뷰-function-based-views-fbv)
  - [1.1. HTTP 응답: 실무 가이드](#11-http-응답-실무-가이드)
    - [1.1.1. **`HttpResponse` - 기본 응답 객체**](#111-httpresponse---기본-응답-객체)
    - [1.1.2. **`JsonResponse` - API를 위한 핵심 응답**](#112-jsonresponse---api를-위한-핵심-응답)
    - [1.1.3. **기타 유용한 `HttpResponse` 자식 클래스**](#113-기타-유용한-httpresponse-자식-클래스)
  - [1.2. 템플릿 렌더링 및 리다이렉트 (render, redirect)](#12-템플릿-렌더링-및-리다이렉트-render-redirect)
  - [1.3. 요청 객체 (HttpRequest) 다루기](#13-요청-객체-httprequest-다루기)
  - [1.4. 페이지네이션 (Paginator)](#14-페이지네이션-paginator)
- [2. 클래스 기반 뷰 (Class-Based Views, CBV): 실무 심화](#2-클래스-기반-뷰-class-based-views-cbv-실무-심화)
  - [2.1. CBV: 왜 클래스를 사용하는가?](#21-cbv-왜-클래스를-사용하는가)
  - [2.2. CBV의 동작 원리: `as_view()` 와 `dispatch()`](#22-cbv의-동작-원리-as_view-와-dispatch)
  - [2.3. 제네릭 뷰 상세 가이드: "Don't Repeat Yourself"](#23-제네릭-뷰-상세-가이드-dont-repeat-yourself)
  - [2.4. CBV 커스터마이징: 실무 핵심 기술](#24-cbv-커스터마이징-실무-핵심-기술)
- [3. 에러 핸들링 및 사용자 친화적 페이지: 실무 가이드](#3-에러-핸들링-및-사용자-친화적-페이지-실무-가이드)
  - [3.1. 사용자 친화적 에러 페이지 (4xx/5xx)](#31-사용자-친화적-에러-페이지-4xx5xx)
  - [3.2. API 예외 처리: 일관된 에러 응답 구조화](#32-api-예외-처리-일관된-에러-응답-구조화)

---



## 1. 함수 기반 뷰 (Function-Based Views, FBV)

뷰(View)는 웹 요청을 수신하고 HTTP 응답을 반환하는 파이썬 함수 또는 클래스입니다. 주로 모델에서 데이터를 가져오거나, 템플릿을 렌더링하거나, 다른 HTTP 응답을 생성하는 로직을 포함합니다.

함수 기반 뷰는 파이썬 함수로 작성된 뷰입니다. 간단한 로직을 처리하거나 특정 HTTP 메서드에 대한 응답을 명확하게 분리할 때 유용합니다.

### 1.1. HTTP 응답: 실무 가이드

모든 Django 뷰의 최종 목적은 `HttpRequest` 객체를 받아 `HttpResponse` 객체 또는 그 자식 클래스의 인스턴스를 반환하는 것입니다. 이 응답 객체는 브라우저에게 어떻게 내용을 표시할지, 상태는 어떠한지 등 중요한 정보를 전달합니다.

#### 1.1.1. **`HttpResponse` - 기본 응답 객체**

가장 기본이 되는 응답 객체로, 주로 HTML 문자열이나 텍스트를 담아 보냅니다. `render()` 단축 함수도 내부적으로는 템플릿을 렌더링한 HTML 문자열을 담은 `HttpResponse`를 생성하여 반환합니다.

- **`content`**: 응답 본문(body)으로, 바이트(bytes)나 문자열(string) 형태입니다.
- **`status`**: HTTP 상태 코드입니다. 기본값은 `200`(OK)이며, `status=403` (Forbidden), `status=404` (Not Found) 등 다른 코드를 명시적으로 지정할 수 있습니다.
- **`content_type`**: 응답의 MIME 타입을 지정합니다. 기본값은 `text/html`입니다. 이를 변경하여 CSV나 XML 등 다양한 형식의 데이터를 반환할 수 있습니다.

**실무 예시:**

```python
from django.http import HttpResponse

# 1. 간단한 텍스트 응답 (주로 테스트나 간단한 확인용)
def health_check(request):
    return HttpResponse("OK", content_type="text/plain", status=200)

# 2. 동적으로 CSV 파일 생성 및 다운로드
def export_users_csv(request):
    import csv
    response = HttpResponse(content_type='text/csv')
    # 브라우저가 응답을 다운로드 파일로 처리하도록 헤더 설정
    response['Content-Disposition'] = 'attachment; filename="users.csv"'

    writer = csv.writer(response)
    writer.writerow(['Username', 'Email', 'First Name', 'Last Name'])

    for user in User.objects.all().values_list('username', 'email', 'first_name', 'last_name'):
        writer.writerow(user)
    
    return response

# 3. 권한이 없을 때 403 Forbidden 응답 반환
def secret_page(request):
    if not request.user.is_staff:
        return HttpResponse("접근 권한이 없습니다.", status=403)
    # ... 스태프에게만 보이는 내용 ...
```

#### 1.1.2. **`JsonResponse` - API를 위한 핵심 응답**

`HttpResponse`의 자식 클래스로, 파이썬 `dict` 객체를 받아 `Content-Type`이 `application/json`으로 설정된 응답을 생성합니다. 현대 웹 프론트엔드(React, Vue 등)와 통신하는 API를 만들 때 필수적입니다.

- **`data`**: JSON으로 변환할 `dict` 객체입니다.
- **`safe=True` (기본값)**: `data` 인자로 `dict` 객체만 허용합니다. 만약 리스트(`list`) 등 다른 타입을 보내려면 보안상의 이유로 `safe=False`를 명시해야 합니다.
- **`json_dumps_params`**: JSON 변환 시 사용할 `json.dumps()`의 파라미터를 지정합니다.
    - `ensure_ascii=False`: 한글 등 비-ASCII 문자가 깨지지 않고 그대로 보이게 합니다.
    - `indent=2`: 개발 환경에서 JSON 응답을 사람이 읽기 좋게 2칸 들여쓰기로 포맷팅합니다.

**실무 예시: ORM 쿼리셋을 JSON으로 변환하기**

ORM이 반환하는 쿼리셋(모델 객체의 리스트)은 JSON으로 바로 변환할 수 없습니다. `.values()`를 사용하여 딕셔너리의 쿼리셋으로 변환한 뒤, 이를 다시 `list()`로 감싸야 `JsonResponse`가 처리할 수 있는 형태가 됩니다.

```python
from django.http import JsonResponse
from .models import Post

# 간단한 게시글 목록 API
def post_list_api(request):
    # 1. 필요한 필드만 .values()로 조회하여 딕셔너리 쿼리셋 생성
    queryset = Post.objects.filter(is_published=True).values(
        'id', 
        'title',
        'created_at',
        'author__username' # ForeignKey 관계 필드 조회
    )

    # 2. 쿼리셋을 리스트로 변환
    post_list = list(queryset)

    # 3. safe=False 옵션과 함께 JsonResponse로 반환
    return JsonResponse(post_list, safe=False)

# API에서 에러 응답 반환하기
def some_api_view(request):
    if not request.user.is_authenticated:
        # 401 Unauthorized (인증되지 않음)
        return JsonResponse({"error": "인증이 필요합니다."}, status=401)

    if request.method != 'POST':
        # 405 Method Not Allowed (허용되지 않은 메소드)
        return JsonResponse({"error": "POST 요청만 가능합니다."}, status=405)
    
    # ... (로직 처리) ...
    return JsonResponse({"message": "성공적으로 처리되었습니다."}, status=200)
```

#### 1.1.3. **기타 유용한 `HttpResponse` 자식 클래스**

특정 HTTP 상태에 대해 Django는 미리 정의된 클래스를 제공하여 코드를 더 명시적으로 만들어 줍니다.

- **`HttpResponseRedirect`**: `redirect()` 단축 함수가 내부적으로 사용하는 클래스. 302 Found 응답을 보냅니다.
- **`HttpResponsePermanentRedirect`**: 301 Moved Permanently 응답을 보냅니다.
- **`HttpResponseNotFound`**: 404 Not Found 응답. `HttpResponse(status=404)`와 동일.
- **`HttpResponseForbidden`**: 403 Forbidden 응답. `HttpResponse(status=403)`와 동일.
- **`HttpResponseBadRequest`**: 400 Bad Request 응답. `HttpResponse(status=400)`와 동일.
- **`HttpResponseNotAllowed`**: 405 Method Not Allowed 응답. 허용되는 메서드 리스트를 첫 인자로 받습니다. `HttpResponseNotAllowed(['GET', 'POST'])`

### 1.2. 템플릿 렌더링 및 리다이렉트 (render, redirect)

*   **`render(request, template_name, context=None)`**: 템플릿 파일을 로드하고, 주어진 컨텍스트(데이터)를 사용하여 렌더링한 후, 그 결과를 `HttpResponse` 객체로 반환합니다. 웹 페이지를 동적으로 생성할 때 가장 흔히 사용됩니다.
    *   **예시 (`myhome1/blog/views.py`):
        ```python
        from django.shortcuts import render
        from .models import Blog

        def list(request):
            blog_list = Blog.objects.all()
            return render(request, "blog/blog_list.html", {"blogList":blog_list})
        ```

*   **`redirect(to, *args, **kwargs)`**: 사용자를 다른 URL로 리다이렉트(재요청)합니다. 주로 폼 제출 후 목록 페이지로 이동하거나, 로그인 후 메인 페이지로 이동하는 등의 상황에서 사용됩니다.
    *   **예시 (`myhome1/blog/views.py`):
        ```python
        from django.shortcuts import redirect

        def save(request):
            # ... (데이터 저장 로직) ...
            return redirect("blog:list") # 'blog' 앱의 'list' URL 패턴으로 리다이렉트
        ```
    *   **예시 (`mysite1/score/views.py`):
        ```python
        def index(request):
            return redirect("score:score_list")
        ```

### 1.3. 요청 객체 (HttpRequest) 다루기

뷰 함수는 첫 번째 인자로 `HttpRequest` 객체를 받습니다. 이 객체는 클라이언트로부터의 요청에 대한 모든 정보를 담고 있습니다.

*   **`request.method`**: 요청의 HTTP 메서드(GET, POST 등)를 문자열로 반환합니다.
    *   **예시 (`mysite1/guestbook/views.py`):
        ```python
        def test3(request):
            if request.method=="POST":
                x = request.POST.get("x")
                y = request.POST.get("y")
                return HttpResponse(int(x)+int(y))
            else:
                return HttpResponse("Error")
        ```

*   **`request.GET`**: GET 방식으로 전달된 모든 파라미터를 담고 있는 딕셔너리 형태의 객체입니다. `request.GET.get('key')`로 값을 가져옵니다.
    *   **예시 (`mysite1/guestbook/views.py`):
        ```python
        def isLeap(request):
            year = request.GET.get("year") # URL 쿼리 파라미터에서 'year' 값 가져오기
            # ...
        ```

*   **`request.POST`**: POST 방식으로 전달된 모든 파라미터를 담고 있는 딕셔너리 형태의 객체입니다. `request.POST.get('key')`로 값을 가져옵니다.
    *   **예시 (`mysite1/guestbook/views.py`):
        ```python
        def save(request):
            flower = request.POST.get("flower") # 폼 데이터에서 'flower' 값 가져오기
            # ...
        ```

*   **URL 파라미터**: `urls.py`에서 `<int:id>`와 같이 정의된 URL 패턴의 값은 뷰 함수의 인자로 직접 전달됩니다.
    *   **예시 (`myhome1/blog/views.py`):
        ```python
        def view(request, id): # id는 URL 패턴에서 전달된 값
            print("id", id)
            blog=Blog.objects.get(id=id)
            # ...
        ```

### 1.4. 페이지네이션 (Paginator)

Django의 `Paginator` 클래스는 대량의 데이터를 여러 페이지로 나누어 표시할 때 유용합니다. 데이터베이스에서 모든 데이터를 한 번에 가져오지 않고, 필요한 페이지의 데이터만 효율적으로 가져올 수 있도록 돕습니다.

*   **`Paginator(object_list, per_page)`**: `object_list` (쿼리셋)를 `per_page` 개수만큼 페이지로 나눕니다.
*   **`paginator.get_page(page_number)`**: 특정 `page_number`에 해당하는 `Page` 객체를 반환합니다. 이 때 실제 데이터베이스 쿼리가 발생합니다.
*   `Page` 객체의 주요 속성:
    *   `object_list`: 현재 페이지의 객체 리스트
    *   `number`: 현재 페이지 번호
    *   `paginator`: 연결된 `Paginator` 객체
    *   `has_previous()`, `has_next()`: 이전/다음 페이지 존재 여부
    *   `previous_page_number()`, `next_page_number()`: 이전/다음 페이지 번호
    *   `paginator.num_pages`: 전체 페이지 수
    *   `paginator.page_range`: 전체 페이지 번호 범위 (예: `range(1, 11)`)

    *   **예시 (`mysite1/score/views.py` - `list` 함수):
        ```python
        from django.core.paginator import Paginator
        from .models import Score

        def list(request):
            scoreList = Score.objects.all().order_by('-id') # 모든 Score 객체를 최신순으로 정렬
            paginator = Paginator(scoreList, 10) # 한 페이지에 10개씩 표시

            page_number = request.GET.get('page') # URL 쿼리에서 'page' 파라미터 가져오기
            page_obj = paginator.get_page(page_number) # 해당 페이지의 Page 객체 가져오기

            context = {
                "page_obj": page_obj, # 템플릿으로 Page 객체 전달
                "title": "성적처리",
            }
            return render(request, "score/score_list.html", context)
        ```
    *   **템플릿 예시 (`mysite1/templates/score/score_list.html`):
        ```html
        {# page_obj.object_list는 현재 페이지에 보여줄 Score 객체들의 리스트입니다. #}
        {% for score in page_obj.object_list %}
            <tr>
                <td>{{ score.id }}</td>
                <td>{{ score.name }}</td>
                <td>{{ score.wdate|date:"Y-m-d" }}</td>
            </tr>
        {% endfor %}

        <div class="pagination">
            {% if page_obj.has_previous %}
                <a href="?page={{ page_obj.previous_page_number }}">이전</a>
            {% else %}
                <span class="disabled">이전</span>
            {% endif %}

            <span>
                페이지 {{ page_obj.number }} / {{ page_obj.paginator.num_pages }}
            </span>

            {% if page_obj.has_next %}
                <a href="?page={{ page_obj.next_page_number }}">다음</a>
            {% else %}
                <span class="disabled">다음</span>
            {% endif %}

            {% for i in page_obj.paginator.page_range %}
                {% if page_obj.number == i %}
                    <span class="current-page">{{ i }}</span>
                {% else %}
                    <a href="?page={{ i }}">{{ i }}</a>
                {% endif %}
            {% endfor %}
        </div>
        ```

## 2. 클래스 기반 뷰 (Class-Based Views, CBV): 실무 심화

클래스 기반 뷰(CBV)는 뷰를 함수가 아닌 파이썬 클래스로 작성하는 방식입니다. HTTP 요청 메서드(GET, POST 등)에 해당하는 클래스 메서드를 각각 정의하여 로직을 구성하며, 상속과 믹스인(Mixin) 같은 객체 지향 프로그래밍의 장점을 활용하여 코드의 재사용성과 구조를 극대화할 수 있습니다.

### 2.1. CBV: 왜 클래스를 사용하는가?

- **코드의 구조화**: `if request.method == 'GET': ... elif request.method == 'POST': ...` 와 같은 조건문 블록 대신, `get()`, `post()` 등 HTTP 메서드 이름과 동일한 별도의 클래스 메서드로 로직이 명확하게 분리됩니다.
- **코드의 재사용성**: 상속을 통해 여러 뷰에서 공통되는 기능을 쉽게 재사용할 수 있습니다. 예를 들어, 로그인한 유저에게만 특정 기능을 제공하는 `LoginRequiredMixin`을 만들어 여러 뷰에 간단히 적용할 수 있습니다.
- **Django 제네릭 뷰**: Django는 목록, 상세, 생성, 수정, 삭제 등 웹 개발에서 반복적으로 나타나는 패턴을 미리 구현해 둔 강력한 **제네릭 뷰(Generic Views)** 세트를 제공합니다. 개발자는 이 클래스들을 상속받고 몇 가지 속성만 지정함으로써, 밑바닥부터 코드를 작성할 필요 없이 매우 빠르게 기능을 구현할 수 있습니다.

- **FBV vs CBV 선택 가이드**
    - **함수 기반 뷰 (FBV) 사용 시점:**
        - **간단한 뷰:** 로직이 복잡하지 않고 몇 줄로 끝나는 간단한 뷰를 작성할 때 빠르고 직관적입니다.
        - **특수한 로직:** 정형화되지 않은 매우 특수한 로직이나 복잡한 조건 분기가 필요할 때 함수형으로 작성하는 것이 더 유연할 수 있습니다.
        - **초심자:** Django를 처음 배울 때 코드의 흐름을 이해하기 더 쉽습니다.
    - **클래스 기반 뷰 (CBV) 사용 시점:**
        - **CRUD 기능:** `ListView`, `DetailView`, `CreateView`, `UpdateView`, `DeleteView`와 같이 데이터베이스와 상호작용하는 정형화된 CRUD 기능을 구현할 때 코드 양을 획기적으로 줄일 수 있습니다.
        - **코드 재사용:** 여러 뷰에서 공통적으로 사용되는 로직이 있을 때, 상속이나 믹스인을 통해 코드를 재사용하기 용이합니다.
        - **대규모 프로젝트:** 프로젝트의 규모가 커질수록 일관된 구조를 유지하고 코드를 체계적으로 관리하는 데 유리합니다.

### 2.2. CBV의 동작 원리: `as_view()` 와 `dispatch()`

`urls.py`에서 CBV를 등록할 때는 항상 `MyView.as_view()`와 같이 `.as_view()` 클래스 메서드를 사용합니다. 그 이유는 다음과 같습니다.

1.  URL이 매칭되면, Django는 `MyView.as_view()`를 호출하여 뷰 클래스의 **인스턴스**를 생성합니다.
2.  생성된 인스턴스의 `dispatch()` 메서드가 호출됩니다.
3.  `dispatch()` 메서드는 `request` 객체를 살펴보고, `request.method`가 `'GET'`이면 클래스 내의 `get()` 메서드를, `'POST'`이면 `post()` 메서드를 찾아 대신 실행해 줍니다.
4.  만약 해당 HTTP 메서드에 대한 메서드가 클래스에 정의되어 있지 않으면, `HttpResponseNotAllowed` (405 Method Not Allowed) 예외를 발생시킵니다.

이러한 내부 동작 덕분에 우리는 각 HTTP 메서드에 따른 로직을 해당 이름의 메서드 안에 깔끔하게 작성하기만 하면 됩니다.

```python
# urls.py
# MyView.as_view()는 요청이 올 때마다 새로운 MyView 인스턴스를 생성하고
# 적절한 메서드(get, post 등)를 실행하는 진입점 역할을 합니다.
path('my-view/', MyView.as_view(), name='my-view')
```

### 2.3. 제네릭 뷰 상세 가이드: "Don't Repeat Yourself"

제네릭 뷰는 CBV의 꽃입니다. 최소한의 코드로 정형화된 웹 기능을 구현할 수 있습니다.

- **`ListView`**: 특정 모델의 객체 목록을 보여줍니다.
    - `model`: 어떤 모델의 목록을 보여줄지 지정합니다. (예: `model = Post`)
    - `template_name`: 템플릿 파일 경로를 지정합니다. (기본값: `<app_name>/<model_name>_list.html`)
    - `context_object_name`: 템플릿에 전달될 객체 목록 변수의 이름을 지정합니다. (기본값: `object_list`)
    - `paginate_by`: 한 페이지에 보여줄 객체의 수를 지정하여 페이지네이션을 활성화합니다.

- **`DetailView`**: 특정 모델의 단일 객체 상세 정보를 보여줍니다.
    - `model`: 대상 모델을 지정합니다.
    - `template_name`: 템플릿 파일 경로를 지정합니다. (기본값: `<app_name>/<model_name>_detail.html`)
    - `context_object_name`: 템플릿에 전달될 단일 객체 변수의 이름을 지정합니다. (기본값: `object`)
    - `pk_url_kwarg`: URLconf에서 객체를 찾기 위해 사용하는 기본 키(PK) 변수의 이름을 지정합니다. (기본값: `'pk'`)



- **`CreateView` / `UpdateView`**: 폼을 통해 객체를 생성하거나 수정합니다.
    - `model`: 대상 모델을 지정합니다.
    - `form_class` 또는 `fields`: 사용할 `ModelForm` 클래스를 직접 지정하거나(`form_class = PostForm`), `fields = ['title', 'content']` 와 같이 필드 목록을 지정하여 Django가 자동으로 폼을 생성하게 할 수 있습니다.
    - `template_name`: 폼을 보여줄 템플릿 경로를 지정합니다. (기본값: `<app_name>/<model_name>_form.html`)
    - `success_url`: 폼 처리가 성공적으로 완료된 후 리다이렉트할 URL을 지정합니다. URL 이름을 사용하는 것이 좋습니다. (예: `success_url = reverse_lazy('blog:post_list')`)

    *   **예시 (`blog/views.py`):
        ```python
        from django.views.generic.edit import CreateView
        from .models import Blog
        from .forms import BlogForms

        class BlogCreateView(CreateView):
            model = Blog
            form_class = BlogForms # 사용할 폼 클래스 지정
            template_name = 'blog/blog_form.html' # 폼을 렌더링할 템플릿
            success_url = '/blog/list/' # 폼 제출 성공 시 리다이렉트할 URL
        ```
    *   **`urls.py`에서 연결:**
        ```python
        from django.urls import path
        from .views import BlogCreateView

        urlpatterns = [
            path('create/', BlogCreateView.as_view(), name='blog_create'),
        ]
        ```

    *   **예시 (`blog/views.py`):
        ```python
        from django.views.generic.edit import UpdateView
        # ... (Blog, BlogForms 임포트)

        class BlogUpdateView(UpdateView):
            model = Blog
            form_class = BlogForms
            template_name = 'blog/blog_form.html'
            success_url = '/blog/list/'
        ```
    *   **`urls.py`에서 연결:**
        ```python
        from django.urls import path
        from .views import BlogUpdateView

        urlpatterns = [
            path('update/<int:pk>/', BlogUpdateView.as_view(), name='blog_update'),
        ]
        ```
- **`DeleteView`**: 특정 객체의 삭제를 확인하고 처리합니다.
    - `model`: 대상 모델을 지정합니다.
    - `template_name`: 삭제 확인 페이지 템플릿 경로를 지정합니다. (기본값: `<app_name>/<model_name>_confirm_delete.html`)
    - `success_url`: 삭제 완료 후 리다이렉트할 URL을 지정합니다.
    *   **예시 (`blog/views.py`):
        ```python
        from django.views.generic.edit import DeleteView
        # ... (Blog 임포트)

        class BlogDeleteView(DeleteView):
            model = Blog
            template_name = 'blog/blog_confirm_delete.html' # 삭제 확인 템플릿
            success_url = '/blog/list/'
        ```
    *   **`urls.py`에서 연결:**
        ```python
        from django.urls import path
        from .views import BlogDeleteView

        urlpatterns = [
            path('delete/<int:pk>/', BlogDeleteView.as_view(), name='blog_delete'),
        ]
        ```
### 2.4. CBV 커스터마이징: 실무 핵심 기술

실무에서는 제네릭 뷰를 그대로 사용하기보다, 상속받아 특정 메서드를 **오버라이딩(재정의)**하여 프로젝트의 요구사항에 맞게 동작을 변경하는 경우가 대부분입니다.

- **`get_queryset()`**: 뷰에 표시할 객체 목록을 동적으로 결정할 때 오버라이딩합니다. 예를 들어, 현재 로그인한 사용자가 작성한 게시글만 필터링하여 보여줄 수 있습니다.
    ```python
    class MyPostListView(ListView):
        model = Post
        # ...
        def get_queryset(self):
            # 기존 쿼리셋을 가져와 필터링 조건을 추가
            queryset = super().get_queryset()
            return queryset.filter(author=self.request.user)
    ```

- **`get_context_data(**kwargs)`**: 템플릿에 기본적으로 전달되는 컨텍스트 데이터 외에 추가적인 정보를 전달하고 싶을 때 오버라이딩합니다.
    ```python
    class PostDetailView(DetailView):
        model = Post
        # ...
        def get_context_data(self, **kwargs):
            # 먼저 부모 클래스의 메서드를 호출하여 기본 컨텍스트를 받음
            context = super().get_context_data(**kwargs)
            # 추가하고 싶은 컨텍스트 정보를 딕셔너리에 추가
            context['all_tags'] = Tag.objects.all()
            return context
    ```

- **`form_valid(form)`**: `CreateView`나 `UpdateView`에서 폼 데이터가 유효성 검사를 통과한 직후, 데이터가 DB에 저장되기 전에 호출됩니다. `request.user`를 작성자로 자동 할당하는 등, 폼에 없는 데이터를 모델에 추가할 때 사용하기 가장 좋은 위치입니다.
    ```python
    class PostCreateView(CreateView):
        model = Post
        fields = ['title', 'content']
        # ...
        def form_valid(self, form):
            # form.instance는 저장될 모델 객체를 의미
            form.instance.author = self.request.user
            # 부모 클래스의 form_valid를 호출하여 최종 저장 및 리다이렉트 수행
            return super().form_valid(form)
    ```

- **믹스인 (Mixins)**: 여러 클래스에서 공통적으로 사용될 특정 기능을 담고 있는 작은 클래스입니다. CBV와 함께 사용하여 뷰에 기능을 손쉽게 추가할 수 있습니다. 대표적인 예가 `LoginRequiredMixin`입니다.
    ```python
    from django.contrib.auth.mixins import LoginRequiredMixin

    # LoginRequiredMixin을 상속받으면, 로그인하지 않은 사용자는 자동으로 로그인 페이지로 리다이렉트됨
    class ProtectedView(LoginRequiredMixin, View):
        # login_url = '/accounts/login/' # 리다이렉트할 로그인 URL을 변경할 수 있음
        
        def get(self, request):
            # 이 코드는 로그인한 사용자에게만 실행됨
            ...
    ```

## 3. 에러 핸들링 및 사용자 친화적 페이지: 실무 가이드

잘 만들어진 웹 애플리케이션은 예외 상황을 우아하게 처리합니다. 에러 핸들링의 목표는 두 가지입니다.

1.  **사용자에게는**: 시스템의 내부 구조를 노출하지 않는, 친절하고 이해하기 쉬운 에러 페이지를 보여줍니다.
2.  **개발자에게는**: 문제의 원인을 신속하게 파악할 수 있는 상세하고 구조화된 로그를 남깁니다.

이 모든 것은 `settings.py`에서 `DEBUG=False`로 설정된 운영 환경을 기준으로 합니다.

### 3.1. 사용자 친화적 에러 페이지 (4xx/5xx)

Django는 특정 HTTP 에러가 발생했을 때, 미리 약속된 이름의 템플릿을 자동으로 렌더링해주는 기능을 제공합니다.

- **방법 1: 템플릿만으로 커스터마이징 (간단한 방법)**

    프로젝트의 루트 `templates` 디렉토리(즉, `settings.py`의 `TEMPLATES` 설정 `DIRS`에 포함된 경로)에 아래 이름으로 HTML 파일을 만들기만 하면 됩니다.

    - `400.html`: Bad Request (잘못된 요청)
    - `403.html`: Permission Denied (권한 없음)
    - `404.html`: Page Not Found (페이지를 찾을 수 없음)
    - `500.html`: Server Error (서버 내부 오류)

    **`templates/404.html` 예시:**
    ```html
    {% extends "base.html" %} {# 사이트의 기본 레이아웃을 상속받아 일관성을 유지합니다 #}

    {% block title %}페이지를 찾을 수 없습니다{% endblock %}

    {% block content %}
      <div class="error-page">
        <h1>404 - Page Not Found</h1>
        <p>요청하신 페이지가 존재하지 않거나, 주소가 변경되었을 수 있습니다.</p>
        <a href="{% url 'main' %}">홈으로 돌아가기</a>
      </div>
    {% endblock %}
    ```

- **방법 2: 커스텀 핸들러 뷰 지정 (고급 방법)**

    단순히 정적인 페이지만 보여주는 것을 넘어, 에러 발생 시 특정 로직(예: 에러 로그 기록, 관리자에게 알림 발송)을 수행하고 싶을 때 사용합니다. 프로젝트의 루트 `urls.py`에 핸들러를 직접 지정합니다.

    **`my_project/urls.py`:**
    ```python
    # ...
    handler404 = 'my_app.views.custom_404_view'
    handler500 = 'my_app.views.custom_500_view'
    handler403 = 'my_app.views.custom_403_view'
    handler400 = 'my_app.views.custom_400_view'
    ```

    **`my_app/views.py`:**
    ```python
    from django.shortcuts import render
    import logging

    logger = logging.getLogger(__name__)

    def custom_404_view(request, exception):
        # exception 객체에는 오류 관련 정보가 담겨 있습니다.
        return render(request, "errors/404.html", {'path': request.path}, status=404)

    def custom_500_view(request):
        # Sentry, Datadog 등 외부 모니터링 서비스에 에러를 리포트하는 로직 추가 가능
        logger.error("Internal Server Error (500) occurred", exc_info=True)
        return render(request, "errors/500.html", status=500)
    ```

### 3.2. API 예외 처리: 일관된 에러 응답 구조화

REST API에서는 HTML이 아닌, 일관된 구조의 JSON으로 에러를 반환하는 것이 매우 중요합니다. 프론트엔드 개발자는 이 구조를 바탕으로 사용자에게 적절한 에러 메시지를 표시할 수 있습니다.

**Django REST Framework(DRF)**는 이를 위한 강력한 **커스텀 예외 핸들러(Custom Exception Handler)** 기능을 제공합니다.

- **목표**: 모든 API 에러 응답을 `{"success": false, "error": {"code": "에러코드", "message": "에러메시지"}}` 와 같은 일관된 형태로 통일합니다.

- **구현 단계:**

    1.  **`exceptions.py` 파일 생성**: 프로젝트의 핵심 로직을 담는 앱(예: `core`)에 `exceptions.py` 파일을 만듭니다.

    2.  **커스텀 핸들러 함수 작성**:
        ```python
        # core/exceptions.py
        from rest_framework.views import exception_handler

        def custom_exception_handler(exc, context):
            # 먼저 DRF의 기본 예외 핸들러를 호출하여 기본적인 응답을 생성합니다.
            response = exception_handler(exc, context)

            # response가 생성되었다면 (즉, DRF가 처리할 수 있는 예외라면)
            if response is not None:
                # 우리가 원하는 포맷으로 데이터를 재구성합니다.
                custom_data = {
                    'success': False,
                    'error': {
                        'code': response.data.get('code', response.status_code),
                        'message': response.data.get('detail', str(exc))
                    }
                }
                response.data = custom_data
            
            return response
        ```

    3.  **`settings.py`에 핸들러 등록**:
        ```python
        # settings.py
        REST_FRAMEWORK = {
            # ... 다른 설정들 ...
            'EXCEPTION_HANDLER': 'core.exceptions.custom_exception_handler',
        }
        ```

- **결과**: 이제 DRF의 `ValidationError`, `PermissionDenied`, `NotFound` 등 모든 예외는 물론, `get_object_or_404`가 발생시키는 `Http404` 예외까지 모두 위에서 정의한 일관된 JSON 형식으로 사용자에게 반환됩니다. 이를 통해 프론트엔드와의 협업이 매우 원활해집니다.
