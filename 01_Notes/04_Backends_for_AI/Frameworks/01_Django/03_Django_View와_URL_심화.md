<h2>Django Backend: View와 URL 심화</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django의 View 계층을 깊이 있게 다룹니다. 함수 기반 뷰(FBV)와 클래스 기반 뷰(CBV)의 작성법, HTTP 요청 및 응답 처리, 페이지네이션, 에러 핸들링 등 View와 관련된 핵심 개념들을 학습하여 동적인 웹 페이지 로직을 효과적으로 구현하는 것을 목표로 합니다.</p>

<h2>목차</h2> 

- [1. Django View](#1-django-view)
  - [1.1. 함수 기반 뷰 (Function-Based Views, FBV)](#11-함수-기반-뷰-function-based-views-fbv)
    - [1.1.1. HTTP 응답 (HttpResponse, JsonResponse)](#111-http-응답-httpresponse-jsonresponse)
    - [1.1.2. 템플릿 렌더링 및 리다이렉트 (render, redirect)](#112-템플릿-렌더링-및-리다이렉트-render-redirect)
    - [1.1.3. 요청 객체 (HttpRequest) 다루기](#113-요청-객체-httprequest-다루기)
    - [1.1.4. 페이지네이션 (Paginator)](#114-페이지네이션-paginator)
  - [1.2. 클래스 기반 뷰 (Class-Based Views, CBV)](#12-클래스-기반-뷰-class-based-views-cbv)
    - [1.2.1. CBV의 장점](#121-cbv의-장점)
    - [1.2.2. `View` 클래스](#122-view-클래스)
    - [1.2.3. 제네릭 편집 뷰 (Generic Editing Views)](#123-제네릭-편집-뷰-generic-editing-views)
    - [1.2.4. 제네릭 디스플레이 뷰 (Generic Display Views)](#124-제네릭-디스플레이-뷰-generic-display-views)
  - [1.3. 에러 핸들링 및 사용자 친화적 페이지](#13-에러-핸들링-및-사용자-친화적-페이지)
    - [1.3.1. 404 및 500 페이지 커스터마이징](#131-404-및-500-페이지-커스터마이징)
    - [1.3.2. 뷰/API 예외 처리](#132-뷰api-예외-처리)

---

## 1. Django View

뷰(View)는 웹 요청을 수신하고 HTTP 응답을 반환하는 파이썬 함수 또는 클래스입니다. 주로 모델에서 데이터를 가져오거나, 템플릿을 렌더링하거나, 다른 HTTP 응답을 생성하는 로직을 포함합니다.

### 1.1. 함수 기반 뷰 (Function-Based Views, FBV)

함수 기반 뷰는 파이썬 함수로 작성된 뷰입니다. 간단한 로직을 처리하거나 특정 HTTP 메서드에 대한 응답을 명확하게 분리할 때 유용합니다.

#### 1.1.1. HTTP 응답 (HttpResponse, JsonResponse)

*   **`HttpResponse`**: 가장 기본적인 HTTP 응답 객체로, 문자열 형태의 콘텐츠를 반환합니다. 주로 간단한 텍스트 응답이나 디버깅에 사용됩니다.
    *   **예시 (`myhome1/blog/views.py`):
        ```python
        from django.http import HttpResponse

        def index(request):
            return HttpResponse("Hello Django")
        ```
    *   **예시 (`mysite1/guestbook/views.py`):
        ```python
        def test1(request):
            x = request.GET.get("x")
            y = request.GET.get("y")
            return HttpResponse(int(x)+int(y))
        ```
    *   **예시 (`장고사이트구축방법.txt` - `views.py` 파일 수정하기 섹션 참고):
        ```python
        # views.py (가상)
        from django.http import HttpRequest, HttpResponse

        def test1(request):
            return HttpResponse("test1")

        def test2(request):
            ua = request.META['HTTP_USER_AGENT']
            return HttpResponse('<H1>'+ua+'</H1>')

        # http://127.0.0.1:8000/blog/4/5
        def test3(request, xvalue, yvalue):
            return HttpResponse("{} + {} = {}".format(xvalue, yvalue, 
                    int(xvalue)+int(yvalue)))

        # http://127.0.0.1:8000/blog?x=4&y=5
        def test4(request):
            xvalue=int(request.GET.get("x"))
            yvalue=int(request.GET.get("y"))
            
            return HttpResponse("{} + {} = {}".format(xvalue, yvalue, 
                    int(xvalue)+int(yvalue)))

        def test5(request):
            if request.method=="POST":
                xvalue=int(request.POST.get("x"))
                yvalue=int(request.POST.get("y"))
                
                return HttpResponse("{} + {} = {}".format(xvalue, yvalue, 
                        int(xvalue)+int(yvalue)))
            else:
                return HttpResponse("Error")
        ```
    *   **예시 (`장고사이트구축방법.txt` - `board/views.py` `index` 함수):
        ```python
        # board/views.py (가상)
        from django.http import HttpResponse

        def index(request):
            return HttpResponse("Hello Django")
        ```

*   **`JsonResponse`**: JSON 형식의 데이터를 반환할 때 사용합니다. RESTful API를 구축할 때 유용하며, 파이썬 딕셔너리나 리스트를 자동으로 JSON으로 직렬화합니다. 한글 깨짐 방지를 위해 `json_dumps_params={'ensure_ascii': False}` 옵션을 사용할 수 있습니다.
    *   **예시 (`mysite1/guestbook/views.py`):
        ```python
        from django.http import JsonResponse

        def getData(request):
            return JsonResponse({"name":"홍길동", "age":23, "phone":"010-0000-0001"},
                                 json_dumps_params={'ensure_ascii': False})
        ```

#### 1.1.2. 템플릿 렌더링 및 리다이렉트 (render, redirect)

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

#### 1.1.3. 요청 객체 (HttpRequest) 다루기

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

#### 1.1.4. 페이지네이션 (Paginator)

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

### 1.2. 클래스 기반 뷰 (Class-Based Views, CBV)

클래스 기반 뷰는 파이썬 클래스로 작성된 뷰입니다. 함수 기반 뷰보다 더 많은 기능을 제공하며, 코드의 재사용성과 유지보수성을 높일 수 있습니다. 특히 Django의 제네릭 뷰(Generic Views)를 활용하면 반복적인 웹 개발 작업을 최소화할 수 있습니다.

#### 1.2.1. CBV의 장점
*   **코드 재사용성:** 상속과 믹스인(Mixin)을 통해 공통 로직을 재사용할 수 있습니다.
*   **유지보수성:** 관련 로직이 클래스 내에 캡슐화되어 있어 코드를 더 깔끔하게 관리할 수 있습니다.
*   **확장성:** 특정 HTTP 메서드(GET, POST 등)에 대한 로직을 별도의 메서드로 분리하여 관리하기 용이합니다.
*   **제네릭 뷰 활용:** Django가 제공하는 강력한 제네릭 뷰를 사용하여 CRUD(Create, Read, Update, Delete)와 같은 일반적인 웹 패턴을 빠르게 구현할 수 있습니다.

- **FBV vs CBV 선택 가이드**
    - **함수 기반 뷰 (FBV) 사용 시점:**
        - **간단한 뷰:** 로직이 복잡하지 않고 몇 줄로 끝나는 간단한 뷰를 작성할 때 빠르고 직관적입니다.
        - **특수한 로직:** 정형화되지 않은 매우 특수한 로직이나 복잡한 조건 분기가 필요할 때 함수형으로 작성하는 것이 더 유연할 수 있습니다.
        - **초심자:** Django를 처음 배울 때 코드의 흐름을 이해하기 더 쉽습니다.
    - **클래스 기반 뷰 (CBV) 사용 시점:**
        - **CRUD 기능:** `ListView`, `DetailView`, `CreateView`, `UpdateView`, `DeleteView`와 같이 데이터베이스와 상호작용하는 정형화된 CRUD 기능을 구현할 때 코드 양을 획기적으로 줄일 수 있습니다.
        - **코드 재사용:** 여러 뷰에서 공통적으로 사용되는 로직이 있을 때, 상속이나 믹스인을 통해 코드를 재사용하기 용이합니다.
        - **대규모 프로젝트:** 프로젝트의 규모가 커질수록 일관된 구조를 유지하고 코드를 체계적으로 관리하는 데 유리합니다.

#### 1.2.2. `View` 클래스

모든 클래스 기반 뷰의 기본이 되는 클래스입니다. `as_view()` 메서드를 통해 URLconf에 연결됩니다.

*   **예시 (`blog/views.py`):
    ```python
    from django.views import View
    from django.http import HttpResponse

    class MyView(View):
        def get(self, request, *args, **kwargs):
            return HttpResponse('Hello, Class-Based View!')

        def post(self, request, *args, **kwargs):
            return HttpResponse('This is a POST request.')
    ```
*   **`urls.py`에서 연결:**
    ```python
    from django.urls import path
    from .views import MyView

    urlpatterns = [
        path('myview/', MyView.as_view()),
    ]
    ```

#### 1.2.3. 제네릭 편집 뷰 (Generic Editing Views)

데이터 생성, 수정, 삭제와 같은 폼 처리와 관련된 작업을 간소화하는 뷰입니다.

*   **`CreateView`:** 모델 객체를 생성하는 폼을 처리합니다.
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

*   **`UpdateView`:** 기존 모델 객체를 수정하는 폼을 처리합니다.
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

*   **`DeleteView`:** 모델 객체를 삭제합니다.
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

#### 1.2.4. 제네릭 디스플레이 뷰 (Generic Display Views)

데이터를 표시하는 작업을 간소화하는 뷰입니다.

*   **`ListView`:** 모델 객체 목록을 표시합니다.
    *   **예시 (`blog/views.py`):
        ```python
        from django.views.generic import ListView
        from .models import Blog

        class BlogListView(ListView):
            model = Blog
            template_name = 'blog/blog_list.html' # 목록을 렌더링할 템플릿
            context_object_name = 'blogList' # 템플릿에서 사용할 변수 이름
            paginate_by = 10 # 페이지당 객체 수 (페이지네이션 자동 적용)
        ```
    *   **`urls.py`에서 연결:**
        ```python
        from django.urls import path
        from .views import BlogListView

        urlpatterns = [
            path('list/', BlogListView.as_view(), name='blog_list_cbv'),
        ]
        ```

*   **`DetailView`:** 단일 모델 객체의 상세 정보를 표시합니다.
    *   **예시 (`blog/views.py`):
        ```python
        from django.views.generic import DetailView
        from .models import Blog

        class BlogDetailView(DetailView):
            model = Blog
            template_name = 'blog/blog_detail.html'
            context_object_name = 'blog' # 템플릿에서 사용할 변수 이름
        ```
    *   **`urls.py`에서 연결:**
        ```python
        from django.urls import path
        from .views import BlogDetailView

        urlpatterns = [
            path('detail/<int:pk>/', BlogDetailView.as_view(), name='blog_detail'),
        ]
        ```

### 1.3. 에러 핸들링 및 사용자 친화적 페이지

운영 환경에서 `DEBUG=False`일 때, Django는 상세한 오류 페이지 대신 일반적인 오류 페이지를 보여줍니다. 사용자에게 더 나은 경험을 제공하고, 시스템 오류를 적절히 처리하기 위해 커스터마이징된 에러 페이지와 뷰/API 예외 처리가 필요합니다.

#### 1.3.1. 404 및 500 페이지 커스터마이징

Django는 `404.html` (페이지를 찾을 수 없음) 및 `500.html` (서버 내부 오류) 템플릿을 자동으로 찾아서 렌더링합니다.

1.  **`settings.py` 설정:**
    ```python
    # settings.py
    # DEBUG가 False일 때만 작동
    # ALLOWED_HOSTS 설정 필수
    # TEMPLATES DIRS에 템플릿 경로가 포함되어 있어야 함
    ```

2.  **템플릿 생성:**
    프로젝트의 `templates` 디렉토리(또는 `TEMPLATES['DIRS']`에 지정된 경로)에 `404.html`과 `500.html` 파일을 생성합니다.
    ```html
    <!-- templates/404.html -->
    <!DOCTYPE html>
    <html>
    <head>
        <title>페이지를 찾을 수 없습니다 (404)</title>
    </head>
    <body>
        <h1>404 - 페이지를 찾을 수 없습니다.</h1>
        <p>요청하신 페이지를 찾을 수 없습니다. 주소를 다시 확인해주세요.</p>
        <a href="/">홈으로 돌아가기</a>
    </body>
    </html>
    ```
    `500.html`도 유사하게 작성합니다.

#### 1.3.2. 뷰/API 예외 처리

뷰나 API 로직 내에서 발생하는 예외를 명시적으로 처리하여 사용자에게 의미 있는 메시지를 반환하거나, 특정 동작을 수행할 수 있습니다.

**예시 (함수 기반 뷰):**
```python
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseBadRequest, HttpResponseServerError
from myapp.models import MyModel

def my_view(request, item_id):
    try:
        item = get_object_or_404(MyModel, pk=item_id)
        # ... 정상 로직 ...
        return render(request, 'myapp/detail.html', {'item': item})
    except ValueError:
        # 잘못된 형식의 ID가 전달된 경우
        return HttpResponseBadRequest("잘못된 요청입니다. ID 형식을 확인해주세요.")
    except MyModel.DoesNotExist:
        # get_object_or_404가 처리하지만, 직접 처리할 경우
        return render(request, 'myapp/item_not_found.html', status=404)
    except Exception as e:
        # 예상치 못한 모든 예외 처리
        logger.error(f"An unexpected error occurred: {e}")
        return HttpResponseServerError("서버에 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
```

**예시 (DRF APIView):**
```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from myapp.models import MyModel
from myapp.serializers import MyModelSerializer

class MyAPIView(APIView):
    def get(self, request, pk, format=None):
        try:
            item = get_object_or_404(MyModel, pk=pk)
            serializer = MyModelSerializer(item)
            return Response(serializer.data)
        except MyModel.DoesNotExist:
            return Response({"detail": "Item not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"API error: {e}")
            return Response({"detail": "An internal server error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
```
