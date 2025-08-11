<h2>Django Backend: REST Framework API 개발</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django REST Framework(DRF)를 사용하여 강력하고 유연한 RESTful API를 구축하는 방법을 학습하는 것을 목표로 합니다. DRF의 핵심 구성 요소인 Serializer, ViewSet, Router의 개념을 이해하고, API 보안을 위한 인증 및 권한 부여 시스템을 다룹니다.</p>

<h2>목차</h2> 

- [1. RESTful API (Django REST Framework): 실무 가이드](#1-restful-api-django-rest-framework-실무-가이드)
  - [1.1. RESTful API의 기본 원칙](#11-restful-api의-기본-원칙)
  - [1.2. 왜 `JsonResponse` 대신 DRF를 사용하는가?](#12-왜-jsonresponse-대신-drf를-사용하는가)
- [2. DRF 설치 및 설정](#2-drf-설치-및-설정)
- [3. Serializer: 데이터 변환과 유효성 검사의 핵심](#3-serializer-데이터-변환과-유효성-검사의-핵심)
  - [3.1. `ModelSerializer`: 모델 기반 시리얼라이저](#31-modelserializer-모델-기반-시리얼라이저)
  - [3.2. `Serializer`: 모델과 무관한 데이터 직렬화](#32-serializer-모델과-무관한-데이터-직렬화)
  - [3.3. 시리얼라이저 사용 흐름 (뷰에서)](#33-시리얼라이저-사용-흐름-뷰에서)
- [4. ViewSet 및 APIView: API 엔드포인트 구현](#4-viewset-및-apiview-api-엔드포인트-구현)
  - [4.1. `APIView`: 저수준 제어](#41-apiview-저수준-제어)
  - [4.2. `ViewSet`: CRUD 작업의 간소화](#42-viewset-crud-작업의-간소화)
- [5. Router: ViewSet의 URL 자동 생성](#5-router-viewset의-url-자동-생성)
  - [5.1. `DefaultRouter` 사용법](#51-defaultrouter-사용법)
  - [5.2. 라우터가 생성하는 URL 패턴 (예시: `PostViewSet`에 대한 `/posts/`)](#52-라우터가-생성하는-url-패턴-예시-postviewset에-대한-posts)
- [6. 인증 및 권한 (Authentication \& Permissions): API 보안의 핵심](#6-인증-및-권한-authentication--permissions-api-보안의-핵심)
  - [6.1. DRF 인증 (Authentication): 사용자 식별](#61-drf-인증-authentication-사용자-식별)
  - [6.2. DRF 권한 (Permissions): 접근 제어](#62-drf-권한-permissions-접근-제어)
- [7. 추가 보안 고려사항 (CORS, XSS, SQL Injection 방지)](#7-추가-보안-고려사항-cors-xss-sql-injection-방지)
  - [7.1. CORS (Cross-Origin Resource Sharing)](#71-cors-cross-origin-resource-sharing)
  - [7.2. XSS (Cross-Site Scripting) 방지](#72-xss-cross-site-scripting-방지)
  - [7.3. SQL Injection 방지 (Raw 쿼리 사용 시 주의점)](#73-sql-injection-방지-raw-쿼리-사용-시-주의점)
- [8. DRF 심화](#8-drf-심화)
  - [8.1. Nested Serializers (중첩 시리얼라이저)](#81-nested-serializers-중첩-시리얼라이저)
  - [8.2. API 문서 자동화](#82-api-문서-자동화)

---

## 1. RESTful API (Django REST Framework): 실무 가이드

Django REST Framework (DRF)는 Django 위에 강력하고 유연한 RESTful API를 쉽게 구축할 수 있도록 돕는 툴킷입니다. DRF는 단순한 `JsonResponse`를 넘어, 복잡한 API 개발에 필요한 다양한 기능과 모범 사례를 제공하여 생산성, 유지보수성, 확장성 모든 면에서 훨씬 효율적입니다.

### 1.1. RESTful API의 기본 원칙

DRF를 사용하기 전에 RESTful API의 핵심 원칙을 이해하는 것이 중요합니다.

-   **자원(Resource)**: API가 제공하는 모든 것은 자원입니다. (예: 사용자, 게시글, 상품) 각 자원은 고유한 URI(Uniform Resource Identifier)로 식별됩니다.
-   **표현(Representation)**: 자원은 JSON, XML 등 다양한 형태로 표현될 수 있습니다. DRF는 주로 JSON을 사용합니다.
-   **상태 비저장(Stateless)**: 각 요청은 필요한 모든 정보를 담고 있어야 하며, 서버는 클라이언트의 이전 요청 상태를 기억하지 않습니다.
-   **균일한 인터페이스(Uniform Interface)**: HTTP 메서드(GET, POST, PUT, DELETE)를 사용하여 자원에 대한 표준화된 작업을 수행합니다.

### 1.2. 왜 `JsonResponse` 대신 DRF를 사용하는가?

Django의 내장 `JsonResponse`만으로도 간단한 API를 만들 수 있지만, 프로젝트 규모가 커지면 다음과 같은 문제에 직면하게 됩니다.

-   **반복적인 작업**: 데이터 직렬화(모델 인스턴스 -> JSON), 역직렬화(JSON -> 파이썬 객체), 유효성 검사, 인증 및 권한 확인 등 API 개발에 필요한 공통적인 작업들을 매번 수동으로 구현해야 합니다.
-   **유지보수의 어려움**: 일관된 구조 없이 API를 개발하면 코드의 유지보수가 어려워지고, API 명세를 관리하기도 힘듭니다.
-   **DRF는 이러한 문제들을 해결해줍니다.**
    -   **시리얼라이저(Serializer)**: 데이터 변환 및 유효성 검사를 자동화합니다.
    -   **인증/권한 시스템**: 토큰 기반 인증, JWT, OAuth2 등 다양한 인증 방식과 유연한 권한 제어 기능을 제공합니다.
    -   **자동 API 문서**: API 엔드포인트를 자동으로 문서화하여 프론트엔드 개발자와의 협업을 원활하게 합니다.
    -   **웹 브라우저블 API**: 개발자가 브라우저에서 직접 API를 테스트하고 디버깅할 수 있는 편리한 UI를 제공합니다.
    -   **필터링 및 페이지네이션**: 복잡한 필터링 조건이나 대량의 데이터를 페이지별로 나누어 제공하는 기능을 손쉽게 구현할 수 있습니다.
    -   **스로틀링(Throttling)**: API 요청 속도를 제한하여 서버 과부하를 방지합니다.
    -   **버저닝(Versioning)**: API의 진화에 따라 여러 버전을 동시에 관리할 수 있습니다.

---

## 2. DRF 설치 및 설정

DRF를 사용하려면 먼저 설치하고 `settings.py`에 등록해야 합니다.

```bash
pip install djangorestframework
```

**`settings.py` 예시:**

DRF의 전역 설정을 `REST_FRAMEWORK` 딕셔너리에 정의합니다.

```python
# settings.py

INSTALLED_APPS = [
    # ...
    'rest_framework',
    # 'rest_framework.authtoken', # 토큰 인증 사용 시 필요
    # 'rest_framework_simplejwt', # JWT 인증 사용 시 필요
]

REST_FRAMEWORK = {
    # 기본 인증 클래스 설정 (API 요청 시 어떤 방식으로 사용자를 인증할지)
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication', # 세션 기반 인증 (웹 브라우저용)
        'rest_framework.authentication.TokenAuthentication',   # 토큰 기반 인증 (API 클라이언트용)
        # 'rest_framework_simplejwt.authentication.JWTAuthentication', # JWT 인증
    ],
    # 기본 권한 클래스 설정 (인증된 사용자가 어떤 리소스에 접근할 수 있는지)
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated', # 기본적으로 인증된 사용자만 접근 허용
        # 'rest_framework.permissions.AllowAny', # 모든 사용자에게 접근 허용 (개발 초기)
    ],
    # 기본 렌더러 클래스 설정 (API 응답을 어떤 형식으로 렌더링할지)
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer', # JSON 응답
        'rest_framework.renderers.BrowsableAPIRenderer', # 웹 브라우저블 API (개발용)
    ],
    # 기본 파서 클래스 설정 (API 요청 본문을 어떤 형식으로 파싱할지)
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser', # JSON 요청 본문 파싱
        'rest_framework.parsers.FormParser', # HTML 폼 데이터 파싱
        'rest_framework.parsers.MultiPartParser', # 파일 업로드 파싱
    ],
    # 페이지네이션 설정 (선택 사항)
    # 'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    # 'PAGE_SIZE': 10,

    # 스로틀링 설정 (선택 사항)
    # 'DEFAULT_THROTTLE_CLASSES': [
    #     'rest_framework.throttling.AnonRateThrottle',
    #     'rest_framework.throttling.UserRateThrottle'
    # ],
    # 'DEFAULT_THROTTLE_RATES': {
    #     'anon': '100/day',
    #     'user': '1000/day'
    # }
}
```

---

## 3. Serializer: 데이터 변환과 유효성 검사의 핵심

시리얼라이저(Serializer)는 DRF의 핵심 구성 요소입니다. Django 모델 인스턴스나 쿼리셋과 같은 복잡한 파이썬 객체를 JSON, XML 등과 같은 파이썬 기본 데이터 타입(딕셔너리, 리스트)으로 변환(직렬화)하고, 반대로 들어오는 데이터(JSON 등)를 파이썬 객체로 변환(역직렬화)하여 유효성 검사를 수행하는 역할을 합니다.

### 3.1. `ModelSerializer`: 모델 기반 시리얼라이저

`ModelSerializer`는 Django의 `ModelForm`과 유사하게, 모델을 기반으로 시리얼라이저 필드를 자동으로 생성해줍니다. 대부분의 경우 이 클래스를 상속받아 사용합니다.

**`serializers.py` 예시:**

```python
# blog/serializers.py

from rest_framework import serializers
from .models import Post # Post 모델을 사용한다고 가정

class PostSerializer(serializers.ModelSerializer):
    # 1. 커스텀 필드 추가 예시: 작성자의 username을 포함
    author_username = serializers.SerializerMethodField()

    class Meta:
        model = Post
        # 2. 폼에 포함할 필드 지정 (명시적으로 지정하는 것이 보안상, 가독성상 권장)
        fields = ['id', 'title', 'content', 'is_published', 'created_at', 'author_username']
        # fields = '__all__' # 모든 필드를 포함 (개발 초기 단계에서만 사용 권장)
        # exclude = ['updated_at'] # 특정 필드를 제외

        # 3. 읽기 전용 필드 지정 (API 요청 시 클라이언트가 수정할 수 없음)
        read_only_fields = ['created_at', 'updated_at', 'author_username']

    # 4. SerializerMethodField에 대한 메서드 구현
    def get_author_username(self, obj):
        return obj.author.username if obj.author else None

    # 5. 단일 필드 유효성 검사 예시: 제목에 특정 단어 포함 여부
    def validate_title(self, value):
        if "비속어" in value.lower():
            raise serializers.ValidationError("제목에 부적절한 단어가 포함되어 있습니다.")
        return value

    # 6. 객체 수준 유효성 검사 예시: 시작일이 종료일보다 빠를 수 없음
    def validate(self, data):
        # data는 cleaned_data와 유사하게 유효성 검사를 통과한 필드들의 딕셔너리
        if data['created_at'] > data['updated_at']: # 예시를 위한 비현실적인 조건
            raise serializers.ValidationError("생성일은 수정일보다 빠를 수 없습니다.")
        return data

```

### 3.2. `Serializer`: 모델과 무관한 데이터 직렬화

`Serializer` 클래스는 `ModelSerializer`와 달리 특정 모델에 묶이지 않습니다. 모델이 없는 데이터(예: 로그인 요청의 사용자 이름/비밀번호)를 직렬화하거나, 복잡한 커스텀 데이터 구조를 다룰 때 사용합니다.

```python
# accounts/serializers.py (로그인 요청을 위한 시리얼라이저 예시)

from rest_framework import serializers

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True) # 응답에 포함되지 않음

    # 객체 수준 유효성 검사 (사용자 인증 로직)
    def validate(self, data):
        username = data.get('username')
        password = data.get('password')

        if username and password:
            # Django의 authenticate 함수를 사용하여 사용자 인증
            user = authenticate(request=self.context.get('request'), username=username, password=password)
            if not user:
                raise serializers.ValidationError('아이디 또는 비밀번호가 올바르지 않습니다.', code='authorization')
        else:
            raise serializers.ValidationError('아이디와 비밀번호를 모두 입력해주세요.', code='authorization')
        
        data['user'] = user # 인증된 사용자 객체를 validated_data에 추가
        return data
```

### 3.3. 시리얼라이저 사용 흐름 (뷰에서)

뷰에서는 시리얼라이저를 사용하여 요청 데이터를 검증하고, 응답 데이터를 생성합니다.

```python
# blog/views.py (APIView 예시)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Post
from .serializers import PostSerializer

class PostListCreateAPIView(APIView):
    def get(self, request, format=None):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True) # 쿼리셋은 many=True
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = PostSerializer(data=request.data) # 요청 데이터로 시리얼라이저 생성
        if serializer.is_valid(): # 유효성 검사
            serializer.save(author=request.user) # 유효성 검사 통과 후 저장 (author는 request.user로 자동 할당)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

```

---

## 4. ViewSet 및 APIView: API 엔드포인트 구현

DRF는 API 엔드포인트를 구현하는 두 가지 주요 방식을 제공합니다.

### 4.1. `APIView`: 저수준 제어

`APIView`는 Django의 `View` 클래스와 유사하지만, DRF의 `Request` 객체, `Response` 객체, 인증, 권한, 렌더링, 파싱 시스템을 활용할 수 있습니다. HTTP 메서드(GET, POST 등)에 따라 로직을 분리하여 작성합니다.

-   **언제 사용하는가?**:
    -   CRUD 패턴에 정확히 맞지 않는 커스텀 엔드포인트 (예: 로그인, 파일 업로드, 특정 계산 API).
    -   요청/응답 사이클에 대한 매우 세밀한 제어가 필요할 때.
    -   `ViewSet`의 자동 생성 URL 패턴이 적합하지 않을 때.

```python
# blog/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

class MyCustomAPIView(APIView):
    permission_classes = [IsAuthenticated] # 이 뷰는 인증된 사용자만 접근 가능

    def get(self, request, format=None):
        # request.user로 현재 인증된 사용자 접근 가능
        return Response({"message": f"Hello, {request.user.username}! This is a custom GET response."})

    def post(self, request, format=None):
        # request.data로 파싱된 요청 본문 데이터 접근
        data = request.data.get('my_field')
        if not data:
            return Response({"error": "my_field is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        # ... 비즈니스 로직 ...
        return Response({"received_data": data, "status": "processed"}, status=status.HTTP_201_CREATED)

```

### 4.2. `ViewSet`: CRUD 작업의 간소화

`ViewSet`은 `APIView`의 확장으로, 하나의 클래스에 `list`, `retrieve`, `create`, `update`, `partial_update`, `destroy`와 같은 CRUD 작업을 위한 여러 뷰 로직을 묶어 관리합니다. 라우터(Router)와 함께 사용하면 URL 설정을 간소화할 수 있습니다.

-   **언제 사용하는가?**:
    -   모델에 대한 표준 CRUD 작업을 구현할 때.
    -   URL 패턴을 자동으로 생성하고 싶을 때.

-   **주요 ViewSet 클래스**:
    -   **`viewsets.ModelViewSet`**: 가장 강력하고 흔하게 사용됩니다. `queryset`과 `serializer_class`만 지정하면 모델에 대한 모든 CRUD 작업을 자동으로 처리합니다.
    -   **`viewsets.ReadOnlyModelViewSet`**: `list`와 `retrieve` (읽기) 작업만 제공합니다.
    -   **`viewsets.GenericViewSet`**: `ViewSet`의 가장 기본적인 형태입니다. `queryset`과 `serializer_class`를 지정하지만, 실제 액션(list, create 등)은 `mixins`를 통해 직접 추가해야 합니다.

**`views.py` 예시 (ModelViewSet):**

```python
# blog/views.py

from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.decorators import action # 커스텀 액션을 위해 임포트
from rest_framework.response import Response # 커스텀 액션 응답을 위해 임포트

from .models import Post
from .serializers import PostSerializer

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    permission_classes = [IsAuthenticatedOrReadOnly] # 인증된 사용자만 쓰기 가능, 나머지는 읽기만

    # 1. 쿼리셋 커스터마이징 예시: 특정 사용자 게시글만 보이게
    def get_queryset(self):
        # 로그인한 사용자에게는 자신의 게시글만 보이도록 필터링
        if self.request.user.is_authenticated:
            return Post.objects.filter(author=self.request.user).order_by('-created_at')
        return Post.objects.filter(is_published=True).order_by('-created_at') # 비로그인 사용자에게는 공개된 게시글만

    # 2. 저장 시 작성자 자동 할당 (create, update 시)
    def perform_create(self, serializer):
        serializer.save(author=self.request.user)

    # 3. 커스텀 액션 예시: 게시글 발행/비발행 토글
    @action(detail=True, methods=['post']) # detail=True: 특정 객체에 대한 액션 (URL: /posts/{pk}/publish/)
    def publish(self, request, pk=None):
        post = self.get_object()
        post.is_published = not post.is_published # 상태 토글
        post.save()
        serializer = self.get_serializer(post) # 변경된 객체를 다시 시리얼라이즈
        return Response(serializer.data)

    # 4. 커스텀 액션 예시: 최근 게시글 목록 (detail=False: 컬렉션에 대한 액션, URL: /posts/recent_posts/)
    @action(detail=False, methods=['get'])
    def recent_posts(self, request):
        recent = Post.objects.order_by('-created_at')[:5]
        serializer = self.get_serializer(recent, many=True)
        return Response(serializer.data)

```

---

## 5. Router: ViewSet의 URL 자동 생성

라우터(Router)는 `ViewSet`을 사용하여 URL 패턴을 자동으로 생성해주는 강력한 도구입니다. 이를 통해 `urls.py` 파일의 코드를 획기적으로 줄이고, 일관된 RESTful URL 구조를 유지할 수 있습니다.

### 5.1. `DefaultRouter` 사용법

`DefaultRouter`는 `ModelViewSet`과 같은 표준 `ViewSet`에 대해 `list`, `detail`, `create`, `update`, `delete` 등의 액션에 해당하는 URL 패턴을 자동으로 생성해줍니다. 또한, 웹 브라우저블 API를 위한 루트 뷰와 포맷 서픽스 패턴도 포함합니다.

**`urls.py` 예시:**

```python
# myproject/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from blog.views import PostViewSet # 위에서 정의한 PostViewSet 임포트

# 1. DefaultRouter 인스턴스 생성
router = DefaultRouter()

# 2. ViewSet을 라우터에 등록
# 첫 번째 인자: URL prefix (예: /posts/)
# 두 번째 인자: ViewSet 클래스
# 세 번째 인자 (선택): base_name. ViewSet에 queryset 속성이 없거나,
#                      여러 ViewSet이 동일한 쿼리셋을 사용하는 경우 필요.
#                      ModelViewSet은 자동으로 model._meta.model_name을 base_name으로 사용
router.register(r'posts', PostViewSet) # /posts/ 경로에 PostViewSet 연결

urlpatterns = [
    path('admin/', admin.site.urls),
    # 3. 라우터가 생성한 URL 패턴들을 포함
    # DRF의 웹 브라우저블 API를 위한 로그인/로그아웃 URL도 함께 포함됩니다.
    path('api/', include(router.urls)),
    # path('api-auth/', include('rest_framework.urls')), # Browsable API 로그인/로그아웃을 위한 URL (선택 사항)
]
```

### 5.2. 라우터가 생성하는 URL 패턴 (예시: `PostViewSet`에 대한 `/posts/`)

`router.register(r'posts', PostViewSet)`를 통해 `DefaultRouter`는 다음과 같은 URL 패턴들을 자동으로 생성합니다.

-   **컬렉션(Collection) URL**: `/posts/`
    -   `GET`: `PostViewSet.list()` (게시글 목록 조회)
    -   `POST`: `PostViewSet.create()` (새 게시글 생성)
-   **멤버(Member) URL**: `/posts/{pk}/`
    -   `GET`: `PostViewSet.retrieve()` (특정 게시글 상세 조회)
    -   `PUT`: `PostViewSet.update()` (특정 게시글 전체 수정)
    -   `PATCH`: `PostViewSet.partial_update()` (특정 게시글 부분 수정)
    -   `DELETE`: `PostViewSet.destroy()` (특정 게시글 삭제)
-   **커스텀 액션 URL**: `@action` 데코레이터를 사용한 메서드에 대해서도 URL이 자동으로 생성됩니다.
    -   `@action(detail=True)`: `/posts/{pk}/publish/` (예: `PostViewSet.publish()`)
    -   `@action(detail=False)`: `/posts/recent_posts/` (예: `PostViewSet.recent_posts()`)

라우터를 사용하면 `urls.py`가 매우 간결해지고, API 엔드포인트의 추가/삭제/변경이 용이해져 유지보수성이 크게 향상됩니다.


## 6. 인증 및 권한 (Authentication & Permissions): API 보안의 핵심

DRF를 사용하여 API를 구축할 때, 누가 API에 접근할 수 있고 어떤 작업을 수행할 수 있는지 제어하는 것은 매우 중요합니다. DRF는 유연하고 강력한 인증(Authentication) 및 권한(Permissions) 시스템을 제공하여 API 보안을 강화하고 접근을 제어합니다.

### 6.1. DRF 인증 (Authentication): 사용자 식별

인증(Authentication)은 클라이언트의 요청에 포함된 자격 증명(credentials)을 확인하여 **사용자를 식별**하는 과정입니다. DRF는 다양한 인증 방식을 지원하며, `settings.py`의 `DEFAULT_AUTHENTICATION_CLASSES`를 통해 전역적으로 설정하거나, 각 뷰/뷰셋에서 `authentication_classes` 속성을 통해 개별적으로 설정할 수 있습니다.

**`settings.py` 예시:**
```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication', # 세션 기반 인증 (웹 브라우저용)
        'rest_framework.authentication.TokenAuthentication',   # 토큰 기반 인증 (API 클라이언트용)
        'rest_framework_simplejwt.authentication.JWTAuthentication', # JWT 인증 (설치 필요)
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated', # 기본 권한: 인증된 사용자만 접근 허용
    ]
}
```

**주요 인증 방식:**

- **`SessionAuthentication`**: Django의 세션 기반 인증 시스템을 사용합니다. 웹 브라우저를 통한 접근(DRF의 Browsable API 포함)에 적합하며, CSRF 토큰과 함께 사용됩니다.

- **`TokenAuthentication`**: 사용자별로 고유한 토큰을 발급하여 API 요청 시 `Authorization: Token <token_value>` 헤더에 포함시켜 인증합니다. 모바일 앱이나 다른 백엔드 서비스와의 연동에 널리 사용됩니다. 
    - **설정**: `INSTALLED_APPS`에 `rest_framework.authtoken`을 추가하고 마이그레이션(`python manage.py migrate`)을 실행하면 `authtoken_token` 테이블이 생성됩니다.
    - **토큰 발급 API 예시**: 클라이언트가 사용자 이름과 비밀번호를 전송하여 토큰을 발급받는 로그인 API 엔드포인트를 구현합니다.
        ```python
        # myapp/views.py (또는 accounts/views.py)
        from rest_framework.authtoken.views import ObtainAuthToken
        from rest_framework.authtoken.models import Token
        from rest_framework.response import Response
        from django.contrib.auth import authenticate # authenticate 함수 임포트

        class CustomAuthToken(ObtainAuthToken):
            def post(self, request, *args, **kwargs):
                serializer = self.serializer_class(data=request.data,
                                                   context={'request': request})
                serializer.is_valid(raise_exception=True)
                user = serializer.validated_data['user']
                token, created = Token.objects.get_or_create(user=user)
                return Response({
                    'token': token.key,
                    'user_id': user.pk,
                    'email': user.email
                })
        ```
        **사용 방법**: 클라이언트는 `POST` 요청으로 `api/token-auth/` 엔드포인트에 `username`과 `password`를 전송합니다. 성공하면 응답으로 `token` 값을 받게 됩니다. 이후 API 요청 시 이 토큰을 `Authorization: Token <token_value>` 헤더에 포함하여 인증합니다.

- **JWT (JSON Web Token) Authentication**: 클라이언트가 로그인 시 JWT를 발급받고, 이후 요청에 JWT를 포함시켜 인증합니다. 토큰 자체에 사용자 정보가 암호화되어 있어 서버가 매번 데이터베이스를 조회할 필요가 없어 확장성이 좋습니다. `djangorestframework-simplejwt` 라이브러리를 사용합니다.
    - **설정**: `pip install djangorestframework-simplejwt` 후 `INSTALLED_APPS`에 `rest_framework_simplejwt`를 추가하고 `settings.py`에 `SIMPLE_JWT` 딕셔너리를 설정합니다.
    - **토큰 발급/갱신**: `simplejwt`가 제공하는 뷰를 `urls.py`에 연결하여 토큰을 발급(`token/`)하고 갱신(`token/refresh/`)할 수 있습니다.
        ```python
        # myproject/urls.py
        from rest_framework_simplejwt.views import (TokenObtainPairView, TokenRefreshView)

        urlpatterns = [
            # ...
            path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
            path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
        ]
        ```
    - **보안 고려사항**: JWT는 토큰이 탈취되면 만료될 때까지 유효하므로, 짧은 만료 시간 설정, Refresh Token 사용, 블랙리스트 관리 등을 고려해야 합니다.

### 6.2. DRF 권한 (Permissions): 접근 제어

권한(Permissions)은 인증된 사용자가 특정 리소스에 대해 **어떤 작업을 수행할 수 있는지** 제어하는 과정입니다. `settings.py`의 `DEFAULT_PERMISSION_CLASSES`를 통해 전역적으로 설정하거나, 각 뷰/뷰셋에서 `permission_classes` 속성을 통해 개별적으로 설정할 수 있습니다.

**주요 권한 클래스:**

- **`AllowAny`**: 모든 사용자(인증 여부 무관)에게 접근을 허용합니다. (기본값)
- **`IsAuthenticated`**: 인증된 사용자에게만 접근을 허용합니다.
- **`IsAdminUser`**: 관리자(`is_staff=True`) 사용자에게만 접근을 허용합니다.
- **`IsAuthenticatedOrReadOnly`**: 인증된 사용자에게는 모든 작업(읽기/쓰기)을 허용하고, 인증되지 않은 사용자에게는 읽기(GET, HEAD, OPTIONS)만 허용합니다. **API에서 가장 흔하게 사용되는 권한입니다.**
- **`DjangoModelPermissions`**: Django의 모델 권한 시스템(`app_label.add_model`, `app_label.change_model` 등)과 연동하여 권한을 확인합니다. 주로 관리자 페이지와 연동되는 API에 사용됩니다.
- **`DjangoObjectPermissions`**: 객체 수준의 권한을 제어합니다. (예: 사용자는 자신이 작성한 게시글만 수정할 수 있다). `DjangoModelPermissions`와 함께 사용되며, `django-guardian` 같은 라이브러리와 연동하여 더 강력한 객체 권한 관리가 가능합니다.

**커스텀 권한 구현 (`IsOwnerOrReadOnly` 예시):**

내장 권한 클래스로는 부족한 복잡한 비즈니스 로직이 있을 때, `rest_framework.permissions.BasePermission`을 상속받아 커스텀 권한 클래스를 구현할 수 있습니다. 

- `has_permission(self, request, view)`: 요청이 뷰에 도달하기 전에 권한을 확인합니다. (전역 권한)
- `has_object_permission(self, request, view, obj)`: 특정 객체에 대한 접근 권한을 확인합니다. (객체 수준 권한)

```python
# blog/permissions.py
from rest_framework import permissions

class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    객체의 소유자에게만 쓰기 권한을 허용하고, 그 외 사용자에게는 읽기 권한만 허용합니다.
    """
    def has_object_permission(self, request, view, obj):
        # 읽기 권한은 모든 요청에 허용됩니다.
        if request.method in permissions.SAFE_METHODS:
            return True

        # 쓰기 권한은 객체의 소유자에게만 허용됩니다.
        return obj.author == request.user
```

**뷰/뷰셋에 권한 적용:**

`permission_classes` 속성을 통해 각 뷰나 뷰셋에 적용할 권한 클래스 리스트를 지정합니다.

```python
# blog/views.py (DRF ViewSet)
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from .permissions import IsOwnerOrReadOnly # 커스텀 권한 임포트

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    
    # 여러 권한 클래스를 리스트로 지정하면, 모든 권한을 통과해야 접근 가능합니다.
    permission_classes = [IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]

    # ... (ViewSet의 다른 로직들) ...
```
DRF를 사용하여 API를 구축할 때, 누가 API에 접근할 수 있고 어떤 작업을 수행할 수 있는지 제어하는 것은 매우 중요합니다. DRF는 유연하고 강력한 인증(Authentication) 및 권한(Permissions) 시스템을 제공합니다.

**실무적 관점:**
*   **보안 강화:** API를 무단 접근으로부터 보호하고, 민감한 데이터가 노출되는 것을 방지합니다.
*   **접근 제어:** 사용자 역할(예: 관리자, 일반 사용자)에 따라 접근 가능한 리소스와 수행 가능한 작업을 세밀하게 제어합니다.
*   **확장성:** 다양한 인증 및 권한 방식을 조합하여 복잡한 비즈니스 요구사항을 충족시킬 수 있습니다.

**DRF 인증 (Authentication):**
클라이언트의 요청에 포함된 자격 증명(credentials)을 확인하여 사용자를 식별하는 과정입니다.
*   **설정:** `settings.py`의 `REST_FRAMEWORK` 설정에 `DEFAULT_AUTHENTICATION_CLASSES`를 추가합니다.
    ```python
    # settings.py
    REST_FRAMEWORK = {
        'DEFAULT_AUTHENTICATION_CLASSES': [
            'rest_framework.authentication.SessionAuthentication', # 세션 기반 인증 (Django 기본)
            'rest_framework.authentication.TokenAuthentication',   # 토큰 기반 인증 (API에 적합)
            # 'rest_framework_simplejwt.authentication.JWTAuthentication', # JWT 인증 (설치 필요)
        ],
        'DEFAULT_PERMISSION_CLASSES': [
            'rest_framework.permissions.IsAuthenticated', # 기본 권한: 인증된 사용자만 접근 허용
        ]
    }
    ```
*   **주요 인증 방식:**
    *   **`SessionAuthentication`:** Django의 세션 기반 인증 시스템을 사용합니다. 웹 브라우저를 통한 접근(DRF의 Browsable API 포함)에 적합합니다.
    *   **`TokenAuthentication`:** 사용자별로 고유한 토큰을 발급하여 API 요청 시 `Authorization: Token <token_value>` 헤더에 포함시켜 인증합니다. 모바일 앱이나 다른 백엔드 서비스와의 연동에 널리 사용됩니다.
        *   **설정:** `INSTALLED_APPS`에 `rest_framework.authtoken`을 추가하고 마이그레이션(`python manage.py migrate`)을 실행하면 `authtoken_token` 테이블이 생성됩니다.
        *   **토큰 생성:** `User` 모델의 `post_save` 시그널을 사용하여 사용자 생성 시 자동으로 토큰을 생성하거나, 관리자 페이지에서 수동으로 생성할 수 있습니다.
  
**토큰 발급 (로그인) API 예시**

클라이언트가 사용자 이름과 비밀번호를 전송하여 토큰을 발급받는 로그인 API 엔드포인트를 구현합니다.

1.  **`rest_framework.authtoken` 앱 등록 확인:**
    `settings.py`의 `INSTALLED_APPS`에 `'rest_framework.authtoken'`이 포함되어 있어야 합니다.

2.  **토큰 발급 뷰 (`myapp/views.py` 또는 `accounts/views.py`):
    ```python
    # myapp/views.py (또는 accounts/views.py)
    from rest_framework.authtoken.views import ObtainAuthToken
    from rest_framework.authtoken.models import Token
    from rest_framework.response import Response

    class CustomAuthToken(ObtainAuthToken):
        def post(self, request, *args, **kwargs):
            serializer = self.serializer_class(data=request.data,
                                               context={'request': request})
            serializer.is_valid(raise_exception=True)
            user = serializer.validated_data['user']
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user_id': user.pk,
                'email': user.email
            })
    ```

3.  **URL 설정 (`project_name/urls.py` 또는 `myapp/urls.py`):
    ```python
    # project_name/urls.py (또는 myapp/urls.py)
    from django.urls import path
    from myapp.views import CustomAuthToken # 위에서 정의한 뷰 임포트

    urlpatterns = [
        # ...
        path('api/token-auth/', CustomAuthToken.as_view(), name='api_token_auth'),
        # ...
    ]
    ```

**사용 방법:**
클라이언트는 `POST` 요청으로 `api/token-auth/` 엔드포인트에 `username`과 `password`를 전송합니다. 성공하면 응답으로 `token` 값을 받게 됩니다. 이후 API 요청 시 이 토큰을 `Authorization: Token <token_value>` 헤더에 포함하여 인증합니다.

*   **JWT (JSON Web Token) Authentication:** 클라이언트가 로그인 시 JWT를 발급받고, 이후 요청에 JWT를 포함시켜 인증합니다. 토큰 자체에 사용자 정보가 암호화되어 있어 서버가 매번 데이터베이스를 조회할 필요가 없어 확장성이 좋습니다. `djangorestframework-simplejwt` 라이브러리를 사용합니다.

**DRF 권한 (Permissions):**
인증된 사용자가 특정 리소스에 대해 어떤 작업을 수행할 수 있는지 제어하는 과정입니다.
*   **설정:** `settings.py`의 `REST_FRAMEWORK` 설정에 `DEFAULT_PERMISSION_CLASSES`를 추가하거나, 각 뷰/뷰셋에 `permission_classes` 속성을 정의합니다.
*   **주요 권한 클래스:**
    *   **`AllowAny`:** 모든 사용자에게 접근을 허용합니다. (기본값)
    *   **`IsAuthenticated`:** 인증된 사용자에게만 접근을 허용합니다.
    *   **`IsAdminUser`:** 관리자(is_staff=True) 사용자에게만 접근을 허용합니다.
    *   **`IsAuthenticatedOrReadOnly`:** 인증된 사용자에게는 모든 작업을 허용하고, 인증되지 않은 사용자에게는 읽기(GET, HEAD, OPTIONS)만 허용합니다.
    *   **`DjangoModelPermissions` / `DjangoObjectPermissions`:** Django의 모델 권한 시스템과 연동하여 객체 수준의 권한을 제어합니다.
    *   **Custom Permissions:** `rest_framework.permissions.BasePermission`을 상속받아 특정 비즈니스 로직에 맞는 커스텀 권한 클래스를 구현할 수 있습니다 (예: `IsOwnerOrReadOnly` - 객체의 소유자만 수정/삭제 가능).

**예시 (뷰셋에 인증 및 권한 적용):**
```python
# blog/views.py (DRF ViewSet)
from rest_framework import viewsets, permissions
from .models import Blog
from .serializers import BlogSerializer

class BlogViewSet(viewsets.ModelViewSet):
    queryset = Blog.objects.all()
    serializer_class = BlogSerializer
    # 인증된 사용자만 접근 가능하며, 객체 소유자만 수정/삭제 가능하도록 커스텀 권한 적용
    permission_classes = [permissions.IsAuthenticatedOrReadOnly] # 또는 CustomPermission
```

## 7. 추가 보안 고려사항: 안전한 API 구축을 위한 필수 지식

Django는 많은 보안 기능을 내장하고 있지만, API를 개발할 때는 몇 가지 추가적인 보안 위협을 인지하고 적절히 대응해야 합니다. 보안은 일회성 설정이 아니라 지속적인 관심과 노력이 필요한 영역입니다.

### 7.1. CORS (Cross-Origin Resource Sharing): 프론트엔드와의 안전한 통신

-   **문제점**: 웹 브라우저는 보안상의 이유로 **동일 출처 정책(Same-Origin Policy)**을 따릅니다. 이는 웹 페이지가 자신과 동일한 도메인, 프로토콜, 포트에서 로드된 리소스만 접근할 수 있도록 제한합니다. 따라서 프론트엔드(예: React, Vue 앱)가 백엔드(Django API)와 다른 도메인에서 실행될 때, 브라우저는 기본적으로 API 요청을 차단합니다.
-   **해결책**: **CORS(Cross-Origin Resource Sharing)**는 서버가 특정 출처(Origin)의 요청을 허용하도록 브라우저에 알려주는 메커니즘입니다. `django-cors-headers` 라이브러리를 사용하여 쉽게 설정할 수 있습니다.

    1.  **설치**: `pip install django-cors-headers`
    2.  **`settings.py` 설정**:
        ```python
        # settings.py
        INSTALLED_APPS = [
            # ...
            'corsheaders',
            # ...
        ]

        MIDDLEWARE = [
            'corsheaders.middleware.CorsMiddleware', # 가장 위에 위치하는 것이 좋음
            'django.middleware.security.SecurityMiddleware',
            # ... 다른 미들웨어 ...
        ]

        # 개발 환경에서 모든 도메인에서의 접근 허용 (운영 환경에서는 절대 사용 금지!)
        # CORS_ALLOW_ALL_ORIGINS = True

        # 운영 환경 권장: 특정 도메인만 허용
        CORS_ALLOWED_ORIGINS = [
            "https://your-frontend-domain.com",
            "https://another-allowed-domain.com",
            "http://localhost:3000", # 개발 환경 프론트엔드
        ]

        # 정규 표현식을 사용하여 유연하게 허용 (예: 모든 서브도메인 허용)
        # CORS_ALLOWED_ORIGIN_REGEXES = [
        #     r"^https://\w+\.your-domain\.com$",
        # ]

        # 자격 증명(쿠키, HTTP 인증)을 포함한 요청 허용 여부
        # True로 설정 시 CORS_ALLOWED_ORIGINS에 '*' 사용 불가
        CORS_ALLOW_CREDENTIALS = True

        # 허용할 HTTP 메서드 (기본값은 모든 메서드 허용)
        # CORS_ALLOW_METHODS = [
        #     'DELETE',
        #     'GET',
        #     'OPTIONS',
        #     'PATCH',
        #     'POST',
        #     'PUT',
        # ]

        # 허용할 HTTP 헤더 (기본값은 모든 헤더 허용)
        # CORS_ALLOW_HEADERS = [
        #     'accept',
        #     'accept-encoding',
        #     'authorization',
        #     'content-type',
        #     'dnt',
        #     'origin',
        #     'user-agent',
        #     'x-csrftoken',
        #     'x-requested-with',
        # ]

        # 브라우저가 접근할 수 있도록 허용할 커스텀 응답 헤더
        # CORS_EXPOSE_HEADERS = ['X-Custom-Header']
        ```

### 7.2. XSS (Cross-Site Scripting) 방지: 사용자 입력의 위험성

-   **문제점**: XSS는 공격자가 웹사이트에 악성 스크립트(예: `<script>alert('해킹!')</script>`)를 삽입하여 다른 사용자의 세션을 탈취하거나 정보를 유출하는 공격입니다.
-   **Django의 방어**: Django 템플릿 시스템은 기본적으로 모든 변수를 HTML 엔티티로 자동 이스케이프(escape)하여 XSS 공격을 방지합니다. (예: `<`는 `&lt;`로, `>`는 `&gt;`로 변환)
-   **`|safe` 필터의 위험성**: `{{ user_input|safe }}`와 같이 `|safe` 필터를 사용하면 이스케이프를 비활성화하고 HTML을 그대로 렌더링합니다. **절대로 신뢰할 수 없는 사용자 입력에 `|safe`를 사용해서는 안 됩니다.** `|safe`는 관리자가 입력한 위지윅 에디터 콘텐츠처럼 **출처가 명확하고 안전하다고 검증된 HTML에만 제한적으로 사용**해야 합니다.
-   **추가 방어**: Content Security Policy (CSP)를 설정하여 브라우저가 로드할 수 있는 리소스(스크립트, 스타일 등)의 출처를 제한하는 것도 강력한 XSS 방어 방법입니다. `django-csp`와 같은 라이브러리를 활용할 수 있습니다.

### 7.3. SQL Injection 방지: 데이터베이스 무결성 보호

-   **문제점**: SQL Injection은 공격자가 사용자 입력 필드에 악의적인 SQL 코드를 삽입하여 데이터베이스를 조작하거나 민감 정보를 탈취하는 공격입니다.
-   **Django ORM의 방어**: Django ORM은 기본적으로 모든 쿼리 파라미터를 안전하게 이스케이프 처리하므로, ORM을 사용하는 한 SQL Injection으로부터 안전합니다.
-   **Raw 쿼리 사용 시 주의점**: `django.db.connection.cursor()`를 사용하여 Raw SQL 쿼리를 직접 실행할 때는 개발자가 직접 SQL Injection을 방지해야 합니다. **절대로 사용자 입력 값을 SQL 쿼리 문자열에 직접 삽입하지 마세요.**

    **안전한 예시 (파라미터 바인딩 사용):**
    ```python
    from django.db import connection

    def get_user_data(username):
        with connection.cursor() as cursor:
            # 사용자 입력 값을 직접 쿼리 문자열에 넣지 않고, 두 번째 인자로 전달
            cursor.execute("SELECT * FROM myapp_user WHERE username = %s", [username])
            row = cursor.fetchone()
        return row
    ```

    **위험한 예시 (SQL Injection 취약! 절대 사용 금지):**
    ```python
    # 절대 이렇게 사용하지 마세요!
    def get_user_data_vulnerable(username):
        with connection.cursor() as cursor:
            # 사용자 입력이 쿼리 문자열에 직접 삽입되어 SQL Injection에 취약
            cursor.execute(f"SELECT * FROM myapp_user WHERE username = '{username}'")
            row = cursor.fetchone()
        return row
    ```

### 7.4. 기타 중요한 보안 고려사항

-   **HTTPS 강제**: 모든 통신은 HTTPS를 통해 암호화되어야 합니다. `settings.py`에서 `SECURE_SSL_REDIRECT = True`를 설정하여 모든 HTTP 요청을 HTTPS로 리다이렉트하고, `SECURE_HSTS_SECONDS`를 설정하여 HSTS(HTTP Strict Transport Security)를 활성화합니다.
-   **CSRF 보호**: Django는 폼과 `SessionAuthentication`을 사용하는 API에 대해 CSRF 보호를 기본으로 제공합니다. `{% csrf_token %}` 태그를 모든 POST 폼에 포함하고, `CsrfViewMiddleware`가 활성화되어 있는지 확인합니다.
-   **Clickjacking 방지**: `X-Frame-Options` 헤더를 사용하여 웹사이트가 `<iframe>`, `<frame>`, `<object>` 태그 내에서 로드되는 것을 방지합니다. `XFrameOptionsMiddleware`가 기본적으로 활성화되어 있습니다.
-   **비밀번호 해싱**: Django는 사용자 비밀번호를 안전하게 해싱하여 저장합니다. `settings.py`의 `PASSWORD_HASHERS`를 통해 해싱 알고리즘을 설정할 수 있으며, 기본값은 강력한 알고리즘을 사용합니다.
-   **민감 데이터 로깅 방지**: 로그에 사용자 비밀번호, API 키, 개인 식별 정보(PII) 등 민감한 데이터가 기록되지 않도록 주의합니다. 로깅 설정을 신중하게 검토해야 합니다.
-   **의존성 관리 및 업데이트**: 프로젝트에 사용되는 모든 서드파파 라이브러리(DRF 포함)를 최신 상태로 유지하고, 알려진 보안 취약점이 있는지 주기적으로 확인합니다. `pip-audit`와 같은 도구를 활용할 수 있습니다.
-   **Rate Limiting / Throttling**: API 엔드포인트에 대한 요청 속도를 제한하여 무차별 대입 공격(Brute-force attack)이나 서비스 거부(DoS) 공격을 방지합니다. DRF는 `DEFAULT_THROTTLE_CLASSES`를 통해 스로틀링 기능을 제공합니다.

이러한 보안 고려사항들을 철저히 적용하여 Django API를 더욱 견고하고 안전하게 구축할 수 있습니다.

## 8. DRF 심화: 복잡한 API 구축을 위한 고급 기법

DRF는 강력한 API 구축 도구이며, 다음 주제들을 학습하면 더욱 효율적이고 확장 가능한 API 개발이 가능합니다.

### 8.1. Nested Serializers (중첩 시리얼라이저): 관계형 데이터의 표현

-   **문제점**: 모델 간에 관계(ForeignKey, ManyToManyField 등)가 있을 때, 기본 시리얼라이저는 관련 객체의 ID만 표시하거나, `depth` 옵션을 사용하면 읽기 전용으로만 중첩됩니다. 하지만 API에서는 관련 객체의 상세 정보를 함께 표시하거나, 한 번의 요청으로 관계된 객체를 함께 생성/수정해야 하는 경우가 많습니다.
-   **해결책**: 중첩 시리얼라이저를 사용하여 관계된 모델의 데이터를 메인 시리얼라이저 내부에 포함시킵니다.

#### 8.1.1. 읽기 전용 중첩 시리얼라이저

관련 객체의 데이터를 표시만 할 때 사용합니다. `depth` 옵션은 간단하지만, 표시할 필드를 세밀하게 제어하기 어렵습니다.

```python
# blog/serializers.py

# 1. 관련 모델(User)을 위한 시리얼라이저 정의
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User # Django의 기본 User 모델 또는 Custom User 모델
        fields = ['username', 'email']

# 2. Post 시리얼라이저 내부에 UserSerializer를 중첩
class PostSerializer(serializers.ModelSerializer):
    author = UserSerializer(read_only=True) # read_only=True로 설정하여 쓰기 방지

    class Meta:
        model = Post
        fields = ['id', 'title', 'content', 'author', 'created_at']

# 응답 예시:
# {"id": 1, "title": "첫 게시글", "content": "내용", "author": {"username": "testuser", "email": "test@example.com"}, ...}
```

#### 8.1.2. 쓰기 가능한 중첩 시리얼라이저

한 번의 API 요청으로 부모 객체와 자식 객체를 함께 생성하거나 수정해야 할 때 사용합니다. 이 경우, 부모 시리얼라이저의 `create()` 또는 `update()` 메서드를 오버라이드하여 중첩된 데이터를 수동으로 처리해야 합니다.

```python
# blog/serializers.py

# Comment 모델을 위한 시리얼라이저 (Post와 author는 부모에서 처리)
class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ['content'] # Post와 author는 부모 시리얼라이저에서 설정

class PostCreateUpdateSerializer(serializers.ModelSerializer):
    comments = CommentSerializer(many=True) # many=True로 여러 댓글 허용

    class Meta:
        model = Post
        fields = ['title', 'content', 'comments']

    # create() 메서드를 오버라이드하여 중첩된 comments 데이터 처리
    def create(self, validated_data):
        comments_data = validated_data.pop('comments') # comments 데이터 분리
        post = Post.objects.create(**validated_data) # Post 객체 생성

        for comment_data in comments_data:
            # Comment 객체 생성 시 post와 author 필드 설정
            Comment.objects.create(post=post, author=self.context['request'].user, **comment_data)
        return post

    # update() 메서드도 유사하게 오버라이드하여 중첩된 데이터 수정 로직 구현
    # def update(self, instance, validated_data):
    #     comments_data = validated_data.pop('comments')
    #     # ... instance 업데이트 로직 ...
    #     # ... comments_data 처리 로직 ...
    #     return instance
```

### 8.2. API 문서 자동화: OpenAPI (Swagger/Redoc)

API 문서는 프론트엔드 개발자, 외부 서비스 연동, 그리고 백엔드 개발자 자신에게도 필수적입니다. `drf-spectacular`와 같은 라이브러리를 사용하면 DRF API를 기반으로 OpenAPI(Swagger) 문서를 자동으로 생성하고, 이를 시각적으로 탐색할 수 있는 UI를 제공합니다.

-   **장점**: API 명세의 일관성 유지, 개발자와 클라이언트 간의 협업 효율성 증대, API 테스트 및 디버깅 편의성 제공.

-   **`drf-spectacular` 설정 방법**:

    1.  **설치**: `pip install drf-spectacular`
    2.  **`settings.py` 설정**:
        ```python
        # settings.py
        INSTALLED_APPS = [
            # ...
            'drf_spectacular',
        ]

        REST_FRAMEWORK = {
            # ...
            'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
        }

        SPECTACULAR_SETTINGS = {
            'TITLE': 'My Awesome Django API', # API 문서 제목
            'DESCRIPTION': '나만의 멋진 Django API 문서입니다.', # API 문서 설명
            'VERSION': '1.0.0', # API 버전
            'SERVE_INCLUDE_SCHEMA': False, # 스키마 파일을 별도로 제공할지 여부
            # 기타 다양한 설정 옵션 (인증, 태그, 응답 예시 등)
        }
        ```
    3.  **`urls.py` 설정**: API 문서와 UI를 위한 URL을 추가합니다.
        ```python
        # myproject/urls.py
        from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView

        urlpatterns = [
            # ...
            # OpenAPI 스키마 파일 제공 (JSON/YAML)
            path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
            # Swagger UI (대화형 API 문서)
            path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
            # Redoc UI (깔끔한 API 문서)
            path('api/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
        ]
        ```

### 8.3. 필터링, 검색, 페이지네이션: 대규모 데이터 처리

DRF는 대규모 데이터를 효율적으로 처리하고 클라이언트가 원하는 데이터를 쉽게 찾을 수 있도록 다양한 기능을 제공합니다.

#### 8.3.1. 필터링 (`django-filter`)

-   **`django-filter`**: DRF와 함께 사용되는 가장 강력한 필터링 라이브러리입니다. 복잡한 필터링 조건을 쉽게 정의할 수 있습니다.

    1.  **설치**: `pip install django-filter`
    2.  **`settings.py` 설정**: `INSTALLED_APPS`에 `django_filters`를 추가합니다.
    3.  **`filters.py` 정의**: 필터링 로직을 담을 `FilterSet` 클래스를 정의합니다.
        ```python
        # blog/filters.py
        import django_filters
        from .models import Post

        class PostFilter(django_filters.FilterSet):
            # 제목에 특정 문자열이 포함된 게시글 (대소문자 구분 안 함)
            title = django_filters.CharFilter(lookup_expr='icontains')
            # 특정 작성자의 게시글
            author = django_filters.CharFilter(field_name='author__username', lookup_expr='exact')
            # 특정 연도에 작성된 게시글
            created_year = django_filters.NumberFilter(field_name='created_at__year')

            class Meta:
                model = Post
                fields = ['title', 'author', 'is_published', 'created_year']
        ```
    4.  **`views.py`에 적용**: `filter_backends`와 `filterset_class`를 지정합니다.
        ```python
        # blog/views.py
        from django_filters.rest_framework import DjangoFilterBackend
        from .filters import PostFilter # 위에서 정의한 필터 임포트

        class PostViewSet(viewsets.ModelViewSet):
            queryset = Post.objects.all()
            serializer_class = PostSerializer
            filter_backends = [DjangoFilterBackend] # 필터 백엔드 활성화
            filterset_class = PostFilter # 사용할 필터셋 클래스 지정
        ```
    5.  **사용 예시**: `/posts/?title=django&author=testuser&created_year=2024`

#### 8.3.2. 검색 (`SearchFilter`)

-   **`SearchFilter`**: 특정 필드에서 키워드 검색을 수행할 때 사용합니다. `search_fields`에 검색 대상 필드를 지정합니다.

    ```python
    # blog/views.py
    from rest_framework.filters import SearchFilter

    class PostViewSet(viewsets.ModelViewSet):
        queryset = Post.objects.all()
        serializer_class = PostSerializer
        filter_backends = [SearchFilter] # 검색 필터 백엔드 활성화
        search_fields = ['title', 'content', 'author__username'] # 검색 대상 필드 지정
    ```
    -   **사용 예시**: `/posts/?search=키워드`

#### 8.3.3. 페이지네이션 (Pagination)

대량의 데이터를 여러 페이지로 나누어 클라이언트에게 제공합니다. DRF는 다양한 페이지네이션 스타일을 지원합니다.

-   **전역 설정**: `settings.py`의 `REST_FRAMEWORK`에서 `DEFAULT_PAGINATION_CLASS`와 `PAGE_SIZE`를 설정하여 모든 뷰셋에 적용할 수 있습니다.
    ```python
    # settings.py
    REST_FRAMEWORK = {
        # ...
        'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
        'PAGE_SIZE': 10, # 한 페이지당 10개 항목
    }
    ```
-   **뷰셋별 설정**: 각 뷰셋에서 `pagination_class` 속성을 통해 전역 설정을 오버라이드하거나, 특정 페이지네이션 클래스를 지정할 수 있습니다.
    ```python
    # blog/views.py
    from rest_framework.pagination import LimitOffsetPagination

    class CustomPostPagination(LimitOffsetPagination):
        default_limit = 5
        max_limit = 20

    class PostViewSet(viewsets.ModelViewSet):
        queryset = Post.objects.all()
        serializer_class = PostSerializer
        pagination_class = CustomPostPagination # 커스텀 페이지네이션 클래스 적용
    ```
-   **주요 페이지네이션 클래스**: `PageNumberPagination` (페이지 번호), `LimitOffsetPagination` (offset/limit), `CursorPagination` (커서 기반, 대규모 데이터셋에 적합)

### 8.4. API 버저닝 (Versioning): API의 진화 관리

API는 시간이 지남에 따라 변경되고 발전합니다. 하위 호환성을 깨는 변경(breaking change)이 발생할 경우, 기존 클라이언트의 동작을 보장하기 위해 API 버저닝 전략을 사용합니다.

-   **DRF 버저닝 방식**: `settings.py`의 `DEFAULT_VERSIONING_CLASS`를 통해 전역적으로 설정합니다.
    -   **`URLPathVersioning` (가장 흔함)**: URL 경로에 버전을 포함합니다. (예: `/api/v1/posts/`, `/api/v2/posts/`)
        ```python
        # settings.py
        REST_FRAMEWORK = {
            # ...
            'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
            'ALLOWED_VERSIONS': ['v1', 'v2'],
            'DEFAULT_VERSION': 'v1',
            'VERSION_PARAM': 'version',
        }
        # urls.py
        urlpatterns = [
            path('api/<str:version>/', include(router.urls)),
        ]
        ```
    -   **`NamespaceVersioning`**: URL 네임스페이스를 사용합니다.
    -   **`QueryParameterVersioning`**: 쿼리 파라미터에 버전을 포함합니다. (예: `/api/posts/?version=v1`)
    -   **`HeaderVersioning`**: HTTP 헤더에 버전을 포함합니다. (예: `Accept: application/json; version=v1`)

API 버저닝은 API의 안정적인 진화를 관리하고, 클라이언트와의 호환성을 유지하는 데 필수적인 전략입니다.
