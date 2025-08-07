<h2>Django Backend: REST Framework API 개발</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django REST Framework(DRF)를 사용하여 강력하고 유연한 RESTful API를 구축하는 방법을 학습하는 것을 목표로 합니다. DRF의 핵심 구성 요소인 Serializer, ViewSet, Router의 개념을 이해하고, API 보안을 위한 인증 및 권한 부여 시스템을 다룹니다.</p>

<h2>목차</h2> 

- [1. RESTful API (Django REST Framework)](#1-restful-api-django-rest-framework)
  - [1.1. DRF 설치 및 설정](#11-drf-설치-및-설정)
  - [1.2. Serializer](#12-serializer)
  - [1.3. ViewSet 및 APIView](#13-viewset-및-apiview)
  - [1.4. Router](#14-router)
  - [1.5. 인증 및 권한 (Authentication & Permissions)](#15-인증-및-권한-authentication--permissions)
  - [1.6. 추가 보안 고려사항 (CORS, XSS, SQL Injection 방지)](#16-추가-보안-고려사항-cors-xss-sql-injection-방지)
    - [1.6.1. CORS (Cross-Origin Resource Sharing)](#161-cors-cross-origin-resource-sharing)
    - [1.6.2. XSS (Cross-Site Scripting) 방지](#162-xss-cross-site-scripting-방지)
    - [1.6.3. SQL Injection 방지 (Raw 쿼리 사용 시 주의점)](#163-sql-injection-방지-raw-쿼리-사용-시-주의점)
  - [1.7. DRF 심화](#17-drf-심화)
    - [1.7.1. Nested Serializers (중첩 시리얼라이저)](#171-nested-serializers-중첩-시리얼라이저)
    - [1.7.2. API 문서 자동화](#172-api-문서-자동화)

---

## 1. RESTful API (Django REST Framework)

Django REST Framework (DRF)는 Django 위에 RESTful API를 쉽게 구축할 수 있도록 돕는 강력하고 유연한 툴킷입니다. 웹 브라우저블 API, 인증 및 권한, 시리얼라이저 등을 제공합니다.

- **왜 `JsonResponse` 대신 DRF를 사용하는가?**
    - Django의 내장 `JsonResponse`만으로도 간단한 API를 만들 수 있지만, 프로젝트 규모가 커지면 여러 문제에 직면하게 됩니다.
    - **반복적인 작업:** 데이터 직렬화(모델 인스턴스 -> JSON), 유효성 검사, 인증 및 권한 확인 등 API 개발에 필요한 공통적인 작업들을 매번 수동으로 구현해야 합니다.
    - **유지보수의 어려움:** 일관된 구조 없이 API를 개발하면 코드의 유지보수가 어려워지고, API 명세를 관리하기도 힘듭니다.
    - **DRF는 이러한 문제들을 해결해줍니다.**
        - **시리얼라이저(Serializer):** 데이터 변환 및 유효성 검사를 자동화합니다.
        - **인증/권한 시스템:** 토큰 기반 인증, JWT, OAuth2 등 다양한 인증 방식과 유연한 권한 제어 기능을 제공합니다.
        - **자동 API 문서:** API 엔드포인트를 자동으로 문서화하여 프론트엔드 개발자와의 협업을 원활하게 합니다.
        - **웹 브라우저블 API:** 개발자가 브라우저에서 직접 API를 테스트하고 디버깅할 수 있는 편리한 UI를 제공합니다.
    - 따라서, 본격적인 API 서버를 구축할 때는 DRF를 사용하는 것이 생산성, 유지보수성, 확장성 모든 면에서 훨씬 효율적입니다.

### 1.1. DRF 설치 및 설정
DRF를 사용하려면 먼저 설치하고 `settings.py`에 등록해야 합니다.

```bash
pip install djangorestframework
```

**`settings.py` 예시 (가상):
```python
INSTALLED_APPS = [
    # ...
    'rest_framework',
]
```

### 1.2. Serializer
시리얼라이저(Serializer)는 Django 모델 인스턴스나 쿼리셋을 JSON, XML 등과 같은 파이썬 데이터 타입으로 변환하고, 반대로 들어오는 데이터를 파이썬 객체로 변환하여 유효성 검사를 수행하는 역할을 합니다.

**예시 (가상 `blog/serializers.py`):
```python
from rest_framework import serializers
from .models import Blog

class BlogSerializer(serializers.ModelSerializer):
    class Meta:
        model = Blog
        fields = ['id', 'title', 'contents', 'writer', 'wdate', 'hit']
```

### 1.3. ViewSet 및 APIView
*   **`APIView`**: Django의 `View` 클래스와 유사하지만, RESTful API에 특화된 기능을 제공합니다. HTTP 메서드(GET, POST 등)에 따라 로직을 분리할 수 있습니다.
*   **`ViewSet`**: `APIView`의 확장으로, CRUD(Create, Retrieve, Update, Delete) 작업을 위한 여러 뷰 로직을 하나의 클래스에 묶어 관리합니다. 라우터와 함께 사용하면 URL 설정을 간소화할 수 있습니다.

**예시 (가상 `blog/views.py` - ViewSet):
```python
from rest_framework import viewsets
from .models import Blog
from .serializers import BlogSerializer

class BlogViewSet(viewsets.ModelViewSet):
    queryset = Blog.objects.all()
    serializer_class = BlogSerializer
```

### 1.4. Router
라우터(Router)는 `ViewSet`을 사용하여 URL 패턴을 자동으로 생성해주는 도구입니다. 이를 통해 `urls.py` 파일의 코드를 줄일 수 있습니다.

**예시 (가상 `config/urls.py`):
```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from blog.views import BlogViewSet

router = DefaultRouter()
router.register(r'blogs', BlogViewSet) # /blogs/ 경로에 BlogViewSet 연결

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)), # DRF 라우터 URL 포함
]
```

### 1.5. 인증 및 권한 (Authentication & Permissions)
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
        * #### 토큰 발급 (로그인) API 예시

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

**예시 (뷰셋에 인증 및 권한 적용):
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

### 1.6. 추가 보안 고려사항 (CORS, XSS, SQL Injection 방지)

Django는 많은 보안 기능을 내장하고 있지만, 개발자가 추가적으로 고려해야 할 사항들이 있습니다.

#### 1.6.1. CORS (Cross-Origin Resource Sharing)

프론트엔드(React, Vue 등)가 백엔드(Django)와 다른 도메인에서 실행될 때, 브라우저의 보안 정책으로 인해 요청이 차단될 수 있습니다. 이를 해결하기 위해 CORS 설정을 해야 합니다.

1.  **`django-cors-headers` 설치:**
    ```bash
    pip install django-cors-headers
    ```

2.  **`settings.py` 설정:**
    ```python
    # settings.py
    INSTALLED_APPS = [
        # ...
        'corsheaders',
        # ...
    ]

    MIDDLEWARE = [
        'corsheaders.middleware.CorsMiddleware', # 가장 위에 위치하는 것이 좋음
        # ... 다른 미들웨어 ...
    ]

    # 모든 도메인에서의 접근 허용 (개발 시 편리, 운영 시에는 특정 도메인만 허용 권장)
    CORS_ALLOW_ALL_ORIGINS = True

    # 특정 도메인만 허용하는 경우 (운영 환경 권장)
    # CORS_ALLOWED_ORIGINS = [
    #     "https://example.com",
    #     "https://sub.example.com",
    #     "http://localhost:3000", # 개발 환경
    # ]

    # 자격 증명(쿠키, HTTP 인증)을 포함한 요청 허용 여부
    CORS_ALLOW_CREDENTIALS = True
    ```

#### 1.6.2. XSS (Cross-Site Scripting) 방지

XSS는 공격자가 웹사이트에 악성 스크립트를 삽입하여 사용자 세션을 탈취하거나 정보를 유출하는 공격입니다. Django 템플릿 시스템은 기본적으로 XSS를 방지하기 위해 모든 변수를 자동으로 이스케이프(escape)합니다.

*   **주의:** `|safe` 필터를 사용하면 이스케이프를 비활성화하므로, 사용자로부터 입력받은 데이터를 `|safe`와 함께 출력할 때는 매우 신중해야 합니다. 신뢰할 수 있는 소스의 HTML만 `|safe`로 출력해야 합니다.

#### 1.6.3. SQL Injection 방지 (Raw 쿼리 사용 시 주의점)

Django ORM은 SQL Injection 공격을 자동으로 방지합니다. 하지만 `django.db.connection.cursor()`를 사용하여 Raw SQL 쿼리를 직접 실행할 때는 개발자가 직접 SQL Injection을 방지해야 합니다.

*   **절대 문자열 포맷팅 사용 금지:** 사용자 입력 값을 SQL 쿼리 문자열에 직접 삽입하지 마세요.
*   **파라미터 바인딩 사용:** `cursor.execute()` 메서드의 두 번째 인자로 파라미터를 튜플이나 딕셔너리 형태로 전달하여 데이터베이스 드라이버가 안전하게 값을 바인딩하도록 해야 합니다.

**안전한 예시:**
```python
from django.db import connection

def get_user_data(username):
    with connection.cursor() as cursor:
        # 사용자 입력 값을 직접 쿼리 문자열에 넣지 않고, 파라미터로 전달
        cursor.execute("SELECT * FROM myapp_user WHERE username = %s", [username])
        row = cursor.fetchone()
    return row
```

**위험한 예시 (SQL Injection 취약):**
```python
# 절대 이렇게 사용하지 마세요!
def get_user_data_vulnerable(username):
    with connection.cursor() as cursor:
        # 사용자 입력이 쿼리 문자열에 직접 삽입되어 SQL Injection에 취약
        cursor.execute(f"SELECT * FROM myapp_user WHERE username = '{username}'")
        row = cursor.fetchone()
    return row
```

### 1.7. DRF 심화

Django REST Framework는 강력한 API 구축 도구이며, 다음 주제들을 학습하면 더욱 효율적인 API 개발이 가능합니다.

#### 1.7.1. Nested Serializers (중첩 시리얼라이저)

*   관계가 있는 모델(예: `Question`과 `Answer`, `User`와 `Profile`)을 하나의 API 응답에서 중첩된 형태로 표현하거나, 한 번의 요청으로 관계된 객체를 함께 생성/수정할 때 사용합니다.
*   **예시:** 게시글 목록을 조회할 때 각 게시글의 작성자 정보(이름, 이메일 등)를 함께 포함시키거나, 주문 생성 시 주문 상세 항목들을 함께 받는 경우.

#### 1.7.2. API 문서 자동화

*   `drf-yasg` 또는 `drf-spectacular`와 같은 라이브러리를 사용하여 DRF API를 기반으로 OpenAPI(Swagger) 문서를 자동으로 생성할 수 있습니다.
*   **장점:** 개발자와 클라이언트(프론트엔드 개발자, 외부 서비스) 간의 API 명세 공유를 용이하게 하고, API 테스트 및 디버깅을 위한 UI를 제공하여 협업 효율성을 크게 향상시킵니다.
