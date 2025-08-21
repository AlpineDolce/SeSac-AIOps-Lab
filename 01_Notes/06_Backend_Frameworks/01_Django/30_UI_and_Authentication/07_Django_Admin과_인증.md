<h2>Django Backend: Admin과 인증</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django의 강력한 관리자 페이지(Admin)를 커스터마이징하는 방법과, 내장된 사용자 인증 및 권한 부여 시스템을 활용하여 안전한 웹 애플리케이션을 구축하는 방법을 학습하는 것을 목표로 합니다.</p>

<h2>목차</h2> 

- [1. Django Admin (관리자 페이지): 실무 가이드](#1-django-admin-관리자-페이지-실무-가이드)
  - [1.1. 관리자 계정 생성 (`createsuperuser`)](#11-관리자-계정-생성-createsuperuser)
  - [1.2. 모델 등록과 기본 `ModelAdmin` 커스터마이징](#12-모델-등록과-기본-modeladmin-커스터마이징)
  - [1.3. 인라인(Inline) 모델 관리: 관련 객체를 한 화면에서](#13-인라인inline-모델-관리-관련-객체를-한-화면에서)
  - [1.4. 관리자 액션 (Admin Actions): 일괄 작업 자동화](#14-관리자-액션-admin-actions-일괄-작업-자동화)
  - [1.5. 관리자 페이지 커스터마이징 심화](#15-관리자-페이지-커스터마이징-심화)
- [2. 사용자 인증 및 권한 부여: 실무 가이드](#2-사용자-인증-및-권한-부여-실무-가이드)
  - [2.1. Custom User 모델: 왜 필요한가?](#21-custom-user-모델-왜-필요한가)
  - [2.2. 사용자 인증 (Authentication)](#22-사용자-인증-authentication)
    - [2.2.1. 로그인/로그아웃 뷰](#221-로그인로그아웃-뷰)
    - [2.2.2. 로그인 폼 커스터마이징](#222-로그인-폼-커스터마이징)
    - [2.2.3. 사용자 등록 (회원가입)](#223-사용자-등록-회원가입)
  - [2.3. 권한 부여 (Authorization)](#23-권한-부여-authorization)
    - [2.3.1. 권한 및 그룹 관리](#231-권한-및-그룹-관리)
    - [2.3.2. 뷰에 권한 적용하기](#232-뷰에-권한-적용하기)
    - [2.3.3. 템플릿에서 권한 확인하기](#233-템플릿에서-권한-확인하기)

---

## 1. Django Admin (관리자 페이지): 실무 가이드

Django Admin은 개발자가 모델 데이터를 손쉽게 관리하고 조작할 수 있도록 자동으로 생성되는 강력한 웹 인터페이스입니다. 개발 초기 단계에서 데이터 확인 및 테스트, 그리고 운영 단계에서 콘텐츠 관리나 사용자 관리 등 백오피스(Back-office) 기능으로 매우 유용하게 활용됩니다.

### 1.1. 관리자 계정 생성 (`createsuperuser`)

Django 관리자 페이지에 접근하려면 먼저 관리자 권한을 가진 사용자 계정(superuser)이 필요합니다. `createsuperuser` 명령어는 이러한 슈퍼유저 계정을 생성하는 가장 기본적인 방법입니다.

**명령어 실행:**

프로젝트의 루트 디렉토리에서 다음 명령어를 실행합니다.

```bash
python manage.py createsuperuser
```

**계정 정보 입력:**

명령어를 실행하면 다음과 같은 정보를 순서대로 입력하라는 프롬프트가 나타납니다.

1.  **Username (사용자 이름):** 관리자 페이지에 로그인할 때 사용할 아이디입니다. (예: `admin`, `superuser`)
2.  **Email address (이메일 주소):** 관리자 계정의 이메일 주소입니다. (선택 사항이지만 입력하는 것을 권장합니다.)
3.  **Password (비밀번호):** 관리자 계정의 비밀번호입니다. 입력 시 화면에 표시되지 않으므로 주의해서 입력해야 합니다.
4.  **Password (again):** 비밀번호를 한 번 더 입력하여 확인합니다.

**예시:**

```
(myenv) C:\path\to\myproject> python manage.py createsuperuser

Username: admin
Email address: admin@example.com
Password: 
Password (again): 
Superuser created successfully.
```

**주의사항:**

*   **비밀번호 보안:** 개발 환경에서는 간단한 비밀번호를 사용할 수 있지만, 실제 운영 환경에서는 반드시 강력하고 복잡한 비밀번호를 사용해야 합니다. Django는 기본적으로 비밀번호의 복잡성을 검사합니다.
*   **최초 생성:** 일반적으로 프로젝트를 처음 설정할 때 한 번만 실행하여 초기 관리자 계정을 생성합니다. 추가적인 관리자 계정은 Django 관리자 페이지 내에서 생성하거나, 필요에 따라 이 명령어를 다시 실행할 수 있습니다.

**관리자 페이지 접근:**

슈퍼유저 계정을 생성한 후, 개발 서버를 실행하고 웹 브라우저에서 `/admin/` 경로로 접속하여 로그인할 수 있습니다.

```bash
python manage.py runserver
```

브라우저에서 `http://127.0.0.1:8000/admin/` (또는 설정된 포트)로 접속한 후, 위에서 생성한 사용자 이름과 비밀번호로 로그인하면 Django 관리자 페이지에 접근할 수 있습니다.

---

### 1.2. 모델 등록과 기본 `ModelAdmin` 커스터마이징

관리자 페이지에서 모델을 관리하려면 해당 모델을 `admin.py` 파일에 등록해야 합니다. 단순히 `admin.site.register()`만 사용하는 대신, `ModelAdmin` 클래스를 함께 사용하여 관리자 인터페이스를 세밀하게 제어하는 것이 실무의 기본입니다.

**`admin.py` 예시:**
```python
# myapp/admin.py

from django.contrib import admin
from .models import Post, Comment # 관리할 모델 임포트

# Post 모델을 관리자 페이지에 등록하고 커스터마이징
@admin.register(Post) # 데코레이터를 사용하여 모델 등록 (더 간결한 방식)
class PostAdmin(admin.ModelAdmin):
    # 1. 목록 페이지(Change List) 설정
    list_display = ('title', 'author', 'is_published', 'created_at', 'updated_at') # 목록에 표시할 필드
    list_filter = ('is_published', 'created_at', 'author') # 사이드바 필터 옵션
    search_fields = ('title', 'content') # 검색 가능한 필드 지정
    ordering = ('-created_at',) # 기본 정렬 순서 (최신순)
    list_per_page = 20 # 페이지당 표시할 객체 수
    raw_id_fields = ('author',) # ForeignKey 필드를 ID 입력 필드로 변경 (객체가 많을 때 유용)

    # 2. 상세 페이지(Change Form) 설정
    # fields = ('title', 'content', 'is_published', 'author') # 필드 순서 지정
    fieldsets = (
        (None, {'fields': ('title', 'content')}),
        ('정보', {'fields': ('author', 'is_published', 'created_at', 'updated_at'), 'classes': ('collapse',)}), # 필드 그룹화 및 접기 기능
    )
    readonly_fields = ('created_at', 'updated_at') # 읽기 전용 필드

    # 3. 폼 저장 시 동작 커스터마이징
    def save_model(self, request, obj, form, change):
        # 객체 생성 시 작성자를 현재 로그인한 사용자로 자동 설정
        if not obj.pk: # 새로 생성되는 객체인 경우
            obj.author = request.user
        super().save_model(request, obj, form, change)

# Comment 모델 등록 (간단한 커스터마이징)
@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ('post', 'author', 'content', 'created_at')
    list_filter = ('author', 'created_at')
    search_fields = ('content',)
    raw_id_fields = ('post', 'author')
```

### 1.3. 인라인(Inline) 모델 관리: 관련 객체를 한 화면에서

인라인 모델은 부모 객체의 관리자 페이지에서 해당 부모 객체와 관련된 자식 객체들을 함께 추가, 편집, 삭제할 수 있도록 해줍니다. 예를 들어, `Post` 객체를 편집하면서 해당 `Post`에 달린 `Comment`들을 바로 관리할 수 있습니다.

- **`admin.StackedInline`**: 관련 객체들을 세로로 쌓아서 표시합니다.
- **`admin.TabularInline`**: 관련 객체들을 테이블 형태로 가로로 표시합니다. (더 간결하여 선호됨)

**`admin.py` 예시 (PostAdmin에 Comment Inline 추가):**
```python
# myapp/admin.py (계속)

class CommentInline(admin.TabularInline): # 또는 admin.StackedInline
    model = Comment # Comment 모델을 Post 모델에 인라인으로 연결
    extra = 1 # 기본으로 보여줄 빈 폼의 개수
    fields = ('author', 'content', 'created_at')
    readonly_fields = ('created_at',)
    # related_name이 comments인 경우, Post 모델에서 comments를 통해 접근
    # fk_name = "post" # ForeignKey가 모호할 경우 명시

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    # ... (기존 PostAdmin 설정) ...
    inlines = [CommentInline] # 여기에 인라인 클래스를 추가
```

### 1.4. 관리자 액션 (Admin Actions): 일괄 작업 자동화

관리자 액션은 목록 페이지에서 여러 객체를 선택한 후, 드롭다운 메뉴를 통해 특정 작업을 일괄적으로 수행할 수 있도록 해줍니다. (예: 선택된 게시글을 발행 상태로 변경, 선택된 사용자에게 이메일 발송)

**`admin.py` 예시 (게시글 발행 액션 추가):**
```python
# myapp/admin.py (계속)

# 액션 함수 정의: modeladmin, request, queryset 세 인자를 받습니다.
@admin.action(description='선택된 게시글을 발행 상태로 변경')
def make_published(modeladmin, request, queryset):
    updated_count = queryset.update(is_published=True)
    modeladmin.message_user(request, f'{updated_count}개의 게시글이 발행 상태로 변경되었습니다.', level='success')

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    # ... (기존 PostAdmin 설정) ...
    actions = [make_published] # 여기에 액션 함수를 추가
```

### 1.5. 관리자 페이지 커스터마이징 심화

Django Admin은 기본적인 커스터마이징 외에도, 더 깊은 수준의 변경을 허용합니다.

- **관리자 사이트 제목/헤더 변경**: `AdminSite` 클래스를 상속받아 `site_header`, `site_title`, `index_title` 속성을 변경할 수 있습니다.
    ```python
    # myproject/admin.py (프로젝트 루트에 새로운 admin.py 파일을 만들거나, 기존 admin.py에 추가)
    from django.contrib.admin import AdminSite

    class MyCustomAdminSite(AdminSite):
        site_header = "내 서비스 관리자 페이지" # 브라우저 탭 제목
        site_title = "내 서비스 관리" # 로그인 페이지 및 헤더 제목
        index_title = "환영합니다!" # 관리자 메인 페이지 제목

    # 커스텀 AdminSite 인스턴스 생성
    custom_admin_site = MyCustomAdminSite(name='my_custom_admin')

    # 모델 등록 시 이 커스텀 AdminSite를 사용
    # custom_admin_site.register(Post, PostAdmin)
    # custom_admin_site.register(Comment, CommentAdmin)
    ```
    그리고 `urls.py`에서 기본 `admin.site.urls` 대신 `myproject.admin.custom_admin_site.urls`를 사용하도록 변경합니다.

- **템플릿 오버라이딩**: Django Admin의 기본 템플릿을 프로젝트의 `templates/admin/` 디렉토리에 동일한 경로로 복사한 후 수정하면, 관리자 페이지의 HTML 구조나 디자인을 완전히 변경할 수 있습니다. (예: `admin/base_site.html`, `admin/change_list.html`)

이러한 커스터마이징 기법들을 활용하면 Django Admin을 단순한 개발 도구를 넘어, 실제 서비스 운영에 필요한 강력한 백오피스 시스템으로 발전시킬 수 있습니다.

## 2. 사용자 인증 및 권한 부여: 실무 가이드

Django는 강력하고 유연한 사용자 인증(Authentication) 및 권한 부여(Authorization) 시스템을 기본으로 제공합니다. 이를 통해 사용자 계정 관리, 로그인/로그아웃, 접근 제어 등을 안전하고 효율적으로 구현할 수 있습니다.

### 2.1. Custom User 모델: 왜 필요한가?

Django는 `django.contrib.auth.models.User`라는 기본 사용자 모델을 제공합니다. 이 모델은 `username`, `password`, `email`, `is_active` 등 기본적인 인증에 필요한 필드들을 포함합니다. 하지만 대부분의 실제 웹 애플리케이션에서는 사용자에게 추가적인 정보(예: 전화번호, 프로필 사진, 주소, 특정 비즈니스 로직에 필요한 필드)를 저장해야 할 필요가 있습니다.

**Custom User 모델을 사용하는 이유:**
- **필드 확장**: 기본 `User` 모델에 없는 필드를 추가할 수 있습니다.
- **인증 방식 변경**: `username` 대신 `email`을 주 인증 수단으로 사용하고 싶을 때.
- **유연성**: 미래의 요구사항 변화에 더 유연하게 대처할 수 있습니다.

**Custom User 모델 정의 방법:**
Django는 두 가지 주요 방법을 제공합니다.

- **`AbstractUser` 상속 (가장 권장)**:
    - Django의 기본 `User` 모델이 제공하는 모든 필드와 기능을 그대로 유지하면서, 추가적인 필드를 손쉽게 확장할 수 있습니다. 대부분의 경우 이 방법을 사용하는 것이 가장 편리하고 안전합니다.
    - `username`, `email`, `password`, `first_name`, `last_name`, `is_active`, `is_staff`, `is_superuser`, `last_login`, `date_joined` 필드와 모든 인증 관련 메서드 및 관계를 상속받습니다.

- **`AbstractBaseUser` 상속**: 
    - 완전히 새로운 사용자 모델을 처음부터 정의할 때 사용합니다. 이 경우 인증 관련 핵심 필드(예: `username`, `password`, `last_login`, `is_active`)를 직접 구현해야 하므로 더 복잡하지만, 최대한의 유연성을 제공합니다. (예: `username` 필드 없이 `email`만으로 인증하는 경우)

**구현 방법 (AbstractUser 예시):**

1.  **`accounts` 앱 생성**: 사용자 모델을 관리할 별도의 앱을 생성하는 것이 일반적입니다.
    ```bash
    python manage.py startapp accounts
    ```

2.  **`settings.py`에 앱 등록**: `accounts` 앱을 `INSTALLED_APPS`에 추가합니다.

3.  **`accounts/models.py`에 Custom User 모델 정의**:
    ```python
    # accounts/models.py
    from django.contrib.auth.models import AbstractUser
    from django.db import models

    class CustomUser(AbstractUser):
        # 기본 User 모델의 필드(username, password, email 등)는 그대로 사용
        # 여기에 추가하고 싶은 필드를 정의합니다.
        phone_number = models.CharField("전화번호", max_length=15, blank=True, null=True)
        profile_picture = models.ImageField("프로필 사진", upload_to='profile_pics/', blank=True, null=True)
        address = models.CharField("주소", max_length=255, blank=True, null=True)

        # __str__ 메서드는 관리자 페이지 등에서 객체를 식별하는 데 사용됩니다.
        def __str__(self):
            return self.username
    ```

4.  **`settings.py`에 Custom User 모델 지정**: Django에게 사용할 사용자 모델이 무엇인지 알려줍니다. **이 설정은 프로젝트의 첫 `makemigrations` 명령을 실행하기 전에 반드시 추가**해야 합니다. 이미 마이그레이션이 생성된 프로젝트에서 변경하려면 복잡한 데이터 마이그레이션 전략이 필요합니다.
    ```python
    # settings.py
    AUTH_USER_MODEL = 'accounts.CustomUser'
    ```

5.  **마이그레이션 실행**: `CustomUser` 모델에 대한 마이그레이션을 생성하고 적용합니다.
    ```bash
    python manage.py makemigrations accounts
    python manage.py migrate
    ```

### 2.2. 사용자 인증 (Authentication)

Django는 내장된 인증 시스템을 통해 사용자 로그인, 로그아웃, 비밀번호 변경 등을 쉽게 처리할 수 있도록 돕습니다.

#### 2.2.1. 로그인/로그아웃 뷰

Django는 `django.contrib.auth.views` 모듈에 미리 구현된 `LoginView`와 `LogoutView`를 제공합니다. 이들을 `urls.py`에 연결하고 템플릿만 지정해주면 됩니다.

**`settings.py` 관련 설정:**
```python
# settings.py
LOGIN_REDIRECT_URL = '/' # 로그인 성공 후 리다이렉트할 URL
LOGOUT_REDIRECT_URL = '/accounts/login/' # 로그아웃 후 리다이렉트할 URL
LOGIN_URL = '/accounts/login/' # @login_required 데코레이터 등이 로그인 페이지로 리다이렉트할 때 사용할 URL
```

**`urls.py` 예시:**
```python
# myproject/urls.py
from django.contrib.auth import views as auth_views
from accounts import views as accounts_views # 회원가입 뷰를 위해 임포트

urlpatterns = [
    # ...
    path('accounts/login/', auth_views.LoginView.as_view(template_name='accounts/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('accounts/signup/', accounts_views.signup, name='signup'), # 회원가입 뷰
    # ...
]
```

#### 2.2.2. 로그인 폼 커스터마이징

`LoginView`는 기본적으로 `AuthenticationForm`을 사용합니다. 이 폼을 커스터마이징하여 추가 필드를 넣거나 유효성 검사 로직을 변경할 수 있습니다.

**`accounts/forms.py` 예시:**
```python
# accounts/forms.py
from django.contrib.auth.forms import AuthenticationForm
from django import forms

class CustomAuthenticationForm(AuthenticationForm):
    # 기본 필드 외에 추가 필드를 넣거나, 기존 필드의 위젯/라벨 변경 가능
    # 예를 들어, username 필드의 placeholder를 변경
    username = forms.CharField(
        label="아이디",
        widget=forms.TextInput(attrs={'placeholder': '아이디를 입력하세요'})
    )
    password = forms.CharField(
        label="비밀번호",
        widget=forms.PasswordInput(attrs={'placeholder': '비밀번호를 입력하세요'})
    )

    # 추가적인 유효성 검사 로직을 clean() 메서드에 구현 가능
    def clean(self):
        cleaned_data = super().clean()
        # ... 추가 로직 ...
        return cleaned_data
```

**`urls.py`에서 커스텀 폼 적용:**
```python
# myproject/urls.py
from django.contrib.auth import views as auth_views
from accounts.forms import CustomAuthenticationForm

urlpatterns = [
    path('accounts/login/', auth_views.LoginView.as_view(
        template_name='accounts/login.html',
        authentication_form=CustomAuthenticationForm # 커스텀 폼 지정
    ), name='login'),
    # ...
]
```

#### 2.2.3. 사용자 등록 (회원가입)

Django는 회원가입 뷰를 기본으로 제공하지 않습니다. `UserCreationForm`을 사용하여 직접 구현해야 합니다.

**`accounts/views.py` 예시:**
```python
# accounts/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm # 기본 User 모델용
# from accounts.forms import CustomUserCreationForm # Custom User 모델용 커스텀 폼

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST) # 또는 CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login') # 회원가입 성공 후 로그인 페이지로 리다이렉트
    else:
        form = UserCreationForm() # 또는 CustomUserCreationForm()
    return render(request, 'accounts/signup.html', {'form': form})
```

### 2.3. 권한 부여 (Authorization)

Django의 권한 시스템은 사용자가 특정 작업을 수행할 수 있는지 여부를 제어합니다. 이는 `User` 모델, `Permission` 모델, `Group` 모델 간의 관계를 통해 구현됩니다.

#### 2.3.1. 권한 및 그룹 관리

- **권한 (Permissions)**: 
    - Django는 각 모델에 대해 `add`, `change`, `delete`, `view` 권한을 자동으로 생성합니다. (예: `blog.add_post`, `blog.change_post`)
    - `class Meta`의 `permissions` 옵션을 통해 커스텀 권한을 정의할 수 있습니다. (예: `("can_publish_post", "Can publish post")`)
- **그룹 (Groups)**: 여러 사용자에게 동일한 권한 집합을 부여할 때 사용합니다. 예를 들어, "편집자" 그룹을 만들고 `blog.add_post`, `blog.change_post` 권한을 부여한 뒤, 편집자 역할을 하는 사용자들을 이 그룹에 추가할 수 있습니다.
- **관리자 페이지**: 모든 권한과 그룹은 Django 관리자 페이지에서 쉽게 관리할 수 있습니다.

#### 2.3.2. 뷰에 권한 적용하기

뷰에 접근 권한을 적용하는 가장 일반적인 방법은 데코레이터(함수 기반 뷰)나 믹스인(클래스 기반 뷰)을 사용하는 것입니다.

- **로그인 필수**: 
    - **FBV**: `@login_required` 데코레이터
        ```python
        from django.contrib.auth.decorators import login_required

        @login_required
        def my_protected_fbv(request):
            return HttpResponse("로그인한 사용자만 볼 수 있습니다.")
        ```
    - **CBV**: `LoginRequiredMixin` 믹스인
        ```python
        from django.contrib.auth.mixins import LoginRequiredMixin
        from django.views import View

        class MyProtectedCBV(LoginRequiredMixin, View):
            def get(self, request, *args, **kwargs):
                return HttpResponse("로그인한 사용자만 볼 수 있습니다.")
        ```

- **특정 권한 필수**: 
    - **FBV**: `@permission_required('app_label.permission_codename')` 데코레이터
        ```python
        from django.contrib.auth.decorators import permission_required

        @permission_required('blog.add_post') # blog 앱의 add_post 권한이 있어야 접근 가능
        def create_post_fbv(request):
            return HttpResponse("게시글 작성 페이지입니다.")
        ```
    - **CBV**: `PermissionRequiredMixin` 믹스인
        ```python
        from django.contrib.auth.mixins import PermissionRequiredMixin
        from django.views.generic.edit import CreateView

        class CreatePostCBV(PermissionRequiredMixin, CreateView):
            permission_required = 'blog.add_post' # blog 앱의 add_post 권한이 있어야 접근 가능
            # ... (CreateView의 다른 설정들) ...
        ```

- **커스텀 로직 기반 권한**: `UserPassesTestMixin`을 사용하면 `test_func` 메서드에 직접 파이썬 로직을 작성하여 복잡한 권한 검사를 수행할 수 있습니다. (예: 게시글의 작성자만 수정 가능)
    ```python
    from django.contrib.auth.mixins import UserPassesTestMixin
    from django.views.generic.edit import UpdateView

    class PostUpdateView(UserPassesTestMixin, UpdateView):
        # ... (UpdateView의 다른 설정들) ...

        def test_func(self):
            # 현재 게시글의 작성자가 요청한 사용자인지 확인
            post = self.get_object()
            return post.author == self.request.user
    ```

#### 2.3.3. 템플릿에서 권한 확인하기

템플릿에서 사용자 권한에 따라 특정 요소를 표시하거나 숨길 수 있습니다.

- **로그인 상태 확인**: `{% if user.is_authenticated %}`
- **스태프/슈퍼유저 확인**: `{% if user.is_staff %}`, `{% if user.is_superuser %}`
- **특정 권한 확인**: `{% if user.has_perm 'app_label.permission_codename' %}`
    - 더 간결한 방법: `{% if perms.app_label.permission_codename %}`

```html
{% if user.is_authenticated %}
    <p>환영합니다, {{ user.username }}님!</p>
    <a href="{% url 'logout' %}">로그아웃</a>
{% else %}
    <a href="{% url 'login' %}">로그인</a>
    <a href="{% url 'signup' %}">회원가입</a>
{% endif %}

{% if perms.blog.add_post %}
    <a href="{% url 'blog:post_create' %}">새 게시글 작성</a>
{% endif %}

{% if perms.blog.change_post and post.author == user %}
    <a href="{% url 'blog:post_update' post.pk %}">게시글 수정</a>
{% endif %}
```
