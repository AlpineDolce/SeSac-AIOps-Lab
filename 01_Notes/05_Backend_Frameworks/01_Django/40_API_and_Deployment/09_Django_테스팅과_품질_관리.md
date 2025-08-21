<h2>Django Backend: 테스팅과 품질 관리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django 애플리케이션의 품질과 안정성을 보장하기 위한 테스트 작성법을 학습하는 것을 목표로 합니다. Django의 내장 테스트 프레임워크를 사용하여 단위 테스트와 통합 테스트를 작성하고, Mocking, Factory Boy, 테스트 커버리지 등 실무적인 테스트 기법을 이해합니다.</p>

<h2>목차</h2> 

- [1. 테스트 (Testing): 소프트웨어 품질 보증의 핵심](#1-테스트-testing-소프트웨어-품질-보증의-핵심)
  - [테스트의 종류 (간략한 개요):](#테스트의-종류-간략한-개요)
- [2. 테스트의 중요성: 왜 테스트해야 하는가?](#2-테스트의-중요성-왜-테스트해야-하는가)
  - [테스팅 전략 및 범위: 무엇을 테스트해야 하는가?](#테스팅-전략-및-범위-무엇을-테스트해야-하는가)
- [3. Django 테스트 프레임워크: 견고한 기반](#3-django-테스트-프레임워크-견고한-기반)
- [4. 단위 테스트 (Unit Tests): 최소 단위의 검증](#4-단위-테스트-unit-tests-최소-단위의-검증)
  - [4.1. 모델 테스트](#41-모델-테스트)
  - [4.2. 폼 테스트](#42-폼-테스트)
- [5. 통합 테스트 (Integration Tests): 컴포넌트 간의 상호작용 검증](#5-통합-테스트-integration-tests-컴포넌트-간의-상호작용-검증)
  - [5.1. 뷰 테스트 (Client 활용)](#51-뷰-테스트-client-활용)
- [6. 테스트 실행: 효율적인 테스트 관리](#6-테스트-실행-효율적인-테스트-관리)
  - [6.1. 기본 테스트 실행 명령어](#61-기본-테스트-실행-명령어)
  - [6.2. 효율적인 테스트 실행을 위한 실무 팁](#62-효율적인-테스트-실행을-위한-실무-팁)
- [7. 테스트 심화 - Mocking: 외부 의존성 제어](#7-테스트-심화---mocking-외부-의존성-제어)
  - [7.1. Mocking의 필요성](#71-mocking의-필요성)
  - [7.2. `unittest.mock` 사용 예시](#72-unittestmock-사용-예시)
- [8. 테스팅 심화: 효율적인 테스트 데이터와 커버리지](#8-테스팅-심화-효율적인-테스트-데이터와-커버리지)
  - [8.1. Factory Boy: 테스트 데이터 생성 자동화](#81-factory-boy-테스트-데이터-생성-자동화)
  - [8.2. Test Coverage (테스트 커버리지): 코드 품질 지표](#82-test-coverage-테스트-커버리지-코드-품질-지표)

---

## 1. 테스트 (Testing): 소프트웨어 품질 보증의 핵심

소프트웨어 개발에서 테스트는 코드의 품질을 보장하고, 버그를 조기에 발견하며, 변경 사항이 기존 기능에 영향을 미치지 않음을 확인하는 데 필수적인 과정입니다. Django는 강력한 테스트 프레임워크를 내장하고 있어 효율적인 테스트 작성을 돕습니다.

테스트는 단순히 버그를 찾는 것을 넘어, **요구사항을 검증**하고, **코드의 유지보수성을 보장**하며, **안심하고 리팩토링**할 수 있도록 돕는 중요한 활동입니다.

### 테스트의 종류 (간략한 개요):
-   **단위 테스트 (Unit Tests)**: 애플리케이션의 가장 작은 단위(함수, 메서드, 클래스)를 독립적으로 테스트합니다.
-   **통합 테스트 (Integration Tests)**: 여러 컴포넌트(모델, 뷰, URL 등)가 함께 작동하여 올바르게 동작하는지 검증합니다.
-   **기능/End-to-End 테스트 (Functional/E2E Tests)**: 사용자 관점에서 전체 시스템의 흐름을 테스트합니다. (예: Selenium을 이용한 브라우저 테스트)

---

## 2. 테스트의 중요성: 왜 테스트해야 하는가?

-   **버그 조기 발견**: 개발 초기 단계에서 버그를 발견하고 수정하는 것이 배포 후 발견하는 것보다 훨씬 비용과 시간이 적게 듭니다.
-   **코드 품질 향상**: 테스트 가능한 코드를 작성하는 과정에서 자연스럽게 모듈화되고 응집도 높은 코드를 작성하게 됩니다.
-   **회귀 방지**: 코드 변경이나 기능 추가 시 기존 기능이 오작동하지 않음을 보장합니다. (Regression Testing)
-   **문서화**: 테스트 코드는 해당 기능이 어떻게 작동해야 하는지에 대한 살아있는 문서 역할을 합니다.
-   **리팩토링 용이성**: 테스트 스위트가 잘 갖춰져 있으면, 코드 리팩토링 시에도 안심하고 변경을 적용할 수 있습니다.

### 테스팅 전략 및 범위: 무엇을 테스트해야 하는가?

-   **Model**: 모델의 커스텀 메서드, `save()` 메서드 오버라이드, `clean()` 메서드 등 비즈니스 로직이 포함된 부분을 테스트합니다. 필드 속성 자체보다는 모델의 '동작'을 검증하는 데 집중합니다.
-   **View**: 뷰의 핵심 로직, 즉 특정 요청(GET, POST 등)에 대해 올바른 응답(상태 코드, 템플릿, 리다이렉트)을 반환하는지, 조건부 로직이 제대로 동작하는지, 권한 제어가 올바른지 등을 테스트합니다.
-   **Form**: 폼의 유효성 검사 로직, 특히 여러 필드에 걸친 복잡한 `clean()` 메서드나 커스텀 validator를 테스트합니다.
-   **Service/Business Logic**: 뷰나 모델에서 분리된 순수 파이썬 비즈니스 로직(서비스 계층)이 있다면, 해당 로직을 독립적으로 테스트하는 것이 중요합니다.
-   **테스트 커버리지(Test Coverage)**: 전체 코드 중 테스트 코드가 얼마나 실행했는지를 나타내는 지표입니다. `coverage.py`와 같은 도구를 사용하여 측정할 수 있으며, 높은 커버리지는 코드의 신뢰도를 높이는 데 도움이 됩니다. 하지만 100% 커버리지가 모든 버그를 막아주는 것은 아니므로, 중요하고 복잡한 로직을 중심으로 효과적인 테스트를 작성하는 것이 더 중요합니다.

---

## 3. Django 테스트 프레임워크: 견고한 기반

Django는 Python의 내장 `unittest` 모듈을 기반으로 확장된 테스트 프레임워크를 제공합니다. Django 애플리케이션을 테스트하기 위한 다양한 유틸리티와 어설션(assertion) 메서드를 포함하고 있습니다.

-   **`django.test.TestCase`**: 
    -   Django 테스트의 기본 클래스입니다. 각 테스트 메서드 실행 전후에 데이터베이스를 깨끗하게 초기화하고 트랜잭션을 롤백하여 **테스트 간의 완벽한 격리**를 보장합니다.
    -   `setUp()`: 각 테스트 메서드 실행 전에 호출되어 테스트에 필요한 데이터를 준비합니다.
    -   `setUpTestData()`: 클래스 내의 모든 테스트 메서드에서 공유될 데이터를 한 번만 생성할 때 사용합니다. `setUp()`보다 효율적입니다.

-   **`django.test.Client`**: 
    -   실제 웹 브라우저처럼 HTTP 요청(GET, POST 등)을 시뮬레이션할 수 있는 가상의 클라이언트입니다. 뷰의 응답을 직접 테스트할 때 사용합니다.
    -   `self.client.get(url, data={}, follow=False)`
    -   `self.client.post(url, data={}, follow=False)`
    -   `follow=True` 옵션을 사용하면 리다이렉트 응답을 자동으로 따라갑니다.

-   **주요 어설션(Assertions)**: `TestCase`는 `unittest.TestCase`의 모든 어설션 외에 Django 테스트에 특화된 어설션들을 제공합니다.
    -   `self.assertEqual(a, b)`: `a`와 `b`가 같은지 확인
    -   `self.assertTrue(x)`: `x`가 `True`인지 확인
    -   `self.assertContains(response, text, status_code=200)`: 응답 본문에 특정 텍스트가 포함되어 있는지 확인
    -   `self.assertTemplateUsed(response, template_name)`: 특정 템플릿이 사용되었는지 확인
    -   `self.assertRedirects(response, expected_url, status_code=302)`: 리다이렉트가 올바른지 확인
    -   `self.assertFormError(response, form_context_name, field_name, error_message)`: 폼 에러 확인

---

## 4. 단위 테스트 (Unit Tests): 최소 단위의 검증

단위 테스트는 애플리케이션의 가장 작은 단위(함수, 메서드, 모델의 특정 로직)가 예상대로 작동하는지 독립적으로 검증하는 테스트입니다. 외부 의존성(DB, 네트워크 등)을 최소화하거나 Mocking을 통해 제거하여 빠르게 실행되고 실패 원인을 명확히 파악할 수 있도록 합니다.

### 4.1. 모델 테스트

모델의 필드 정의, `__str__` 메서드, 커스텀 메서드, `clean()` 메서드 등 모델에 정의된 비즈니스 로직을 검증합니다.

```python
# myapp/tests/test_models.py

from django.test import TestCase
from django.core.exceptions import ValidationError
from django.contrib.auth import get_user_model # Custom User 모델을 안전하게 가져옴
from myapp.models import Post

User = get_user_model()

class PostModelTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        # 테스트 클래스 전체에서 공유될 데이터 (한 번만 생성)
        cls.user = User.objects.create_user(username='testuser', password='password')
        cls.post = Post.objects.create(author=cls.user, title='테스트 게시글', content='테스트 내용')

    def test_post_creation(self):
        # 객체가 올바르게 생성되었는지 확인
        self.assertEqual(Post.objects.count(), 1)
        self.assertEqual(self.post.title, '테스트 게시글')
        self.assertTrue(self.post.is_published) # 기본값이 True라고 가정

    def test_post_str_representation(self):
        # __str__ 메서드가 올바른 문자열을 반환하는지 확인
        self.assertEqual(str(self.post), '테스트 게시글')

    def test_post_custom_method(self):
        # 모델에 정의된 커스텀 메서드 테스트 (예: get_word_count)
        # self.assertEqual(self.post.get_word_count(), 2) # Post 모델에 get_word_count 메서드가 있다고 가정
        pass

    def test_post_clean_method_validation(self):
        # 모델의 clean() 메서드 유효성 검사 테스트
        # (예: 제목에 특정 단어가 포함되면 안 되는 로직이 clean()에 있다고 가정)
        invalid_post = Post(author=self.user, title='금지어 포함', content='내용')
        with self.assertRaises(ValidationError):
            invalid_post.full_clean() # full_clean()은 모델의 모든 유효성 검사를 실행
```

### 4.2. 폼 테스트

폼의 유효성 검사 로직, 필드 렌더링, 데이터 저장 등이 올바르게 작동하는지 확인합니다. 특히 커스텀 `clean_<field>()`나 `clean()` 메서드를 테스트하는 것이 중요합니다.

```python
# myapp/tests/test_forms.py

from django.test import TestCase
from myapp.forms import PostForm # PostForm은 Post 모델에 연결된 ModelForm이라고 가정

class PostFormTest(TestCase):
    def test_valid_form(self):
        # 유효한 데이터로 폼 생성 및 유효성 검사
        form = PostForm(data={'title': '유효한 제목', 'content': '유효한 내용', 'is_published': True})
        self.assertTrue(form.is_valid()) # 폼이 유효해야 함

    def test_invalid_form_missing_title(self):
        # 필수 필드(title)가 누락된 경우
        form = PostForm(data={'content': '내용', 'is_published': True})
        self.assertFalse(form.is_valid()) # 폼이 유효하지 않아야 함
        self.assertIn('title', form.errors) # title 필드에 에러가 있어야 함

    def test_invalid_form_custom_validation(self):
        # 폼의 커스텀 유효성 검사(예: clean_title) 테스트
        form = PostForm(data={'title': '비속어 포함', 'content': '내용', 'is_published': True})
        self.assertFalse(form.is_valid()) # 폼이 유효하지 않아야 함
        self.assertIn('title', form.errors) # title 필드에 에러가 있어야 함
        self.assertIn('제목에 부적절한 단어가 포함되어 있습니다.', form.errors['title'][0])

    def test_form_save(self):
        # 폼을 통한 데이터 저장 테스트
        user = get_user_model().objects.create_user(username='testuser', password='password')
        form = PostForm(data={'title': '저장될 제목', 'content': '저장될 내용', 'is_published': True})
        self.assertTrue(form.is_valid())
        
        # form.save()는 모델 인스턴스를 반환
        post = form.save(commit=False)
        post.author = user # author 필드는 폼에 없으므로 수동 할당
        post.save()

        self.assertIsNotNone(post.pk) # PK가 할당되었는지 확인
        self.assertEqual(Post.objects.count(), 1) # DB에 저장되었는지 확인
```

---

## 5. 통합 테스트 (Integration Tests): 컴포넌트 간의 상호작용 검증

통합 테스트는 여러 컴포넌트(모델, 뷰, 템플릿, URL, 폼 등)가 함께 작동하여 전체 기능이 올바르게 수행되는지 검증하는 테스트입니다. Django의 `Client` 클래스를 사용하여 HTTP 요청을 시뮬레이션하고, 실제 데이터베이스와 상호작용하며 테스트를 수행합니다.

### 5.1. 뷰 테스트 (Client 활용)

뷰가 올바른 HTTP 응답을 반환하고, 템플릿을 제대로 렌더링하며, 데이터베이스와 상호작용하는지 확인합니다. 로그인, POST 요청 등 실제 사용자 시나리오를 모방합니다.

```python
# myapp/tests/test_views.py

from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth import get_user_model
from myapp.models import Post

User = get_user_model()

class PostViewTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        # 테스트 클래스 전체에서 공유될 데이터 (한 번만 생성)
        cls.user = User.objects.create_user(username='testuser', password='password')
        cls.admin_user = User.objects.create_superuser(username='admin', password='adminpass')
        cls.post = Post.objects.create(author=cls.user, title='테스트 게시글', content='테스트 내용', is_published=True)
        cls.draft_post = Post.objects.create(author=cls.user, title='임시 게시글', content='임시 내용', is_published=False)

    def setUp(self):
        # 각 테스트 메서드 실행 전에 호출
        self.client = Client()

    def test_post_list_view(self):
        # 게시글 목록 페이지 접근 테스트
        response = self.client.get(reverse('post_list')) # urls.py에 정의된 URL name 사용
        self.assertEqual(response.status_code, 200) # HTTP 200 OK 확인
        self.assertContains(response, self.post.title) # 템플릿에 게시글 제목이 포함되어 있는지 확인
        self.assertNotContains(response, self.draft_post.title) # 발행되지 않은 글은 보이지 않아야 함
        self.assertTemplateUsed(response, 'myapp/post_list.html') # 사용된 템플릿 확인

    def test_post_detail_view(self):
        # 게시글 상세 페이지 접근 테스트
        response = self.client.get(reverse('post_detail', args=[self.post.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, self.post.title)
        self.assertContains(response, self.post.content)

    def test_post_create_view_authenticated(self):
        # 로그인한 사용자가 게시글 생성 폼 제출 테스트
        self.client.login(username='testuser', password='password') # 사용자 로그인
        response = self.client.post(reverse('post_create'), {
            'title': '새로운 게시글',
            'content': '새로운 내용입니다.',
            'is_published': True,
            # 'author': self.user.pk, # ModelForm에서 request.user로 자동 할당된다고 가정
        }, follow=True) # follow=True: 리다이렉트 응답을 자동으로 따라감
        
        # 성공 시 리다이렉트 (일반적으로 목록 페이지나 상세 페이지로)
        self.assertEqual(response.status_code, 200) # 리다이렉트 후 최종 응답 코드
        self.assertContains(response, '새로운 게시글') # 최종 페이지에 내용이 포함되어 있는지 확인
        self.assertTrue(Post.objects.filter(title='새로운 게시글').exists()) # DB에 저장되었는지 확인
        self.assertEqual(Post.objects.last().author, self.user) # 작성자가 올바른지 확인

    def test_post_create_view_unauthenticated(self):
        # 비로그인 사용자가 게시글 생성 시도 테스트 (권한 없음)
        response = self.client.post(reverse('post_create'), {
            'title': '비로그인 게시글',
            'content': '내용',
            'is_published': True,
        })
        # 로그인 페이지로 리다이렉트되거나 403 Forbidden 응답을 기대
        self.assertRedirects(response, reverse('login') + '?next=' + reverse('post_create'))
        self.assertFalse(Post.objects.filter(title='비로그인 게시글').exists()) # DB에 저장되지 않아야 함

    def test_post_update_view_owner(self):
        # 게시글 소유자가 수정 폼 제출 테스트
        self.client.login(username='testuser', password='password')
        response = self.client.post(reverse('post_update', args=[self.post.pk]), {
            'title': '수정된 게시글 제목',
            'content': '수정된 내용입니다.',
            'is_published': True,
        }, follow=True)
        self.assertEqual(response.status_code, 200)
        self.post.refresh_from_db() # DB에서 최신 데이터 로드
        self.assertEqual(self.post.title, '수정된 게시글 제목')

    def test_post_update_view_non_owner(self):
        # 다른 사용자가 게시글 수정 시도 테스트 (권한 없음)
        other_user = User.objects.create_user(username='otheruser', password='password')
        self.client.login(username='otheruser', password='password')
        response = self.client.post(reverse('post_update', args=[self.post.pk]), {
            'title': '다른 사용자가 수정',
            'content': '내용',
            'is_published': True,
        })
        self.assertEqual(response.status_code, 403) # 403 Forbidden 응답 기대
        self.post.refresh_from_db()
        self.assertNotEqual(self.post.title, '다른 사용자가 수정') # 제목이 변경되지 않아야 함

    def test_post_delete_view(self):
        # 게시글 삭제 테스트
        self.client.login(username='testuser', password='password')
        response = self.client.post(reverse('post_delete', args=[self.post.pk]), follow=True)
        self.assertEqual(response.status_code, 200) # 삭제 후 리다이렉트된 페이지
        self.assertFalse(Post.objects.filter(pk=self.post.pk).exists()) # DB에서 삭제되었는지 확인
```

소프트웨어 개발에서 테스트는 코드의 품질을 보장하고, 버그를 조기에 발견하며, 변경 사항이 기존 기능에 영향을 미치지 않음을 확인하는 데 필수적인 과정입니다. Django는 강력한 테스트 프레임워크를 내장하고 있어 효율적인 테스트 작성을 돕습니다.



## 6. 테스트 실행: 효율적인 테스트 관리

작성된 테스트 코드를 실행하고 관리하는 것은 테스트 프로세스의 중요한 부분입니다. Django는 `manage.py test` 명령어를 통해 유연한 테스트 실행 옵션을 제공합니다.

### 6.1. 기본 테스트 실행 명령어

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 모든 테스트를 실행할 수 있습니다.

```bash
python manage.py test
```

-   **특정 앱의 테스트만 실행**: `python manage.py test myapp`
-   **특정 테스트 케이스만 실행**: `python manage.py test myapp.tests.MyModelTest`
-   **특정 테스트 메서드만 실행**: `python manage.py test myapp.tests.MyModelTest.test_my_model_creation`

### 6.2. 효율적인 테스트 실행을 위한 실무 팁

-   **병렬 테스트 실행**: 멀티코어 CPU를 활용하여 테스트 실행 속도를 크게 향상시킬 수 있습니다. 특히 테스트 수가 많아질수록 효과적입니다.
    ```bash
    python manage.py test --parallel [N] # N은 사용할 프로세스 수 (CPU 코어 수와 동일하게 설정 권장)
    # 예: python manage.py test --parallel 4
    ```
-   **테스트 실패 시 중단**: 첫 번째 테스트 실패 시 바로 실행을 중단하여 빠르게 피드백을 얻을 수 있습니다.
    ```bash
    python manage.py test --failfast
    ```
-   **테스트 건너뛰기**: 특정 테스트가 아직 구현 중이거나, 특정 환경에서만 실행되어야 할 때 사용합니다.
    ```python
    import unittest

    class MyTest(unittest.TestCase):
        @unittest.skip("아직 구현 중인 기능")
        def test_something_not_ready(self):
            pass

        @unittest.skipIf(sys.version_info.minor < 9, "Python 3.9 이상에서만 지원")
        def test_python_version_specific_feature(self):
            pass
    ```

---

## 7. 테스트 심화 - Mocking: 외부 의존성 제어

단위 테스트를 작성할 때, 테스트 대상 코드가 외부 서비스(API), 데이터베이스, 파일 시스템, 현재 시간 등과 같은 복잡한 의존성을 가질 수 있습니다. **Mocking**은 이러한 의존성을 가짜(mock) 객체로 대체하여 테스트 대상 코드만 독립적으로, 빠르게, 그리고 일관되게 테스트할 수 있도록 돕는 기법입니다.

### 7.1. Mocking의 필요성

-   **독립성**: 테스트가 외부 요인(네트워크 지연, DB 상태, 외부 API 응답)에 의존하지 않고 독립적으로 실행될 수 있도록 합니다.
-   **속도**: 실제 데이터베이스 접근이나 네트워크 요청 없이 메모리 내에서 빠르게 테스트를 실행할 수 있습니다.
-   **재현성**: 외부 서비스의 상태 변화에 관계없이 테스트 결과를 일관되게 재현할 수 있습니다.
-   **특정 시나리오 테스트**: 오류 발생, 특정 값 반환 등 실제로는 발생하기 어렵거나 제어하기 어려운 시나리오를 쉽게 시뮬레이션할 수 있습니다.

### 7.2. `unittest.mock` 사용 예시

Python의 내장 `unittest.mock` 모듈을 사용하여 Mock 객체를 생성하고 관리할 수 있습니다. `patch` 데코레이터는 특정 객체나 함수의 동작을 임시로 변경하는 데 사용됩니다.

**가상 시나리오**: 외부 날씨 API를 호출하여 날씨 정보를 가져오는 `weather_service.py` 모듈의 `get_current_weather` 함수를 테스트하고 싶습니다.

```python
# myapp/weather_service.py
import requests

def get_current_weather(city, api_key):
    url = f"http://api.weather.com/current?city={city}&apiKey={api_key}"
    response = requests.get(url)
    response.raise_for_status() # HTTP 에러 발생 시 예외 처리
    return response.json()

# myapp/tests/test_weather_service.py
from django.test import TestCase
from unittest.mock import patch, MagicMock # patch, MagicMock 임포트
from myapp.weather_service import get_current_weather

class WeatherServiceTest(TestCase):
    # requests.get 함수를 Mocking합니다. patch의 경로는 해당 함수가 '사용되는' 곳을 기준으로 합니다.
    @patch('myapp.weather_service.requests.get') 
    def test_get_current_weather_success(self, mock_get):
        # Mock 객체의 반환 값 설정 (requests.Response 객체를 모방)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "city": "Seoul",
            "temperature": 25,
            "condition": "Sunny"
        }
        mock_get.return_value = mock_response

        weather_data = get_current_weather("Seoul", "TEST_API_KEY")

        # requests.get이 올바른 인자로 호출되었는지 확인
        mock_get.assert_called_once_with(
            "http://api.weather.com/current?city=Seoul&apiKey=TEST_API_KEY"
        )
        # 반환된 데이터가 예상과 일치하는지 확인
        self.assertEqual(weather_data['temperature'], 25)
        self.assertEqual(weather_data['condition'], "Sunny")

    @patch('myapp.weather_service.requests.get')
    def test_get_current_weather_api_error(self, mock_get):
        # Mock 객체가 HTTP 에러를 발생시키도록 설정
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            get_current_weather("NonExistentCity", "TEST_API_KEY")

        mock_get.assert_called_once()

### 7.3. Django ORM Mocking (데이터베이스 접근 방지)

단위 테스트에서 데이터베이스 접근을 피하고 싶을 때 ORM 메서드를 Mocking할 수 있습니다. 이는 테스트 속도를 높이고 테스트의 독립성을 보장합니다.

```python
# myapp/tests/test_views.py (일부 로직만 테스트할 때)

from django.test import TestCase
from unittest.mock import patch, MagicMock
from myapp.models import Post # Post 모델 사용

class PostViewUnitTest(TestCase):
    @patch('myapp.models.Post.objects.get') # Post.objects.get 메서드를 Mocking
    def test_get_post_detail_mocked(self, mock_get_post):
        # Mock 객체가 반환할 가짜 Post 인스턴스 생성
        mock_post_instance = MagicMock(title='Mocked Post', content='Mocked Content')
        mock_get_post.return_value = mock_post_instance

        # 뷰 함수를 직접 호출 (request 객체는 MagicMock으로 대체)
        mock_request = MagicMock()
        response = my_post_detail_view_function(mock_request, post_id=1) # 뷰 함수가 있다고 가정

        # Post.objects.get이 올바른 인자로 호출되었는지 확인
        mock_get_post.assert_called_once_with(pk=1)
        # 응답 내용 확인
        self.assertContains(response, 'Mocked Post')
```

---

## 8. 테스팅 심화: 효율적인 테스트 데이터와 커버리지

### 8.1. Factory Boy: 테스트 데이터 생성 자동화

-   **문제점**: 테스트를 위해 모델 객체를 수동으로 `MyModel.objects.create(...)` 하는 것은 반복적이고 지루하며, 모델의 필드가 많아지거나 관계가 복잡해지면 테스트 코드의 가독성을 해치고 작성 시간을 늘립니다.
-   **해결책**: `factory-boy` 라이브러리는 모델 객체 생성을 자동화하고, 더 깔끔하고 효율적인 테스트 코드를 작성할 수 있도록 돕는 팩토리(Factory) 패턴을 제공합니다.

    1.  **설치**: `pip install factory-boy`
    2.  **팩토리 정의**: `myapp/tests/factories.py` 파일에 모델별 팩토리를 정의합니다.
        ```python
        # myapp/tests/factories.py
        import factory
        from django.contrib.auth import get_user_model
        from myapp.models import Post, Comment

        User = get_user_model()

        class UserFactory(factory.django.DjangoModelFactory):
            class Meta:
                model = User
            username = factory.Sequence(lambda n: f'testuser{n}') # 고유한 username 자동 생성
            email = factory.LazyAttribute(lambda o: f'{o.username}@example.com')
            # 비밀번호는 set_password 메서드를 통해 해싱하여 저장
            password = factory.PostGenerationMethodCall('set_password', 'defaultpassword')

        class PostFactory(factory.django.DjangoModelFactory):
            class Meta:
                model = Post
            # ForeignKey 관계는 SubFactory를 통해 자동으로 생성/연결
            author = factory.SubFactory(UserFactory)
            title = factory.Sequence(lambda n: f'게시글 제목 {n}')
            content = factory.Faker('paragraph') # 가짜 텍스트 데이터 생성
            is_published = True

        class CommentFactory(factory.django.DjangoModelFactory):
            class Meta:
                model = Comment
            post = factory.SubFactory(PostFactory)
            author = factory.SubFactory(UserFactory)
            content = factory.Faker('sentence')
        ```
    3.  **테스트에서 사용**: `setUpTestData`나 `setUp` 메서드에서 팩토리를 호출하여 테스트 데이터를 생성합니다.
        ```python
        # myapp/tests/test_models.py (또는 test_views.py)
        from django.test import TestCase
        from myapp.models import Post, Comment
        from myapp.tests.factories import UserFactory, PostFactory, CommentFactory

        class PostCommentTest(TestCase):
            def test_post_and_comment_creation(self):
                # 팩토리를 사용하여 Post 객체 생성 (관련 User도 자동 생성)
                post = PostFactory(title='팩토리로 만든 게시글')
                # 팩토리를 사용하여 Comment 객체 생성 (관련 Post, User도 자동 생성)
                comment = CommentFactory(post=post, content='팩토리로 만든 댓글')

                self.assertEqual(Post.objects.count(), 1)
                self.assertEqual(Comment.objects.count(), 1)
                self.assertEqual(comment.post, post)
                self.assertEqual(comment.author.username, post.author.username)

            def test_create_batch_of_posts(self):
                # 여러 개의 Post 객체를 한 번에 생성 (각각 다른 User와 연결)
                posts = PostFactory.create_batch(5)
                self.assertEqual(Post.objects.count(), 5)
                self.assertEqual(User.objects.count(), 5) # SubFactory 덕분에 User도 5명 생성

            def test_create_batch_of_comments_for_post(self):
                post = PostFactory()
                # 특정 Post에 대한 댓글 3개 생성
                comments = CommentFactory.create_batch(3, post=post)
                self.assertEqual(post.comments.count(), 3)
        ```

### 8.2. Test Coverage (테스트 커버리지): 코드 품질 지표

-   **`coverage.py`**: 테스트 코드가 실제 애플리케이션 코드의 몇 퍼센트를 실행하는지 측정하는 도구입니다. 코드의 어떤 부분이 테스트되지 않았는지 시각적으로 보여주어, 테스트를 보강해야 할 부분을 식별하는 데 도움을 줍니다.

    1.  **설치**: `pip install coverage`
    2.  **테스트 실행 및 커버리지 측정**: `coverage run manage.py test`
    3.  **보고서 생성**: `coverage html` (프로젝트 루트에 `htmlcov/` 디렉토리가 생성되고, 웹 브라우저로 열어볼 수 있는 상세 보고서가 포함됩니다.)
    4.  **보고서 확인**: `htmlcov/index.html` 파일을 웹 브라우저로 열어 각 파일별 커버리지와 테스트되지 않은 코드 라인을 확인할 수 있습니다.

-   **실무적 관점**: 
    -   **목표 설정**: 100% 커버리지가 항상 목표가 될 필요는 없습니다. 중요한 비즈니스 로직, 복잡한 조건 분기, 에러 처리 로직 등 핵심적인 부분에 대한 커버리지를 높이는 것이 더 중요합니다.
    -   **CI/CD 통합**: CI/CD 파이프라인에 커버리지 측정을 통합하고, 특정 임계값(예: 80%) 미만일 경우 빌드를 실패하도록 설정하여 코드 품질을 지속적으로 관리합니다.
        ```ini
        # .coveragerc (프로젝트 루트에 생성)
        [run]
        source = myapp, another_app # 커버리지를 측정할 앱 지정

        [report]
        show_missing = True # 테스트되지 않은 라인 표시
        fail_under = 80 # 커버리지가 80% 미만이면 실패
        ```
        `coverage report --fail-under=80` 명령으로 CI/CD에서 커버리지 임계값을 확인할 수 있습니다.

-   **테스트 커버리지의 한계**: 커버리지가 높다고 해서 버그가 없다는 것을 의미하지는 않습니다. 테스트 코드가 단순히 코드를 실행만 하고 올바른 결과를 검증하지 않는다면, 커버리지는 높지만 실제로는 무의미한 테스트가 될 수 있습니다. 중요한 것은 **의미 있는 테스트**를 작성하는 것입니다.