<h2>Django Backend: 테스팅과 품질 관리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django 애플리케이션의 품질과 안정성을 보장하기 위한 테스트 작성법을 학습하는 것을 목표로 합니다. Django의 내장 테스트 프레임워크를 사용하여 단위 테스트와 통합 테스트를 작성하고, Mocking, Factory Boy, 테스트 커버리지 등 실무적인 테스트 기법을 이해합니다.</p>

<h2>목차</h2> 

- [1. 테스트 (Testing)](#1-테스트-testing)
  - [1.1. 테스트의 중요성](#11-테스트의-중요성)
  - [1.2. Django 테스트 프레임워크](#12-django-테스트-프레임워크)
  - [1.3. 단위 테스트 (Unit Tests)](#13-단위-테스트-unit-tests)
  - [1.4. 통합 테스트 (Integration Tests)](#14-통합-테스트-integration-tests)
  - [1.5. 테스트 실행](#15-테스트-실행)
  - [1.6. 테스트 심화 - Mocking](#16-테스트-심화---mocking)
    - [1.6.1. Mocking의 필요성](#161-mocking의-필요성)
    - [1.6.2. `unittest.mock` 사용 예시](#162-unittestmock-사용-예시)
  - [1.7. 테스팅 심화](#17-테스팅-심화)
    - [1.7.1. Factory Boy](#171-factory-boy)
    - [1.7.2. Test Coverage (테스트 커버리지)](#172-test-coverage-테스트-커버리지)

---

## 1. 테스트 (Testing)

소프트웨어 개발에서 테스트는 코드의 품질을 보장하고, 버그를 조기에 발견하며, 변경 사항이 기존 기능에 영향을 미치지 않음을 확인하는 데 필수적인 과정입니다. Django는 강력한 테스트 프레임워크를 내장하고 있어 효율적인 테스트 작성을 돕습니다.

### 1.1. 테스트의 중요성
*   **버그 조기 발견:** 개발 초기 단계에서 버그를 발견하고 수정하는 것이 배포 후 발견하는 것보다 훨씬 비용이 적게 듭니다.
*   **코드 품질 향상:** 테스트 가능한 코드를 작성하는 과정에서 자연스럽게 모듈화되고 응집도 높은 코드를 작성하게 됩니다.
*   **회귀 방지:** 코드 변경이나 기능 추가 시 기존 기능이 오작동하지 않음을 보장합니다.
*   **문서화:** 테스트 코드는 해당 기능이 어떻게 작동해야 하는지에 대한 살아있는 문서 역할을 합니다.
*   **리팩토링 용이성:** 테스트 스위트가 잘 갖춰져 있으면, 코드 리팩토링 시에도 안심하고 변경을 적용할 수 있습니다.

- **테스팅 전략 및 범위: 무엇을 테스트해야 하는가?**
    - **Model:** 모델의 커스텀 메서드, `save()` 메서드 오버라이드 등 비즈니스 로직이 포함된 부분을 테스트합니다. 필드 속성 자체보다는 모델의 '동작'을 검증하는 데 집중합니다.
    - **View:** 뷰의 핵심 로직, 즉 특정 요청(GET, POST 등)에 대해 올바른 응답(상태 코드, 템플릿, 리다이렉트)을 반환하는지, 조건부 로직이 제대로 동작하는지, 권한 제어가 올바른지 등을 테스트합니다.
    - **Form:** 폼의 유효성 검사 로직, 특히 여러 필드에 걸친 복잡한 `clean()` 메서드나 커스텀 validator를 테스트합니다.
    - **Service/Business Logic:** 뷰나 모델에서 분리된 비즈니스 로직(서비스 계층)이 있다면, 해당 로직을 독립적으로 테스트하는 것이 중요합니다.
    - **테스트 커버리지(Test Coverage):** 전체 코드 중 테스트 코드가 얼마나 실행했는지를 나타내는 지표입니다. `coverage.py`와 같은 도구를 사용하여 측정할 수 있으며, 높은 커버리지는 코드의 신뢰도를 높이는 데 도움이 됩니다. 하지만 100% 커버리지가 모든 버그를 막아주는 것은 아니므로, 중요하고 복잡한 로직을 중심으로 효과적인 테스트를 작성하는 것이 더 중요합니다.

### 1.2. Django 테스트 프레임워크
Django는 Python의 `unittest` 모듈을 기반으로 확장된 테스트 프레임워크를 제공합니다. `django.test.TestCase` 클래스는 Django 환경(데이터베이스, 설정 등)을 자동으로 설정하여 테스트를 용이하게 합니다.

### 1.3. 단위 테스트 (Unit Tests)
단위 테스트는 애플리케이션의 가장 작은 단위(함수, 메서드, 모델)가 예상대로 작동하는지 독립적으로 검증하는 테스트입니다.

*   **모델 테스트:** 모델의 필드 정의, `__str__` 메서드, 커스텀 메서드 등이 올바르게 작동하는지 확인합니다.
    ```python
    # myapp/tests.py
    from django.test import TestCase
    from myapp.models import MyModel

    class MyModelTest(TestCase):
        def setUp(self):
            # 테스트를 위한 데이터 생성
            MyModel.objects.create(name="Test Item", value=10)

        def test_my_model_creation(self):
            item = MyModel.objects.get(name="Test Item")
            self.assertEqual(item.value, 10)
            self.assertEqual(str(item), "Test Item") # __str__ 메서드 테스트
    ```

*   **폼 테스트:** 폼의 유효성 검사, 필드 렌더링, 데이터 저장 등이 올바르게 작동하는지 확인합니다.
    ```python
    # myapp/tests.py
    from django.test import TestCase
    from myapp.forms import MyForm

    class MyFormTest(TestCase):
        def test_valid_form(self):
            form = MyForm({'name': 'Valid Name', 'value': 20})
            self.assertTrue(form.is_valid())

        def test_invalid_form(self):
            form = MyForm({'name': '', 'value': 'abc'}) # 이름 누락, 값 타입 오류
            self.assertFalse(form.is_valid())
            self.assertIn('name', form.errors)
            self.assertIn('value', form.errors)
    ```

### 1.4. 통합 테스트 (Integration Tests)
통합 테스트는 여러 컴포넌트(모델, 뷰, 템플릿, URL)가 함께 작동하여 전체 기능이 올바르게 수행되는지 검증하는 테스트입니다. Django의 `Client` 클래스를 사용하여 HTTP 요청을 시뮬레이션할 수 있습니다.

*   **뷰 테스트:** 뷰가 올바른 HTTP 응답을 반환하고, 템플릿을 제대로 렌더링하며, 데이터베이스와 상호작용하는지 확인합니다.
    ```python
    # myapp/tests.py
    from django.test import TestCase, Client
    from django.urls import reverse
    from myapp.models import MyModel

    class MyViewTest(TestCase):
        def setUp(self):
            self.client = Client()
            self.item = MyModel.objects.create(name="Test Item", value=10)

        def test_list_view(self):
            response = self.client.get(reverse('my_list_url_name')) # urls.py에 정의된 URL name 사용
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, "Test Item") # 템플릿에 내용이 포함되어 있는지 확인
            self.assertTemplateUsed(response, 'myapp/my_list.html') # 사용된 템플릿 확인

        def test_detail_view(self):
            response = self.client.get(reverse('my_detail_url_name', args=[self.item.id]))
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, "Test Item")

        def test_create_view_post(self):
            response = self.client.post(reverse('my_create_url_name'), {'name': 'New Item', 'value': 30})
            self.assertEqual(response.status_code, 302) # 성공 시 리다이렉트 (302 Found)
            self.assertTrue(MyModel.objects.filter(name='New Item').exists())
    ```

### 1.5. 테스트 실행
프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 모든 테스트를 실행할 수 있습니다.
```bash
python manage.py test
```
*   특정 앱의 테스트만 실행: `python manage.py test myapp`
*   특정 테스트 케이스만 실행: `python manage.py test myapp.tests.MyModelTest`
*   특정 테스트 메서드만 실행: `python manage.py test myapp.tests.MyModelTest.test_my_model_creation`

**실무적 관점:**
*   **테스트 주도 개발 (TDD):** 기능을 구현하기 전에 테스트 코드를 먼저 작성하는 TDD 방법론을 적용하면 코드의 설계 품질을 높일 수 있습니다.
*   **CI/CD 파이프라인 통합:** 테스트는 CI/CD(지속적 통합/지속적 배포) 파이프라인의 핵심 단계입니다. 코드가 변경될 때마다 자동으로 테스트를 실행하여 문제가 없는지 확인하고, 통과해야만 다음 단계(배포)로 진행되도록 설정합니다.
*   **테스트 커버리지:** 코드의 몇 퍼센트가 테스트 코드로 커버되는지 측정하는 지표입니다. `coverage.py`와 같은 도구를 사용하여 테스트 커버리지를 측정하고 관리할 수 있습니다. 높은 테스트 커버리지는 코드의 안정성을 높이는 데 기여합니다.

### 1.6. 테스트 심화 - Mocking

단위 테스트를 작성할 때, 테스트 대상 코드가 외부 서비스, 데이터베이스, 파일 시스템 등과 같은 복잡한 의존성을 가질 수 있습니다. Mocking은 이러한 의존성을 가짜(mock) 객체로 대체하여 테스트 대상 코드만 독립적으로 테스트할 수 있도록 돕는 기법입니다.

#### 1.6.1. Mocking의 필요성

*   **독립성:** 테스트가 외부 요인에 의존하지 않고 독립적으로 실행될 수 있도록 합니다.
*   **속도:** 실제 데이터베이스 접근이나 네트워크 요청 없이 메모리 내에서 빠르게 테스트를 실행할 수 있습니다.
*   **재현성:** 외부 서비스의 상태 변화에 관계없이 테스트 결과를 일관되게 재현할 수 있습니다.
*   **특정 시나리오 테스트:** 오류 발생, 특정 값 반환 등 실제로는 발생하기 어렵거나 제어하기 어려운 시나리오를 쉽게 시뮬레이션할 수 있습니다.

#### 1.6.2. `unittest.mock` 사용 예시

Python의 내장 `unittest.mock` 모듈을 사용하여 Mock 객체를 생성하고 관리할 수 있습니다.

**가상 시나리오:** 외부 API를 호출하여 날씨 정보를 가져오는 함수를 테스트하고 싶습니다.

```python
# myapp/weather_service.py
import requests

def get_current_weather(city):
    api_key = "YOUR_API_KEY" # 실제 API 키
    url = f"http://api.weather.com/current?city={city}&apiKey={api_key}"
    response = requests.get(url)
    response.raise_for_status() # HTTP 에러 발생 시 예외 처리
    return response.json()

# myapp/tests.py
from django.test import TestCase
from unittest.mock import patch # patch 데코레이터 임포트
from myapp.weather_service import get_current_weather

class WeatherServiceTest(TestCase):
    @patch('myapp.weather_service.requests.get') # requests.get 함수를 Mocking
    def test_get_current_weather_success(self, mock_get):
        # Mock 객체의 반환 값 설정
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "city": "Seoul",
            "temperature": 25,
            "condition": "Sunny"
        }

        weather_data = get_current_weather("Seoul")

        # requests.get이 올바른 인자로 호출되었는지 확인
        mock_get.assert_called_once_with(
            "http://api.weather.com/current?city=Seoul&apiKey=YOUR_API_KEY"
        )
        # 반환된 데이터가 예상과 일치하는지 확인
        self.assertEqual(weather_data['temperature'], 25)
        self.assertEqual(weather_data['condition'], "Sunny")

    @patch('myapp.weather_service.requests.get')
    def test_get_current_weather_api_error(self, mock_get):
        # Mock 객체가 HTTP 에러를 발생시키도록 설정
        mock_get.return_value.status_code = 404
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError

        with self.assertRaises(requests.exceptions.HTTPError):
            get_current_weather("NonExistentCity")

        mock_get.assert_called_once()
```
### 1.7. 테스팅 심화

테스트는 코드 품질과 안정성을 보장하는 핵심 요소입니다. 다음 도구와 개념을 익히면 테스트 작성 및 관리가 더욱 효율적입니다.

#### 1.7.1. Factory Boy

*   테스트를 위해 모델 객체를 생성할 때 `factory-boy` 라이브러리를 사용하면 반복적이고 복잡한 테스트 데이터 생성을 자동화하고, 더 깔끔하고 효율적인 테스트 코드를 작성할 수 있습니다.
*   **장점:** 테스트 코드의 가독성을 높이고, 테스트 데이터의 일관성을 유지하며, 테스트 작성 시간을 단축시킵니다.

#### 1.7.2. Test Coverage (테스트 커버리지)

*   `coverage.py`와 같은 도구를 사용하여 테스트 코드가 실제 애플리케이션 코드의 몇 퍼센트를 실행하는지 측정할 수 있습니다.
*   **장점:** 테스트되지 않은 코드 영역을 식별하여 잠재적인 버그를 줄이고, 코드 품질을 객관적으로 평가하는 지표로 활용할 수 있습니다. 높은 테스트 커버리지는 코드의 안정성을 높이는 데 기여합니다.
