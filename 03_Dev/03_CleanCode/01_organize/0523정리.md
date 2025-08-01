# 🧹 클린 코드(Clean Code) 핵심 가이드 (Day 21)

> **“나중에”는 결코 오지 않는다. - 르블랑의 법칙**

> 이 문서는 로버트 C. 마틴의 "클린 코드"를 기반으로, 읽기 쉽고 유지보수하기 쉬우며 견고한 코드를 작성하기 위한 핵심 원칙과 실천 방법을 정리한 가이드입니다. 단순히 동작하는 코드를 넘어, 동료와 미래의 나를 위한 전문성 있는 코드를 지향합니다.

---

## 목차

1.  [**클린 코드란?**](#1-클린-코드란)
2.  [**의미 있는 이름 (Meaningful Names)**](#2-의미-있는-이름-meaningful-names)
3.  [**함수 (Functions)**](#3-함수-functions)
    -   [3.1. 작게 만들어라 (하나의 일만 할 것)](#31-작게-만들어라-하나의-일만-할-것)
    -   [3.2. 부수 효과를 일으키지 마라 (No Side Effects)](#32-부수-효과를-일으키지-마라-no-side-effects)
    -   [3.3. 명령-조회 분리 (Command-Query Separation)](#33-명령-조회-분리-command-query-separation)
4.  [**주석 (Comments)**](#4-주석-comments)
5.  [**형식 (Formatting)**](#5-형식-formatting)
6.  [**객체와 자료 구조 (Objects and Data Structures)**](#6-객체와-자료-구조-objects-and-data-structures)
    -   [6.1. 자료 추상화와 디미터 법칙](#61-자료-추상화와-디미터-법칙)
    -   [6.2. 리플렉션(Reflection)과 매직 메서드](#62-리플렉션reflection과-매직-메서드)
    -   [6.3. 파이썬의 매직 메서드 활용](#63-파이썬의-매직-메서드-활용)
7.  [**오류 처리 (Error Handling)**](#7-오류-처리-error-handling)
    -   [7.1. 오류 코드보다 예외를 사용하라](#71-오류-코드보다-예외를-사용하라)
    -   [7.2. null을 반환하거나 전달하지 마라](#72-null을-반환하거나-전달하지-마라)
8.  [**경계 (Boundaries)**](#8-경계-boundaries)
9.  [**단위 테스트 (Unit Tests)와 TDD**](#9-단위-테스트-unit-tests와-tdd)
    -   [9.1. TDD 3원칙과 F.I.R.S.T 원칙](#91-tdd-3원칙과-first-원칙)
10. [**클래스 (Classes)와 SOLID 원칙**](#10-클래스-classes와-solid-원칙)
    -   [10.1. 클래스는 작아야 한다 (단일 책임 원칙)](#101-클래스는-작아야-한다-단일-책임-원칙)
    -   [10.2. SOLID 객체지향 설계 원칙](#102-solid-객체지향-설계-원칙)
11. [**점진적인 개선**](#11-점진적인-개선)
---

## 1. 클린 코드란?

클린 코드란 단순히 동작하는 코드를 넘어, **가독성이 높고, 유지보수가 용이하며, 중복이 없고, 테스트가 쉬운 코드**를 의미합니다. 이는 장기적인 관점에서 프로젝트의 생산성과 안정성을 결정하는 핵심 요소입니다.

---

## 2. 의미 있는 이름 (Meaningful Names)

변수, 함수, 클래스의 이름은 그것이 무엇을 하고, 왜 존재하는지를 명확히 설명해야 합니다.

-   **의도를 분명히 밝혀라**: 이름만 보고도 용도를 짐작할 수 있어야 합니다.
    ```python
    # 나쁜 예시
    d = 10 # 경과 시간(단위: 날짜)

    # 좋은 예시
    elapsed_time_in_days = 10
    ```
-   **검색하기 쉬운 이름을 사용하라**: 간단한 이름보다 검색 가능한 이름을 사용하는 것이 디버깅에 유리합니다.
    ```python
    # 나쁜 예시
    if a > 80:
        # ...

    # 좋은 예시
    MAX_SCORE = 80
    if score > MAX_SCORE:
        # ...
    ```
-   **자신의 프로젝트 네이밍 컨벤션을 따르라**: `snake_case`, `camelCase`, `PascalCase` 등 프로젝트 전체에서 일관된 규칙을 적용해야 합니다.
-   **그릇된 정보를 피하라**: 실제 `List`가 아닌데 `accountList`라고 이름 짓거나, 타입이 다른데 비슷한 이름을 사용하지 마세요. `hp`라는 이름은 유닉스 시스템 이름을 연상시키므로 `hypotenuse`라고 명확히 해야 합니다.
-   **검색하기 쉬운 이름을 사용하라**: `WORK_DAYS_PER_WEEK = 5`는 숫자 `5`보다 검색과 의미 파악이 훨씬 쉽습니다. 이름 길이는 짧은 것보다 명확한 것이 중요합니다.
-   **자신의 프로젝트 네이밍 컨벤션을 따르라**: `snake_case`, `camelCase`, `PascalCase` 등 프로젝트 전체에서 일관된 규칙을 적용해야 합니다.
---

## 3. 함수 (Functions)

### 3.1. 작게 만들어라 (하나의 일만 할 것)
-   함수는 **단 한 가지의 일**만 책임져야 하며, 그 일을 잘해야 합니다. (단일 책임 원칙, SRP)
-   함수 내에서 여러 논리적 단계를 수행한다면, 각 단계를 더 작은 함수로 분리하는 것을 고려해야 합니다.

    ```python
    # 나쁜 예시
    def process_data(data):
        clean = [d.strip() for d in data]
        print(clean)
        return len(clean)

    # 좋은 예시
    def clean_data(data):
        return [d.strip() for d in data]

    def print_data(data):
        print(data)

    def count_data(data):
        return len(data)
    ```

### 3.2. 부수 효과를 일으키지 마라 (No Side Effects)
-   함수는 입력값을 받아 결과물을 반환하는 것 외에, 함수 외부의 상태를 변경하는 **부수 효과(Side Effect)**를 일으켜서는 안 됩니다.
-   예를 들어, 함수가 전역 변수나 인자로 받은 객체의 상태를 직접 수정하는 것은 예측 불가능한 버그의 원인이 됩니다.

    ```python
    # 나쁜 예시 (전역 변수를 수정하는 부수 효과)
    user_name = "Guest"
    def greet_user(name):
        global user_name
        user_name = name # 함수 외부의 상태를 변경
        return f"Hello, {name}"

    # 좋은 예시 (부수 효과 없음)
    def create_greeting(name):
        return f"Hello, {name}"
    ```

### 3.3. 명령-조회 분리 (Command-Query Separation)
-   함수는 **무언가를 수행(명령)하거나, 무언가를 반환(조회)하거나** 둘 중 하나만 해야 합니다.
-   객체의 상태를 변경하는 함수(명령)가 값을 반환하면 혼란을 야기할 수 있습니다.

    ```python
    # 나쁜 예시 (상태를 변경하면서 결과를 반환)
    def set_attribute(obj, key, value):
        obj[key] = value
        return True # 성공 여부를 반환? 혼란스럽다.

    # 좋은 예시 (명령과 조회를 분리)
    def set_attribute(obj, key, value):
        obj[key] = value

    def attribute_exists(obj, key):
        return key in obj
    ```
-   **함수 인수(Arguments)는 적을수록 좋다**: 최적의 인수 개수는 0개이며, 1개, 2개 순으로 좋습니다. 3개를 넘어가면 함수가 너무 많은 일을 하려 한다는 신호일 수 있습니다. 인수가 3개 이상 필요하다면, `Point(x, y)`처럼 별도의 클래스로 묶는 것을 고려하세요.
-   **플래그 인수를 쓰지 마라**: `render(is_test_mode=True)`와 같이 `boolean` 플래그를 넘기는 것은 함수 내부에 `if/else` 분기가 있다는 뜻이며, 이는 함수가 두 가지 이상의 일을 한다는 증거입니다. `render_for_test()`와 `render_for_production()`으로 함수를 분리하세요.

---

## 4. 주석 (Comments)

> **나쁜 코드에 주석을 달지 마라. 코드를 새로 짜라. - 브라이언 커니핸, P.J. 플라우거**

-   주석은 코드로 의도를 표현하지 못했을 때 사용하는 최후의 수단입니다.
-   **좋은 주석**:
    -   법적인 정보: 법적인 정보(`Copyright...`), 복잡한 알고리즘에 대한 의도 설명, 결과를 경고하는 주석(`// WARNING: ...`), `TODO`나 `FIXME` 같은 태그
    -   의도를 설명하는 주석: 복잡한 정규식이나 알고리즘의 의도를 설명
    -   결과를 경고하는 주석: `// WARNING: 이 테스트는 실행하는 데 오래 걸립니다.`
    -   `TODO` 주석: 앞으로 해야 할 일을 명시
-   **나쁜 주석**:
    -   코드를 그대로 설명하는 주석: `i += 1 # i를 1 증가시킴`
    -   변경 이력을 기록하는 주석(Git이 할 일), 오해의 소지가 있는 주석, 주석 처리된 코드 등


---

## 5. 형식 (Formatting)

-   팀의 코드 스타일 가이드를 따르는 것이 가장 중요합니다. 일관된 형식은 코드 가독성을 크게 향상시킵니다.
-   **수직 거리**: 연관된 코드는 가까이 배치하고, 개념적으로 다른 코드는 빈 줄로 분리합니다.
-   **수평 거리**: 적절한 들여쓰기와 공백을 사용하여 코드의 계층과 연산자 우선순위를 명확히 표현합니다.
-   **자동화 도구**: `black`, `isort`, `flake8`과 같은 코드 포맷터를 사용하여 팀의 스타일을 강제하고 일관성을 유지하세요.

---

## 6. 객체와 자료 구조 (Objects and Data Structures)

-   **자료 구조(Data Structures)**: 자료를 그대로 노출하며, 별다른 동작을 제공하지 않습니다. (예: `dict`, `list`)
-   **객체(Objects)**: 자료를 숨기고, 해당 자료를 다루는 동작(메서드)을 공개합니다.


### 6.1. 자료 추상화와 디미터 법칙
-   객체는 내부 구현을 숨기고, 사용자가 조작할 수 있는 추상적인 인터페이스를 제공해야 합니다.
-   **디미터 법칙(Law of Demeter)**: "낯선 이에게 말하지 마라." 즉, 모듈은 자신이 조작하는 객체의 속사정을 몰라야 합니다. `a.getB().getC().doSomething()`과 같은 코드는 피해야 합니다.

    ```python
    # 나쁜 예시 (기차 충돌 코드)
    final_destination = user.get_profile().get_address().get_city()

    # 좋은 예시 (객체에 메시지를 보내 일을 시킴)
    final_destination = user.get_destination_city()
    ```
### 6.2. 리플렉션(Reflection)과 매직 메서드
-   **리플렉션**: 동적으로 객체의 속성이나 메서드에 접근하는 기능입니다. 코드의 유연성을 높이지만, 남용하면 코드 추적이 어려워집니다.
    ```python
    class User:
        def __init__(self, name):
            self.name = name

    user = User("Alice")
    # getattr을 사용한 동적 접근
    print(getattr(user, "name", "Unknown"))

### 6.3. 파이썬의 매직 메서드 활용
-   연산자 오버로딩(`__add__`, `__sub__` 등)과 내장 함수 동작(`__str__`, `__len__` 등)을 위한 매직 메서드를 구현하여 객체를 파이썬의 기본 타입처럼 자연스럽게 사용할 수 있도록 만드세요.

    ```python
    class Money:
        def __init__(self, amount):
            self.amount = amount

        def __add__(self, other):
            return Money(self.amount + other.amount)

        def __str__(self):
            return f"{self.amount}원"

    wallet1 = Money(1000)
    wallet2 = Money(5000)
    total = wallet1 + wallet2
    print(total) # 출력: 6000원
    ```

---

## 7. 오류 처리 (Error Handling)

오류 처리는 프로그램의 정상적인 로직과 분리되어야 코드가 깔끔해집니다.

### 7.1. 오류 코드보다 예외를 사용하라
-   오류가 발생할 때마다 오류 코드를 반환하고 호출자가 이를 확인하는 방식은 코드를 복잡하게 만듭니다.
-   대신, 예외(Exception)를 발생시켜 오류 처리 로직을 명확히 분리하세요.

    ```python
    # 나쁜 예시
    def get_user(id):
        if not user_exists(id):
            return -1 # 오류 코드 반환
        # ...

    # 좋은 예시
    def get_user(id):
        if not user_exists(id):
            raise UserNotFoundError(f"User with id {id} not found.")
        # ...
    ```

### 7.2. null을 반환하거나 전달하지 마라
-   `null`을 반환하는 코드는 호출자에게 `null` 체크를 강제하며, 누락 시 `NullPointerException`을 유발합니다.
-   `null` 대신 예외를 던지거나, 특수 사례 객체(Special Case Object, 예: 빈 리스트)를 반환하는 것을 고려하세요.

---

## 8. 경계 (Boundaries)

-   외부 라이브러리나 시스템 API를 사용할 때는 이를 직접 코드 전체에 흩뿌려 놓지 말고, **경계 인터페이스**로 감싸서 사용하세요.
-   이렇게 하면 외부 시스템의 변경이 우리 코드에 미치는 영향을 최소화할 수 있고, 테스트가 용이해집니다.

    ```python
    # 외부 라이브러리: payment_sdk

    # 경계 인터페이스 (Adapter 패턴)
    class PaymentGateway:
        def process_payment(self, amount, card_info):
            # payment_sdk의 복잡한 로직을 여기서 처리
            payment_sdk.initialize(API_KEY)
            result = payment_sdk.charge(amount, card_info)
            return result.is_success()

    # 우리 시스템의 코드
    def purchase_item(item, user):
        gateway = PaymentGateway()
        if gateway.process_payment(item.price, user.card):
            # 구매 성공 로직
            pass
    ```

---

## 9. 단위 테스트 (Unit Tests)와 TDD

-   클린 코드는 테스트 가능한 코드이며, 테스트 코드는 클린 코드를 유지하는 핵심적인 역할을 합니다.
-   테스트 코드는 실제 코드만큼 중요하며, 깨끗하고 가독성 높게 유지되어야 합니다.

### 9.1. TDD 3원칙과 F.I.R.S.T 원칙
-   **TDD(Test-Driven Development) 3원칙**:
    1.  실패하는 단위 테스트를 작성할 때까지 실제 코드를 작성하지 않는다.
    2.  컴파일은 실패하지 않으면서, 실행이 실패하는 정도로만 단위 테스트를 작성한다.
    3.  현재 실패하는 테스트를 통과할 정도로만 실제 코드를 작성한다.
-   **클린 테스트의 F.I.R.S.T 원칙**:
    -   **Fast**: 테스트는 빨라야 자주 돌릴 수 있다.
    -   **Independent**: 각 테스트는 서로 독립적이어야 한다.
    -   **Repeatable**: 어떤 환경에서도 반복 가능해야 한다.
    -   **Self-Validating**: 테스트는 `True`/`False`로 결과를 내어 자체적으로 검증되어야 한다.
    -   **Timely**: 테스트는 실제 코드를 구현하기 직전에 제때 작성해야 한다.

---

## 10. 클래스 (Classes)와 SOLID 원칙

### 10.1. 클래스는 작아야 한다 (단일 책임 원칙)
-   함수와 마찬가지로 클래스도 **단 하나의 책임**을 가져야 합니다. 클래스의 이름은 그 책임을 명확히 설명해야 합니다.
-   클래스가 너무 많은 책임을 가지면(응집도가 낮아지면), 관련된 변경이 필요할 때 수정해야 할 부분이 많아지고 버그가 발생하기 쉽습니다.

### 10.2. SOLID 객체지향 설계 원칙

SOLID는 유지보수 및 확장이 용이한 소프트웨어를 만들기 위한 다섯 가지 핵심 설계 원칙입니다.

1.  **SRP (Single Responsibility Principle)**: 단일 책임 원칙
    -   클래스는 변경의 이유가 단 하나여야 한다.
2.  **OCP (Open-Closed Principle)**: 개방-폐쇄 원칙
    -   확장에는 열려 있어야 하고, 변경에는 닫혀 있어야 한다.
3.  **LSP (Liskov Substitution Principle)**: 리스코프 치환 원칙
    -   하위 타입은 언제나 상위 타입으로 교체할 수 있어야 한다.
4.  **ISP (Interface Segregation Principle)**: 인터페이스 분리 원칙
    -   클라이언트는 자신이 사용하지 않는 인터페이스에 의존해서는 안 된다.
5.  **DIP (Dependency Inversion Principle)**: 의존성 역전 원칙
    -   상위 모듈은 하위 모듈에 의존해서는 안 되며, 둘 모두 추상화에 의존해야 한다.

---

## 11. 점진적인 개선

> **“보이스카우트 규칙: 캠프장을 처음 왔을 때보다 더 깨끗하게 해놓고 떠나라.”**

-   기존 코드를 한 번에 모두 개선하려는 시도는 위험합니다. 코드를 수정할 때마다 조금씩 더 깨끗하게 만드는 **점진적인 개선**을 습관화하세요. 변수 이름을 바꾸고, 긴 함수를 나누는 작은 노력들이 모여 시스템 전체를 건강하게 만듭니다.

---
[⏮️ Linux 문서](../../01_Linux/0510_Linux정리.md) | [ Git 문서](../../02_Git/0522_Git.Github정리.md) | [알고리즘 문서 ⏭️](../../../04_Algorithm/01_organize/0602_Algorithm정리.md)