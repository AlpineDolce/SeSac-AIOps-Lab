# 🐍 Python 객체 지향 프로그래밍(OOP) 기초 (Day 8)

> **이 문서의 목적**: 이 문서는 부트캠프 8일차에 학습한 **객체 지향 프로그래밍(Object-Oriented Programming, OOP)**의 핵심 개념과 Python에서의 구현 방법을 깊이 있게 정리한 자료입니다. 클래스, 객체, 모듈, 패키지의 개념을 명확히 이해하고, 이를 실제 프로젝트에 적용하여 코드의 재사용성과 구조를 개선하는 능력을 보여주는 것을 목표로 합니다.

---

## 목차

1.  [**객체 지향 프로그래밍(OOP)이란?**](#1-객체-지향-프로그래밍oop이란)
    -   [OOP의 핵심: 클래스와 객체](#oop의-핵심-클래스와-객체)
    -   [클래스, 객체, 인스턴스의 관계](#클래스-객체-인스턴스의-관계)
2.  [**Python 클래스(Class) 마스터하기**](#2-python-클래스class-마스터하기)
    -   [클래스의 기본 구조](#클래스의-기본-구조)
    -   [생성자: `__init__` 메서드](#생성자-__init__-메서드)
    -   [인스턴스 변수 vs 클래스 변수](#인스턴스-변수-vs-클래스-변수)
3.  [**코드 모듈화: 모듈과 패키지**](#3-코드-모듈화-모듈과-패키지)
    -   [모듈(Module): 코드 재사용의 기본 단위](#모듈module-코드-재사용의-기본-단위)
    -   [패키지(Package): 모듈을 체계적으로 관리하기](#패키지package-모듈을-체계적으로-관리하기)
    -   [`if __name__ == "__main__":`의 의미](#if-__name__--__main__의-의미)
4.  [**실전 프로젝트: 객체 지향 주급 관리 시스템**](#4-실전-프로젝트-객체-지향-주급-관리-시스템)
    -   [프로젝트 구조 설계](#프로젝트-구조-설계)
    -   [클래스 및 모듈 구현](#클래스-및-모듈-구현)

---

## 1. 객체 지향 프로그래밍(OOP)이란?

객체 지향 프로그래밍은 현실 세계의 사물이나 개념을 **객체(Object)**로 모델링하고, 이 객체들 간의 상호작용을 통해 프로그램을 설계하는 방식입니다. 이는 코드의 재사용성을 높이고, 유지보수를 용이하게 만들어 대규모 프로젝트 개발에 매우 효과적입니다.

### OOP의 핵심: 클래스와 객체

-   **클래스(Class)**: 객체를 만들기 위한 **설계도** 또는 **틀(template)**입니다. 클래스에는 객체가 가질 데이터(**속성, Attribute**)와 객체가 수행할 동작(**메서드, Method**)이 정의되어 있습니다.
    -   *비유: `붕어빵 틀`, `자동차 설계도`*

-   **객체(Object)**: 클래스라는 설계도를 바탕으로 메모리에 **실체화된 존재**입니다. 각각의 객체는 자신만의 고유한 속성 값을 가질 수 있습니다.
    -   *비유: `팥 붕어빵`, `슈크림 붕어빵`, `실제로 만들어진 나의 자동차`*

### 클래스, 객체, 인스턴스의 관계

| 용어 | 설명 | 예시 |
| :--- | :--- | :--- |
| **클래스** | 객체를 정의하는 설계도. | `class Car:` |
| **객체** | 클래스로부터 생성된 메모리상의 실체. | `my_car = Car()`에서 `my_car`가 가리키는 메모리 공간. |
| **인스턴스** | 특정 클래스로부터 생성된 객체임을 강조하는 용어. | `my_car`는 `Car` 클래스의 **인스턴스**이다. |

> **결론**: "객체"와 "인스턴스"는 거의 같은 의미로 사용되지만, "인스턴스"는 특정 클래스와의 관계를 명확히 할 때 주로 사용됩니다.

---

## 2. Python 클래스(Class) 마스터하기

### 클래스의 기본 구조

```python
class ClassName:
    # 클래스 변수 (모든 인스턴스가 공유)
    class_variable = "I am a class variable"

    # 생성자 메서드
    def __init__(self, param1, param2):
        # 인스턴스 변수 (각 인스턴스마다 고유)
        self.attribute1 = param1
        self.attribute2 = param2

    # 인스턴스 메서드
    def method(self, param):
        # self를 통해 인스턴스 변수에 접근
        print(f"Attribute1 is {self.attribute1}")
```

### 생성자: `__init__` 메서드

`__init__`은 **생성자(Constructor)**라 불리는 특별한 메서드입니다. 클래스로부터 객체(인스턴스)가 생성될 때 **자동으로 호출**되며, 주로 해당 인스턴스가 가질 초기 속성(인스턴스 변수)을 설정하는 역할을 합니다.

-   **`self`**: `__init__`의 첫 번째 매개변수인 `self`는 **생성되는 인스턴스 자신**을 가리킵니다. Python이 자동으로 전달해주므로, 호출 시에는 `self`에 해당하는 인자를 넘기지 않습니다.

```python
class Person:
    def __init__(self, name, age):
        print(f"'{name}' 객체를 생성합니다.")
        self.name = name # 인스턴스 변수 name 초기화
        self.age = age   # 인스턴스 변수 age 초기화

# Person 클래스의 인스턴스 생성
# 이 때 __init__ 메서드가 자동으로 호출됨
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

print(f"{person1.name} is {person1.age} years old.") # Alice is 30 years old.
```

### 인스턴스 변수 vs 클래스 변수

-   **인스턴스 변수 (Instance Variable)**:
    -   `self.변수명` 형태로 `__init__` 안에서 주로 정의됩니다.
    -   각 인스턴스마다 **독립적인 공간**을 가지므로, 한 인스턴스에서 값을 변경해도 다른 인스턴스에 영향을 주지 않습니다.

-   **클래스 변수 (Class Variable)**:
    -   클래스 선언 바로 아래에 정의됩니다.
    -   해당 클래스로부터 생성된 **모든 인스턴스가 공유**하는 변수입니다.

```python
class Car:
    # 클래스 변수: 모든 Car 인스턴스가 공유
    wheel_count = 4

    def __init__(self, color):
        # 인스턴스 변수: 각 Car 인스턴스마다 고유
        self.color = color

car1 = Car("Red")
car2 = Car("Blue")

print(f"Car 1: {car1.color}, Wheels: {car1.wheel_count}") # Red, 4
print(f"Car 2: {car2.color}, Wheels: {car2.wheel_count}") # Blue, 4

# 클래스 변수 변경 (모든 인스턴스에 반영됨)
Car.wheel_count = 3
print(f"After change, Car 1 wheels: {car1.wheel_count}") # 3
```

> **⚠️ 주의**: `list`나 `dict`와 같은 **변경 가능한(Mutable) 객체**를 클래스 변수로 사용하면, 한 인스턴스에서의 변경이 모든 인스턴스에 영향을 미쳐 예상치 못한 버그를 유발할 수 있습니다. 이런 경우, `__init__`에서 인스턴스 변수로 초기화하는 것이 안전합니다.

---

## 3. 코드 모듈화: 모듈과 패키지

### 모듈(Module): 코드 재사용의 기본 단위

모듈은 함수, 클래스, 변수 등을 모아놓은 하나의 Python 파일(`.py`)입니다. `import` 키워드를 사용하여 다른 파일에서 모듈의 내용을 가져와 재사용할 수 있습니다.

```python
# my_math.py (모듈 파일)
PI = 3.14159

def add(a, b):
    return a + b

class Circle:
    def __init__(self, radius):
        self.radius = radius
    def get_area(self):
        return PI * self.radius**2
```

```python
# main.py (모듈 사용 파일)
import my_math

print(my_math.add(5, 3)) # 8

c = my_math.Circle(10)
print(c.get_area()) # 314.159
```

### 패키지(Package): 모듈을 체계적으로 관리하기

패키지는 여러 관련 모듈들을 디렉터리 구조로 묶어 관리하는 방법입니다. 패키지로 인식되려면 해당 디렉터리 안에 `__init__.py` 파일이 있어야 합니다 (Python 3.3+ 에서는 필수는 아니지만, 호환성을 위해 포함하는 것이 좋습니다).

```
mypackage/
├── main.py
└── my_package/
    ├── __init__.py
    ├── math_ops.py
    └── string_ops.py
```

```python
# main.py
from my_package import math_ops, string_ops

sum_result = math_ops.add(10, 5)
upper_string = string_ops.to_upper("hello")
```

### `if __name__ == "__main__":`의 의미

이 조건문은 Python 파일이 **직접 실행되었는지, 아니면 다른 파일에서 모듈로 임포트되었는지**를 구분하는 역할을 합니다.

-   **직접 실행**: `python my_module.py`와 같이 터미널에서 직접 실행하면, 해당 모듈의 `__name__` 변수에는 `"__main__"`이라는 문자열이 할당됩니다.
-   **모듈로 임포트**: 다른 파일에서 `import my_module`로 불러오면, `__name__` 변수에는 모듈의 이름(`"my_module"`)이 할당됩니다.

이를 통해, 모듈이 직접 실행될 때만 테스트 코드를 동작시키는 등의 용도로 활용할 수 있습니다.

```python
# my_module.py

def my_function():
    return "This is my function."

# 이 파일이 직접 실행될 때만 아래 코드가 동작
if __name__ == "__main__":
    print("모듈을 직접 실행했습니다. 테스트를 시작합니다.")
    result = my_function()
    print(f"테스트 결과: {result}")
```

---

## 4. 실전 프로젝트: 객체 지향 주급 관리 시스템

### 프로젝트 구조 설계

기능별로 역할을 분리하여 두 개의 클래스를 설계하고, 이를 각각의 모듈로 관리합니다.

-   **`WeekPay` 클래스 (`weekpay.py`)**: 한 명의 직원에 대한 정보(이름, 근무시간, 시급)와 주급 계산 기능을 담당합니다. (데이터 중심 객체)
-   **`WeekPayManager` 클래스 (`manager.py`)**: 여러 `WeekPay` 객체들을 리스트로 관리하며, 직원 정보 추가, 검색, 수정, 전체 출력 등 시스템의 전체 로직을 담당합니다. (관리 중심 객체)

### 클래스 및 모듈 구현

#### `weekpay.py`

```python
class WeekPay:
    """한 명의 직원 정보를 관리하는 클래스."""
    def __init__(self, name, work_time, per_pay):
        self.name = name
        self.work_time = work_time
        self.per_pay = per_pay
        self.pay = self.calculate_pay()

    def calculate_pay(self):
        """주급을 계산하여 반환합니다."""
        return self.work_time * self.per_pay

    def display(self):
        """직원 정보를 출력합니다."""
        print(f"{self.name}\t{self.work_time}시간\t{self.per_pay}원\t{self.pay}원")

if __name__ == "__main__":
    # 모듈 테스트 코드
    wp = WeekPay("TestUser", 40, 10000)
    wp.display()
```

#### `manager.py`

```python
from weekpay import WeekPay # weekpay 모듈에서 WeekPay 클래스를 임포트

class WeekPayManager:
    """전체 직원 정보를 관리하는 시스템 클래스."""
    def __init__(self):
        self.worker_list = []

    def add_worker(self):
        # ... 직원 추가 로직 ...
        pass

    def search_worker(self):
        # ... 직원 검색 로직 ...
        pass
    
    def print_all(self):
        print("\n--- 전체 직원 주급 정보 ---")
        print("이름\t근무시간\t시급\t주급")
        for worker in self.worker_list:
            worker.display()
        print("-" * 30)

    def start_system(self):
        # ... 메인 메뉴 루프 ...
        pass

if __name__ == "__main__":
    manager = WeekPayManager()
    # 테스트를 위해 기본 데이터 추가
    manager.worker_list.append(WeekPay("Alice", 40, 15000))
    manager.worker_list.append(WeekPay("Bob", 35, 18000))
    manager.print_all()
```

> **설계의 장점**:
> - **역할 분리**: `WeekPay`는 데이터 표현에, `WeekPayManager`는 시스템 로직에 집중하여 코드가 명확해집니다.
> - **유지보수 용이**: 주급 계산 방식이 변경되면 `WeekPay` 클래스만 수정하면 되고, 검색 기능이 변경되면 `WeekPayManager`만 수정하면 됩니다.
> - **확장성**: 새로운 기능(예: 파일 저장, 삭제)을 추가할 때 `WeekPayManager`에 새로운 메서드를 추가하는 방식으로 쉽게 확장할 수 있습니다.

---

[⏮️ 이전 문서](./0502_Python정리.md) | [다음 문서 ⏭️](./0508_Python정리.md)
