# 🐍 Python 심화: OOP와 코드 구조화, 동시성 프로그래밍 (Day 13)

> **이 문서의 목적**: 이 문서는 부트캠프 13일차에 학습한 Python의 **고급 객체 지향 개념(상속)**, **코드 구조화(패키지)**, 그리고 **성능 향상을 위한 동시성 프로그래밍(`multiprocessing`)**을 깊이 있게 정리한 자료입니다. 복잡한 상속 구조를 이해하고, 잘 구조화된 패키지를 만들며, 병렬 처리를 통해 CPU를 최대한 활용하는 능력을 보여주는 것을 목표로 합니다.

---

## 목차

1.  [**상속(Inheritance): 코드 재사용의 기술**](#1-상속inheritance-코드-재사용의-기술)
    -   [기본 상속과 메서드 오버라이딩](#11-기본-상속과-메서드-오버라이딩)
    -   [다중 상속과 메서드 결정 순서(MRO)](#12-다중-상속과-메서드-결정-순서mro)
    -   [다이아몬드 상속 문제와 `super()`](#13-다이아몬드-상속-문제와-super)
2.  [**코드 구조화: 패키지와 모듈**](#2-코드-구조화-패키지와-모듈)
    -   [패키지란 무엇인가?](#21-패키지란-무엇인가)
    -   [`__init__.py`의 역할](#22-__init__py의-역할)
3.  [**Python의 기본기: 내장 함수와 표준 라이브러리**](#3-python의-기본기-내장-함수와-표준-라이브러리)
    -   [자주 사용하는 내장 함수](#31-자주-사용하는-내장-함수)
    -   ["Batteries Included": 표준 라이브러리](#32-batteries-included-표준-라이브러리)
4.  [**동시성 프로그래밍: `multiprocessing`**](#4-동시성-프로그래밍-multiprocessing)
    -   [프로세스 병렬 처리의 이해](#41-프로세스-병렬-처리의-이해)
    -   [`Process`와 `Pool` 활용법](#42-process와-pool-활용법)
5.  [**마무리 요약**](#5-마무리-요약)

---

## 1. 상속(Inheritance): 코드 재사용의 기술

상속은 기존 클래스(부모 클래스)의 속성과 메서드를 새로운 클래스(자식 클래스)가 물려받아, 코드를 재사용하고 기능을 확장하는 객체 지향의 핵심 원리입니다. 클래스 간의 "IS-A"(~는 ~의 한 종류다) 관계를 표현합니다.

### 1.1. 기본 상속과 메서드 오버라이딩

자식 클래스는 부모의 기능을 그대로 사용할 수 있으며, 필요에 따라 부모의 메서드를 **재정의(Override)**하여 자신만의 동작을 구현할 수 있습니다. 이때 `super()`를 사용하면 부모 클래스의 메서드를 명시적으로 호출할 수 있습니다.

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        print(f"{self.name}이(가) 소리를 냅니다.")

# Animal을 상속받는 Dog 클래스
class Dog(Animal):
    # 메서드 오버라이딩
    def speak(self):
        # super()를 통해 부모의 speak() 메서드를 먼저 호출
        super().speak()
        print("멍멍!")

my_dog = Dog("보리")
my_dog.speak()
# 출력:
# 보리이(가) 소리를 냅니다.
# 멍멍!
```

### 1.2. 다중 상속과 메서드 결정 순서(MRO)

Python은 두 개 이상의 클래스로부터 상속받는 **다중 상속**을 지원합니다. 여러 부모 클래스에 동일한 이름의 메서드가 있을 경우, 어떤 메서드를 먼저 호출할지 결정하는 규칙이 필요한데, 이를 **MRO(Method Resolution Order)**라고 합니다.

MRO는 `클래스.__mro__` 또는 `클래스.mro()`를 통해 확인할 수 있으며, C3 선형화 알고리즘에 따라 일관되고 예측 가능한 순서를 보장합니다.

```python
class Engine:
    def start(self):
        print("엔진 시동")

class Radio:
    def play_music(self):
        print("음악 재생")

# Engine과 Radio를 다중 상속
class Car(Engine, Radio):
    pass

my_car = Car()
my_car.start()       # Engine의 메서드 사용
my_car.play_music()  # Radio의 메서드 사용

# MRO 확인: Car -> Engine -> Radio -> object 순으로 탐색
print(Car.__mro__)
# (<class '__main__.Car'>, <class '__main__.Engine'>, <class '__main__.Radio'>, <class 'object'>)
```

### 1.3. 다이아몬드 상속 문제와 `super()`

**다이아몬드 상속**은 서로 다른 두 자식 클래스가 동일한 조상 클래스를 상속하고, 또 다른 클래스가 이 두 자식 클래스를 다중 상속받는 마름모 형태의 구조입니다. 이때 `super()`는 단순히 부모를 호출하는 것이 아니라, **MRO 순서에 따라 다음 클래스의 메서드를 호출**하여 중복 호출을 방지하고 모든 부모의 메서드가 한 번씩만 호출되도록 보장합니다.

```python
class A:
    def whoami(self):
        print("A.whoami")

class B(A):
    def whoami(self):
        print("B.whoami")
        super().whoami()

class C(A):
    def whoami(self):
        print("C.whoami")
        super().whoami()

class D(B, C):
    def whoami(self):
        print("D.whoami")
        super().whoami()

d = D()
d.whoami()
# MRO: D -> B -> C -> A -> object
# 출력:
# D.whoami
# B.whoami
# C.whoami
# A.whoami
```

---

## 2. 코드 구조화: 패키지와 모듈

### 2.1. 패키지란 무엇인가?

**패키지**는 여러 관련 모듈들을 모아놓은 디렉터리입니다. 이를 통해 코드를 기능별로 구조화하고, 이름 충돌(namespace collision)을 방지하며, 재사용성을 높일 수 있습니다.

```
my_app/
├── main.py
└── my_package/
    ├── __init__.py
    ├── utils.py
    └── models.py
```

### 2.2. `__init__.py`의 역할

이 파일이 있는 디렉터리는 Python 패키지로 인식됩니다. `__init__.py` 파일은 다음과 같은 역할을 합니다.

1.  **패키지 식별**: 가장 기본적인 역할입니다. 파일이 비어 있어도 패키지로 인식됩니다.
2.  **초기화 코드 실행**: 패키지가 임포트될 때 실행되어야 할 코드를 넣을 수 있습니다.
3.  **네임스페이스 관리**: `__all__` 변수를 정의하여 `from package import *` 시 노출할 모듈을 제한하거나, 하위 모듈의 함수/클래스를 패키지의 최상위 레벨로 끌어올릴 수 있습니다.

```python
# my_package/__init__.py

# utils 모듈의 helper 함수를 my_package 레벨로 노출
from .utils import helper

# from my_package import * 시 노출할 이름 정의
__all__ = ['helper', 'models']
```

```python
# main.py

# __init__.py 덕분에 바로 helper 함수를 임포트 가능
from my_package import helper

helper()
```

---

## 3. Python의 기본기: 내장 함수와 표준 라이브러리

### 3.1. 자주 사용하는 내장 함수

`import` 없이 바로 사용할 수 있는 Python의 기본 함수들입니다. 특히 `map`, `filter`, `zip` 등은 함수형 프로그래밍 스타일의 코드 작성에 유용합니다.

-   **`map(function, iterable)`**: `iterable`의 각 요소에 `function`을 적용한 결과를 반환합니다.
-   **`filter(function, iterable)`**: `iterable`의 각 요소 중 `function`의 결과가 `True`인 것만 걸러서 반환합니다.
-   **`zip(*iterables)`**: 여러 `iterable`을 동일한 인덱스끼리 묶어 튜플의 형태로 반환합니다.

```python
# map: 모든 요소를 제곱
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, numbers))  # [1, 4, 9, 16]

# filter: 짝수만 필터링
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4]

# zip: 이름과 역할을 묶기
names = ["Alice", "Bob", "Charlie"]
roles = ["Admin", "User", "Guest"]
users = list(zip(names, roles))  # [('Alice', 'Admin'), ('Bob', 'User'), ('Charlie', 'Guest')]
```

### 3.2. "Batteries Included": 표준 라이브러리

Python은 "건전지 포함(Batteries Included)" 철학에 따라, 추가 설치 없이 바로 사용할 수 있는 방대하고 강력한 **표준 라이브러리**를 제공합니다.

| 모듈 | 주요 용도 |
| :--- | :--- |
| `os` | 운영체제 상호작용 (파일 경로, 디렉터리 등) |
| `sys` | Python 인터프리터 제어 (명령행 인자 등) |
| `datetime` | 날짜 및 시간 처리 |
| `math` | 복잡한 수학 연산 |
| `random` | 난수 생성 및 랜덤 선택 |
| `json` | JSON 데이터 직렬화 및 역직렬화 |
| `collections` | 특수 컨테이너 자료형 (`deque`, `Counter`, `defaultdict`) |
| `re` | 정규 표현식을 이용한 문자열 처리 |

---

## 4. 동시성 프로그래밍: `multiprocessing`

### 4.1. 프로세스 병렬 처리의 이해

`multiprocessing`은 여러 개의 CPU 코어를 동시에 활용하여 작업을 **병렬(parallel)**로 처리하는 라이브러리입니다. 각 프로세스는 독립적인 메모리 공간과 GIL(Global Interpreter Lock)을 가지므로, CPU 집약적인 작업에서 `threading`보다 훨씬 높은 성능을 낼 수 있습니다.

> **💡 왜 `if __name__ == "__main__"`이 필요한가요?**
> `multiprocessing`은 새로운 프로세스를 생성(spawn)할 때 메인 스크립트를 다시 임포트합니다. 만약 `Process` 생성 코드가 이 가드 블록 밖에 있다면, 자식 프로세스가 코드를 임포하면서 또 다른 자식 프로세스를 무한히 생성하는 재귀 폭탄이 발생할 수 있습니다. 따라서 프로세스를 생성하는 코드는 반드시 이 블록 안에 두어 메인 스크립트로 직접 실행될 때만 동작하도록 해야 합니다.

### 4.2. `Process`와 `Pool` 활용법

-   **`Process`**: 단일 작업을 별도의 프로세스에서 실행할 때 사용합니다.
-   **`Pool`**: 여러 개의 데이터를 함수에 매핑하여 병렬 처리할 때 유용합니다. 정해진 개수의 프로세스 풀(pool)을 만들어 작업을 분배합니다.

```python
from multiprocessing import Process, Pool
import time
import os

def worker(name):
    """단일 작업을 수행하는 함수"""
    print(f"[{os.getpid()}] {name} 작업 시작")
    time.sleep(1)
    print(f"[{os.getpid()}] {name} 작업 종료")

def square(x):
    """계산 작업을 수행하는 함수"""
    return x * x

if __name__ == "__main__":
    # 1. Process 사용 예시
    print("--- Process 예시 ---")
    p1 = Process(target=worker, args=("프로세스 1",))
    p2 = Process(target=worker, args=("프로세스 2",))
    p1.start()  # 프로세스 시작
    p2.start()
    p1.join()   # 프로세스가 끝날 때까지 대기
    p2.join()
    print("모든 Process 작업 완료")

    # 2. Pool 사용 예시
    print("\n--- Pool 예시 ---")
    with Pool(processes=4) as pool: # 4개의 프로세스를 가진 풀 생성
        result = pool.map(square, [1, 2, 3, 4, 5, 6, 7, 8])
    print(f"Pool 작업 결과: {result}")
```

---

## 5. 마무리 요약

| 개념 | 핵심 설명 | 주요 키워드 |
| :--- | :--- | :--- |
| **상속** | 부모 클래스의 코드를 재사용하고 확장하는 기술 | `class Child(Parent)`, `super()`, 오버라이딩 |
| **다중 상속** | 여러 부모로부터 상속. MRO 규칙에 따라 메서드 탐색 | `MRO`, `__mro__`, 다이아몬드 문제 |
| **패키지** | 모듈을 디렉터리 단위로 구조화하는 방법 | `__init__.py`, `import`, 네임스페이스 |
| **내장 함수** | `import` 없이 사용하는 Python의 기본 기능 | `map`, `filter`, `zip`, `enumerate` |
| **표준 라이브러리** | "건전지 포함". 설치 없이 사용하는 강력한 모듈 모음 | `os`, `datetime`, `json`, `collections` |
| **멀티프로세싱** | 여러 CPU 코어를 활용한 진정한 병렬 처리 | `Process`, `Pool`, `GIL`, `if __name__ == "__main__"` |

---

[⏮️ 이전 문서](./0512_Python정리.md) | [다음 문서 ⏭️](./0514_Python정리.md)
