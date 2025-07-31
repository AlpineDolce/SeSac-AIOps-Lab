# 🐍 Python 객체 지향 프로그래밍(OOP) 심화 (Day 9)

> **이 문서의 목적**: 이 문서는 부트캠프 9일차에 학습한 **객체 지향 프로그래밍(OOP)의 4대 핵심 원칙**(추상화, 캡슐화, 상속, 다형성)을 깊이 있게 정리한 자료입니다. 각 원칙의 개념을 명확히 이해하고, Python 코드로 어떻게 구현되는지 실전 예제와 함께 다루어 OOP 설계 능력을 보여주는 것을 목표로 합니다.

---

## 목차

1.  [**객체 지향 프로그래밍(OOP)의 4대 원칙**](#1-객체-지향-프로그래밍oop의-4대-원칙)
    -   [추상화 (Abstraction)](#추상화-abstraction)
    -   [캡슐화 (Encapsulation)](#캡슐화-encapsulation)
    -   [상속 (Inheritance)](#상속-inheritance)
    -   [다형성 (Polymorphism)](#다형성-polymorphism)
2.  [**실전 프로젝트: 객체 지향 숫자 야구 게임**](#2-실전-프로젝트-객체-지향-숫자-야구-게임)
    -   [프로젝트 구조 설계: 역할 분리](#프로젝트-구조-설계-역할-분리)
    -   [클래스별 책임과 구현](#클래스별-책임과-구현)
    -   [전체 코드 및 실행 흐름](#전체-코드-및-실행-흐름)

---

## 1. 객체 지향 프로그래밍(OOP)의 4대 원칙

객체 지향 프로그래밍은 단순히 클래스와 객체를 사용하는 것을 넘어, 네 가지 핵심 원칙을 통해 코드의 유연성, 재사용성, 유지보수성을 극대화합니다.

### 추상화 (Abstraction)

-   **개념**: 복잡한 내부 구현은 숨기고, 사용자가 알아야 할 **핵심적인 인터페이스(메서드)만 노출**하는 것을 의미합니다. 사용자는 내부가 어떻게 동작하는지 몰라도, 제공된 기능을 쉽게 사용할 수 있습니다.
-   **비유**: 우리가 자동차를 운전할 때, 엔진의 복잡한 원리를 몰라도 핸들, 페달, 기어라는 단순한 인터페이스만으로 조작할 수 있는 것과 같습니다.

```python
class Car:
    def __init__(self, model):
        self._model = model
        self._is_engine_on = False

    def start_engine(self):
        """엔진 시동을 거는 기능 (내부 로직은 숨겨져 있음)"""
        if not self._is_engine_on:
            print(f"{self._model}의 엔진 시동을 겁니다.")
            # ... 내부적으로는 연료 분사, 점화 플러그 작동 등 복잡한 과정이 일어남 ...
            self._is_engine_on = True
        else:
            print("엔진이 이미 켜져 있습니다.")

# 사용자는 start_engine() 메서드만 호출하면 됨
my_car = Car("Tesla Model 3")
my_car.start_engine()
```

### 캡슐화 (Encapsulation)

-   **개념**: 데이터(속성)와 해당 데이터를 처리하는 메서드를 하나의 **객체로 묶고, 데이터에 대한 직접적인 접근을 제한**하는 원칙입니다. 이를 통해 데이터의 무결성을 보장하고, 의도치 않은 수정을 방지합니다.
-   **Python에서의 구현**:
    -   Python은 `public`, `private` 같은 명시적인 접근 제어자가 없습니다.
    -   대신, 변수나 메서드 이름 앞에 **언더스코어(`_` 또는 `__`)**를 붙여 접근 수준을 암시하는 **관례**를 사용합니다.
        -   `_variable`: **Protected**. "외부에서 직접 접근하지 마세요"라는 약한 권고.
        -   `__variable`: **Private**. Python이 내부적으로 이름을 변경(`_ClassName__variable`)하여 외부에서의 직접 접근을 어렵게 만듭니다. (Name Mangling)

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance # Private 속성으로 잔액 은닉

    def deposit(self, amount):
        """입금 기능. 양수만 입금 가능하도록 제어."""
        if amount > 0:
            self.__balance += amount
            print(f"{amount}원이 입금되었습니다.")
        else:
            print("입금액은 0보다 커야 합니다.")

    def get_balance(self):
        """잔액 조회 기능 (읽기 전용 접근)."""
        return self.__balance

# 사용자는 제공된 메서드를 통해서만 잔액에 접근 가능
acc = BankAccount("Alice")
acc.deposit(1000)
# acc.__balance = -5000 # 직접 수정 시도 -> AttributeError 발생
print(f"현재 잔액: {acc.get_balance()}") # 현재 잔액: 1000
```

### 상속 (Inheritance)

-   **개념**: 기존 클래스(**부모/슈퍼 클래스**)의 속성과 메서드를 새로운 클래스(**자식/서브 클래스**)가 물려받는 기능입니다. 코드의 **재사용성**을 높이고, 클래스 간의 계층 구조를 형성하여 코드를 체계적으로 관리할 수 있게 합니다.
-   **메서드 오버라이딩 (Method Overriding)**: 자식 클래스에서 부모 클래스로부터 물려받은 메서드를 자신에게 맞게 **재정의**하는 것입니다.

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("자식 클래스에서 이 메서드를 반드시 재정의해야 합니다.")

class Dog(Animal): # Animal 클래스를 상속
    def speak(self): # 부모의 speak 메서드를 오버라이딩
        return "멍멍!"

class Cat(Animal): # Animal 클래스를 상속
    def speak(self): # 부모의 speak 메서드를 오버라이딩
        return "야옹"

my_dog = Dog("해피")
print(f"{my_dog.name}가 말하길: {my_dog.speak()}") # 해피가 말하길: 멍멍!
```

### 다형성 (Polymorphism)

-   **개념**: "여러 형태를 가질 수 있는 능력"이라는 뜻으로, **동일한 형태의 코드(예: 같은 메서드 호출)가 객체의 타입에 따라 다른 동작을 하는 것**을 의미합니다.
-   **장점**: 코드를 더 유연하고 확장 가능하게 만듭니다. 새로운 클래스가 추가되어도 기존 코드를 수정할 필요가 없습니다.

```python
# 위에서 정의한 Dog와 Cat 클래스 사용
animals = [Dog("해피"), Cat("나비"), Dog("초코")]

# animal 변수는 루프마다 Dog 객체였다가 Cat 객체가 됨
# 동일한 animal.speak() 호출이 각 객체의 타입에 맞게 다른 결과를 출력
for animal in animals:
    print(f"{animal.name}가 말하길: {animal.speak()}")

# 해피가 말하길: 멍멍!
# 나비가 말하길: 야옹
# 초코가 말하길: 멍멍!
```

---

## 2. 실전 프로젝트: 객체 지향 숫자 야구 게임

### 프로젝트 구조 설계: 역할 분리

숫자 야구 게임을 객체 지향적으로 설계하기 위해, 각 기능의 **책임(Role)**에 따라 클래스를 분리합니다.

| 클래스명 | 책임 (역할) | 주요 속성 및 메서드 |
| :--- | :--- | :--- |
| `BaseballGame` | **한 판의 게임**에 대한 데이터와 로직을 관리 | `computer_nums`, `user_guesses`, `judge()` |
| `GameManager` | **전체 게임의 흐름**과 여러 판의 기록을 관리 | `game_history`, `start_new_game()`, `show_stats()` |

### 클래스별 책임과 구현

#### `baseball_game.py` (게임 한 판을 담당하는 모듈)

```python
import random

class BaseballGame:
    """숫자 야구 게임 한 판의 데이터와 로직을 관리하는 클래스."""
    def __init__(self, digits=3):
        self.digits = digits
        self.computer_nums = self._generate_numbers()
        self.user_guesses = []
        self.is_finished = False

    def _generate_numbers(self):
        """중복 없는 무작위 숫자를 생성합니다."""
        return random.sample(range(1, 10), self.digits)

    def guess(self, user_input):
        """사용자의 추측을 받아 결과를 판정하고 기록합니다."""
        if self.is_finished:
            return "이미 종료된 게임입니다."

        # 입력 유효성 검사 (3자리 숫자만 허용)
        if len(user_input) != self.digits or not user_input.isdigit():
            return f"잘못된 입력입니다. {self.digits}자리 숫자를 입력해주세요."

        user_nums = [int(d) for d in user_input]
        
        strikes, balls = 0, 0
        for i, num in enumerate(user_nums):
            if num == self.computer_nums[i]:
                strikes += 1
            elif num in self.computer_nums:
                balls += 1
        
        self.user_guesses.append((user_nums, strikes, balls))

        if strikes == self.digits:
            self.is_finished = True
        
        return strikes, balls
```

#### `game_manager.py` (전체 게임 흐름을 담당하는 모듈)

```python
from baseball_game import BaseballGame

class GameManager:
    """전체 숫자 야구 게임의 흐름과 기록을 관리하는 클래스."""
    def __init__(self):
        self.game_history = []

    def start_new_game(self):
        """새로운 게임을 시작하고 플레이 루프를 실행합니다."""
        game = BaseballGame()
        self.game_history.append(game)
        attempts = 0
        print("\n--- 새로운 숫자 야구 게임을 시작합니다! ---")

        while not game.is_finished and attempts < 9:
            attempts += 1
            user_input = input(f"[{attempts}번째 시도] 세 자리 숫자를 입력하세요: ")
            
            result = game.guess(user_input)
            if isinstance(result, tuple):
                strikes, balls = result
                print(f"결과: {strikes} 스트라이크, {balls} 볼")
            else:
                print(result)  # 잘못된 입력에 대한 에러 메시지

        if game.is_finished:
            print(f"축하합니다! {attempts}번 만에 정답을 맞혔습니다!")
        else:
            print(f"아쉽네요. 정답은 {game.computer_nums}였습니다.")

    def show_stats(self):
        """전체 게임 기록을 보여주는 메소드"""
        if not self.game_history:
            print("게임 기록이 없습니다.")
            return
        
        print("\n--- 게임 기록 ---")
        for i, game in enumerate(self.game_history, 1):
            print(f"게임 {i}:")
            for guess, strikes, balls in game.user_guesses:
                print(f"  추측: {guess}, 스트라이크: {strikes}, 볼: {balls}")
            if game.is_finished:
                print("  게임 종료: 맞혔습니다!")
            else:
                print("  게임 종료: 기회 종료")
            print("-" * 20)

    def main_menu(self):
        """메인 메뉴에서 게임 선택 및 기록 보기"""
        while True:
            choice = input("\n1. 새 게임 시작\n2. 기록 보기\n0. 종료\n> ")
            if choice == '1':
                self.start_new_game()
            elif choice == '2':
                self.show_stats()
            elif choice == '0':
                print("게임을 종료합니다. 감사합니다!")
                break
            else:
                print("잘못된 선택입니다. 다시 입력해주세요.")
```

#### `main.py` (게임 실행 모듈)

```python
from game_manager import GameManager

def main():
    manager = GameManager()
    manager.main_menu()

if __name__ == "__main__":
    main()
```

### 전체 코드 및 실행 흐름

1.  사용자가 `game_manager.py`를 실행합니다.
2.  `GameManager` 인스턴스가 생성되고, `main_menu()`가 호출됩니다.
3.  사용자가 "새 게임 시작"을 선택하면, `start_new_game()` 메서드가 호출됩니다.
4.  `start_new_game()`은 `BaseballGame`의 새로운 인스턴스를 생성합니다. 이 인스턴스는 한 판의 게임을 책임집니다.
5.  게임 루프가 돌면서 사용자의 입력을 받고, `game.guess()`를 호출하여 결과를 판정합니다.
6.  게임이 끝나면, 해당 `BaseballGame` 인스턴스는 모든 기록(정답, 사용자 추측 등)을 가진 채로 `GameManager`의 `game_history` 리스트에 저장됩니다.

> **설계의 장점**:
> - **단일 책임 원칙(SRP)**: `BaseballGame`은 게임 한 판의 규칙에만, `GameManager`는 전체 어플리케이션의 흐름에만 집중하여 각 클래스의 역할이 명확합니다.
> - **재사용성**: `BaseballGame` 클래스는 다른 종류의 UI(웹, GUI 등)에서도 재사용할 수 있습니다.
> - **테스트 용이성**: 각 클래스를 독립적으로 테스트하기 용이합니다.

---

[⏮️ 이전 문서](./0507_Python정리.md) | [다음 문서 ⏭️](./0509_Python정리.md)
