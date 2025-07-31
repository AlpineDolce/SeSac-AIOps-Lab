# 🐍 Python 심화: 반복문과 함수 (Day 4)

> **이 문서의 목적**: 이 문서는 부트캠프 4일차에 학습한 **중첩 반복문(Nested Loops)**과 프로그래밍의 핵심 구성 요소인 **함수(Function)**에 대해 깊이 있게 정리한 자료입니다. 복잡한 패턴을 구현하는 방법과 코드를 구조화하고 재사용하는 기술을 상세한 예제와 함께 다루어, 논리적 사고와 문제 해결 능력을 보여주는 것을 목표로 합니다.

---

## 목차

1.  [**중첩 반복문 (Nested Loops)**](#1-중첩-반복문-nested-loops)
    -   [개념 및 동작 원리](#개념-및-동작-원리)
    -   [활용 사례: 2차원 데이터 처리 및 패턴 출력](#활용-사례-2차원-데이터-처리-및-패턴-출력)
2.  [**함수 (Functions): 코드의 재사용과 구조화**](#2-함수-functions-코드의-재사용과-구조화)
    -   [함수란 무엇인가?](#함수란-무엇인가)
    -   [함수 정의와 호출](#함수-정의와-호출)
    -   [매개변수(Parameters)와 인자(Arguments)](#매개변수parameters와-인자arguments)
    -   [값의 전달 방식: Call by Value vs Call by Object Reference](#값의-전달-방식-call-by-value-vs-call-by-object-reference)
    -   [반환 값 (Return Values)](#반환-값-return-values)
    -   [매개변수 기본값 (Default Parameter Values)](#매개변수-기본값-default-parameter-values)
3.  [**실전 예제: 주급 계산기 모듈화**](#3-실전-예제-주급-계산기-모듈화)

---

## 1. 중첩 반복문 (Nested Loops)

### 개념 및 동작 원리

**중첩 반복문**은 하나의 `for`문 또는 `while`문 내부에 또 다른 반복문이 포함된 구조를 말합니다. 바깥쪽 반복문이 한 번 실행될 때마다, 안쪽 반복문은 처음부터 끝까지 전체가 실행됩니다.

-   **실행 횟수**: 바깥쪽 루프가 M번, 안쪽 루프가 N번 반복한다면, 내부 코드는 총 **M × N** 번 실행됩니다.
-   **주의점**: 중첩이 깊어질수록 실행 시간은 기하급수적으로 증가하므로(시간 복잡도 증가), 불필요한 중첩은 피하고 알고리즘을 최적화하는 것이 중요합니다. 일반적으로 2중, 최대 3중 `for`문까지만 사용하는 것이 권장됩니다.

```python
# 동작 원리 예시
for i in range(2):  # 바깥쪽 루프 (2번 실행)
    print(f"--- 바깥쪽 루프 {i+1}번째 시작 ---")
    for j in range(3):  # 안쪽 루프 (3번 실행)
        print(f"  안쪽 루프: i={i}, j={j}")
    print(f"--- 바깥쪽 루프 {i+1}번째 종료 ---\n")

# --- 바깥쪽 루프 1번째 시작 ---
#   안쪽 루프: i=0, j=0
#   안쪽 루프: i=0, j=1
#   안쪽 루프: i=0, j=2
# --- 바깥쪽 루프 1번째 종료 ---

# --- 바깥쪽 루프 2번째 시작 ---
#   안쪽 루프: i=1, j=0
#   안쪽 루프: i=1, j=1
#   안쪽 루프: i=1, j=2
# --- 바깥쪽 루프 2번째 종료 ---
```

### 활용 사례: 2차원 데이터 처리 및 패턴 출력

#### 2차원 리스트(행렬) 순회

중첩 반복문은 행과 열로 구성된 2차원 데이터를 처리하는 데 매우 효과적입니다.

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# 행(row)을 기준으로 순회
for row in matrix:
    # 각 행(row) 내부의 열(col)을 순회
    for col_element in row:
        print(col_element, end=' ')
    print() # 한 행의 출력이 끝나면 줄바꿈
```

#### 패턴 출력: 다이아몬드 별 찍기

논리적 사고를 기르는 데 도움이 되는 고전적인 예제입니다.

```python
n = 5 # 다이아몬드의 절반 높이

# 위쪽 삼각형
for i in range(n):
    # 공백 출력: n-1, n-2, ..., 0 순으로 감소
    spaces = " " * (n - 1 - i)
    # 별 출력: 1, 3, 5, ... 순으로 증가
    stars = "*" * (2 * i + 1)
    print(spaces + stars)

# 아래쪽 역삼각형 (중앙선을 제외하고 n-1번 반복)
for i in range(n - 2, -1, -1): # n-2부터 0까지 감소
    spaces = " " * (n - 1 - i)
    stars = "*" * (2 * i + 1)
    print(spaces + stars)

#     *        # i=0, 공백4 + 별1
#    ***       # i=1, 공백3 + 별3
#   *****      # i=2, 공백2 + 별5
#  *******     # i=3, 공백1 + 별7
# *********    # i=4, 공백0 + 별9  ← 중앙 줄
#  *******     # i=3, 공백1 + 별7
#   *****      # i=2, 공백2 + 별5
#    ***       # i=1, 공백3 + 별3
#     *        # i=0, 공백4 + 별1
```

---

## 2. 함수 (Functions): 코드의 재사용과 구조화

### 함수란 무엇인가?

**함수**는 특정 작업을 수행하기 위해 설계된, **재사용 가능한 코드 블록**입니다. 복잡한 문제를 여러 개의 작은 기능 단위로 나누어 해결(모듈화)함으로써 코드의 가독성과 유지보수성을 획기적으로 향상시킵니다.

-   **장점**:
    -   **재사용성**: 반복되는 코드를 한 번만 작성하고 필요할 때마다 호출하여 사용합니다.
    -   **모듈화**: 기능별로 코드를 분리하여 프로그램의 구조를 명확하게 만듭니다.
    -   **가독성**: 함수 이름만으로도 코드의 역할을 짐작할 수 있어 이해하기 쉽습니다.
    -   **유지보수**: 특정 기능에 문제가 생기면 해당 함수만 수정하면 되므로 관리가 용이합니다.

### 함수 정의와 호출

-   **정의(Definition)**: `def` 키워드를 사용하여 함수를 만듭니다.
-   **호출(Call)**: 함수 이름을 사용하여 해당 코드 블록을 실행합니다.

```python
# 함수 정의
def greet():
    print("=" * 20)
    print("Hello, Function!")
    print("=" * 20)

# 함수 호출
greet()
```

### 매개변수(Parameters)와 인자(Arguments)

-   **매개변수(Parameter)**: 함수를 **정의**할 때, 함수 내부에서 사용될 변수. 함수의 입력값을 받는 "자리표시자"입니다.
-   **인자(Argument)**: 함수를 **호출**할 때, 매개변수에 실제로 전달되는 값.

```python
# 'name'은 매개변수
def greet_by_name(name):
    print(f"Hello, {name}!")

# "Alice"와 "Bob"은 인자
greet_by_name("Alice")
greet_by_name("Bob")
```

### 값의 전달 방식: Call by Value vs Call by Object Reference

Python은 "Call by Object Reference" 방식을 사용합니다. 이는 전달되는 인자의 타입에 따라 동작이 다르게 보일 수 있습니다.

-   **Immutable 객체 전달 (숫자, 문자열, 튜플 등)**:
    값이 복사되어 전달되는 것처럼 동작합니다. 함수 내에서 매개변수를 수정해도 원본 변수에는 영향을 주지 않습니다. (결과적으로 **Call by Value**와 유사)

    ```python
    def change_value(x):
        x = 100 # 함수 내에서 x는 새로운 객체를 가리킴
        print(f"함수 안: {x}") # 100

    a = 10
    change_value(a)
    print(f"함수 밖: {a}") # 10 (원본 a는 변하지 않음)
    ```

-   **Mutable 객체 전달 (리스트, 딕셔너리 등)**:
    객체의 **참조(메모리 주소)**가 전달됩니다. 함수 내에서 해당 객체의 내용을 변경하면 원본 객체도 함께 변경됩니다. (결과적으로 **Call by Reference**와 유사)

    ```python
    def add_element(my_list):
        my_list.append(4)
        print(f"함수 안: {my_list}") # [1, 2, 3, 4]

    original_list = [1, 2, 3]
    add_element(original_list)
    print(f"함수 밖: {original_list}") # [1, 2, 3, 4] (원본 리스트가 변경됨)
    ```

### 반환 값 (Return Values)

`return` 키워드는 함수의 실행 결과를 함수를 호출한 곳으로 돌려줍니다.

-   `return`이 없는 함수는 기본적으로 `None`을 반환합니다.
-   여러 값을 반환하고 싶을 때는 쉼표로 구분하며, 이 경우 값들은 **튜플(Tuple)**로 묶여 반환됩니다.

```python
def calculate_circle(radius):
    area = 3.14 * radius**2
    circumference = 2 * 3.14 * radius
    return area, circumference # (area, circumference) 튜플로 반환

# 언패킹을 통해 여러 변수에 나누어 받을 수 있음
a, c = calculate_circle(5)
print(f"넓이: {a}, 둘레: {c}")
```

### 매개변수 기본값 (Default Parameter Values)

매개변수에 기본값을 설정해두면, 함수 호출 시 해당 인자를 생략할 수 있습니다.

-   **규칙**: 기본값이 있는 매개변수는 반드시 기본값이 없는 매개변수 **뒤에** 위치해야 합니다.

```python
# 'role' 매개변수에 기본값 "User" 설정
def register_user(username, role="User"):
    print(f"사용자: {username}, 역할: {role}")

register_user("Alice") # role 인자 생략 -> 기본값 "User" 사용
register_user("Bob", "Admin") # role 인자 전달 -> "Admin" 사용
```

> **💡 함수 오버로딩(Overloading)과의 관계**
> Python은 C++이나 Java와 달리 함수 오버로딩(동일한 이름의 함수를 매개변수의 개수나 타입에 따라 다르게 정의하는 것)을 공식적으로 지원하지 않습니다. 하지만 매개변수 기본값을 활용하면 이와 유사한 효과를 낼 수 있습니다.

---

## 3. 실전 예제: 주급 계산기 모듈화

기능별로 코드를 함수로 분리하여 주급 계산 프로그램을 더 구조적으로 만들어 보겠습니다.

```python
# 전역 변수로 직원 목록 관리
worker_list = [
    {"name": "홍길동", "work_time": 40, "per_pay": 15000, "pay": 0},
    {"name": "임꺽정", "work_time": 30, "per_pay": 20000, "pay": 0}
]

# 기능 1: 새로운 직원 정보 입력받아 추가하는 함수
def add_worker():
    """사용자로부터 이름, 근무시간, 시급을 입력받아 worker_list에 추가합니다."""
    worker = {}
    worker["name"] = input("이름: ")
    worker["work_time"] = int(input("근무 시간: "))
    worker["per_pay"] = int(input("시간당 급여: "))
    worker["pay"] = 0
    worker_list.append(worker)
    print(f"{worker['name']}님의 정보가 추가되었습니다.")

# 기능 2: 모든 직원의 주급을 계산하는 함수
def calculate_all_pays():
    """worker_list를 순회하며 각 직원의 주급을 계산하여 업데이트합니다."""
    for worker in worker_list:
        worker['pay'] = worker['work_time'] * worker['per_pay']
    print("모든 직원의 주급 계산이 완료되었습니다.")

# 기능 3: 모든 직원 정보를 출력하는 함수
def print_all_workers():
    """worker_list에 있는 모든 직원의 정보를 출력합니다."""
    print("\n--- 전체 직원 정보 ---")
    for w in worker_list:
        print(f"이름: {w['name']}, 근무시간: {w['work_time']}, 시급: {w['per_pay']}, 주급: {w['pay']}")
    print("-" * 25)

# 메인 로직을 담당하는 함수
def main():
    """프로그램의 메인 메뉴를 보여주고 사용자 입력에 따라 각 기능을 호출합니다."""
    while True:
        print("\n[ 주급 계산 프로그램 ]")
        print("1. 직원 추가")
        print("2. 전체 주급 계산")
        print("3. 전체 정보 출력")
        print("0. 종료")
        
        choice = input("선택: ")
        
        if choice == "1":
            add_worker()
        elif choice == "2":
            calculate_all_pays()
        elif choice == "3":
            print_all_workers()
        elif choice == "0":
            print("프로그램을 종료합니다.")
            break # 무한 루프 종료
        else:
            print("잘못된 입력입니다. 다시 선택해주세요.")

# 프로그램 시작
main()
```

---

[⏮️ 이전 문서](./0425_Python정리.md) | [다음 문서 ⏭️](./0429-Python정리.md)
