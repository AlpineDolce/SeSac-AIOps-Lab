# 🐍 Python 심화: 고차 함수와 람다 (Day 5)

> **이 문서의 목적**: 이 문서는 부트캠프 5일차에 학습한 Python의 강력한 기능인 **고급 함수 기법**들을 깊이 있게 정리한 자료입니다. 가변 인자, 키워드 인자, 그리고 `lambda`와 고차 함수(`map`, `filter`, `sorted`)의 개념과 활용법을 상세한 예제와 함께 다루어, Python을 더욱 Python답게 사용하는 능력을 보여주는 것을 목표로 합니다.

---

## 목차

1.  [**유연한 함수 만들기: 고급 매개변수**](#1-유연한-함수-만들기-고급-매개변수)
    -   [가변 위치 인자: `*args`](#가변-위치-인자-args)
    -   [가변 키워드 인자: `**kwargs`](#가변-키워드-인자-kwargs)
    -   [키워드 전용 인자: 명시적 호출 강제](#키워드-전용-인자-명시적-호출-강제)
    -   [매개변수 정의 순서](#매개변수-정의-순서)
2.  [**람다(Lambda): 이름 없는 한 줄 함수**](#2-람다lambda-이름-없는-한-줄-함수)
    -   [람다란 무엇인가?](#람다란-무엇인가)
    -   [람다의 한계와 사용 시점](#람다의-한계와-사용-시점)
3.  [**고차 함수(Higher-Order Functions): 함수를 다루는 함수**](#3-고차-함수higher-order-functions-함수를-다루는-함수)
    -   [`map()`: 모든 요소에 함수 적용하기](#map-모든-요소에-함수-적용하기)
    -   [`filter()`: 조건에 맞는 요소만 걸러내기](#filter-조건에-맞는-요소만-걸러내기)
    -   [`sorted()` vs `.sort()`: 정렬의 두 가지 방법](#sorted-vs-sort-정렬의-두-가지-방법)
4.  [**실전 예제: `lambda`와 고차 함수 활용**](#4-실전-예제-lambda와-고차-함수-활용)

---

## 1. 유연한 함수 만들기: 고급 매개변수

Python 함수는 정해진 개수의 인자 외에도, 유연하게 인자를 받을 수 있는 고급 매개변수 기법을 제공합니다.

### 가변 위치 인자: `*args`

`*args`는 **개수가 정해지지 않은 위치 인자**들을 **튜플(Tuple)**로 묶어 받습니다. 함수가 몇 개의 인자를 받을지 예측할 수 없을 때 유용합니다.

```python
def sum_all(*numbers):
    """전달된 모든 숫자 인자들의 합계를 반환합니다."""
    print(f"받은 인자 (튜플): {numbers}")
    total = 0
    for num in numbers:
        total += num
    return total

print(sum_all(1, 2, 3))       # 받은 인자 (튜플): (1, 2, 3) -> 결과: 6
print(sum_all(10, 20, 30, 40)) # 받은 인자 (튜플): (10, 20, 30, 40) -> 결과: 100
```

### 가변 키워드 인자: `**kwargs`

`**kwargs`는 **개수가 정해지지 않은 키워드 인자**들을 **딕셔너리(Dictionary)**로 묶어 받습니다. 함수의 옵션이나 추가 정보를 유연하게 전달받을 때 사용됩니다.

```python
def display_profile(**user_info):
    """전달된 사용자 정보를 출력합니다."""
    print(f"받은 정보 (딕셔너리): {user_info}")
    for key, value in user_info.items():
        print(f"- {key}: {value}")

display_profile(name="Alice", age=30, city="Seoul")
# 받은 정보 (딕셔너리): {'name': 'Alice', 'age': 30, 'city': 'Seoul'}
# - name: Alice
# - age: 30
# - city: Seoul
```

### 키워드 전용 인자: 명시적 호출 강제

`*` 뒤에 오는 매개변수는 **반드시 키워드를 사용하여 인자를 전달**하도록 강제할 수 있습니다. 이는 함수의 가독성을 높이고, 인자의 의미를 명확하게 만들어 실수를 방지합니다.

```python
# is_admin과 is_active는 반드시 키워드로 전달해야 함
def create_user(username, *, is_admin=False, is_active=True):
    print(f"User: {username}, Admin: {is_admin}, Active: {is_active}")

# 올바른 호출
create_user("bob", is_admin=True)

# 잘못된 호출 (TypeError 발생)
# create_user("bob", True)
```

### 매개변수 정의 순서

함수 정의 시 매개변수는 정해진 순서를 따라야 합니다.

> **순서**: `위치 인자` → `기본값 인자` → `*args` → `키워드 전용 인자` → `**kwargs`

```python
def comprehensive_function(pos1, default_val="default", *args, kw_only, **kwargs):
    print(f"pos1: {pos1}")
    print(f"default_val: {default_val}")
    print(f"args: {args}")
    print(f"kw_only: {kw_only}")
    print(f"kwargs: {kwargs}")

comprehensive_function(1, "val", 2, 3, kw_only="must", ext1="a", ext2="b")
```

---

## 2. 람다(Lambda): 이름 없는 한 줄 함수

### 람다란 무엇인가?

람다는 **이름이 없는(anonymous) 간단한 한 줄짜리 함수**를 정의할 때 사용하는 키워드입니다. `def`를 사용하기에는 너무 간단한 기능을 임시로 만들 때 유용하며, 주로 다른 함수의 인자로 전달될 때 그 진가를 발휘합니다.

-   **문법**: `lambda 인자: 표현식`
-   **특징**:
    -   `return` 키워드 없이 표현식의 결과가 자동으로 반환됩니다.
    -   복잡한 로직(여러 줄, 반복문, 조건문 등)은 작성할 수 없습니다.

```python
# 두 수를 더하는 람다 함수
add = lambda x, y: x + y
result = add(5, 3) # result는 8
```

### 람다의 한계와 사용 시점

람다는 매우 편리하지만, 모든 상황에 적합한 것은 아닙니다.

-   **한계**:
    -   한 줄의 표현식만 허용됩니다.
    -   복잡한 로직이나 여러 줄의 코드가 필요하면 가독성이 급격히 떨어집니다.
-   **최적의 사용 시점**:
    -   `map()`, `filter()`, `sorted()`와 같은 **고차 함수의 인자**로 전달될 때.
    -   한 번만 사용되고 버려질 간단한 콜백(callback) 함수가 필요할 때.

> **결론**: 로직이 복잡하다면 주저하지 말고 `def`를 사용하여 명시적인 함수를 만드는 것이 좋습니다.

---

## 3. 고차 함수(Higher-Order Functions): 함수를 다루는 함수

고차 함수는 **다른 함수를 인자로 받거나, 함수를 결과로 반환**하는 함수를 말합니다. Python의 `map`, `filter`, `sorted`가 대표적인 예입니다.

### `map()`: 모든 요소에 함수 적용하기

`map(function, iterable)`은 `iterable`의 각 요소에 `function`을 순서대로 적용하여, 그 결과들을 모은 **map 객체(iterator)**를 반환합니다.

```python
numbers = [1, 2, 3, 4]

# 각 숫자를 문자열로 변환
str_numbers = list(map(str, numbers))
print(str_numbers) # ['1', '2', '3', '4']

# 각 숫자를 제곱 (람다 활용)
squared_numbers = list(map(lambda x: x**2, numbers))
print(squared_numbers) # [1, 4, 9, 16]
```

### `filter()`: 조건에 맞는 요소만 걸러내기

`filter(function, iterable)`은 `iterable`의 각 요소에 `function`을 적용하여, 결과가 `True`인 요소들만 모아 **filter 객체(iterator)**를 반환합니다.

```python
numbers = [1, 2, 3, 4, 5, 6]

# 짝수만 필터링
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers) # [2, 4, 6]

# 3보다 큰 수만 필터링
greater_than_3 = list(filter(lambda x: x > 3, numbers))
print(greater_than_3) # [4, 5, 6]
```

> **💡 Iterator란?**
> `map`과 `filter`는 결과를 리스트가 아닌 **이터레이터(iterator)**로 반환합니다. 이터레이터는 값이 필요한 시점에 하나씩 계산하여 반환하는 **메모리 효율적인 객체**입니다. 모든 결과를 보려면 `list()`로 형 변환해야 합니다.

### `sorted()` vs `.sort()`: 정렬의 두 가지 방법

| 구분 | `sorted(iterable, ...)` | `list.sort(...)` |
| :--- | :--- | :--- |
| **동작 방식** | **새로운 정렬된 리스트**를 생성하여 **반환** | **원본 리스트**를 직접 수정 (in-place) |
| **원본 변경** | **아니요** (원본은 그대로 유지됨) | **예** (원본이 변경됨) |
| **반환 값** | 정렬된 **새 리스트** | **`None`** |
| **사용 대상** | **모든 반복 가능한 객체** (리스트, 튜플, 문자열 등) | **리스트**에만 사용 가능 |

```python
my_list = [3, 1, 4, 2]

# sorted() 사용
new_sorted_list = sorted(my_list)
print(f"sorted() 후 원본: {my_list}")       # [3, 1, 4, 2] (변경 없음)
print(f"sorted() 결과: {new_sorted_list}") # [1, 2, 3, 4]

# .sort() 사용
my_list.sort()
print(f".sort() 후 원본: {my_list}") # [1, 2, 3, 4] (변경됨)
```

#### `key`를 이용한 복잡한 정렬

`sorted()`와 `.sort()` 모두 `key` 매개변수를 받아 정렬 기준을 지정할 수 있습니다. `lambda`와 함께 사용하면 매우 강력합니다.

```python
students = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 88}
]

# 점수(score)를 기준으로 오름차순 정렬
sorted_by_score = sorted(students, key=lambda s: s["score"])
print(sorted_by_score)

# 이름을 기준으로 내림차순 정렬
sorted_by_name_desc = sorted(students, key=lambda s: s["name"], reverse=True)
print(sorted_by_name_desc)
```

---

## 4. 실전 예제: `lambda`와 고차 함수 활용

```python
words = ["assembly", "java", "rain", "notebook", "north", 
         "south", "hospital", "programming", "house", "hour"]

# 1. 글자 수가 6글자 이상인 단어만 필터링
long_words = list(filter(lambda w: len(w) >= 6, words))
print(f"6글자 이상 단어: {long_words}")

# 2. 모든 단어를 대문자로 변환하고, 길이를 함께 튜플로 묶기
word_info = list(map(lambda w: (w.upper(), len(w)), words))
print(f"단어 정보: {word_info}")

# 3. 단어들을 길이순으로 오름차순 정렬
sorted_by_length = sorted(words, key=len)
print(f"길이순 정렬: {sorted_by_length}")

# 4. 'o'가 포함된 단어만 필터링한 후, 알파벳 역순으로 정렬
o_words_sorted_desc = sorted(
    filter(lambda w: 'o' in w, words), 
    reverse=True
)
print(f"'o' 포함 단어 역순 정렬: {o_words_sorted_desc}")
```

---

[⏮️ 이전 문서](./0428_Python정리.md) | [다음 문서 ⏭️](./0430_Python정리.md)
