# 🐍 Python 심화: OOP, 함수형 프로그래밍, 예외 처리 (Day 12)

> **이 문서의 목적**: 이 문서는 Python의 **고급 객체 지향 프로그래밍(OOP)**, **함수형 프로그래밍** 및 **예외 처리 기법**을 깊이 있게 정리한 자료입니다. `@classmethod`, `@staticmethod`, `@property` 등의 특별한 메서드의 차이점을 명확히 이해하고, 싱글톤, 데코레이터, 클로저와 같은 디자인 패턴  및 프로그래밍 기법을 실제 프로젝트에 적용하는 방법을 설명합니다. 또한, 예외 처리 기법을 통해 견고하고 안정적인 코드를 작성하는 법을 학습하는 것을 목표로 합니다.

---

## 목차

1.  [**파이썬의 특별한 메서드: 언제 무엇을 써야 할까?**](#1-파이썬의-특별한-메서드-언제-무엇을-써야-할까)
    -   [클래스 메서드 (`@classmethod`)](#클래스-메서드-classmethod)
    -   [스태틱 메서드 (`@staticmethod`)](#스태틱-메서드-staticmethod)
    -   [프로퍼티 (`@property`)](#프로퍼티-property)
    -   [한눈에 보는 메서드 비교](#한눈에-보는-메서드-비교)
2.  [**핵심 디자인 패턴과 프로그래밍 기법**](#2-핵심-디자인-패턴과-프로그래밍-기법)
    -   [싱글톤 패턴 (Singleton Pattern)](#싱글톤-패턴-singleton-pattern)
    -   [데코레이터 (Decorator)](#데코레이터-decorator)
    -   [클로저 (Closure)](#클로저-closure)
3.  [**견고한 코드를 위한 예외 처리 (Exception Handling)**](#3-견고한-코드를-위한-예외-처리-exception-handling)
    -   [`try...except`: 오류를 제어하는 기본](#tryexcept-오류를-제어하는-기본)
    -   [`else`와 `finally`: 보너스 제어 구문](#else와-finally-보너스-제어-구문)
4.  [**실전 프로젝트: 고급 기능을 활용한 회원 관리 시스템**](#4-실전-프로젝트-고급-기능을-활용한-회원-관리-시스템)
    -   [프로젝트 구조와 설계](#프로젝트-구조와-설계)
    -   [컴포넌트별 핵심 코드 분석](#컴포넌트별-핵심-코드-분석)
    -   [학습 내용의 적용](#학습-내용의-적용)

---

## 1. 파이썬의 특별한 메서드: 언제 무엇을 써야 할까?

### 클래스 메서드 (`@classmethod`)

**인스턴스가 아닌 클래스 자체에 바인딩되는 메서드**입니다. 첫 번째 인자로 클래스 객체(`cls`)를 받아 클래스 속성을 조작하거나, 특정 조건에 맞는 인스턴스를 생성하는 **팩토리 메서드**로 주로 활용됩니다.

```python
class User:
    user_count = 0

    def __init__(self, name):
        self.name = name
        User.user_count += 1

    @classmethod
    def get_user_count(cls):
        # cls는 User 클래스를 가리킴
        return f"총 사용자 수: {cls.user_count}명"

    @classmethod
    def from_csv(cls, csv_string):
        # "이름,나이" 형태의 CSV 문자열에서 User 객체를 생성하는 팩토리 메서드
        name, _ = csv_string.split(',')
        return cls(name)

# 인스턴스 없이 클래스에서 직접 호출
print(User.get_user_count())

user = User.from_csv("홍길동,30")
print(f"생성된 사용자: {user.name}")
```

### 스태틱 메서드 (`@staticmethod`)

**클래스나 인스턴스 상태와 완전히 독립적인 메서드**입니다. 논리적으로는 클래스에 속하지만, `self`나 `cls`를 인자로 받지 않아 내용적으로는 일반 함수와 같습니다. **유틸리티 함수**를 클래스 네임스페이스 안에 묶어두고 싶을 때 사용합니다.

```python
class StringUtils:
    @staticmethod
    def is_palindrome(s):
        # 회문(palindrome)인지 확인하는 유틸리티
        return s == s[::-1]

    @staticmethod
    def count_char(s, char):
        return s.count(char)

# 클래스 이름을 통해 직접 호출
print(StringUtils.is_palindrome("level"))  # True
print(StringUtils.count_char("hello world", "l")) # 3
```

### 프로퍼티 (`@property`)

**메서드를 속성(attribute)처럼 보이게 만드는 기능**입니다. 속성 값을 가져오거나(`getter`), 설정하거나(`setter`), 삭제할 때(`deleter`) 추가적인 로직을 실행할 수 있어 **캡슐화**와 **데이터 유효성 검사**에 매우 유용합니다.

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius  # 내부적으로 사용할 비공개 속성

    @property
    def radius(self):
        """반지름 값을 반환 (getter)"""
        return self._radius

    @radius.setter
    def radius(self, value):
        """반지름 값을 설정하며 유효성 검사 (setter)"""
        if value <= 0:
            raise ValueError("반지름은 0보다 커야 합니다.")
        self._radius = value

    @property
    def area(self):
        """계산된 속성 (다른 속성을 기반으로 동적으로 값 계산)"""
        return 3.14 * self._radius ** 2

# 사용 예시
c = Circle(10)
print(f"반지름: {c.radius}")  # 메서드지만 속성처럼 호출
print(f"넓이: {c.area}")

c.radius = 12  # setter가 호출되어 유효성 검사 후 값 변경
print(f"변경된 넓이: {c.area}")
```

### 한눈에 보는 메서드 비교

| 구분 | 첫 번째 인자 | 상태 접근 | 주 용도 |
| :--- | :--- | :--- | :--- |
| **인스턴스 메서드** | `self` (인스턴스) | 인스턴스 상태 | 객체의 동작 정의 |
| **클래스 메서드** | `cls` (클래스) | 클래스 상태 | 팩토리 메서드, 클래스 수준 조작 |
| **스태틱 메서드** | 없음 | 상태 접근 불가 | 유틸리티, 도우미 함수 |
| **프로퍼티** | `self` (인스턴스) | 인스턴스 상태 | 속성 접근 제어, 캡슐화 |

---

## 2. 핵심 디자인 패턴과 프로그래밍 기법

### 싱글톤 패턴 (Singleton Pattern)

**클래스의 인스턴스가 오직 하나만 생성되도록 보장**하는 디자인 패턴입니다. 시스템 전역에서 공유해야 하는 설정 객체, 데이터베이스 연결 풀 등에 사용됩니다.

```python
class Settings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            print("Settings 객체를 생성했습니다.")
        return cls._instance

    def __init__(self):
        # __init__은 매번 호출될 수 있으므로, 초기화는 한 번만 수행되도록 방어
        if not hasattr(self, 'initialized'):
            self.theme = "Light"
            self.version = "1.0"
            self.initialized = True

s1 = Settings()
s2 = Settings()

print(f"s1과 s2는 같은 객체인가? {s1 is s2}") # True
s1.theme = "Dark"
print(f"s2의 테마: {s2.theme}") # Dark
```

### 데코레이터 (Decorator)

**기존 함수의 코드를 수정하지 않고, 추가 기능을 덧붙이는** 강력한 도구입니다. 함수의 실행 전후에 로깅, 인증, 캐싱, 시간 측정 등의 공통 관심사를 처리하는 데 널리 사용됩니다.

```python
import time
import functools

def log_execution_time(func):
    @functools.wraps(func)  # 원본 함수의 메타정보(이름, docstring 등) 보존
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"'{func.__name__}' 실행 시간: {end - start:.4f}초")
        return result
    return wrapper

@log_execution_time
def fetch_large_data():
    """시간이 오래 걸리는 데이터 처리 작업을 시뮬레이션합니다."""
    time.sleep(1.5)
    return "데이터 로딩 완료"

fetch_large_data()
# 출력: 'fetch_large_data' 실행 시간: 1.5023초
```

### 클로저 (Closure)

**자신이 정의된 시점의 환경(스코프)을 기억하는 함수**입니다. 외부 함수의 실행이 끝나도, 내부 함수는 외부 함수의 변수(자유 변수)에 접근하여 상태를 유지하고 조작할 수 있습니다.

```python
def power_factory(exponent):
    # exponent는 자유 변수
    def power(base):
        # 내부 함수 power는 exponent를 기억
        return base ** exponent
    return power

# 2의 거듭제곱을 계산하는 함수 생성
square = power_factory(2)
# 3의 거듭제곱을 계산하는 함수 생성
cube = power_factory(3)

print(f"5의 제곱: {square(5)}")  # 25
print(f"5의 세제곱: {cube(5)}")  # 125
```

---

## 3. 견고한 코드를 위한 예외 처리 (Exception Handling)

프로그램 실행 중 발생할 수 있는 오류(예외)에 대비하여 코드를 안정적으로 만드는 것은 매우 중요합니다. Python은 `try...except` 구문을 통해 강력한 예외 처리 메커니즘을 제공합니다.

### `try...except`: 오류를 제어하는 기본

-   **`try`**: 예외가 발생할 가능성이 있는 코드 블록을 감쌉니다.
-   **`except [오류 종류]`**: `try` 블록에서 특정 오류가 발생했을 때 실행할 코드를 정의합니다.

```python
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("[에러] 0으로 나눌 수 없습니다.")
        return None
    return result

divide(10, 0)
```

### `else`와 `finally`: 보너스 제어 구문

-   **`else`**: `try` 블록에서 **예외가 발생하지 않았을 때만** 실행됩니다.
-   **`finally`**: 예외 발생 여부와 **상관없이 항상** 실행됩니다. 주로 파일 닫기, 데이터베이스 연결 해제 등 리소스 정리 작업에 사용됩니다.

```python
try:
    file = open('data.txt', 'r')
except FileNotFoundError:
    print("파일이 존재하지 않습니다.")
else:
    # 예외가 발생하지 않았을 때만 실행
    print("파일을 성공적으로 열었습니다.")
    file.close()
finally:
    # 성공하든 실패하든 항상 실행
    print("파일 처리 시도를 종료합니다.")
```
---

## 4. 실전 프로젝트: 고급 기능을 활용한 회원 관리 시스템

### 프로젝트 구조와 설계

```python
project/
│
├── main.py
├── Membership.py
├── MembershipManager.py
├── NoticeBoard.py
├── NoticeBoardManager.py
├── SingletonMeta.py
├── members.pkl
└── posts.pkl
```
---

### 컴포넌트별 핵심 코드 분석

`Membership.py` - `사용자 정보 저장`
```python
import re

class Membership:
    def __init__(self, member_id, username, password, name, phone, email):
        self._member_id = member_id
        self._username = username
        self._password = password
        self._name = name
        self._phone = phone
        self._email = email

    # 캡슐화된 속성 접근을 위한 property
    @property
    def member_id(self):
        return self._member_id

    @property
    def username(self):
        return self._username

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not value:
            raise ValueError("이름은 비워둘 수 없습니다.")
        self._name = value

    @property
    def phone(self):
        return self._phone

    @phone.setter
    def phone(self, value):
        if not value.isdigit():
            raise ValueError("전화번호는 숫자만 가능합니다.")
        self._phone = value

    @property
    def email(self):
        return self._email

    @email.setter
    def email(self, value):
        if not self.is_valid_email(value):
            raise ValueError("유효하지 않은 이메일 형식입니다.")
        self._email = value

    def check_password(self, input_password):
        return self._password == input_password

    def show_info(self):
        return (f"[회원번호: {self._member_id}]\n"
                f"이름: {self._name}\n"
                f"아이디: {self._username}\n"
                f"전화번호: {self._phone}\n"
                f"이메일: {self._email}")

    @staticmethod
    def is_valid_email(email):
        return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            member_id=data["member_id"],
            username=data["username"],
            password=data["password"],
            name=data["name"],
            phone=data["phone"],
            email=data["email"]
        )

    def to_dict(self):
        return {
            "member_id": self._member_id,
            "username": self._username,
            "password": self._password,
            "name": self._name,
            "phone": self._phone,
            "email": self._email
        }
```

#### 회원 관리 `Membership`

- 목적 : 사용자 정보 저장
- 필드 :
    + 회원 번호     : `member_id`   (회원 번호, 시스템에서 자동 부여)
    + 회원 아이디   : `username`    (아이디)
    + 패스워드      : `password`
    + 이름          : `name`
    + 전화번호      : `phone`
    + 이메일        : `email`
- 메서드(예시) :
    + `check_password(input_pw)` -> 비밀번호 검증
- 기능 적용 :

| 기능                 | 설명                                |
| ------------------ | --------------------------------- |
| `property`         | 이름, 전화번호, 이메일에 대해 캡슐화 및 유효성 검사 제공 |
| `staticmethod`     | `is_valid_email()` 로 이메일 유효성 체크   |
| `classmethod`      | `from_dict()`로 저장된 데이터를 객체로 복원    |
| `to_dict()`        | pickle 저장을 위한 dict 변환 지원          |
| `check_password()` | 비밀번호 확인                           |
    
---
`NoticeBoard.py` - `게시글 한건의 정보 저장`
```python
from datetime import datetime
from abc import ABC, abstractmethod


class PostBase(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def update(self, title, content):
        pass

    @abstractmethod
    def delete(self):
        pass


class NoticeBoard(PostBase):
    def __init__(self, post_id, member_id, title, content, created_at=None, views=0):
        self._post_id = post_id
        self._member_id = member_id
        self._title = title
        self._content = content
        self._created_at = created_at if created_at else datetime.now()
        self._views = views

    # Property encapsulation
    @property
    def post_id(self):
        return self._post_id

    @property
    def member_id(self):
        return self._member_id

    @property
    def title(self):
        return self._title

    @title.setter
    def title(self, value):
        self._title = value

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    @property
    def created_at(self):
        return self._created_at

    @property
    def views(self):
        return self._views

    # 클로저를 이용한 조회수 증가 함수
    def view_counter(self):
        def increment():
            self._views += 1
            return self._views
        return increment

    # 게시글 읽기 (조회수 증가 포함)
    def read(self):
        increment_views = self.view_counter()
        increment_views()
        return f"제목: {self._title}\n내용: {self._content}\n조회수: {self._views}"

    # 게시글 수정
    def update(self, title, content):
        self._title = title
        self._content = content

    # 게시글 삭제 (실제로는 호출자 측에서 리스트에서 삭제)
    def delete(self):
        print(f"[삭제 완료] 게시글 ID: {self._post_id}")

    def summary(self):
        return f"[{self._post_id}] {self._title} by {self._member_id} | 조회수: {self._views}"

    # 직렬화 지원
    def to_dict(self):
        return {
            "post_id": self._post_id,
            "member_id": self._member_id,
            "title": self._title,
            "content": self._content,
            "created_at": self._created_at.isoformat(),
            "views": self._views
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            post_id=data["post_id"],
            member_id=data["member_id"],
            title=data["title"],
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            views=data["views"]
        )


# 테스트 예시
if __name__ == "__main__":
    post = NoticeBoard("P001", 1, "공지사항", "환영합니다!")
    print(post.read())
    print(post.read())  # 조회수 2로 증가
    post.update("수정된 제목", "새로운 내용")
    print(post.read())
```
#### 게시판 `NoticeBoard`

- 목적 : 게시글 한 건의 정보 저장
- 필드 :
    + 글 번호       : `post_id`     (글 번호, 시스템에서 자동 부여)
    + 작성자 번호   : `member_id`   (작성자 번호)
    + 제목          : `title`
    + 내용          : `content`
    + 작성일        : `created_at`  (작성일, datatime)
    + 조회수        : `views`
- 메서드(예시) :
    + `read()`      -> 조회수 증가 후 내용 출력
    + `update()`    -> 글 수정(작성자 + 비밀번호 일치 시)
    + `delete()`    -> 삭제 허용 조건 판단
- 기능 적용 :

| 기능                    | 설명                    |
| --------------------- | --------------------- |
| `PostBase`            | 추상 클래스, 게시판 규칙 정의     |
| `@property`           | 캡슐화 적용                |
| `view_counter()`      | 클로저 예제 (조회수 증가 함수 반환) |
| `to_dict/from_dict()` | 직렬화/역직렬화 대응           |
| `summary()`           | 목록 요약용 문자열 제공         |
| `read()`              | 자동 조회수 증가 포함          |


---
`MembershipManager.py` - `회원가입 수정, 탈퇴, 조회(내 정보)`
```python
import pickle
import os
from Membership import Membership
from SingletonMeta import SingletonMeta


# 데코레이터: 로그인 확인
def login_required(func):
    def wrapper(self, *args, **kwargs):
        if not self.current_user:
            print("로그인이 필요합니다.")
            return
        return func(self, *args, **kwargs)
    return wrapper


class MembershipManager(metaclass=SingletonMeta):
    DATA_FILE = "members.pkl"

    def __init__(self):
        self._membersList = []
        self._next_member_id = 1
        self.current_user = None
        self.load_members()

    def sign_up(self):
        print("\n[회원가입]")
        username = input("아이디: ")
        if self.find_by_username(username):
            print("이미 존재하는 아이디입니다.")
            return

        password = input("비밀번호: ")
        name = input("이름: ")
        phone = input("전화번호: ")
        email = input("이메일: ")

        try:
            new_member = Membership(
                member_id=self._next_member_id,
                username=username,
                password=password,
                name=name,
                phone=phone,
                email=email
            )
            self._membersList.append(new_member)
            self._next_member_id += 1
            print(f"{name}님, 회원가입이 완료되었습니다.")
            self.save_members()
        except ValueError as e:
            print(f"오류: {e}")

    def login(self):
        print("\n[로그인]")
        username = input("아이디: ")
        password = input("비밀번호: ")

        member = self.find_by_username(username)
        if member and member.check_password(password):
            self.current_user = member
            print(f"{member.name}님, 로그인 되었습니다.")
        else:
            print("아이디 또는 비밀번호가 잘못되었습니다.")

    @login_required
    def get_my_info(self):
        print("\n[내 정보 조회]")
        print(self.current_user.show_info())

    @login_required
    def update_member(self):
        print("\n[회원 정보 수정]")
        pw = input("비밀번호 확인: ")
        if not self.current_user.check_password(pw):
            print("비밀번호가 틀렸습니다.")
            return

        new_phone = input("새 전화번호: ")
        new_email = input("새 이메일: ")

        try:
            self.current_user.phone = new_phone
            self.current_user.email = new_email
            self.save_members()
            print("회원 정보가 수정되었습니다.")
        except ValueError as e:
            print(f"오류: {e}")

    @login_required
    def delete_member(self):
        print("\n[회원 탈퇴]")
        pw = input("비밀번호 확인: ")
        if not self.current_user.check_password(pw):
            print("비밀번호가 틀렸습니다.")
            return

        self._membersList = [m for m in self._membersList if m != self.current_user]
        print("회원 탈퇴가 완료되었습니다.")
        self.current_user = None
        self.save_members()

    def find_by_username(self, username):
        return next((m for m in self._membersList if m.username == username), None)

    def save_members(self):
        with open(self.DATA_FILE, "wb") as f:
            pickle.dump([m.to_dict() for m in self._membersList], f)

    def load_members(self):
        try:
            with open(self.DATA_FILE, "rb") as f:
                data_list = pickle.load(f)
                self._membersList = [Membership.from_dict(d) for d in data_list]
        except FileNotFoundError:
            # 파일이 없을 경우, 오류를 출력하고 빈 리스트로 시작
            print(f"데이터 파일({self.DATA_FILE})이 없어 새로 시작합니다.")
        except Exception as e:
            # 그 외 다른 예외 (파일 손상 등) 처리
            print(f"데이터 로딩 중 오류 발생: {e}")
```



#### 회원가입 수정, 탈퇴, 조회(내 정보) `MembershipManager`

- 필드 : `members` (회원 리스트, `list` 또는 `dict`)
- 기능 메서드 :
    + `sign_up()`           -> 회원가입
    + `login()`             -> 로그인
    + `get_my_info()`       -> 내 정보 조회
    + `update_member()`     -> 회원 정보 수정
    + `delele_member()`     -> 회원 탈퇴

- 기능 적용 :

| 기능                             | 설명                              |
| ------------------------------ | ------------------------------- |
| `SingletonMeta`                | 인스턴스 단일화 (모든 앱에서 하나의 관리자 사용)    |
| `@login_required`              | 로그인 확인 데코레이터로 중복 제거             |
| `save_members`, `load_members` | `pickle`을 통해 파일 저장/복원 지원        |
| `lambda + next()`              | 빠른 유저 검색에 `find_by_username` 사용 |
| `current_user`                 | 현재 로그인 사용자 상태 저장                |
| `try/except`                   | setter 유효성 검사를 통한 안정성 확보        |

---
`NoticeBoardManager.py` - `게시글 작성`
```python
import pickle
import os
from NoticeBoard import NoticeBoard
from SingletonMeta import SingletonMeta
from MembershipManager import MembershipManager


# 데코레이터: 로그인 필요
def login_required(func):
    def wrapper(self, *args, **kwargs):
        if not self.current_user:
            print("로그인이 필요합니다.")
            return
        return func(self, *args, **kwargs)
    return wrapper


class NoticeBoardManager(metaclass=SingletonMeta):
    DATA_FILE = "posts.pkl"

    def __init__(self):
        self._posts = []
        self._next_post_id = 1
        self.current_user = None  # 로그인 사용자 연동
        self.load_posts()

    def set_current_user(self, user):
        self.current_user = user

    @login_required
    def write_post(self):
        print("\n[게시글 작성]")
        title = input("제목: ")
        content = input("내용: ")

        post = NoticeBoard(
            post_id=self._next_post_id,
            member_id=self.current_user.member_id,
            title=title,
            content=content
        )
        self._posts.append(post)
        self._next_post_id += 1
        self.save_posts()
        print("게시글이 등록되었습니다.")

    def list_posts(self):
        print("\n[게시글 목록]")
        if not self._posts:
            print("게시글이 없습니다.")
            return
        for post in sorted(self._posts, key=lambda p: p.created_at, reverse=True):
            print(post.summary())

    def find_post_by_id(self, post_id):
        return next((p for p in self._posts if p.post_id == post_id), None)

    def read_post(self):
        print("\n[게시글 조회]")
        pid = int(input("게시글 번호 입력: "))
        post = self.find_post_by_id(pid)
        if post:
            print(post.read())
        else:
            print("게시글을 찾을 수 없습니다.")

    @login_required
    def update_post(self):
        print("\n[게시글 수정]")
        pid = int(input("수정할 게시글 번호: "))
        post = self.find_post_by_id(pid)
        if not post:
            print("해당 게시글이 없습니다.")
            return
        if post.member_id != self.current_user.member_id:
            print("본인이 작성한 글만 수정할 수 있습니다.")
            return

        pw = input("비밀번호 확인: ")
        if not self.current_user.check_password(pw):
            print("비밀번호가 틀렸습니다.")
            return

        new_title = input("새 제목: ")
        new_content = input("새 내용: ")
        post.update(new_title, new_content)
        self.save_posts()
        print("게시글이 수정되었습니다.")

    @login_required
    def delete_post(self):
        print("\n[게시글 삭제]")
        pid = int(input("삭제할 게시글 번호: "))
        post = self.find_post_by_id(pid)
        if not post:
            print("해당 게시글이 없습니다.")
            return
        if post.member_id != self.current_user.member_id:
            print("본인이 작성한 글만 삭제할 수 있습니다.")
            return

        pw = input("비밀번호 확인: ")
        if not self.current_user.check_password(pw):
            print("비밀번호가 틀렸습니다.")
            return

        self._posts = [p for p in self._posts if p.post_id != pid]
        self.save_posts()
        print("게시글이 삭제되었습니다.")

    def save_posts(self):
        with open(self.DATA_FILE, "wb") as f:
            pickle.dump([p.to_dict() for p in self._posts], f)

    def load_posts(self):
        if os.path.exists(self.DATA_FILE):
            with open(self.DATA_FILE, "rb") as f:
                data = pickle.load(f)
                self._posts = [NoticeBoard.from_dict(d) for d in data]
            if self._posts:
                self._next_post_id = max(p.post_id for p in self._posts) + 1
```
#### 게시글 작성 `NoticeBoardManager`

- 필드: posts (게시글 리스트)
- 기능 메서드:
    + `write_post(member_id, title, content)` → 글 작성
    + `read_post(post_id)` → 글 읽기 (조회수 증가 포함)
    + `update_post(post_id, member_id, password)` → 글 수정 (검증 포함)
    + `delete_post(post_id, member_id, password)` → 글 삭제 (검증 포함)
    + `list_posts()` → 모든 글 요약 출력
  
- 회원 번호, 제목, 내용, 작성일(Date), 조회수 0
  + 읽어보기
  + 수정 (작성자만); 회원 번호랑 패스워드 입력 시 가능
  + 삭제 (작성자만); 회원 번호랑 패스워드 입력 시 가능
  
- 주요 기능 :

| 기능              | 설명                   |
| --------------- | -------------------- |
| `write_post()`  | 로그인한 사용자가 글 작성       |
| `read_post()`   | 글 번호로 조회, 조회수 자동 증가  |
| `update_post()` | 본인 + 비밀번호 일치 시 수정 허용 |
| `delete_post()` | 본인 + 비밀번호 일치 시 삭제 허용 |
| `save/load`     | 게시글 자동 저장 및 복구       |
| `list_posts()`  | 최신순 정렬(lambda 활용)    |

---
`SingletonMeta.py` - `전체 실행 흐름`
```python
class SingletonMeta(type):
    _instances = {}  # 클래스별로 생성된 인스턴스를 저장할 딕셔너리

    def __call__(cls, *args, **kwargs):
        # cls: 현재 클래스를 의미 (예: MembershipManager)

        if cls not in cls._instances:
            # 인스턴스가 아직 없으면 생성
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        # 이미 생성된 인스턴스가 있다면 그것을 반환
        return cls._instances[cls]
```
- **클래스의 인스턴스가 오직 하나만 생성**되도록 보장하며, 여러 곳에서 동일한 인스턴스를 공유할 수 있도록 유지
---
`main.py` - `전체 실행 흐름`
```python
from MembershipManager import MembershipManager
from NoticeBoardManager import NoticeBoardManager

def main():
    member_manager = MembershipManager()
    board_manager = NoticeBoardManager()

    while True:
        print("\n====== 회원 & 게시판 시스템 ======")
        print("1. 회원가입")
        print("2. 로그인")
        print("3. 내 정보 조회")
        print("4. 회원 정보 수정")
        print("5. 회원 탈퇴")
        print("6. 게시글 작성")
        print("7. 게시글 목록 보기")
        print("8. 게시글 조회")
        print("9. 게시글 수정")
        print("10. 게시글 삭제")
        print("0. 종료")

        choice = input("선택> ")

        if choice == '1':
            member_manager.sign_up()
        elif choice == '2':
            member_manager.login()
            board_manager.set_current_user(member_manager.current_user)  # 게시판에 로그인 상태 반영
        elif choice == '3':
            member_manager.get_my_info()
        elif choice == '4':
            member_manager.update_member()
        elif choice == '5':
            member_manager.delete_member()
        elif choice == '6':
            board_manager.write_post()
        elif choice == '7':
            board_manager.list_posts()
        elif choice == '8':
            board_manager.read_post()
        elif choice == '9':
            board_manager.update_post()
        elif choice == '10':
            board_manager.delete_post()
        elif choice == '0':
            print("시스템을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    main()
```


### 학습 내용의 적용

| 적용된 개념 | 클래스 및 메서드 | 구현 내용 및 목적 |
| :--- | :--- | :--- |
| **예외 처리** | `load_members`, `sign_up` | `FileNotFoundError`를 처리하여 프로그램의 안정성을 높이고, `ValueError`를 잡아 사용자 입력 오류에 대응합니다. |
| **싱글톤 패턴** | `MembershipManager`, `NoticeBoardManager` | 시스템 전체에서 회원과 게시판 관리자가 단 하나의 인스턴스만 갖도록 하여 데이터 일관성을 유지합니다. |
| **데코레이터** | `@login_required` | 정보 조회, 수정, 탈퇴 등 로그인이 필요한 기능에 적용하여 인증 로직의 중복을 제거하고 가독성을 높입니다. |
| **프로퍼티** | `Membership.email`, `Membership.phone` | `setter`를 통해 이메일 형식이나 전화번호 숫자 여부 등 데이터 유효성을 검사하여 잘못된 값이 입력되는 것을 방지합니다. |
| **클래스 메서드** | `Membership.from_dict` | 파일에서 읽어온 딕셔너리 데이터를 `Membership` 객체로 변환하는 팩토리 역할을 수행하여 객체 생성 로직을 캡슐화합니다. |
| **스태틱 메서드** | `Membership.is_valid_email` | 이메일 유효성 검사처럼 인스턴스 상태와 무관한 순수 함수를 클래스 네임스페이스 내에 유틸리티로 제공합니다. |
| **클로저** | `NoticeBoard.view_counter` | 게시글을 읽을 때마다 조회수가 안전하게 1씩 증가하는 상태를 갖는 함수를 생성하여 `read()` 메서드에 적용합니다. |

---

[⏮️ 이전 문서](./0510_Linux정리.md) | [다음 문서 ⏭️](./0513_Python정리.md)