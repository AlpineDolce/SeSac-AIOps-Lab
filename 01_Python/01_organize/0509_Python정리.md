# 🐍 Python 심화: 파일 I/O와 객체 직렬화 (Day 10)

> **이 문서의 목적**: 이 문서는 부트캠프 10일차에 학습한 **파일 입출력(I/O)**의 기본 원리와 Python의 강력한 **객체 직렬화(Serialization)** 라이브러리인 `pickle`의 사용법을 깊이 있게 정리한 자료입니다. 프로그램의 데이터를 영구적으로 저장하고 복원하는 방법을 이해하고, 이를 객체 지향 설계에 통합하여 완성도 높은 애플리케이션을 만드는 능력을 보여주는 것을 목표로 합니다.

---

## 목차

1.  [**파일 입출력(File I/O)의 기초**](#1-파일-입출력file-io의-기초)
    -   [파일 처리의 핵심: `with open()`](#파일-처리의-핵심-with-open)
    -   [파일 열기 모드(Mode)](#파일-열기-모드mode)
    -   [텍스트 파일 읽고 쓰기](#텍스트-파일-읽고-쓰기)
2.  [**객체 직렬화(Serialization)와 `pickle`**](#2-객체-직렬화serialization와-pickle)
    -   [직렬화란 무엇인가?](#직렬화란-무엇인가)
    -   [`pickle` 모듈: Python 객체를 그대로 저장하기](#pickle-모듈-python-객체를-그대로-저장하기)
    -   [직렬화(`dump`)와 역직렬화(`load`)](#직렬화dump와-역직렬화load)
3.  [**실전 프로젝트: `pickle`을 이용한 성적 관리 시스템**](#3-실전-프로젝트-pickle을-이용한-성적-관리-시스템)
    -   [프로젝트 구조 설계](#프로젝트-구조-설계)
    -   [클래스별 책임과 구현](#클래스별-책임과-구현)
    -   [데이터 영속성 확보](#데이터-영속성-확보)

---

## 1. 파일 입출력(File I/O)의 기초

파일 입출력은 프로그램이 메모리상에서 다루는 데이터를 파일 시스템에 영구적으로 저장하거나, 파일로부터 데이터를 읽어오는 모든 과정을 의미합니다.

### 파일 처리의 핵심: `with open()`

Python에서는 `with open(...) as ...:` 구문을 사용하여 파일을 처리하는 것이 표준적인 방법입니다.

-   **장점**:
    -   **자동 리소스 관리**: `with` 블록이 끝나면 파일이 **자동으로 닫힙니다 (`.close()`)**.
    -   **예외 안전성**: 파일 처리 중 오류가 발생해도 파일이 항상 안전하게 닫히는 것을 보장합니다.
    -   **가독성**: 코드 블록이 명확하여 파일 작업의 범위를 쉽게 파악할 수 있습니다.

```python
# "example.txt" 파일을 쓰기 모드('w')로 열기
# 한글 처리를 위해 encoding='utf-8' 지정
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("안녕하세요, 파일 쓰기 예제입니다.")
# with 블록이 끝나면 f는 자동으로 닫힘
```

### 파일 열기 모드(Mode)

`open()` 함수의 두 번째 인자로 전달되는 모드는 파일에 대한 작업 유형을 결정합니다.

| 모드 | 설명 | 파일이 없을 때 | 파일이 있을 때 |
| :--: | :--- | :--- | :--- |
| `'r'` | **읽기 (Read)** | `FileNotFoundError` 발생 | 내용을 읽음 |
| `'w'` | **쓰기 (Write)** | 새로 생성 | **기존 내용을 모두 삭제**하고 새로 씀 |
| `'a'` | **추가 (Append)** | 새로 생성 | 기존 내용 **끝에** 이어서 씀 |
| `'x'` | **배타적 생성** | 새로 생성 | `FileExistsError` 발생 |
| `'+'` | **읽기/쓰기** | 모드에 따라 다름 | 읽고 쓸 수 있음 (예: `r+`, `w+`) |
| `'t'` | **텍스트 모드** | (기본값) | 문자열로 데이터를 다룸 |
| `'b'` | **바이너리 모드** | | 바이트(bytes) 단위로 데이터를 다룸 |

> **💡 Tip**: `encoding='utf-8'`은 텍스트 모드에서 한글이나 특수문자가 깨지는 것을 방지하기 위한 필수 옵션입니다.

### 텍스트 파일 읽고 쓰기

-   **쓰기**:
    -   `f.write(string)`: 단일 문자열을 씁니다. (줄바꿈 문자 `\n` 없음)
    -   `f.writelines(list_of_strings)`: 문자열 리스트를 씁니다. (마찬가지로 `\n` 자동 추가 안 됨)

-   **읽기**:
    -   `f.read()`: 파일 전체 내용을 하나의 문자열로 읽습니다.
    -   `f.readline()`: 파일에서 한 줄만 읽습니다.
    -   `f.readlines()`: 파일의 모든 줄을 리스트 형태로 읽습니다.
    -   `for line in f:`: 파일을 한 줄씩 순회하는 가장 효율적이고 Pythonic한 방법입니다.

```python
# 파일의 모든 줄을 읽어 대문자로 출력하기
with open("example.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip().upper()) # .strip()으로 양 끝의 공백 및 줄바꿈 문자 제거
```

---

## 2. 객체 직렬화(Serialization)와 `pickle`

### 직렬화란 무엇인가?

**직렬화(Serialization)**는 프로그램 메모리에서 사용되는 **객체(리스트, 딕셔너리, 클래스 인스턴스 등)를 파일에 저장하거나 네트워크로 전송할 수 있는 형태(주로 바이트 스트림)로 변환**하는 과정을 말합니다. 반대로 파일이나 바이트 스트림으로부터 원래의 객체 구조를 복원하는 것을 **역직렬화(Deserialization)**라고 합니다.

### `pickle` 모듈: Python 객체를 그대로 저장하기

`pickle`은 Python의 내장 모듈로, 거의 모든 Python 객체를 있는 그대로의 모습으로 직렬화하고 복원할 수 있게 해줍니다.

-   **장점**:
    -   사용법이 매우 간단합니다.
    -   리스트, 딕셔너리는 물론, 사용자가 직접 정의한 클래스의 인스턴스까지 대부분의 Python 객체를 처리할 수 있습니다.
-   **단점**:
    -   **Python 전용 포맷**: 다른 프로그래밍 언어와는 호환되지 않습니다.
    -   **보안 취약점**: 신뢰할 수 없는 출처의 `pickle` 파일을 역직렬화하면 악의적인 코드가 실행될 수 있습니다. **절대로 신뢰할 수 없는 데이터를 `pickle.load()` 하지 마세요.**

### 직렬화(`dump`)와 역직렬화(`load`)

-   **`pickle.dump(obj, file)`**: 객체(`obj`)를 파일 객체(`file`)에 직렬화하여 저장합니다. 파일은 반드시 **바이너리 쓰기 모드(`'wb'`)**로 열어야 합니다.
-   **`pickle.load(file)`**: 파일 객체(`file`)로부터 데이터를 읽어 원래의 Python 객체로 역직렬화합니다. 파일은 반드시 **바이너리 읽기 모드(`'rb'`)**로 열어야 합니다.

```python
import pickle

# 직렬화할 데이터 (복잡한 구조의 리스트)
data_to_save = [
    {'name': 'Alice', 'scores': [90, 85, 95]},
    {'name': 'Bob', 'scores': [88, 92, 80]}
]

# 1. 직렬화 (객체 -> 파일)
with open("students.pkl", "wb") as f:
    pickle.dump(data_to_save, f)
    print("데이터가 students.pkl 파일에 저장되었습니다.")

# 2. 역직렬화 (파일 -> 객체)
with open("students.pkl", "rb") as f:
    loaded_data = pickle.load(f)
    print("파일로부터 데이터를 복원했습니다.")

print(loaded_data)
# 출력: [{'name': 'Alice', 'scores': [90, 85, 95]}, {'name': 'Bob', 'scores': [88, 92, 80]}]
```

---

## 3. 실전 프로젝트: `pickle`을 이용한 성적 관리 시스템

### 프로젝트 구조 설계

객체 지향 원칙에 따라 데이터와 관리 로직을 분리합니다.

| 모듈 | 클래스 | 책임 (역할) |
| :--- | :--- | :--- |
| `score_data.py` | `ScoreData` | 한 학생의 성적 정보(이름, 점수, 총점, 평균, 등급)를 **캡슐화**하는 데이터 클래스. |
| `score_manager.py` | `ScoreManager` | 여러 `ScoreData` 객체들을 리스트로 관리하며, 추가/검색/수정/삭제/정렬 및 **파일 저장/불러오기** 등 전체 시스템 로직을 담당. |

### 클래스별 책임과 구현

#### `score_data.py`

```python
class ScoreData:
    """한 학생의 성적 정보를 표현하는 데이터 클래스."""
    def __init__(self, name, kor, eng, mat):
        self.name = name
        self.kor = kor
        self.eng = eng
        self.mat = mat
        self.process()

    def process(self):
        """총점, 평균, 등급을 계산합니다."""
        self.total = self.kor + self.eng + self.mat
        self.avg = self.total / 3
        if self.avg >= 90: self.grade = "수"
        elif self.avg >= 80: self.grade = "우"
        elif self.avg >= 70: self.grade = "미"
        elif self.avg >= 60: self.grade = "양"
        else: self.grade = "가"

    def display(self):
        """성적 정보를 포맷에 맞게 출력합니다."""
        print(f"{self.name}\t{self.kor}\t{self.eng}\t{self.mat}\t{self.total}\t{self.avg:.2f}\t{self.grade}")
```

#### `score_manager.py`

```python
import pickle
from score_data import ScoreData

class ScoreManager:
    """성적 데이터를 관리하고 파일 I/O를 처리하는 클래스."""
    def __init__(self, filename="scores.pkl"):
        self.filename = filename
        self.score_list = []
        self.load_scores()  # 프로그램 시작 시 자동으로 데이터 불러오기

    def save_scores(self):
        """현재 성적 리스트를 pickle 파일에 저장합니다."""
        with open(self.filename, "wb") as f:
            pickle.dump(self.score_list, f)
        print("데이터가 저장되었습니다.")

    def load_scores(self):
        """pickle 파일에서 성적 리스트를 불러옵니다."""
        try:
            with open(self.filename, "rb") as f:
                self.score_list = pickle.load(f)
            print("데이터를 불러왔습니다.")
        except FileNotFoundError:
            print("저장된 데이터 파일이 없습니다. 새로운 목록으로 시작합니다.")

    def append_score(self):
        """새로운 성적을 추가합니다."""
        name = input("학생 이름을 입력하세요: ")
        kor = int(input("국어 점수를 입력하세요: "))
        eng = int(input("영어 점수를 입력하세요: "))
        mat = int(input("수학 점수를 입력하세요: "))
        
        score = ScoreData(name, kor, eng, mat)
        self.score_list.append(score)
        print(f"{name}의 성적이 추가되었습니다.")

    def search_score(self):
        """학생 이름으로 성적을 검색합니다."""
        name = input("성적을 검색할 학생의 이름을 입력하세요: ")
        found = False
        for score in self.score_list:
            if score.name == name:
                score.display()
                found = True
                break
        if not found:
            print(f"{name}의 성적을 찾을 수 없습니다.")

    def modify_score(self):
        """학생 성적을 수정합니다."""
        name = input("성적을 수정할 학생의 이름을 입력하세요: ")
        found = False
        for score in self.score_list:
            if score.name == name:
                kor = int(input("수정할 국어 점수를 입력하세요: "))
                eng = int(input("수정할 영어 점수를 입력하세요: "))
                mat = int(input("수정할 수학 점수를 입력하세요: "))
                score.kor = kor
                score.eng = eng
                score.mat = mat
                score.process()  # 수정 후 총점, 평균, 등급 갱신
                print(f"{name}의 성적이 수정되었습니다.")
                found = True
                break
        if not found:
            print(f"{name}의 성적을 찾을 수 없습니다.")

    def delete_score(self):
        """학생 성적을 삭제합니다."""
        name = input("삭제할 학생의 이름을 입력하세요: ")
        found = False
        for score in self.score_list:
            if score.name == name:
                self.score_list.remove(score)
                print(f"{name}의 성적이 삭제되었습니다.")
                found = True
                break
        if not found:
            print(f"{name}의 성적을 찾을 수 없습니다.")

    def sort_scores(self):
        """성적을 총점 기준으로 내림차순 정렬합니다."""
        self.score_list.sort(key=lambda score: score.total, reverse=True)
        print("성적이 총점 기준으로 내림차순 정렬되었습니다.")

    def show_all_scores(self):
        """전체 성적을 출력합니다."""
        if not self.score_list:
            print("저장된 성적이 없습니다.")
        else:
            print("이름\t국어\t영어\t수학\t총점\t평균\t등급")
            for score in self.score_list:
                score.display()

    def start(self):
        """메인 메뉴 루프를 실행합니다."""
        while True:
            print("\n--- 성적 관리 시스템 ---")
            print("1. 성적 추가")
            print("2. 성적 검색")
            print("3. 성적 수정")
            print("4. 성적 삭제")
            print("5. 성적 정렬")
            print("6. 전체 성적 보기")
            print("7. 저장")
            print("0. 종료")

            choice = input("> ")

            if choice == '1':
                self.append_score()
            elif choice == '2':
                self.search_score()
            elif choice == '3':
                self.modify_score()
            elif choice == '4':
                self.delete_score()
            elif choice == '5':
                self.sort_scores()
            elif choice == '6':
                self.show_all_scores()
            elif choice == '7':
                self.save_scores()
            elif choice == '0':
                print("프로그램을 종료합니다.")
                break
            else:
                print("잘못된 선택입니다. 다시 입력해주세요.")
```

### 데이터 영속성 확보

-   **저장 (`save_scores`)**: 사용자가 "저장" 메뉴를 선택하면, `ScoreManager`의 `score_list` (이 리스트에는 `ScoreData` 인스턴스들이 들어있음)가 `pickle.dump`를 통해 지정된 파일에 통째로 직렬화됩니다.
-   **불러오기 (`load_scores`)**: `ScoreManager` 객체가 생성될 때(`__init__`) 자동으로 `load_scores`를 호출합니다. 파일이 존재하면 `pickle.load`를 통해 파일의 내용을 역직렬화하여 `score_list`를 복원합니다. 이를 통해 프로그램을 종료했다가 다시 켜도 이전 작업 내용을 그대로 이어서 사용할 수 있게 됩니다.

> **설계의 장점**:
> - **데이터 영속성**: 프로그램이 종료되어도 데이터가 사라지지 않습니다.
> - **객체 지향 구조 유지**: 파일 I/O 로직이 `ScoreManager` 클래스 내에 캡슐화되어 있어, 다른 클래스들은 파일 처리 방식에 대해 알 필요가 없습니다.
> - **효율성**: 복잡한 객체 구조를 단 두 줄의 코드로 저장하고 불러올 수 있어 매우 효율적입니다.

---

[⏮️ 이전 문서](./0508_Python정리.md) | [다음 문서 ⏭️](./0512_Python정리.md)