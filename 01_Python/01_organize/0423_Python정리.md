# 🐍 Python 프로그래밍 기초: Day 1 학습 노트

> **이 문서의 목적**: 이 문서는 데이터 분석/AI 부트캠프 첫날 학습한 Python의 핵심 기초를 포트폴리오용으로 정리한 것입니다. 단순한 요약을 넘어, 
각 개념의 **Why**와 **How**를 깊이 있게 다루어, 탄탄한 기본기를 갖추었음을 보여주는 것을 목표로 합니다.

---

## 목차

1.  [**Python: 언어 철학과 특징**](#1-python-언어-철학과-특징)
    -   [Python의 설계 철학: The Zen of Python](#python의-설계-철학-the-zen-of-python)
    -   [핵심 특징 요약](#핵심-특징-요약)
2.  [**개발 환경 구축: Anaconda와 VSCode**](#2-개발-환경-구축-anaconda와-vscode)
    -   [왜 Anaconda를 사용하는가?: `pip`와의 차이점](#왜-anaconda를-사용하는가-pip와의-차이점)
    -   [Conda 가상 환경 설정 및 관리](#conda-가상-환경-설정-및-관리)
    -   [VSCode 연동 및 필수 확장 프로그램](#vscode-연동-및-필수-확장-프로그램)
3.  [**Python 프로그래밍 첫걸음: 필수 문법**](#3-python-프로그래밍-첫걸음-필수-문법)
    -   [주석(Comments): 코드의 길잡이](#주석comments-코드의-길잡이)
    -   [변수(Variables): 데이터에 이름표 붙이기](#변수variables-데이터에-이름표-붙이기)
    -   [자료형(Data Types): 데이터의 종류](#자료형data-types-데이터의-종류)
    -   [형 변환(Type Casting): 데이터의 변신](#형-변환type-casting-데이터의-변신)
    -   [기본 입출력(I/O)](#기본-입출력io)
    -   [연산자(Operators): 계산과 처리](#연산자operators-계산과-처리)
4.  [**실전 예제: 개념 응용하기**](#4-실전-예제-개념-응용하기)
    -   [예제 1: 단위 변환기 (섭씨 ↔ 화씨)](#예제-1-단위-변환기-섭씨--화씨)
5.  [**핵심 개념 요약표**](#5-핵심-개념-요약표)

---

## 1. Python: 언어 철학과 특징

Python은 단순한 프로그래밍 언어를 넘어, 특유의 철학을 가진 강력한 도구입니다.

### Python의 설계 철학: The Zen of Python

Python은 가독성과 단순함을 중시하는 고유의 설계 철학을 가지고 있습니다. 이는 `import this`를 실행하면 나타나는 "Python의 선(Zen of Python)"이라는 20가지 원칙에 잘 나타나 있습니다.

> **The Zen of Python (일부 발췌)**
>
> -   Beautiful is better than ugly. (아름다운 것이 추한 것보다 낫다.)
> -   Explicit is better than implicit. (명시적인 것이 암시적인 것보다 낫다.)
> -   Simple is better than complex. (단순한 것이 복잡한 것보다 낫다.)
> -   Readability counts. (가독성은 중요하다.)

이러한 철학 덕분에 Python 코드는 다른 사람이 읽고 이해하기 쉬우며, 유지보수가 용이합니다.

### 핵심 특징 요약

| 특징 | 설명 |
| :--- | :--- |
| **높은 생산성** | 간결한 문법과 방대한 라이브러리 덕분에 적은 양의 코드로 복잡한 작업을 빠르게 구현할 수 있습니다. |
| **인터프리터 언어** | 코드를 한 줄씩 해석하고 즉시 실행하여, 빠른 프로토타이핑과 디버깅이 가능합니다. |
| **동적 타이핑** | 변수 선언 시 타입을 명시할 필요가 없어 유연하고 신속한 개발이 가능합니다. (단, 런타임 오류의 가능성이 있어 주의가 필요합니다.) |
| **강력한 생태계** | 데이터 과학(`Pandas`, `NumPy`), AI(`TensorFlow`, `PyTorch`), 웹(`Django`, `FastAPI`) 등 특정 목적을 위한 수많은 라이브러리를 통해 무한한 확장이 가능합니다. |

---

## 2. 개발 환경 구축: Anaconda와 VSCode

전문적인 데이터 분석 및 AI 개발을 위해서는 안정적이고 격리된 개발 환경이 필수적입니다. Anaconda는 이를 위한 산업 표준 도구입니다.

### 왜 Anaconda를 사용하는가?: `pip`와의 차이점

`pip`는 Python의 기본 패키지 관리자이지만, 데이터 과학 분야에서는 `conda`가 더 선호됩니다.

-   **의존성 관리의 차이**:
    -   `pip`: Python 패키지만을 관리합니다. C, C++ 등으로 빌드된 라이브러리(예: `NumPy`, `SciPy`)의 복잡한 비(非)Python 의존성을 해결하지 못해 종종 빌드 오류가 발생합니다.
    -   `conda`: Python 패키지뿐만 아니라, 그 패키지가 의존하는 **모든 외부 라이브러리(C, Fortran 등)까지 포함하여 패키지를 관리**합니다. 미리 컴파일된 바이너리 형태로 설치하므로, 복잡한 의존성 문제를 원천적으로 방지하고 설치 속도가 빠릅니다.

-   **환경 격리**:
    -   `pip`는 `venv`와 함께 사용하여 가상 환경을 관리하지만, `conda`는 **환경 관리 기능이 내장**되어 있어 `conda activate` 명령어로 Python 버전까지 완벽하게 격리된 환경을 손쉽게 만들고 전환할 수 있습니다.

> **결론**: `conda`는 복잡한 과학 계산 라이브러리의 의존성 문제를 해결하고, 언어 버전까지 포함한 완벽한 환경 격리를 제공하므로 데이터 과학 프로젝트에 훨씬 안정적이고 적합합니다.

### Conda 가상 환경 설정 및 관리

1. **Anaconda 다운로드**
- 공식 페이지: https://www.anaconda.com/products/distribution
2. **설치 옵션**
- "Just Me" 또는 "All Users" 선택
- "Add Anaconda to my PATH environment variable" 체크 안 함 (권장)
설치 후 Anaconda Prompt 실행

3.  **가상 환경 생성**:
    ```bash
    # 'sesac_env'라는 이름으로 Python 3.9 버전의 가상 환경 생성
    conda create -n sesac_env python=3.9
    ```
    > **실무 Tip**: 프로젝트마다 별도의 환경을 생성하는 것은 협업과 유지보수의 기본입니다. TensorFlow, PyTorch 등 주요 프레임워크들은 특정 Python 버전에 가장 안정적이므로, `python=3.9`와 같이 버전을 명시하는 것이 호환성 문제를 예방하는 좋은 습관입니다.

4.  **가상 환경 활성화 및 비활성화**:
    ```bash
    # 생성한 가상 환경 활성화
    conda activate sesac_env

    # (작업 완료 후) 가상 환경 비활성화
    conda deactivate
    ```

5.  **설치된 패키지 목록 확인**:
    ```bash
    conda list
    ```

#### 문제 해결: 'conda' 명령어를 찾을 수 없을 때

> **Note**: Anaconda 설치 시 `conda init` 명령어가 대부분의 셸(Anaconda Prompt, Git Bash, Powershell) 설정을 자동으로 처리해줍니다. 아래의 수동 설정은 `cmd.exe`와 같은 기본 터미널에서 `conda` 명령어가 인식되지 않는 예외적인 경우에만 필요합니다.

Anaconda 설치 프로그램은 시스템의 다른 Python 버전과의 충돌을 피하기 위해 기본적으로 Anaconda를 시스템 PATH 환경 변수에 추가하는 것을 권장하지 않습니다. 하지만 필요에 따라 수동으로 추가할 수 있습니다.

1.  **시스템 환경 변수 편집 열기**:
    -   Windows 검색창에서 `시스템 환경 변수 편집`을 검색하여 실행합니다.

2.  **환경 변수 설정 창 열기**:
    -   `시스템 속성` 창에서 `환경 변수(N)...` 버튼을 클릭합니다.

3.  **Path 변수 편집**:
    -   `시스템 변수` 섹션에서 `Path` 변수를 찾아 선택한 후 `편집(I)...`을 클릭합니다.

4.  **Anaconda 경로 추가**:
    -   `새로 만들기(N)`를 클릭하여 아래의 경로들을 순서대로 추가합니다. `<UserName>` 부분은 본인의 Windows 사용자 이름으로 변경해야 합니다.
        -   `C:\Users\<UserName>\anaconda3`
        -   `C:\Users\<UserName>\anaconda3\Scripts`
        -   `C:\Users\<UserName>\anaconda3\Library\bin`

5.  **확인 및 재시작**:
    -   모든 창에서 `확인`을 눌러 변경 사항을 저장합니다.
    -   변경 사항을 적용하려면 열려 있는 모든 터미널 창을 닫고 새로 시작해야 합니다.

### VSCode 연동 및 필수 확장 프로그램

-   **Python 인터프리터 연결**:
    1.  VSCode(https://code.visualstudio.com/) 설치 이후 `Ctrl+Shift+P`를 눌러 명령어 팔레트를 엽니다.
    2.  `Python: Select Interpreter`를 검색하여 선택합니다.
    3.  목록에서 방금 생성한 `conda` 가상 환경(`sesac_env:conda`)을 선택합니다.
    4.  이제 VSCode의 터미널과 코드 실행이 선택된 `conda` 환경을 기반으로 동작합니다.

-   **추천 확장 프로그램**:
    -   **Python (Microsoft)**: Python 언어 지원을 위한 필수 확장 프로그램.
    -   **Pylance (Microsoft)**: 빠른 정적 타입 검사, 자동 완성, 코드 탐색 등 강력한 인텔리센스 기능을 제공하여 생산성을 극대화합니다. 코드 작성 시 잠재적인 타입 오류를 미리 발견해줍니다.

---

## 3. Python 프로그래밍 첫걸음: 필수 문법

### 주석(Comments): 코드의 길잡이

주석은 코드가 **"무엇을"** 하는지보다 **"왜"** 그렇게 작성되었는지를 설명하는 데 사용하는 것이 좋습니다.

```python
# 한 줄 주석: 코드 라인에 대한 간단한 설명
# 예: 사용자의 입력을 받아 정수로 변환 (명확한 의도 전달)
user_age = int(input("나이를 입력하세요: "))

"""
여러 줄 주석:
함수나 클래스의 목적, 사용법, 주요 로직 등
복잡한 내용을 설명할 때 유용합니다.
이 함수는 사용자 정보를 받아 데이터베이스에 저장하는 역할을 합니다.
"""
```

### 변수(Variables): 데이터에 이름표 붙이기

변수는 특정 데이터(객체)를 담는 상자가 아니라, 데이터가 저장된 **메모리상의 위치를 가리키는 이름표(label)**와 같습니다.

```python
# weight 변수는 메모리에 생성된 9.8이라는 float 객체를 가리킵니다.
weight = 9.8

# message 변수는 "Hello"라는 str 객체를 가리킵니다.
message = "Hello"
```

-   **변수명 규칙 (PEP 8)**: Python에서는 코드의 가독성을 위해 **`snake_case`** 표기법을 권장합니다.
    -   **Good**: `user_input`, `weekly_salary`
    -   **Bad**: `userInput`, `WeeklySalary`, `data1`

### 자료형(Data Types): 데이터의 종류

Python의 기본 자료형은 다음과 같습니다.

| 자료형 | 설명 | 예시 |
| :--- | :--- | :--- |
| `int` | 정수 (Integer) | `10`, `-5`, `0` |
| `float` | 실수 (Floating-point) | `3.14`, `-0.01` |
| `str` | 문자열 (String) | `"Hello"`, `'Python'` |
| `bool` | 불리언 (Boolean) | `True`, `False` (첫 글자는 대문자) |

### 형 변환(Type Casting): 데이터의 변신

`input()`으로 받은 문자열을 숫자로 바꾸거나, 숫자를 문자열에 포함시키는 등 데이터의 타입을 의도적으로 변경하는 작업입니다.

```python
# input()은 항상 str 타입을 반환
age_str = input("나이: ") # 예: "25"

# 숫자 연산을 위해 int로 형 변환
age_int = int(age_str)
print(f"10년 후 당신의 나이는 {age_int + 10}살 입니다.")

# 숫자를 문자열과 연결하기 위해 str로 형 변환
year = 2024
print("올해는 " + str(year) + "년입니다.")
```

> **⚠️ 주의**: 변환할 수 없는 값을 형 변환 시 `ValueError`가 발생합니다.
> `int("스물다섯")` # ValueError: invalid literal for int() with base 10: '스물다섯'
> 안정적인 프로그램은 이러한 예외를 처리하는 로직이 반드시 필요합니다.

### 기본 입출력(I/O)

#### `print()`: 화면에 정보 출력하기

`print()` 함수는 다양한 옵션을 통해 출력 형식을 제어할 수 있습니다.

```python
# 기본 사용법
print("Hello", "Python") # 출력: Hello Python (기본적으로 띄어쓰기로 구분)

# sep: 구분자(separator) 지정
print("2024", "07", "26", sep="-") # 출력: 2024-07-26

# end: 출력 끝에 추가할 문자 지정
print("줄이 바뀌지 않습니다.", end=" ")
print("바로 옆에 출력됩니다.")
# 출력: 줄이 바뀌지 않습니다. 바로 옆에 출력됩니다.
```

#### `input()`: 사용자로부터 정보 입력받기

`input()` 함수는 사용자에게 안내 메시지(prompt)를 보여주고 키보드 입력을 기다립니다. **모든 입력은 문자열(`str`)로 반환**된다는 점을 항상 기억해야 합니다.

#### 문자열 포매팅: f-string (Python 3.6+)

변수와 문자열을 조합할 때 가장 권장되는 현대적인 방식입니다. 가독성이 뛰어나고 사용이 간편합니다.

```python
name = "Alice"
age = 30

# f-string: 문자열 앞에 'f'를 붙이고, 변수를 {}로 감싸 사용
greeting = f"안녕하세요, 제 이름은 {name}이고 나이는 {age}살입니다."
print(greeting)
# 출력: 안녕하세요, 제 이름은 Alice이고 나이는 30살입니다.
```

### 연산자(Operators): 계산과 처리

| 종류 | 연산자 | 설명 | 예시 (`a=13`, `b=5`) | 결과 |
| :--- | :---: | :--- | :--- | :---: |
| **산술** | `+`, `-`, `*` | 덧셈, 뺄셈, 곱셈 | `a * b` | `65` |
| | `/` | 나눗셈 (결과는 항상 `float`) | `a / b` | `2.6` |
| | `//` | **몫** (정수 나눗셈) | `a // b` | `2` |
| | `%` | **나머지** (짝수/홀수 판별, 주기 계산 등) | `a % b` | `3` |
| | `**` | 거듭제곱 | `b ** 2` | `25` |
| **문자열** | `+` | 문자열 연결(Concatenation) | `"Hi" + "There"` | `"HiThere"` |
| | `*` | 문자열 반복 | `"Hi" * 3` | `"HiHiHi"` |

---

## 4. 실전 예제: 개념 응용하기

### 예제 1: 단위 변환기 (섭씨 ↔ 화씨)

지금까지 배운 `input`, 형 변환, 연산자, f-string을 모두 활용하여 실용적인 프로그램을 만들어 봅니다.

> **공식**:
> - 섭씨(℃) → 화씨(℉): `(섭씨 온도 × 9/5) + 32`
> - 화씨(℉) → 섭씨(℃): `(화씨 온도 - 32) × 5/9`

```python
# 1. 사용자로부터 변환하고 싶은 온도를 입력받습니다.
#    소수점 입력을 고려하여 float으로 형 변환합니다.
celsius_temp_str = input("화씨로 변환할 섭씨 온도를 입력하세요: ")
celsius_temp = float(celsius_temp_str)

# 2. 섭씨를 화씨로 변환하는 공식을 적용합니다.
#    연산자 우선순위에 따라 곱셈/나눗셈이 먼저 계산됩니다.
fahrenheit_temp = (celsius_temp * 9/5) + 32

# 3. f-string을 사용하여 결과를 명확하고 읽기 쉽게 출력합니다.
#    소수점 둘째 자리까지만 표시하기 위해 :.2f 포맷팅을 사용합니다.
print(f"섭씨 {celsius_temp}°C는 화씨 {fahrenheit_temp:.2f}°F 입니다.")

# --- 반대 경우도 계산 ---

# 4. 사용자로부터 화씨 온도를 입력받습니다.
fahrenheit_temp_str = input("섭씨로 변환할 화씨 온도를 입력하세요: ")
fahrenheit_temp_2 = float(fahrenheit_temp_str)

# 5. 화씨를 섭씨로 변환합니다.
celsius_temp_2 = (fahrenheit_temp_2 - 32) * 5/9

# 6. 결과를 소수점 둘째 자리까지 출력합니다.
print(f"화씨 {fahrenheit_temp_2}°F는 섭씨 {celsius_temp_2:.2f}°C 입니다.")
```

---

## 5. 핵심 개념 요약표

| 구분 | 개념 | 핵심 설명 | 실무 Tip / 예시 |
| :--- | :--- | :--- | :--- |
| **환경** | `conda` | Python 및 외부 라이브러리 의존성을 관리하는 **패키지 및 환경 관리자**. | `conda create -n proj_env python=3.9` |
| **문법** | `print()` | 디버깅 및 결과 확인을 위한 기본 출력 함수. | `print(f"Value: {my_var}", sep=" | ")` |
| | `input()` | 사용자 입력을 **문자열(str)**로 받음. 숫자 계산 시 반드시 형 변환 필요. | `age = int(input("나이: "))` |
| **변수** | `snake_case` | PEP 8에서 권장하는 변수명 규칙. 가독성을 높임. | `weekly_salary`, `user_name` |
| **자료형** | 동적 타이핑 | 변수에 값이 할당될 때 타입이 결정됨. 유연하지만 타입 오류에 주의. | `x = 10` (int) → `x = "ten"` (str) |
| | 형 변환 | `int()`, `float()`, `str()` 등을 사용해 타입을 명시적으로 변경. | `float("3.14")` |
| **문자열** | f-string | 변수를 문자열에 삽입하는 가장 현대적이고 가독성 좋은 방법. | `f"User {user_id} logged in."` |
| **연산자** | `//` vs `/` | `//`는 몫(정수), `/`는 나눗셈(실수) 결과를 반환. | `10 // 3` → `3`, `10 / 3` → `3.33...` |

---

##### 🏫 <a href="https://sesac.seoul.kr/course/active/detail.do?courseActiveSeq=2866" target="_blank">(청년취업사관학교-영등포 7기) 데이터/AI 개발자 과정</a>

[다음 문서 ⏭️](./0424_Python정리.md)
