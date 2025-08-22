<h2>LangChain 학습 가이드: 체인(Chains)으로 워크플로우 구축하기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LangChain의 핵심 개념인 '체인(Chain)'의 원리를 이해하고, 다양한 종류의 체인을 활용하여 여러 컴포넌트(모델, 프롬프트 등)를 연결하고 복잡한 워크플로우를 구축하는 방법을 학습하는 것을 목표로 합니다. 특히 최신 LangChain의 표준이 된 LCEL(LangChain Expression Language)을 중심으로 체인을 구성하는 방법을 익힙니다.

<h2>목차</h2>

- [1. 왜 체인이 필요한가?](#1-왜-체인이-필요한가)
- [2. LCEL: 체인 구성을 위한 새로운 표준](#2-lcel-체인-구성을-위한-새로운-표준)
  - [2.1. LCEL의 기본 구조](#21-lcel의-기본-구조)
  - [2.2. Runnable 프로토콜](#22-runnable-프로토콜)
- [3. 기본적인 체인: LLMChain](#3-기본적인-체인-llmchain)
  - [3.1. 구성 요소와 데이터 흐름](#31-구성-요소와-데이터-흐름)
  - [3.2. 코드 예제](#32-코드-예제)
- [4. 순차적인 체인: SequentialChain](#4-순차적인-체인-sequentialchain)
  - [4.1. 여러 체인의 연결](#41-여러-체인의-연결)
  - [4.2. 코드 예제](#42-코드-예제)

---

## 1. 왜 체인이 필요한가?

LLM 애플리케이션은 단순히 LLM을 한 번 호출하는 것으로 끝나지 않는 경우가 많습니다. 사용자의 입력을 받아 프롬프트를 만들고, LLM을 호출한 뒤, 그 결과를 파싱하여 다른 LLM에 전달하거나, 데이터베이스에 저장하는 등 여러 단계의 작업이 필요합니다.

체인은 이러한 다단계 워크플로우를 **하나의 응집력 있는 단위**로 묶어주는 역할을 합니다. 이를 통해 코드의 구조가 명확해지고, 재사용성이 높아지며, 복잡한 로직을 보다 쉽게 관리할 수 있습니다.

## 2. LCEL: 체인 구성을 위한 새로운 표준

LCEL(LangChain Expression Language)은 LangChain에서 체인을 구성하는 선언적인 방법입니다. 파이썬의 파이프(`|`) 연산자를 사용하여, 마치 데이터가 흘러가는 것처럼 각 컴포넌트를 자연스럽게 연결할 수 있습니다.

### 2.1. LCEL의 기본 구조
```python
chain = component1 | component2 | component3
```
위 코드는 `component1`의 출력이 `component2`의 입력으로, `component2`의 출력이 `component3`의 입력으로 전달되는 데이터 파이프라인을 의미합니다.

### 2.2. Runnable 프로토콜
LCEL로 연결될 수 있는 모든 컴포넌트(모델, 프롬프트, 파서 등)는 `Runnable`이라는 공통 프로토콜을 따릅니다. 이 프로토콜은 `invoke`, `stream`, `batch` 등 통일된 실행 메소드를 제공하여, 어떤 체인이든 일관된 방식으로 호출할 수 있게 해줍니다.

## 3. 기본적인 체인: LLMChain

`LLMChain`은 가장 기본적이면서도 널리 사용되는 체인으로, **프롬프트 템플릿 + 모델 (+ 출력 파서)** 의 조합으로 이루어집니다.

### 3.1. 구성 요소와 데이터 흐름
1.  **입력**: 딕셔너리 형태의 사용자 입력 (예: `{"product": "AI 스피커"}`)
2.  **프롬프트 템플릿**: 입력을 받아 완전한 프롬프트를 생성
3.  **LLM 모델**: 생성된 프롬프트를 받아 응답 생성
4.  **출력**: LLM이 생성한 텍스트 (또는 출력 파서가 변환한 객체)

### 3.2. 코드 예제
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 컴포넌트 준비
prompt = ChatPromptTemplate.from_template("{product}의 특징을 설명하는 세일즈 포인트를 3가지 알려주세요.")
model = ChatOpenAI()
output_parser = StrOutputParser() # 모델의 출력(AIMessage)을 문자열로 변환

# LCEL을 사용하여 체인 구성
chain = prompt | model | output_parser

# 체인 실행
response = chain.invoke({"product": "AI 스마트 워치"})

print(response)
```

## 4. 순차적인 체인: SequentialChain

`SequentialChain`은 여러 개의 체인을 순차적으로 연결하여, 한 체인의 출력을 다음 체인의 입력으로 사용하는 복잡한 워크플로우를 만들 때 사용됩니다.

### 4.1. 여러 체인의 연결
예를 들어, 다음과 같은 2단계 작업을 생각할 수 있습니다.
1.  **1단계**: 특정 주제에 대한 희곡 제목을 생성한다.
2.  **2단계**: 생성된 희곡 제목을 바탕으로 간단한 시놉시스를 작성한다.

### 4.2. 코드 예제
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SequentialChain, LLMChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

# 1. 첫 번째 체인: 희곡 제목 생성
prompt1 = ChatPromptTemplate.from_template("{topic}을 주제로 하는 연극 제목을 하나 제안해주세요.")
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="title") # 출력 변수 이름을 'title'로 지정

# 2. 두 번째 체인: 시놉시스 작성
# chain1의 출력인 'title'을 입력 변수로 받음
prompt2 = ChatPromptTemplate.from_template("연극 제목이 \"{title}\"일 때, 간단한 시놉시스를 작성해주세요.")
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="synopsis")

# 두 체인을 SequentialChain으로 연결
# input_variables: 전체 체인의 시작점이 되는 입력 변수
# output_variables: 최종적으로 반환될 출력 변수들
overall_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["topic"],
    output_variables=["title", "synopsis"],
    verbose=True # 체인의 실행 과정을 로그로 출력
)

# 전체 체인 실행
response = overall_chain.invoke({"topic": "시간 여행자의 딜레마"})

print(response)
# 예상 출력:
# {'topic': '시간 여행자의 딜레마', 'title': '어제의 그림자', 'synopsis': '과거를 바꾸려는 한 시간 여행자가 자신의 선택이 현재에 예상치 못한 비극을 불러오는 것을 깨닫고, 모든 것을 되돌리기 위해 고군분투하는 이야기입니다.'}
```
이처럼 체인을 사용하면 복잡한 LLM 기반의 워크플로우를 체계적으로 설계하고 구현할 수 있습니다.