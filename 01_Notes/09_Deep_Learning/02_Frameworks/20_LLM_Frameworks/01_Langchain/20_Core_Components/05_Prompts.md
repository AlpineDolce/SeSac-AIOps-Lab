<h2>LangChain 학습 가이드: 프롬프트(Prompts) 설계의 모든 것</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LLM의 행동을 제어하고 원하는 결과물을 이끌어내는 핵심 요소인 프롬프트를 효과적으로 설계하고 관리하는 방법을 다룹니다. 정적인 프롬프트를 넘어, 사용자의 입력과 상황에 따라 동적으로 변화하는 프롬프트 템플릿의 개념을 이해하고, 이를 활용하여 유연하고 강력한 LLM 애플리케이션을 구축하는 것을 목표로 합니다.

<h2>목차</h2>

- [1. 프롬프트의 중요성](#1-프롬프트의-중요성)
- [2. 프롬프트 템플릿 (Prompt Template)](#2-프롬프트-템플릿-prompt-template)
  - [2.1. 왜 템플릿을 사용하는가?](#21-왜-템플릿을-사용하는가)
  - [2.2. 기본적인 프롬프트 템플릿 만들기](#22-기본적인-프롬프트-템플릿-만들기)
- [3. 챗 프롬프트 템플릿 (Chat Prompt Template)](#3-챗-프롬프트-템플릿-chat-prompt-template)
  - [3.1. 역할 기반의 메시지 템플릿](#31-역할-기반의-메시지-템플릿)
  - [3.2. 코드 예제](#32-코드-예제)
- [4. 고급 프롬프트 기법: Few-Shot](#4-고급-프롬프트-기법-few-shot)
  - [4.1. Few-Shot 프롬프팅이란?](#41-few-shot-프롬프팅이란)
  - [4.2. FewShotPromptTemplate 사용법](#42-fewshotprompttemplate-사용법)

--- 

## 1. 프롬프트의 중요성

LLM은 프롬프트에 따라 전혀 다른 결과물을 생성합니다. 'Garbage in, garbage out'이라는 말처럼, 잘 설계된 프롬프트는 LLM의 성능을 극대화하는 반면, 모호하거나 잘못된 프롬프트는 엉뚱한 답변을 유도할 수 있습니다. 따라서 프롬프트 엔지니어링은 LLM 애플리케이션 개발의 핵심 기술 중 하나입니다.

## 2. 프롬프트 템플릿 (Prompt Template)

### 2.1. 왜 템플릿을 사용하는가?
애플리케이션에서는 사용자의 입력이나 특정 변수에 따라 프롬프트의 내용이 동적으로 바뀌어야 하는 경우가 많습니다. 프롬프트 템플릿은 이러한 동적인 프롬프트를 생성하기 위한 '틀'을 제공합니다.

- **재사용성**: 동일한 프롬프트 구조를 여러 곳에서 재사용할 수 있습니다.
- **관리 용이성**: 프롬프트의 구조와 내용을 분리하여 관리하기 용이합니다.
- **유연성**: 변수만 변경하여 다양한 상황에 맞는 프롬프트를 쉽게 생성할 수 있습니다.

### 2.2. 기본적인 프롬프트 템플릿 만들기
`PromptTemplate` 클래스를 사용하여 간단한 템플릿을 만들 수 있습니다.

```python
from langchain_core.prompts import PromptTemplate

# input_variables: 템플릿 안에서 사용될 변수들의 이름을 리스트로 지정
# template: 실제 프롬프트의 틀
prompt_template = PromptTemplate(
    input_variables=["product"],
    template="{product}의 새로운 광고 문구를 3개 제안해주세요."
)

# format 메소드를 사용하여 변수에 값을 채워넣음
formatted_prompt = prompt_template.format(product="무선 이어폰")

print(formatted_prompt)
# 출력:
# 무선 이어폰의 새로운 광고 문구를 3개 제안해주세요.
```

## 3. 챗 프롬프트 템플릿 (Chat Prompt Template)

Chat Model은 단순한 문자열이 아닌, 역할(role)이 부여된 메시지 리스트를 입력으로 받습니다. `ChatPromptTemplate`은 이러한 메시지 리스트를 동적으로 생성하는 데 사용됩니다.

### 3.1. 역할 기반의 메시지 템플릿
`SystemMessage`, `HumanMessage`, `AIMessage`에 해당하는 각각의 템플릿을 만들고 이를 조합하여 전체 대화의 템플릿을 구성합니다.

### 3.2. 코드 예제
```python
from langchain_core.prompts import ChatPromptTemplate

# ChatPromptTemplate.from_messages를 사용하여 메시지 템플릿 리스트를 생성
chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {language}로 응답하는 번역 전문가입니다."),
    ("human", "{text}를 번역해주세요.")
])

# format_messages 메소드를 사용하여 변수 값을 채워넣음
formatted_messages = chat_template.format_messages(language="영어", text="안녕하세요, 반갑습니다.")

print(formatted_messages)
# 출력:
# [
#   SystemMessage(content='당신은 영어로 응답하는 번역 전문가입니다.'),
#   HumanMessage(content='안녕하세요, 반갑습니다.를 번역해주세요.')
# ]
```

## 4. 고급 프롬프트 기법: Few-Shot

### 4.1. Few-Shot 프롬프팅이란?
LLM에게 단순히 작업만 지시하는 것(Zero-Shot)을 넘어, 몇 개의 예시(example)를 함께 제공하여 원하는 결과물의 형식이나 스타일을 더 명확하게 알려주는 기법입니다. 이를 통해 모델의 성능을 크게 향상시킬 수 있습니다.

### 4.2. FewShotPromptTemplate 사용법
`FewShotPromptTemplate`은 이러한 예시들을 동적으로 선택하고 프롬프트에 포함시키는 과정을 자동화해줍니다.

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 제공할 예시 데이터
examples = [
    {"input": "행복", "output": "Happy"},
    {"input": "슬픔", "output": "Sad"},
]

# 각 예시를 어떻게 포맷할지 정의하는 템플릿
example_prompt = PromptTemplate.from_template("입력: {input}\n출력: {output}")

# Few-shot 프롬프트 템플릿 생성
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="다음 단어를 영어로 번역하세요:",
    suffix="입력: {user_input}\n출력:",
    input_variables=["user_input"],
)

# 템플릿 포맷팅
formatted_prompt = few_shot_prompt.format(user_input="사랑")

print(formatted_prompt)
# 출력:
# 다음 단어를 영어로 번역하세요:
#
# 입력: 행복
# 출력: Happy
#
# 입력: 슬픔
# 출력: Sad
#
# 입력: 사랑
# 출력:
```
이처럼 Few-shot 프롬프팅을 사용하면, LLM은 주어진 예시의 패턴을 학습하여 '사랑'에 대한 번역 결과로 'Love'를 출력할 확률이 매우 높아집니다.
