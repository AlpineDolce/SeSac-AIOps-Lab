<h2>LangChain 학습 가이드: 커스터마이제이션 - 나만의 컴포넌트 만들기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LangChain이 제공하는 기본 컴포넌트를 넘어, 특정 도메인이나 문제 상황에 맞는 자신만의 도구(Tool), 체인(Chain), 에이전트(Agent)를 만드는 방법을 학습하는 것을 목표로 합니다. 커스터마이제이션을 통해 LangChain의 기능을 무한히 확장하고, 복잡하고 독창적인 LLM 애플리케이션을 구현하는 능력을 기릅니다.

<h2>목차</h2>

- [1. 왜 커스터마이제이션이 필요한가?](#1-왜-커스터마이제이션이-필요한가)
- [2. 커스텀 도구 (Custom Tool) 만들기](#2-커스텀-도구-custom-tool-만들기)
  - [2.1. `@tool` 데코레이터 활용](#21-tool-데코레이터-활용)
- [3. 커스텀 체인 (Custom Chain) 만들기](#3-커스텀-체인-custom-chain-만들기)
  - [3.1. LCEL을 이용한 조합](#31-lcel을-이용한-조합)
  - [3.2. `Runnable` 클래스 상속 (고급)](#32-runnable-클래스-상속-고급)
- [4. 커스텀 에이전트 (Custom Agent) 만들기](#4-커스텀-에이전트-custom-agent-만들기)
  - [4.1. 커스텀 에이전트의 구성 요소](#41-커스텀-에이전트의-구성-요소)
  - [4.2. 구현 단계](#42-구현-단계)

---

## 1. 왜 커스터마이제이션이 필요한가?

LangChain은 수많은 유용한 컴포넌트를 기본적으로 제공하지만, 세상의 모든 문제를 해결할 수는 없습니다. 실제 프로젝트에서는 다음과 같은 상황을 마주하게 됩니다.

- 우리 회사 내부 데이터베이스에만 접근할 수 있는 특별한 기능이 필요할 때
- 여러 API를 조합하여 특정 비즈니스 로직을 수행해야 할 때
- 기존 에이전트의 작동 방식(추론 방식)을 우리 서비스에 맞게 바꾸고 싶을 때

이러한 경우, 직접 컴포넌트를 만들어 LangChain의 생태계 안에서 유기적으로 동작하도록 해야 합니다.

## 2. 커스텀 도구 (Custom Tool) 만들기

에이전트가 사용할 수 있는 나만의 기능을 만드는 가장 쉬운 방법입니다.

### 2.1. `@tool` 데코레이터 활용
일반 파이썬 함수 위에 `@tool` 데코레이터를 붙이기만 하면 간단하게 도구를 만들 수 있습니다. 함수의 독스트링(docstring)은 LLM이 도구의 용도를 파악하는 데 사용되므로, 명확하게 작성하는 것이 매우 중요합니다.

```python
from langchain.tools import tool
import datetime

@tool
def get_current_day_of_week() -> str:
    """오늘이 무슨 요일인지 한국어로 반환합니다."""
    days = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
    return days[datetime.datetime.today().weekday()]

# 생성된 도구는 에이전트의 tools 리스트에 포함시켜 사용
tools = [get_current_day_of_week]
```

## 3. 커스텀 체인 (Custom Chain) 만들기

### 3.1. LCEL을 이용한 조합
대부분의 커스텀 체인은 LCEL을 사용하여 기존 컴포넌트들을 새롭게 조합하는 것만으로도 충분히 만들 수 있습니다. 이것이 가장 권장되는 방법입니다.

```python
# 예시: 질문을 받아서 먼저 영어로 번역하고, 그 다음에 LLM에게 답변을 요청하는 체인
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

translate_prompt = ChatPromptTemplate.from_template("Translate the following text to English: {text}")
answer_prompt = ChatPromptTemplate.from_template("Answer the following question: {english_question}")
model = ChatOpenAI()

# 번역 체인
translation_chain = translate_prompt | model | StrOutputParser()

# 답변 체인
answer_chain = answer_prompt | model | StrOutputParser()

# 두 체인을 LCEL로 연결
# translation_chain의 출력을 answer_chain의 입력(english_question)으로 매핑
custom_chain = {"english_question": translation_chain} | answer_chain

response = custom_chain.invoke({"text": "강아지의 특징은?"})
```

### 3.2. `Runnable` 클래스 상속 (고급)
체인 내에서 복잡한 상태를 관리하거나, 비표준적인 입출력을 다뤄야 하는 매우 드문 경우에는 `Runnable` 클래스를 직접 상속하여 자신만의 커스텀 `invoke` 메소드를 구현할 수 있습니다.

## 4. 커스텀 에이전트 (Custom Agent) 만들기

LLM이 생각하고 행동하는 방식 자체를 바꾸고 싶을 때 커스텀 에이전트를 만듭니다. 이는 가장 고급 수준의 커스터마이제이션입니다.

### 4.1. 커스텀 에이전트의 구성 요소
- **커스텀 프롬프트**: LLM이 어떻게 생각하고, 어떤 형식으로 답변을 출력해야 하는지를 정의하는 새로운 지시문을 만듭니다.
- **커스텀 출력 파서**: LLM의 출력(문자열)을 파싱하여 다음 행동(Action)이나 최종 답변(Final Answer)으로 변환하는 로직을 구현합니다.

### 4.2. 구현 단계
1.  **도구 정의**: 에이전트가 사용할 도구들을 정의합니다.
2.  **프롬프트 템플릿 생성**: LLM에게 역할을 부여하고, 도구 사용법과 출력 형식을 안내하는 프롬프트를 상세하게 작성합니다.
3.  **출력 파서 작성**: 프롬프트에서 정의한 출력 형식에 맞춰 LLM의 텍스트 출력을 파싱하는 파이썬 클래스나 함수를 만듭니다.
4.  **LLM과 컴포넌트 연결**: 프롬프트, LLM, 출력 파서를 `|` 연산자로 연결하여 에이전트의 핵심 로직(Agent 객체)을 완성합니다.
5.  **AgentExecutor로 실행**: 완성된 Agent 객체와 도구들을 `AgentExecutor`에 전달하여 실행 가능한 에이전트를 만듭니다.

커스텀 에이전트 구현은 매우 강력하지만 복잡하므로, 대부분의 경우 LangChain Hub에서 제공하는 검증된 프롬프트(`hwchase17/react` 등)를 수정하거나, 기존 에이전트의 프롬프트를 약간 변경하는 것만으로도 원하는 결과를 얻을 수 있습니다.