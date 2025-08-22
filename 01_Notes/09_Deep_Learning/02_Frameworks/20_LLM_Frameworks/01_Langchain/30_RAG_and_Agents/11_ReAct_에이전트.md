<h2>LangChain 학습 가이드: ReAct 에이전트 - 추론과 행동의 시너지</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 에이전트의 성능을 극대화하는 핵심 프레임워크 중 하나인 ReAct(Reason + Act)의 원리를 깊이 있게 이해하고, 직접 커스텀 도구를 만들어 ReAct 기반의 에이전트를 구축하는 것을 목표로 합니다. 이를 통해 LLM이 복잡한 문제를 해결하기 위해 어떻게 추론하고 도구를 활용하는지에 대한 통찰을 얻고, 특정 도메인에 맞는 맞춤형 에이전트를 개발하는 능력을 기릅니다.

<h2>목차</h2>

- [1. ReAct 프레임워크란?](#1-react-프레임워크란)
  - [1.1. 추론(Reason)과 행동(Act)의 결합](#11-추론reason과-행동act의-결합)
  - [1.2. ReAct 프롬프트의 구조](#12-react-프롬프트의-구조)
- [2. 커스텀 도구 만들기](#2-커스텀-도구-만들기)
  - [2.1. 왜 커스텀 도구가 필요한가?](#21-왜-커스텀-도구가-필요한가)
  - [2.2. `@tool` 데코레이터를 사용한 손쉬운 도구 생성](#22-tool-데코레이터를-사용한-손쉬운-도구-생성)
- [3. ReAct 에이전트 직접 구축하기](#3-react-에이전트-직접-구축하기)
  - [3.1. 도구 정의](#31-도구-정의)
  - [3.2. 프롬프트 설정](#32-프롬프트-설정)
  - [3.3. 에이전트 및 실행기 생성](#33-에이전트-및-실행기-생성)
  - [3.4. 실행 및 결과 분석](#34-실행-및-결과-분석)

---

## 1. ReAct 프레임워크란?

ReAct는 2022년 Google 연구팀이 발표한 논문 "ReAct: Synergizing Reasoning and Acting in Language Models"에서 제안된 프레임워크입니다. LLM이 단순히 행동(Act)만 결정하는 것이 아니라, 행동 계획을 세우고(Reason), 그 계획에 따라 행동하며, 결과를 다시 관찰하여 다음 계획을 세우는, 인간의 문제 해결 방식과 유사한 접근법입니다.

### 1.1. 추론(Reason)과 행동(Act)의 결합
ReAct의 핵심 아이디어는 LLM이 생성하는 텍스트 안에 **생각(Thought)** 과 **행동(Action)** 을 명시적으로 분리하여 포함시키는 것입니다.

- **Thought**: 현재 상황을 분석하고, 최종 목표를 달성하기 위한 다음 단계의 전략을 서술합니다.
- **Action**: 생각의 결과로, 사용할 도구와 그 도구에 전달할 입력을 지정합니다.

이러한 `Thought -> Action -> Observation` 사이클을 통해 에이전트는 더 복잡하고 여러 단계에 걸친 문제를 체계적으로 해결할 수 있으며, 중간 추론 과정을 우리가 직접 확인할 수 있어 디버깅에도 매우 용이합니다.

### 1.2. ReAct 프롬프트의 구조
ReAct 에이전트가 사용하는 프롬프트에는 LLM이 따라야 할 명확한 지침이 포함되어 있습니다. 보통 다음과 같은 내용이 들어갑니다.

- 최종 답변을 내기 전까지 `Thought`, `Action`, `Action Input`, `Observation`의 패턴을 반복하라는 지시
- 사용할 수 있는 도구의 목록과 각 도구의 설명
- 사용자의 질문과 이전 대화 기록

## 2. 커스텀 도구 만들기

### 2.1. 왜 커스텀 도구가 필요한가?
LangChain이 제공하는 기본 도구만으로는 해결할 수 없는, 우리 서비스만의 고유한 기능이 필요할 때가 많습니다. 예를 들어, 우리 회사 데이터베이스에서 고객 정보를 조회하거나, 내부 API를 호출하여 재고를 확인하는 등의 작업입니다. 이때 우리는 직접 파이썬 함수를 작성하여 에이전트가 사용할 수 있는 커스텀 도구를 만들 수 있습니다.

### 2.2. `@tool` 데코레이터를 사용한 손쉬운 도구 생성
LangChain은 일반 파이썬 함수를 `@tool` 데코레이터 하나로 손쉽게 도구로 변환할 수 있는 기능을 제공합니다. 이때 함수의 **docstring**은 LLM이 도구의 역할을 이해하는 데 사용되므로, 명확하고 상세하게 작성하는 것이 매우 중요합니다.

```python
from langchain.tools import tool

@tool
def get_word_length(word: str) -> int:
    """특정 단어의 길이를 반환합니다."""
    return len(word)

# 도구의 이름과 설명 확인
print(get_word_length.name) # get_word_length
print(get_word_length.description) # 특정 단어의 길이를 반환합니다.
print(get_word_length.args) # {'word': {'title': 'Word', 'type': 'string'}}
```

## 3. ReAct 에이전트 직접 구축하기

이제 위에서 만든 커스텀 도구를 사용하는 ReAct 에이전트를 직접 만들어 보겠습니다.

### 3.1. 도구 정의
`get_word_length` 도구와 웹 검색 도구를 함께 사용해 보겠습니다.

```python
from langchain_community.tools import DuckDuckGoSearchRun

tools = [get_word_length, DuckDuckGoSearchRun()]
```

### 3.2. 프롬프트 설정
LangChain Hub에 있는 표준 ReAct 프롬프트를 그대로 활용합니다. 이 프롬프트에는 `tools`와 `tool_names` 변수를 통해 우리가 정의한 도구 목록과 설명을 동적으로 주입할 수 있는 공간이 마련되어 있습니다.

```python
from langchain import hub

prompt = hub.pull("hwchase17/react")
```

### 3.3. 에이전트 및 실행기 생성
`create_react_agent` 함수에 LLM, 도구 리스트, 프롬프트를 전달하여 에이전트를 생성하고, `AgentExecutor`로 감싸서 실행 환경을 만듭니다.

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### 3.4. 실행 및 결과 분석

```python
query = "'LangChain'이라는 단어의 길이를 알려주고, 이 단어가 무엇인지 웹에서 검색해서 설명해줘."
response = agent_executor.invoke({"input": query})

print(f"\n최종 답변: {response['output']}")
```

`verbose=True`로 설정된 출력 로그를 보면, 에이전트가 다음과 같이 추론하고 행동하는 것을 볼 수 있습니다.

1.  **Thought**: 먼저 'LangChain'의 길이를 구하고, 그 다음에 웹 검색을 해야겠다. `get_word_length` 도구가 적합해 보인다.
2.  **Action**: `get_word_length`
3.  **Action Input**: `LangChain`
4.  **Observation**: (도구 실행 결과) `9`
5.  **Thought**: 이제 단어의 길이는 알았으니, 웹에서 'LangChain'이 무엇인지 검색해야겠다. `DuckDuckGoSearchRun` 도구를 사용해야지.
6.  **Action**: `DuckDuckGoSearchRun`
7.  **Action Input**: `LangChain`
8.  **Observation**: (웹 검색 결과) "LangChain은 LLM을 활용한 애플리케이션 개발을 돕는 프레임워크입니다..."
9.  **Thought**: 이제 모든 정보를 얻었으니, 종합해서 최종 답변을 만들자.
10. **Final Answer**: 'LangChain'이라는 단어의 길이는 9이며, 웹 검색 결과에 따르면 LangChain은 LLM을 활용한 애플리케이션 개발을 돕는 프레임워크입니다.

이처럼 ReAct 에이전트는 복잡한 문제를 논리적인 단계로 분해하고, 각 단계에 맞는 적절한 도구를 활용하여 체계적으로 해결해 나가는 강력한 능력을 보여줍니다.
