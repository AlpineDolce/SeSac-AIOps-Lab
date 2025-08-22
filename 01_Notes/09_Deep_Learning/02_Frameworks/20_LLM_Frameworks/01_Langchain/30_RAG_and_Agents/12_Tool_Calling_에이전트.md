<h2>LangChain 학습 가이드: Tool Calling 에이전트 - LLM의 직접적인 도구 활용</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 최신 LLM(Large Language Model)들이 제공하는 강력한 기능인 Tool Calling (또는 Function Calling)의 개념을 이해하고, 이를 LangChain에서 어떻게 활용하여 더욱 견고하고 효율적인 에이전트를 구축하는지 학습하는 것을 목표로 합니다. ReAct 에이전트와 비교하여 Tool Calling의 장점을 파악하고, 실제 코드를 통해 구현 능력을 기릅니다.

<h2>목차</h2>

- [1. Tool Calling / Function Calling 이란?](#1-tool-calling--function-calling-이란)
  - [1.1. LLM의 새로운 능력](#11-llm의-새로운-능력)
  - [1.2. ReAct와의 차이점](#12-react와의-차이점)
- [2. LangChain에서 Tool Calling 에이전트 구축하기](#2-langchain에서-tool-calling-에이전트-구축하기)
  - [2.1. 도구 정의: Pydantic 모델 활용](#21-도구-정의-pydantic-모델-활용)
  - [2.2. LLM에 도구 바인딩](#22-llm에-도구-바인딩)
  - [2.3. Tool Calling 에이전트 생성 및 실행](#23-tool-calling-에이전트-생성-및-실행)
- [3. Tool Calling 에이전트의 장점](#3-tool-calling-에이전트의-장점)
  - [3.1. 견고성과 안정성](#31-견고성과-안정성)
  - [3.2. 효율적인 프롬프트 관리](#32-효율적인-프롬프트-관리)
  - [3.3. 복잡한 도구 인자 처리](#33-복잡한-도구-인자-처리)

--- 

## 1. Tool Calling / Function Calling 이란?

Tool Calling (또는 Function Calling)은 LLM이 사용자의 요청을 이해하고, 특정 작업을 수행하기 위해 외부 도구(함수)를 호출해야 한다고 판단했을 때, 해당 도구의 이름과 필요한 인자(arguments)를 **구조화된 형식(예: JSON)**으로 출력하는 능력입니다. 이는 LLM이 단순히 텍스트를 생성하는 것을 넘어, 외부 시스템과 직접적으로 상호작용할 수 있는 강력한 인터페이스를 제공합니다.

### 1.1. LLM의 새로운 능력
과거에는 LLM이 도구를 사용하려면 ReAct와 같은 프롬프트 엔지니어링 기법을 통해 LLM이 '생각(Thought)'하고 '행동(Action)'을 텍스트로 출력하도록 유도해야 했습니다. 하지만 Tool Calling 기능이 내장된 LLM은 사용자의 의도를 파악하여 직접적으로 도구 호출을 위한 구조화된 데이터를 생성합니다. 이는 LLM이 도구 사용에 대한 '의도'를 더 명확하게 표현할 수 있게 합니다.

예를 들어, "오늘 날씨 어때?"라는 질문에 대해 LLM은 "날씨 정보를 가져오는 `get_weather` 함수를 호출해야겠군. 지역은 '서울'로 해야겠다."와 같은 내부 추론을 거쳐, 최종적으로 `{"tool_name": "get_weather", "arguments": {"location": "서울"}}`과 같은 JSON 객체를 출력합니다. 이 JSON 객체를 파싱하여 실제 도구를 실행하는 것은 외부 시스템(LangChain 에이전트 실행기)의 역할입니다.

### 1.2. ReAct와의 차이점
ReAct와 Tool Calling은 모두 LLM이 도구를 사용하도록 하는 프레임워크이지만, 작동 방식에 큰 차이가 있습니다.

| 특징         | ReAct 에이전트                                     | Tool Calling 에이전트                               
| :----------- | :------------------------------------------------- | :-------------------------------------------------- 
| **LLM 출력** | `Thought`, `Action`, `Action Input` 등 텍스트 기반 | `tool_name`, `arguments` 등 구조화된 JSON 객체      
| **파싱 복잡성**| LLM의 텍스트 출력을 파싱해야 하므로 오류 발생 가능성 높음 | 구조화된 JSON이므로 파싱이 간단하고 오류 발생 가능성 낮음 
| **견고성**   | LLM의 텍스트 생성에 의존하여 불안정할 수 있음      | LLM이 직접 구조화된 호출을 생성하여 더 견고함      
| **프롬프트** | LLM에게 추론 및 행동 패턴을 지시하는 복잡한 프롬프트 필요 | 도구의 스키마만 제공하면 LLM이 알아서 호출 패턴 학습 

Tool Calling은 LLM이 도구 사용에 대한 '의도'를 더 명확하고 안정적으로 표현할 수 있게 하여, 에이전트의 전반적인 견고성과 개발 편의성을 크게 향상시킵니다.

## 2. LangChain에서 Tool Calling 에이전트 구축하기

LangChain은 LLM의 Tool Calling 기능을 추상화하여 개발자가 쉽게 에이전트를 구축할 수 있도록 지원합니다.

### 2.1. 도구 정의: Pydantic 모델 활용

Tool Calling 에이전트에서는 도구의 인자(arguments)를 명확하게 정의하는 것이 중요합니다. LangChain은 이를 위해 Pydantic 모델을 활용하는 것을 권장합니다. Pydantic 모델은 도구의 입력 스키마를 정의하고, LLM이 생성한 인자가 이 스키마에 맞는지 자동으로 검증해 줍니다.

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 도구의 입력 스키마를 정의하는 Pydantic 모델
class GetWeatherInput(BaseModel):
    """날씨 정보를 가져오는 도구의 입력 스키마"""
    location: str = Field(description="날씨 정보를 가져올 도시 이름")
    unit: str = Field(default="celsius", description="온도 단위 (celsius 또는 fahrenheit)")

@tool(args_schema=GetWeatherInput)
def get_weather(location: str, unit: str = "celsius") -> str:
    """특정 도시의 현재 날씨 정보를 반환합니다."""
    # 실제 날씨 API 호출 로직 (예시)
    if location == "서울":
        return f"서울의 현재 날씨는 맑고, 온도는 25 {unit}입니다."
    elif location == "부산":
        return f"부산의 현재 날씨는 흐리고, 온도는 22 {unit}입니다."
    else:
        return "해당 도시의 날씨 정보를 찾을 수 없습니다."

# 일반 파이썬 함수도 @tool 데코레이터로 도구화 가능 (인자 스키마는 자동으로 추론됨)
@tool
def get_word_length(word: str) -> int:
    """특정 단어의 길이를 반환합니다."""
    return len(word)
```

### 2.2. LLM에 도구 바인딩

Tool Calling 기능을 지원하는 LLM(예: `ChatOpenAI`, `ChatGoogleGenerativeAI`)에 정의한 도구들을 `bind_tools()` 메서드를 사용하여 바인딩합니다. 이렇게 바인딩된 LLM은 사용자의 질문에 따라 적절한 도구를 호출할 수 있는 능력을 갖게 됩니다.

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [get_weather, get_word_length]

# LLM에 도구 바인딩
llm_with_tools = llm.bind_tools(tools)
```

### 2.3. Tool Calling 에이전트 생성 및 실행

LangChain은 Tool Calling 기능을 활용하는 에이전트를 쉽게 생성할 수 있는 `create_tool_calling_agent` 함수를 제공합니다. 이 함수는 LLM, 도구 리스트, 그리고 프롬프트를 인자로 받아 에이전트를 생성합니다.

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Tool Calling 에이전트에 적합한 프롬프트 템플릿
# {input}과 {agent_scratchpad}는 필수
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 유용한 AI 비서입니다."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), # 에이전트의 중간 생각과 도구 호출 기록
])

# 에이전트 생성
agent = create_tool_calling_agent(llm_with_tools, tools, prompt)

# 에이전트 실행기 생성
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

이제 에이전트를 실행하여 Tool Calling이 어떻게 작동하는지 확인해 봅시다.

```python
# 예시 1: 단어 길이 도구 호출
query1 = "'LangChain'이라는 단어의 길이를 알려줘."
response1 = agent_executor.invoke({"input": query1})
print(f"\n최종 답변 1: {response1['output']}")

# 예시 2: 날씨 도구 호출
query2 = "서울의 오늘 날씨는 어때? 온도는 화씨로 알려줘."
response2 = agent_executor.invoke({"input": query2})
print(f"\n최종 답변 2: {response2['output']}")

# 예시 3: 도구 호출 없이 직접 답변
query3 = "안녕하세요, 당신은 누구인가요?"
response3 = agent_executor.invoke({"input": query3})
print(f"\n최종 답변 3: {response3['output']}")
```

`verbose=True`로 설정된 출력 로그를 보면, LLM이 직접 `tool_calls`를 생성하고, 에이전트 실행기가 이를 받아 도구를 실행한 후, 그 결과를 다시 LLM에게 전달하여 최종 답변을 생성하는 과정을 확인할 수 있습니다.

## 3. Tool Calling 에이전트의 장점

### 3.1. 견고성과 안정성
Tool Calling은 LLM이 도구 호출을 위한 구조화된 JSON 객체를 직접 생성하기 때문에, ReAct처럼 LLM의 텍스트 출력을 파싱하는 과정에서 발생할 수 있는 오류가 현저히 줄어듭니다. 이는 에이전트의 전반적인 안정성과 신뢰성을 크게 향상시킵니다.

### 3.2. 효율적인 프롬프트 관리
개발자는 LLM에게 도구 사용 패턴을 일일이 지시하는 복잡한 프롬프트를 작성할 필요가 없습니다. 단순히 도구의 스키마(Pydantic 모델)만 제공하면, LLM이 사용자의 질문과 도구 스키마를 기반으로 어떤 도구를 호출해야 할지, 어떤 인자를 전달해야 할지 스스로 판단합니다. 이는 프롬프트 엔지니어링의 부담을 줄여줍니다.

### 3.3. 복잡한 도구 인자 처리
Pydantic 모델을 사용하여 도구의 입력 스키마를 명확하게 정의할 수 있으므로, LLM은 여러 개의 인자를 가진 복잡한 도구도 정확하게 호출할 수 있습니다. 또한, Pydantic의 유효성 검사 기능을 통해 LLM이 잘못된 형식의 인자를 생성했을 때 이를 감지하고 처리할 수 있습니다.

Tool Calling은 LLM 기반 에이전트 개발의 새로운 표준으로 자리매김하고 있으며, LangChain을 통해 이 강력한 기능을 쉽게 활용할 수 있습니다.