<h2>LangChain 학습 가이드: LangGraph 핵심 개념 - 동적 워크플로우의 설계</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LangGraph의 핵심 개념을 이해하고, 이를 활용하여 LLM 기반 애플리케이션의 복잡하고 동적인 워크플로우를 설계하는 방법을 학습하는 것을 목표로 합니다. LangGraph의 주요 구성 요소인 StateGraph, 노드, 엣지, 조건부 분기 등을 이해하고 간단한 그래프를 직접 구축해봅니다.

<h2>목차</h2>

- [1. LangGraph란?](#1-langgraph란)
  - [1.1. 왜 LangGraph가 필요한가?](#11-왜-langgraph가-필요한가)
  - [1.2. LangChain Expression Language (LCEL)와의 관계](#12-langchain-expression-language-lcel와의-관계)
- [2. LangGraph의 핵심 구성 요소](#2-langgraph의-핵심-구성-요소)
  - [2.1. StateGraph: 그래프의 상태 정의](#21-stategraph-그래프의-상태-정의)
  - [2.2. 노드 (Nodes): 작업 단위](#22-노드-nodes-작업-단위)
  - [2.3. 엣지 (Edges): 흐름 제어](#23-엣지-edges-흐름-제어)
    - [2.3.1. 일반 엣지 (Normal Edges)](#231-일반-엣지-normal-edges)
    - [2.3.2. 조건부 엣지 (Conditional Edges)](#232-조건부-엣지-conditional-edges)
  - [2.4. 중간 단계 스트리밍 (Streaming Intermediate Steps)](#24-중간-단계-스트리밍-streaming-intermediate-steps)
- [3. 간단한 LangGraph 구축하기](#3-간단한-langgraph-구축하기)
  - [3.1. 상태 정의](#31-상태-정의)
  - [3.2. 노드 정의](#32-노드-정의)
  - [3.3. 그래프 구성 및 컴파일](#33-그래프-구성-및-컴파일)
  - [3.4. 그래프 실행](#34-그래프-실행)

--- 

## 1. LangGraph란?

LangGraph는 LangChain의 확장 라이브러리로, LLM 기반 애플리케이션에서 **순환(loops)**, **분기(branching)**, **다중 에이전트(multi-agent)** 와 같은 복잡하고 동적인 워크플로우를 구축할 수 있도록 돕는 프레임워크입니다. 이는 유한 상태 머신(Finite State Machine) 개념을 기반으로 하며, 각 단계의 상태를 명시적으로 관리하면서 LLM의 추론 및 행동 과정을 제어합니다.

### 1.1. 왜 LangGraph가 필요한가?
기존 LangChain의 체인(Chain)은 주로 선형적인(linear) 흐름을 처리하는 데 적합했습니다. 하지만 실제 LLM 애플리케이션은 다음과 같은 복잡한 시나리오를 요구합니다.
-   **반복적인 추론**: 에이전트가 목표를 달성할 때까지 여러 도구를 반복적으로 사용해야 하는 경우.
-   **조건부 분기**: LLM의 판단에 따라 워크플로우의 다음 단계가 동적으로 변경되어야 하는 경우.
-   **사람의 개입**: 특정 단계에서 사람의 피드백이나 승인이 필요한 경우.
-   **멀티 에이전트 협업**: 여러 LLM 기반 에이전트가 서로 소통하며 복잡한 문제를 해결하는 경우.

LangGraph는 이러한 복잡한 흐름을 명시적인 그래프 구조로 정의하고 실행할 수 있게 하여, LLM 애플리케이션의 자율성과 견고성을 크게 향상시킵니다.

### 1.2. LangChain Expression Language (LCEL)와의 관계
LangGraph는 LCEL(LangChain Expression Language) 위에 구축됩니다. 즉, LangGraph의 노드 내에서 실행되는 모든 작업은 LCEL 표현식으로 정의될 수 있습니다. 이는 LangGraph가 LCEL의 유연성과 확장성을 그대로 활용하면서, LCEL만으로는 어려웠던 순환 및 분기 로직을 그래프 형태로 시각화하고 제어할 수 있게 합니다.

## 2. LangGraph의 핵심 구성 요소

LangGraph는 주로 `StateGraph`, `노드(Nodes)`, `엣지(Edges)` 세 가지 핵심 요소로 구성됩니다.

### 2.1. StateGraph: 그래프의 상태 정의
`StateGraph`는 그래프의 현재 상태를 정의하는 역할을 합니다. 각 노드가 실행될 때마다 이 상태가 업데이트되며, 다음 노드의 실행에 영향을 미칩니다. 상태는 일반적으로 Python 딕셔너리나 Pydantic 모델로 정의됩니다.

```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

# 그래프의 상태를 정의합니다.
class AgentState(TypedDict):
    # 대화 기록을 저장합니다.
    messages: Annotated[List[BaseMessage], operator.add]
    # 다음으로 실행할 노드의 이름을 저장합니다.
    next: str
```
여기서 `Annotated[List[BaseMessage], operator.add]`는 `messages` 리스트에 새로운 메시지가 추가될 때 `operator.add` 함수를 사용하여 기존 리스트에 이어 붙이도록 지시합니다.

### 2.2. 노드 (Nodes): 작업 단위
노드는 그래프 내에서 특정 작업을 수행하는 단위입니다. 각 노드는 파이썬 함수나 LCEL 표현식으로 정의될 수 있으며, 상태를 입력으로 받아 상태를 업데이트하거나 새로운 값을 반환합니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# LLM 호출을 수행하는 노드
def call_llm(state: AgentState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# 도구 호출을 수행하는 노드 (예시)
def call_tool(state: AgentState):
    # 실제 도구 호출 로직
    tool_output = "도구 실행 결과입니다."
    return {"messages": [HumanMessage(content=tool_output)]}
```

### 2.3. 엣지 (Edges): 흐름 제어
엣지는 노드 간의 전환을 정의하여 그래프의 실행 흐름을 제어합니다.

#### 2.3.1. 일반 엣지 (Normal Edges)
특정 노드에서 다른 특정 노드로 항상 전환되는 엣지입니다.

```python
# graph.add_edge("노드_A", "노드_B")
```

#### 2.3.2. 조건부 엣지 (Conditional Edges)
가장 강력한 엣지 유형으로, 특정 노드의 출력(또는 현재 상태)에 따라 다음으로 실행될 노드가 동적으로 결정됩니다. 조건부 엣지는 일반적으로 파이썬 함수를 사용하여 다음 노드의 이름을 반환합니다.

```python
# 조건부 엣지 함수
def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    # LLM의 마지막 메시지가 도구 호출을 포함하는지 여부에 따라 분기
    if "tool_calls" in last_message.additional_kwargs:
        return "call_tool"
    else:
        return "end"

# graph.add_conditional_edges("llm_node", should_continue, {"call_tool": "tool_node", "end": END})
```
여기서 `END`는 LangGraph에서 제공하는 특별한 노드로, 그래프 실행을 종료시킵니다.

### 2.4. 중간 단계 스트리밍 (Streaming Intermediate Steps)
LangGraph는 그래프가 실행되는 동안 각 노드의 중간 결과(intermediate steps)를 실시간으로 스트리밍할 수 있는 기능을 제공합니다. 이는 복잡한 에이전트의 추론 과정을 사용자에게 투명하게 보여주거나, 디버깅에 매우 유용합니다.

```python
# 그래프 실행 시 스트리밍 옵션 활용
# for s in app.stream({"messages": [HumanMessage(content="질문")]}):
#     print(s)
```

## 3. 간단한 LangGraph 구축하기

이제 위에서 배운 개념들을 활용하여 간단한 LangGraph를 구축해 보겠습니다.

### 3.1. 상태 정의
```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
```

### 3.2. 노드 정의
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

llm = ChatOpenAI(model="gpt-4o", temperature=0)

@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    print(f"\n--- 웹 검색 실행: {query} ---")
    return f"웹 검색 결과: {query}에 대한 정보입니다."

tools = [search_web]
llm_with_tools = llm.bind_tools(tools)

def call_llm_node(state: AgentState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tool_node(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    tool_output = ""
    for tool_call in last_message.tool_calls:
        if tool_call.name == "search_web":
            tool_output += search_web.invoke(tool_call.args['query'])
    return {"messages": [HumanMessage(content=tool_output)]}

def decide_next_step(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if "tool_calls" in last_message.additional_kwargs:
        return "call_tool_node"
    else:
        return "end"
```

### 3.3. 그래프 구성 및 컴파일
```python
from langgraph.graph import StateGraph, END

graph_builder = StateGraph(AgentState)

# 노드 추가
graph_builder.add_node("llm_node", call_llm_node)
graph_builder.add_node("call_tool_node", call_tool_node)

# 시작점 설정
graph_builder.set_entry_point("llm_node")

# 엣지 추가
graph_builder.add_conditional_edges(
    "llm_node", # 현재 노드
    decide_next_step,
    {
        "call_tool_node": "call_tool_node", # 조건부 결과가 "call_tool_node"이면 해당 노드로 이동
        "end": END # 조건부 결과가 "end"이면 그래프 종료
    }
)
graph_builder.add_edge("call_tool_node", "llm_node") # 도구 호출 후 다시 LLM으로 돌아감

# 그래프 컴파일
app = graph_builder.compile()
```

### 3.4. 그래프 실행
```python
from langchain_core.messages import HumanMessage

# 예시 1: 도구 호출이 필요한 질문
result1 = app.invoke({"messages": [HumanMessage(content="웹에서 'LangGraph'에 대해 검색해줘.")]})
print(f"최종 결과 1: {result1['messages'][-1].content}")

# 예시 2: 도구 호출이 필요 없는 질문
result2 = app.invoke({"messages": [HumanMessage(content="안녕하세요.")]})
print(f"최종 결과 2: {result2['messages'][-1].content}")
```
```python
# 스트리밍 예시
# for s in app.stream({"messages": [HumanMessage(content="웹에서 'LangGraph'에 대해 검색해줘.")]}):
#     print(s)
```
```python
# 그래프 시각화 (Graphviz 설치 필요)
# from IPython.display import Image, display
# Image(app.get_graph().draw_png())
```