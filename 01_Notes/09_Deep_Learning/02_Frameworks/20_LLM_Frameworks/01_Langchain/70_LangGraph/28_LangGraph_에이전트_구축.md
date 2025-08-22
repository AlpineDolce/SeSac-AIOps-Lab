<h2>LangChain 학습 가이드: LangGraph 에이전트 구축 - 자율성과 제어의 결합</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LangGraph를 활용하여 LLM 기반 에이전트를 구축하는 방법을 학습합니다. 특히 에이전트의 핵심 기능인 메모리(Memory) 통합, 중간 단계 스트리밍(Streaming), 그리고 사람의 개입(Human-in-the-loop) 기능을 LangGraph 그래프 내에 어떻게 구현하는지에 초점을 맞춥니다. 이를 통해 더욱 강력하고 유연한 대화형 에이전트를 설계하는 능력을 기릅니다.

<h2>목차</h2>

- [1. LangGraph 기반 에이전트의 특징](#1-langgraph-기반-에이전트의-특징)
  - [1.1. ReAct 에이전트의 확장](#11-react-에이전트의-확장)
  - [1.2. 상태 관리의 이점](#12-상태-관리의-이점)
- [2. 에이전트 핵심 기능 구현](#2-에이전트-핵심-기능-구현)
  - [2.1. 메모리(Memory) 통합](#21-메모리memory-통합)
    - [2.1.1. 대화 기록 관리](#211-대화-기록-관리)
    - [2.1.2. RunnableWithMessageHistory 활용](#212-runnablewithmessagehistory-활용)
  - [2.2. 중간 단계 스트리밍 (Streaming)](#22-중간-단계-스트리밍-streaming)
  - [2.3. 사람의 개입 (Human-in-the-loop)](#23-사람의-개입-human-in-the-loop)
    - [2.3.1. 사용자 승인 노드](#231-사용자-승인-노드)
    - [2.3.2. 상태 수정 및 되돌림](#232-상태-수정-및-되돌림)
- [3. LangGraph 에이전트 구축 실습](#3-langgraph-에이전트-구축-실습)
  - [3.1. 기본 에이전트 그래프 설계](#31-기본-에이전트-그래프-설계)
  - [3.2. 메모리 추가](#32-메모리-추가)
  - [3.3. 스트리밍 구현](#33-스트리밍-구현)
  - [3.4. 사람의 개입 노드 추가](#34-사람의-개입-노드-추가)

--- 

## 1. LangGraph 기반 에이전트의 특징

LangGraph는 LLM 기반 에이전트가 복잡한 작업을 수행할 때 필요한 순환(loops), 조건부 분기(conditional branching), 그리고 명시적인 상태 관리 기능을 제공합니다. 이는 기존 LangChain의 선형적인 체인(Chain)으로는 구현하기 어려웠던 자율적이고 동적인 에이전트의 설계를 가능하게 합니다.

### 1.1. ReAct 에이전트의 확장
LangGraph는 ReAct(Reason + Act) 프레임워크의 아이디어를 그래프 형태로 확장합니다. 에이전트는 특정 상태에서 추론(Reason)하고, 행동(Act)을 결정하며, 그 결과(Observation)에 따라 다음 상태로 전환하거나, 다시 추론-행동 사이클을 반복할 수 있습니다. 이는 에이전트가 목표를 달성할 때까지 스스로 계획을 수정하고 실행하는 능력을 부여합니다.

### 1.2. 상태 관리의 이점
LangGraph의 핵심은 `StateGraph`를 통한 명시적인 상태 관리입니다. 에이전트의 모든 중간 단계(대화 기록, 도구 호출 결과, 내부 추론 등)가 상태 객체에 저장되므로, 개발자는 에이전트의 작동 과정을 쉽게 추적하고 디버깅할 수 있습니다. 또한, 특정 상태로 되돌아가거나 상태를 수정하여 에이전트의 행동을 재조정하는 것도 가능해집니다.

## 2. 에이전트 핵심 기능 구현

LangGraph는 에이전트의 필수 기능을 그래프 내에 유연하게 통합할 수 있도록 설계되었습니다.

### 2.1. 메모리(Memory) 통합
에이전트가 이전 대화 내용을 기억하고 활용하는 것은 대화형 애플리케이션에서 필수적입니다. LangGraph는 `langchain_core.messages.BaseMessage`를 사용하여 대화 기록을 상태에 저장하고 관리합니다.

#### 2.1.1. 대화 기록 관리
`StateGraph`의 상태에 `messages` 필드를 추가하여 대화 기록을 누적합니다. `operator.add`를 사용하여 새로운 메시지가 기존 메시지 리스트에 이어 붙도록 설정할 수 있습니다.

```python
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
```

#### 2.1.2. RunnableWithMessageHistory 활용
LangChain의 `RunnableWithMessageHistory`를 LangGraph와 함께 사용하여 대화 기록을 외부 저장소(예: SQLite)에 영구적으로 저장하고 관리할 수 있습니다. 이는 에이전트가 세션 간에 대화 기록을 유지할 수 있도록 합니다.

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 세션 ID에 따라 대화 기록을 관리하는 함수
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# LangGraph 앱을 RunnableWithMessageHistory로 감싸기
# app_with_history = RunnableWithMessageHistory(app, get_session_history, input_messages_key="messages")
# app_with_history.invoke({"messages": [HumanMessage(content="안녕?")]}, config={"configurable": {"session_id": "test_session"}})
```

### 2.2. 중간 단계 스트리밍 (Streaming)
LangGraph는 그래프가 실행되는 동안 각 노드의 중간 결과(intermediate steps)를 실시간으로 스트리밍할 수 있는 기능을 제공합니다. 이는 사용자에게 에이전트의 추론 과정을 투명하게 보여주거나, 긴 응답이 생성되는 동안 사용자 경험을 개선하는 데 매우 유용합니다.

```python
# LangGraph 앱 실행 시 .stream() 메서드 활용
# for s in app.stream({"messages": [HumanMessage(content="질문")]}):
#     print(s)
# print("--- 스트리밍 완료 ---")
```
스트리밍을 통해 LLM의 생각(Thought), 도구 호출(Tool Call), 도구 실행 결과(Observation) 등을 실시간으로 확인할 수 있습니다.

### 2.3. 사람의 개입 (Human-in-the-loop)
복잡하거나 민감한 작업의 경우, 에이전트의 결정에 사람의 승인이나 피드백이 필요할 수 있습니다. LangGraph는 특정 노드에서 실행을 일시 중지하고 사람의 입력을 기다리는 기능을 쉽게 구현할 수 있습니다.

#### 2.3.1. 사용자 승인 노드
특정 노드에서 `input()` 함수를 사용하거나, 외부 인터페이스를 통해 사용자 입력을 기다리도록 설정할 수 있습니다. LangGraph는 이러한 일시 중지된 상태를 관리하고, 사용자 입력이 들어오면 중단된 지점부터 실행을 재개합니다.

```python
# 사람의 개입이 필요한 노드 (예시)
def human_approval_node(state: AgentState):
    print("\n--- 사람의 승인이 필요합니다 ---")
    user_input = input("계속 진행하시겠습니까? (y/n): ")
    if user_input.lower() == 'y':
        return {"messages": [HumanMessage(content="사용자가 승인했습니다.")]}
    else:
        return {"messages": [HumanMessage(content="사용자가 거부했습니다.")]}
```

#### 2.3.2. 상태 수정 및 되돌림
LangGraph는 실행 중인 그래프의 상태를 검사하고, 필요에 따라 상태를 수정하거나 특정 과거 상태로 되돌리는 기능을 제공합니다. 이는 디버깅, 오류 복구, 또는 사용자의 피드백에 따라 에이전트의 행동을 재조정하는 데 매우 유용합니다.

```python
# 특정 상태로 되돌아가기 (예시)
# app.get_graph().get_state_at_step(step_index)
# app.invoke(new_input, config={"configurable": {"thread_id": "...", "checkpoint": "..."}})
```

## 3. LangGraph 에이전트 구축 실습

여기서는 간단한 ReAct 기반 에이전트를 LangGraph로 구현하고, 위에서 설명한 기능들을 통합하는 방법을 실습합니다.

### 3.1. 기본 에이전트 그래프 설계
`27_LangGraph_핵심_개념.md`에서 다룬 기본 그래프 구조를 활용하여 LLM과 도구 호출 노드를 정의합니다.

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, List
import operator
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 도구 정의
@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    print(f"\n--- 웹 검색 실행: {query} ---")
    return f"웹 검색 결과: {query}에 대한 정보입니다."

tools = [search_web]
llm_with_tools = llm.bind_tools(tools)

# 노드 정의
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
    return {"messages": [AIMessage(content=tool_output, tool_calls=last_message.tool_calls)]}

# 조건부 엣지 함수
def decide_next_step(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls: # tool_calls가 있으면 도구 호출 노드로
        return "call_tool_node"
    else: # 없으면 종료
        return "end"

# 그래프 구축
graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm_node", call_llm_node)
graph_builder.add_node("call_tool_node", call_tool_node)

graph_builder.set_entry_point("llm_node")

graph_builder.add_conditional_edges(
    "llm_node",
    decide_next_step,
    {"call_tool_node": "call_tool_node", "end": END}
)
graph_builder.add_edge("call_tool_node", "llm_node") # 도구 호출 후 다시 LLM으로 돌아감

app = graph_builder.compile()
```

### 3.2. 메모리 추가
`RunnableWithMessageHistory`를 사용하여 대화 기록을 관리합니다.

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

app_with_history = RunnableWithMessageHistory(
    app,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="messages"
)

# 실행 예시
# print("\n--- 메모리 테스트 ---")
# config = {"configurable": {"session_id": "memory_test_session"}}
# app_with_history.invoke({"messages": [HumanMessage(content="내 이름은 홍길동이야.")]}, config=config)
# response = app_with_history.invoke({"messages": [HumanMessage(content="내 이름이 뭐였지?")]}, config=config)
# print(f"메모리 테스트 결과: {response['messages'][-1].content}")
```

### 3.3. 스트리밍 구현
`app.stream()` 메서드를 사용하여 중간 단계의 출력을 스트리밍합니다.

```python
# print("\n--- 스트리밍 테스트 ---")
# for s in app.stream({"messages": [HumanMessage(content="웹에서 'LangGraph'에 대해 검색해줘.")]}):
#     if "__end__" not in s:
#         print(s)
# print("--- 스트리밍 완료 ---")
```

### 3.4. 사람의 개입 노드 추가
특정 조건에서 사람의 승인을 기다리는 노드를 추가합니다.

```python
# 사람의 개입이 필요한 노드 (예시)
def human_approval_node(state: AgentState):
    print("\n--- 사람의 승인이 필요합니다 ---")
    user_input = input("계속 진행하시겠습니까? (y/n): ")
    if user_input.lower() == 'y':
        return {"messages": [HumanMessage(content="사용자가 승인했습니다.")]}
    else:
        return {"messages": [HumanMessage(content="사용자가 거부했습니다.")]}

# 그래프에 사람의 개입 노드 추가 (예시)
# graph_builder_human = StateGraph(AgentState)
# graph_builder_human.add_node("llm_node", call_llm_node)
# graph_builder_human.add_node("human_approval", human_approval_node)
# graph_builder_human.add_node("call_tool_node", call_tool_node)

# graph_builder_human.set_entry_point("llm_node")

# graph_builder_human.add_conditional_edges(
#     "llm_node",
#     lambda state: "human_approval" if "민감한" in state['messages'][-1].content else decide_next_step(state),
#     {"human_approval": "human_approval", "call_tool_node": "call_tool_node", "end": END}
# )
# graph_builder_human.add_edge("human_approval", "llm_node") # 승인 후 다시 LLM으로
# graph_builder_human.add_edge("call_tool_node", "llm_node")

# app_human = graph_builder_human.compile()

# 실행 예시 (민감한 내용 포함 시 사람의 개입)
# print("\n--- 사람의 개입 테스트 ---")
# try:
#     app_human.invoke({"messages": [HumanMessage(content="민감한 정보를 검색해줘.")]})
# except Exception as e:
#     print(f"예외 발생: {e}")

```