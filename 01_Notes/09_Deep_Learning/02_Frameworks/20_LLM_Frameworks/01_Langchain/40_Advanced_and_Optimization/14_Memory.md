<h2>LangChain 학습 가이드: 메모리(Memory) - 대화의 맥락을 기억하는 법</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LLM이 이전 대화 내용을 기억하게 만들어, 사용자와의 연속적인 상호작용이 가능한 챗봇과 같은 애플리케이션을 구축하는 '메모리(Memory)' 기능에 대해 학습하는 것을 목표로 합니다. 다양한 메모리 유형의 특징과 장단점을 이해하고, 이를 체인에 통합하여 맥락을 유지하는 대화형 AI를 구현하는 방법을 익힙니다.

<h2>목차</h2>

- [1. 왜 메모리가 필요한가? (Stateless의 한계)](#1-왜-메모리가-필요한가-stateless의-한계)
- [2. LangChain의 메모리 컴포넌트](#2-langchain의-메모리-컴포넌트)
- [3. 주요 메모리 유형](#3-주요-메모리-유형)
  - [3.1. `ConversationBufferMemory`](#31-conversationbuffermemory)
  - [3.2. `ConversationBufferWindowMemory`](#32-conversationbufferwindowmemory)
  - [3.3. `ConversationSummaryMemory`](#33-conversationsummarymemory)
- [4. 체인에 메모리 통합하기](#4-체인에-메모리-통합하기)
  - [4.1. `ConversationChain` 활용](#41-conversationchain-활용)
  - [4.2. 코드 예제](#42-코드-예제)

--- 

## 1. 왜 메모리가 필요한가? (Stateless의 한계)

LLM과 Chat Model은 기본적으로 **상태가 없습니다(Stateless)**. 즉, 이전의 요청을 기억하지 못합니다. 매번의 `invoke` 호출은 완전히 독립적인 사건으로 처리됩니다. 따라서 다음과 같은 대화는 불가능합니다.

> **사용자**: 안녕, 내 이름은 철수야.
> **AI**: 안녕하세요, 철수님! 만나서 반갑습니다.
> **사용자**: 내 이름이 뭐라고 했지?
> **AI**: 죄송하지만, 당신의 이름을 알지 못합니다.

이러한 한계를 극복하고 자연스러운 대화를 이어가기 위해, 이전 대화 기록을 저장하고 다음 요청 시 LLM에게 함께 전달해주는 기능이 필요한데, 이것이 바로 **메모리**의 역할입니다.

## 2. LangChain의 메모리 컴포넌트

LangChain의 메모리 컴포넌트는 대화 기록을 읽고 쓰는 역할을 담당합니다. 체인이 실행될 때, 메모리는 이전 대화 기록을 컨텍스트에 추가하여 프롬프트를 만들고, 체인 실행이 끝난 후에는 새로운 대화(사용자 입력, AI 응답)를 기록하여 다음을 위해 저장합니다.

## 3. 주요 메모리 유형

LangChain은 다양한 메모리 전략을 제공합니다. 대표적인 세 가지는 다음과 같습니다.

### 3.1. `ConversationBufferMemory`
- **작동 방식**: 모든 대화 기록을 있는 그대로 전부 저장합니다.
- **장점**: 전체 대화의 맥락을 완벽하게 보존합니다.
- **단점**: 대화가 길어지면 프롬프트가 너무 길어져 토큰 제한을 초과할 수 있고, API 비용이 증가합니다.
- **사용 사례**: 짧고 간단한 대화에 적합합니다.

### 3.2. `ConversationBufferWindowMemory`
- **작동 방식**: 가장 최근의 `k`개 대화만 저장합니다.
- **장점**: 토큰 길이를 일정하게 유지하여 토큰 제한과 비용 문제를 해결할 수 있습니다.
- **단점**: `k`개 이전의 중요한 대화 내용을 잊어버릴 수 있습니다.
- **사용 사례**: 최근 대화의 흐름이 중요한 일반적인 챗봇에 적합합니다. (예: `k=5`)

### 3.3. `ConversationSummaryMemory`
- **작동 방식**: 전체 대화 기록을 저장하는 대신, 대화가 진행됨에 따라 LLM을 사용하여 대화 내용을 요약하고, 이 요약본을 저장합니다.
- **장점**: 긴 대화의 핵심 내용을 압축하여 보존하므로 토큰 사용량이 효율적입니다.
- **단점**: 요약 과정에서 LLM을 추가로 호출하므로 약간의 지연과 추가 비용이 발생할 수 있으며, 세부 정보가 손실될 수 있습니다.
- **사용 사례**: 긴 시간 동안 진행되는 상담이나 회의록 요약과 같은 작업에 적합합니다.

## 4. 체인에 메모리 통합하기

### 4.1. `ConversationChain` 활용
`ConversationChain`은 메모리 기능을 편리하게 사용할 수 있도록 미리 구성된 체인입니다. `LLMChain`과 유사하지만, `memory` 객체를 인자로 받아 대화 기록을 자동으로 관리해줍니다.

### 4.2. 코드 예제

`ConversationBufferWindowMemory`를 사용하는 예제입니다.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()

# 1. 메모리 객체 생성 (최근 3개의 대화만 기억)
memory = ConversationBufferWindowMemory(k=3)

# 2. LLM 모델 준비
llm = ChatOpenAI(temperature=0)

# 3. ConversationChain 생성
# verbose=True로 프롬프트의 변화를 확인
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 4. 대화 진행
print("첫 번째 대화")
conversation.invoke("안녕, 내 이름은 영희야.")

print("\n두 번째 대화")
conversation.invoke("나는 서울에 살아.")

print("\n세 번째 대화")
response = conversation.invoke("내 이름이 뭐라고 했지?")

print(f"\nAI 응답: {response['response']}")
# 예상 출력:
# AI 응답: 당신의 이름은 영희입니다.
```

`verbose=True`로 설정된 로그를 보면, `invoke`가 호출될 때마다 `memory`가 `history` 변수에 이전 대화 기록을 채워넣어 프롬프트에 전달하는 것을 확인할 수 있습니다. 이를 통해 LLM은 이전 대화의 맥락을 파악하고 "당신의 이름은 영희입니다."라고 정확하게 답변할 수 있게 됩니다.
```