<h2>LangChain 학습 가이드: 스트리밍(Streaming) - 실시간 응답 구현하기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LLM 애플리케이션의 사용자 경험(UX)을 극대화하는 핵심 기술인 '스트리밍(Streaming)'의 원리를 이해하고, LCEL을 사용하여 LLM의 응답을 실시간으로 처리하는 방법을 학습하는 것을 목표로 합니다. 이를 통해 사용자가 답변을 기다리는 시간을 최소화하고, 마치 사람과 대화하는 듯한 동적인 인터페이스를 구현하는 능력을 기릅니다.

<h2>목차</h2>

- [1. 왜 스트리밍이 중요한가?](#1-왜-스트리밍이-중요한가)
  - [1.1. 사용자 경험의 차이](#11-사용자-경험의-차이)
- [2. LCEL과 스트리밍](#2-lcel과-스트리밍)
  - [2.1. `.stream()` 메소드](#21-stream-메소드)
  - [2.2. 스트리밍의 기본 원리](#22-스트리밍의-기본-원리)
- [3. 스트리밍 구현 실습](#3-스트리밍-구현-실습)
  - [3.1. 기본 체인 스트리밍](#31-기본-체인-스트리밍)
  - [3.2. RAG 파이프라인 스트리밍](#32-rag-파이프라인-스트리밍)

--- 

## 1. 왜 스트리밍이 중요한가?

### 1.1. 사용자 경험의 차이
LLM이 복잡한 질문에 대한 긴 답변을 생성하는 데는 수 초에서 수십 초가 걸릴 수 있습니다. 스트리밍이 없다면, 사용자는 모든 답변이 생성될 때까지 아무런 피드백 없이 로딩 화면만 바라봐야 합니다. 이는 사용자를 지루하게 만들고 서비스 이탈률을 높이는 원인이 됩니다.

**스트리밍**은 LLM이 생성하는 텍스트(토큰)를 완성될 때까지 기다리는 대신, 생성되는 즉시 사용자에게 전달하는 기술입니다. ChatGPT가 답변을 한 글자씩 타이핑하듯 보여주는 것이 바로 스트리밍의 대표적인 예입니다. 이를 통해 사용자는 시스템이 작동하고 있음을 인지하고, 답변의 일부를 미리 읽으며 지루함을 덜 느끼게 됩니다.

## 2. LCEL과 스트리밍

LCEL로 구성된 체인은 스트리밍 기능을 **기본적으로 지원**합니다. 별도의 복잡한 설정 없이, 체인의 `.invoke()` 메소드 대신 `.stream()` 메소드를 호출하기만 하면 됩니다.

### 2.1. `.stream()` 메소드
- **역할**: 체인을 스트리밍 모드로 실행합니다.
- **반환값**: 이터레이터(iterator)를 반환합니다. `for` 루프를 사용하여 이 이터레이터를 순회하면, 체인의 각 단계에서 생성되는 출력 조각(chunk)들을 실시간으로 얻을 수 있습니다.

### 2.2. 스트리밍의 기본 원리
LCEL 체인(`prompt | model | parser`)에서 `.stream()`을 호출하면 다음과 같이 동작합니다.
1.  프롬프트는 입력을 받아 즉시 전체 프롬프트를 생성하여 모델에 전달합니다.
2.  모델은 응답을 토큰 단위로 생성하기 시작하며, 생성될 때마다 토큰 조각을 출력 파서로 보냅니다.
3.  출력 파서 또한 이 조각들을 받아 처리하고, 그 결과를 즉시 반환합니다.

결과적으로, 최종 사용자는 전체 과정이 끝날 때까지 기다릴 필요 없이, 모델이 생성하는 텍스트를 실시간으로 받아볼 수 있습니다.

## 3. 스트리밍 구현 실습

### 3.1. 기본 체인 스트리밍

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# 1. 체인 구성
prompt = ChatPromptTemplate.from_template("{country}의 역사에 대해 세 문장으로 요약해줘.")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# 2. .stream() 메소드 호출 및 결과 처리
query = {"country": "대한민국"}

print(f"질문: {query['country']}")
print("답변 (스트리밍): ", end="")

for chunk in chain.stream(query):
    print(chunk, end="", flush=True) # flush=True로 버퍼를 즉시 비움

print("\n\n스트리밍 완료.")
```
위 코드를 실행하면, 답변이 한 번에 나타나는 것이 아니라, 마치 타이핑되듯이 순차적으로 콘솔에 출력되는 것을 확인할 수 있습니다.

### 3.2. RAG 파이프라인 스트리밍
더 복잡한 RAG 파이프라인에서도 LCEL을 사용했다면 스트리밍은 동일하게 작동합니다. 중간의 Retriever 단계는 스트리밍이 불가능하므로 문서를 한 번에 검색하지만, 그 결과를 받은 최종 LLM 단계부터는 스트리밍이 적용됩니다.

```python
# RAG 체인 구성 (이전 예제와 유사)
# chain = setup_and_retrieval | prompt | model | output_parser

# 스트리밍 실행
for chunk in chain.stream({"question": "강아지의 특징은?"}):
    print(chunk, end="", flush=True)
```

이처럼 LCEL은 복잡한 체인에서도 스트리밍을 매우 간단하게 구현할 수 있도록 지원하여, 개발자가 손쉽게 반응성 높은 애플리케이션을 만들 수 있게 해줍니다.
