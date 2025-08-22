<h2>LangChain 학습 가이드: LCEL - 체인을 자유자재로 조립하기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LangChain의 선언적 체인 구성 방식인 LCEL(LangChain Expression Language)의 강력한 기능과 활용법을 깊이 있게 탐구하는 것을 목표로 합니다. LCEL의 핵심인 `Runnable` 프로토콜을 이해하고, 병렬 실행, 데이터 전달, 커스텀 함수 연동 등 고급 기법을 활용하여 복잡하고 효율적인 데이터 파이프라인을 구축하는 능력을 기릅니다.

<h2>목차</h2>

- [1. LCEL이란 무엇인가?](#1-lcel이란-무엇인가)
  - [1.1. 선언적 파이프라인](#11-선언적-파이프라인)
  - [1.2. LCEL의 장점](#12-lcel의-장점)
- [2. Runnable 프로토콜](#2-runnable-프로토콜)
  - [2.1. 통일된 인터페이스](#21-통일된-인터페이스)
- [3. LCEL 고급 활용 기법](#3-lcel-고급-활용-기법)
  - [3.1. 병렬 실행: `RunnableParallel`](#31-병렬-실행-runnableparallel)
  - [3.2. 데이터 추가 및 전달: `RunnablePassthrough`](#32-데이터-추가-및-전달-runnablepassthrough)
  - [3.3. 커스텀 함수 연동: `RunnableLambda`](#33-커스텀-함수-연동-runnablelambda)
- [4. 종합 실습 예제](#4-종합-실습-예제)

--- 

## 1. LCEL이란 무엇인가?

### 1.1. 선언적 파이프라인
LCEL(LangChain Expression Language)은 LangChain v0.1.0부터 핵심 기능으로 자리 잡은, 체인을 만드는 새로운 표준 방식입니다. 파이썬의 파이프 연산자 `|`를 사용하여, 데이터 처리 단계를 마치 쉘 스크립트처럼 직관적으로 연결할 수 있게 해줍니다. 이는 복잡한 워크플로우를 더 읽기 쉽고, 유지보수하기 쉬운 코드로 만들어 줍니다.

### 1.2. LCEL의 장점
LCEL로 구성된 체인은 기존 방식(Legacy Chain)에 비해 다음과 같은 강력한 장점을 기본적으로 제공합니다.
- **스트리밍 (Streaming)**: `chain.stream()`을 호출하기만 하면, LLM의 출력을 토큰 단위로 실시간 스트리밍할 수 있습니다. 사용자 경험을 크게 향상시킵니다.
- **비동기 지원 (Async Support)**: `chain.ainvoke()`를 통해 비동기 호출을 손쉽게 구현할 수 있어, 웹 서버 등에서 높은 처리량을 달성할 수 있습니다.
- **배치 처리 (Batch Processing)**: `chain.batch()`를 사용하여 여러 입력을 한 번에 처리하여 효율성을 높일 수 있습니다.
- **실행 추적**: LangSmith와 연동 시, 체인의 모든 단계와 입출력을 시각적으로 추적하고 디버깅할 수 있습니다.

## 2. Runnable 프로토콜

LCEL의 모든 구성요소(프롬프트, 모델, 파서 등)는 `Runnable`이라는 공통 프로토콜을 구현합니다. 이는 모든 컴포넌트가 `invoke`, `stream`, `batch`, `ainvoke` 등 통일된 인터페이스를 갖게 함을 의미합니다. 덕분에 어떤 컴포넌트든 자유롭게 조합하고 일관된 방식으로 실행할 수 있습니다.

## 3. LCEL 고급 활용 기법

### 3.1. 병렬 실행: `RunnableParallel`
여러 작업을 동시에 실행하고 그 결과를 하나의 딕셔너리로 묶고 싶을 때 사용합니다. 예를 들어, 동일한 주제에 대해 하나는 농담을, 다른 하나는 시를 생성하도록 병렬로 실행할 수 있습니다.

```python
from langchain_core.runnables import RunnableParallel

chain1 = ChatPromptTemplate.from_template("{topic}에 대한 농담") | model | StrOutputParser()
chain2 = ChatPromptTemplate.from_template("{topic}에 대한 시") | model | StrOutputParser()

# 두 체인을 병렬로 실행하도록 구성
combined_chain = RunnableParallel(joke=chain1, poem=chain2)

response = combined_chain.invoke({"topic": "인공지능"})

print(response)
# 예상 출력:
# {'joke': '인공지능이 가장 좋아하는 간식은? 칩(chip)!', 'poem': '차가운 실리콘 위, 논리의 꽃 피어나...'}
```

### 3.2. 데이터 추가 및 전달: `RunnablePassthrough`
체인의 중간 단계에서 초기 입력값을 잃지 않고, 다음 단계로 그대로 전달하고 싶을 때 사용합니다. RAG 파이프라인에서 검색된 문서(context)와 원본 질문(question)을 모두 LLM에게 전달해야 할 때 매우 유용합니다.

```python
from langchain_core.runnables import RunnablePassthrough

# retriever: 질문을 받아 관련 문서를 검색하는 컴포넌트라고 가정
# setup_and_retrieval 체인은 context와 question을 키로 갖는 딕셔너리를 반환
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

# 검색된 context와 원본 question을 모두 프롬프트에 전달
chain = setup_and_retrieval | prompt | model | StrOutputParser()
```

### 3.3. 커스텀 함수 연동: `RunnableLambda`
일반적인 파이썬 함수를 LCEL 체인 안에 쉽게 통합할 수 있게 해줍니다. 이를 통해 LangChain이 제공하지 않는 커스텀 로직을 체인 중간에 삽입할 수 있습니다.

```python
from langchain_core.runnables import RunnableLambda

def get_length(text):
    return len(text)

chain = model | StrOutputParser() | RunnableLambda(get_length)

response = chain.invoke("안녕하세요")
print(response) # 5
```

## 4. 종합 실습 예제

지금까지 배운 LCEL 기법들을 종합하여 RAG 파이프라인을 구성하는 예제입니다.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 1. 벡터 저장소 및 Retriever 준비 (가정)
texts = ["강아지는 귀엽다", "고양이는 독립적이다"]
vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 2. 컴포넌트 준비
model = ChatOpenAI()
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

# 3. LCEL을 사용하여 체인 구성
# RunnablePassthrough()는 딕셔너리 전체(이 경우 {"question": ...})를 전달
setup_and_retrieval = RunnableParallel(
    context=(RunnablePassthrough() | retriever),
    question=RunnablePassthrough()
)

# RunnableLambda를 사용하여 question 키의 값만 추출
question_retriever = RunnableLambda(lambda x: x["question"])

# 최종 체인
chain = {"context": retriever, "question": question_retriever} | prompt | model | output_parser

# 4. 실행
response = chain.invoke({"question": "강아지의 특징은?"})
print(response) # "귀엽다"
```
이처럼 LCEL을 활용하면 복잡한 데이터 흐름과 로직을 매우 직관적이고 유연하게 구성할 수 있으며, 이는 현대적인 LangChain 애플리케이션 개발의 핵심입니다.
