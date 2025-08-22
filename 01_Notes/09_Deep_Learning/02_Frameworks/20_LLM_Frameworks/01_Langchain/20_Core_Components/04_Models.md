<h2>LangChain 학습 가이드: 모델(Models) 완전 정복</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LangChain에서 사용되는 세 가지 주요 모델 유형인 LLM, Chat Model, Text Embedding Model의 차이점을 명확히 이해하고, 각각의 모델을 언제 어떻게 사용해야 하는지 학습하는 것을 목표로 합니다. 각 모델의 입출력 형태와 특징을 파악하여 다양한 시나리오에 맞는 최적의 모델을 선택하고 활용하는 능력을 기릅니다.

<h2>목차</h2>

- [1. LangChain의 모델 추상화](#1-langchain의-모델-추상화)
- [2. LLM: 순수한 텍스트 입출력](#2-llm-순수한-텍스트-입출력)
  - [2.1. 특징 및 사용 사례](#21-특징-및-사용-사례)
  - [2.2. 코드 예제](#22-코드-예제)
- [3. Chat Model: 대화형 인터페이스](#3-chat-model-대화형-인터페이스)
  - [3.1. 특징 및 사용 사례](#31-특징-및-사용-사례)
  - [3.2. 메시지 타입의 이해](#32-메시지-타입의-이해)
  - [3.3. 코드 예제](#33-코드-예제)
- [4. Text Embedding Model: 텍스트의 벡터화](#4-text-embedding-model-텍스트의-벡터화)
  - [4.1. 특징 및 사용 사례](#41-특징-및-사용-사례)
  - [4.2. 코드 예제](#42-코드-예제)

---

## 1. LangChain의 모델 추상화

LangChain의 가장 큰 장점 중 하나는 다양한 모델 제공자(OpenAI, Google, Anthropic 등)의 API를 일관된 인터페이스로 추상화했다는 점입니다. 이를 통해 개발자는 기본 모델을 교체하더라도 애플리케이션의 나머지 코드를 거의 수정할 필요가 없어 유연성과 확장성이 크게 향상됩니다. LangChain은 크게 세 가지 유형의 모델 인터페이스를 제공합니다.

## 2. LLM: 순수한 텍스트 입출력

### 2.1. 특징 및 사용 사례
- **입력**: 단순한 문자열 (string)
- **출력**: 단순한 문자열 (string)
- **설명**: 가장 기본적인 모델 인터페이스입니다. 주어진 텍스트에 이어질 내용을 예측하거나, 텍스트 요약, 번역 등 간단한 텍스트 생성 작업에 적합합니다. 챗봇과 같은 대화형 시나리오보다는 단일 작업을 처리하는 데 주로 사용됩니다.
- **대표 모델**: OpenAI의 `gpt-3.5-turbo-instruct`

### 2.2. 코드 예제
```python
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# LLM 모델 초기화
llm = OpenAI()

# 문자열을 입력으로 제공
text = "대한민국의 수도는?"
response = llm.invoke(text)

print(response)
# 예상 출력: 
# 서울입니다.
```

## 3. Chat Model: 대화형 인터페이스

### 3.1. 특징 및 사용 사례
- **입력**: 메시지 객체의 리스트 (list of message objects)
- **출력**: 단일 메시지 객체 (single message object)
- **설명**: 최신 LLM들은 대부분 대화 형식에 최적화되어 있습니다. Chat Model 인터페이스는 이러한 모델들과 상호작용하기 위해 설계되었으며, 역할(role)을 가진 메시지들을 통해 대화의 맥락을 모델에 전달할 수 있습니다. 챗봇, 역할극, 멀티턴(multi-turn) 대화 등 복잡한 상호작용에 필수적입니다.
- **대표 모델**: OpenAI의 `gpt-4`, `gpt-3.5-turbo`

### 3.2. 메시지 타입의 이해
Chat Model은 다음과 같은 메시지 타입을 사용하여 대화의 맥락을 구성합니다.
- `SystemMessage`: 대화 전체에 걸쳐 AI의 역할이나 지켜야 할 규칙을 설정합니다. (예: "너는 친절한 AI 비서야.")
- `HumanMessage`: 사용자의 입력을 나타냅니다.
- `AIMessage`: AI의 응답을 나타냅니다. 이전 대화 기록을 전달할 때 사용됩니다.

### 3.3. 코드 예제
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Chat Model 초기화
chat = ChatOpenAI()

# 메시지 리스트를 입력으로 제공
messages = [
    SystemMessage(content="당신은 여행 전문가입니다."),
    HumanMessage(content="대한민국의 수도는 어디인가요? 그 곳의 대표적인 관광지 2곳을 추천해주세요."),
]

response = chat.invoke(messages)

print(response.content)
# 예상 출력:
# 대한민국의 수도는 서울입니다. 
# 서울의 대표적인 관광지로는 다음과 같은 곳들을 추천해 드릴 수 있습니다:
# 1. 경복궁: 조선 시대의 웅장한 궁궐로, 한국의 역사와 아름다움을 느낄 수 있는 곳입니다.
# 2. 명동: 쇼핑과 먹거리가 가득한 활기찬 거리로, 현대적인 서울을 경험할 수 있습니다.
```

## 4. Text Embedding Model: 텍스트의 벡터화

### 4.1. 특징 및 사용 사례
- **입력**: 텍스트 또는 텍스트 리스트 (string or list of strings)
- **출력**: 숫자 리스트(벡터) 또는 벡터 리스트 (list of floats or list of list of floats)
- **설명**: 텍스트의 의미를 고차원 공간의 벡터(숫자 배열)로 변환하는 모델입니다. 이 벡터들은 단어, 문장, 문서 간의 의미적 유사성을 계산하는 데 사용됩니다. RAG(Retrieval-Augmented Generation), 시맨틱 검색, 텍스트 군집화, 이상 탐지 등 다양한 작업의 기반 기술입니다.
- **대표 모델**: OpenAI의 `text-embedding-3-small`

### 4.2. 코드 예제
```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Embedding Model 초기화
embeddings = OpenAIEmbeddings()

# 단일 문서 임베딩
text = "안녕하세요, LangChain입니다."
text_embedding = embeddings.embed_query(text)

print(f"단일 문서 임베딩 벡터의 일부: {text_embedding[:5]}...")
print(f"임베딩 벡터의 차원: {len(text_embedding)}")

# 여러 문서 임베딩
documents = ["사과", "바나나", "컴퓨터", "스마트폰"]
document_embeddings = embeddings.embed_documents(documents)

print(f"\n{len(documents)}개의 문서를 임베딩했습니다.")
print(f"첫 번째 문서(사과)의 임베딩 벡터 일부: {document_embeddings[0][:5]}...")
```