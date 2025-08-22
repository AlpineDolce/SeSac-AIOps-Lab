<h2>LangChain 학습 가이드: 출력 파서(Output Parsers)로 LLM 결과 제어하기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LLM이 생성한 순수한 텍스트(string)를 개발자가 원하는 구조화된 데이터(JSON, list, Pydantic 객체 등)로 변환하는 '출력 파서(Output Parser)'의 역할과 사용법을 학습하는 것을 목표로 합니다. 출력 파서를 통해 LLM의 응답을 안정적으로 파싱하고, 후속 처리나 데이터 검증을 용이하게 만드는 방법을 익힙니다.

<h2>목차</h2>

- [1. 왜 출력 파서가 필요한가?](#1-왜-출력-파서가-필요한가)
- [2. 기본적인 출력 파서](#2-기본적인-출력-파서)
  - [2.1. StrOutputParser](#21-stroutputparser)
  - [2.2. CommaSeparatedListOutputParser](#22-commaseparatedlistoutputparser)
- [3. 구조화된 데이터 파싱](#3-구조화된-데이터-파싱)
  - [3.1. PydanticOutputParser](#31-pydanticoutputparser)
  - [3.2. StructuredOutputParser](#32-structuredoutputparser)
- [4. 파싱 오류 처리: RetryOutputParser](#4-파싱-오류-처리-retryoutputparser)

---

## 1. 왜 출력 파서가 필요한가?

LLM은 기본적으로 텍스트를 생성하지만, 애플리케이션에서는 이 텍스트를 특정 형식의 데이터로 변환하여 사용해야 하는 경우가 많습니다. 예를 들어, LLM이 생성한 제품 정보를 JSON 객체로 변환하여 데이터베이스에 저장하거나, 사용자 목록을 파이썬 리스트로 받아 처리하는 경우입니다.

하지만 LLM은 항상 우리가 원하는 형식대로 정확하게 출력물을 생성한다고 보장할 수 없습니다. 출력 파서는 이러한 LLM의 응답을 우리가 정의한 형식에 맞게 파싱하고, 필요한 경우 형식을 맞추도록 LLM에게 재요청하는 등 **안정적인 데이터 변환**을 도와주는 필수적인 도구입니다.

## 2. 기본적인 출력 파서

### 2.1. StrOutputParser
가장 간단한 파서로, Chat Model이 출력하는 `AIMessage` 객체에서 내용(content) 부분만 추출하여 순수한 문자열로 변환합니다.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_template("안녕! {name}") | ChatOpenAI() | StrOutputParser()
response = chain.invoke({"name": "철수"})

print(type(response)) # <class 'str'>
print(response) # "안녕하세요, 철수님! 만나서 반갑습니다."
```

### 2.2. CommaSeparatedListOutputParser
LLM이 쉼표로 구분된 목록을 생성하도록 유도하고, 이를 파이썬 리스트로 변환합니다.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()

# 파서가 LLM에게 어떤 형식으로 출력해야 할지 알려주는 지침을 포함시킴
format_instructions = output_parser.get_format_instructions()
# format_instructions -> "Your response should be a list of comma separated values, eg: `foo, bar, baz`"

chain = prompt | model | output_parser
response = chain.invoke({"query": "AI의 3가지 주요 분야를 알려줘"}) # LLM은 '기계 학습, 자연어 처리, 컴퓨터 비전'과 같이 출력

print(type(response)) # <class 'list'>
print(response) # ['기계 학습', '자연어 처리', '컴퓨터 비전']
```

## 3. 구조화된 데이터 파싱

### 3.1. PydanticOutputParser
가장 강력하고 권장되는 방법 중 하나입니다. Pydantic 모델을 사용하여 원하는 데이터 구조를 정의하면, 파서가 알아서 형식 지침을 생성하고 출력물을 Pydantic 객체로 변환해 줍니다.

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# 1. 원하는 데이터 구조를 Pydantic 모델로 정의
class Joke(BaseModel):
    setup: str = Field(description="농담의 배경 설정")
    punchline: str = Field(description="농담의 핵심 부분")

# 2. 파서 생성
parser = PydanticOutputParser(pydantic_object=Joke)

# 3. 프롬프트에 형식 지침 주입
prompt = ChatPromptTemplate.from_messages([
    ("system", "{format_instructions}"),
    ("human", "{topic}에 대한 농담을 만들어줘.")
]).partial(format_instructions=parser.get_format_instructions())

# 4. 체인 구성 및 실행
chain = prompt | ChatOpenAI() | parser
response = chain.invoke({"topic": "개발자"})

print(type(response)) # <class '__main__.Joke'>
print(response) # setup='왜 개발자는 항상 헤드폰을 끼고 있을까?' punchline='시끄러운 버그들의 소리를 듣지 않으려고!'
print(response.punchline) # '시끄러운 버그들의 소리를 듣지 않으려고!'
```

### 3.2. StructuredOutputParser
Pydantic을 사용하지 않고, 간단한 키-값 쌍의 JSON 객체를 파싱하고 싶을 때 사용합니다.

```python
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="answer", description="사용자의 질문에 대한 답변"),
    ResponseSchema(name="source", description="답변의 출처")
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 이후 과정은 PydanticOutputParser와 유사
```

## 4. 파싱 오류 처리: RetryOutputParser

LLM이 정의된 형식에 맞지 않는 결과물을 생성하여 파싱에 실패하는 경우가 있습니다. `RetryOutputParser`는 이러한 경우, 파싱 오류 메시지와 함께 동일한 프롬프트를 다시 LLM에게 보내 출력을 교정하도록 자동으로 재시도합니다. 이는 애플리케이션의 안정성을 크게 높여줍니다.

```python
from langchain.output_parsers import RetryOutputParser

# 기본 파서(Pydantic 파서 등)와 LLM을 인자로 받음
retry_parser = RetryOutputParser.from_llm(parser=parser, llm=ChatOpenAI())

# 사용법은 기본 파서와 동일
# chain = prompt | model | retry_parser
```