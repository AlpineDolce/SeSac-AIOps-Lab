<h2>LangChain 학습 가이드: RAG 실전 - 나만의 문서 Q&A 챗봇 만들기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 RAG(Retrieval-Augmented Generation)의 이론적 개념을 바탕으로, 실제 텍스트 파일을 기반으로 질문에 답변하는 Q&A 시스템을 처음부터 끝까지 구축하는 과정을 안내합니다. 각 단계별 코드 구현을 통해 RAG 파이프라인의 전체적인 흐름을 체득하고, 이를 응용하여 자신만의 데이터 기반 챗봇을 만들 수 있는 능력을 기르는 것을 목표로 합니다.

<h2>목차</h2>

- [1. RAG Q&A 시스템 아키텍처](#1-rag-qa-시스템-아키텍처)
- [2. 사전 준비](#2-사전-준비)
- [3. 단계별 구현](#3-단계별-구현)
  - [3.1. 1단계: 문서 로딩 (Load)](#31-1단계-문서-로딩-load)
  - [3.2. 2단계: 문서 분할 (Split)](#32-2단계-문서-분할-split)
  - [3.3. 3단계: 벡터 저장소에 저장 (Store)](#33-3단계-벡터-저장소에-저장-store)
  - [3.4. 4-5단계: 검색 및 생성 (Retrieve & Generate)](#34-4-5단계-검색-및-생성-retrieve--generate)
- [4. 전체 실행 코드](#4-전체-실행-코드)

---

## 1. RAG Q&A 시스템 아키텍처

우리가 만들 시스템은 다음 흐름으로 동작합니다.
1.  **Indexing (사전 준비 단계)**: PDF, TXT 등 원본 문서를 불러와 의미 단위로 분할하고, 각 조각을 벡터로 변환하여 벡터 데이터베이스에 저장합니다.
2.  **Retrieval & Generation (실시간 응답 단계)**: 사용자의 질문이 들어오면, 해당 질문과 가장 관련성 높은 문서 조각을 벡터 DB에서 검색하고, 검색된 내용과 원본 질문을 함께 LLM에 전달하여 최종 답변을 생성합니다.

## 2. 사전 준비

필요한 라이브러리를 설치합니다. `faiss-cpu`는 Facebook에서 개발한 효율적인 벡터 검색 라이브러리입니다.

```bash
pip install langchain langchain-openai faiss-cpu python-dotenv
```

그리고 답변의 근거가 될 텍스트 파일을 하나 준비합니다. 예를 들어, `sample.txt`라는 이름으로 다음과 같이 저장합니다.

`sample.txt`
```
LangChain은 LLM을 활용한 애플리케이션 개발을 돕는 프레임워크입니다.
주요 기능으로는 모델 I/O, 체인, 에이전트, RAG 등이 있습니다.
2022년 Harrison Chase에 의해 처음 출시되었습니다.
LangChain을 사용하면 복잡한 LLM 워크플로우를 쉽게 구현할 수 있습니다.
```

## 3. 단계별 구현

### 3.1. 1단계: 문서 로딩 (Load)
`TextLoader`를 사용하여 `sample.txt` 파일을 불러옵니다.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample.txt", encoding="utf-8")
docs = loader.load()
```

### 3.2. 2단계: 문서 분할 (Split)
`RecursiveCharacterTextSplitter`를 사용하여 문서를 적절한 크기의 청크로 분할합니다.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
splits = text_splitter.split_documents(docs)
```

### 3.3. 3단계: 벡터 저장소에 저장 (Store)
분할된 청크들을 임베딩 모델(`OpenAIEmbeddings`)을 통해 벡터로 변환하고, `FAISS` 벡터 저장소에 저장합니다. 이 과정을 통해 나중에 질문과 관련된 내용을 빠르게 찾을 수 있는 인덱스가 생성됩니다.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# OpenAI 임베딩 모델 초기화
embeddings = OpenAIEmbeddings()

# 분할된 문서들을 기반으로 FAISS 벡터 저장소 생성
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
```

### 3.4. 4-5단계: 검색 및 생성 (Retrieve & Generate)
이제 사용자의 질문에 답변할 준비가 되었습니다. LangChain은 이 과정을 편리하게 처리할 수 있는 `create_retrieval_chain` 함수를 제공합니다.

1.  **Retriever 생성**: 벡터 저장소를 검색기(Retriever)로 변환합니다.
2.  **Prompt Template 설정**: 검색된 문서(`context`)와 사용자 질문(`input`)을 조합하여 LLM에게 전달할 프롬프트를 정의합니다.
3.  **Chain 생성**: `create_retrieval_chain`을 사용하여 Retriever와 LLM을 연결하는 전체 체인을 만듭니다.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LLM 모델 초기화
llm = ChatOpenAI()

# Retriever 생성
retriever = vectorstore.as_retriever()

# Prompt Template 설정
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# 문서들을 통합하여 하나의 체인으로 만드는 부분
document_chain = create_stuff_documents_chain(llm, prompt)

# Retriever와 document_chain을 연결하여 최종 체인 생성
retrieval_chain = create_retrieval_chain(retriever, document_chain)
```

## 4. 전체 실행 코드

```python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# API 키 로드
load_dotenv()

# 1. 문서 로딩
loader = TextLoader("sample.txt", encoding="utf-8")
docs = loader.load()

# 2. 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
splits = text_splitter.split_documents(docs)

# 3. 벡터 저장소 생성
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

# 4. LLM 및 체인 설정
llm = ChatOpenAI()
retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 5. 체인 실행 및 결과 확인
question = "LangChain은 언제 출시되었나요?"
response = retrieval_chain.invoke({"input": question})

print(f"질문: {question}")
print(f"답변: {response['answer']}")
# 예상 출력:
# 답변: LangChain은 2022년 Harrison Chase에 의해 처음 출시되었습니다.
```
이 코드를 통해 우리는 `sample.txt` 파일의 내용을 기반으로 정확하게 답변하는 Q&A 시스템을 성공적으로 구축했습니다.