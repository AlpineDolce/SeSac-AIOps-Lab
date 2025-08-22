<h2>LlamaIndex 학습 가이드: 5줄 코드로 만드는 첫 RAG 파이프라인</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LlamaIndex의 강력함과 간결함을 직접 체험하는 것을 목표로 합니다. 단 5줄의 핵심 코드로 로컬 문서를 기반으로 질문에 답변하는 완전한 RAG(Retrieval-Augmented Generation) 파이프라인을 구축하는 과정을 안내합니다. 이 실습을 통해 LlamaIndex의 전체적인 작동 흐름을 직관적으로 이해하게 될 것입니다.

<h2>목차</h2>

- [1. 실습 개요](#1-실습-개요)
- [2. 사전 준비](#2-사전-준비)
- [3. 5줄의 핵심 코드](#3-5줄의-핵심-코드)
  - [3.1. 코드 분석](#31-코드-분석)
- [4. 전체 실행 코드 및 결과](#4-전체-실행-코드-및-결과)
- [5. 내부 작동 원리](#5-내부-작동-원리)

---

## 1. 실습 개요
우리의 목표는 특정 텍스트 파일의 내용에 대해서만 질문하고 답변을 받는 간단한 Q&A 봇을 만드는 것입니다. LlamaIndex를 사용하면 이 과정이 놀랍도록 간단해집니다.

## 2. 사전 준비
1.  `02_환경_설정.md` 가이드에 따라 환경 설정 및 라이브러리 설치가 완료되어야 합니다.
2.  질문의 기반이 될 텍스트 파일을 준비합니다. 프로젝트 디렉터리에 `data` 폴더를 만들고, 그 안에 `paul_graham_essay.txt` 라는 이름으로 폴 그레이엄의 에세이 "What I Worked On"의 일부를 저장했다고 가정하겠습니다. (실제 에세이 텍스트는 웹에서 쉽게 구할 수 있습니다.)

## 3. 5줄의 핵심 코드

LlamaIndex의 마법은 다음 5줄의 코드에 압축되어 있습니다.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. 데이터 로딩
documents = SimpleDirectoryReader("data").load_data()

# 2. 인덱스 생성 (내부적으로 임베딩 및 저장이 일어남)
index = VectorStoreIndex.from_documents(documents)

# 3. 쿼리 엔진 생성
query_engine = index.as_query_engine()

# 4. 쿼리 및 응답 생성
response = query_engine.query("What did the author do growing up?")
```

### 3.1. 코드 분석
- **`SimpleDirectoryReader("data")`**: `data` 폴더 안의 모든 파일을 읽어오는 로더입니다.
- **`.load_data()`**: 파일을 `Document` 객체의 리스트로 변환합니다.
- **`VectorStoreIndex.from_documents(documents)`**: LlamaIndex의 핵심입니다. 이 한 줄의 코드 안에서 다음 작업이 모두 자동으로 일어납니다.
    - 문서를 `Node`로 분할 (Chunking)
    - 각 `Node`를 OpenAI 임베딩 모델을 사용해 벡터로 변환 (Embedding)
    - 텍스트와 벡터를 메모리 상의 간단한 벡터 저장소에 저장 (Storing)
- **`index.as_query_engine()`**: 생성된 인덱스를 사용하여 질문에 답변할 수 있는 쿼리 엔진을 만듭니다.
- **`query_engine.query(...)`**: RAG 파이프라인 전체를 실행합니다.
    - 입력된 질문을 벡터로 변환
    - 인덱스에서 관련성 높은 문서를 검색
    - 검색된 문서와 질문을 함께 LLM에 전달하여 답변 생성

## 4. 전체 실행 코드 및 결과

`first_rag.py`
```python
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# .env 파일에서 OPENAI_API_KEY 로드
load_dotenv()

# 1. 데이터 로딩
print("데이터를 로딩합니다...")
documents = SimpleDirectoryReader("data").load_data()
print(f"{len(documents)}개의 문서를 로드했습니다.")

# 2. 인덱스 생성
print("인덱스를 생성하는 중입니다...")
index = VectorStoreIndex.from_documents(documents)
print("인덱스 생성이 완료되었습니다.")

# 3. 쿼리 엔진 생성
query_engine = index.as_query_engine()

# 4. 쿼리 및 응답 생성
print("쿼리를 실행합니다...")
question = "What did the author do growing up?"
response = query_engine.query(question)

# 5. 결과 출력
print("\n--- 응답 ---")
print(response)
print("--------------")
```

**실행 결과 (예시):**
```
데이터를 로딩합니다...
1개의 문서를 로드했습니다.
인덱스를 생성하는 중입니다...
인덱스 생성이 완료되었습니다.
쿼리를 실행합니다...

--- 응답 ---
The author worked on writing and programming outside of school. He wrote short stories and also tried to program on an IBM 1401 in 9th grade. He later got a TRS-80 microcomputer and started programming on it, writing simple games and a word processor.
--------------
```

## 5. 내부 작동 원리
단 5줄의 코드로, 우리는 LlamaIndex가 내부적으로 수행하는 복잡한 RAG 파이프라인(Load -> Split -> Embed -> Store -> Retrieve -> Generate)을 모두 실행했습니다. LlamaIndex는 이처럼 개발자가 RAG의 본질에만 집중할 수 있도록 높은 수준의 추상화를 제공하며, 이것이 LlamaIndex가 가진 가장 큰 매력입니다.
