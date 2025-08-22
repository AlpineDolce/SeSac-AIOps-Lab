<h2>LlamaIndex 학습 가이드: 지식 그래프 RAG - 구조화된 지식으로 LLM 강화하기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 LlamaIndex를 활용하여 비정형 텍스트 데이터에서 지식 그래프(Knowledge Graph)를 구축하고, 이를 RAG(Retrieval-Augmented Generation) 시스템에 통합하는 방법을 학습하는 것을 목표로 합니다. 구조화된 지식을 통해 LLM의 추론 능력을 강화하고, 더욱 정확하고 신뢰할 수 있는 답변을 생성하는 고급 RAG 전략을 구현합니다.

<h2>목차</h2>

- [1. 지식 그래프 RAG의 필요성](#1-지식-그래프-rag의-필요성)
  - [1.1. 비정형 텍스트 RAG의 한계](#11-비정형-텍스트-rag의-한계)
  - [1.2. 지식 그래프의 역할](#12-지식-그래프의-역할)
- [2. LlamaIndex에서 지식 그래프 구축](#2-llamaindex에서-지식-그래프-구축)
  - [2.1. Graph Store 설정](#21-graph-store-설정)
  - [2.2. 지식 그래프 인덱스 (KnowledgeGraphIndex) 생성](#22-지식-그래프-인덱스-knowledgegraphindex-생성)
  - [2.3. 엔티티 및 관계 추출](#23-엔티티-및-관계-추출)
- [3. 지식 그래프를 활용한 RAG 쿼리](#3-지식-그래프를-활용한-rag-쿼리)
  - [3.1. 지식 그래프 쿼리 엔진](#31-지식-그래프-쿼리-엔진)
  - [3.2. 하이브리드 검색 (텍스트 + 지식 그래프)](#32-하이브리드-검색-텍스트--지식-그래프)
- [4. 지식 그래프 RAG의 장점](#4-지식-그래프-rag의-장점)
  - [4.1. 추론 능력 강화](#41-추론-능력-강화)
  - [4.2. 환각 감소](#42-환각-감소)
  - [4.3. 설명 가능성 (Explainability)](#43-설명-가능성-explainability)

---

## 1. 지식 그래프 RAG의 필요성

RAG(Retrieval-Augmented Generation)는 LLM의 답변을 외부 데이터로 보강하여 환각을 줄이는 강력한 방법입니다. 하지만 대부분의 RAG 시스템은 비정형 텍스트(unstructured text)를 기반으로 합니다. 복잡한 질문이나 다단계 추론이 필요한 경우, 비정형 텍스트만으로는 LLM이 정확한 답변을 생성하는 데 한계가 있습니다.

### 1.1. 비정형 텍스트 RAG의 한계
-   **복잡한 추론 어려움**: 여러 문서에 흩어진 정보를 조합하거나, 엔티티 간의 관계를 파악하여 추론하는 데 어려움이 있습니다.
-   **환각 발생 가능성**: LLM이 잘못된 관계를 추론하거나, 존재하지 않는 사실을 만들어낼 수 있습니다.
-   **설명 가능성 부족**: LLM이 왜 특정 답변을 생성했는지 추적하기 어렵습니다.

### 1.2. 지식 그래프의 역할
**지식 그래프(Knowledge Graph)**는 엔티티(개념, 사물, 사람 등)와 그들 간의 관계를 노드와 엣지로 표현하는 구조화된 데이터베이스입니다. 지식 그래프는 다음과 같은 장점을 통해 비정형 텍스트 RAG의 한계를 보완합니다.
-   **명시적인 관계**: 엔티티 간의 관계가 명확하게 정의되어 LLM의 추론을 돕습니다.
-   **정확한 정보**: 구조화된 데이터이므로 환각 발생 가능성이 낮습니다.
-   **추론 경로 제공**: 관계를 따라 추론 경로를 시각화하고 설명할 수 있습니다.

**지식 그래프 RAG**는 비정형 텍스트에서 정보를 검색하는 것 외에, 지식 그래프에서 구조화된 정보를 검색하여 LLM의 답변을 보강하는 전략입니다.

## 2. LlamaIndex에서 지식 그래프 구축

LlamaIndex는 비정형 텍스트에서 엔티티와 관계를 추출하여 지식 그래프를 구축하고, 이를 RAG에 활용하는 기능을 제공합니다.

### 2.1. Graph Store 설정
지식 그래프 데이터를 저장할 `Graph Store`를 설정합니다. LlamaIndex는 다양한 그래프 데이터베이스(예: Neo4j, Kuzu)를 지원하며, 간단한 테스트를 위해서는 인메모리(in-memory) 그래프 스토어를 사용할 수 있습니다.

```python
from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 인메모리 그래프 스토어 설정
graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

llm = OpenAI(model="gpt-4o", temperature=0)
```

### 2.2. 지식 그래프 인덱스 (KnowledgeGraphIndex) 생성
`KnowledgeGraphIndex`는 문서에서 엔티티와 관계를 추출하여 지식 그래프를 생성하고, 이를 인덱싱합니다.

```python
# 문서 로드 (예시)
documents = SimpleDirectoryReader(input_files=["./data/knowledge_graph_sample.txt"]).load_data()

# 지식 그래프 인덱스 생성
# LLM이 문서에서 엔티티와 관계를 추출하여 그래프를 구축합니다.
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    llm=llm,
    max_triplets_per_chunk=2, # 한 청크에서 추출할 최대 삼중항(triplet) 수
    include_embeddings=True, # 노드 임베딩 포함 여부
)
```
`knowledge_graph_sample.txt` 예시 내용:
```
이순신은 조선의 장군이다. 그는 거북선을 만들었다. 거북선은 임진왜란에서 사용되었다. 임진왜란은 1592년에 발발했다.
```

### 2.3. 엔티티 및 관계 추출
`KnowledgeGraphIndex`는 내부적으로 LLM을 사용하여 문서에서 엔티티(예: 이순신, 거북선, 임진왜란)와 그들 간의 관계(예: 이순신-장군-조선, 이순신-만들었다-거북선)를 삼중항(Subject-Predicate-Object) 형태로 추출합니다.

```python
# 생성된 지식 그래프 확인 (예시)
# print(kg_index.get_knowledge_graph_triplets())
# [('이순신', '장군', '조선'), ('이순신', '만들었다', '거북선'), ('거북선', '사용되었다', '임진왜란'), ('임진왜란', '발발했다', '1592년')]
```

## 3. 지식 그래프를 활용한 RAG 쿼리

지식 그래프가 구축되면, 이를 활용하여 LLM의 추론을 보강하는 RAG 쿼리를 수행할 수 있습니다.

### 3.1. 지식 그래프 쿼리 엔진
`KnowledgeGraphIndex`는 자체적인 쿼리 엔진을 제공합니다. 이 쿼리 엔진은 질문에서 엔티티를 식별하고, 지식 그래프에서 해당 엔티티와 관련된 정보를 검색하여 LLM에게 전달합니다.

```python
# 지식 그래프 쿼리 엔진 생성
kg_query_engine = kg_index.as_query_engine(
    include_text=False, # 텍스트 검색 결과는 포함하지 않음
    response_mode="tree_summarize", # 검색된 지식 그래프 정보를 요약하여 답변 생성
    embedding_mode="hybrid", # 임베딩과 키워드 매칭을 함께 사용
    llm=llm,
)

# 질의 예시
response = kg_query_engine.query("이순신이 만든 것은 무엇이며, 그것은 언제 사용되었나요?")
print(response)
# 예상 답변: 이순신은 거북선을 만들었으며, 거북선은 임진왜란에서 사용되었습니다.
```

### 3.2. 하이브리드 검색 (텍스트 + 지식 그래프)
가장 강력한 형태의 지식 그래프 RAG는 비정형 텍스트 검색과 지식 그래프 검색을 결합하는 하이브리드 검색입니다. LlamaIndex의 `RouterQueryEngine`을 활용하여 질문의 유형에 따라 적절한 검색 엔진을 동적으로 선택하거나, 두 가지 검색 결과를 통합할 수 있습니다.

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import VectorStoreIndex

# 텍스트 기반 인덱스 생성 (예시)
documents_text = SimpleDirectoryReader(input_files=["./data/text_sample.txt"]).load_data()
text_index = VectorStoreIndex.from_documents(documents_text)

# 텍스트 쿼리 엔진
text_query_engine = text_index.as_query_engine(llm=llm)

# 라우터 쿼리 엔진 생성
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=llm),
    query_engine_tools=[
        text_query_engine.as_query_engine_tool(
            "text_query_engine",
            description="텍스트 기반 질문에 답변합니다.",
        ),
        kg_query_engine.as_query_engine_tool(
            "kg_query_engine",
            description="엔티티 간의 관계나 사실에 대한 질문에 답변합니다.",
        ),
    ],
)

# 질의 예시
# response = query_engine.query("이순신이 만든 것은 무엇인가요?") # 지식 그래프 쿼리 엔진 선택
# response = query_engine.query("이 문서의 핵심 내용은 무엇인가요?") # 텍스트 쿼리 엔진 선택
# print(response)
```

## 4. 지식 그래프 RAG의 장점

### 4.1. 추론 능력 강화
지식 그래프는 엔티티 간의 명시적인 관계를 제공하여 LLM이 복잡한 추론을 수행하는 데 필요한 구조화된 정보를 제공합니다. 이는 LLM이 여러 사실을 연결하여 새로운 통찰을 도출하는 데 도움을 줍니다.

### 4.2. 환각 감소
구조화된 지식 그래프는 정확하고 검증된 사실을 기반으로 하므로, LLM이 존재하지 않는 사실을 만들어내는 환각 현상을 줄이는 데 기여합니다.

### 4.3. 설명 가능성 (Explainability)
지식 그래프는 LLM이 답변을 생성하는 데 사용한 정보의 출처와 추론 경로를 시각적으로 추적하고 설명할 수 있게 합니다. 이는 LLM 애플리케이션의 신뢰성을 높이는 데 중요합니다.