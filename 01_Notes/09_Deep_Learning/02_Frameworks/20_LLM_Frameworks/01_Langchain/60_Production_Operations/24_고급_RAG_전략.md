<h2>LangChain 학습 가이드: 고급 RAG 전략 - 검색 성능 극대화하기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 기본적인 RAG 시스템의 한계를 극복하고, 더 복잡하고 미묘한 사용자 질문에 대해 정확한 답변을 생성하기 위한 고급 RAG 전략들을 학습하는 것을 목표로 합니다. 쿼리 변환, 재랭킹, 하이브리드 검색 등 검색(Retrieval) 단계의 성능을 극대화하는 기법들을 이해하고, 이를 통해 RAG 시스템의 전반적인 품질을 한 단계 끌어올리는 능력을 기릅니다.

<h2>목차</h2>

- [1. 기본 RAG의 한계](#1-기본-rag의-한계)
- [2. 쿼리 변환 (Query Transformation)](#2-쿼리-변환-query-transformation)
  - [2.1. HyDE (Hypothetical Document Embeddings)](#21-hyde-hypothetical-document-embeddings)
  - [2.2. 다중 쿼리 (Multi-Query)](#22-다중-쿼리-multi-query)
- [3. 재랭킹 (Re-ranking)](#3-재랭킹-re-ranking)
  - [3.1. 재랭킹의 필요성](#31-재랭킹의-필요성)
  - [3.2. Cohere Rerank API 활용](#32-cohere-rerank-api-활용)
- [4. 하이브리드 검색 (Hybrid Search)](#4-하이브리드-검색-hybrid-search)
  - [4.1. 의미론적 검색 vs 키워드 검색](#41-의미론적-검색-vs-키워드-검색)
  - [4.2. 두 방식의 결합](#42-두-방식의-결합)

---

## 1. 기본 RAG의 한계

기본적인 RAG는 사용자의 질문을 그대로 벡터로 변환하여 유사한 문서를 찾습니다. 이 방식은 간단하지만 다음과 같은 경우에 한계를 보입니다.

- **질문과 문서의 표현 불일치**: 사용자의 질문이 매우 짧거나, 문서에서 사용하는 용어와 다른 용어를 사용할 경우, 의미는 같더라도 벡터 공간에서 거리가 멀어져 검색 성능이 저하될 수 있습니다.
- **복합적인 질문**: 여러 개의 소주제를 포함하는 복잡한 질문에 대해, 일부 주제와 관련된 문서만 검색되고 나머지는 누락될 수 있습니다.
- **키워드의 중요성**: 'GPT-4'와 같은 특정 고유명사나 전문 용어는 의미보다는 키워드 자체가 중요하지만, 의미론적 검색은 이를 놓칠 수 있습니다.

## 2. 쿼리 변환 (Query Transformation)

쿼리 변환은 사용자의 원본 질문을 검색에 더 유리한 형태로 가공하는 기술입니다.

### 2.1. HyDE (Hypothetical Document Embeddings)
- **작동 원리**: 사용자의 질문에 대해 LLM이 먼저 **가상의 답변(Hypothetical Answer)** 을 생성합니다. 그 다음, 원본 질문 대신 이 가상의 답변을 벡터로 변환하여 문서를 검색합니다. 질문보다 상세한 가상의 답변이 실제 문서와 벡터 공간에서 더 가까울 것이라는 아이디어에 기반합니다.
- **사용 사례**: "RAG의 단점은?"과 같이 짧고 추상적인 질문을 "RAG는 환각을 줄이지만, 검색 성능에 따라 품질이 좌우되는 단점이 있다..."와 같은 구체적인 문장으로 변환하여 검색 품질을 높일 수 있습니다.

### 2.2. 다중 쿼리 (Multi-Query)
- **작동 원리**: 하나의 복잡한 질문을 LLM을 사용하여 여러 개의 다른 관점을 가진 간단한 질문으로 분해합니다. 그리고 이 모든 질문들을 각각 사용하여 문서를 검색한 뒤, 결과를 통합합니다.
- **사용 사례**: "LangChain과 LlamaIndex의 장단점을 비교하고, 에이전트 구현에 더 적합한 것을 알려줘"라는 질문을 다음과 같은 여러 쿼리로 변환할 수 있습니다.
    1. "LangChain의 장단점"
    2. "LlamaIndex의 장단점"
    3. "LangChain 에이전트 구현 방법"
    4. "LlamaIndex 에이전트 구현 방법"

## 3. 재랭킹 (Re-ranking)

### 3.1. 재랭킹의 필요성
1차 Retriever(예: 벡터 검색)는 속도가 빠르지만, 정확도가 완벽하지 않을 수 있습니다. 보통 관련성이 높은 문서를 상위에 반환하지만, 때로는 덜 중요한 문서가 섞여 있을 수 있습니다. 재랭킹은 1차적으로 검색된 문서들의 순서를 더 정교한 모델을 사용하여 다시 매기는 과정입니다.

- **작동 방식**: 1차 Retriever가 상위 N개의 문서(예: 20개)를 가져오면, 더 강력하지만 느린 재랭킹 모델이 이 N개의 문서를 대상으로 사용자 질문과의 실제 관련도를 다시 계산하여 상위 K개(예: 5개)를 최종 선택합니다.

### 3.2. Cohere Rerank API 활용
Cohere는 상업적으로 사용할 수 있는 고성능 재랭킹 모델 API를 제공하며, LangChain은 이를 쉽게 연동할 수 있는 `CohereRerank` 통합을 지원합니다.

```python
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# 기본 Retriever 준비
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Cohere 재랭킹 모델 설정
cohere_reranker = CohereRerank(top_n=5)

# 압축 Retriever 생성: 재랭킹을 통해 문서를 '압축'(선별)하는 효과
compression_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_reranker, base_retriever=base_retriever
)

# 이제 compression_retriever를 사용하면, 20개를 검색한 후 5개로 재랭킹된 결과가 나옴
```

## 4. 하이브리드 검색 (Hybrid Search)

### 4.1. 의미론적 검색 vs 키워드 검색
- **의미론적 검색 (Semantic Search)**: 벡터 유사도에 기반하며, 문장의 의미를 이해하여 관련 문서를 찾습니다. (예: "배 아플 때 먹는 약" -> "소화제")
- **키워드 검색 (Keyword Search)**: BM25와 같은 알고리즘에 기반하며, 단어의 출현 빈도와 중요도를 계산하여 문서를 찾습니다. 특정 용어나 고유명사 검색에 강합니다.

### 4.2. 두 방식의 결합
하이브리드 검색은 이 두 가지 방식의 장점을 모두 취하는 전략입니다. 의미론적 검색과 키워드 검색을 동시에 수행한 후, 각 점수를 조합하여 최종 순위를 결정합니다. 이를 통해 의미적으로 관련성이 높으면서도 중요한 키워드를 포함하는 문서를 효과적으로 찾아낼 수 있습니다. 많은 최신 벡터 DB(Pinecone, Weaviate 등)가 하이브리드 검색 기능을 자체적으로 지원합니다.