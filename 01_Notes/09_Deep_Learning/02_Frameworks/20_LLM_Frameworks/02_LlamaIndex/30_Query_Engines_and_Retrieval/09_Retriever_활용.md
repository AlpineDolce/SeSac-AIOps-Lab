<h2>LlamaIndex 학습 가이드: Retriever 활용 - 검색 과정 제어하기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 쿼리 엔진의 핵심 구성 요소인 `Retriever`를 직접 다루는 방법을 학습하는 것을 목표로 합니다. `Retriever`의 역할을 명확히 이해하고, `index.as_retriever()`를 통해 검색 과정을 세밀하게 제어하며, 벡터 저장소의 다양한 쿼리 모드를 활용하여 특정 시나리오에 맞는 검색 전략을 구사하는 능력을 기릅니다.

<h2>목차</h2>

- [1. 쿼리 엔진 vs. Retriever](#1-쿼리-엔진-vs-retriever)
- [2. Retriever 직접 사용하기](#2-retriever-직접-사용하기)
  - [2.1. `index.as_retriever()`](#21-indexas_retriever)
  - [2.2. `retrieve` 메소드](#22-retrieve-메소드)
- [3. Retriever 커스터마이징](#3-retriever-커스터마이징)
  - [3.1. `similarity_top_k` 설정](#31-similarity_top_k-설정)
  - [3.2. 메타데이터 필터링](#32-메타데이터-필터링)

---

## 1. 쿼리 엔진 vs. Retriever

쿼리 엔진과 Retriever는 밀접하게 관련되어 있지만, 역할이 다릅니다.

- **Retriever**: 이름 그대로, 인덱스에서 쿼리와 관련된 `Node`(문서 조각)를 **검색(Retrieve)**하는 역할만 담당합니다. Retriever는 LLM을 호출하지 않습니다.
- **Query Engine**: Retriever를 사용하여 `Node`를 검색하고, 검색된 `Node`들을 LLM에 전달하여 최종 답변을 **생성(Generate)**하는 전체 과정을 담당합니다.

즉, **Query Engine = Retriever + Response Synthesizer (LLM)** 의 구조를 가집니다. 대부분의 경우 쿼리 엔진을 사용하는 것이 편리하지만, 검색 과정 자체를 더 세밀하게 제어하거나, 검색된 결과를 LLM에 전달하기 전에 별도의 처리를 하고 싶을 때는 Retriever를 직접 사용하는 것이 유용합니다.

## 2. Retriever 직접 사용하기

### 2.1. `index.as_retriever()`
인덱스 객체의 `as_retriever()` 메소드를 호출하여 Retriever 객체를 얻을 수 있습니다.

```python
# 인덱스로부터 Retriever 생성
retriever = index.as_retriever()
```

### 2.2. `retrieve` 메소드
Retriever 객체의 `retrieve()` 메소드에 쿼리 문자열을 전달하면, 관련성 높은 `Node` 객체의 리스트를 반환합니다.

```python
query_str = "What did the author do after college?"

# 쿼리를 사용하여 관련 노드 검색
retrieved_nodes = retriever.retrieve(query_str)

# 검색된 노드의 내용과 유사도 점수 확인
for node in retrieved_nodes:
    print(f"Score: {node.score:.4f}")
    print(f"Text: {node.text}")
    print("---")
```
`node.score`는 쿼리와의 유사도를 나타내며, LlamaIndex는 기본적으로 0.0 ~ 1.0 사이의 값으로 정규화하여 보여줍니다. (1.0에 가까울수록 유사함)

## 3. Retriever 커스터마이징

`as_retriever()`를 호출할 때 다양한 파라미터를 전달하여 검색 동작을 세밀하게 제어할 수 있습니다.

### 3.1. `similarity_top_k` 설정
쿼리 엔진과 마찬가지로, 검색할 상위 K개의 노드 수를 지정할 수 있습니다.

```python
# 상위 5개의 노드만 검색하도록 설정
retriever_top5 = index.as_retriever(similarity_top_k=5)
```

### 3.2. 메타데이터 필터링
`Node`를 인덱싱할 때 메타데이터를 함께 저장했다면, 검색 시 이 메타데이터를 필터링 조건으로 사용할 수 있습니다. 이는 RAG 시스템의 정확도를 높이는 매우 강력한 기능입니다.

**예시**: `category`가 `tech`인 문서 중에서만 검색하기

```python
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

# 필터 조건 정의
filters = MetadataFilters(
    filters=[ExactMatchFilter(key="category", value="tech")]
)

# 필터를 적용하여 Retriever 생성
retriever_filtered = index.as_retriever(
    similarity_top_k=3,
    filters=filters
)

retrieved_nodes = retriever_filtered.retrieve("Tell me about Large Language Models")
```
이제 `retrieved_nodes`에는 `category`가 `tech`인 노드들만 포함되게 됩니다. 이를 통해 관련 없는 분야의 문서가 검색 결과에 포함되는 것을 방지하여 LLM이 더 정확한 답변을 생성하도록 도울 수 있습니다.