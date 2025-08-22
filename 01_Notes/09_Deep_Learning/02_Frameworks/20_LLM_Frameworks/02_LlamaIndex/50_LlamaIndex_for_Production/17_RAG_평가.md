<h2>LlamaIndex 학습 가이드: RAG 시스템 정량적 평가</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 개발된 RAG 시스템의 성능을 주관적인 '감'이 아닌, 객관적이고 정량적인 지표로 평가하는 방법을 학습하는 것을 목표로 합니다. LlamaIndex가 제공하는 평가 모듈을 사용하여 답변의 충실성(Faithfulness), 관련성(Relevancy) 등을 자동으로 측정하고, 이를 통해 시스템의 문제점을 진단하고 개선 방향을 설정하는 능력을 기릅니다.

<h2>목차</h2>

- [1. 왜 정량적 평가가 중요한가?](#1-왜-정량적-평가가-중요한가)
- [2. LlamaIndex 평가 모듈의 핵심](#2-llamaindex-평가-모듈의-핵심)
  - [2.1. 평가 데이터셋 생성](#21-평가-데이터셋-생성)
  - [2.2. 평가기 (Evaluators)](#22-평가기-evaluators)
- [3. RAG 평가 실습](#3-rag-평가-실습)
  - [3.1. 평가용 질문 자동 생성](#31-평가용-질문-자동-생성)
  - [3.2. 응답 및 평가 실행](#32-응답-및-평가-실행)

---

## 1. 왜 정량적 평가가 중요한가?

프롬프트를 약간 수정하거나, `chunk_size`를 변경했을 때, RAG 시스템의 성능이 정말로 좋아졌는지 어떻게 확신할 수 있을까요? 몇 가지 샘플 질문에 대한 답변이 좋아 보인다고 해서 전체적인 성능 향상을 보장할 수는 없습니다.

**정량적 평가**는 일관된 기준(metric)과 데이터셋을 사용하여 시스템의 성능을 숫자로 표현하는 과정입니다. 이를 통해 다음과 같은 이점을 얻을 수 있습니다.

- **객관적인 비교**: 변경 전후의 성능을 객관적인 점수로 비교하여 개선 여부를 명확히 판단할 수 있습니다.
- **문제 진단**: 어떤 부분(Retriever, Generator)에서 성능 저하가 발생하는지 구체적으로 파악할 수 있습니다.
- **지속적인 개선**: 평가-개선 사이클을 자동화하여 시스템의 품질을 지속적으로 관리하고 향상시킬 수 있습니다.

## 2. LlamaIndex 평가 모듈의 핵심

LlamaIndex는 RAG 시스템 평가를 위한 강력한 도구들을 `llama-index-evaluation` 패키지에 제공합니다.

```bash
pip install llama-index-evaluation
```

### 2.1. 평가 데이터셋 생성
좋은 평가는 좋은 데이터셋에서 시작됩니다. LlamaIndex는 `generate_question_context_pairs` 함수를 제공하여, 보유한 문서(`Node`)로부터 평가에 사용할 (질문, 컨텍스트) 쌍을 LLM을 통해 자동으로 생성하게 할 수 있습니다. 이는 평가 데이터셋 구축에 드는 노력을 크게 줄여줍니다.

### 2.2. 평가기 (Evaluators)
LlamaIndex는 RAG의 품질을 다양한 관점에서 측정하는 여러 평가기를 제공합니다.

- **`FaithfulnessEvaluator` (충실성)**: 생성된 답변이 검색된 컨텍스트(문서 조각)에 얼마나 충실한지를 평가합니다. LLM이 컨텍스트에 없는 내용을 지어내는지(환각)를 측정합니다.
- **`RelevancyEvaluator` (관련성)**: 생성된 답변과 검색된 컨텍스트가 모두 사용자의 원본 질문과 얼마나 관련이 있는지를 평가합니다. 동문서답 여부를 측정합니다.
- **`CorrectnessEvaluator` (정확성)**: 생성된 답변이 우리가 미리 준비한 정답(reference answer)과 얼마나 일치하는지를 평가합니다.

이 평가기들은 내부적으로 LLM을 사용하여 점수(1.0 ~ 5.0)와 평가 이유를 함께 반환합니다.

## 3. RAG 평가 실습

### 3.1. 평가용 질문 자동 생성

```python
from llama_index.core.evaluation import generate_question_context_pairs

# 문서(노드)로부터 평가용 데이터셋 생성
qa_dataset = generate_question_context_pairs(nodes)
```

### 3.2. 응답 및 평가 실행

```python
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# 1. 쿼리 엔진 준비
query_engine = index.as_query_engine()

# 2. 평가기 준비
faithfulness_evaluator = FaithfulnessEvaluator()
relevancy_evaluator = RelevancyEvaluator()

# 3. 첫 번째 질문에 대해 응답 생성 및 평가
eval_question = qa_dataset.queries["some-query-id"]
response_vector = query_engine.query(eval_question)

# 4. 충실성 평가
faithfulness_result = faithfulness_evaluator.evaluate_response(
    response=response_vector
)
print(f"Faithfulness: {faithfulness_result.passing}, Score: {faithfulness_result.score}")

# 5. 관련성 평가
relevancy_result = relevancy_evaluator.evaluate_response(
    query=eval_question,
    response=response_vector
)
print(f"Relevancy: {relevancy_result.passing}, Score: {relevancy_result.score}")
```

이러한 평가 파이프라인을 구축하면, RAG 시스템의 구성 요소(청킹 전략, 임베딩 모델, 응답 모드 등)를 변경할 때마다 성능 변화를 정량적으로 추적하고, 데이터에 기반한 의사결정을 내릴 수 있게 됩니다.