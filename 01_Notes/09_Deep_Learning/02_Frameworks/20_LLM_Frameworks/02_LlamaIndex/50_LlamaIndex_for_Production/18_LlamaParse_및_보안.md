<h2>LlamaIndex 학습 가이드: LlamaParse 및 보안 고려사항</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-22

<h2>문서 목표</h2>
이 문서는 테이블, 이미지, 복잡한 레이아웃을 포함하는 PDF 문서를 효과적으로 파싱하기 위한 전문 서비스인 LlamaParse의 사용법을 학습하고, RAG 시스템 구축 시 반드시 고려해야 할 데이터 보안 및 접근 제어 문제를 이해하는 것을 목표로 합니다. 이를 통해 프로덕션 환경에서 마주할 수 있는 복잡한 데이터 처리와 보안 요구사항에 대응하는 능력을 기릅니다.

<h2>목차</h2>

- [1. 복잡한 PDF 파싱의 어려움](#1-복잡한-pdf-파싱의-어려움)
- [2. LlamaParse: 지능형 문서 파싱 서비스](#2-llamaparse-지능형-문서-파싱-서비스)
  - [2.1. LlamaParse란?](#21-llamaparse란)
  - [2.2. 사용 방법](#22-사용-방법)
- [3. RAG 시스템의 보안](#3-rag-시스템의-보안)
  - [3.1. 데이터 접근 제어](#31-데이터-접근-제어)
  - [3.2. 개인정보보호(PII)](#32-개인정보보호pii)

---

## 1. 복잡한 PDF 파싱의 어려움

일반적인 텍스트 파일과 달리, PDF는 텍스트, 테이블, 이미지, 다단(multi-column) 레이아웃 등 복잡한 구조를 가지고 있습니다. 기존의 PDF 파싱 라이브러리들은 종종 테이블 구조를 깨뜨리거나, 텍스트의 논리적인 순서를 무시하고 잘못된 순서로 추출하는 문제를 보입니다. 이는 RAG 시스템의 검색 품질을 저하시키는 주요 원인이 됩니다.

## 2. LlamaParse: 지능형 문서 파싱 서비스

### 2.1. LlamaParse란?
LlamaParse는 LlamaIndex 개발팀이 직접 제공하는 **지능형 문서 파싱 API 서비스**입니다. 복잡한 PDF 문서에 포함된 테이블과 같은 구조적인 요소를 정확하게 인식하고, 이를 마크다운(Markdown) 형식으로 변환하여 텍스트의 구조와 의미를 최대한 보존해줍니다. 이는 RAG 시스템이 문서의 내용을 더 잘 이해하고 정확한 답변을 생성하도록 돕습니다.

### 2.2. 사용 방법

1.  **API 키 발급**: [LlamaCloud](https://cloud.llamaindex.ai/)에 가입하여 LlamaParse API 키를 발급받습니다.
2.  **라이브러리 설치**:
    ```bash
    pip install llama-parse
    ```
3.  **파서 실행**:

    ```python
    import os
    from llama_parse import LlamaParse

    # .env 등에 LLAMA_CLOUD_API_KEY="..." 설정
    parser = LlamaParse(
        result_as_markdown=True, # 결과를 마크다운으로 변환
        verbose=True
    )

    # 파일 로더에 파서를 연결
    from llama_index.core import SimpleDirectoryReader
    file_extractor = { ".pdf": parser }
    documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

    # 이제 documents에는 테이블 등이 마크다운 형식으로 변환된 내용이 포함됨
    ```

## 3. RAG 시스템의 보안

프로덕션 RAG 시스템은 민감한 데이터를 다루는 경우가 많으므로, 보안은 매우 중요한 고려사항입니다.

### 3.1. 데이터 접근 제어
기업 환경에서는 사용자의 역할이나 부서에 따라 접근할 수 있는 문서가 다릅니다. RAG 시스템 또한 이러한 접근 권한을 존중해야 합니다.

- **해결 전략: 메타데이터 필터링**
  1.  **인덱싱 단계**: 각 `Node`를 생성할 때, 해당 문서에 접근 가능한 사용자 그룹이나 부서 ID를 메타데이터로 추가합니다. (예: `metadata={"allowed_groups": ["engineering", "product"]}`)
  2.  **검색 단계**: 사용자의 쿼리가 들어오면, 해당 사용자의 인증 정보(소속 그룹 등)를 확인합니다. `Retriever`를 설정할 때, 이 사용자 그룹 정보와 일치하는 `allowed_groups` 메타데이터를 가진 노드만 검색하도록 필터를 동적으로 적용합니다.

    ```python
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

    # 현재 사용자가 'engineering' 그룹에 속한다고 가정
    user_group = "engineering"

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="allowed_groups", value=user_group)]
    )

    retriever = index.as_retriever(filters=filters)
    ```

### 3.2. 개인정보보호(PII)
사용자의 질문이나 소스 문서에 포함된 개인정보(이름, 전화번호, 주민등록번호 등)가 LLM API를 통해 외부로 전송되거나, 로그에 기록되지 않도록 주의해야 합니다.

- **해결 전략: PII 탐지 및 마스킹**
  - LlamaIndex의 데이터 변환(Transformation) 파이프라인 단계에서 PII를 탐지하고 익명화(`"John Doe" -> "[PERSON]"`)하는 커스텀 `NodePostprocessor`를 추가할 수 있습니다. 이를 위해 `Microsoft Presidio`와 같은 전문 PII 처리 라이브러리를 활용할 수 있습니다.

보안은 애플리케이션 설계 초기부터 반드시 고려되어야 할 핵심 요소이며, 신뢰할 수 있는 서비스를 구축하기 위한 필수 조건입니다.