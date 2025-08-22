# LlamaIndex 실무 역량 강화 가이드

**데이터와 LLM의 완벽한 연결: 최첨단 RAG 애플리케이션 구축의 모든 것**

이 가이드는 LLM(거대 언어 모델)에 외부 데이터를 효과적으로 연결하는 RAG(Retrieval-Augmented Generation) 기술에 특화된 프레임워크, LlamaIndex의 핵심 원리부터 실전 응용까지 체계적으로 안내합니다. LlamaIndex는 단순한 데이터 연결을 넘어, 복잡한 데이터를 최적으로 인덱싱하고, 정교하게 검색하며, 신뢰도 높은 답변을 생성하는 최첨단 RAG 파이프라인 구축을 목표로 합니다. 이 과정을 통해 여러분은 데이터의 잠재력을 최대한 활용하여 정확하고 지능적인 LLM 애플리케이션을 개발하는 핵심 역량을 갖추게 될 것입니다.

---

### Part 1: LlamaIndex 시작하기 (Getting Started)
- **학습 목표:** LlamaIndex의 핵심 사상과 RAG에서의 역할을 이해하고, 기본적인 데이터 질의응답 파이프라인을 직접 구축해봅니다.
- **왜 중요한가?** LlamaIndex가 어떤 문제를 해결하기 위해 탄생했는지, 그리고 LangChain과 어떤 철학적 차이가 있는지를 이해하는 것은 LlamaIndex를 효과적으로 활용하기 위한 첫걸음입니다.
- **주요 내용:**
    - [**01_LlamaIndex_소개.md**](./10_Getting_Started/01_LlamaIndex_소개.md): LlamaIndex의 정의, RAG 특화 프레임워크로서의 강점, LangChain과의 차이점을 학습합니다.
    - [**02_환경_설정.md**](./10_Getting_Started/02_환경_설정.md): 개발 환경 설정 및 필수 라이브러리를 설치합니다.
    - [**03_첫_RAG_파이프라인.md**](./10_Getting_Started/03_첫_RAG_파이프라인.md): 단 5줄의 코드로 문서에 대해 질문하고 답변을 받는 가장 간단한 RAG 파이프라인을 경험합니다.

### Part 2: 데이터 인덱싱 (Data Indexing)
- **학습 목표:** 다양한 소스로부터 데이터를 불러오고, 이를 검색에 최적화된 구조인 '인덱스(Index)'로 변환하는 LlamaIndex의 핵심 메커니즘을 학습합니다.
- **왜 중요한가?** 얼마나 효과적으로 데이터를 인덱싱하는지가 RAG 시스템 전체의 성능을 좌우합니다. 데이터의 특성에 맞는 인덱스를 구성하는 능력은 RAG 전문가의 핵심 역량입니다.
- **주요 내용:**
    - [**04_데이터_로더.md**](./20_Data_Indexing/04_데이터_로더.md): PDF, 웹페이지, 데이터베이스 등 다양한 데이터 소스를 로드하는 `Document Loader`의 활용법을 익힙니다.
    - [**05_노드와_변환.md**](./20_Data_Indexing/05_노드와_변환.md): 문서를 의미 있는 단위인 `Node`로 분할하고, 메타데이터를 추가하는 등 파싱 및 변환 과정을 학습합니다.
    - [**06_벡터_인덱스.md**](./20_Data_Indexing/06_벡터_인덱스.md): 가장 기본이 되는 `VectorStoreIndex`의 작동 원리와 임베딩 모델과의 관계를 이해합니다.
    - [**07_인덱스_저장_및_로드.md**](./20_Data_Indexing/07_인덱스_저장_및_로드.md): 한 번 생성한 인덱스를 디스크에 저장하고, 필요할 때 다시 불러와 재사용하는 방법을 학습하여 시간과 비용을 절약합니다.

### Part 3: 쿼리 엔진과 검색 (Query Engines & Retrieval)
- **학습 목표:** 인덱싱된 데이터를 실제로 활용하여 질문에 답변하는 `쿼리 엔진`의 작동 방식을 이해하고, 다양한 검색 전략과 응답 모드를 제어하는 방법을 학습합니다.
- **왜 중요한가?** 단순히 질문하고 답을 얻는 것을 넘어, 어떤 방식으로 문서를 검색하고, 검색된 정보를 어떻게 조합하여 답변을 생성할지를 제어함으로써 RAG 시스템의 정확도와 응답 품질을 크게 향상시킬 수 있습니다.
- **주요 내용:**
    - [**08_쿼리_엔진_기초.md**](./30_Query_Engines_and_Retrieval/08_쿼리_엔진_기초.md): `index.as_query_engine()`을 통해 생성되는 쿼리 엔진의 기본 사용법을 익힙니다.
    - [**09_Retriever_활용.md**](./30_Query_Engines_and_Retrieval/09_Retriever_활용.md): 쿼리 엔진의 하위 컴포넌트인 `Retriever`를 직접 사용하여 검색 과정을 세밀하게 제어하는 방법을 학습합니다.
    - [**10_응답_모드.md**](./30_Query_Engines_and_Retrieval/10_응답_모드.md): `refine`, `compact`, `tree_summarize` 등 검색된 여러 문서를 종합하여 답변을 생성하는 다양한 전략을 비교하고 활용합니다.
    - [**11_서브_쿼리_엔진.md**](./30_Query_Engines_and_Retrieval/11_서브_쿼리_엔진.md): 여러 개의 다른 문서를 각각 인덱싱하고, 복잡한 질문에 대해 각 문서에 맞는 하위 질문을 생성하여 종합적으로 답변하는 고급 기법을 학습합니다.

### Part 4: 고급 검색 및 RAG 최적화 (Advanced Retrieval & RAG Optimization)
- **학습 목표:** 기본적인 RAG를 넘어, 실제 비즈니스 문제 해결에 필요한 고성능 검색 및 RAG 파이프라인 최적화 기법들을 학습합니다.
- **왜 중요한가?** 실제 데이터는 복잡하고 노이즈가 많습니다. 고급 검색 기법과 최적화 전략은 이러한 한계를 극복하고, RAG 시스템의 성능을 프로덕션 수준으로 끌어올리는 데 필수적입니다.
- **주요 내용:**
    - [**12_임베딩_및_재랭킹.md**](./40_Advanced_Retrieval_and_Optimization/12_임베딩_및_재랭킹.md): 임베딩 모델을 교체하고, `SentenceTransformerRerank`와 같은 재랭커를 사용하여 검색 정확도를 극대화하는 방법을 익힙니다.
    - [**13_라우터_쿼리_엔진.md**](./40_Advanced_Retrieval_and_Optimization/13_라우터_쿼리_엔진.md): 사용자의 질문 유형에 따라 다른 인덱스나 쿼리 엔진을 동적으로 선택하는 `RouterQueryEngine`을 구축합니다.
    - [**14_노드_후처리.md**](./40_Advanced_Retrieval_and_Optimization/14_노드_후처리.md): 검색된 노드(문서 조각)에 대해 특정 키워드를 포함하거나, 유사도 점수가 일정 기준 이상인 것만 필터링하는 등 후처리 단계를 추가합니다.
    - [**15_에이전트_기초.md**](./40_Advanced_Retrieval_and_Optimization/15_에이전트_기초.md): LlamaIndex의 쿼리 엔진을 하나의 '도구'로 사용하여 외부 서비스와 상호작용하는 LlamaIndex 에이전트의 기본을 학습합니다.

### Part 5: 프로덕션을 위한 LlamaIndex (LlamaIndex for Production)
- **학습 목표:** 개발된 RAG 시스템의 성능을 체계적으로 관찰 및 평가하고, LlamaParse와 같은 전문 파싱 도구를 활용하며, 최종적으로 API 서버로 배포하는 프로덕션 전 과정을 학습합니다.
- **왜 중요한가?** 성공적인 서비스는 구현에서 끝나지 않습니다. 지속적인 성능 모니터링과 평가를 통해 품질을 유지하고, 안정적인 인프라 위에서 서비스를 제공하는 능력이 비즈니스 성공의 핵심입니다.
- **주요 내용:**
    - [**16_관찰성_및_디버깅.md**](./50_LlamaIndex_for_Production/16_관찰성_및_디버깅.md): 콜백(Callback) 기능을 활용하여 RAG 파이프라인의 각 단계별 소요 시간, 입출력 데이터 등을 추적하고 디버깅하는 방법을 익힙니다.
    - [**17_RAG_평가.md**](./50_LlamaIndex_for_Production/17_RAG_평가.md): LlamaIndex의 `ResponseEvaluator`와 `FaithfulnessEvaluator`를 사용하여 RAG 시스템의 품질을 정량적으로 평가하는 방법을 학습합니다.
    - [**18_LlamaParse_및_보안.md**](./50_LlamaIndex_for_Production/18_LlamaParse_및_보안.md): 복잡한 PDF(테이블, 이미지 포함)를 효과적으로 파싱하는 LlamaParse 서비스를 활용하고, 데이터 처리 시 보안 고려사항을 점검합니다.
    - [**19_API_서버_배포.md**](./50_LlamaIndex_for_Production/19_API_서버_배포.md): FastAPI 등 웹 프레임워크와 연동하여 LlamaIndex 쿼리 엔진을 외부에서 호출할 수 있는 API로 배포하는 방법을 학습합니다.

### Part 6: 고급 인덱스 및 쿼리 전략 (Advanced Indexes & Query Strategies)
- **학습 목표:** `VectorStoreIndex`를 넘어선 다양한 인덱스 유형과, 쿼리 자체를 최적화하여 검색 정확도를 극대화하는 고급 전략을 학습합니다.
- **왜 중요한가?** 실제 데이터는 복잡하고 노이즈가 많습니다. 데이터의 특성과 질문의 의도에 맞는 인덱스 및 쿼리 전략을 선택하는 것은 RAG 시스템의 성능을 한 단계 끌어올리는 핵심 역량입니다.
- **주요 내용:**
    - [**20_다양한_인덱스_유형.md**](./60_Advanced_Indexes_and_Query_Strategies/20_다양한_인덱스_유형.md): `TreeIndex`, `KeywordTableIndex`, `ListIndex`, `CompositeIndex` 등 각 인덱스의 특징과 활용 시나리오를 학습합니다.
    - [**21_대화형_RAG_Chat_Engines.md**](./60_Advanced_Indexes_and_Query_Strategies/21_대화형_RAG_Chat_Engines.md): 챗봇과 같은 대화형 애플리케이션을 위한 `Chat Engines`의 구현 및 활용 방법을 학습합니다.
    - [**22_고급_쿼리_변환.md**](./60_Advanced_Indexes_and_Query_Strategies/22_고급_쿼리_변환.md): `HyDE`, `Multi-query` 등 LLM을 활용한 쿼리 변환 기법을 학습하여 검색 성능을 향상시킵니다.

### Part 7: 지식 그래프 및 고급 에이전트 (Knowledge Graphs & Advanced Agents)
- **학습 목표:** 구조화된 지식 그래프를 RAG에 통합하는 방법과, LlamaIndex의 쿼리 엔진을 활용한 더욱 복잡하고 자율적인 에이전트 패턴을 구축합니다.
- **왜 중요한가?** 비정형 텍스트를 넘어 구조화된 지식과 복잡한 추론을 결합하는 것은 LLM 애플리케이션의 지능을 한 차원 높이는 핵심 기술입니다.
- **주요 내용:**
    - [**23_지식_그래프_RAG.md**](./70_Knowledge_Graphs_and_Advanced_Agents/23_지식_그래프_RAG.md): `Graph Store`를 활용한 지식 그래프 구축 및 `KnowledgeGraphIndex`를 통한 RAG 구현 방법을 학습합니다.
    - [**24_고급_에이전트_패턴.md**](./70_Knowledge_Graphs_and_Advanced_Agents/24_고급_에이전트_패턴.md): LlamaIndex의 쿼리 엔진과 도구를 활용한 복잡한 에이전트 워크플로우 설계 및 구현 방법을 학습합니다.
