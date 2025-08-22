# 딥러닝 마스터 로드맵: 이론부터 실전 프레임워크까지

**복잡한 딥러닝의 세계를 체계적으로 탐험하고, 최신 AI 기술을 마스터하기 위한 종합 가이드**

이 로드맵은 딥러닝의 핵심 이론부터 TensorFlow/Keras, PyTorch와 같은 주요 프레임워크, 그리고 LangChain, LlamaIndex, Hugging Face와 같은 LLM 특화 프레임워크까지, 딥러닝 개발에 필요한 모든 지식을 체계적으로 안내합니다. 각 섹션은 독립적인 학습 경로를 제공하며, 상호 참조를 통해 지식을 확장할 수 있도록 설계되었습니다. 이 여정을 통해 여러분은 딥러닝의 원리를 깊이 이해하고, 실제 AI 애플리케이션을 구축하는 데 필요한 실무 역량을 갖추게 될 것입니다.

---

### Part 1: 딥러닝 이론 (Deep Learning Theory)
- **개요:** 딥러닝 모델의 근본 원리를 이해하기 위한 수학적 기초, 신경망의 구조, 학습 알고리즘, 그리고 모델의 성능을 최적화하는 다양한 기법들을 다룹니다.
- **주요 내용:**
    - [**핵심 개념**](./01_Deep_Learning_Theory/10_Core_Concepts/README.md): 딥러닝 수학 핵심, 딥러닝 소개 및 역사, 신경망 기초, 최적화 이론, 역전파 알고리즘, 활성화 함수, 가중치 초기화, 과적합 방지 및 정규화.
    - [**아키텍처**](./01_Deep_Learning_Theory/20_Architectures/README.md): 합성곱 신경망(CNN), 순환 신경망(RNN, LSTM, GRU), 어텐션 메커니즘과 트랜스포머, 생성 모델(VAE, GAN), 비지도 학습, 자기지도 학습, 강화 학습, 그래프 신경망(GNN) 등 주요 딥러닝 모델 아키텍처.
    - [**평가 및 윤리**](./01_Deep_Learning_Theory/30_Evaluation_and_Ethics/README.md): 모델 평가 및 성능 지표, 모델 해석 가능성(XAI), 확률적 딥러닝 및 불확실성, 딥러닝 윤리 및 편향.

### Part 2: 딥러닝 프레임워크 (Deep Learning Frameworks)
- **개요:** 딥러닝 모델을 효율적으로 구축하고 학습하며 배포하기 위한 주요 프레임워크들의 사용법과 실무적 활용 전략을 다룹니다.
- **주요 내용:**
    - [**DL 프레임워크**](./02_Frameworks/10_DL_Frameworks/README.md):
        - [**TensorFlow & Keras**](./02_Frameworks/10_DL_Frameworks/01_TensorFlow_Keras/README.md): TensorFlow와 Keras의 핵심 개념, 모델 아키텍처 설계, 데이터 파이프라인, 학습/평가/최적화, 모델 배포 및 MLOps, 생태계.
        - [**PyTorch**](./02_Frameworks/10_DL_Frameworks/02_Pytorch/README.md): PyTorch의 기본 개념, 모델 아키텍처 설계, 데이터 파이프라인, 학습/평가/최적화, 모델 배포 및 MLOps, 생태계.
    - [**LLM 프레임워크**](./02_Frameworks/20_LLM_Frameworks/README.md):
        - [**LangChain**](./02_Frameworks/20_LLM_Frameworks/01_Langchain/README.md): LLM 애플리케이션 개발의 핵심 프레임워크, RAG, 에이전트, 프로덕션 생태계, LangGraph.
        - [**LlamaIndex**](./02_Frameworks/20_LLM_Frameworks/02_LlamaIndex/README.md): 데이터와 LLM의 완벽한 연결, RAG 특화 프레임워크, 다양한 인덱스 및 쿼리 전략, 지식 그래프.
        - [**Hugging Face**](./02_Frameworks/20_LLM_Frameworks/03_HuggingFace/README.md): LLM 개발 및 배포의 핵심 플랫폼, Transformers, 모델 학습/파인튜닝, 최적화/배포, 고급 주제.
