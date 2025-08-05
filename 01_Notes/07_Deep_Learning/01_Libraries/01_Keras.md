<h2>Keras 핵심 개념 정리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Keras를 활용한 딥러닝 모델 개발의 전반적인 과정을 실무 관점에서 체계적으로 다룹니다. Keras의 핵심 API를 이용한 모델 구축, 데이터 전처리, 학습 및 평가, 그리고 모델 배포 전략까지, 실제 프로젝트에 바로 적용할 수 있는 실용적인 지식과 팁을 제공하는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. Keras 개요 및 기본 개념](#1-keras-개요-및-기본-개념)
  - [1.1. Keras란? (독립성과 철학)](#11-keras란-독립성과-철학)
  - [1.2. Keras의 멀티 백엔드 지원 (TensorFlow, JAX, PyTorch)](#12-keras의-멀티-백엔드-지원-tensorflow-jax-pytorch)
    - [1.2.1. Keras 3.0 멀티-백엔드의 현실적 측면](#121-keras-30-멀티-백엔드의-현실적-측면)
  - [1.3. 설치 및 환경 설정](#13-설치-및-환경-설정)
  - [1.4. Keras의 주요 구성 요소 (Models, Layers, Optimizers, Losses, Metrics)](#14-keras의-주요-구성-요소-models-layers-optimizers-losses-metrics)
- [2. Keras 모델 아키텍처 설계](#2-keras-모델-아키텍처-설계)
  - [2.1. Sequential API를 이용한 모델 구축](#21-sequential-api를-이용한-모델-구축)
  - [2.2. Functional API를 이용한 복잡한 모델 구축](#22-functional-api를-이용한-복잡한-모델-구축)
  - [2.3. Subclassing API를 이용한 Custom Model 구현](#23-subclassing-api를-이용한-custom-model-구현)
  - [2.4. Custom Layer 및 Custom Loss/Metric 구현](#24-custom-layer-및-custom-lossmetric-구현)
  - [2.5. Keras Applications (사전 학습된 모델)](#25-keras-applications-사전-학습된-모델)
- [3. 데이터 준비 및 Keras 전처리 레이어](#3-데이터-준비-및-keras-전처리-레이어)
  - [3.1. NumPy 배열을 이용한 데이터 준비](#31-numpy-배열을-이용한-데이터-준비)
  - [3.2. Keras 전처리 레이어 (TextVectorization, Image Augmentation 등)](#32-keras-전처리-레이어-textvectorization-image-augmentation-등)
  - [3.3. `tf.data`와의 연동 (Keras 모델 학습을 위한 데이터 파이프라인 구성)](#33-tfdata와의-연동-keras-모델-학습을-위한-데이터-파이프라인-구성)
    - [3.3.1. 현실적인 데이터 처리 및 파이프라인](#331-현실적인-데이터-처리-및-파이프라인)
  - [3.4. 불균형 데이터셋 처리 전략](#34-불균형-데이터셋-처리-전략)
- [4. 모델 학습, 평가 및 예측](#4-모델-학습-평가-및-예측)
  - [4.1. 모델 컴파일 (Optimizer, Loss, Metrics 설정)](#41-모델-컴파일-optimizer-loss-metrics-설정)
  - [4.2. `model.fit()`을 이용한 모델 학습](#42-modelfit을-이용한-모델-학습)
  - [4.3. `model.evaluate()`, `model.predict()`](#43-modelevaluate-modelpredict)
  - [4.4. 콜백 (Callbacks) 활용 (EarlyStopping, ModelCheckpoint, Custom Callbacks)](#44-콜백-callbacks-활용-earlystopping-modelcheckpoint-custom-callbacks)
  - [4.5. 학습률 스케줄러 (Learning Rate Schedulers)](#45-학습률-스케줄러-learning-rate-schedulers)
  - [4.6. Custom Training Loop](#46-custom-training-loop)
  - [4.7. Multi-GPU/TPU 학습 전략](#47-multi-gputpu-학습-전략)
  - [4.8. Gradient Accumulation](#48-gradient-accumulation)
  - [4.9. 혼합 정밀도(Mixed Precision)를 이용한 학습 가속화](#49-혼합-정밀도mixed-precision를-이용한-학습-가속화)
- [5. 모델 저장 및 로드](#5-모델-저장-및-로드)
  - [5.1. Keras 모델 저장 형식 (.keras)](#51-keras-모델-저장-형식-keras)
  - [5.2. 가중치만 저장 및 로드](#52-가중치만-저장-및-로드)
  - [5.3. 모델 아키텍처만 저장 및 로드](#53-모델-아키텍처만-저장-및-로드)
  - [5.4. TensorFlow SavedModel 형식으로 내보내기 (배포를 위한 준비)](#54-tensorflow-savedmodel-형식으로-내보내기-배포를-위한-준비)
- [6. 하이퍼파라미터 튜닝 (KerasTuner)](#6-하이퍼파라미터-튜닝-kerastuner)
  - [6.1. KerasTuner 개요](#61-kerastuner-개요)
  - [6.2. 하이퍼모델 정의 (HyperModel)](#62-하이퍼모델-정의-hypermodel)
  - [6.3. 튜너 (Tuner) 선택 및 실행](#63-튜너-tuner-선택-및-실행)
  - [6.4. 최적의 하이퍼파라미터 검색](#64-최적의-하이퍼파라미터-검색)
- [7. Keras 모델 배포 준비 및 최적화](#7-keras-모델-배포-준비-및-최적화)
  - [7.1. Keras 모델을 TensorFlow Serving, Lite, TF.js 등으로 내보내기](#71-keras-모델을-tensorflow-serving-lite-tfjs-등으로-내보내기)
  - [7.2. ONNX/TensorRT 변환을 통한 추론 최적화](#72-onnxtensorrt-변환을-통한-추론-최적화)
  - [7.3. 모델 최적화 (양자화, 가지치기, 지식 증류)](#73-모델-최적화-양자화-가지치기-지식-증류)
  - [7.4. MLOps 통합 및 고려사항](#74-mlops-통합-및-고려사항)
    - [7.4.1 모델 배포 및 MLOps 심화](#741-모델-배포-및-mlops-심화)
  - [7.4.2. ML 모델을 위한 테스트 및 검증 전략](#742-ml-모델을-위한-테스트-및-검증-전략)
  - [7.5. 체계적인 실험 관리 및 재현성 확보 (MLflow, W\&B 연동)](#75-체계적인-실험-관리-및-재현성-확보-mlflow-wb-연동)
- [8. TensorBoard를 활용한 Keras 모델 시각화 및 디버깅](#8-tensorboard를-활용한-keras-모델-시각화-및-디버깅)
  - [8.1. TensorBoard 개요 및 Keras 연동](#81-tensorboard-개요-및-keras-연동)
  - [8.2. Keras 모델 학습 과정 모니터링](#82-keras-모델-학습-과정-모니터링)
  - [8.3. Keras 모델 그래프 시각화](#83-keras-모델-그래프-시각화)
  - [8.4. Keras Profiler를 이용한 성능 분석](#84-keras-profiler를-이용한-성능-분석)
- [9. 실전 프로젝트 예제 및 팁](#9-실전-프로젝트-예제-및-팁)
  - [9.1. 이미지 처리 모델](#91-이미지-처리-모델)
    - [9.1.1. 이미지 분류 (CNN, ResNet, EfficientNet)](#911-이미지-분류-cnn-resnet-efficientnet)
    - [9.1.2. 객체 탐지 (Object Detection - YOLO, SSD)](#912-객체-탐지-object-detection---yolo-ssd)
    - [9.1.3. 이미지 분할 (Image Segmentation - U-Net, Mask R-CNN)](#913-이미지-분할-image-segmentation---u-net-mask-r-cnn)
    - [9.1.4. 이미지 생성 (GAN, VAE, StyleGAN)](#914-이미지-생성-gan-vae-stylegan)
    - [9.1.5. 초해상도 (Super-resolution)](#915-초해상도-super-resolution)
    - [9.1.6. 스타일 전이 (Style Transfer)](#916-스타일-전이-style-transfer)
    - [9.1.7. 이미지 캡셔닝 (Image Captioning)](#917-이미지-캡셔닝-image-captioning)
  - [9.2. 시퀀스 및 텍스트 처리 모델](#92-시퀀스-및-텍스트-처리-모델)
    - [9.2.1. 텍스트 분류 (RNN, LSTM, GRU)](#921-텍스트-분류-rnn-lstm-gru)
    - [9.2.2. 시계열 예측 (LSTM, Transformer)](#922-시계열-예측-lstm-transformer)
    - [9.2.3. 자연어 처리 (Transformer, BERT, GPT)](#923-자연어-처리-transformer-bert-gpt)
    - [9.2.4. 시퀀스-투-시퀀스 모델 (Seq2Seq, NMT)](#924-시퀀스-투-시퀀스-모델-seq2seq-nmt)
    - [9.2.5. 개체명 인식 (Named Entity Recognition, NER)](#925-개체명-인식-named-entity-recognition-ner)
    - [9.2.6. 질의응답 (Question Answering)](#926-질의응답-question-answering)
    - [9.2.7. 음성 인식 (Speech Recognition)](#927-음성-인식-speech-recognition)
    - [9.2.8. 텍스트 요약 (Text Summarization)](#928-텍스트-요약-text-summarization)
  - [9.3. 그래프 신경망 (Graph Neural Networks, GNN)](#93-그래프-신경망-graph-neural-networks-gnn)
    - [9.3.1. GNN 기본 개념 및 Keras 구현](#931-gnn-기본-개념-및-keras-구현)
    - [9.3.2. 그래프 분류 및 노드 분류](#932-그래프-분류-및-노드-분류)
    - [9.3.3. GCN (Graph Convolutional Networks)](#933-gcn-graph-convolutional-networks)
    - [9.3.4. GAT (Graph Attention Networks)](#934-gat-graph-attention-networks)
  - [9.4. 추천 시스템 (Recommendation Systems)](#94-추천-시스템-recommendation-systems)
    - [9.4.1. 협업 필터링 (Collaborative Filtering)](#941-협업-필터링-collaborative-filtering)
    - [9.4.2. 콘텐츠 기반 필터링 (Content-based Filtering)](#942-콘텐츠-기반-필터링-content-based-filtering)
    - [9.4.3. 하이브리드 추천 시스템](#943-하이브리드-추천-시스템)
  - [9.5. 멀티모달 모델 (Multimodal Models)](#95-멀티모달-모델-multimodal-models)
    - [9.5.1. 이미지-텍스트 융합 (Image-Text Fusion)](#951-이미지-텍스트-융합-image-text-fusion)
    - [9.5.2. 비디오-텍스트 이해 (Video-Text Understanding)](#952-비디오-텍스트-이해-video-text-understanding)
  - [9.6. 기타 고급 모델 및 기법](#96-기타-고급-모델-및-기법)
    - [9.6.1. Autoencoders 및 Variational Autoencoders (VAEs)](#961-autoencoders-및-variational-autoencoders-vaes)
    - [9.6.2. 강화 학습 (Reinforcement Learning) 개요 및 Keras 활용](#962-강화-학습-reinforcement-learning-개요-및-keras-활용)
    - [9.6.3. 자기지도 학습 (Self-supervised Learning)](#963-자기지도-학습-self-supervised-learning)
    - [9.6.4. 확산 모델 (Diffusion Models)](#964-확산-모델-diffusion-models)
    - [9.6.5. 신경망 구조 탐색 (Neural Architecture Search, NAS)](#965-신경망-구조-탐색-neural-architecture-search-nas)
  - [9.7. 최신 연구 및 실무 적용 논의 모델](#97-최신-연구-및-실무-적용-논의-모델)
    - [9.7.1. 대규모 언어 모델 (LLM) 응용 (Fine-tuning, Prompt Engineering)](#971-대규모-언어-모델-llm-응용-fine-tuning-prompt-engineering)
    - [9.7.2. 고급 생성형 AI (Text-to-Image, Video Generation)](#972-고급-생성형-ai-text-to-image-video-generation)
    - [9.7.3. 고급 강화 학습 (Multi-agent RL, Offline RL)](#973-고급-강화-학습-multi-agent-rl-offline-rl)
    - [9.7.4. 인과 관계 추론 모델 (Causal Inference Models)](#974-인과-관계-추론-모델-causal-inference-models)
  - [9.8. 일반적인 팁 및 전략](#98-일반적인-팁-및-전략)
    - [9.8.1. 모델 최적화 및 성능 튜닝 팁](#981-모델-최적화-및-성능-튜닝-팁)
    - [9.8.2. 일반적인 에러 처리 및 디버깅 전략](#982-일반적인-에러-처리-및-디버깅-전략)
    - [9.8.3. 고급 디버깅 기법 (NaN 값, 기울기 문제 해결)](#983-고급-디버깅-기법-nan-값-기울기-문제-해결)
    - [9.8.3.1 고급 디버깅 및 문제 해결 전략](#9831-고급-디버깅-및-문제-해결-전략)
    - [9.8.3.1.1 Eager Execution을 활용한 대화형 디버깅](#98311-eager-execution을-활용한-대화형-디버깅)
    - [9.8.3.2 `tf.print()`를 이용한 그래프 내부 값 확인](#9832-tfprint를-이용한-그래프-내부-값-확인)
    - [9.8.4. Keras 모델의 인터프리터빌리티 (LIME, SHAP)](#984-keras-모델의-인터프리터빌리티-lime-shap)
- [10. Keras 생태계: 도메인 특화 라이브러리](#10-keras-생태계-도메인-특화-라이브러리)
  - [10.1. KerasCV (컴퓨터-비전)](#101-kerascv-컴퓨터-비전)
  - [10.2. KerasNLP (자연어-처리)](#102-kerasnlp-자연어-처리)
  - [10.3. Keras-RL (강화 학습 라이브러리)](#103-keras-rl-강화-학습-라이브러리)
- [11. 책임감 있는 AI (Responsible AI) 및 모델 해석](#11-책임감-있는-ai-responsible-ai-및-모델-해석)
  - [11.1. Explainable AI (XAI) 개요 및 Keras 모델 해석](#111-explainable-ai-xai-개요-및-keras-모델-해석)
  - [11.2. 모델 공정성 (Fairness) 및 편향 감지](#112-모델-공정성-fairness-및-편향-감지)
  - [11.3. 프라이버시 보호 (Differential Privacy)](#113-프라이버시-보호-differential-privacy)
  - [11.4. 모델 견고성 (Robustness)](#114-모델-견고성-robustness)
  - [**12. 지속 가능한 ML 시스템을 위한 설계 원칙과 트레이드오프**](#12-지속-가능한-ml-시스템을-위한-설계-원칙과-트레이드오프)
    - [12.1. 문제 정의와 기술 선택의 기준](#121-문제-정의와-기술-선택의-기준)
    - [12.2. 개발 속도와 시스템 안정성의 균형](#122-개발-속도와-시스템-안정성의-균형)
    - [12.3. '기술 부채'를 경계하는 MLOps 설계](#123-기술-부채를-경계하는-mlops-설계)
    - [12.4. Keras를 넘어: 미래를 위한 제언](#124-keras를-넘어-미래를-위한-제언)

---

## 1. Keras 개요 및 기본 개념

### 1.1. Keras란? (독립성과 철학)

Keras는 딥러닝 모델을 빠르고 쉽게 개발할 수 있도록 설계된 고수준 신경망 API입니다. "인간을 위한 딥러닝"이라는 철학 아래, 사용자 친화적이고 모듈화된 접근 방식을 제공하여 복잡한 딥러닝 모델 구축을 간소화합니다. Keras는 백엔드 엔진(TensorFlow, JAX, PyTorch 등) 위에서 동작하는 추상화 계층으로, 사용자는 백엔드의 복잡성을 직접 다루지 않고도 딥러닝 모델을 구현할 수 있습니다.

**실무 관점:**
*   **생산성 향상:** 직관적인 API와 간결한 코드로 모델을 빠르게 프로토타이핑하고 실험할 수 있어 개발 시간을 단축시킵니다.
*   **쉬운 학습 곡선:** 딥러닝 초보자도 쉽게 접근할 수 있도록 설계되어, 복잡한 개념보다는 모델 구현 자체에 집중할 수 있게 돕습니다.
*   **모듈성 및 재사용성:** 레이어, 모델, 옵티마이저, 손실 함수 등이 독립적인 모듈로 구성되어 있어, 필요에 따라 조합하고 재사용하기 용이합니다. 이는 코드의 유지보수성을 높입니다.
*   **유연성:** 고수준 API를 제공하면서도, 필요에 따라 저수준의 백엔드 연산에 접근할 수 있는 유연성을 제공하여 고급 사용자도 만족시킬 수 있습니다.

### 1.2. Keras의 멀티 백엔드 지원 (TensorFlow, JAX, PyTorch)

Keras의 가장 큰 특징 중 하나는 다양한 딥러닝 프레임워크를 백엔드로 사용할 수 있다는 점입니다. 이는 Keras가 특정 프레임워크에 종속되지 않고, 사용자가 선호하는 백엔드를 선택하여 활용할 수 있도록 합니다.

*   **TensorFlow:** Keras의 기본이자 가장 널리 사용되는 백엔드입니다. TensorFlow 2.x부터 Keras는 TensorFlow의 공식 고수준 API로 통합되어 `tf.keras`로 제공됩니다. TensorFlow의 강력한 분산 학습, SavedModel 형식, TFX 생태계 등을 활용할 수 있습니다.
*   **JAX:** Google에서 개발한 고성능 수치 계산 라이브러리로, 자동 미분 및 XLA 컴파일러를 통한 JIT(Just-In-Time) 컴파일을 지원합니다. Keras 3.0부터 JAX 백엔드를 지원하여 연구 및 고성능 컴퓨팅 환경에서 활용될 수 있습니다.
*   **PyTorch:** Facebook에서 개발한 딥러닝 프레임워크로, 동적 계산 그래프와 Pythonic한 인터페이스로 인해 연구자들 사이에서 인기가 많습니다. Keras 3.0부터 PyTorch 백엔드를 지원하여 PyTorch 생태계의 장점을 Keras에서 활용할 수 있게 되었습니다.

**실무 관점:**
*   **유연한 선택:** 프로젝트 요구사항이나 팀의 숙련도에 따라 최적의 백엔드를 선택할 수 있습니다. 예를 들어, 프로덕션 배포 및 MLOps에 강점이 있는 TensorFlow, 연구 및 고성능 컴퓨팅에 적합한 JAX, 유연한 디버깅 및 연구에 유리한 PyTorch를 선택할 수 있습니다.
*   **코드 재사용성:** 백엔드가 변경되어도 Keras 코드는 거의 동일하게 유지되므로, 다른 프레임워크로의 전환 비용이 낮습니다.
*   **최신 기술 수용:** 각 백엔드 프레임워크의 최신 기능과 성능 최적화를 Keras를 통해 활용할 수 있습니다.

#### 1.2.1. Keras 3.0 멀티-백엔드의 현실적 측면

**개요**: Keras 3.0의 멀티-백엔드 기능을 최대한 활용하기 위한 실질적인 코드 작성법을 추가합니다.

백엔드 중립적 코드 작성을 위한 `keras.ops` 활용

*   **제안 내용**: TensorFlow, PyTorch, JAX 백엔드 간의 완벽한 코드 호환성을 위해, `tf.math`, `torch.nn.functional`과 같은 백엔드 종속적인 함수 대신, Keras 3.0에서 도입된 백엔드 중립적인 연산 라이브러리인 `keras.ops`를 사용해야 함을 강조합니다.
*   **실무적 중요성**: `keras.ops`를 사용하면 단 한 줄의 코드 변경 없이 Keras 코드를 다른 백엔드에서 실행할 수 있습니다. 이는 진정한 프레임워크 독립성을 달성하는 핵심입니다.
*   **삽입 위치**: `01_Keras.md`의 `1.2. Keras의 멀티 백엔드 지원 (TensorFlow, JAX, PyTorch)` 섹션의 실무 관점 파트에 추가합니다.
*   **코드 예제**:
    ```python
    import keras
    import numpy as np

    # 백엔드 종속적인 코드 (TensorFlow)
    # import tensorflow as tf
    # def my_custom_layer_tf(inputs):
    #     return tf.matmul(inputs, tf.transpose(inputs))

    # 백엔드 중립적인 코드 (Keras 3.0)
    def my_custom_layer_keras_ops(inputs):
        # 이 코드는 TensorFlow, PyTorch, JAX 백엔드 모두에서 동작합니다.
        return keras.ops.matmul(inputs, keras.ops.transpose(inputs))

    # 예시 실행
    data = np.random.rand(10, 5).astype("float32")
    output = my_custom_layer_keras_ops(data)
    print("Output shape with keras.ops:", output.shape)
    ```

### 1.3. 설치 및 환경 설정

Keras를 사용하기 위한 설치 및 환경 설정은 선택하는 백엔드에 따라 달라질 수 있습니다. 여기서는 가장 일반적인 TensorFlow 백엔드를 기준으로 설명합니다.

*   **Python 환경:**
    *   **가상 환경:** `conda` 또는 `venv`를 사용하여 독립적인 Python 가상 환경을 구축하는 것이 권장됩니다. 이는 프로젝트 간의 의존성 충돌을 방지하고 환경 관리를 용이하게 합니다.
    *   **설치:**
        *   TensorFlow 백엔드를 사용하는 경우: `pip install tensorflow` (CPU 버전) 또는 `pip install tensorflow[and-cuda]` (GPU 버전, CUDA 및 cuDNN 자동 설치 시도)
        *   Keras 3.0 이상에서 특정 백엔드를 명시적으로 설치하는 경우: `pip install keras` 후 `pip install keras-tensorflow` 또는 `pip install keras-jax` 또는 `pip install keras-pytorch`
*   **GPU 설정 (NVIDIA GPU 기준, TensorFlow 백엔드 사용 시):**
    *   TensorFlow 2.x는 CUDA Toolkit 및 cuDNN 라이브러리와의 호환성이 중요합니다. TensorFlow 공식 문서에서 권장하는 버전을 확인하고 설치해야 합니다.
    *   **도커(Docker) 활용:** 가장 안정적이고 권장되는 방법은 TensorFlow가 미리 설치되고 CUDA/cuDNN 설정이 완료된 공식 TensorFlow Docker 이미지를 사용하는 것입니다. 이는 환경 설정의 번거로움을 줄이고 재현 가능한 개발 환경을 제공합니다.
    *   **클라우드 환경:** AWS SageMaker, Google Cloud AI Platform, Azure Machine Learning 등 클라우드 서비스는 GPU 환경이 미리 구성되어 있어 별도의 설정 없이 바로 딥러닝 개발을 시작할 수 있습니다.

**실무 관점:**
*   **환경 재현성:** `requirements.txt` 또는 `environment.yml` 파일을 사용하여 프로젝트의 모든 의존성을 명시적으로 관리하고, 가상 환경을 사용하여 개발 환경의 재현성을 보장해야 합니다.
*   **GPU 활용:** 대규모 딥러닝 모델 학습에는 GPU가 필수적입니다. GPU 설정이 복잡할 경우, 클라우드 환경이나 Docker를 적극적으로 활용하는 것이 효율적입니다.

### 1.4. Keras의 주요 구성 요소 (Models, Layers, Optimizers, Losses, Metrics)

Keras는 딥러닝 모델을 구성하는 핵심 요소들을 직관적인 API로 제공합니다.

*   **Models (`keras.Model`):**
    *   **개념:** 레이어들을 조직화하여 입력으로부터 출력으로의 변환을 정의하는 객체입니다. Keras는 `Sequential` 모델, `Functional API` 모델, `Subclassing` 모델의 세 가지 방식으로 모델을 구축할 수 있습니다.
    *   **실무 관점:** `model.compile()`, `model.fit()`, `model.evaluate()`, `model.predict()`와 같은 메서드를 통해 모델의 학습, 평가, 예측 과정을 쉽게 관리할 수 있습니다. 모델의 복잡성과 유연성 요구사항에 따라 적절한 모델 구축 방식을 선택합니다.

*   **Layers (`keras.layers`):**
    *   **개념:** 신경망의 기본 빌딩 블록입니다. 입력 텐서를 받아 변환을 수행하고 출력 텐서를 반환합니다. `Dense`, `Conv2D`, `MaxPooling2D`, `LSTM`, `Dropout`, `BatchNormalization` 등 다양한 종류의 레이어를 제공합니다.
    *   **실무 관점:** 각 레이어는 학습 가능한 가중치(weights)를 가질 수 있으며, 모델의 복잡성과 표현력을 결정합니다. 적절한 레이어 선택과 조합은 모델 성능에 큰 영향을 미칩니다.

*   **Optimizers (`keras.optimizers`):**
    *   **개념:** 모델의 가중치를 업데이트하여 손실 함수를 최소화하는 알고리즘입니다. `Adam`, `SGD`, `RMSprop`, `Adagrad` 등 다양한 최적화 알고리즘을 제공합니다.
    *   **실무 관점:** 옵티마이저는 학습률(learning rate)과 같은 하이퍼파라미터를 가집니다. 적절한 옵티마이저와 학습률 선택은 모델의 수렴 속도와 최종 성능에 매우 중요합니다.

*   **Losses (`keras.losses`):**
    *   **개념:** 모델의 예측과 실제 정답 간의 오차를 측정하는 함수입니다. 모델이 학습해야 할 목표를 정의합니다. `CategoricalCrossentropy`, `SparseCategoricalCrossentropy`, `MeanSquaredError`, `BinaryCrossentropy` 등 다양한 손실 함수를 제공합니다.
    *   **실무 관점:** 태스크의 특성(분류, 회귀 등)과 출력 데이터의 형태에 따라 적절한 손실 함수를 선택해야 합니다. 예를 들어, 다중 클래스 분류에는 `CategoricalCrossentropy` 또는 `SparseCategoricalCrossentropy`를, 회귀에는 `MeanSquaredError`를 사용합니다.

*   **Metrics (`keras.metrics`):**
    *   **개념:** 모델의 성능을 평가하는 지표입니다. 손실 함수와 달리, 메트릭은 모델 학습에 직접적으로 사용되지 않지만, 모델의 성능을 사람이 이해하기 쉬운 형태로 보여줍니다. `Accuracy`, `Precision`, `Recall`, `AUC` 등 다양한 메트릭을 제공합니다.
    *   **실무 관점:** 학습 과정에서 모델의 성능 변화를 모니터링하고, 최종 모델의 성능을 평가하는 데 사용됩니다. 비즈니스 목표에 부합하는 메트릭을 선택하는 것이 중요합니다.

## 2. Keras 모델 아키텍처 설계

Keras는 모델을 구축하는 세 가지 주요 방법을 제공하며, 각 방법은 유연성과 사용 편의성 면에서 장단점을 가집니다.

### 2.1. Sequential API를 이용한 모델 구축

가장 간단하고 직관적인 모델 구축 방법으로, 레이어들을 순서대로 쌓아 올리는 선형적인(linear stack) 모델에 적합합니다.

*   **개념:** `keras.Sequential` 클래스를 사용하여 레이어 리스트를 전달하거나, `add()` 메서드를 통해 레이어를 하나씩 추가하여 모델을 정의합니다.
*   **실무 관점:**
    *   **장점:** 코드가 간결하고 이해하기 쉬워 딥러닝 초보자나 간단한 모델을 빠르게 구현할 때 매우 유용합니다.
    *   **단점:** 다중 입력/출력 모델, 공유 레이어, 비선형적인 연결(예: 잔차 연결, 인셉션 모듈)과 같은 복잡한 아키텍처를 표현하기 어렵습니다.
    *   **활용 사례:** 기본적인 MLP(Multi-Layer Perceptron), 간단한 CNN(Convolutional Neural Network), RNN(Recurrent Neural Network) 모델 등에 적합합니다.

```python
import keras
from keras import layers

# Sequential 모델 정의 예시
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(784,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 또는 add() 메서드 사용
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(784,)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

model.summary()
```

### 2.2. Functional API를 이용한 복잡한 모델 구축

Sequential API보다 훨씬 유연하며, 다중 입력/출력, 공유 레이어, 비선형적인 연결 등 복잡한 모델 아키텍처를 구축할 수 있게 합니다.

*   **개념:** 입력 텐서(`keras.Input`)를 정의하고, 이 입력 텐서에 레이어를 함수처럼 적용하여 출력 텐서를 얻는 방식으로 모델을 구성합니다. 최종적으로 입력 텐서와 출력 텐서를 `keras.Model` 클래스에 전달하여 모델을 생성합니다.
*   **실무 관점:**
    *   **장점:** 모델의 구조를 명시적으로 정의하므로 가독성이 좋고, 복잡한 신경망 아키텍처(예: ResNet, Inception, Siamese Network)를 구현하는 데 필수적입니다.
    *   **단점:** Sequential API보다 코드가 길어지고, 개념적으로 약간 더 복잡할 수 있습니다.
    *   **활용 사례:** 대부분의 실제 딥러닝 프로젝트에서 복잡한 모델을 구축할 때 선호되는 방식입니다.

```python
import keras
from keras import layers

# Functional API 모델 정의 예시 (다중 입력/출력)
input_a = keras.Input(shape=(64,), name="branch_a_input")
input_b = keras.Input(shape=(128,), name="branch_b_input")

x = layers.Dense(32, activation="relu")(input_a)
x = layers.Dense(16, activation="relu")(x)

y = layers.Dense(64, activation="relu")(input_b)
y = layers.Dense(16, activation="relu")(y)

# 두 브랜치의 출력을 연결
combined = layers.concatenate([x, y])

# 최종 출력 레이어
output_c = layers.Dense(10, activation="softmax", name="classification_output")(combined)
output_d = layers.Dense(1, activation="sigmoid", name="regression_output")(combined)

model = keras.Model(inputs=[input_a, input_b], outputs=[output_c, output_d])
model.summary()
```

### 2.3. Subclassing API를 이용한 Custom Model 구현

가장 유연한 모델 구축 방법으로, Python 클래스 상속을 통해 모델의 `__init__` 메서드에서 레이어를 정의하고, `call` 메서드에서 순전파 로직을 직접 구현합니다.

*   **개념:** `keras.Model` 클래스를 상속받아 새로운 클래스를 정의하고, `__init__` 메서드에서 필요한 레이어들을 인스턴스화하며, `call(self, inputs)` 메서드에서 입력 텐서가 레이어들을 통과하는 순전파 로직을 작성합니다.
*   **실무 관점:**
    *   **장점:** 가장 높은 수준의 유연성을 제공하여, Keras의 기본 레이어로는 표현하기 어려운 매우 복잡하거나 동적인 모델 아키텍처(예: 조건부 로직, 루프, 재귀적 구조)를 구현할 때 적합합니다. Python의 모든 기능을 활용할 수 있습니다.
    *   **단점:** 디버깅이 더 어렵고, 모델의 구조가 코드에 숨겨져 있어 가독성이 떨어질 수 있습니다. 또한, `model.save()` 시 모델 아키텍처가 아닌 가중치만 저장될 수 있으므로 주의가 필요합니다 (SavedModel로 저장 시 `call` 메서드에 `tf.function` 데코레이터를 적용하는 것이 권장됨).
    *   **활용 사례:** 연구 목적의 새로운 모델 아키텍처를 구현하거나, 기존 Keras API로 표현하기 어려운 특수한 모델에 사용됩니다.

```python
import keras
from keras import layers
import tensorflow as tf

# Subclassing API 모델 정의 예시
class CustomModel(keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.classifier = layers.Dense(num_classes, activation="softmax")

    @tf.function # 성능 최적화 및 SavedModel 저장을 위해 권장
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.classifier(x)

# 모델 인스턴스 생성 및 사용
model = CustomModel(num_classes=10)
# 모델의 입력 형태를 정의하기 위해 더미 데이터로 빌드
model.build(input_shape=(None, 784)) # None은 배치 크기를 의미
model.summary()
```

### 2.4. Custom Layer 및 Custom Loss/Metric 구현

Keras의 기본 레이어나 손실 함수, 메트릭으로는 표현할 수 없는 특수한 연산이나 평가 지표가 필요할 때 사용자 정의 기능을 구현할 수 있습니다.

*   **Custom Layer:**
    *   **개념:** `keras.layers.Layer` 클래스를 상속받아 `__init__`, `build`, `call` 메서드를 오버라이드하여 새로운 레이어를 정의합니다.
        *   `__init__(self, **kwargs)`: 레이어의 하이퍼파라미터 등을 초기화합니다.
        *   `build(self, input_shape)`: 입력 형태가 주어졌을 때 레이어의 가중치(변수)를 생성합니다. `self.add_weight()`를 사용합니다.
        *   `call(self, inputs)`: 입력 텐서에 대한 순전파 연산을 정의합니다.
    *   **실무 관점:** 특정 도메인 지식을 반영한 새로운 연산, 복잡한 어텐션 메커니즘, 특수한 정규화 기법 등을 구현할 때 사용됩니다. 재사용 가능한 컴포넌트를 만들어 코드의 모듈성을 높일 수 있습니다.

```python
import keras
from keras import layers
import tensorflow as tf

class CustomDense(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )
        super().build(input_shape)

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)

# Custom Layer 사용 예시
model = keras.Sequential([
    keras.Input(shape=(784,)),
    CustomDense(64, activation="relu"),
    CustomDense(10, activation="softmax")
])
model.summary()
```

*   **Custom Loss Function:**
    *   **개념:** `keras.losses.Loss` 클래스를 상속받아 `call(self, y_true, y_pred)` 메서드를 오버라이드하거나, 단순히 `y_true`와 `y_pred`를 인자로 받는 함수를 정의합니다.
    *   **실무 관점:** 특정 문제에 최적화된 손실 함수(예: Focal Loss for imbalanced data, IoU Loss for object detection)를 구현할 때 사용됩니다.

```python
import keras
import tensorflow as tf

# Custom Loss Function (함수형)
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Custom Loss Function (클래스형)
class CustomHuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, name="huber_loss"):
        super().__init__(name=name)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * (tf.abs(error) - self.threshold / 2)
        return tf.where(is_small_error, squared_loss, linear_loss)

# 모델 컴파일 시 사용
# model.compile(optimizer="adam", loss=custom_mse)
# model.compile(optimizer="adam", loss=CustomHuberLoss(threshold=0.5))
```

*   **Custom Metric:**
    *   **개념:** `keras.metrics.Metric` 클래스를 상속받아 `__init__`, `update_state`, `result`, `reset_state` 메서드를 오버라이드하여 새로운 메트릭을 정의합니다.
        *   `__init__(self, name='custom_metric', **kwargs)`: 메트릭의 상태 변수를 초기화합니다.
        *   `update_state(self, y_true, y_pred, sample_weight=None)`: 배치별로 메트릭의 상태를 업데이트합니다.
        *   `result(self)`: 현재까지의 상태를 기반으로 최종 메트릭 값을 계산하여 반환합니다.
        *   `reset_state(self)`: 메트릭의 상태를 초기화합니다 (새로운 에포크 시작 시).
    *   **실무 관점:** 특정 도메인에 특화된 평가 지표(예: F1-score for classification, IoU for segmentation)를 학습 과정에서 모니터링할 때 사용됩니다.

```python
import keras
import tensorflow as tf

class F1Score(keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.round(y_pred), tf.bool) # 이진 분류 가정

        tp = tf.cast(tf.reduce_sum(tf.cast(y_true & y_pred, tf.float32)), tf.float32)
        fp = tf.cast(tf.reduce_sum(tf.cast(~y_true & y_pred, tf.float32)), tf.float32)
        fn = tf.cast(tf.reduce_sum(tf.cast(y_true & ~y_pred, tf.float32)), tf.float32)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + keras.backend.epsilon())
        f1 = 2 * ((precision * recall) / (precision + recall + keras.backend.epsilon()))
        return f1

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

# 모델 컴파일 시 사용
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[F1Score()])
```

### 2.5. Keras Applications (사전 학습된 모델)

Keras Applications는 ImageNet과 같은 대규모 데이터셋으로 미리 학습된(pre-trained) 인기 있는 모델 아키텍처들을 제공합니다. 이 모델들은 컴퓨터 비전 태스크에서 강력한 특징 추출기로 사용될 수 있습니다.

*   **개념:** `keras.applications` 모듈에서 `VGG16`, `ResNet50`, `InceptionV3`, `MobileNetV2`, `EfficientNetB0` 등 다양한 CNN 모델을 제공합니다. `weights='imagenet'` 옵션을 통해 ImageNet으로 학습된 가중치를 로드할 수 있습니다.
*   **실무 관점:**
    *   **전이 학습 (Transfer Learning):** 새로운 이미지 분류 태스크를 해결할 때, 처음부터 모델을 학습시키는 대신 사전 학습된 모델을 특징 추출기(feature extractor)로 사용하거나 미세 조정(fine-tuning)하여 성능을 크게 향상시키고 학습 시간을 단축할 수 있습니다.
    *   **데이터 부족 문제 해결:** 특히 데이터셋이 작을 때 사전 학습된 모델은 과적합을 방지하고 일반화 성능을 높이는 데 매우 효과적입니다.
    *   **활용 전략:**
        *   **특징 추출기로 사용:** 사전 학습된 모델의 컨볼루션 베이스(Convolutional Base)를 고정(`trainable=False`)하고, 그 위에 새로운 분류기 레이어를 추가하여 학습합니다.
        *   **미세 조정 (Fine-tuning):** 사전 학습된 모델의 일부 또는 전체 레이어를 새로운 데이터에 맞게 추가적으로 학습합니다. 일반적으로 컨볼루션 베이스의 상위 레이어는 고정하고 하위 레이어는 학습률을 낮춰 미세 조정합니다.
    *   **예시:**

```python
import keras
from keras import layers
from keras.applications import ResNet50

# ImageNet으로 사전 학습된 ResNet50 모델 로드 (최상위 분류 레이어 제외)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 사전 학습된 모델의 가중치를 고정 (특징 추출기로 사용)
base_model.trainable = False

# 새로운 분류기 레이어 추가
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False) # training=False는 BatchNormalization 레이어가 추론 모드로 작동하도록 함
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x) # 10개 클래스 분류

model = keras.Model(inputs, outputs)
model.summary()

# 모델 컴파일 및 학습 (새로운 분류기만 학습)
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model.fit(train_dataset, epochs=10)

# 미세 조정을 위한 모델 설정 (일부 레이어만 학습 가능하게)
# base_model.trainable = True
# for layer in base_model.layers[:-50]: # 마지막 50개 레이어만 학습 가능하게
#     layer.trainable = False
# model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model.fit(train_dataset, epochs=10, initial_epoch=10) # 이어서 학습
```

## 3. 데이터 준비 및 Keras 전처리 레이어

딥러닝 모델 학습의 성공은 데이터 준비에 크게 좌우됩니다. Keras는 다양한 데이터 형식과 전처리 요구사항을 처리할 수 있는 유연한 도구들을 제공합니다.

### 3.1. NumPy 배열을 이용한 데이터 준비

가장 기본적인 데이터 준비 방법으로, 소규모 데이터셋이나 이미 메모리에 로드된 데이터를 학습에 사용할 때 유용합니다.

*   **개념:** 입력 데이터(X)와 타겟 데이터(y)를 NumPy 배열 형태로 준비하여 `model.fit()` 메서드에 직접 전달합니다.
*   **실무 관점:**
    *   **간편성:** 데이터셋의 크기가 작거나, 복잡한 데이터 파이프라인이 필요하지 않을 때 빠르게 모델을 학습시킬 수 있습니다.
    *   **메모리 제약:** 대규모 데이터셋의 경우, 전체 데이터를 메모리에 로드하는 것이 불가능하거나 비효율적일 수 있습니다. 이 경우 `tf.data` API를 사용하는 것이 좋습니다.
    *   **예시:**

```python
import numpy as np
import keras

# 더미 데이터 생성
num_samples = 1000
num_features = 784
num_classes = 10

x_train = np.random.rand(num_samples, num_features).astype("float32")
y_train = np.random.randint(0, num_classes, size=(num_samples,)).astype("int32")

# 모델 정의 (Sequential API 예시)
model = keras.Sequential([
    keras.Input(shape=(num_features,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# NumPy 배열로 모델 학습
# model.fit(x_train, y_train, epochs=10, batch_size=32)
```

### 3.2. Keras 전처리 레이어 (TextVectorization, Image Augmentation 등)

Keras 2.6부터 도입된 전처리 레이어는 모델 내부에 데이터 전처리 로직을 포함할 수 있게 하여, 학습 및 배포 시 일관된 전처리를 보장하고 MLOps 파이프라인을 간소화합니다.

*   **개념:** `keras.layers.TextVectorization`, `keras.layers.Resizing`, `keras.layers.Rescaling`, `keras.layers.RandomFlip`, `keras.layers.RandomRotation` 등 다양한 전처리 레이어를 제공합니다. 이 레이어들은 모델의 첫 번째 레이어로 추가되거나, `tf.data` 파이프라인 내에서 `map()` 함수와 함께 사용될 수 있습니다.
*   **실무 관점:**
    *   **배포 용이성:** 전처리 로직이 모델 그래프의 일부가 되므로, 모델을 SavedModel로 저장할 때 전처리 로직도 함께 저장됩니다. 이는 TensorFlow Serving, TensorFlow Lite 등으로 모델을 배포할 때 별도의 전처리 코드를 관리할 필요 없이 일관된 추론을 가능하게 합니다.
    *   **성능 최적화:** GPU에서 실행될 수 있는 전처리 레이어는 CPU에서 Python 코드로 전처리하는 것보다 효율적일 수 있습니다.
    *   **데이터 증강 (Data Augmentation):** 이미지 분류와 같은 태스크에서 데이터 증강은 모델의 과적합을 방지하고 일반화 성능을 높이는 데 매우 중요합니다. Keras 전처리 레이어를 사용하면 학습 중에 실시간으로 데이터를 증강할 수 있습니다.
    *   **예시:**

```python
import keras
from keras import layers
import tensorflow as tf

# 텍스트 전처리 레이어 예시
text_data = tf.constant(["hello world", "keras is great", "deep learning"])
text_vectorization_layer = layers.TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=20
)
text_vectorization_layer.adapt(text_data) # 데이터에 맞춰 어휘 학습

# 이미지 전처리 및 증강 레이어 예시
image_augmentation_layers = keras.Sequential([
    layers.Resizing(224, 224),
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# 모델에 전처리 레이어 포함
# 텍스트 모델
# inputs = keras.Input(shape=(1,), dtype=tf.string)
# x = text_vectorization_layer(inputs)
# x = layers.Embedding(10000, 128)(x)
# outputs = layers.Dense(1, activation="sigmoid")(x)
# text_model = keras.Model(inputs, outputs)

# 이미지 모델
# inputs = keras.Input(shape=(None, None, 3)) # 가변 크기 이미지 입력
# x = image_augmentation_layers(inputs)
# x = layers.Conv2D(32, 3, activation="relu")(x)
# outputs = layers.Dense(10, activation="softmax")(x)
# image_model = keras.Model(inputs, outputs)
```

### 3.3. `tf.data`와의 연동 (Keras 모델 학습을 위한 데이터 파이프라인 구성)

`tf.data` API는 대규모 데이터셋을 효율적으로 로드하고 전처리하며 모델에 공급하기 위한 강력한 도구입니다. Keras 모델은 `tf.data.Dataset` 객체를 `model.fit()`에 직접 전달하여 학습할 수 있습니다.

*   **개념:** `tf.data.Dataset`은 데이터셋을 나타내는 추상화된 클래스로, `map()`, `batch()`, `shuffle()`, `prefetch()`, `cache()` 등 다양한 변환(transformation)을 지원합니다.
*   **실무 관점:**
    *   **메모리 효율성:** 대용량 데이터셋을 다룰 때, 전체 데이터를 메모리에 로드하는 대신 스트리밍 방식으로 처리하여 메모리 부족 문제를 해결합니다.
    *   **성능 최적화:** 데이터 로드 및 전처리 연산을 병렬화하고 비동기적으로 수행하여 GPU/TPU가 데이터 입력을 기다리는 시간을 최소화합니다. `dataset.prefetch(tf.data.AUTOTUNE)`와 `dataset.cache()`는 필수적인 성능 최적화 기법입니다.
    *   **유연성:** 이미지, 텍스트, 오디오 등 다양한 형식의 데이터를 처리하고, 복잡한 전처리 파이프라인을 구축할 수 있습니다.
    *   **재현성:** 데이터 파이프라인을 코드로 명확하게 정의함으로써, 데이터 전처리 과정의 재현성을 보장하고 MLOps 파이프라인에 쉽게 통합할 수 있습니다. 대규모 프로젝트에서는 데이터의 버전 관리(예: DVC, Pachyderm 등) 및 데이터 거버넌스 전략을 수립하여 데이터의 일관성과 신뢰성을 확보하는 것이 중요합니다.
    *   **예시:**

```python
import tensorflow as tf
import numpy as np
import keras

# 더미 데이터 생성
num_samples = 1000
num_features = 784
num_classes = 10

x_data = np.random.rand(num_samples, num_features).astype("float32")
y_data = np.random.randint(0, num_classes, size=(num_samples,)).astype("int32")

# tf.data.Dataset 생성
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

# 데이터 전처리 및 파이프라인 구성
def preprocess_data(x, y):
    x = x / 255.0 # 예시: 스케일링
    return x, y

dataset = dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE) # 학습 속도 최적화

# 모델 정의
model = keras.Sequential([
    keras.Input(shape=(num_features,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# tf.data.Dataset으로 모델 학습
# model.fit(dataset, epochs=10)
```
#### 3.3.1. 현실적인 데이터 처리 및 파이프라인

**개요**: 실무에서 마주치는 대용량, 비정형, 손상된 데이터를 효과적으로 처리하기 위한 `tf.data` 고급 기법을 추가합니다.

`TFRecord`를 활용한 대용량 데이터셋 처리

*   **제안 내용**: 대용량 데이터셋의 I/O 병목 현상을 줄이기 위해, 데이터를 바이너리 형식으로 직렬화하여 저장하는 `TFRecord` 포맷을 생성하고 `tf.data`로 읽는 방법을 소개합니다.
*   **실무적 중요성**: 여러 개의 작은 파일을 읽는 것보다 하나의 큰 바이너리 파일을 순차적으로 읽는 것이 훨씬 빠릅니다. 특히 클라우드 스토리지 환경에서 학습할 때 성능 향상에 큰 도움이 됩니다.
*   **삽입 위치**: `01_Keras.md`의 `3.3. tf.data와의 연동` 섹션의 심화 내용으로 추가합니다.
*   **코드 예제**:
    ```python
    import tensorflow as tf
    import numpy as np

    # TFRecord 파일 생성 (예시)
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    with tf.io.TFRecordWriter('data.tfrecord') as writer:
        for i in range(5):
            image = np.random.rand(64, 64, 3).astype(np.float32).tobytes()
            label = i
            feature = {
                'image': _bytes_feature(image),
                'label': _int64_feature(label)
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    # TFRecord 파일 읽기 및 파싱
    def parse_tfrecord_fn(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.io.decode_raw(example['image'], tf.float32)
        image = tf.reshape(image, (64, 64, 3))
        return image, example['label']

    dataset = tf.data.TFRecordDataset('data.tfrecord')
    parsed_dataset = dataset.map(parse_tfrecord_fn)
    print(next(iter(parsed_dataset)))
    ```
### 3.4. 불균형 데이터셋 처리 전략

**개요**: 사기 탐지, 의료 진단 등 실제 데이터에서 흔히 발생하는 클래스 불균형 문제를 해결하기 위한 실용적인 방법을 추가합니다.

*   **제안 내용**: `model.fit()`의 `class_weight` 파라미터를 사용하여 학습 시 소수 클래스의 중요도를 높이는 가장 간단하고 효과적인 방법을 소개합니다. `sklearn.utils.class_weight.compute_class_weight`를 활용하여 자동으로 가중치를 계산하는 예제를 포함합니다.
*   **실무적 중요성**: 모델이 다수 클래스만 예측하는 함정에 빠지는 것을 방지하고, F1-score, Precision-Recall AUC 등 실질적인 비즈니스 가치와 연결된 지표를 향상시키는 데 핵심적입니다.
*   **코드 예제**:
    ```python
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    import keras

    # 불균형한 더미 데이터 생성 (클래스 0: 900개, 클래스 1: 100개)
    y_train = np.concatenate([np.zeros(900), np.ones(100)])
    x_train = np.random.rand(1000, 10)

    # Scikit-learn을 사용하여 클래스 가중치 계산
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"계산된 클래스 가중치: {class_weight_dict}")
    # 예시 출력: {0: 0.555..., 1: 5.0}
    # 이는 클래스 1의 손실에 5배의 가중치를 부여함을 의미합니다.

    # 모델 정의 (예시)
    model = keras.Sequential([
        keras.Input(shape=(10,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.fit()에 class_weight 전달
    # model.fit(x_train, y_train, epochs=5, batch_size=32, class_weight=class_weight_dict)
    ```

## 4. 모델 학습, 평가 및 예측

Keras는 모델 학습, 평가, 예측을 위한 직관적이고 강력한 메서드들을 제공합니다.

### 4.1. 모델 컴파일 (Optimizer, Loss, Metrics 설정)

모델을 학습하기 전에 `model.compile()` 메서드를 사용하여 학습 설정을 구성해야 합니다.

*   **개념:**
    *   `optimizer`: 모델의 가중치를 업데이트할 최적화 알고리즘을 지정합니다 (예: `'adam'`, `'sgd'`, `keras.optimizers.Adam()`).
    *   `loss`: 모델의 예측과 실제 정답 간의 오차를 측정할 손실 함수를 지정합니다 (예: `'sparse_categorical_crossentropy'`, `'mse'`, `keras.losses.MeanSquaredError()`).
    *   `metrics`: 학습 및 평가 과정에서 모니터링할 성능 지표를 지정합니다 (예: `'accuracy'`, `'mae'`, `keras.metrics.Precision()`).
*   **실무 관점:**
    *   **적절한 선택:** 태스크의 종류(분류, 회귀 등)와 데이터 특성에 따라 최적의 옵티마이저, 손실 함수, 메트릭을 선택하는 것이 중요합니다.
    *   **학습률:** 옵티마이저의 학습률은 모델 수렴에 큰 영향을 미치므로, 초기 학습률 설정과 학습률 스케줄링 전략을 고려해야 합니다.
    *   **다중 출력/손실:** 모델이 여러 개의 출력을 가지거나 여러 손실 함수를 사용하는 경우, `loss`와 `metrics` 인자에 딕셔너리 또는 리스트 형태로 각 출력에 대한 설정을 지정할 수 있습니다. `loss_weights` 인자를 사용하여 각 손실의 중요도를 조절할 수도 있습니다.
    *   **예시:**

```python
import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])

# 기본적인 컴파일
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 커스텀 옵티마이저, 손실, 메트릭 사용
# model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01),
#               loss=keras.losses.MeanSquaredError(),
#               metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

# 다중 출력 모델 컴파일 예시 (Functional API 모델 가정)
# model_multi_output.compile(
#     optimizer="adam",
#     loss={"classification_output": "sparse_categorical_crossentropy",
#           "regression_output": "mse"},
#     loss_weights={"classification_output": 1.0, "regression_output": 0.5}, # 손실 가중치
#     metrics={"classification_output": ["accuracy"],
#              "regression_output": ["mae"]}
# )
```

### 4.2. `model.fit()`을 이용한 모델 학습

컴파일된 모델을 학습 데이터에 맞춰 훈련시키는 메서드입니다.

*   **개념:** `model.fit(x=None, y=None, batch_size=None, epochs=1, verbose='auto', callbacks=None, validation_data=None, ...)`
    *   `x`, `y`: 학습 데이터와 타겟 데이터 (NumPy 배열 또는 `tf.data.Dataset`).
    *   `batch_size`: 한 번에 처리할 샘플 수.
    *   `epochs`: 전체 데이터셋을 반복할 횟수.
    *   `validation_data`: 검증 데이터셋 (튜플 또는 `tf.data.Dataset`). 학습 중 모델 성능을 모니터링하는 데 사용됩니다.
    *   `callbacks`: 학습 과정에 특정 동작을 수행할 콜백 함수 리스트.
*   **실무 관점:**
    *   **학습 진행 상황 모니터링:** `verbose` 인자를 통해 학습 진행 상황을 출력하고, `validation_data`를 사용하여 과적합 여부를 조기에 파악합니다.
    *   **콜백 활용:** `callbacks` 인자를 적극적으로 활용하여 학습률 조정, 조기 종료, 모델 체크포인트 저장, TensorBoard 로깅 등 다양한 자동화된 작업을 수행합니다.
    *   **데이터 공급:** `tf.data.Dataset`을 사용하는 것이 대규모 데이터셋 학습에 가장 효율적입니다.
    *   **예시:**

```python
import numpy as np
import keras

# 더미 데이터
x_train = np.random.rand(1000, 784).astype("float32")
y_train = np.random.randint(0, 10, size=(1000,)).astype("int32")
x_val = np.random.rand(200, 784).astype("float32")
y_val = np.random.randint(0, 10, size=(200,)).astype("int32")

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 모델 학습
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=32,
#                     validation_data=(x_val, y_val),
#                     verbose=1)
```

### 4.3. `model.evaluate()`, `model.predict()`

학습된 모델의 성능을 평가하고 새로운 데이터에 대한 예측을 수행하는 메서드입니다.

*   **`model.evaluate()`:**
    *   **개념:** 모델의 손실과 메트릭을 계산하여 반환합니다. 학습 데이터와 독립적인 테스트 데이터셋으로 모델의 일반화 성능을 측정하는 데 사용됩니다.
    *   **실무 관점:** 모델 학습이 완료된 후 최종 성능을 확인하는 데 필수적입니다. `verbose=0`으로 설정하여 출력 없이 결과만 얻을 수 있습니다.
    *   **예시:**
        ```python
        # test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        ```

*   **`model.predict()`:**
    *   **개념:** 새로운 입력 데이터에 대한 모델의 예측(출력)을 생성합니다.
    *   **실무 관점:** 모델을 실제 서비스에 적용하기 전에 예측 결과를 확인하거나, 추론 파이프라인의 일부로 사용됩니다. 출력은 NumPy 배열 형태로 반환됩니다.
    *   **예시:**
        ```python
        # new_data = np.random.rand(5, 784).astype("float32")
        # predictions = model.predict(new_data)
        # print(predictions.shape) # (5, 10)
        # print(np.argmax(predictions, axis=1)) # 예측된 클래스
        ```

### 4.4. 콜백 (Callbacks) 활용 (EarlyStopping, ModelCheckpoint, Custom Callbacks)

콜백은 `model.fit()` 메서드에 전달되어 학습 과정의 특정 시점(예: 에포크 시작/종료, 배치 시작/종료)에 호출되는 객체입니다. 학습 과정을 제어하고 모니터링하며, 자동화된 작업을 수행하는 데 매우 유용합니다.

*   **`EarlyStopping`:**
    *   **개념:** 검증 손실(또는 다른 모니터링 메트릭)이 더 이상 개선되지 않을 때 학습을 조기에 중단하여 과적합을 방지합니다.
    *   **실무 관점:** `monitor` (모니터링할 메트릭), `patience` (개선이 없는 에포크 수), `mode` (min/max) 등을 설정하여 사용합니다. `restore_best_weights=True`를 설정하면 가장 좋은 성능을 보였던 시점의 가중치를 복원합니다.
    *   **예시:**
        ```python
        # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # model.fit(..., callbacks=[early_stopping])
        ```

*   **`ModelCheckpoint`:**
    *   **개념:** 학습 중 특정 조건(예: 검증 손실이 최소일 때)을 만족할 때마다 모델의 가중치 또는 전체 모델을 저장합니다.
    *   **실무 관점:** 학습 중단 시 재시작하거나, 학습이 가장 잘 된 시점의 모델을 복원하는 데 사용됩니다. `filepath`, `monitor`, `save_best_only`, `save_weights_only` 등을 설정합니다.
    *   **예시:**
        ```python
        # model_checkpoint = keras.callbacks.ModelCheckpoint(
        #     filepath='best_model.keras', # Keras 3.0부터 권장되는 확장자
        #     monitor='val_loss',
        #     save_best_only=True,
        #     save_weights_only=False, # 전체 모델 저장
        #     verbose=1
        # )
        # model.fit(..., callbacks=[model_checkpoint])
        ```

*   **`TensorBoard`:**
    *   **개념:** 학습 과정의 손실, 메트릭, 그래프, 이미지 등을 TensorBoard로 시각화하기 위한 로그를 생성합니다.
    *   **실무 관점:** 학습 진행 상황을 시각적으로 파악하고, 여러 실험 결과를 비교하며, 모델의 디버깅 및 성능 분석에 필수적입니다.
    *   **예시:**
        ```python
        # tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
        # model.fit(..., callbacks=[tensorboard_callback])
        # 터미널에서 tensorboard --logdir ./logs 실행
        ```

*   **Custom Callbacks:**
    *   **개념:** `keras.callbacks.Callback` 클래스를 상속받아 `on_epoch_begin`, `on_epoch_end`, `on_batch_begin`, `on_batch_end`, `on_train_begin`, `on_train_end` 등 다양한 메서드를 오버라이드하여 사용자 정의 동작을 구현합니다.
    *   **실무 관점:** 학습 중 특정 조건에 따라 동적으로 학습률을 변경하거나, 특정 메트릭을 계산하여 로깅하거나, 외부 시스템과 연동하는 등 복잡한 학습 제어 로직을 구현할 때 사용됩니다.

```python
import keras

class CustomLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"
Epoch {epoch+1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")
        if 'val_loss' in logs:
            print(f"Validation Loss = {logs['val_loss']:.4f}, Validation Accuracy = {logs['val_accuracy']:.4f}")

# model.fit(..., callbacks=[CustomLogger()])
```

### 4.5. 학습률 스케줄러 (Learning Rate Schedulers)

학습률 스케줄러는 학습이 진행됨에 따라 학습률을 동적으로 조정하여 모델의 수렴을 돕고 성능을 향상시키는 기법입니다.

*   **개념:** 초기에는 높은 학습률로 빠르게 수렴하고, 학습이 진행될수록 학습률을 점진적으로 감소시켜 미세 조정을 통해 최적점에 도달하도록 합니다.
*   **실무 관점:**
    *   **`ReduceLROnPlateau`:** 검증 손실이 일정 에포크 동안 개선되지 않을 때 학습률을 감소시킵니다. 가장 널리 사용되는 스케줄러 중 하나입니다.
    *   **`ExponentialDecay`, `PolynomialDecay`, `CosineDecay`:** 미리 정의된 스케줄에 따라 학습률을 감소시킵니다.
    *   **Custom Learning Rate Schedule:** `keras.optimizers.schedules.LearningRateSchedule` 클래스를 상속받아 사용자 정의 스케줄을 구현할 수 있습니다.
    *   **예시:**

```python
import keras
import tensorflow as tf

# ReduceLROnPlateau 콜백 사용
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
# model.fit(..., callbacks=[reduce_lr])

# LearningRateSchedule 사용 (옵티마이저에 직접 전달)
# initial_learning_rate = 0.1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True
# )
# optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
# model.compile(optimizer=optimizer, ...)
```

### 4.6. Custom Training Loop

Keras는 `model.fit()`이라는 고수준 API를 통해 대부분의 학습 시나리오를 커버하지만, 때로는 더 세밀한 제어가 필요한 경우가 있습니다. 이럴 때 Custom Training Loop를 구현할 수 있습니다.

*   **개념:** `tf.GradientTape`를 사용하여 순전파, 손실 계산, 기울기 계산, 가중치 업데이트 등 학습의 모든 단계를 직접 코드로 작성하는 방식입니다. Keras 모델과 레이어는 Custom Training Loop 내에서도 재사용될 수 있습니다.
*   **실무 관점:**
    *   **필요성:** GAN(Generative Adversarial Networks), 강화 학습, 메타 학습 등 복잡한 학습 알고리즘이나 여러 개의 옵티마이저를 사용하는 경우 Custom Training Loop가 필수적입니다.
    *   **Keras와의 연동:** Keras 모델(`keras.Model`)과 레이어(`keras.layers`)는 `tf.Module`을 상속받으므로, `tf.GradientTape`와 함께 사용하여 Custom Training Loop를 구현할 수 있습니다.
    *   **성능 최적화:** Custom Training Loop를 구현할 때는 `tf.function` 데코레이터를 사용하여 성능을 최적화하는 것이 중요합니다.
    *   **참고:** Keras는 `model.train_step`, `model.test_step`, `model.predict_step` 메서드를 오버라이드하여 `model.fit()`의 동작을 커스터마이징하는 방법도 제공합니다. 이는 Custom Training Loop와 `model.fit()`의 중간 지점이라고 볼 수 있습니다.

    **예시 (간단한 Custom Training Loop):**
    ```python
    import tensorflow as tf
    import numpy as np
    import keras

    # 더미 데이터
    x_train_custom = np.random.rand(1000, 784).astype("float32")
    y_train_custom = np.random.randint(0, 10, size=(1000,)).astype("int32")

    # tf.data.Dataset으로 변환
    train_dataset_custom = tf.data.Dataset.from_tensor_slices((x_train_custom, y_train_custom)).batch(32)

    # 모델 정의
    model_custom_loop = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=(784,)),
        keras.layers.Dense(10, activation="softmax")
    ])

    # 옵티마이저와 손실 함수 정의
    optimizer_custom = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn_custom = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # 학습 스텝 함수 정의
    @tf.function
    def train_step_custom(x, y):
        with tf.GradientTape() as tape:
            logits = model_custom_loop(x, training=True)
            loss_value = loss_fn_custom(y, logits)
        grads = tape.gradient(loss_value, model_custom_loop.trainable_variables)
        optimizer_custom.apply_gradients(zip(grads, model_custom_loop.trainable_variables))
        return loss_value

    # 학습 루프 실행
    epochs_custom = 3
    for epoch in range(epochs_custom):
        for batch_idx, (x_batch, y_batch) in enumerate(train_dataset_custom):
            loss_val = train_step_custom(x_batch, y_batch)
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss_val:.4f}")
    print("Custom Training Loop finished.")
    ```

### 4.7. Multi-GPU/TPU 학습 전략

대규모 모델과 데이터셋을 효율적으로 학습하기 위해 여러 GPU, TPU 또는 여러 머신에 걸쳐 학습을 분산하는 기술입니다. Keras는 TensorFlow의 `tf.distribute.Strategy` API를 통해 이를 지원합니다.

*   **개념:** `tf.distribute.Strategy` 객체를 생성하고, 이 전략의 `scope()` 내에서 Keras 모델을 생성하고 컴파일하면, Keras는 자동으로 학습을 분산 처리합니다.
*   **실무 관점:**
    *   **학습 시간 단축:** 대규모 모델 학습 시간을 획기적으로 줄여줍니다.
    *   **대규모 모델 학습:** 단일 디바이스 메모리에 들어가지 않는 모델도 학습할 수 있게 합니다.
    *   **코드 변경 최소화:** 기존 Keras 모델 코드를 최소한으로 변경하여 분산 학습을 적용할 수 있습니다.
    *   **주요 전략:**
        *   `tf.distribute.MirroredStrategy`: 단일 머신 내의 여러 GPU에 데이터 병렬화를 적용합니다. 가장 흔히 사용됩니다.
        *   `tf.distribute.MultiWorkerMirroredStrategy`: 여러 머신에 걸쳐 데이터 병렬화를 적용합니다.
        *   `tf.distribute.TPUStrategy`: Google Cloud TPU에서 학습을 수행하기 위한 전략입니다.
    *   **예시 (`MirroredStrategy`):**

```python
import keras
import tensorflow as tf

# GPU가 여러 개 있는 환경에서 실행
# strategy = tf.distribute.MirroredStrategy()
# print(f"Number of devices: {strategy.num_replicas_in_sync}")

# with strategy.scope():
#     model = keras.Sequential([
#         keras.layers.Dense(64, activation="relu", input_shape=(784,)),
#         keras.layers.Dense(10, activation="softmax")
#     ])
#     model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# model.fit(x_train, y_train, epochs=10, batch_size=strategy.num_replicas_in_sync * 32) # 배치 크기는 복제본 수에 비례하여 증가
```

### 4.8. Gradient Accumulation

메모리 제약으로 인해 큰 배치 크기를 사용할 수 없을 때, 작은 배치로 여러 번 순전파 및 역전파를 수행한 후, 기울기를 누적하여 한 번에 가중치를 업데이트하는 기법입니다. 이는 큰 배치 크기를 사용하는 것과 유사한 효과를 낼 수 있습니다.

*   **개념:** `model.fit()`에서는 직접 지원하지 않으므로, Custom Training Loop를 구현하여 적용해야 합니다. 여러 미니 배치에 대한 기울기를 계산하고, 이들을 합산한 후 옵티마이저를 한 번 호출하여 가중치를 업데이트합니다.
*   **실무 관점:**
    *   **메모리 효율성:** GPU 메모리가 부족하여 원하는 배치 크기를 설정할 수 없을 때 유용합니다.
    *   **성능 유사성:** 큰 배치 크기를 사용하는 것과 유사한 학습 안정성과 성능을 얻을 수 있습니다.
    *   **구현 복잡성:** `model.fit()`을 사용할 수 없으므로 Custom Training Loop를 직접 구현해야 하는 복잡성이 있습니다.
    *   **예시 (개념적 코드):**

```python
import keras
import tensorflow as tf
import numpy as np

# 더미 데이터
x_train = np.random.rand(1000, 784).astype("float32")
y_train = np.random.randint(0, 10, size=(1000,)).astype("int32")

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

gradient_accumulation_steps = 4 # 4개의 미니 배치 기울기를 누적

@tf.function # 성능 최적화를 위해 tf.function으로 래핑 가능
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

# 더미 데이터셋 (실제 사용 시에는 tf.data.Dataset으로 대체)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

epochs = 3 # 예시 에포크 수
for epoch in range(epochs):
    accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
    for i, (x_batch, y_batch) in enumerate(train_dataset):
        loss_value, gradients = train_step(x_batch, y_batch)
        for j in range(len(accumulated_gradients)):
            if gradients[j] is not None: # None gradient 처리
                accumulated_gradients[j] += gradients[j] / gradient_accumulation_steps
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
            accumulated_gradients = [tf.zeros_like(var) for var in model.trainable_variables]
    # 에포크 종료 후 남은 기울기 적용 (선택 사항)
    if any(tf.reduce_sum(tf.abs(g)) > 0 for g in accumulated_gradients if g is not None):
        optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
    print(f"Epoch {epoch+1}, Loss: {loss_value:.4f}")
```
### 4.9. 혼합 정밀도(Mixed Precision)를 이용한 학습 가속화

**개요**: 최신 GPU(NVIDIA Tensor Core)의 성능을 최대로 활용하여 학습 속도를 획기적으로 개선하고 메모리 사용량을 줄이는 혼합 정밀도 학습 방법을 추가합니다.

*   **제안 내용**: `float32`와 `float16` 연산을 혼합하여 학습하는 `Mixed Precision`의 개념과 Keras에서의 설정 방법을 소개합니다. `float16` 사용 시 발생할 수 있는 수치 불안정성을 막기 위한 `Loss Scaling`이 Keras 옵티마이저에 의해 자동으로 처리된다는 점을 강조합니다.
*   **실무적 중요성**: 대규모 모델(LLM, Diffusion Model 등)을 학습하거나, 제한된 시간 안에 많은 실험을 반복해야 할 때 필수적인 기술입니다. 학습 속도를 1.5배에서 3배까지 가속할 수 있습니다.
*   **코드 예제**:
    ```python
    import keras
    import tensorflow as tf # TensorFlow 백엔드 기준

    # 1. 전역 정책 설정 (코드 시작 부분)
    # NVIDIA GPU에서는 'mixed_float16'을, TPU에서는 'mixed_bfloat16'을 사용합니다.
    keras.mixed_precision.set_global_policy('mixed_float16')

    # 2. 모델 정의
    inputs = keras.Input(shape=(784,), name="digits")
    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dense(64, activation="relu")(x)
    # 출력 레이어는 수치 안정성을 위해 float32로 유지하는 것이 좋습니다.
    outputs = keras.layers.Dense(10, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # 3. 모델 컴파일
    # 옵티마이저는 자동으로 Loss Scaling을 처리하는 Wrapper로 감싸집니다.
    # 별도의 추가 설정이 필요 없습니다.
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])

    # 이제 평소와 같이 model.fit()을 호출하면 혼합 정밀도로 학습이 진행됩니다.
    # (더미 데이터)
    # x_train = tf.random.uniform((1000, 784))
    # y_train = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)
    # model.fit(x_train, y_train, epochs=2)
    ```

## 5. 모델 저장 및 로드

학습된 모델을 저장하고 필요할 때 다시 로드하는 것은 딥러닝 워크플로우의 필수적인 부분입니다. Keras는 다양한 저장 형식을 제공합니다.

### 5.1. Keras 모델 저장 형식 (.keras)

Keras 3.0부터 `.keras` 확장자는 Keras 모델을 저장하는 권장 형식입니다. 이 형식은 모델의 아키텍처, 가중치, 컴파일 설정(옵티마이저, 손실, 메트릭), 그리고 사용자 정의 객체까지 모두 포함하는 포괄적인 형식입니다.

*   **개념:** `model.save('my_model.keras')`를 사용하여 모델을 저장하고, `keras.models.load_model('my_model.keras')`를 사용하여 로드합니다.
*   **실무 관점:**
    *   **간편성:** 모델의 모든 정보를 한 파일에 저장하므로, 모델을 다른 환경으로 이동하거나 재사용할 때 매우 편리합니다.
    *   **재현성:** 학습된 모델을 완벽하게 재현할 수 있습니다.
    *   **사용자 정의 객체:** Custom Layer, Custom Loss, Custom Metric 등을 포함하는 모델도 이 형식으로 저장하고 로드할 수 있습니다. 이때 `custom_objects` 인자를 `load_model`에 전달해야 할 수 있습니다.
    *   **예시:**

```python
import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 더미 학습
# x_train = np.random.rand(100, 784).astype("float32")
# y_train = np.random.randint(0, 10, size=(100,)).astype("int32")
# model.fit(x_train, y_train, epochs=1)

# 모델 저장
# model.save('my_model.keras')

# 모델 로드
# loaded_model = keras.models.load_model('my_model.keras')
# loaded_model.summary()
```

### 5.2. 가중치만 저장 및 로드

모델의 아키텍처는 코드에 정의되어 있고, 학습된 가중치만 저장하거나 로드해야 할 때 사용합니다.

*   **개념:** `model.save_weights('my_model_weights.weights.h5')` 또는 `model.save_weights('my_model_weights')` (Keras 3.0부터 권장되는 SavedModel 형식의 가중치 저장)를 사용하여 가중치만 저장하고, `model.load_weights('my_model_weights.weights.h5')` 또는 `model.load_weights('my_model_weights')`를 사용하여 로드합니다.
*   **실무 관점:**
    *   **전이 학습:** 사전 학습된 모델의 가중치를 로드하여 새로운 모델에 적용하거나, 미세 조정을 위해 사용될 때 유용합니다.
    *   **체크포인트:** 학습 중 주기적으로 가중치를 저장하여 학습 중단 시 재시작하거나, 가장 좋은 성능을 보인 시점의 가중치를 복원하는 데 사용됩니다. `keras.callbacks.ModelCheckpoint` 콜백과 함께 자주 사용됩니다.
    *   **예시:**

```python
import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 더미 학습
# x_train = np.random.rand(100, 784).astype("float32")
# y_train = np.random.randint(0, 10, size=(100,)).astype("int32")
# model.fit(x_train, y_train, epochs=1)

# 가중치만 저장 (HDF5 형식)
# model.save_weights('my_model_weights.weights.h5')

# 새로운 모델 인스턴스 생성 후 가중치 로드
# new_model = keras.Sequential([
#     keras.layers.Dense(64, activation="relu", input_shape=(784,)),
#     keras.layers.Dense(10, activation="softmax")
# ])
# new_model.load_weights('my_model_weights.weights.h5')
# new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

### 5.3. 모델 아키텍처만 저장 및 로드

모델의 구조(레이어 구성)만 저장하고 로드해야 할 때 사용합니다. 가중치는 포함되지 않습니다.

*   **개념:** `model.to_json()` 또는 `model.to_yaml()`을 사용하여 모델 아키텍처를 JSON 또는 YAML 문자열로 저장하고, `keras.models.model_from_json()` 또는 `keras.models.model_from_yaml()`을 사용하여 로드합니다.
*   **실무 관점:**
    *   **경량화된 저장:** 모델의 가중치 없이 구조만 공유하거나 저장할 때 유용합니다.
    *   **재현성:** 모델 아키텍처를 텍스트 파일로 저장하여 버전 관리 시스템에 포함시키기 용이합니다.
    *   **주의:** 이 방법은 `Sequential` 및 `Functional API`로 구축된 모델에만 작동합니다. `Subclassing API`로 구축된 모델은 Python 코드에 아키텍처가 정의되어 있으므로 이 방법으로 저장할 수 없습니다.
    *   **예시:**

```python
import keras

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])

# 모델 아키텍처를 JSON 문자열로 저장
# json_config = model.to_json()
# print(json_config)

# JSON 문자열로부터 모델 로드
# loaded_model_from_json = keras.models.model_from_json(json_config)
# loaded_model_from_json.summary()
```

### 5.4. TensorFlow SavedModel 형식으로 내보내기 (배포를 위한 준비)

Keras 모델은 TensorFlow의 SavedModel 형식으로 내보낼 수 있습니다. SavedModel은 TensorFlow 모델을 프로덕션 환경에 배포하기 위한 표준 형식입니다.

*   **개념:** `model.save('my_saved_model', save_format='tf')`를 사용하여 SavedModel 형식으로 저장합니다. 이 형식은 모델의 아키텍처, 가중치, 컴파일 설정, 그리고 `tf.function`으로 컴파일된 연산 그래프까지 모두 포함합니다.
*   **실무 관점:**
    *   **배포 표준:** TensorFlow Serving, TensorFlow Lite, TensorFlow.js 등 다양한 TensorFlow 생태계 도구를 통해 모델을 배포할 때 필수적으로 사용됩니다.
    *   **언어 독립적:** Python 환경 없이도 로드하고 실행할 수 있습니다.
    *   **서명(Signature):** 모델의 입력 및 출력 텐서의 형태와 데이터 타입을 정의하는 서명을 포함하여, 모델을 쉽게 호출할 수 있도록 합니다.
    *   **`Subclassing` 모델:** `Subclassing API`로 구축된 모델의 경우, `call` 메서드에 `tf.function` 데코레이터를 적용해야 SavedModel로 올바르게 저장되고 추론 시 성능 최적화를 얻을 수 있습니다.
    *   **예시:**

```python
import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 더미 학습
x_train = np.random.rand(100, 784).astype("float32")
y_train = np.random.randint(0, 10, size=(100,)).astype("int32")
model.fit(x_train, y_train, epochs=1)

# SavedModel 형식으로 저장
model.save('my_saved_model_directory', save_format='tf')

# SavedModel 로드 (TensorFlow API 사용)
import tensorflow as tf
loaded_saved_model = tf.saved_model.load('my_saved_model_directory')
print(loaded_saved_model.signatures['serving_default'])
```

## 6. 하이퍼파라미터 튜닝 (KerasTuner)

하이퍼파라미터 튜닝은 모델의 성능을 최적화하기 위해 학습률, 배치 크기, 레이어 수, 뉴런 수 등 모델의 구조나 학습 과정에 영향을 미치는 파라미터들을 탐색하는 과정입니다. KerasTuner는 Keras 모델을 위한 강력하고 사용하기 쉬운 하이퍼파라미터 튜닝 라이브러리입니다.

### 6.1. KerasTuner 개요

*   **개념:** KerasTuner는 하이퍼파라미터 탐색 공간을 정의하고, 다양한 탐색 알고리즘(RandomSearch, Hyperband, BayesianOptimization)을 사용하여 최적의 하이퍼파라미터 조합을 찾아주는 도구입니다.
*   **실무 관점:**
    *   **자동화:** 수동으로 하이퍼파라미터를 변경하며 실험하는 번거로움을 줄여줍니다.
    *   **효율성:** 체계적인 탐색 알고리즘을 통해 더 적은 시도로 더 좋은 성능의 모델을 찾을 수 있습니다.
    *   **재현성:** 튜닝 과정을 코드로 관리하여 실험의 재현성을 높입니다.
    *   **설치:** `pip install keras-tuner`

### 6.2. 하이퍼모델 정의 (HyperModel)

KerasTuner를 사용하려면 하이퍼파라미터에 따라 모델을 빌드하는 함수 또는 클래스를 정의해야 합니다.

*   **개념:**
    *   **함수형:** `build_model(hp)`와 같이 `hp` (HyperParameters) 객체를 인자로 받아 모델을 정의하고 컴파일하는 함수를 작성합니다. `hp.Int()`, `hp.Float()`, `hp.Choice()` 등을 사용하여 탐색할 하이퍼파라미터 범위를 지정합니다.
    *   **클래스형 (`HyperModel`):** `keras_tuner.HyperModel` 클래스를 상속받아 `build(self, hp)` 메서드를 오버라이드하여 모델을 정의합니다.
*   **실무 관점:**
    *   **탐색 공간 정의:** 어떤 하이퍼파라미터를 탐색할지, 그리고 각 하이퍼파라미터의 탐색 범위(최소/최대 값, 선택지)를 명확하게 정의하는 것이 중요합니다.
    *   **조건부 하이퍼파라미터:** 특정 하이퍼파라미터의 값에 따라 다른 하이퍼파라미터가 활성화되도록 조건부 로직을 구현할 수 있습니다.
    *   **예시:**

```python
import keras
from keras import layers
import keras_tuner as kt

# 함수형 HyperModel 정의
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(784,)))

    # 은닉층의 뉴런 수 탐색 (32, 64, 128 중 선택)
    hp_units = hp.Choice('units', values=[32, 64, 128])
    model.add(layers.Dense(units=hp_units, activation='relu'))

    # 은닉층의 개수 탐색 (1 또는 2)
    if hp.Boolean("add_second_dense_layer"):
        hp_units_2 = hp.Choice('units_2', values=[32, 64, 128])
        model.add(layers.Dense(units=hp_units_2, activation='relu'))

    # 학습률 탐색 (로그 스케일)
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 클래스형 HyperModel 정의 (더 복잡한 시나리오에 적합)
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(784,)))
        
        hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(layers.Dense(units=hp_units, activation='relu'))

        if hp.Boolean("add_dropout"):
            hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
            model.add(layers.Dropout(rate=hp_dropout_rate))

        model.add(layers.Dense(10, activation='softmax'))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
```

### 6.3. 튜너 (Tuner) 선택 및 실행

하이퍼모델을 정의한 후, 어떤 탐색 알고리즘을 사용할지 튜너를 선택하고 실행합니다.

*   **개념:**
    *   **`RandomSearch`:** 하이퍼파라미터 공간에서 무작위로 조합을 선택하여 시도합니다. 간단하고 빠르게 시작할 수 있습니다.
    *   **`Hyperband`:** 리소스(예: 에포크 수)를 효율적으로 할당하여 비효율적인 모델을 조기에 중단하고 유망한 모델에 더 많은 리소스를 할당합니다. RandomSearch보다 효율적입니다.
    *   **`BayesianOptimization`:** 이전 실험 결과를 바탕으로 다음 실험할 하이퍼파라미터 조합을 예측하여 탐색 효율을 높입니다.
*   **실무 관점:**
    *   **탐색 알고리즘 선택:** 초기 탐색에는 `RandomSearch`나 `Hyperband`가 좋고, 더 정교한 탐색이 필요할 때는 `BayesianOptimization`을 고려할 수 있습니다.
    *   **`objective`:** 튜닝의 목표가 되는 메트릭(예: `val_accuracy`, `val_loss`)과 `direction` (최대화 'max' 또는 최소화 'min')을 지정합니다.
    *   **`max_trials`:** 시도할 모델의 최대 개수를 지정합니다.
    *   **`executions_per_trial`:** 각 하이퍼파라미터 조합에 대해 모델을 몇 번 학습시킬지 지정합니다. (모델 학습의 변동성을 줄이기 위함)
    *   **`directory`, `project_name`:** 튜닝 결과를 저장할 디렉토리와 프로젝트 이름을 지정합니다.
    *   **예시:**

```python
import numpy as np
import keras_tuner as kt

# 더미 데이터
x_train = np.random.rand(1000, 784).astype("float32")
y_train = np.random.randint(0, 10, size=(1000,)).astype("int32")
x_val = np.random.rand(200, 784).astype("float32")
y_val = np.random.randint(0, 10, size=(200,)).astype("int32")

# 튜너 인스턴스 생성 및 실행
tuner = kt.RandomSearch(
    hypermodel=build_model, # 또는 MyHyperModel()
    objective='val_accuracy',
    max_trials=10, # 최대 10개의 다른 모델 조합 시도
    executions_per_trial=2, # 각 조합을 2번 학습시켜 평균 성능 측정
    directory='my_dir',
    project_name='intro_to_kt'
)

# 튜닝 시작
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
```

### 6.4. 최적의 하이퍼파라미터 검색

튜닝이 완료되면, KerasTuner는 탐색된 모델 중 최적의 성능을 보인 모델과 해당 하이퍼파라미터 조합을 제공합니다.

*   **개념:** `tuner.get_best_hyperparameters()` 메서드를 사용하여 최적의 하이퍼파라미터 조합을 얻고, `tuner.get_best_models()` 메서드를 사용하여 최적의 모델을 로드합니다.
*   **실무 관점:**
    *   **최적 모델 로드:** 튜닝을 통해 얻은 최적의 하이퍼파라미터로 모델을 다시 빌드하고, 전체 데이터셋(학습 + 검증)으로 최종 학습을 수행하여 프로덕션에 배포할 모델을 준비합니다.
    *   **결과 분석:** `tuner.results_summary()`를 통해 각 시도의 결과와 하이퍼파라미터 조합을 요약하여 볼 수 있습니다. 이를 통해 어떤 하이퍼파라미터가 모델 성능에 큰 영향을 미치는지 분석할 수 있습니다.
    *   **예시 (KerasTuner End-to-End 워크플로우):**

```python
import numpy as np
import keras
from keras import layers
import keras_tuner as kt

# 1. 데이터 준비 (더미 데이터)
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_full = x_train_full.reshape(-1, 784).astype("float32") / 255.0
y_train_full = y_train_full.astype("int32")
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
y_test = y_test.astype("int32")

# 학습 데이터와 검증 데이터 분리
x_train, x_val = x_train_full[:-10000], x_train_full[-10000:]
y_train, y_val = y_train_full[:-10000], y_train_full[-10000:]

# 2. HyperModel 정의 (함수형)
def build_model_for_tuning(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(784,)))

    # 은닉층의 뉴런 수 탐색 (32, 64, 128 중 선택)
    hp_units = hp.Choice('units', values=[32, 64, 128])
    model.add(layers.Dense(units=hp_units, activation='relu'))

    # 두 번째 은닉층 추가 여부 및 뉴런 수 탐색
    if hp.Boolean("add_second_dense_layer"):
        hp_units_2 = hp.Choice('units_2', values=[32, 64, 128])
        model.add(layers.Dense(units=hp_units_2, activation='relu'))

    # 학습률 탐색 (로그 스케일)
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. 튜너 (Tuner) 선택 및 실행
tuner = kt.RandomSearch(
    hypermodel=build_model_for_tuning,
    objective='val_accuracy',
    max_trials=10, # 최대 10개의 다른 모델 조합 시도
    executions_per_trial=2, # 각 조합을 2번 학습시켜 평균 성능 측정 (변동성 감소)
    directory='keras_tuner_demo',
    project_name='mnist_tuning'
)

print("\n--- KerasTuner Search Space Summary ---")
tuner.search_space_summary()

print("\n--- Starting KerasTuner Search ---")
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# 4. 최적의 하이퍼파라미터 및 모델 검색
print("\n--- KerasTuner Results Summary ---")
tuner.results_summary()

# 최적의 하이퍼파라미터 얻기
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"\nOptimal units for first dense layer: {best_hps.get('units')}")
if best_hps.get("add_second_dense_layer"):
    print(f"Optimal units for second dense layer: {best_hps.get('units_2')}")
print(f"Optimal learning rate: {best_hps.get('learning_rate')}")

# 최적의 모델 얻기
best_model = tuner.get_best_models(num_models=1)[0]
print("\n--- Best Model Summary ---")
best_model.summary()

# 최적의 모델로 최종 학습 (전체 학습 데이터셋 사용)
print("\n--- Final Training with Best Model ---")
# EarlyStopping 콜백 추가
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = best_model.fit(x_train_full, y_train_full, 
                         epochs=50, 
                         validation_split=0.1, # 전체 데이터셋에서 10%를 검증용으로 사용
                         callbacks=[early_stopping])

# 최종 모델 평가
print("\n--- Final Model Evaluation on Test Set ---")
loss, accuracy = best_model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
```

## 7. Keras 모델 배포 준비 및 최적화

학습된 Keras 모델을 실제 서비스 환경에 배포하기 위해서는 모델을 특정 배포 환경에 맞게 변환하고 최적화하는 과정이 필요합니다.

### 7.1. Keras 모델을 TensorFlow Serving, Lite, TF.js 등으로 내보내기

Keras 모델은 TensorFlow의 백엔드를 사용하므로, TensorFlow의 다양한 배포 도구와 호환됩니다.

*   **TensorFlow Serving:**
    *   **개념:** 프로덕션 환경에서 고성능으로 모델을 서빙하기 위한 시스템입니다. Keras 모델을 SavedModel 형식으로 저장하여 TensorFlow Serving에 배포합니다.
    *   **실무 관점:** `model.save('path/to/saved_model', save_format='tf')`를 사용하여 SavedModel로 내보냅니다. TensorFlow Serving은 모델 버전 관리, 배치 처리 등을 지원하여 대규모 서비스에 적합합니다.
        *   **배포 다음 단계:** SavedModel로 저장된 모델은 Docker 컨테이너나 Kubernetes 클러스터를 통해 TensorFlow Serving으로 배포할 수 있습니다. 예를 들어, Docker를 사용하여 로컬에서 모델을 서빙하는 명령어는 다음과 같습니다.
            ```bash
            # SavedModel이 /tmp/my_model/1 에 저장되어 있다고 가정
            # (1은 모델 버전 번호)
            docker run -p 8501:8501 --name tfserving_model \
              --mount type=bind,source=/tmp/my_model,target=/models/my_model \
              -e MODEL_NAME=my_model -t tensorflow/serving &
            
            # 모델 추론 요청 예시 (Python requests)
            # import requests
            # import json
            # data = json.dumps({"signature_name": "serving_default", "instances": input_data.tolist()})
            # headers = {"content-type": "application/json"}
            # json_response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data, headers=headers)
            # predictions = json.loads(json_response.text) 
            ```

*   **TensorFlow Lite:**
    *   **개념:** 모바일, 임베디드, 엣지 디바이스에서 머신러닝 모델을 실행하기 위한 경량 솔루션입니다. Keras 모델을 `.tflite` 형식으로 변환하여 배포합니다.
    *   **실무 관점:** `tf.lite.TFLiteConverter`를 사용하여 Keras 모델을 `.tflite` 형식으로 변환합니다. 이 과정에서 양자화(Quantization)를 적용하여 모델 크기를 줄이고 추론 속도를 높일 수 있습니다. 온디바이스 ML에 필수적입니다.
        *   **변환 및 배포 다음 단계:**
            ```python
            import tensorflow as tf
            # model은 학습된 Keras 모델
            # converter = tf.lite.TFLiteConverter.from_keras_model(model)
            # tflite_model = converter.convert()
            # with open('model.tflite', 'wb') as f:
            #     f.write(tflite_model)
            ```
            변환된 `.tflite` 모델은 Android, iOS, Raspberry Pi 등 다양한 플랫폼의 애플리케이션에 통합하여 사용할 수 있습니다. 각 플랫폼별 TensorFlow Lite 인터프리터 API를 사용하여 모델을 로드하고 추론을 실행합니다.

*   **TensorFlow.js:**
    *   **개념:** 웹 브라우저나 Node.js 환경에서 머신러닝 모델을 개발하고 실행하기 위한 JavaScript 라이브러리입니다. Keras 모델을 TensorFlow.js 형식으로 변환하여 웹 애플리케이션에 포함합니다.
    *   **실무 관점:** `tensorflowjs_converter` 도구를 사용하여 Keras 모델을 TensorFlow.js 형식으로 변환합니다. 클라이언트 측 ML을 통해 개인 정보 보호, 낮은 지연 시간, 서버 부하 감소 등의 이점을 얻을 수 있습니다.
        *   **변환 및 배포 다음 단계:**
            ```bash
            # Keras 모델을 SavedModel 형식으로 먼저 저장
            # model.save('my_tf_model', save_format='tf')
            
            # TensorFlow.js 형식으로 변환
            # pip install tensorflowjs
            # tensorflowjs_converter --input_format=tf_saved_model \
            #                        --output_node_names='output_node_name' \
            #                        --output_format=tfjs_graph_model \
            #                        ./my_tf_model ./web_model
            ```
            변환된 모델은 웹 페이지에서 `<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>`를 통해 TensorFlow.js 라이브러리를 로드한 후, JavaScript 코드로 모델을 로드하고 추론을 실행할 수 있습니다.



### 7.2. ONNX/TensorRT 변환을 통한 추론 최적화

TensorFlow 생태계 외의 다른 추론 엔진이나 하드웨어 가속기를 활용하기 위해 모델을 다른 형식으로 변환할 수 있습니다.

*   **ONNX (Open Neural Network Exchange):**
    *   **개념:** 딥러닝 모델을 표현하기 위한 개방형 표준 형식입니다. 다양한 프레임워크(PyTorch, TensorFlow, Keras 등)에서 학습된 모델을 ONNX 형식으로 변환하여 다른 프레임워크나 추론 엔진(ONNX Runtime)에서 실행할 수 있게 합니다.
    *   **실무 관점:** 모델의 프레임워크 종속성을 줄이고, 다양한 배포 환경에 유연하게 대응할 수 있게 합니다. `tf2onnx`와 같은 도구를 사용하여 TensorFlow/Keras 모델을 ONNX로 변환할 수 있습니다.
*   **TensorRT:**
    *   **개념:** NVIDIA GPU에서 딥러닝 추론을 최적화하기 위한 SDK입니다. 모델을 TensorRT 엔진으로 컴파일하여 GPU에서의 추론 성능을 극대화합니다.
    *   **실무 관점:** NVIDIA GPU 환경에서 최고 수준의 추론 성능이 요구될 때 사용됩니다. TensorFlow는 `tf.saved_model.experimental.build_signature_def`와 `tf.experimental.tensorrt.Converter`를 통해 TensorRT 통합을 지원합니다.

### 7.3. 모델 최적화 (양자화, 가지치기, 지식 증류)

모델의 크기를 줄이고 추론 속도를 높여 배포 환경에서의 효율성을 극대화하는 기법들입니다.

*   **양자화 (Quantization):**
    *   **개념:** 모델의 가중치와 활성화 값을 `float32`에서 `float16`, `int8` 등 더 낮은 비트 정밀도로 변환하여 모델 크기를 줄이고 추론 속도를 높이는 기법입니다.
    *   **실무 관점:** 모바일, 엣지 디바이스 등 리소스가 제한된 환경에 모델을 배포할 때 필수적입니다. TensorFlow Lite Converter는 학습 후 양자화(Post-training Quantization) 및 양자화 인식 학습(Quantization-aware Training)을 지원합니다.
*   **가지치기 (Pruning):**
    *   **개념:** 모델의 중요하지 않은 가중치(예: 값이 0에 가까운 가중치)를 제거하여 모델의 희소성(sparsity)을 높이고 크기를 줄이는 기법입니다.
    *   **실무 관점:** 모델 크기를 줄이고 추론 속도를 향상시킬 수 있지만, 양자화와 마찬가지로 정확도 손실이 발생할 수 있습니다. TensorFlow Model Optimization Toolkit에서 가지치기 기능을 제공합니다.
*   **지식 증류 (Knowledge Distillation):**
    *   **개념:** 크고 복잡한 "교사(Teacher)" 모델의 지식(예측 분포)을 작고 효율적인 "학생(Student)" 모델에게 전달하여, 학생 모델이 교사 모델과 유사한 성능을 내도록 학습시키는 기법입니다.
    *   **실무 관점:** 배포 환경에서 리소스 제약이 있을 때, 작은 모델로 큰 모델의 성능을 모방하여 효율적인 배포를 가능하게 합니다. Keras에서는 Custom Training Loop를 통해 구현할 수 있습니다.

*   **코드 예제**:
    ```python
    import keras
    import tensorflow as tf

    class Distiller(keras.Model):
        def __init__(self, student, teacher):
            super().__init__()
            self.student = student
            self.teacher = teacher

        def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=3):
            super().compile(optimizer=optimizer, metrics=metrics)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha
            self.temperature = temperature

        def train_step(self, data):
            x, y = data

            # 교사 모델은 추론 모드로, 가중치가 업데이트되지 않음
            teacher_predictions = self.teacher(x, training=False)

            with tf.GradientTape() as tape:
                # 학생 모델은 학습 모드로 순전파
                student_predictions = self.student(x, training=True)

                # 1. 학생 모델의 실제 레이블에 대한 손실 (기본 손실)
                student_loss = self.student_loss_fn(y, student_predictions)

                # 2. 교사 모델의 예측(soft label)에 대한 손실 (증류 손실)
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                # 최종 손실 = 기본 손실 + (alpha * 증류 손실)
                loss = student_loss * (1 - self.alpha) + distillation_loss * self.alpha

            # 학생 모델의 가중치 업데이트
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # 메트릭 업데이트
            self.compiled_metrics.update_state(y, student_predictions)
            return {m.name: m.result() for m in self.metrics}

    # --- 사용 예시 ---
    # 1. 교사/학생 모델 생성
    teacher = keras.Sequential([...]) # 크고 복잡한 모델
    student = keras.Sequential([...]) # 작고 효율적인 모델

    # 2. Distiller 인스턴스화 및 컴파일
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    # 3. model.fit()으로 학습
    # distiller.fit(x_train, y_train, epochs=5)
    ```


### 7.4. MLOps 통합 및 고려사항

MLOps (Machine Learning Operations)는 머신러닝 모델의 개발부터 배포, 운영, 모니터링까지 전체 라이프사이클을 자동화하고 관리하는 프로세스입니다. Keras 모델을 실무에 적용할 때는 다음과 같은 MLOps 관점을 고려해야 합니다.

*   **모델 버전 관리:** 학습된 모델과 그에 해당하는 코드, 데이터셋을 체계적으로 관리하여 재현성을 보장합니다. Git (코드), DVC (데이터), MLflow (모델) 등의 도구를 활용할 수 있습니다.
*   **데이터 파이프라인 자동화:** 데이터 수집, 전처리, 검증 과정을 자동화하여 모델 학습에 필요한 데이터를 안정적으로 공급합니다. Apache Airflow, Kubeflow Pipelines 등이 사용될 수 있습니다.
*   **학습 파이프라인 자동화:** 모델 학습 과정을 자동화하고, 하이퍼파라미터 튜닝, 분산 학습 등을 효율적으로 관리합니다. Kubeflow Training Operator, Vertex AI Training 등이 활용됩니다.
*   **모델 배포 자동화:** 학습된 모델을 다양한 환경(온프레미스, 클라우드, 엣지 디바이스)에 자동으로 배포하고 업데이트합니다. TensorFlow Serving, Kubeflow Serving, SageMaker Endpoints 등이 사용됩니다.
*   **모델 모니터링:** 배포된 모델의 성능(정확도, 지연 시간 등)과 데이터 드리프트(Data Drift), 모델 드리프트(Model Drift) 등을 지속적으로 모니터링하여 모델 재학습 필요성을 감지합니다. Prometheus, Grafana, Arize AI 등이 활용될 수 있습니다.
*   **테스트 및 검증:** 모델의 기능적 정확성뿐만 아니라, 공정성, 견고성, 성능 등 비기능적 요구사항에 대한 테스트 및 검증을 자동화합니다.

Keras는 TensorFlow 생태계의 일부이므로, TensorFlow Extended (TFX)와 같은 MLOps 플랫폼과 긴밀하게 통합될 수 있습니다. TFX는 데이터 검증, 특징 변환, 모델 학습, 모델 평가, 모델 서빙 등 ML 파이프라인의 모든 단계를 위한 컴포넌트를 제공합니다.

#### 7.4.1 모델 배포 및 MLOps 심화

**개요**: MLOps의 실질적인 구현을 돕기 위해, 설정 관리와 CI/CT/CD 파이프라인의 구체적인 예시를 추가합니다.

설정 파일(YAML/JSON)을 이용한 하이퍼파라미터 관리

*   **제안 내용**: 학습률, 배치 크기, 모델 구조 등 실험에 필요한 모든 설정을 코드에서 분리하여 `config.yaml`과 같은 파일로 관리하는 패턴을 소개합니다.
*   **실무적 중요성**: 코드 변경 없이 설정 파일만 수정하여 다양한 실험을 쉽게 시도할 수 있습니다. 이는 실험의 재현성을 보장하고, 여러 실험 결과를 체계적으로 관리하는 데 필수적입니다.
*   **삽입 위치**: `01_Keras.md`의 `7.4. MLOps 통합 및 고려사항` 섹션에 추가합니다.
*   **코드 예제**:
    *   **`config.yaml` 파일 예시**:
        ```yaml
        data:
          path: "/path/to/dataset"
          batch_size: 64

        model:
          name: "ResNet50"
          params:
            num_classes: 10
            include_top: false

        train:
          epochs: 50
          optimizer:
            name: "Adam"
            learning_rate: 0.001
        ```
    *   **Python에서 로드하여 사용**:
        ```python
        import yaml
        import keras

        # with open("config.yaml", "r") as f:
        #     config = yaml.safe_load(f)

        # model = keras.applications.ResNet50(
        #     include_top=config['model']['params']['include_top'],
        #     classes=config['model']['params']['num_classes']
        # )
        # optimizer = keras.optimizers.get(config['train']['optimizer']['name'])
        # optimizer.learning_rate = config['train']['optimizer']['learning_rate']

        # print(f"Batch size: {config['data']['batch_size']}")
        ```

### 7.4.2. ML 모델을 위한 테스트 및 검증 전략

**개요**: 소프트웨어 테스트와는 다른, 머신러닝 모델의 특수성을 고려한 테스트 및 검증 방법론을 구체적으로 제시하여 모델의 신뢰도를 높입니다.

*   **제안 내용**: 전체 테스트셋의 평균 성능 지표 뒤에 숨겨진 모델의 취약점을 발견하기 위한 '데이터 슬라이스 기반 평가'와 '모델 강건성(Robustness) 테스트'의 개념과 간단한 구현 예시를 소개합니다.
*   **실무적 중요성**: 특정 사용자 그룹이나 특정 상황에서 모델이 치명적인 오류를 일으키는 것을 사전에 방지하고, 예측할 수 없는 실제 환경 변화에 더 잘 대응하는 안정적인 모델을 구축할 수 있습니다.
*   **추천 삽입 위치**: `01_Keras.md`의 `7.4. MLOps 통합 및 고려사항` 섹션 하위에 **`7.4.2. ML 모델을 위한 테스트 및 검증 전략`** 항목을 신설하여 추가합니다.
*   **코드 예제**:
    ```python
    import numpy as np
    import tensorflow as tf
    import keras

    # 더미 모델 및 데이터
    model = keras.Sequential([keras.Input(shape=(10,)), keras.layers.Dense(1, activation='sigmoid')])
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    x_test = np.random.rand(100, 10)
    y_test = np.random.randint(0, 2, 100)
    # 데이터 슬라이스를 위한 그룹 정보 (예: 남성/여성, 국가 등)
    test_groups = np.random.choice(['group_A', 'group_B'], 100)

    # 1. 데이터 슬라이스 기반 평가
    print("--- 데이터 슬라이스 기반 평가 ---")
    for group in np.unique(test_groups):
        slice_indices = np.where(test_groups == group)
        x_slice, y_slice = x_test[slice_indices], y_test[slice_indices]
        loss, accuracy = model.evaluate(x_slice, y_slice, verbose=0)
        print(f"'{group}' 슬라이스 성능: Loss={loss:.4f}, Accuracy={accuracy:.4f}")

    # 2. 모델 강건성(Robustness) 테스트 (가우시안 노이즈 추가)
    print("\n--- 모델 강건성 테스트 ---")
    noise_factor = 0.1
    x_test_noisy = x_test + np.random.normal(loc=0.0, scale=noise_factor, size=x_test.shape)
    loss_noisy, acc_noisy = model.evaluate(x_test_noisy, y_test, verbose=0)
    print(f"노이즈 추가 후 성능: Loss={loss_noisy:.4f}, Accuracy={acc_noisy:.4f}")
    ```

### 7.5. 체계적인 실험 관리 및 재현성 확보 (MLflow, W&B 연동)

**개요**: 수십, 수백 번의 실험 이력(하이퍼파라미터, 코드 버전, 데이터셋, 성능 지표, 산출물 등)을 체계적으로 기록하고 관리하여, 완벽한 재현성을 보장하고 프로젝트의 지식을 자산화하는 방법을 추가합니다.

*   **제안 내용**: Keras 콜백을 사용하여 학습 과정을 `MLflow`나 `Weights & Biases (W&B)` 같은 전문 실험 관리 도구에 자동으로 로깅하는 방법을 소개합니다. 이를 통해 제공되는 대시보드에서 여러 실험 결과를 시각적으로 비교하고 분석하는 전체 워크플로우를 안내합니다.
*   **실무적 중요성**: "지난주에 성능이 가장 좋았던 그 모델이 어떤 조건이었지?"와 같은 재앙을 원천적으로 방지합니다. 팀원 간의 협업 효율성을 극대화하고, 모든 실험 과정을 프로젝트의 중요한 지식 자산으로 축적할 수 있습니다.
*   **코드 예제 (MLflow)**:
    ```python
    # MLflow 설치 필요: pip install mlflow
    import mlflow
    import keras
    import numpy as np

    # 1. MLflow 서버 실행 (터미널): mlflow ui

    # 2. MLflow 로깅 시작
    mlflow.set_experiment("Keras MNIST Experiment")
    with mlflow.start_run():
        # 3. 하이퍼파라미터 및 태그 로깅
        params = {"learning_rate": 0.001, "epochs": 5, "batch_size": 64}
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "Simple_CNN")

        # 모델 정의 및 컴파일
        model = keras.Sequential([
            keras.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # 4. MLflow 콜백 생성
        # 이 콜백이 매 에포크마다 loss, accuracy 등의 지표를 자동으로 로깅합니다.
        mlflow_callback = mlflow.keras.MLflowCallback(run=mlflow.active_run())

        # 더미 데이터
        x_train = np.random.rand(100, 28, 28, 1)
        y_train = np.random.randint(0, 10, 100)

        # 5. 모델 학습 (콜백과 함께)
        model.fit(x_train, y_train, 
                  epochs=params["epochs"], 
                  batch_size=params["batch_size"], 
                  callbacks=[mlflow_callback])

        # 6. 모델 아티팩트 로깅
        mlflow.keras.log_model(model, "keras_model")
        print("MLflow 로깅 완료. http://127.0.0.1:5000 에서 확인하세요.")
    ```

## 8. TensorBoard를 활용한 Keras 모델 시각화 및 디버깅

TensorBoard는 TensorFlow와 Keras 모델의 학습 과정을 시각화하고 디버깅하며 최적화하는 데 필수적인 도구입니다.

### 8.1. TensorBoard 개요 및 Keras 연동

*   **개념:** 학습 과정의 다양한 메트릭, 모델 그래프, 이미지, 오디오, 텍스트, 프로파일링 데이터 등을 시각화하여 보여주는 웹 기반 대시보드입니다.
*   **Keras 연동:** `keras.callbacks.TensorBoard` 콜백을 사용하여 Keras 모델 학습 시 자동으로 TensorBoard 로그를 생성할 수 있습니다.
*   **실무 관점:**
    *   **설치 및 실행:** `pip install tensorboard`로 설치하고, `tensorboard --logdir /path/to/logs` 명령어로 실행합니다.
    *   **로그 디렉토리:** 각 실험마다 별도의 로그 디렉토리를 사용하여 여러 실험 결과를 쉽게 비교할 수 있도록 관리하는 것이 중요합니다.
    *   **예시:**

```python
import keras
import numpy as np
import datetime

# 더미 데이터
x_train = np.random.rand(100, 784).astype("float32")
y_train = np.random.randint(0, 10, size=(100,)).astype("int32")

model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(784,)),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
# 터미널에서 `tensorboard --logdir logs/fit` 실행
```

### 8.2. Keras 모델 학습 과정 모니터링

TensorBoard의 Scalars 대시보드를 통해 학습 과정의 주요 지표들을 실시간으로 모니터링할 수 있습니다.

*   **개념:** 손실(loss), 정확도(accuracy), 학습률(learning rate) 등 시간에 따라 변하는 단일 숫자 값을 그래프로 시각화합니다.
*   **실무 관점:**
    *   **과적합/과소적합 진단:** 학습 손실과 검증 손실의 추이를 비교하여 모델의 과적합 또는 과소적합 여부를 판단합니다.
    *   **학습 안정성:** 손실 값의 변동성을 확인하여 학습이 안정적으로 진행되는지 파악합니다.
    *   **하이퍼파라미터 튜닝:** 여러 실험의 스칼라 그래프를 겹쳐서 비교하여 최적의 하이퍼파라미터를 찾을 수 있습니다.
    *   **`histogram_freq`:** `TensorBoard` 콜백의 `histogram_freq` 인자를 설정하여 가중치, 편향, 활성화 값의 분포 변화를 히스토그램으로 시각화할 수 있습니다. 이는 기울기 소실/폭주 문제나 Dying ReLU와 같은 문제를 진단하는 데 유용합니다.

### 8.3. Keras 모델 그래프 시각화

TensorBoard의 Graphs 대시보드는 Keras 모델의 내부 구조를 시각적으로 탐색할 수 있게 합니다.

*   **개념:** Keras 모델의 레이어 구성, 연결 관계, 데이터 흐름을 노드와 엣지로 표현된 계산 그래프 형태로 보여줍니다.
*   **실무 관점:**
    *   **모델 이해:** 복잡한 신경망의 아키텍처를 직관적으로 이해하고, 각 레이어의 역할과 연결 방식을 파악하는 데 도움을 줍니다.
    *   **디버깅:** 예상치 못한 연결이나 누락된 연산을 찾아내어 모델 정의 오류를 디버깅할 수 있습니다. 특히 Functional API나 Subclassing API로 구축된 복잡한 모델의 구조를 검증할 때 유용합니다.
    *   **`tf.function`:** `tf.function`으로 래핑된 Custom Training Loop나 Custom Layer의 내부 그래프도 시각화하여 연산 흐름을 분석할 수 있습니다.

### 8.4. Keras Profiler를 이용한 성능 분석

TensorBoard의 Profile 대시보드는 Keras 모델 학습 및 추론 과정의 성능 병목 현상을 진단하고 최적화하는 데 사용되는 강력한 도구입니다.

*   **개념:** CPU, GPU 등 다양한 디바이스에서 Keras/TensorFlow 연산의 실행 시간, 메모리 사용량, 디바이스 간 통신 등을 상세하게 기록하고 분석합니다.
*   **실무 관점:**
    *   **병목 현상 식별:** 어떤 연산이 가장 많은 시간을 소모하는지, CPU와 GPU 간의 데이터 전송이 비효율적인지 등을 파악하여 최적화 포인트를 찾습니다.
    *   **GPU 활용률 분석:** GPU가 얼마나 효율적으로 사용되고 있는지 (예: GPU 유휴 시간, Tensor Core 활용률)를 확인하여 학습 속도 향상 방안을 모색합니다.
    *   **메모리 분석:** 각 연산이 사용하는 메모리 양을 분석하여 메모리 부족 문제를 해결하거나 메모리 사용량을 최적화합니다.
    *   **데이터 파이프라인 분석:** `tf.data` 파이프라인의 각 단계에서 데이터가 얼마나 빠르게 준비되고 모델에 공급되는지 분석하여 입력 파이프라인의 병목을 해결합니다.
    *   **사용법:** `keras.callbacks.TensorBoard` 콜백에 `profile_batch` 인자를 설정하거나, `tf.profiler.experimental.start()` 및 `tf.profiler.experimental.stop()` 함수를 사용하여 프로파일링을 시작하고 중지할 수 있습니다.

## 9. 실전 프로젝트 예제 및 팁

Keras는 다양한 딥러닝 태스크에 적용될 수 있는 유연한 프레임워크입니다. 여기서는 주요 도메인별 모델 구축 예시와 일반적인 팁을 다룹니다.

### 9.1. 이미지 처리 모델

Keras는 컴퓨터 비전 태스크를 위한 강력한 레이어(`Conv2D`, `MaxPooling2D`, `BatchNormalization` 등)와 사전 학습된 모델(`keras.applications`)을 제공합니다.

#### 9.1.1. 이미지 분류 (CNN, ResNet, EfficientNet)

*   **개념:** 이미지를 입력으로 받아 미리 정의된 클래스 중 하나로 분류하는 태스크입니다. CNN(Convolutional Neural Network)이 주로 사용됩니다.
*   **실무 관점:**
    *   **CNN:** 이미지의 공간적 특징을 효과적으로 추출합니다. 초기 레이어는 저수준 특징(엣지, 코너)을, 깊은 레이어는 고수준 특징(객체 부분)을 학습합니다.
    *   **ResNet (Residual Network):** 깊은 신경망에서 발생하는 기울기 소실 문제를 해결하기 위해 잔차 연결(Residual Connection)을 도입했습니다. 매우 깊은 모델을 안정적으로 학습할 수 있게 합니다.
    *   **EfficientNet:** 모델의 깊이, 너비, 해상도를 효율적으로 스케일링하여 높은 정확도와 효율성을 동시에 달성합니다.
    *   **데이터 증강:** `keras.layers.RandomFlip`, `RandomRotation`, `RandomZoom` 등 Keras 전처리 레이어를 사용하여 학습 데이터의 다양성을 늘리고 과적합을 방지합니다.
    *   **전이 학습:** ImageNet으로 사전 학습된 모델(`keras.applications`)을 활용하여 새로운 이미지 분류 태스크에 적용하는 것이 일반적입니다.
    *   **예시:**

```python
import keras
from keras import layers
from keras.applications import ResNet50

간단한 CNN 모델
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# ResNet50을 이용한 전이 학습 (특징 추출)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)
```

#### 9.1.2. 객체 탐지 (Object Detection - YOLO, SSD)

*   **개념:** 이미지 내에서 객체의 위치(바운딩 박스)와 클래스를 동시에 예측하는 태스크입니다.
*   **실무 관점:**
    *   **YOLO (You Only Look Once):** 단일 신경망으로 바운딩 박스와 클래스 확률을 직접 예측하여 매우 빠른 추론 속도를 제공합니다. 실시간 객체 탐지에 적합합니다.
    *   **SSD (Single Shot MultiBox Detector):** 다양한 스케일의 특징 맵에서 객체를 탐지하여 YOLO보다 정확도를 높이면서도 빠른 속도를 유지합니다.
    *   **Keras 구현:** Keras-RetinaNet, Keras-YOLOv3 등 커뮤니티에서 제공하는 구현체를 활용하거나, Functional API를 사용하여 직접 구현할 수 있습니다. 복잡한 앵커 박스(anchor box) 생성, NMS(Non-Maximum Suppression) 등의 후처리 로직이 필요합니다.
    *   **데이터셋:** COCO, PASCAL VOC 등 객체 탐지 전용 데이터셋을 사용합니다.
    *   **손실 함수:** 분류 손실(Categorical Crossentropy)과 바운딩 박스 회귀 손실(Smooth L1 Loss)을 조합하여 사용합니다.

```python
import keras
import keras_cv
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# In a real scenario, you would use a dataset like COCO or Pascal VOC.
# Here, we simulate a very small dataset of images and bounding box annotations.

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

# Number of samples and classes
NUM_SAMPLES = 10
NUM_CLASSES = 2 # e.g., 'car', 'person'

# Generate dummy images
dummy_images = np.random.rand(NUM_SAMPLES, IMG_HEIGHT, IMG_WIDTH, CHANNELS).astype(np.float32)

# Generate dummy bounding box annotations
# Format: { 'boxes': (num_boxes, 4), 'classes': (num_boxes,) }
# Boxes are in normalized [ymin, xmin, ymax, xmax] format.
# Classes are integer IDs.

dummy_bounding_boxes = []
for _ in range(NUM_SAMPLES):
    num_boxes_per_image = np.random.randint(1, 4) # 1 to 3 boxes per image
    boxes = np.random.rand(num_boxes_per_image, 4).astype(np.float32)
    classes = np.random.randint(0, NUM_CLASSES, size=(num_boxes_per_image,)).astype(np.int32)
    dummy_bounding_boxes.append({'boxes': boxes, 'classes': classes})

# Convert to tf.data.Dataset
def format_dataset_entry(image, bboxes):
    return {'images': image, 'bounding_boxes': bboxes}

dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_bounding_boxes))
dataset = dataset.map(format_dataset_entry, num_parallel_calls=tf.data.AUTOTUNE)
BATCH_SIZE = 2
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

print(f"Dummy images shape: {dummy_images.shape}")
print(f"Dummy bounding boxes example: {dummy_bounding_boxes[0]}")

# --- KerasCV Object Detection Model (YOLOv8) ---
# KerasCV provides pre-trained object detection models like YOLOv8.
# These models come with built-in preprocessing and postprocessing.

# Load a pre-trained YOLOv8 model from KerasCV
# `num_classes` should include the background class if applicable, or just your object classes.
# `bounding_box_format` is crucial for correct interpretation of annotations.

yolov8_model = keras_cv.models.YOLOV8Detector(
    num_classes=NUM_CLASSES + 1, # +1 for background class
    bounding_box_format="xywh", # KerasCV models often expect xywh format internally
    backbone=keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone"),
    fpn_depth=1 # Simplified FPN depth for example
)

# Compile the model
# Object detection models often use specialized losses (e.g., Focal Loss, GIoU Loss)
# KerasCV models usually handle this internally when compiled.

yolov8_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    classification_loss="focal",
    box_loss="ciou"
)

yolov8_model.summary()

# --- Training Example ---
print("\n--- Training KerasCV YOLOv8 Model ---")
# KerasCV models expect inputs in a specific dictionary format for training.
# The `bounding_box_format` in the dataset should match the model's expectation.
# Here, we convert our dummy [ymin, xmin, ymax, xmax] to "xywh" for the model.

def convert_boxes_to_xywh(inputs):
    images = inputs['images']
    boxes = inputs['bounding_boxes']['boxes']
    classes = inputs['bounding_boxes']['classes']

    # Convert [ymin, xmin, ymax, xmax] to [x, y, width, height]
    # x = xmin + (xmax - xmin) / 2
    # y = ymin + (ymax - ymin) / 2
    # width = xmax - xmin
    # height = ymax - ymin
    
    # Assuming normalized coordinates (0-1)
    x_center = (boxes[:, 1] + boxes[:, 3]) / 2.0
    y_center = (boxes[:, 0] + boxes[:, 2]) / 2.0
    width = boxes[:, 3] - boxes[:, 1]
    height = boxes[:, 2] - boxes[:, 0]

    xywh_boxes = tf.stack([x_center, y_center, width, height], axis=-1)

    return {
        'images': images,
        'bounding_boxes': {
            'boxes': xywh_boxes,
            'classes': classes
        }
    }

train_ds_formatted = dataset.map(convert_boxes_to_xywh, num_parallel_calls=tf.data.AUTOTUNE)

history_yolov8 = yolov8_model.fit(
    train_ds_formatted,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making an Object Detection Prediction ---")
# Get a sample image from the dummy dataset
sample_image = dummy_images[0:1] # Keep batch dimension

# Predict bounding boxes and classes
# The model outputs raw predictions, which need to be decoded.
# KerasCV models often have a `decode_predictions` method.
raw_predictions = yolov8_model.predict(sample_image)

# Decode predictions (e.g., apply NMS, convert to readable format)
# KerasCV provides a utility for this.
# Note: The `decode_predictions` method might require specific arguments
# depending on the model and task.

# For demonstration, we'll just print the raw output structure.
print(f"Raw prediction output structure: {raw_predictions.keys()}")
print(f"Raw prediction boxes shape: {raw_predictions['boxes'].shape}")
print(f"Raw prediction classes shape: {raw_predictions['classes'].shape}")

# A real decoding step would look something like this (conceptual):
# import matplotlib.pyplot as plt
# import keras_cv
# decoded_predictions = yolov8_model.decode_predictions(raw_predictions, 
#                                                     confidence_threshold=0.5, 
#                                                     iou_threshold=0.4)
# print(f"Decoded predictions: {decoded_predictions}")

# You can visualize the predictions on the image using KerasCV's visualization utilities.
# import matplotlib.pyplot as plt
# import keras_cv
# plt.figure(figsize=(10, 10))
# keras_cv.visualization.plot_bounding_box_gallery(
#     sample_image,
#     value_range=(0, 1),
#     bounding_box_format="xywh", # Or whatever format your decoded predictions are in
#     bounding_boxes=decoded_predictions
# )
# plt.show()
```

#### 9.1.3. 이미지 분할 (Image Segmentation - U-Net, Mask R-CNN)

*   **개념:** 이미지의 각 픽셀에 대해 클래스 레이블을 할당하여 객체의 경계를 정확하게 분할하는 태스크입니다.
*   **실무 관점:**
    *   **U-Net:** 의료 영상 분할에 널리 사용되는 인코더-디코더 구조의 모델입니다. 인코더에서 특징을 추출하고, 디코더에서 이를 업샘플링하여 픽셀 단위 예측을 수행합니다. 인코더와 디코더 간의 스킵 연결(skip connection)이 특징입니다.
    *   **Mask R-CNN:** 객체 탐지와 인스턴스 분할(각 객체 인스턴스별로 마스크 생성)을 동시에 수행하는 모델입니다. Faster R-CNN에 마스크 예측 브랜치를 추가한 형태입니다.
    *   **Keras 구현:** Keras-Unet, Keras-Mask-RCNN 등 커뮤니티 구현체를 활용하거나, Functional API를 사용하여 직접 구현합니다.
    *   **손실 함수:** 픽셀 단위 분류를 위해 `BinaryCrossentropy` 또는 `CategoricalCrossentropy`를 사용하며, Dice Loss, IoU Loss 등 분할 태스크에 특화된 손실 함수를 함께 사용하기도 합니다.

```python
import keras
from keras import layers
import tensorflow as tf

def double_conv_block(x, n_filters):
    """A block of two 3x3 convolution layers followed by ReLU activations."""
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

def downsample_block(x, n_filters):
    """Downsampling block using max pooling."""
    f = double_conv_block(x, n_filters)
    p = layers.MaxPooling2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x, conv_features, n_filters):
    """Upsampling block using Conv2DTranspose."""
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

def build_unet_model(input_shape=(128, 128, 3), num_classes=1):
    """Builds a U-Net model using the Keras Functional API."""
    inputs = layers.Input(shape=input_shape)

    # Encoder (Downsampling Path)
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    # Bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # Decoder (Upsampling Path)
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    # Output Layer
    # Use 'sigmoid' for binary segmentation, 'softmax' for multi-class
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation=activation)(u9)

    unet_model = keras.Model(inputs, outputs, name="U-Net")
    return unet_model

# --- Model Instantiation and Compilation ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
NUM_CLASSES = 1 # Binary segmentation (e.g., background vs. foreground)

# Build the model
unet_model = build_unet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES)

# Compile the model
# For binary segmentation, use binary_crossentropy.
# For multi-class, use sparse_categorical_crossentropy or categorical_crossentropy.
# Common metrics for segmentation are IoU (Intersection over Union) and Dice coefficient.
unet_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=[tf.keras.metrics.MeanIoU(num_classes=2)]) # Adjust num_classes for multi-class

# Print model summary
unet_model.summary()

# --- Dummy Data Example ---
# Create a dummy input tensor to verify the model's output shape
dummy_input = tf.random.normal((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
dummy_output = unet_model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {dummy_output.shape}")
```

#### 9.1.4. 이미지 생성 (GAN, VAE, StyleGAN)

*   **개념:** 훈련 데이터와 유사한 새로운 이미지를 생성하는 태스크입니다.
*   **실무 관점:**
    *   **GAN (Generative Adversarial Networks):** 생성자(Generator)와 판별자(Discriminator)라는 두 개의 신경망이 서로 경쟁하며 학습하여 사실적인 이미지를 생성합니다. Keras에서는 Custom Training Loop를 사용하여 GAN을 구현하는 것이 일반적입니다.
    *   **VAE (Variational Autoencoders):** 잠재 공간(latent space)에서 샘플링하여 이미지를 생성하는 생성 모델입니다. 잠재 공간이 연속적이고 의미론적인 특성을 가지도록 학습됩니다.
    *   **StyleGAN:** 고해상도 이미지를 생성하고, 스타일을 제어할 수 있는 GAN의 변형입니다. 얼굴 이미지 생성에 특히 강력한 성능을 보입니다.
    *   **활용 사례:** 이미지 합성, 데이터 증강, 예술 작품 생성, 이미지 복원 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# VAE의 샘플링 레이어 (재매개변수화 트릭)
class Sampling(layers.Layer):
    """평균(z_mean)과 로그 분산(z_log_var)으로부터 잠재 공간 벡터를 샘플링합니다."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# VAE 인코더
def build_vae_encoder(input_shape, latent_dim):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# VAE 디코더
def build_vae_decoder(latent_dim, output_shape):
    decoder_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(np.prod(output_shape[0]//4, output_shape[1]//4, 64), activation="relu")(decoder_inputs) # Adjust based on encoder output
    x = layers.Reshape((output_shape[0]//4, output_shape[1]//4, 64))(x) # Adjust based on encoder output
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(output_shape[-1], 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
    return decoder

# VAE 모델 (인코더 + 디코더)
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# --- Model Instantiation and Training Example ---
# Dummy Data (e.g., MNIST-like images 28x28x1)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

# VAE Parameters
input_shape = (28, 28, 1)
latent_dim = 2

# Build Encoder and Decoder
encoder = build_vae_encoder(input_shape, latent_dim)
decoder = build_vae_decoder(latent_dim, input_shape)

# Build VAE model
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# Train VAE
print("\n--- Training VAE ---")
vae.fit(mnist_digits, epochs=1) # epochs can be increased for better results

# --- Generate new images from VAE ---
print("\n--- Generating new images ---")
# Sample points from the latent space
num_generation_samples = 10
random_latent_vectors = tf.random.normal(shape=(num_generation_samples, latent_dim))
generated_images = vae.decoder(random_latent_vectors)

print(f"Generated images shape: {generated_images.shape}") # Expected: (10, 28, 28, 1)

# You can save and visualize these images
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 1))
for i in range(num_generation_samples):
    ax = plt.subplot(1, num_generation_samples, i + 1)
    plt.imshow(generated_images[i].numpy().squeeze(), cmap="gray")
    plt.axis("off")
plt.show()
```

#### 9.1.5. 초해상도 (Super-resolution)

*   **개념:** 저해상도 이미지를 입력으로 받아 고해상도 이미지를 생성하는 태스크입니다.
*   **실무 관점:**
    *   **모델 아키텍처:** 주로 CNN 기반의 인코더-디코더 구조나 GAN 기반의 모델이 사용됩니다. 픽셀 단위의 정확한 복원이 중요하므로, 손실 함수로 MSE(Mean Squared Error) 외에 perceptual loss, adversarial loss 등을 함께 사용하기도 합니다.
    *   **활용 사례:** 오래된 사진 복원, 의료 영상 해상도 개선, CCTV 영상 화질 개선 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Simulate low-resolution (LR) and high-resolution (HR) image pairs.
# In a real scenario, LR images would be downsampled from HR images.

# Define image dimensions
HR_HEIGHT = 64
HR_WIDTH = 64
CHANNELS = 3
LR_SCALE_FACTOR = 2 # LR image will be HR_HEIGHT/2 x HR_WIDTH/2

LR_HEIGHT = HR_HEIGHT // LR_SCALE_FACTOR
LR_WIDTH = HR_WIDTH // LR_SCALE_FACTOR

num_samples = 100

# Generate dummy HR images (e.g., random noise for simplicity)
dummy_hr_images = np.random.rand(num_samples, HR_HEIGHT, HR_WIDTH, CHANNELS).astype(np.float32)

# Simulate LR images by simple downsampling (e.g., using TensorFlow's resize)
dummy_lr_images = tf.image.resize(dummy_hr_images, (LR_HEIGHT, LR_WIDTH), method=tf.image.ResizeMethod.BICUBIC).numpy()

print(f"Dummy HR images shape: {dummy_hr_images.shape}")
print(f"Dummy LR images shape: {dummy_lr_images.shape}")

# --- Super-Resolution Convolutional Neural Network (SRCNN) Model ---
# A simple, early deep learning model for super-resolution.
# It consists of three convolutional layers:
# 1. Patch extraction and representation
# 2. Non-linear mapping
# 3. Reconstruction

def build_srcnn_model(input_shape, scale_factor, channels):
    # Input is the low-resolution image
    inputs = layers.Input(shape=input_shape)

    # Upsample the LR image to HR size using Bicubic interpolation
    # This is a common pre-processing step for many SR models
    upscaled_lr = layers.Lambda(lambda x: tf.image.resize(x, (input_shape[0] * scale_factor, input_shape[1] * scale_factor), method=tf.image.ResizeMethod.BICUBIC))(inputs)

    # Layer 1: Patch extraction and representation (Conv + ReLU)
    # Filters: 64, Kernel: 9x9
    x = layers.Conv2D(64, (9, 9), activation='relu', padding='same')(upscaled_lr)

    # Layer 2: Non-linear mapping (Conv + ReLU)
    # Filters: 32, Kernel: 1x1
    x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)

    # Layer 3: Reconstruction (Conv)
    # Filters: Number of channels, Kernel: 5x5
    outputs = layers.Conv2D(channels, (5, 5), activation='linear', padding='same')(x) # Use linear for image reconstruction

    model = keras.Model(inputs, outputs, name="SRCNN")
    return model

# Build the SRCNN model
srcnn_model = build_srcnn_model(
    input_shape=(LR_HEIGHT, LR_WIDTH, CHANNELS),
    scale_factor=LR_SCALE_FACTOR,
    channels=CHANNELS
)

# Compile the model
# MSE is a common loss for super-resolution (pixel-wise difference)
srcnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

srcnn_model.summary()

# --- Training Example ---
print("\n--- Training SRCNN Model ---")
# In a real scenario, you would use a proper dataset (e.g., DIV2K, Set5)
# and train for many epochs.
history_srcnn = srcnn_model.fit(
    dummy_lr_images,
    dummy_hr_images,
    batch_size=16,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making a Super-resolution Prediction ---")
# Take one LR image from the dummy dataset
sample_lr_image = dummy_lr_images[0:1] # Keep batch dimension

# Predict the HR version
predicted_hr_image = srcnn_model.predict(sample_lr_image)

print(f"Input LR image shape: {sample_lr_image.shape}")
print(f"Predicted HR image shape: {predicted_hr_image.shape}")

# You can visualize the original HR, LR, and predicted HR images
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original HR")
plt.imshow(dummy_hr_images[0])
plt.axis("off")
plt.subplot(1, 3, 2)
plt.title("Low Resolution")
plt.imshow(sample_lr_image[0])
plt.axis("off")
plt.subplot(1, 3, 3)
plt.title("Predicted HR")
plt.imshow(predicted_hr_image[0])
plt.axis("off")
plt.show()
```

#### 9.1.6. 스타일 전이 (Style Transfer)

*   **개념:** 내용 이미지의 내용과 스타일 이미지의 스타일을 결합하여 새로운 이미지를 생성하는 태스크입니다.
*   **실무 관점:**
    *   **VGG 네트워크:** 사전 학습된 VGG 네트워크의 중간 레이어에서 내용 특징과 스타일 특징을 추출하여 손실 함수를 구성합니다.
    *   **손실 함수:** 내용 손실(Content Loss)과 스타일 손실(Style Loss)을 조합하여 사용합니다.
    *   **활용 사례:** 예술 작품 생성, 사진 편집 앱 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np
import PIL.Image

# --- Helper Functions for Image Loading and Preprocessing ---
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

# --- Define VGG19 for Feature Extraction ---
# We will use the VGG19 model pre-trained on ImageNet to extract features.
# We need access to intermediate layers for content and style representations.

def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values."""
    # Load our pre-trained VGG19 model without the top classification layer
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# Content layer where we will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# --- Calculate Style Loss (Gram Matrix) ---
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# --- StyleContentModel ---
# This model will take an image and return the content and style representations.
class StyleContentModel(keras.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0,1]."""
        inputs = inputs*255.0 # Scale back to VGG input range
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value in zip(self.style_layers, style_outputs)}
        
        return {'content': content_dict, 'style': style_dict}

# --- Optimization Parameters ---
# Content and style weight balance
content_weight=1e3
style_weight=1e-2

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Loss function (MSE for content, MSE for style)
# We will define custom loss functions within the training loop.

# --- Training Loop ---
@tf.function()
def train_step(image, style_targets, content_targets, extractor):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers

        total_loss = style_loss + content_loss

    grad = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
    return total_loss, style_loss, content_loss

# --- Main Execution (requires image files) ---
# For this example to run, you need to provide actual image paths.
# Replace 'path/to/content_image.jpg' and 'path/to/style_image.jpg'
# with valid paths to your content and style images.

# Example: Download images if you don't have them locally
# content_path = tf.keras.utils.get_file('YellowLabradorPuppy.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorPuppy.jpg')
# style_path = tf.keras.utils.get_file('kandinsky.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913._Composition_7.jpg')

# Dummy paths for demonstration (will fail without actual files)
content_path = "dummy_content.jpg"
style_path = "dummy_style.jpg"

# Create dummy image files for the example to run without error
# In a real scenario, you would use actual images.
if not tf.io.gfile.exists(content_path):
    dummy_content_img = PIL.Image.new('RGB', (256, 256), color = 'red')
    dummy_content_img.save(content_path)
if not tf.io.gfile.exists(style_path):
    dummy_style_img = PIL.Image.new('RGB', (256, 256), color = 'blue')
    dummy_style_img.save(style_path)

content_image = load_img(content_path)
style_image = load_img(style_path)

# Create the extractor model
extractor = StyleContentModel(style_layers, content_layers)

# Get targets for style and content
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Initialize the generated image with the content image
image = tf.Variable(content_image)

epochs = 1 # Increase for better results
steps_per_epoch = 10 # Increase for better results

print("\n--- Starting Style Transfer Training ---")
import time
start_time = time.time()

for n in range(epochs):
    for m in range(steps_per_epoch):
        total_loss, s_loss, c_loss = train_step(image, style_targets, content_targets, extractor)
        if m % 10 == 0:
            print(".", end='')
    print(f"Epoch {n+1}/{epochs} - Total Loss: {total_loss:.2f}, Style Loss: {s_loss:.2f}, Content Loss: {c_loss:.2f}")

end_time = time.time()
print(f"\nTotal time: {end_time - start_time:.2f} seconds")

# --- Display Result (requires matplotlib) ---
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))
# imshow(image, 'Generated Image')
# plt.show()

# Clean up dummy files
tf.io.gfile.remove(content_path)
tf.io.gfile.remove(style_path)
```

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np
import PIL.Image

# --- Helper Functions for Image Loading and Preprocessing ---
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

# --- Define VGG19 for Feature Extraction ---
# We will use the VGG19 model pre-trained on ImageNet to extract features.
# We need access to intermediate layers for content and style representations.

def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values."""
    # Load our pre-trained VGG19 model without the top classification layer
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

# Content layer where we will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# --- Calculate Style Loss (Gram Matrix) ---
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# --- StyleContentModel ---
# This model will take an image and return the content and style representations.
class StyleContentModel(keras.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """Expects float input in [0,1]."""
        inputs = inputs*255.0 # Scale back to VGG input range
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {
            content_name: value
            for content_name, value in zip(self.content_layers, content_outputs)
        }

        style_dict = {
            style_name: value
            for style_name, value in zip(self.style_layers, style_outputs)
        }
        
        return {'content': content_dict, 'style': style_dict}

# --- Optimization Parameters ---
# Content and style weight balance
content_weight=1e3
style_weight=1e-2

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Loss function (MSE for content, MSE for style)
# We will define custom loss functions within the training loop.

# --- Training Loop ---
@tf.function()
def train_step(image, style_targets, content_targets, extractor):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                               for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers

        total_loss = style_loss + content_loss

    grad = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
    return total_loss, style_loss, content_loss

# --- Main Execution (requires image files) ---
# For this example to run, you need to provide actual image paths.
# Replace 'path/to/content_image.jpg' and 'path/to/style_image.jpg'
# with valid paths to your content and style images.

# Example: Download images if you don't have them locally
# content_path = tf.keras.utils.get_file('YellowLabradorPuppy.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorPuppy.jpg')
# style_path = tf.keras.utils.get_file('kandinsky.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913._Composition_7.jpg')

# Dummy paths for demonstration (will fail without actual files)
content_path = "dummy_content.jpg"
style_path = "dummy_style.jpg"

# Create dummy image files for the example to run without error
# In a real scenario, you would use actual images.
if not tf.io.gfile.exists(content_path):
    dummy_content_img = PIL.Image.new('RGB', (256, 256), color = 'red')
    dummy_content_img.save(content_path)
if not tf.io.gfile.exists(style_path):
    dummy_style_img = PIL.Image.new('RGB', (256, 256), color = 'blue')
    dummy_style_img.save(style_path)

content_image = load_img(content_path)
style_image = load_img(style_path)

# Create the extractor model
extractor = StyleContentModel(style_layers, content_layers)

# Get targets for style and content
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Initialize the generated image with the content image
image = tf.Variable(content_image)

epochs = 1 # Increase for better results
steps_per_epoch = 10 # Increase for better results

print("\n--- Starting Style Transfer Training ---")
import time
start_time = time.time()

for n in range(epochs):
    for m in range(steps_per_epoch):
        total_loss, s_loss, c_loss = train_step(image, style_targets, content_targets, extractor)
        if m % 10 == 0:
            print(".", end='')
    print(f"Epoch {n+1}/{epochs} - Total Loss: {total_loss:.2f}, Style Loss: {s_loss:.2f}, Content Loss: {c_loss:.2f}")

end_time = time.time()
print(f"\nTotal time: {end_time - start_time:.2f} seconds")

# --- Display Result (requires matplotlib) ---
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 10))
# imshow(image, 'Generated Image')
# plt.show()

# Clean up dummy files
tf.io.gfile.remove(content_path)
tf.io.gfile.remove(style_path)
```

#### 9.1.7. 이미지 캡셔닝 (Image Captioning)

*   **개념:** 이미지를 입력으로 받아 이미지의 내용을 설명하는 텍스트 문장을 생성하는 태스크입니다.
*   **실무 관점:**
    *   **모델 아키텍처:** 주로 CNN(이미지 특징 추출)과 RNN/Transformer(텍스트 시퀀스 생성)를 결합한 인코더-디코더 구조가 사용됩니다.
    *   **데이터셋:** MS COCO, Flickr30k 등 이미지-캡션 쌍으로 구성된 데이터셋을 사용합니다.
    *   **평가 메트릭:** BLEU, CIDEr, METEOR 등 텍스트 생성 평가 메트릭을 사용합니다.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np
import collections
import random

# --- Dummy Data Generation ---
# In a real scenario, you would use a dataset like MS COCO or Flickr30k.
# Here, we simulate a small dataset of image features and corresponding captions.

# Vocabulary parameters
VOCAB_SIZE = 5000
MAX_CAPTION_LENGTH = 20
EMBEDDING_DIM = 256

# Image features (e.g., extracted from a pre-trained CNN like InceptionV3)
# Simulate 100 images, each with a 2048-dim feature vector
NUM_IMAGES = 100
IMAGE_FEATURE_DIM = 2048
dummy_image_features = np.random.rand(NUM_IMAGES, IMAGE_FEATURE_DIM).astype(np.float32)

# Dummy captions
dummy_captions_raw = [
    "<start> a dog is playing in the park <end>",
    "<start> a cat is sleeping on the couch <end>",
    "<start> a group of people are walking on the street <end>",
    "<start> a car is driving on the road <end>",
    "<start> a bird is flying in the sky <end>",
    "<start> a dog is running on the grass <end>",
    "<start> a cat is looking out the window <end>",
    "<start> people are eating at a restaurant <end>",
    "<start> a red car is parked on the side <end>",
    "<start> birds are sitting on a branch <end>",
] * (NUM_IMAGES // 10) # Repeat to match number of images

# Simple TextVectorization for dummy captions
text_vectorization = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_CAPTION_LENGTH,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ragged=False,
)
text_vectorization.adapt(dummy_captions_raw)

# Convert raw captions to integer sequences
dummy_captions_vectorized = text_vectorization(tf.constant(dummy_captions_raw)).numpy()

# Prepare dataset: (image_features, caption_input_sequence), caption_target_sequence
# caption_input_sequence: <start> word1 word2 ... wordN
# caption_target_sequence: word1 word2 ... wordN <end>

dummy_caption_inputs = dummy_captions_vectorized[:, :-1]
dummy_caption_targets = dummy_captions_vectorized[:, 1:]

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(
    ((dummy_image_features, dummy_caption_inputs), dummy_caption_targets)
)
BATCH_SIZE = 32
dataset = dataset.shuffle(NUM_IMAGES).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Image Captioning Model (Encoder-Decoder with Attention) ---

# Encoder: CNN features to a fixed-size vector
class CNN_Encoder(keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc layer == (batch_size, embedding_dim)
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

# Decoder: RNN (GRU) with Attention
class RNN_Decoder(keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(self.units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')
        self.fc1 = layers.Dense(self.units)
        self.fc2 = layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # features shape == (batch_size, 64, embedding_dim) (if using attention on image grid)
        # For now, features is (batch_size, embedding_dim) from CNN_Encoder

        # x shape == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # features shape == (batch_size, embedding_dim) -> expand for attention
        features_expanded = tf.expand_dims(features, 1) # (batch_size, 1, embedding_dim)

        # context_vector shape == (batch_size, hidden_size)
        # attention_weights shape == (batch_size, 1, 1) (for single feature vector)
        context_vector, attention_weights = self.attention(features_expanded, hidden)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x, initial_state=hidden)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# Bahdanau Attention Mechanism
class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 1, embedding_dim)
        # hidden shape == (batch_size, hidden_size)

        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 1, 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, 1, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# --- Model Instantiation ---
UNITS = 512 # RNN units

encoder = CNN_Encoder(EMBEDDING_DIM)
de תecoder = RNN_Decoder(EMBEDDING_DIM, UNITS, VOCAB_SIZE)

# --- Loss Function and Optimizer ---
optimizer = keras.optimizers.Adam()
loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# --- Training Step (Custom Training Loop) ---
@tf.function
def train_step_captioning(img_tensor, target):
    loss = 0

    # Initialize hidden state for the decoder
    hidden = decoder.reset_state(batch_size=target.shape[0])

    # Encode image features
    features = encoder(img_tensor)

    # Decoder input starts with <start> token
    dec_input = tf.expand_dims([text_vectorization.vocabulary()[-2]] * target.shape[0], 1) # Assuming <start> is the second to last token

    with tf.GradientTape() as tape:
        for i in range(1, target.shape[1]):
            # passing the features and the hidden state to the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return total_loss

# --- Training Loop ---
EPOCHS = 1 # Increase for better results

print("\n--- Training Image Captioning Model ---")
for epoch in range(EPOCHS):
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss = train_step_captioning(img_tensor, target)
        total_loss += batch_loss

        if batch % 10 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
    
    print(f'Epoch {epoch+1} Loss {total_loss/len(dataset):.4f}')

# --- Caption Generation (Inference) Example ---
print("\n--- Generating Caption for a Dummy Image ---")

def evaluate_caption(image_features_input):
    attention_plot = np.zeros((MAX_CAPTION_LENGTH, UNITS))

    hidden = decoder.reset_state(batch_size=1)

    features = encoder(image_features_input)

    dec_input = tf.expand_dims([text_vectorization.vocabulary()[-2]], 0) # <start> token
    result = []

    for i in range(MAX_CAPTION_LENGTH):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        # attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = text_vectorization.get_vocabulary()[predicted_id]
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    # attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

# Select a dummy image feature for inference
sample_image_feature = dummy_image_features[0:1]

# Generate caption
result_caption, _ = evaluate_caption(sample_image_feature)
print(f"Generated Caption: {' '.join(result_caption)}")
```


### 9.2. 시퀀스 및 텍스트 처리 모델

Keras는 시퀀스 데이터(텍스트, 시계열) 처리를 위한 `Embedding`, `LSTM`, `GRU`, `Conv1D` 등 다양한 레이어를 제공합니다.

#### 9.2.1. 텍스트 분류 (RNN, LSTM, GRU)

*   **개념:** 텍스트 문서를 입력으로 받아 미리 정의된 클래스 중 하나로 분류하는 태스크입니다 (예: 스팸 메일 분류, 감성 분석).
*   **실무 관점:**
    *   **전처리:** `keras.layers.TextVectorization`을 사용하여 텍스트를 토큰화하고 정수 시퀀스로 변환합니다.
    *   **임베딩:** `keras.layers.Embedding` 레이어를 사용하여 단어를 저차원 벡터 공간으로 매핑합니다. 사전 학습된 워드 임베딩(Word2Vec, GloVe, FastText)을 활용할 수도 있습니다.
    *   **RNN/LSTM/GRU:** 텍스트의 순차적인 특성을 학습하는 데 효과적입니다. LSTM과 GRU는 RNN의 장기 의존성 문제를 해결합니다.
    *   **Conv1D:** 텍스트에서 지역적인 특징(n-gram)을 추출하는 데 사용될 수 있습니다.
    *   **예시:**

```python
import keras
from keras import layers
import tensorflow as tf

# 텍스트 분류 모델 (LSTM 기반)
max_features = 20000 # 어휘 사전 크기
embedding_dim = 128
sequence_length = 500 # 최대 시퀀스 길이

text_vectorization_layer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length
)
# text_vectorization_layer.adapt(train_texts) # 실제 데이터로 adapt 필요

model = keras.Sequential([
    keras.Input(shape=(1,), dtype=tf.string),
    text_vectorization_layer,
    layers.Embedding(max_features, embedding_dim),
    layers.LSTM(128),
    layers.Dense(10, activation="softmax") # 10개 클래스 분류
])
```

#### 9.2.2. 시계열 예측 (LSTM, Transformer)

*   **개념:** 과거 시계열 데이터를 기반으로 미래 값을 예측하는 태스크입니다 (예: 주가 예측, 기온 예측).
*   **실무 관점:**
    *   **LSTM/GRU:** 시계열 데이터의 장기적인 패턴을 학습하는 데 효과적입니다.
    *   **Transformer:** 어텐션 메커니즘을 사용하여 시계열 데이터의 장거리 의존성을 효과적으로 포착합니다. 특히 복잡한 패턴이나 다변량 시계열 데이터에 강점을 보입니다.
    *   **데이터 전처리:** 시계열 데이터는 일반적으로 정규화/표준화가 필요하며, 시퀀스 형태로 데이터를 구성해야 합니다.
    *   **활용 사례:** 금융 시장 예측, 에너지 소비 예측, 교통량 예측 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Create a simple sine wave time series for demonstration.
# In a real scenario, this would be sensor data, stock prices, etc.

def generate_time_series_data(num_points, seq_length, prediction_horizon):
    time = np.linspace(0, 100, num_points + seq_length + prediction_horizon)
    data = np.sin(time) + np.random.normal(0, 0.1, num_points + seq_length + prediction_horizon)

    X, y = [], []
    for i in range(num_points):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length : i+seq_length+prediction_horizon])
    
    return np.array(X), np.array(y)

SEQ_LENGTH = 10 # Number of past time steps to consider
PREDICTION_HORIZON = 1 # Number of future time steps to predict
NUM_DATA_POINTS = 1000

x_data, y_data = generate_time_series_data(NUM_DATA_POINTS, SEQ_LENGTH, PREDICTION_HORIZON)

# Reshape for LSTM input: (samples, timesteps, features)
x_data = x_data.reshape((NUM_DATA_POINTS, SEQ_LENGTH, 1)) # 1 feature (univariate)
y_data = y_data.reshape((NUM_DATA_POINTS, PREDICTION_HORIZON, 1)) # 1 feature (univariate)

print(f"Input data shape (X): {x_data.shape}")
print(f"Target data shape (y): {y_data.shape}")

# Split data (simple split for demonstration)
split_ratio = 0.8
split_index = int(NUM_DATA_POINTS * split_ratio)

x_train, x_val = x_data[:split_index], x_data[split_index:]
y_train, y_val = y_data[:split_index], y_data[split_index:]

# Convert to tf.data.Dataset for efficient loading
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=100).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# --- LSTM Model for Time Series Prediction ---
# A simple LSTM model to predict the next value(s) in a sequence.

def build_lstm_model(input_shape, output_units):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(50, return_sequences=False), # return_sequences=True if stacking LSTMs
        layers.Dense(output_units)
    ])
    return model

lstm_model = build_lstm_model(input_shape=(SEQ_LENGTH, 1), output_units=PREDICTION_HORIZON)

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()

# --- Training Example ---
print("\n--- Training LSTM Time Series Prediction Model ---")
history_lstm = lstm_model.fit(
    train_dataset,
    epochs=1, # Increase epochs for better results
    validation_data=val_dataset,
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making a Time Series Prediction ---")
# Take a sample from the validation set
sample_input = x_val[0:1] # Keep batch dimension

# Predict the next value(s)
predicted_output = lstm_model.predict(sample_input)

print(f"Sample Input Sequence (first 5 values): {sample_input[0, :5, 0]}")
print(f"True Next Value(s): {y_val[0, :, 0]}")
print(f"Predicted Next Value(s): {predicted_output[0, :]}")

# You can visualize the predictions vs. actuals
# import matplotlib.pyplot as plt
# plt.plot(np.arange(SEQ_LENGTH), sample_input[0, :, 0], label='Input Sequence')
# plt.plot(np.arange(SEQ_LENGTH, SEQ_LENGTH + PREDICTION_HORIZON), y_val[0, :, 0], label='True Future', marker='o')
# plt.plot(np.arange(SEQ_LENGTH, SEQ_LENGTH + PREDICTION_HORIZON), predicted_output[0, :], label='Predicted Future', marker='x')
# plt.legend()
# plt.title('Time Series Prediction')
# plt.show()
```

#### 9.2.3. 자연어 처리 (Transformer, BERT, GPT)

*   **개념:** 텍스트 데이터를 이해하고 생성하는 다양한 태스크를 포함합니다 (예: 기계 번역, 텍스트 요약, 질의응답).
*   **실무 관점:**
    *   **Transformer:** RNN의 순차적 처리 한계를 극복하고 병렬 처리를 가능하게 한 아키텍처로, 현재 대부분의 최신 NLP 모델의 기반이 됩니다. Keras에서는 `layers.MultiHeadAttention`, `layers.LayerNormalization` 등을 조합하여 Transformer 블록을 구현할 수 있습니다.
    *   **BERT (Bidirectional Encoder Representations from Transformers):** 양방향으로 문맥을 학습하는 사전 학습된 언어 모델입니다. 다양한 NLP 다운스트림 태스크에 미세 조정(fine-tuning)하여 높은 성능을 달성합니다.
    *   **GPT (Generative Pre-trained Transformer):** Transformer의 디코더 부분을 기반으로 한 생성형 언어 모델입니다. 텍스트 생성, 요약, 번역 등 다양한 생성 태스크에 활용됩니다.
    *   **KerasNLP:** Keras 생태계의 일부로, 사전 학습된 BERT, GPT 등의 모델과 NLP 전처리 도구를 제공하여 NLP 모델 개발을 간소화합니다.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Simulate a simple sequence-to-sequence task, e.g., adding numbers.
# Input: "1 2 3 4 5" -> Output: "6 7 8 9 10"

VOCAB_SIZE = 20 # Digits 0-9, space, <start>, <end>, <pad>
MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 64
NUM_SAMPLES = 100

# Create a simple vocabulary
char_to_id = {str(i): i for i in range(10)}
char_to_id[" "] = 10
char_to_id["<start>"] = 11
char_to_id["<end>"] = 12
char_to_id["<pad>"] = 13

id_to_char = {i: char for char, i in char_to_id.items()}

def encode_sequence(text, max_len):
    encoded = [char_to_id.get(c, char_to_id["<pad>"]) for c in text]
    if len(encoded) < max_len:
        encoded += [char_to_id["<pad>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return np.array(encoded)

def decode_sequence(sequence):
    return "".join([id_to_char.get(i, '') for i in sequence if i != char_to_id["<pad>"]])

x_data, y_data_input, y_data_target = [], [], []
for _ in range(NUM_SAMPLES):
    num1 = np.random.randint(1, 5)
    num2 = np.random.randint(1, 5)
    input_seq = f"{num1} {num2}"
    output_seq = f"{num1 + num2}"

    x_data.append(encode_sequence(input_seq, MAX_SEQUENCE_LENGTH))
    y_data_input.append(encode_sequence(f"<start> {output_seq}", MAX_SEQUENCE_LENGTH))
    y_data_target.append(encode_sequence(f"{output_seq} <end>", MAX_SEQUENCE_LENGTH))

x_data = np.array(x_data)
y_data_input = np.array(y_data_input)
y_data_target = np.array(y_data_target)

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(((x_data, y_data_input), y_data_target))
BATCH_SIZE = 4
dataset = dataset.shuffle(NUM_SAMPLES).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Input data shape: {x_data.shape}")
print(f"Target input shape: {y_data_input.shape}")
print(f"Target output shape: {y_data_target.shape}")

# --- Transformer Block ---
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# --- Positional Embedding ---
class TokenAndPositionalEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_emb = layers.Embedding(maxlen, embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# --- Encoder (for input sequence) ---
def build_encoder(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks):
    encoder_inputs = layers.Input(shape=(maxlen,))
    x = TokenAndPositionalEmbedding(maxlen, vocab_size, embed_dim)(encoder_inputs)
    for _ in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    encoder_outputs = x
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")

# --- Decoder (for output sequence) ---
class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, inputs, encoder_outputs, training, look_ahead_mask=None, padding_mask=None):
        # Self-attention (masked)
        attn1_output = self.att1(inputs, inputs, attention_mask=look_ahead_mask)
        attn1_output = self.dropout1(attn1_output, training=training)
        out1 = self.layernorm1(inputs + attn1_output)

        # Encoder-decoder attention
        attn2_output = self.att2(out1, encoder_outputs, attention_mask=padding_mask)
        attn2_output = self.dropout2(attn2_output, training=training)
        out2 = self.layernorm2(out1 + attn2_output)

        # Feed-forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

# --- Full Transformer Model (Encoder-Decoder) ---
class Transformer(keras.Model):
    def __init__(self, encoder, decoder, target_vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.final_layer = layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        inp, tar = inputs

        # Encoder output
        enc_output = self.encoder(inp, training=training)

        # Decoder input and masks
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        padding_mask = self.create_padding_mask(inp)

        # Decoder output
        dec_output = self.decoder(tar, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, char_to_id["<pad>"]), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask # (seq_len, seq_len)

# --- Model Instantiation and Training ---
NUM_HEADS = 2
FF_DIM = 128 # Feed-forward network dimension
NUM_TRANSFORMER_BLOCKS = 2

encoder_model = build_encoder(MAX_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, FF_DIM, NUM_TRANSFORMER_BLOCKS)
decoder_model = DecoderBlock(EMBEDDING_DIM, NUM_HEADS, FF_DIM)

transformer = Transformer(encoder_model, decoder_model, VOCAB_SIZE)

# Custom loss function to ignore padding in target
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, char_to_id["<pad>"]))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# Custom accuracy metric
def masked_accuracy(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, char_to_id["<pad>"]))
    accuracies = tf.cast(tf.equal(real, tf.argmax(pred, axis=2)), tf.float32)
    mask = tf.cast(mask, dtype=accuracies.dtype)
    accuracies *= mask
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

transformer.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=masked_loss, metrics=[masked_accuracy])

transformer.summary()

print("\n--- Training Transformer Model ---")
history_transformer = transformer.fit(
    dataset,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Inference Example ---
print("\n--- Making a Transformer Prediction ---")

def predict_transformer(input_sentence):
    encoder_input = encode_sequence(input_sentence, MAX_SEQUENCE_LENGTH)
    encoder_input = tf.expand_dims(encoder_input, 0) # Add batch dim

    decoder_input = encode_sequence("<start>", MAX_SEQUENCE_LENGTH)
    decoder_input = tf.expand_dims(decoder_input, 0) # Add batch dim

    output_sequence = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    output_sequence = output_sequence.write(0, char_to_id["<start>"])

    for i in tf.range(MAX_SEQUENCE_LENGTH):
        predictions = transformer((encoder_input, decoder_input), training=False)
        predictions = predictions[:, i, :]
        predicted_id = tf.argmax(predictions, axis=-1)[0].numpy()

        output_sequence = output_sequence.write(i+1, predicted_id)

        if predicted_id == char_to_id["<end>"]:
            break

        # Update decoder input for next step
        new_decoder_input = tf.TensorArray(tf.int32, size=MAX_SEQUENCE_LENGTH)
        for j in tf.range(i + 2):
            new_decoder_input = new_decoder_input.write(j, output_sequence.read(j))
        decoder_input = tf.expand_dims(new_decoder_input.stack(), 0)
        decoder_input = tf.cast(decoder_input, tf.int64) # Ensure correct dtype

    return decode_sequence(output_sequence.stack().numpy())

sample_input_text = "3 4"
predicted_output_text = predict_transformer(sample_input_text)

print(f"Input: '{sample_input_text}'")
print(f"Predicted Output: '{predicted_output_text}'")
```

#### 9.2.4. 시퀀스-투-시퀀스 모델 (Seq2Seq, NMT)

*   **개념:** 하나의 시퀀스를 입력으로 받아 다른 시퀀스를 출력으로 생성하는 모델입니다 (예: 기계 번역, 챗봇).
*   **실무 관점:**
    *   **인코더-디코더 구조:** 입력 시퀀스를 인코더가 고정된 크기의 컨텍스트 벡터로 압축하고, 디코더가 이 컨텍스트 벡터를 기반으로 출력 시퀀스를 생성합니다.
    *   **어텐션 메커니즘:** 디코더가 출력 시퀀스를 생성할 때 입력 시퀀스의 특정 부분에 집중할 수 있도록 하여 성능을 크게 향상시킵니다.
    *   **활용 사례:** 기계 번역(NMT), 텍스트 요약, 챗봇 대화 생성 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Simple sequence-to-sequence task: reverse a sequence of numbers.
# Input: "1 2 3 4" -> Output: "4 3 2 1"

VOCAB_SIZE = 15 # Digits 0-9, space, <start>, <end>, <pad>
MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 64
NUM_SAMPLES = 100

# Create a simple vocabulary
char_to_id = {str(i): i for i in range(10)}
char_to_id[" "] = 10
char_to_id["<start>"] = 11
char_to_id["<end>"] = 12
char_to_id["<pad>"] = 13

id_to_char = {i: char for char, i in char_to_id.items()}

def encode_sequence(text, max_len, add_start_end=False):
    if add_start_end:
        text = "<start> " + text + " <end>"
    encoded = [char_to_id.get(c, char_to_id["<pad>"]) for c in text]
    if len(encoded) < max_len:
        encoded += [char_to_id["<pad>"]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return np.array(encoded)

def decode_sequence(sequence):
    return "".join([id_to_char.get(i, "") for i in sequence if i != char_to_id["<pad>"] and i != char_to_id["<start>"] and i != char_to_id["<end>"]])

x_data, y_data_input, y_data_target = [], [], []
for _ in range(NUM_SAMPLES):
    nums = [str(np.random.randint(0, 10)) for _ in range(np.random.randint(3, 8))] # 3 to 7 digits
    input_seq = " ".join(nums)
    output_seq = " ".join(nums[::-1]) # Reversed

    x_data.append(encode_sequence(input_seq, MAX_SEQUENCE_LENGTH))
    y_data_input.append(encode_sequence(output_seq, MAX_SEQUENCE_LENGTH, add_start_end=True))
    y_data_target.append(encode_sequence(output_seq, MAX_SEQUENCE_LENGTH, add_start_end=True))

x_data = np.array(x_data)
y_data_input = np.array(y_data_input)
y_data_target = np.array(y_data_target)

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(((x_data, y_data_input), y_data_target))
BATCH_SIZE = 4
dataset = dataset.shuffle(NUM_SAMPLES).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Input data shape: {x_data.shape}")
print(f"Target input shape: {y_data_input.shape}")
print(f"Target output shape: {y_data_target.shape}")

# --- Encoder ---
class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = layers.GRU(self.enc_units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# --- Decoder (with simple attention for context) ---
class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = layers.GRU(self.dec_units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')
        self.fc = layers.Dense(vocab_size)

        # Simple attention mechanism (dot product attention)
        self.W1 = layers.Dense(self.dec_units)
        self.W2 = layers.Dense(self.dec_units)
        self.V = layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, enc_units)
        # hidden shape == (batch_size, dec_units)

        # context_vector, attention_weights = self.attention(hidden, enc_output)
        # Simplified dot product attention
        score = tf.matmul(tf.expand_dims(hidden, 1), enc_output, transpose_b=True) # (batch_size, 1, max_length)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, enc_output) # (batch_size, 1, enc_units)
        context_vector = tf.squeeze(context_vector, axis=1) # (batch_size, enc_units)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + enc_units)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x, initial_state=hidden)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state

# --- Model Instantiation ---
ENCODER_UNITS = 256
DECODER_UNITS = 256

encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, ENCODER_UNITS, BATCH_SIZE)
decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM, DECODER_UNITS, BATCH_SIZE)

# --- Loss Function and Optimizer ---
optimizer = keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, char_to_id["<pad>"]))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# --- Training Step (Custom Training Loop) ---
@tf.function
def train_step_seq2seq(inp, targ_inp, targ_real):
    loss = 0

    with tf.GradientTape() as tape:
        enc_hidden = encoder.initialize_hidden_state()
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ_real.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden = decoder(tf.expand_dims(targ_inp[:, t], 1), dec_hidden, enc_output)

            loss += loss_function(targ_real[:, t], predictions)

    batch_loss = (loss / int(targ_real.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# --- Training Loop ---
EPOCHS = 1 # Increase for better results

print("\n--- Training Seq2Seq Model ---")
for epoch in range(EPOCHS):
    total_loss = 0

    for (batch, ((inp, targ_inp), targ_real)) in enumerate(dataset):
        batch_loss = train_step_seq2seq(inp, targ_inp, targ_real)
        total_loss += batch_loss

        if batch % 10 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
    
    print(f'Epoch {epoch+1} Loss {total_loss/len(dataset):.4f}')

# --- Inference Example ---
print("\n--- Generating Sequence from Seq2Seq Model ---")

def translate_sequence(input_sequence_raw):
    input_sequence = encode_sequence(input_sequence_raw, MAX_SEQUENCE_LENGTH)
    input_sequence = tf.expand_dims(input_sequence, 0) # Add batch dim

    enc_hidden = encoder.initialize_hidden_state()
    enc_output, enc_hidden = encoder(input_sequence, enc_hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([char_to_id["<start>"]], 0)

    result = []

    for t in range(MAX_SEQUENCE_LENGTH):
        predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)

        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_char = id_to_char[predicted_id]
        result.append(predicted_char)

        if predicted_char == '<end>':
            break

        dec_input = tf.expand_dims([predicted_id], 0)

    return ''.join(result)

# Test with a sample input
sample_input = "1 2 3"
predicted_output = translate_sequence(sample_input)

print(f"Input: '{sample_input}'")
print(f"Predicted Output: '{predicted_output}'")
```

#### 9.2.5. 개체명 인식 (Named Entity Recognition, NER)

*   **개념:** 텍스트에서 사람 이름, 장소, 조직, 날짜 등 미리 정의된 개체명을 식별하고 분류하는 태스크입니다.
*   **실무 관점:**
    *   **모델 아키텍처:** 주로 Bi-LSTM-CRF(Conditional Random Field) 또는 Transformer 기반 모델이 사용됩니다. CRF 레이어는 출력 레이블 간의 의존성을 모델링하여 예측의 일관성을 높입니다.
    *   **활용 사례:** 정보 추출, 검색 엔진, 챗봇, 의료 기록 분석 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# In a real scenario, you would use a dataset like CoNLL-2003.
# Here, we simulate a small dataset of sentences and their corresponding NER tags.

# Define vocabulary and tag set
words = ["<PAD>", "<UNK>", "apple", "banana", "john", "doe", "london", "paris", "eats", "visits", "is", "a", "."]
tags = ["<PAD>", "O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]

word_to_id = {word: i for i, word in enumerate(words)}
id_to_word = {i: word for i, word in enumerate(words)}
tag_to_id = {tag: i for i, tag in enumerate(tags)}
id_to_tag = {i: tag for i, tag in enumerate(tags)}

VOCAB_SIZE = len(words)
NUM_TAGS = len(tags)
MAX_SEQUENCE_LENGTH = 10

# Dummy sentences and their NER tags
dummy_sentences = [
    ["john", "doe", "visits", "london", "."],
    ["apple", "is", "a", "fruit", "."], # "fruit" will be <UNK>
    ["john", "eats", "banana", "."],
    ["doe", "from", "paris", "."],
]
dummy_tags = [
    ["B-PER", "I-PER", "O", "B-LOC", "O"],
    ["O", "O", "O", "O", "O"],
    ["B-PER", "O", "O", "O"],
    ["I-PER", "O", "B-LOC", "O"],
]

# Convert to integer sequences and pad
def preprocess_sentence(sentence, tags):
    word_ids = [word_to_id.get(w, word_to_id["<UNK>"]) for w in sentence]
    tag_ids = [tag_to_id.get(t, tag_to_id["O"]) for t in tags]

    # Pad sequences
    word_ids = word_ids + [word_to_id["<PAD>"]] * (MAX_SEQUENCE_LENGTH - len(word_ids))
    tag_ids = tag_ids + [tag_to_id["<PAD>"]] * (MAX_SEQUENCE_LENGTH - len(tag_ids))
    
    return np.array(word_ids[:MAX_SEQUENCE_LENGTH]), np.array(tag_ids[:MAX_SEQUENCE_LENGTH])

x_data, y_data = [], []
for sentence, tag_seq in zip(dummy_sentences, dummy_tags):
    x, y = preprocess_sentence(sentence, tag_seq)
    x_data.append(x)
    y_data.append(y)

x_data = np.array(x_data)
y_data = np.array(y_data)

print(f"Input data shape (X): {x_data.shape}")
print(f"Target data shape (y): {y_data.shape}")

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
BATCH_SIZE = 2
dataset = dataset.shuffle(len(x_data)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Bi-LSTM-CRF Model for NER ---
# This model combines Bidirectional LSTMs for sequence modeling
# and a Conditional Random Field (CRF) layer for sequence tagging.
# The CRF layer learns dependencies between output tags.

# Note: Keras does not have a built-in CRF layer. You would typically use
# a custom implementation or a library like `tensorflow_addons` or `keras_contrib`.
# For this example, we will simulate the CRF loss and decoding.
# A full CRF implementation is complex and beyond a simple example.
# We will use a TimeDistributed Dense layer as a proxy for simplicity.

# A more complete implementation would involve:
# from tensorflow_addons.text import crf_log_likelihood, crf_decode
# from tensorflow_addons.layers import CRF

class NERModel(keras.Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_tags, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.bi_lstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.TimeDistributed(layers.Dense(num_tags, activation='softmax'))
        # In a real Bi-LSTM-CRF, a CRF layer would go here instead of softmax
        # self.crf = CRF(num_tags)

    def call(self, inputs):
        mask = self.embedding.compute_mask(inputs)
        x = self.embedding(inputs)
        x = self.bi_lstm(x, mask=mask)
        x = self.dropout(x)
        logits = self.classifier(x)
        # In a real Bi-LSTM-CRF, you'd pass logits to CRF layer
        # output = self.crf(logits)
        return logits # Return logits for custom loss

# --- Custom Loss Function for NER (with masking) ---
# For sequence tagging, we need to ignore padding tokens in loss calculation.

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    # y_true: (batch_size, sequence_length)
    # y_pred: (batch_size, sequence_length, num_tags)

    # Create a mask from y_true (where y_true is not <PAD>_id)
    mask = tf.math.logical_not(tf.math.equal(y_true, tag_to_id["<PAD>"]))
    
    # Calculate sparse categorical crossentropy
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    loss_ = loss_object(y_true, y_pred)

    # Apply the mask
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    # Return mean loss over non-padded elements
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

# --- Model Instantiation and Compilation ---
EMBEDDING_DIM = 100
LSTM_UNITS = 128

ner_model = NERModel(VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS, NUM_TAGS)

ner_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=masked_sparse_categorical_crossentropy, # Use custom masked loss
    metrics=['accuracy'] # Accuracy will also be affected by masking
)

ner_model.summary()

# --- Training Example ---
print("\n--- Training NER Model ---")
history_ner = ner_model.fit(
    dataset,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making an NER Prediction ---")

def predict_ner(model, sentence_raw):
    word_ids, _ = preprocess_sentence(sentence_raw, ["O"] * len(sentence_raw)) # Dummy tags for preprocessing
    word_ids = np.array([word_ids]) # Add batch dimension

    predictions = model.predict(word_ids)
    predicted_tag_ids = np.argmax(predictions[0], axis=-1) # Get the tag with highest probability

    # Convert predicted IDs back to tags, ignoring padding
    predicted_tags = []
    for i, tag_id in enumerate(predicted_tag_ids):
        if word_ids[0, i] == word_to_id["<PAD>"]:
            break # Stop at padding
        predicted_tags.append(id_to_tag[tag_id])
    return predicted_tags

sample_sentence = ["john", "doe", "from", "london"]
predicted_tags = predict_ner(ner_model, sample_sentence)

print(f"Sentence: {' '.join(sample_sentence)}")
print(f"Predicted Tags: {predicted_tags}")
```

#### 9.2.6. 질의응답 (Question Answering)

*   **개념:** 주어진 텍스트(문맥)에서 질문에 대한 답변을 찾아내거나 생성하는 태스크입니다.
*   **실무 관점:**
    *   **모델 아키텍처:** BERT, RoBERTa 등 사전 학습된 Transformer 모델이 주로 사용됩니다. SQuAD(Stanford Question Answering Dataset)와 같은 데이터셋으로 학습됩니다.
    *   **활용 사례:** 고객 서비스 챗봇, 지식 검색 시스템, 법률 문서 분석 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# In a real scenario, you would use a dataset like SQuAD (Stanford Question Answering Dataset).
# Here, we simulate a very small dataset for a simplified extractive QA task.

# Contexts, Questions, and Answers
contexts = [
    "The quick brown fox jumps over the lazy dog. John is a student.",
    "Paris is the capital of France. It is known for its Eiffel Tower.",
    "Artificial intelligence is a rapidly developing field. Machine learning is a subfield of AI."
]
questions = [
    "Who is a student?",
    "What is the capital of France?",
    "What is a subfield of AI?"
]
answers = [
    "John",
    "Paris",
    "Machine learning"
]

# For extractive QA, we need the start and end character indices of the answer in the context.
# This is a simplified manual mapping for dummy data.
# In real datasets, these indices are provided.
answer_start_indices = [39, 0, 31]
answer_end_indices = [42, 4, 46]

# --- Tokenization and Input Preparation (Simplified BERT-like) ---
# A real BERT model uses a WordPiece tokenizer and specific input formats
# (token_ids, segment_ids, attention_mask).
# For this example, we'll use a simple TextVectorization and simulate the input structure.

VOCAB_SIZE = 200 # Example vocabulary size
MAX_SEQUENCE_LENGTH = 64 # Max length of [CLS] context [SEP] question [SEP]

text_vectorization = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_SEQUENCE_LENGTH,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ragged=False,
)

# Adapt the TextVectorization layer to all text (contexts + questions)
all_texts = contexts + questions
text_vectorization.adapt(all_texts)

# Get special tokens IDs (assuming they are at the end of vocabulary after adaptation)
CLS_ID = text_vectorization.vocabulary_size() - 3 # [CLS]
SEP_ID = text_vectorization.vocabulary_size() - 2 # [SEP]
PADDING_ID = 0 # Default padding token

# Function to create BERT-like input
def create_qa_input(context, question, answer_start, answer_end):
    # Tokenize context and question
    context_tokens = text_vectorization([context]).numpy()[0]
    question_tokens = text_vectorization([question]).numpy()[0]

    # Remove padding from tokens for concatenation
    context_tokens = context_tokens[context_tokens != PADDING_ID]
    question_tokens = question_tokens[question_tokens != PADDING_ID]

    # Construct input sequence: [CLS] context [SEP] question [SEP]
    input_tokens = np.concatenate([
        [CLS_ID],
        context_tokens,
        [SEP_ID],
        question_tokens,
        [SEP_ID]
    ])

    # Pad or truncate to MAX_SEQUENCE_LENGTH
    if len(input_tokens) > MAX_SEQUENCE_LENGTH:
        input_tokens = input_tokens[:MAX_SEQUENCE_LENGTH]
    else:
        input_tokens = np.pad(input_tokens, (0, MAX_SEQUENCE_LENGTH - len(input_tokens)), 'constant', constant_values=PADDING_ID)

    # Create segment IDs (0 for context, 1 for question)
    segment_ids = np.zeros_like(input_tokens)
    context_len = len(context_tokens)
    question_len = len(question_tokens)
    segment_ids[1 + context_len + 1 : 1 + context_len + 1 + question_len + 1] = 1

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_tokens != PADDING_ID).astype(np.int32)

    # Calculate answer token start/end indices within the new tokenized sequence
    # This is highly simplified and might not be accurate for complex tokenizations.
    # In real BERT, you map char indices to token indices.
    start_token_idx = -1
    end_token_idx = -1
    current_char_idx = 0
    for i, token_id in enumerate(input_tokens):
        if token_id == CLS_ID or token_id == SEP_ID or token_id == PADDING_ID:
            continue
        word = id_to_word.get(token_id, '')
        if current_char_idx <= answer_start < current_char_idx + len(word):
            start_token_idx = i
        if current_char_idx <= answer_end < current_char_idx + len(word):
            end_token_idx = i
            break
        current_char_idx += len(word) + 1 # +1 for space
    
    # If answer not found or spans across tokens, set to CLS token (0) for unanswerable
    if start_token_idx == -1 or end_token_idx == -1:
        start_token_idx = 0
        end_token_idx = 0

    return {
        'input_token_ids': input_tokens,
        'input_segment_ids': segment_ids,
        'input_attention_mask': attention_mask
    }, {
        'start_token_idx': start_token_idx,
        'end_token_idx': end_token_idx
    }

x_inputs, y_targets = [], []
for i in range(len(contexts)):
    x, y = create_qa_input(contexts[i], questions[i], answer_start_indices[i], answer_end_indices[i])
    x_inputs.append(x)
    y_targets.append(y)

# Convert lists of dicts to dict of arrays for tf.data.Dataset
x_input_token_ids = np.array([x['input_token_ids'] for x in x_inputs])
x_input_segment_ids = np.array([x['input_segment_ids'] for x in x_inputs])
x_input_attention_mask = np.array([x['input_attention_mask'] for x in x_inputs])

y_start_token_idx = np.array([y['start_token_idx'] for y in y_targets])
y_end_token_idx = np.array([y['end_token_idx'] for y in y_targets])

qa_dataset = tf.data.Dataset.from_tensor_slices(
    ({
        'input_token_ids': x_input_token_ids,
        'input_segment_ids': x_input_segment_ids,
        'input_attention_mask': x_input_attention_mask
    },
    {
        'start_token_idx': y_start_token_idx,
        'end_token_idx': y_end_token_idx
    })
)
BATCH_SIZE = 1 # Small batch size for dummy data
qa_dataset = qa_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Input token IDs shape: {x_input_token_ids.shape}")
print(f"Target start token IDs shape: {y_start_token_idx.shape}")

# --- BERT-like QA Model ---
# This is a simplified model. A real BERT QA model would use a pre-trained BERT
# encoder and add a classification head on top to predict start/end logits.

class QAModel(keras.Model):
    def __init__(self, max_sequence_length, vocab_size, embedding_dim, num_transformer_blocks, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = layers.Embedding(vocab_size, embedding_dim)
        self.position_embedding = layers.Embedding(max_sequence_length, embedding_dim)
        self.segment_embedding = layers.Embedding(2, embedding_dim) # 0 for context, 1 for question

        self.transformer_blocks = []
        for _ in range(num_transformer_blocks):
            self.transformer_blocks.append(TransformerBlock(embedding_dim, num_heads, ff_dim))
        
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(0.1)

        # Output heads for start and end token prediction
        self.start_head = layers.Dense(1, name="start_token_idx")
        self.end_head = layers.Dense(1, name="end_token_idx")

    def call(self, inputs):
        input_token_ids = inputs['input_token_ids']
        input_segment_ids = inputs['input_segment_ids']
        input_attention_mask = inputs['input_attention_mask']

        seq_len = tf.shape(input_token_ids)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)

        token_embeddings = self.token_embedding(input_token_ids)
        position_embeddings = self.position_embedding(positions)
        segment_embeddings = self.segment_embedding(input_segment_ids)

        x = token_embeddings + position_embeddings + segment_embeddings
        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, input_attention_mask)
        
        x = self.norm(x)

        # Predict start and end logits for each token in the sequence
        start_logits = self.start_head(x) # (batch_size, sequence_length, 1)
        end_logits = self.end_head(x)   # (batch_size, sequence_length, 1)

        # Squeeze the last dimension
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        return {'start_token_idx': start_logits, 'end_token_idx': end_logits}

# Transformer Block (simplified)
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, attention_mask=None, training=False):
        attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# --- Model Instantiation and Compilation ---
EMBEDDING_DIM = 128
NUM_TRANSFORMER_BLOCKS = 2
NUM_HEADS = 2
FF_DIM = 512

qa_model = QAModel(
    max_sequence_length=MAX_SEQUENCE_LENGTH,
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM
)

# Custom loss for QA: SparseCategoricalCrossentropy for start and end logits
qa_losses = {
    'start_token_idx': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'end_token_idx': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
}

qa_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=qa_losses
)

qa_model.summary()

# --- Training Example ---
print("\n--- Training QA Model ---")
history_qa = qa_model.fit(
    qa_dataset,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making a QA Prediction ---")

def predict_answer(model, context, question):
    # Prepare input for prediction
    x_pred, _ = create_qa_input(context, question, -1, -1) # No answer indices needed for prediction
    
    # Add batch dimension
    x_pred_batched = {
        'input_token_ids': tf.expand_dims(x_pred['input_token_ids'], 0),
        'input_segment_ids': tf.expand_dims(x_pred['input_segment_ids'], 0),
        'input_attention_mask': tf.expand_dims(x_pred['input_attention_mask'], 0)
    }

    # Get logits from the model
    predictions = model.predict(x_pred_batched)
    start_logits = predictions['start_token_idx'][0].numpy()
    end_logits = predictions['end_token_idx'][0].numpy()

    # Find the best span (simplified: just take argmax)
    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)

    # Convert token IDs back to words
    input_tokens = x_pred['input_token_ids']
    tokens = [id_to_word.get(token_id, '') for token_id in input_tokens]

    # Extract answer span
    if start_index <= end_index < len(tokens):
        predicted_answer_tokens = tokens[start_index : end_index + 1]
        predicted_answer = " ".join(predicted_answer_tokens)
    else:
        predicted_answer = ""

    return predicted_answer

# Test with a sample
sample_context = "The Amazon river is the largest river by discharge volume of water in the world."
sample_question = "What is the largest river by discharge volume?"

predicted_answer = predict_answer(qa_model, sample_context, sample_question)
print(f"Context: {sample_context}")
print(f"Question: {sample_question}")
print(f"Predicted Answer: {predicted_answer}")
```

#### 9.2.7. 음성 인식 (Speech Recognition)

*   **개념:** 음성 오디오를 입력으로 받아 텍스트로 변환하는 태스크입니다.
*   **실무 관점:**
    *   **전처리:** 음성 신호를 MFCC(Mel-Frequency Cepstral Coefficients)와 같은 특징 벡터로 변환합니다.
    *   **모델 아키텍처:** RNN, LSTM, GRU, Transformer 기반 모델이 사용됩니다. CTC(Connectionist Temporal Classification) 손실 함수가 자주 사용됩니다.
    *   **활용 사례:** 음성 비서, 음성 명령 시스템, 회의록 자동 생성 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# In a real scenario, you would use an audio dataset like LibriSpeech or Common Voice.
# Here, we simulate a very small dataset of audio features (MFCCs) and corresponding transcripts.

# Audio features: MFCCs (Mel-Frequency Cepstral Coefficients)
# Simulate 10 audio samples, each with 100 time steps and 13 MFCC features.
NUM_AUDIO_SAMPLES = 10
MAX_AUDIO_TIMESTEPS = 100
NUM_MFCCS = 13

dummy_audio_features = np.random.rand(NUM_AUDIO_SAMPLES, MAX_AUDIO_TIMESTEPS, NUM_MFCCS).astype(np.float32)

# Transcripts (sequences of characters/phonemes)
# We'll use a simple character-level vocabulary.

characters = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_num = layers.StringLookup(vocabulary=list(characters), oov_token="")
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

MAX_TRANSCRIPT_LENGTH = 20

dummy_transcripts_raw = [
    "hello world",
    "speech recognition",
    "keras example",
    "deep learning",
    "audio processing",
    "neural networks",
    "python code",
    "machine learning",
    "data science",
    "artificial intelligence"
]

# Convert raw transcripts to padded integer sequences
def text_to_sequence(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^a-z ]', '') # Remove non-alphabetic chars
    sequence = char_to_num(tf.strings.unicode_split(text, input_encoding='UTF-8'))
    # Pad or truncate
    if tf.shape(sequence)[0] < MAX_TRANSCRIPT_LENGTH:
        padding = tf.zeros(MAX_TRANSCRIPT_LENGTH - tf.shape(sequence)[0], dtype=tf.int64)
        sequence = tf.concat([sequence, padding], axis=0)
    else:
        sequence = sequence[:MAX_TRANSCRIPT_LENGTH]
    return sequence

dummy_transcripts_sequences = tf.map_fn(text_to_sequence, tf.constant(dummy_transcripts_raw), fn_output_signature=tf.TensorSpec(shape=(MAX_TRANSCRIPT_LENGTH,), dtype=tf.int64))

# Create tf.data.Dataset
# For CTC, we need input_length and label_length for each sample.
# input_length: number of timesteps in audio features
# label_length: number of characters in transcript

def prepare_ctc_data(audio_features, transcript_sequence):
    audio_len = tf.shape(audio_features)[0]
    label_len = tf.reduce_sum(tf.cast(tf.not_equal(transcript_sequence, 0), tf.int32)) # Count non-padding chars
    return audio_features, transcript_sequence, audio_len, label_len


dataset = tf.data.Dataset.from_tensor_slices((dummy_audio_features, dummy_transcripts_sequences))
dataset = dataset.map(prepare_ctc_data, num_parallel_calls=tf.data.AUTOTUNE)
BATCH_SIZE = 2
dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=(
    tf.TensorShape([None, NUM_MFCCS]),  # Audio features (variable length)
    tf.TensorShape([None]),             # Transcript sequence (variable length)
    tf.TensorShape([]),                 # Audio length
    tf.TensorShape([])                  # Label length
)).prefetch(tf.data.AUTOTUNE)

print(f"Dummy audio features shape: {dummy_audio_features.shape}")
print(f"Dummy transcripts sequences shape: {dummy_transcripts_sequences.shape}")

# --- Speech Recognition Model (DeepSpeech-like CNN-RNN with CTC) ---
# This model uses convolutional layers for feature extraction, recurrent layers
# (GRU/LSTM) for sequence modeling, and a CTC loss layer for alignment.

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the CTC loss value
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the predictions
        return y_pred

def build_speech_recognition_model(input_shape, output_dim):
    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name="audio_features")
    labels = layers.Input(shape=(None,), dtype=tf.int64, name="labels")
    input_length = layers.Input(shape=(), dtype=tf.int32, name="input_length")
    label_length = layers.Input(shape=(), dtype=tf.int32, name="label_length")

    # CNN layers for feature extraction
    x = layers.Conv1D(32, 5, strides=2, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 5, strides=2, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 5, strides=1, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Reshape for RNN (flatten time and features if needed, or use TimeDistributed)
    # If strides in Conv1D reduce time dimension, adjust accordingly.
    # Here, we assume the time dimension is still significant.

    # RNN layers (GRU or LSTM)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True, dropout=0.25))(x)

    # Output layer (predicts characters/phonemes)
    outputs = layers.Dense(output_dim + 1, activation="softmax")(x) # +1 for blank token in CTC

    # Add CTC layer for training
    output_with_ctc_loss = CTCLayer(name="ctc_loss")(labels, outputs, input_length, label_length)

    # Define the model for training
    model = keras.Model(
        inputs=[inputs, labels, input_length, label_length],
        outputs=output_with_ctc_loss
    )

    # Define the model for inference (without CTC layer)
    inference_model = keras.Model(inputs=inputs, outputs=outputs)

    return model, inference_model

# Build the model
training_model, inference_model = build_speech_recognition_model(
    input_shape=(None, NUM_MFCCS), # Time dimension is None for variable length
    output_dim=len(characters) # Number of unique characters
)

training_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))

training_model.summary()

# --- Training Example ---
print("\n--- Training Speech Recognition Model ---")
history_asr = training_model.fit(
    dataset,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Inference Example ---
print("\n--- Making a Speech Recognition Prediction ---")

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For a real application, beam search is better.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for res in results.numpy():
        output_text.append(tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8'))
    return output_text

# Get a batch of dummy audio features for prediction
sample_audio_batch = dummy_audio_features[0:BATCH_SIZE]

# Get predictions from the inference model
raw_predictions = inference_model.predict(sample_audio_batch)

# Decode the predictions
decoded_texts = decode_batch_predictions(raw_predictions)

print("\n--- Decoded Predictions ---")
for i, text in enumerate(decoded_texts):
    print(f"Sample {i+1}: '{text}'")
    print(f"  Original: '{dummy_transcripts_raw[i]}'")
```

#### 9.2.8. 텍스트 요약 (Text Summarization)

*   **개념:** 긴 텍스트 문서를 입력으로 받아 핵심 내용을 담은 짧은 요약문을 생성하는 태스크입니다.
*   **실무 관점:**
    *   **추출 요약 (Extractive Summarization):** 원문에서 중요한 문장이나 구절을 추출하여 요약합니다.
    *   **추상 요약 (Abstractive Summarization):** 원문의 내용을 이해하고 새로운 문장으로 요약문을 생성합니다. Seq2Seq 모델과 Transformer 기반 모델이 주로 사용됩니다.
    *   **활용 사례:** 뉴스 기사 요약, 보고서 요약, 논문 요약 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# In a real scenario, you would use a dataset like CNN/Daily Mail or XSum.
# Here, we simulate a small dataset of (document, summary) pairs.

# Vocabulary parameters
VOCAB_SIZE = 2000
MAX_DOC_LENGTH = 50
MAX_SUMMARY_LENGTH = 15
EMBEDDING_DIM = 128

# Dummy documents and summaries
dummy_documents_raw = [
    "The quick brown fox jumps over the lazy dog. This is a test document for summarization.",
    "Artificial intelligence is a rapidly developing field. Machine learning is a subfield of AI and is used in many applications.",
    "Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.",
    "The weather today is sunny and warm. It is a perfect day for outdoor activities like hiking or swimming.",
    "Global warming is a serious issue. Rising temperatures are causing significant changes to our planet."
] * 5 # Repeat to get more samples

dummy_summaries_raw = [
    "Fox jumps over dog. Test summarization.",
    "AI is developing fast. ML is subfield of AI.",
    "Keras is a high-level API for neural networks.",
    "Sunny and warm weather today. Perfect for outdoors.",
    "Global warming is serious. Temperatures are rising."
] * 5

# Add <start> and <end> tokens to summaries for sequence generation
dummy_summaries_input = ["<start> " + s for s in dummy_summaries_raw]
dummy_summaries_target = [s + " <end>" for s in dummy_summaries_raw]

# TextVectorization for documents and summaries
doc_vectorization = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_DOC_LENGTH,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ragged=False,
)
doc_vectorization.adapt(dummy_documents_raw)

summary_vectorization = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_SUMMARY_LENGTH,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ragged=False,
)
summary_vectorization.adapt(dummy_summaries_input + dummy_summaries_target)

# Convert raw texts to integer sequences
x_docs = doc_vectorization(tf.constant(dummy_documents_raw)).numpy()
x_summaries_input = summary_vectorization(tf.constant(dummy_summaries_input)).numpy()
y_summaries_target = summary_vectorization(tf.constant(dummy_summaries_target)).numpy()

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(
    ((x_docs, x_summaries_input), y_summaries_target)
)
BATCH_SIZE = 4
dataset = dataset.shuffle(len(x_docs)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Document input shape: {x_docs.shape}")
print(f"Summary input shape: {x_summaries_input.shape}")
print(f"Summary target shape: {y_summaries_target.shape}")

# --- Seq2Seq Model with Attention for Text Summarization ---

# Encoder
class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(self.enc_units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# Attention Mechanism (Bahdanau Attention)
class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# Decoder
class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(self.dec_units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')
        self.fc = layers.Dense(vocab_size)

        # Used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, enc_units)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x, initial_state=hidden)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

# --- Model Instantiation ---
ENCODER_UNITS = 256
DECODER_UNITS = 256

encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, ENCODER_UNITS, BATCH_SIZE)
decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM, DECODER_UNITS, BATCH_SIZE)

# --- Loss Function and Optimizer ---
optimizer = keras.optimizers.Adam()
loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0)) # Ignore padding
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# --- Training Step (Custom Training Loop) ---
@tf.function
def train_step_summarization(inp, targ):
    loss = 0

    with tf.GradientTape() as tape:
        enc_hidden = encoder.initialize_hidden_state()
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        # Decoder input starts with <start> token
        dec_input = tf.expand_dims([summary_vectorization.vocabulary().index("<start>")] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# --- Training Loop ---
EPOCHS = 1 # Increase for better results

print("\n--- Training Text Summarization Model ---")
for epoch in range(EPOCHS):
    total_loss = 0

    for (batch, ((inp, targ_inp), targ_real)) in enumerate(dataset):
        batch_loss = train_step_summarization(inp, targ_real)
        total_loss += batch_loss

        if batch % 10 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
    
    print(f'Epoch {epoch+1} Loss {total_loss/len(dataset):.4f}')

# --- Summarization Inference Example ---
print("\n--- Generating Summary for a Dummy Document ---")

def summarize_text(document_raw):
    document_seq = doc_vectorization(tf.constant([document_raw])).numpy()
    
    enc_hidden = encoder.initialize_hidden_state()
    enc_output, enc_hidden = encoder(document_seq, enc_hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([summary_vectorization.vocabulary().index("<start>")], 0)

    result = []

    for t in range(MAX_SUMMARY_LENGTH):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_output)

        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = summary_vectorization.get_vocabulary()[predicted_id]
        result.append(predicted_word)

        if predicted_word == '<end>':
            break

        dec_input = tf.expand_dims([predicted_id], 0)

    return ' '.join(result)

# Test with a sample document
sample_document = "The Amazon rainforest is the largest tropical rainforest in the world. It is home to an incredible diversity of plants and animals. Deforestation is a major threat to this vital ecosystem."

predicted_summary = summarize_text(sample_document)
print(f"Document: {sample_document}")
print(f"Generated Summary: {predicted_summary}")
```

### 9.3. 그래프 신경망 (Graph Neural Networks, GNN)

그래프 신경망은 그래프 구조의 데이터를 처리하는 딥러닝 모델입니다. Keras는 GNN을 직접 지원하는 레이어를 기본으로 제공하지는 않지만, Functional API나 Subclassing API를 사용하여 구현할 수 있습니다.

#### 9.3.1. GNN 기본 개념 및 Keras 구현

*   **개념:** 노드(node)와 엣지(edge)로 구성된 그래프 데이터에서 노드 분류, 링크 예측, 그래프 분류 등의 태스크를 수행합니다. GNN은 이웃 노드의 정보를 집계하여 각 노드의 임베딩을 학습합니다.
*   **Keras 구현:**
    *   **Custom Layer:** `keras.layers.Layer`를 상속받아 그래프 연산을 수행하는 Custom Layer를 구현합니다. 이웃 노드의 특징을 집계하는 로직(`tf.gather`, `tf.math.segment_sum` 등)을 `call` 메서드에 작성합니다.
    *   **입력:** 그래프 데이터는 일반적으로 인접 행렬(adjacency matrix) 또는 엣지 리스트(edge list)와 노드 특징 행렬(node feature matrix) 형태로 모델에 입력됩니다.
*   **실무 관점:**
    *   **데이터 표현:** 그래프 데이터를 Keras 모델에 입력할 수 있는 텐서 형태로 변환하는 것이 중요합니다.
    *   **메모리 효율성:** 대규모 그래프의 경우, 전체 인접 행렬을 메모리에 로드하는 것이 어려울 수 있으므로, 희소 행렬(sparse matrix) 표현이나 샘플링 기법을 고려해야 합니다.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Simulate a simple graph with node features and an adjacency list.
# This is a conceptual example to show how GNNs can be structured in Keras.

num_nodes = 10
feature_dim = 8 # Features for each node

# Node features (e.g., attributes of each node)
node_features = np.random.normal(size=(num_nodes, feature_dim)).astype(np.float32)

# Adjacency list (representing graph connections)
# For simplicity, let's say node 0 is connected to 1, 2; node 1 to 0, 3, etc.
# In a real scenario, this would be derived from graph data.
# Format: list of lists, where each inner list contains neighbors of a node.
adjacency_list = [
    [1, 2],       # Node 0 connected to 1, 2
    [0, 3, 4],    # Node 1 connected to 0, 3, 4
    [0, 5],       # Node 2 connected to 0, 5
    [1, 6],       # Node 3 connected to 1, 6
    [1, 7],       # Node 4 connected to 1, 7
    [2, 8],       # Node 5 connected to 2, 8
    [3, 9],       # Node 6 connected to 3, 9
    [4],          # Node 7 connected to 4
    [5],          # Node 8 connected to 5
    [6]           # Node 9 connected to 6
]

print(f"Node features shape: {node_features.shape}")
print(f"Adjacency list (first 3 nodes): {adjacency_list[:3]}")

# --- Custom Graph Neural Network Layer (Conceptual) ---
# This custom layer demonstrates the core idea of message passing in GNNs:
# 1. Aggregate information from neighbors.
# 2. Transform the aggregated information.

class ConceptualGNNLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape[0] is for node features (batch, N, F)
        # input_shape[1] is for adjacency list (ragged tensor or padded)
        input_dim = input_shape[0][-1]
        
        # Weight matrix for transforming node features
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            name="kernel"
        )
        # Weight matrix for transforming aggregated neighbor features
        self.neighbor_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            name="neighbor_kernel"
        )
        if self.activation is None:
            self.bias = None
        else:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                name="bias"
            )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: [node_features (batch, N, F), adjacency_list (ragged or padded)]
        node_features_batch, adjacency_list_batch = inputs
        
        batch_size = tf.shape(node_features_batch)[0]
        num_nodes = tf.shape(node_features_batch)[1]
        
        output_features = tf.TensorArray(tf.float32, size=batch_size)

        for b in tf.range(batch_size):
            current_node_features = node_features_batch[b] # (N, F)
            current_adj_list = adjacency_list_batch[b] # (N, max_neighbors) or RaggedTensor

            transformed_self = tf.matmul(current_node_features, self.kernel) # (N, units) 
            
            # Aggregate neighbor features (simplified: average of neighbors)
            aggregated_neighbors = tf.TensorArray(tf.float32, size=num_nodes)
            for i in tf.range(num_nodes):
                neighbors = tf.boolean_mask(current_adj_list[i], current_adj_list[i] != -1) # Assuming -1 for padding
                if tf.size(neighbors) > 0:
                    neighbor_features = tf.gather(current_node_features, neighbors) # (num_neighbors, F)
                    mean_neighbor_feature = tf.reduce_mean(neighbor_features, axis=0) # (F,)
                else:
                    mean_neighbor_feature = tf.zeros(feature_dim) # No neighbors
                aggregated_neighbors = aggregated_neighbors.write(i, mean_neighbor_feature)
            
            aggregated_neighbors_tensor = aggregated_neighbors.stack() # (N, F)
            transformed_neighbors = tf.matmul(aggregated_neighbors_tensor, self.neighbor_kernel) # (N, units)

            # Combine self and neighbor information
            combined_features = transformed_self + transformed_neighbors
            
            if self.bias is not None:
                combined_features = combined_features + self.bias
            
            if self.activation is not None:
                combined_features = self.activation(combined_features)
            
            output_features = output_features.write(b, combined_features)
        
        return output_features.stack()

# --- GNN Model for Node Classification (Conceptual) ---
def build_conceptual_gnn_model(input_feature_shape, input_adj_shape, num_classes):
    node_features_input = layers.Input(shape=input_feature_shape, name="node_features_input")
    # For simplicity, we'll use a padded adjacency list. In real GNNs, often use sparse tensors.
    adj_list_input = layers.Input(shape=input_adj_shape, dtype=tf.int32, name="adj_list_input")

    # First GNN layer
    x = ConceptualGNNLayer(64, activation='relu')([node_features_input, adj_list_input])
    x = layers.Dropout(0.5)(x)

    # Second GNN layer
    x = ConceptualGNNLayer(32, activation='relu')([x, adj_list_input])
    x = layers.Dropout(0.5)(x)

    # Output layer for node classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=[node_features_input, adj_list_input], outputs=outputs, name="Conceptual_GNN_Classifier")
    return model

# Prepare dummy adjacency list for batching (pad with -1)
max_neighbors = max(len(n) for n in adjacency_list)
padded_adj_list = []
for neighbors in adjacency_list:
    padded_neighbors = neighbors + [-1] * (max_neighbors - len(neighbors))
    padded_adj_list.append(padded_neighbors)
padded_adj_list = np.array(padded_adj_list, dtype=np.int32)

# Add batch dimension to inputs
x_train_features = np.expand_dims(node_features, axis=0) # (1, N, F)
x_train_adj_list = np.expand_dims(padded_adj_list, axis=0) # (1, N, max_neighbors)

# Dummy node labels for training
node_labels = np.random.randint(0, num_classes, size=(num_nodes,)).astype(np.int32)
y_train_labels = np.expand_dims(node_labels, axis=0) # (1, N)

# Build the conceptual GNN model
conceptual_gnn_model = build_conceptual_gnn_model(
    input_feature_shape=(num_nodes, feature_dim),
    input_adj_shape=(num_nodes, max_neighbors),
    num_classes=num_classes
)

# Compile the model
conceptual_gnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

conceptual_gnn_model.summary()

# --- Training Example ---
print("\n--- Training Conceptual GNN Model (Node Classification) ---")
history_conceptual_gnn = conceptual_gnn_model.fit(
    [x_train_features, x_train_adj_list],
    y_train_labels,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making a Conceptual GNN Prediction (Node Classification) ---")
predictions = conceptual_gnn_model.predict([x_train_features, x_train_adj_list])
predicted_classes = np.argmax(predictions[0], axis=-1)

print(f"True Node Labels: {node_labels}")
print(f"Predicted Node Labels: {predicted_classes}")
```

#### 9.3.2. 그래프 분류 및 노드 분류

*   **그래프 분류:** 전체 그래프를 하나의 클래스로 분류하는 태스크입니다 (예: 분자 구조의 독성 예측).
    *   **Keras 구현:** 각 노드의 임베딩을 얻은 후, 이를 풀링(예: `GlobalAveragePooling1D` 또는 `GlobalMaxPooling1D`를 사용하여 노드 임베딩을 집계)하여 그래프 전체의 임베딩을 생성하고, 이 임베딩을 분류기 레이어에 연결합니다.
*   **노드 분류:** 그래프 내의 각 노드를 특정 클래스로 분류하는 태스크입니다 (예: 소셜 네트워크에서 사용자의 관심사 분류).
    *   **Keras 구현:** GNN 레이어를 통해 각 노드의 임베딩을 얻은 후, 각 노드 임베딩에 대해 독립적으로 분류기 레이어를 적용합니다.

#### 9.3.3. GCN (Graph Convolutional Networks)

*   **개념:** 그래프 컨볼루션 연산을 통해 이웃 노드의 특징을 집계하고 변환하여 노드 임베딩을 학습하는 GNN의 한 종류입니다.
*   **Keras 구현:** Custom Layer로 GCN 레이어를 구현할 수 있습니다. 인접 행렬과 노드 특징 행렬을 입력으로 받아, 행렬 곱셈과 활성화 함수를 통해 노드 특징을 업데이트합니다.

 ```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Simulate a simple graph for node classification.
# Nodes have features, and there are connections (edges) between them.

num_nodes = 10
feature_dim = 16
num_classes = 2 # e.g., two types of nodes

# Node features (e.g., attributes of each node)
node_features = np.random.normal(size=(num_nodes, feature_dim)).astype(np.float32)

# Adjacency matrix (defines connections between nodes)
# Create a random symmetric adjacency matrix (undirected graph)
adj_matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes)).astype(np.float32)
adj_matrix = adj_matrix + adj_matrix.T # Make it symmetric
adj_matrix[adj_matrix > 1] = 1 # Ensure binary
np.fill_diagonal(adj_matrix, 0) # No self-loops initially

# Node labels (for node classification task)
node_labels = np.random.randint(0, num_classes, size=(num_nodes,)).astype(np.int32)

print(f"Node features shape: {node_features.shape}")
print(f"Adjacency matrix shape: {adj_matrix.shape}")
print(f"Node labels shape: {node_labels.shape}")

# --- Graph Convolutional Layer (Custom Keras Layer) ---
# This layer implements the core GCN operation:
# H^(l+1) = sigma(D^(-1/2) * A_hat * D^(-1/2) * H^(l) * W^(l))
# where A_hat = A + I (adjacency matrix with self-loops)
# D is the degree matrix of A_hat

class GraphConvolution(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape[0] is for node features (batch, N, F)
        # input_shape[1] is for adjacency matrix (batch, N, N)
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            name="kernel"
        )
        if self.activation is None:
            self.bias = None
        else:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                name="bias"
            )
        super().build(input_shape)

    def call(self, inputs):
        features, adj_matrix = inputs # (batch, N, F), (batch, N, N)
        
        # Add self-loops to adjacency matrix (A_hat = A + I)
        adj_matrix_with_self_loops = adj_matrix + tf.eye(tf.shape(adj_matrix)[1], batch_shape=tf.shape(adj_matrix)[:-2])
        
        # Calculate degree matrix (D_hat) and its inverse square root
        row_sum = tf.reduce_sum(adj_matrix_with_self_loops, axis=-1) # (batch, N)
        degree_inv_sqrt = tf.pow(row_sum, -0.5) # (batch, N)
        # Handle potential Inf values if a node has degree 0
        degree_inv_sqrt = tf.where(tf.math.is_inf(degree_inv_sqrt), 0., degree_inv_sqrt)
        
        # Create diagonal matrix D_hat^(-1/2)
        # This needs to be done carefully for batched input
        D_inv_sqrt = tf.linalg.diag(degree_inv_sqrt) # (batch, N, N)

        # Normalize adjacency matrix: D_hat^(-1/2) * A_hat * D_hat^(-1/2)
        normalized_adj = tf.matmul(tf.matmul(D_inv_sqrt, adj_matrix_with_self_loops), D_inv_sqrt)
        
        # Feature transformation: H^(l) * W^(l)
        transformed_features = tf.matmul(features, self.kernel)
        
        # Graph convolution: normalized_adj * transformed_features
        output = tf.matmul(normalized_adj, transformed_features)
        
        if self.bias is not None:
            output = output + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
        
        return output

# --- GCN Model for Node Classification ---
def build_gcn_model(input_feature_shape, input_adj_shape, num_classes):
    node_features_input = layers.Input(shape=input_feature_shape, name="node_features_input")
    adj_matrix_input = layers.Input(shape=input_adj_shape, name="adj_matrix_input")

    # First GCN layer
    x = GraphConvolution(64, activation='relu')([node_features_input, adj_matrix_input])
    x = layers.Dropout(0.5)(x)

    # Second GCN layer
    x = GraphConvolution(32, activation='relu')([x, adj_matrix_input])
    x = layers.Dropout(0.5)(x)

    # Output layer for node classification
    # Apply Dense layer to each node's output features (TimeDistributed is not needed for (batch, N, F) -> (batch, N, C) directly)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=[node_features_input, adj_matrix_input], outputs=outputs, name="GCN_Node_Classifier")
    return model

# Build the GCN model
gcn_model = build_gcn_model(
    input_feature_shape=(num_nodes, feature_dim), # N, F
    input_adj_shape=(num_nodes, num_nodes), # N, N
    num_classes=num_classes
)

# Compile the model
gcn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

gcn_model.summary()

# --- Training Example ---
print("\n--- Training GCN Model (Node Classification) ---")
# Prepare data for training (add batch dimension)
x_train_features = np.expand_dims(node_features, axis=0) # (1, N, F)
x_train_adj = np.expand_dims(adj_matrix, axis=0) # (1, N, N)
y_train_labels = np.expand_dims(node_labels, axis=0) # (1, N)

# In a real scenario, you'd have multiple graphs or batches of nodes.
# For this dummy example, we train on a single graph.

history_gcn = gcn_model.fit(
    [x_train_features, x_train_adj],
    y_train_labels,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making a GCN Prediction (Node Classification) ---")
# Predict on the same dummy graph
predictions = gcn_model.predict([x_train_features, x_train_adj])

# Get predicted classes for each node
predicted_classes = np.argmax(predictions[0], axis=-1)

print(f"True Node Labels: {node_labels}")
print(f"Predicted Node Labels: {predicted_classes}")
```

#### 9.3.4. GAT (Graph Attention Networks)

*   **개념:** GCN과 유사하지만, 각 노드가 이웃 노드에 대해 다른 중요도(어텐션 가중치)를 부여하여 특징을 집계하는 GNN입니다.
*   **Keras 구현:** Custom Layer로 GAT 레이어를 구현할 수 있습니다. 어텐션 메커니즘을 추가하여 이웃 노드 간의 관계를 더 유연하게 모델링합니다.

    **예시 (간단한 GAT Layer):**
    ```python
    import keras
    from keras import layers
    import tensorflow as tf

    class GraphAttention(layers.Layer):
        def __init__(self, units, num_heads, activation=None, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.num_heads = num_heads
            self.activation = keras.activations.get(activation)

            self.att_kernels = []
            self.bias_kernels = []
            for _ in range(self.num_heads):
                self.att_kernels.append(self.add_weight(
                    shape=(self.units * 2, 1),
                    initializer="glorot_uniform",
                    name=f"att_kernel_{_}"
                ))
                self.bias_kernels.append(self.add_weight(
                    shape=(self.units,),
                    initializer="zeros",
                    name=f"bias_kernel_{_}"
                ))

        def build(self, input_shape):
            # input_shape[0]은 노드 특징 (N, F), input_shape[1]은 인접 행렬 (N, N)
            input_dim = input_shape[0][-1]
            self.kernel = self.add_weight(
                shape=(input_dim, self.units * self.num_heads),
                initializer="glorot_uniform",
                name="kernel"
            )
            super().build(input_shape)

        def call(self, inputs):
            features, adj_matrix = inputs # (N, F), (N, N)
            num_nodes = tf.shape(features)[0]

            # 특징 변환 (F' = F * W)
            features_transformed = tf.matmul(features, self.kernel)
            
            outputs = []
            for i in range(self.num_heads):
                head_features = features_transformed[:, i*self.units:(i+1)*self.units]

                # 어텐션 계수 계산
                # a(Wh_i, Wh_j) = LeakyReLU(W_a * [Wh_i || Wh_j])
                a_input = tf.concat([
                    tf.repeat(tf.expand_dims(head_features, axis=1), num_nodes, axis=1),
                    tf.repeat(tf.expand_dims(head_features, axis=0), num_nodes, axis=0)
                ], axis=-1) # (N, N, 2*units)
                
                e = tf.matmul(a_input, self.att_kernels[i]) # (N, N, 1)
                e = tf.squeeze(e, axis=-1) # (N, N)
                e = tf.nn.leaky_relu(e, alpha=0.2)

                # 마스킹 (연결되지 않은 노드는 어텐션 0)
                zero_mask = -1e9 * (1.0 - adj_matrix) # 연결 없는 곳은 -inf
                e = e + zero_mask

                # 소프트맥스 적용하여 어텐션 가중치 얻기
                attention = tf.nn.softmax(e, axis=-1) # (N, N)

                # 어텐션 가중치와 특징 곱하여 집계
                head_output = tf.matmul(attention, head_features) # (N, units)
                outputs.append(head_output)
            
            # 여러 헤드 결과 연결
            output = tf.concat(outputs, axis=-1)

            if self.activation is not None:
                output = self.activation(output)
            
            return output

    # GAT Layer 사용 예시 (더미 데이터)
    num_nodes = 10
    feature_dim = 16
    output_units = 32
    num_heads = 2

    # 더미 노드 특징 (10개 노드, 각 16차원 특징)
    node_features = tf.random.normal((num_nodes, feature_dim))
    # 더미 인접 행렬 (10x10, 희소 그래프)
    adj_matrix = tf.cast(tf.random.uniform((num_nodes, num_nodes), maxval=2, dtype=tf.int32), tf.float32)
    adj_matrix = tf.cast((adj_matrix + tf.transpose(adj_matrix)) > 0, tf.float32) # 대칭 행렬

    gat_layer = GraphAttention(output_units, num_heads, activation='relu')
    output_features = gat_layer([node_features, adj_matrix])
    print(f"Input features shape: {node_features.shape}")
    print(f"Output features shape: {output_features.shape}") # (10, 32 * 2)
    ```

### 9.4. 추천 시스템 (Recommendation Systems)

Keras는 추천 시스템 구축을 위한 다양한 모델 아키텍처를 구현하는 데 사용될 수 있습니다.

#### 9.4.1. 협업 필터링 (Collaborative Filtering)

*   **개념:** 사용자-아이템 상호작용 데이터(예: 평점, 구매 기록)를 기반으로 유사한 사용자 또는 아이템을 찾아 추천하는 기법입니다.
*   **실무 관점:**
    *   **행렬 분해 (Matrix Factorization):** 사용자 및 아이템을 저차원 잠재 공간으로 임베딩하여 평점을 예측합니다. Keras에서는 `Embedding` 레이어를 사용하여 사용자 및 아이템 임베딩을 학습하고, 내적(dot product)을 통해 평점을 예측하는 모델을 구축할 수 있습니다.
    *   **딥러닝 기반 협업 필터링:** 사용자 및 아이템 임베딩을 입력으로 받아 MLP(Multi-Layer Perceptron)를 통해 상호작용을 모델링합니다.
    *   **활용 사례:** 영화 추천, 상품 추천, 음악 추천 등.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_items, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1, embeddings_initializer="zeros")
        self.item_embedding = layers.Embedding(
            num_items,
            embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.item_bias = layers.Embedding(num_items, 1, embeddings_initializer="zeros")

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])

        dot_product_users_items = tf.tensordot(user_vector, item_vector, 2)

        # Add biases
        x = dot_product_users_items + user_bias + item_bias
        
        # Sigmoid activation to scale ratings between 0 and 1 (or 1 and 5 if scaled)
        return tf.nn.sigmoid(x)

# --- Dummy Data Example ---
# Assume we have 100 users and 50 items
num_users = 100
num_items = 50
embedding_dim = 50 # Latent factors

# Generate dummy data: user_id, item_id, rating
# In a real scenario, this would come from a dataset like MovieLens
dummy_ratings = []
for user_id in range(num_users):
    for _ in range(np.random.randint(5, 20)): # Each user rates 5 to 20 items
        item_id = np.random.randint(0, num_items)
        rating = np.random.rand() * 4 + 1 # Ratings between 1 and 5
        dummy_ratings.append([user_id, item_id, rating])

dummy_ratings = np.array(dummy_ratings)

user_ids = dummy_ratings[:, 0]
item_ids = dummy_ratings[:, 1]
ratings = dummy_ratings[:, 2]

# Normalize ratings to be between 0 and 1 for sigmoid output
min_rating = np.min(ratings)
max_rating = np.max(ratings)
ratings = (ratings - min_rating) / (max_rating - min_rating)

# Prepare data for the model
x = np.hstack([user_ids[:, None], item_ids[:, None]])
y = ratings

# Build and compile the model
model = RecommenderNet(num_users, num_items, embedding_dim)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train the model (using dummy data)
print("\n--- Training Collaborative Filtering Model ---")
history = model.fit(x, y, batch_size=64, epochs=1, verbose=1) # epochs can be increased

# --- Make a prediction example ---
print("\n--- Making a prediction ---")
# Predict rating for user 10, item 5
user_id_to_predict = 10
item_id_to_predict = 5

predicted_rating_normalized = model.predict(np.array([[user_id_to_predict, item_id_to_predict]]))[0][0]
predicted_rating_original_scale = predicted_rating_normalized * (max_rating - min_rating) + min_rating

print(f"Predicted rating for user {user_id_to_predict} and item {item_id_to_predict}: {predicted_rating_original_scale:.2f}")
```

#### 9.4.2. 콘텐츠 기반 필터링 (Content-based Filtering)

*   **개념:** 아이템의 속성(예: 영화의 장르, 배우)이나 사용자의 프로필(예: 나이, 성별)을 기반으로 추천하는 기법입니다.
*   **실무 관점:**
    *   **Keras 구현:** 아이템 속성이나 사용자 프로필을 입력으로 받아 `Embedding` 레이어 또는 `Dense` 레이어를 통해 특징을 추출하고, 이를 결합하여 추천 모델을 구축합니다.
    *   **활용 사례:** 뉴스 기사 추천(사용자가 읽은 기사의 키워드 기반), 상품 추천(상품의 카테고리, 설명 기반).

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# Dummy Data: User features and Item features
# User features could be age, gender, demographics, etc.
# Item features could be genre, director, actors for movies; category, brand for products.

num_users = 100
num_items = 50
user_feature_dim = 5 # e.g., age, gender, income_level, etc.
item_feature_dim = 10 # e.g., genre_one_hot, director_embedding, etc.

# Generate dummy user and item features
user_features = np.random.rand(num_users, user_feature_dim).astype(np.float32)
item_features = np.random.rand(num_items, item_feature_dim).astype(np.float32)

# Generate dummy interaction data (user_id, item_id, rating)
# In a real system, this would be explicit ratings or implicit interactions
dummy_interactions = []
for user_id in range(num_users):
    for _ in range(np.random.randint(5, 20)): # Each user interacts with 5 to 20 items
        item_id = np.random.randint(0, num_items)
        rating = np.random.rand() * 4 + 1 # Ratings between 1 and 5
        dummy_interactions.append([user_id, item_id, rating])

dummy_interactions = np.array(dummy_interactions)

user_ids_interaction = dummy_interactions[:, 0].astype(np.int32)
item_ids_interaction = dummy_interactions[:, 1].astype(np.int32)
ratings_interaction = dummy_interactions[:, 2].astype(np.float32)

# Normalize ratings to be between 0 and 1 for sigmoid output
min_rating = np.min(ratings_interaction)
max_rating = np.max(ratings_interaction)
ratings_interaction_normalized = (ratings_interaction - min_rating) / (max_rating - min_rating)

# --- Content-based Filtering Model ---
# Model takes user_id and item_id as input, then looks up their features
# and combines them to predict a rating.

class ContentBasedRecommender(keras.Model):
    def __init__(self, user_features_data, item_features_data, **kwargs):
        super().__init__(**kwargs)
        self.user_features_data = tf.constant(user_features_data, dtype=tf.float32)
        self.item_features_data = tf.constant(item_features_data, dtype=tf.float32)

        # User tower
        self.user_dense1 = layers.Dense(32, activation="relu")
        self.user_dense2 = layers.Dense(16, activation="relu")

        # Item tower
        self.item_dense1 = layers.Dense(32, activation="relu")
        self.item_dense2 = layers.Dense(16, activation="relu")

        # Combination and prediction
        self.concat = layers.concatenate
        self.output_dense = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        user_id_input = inputs[:, 0] # User ID
        item_id_input = inputs[:, 1] # Item ID

        # Look up features based on IDs
        user_feat = tf.gather(self.user_features_data, user_id_input)
        item_feat = tf.gather(self.item_features_data, item_id_input)

        # Process user features
        user_processed = self.user_dense1(user_feat)
        user_processed = self.user_dense2(user_processed)

        # Process item features
        item_processed = self.item_dense1(item_feat)
        item_processed = self.item_dense2(item_processed)

        # Concatenate processed features and predict
        combined = self.concat([user_processed, item_processed])
        output = self.output_dense(combined)
        return output

# Build and compile the model
content_model = ContentBasedRecommender(user_features, item_features)
content_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Prepare input for training: user_id and item_id pairs
x_train_content = np.hstack([user_ids_interaction[:, None], item_ids_interaction[:, None]])
y_train_content = ratings_interaction_normalized

# Train the model
print("\n--- Training Content-based Filtering Model ---")
history_content = content_model.fit(x_train_content, y_train_content, batch_size=64, epochs=1, verbose=1) # epochs can be increased

# --- Make a prediction example ---
print("\n--- Making a prediction ---")
user_id_to_predict_content = 5
item_id_to_predict_content = 25

predicted_rating_normalized_content = content_model.predict(np.array([[user_id_to_predict_content, item_id_to_predict_content]]))[0][0]
predicted_rating_original_scale_content = predicted_rating_normalized_content * (max_rating - min_rating) + min_rating

print(f"Predicted rating for user {user_id_to_predict_content} and item {item_id_to_predict_content} (content-based): {predicted_rating_original_scale_content:.2f}")
```

#### 9.4.3. 하이브리드 추천 시스템

*   **개념:** 협업 필터링과 콘텐츠 기반 필터링의 장점을 결합하여 추천 성능을 향상시키는 기법입니다.
*   **실무 관점:**
    *   **Keras 구현:** Functional API를 사용하여 협업 필터링 브랜치와 콘텐츠 기반 필터링 브랜치를 각각 구축하고, 이들의 출력을 `concatenate`하거나 `add`하여 최종 예측을 수행하는 모델을 만들 수 있습니다.
    *   **활용 사례:** 대부분의 실제 추천 시스템은 하이브리드 방식을 사용합니다.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Simulate data for a hybrid recommendation system.
# We'll combine collaborative filtering (user-item interactions) and content-based features.

num_users = 100
num_items = 50
embedding_dim_cf = 32 # Latent factors for collaborative filtering

user_feature_dim = 5 # e.g., age, gender, income_level, etc.
item_content_feature_dim = 10 # e.g., genre_one_hot, director_embedding, etc.

# Generate dummy user and item content features
user_content_features = np.random.rand(num_users, user_feature_dim).astype(np.float32)
item_content_features = np.random.rand(num_items, item_content_feature_dim).astype(np.float32)

# Generate dummy interaction data (user_id, item_id, rating)
# Ratings are between 1 and 5
dummy_interactions = []
for user_id in range(num_users):
    for _ in range(np.random.randint(5, 20)): # Each user rates 5 to 20 items
        item_id = np.random.randint(0, num_items)
        rating = np.random.rand() * 4 + 1 # Ratings between 1 and 5
        dummy_interactions.append([user_id, item_id, rating])

dummy_interactions = np.array(dummy_interactions)

user_ids_interaction = dummy_interactions[:, 0].astype(np.int32)
item_ids_interaction = dummy_interactions[:, 1].astype(np.int32)
ratings_interaction = dummy_interactions[:, 2].astype(np.float32)

# Normalize ratings to be between 0 and 1 for sigmoid output
min_rating = np.min(ratings_interaction)
max_rating = np.max(ratings_interaction)
ratings_interaction_normalized = (ratings_interaction - min_rating) / (max_rating - min_rating)

# Prepare data for the model
# Input will be (user_id, item_id)
x_train_hybrid = np.hstack([user_ids_interaction[:, None], item_ids_interaction[:, None]])
y_train_hybrid = ratings_interaction_normalized

# --- Hybrid Recommendation System Model (Functional API) ---
# This model combines two branches:
# 1. Collaborative Filtering (Matrix Factorization-like) branch
# 2. Content-Based Filtering branch (using pre-defined content features)

class HybridRecommender(keras.Model):
    def __init__(self, num_users, num_items, embedding_dim_cf, 
                 user_content_features_data, item_content_features_data, **kwargs):
        super().__init__(**kwargs)
        self.user_content_features_data = tf.constant(user_content_features_data, dtype=tf.float32)
        self.item_content_features_data = tf.constant(item_content_features_data, dtype=tf.float32)

        # Collaborative Filtering Branch
        self.user_embedding_cf = layers.Embedding(
            num_users,
            embedding_dim_cf,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="user_embedding_cf"
        )
        self.item_embedding_cf = layers.Embedding(
            num_items,
            embedding_dim_cf,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="item_embedding_cf"
        )
        self.user_bias_cf = layers.Embedding(num_users, 1, embeddings_initializer="zeros", name="user_bias_cf")
        self.item_bias_cf = layers.Embedding(num_items, 1, embeddings_initializer="zeros", name="item_bias_cf")

        # Content-Based Filtering Branch
        self.user_content_dense = layers.Dense(32, activation="relu", name="user_content_dense")
        self.item_content_dense = layers.Dense(32, activation="relu", name="item_content_dense")

        # Fusion and Final Prediction
        self.fusion_dense1 = layers.Dense(64, activation="relu", name="fusion_dense1")
        self.output_layer = layers.Dense(1, activation="sigmoid", name="output_layer")

    def call(self, inputs):
        user_id = inputs[:, 0]
        item_id = inputs[:, 1]

        # Collaborative Filtering Path
        user_vector_cf = self.user_embedding_cf(user_id)
        item_vector_cf = self.item_embedding_cf(item_id)
        user_bias_cf = self.user_bias_cf(user_id)
        item_bias_cf = self.item_bias_cf(item_id)
        
        dot_product_cf = tf.reduce_sum(user_vector_cf * item_vector_cf, axis=1, keepdims=True)
        cf_output = dot_product_cf + user_bias_cf + item_bias_cf

        # Content-Based Path
        user_content_feat = tf.gather(self.user_content_features_data, user_id)
        item_content_feat = tf.gather(self.item_content_features_data, item_id)

        user_content_processed = self.user_content_dense(user_content_feat)
        item_content_processed = self.item_content_dense(item_content_feat)

        content_output = layers.concatenate([user_content_processed, item_content_processed])
        content_output = layers.Dense(32, activation="relu")(content_output) # Additional layer for content

        # Fusion
        fused_features = layers.concatenate([cf_output, content_output])
        x = self.fusion_dense1(fused_features)
        output = self.output_layer(x)
        
        return output

# Build and compile the model
hybrid_model = HybridRecommender(
    num_users=num_users,
    num_items=num_items,
    embedding_dim_cf=embedding_dim_cf,
    user_content_features_data=user_content_features,
    item_content_features_data=item_content_features
)

hybrid_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

hybrid_model.summary()

# --- Training Example ---
print("\n--- Training Hybrid Recommendation System Model ---")
history_hybrid = hybrid_model.fit(
    x_train_hybrid,
    y_train_hybrid,
    batch_size=64,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making a Hybrid Recommendation Prediction ---")
user_id_to_predict = 15
item_id_to_predict = 30

predicted_rating_normalized = hybrid_model.predict(np.array([[user_id_to_predict, item_id_to_predict]]))[0][0]
predicted_rating_original_scale = predicted_rating_normalized * (max_rating - min_rating) + min_rating

print(f"Predicted rating for user {user_id_to_predict} and item {item_id_to_predict} (hybrid): {predicted_rating_original_scale:.2f}")
```

### 9.5. 멀티모달 모델 (Multimodal Models)

멀티모달 모델은 여러 종류의 데이터(예: 이미지와 텍스트, 비디오와 오디오)를 함께 처리하여 더 풍부한 정보를 학습하고 복잡한 태스크를 해결합니다.

#### 9.5.1. 이미지-텍스트 융합 (Image-Text Fusion)

*   **개념:** 이미지와 텍스트 데이터를 동시에 입력으로 받아 처리하는 모델입니다 (예: 이미지 캡셔닝, VQA(Visual Question Answering)).
*   **실무 관점:**
    *   **Keras 구현:** Functional API를 사용하여 이미지 처리 브랜치(CNN)와 텍스트 처리 브랜치(RNN/Transformer)를 각각 구축하고, 두 브랜치의 출력을 `concatenate`하거나 `add`하여 융합합니다. 융합된 특징은 최종 태스크(분류, 생성)를 위한 레이어에 연결됩니다.
    *   **활용 사례:** 이미지 검색(텍스트 쿼리로 이미지 검색), 이미지-텍스트 매칭, 이미지 설명 생성.
```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Assume image data is 64x64 RGB images
img_height = 64
img_width = 64
img_channels = 3

# Assume text data is sequences of integers (word IDs)
max_text_tokens = 10000 # Vocabulary size
max_sequence_length = 20 # Max words in a caption

num_samples = 100
num_classes = 5 # Example: 5 categories for image-text pairs

dummy_images = np.random.rand(num_samples, img_height, img_width, img_channels).astype(np.float32)
dummy_texts = np.random.randint(0, max_text_tokens, size=(num_samples, max_sequence_length)).astype(np.int32)
dummy_labels = np.random.randint(0, num_classes, size=(num_samples,)).astype(np.int32)

# --- Image-Text Fusion Model ---
# This model will have two branches: one for image processing (CNN) and one for text processing (Embedding + LSTM).
# Their outputs will be concatenated and fed into a final classifier.

def build_image_text_fusion_model(img_shape, text_vocab_size, text_sequence_length, embedding_dim, num_classes):
    # Image Branch (CNN)
    image_input = keras.Input(shape=img_shape, name="image_input")
    x_img = layers.Conv2D(32, 3, activation="relu", padding="same")(image_input)
    x_img = layers.MaxPooling2D(2)(x_img)
    x_img = layers.Conv2D(64, 3, activation="relu", padding="same")(x_img)
    x_img = layers.MaxPooling2D(2)(x_img)
    x_img = layers.Flatten()(x_img)
    x_img = layers.Dense(128, activation="relu")(x_img)

    # Text Branch (Embedding + LSTM)
    text_input = keras.Input(shape=(text_sequence_length,), name="text_input")
    x_txt = layers.Embedding(text_vocab_size, embedding_dim)(text_input)
    x_txt = layers.LSTM(128)(x_txt)
    x_txt = layers.Dense(128, activation="relu")(x_txt)

    # Fusion Layer
    # Concatenate the outputs of both branches
    fused_features = layers.concatenate([x_img, x_txt], name="fused_features")

    # Classifier
    output = layers.Dense(num_classes, activation="softmax", name="output")(fused_features)

    model = keras.Model(inputs=[image_input, text_input], outputs=output, name="Image_Text_Fusion_Model")
    return model

# --- Model Instantiation and Compilation ---
embedding_dim = 64

fusion_model = build_image_text_fusion_model(
    img_shape=(img_height, img_width, img_channels),
    text_vocab_size=max_text_tokens,
    text_sequence_length=max_sequence_length,
    embedding_dim=embedding_dim,
    num_classes=num_classes
)

fusion_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fusion_model.summary()

# --- Training Example ---
print("\n--- Training Image-Text Fusion Model ---")
history_fusion = fusion_model.fit(
    {"image_input": dummy_images, "text_input": dummy_texts},
    dummy_labels,
    batch_size=32,
    epochs=1, # Increase epochs for better training
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making a prediction ---")
# Create dummy new data for prediction
new_image = np.random.rand(1, img_height, img_width, img_channels).astype(np.float32)
new_text = np.random.randint(0, max_text_tokens, size=(1, max_sequence_length)).astype(np.int32)

predictions = fusion_model.predict({"image_input": new_image, "text_input": new_text})
predicted_class = np.argmax(predictions[0])

print(f"Predicted class for new image-text pair: {predicted_class}")
```

#### 9.5.2. 비디오-텍스트 이해 (Video-Text Understanding)

*   **개념:** 비디오 데이터(프레임 시퀀스)와 텍스트 데이터를 함께 처리하여 비디오의 내용을 이해하거나 질문에 답변하는 모델입니다.
*   **실무 관점:**
    *   **Keras 구현:** 비디오 프레임 시퀀스를 처리하는 3D CNN 또는 RNN/Transformer 기반 브랜치와 텍스트를 처리하는 브랜치를 구축하고, 이들을 융합합니다.
    *   **활용 사례:** 비디오 요약, 비디오 질의응답, 비디오 검색.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Simulate video frames and corresponding text descriptions.
# In a real scenario, this would involve actual video and text datasets.

# Video parameters
NUM_VIDEOS = 10
MAX_FRAMES_PER_VIDEO = 10
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
CHANNELS = 3

# Text parameters
VOCAB_SIZE = 1000
MAX_TEXT_LENGTH = 15
EMBEDDING_DIM = 64

# Generate dummy video frames (sequences of images)
dummy_video_frames = np.random.rand(NUM_VIDEOS, MAX_FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS).astype(np.float32)

# Generate dummy text descriptions
dummy_texts_raw = [
    "a person is running",
    "a car is driving fast",
    "birds are flying high",
    "a dog is playing fetch",
    "people are talking",
    "a ball is bouncing",
    "water is flowing",
    "trees are swaying",
    "a cat is jumping",
    "children are laughing"
]

# Simple TextVectorization for dummy texts
text_vectorization = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_TEXT_LENGTH,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    ragged=False,
)
txt_vectorization.adapt(dummy_texts_raw)

# Convert raw texts to integer sequences
dummy_texts_vectorized = text_vectorization(tf.constant(dummy_texts_raw)).numpy()

# Dummy labels for a conceptual task (e.g., video classification based on content)
NUM_CLASSES = 2 # e.g., 'action', 'no_action'
dummy_labels = np.random.randint(0, NUM_CLASSES, size=(NUM_VIDEOS,)).astype(np.int32)

# Create tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices(
    ((dummy_video_frames, dummy_texts_vectorized), dummy_labels)
)
BATCH_SIZE = 2
dataset = dataset.shuffle(NUM_VIDEOS).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Dummy video frames shape: {dummy_video_frames.shape}")
print(f"Dummy texts vectorized shape: {dummy_texts_vectorized.shape}")

# --- Video-Text Understanding Model (Functional API) ---
# This model will have two branches: one for video processing (3D CNN) and one for text processing (Embedding + LSTM).
# Their outputs will be concatenated and fed into a final classifier.

def build_video_text_model(video_shape, text_vocab_size, text_sequence_length, embedding_dim, num_classes):
    # Video Branch (3D CNN)
    video_input = keras.Input(shape=video_shape, name="video_input")
    x_vid = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(video_input)
    x_vid = layers.MaxPooling3D((1, 2, 2))(x_vid) # Pool spatially, keep temporal
    x_vid = layers.Conv3D(64, (3, 3, 3), activation="relu", padding="same")(x_vid)
    x_vid = layers.MaxPooling3D((1, 2, 2))(x_vid)
    x_vid = layers.TimeDistributed(layers.Flatten())(x_vid) # Flatten each frame's features
    x_vid = layers.LSTM(128)(x_vid) # Process sequence of flattened frames
    x_vid = layers.Dense(128, activation="relu")(x_vid)

    # Text Branch (Embedding + LSTM)
    text_input = keras.Input(shape=(text_sequence_length,), name="text_input")
    x_txt = layers.Embedding(text_vocab_size, embedding_dim)(text_input)
    x_txt = layers.LSTM(128)(x_txt)
    x_txt = layers.Dense(128, activation="relu")(x_txt)

    # Fusion Layer
    # Concatenate the outputs of both branches
    fused_features = layers.concatenate([x_vid, x_txt], name="fused_features")

    # Classifier
    output = layers.Dense(num_classes, activation="softmax", name="output")(fused_features)

    model = keras.Model(inputs=[video_input, text_input], outputs=output, name="Video_Text_Understanding_Model")
    return model

# --- Model Instantiation and Compilation ---
video_text_model = build_video_text_model(
    video_shape=(MAX_FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS),
    text_vocab_size=VOCAB_SIZE,
    text_sequence_length=MAX_TEXT_LENGTH,
    embedding_dim=EMBEDDING_DIM,
    num_classes=NUM_CLASSES
)

video_text_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

video_text_model.summary()

# --- Training Example ---
print("\n--- Training Video-Text Understanding Model ---")
history_video_text = video_text_model.fit(
    dataset,
    epochs=1, # Increase epochs for better training
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making a Prediction ---")
# Create dummy new data for prediction
new_video = np.random.rand(1, MAX_FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS).astype(np.float32)
new_text = text_vectorization(tf.constant(["a dog is running"])) # Example text

predictions = video_text_model.predict({"video_input": new_video, "text_input": new_text})
predicted_class = np.argmax(predictions[0])

print(f"Predicted class for new video-text pair: {predicted_class}")
```

### 9.6. 기타 고급 모델 및 기법

#### 9.6.1. Autoencoders 및 Variational Autoencoders (VAEs)

*   **개념:**
    *   **Autoencoder:** 입력 데이터를 저차원 잠재 공간으로 압축(인코딩)한 후 다시 원본 데이터로 복원(디코딩)하도록 학습하는 비지도 학습 모델입니다. 데이터 압축, 특징 학습, 노이즈 제거 등에 사용됩니다.
    *   **VAE:** 오토인코더의 변형으로, 잠재 공간이 특정 확률 분포(예: 정규 분포)를 따르도록 학습하여 새로운 데이터를 생성할 수 있게 합니다.
*   **실무 관점:**
    *   **Keras 구현:** `Sequential` 또는 `Functional API`를 사용하여 인코더와 디코더 네트워크를 구축합니다. VAE의 경우, 잠재 공간에서 샘플링하는 레이어와 KL 발산 손실을 추가해야 합니다.
    *   **활용 사례:** 차원 축소, 이상 탐지, 데이터 생성, 특징 학습.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Using MNIST digits for demonstration.
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

print(f"MNIST digits shape: {mnist_digits.shape}")

# --- Autoencoder Model ---
# An Autoencoder consists of an encoder and a decoder.
# The encoder compresses the input into a latent-space representation,
# and the decoder reconstructs the input from this representation.

LATENT_DIM = 32 # Dimension of the latent space

# Encoder
def build_encoder(input_shape, latent_dim):
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    encoder_outputs = layers.Dense(latent_dim, activation="relu")(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs, name="encoder")
    return encoder

# Decoder
def build_decoder(latent_dim, output_shape):
    decoder_inputs = keras.Input(shape=(latent_dim,))
    # Calculate initial dense layer size based on output_shape and conv strides
    # Assuming 2 Conv2DTranspose layers with stride 2, so output_shape is 4x larger than initial reshape
    initial_dense_units = (output_shape[0] // 4) * (output_shape[1] // 4) * 64 # 64 is the last conv filter size
    x = layers.Dense(initial_dense_units, activation="relu")(decoder_inputs)
    x = layers.Reshape((output_shape[0] // 4, output_shape[1] // 4, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(output_shape[-1], 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
    return decoder

# Autoencoder (Encoder + Decoder)
class Autoencoder(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        return reconstructed

# --- Model Instantiation and Training Example ---
input_shape = (28, 28, 1)

encoder = build_encoder(input_shape, LATENT_DIM)
decoder = build_decoder(LATENT_DIM, input_shape)

autoencoder = Autoencoder(encoder, decoder)
autoencoder.compile(optimizer=keras.optimizers.Adam(), loss='mse') # Mean Squared Error for reconstruction

autoencoder.summary()

# Train Autoencoder
print("\n--- Training Autoencoder ---")
autoencoder.fit(mnist_digits, mnist_digits, # Input and target are the same
                epochs=1, # Increase epochs for better results
                batch_size=128,
                shuffle=True,
                validation_split=0.2)

# --- Reconstruction Example ---
print("\n--- Reconstructing Images ---")
num_reconstruction_samples = 10
sample_images = mnist_digits[:num_reconstruction_samples]
reconstructed_images = autoencoder.predict(sample_images)

print(f"Sample images shape: {sample_images.shape}")
print(f"Reconstructed images shape: {reconstructed_images.shape}")

# You can visualize original and reconstructed images
# import matplotlib.pyplot as plt
# plt.figure(figsize=(20, 4))
# for i in range(num_reconstruction_samples):
#     # Original
#     ax = plt.subplot(2, num_reconstruction_samples, i + 1)
#     plt.imshow(sample_images[i].squeeze(), cmap="gray")
#     plt.title("Original")
#     plt.axis("off")

#     # Reconstruction
#     ax = plt.subplot(2, num_reconstruction_samples, i + 1 + num_reconstruction_samples)
#     plt.imshow(reconstructed_images[i].squeeze(), cmap="gray")
#     plt.title("Reconstructed")
#     plt.axis("off")
# plt.show()
```

#### 9.6.2. 강화 학습 (Reinforcement Learning) 개요 및 Keras 활용

*   **개념:** 에이전트가 환경과 상호작용하며 보상을 최대화하는 방향으로 행동을 학습하는 머신러닝 패러다임입니다.
*   **Keras 활용:** Keras는 강화 학습 알고리즘의 핵심 구성 요소인 정책(Policy) 네트워크나 가치(Value) 네트워크를 구축하는 데 사용될 수 있습니다. `keras-rl`과 같은 라이브러리는 Keras를 기반으로 다양한 강화 학습 알고리즘을 구현합니다.
*   **실무 관점:**
    *   **Custom Training Loop:** 강화 학습은 `model.fit()`과 같은 표준 학습 루프와는 다른 학습 방식을 요구하므로, Custom Training Loop를 구현해야 합니다.
    *   **활용 사례:** 게임 플레이, 로봇 제어, 자율 주행, 자원 관리 등.

    **예시 (간단한 정책 네트워크):**
    ```python
    import keras
    from keras import layers
    import tensorflow as tf
    import numpy as np

    # 더미 환경 (간단한 이산 행동 공간)
    class SimpleEnvironment:
        def __init__(self):
            self.state = 0
            self.max_steps = 10
            self.current_step = 0

        def reset(self):
            self.state = 0
            self.current_step = 0
            return self.state

        def step(self, action):
            if action == 0: # 왼쪽
                self.state = max(0, self.state - 1)
            else: # 오른쪽
                self.state = min(4, self.state + 1)
            
            self.current_step += 1
            done = self.current_step >= self.max_steps
            reward = 1 if self.state == 4 else 0 # 목표 상태에 도달하면 보상
            return self.state, reward, done

    # 정책 네트워크 (Policy Network) 정의
    class PolicyNetwork(keras.Model):
        def __init__(self, num_actions):
            super().__init__()
            self.dense1 = layers.Dense(32, activation="relu")
            self.dense2 = layers.Dense(num_actions, activation="softmax")

        def call(self, inputs):
            x = self.dense1(inputs)
            return self.dense2(x)

    # 강화 학습 Custom Training Loop (개념적)
    policy_net = PolicyNetwork(num_actions=2) # 0: 왼쪽, 1: 오른쪽
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    env = SimpleEnvironment()

    num_episodes = 100
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        episode_log_probs = []

        while not done:
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            with tf.GradientTape() as tape:
                action_probs = policy_net(state_tensor)
                action = tf.random.categorical(tf.math.log(action_probs), 1)[0, 0].numpy()
                log_prob = tf.math.log(action_probs[0, action])

            next_state, reward, done = env.step(action)
            
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            state = next_state

        # 에피소드 종료 후 정책 업데이트 (REINFORCE 알고리즘 개념)
        # 감가율 적용된 보상 계산
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + 0.99 * G # 감가율 0.99
            returns.insert(0, G)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8) # 정규화

        # 손실 계산 및 기울기 적용
        policy_loss = -tf.reduce_sum(tf.stack(episode_log_probs) * returns)
        grads = tape.gradient(policy_loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

        print(f"Episode {episode+1}, Total Reward: {sum(episode_rewards)}")
    ```

#### 9.6.3. 자기지도 학습 (Self-supervised Learning)

*   **개념:** 레이블이 없는 대규모 데이터셋에서 데이터 자체의 구조나 관계를 이용하여 학습 목표(pretext task)를 설정하고, 이를 통해 유용한 특징 표현(representation)을 학습하는 비지도 학습의 한 형태입니다.
*   **실무 관점:**
    *   **활용:** 레이블된 데이터가 부족한 상황에서 사전 학습(pre-training) 단계에 사용되어, 다운스트림 태스크의 성능을 향상시킵니다.
    *   **예시:** 이미지의 회전 각도 예측, 이미지 패치 순서 예측, 마스킹된 단어 예측(BERT의 MLM).
    *   **Keras 구현:** Custom Training Loop를 사용하여 pretext task를 위한 모델을 구축하고 학습합니다.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Assume we have a dataset of unlabeled images (e.g., CIFAR-10 without labels)
(x_train_cifar, _), (x_test_cifar, _) = tf.keras.datasets.cifar10.load_data()
# Combine for a larger unlabeled dataset
unlabeled_images = np.concatenate([x_train_cifar, x_test_cifar], axis=0).astype(np.float32) / 255.0

# --- Self-supervised Pretext Task: Rotation Prediction ---
# The model will be trained to predict the rotation angle (0, 90, 180, 270 degrees)
# applied to an input image. This forces the model to learn meaningful visual features.

def rotate_image(image, angle_idx):
    """Rotates an image by 0, 90, 180, or 270 degrees."""
    if angle_idx == 0:
        return image
    elif angle_idx == 1: # 90 degrees
        return tf.image.rot90(image, k=1)
    elif angle_idx == 2: # 180 degrees
        return tf.image.rot90(image, k=2)
    elif angle_idx == 3: # 270 degrees
        return tf.image.rot90(image, k=3)

def create_rotated_dataset(images, num_rotation_angles=4):
    rotated_images = []
    rotation_labels = []
    for img in images:
        for angle_idx in range(num_rotation_angles):
            rotated_images.append(rotate_image(img, angle_idx))
            rotation_labels.append(angle_idx)
    return np.array(rotated_images), np.array(rotation_labels)

# Create the self-supervised dataset
print("\n--- Creating Self-supervised Dataset (Rotation Prediction) ---")
x_pretext, y_pretext = create_rotated_dataset(unlabeled_images)

# Convert to tf.data.Dataset for efficient loading
pretext_dataset = tf.data.Dataset.from_tensor_slices((x_pretext, y_pretext))
pretext_dataset = pretext_dataset.shuffle(buffer_size=10000).batch(64).prefetch(tf.data.AUTOTUNE)

# --- Self-supervised Model Architecture ---
# The encoder learns features, and a small classification head predicts the rotation.
# The encoder's weights will then be used for a downstream task.

def build_self_supervised_model(input_shape, num_rotation_angles):
    inputs = keras.Input(shape=input_shape, name="image_input")

    # Encoder (Feature Extractor)
    # This part will be reused for downstream tasks
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    encoder_output = layers.Dense(256, activation="relu", name="encoder_output")(x)

    # Rotation Prediction Head
    rotation_output = layers.Dense(num_rotation_angles, activation="softmax", name="rotation_output")(encoder_output)

    model = keras.Model(inputs=inputs, outputs=rotation_output, name="Self_Supervised_Rotation_Model")
    return model

# --- Model Instantiation and Training ---
input_shape_pretext = unlabeled_images.shape[1:]
num_rotation_angles = 4

self_supervised_model = build_self_supervised_model(input_shape_pretext, num_rotation_angles)

self_supervised_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

self_supervised_model.summary()

print("\n--- Training Self-supervised Model (Rotation Prediction) ---")
history_self_supervised = self_supervised_model.fit(
    pretext_dataset,
    epochs=1, # Increase epochs for better feature learning
    verbose=1
)

# --- Downstream Task Example (using the pre-trained encoder) ---
print("\n--- Using Pre-trained Encoder for Downstream Task (e.g., Image Classification) ---")

# Extract the encoder part
encoder_for_downstream = keras.Model(inputs=self_supervised_model.input, 
                                     outputs=self_supervised_model.get_layer("encoder_output").output,
                                     name="Pretrained_Encoder")

# Build a new classifier on top of the pre-trained encoder
num_downstream_classes = 10 # e.g., CIFAR-10 classes

downstream_inputs = keras.Input(shape=input_shape_pretext)
# Freeze the encoder layers (optional, can also fine-tune)
x_downstream = encoder_for_downstream(downstream_inputs, training=False) 
output_downstream = layers.Dense(num_downstream_classes, activation="softmax")(x_downstream)

downstream_model = keras.Model(inputs=downstream_inputs, outputs=output_downstream, name="Downstream_Classifier")

downstream_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

downstream_model.summary()

# Dummy training for downstream task (e.g., with CIFAR-10 labeled data)
(x_train_labeled, y_train_labeled), (x_test_labeled, y_test_labeled) = tf.keras.datasets.cifar10.load_data()
x_train_labeled = x_train_labeled.astype(np.float32) / 255.0
y_train_labeled = y_train_labeled.astype(np.int32)
x_test_labeled = x_test_labeled.astype(np.float32) / 255.0
y_test_labeled = y_test_labeled.astype(np.int32)

print("\n--- Training Downstream Classifier ---")
downstream_model.fit(x_train_labeled, y_train_labeled, epochs=1, batch_size=64, verbose=1)

loss, accuracy = downstream_model.evaluate(x_test_labeled, y_test_labeled, verbose=0)
print(f"Downstream Task Test Accuracy: {accuracy:.4f}")
```

#### 9.6.4. 확산 모델 (Diffusion Models)

*   **개념:** 최근 이미지 생성 분야에서 GAN을 능가하는 성능을 보여주는 생성 모델입니다. 점진적으로 노이즈를 추가하는 확산 과정(forward process)과 노이즈를 제거하여 데이터를 복원하는 역확산 과정(reverse process)을 학습합니다.
*   **실무 관점:**
    *   **활용:** 고품질 이미지 생성, 이미지 편집, 조건부 이미지 생성(텍스트-이미지 변환) 등.
    *   **Keras 구현:** U-Net과 같은 아키텍처를 기반으로 노이즈 예측 네트워크를 구축하고, Custom Training Loop를 통해 학습합니다.

    **예시 (간단한 확산 모델의 노이즈 예측 네트워크):**
    ```python
    import keras
    from keras import layers
    import tensorflow as tf
    import numpy as np

    # 간단한 U-Net 형태의 노이즈 예측 네트워크
    class NoisePredictor(keras.Model):
        def __init__(self, img_size, num_channels, widths, block_depth):
            super().__init__()
            self.img_size = img_size
            self.num_channels = num_channels
            self.widths = widths
            self.block_depth = block_depth

            self.conv_in = layers.Conv2D(widths[0], kernel_size=3, padding="same", activation="relu")
            
            self.down_blocks = []
            self.up_blocks = []
            
            for i in range(len(widths) - 1):
                self.down_blocks.append(DownBlock(widths[i], widths[i+1], block_depth))
                self.up_blocks.append(UpBlock(widths[len(widths)-1-i], widths[len(widths)-2-i], block_depth))

            self.conv_out = layers.Conv2D(num_channels, kernel_size=3, padding="same")

        def call(self, inputs):
            x = self.conv_in(inputs)
            
            skips = []
            for block in self.down_blocks:
                x, skip = block(x)
                skips.append(skip)
            
            # Bottleneck (가장 깊은 부분)
            x = layers.Conv2D(self.widths[-1], kernel_size=3, padding="same", activation="relu")(x)

            for i, block in enumerate(self.up_blocks):
                x = block(x, skips[len(skips)-1-i])
            
            return self.conv_out(x)

    class DownBlock(layers.Layer):
        def __init__(self, filters_in, filters_out, block_depth):
            super().__init__()
            self.convs = []
            for _ in range(block_depth):
                self.convs.append(layers.Conv2D(filters_in, kernel_size=3, padding="same", activation="relu"))
            self.downsample = layers.Conv2D(filters_out, kernel_size=3, strides=2, padding="same", activation="relu")

        def call(self, inputs):
            x = inputs
            for conv in self.convs:
                x = conv(x)
            skip = x
            x = self.downsample(x)
            return x, skip

    class UpBlock(layers.Layer):
        def __init__(self, filters_in, filters_out, block_depth):
            super().__init__()
            self.upsample = layers.Conv2DTranspose(filters_in, kernel_size=3, strides=2, padding="same", activation="relu")
            self.convs = []
            for _ in range(block_depth):
                self.convs.append(layers.Conv2D(filters_out, kernel_size=3, padding="same", activation="relu"))

        def call(self, inputs, skip_connection):
            x = self.upsample(inputs)
            x = layers.concatenate([x, skip_connection])
            for conv in self.convs:
                x = conv(x)
            return x

    # 확산 모델 학습 루프 (개념적)
    image_size = 32
    num_channels = 3
    widths = [64, 128, 256]
    block_depth = 2

    noise_predictor = NoisePredictor(image_size, num_channels, widths, block_depth)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    mse_loss = keras.losses.MeanSquaredError()

    @tf.function
    def train_step(images):
        batch_size = tf.shape(images)[0]
        # 1. 랜덤한 시간 스텝 샘플링
        t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=1.0)
        
        # 2. 랜덤한 노이즈 샘플링
        noise = tf.random.normal(shape=tf.shape(images))
        
        # 3. 노이즈가 추가된 이미지 생성 (확산 과정)
        # 이 부분은 실제 확산 모델 구현에 따라 달라짐 (예: DDPM)
        # 여기서는 간단히 원본 이미지와 노이즈를 선형 결합하는 것으로 가정
        noisy_images = images * (1 - t) + noise * t 

        with tf.GradientTape() as tape:
            # 4. 노이즈 예측 네트워크로 노이즈 예측
            predicted_noise = noise_predictor(noisy_images)
            # 5. 손실 계산 (예측된 노이즈와 실제 노이즈 간의 MSE)
            loss = mse_loss(noise, predicted_noise)

        # 6. 기울기 계산 및 가중치 업데이트
        gradients = tape.gradient(loss, noise_predictor.trainable_variables)
        optimizer.apply_gradients(zip(gradients, noise_predictor.trainable_variables))
        return loss

    # 더미 데이터셋
    dummy_images = tf.random.normal((64, image_size, image_size, num_channels))
    dataset = tf.data.Dataset.from_tensor_slices(dummy_images).batch(16).prefetch(tf.data.AUTOTUNE)

    for epoch in range(3):
        for batch in dataset:
            loss_value = train_step(batch)
        print(f"Epoch {epoch+1}, Loss: {loss_value:.4f}")
    ```

#### 9.6.5. 신경망 구조 탐색 (Neural Architecture Search, NAS)

*   **개념:** 최적의 신경망 아키텍처를 자동으로 탐색하는 기술입니다. 수동으로 아키텍처를 설계하는 대신, 알고리즘이 주어진 태스크에 가장 적합한 모델 구조를 찾아냅니다.
*   **실무 관점:**
    *   **활용:** 특정 태스크에 대한 최적의 모델을 찾거나, 리소스 제약이 있는 환경(예: 모바일)에 맞는 경량 모델을 설계할 때 사용됩니다.
    *   **KerasTuner:** KerasTuner는 NAS의 한 형태로 볼 수 있으며, 하이퍼파라미터 탐색을 통해 모델 구조를 포함한 최적의 조합을 찾을 수 있습니다.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Conceptual NAS Example (Random Search for a simple MLP) ---
# In a real NAS scenario, you would use libraries like KerasTuner (already covered),
# AutoKeras, or more advanced algorithms like Reinforcement Learning-based NAS.
# This example conceptually shows how different architectures can be sampled and evaluated.

# Define the search space for a simple MLP
def build_nas_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(784,)))

    # Search for number of hidden layers (1 to 3)
    for i in range(hp.Int('num_layers', 1, 3)):
        # Search for number of units in each layer (32, 64, 128)
        model.add(layers.Dense(units=hp.Choice(f'units_{i}', [32, 64, 128]), activation='relu'))
        # Search for whether to add dropout
        if hp.Boolean(f'dropout_{i}'):
            model.add(layers.Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(layers.Dense(10, activation='softmax'))

    # Search for optimizer learning rate
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- KerasTuner Integration (as a practical NAS tool) ---
# KerasTuner is a powerful library for hyperparameter tuning, which can be used
# for Neural Architecture Search by defining the model architecture as part of the search space.

import keras_tuner as kt

# Dummy Data (e.g., MNIST)
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_full = x_train_full.reshape(-1, 784).astype("float32") / 255.0
y_train_full = y_train_full.astype("int32")
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
y_test = y_test.astype("int32")

# Split data for tuning
x_train, x_val = x_train_full[:-10000], x_train_full[-10000:]
y_train, y_val = y_train_full[:-10000], y_train_full[-10000:]

# Instantiate the tuner (e.g., RandomSearch)
tuner = kt.RandomSearch(
    hypermodel=build_nas_model,
    objective='val_accuracy',
    max_trials=5, # Number of different architectures to try
    executions_per_trial=1, # Number of runs per architecture
    directory='nas_demo',
    project_name='simple_mlp_nas'
)

print("\n--- Starting Conceptual NAS (via KerasTuner) ---")
# tuner.search(x_train, y_train, epochs=1, validation_data=(x_val, y_val)) # Run for more epochs in real scenario

# Get the best model found
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# best_model = tuner.get_best_models(num_models=1)[0]

# print(f"\nBest Hyperparameters: {best_hps.values}")
# print("Best Model Summary:")
# best_model.summary()

print("Conceptual NAS example setup complete. To run, uncomment tuner.search() and related lines.")
print("This demonstrates how KerasTuner can be used for NAS by defining architectural choices as hyperparameters.")
```


### 9.7. 최신 연구 및 실무 적용 논의 모델

#### 9.7.1. 대규모 언어 모델 (LLM) 응용 (Fine-tuning, Prompt Engineering)

*   **개념:** GPT-3, BERT, T5 등 방대한 텍스트 데이터로 사전 학습된 거대 신경망 모델입니다. 다양한 자연어 처리 태스크에서 뛰어난 성능을 보입니다.
*   **실무 관점:**
    *   **Fine-tuning:** 특정 다운스트림 태스크(예: 감성 분석, 텍스트 분류)에 맞게 LLM의 가중치를 추가적으로 학습시킵니다. Keras에서는 `keras.applications`와 유사하게 사전 학습된 LLM을 로드하고 그 위에 새로운 레이어를 추가하여 미세 조정할 수 있습니다 (KerasNLP 활용).
    *   **Prompt Engineering:** LLM에 질문이나 지시(프롬프트)를 적절히 구성하여 모델이 원하는 작업을 수행하도록 유도하는 기법입니다. 미세 조정 없이도 다양한 태스크를 해결할 수 있게 합니다.
    *   **활용 사례:** 챗봇, 콘텐츠 생성, 번역, 코드 생성, 정보 검색 등.

```python
import keras
import keras_nlp
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Simulate a small dataset for a text classification task (e.g., sentiment analysis)
# In a real scenario, you would load a larger, more diverse dataset.

dummy_texts = [
    "This movie was fantastic! I loved every moment.",
    "Absolutely terrible, a complete waste of time.",
    "It was okay, nothing special but not bad either.",
    "Highly recommend, great acting and story.",
    "Couldn't stand it, so boring and predictable.",
    "The plot was engaging and the characters were well-developed.",
    "A masterpiece of modern cinema, truly inspiring.",
    "I fell asleep halfway through, utterly dull.",
    "Not the worst, but definitely not the best.",
    "Brilliant direction and powerful performances."
]
dummy_labels = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1] # 1 for positive, 0 for negative/neutral

# Convert to tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((dummy_texts, dummy_labels))
BATCH_SIZE = 2
dataset = dataset.shuffle(len(dummy_texts)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- LLM Fine-tuning with KerasNLP (BERT Classifier) ---
# KerasNLP provides easy-to-use APIs for loading and fine-tuning pre-trained LLMs.

# 1. Load a pre-trained BERT Classifier from a preset
# This will download the model weights and the associated preprocessor.
# Common presets: 'bert_base_en_uncased', 'bert_large_en_uncased', etc.
# num_classes should match your downstream task (e.g., 2 for binary sentiment).

# Note: Ensure you have `tensorflow_text` installed for some tokenizers.
# pip install tensorflow_text

print("\n--- Loading Pre-trained BERT Classifier (KerasNLP) ---")
# Using a smaller preset for faster download and demonstration
classifier_model = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_uncased", # A very small BERT for quick demo
    num_classes=2, # Binary classification (positive/negative)
    preprocessor=keras_nlp.models.BertPreprocessor.from_preset(
        "bert_tiny_uncased",
        sequence_length=128 # Max sequence length for input
    )
)

# 2. Compile the model
classifier_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5), # Common learning rate for fine-tuning LLMs
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Use from_logits=True if last layer has no activation
    metrics=['accuracy']
)

classifier_model.summary()

# --- Training (Fine-tuning) Example ---
print("\n--- Fine-tuning LLM (BERT Classifier) ---")
history_llm_finetune = classifier_model.fit(
    dataset,
    epochs=1, # Increase epochs for better fine-tuning
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making LLM Predictions ---")
new_texts_to_predict = [
    "This is an absolutely brilliant film, I highly recommend it!",
    "The worst experience ever, I would not watch it again.",
    "It was neither good nor bad, just average."
]

predictions = classifier_model.predict(tf.constant(new_texts_to_predict))

# Interpret predictions (assuming 2 classes: 0=negative, 1=positive)
predicted_sentiments = tf.argmax(predictions, axis=-1).numpy()

for text, sentiment_id in zip(new_texts_to_predict, predicted_sentiments):
    sentiment_label = "Positive" if sentiment_id == 1 else "Negative/Neutral"
    print(f"Text: \"{text}\" -> Predicted Sentiment: {sentiment_label}")

# --- Conceptual Prompt Engineering (without Keras code) ---
print("\n--- Conceptual Prompt Engineering ---")
print("Prompt engineering involves crafting inputs to guide LLMs without fine-tuning.")
print("Example: Instead of training, you might prompt a large generative LLM like:")
print("\"Classify the sentiment of the following movie review as positive or negative: 'This movie was amazing!'\"")
print("Or: \"Translate 'Hello world' to French.\"")
print("KerasNLP focuses on fine-tuning, but prompt engineering is a key complementary technique.")
```

#### 9.7.2. 고급 생성형 AI (Text-to-Image, Video Generation)

*   **개념:** 텍스트 설명이나 다른 형태의 입력을 기반으로 사실적인 이미지나 비디오를 생성하는 AI 모델입니다 (예: DALL-E, Stable Diffusion, Midjourney).
*   **실무 관점:**
    *   **모델 아키텍처:** 주로 확산 모델(Diffusion Models)이나 GAN 기반 모델이 사용되며, 텍스트 인코더(예: CLIP의 텍스트 인코더)와 이미지/비디오 생성 네트워크가 결합됩니다.
    *   **활용 사례:** 예술 작품 생성, 디자인 시안 생성, 가상 현실 콘텐츠 제작, 광고 콘텐츠 생성 등.

```python
import keras
import keras_cv
import tensorflow as tf
import numpy as np

# --- Text-to-Image Generation with KerasCV Stable Diffusion ---
# KerasCV provides an implementation of Stable Diffusion, a powerful text-to-image model.
# This example shows how to generate images from text prompts.

# Note: Stable Diffusion models are large and require significant GPU memory.
# The first run will download model weights (several GBs).
# Consider running this on a GPU-enabled environment (e.g., Google Colab with GPU runtime).

# 1. Load the Stable Diffusion model
print("\n--- Loading KerasCV Stable Diffusion Model (This may take a while) ---")
# Use a smaller resolution for faster generation and less memory usage for demonstration
# The default resolution is 512x512, which is very memory intensive.
# For a quick test, you might try 256x256 or 128x128 if your GPU memory is limited.

# stable_diffusion_model = keras_cv.models.StableDiffusion(
#     img_width=256, img_height=256
# )

# For demonstration purposes, we will skip loading the full model
# as it requires significant resources and download time.
# Instead, we'll show the conceptual usage.

print("Skipping actual Stable Diffusion model loading for faster demonstration.")
print("To run the full example, uncomment the `stable_diffusion_model` instantiation.")

# --- Generate Images from Text Prompts (Conceptual Usage) ---
# Once the model is loaded, you can generate images by calling its `text_to_image` method.

# Example prompts
# prompts = [
#     "photograph of an astronaut riding a horse",
#     "A futuristic city at sunset, digital art",
#     "A cat wearing a tiny hat, watercolor painting"
# ]

# Number of images to generate per prompt
# batch_size = 1

# print("\n--- Generating Images from Prompts (Conceptual) ---")
# for prompt in prompts:
#     print(f"Generating for prompt: \"{prompt}\" ")
#     # Generated images will be a TensorFlow tensor of shape (batch_size, img_height, img_width, 3)
#     # generated_images = stable_diffusion_model.text_to_image(prompt, batch_size=batch_size)

#     # You can then save or display these images
#     # import matplotlib.pyplot as plt
#     # for i in range(batch_size):
#     #     plt.imshow(generated_images[i].numpy().astype(np.uint8))
#     #     plt.axis("off")
#     #     plt.title(prompt)
#     #     plt.show()

print("Conceptual text-to-image generation demonstrated. Actual generation requires model loading.")

# --- Video Generation (Conceptual) ---
print("\n--- Conceptual Video Generation ---")
print("Video generation is an extension of image generation, often involving recurrent or temporal models.")
print("It typically involves generating a sequence of coherent images (frames) over time.")
print("KerasCV currently focuses on image generation, but the principles extend to video.")
print("Conceptual approach: Generate a sequence of images conditioned on a text prompt and temporal information.")
print("Example: A model might generate [frame_1, frame_2, ..., frame_N] from a prompt like \"a dog running\".")
print("This often involves combining image generation techniques with video-specific architectures (e.g., 3D convolutions, recurrent layers).")
```

#### 9.7.3. 고급 강화 학습 (Multi-agent RL, Offline RL)

*   **개념:**
    *   **Multi-agent RL:** 여러 에이전트가 동시에 환경과 상호작용하며 학습하는 강화 학습입니다. 협력 또는 경쟁 환경에서 복잡한 행동을 학습합니다.
    *   **Offline RL (Batch RL):** 미리 수집된 고정된 데이터셋만을 사용하여 정책을 학습하는 강화 학습입니다. 환경과의 추가적인 상호작용 없이 학습이 이루어지므로, 실제 환경에서 상호작용 비용이 높거나 위험할 때 유용합니다.
*   **실무 관점:**
    *   **Keras 활용:** 각 에이전트의 정책 네트워크나 가치 네트워크를 Keras 모델로 구축하고, Custom Training Loop를 통해 강화 학습 알고리즘을 구현합니다.
    *   **활용 사례:** 자율 주행 차량의 협력 제어, 로봇 군집 제어, 게임 AI, 추천 시스템의 사용자 행동 최적화.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# --- Dummy Multi-Agent Environment (e.g., Cooperative Grid World) ---
# Two agents trying to reach a common goal, needing to coordinate.
class MultiAgentGridEnv:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.agent_pos = [np.array([0, 0]), np.array([grid_size-1, grid_size-1])]
        self.goal_pos = np.array([grid_size//2, grid_size//2])
        self.max_steps = 20
        self.current_step = 0

    def reset(self):
        self.agent_pos = [np.array([0, 0]), np.array([self.grid_size-1, self.grid_size-1])]
        self.current_step = 0
        return [pos.tolist() for pos in self.agent_pos] # Return list of agent positions

    def step(self, actions):
        # actions = [action_agent1, action_agent2]
        rewards = [0, 0]
        done = False

        for i, action in enumerate(actions):
            if action == 0: # Up
                self.agent_pos[i][0] = max(0, self.agent_pos[i][0] - 1)
            elif action == 1: # Down
                self.agent_pos[i][0] = min(self.grid_size - 1, self.agent_pos[i][0] + 1)
            elif action == 2: # Left
                self.agent_pos[i][1] = max(0, self.agent_pos[i][1] - 1)
            elif action == 3: # Right
                self.agent_pos[i][1] = min(self.grid_size - 1, self.agent_pos[i][1] + 1)
            # No-op for action 4 (if applicable)

        # Common reward: both agents reach the goal
        if np.array_equal(self.agent_pos[0], self.goal_pos) and np.array_equal(self.agent_pos[1], self.goal_pos):
            rewards = [10, 10] # High reward for reaching goal together
            done = True
        elif np.array_equal(self.agent_pos[0], self.goal_pos) or np.array_equal(self.agent_pos[1], self.goal_pos):
            rewards = [-1, -1] # Penalty if only one reaches (bad coordination)
        else:
            rewards = [-0.1, -0.1] # Small penalty for each step

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return [pos.tolist() for pos in self.agent_pos], rewards, done

# --- 2. Build Keras Models for Each Agent (e.g., Policy Networks) ---
# Each agent has its own policy network. For simplicity, they share the same architecture.

state_dim = 2 # (row, col) for each agent
num_actions = 4 # Up, Down, Left, Right

def build_agent_policy_network(input_dim, num_actions):
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_actions, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model

# Create two policy networks for two agents
agent1_policy = build_agent_policy_network(state_dim, num_actions)
agent2_policy = build_agent_policy_network(state_dim, num_actions)

agent1_policy.summary()
agent2_policy.summary()

# --- 3. Multi-Agent Training Loop (Conceptual - Centralized Training, Decentralized Execution) ---
# This is a simplified conceptual loop. Real multi-agent RL is complex.
# Here, we assume a centralized critic or shared experience replay.

optimizer = keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step_multi_agent(states, actions, rewards, next_states, dones):
    # This is highly simplified. In reality, you'd have a loss function
    # that considers the interaction between agents (e.g., Q-mix, MADDPG).
    # For now, we'll just do a dummy update based on individual rewards.

    # Agent 1 update
    with tf.GradientTape() as tape1:
        # Predict action probabilities for current state
        action_probs1 = agent1_policy(states[0])
        # Calculate log probability of taken action
        log_prob1 = tf.math.log(tf.gather_nd(action_probs1, tf.stack([tf.range(tf.shape(actions[0])[0]), actions[0]], axis=1)))
        # Simple policy gradient loss (rewards as advantage)
        loss1 = -tf.reduce_mean(log_prob1 * rewards[0])
    grads1 = tape1.gradient(loss1, agent1_policy.trainable_variables)
    optimizer.apply_gradients(zip(grads1, agent1_policy.trainable_variables))

    # Agent 2 update (similar)
    with tf.GradientTape() as tape2:
        action_probs2 = agent2_policy(states[1])
        log_prob2 = tf.math.log(tf.gather_nd(action_probs2, tf.stack([tf.range(tf.shape(actions[1])[0]), actions[1]], axis=1)))
        loss2 = -tf.reduce_mean(log_prob2 * rewards[1])
    grads2 = tape2.gradient(loss2, agent2_policy.trainable_variables)
    optimizer.apply_gradients(zip(grads2, agent2_policy.trainable_variables))

    return loss1, loss2

# Training loop
env = MultiAgentGridEnv()
num_episodes = 100

print("\n--- Training Multi-Agent RL (Conceptual) ---")
for episode in range(num_episodes):
    current_states = env.reset()
    done = False
    episode_rewards = [0, 0]

    # Store experiences for this episode
    episode_states = [[], []]
    episode_actions = [[], []]
    episode_rewards_list = [[], []]
    episode_next_states = [[], []]
    episode_dones = []

    while not done:
        # Agents choose actions based on their current observation
        state_tensor1 = tf.constant([current_states[0]], dtype=tf.float32)
        state_tensor2 = tf.constant([current_states[1]], dtype=tf.float32)

        action_probs1 = agent1_policy(state_tensor1)
        action1 = tf.random.categorical(tf.math.log(action_probs1), 1)[0, 0].numpy()

        action_probs2 = agent2_policy(state_tensor2)
        action2 = tf.random.categorical(tf.math.log(action_probs2), 1)[0, 0].numpy()

        actions = [action1, action2]
        next_states, rewards, done = env.step(actions)

        episode_states[0].append(current_states[0])
        episode_states[1].append(current_states[1])
        episode_actions[0].append(action1)
        episode_actions[1].append(action2)
        episode_rewards_list[0].append(rewards[0])
        episode_rewards_list[1].append(rewards[1])
        episode_next_states[0].append(next_states[0])
        episode_next_states[1].append(next_states[1])
        episode_dones.append(done)

        episode_rewards[0] += rewards[0]
        episode_rewards[1] += rewards[1]
        current_states = next_states

    # Convert lists to tensors for training step
    states_tensor = [tf.constant(episode_states[0], dtype=tf.float32), tf.constant(episode_states[1], dtype=tf.float32)]
    actions_tensor = [tf.constant(episode_actions[0], dtype=tf.int32), tf.constant(episode_actions[1], dtype=tf.int32)]
    rewards_tensor = [tf.constant(episode_rewards_list[0], dtype=tf.float32), tf.constant(episode_rewards_list[1], dtype=tf.float32)]
    next_states_tensor = [tf.constant(episode_next_states[0], dtype=tf.float32), tf.constant(episode_next_states[1], dtype=tf.float32)]
    dones_tensor = tf.constant(episode_dones, dtype=tf.bool)

    loss1, loss2 = train_step_multi_agent(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
    print(f"Episode {episode+1}: Agent 1 Reward = {episode_rewards[0]:.2f}, Agent 2 Reward = {episode_rewards[1]:.2f}")

print("Multi-Agent RL training finished.")
```

#### 9.7.4. 인과 관계 추론 모델 (Causal Inference Models)

*   **개념:** 데이터 간의 단순한 상관관계가 아닌, 원인과 결과의 인과 관계를 추론하는 모델입니다.
*   **실무 관점:**
    *   **활용:** 특정 개입(treatment)이 결과에 미치는 영향을 정량화하거나, 정책 결정의 효과를 예측하는 데 사용됩니다. (예: 특정 마케팅 캠페인이 매출에 미치는 인과적 효과).
    *   **Keras 활용:** 인과 관계 추론 모델은 종종 딥러닝 모델을 구성 요소로 포함합니다. 예를 들어, 특징을 임베딩하거나 예측 모델을 구축하는 데 Keras 모델을 사용할 수 있습니다.
    *   **도구:** `DoWhy`, `CausalML` 등 인과 관계 추론 라이브러리와 함께 Keras 모델을 활용할 수 있습니다.

    **예시 (Keras 모델을 활용한 인과 관계 추론의 구성 요소):**
    ```python
    import keras
    from keras import layers
    import tensorflow as tf
    import numpy as np

    # 가상의 데이터 생성: 치료(treatment), 공변량(covariates), 결과(outcome)
    # 목표: treatment가 outcome에 미치는 인과적 효과 추정
    np.random.seed(42)
    num_samples = 1000
    
    # 공변량 (예: 나이, 소득 등)
    covariates = np.random.normal(loc=0, scale=1, size=(num_samples, 5)).astype(np.float32)
    # 치료 (0: 대조군, 1: 치료군)
    treatment = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
    
    # 결과 (treatment와 covariates에 따라 달라짐)
    # 실제 인과 관계: outcome = 2 * treatment + 0.5 * covariates[:, 0] + noise
    outcome = (2 * treatment + 0.5 * covariates[:, 0].reshape(-1, 1) + np.random.normal(loc=0, scale=0.5, size=(num_samples, 1))).astype(np.float32)

    # Keras 모델을 사용하여 공변량의 특징 임베딩 학습
    # 또는 outcome 예측 모델의 일부로 활용
    class FeatureExtractor(keras.Model):
        def __init__(self, output_dim):
            super().__init__()
            self.dense1 = layers.Dense(64, activation="relu")
            self.dense2 = layers.Dense(output_dim, activation="relu")

        def call(self, inputs):
            x = self.dense1(inputs)
            return self.dense2(x)

    # Keras 모델을 활용한 이중 강건(Doubly Robust) 추정의 예시 (개념적)
    # 1. Outcome 모델 (E[Y|T, X]) - treatment와 covariates를 이용해 outcome 예측
    # 2. Propensity Score 모델 (P(T=1|X)) - covariates를 이용해 treatment 받을 확률 예측

    # Outcome 예측 모델 (Keras Functional API 예시)
    input_covariates = keras.Input(shape=(covariates.shape[1],), name="covariates_input")
    input_treatment = keras.Input(shape=(1,), name="treatment_input")

    # 공변량 특징 추출
    extracted_features = FeatureExtractor(output_dim=32)(input_covariates)

    # 특징과 치료를 결합하여 outcome 예측
    combined_features = layers.concatenate([extracted_features, input_treatment])
    outcome_prediction = layers.Dense(1, activation="linear", name="outcome_prediction")(combined_features)

    outcome_model = keras.Model(inputs=[input_covariates, input_treatment], outputs=outcome_prediction)
    outcome_model.compile(optimizer="adam", loss="mse")
    outcome_model.fit([covariates, treatment], outcome, epochs=10, batch_size=32)

    # Propensity Score 예측 모델 (Keras Sequential API 예시)
    propensity_model = keras.Sequential([
        keras.Input(shape=(covariates.shape[1],)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid", name="propensity_score_prediction")
    ])
    propensity_model.compile(optimizer="adam", loss="binary_crossentropy")
    propensity_model.fit(covariates, treatment, epochs=10, batch_size=32)

    # 인과 효과 추정은 이 모델들의 예측을 기반으로 통계적/알고리즘적 방법을 통해 수행됩니다.
    # (예: Inverse Probability Weighting, G-computation, Double Machine Learning 등)
    # Keras는 이러한 추정 방법의 구성 요소인 예측 모델을 구축하는 데 활용됩니다.
    ```

### 9.8. 일반적인 팁 및 전략

#### 9.8.1. 모델 최적화 및 성능 튜닝 팁

*   **데이터 전처리:** 데이터의 스케일링, 정규화, 이상치 처리 등은 모델 성능에 큰 영향을 미칩니다. Keras 전처리 레이어나 `tf.data`를 활용하여 효율적인 파이프라인을 구축합니다.
*   **학습률:** 가장 중요한 하이퍼파라미터 중 하나입니다. 너무 높으면 발산하고, 너무 낮으면 수렴이 느립니다. 학습률 스케줄러를 사용하거나, Learning Rate Finder와 같은 도구를 사용하여 최적의 학습률을 찾습니다.
*   **배치 크기:** GPU 메모리 사용량과 학습 안정성에 영향을 미칩니다. 일반적으로 큰 배치 크기가 학습 속도에 유리하지만, 일반화 성능에는 작은 배치 크기가 더 좋을 수 있습니다.
*   **옵티마이저:** Adam이 대부분의 경우 좋은 성능을 보이지만, SGD with Momentum, RMSprop 등 다른 옵티마이저도 시도해 볼 가치가 있습니다.
*   **정규화:** 과적합을 방지하기 위해 Dropout, L1/L2 정규화, Batch Normalization 등을 적절히 사용합니다.
*   **조기 종료 (Early Stopping):** 검증 성능이 더 이상 개선되지 않을 때 학습을 중단하여 과적합을 방지하고 학습 시간을 절약합니다.
*   **모델 앙상블:** 여러 모델의 예측을 결합하여 단일 모델보다 더 좋은 성능을 얻을 수 있습니다.
*   **코드 예제 및 최적화 패턴**:
    ```python
    import tensorflow as tf

    # 파일 경로 리스트로부터 Dataset 생성
    # list_of_filenames = ["file1.tfrecord", "file2.tfrecord", ...]
    # dataset = tf.data.TFRecordDataset(list_of_filenames)
    dataset = tf.data.Dataset.from_tensor_slices(tf.range(1000))

    # --- 비효율적인 순서 (안티 패턴) ---
    # dataset.map(heavy_preprocessing_func) # 1. 모든 개별 데이터에 전처리 적용
    #        .shuffle(1000)                 # 2. 전처리된 모든 데이터를 메모리에 올려 셔플
    #        .batch(64)                     # 3. 배치화
    #        .prefetch(tf.data.AUTOTUNE)    # 4. 프리페치

    # --- 효율적인 순서 (최적 패턴) ---
    # 1. 로딩 후 바로 캐싱 (원본 데이터가 메모리에 적합할 경우)
    # 전처리 비용이 비쌀 경우, 전처리 후의 데이터를 캐싱하는 것이 더 효율적일 수 있음
    dataset = dataset.cache()

    # 2. 셔플 (전체 데이터에 대해 수행)
    dataset = dataset.shuffle(buffer_size=1000)

    # 3. 전처리 (map): 셔플된 데이터에 대해 병렬로 수행
    # dataset = dataset.map(preprocessing_func, num_parallel_calls=tf.data.AUTOTUNE)

    # 4. 배치화
    dataset = dataset.batch(64)

    # 5. 프리페치 (가장 마지막 단계)
    # GPU가 현재 배치를 학습하는 동안, CPU는 다음 배치를 미리 준비하도록 함
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print("최적화된 tf.data 파이프라인:", dataset)
    ```

#### 9.8.2. 일반적인 에러 처리 및 디버깅 전략

*   **Shape 불일치:** Keras 모델에서 가장 흔히 발생하는 오류 중 하나입니다. `model.summary()`를 통해 각 레이어의 입출력 형태를 확인하고, `tf.print(tensor.shape)` 또는 `tensor.shape`를 사용하여 텐서의 형태를 추적합니다.
*   **NaN/Inf 값:** 손실이나 가중치에서 `NaN` 또는 `Inf` 값이 발생하면 학습이 중단됩니다. (TensorFlow.md의 3.8절 참조)
    *   **원인:** 높은 학습률, 데이터 스케일링 문제, 손실 함수 문제, 기울기 폭주 등.
    *   **해결:** 학습률 감소, 데이터 정규화, 손실 함수 확인, `tf.debugging.check_numerics` 사용, `tf.clip_by_value` 또는 `tf.clip_by_norm`를 이용한 기울기 클리핑.
*   **과적합/과소적합:**
    *   **과소적합 (Underfitting):** 학습 데이터와 검증 데이터 모두에서 성능이 낮은 경우. 모델의 복잡도를 높이거나, 더 오래 학습시키거나, 더 좋은 특징을 사용해야 합니다.
    *   **과적합 (Overfitting):** 학습 데이터에서는 성능이 좋지만 검증 데이터에서는 성능이 낮은 경우. 데이터 증강, 정규화 기법(Dropout, L1/L2, Batch Normalization), 모델 복잡도 감소, 더 많은 데이터 확보 등으로 해결합니다.
*   **TensorBoard 활용:** 학습 과정의 손실, 메트릭, 가중치 분포 등을 시각화하여 문제의 원인을 파악합니다.

*   **`tf.function`과 Eager Execution 디버깅:**
    *   **개념:** TensorFlow 2.x는 기본적으로 Eager Execution을 사용하여 즉시 실행되는 Python 코드를 작성할 수 있게 합니다. 이는 디버깅을 용이하게 하지만, 성능 최적화를 위해 `tf.function` 데코레이터를 사용하여 그래프 모드로 컴파일할 수 있습니다. `tf.function`으로 컴파일된 코드는 더 빠르지만, Python 디버거로 내부를 들여다보기 어렵습니다.
    *   **해결:** `tf.function` 내부에서 문제가 발생할 경우, 일시적으로 Eager Execution을 강제하여 디버깅할 수 있습니다. `tf.config.run_functions_eagerly(True)`를 코드 시작 부분에 추가하면 모든 `tf.function`이 Eager 모드로 실행되어 표준 Python 디버거를 사용할 수 있게 됩니다. 디버깅 후에는 이 설정을 다시 `False`로 변경하거나 제거하여 성능을 복원해야 합니다.
    *   **예시:**
        ```python
        import tensorflow as tf

        # 디버깅을 위해 tf.function을 Eager 모드로 실행
        tf.config.run_functions_eagerly(True)

        @tf.function
        def my_function(x):
            # 여기에 디버깅이 필요한 복잡한 로직이 있다고 가정
            if tf.reduce_sum(x) < 0:
                tf.print("Warning: Sum is negative!") # tf.print는 그래프 모드에서도 작동
            return x * 2

        print(my_function(tf.constant([-1.0, 2.0])))

        # 디버깅 완료 후 다시 그래프 모드로 전환 (선택 사항)
        tf.config.run_functions_eagerly(False)
        ```

#### 9.8.3. 고급 디버깅 기법 (NaN 값, 기울기 문제 해결)

#### 9.8.3.1 고급 디버깅 및 문제 해결 전략

**개요**: 모델이 예상대로 동작하지 않거나, 학습 중 `NaN`이 발생하는 등 복잡한 문제를 해결하기 위한 구체적인 디버깅 기술을 추가합니다.

#### 9.8.3.1.1 Eager Execution을 활용한 대화형 디버깅

*   **제안 내용**: `tf.function`으로 컴파일된 그래프 내부를 일반 Python 디버거(pdb, IDE 디버거 등)로 단계별 실행하며 디버깅하는 방법을 소개합니다. `tf.config.run_functions_eagerly(True)`를 사용하면 Keras 모델의 `call` 함수나 `train_step` 내부의 모든 연산을 즉시 실행 모드로 전환하여, 중간 텐서의 값과 shape을 쉽게 확인할 수 있습니다.
*   **실무적 중요성**: 복잡한 Subclassing 모델이나 Custom Training Loop에서 발생하는 미묘한 로직 오류나 shape 불일치 문제를 찾는 데 매우 효과적입니다.
*   **코드 예제**:
    ```python
    import tensorflow as tf
    import keras

    # 디버깅이 필요할 때 코드 상단에 추가
    tf.config.run_functions_eagerly(True)

    class MyModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense1 = keras.layers.Dense(10)
            self.dense2 = keras.layers.Dense(5)

        def call(self, x):
            # 이제 여기에 breakpoint를 설정하고 디버거로 x의 값을 확인할 수 있습니다.
            # import pdb; pdb.set_trace()
            x = self.dense1(x)
            # 여기에서도 중간 텐서 x를 확인할 수 있습니다.
            x = self.dense2(x)
            return x

    model = MyModel()
    dummy_data = tf.random.uniform((1, 20))
    output = model(dummy_data)
    print("Eager execution successful, debugging is possible.")

    # 디버깅 완료 후 성능을 위해 다시 비활성화
    tf.config.run_functions_eagerly(False)
    ```

#### 9.8.3.2 `tf.print()`를 이용한 그래프 내부 값 확인

*   **제안 내용**: Eager 모드로 전환하는 것이 부담스러울 때, `tf.function` 내부에서 `print()` 대신 `tf.print()`를 사용하여 그래프 실행 중에도 중간 텐서의 값, 모양, 타입을 출력하는 방법을 소개합니다.
*   **실무적 중요성**: 학습 루프의 특정 단계에서만 발생하는 문제를 진단하거나, GPU에서 실행되는 그래프의 중간 상태를 확인하는 데 유용합니다.
*   **코드 예제**:
    ```python
    import tensorflow as tf
    import keras

    class CustomLayerWithPrint(keras.layers.Layer):
        def call(self, inputs):
            # tf.print는 그래프 모드에서도 동작합니다.
            tf.print("Inside CustomLayer - Input shape:", tf.shape(inputs), "Input mean:", tf.reduce_mean(inputs))
            return inputs * 2

    model = keras.Sequential([
        keras.Input(shape=(10,)),
        CustomLayerWithPrint()
    ])

    # 모델을 실행하면 tf.print의 출력이 표시됩니다.
    model(tf.random.uniform((2, 10)))
    ```

#### 9.8.4. Keras 모델의 인터프리터빌리티 (LIME, SHAP)

*   **개념:** 딥러닝 모델의 예측이 왜 그렇게 나왔는지 이해하고 설명할 수 있도록 돕는 기술입니다.
*   **실무 관점:**
    *   **LIME (Local Interpretable Model-agnostic Explanations):** 개별 예측에 대한 설명을 제공합니다. 모델에 독립적이므로 Keras 모델에도 적용할 수 있습니다.
    *   **SHAP (SHapley Additive exPlanations):** 게임 이론 기반으로 각 특징이 예측에 기여하는 정도를 정량화합니다. 역시 모델에 독립적이므로 Keras 모델에 적용할 수 있습니다.
    *   **활용 사례:** 의료 진단, 금융 사기 탐지 등 모델의 의사 결정 과정을 이해하고 신뢰성을 확보해야 하는 분야.

## 10. Keras 생태계: 도메인 특화 라이브러리

Keras는 핵심 라이브러리 외에도 특정 도메인에 특화된 기능을 제공하는 확장 라이브러리들을 통해 생태계를 확장하고 있습니다.

### 10.1. KerasCV (컴퓨터-비전)

*   **개념:** 컴퓨터 비전 태스크를 위한 Keras 네이티브 레이어, 모델, 유틸리티를 제공하는 라이브러리입니다. 이미지 분류, 객체 탐지, 이미지 분할 등 다양한 비전 태스크를 위한 빌딩 블록을 포함합니다.
*   **실무 관점:**
    *   **데이터 증강:** `keras_cv.layers.RandAugment`, `keras_cv.layers.CutMix` 등 고급 데이터 증강 기법을 쉽게 적용할 수 있습니다.
    *   **모델 아키텍처:** 사전 학습된 최신 비전 모델(예: `YOLOV8Detector`, `StableDiffusion`)을 제공하여 빠르게 모델을 구축하고 실험할 수 있습니다.
    *   **평가 메트릭:** 비전 태스크에 특화된 메트릭(예: `COCOMeanAveragePrecision`)을 제공합니다.
    *   **활용 사례:** 이미지 분류, 객체 탐지, 이미지 분할 등 컴퓨터 비전 프로젝트의 개발 생산성을 높입니다.

```python
import keras
import keras_cv
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Using CIFAR-10 for demonstration (32x32 images)
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = tf.keras.datasets.cifar10.load_data()
x_train_cifar = x_train_cifar.astype(np.float32) / 255.0
y_train_cifar = y_train_cifar.astype(np.int32)
x_test_cifar = x_test_cifar.astype(np.float32) / 255.0
y_test_cifar = y_test_cifar.astype(np.int32)

# Convert to tf.data.Dataset
batch_size = 32

train_ds = tf.data.Dataset.from_tensor_slices((x_train_cifar, y_train_cifar))
train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test_cifar, y_test_cifar))
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- KerasCV Preprocessing and Model ---
# 1. Data Augmentation with KerasCV
# KerasCV provides powerful and efficient data augmentation layers.
# We'll use RandAugment for advanced augmentation.

# Define a simple augmentation pipeline
augmentation_layers = keras.Sequential([
    keras_cv.layers.RandomFlip(mode="horizontal"),
    keras_cv.layers.RandomRotation(factor=0.1),
    keras_cv.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    # keras_cv.layers.RandAugment(value_range=(0, 255), augmentations_per_call=2, magnitude=0.5) # More advanced
], name="augmentation_pipeline")

# Apply augmentation to the dataset
def apply_augmentation(images, labels):
    return augmentation_layers(images), labels

train_ds_augmented = train_ds.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

# 2. KerasCV Model: Pre-trained ResNet50 for Image Classification
# KerasCV provides pre-trained models with built-in preprocessing.

# Load a pre-trained ResNet50 model from KerasCV
# `num_classes` will add a classification head for CIFAR-10
# `input_shape` is important for the model to adapt
keras_cv_model = keras_cv.models.ResNet50(num_classes=10, input_shape=(32, 32, 3), include_rescaling=True)

# Compile the model
keras_cv_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

keras_cv_model.summary()

# --- Training Example ---
print("\n--- Training KerasCV Image Classification Model ---")
history_keras_cv = keras_cv_model.fit(
    train_ds_augmented,
    validation_data=test_ds,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Evaluation Example ---
print("\n--- Evaluating KerasCV Model ---")
loss, accuracy = keras_cv_model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# --- Prediction Example ---
print("\n--- Making a prediction ---")
# Get a batch of test images
sample_images, sample_labels = next(iter(test_ds))

# Predict on the first image in the batch
prediction = keras_cv_model.predict(tf.expand_dims(sample_images[0], axis=0))
predicted_class = np.argmax(prediction[0])
true_class = sample_labels[0].numpy()

print(f"True class: {true_class}, Predicted class: {predicted_class}")
```

### 10.2. KerasNLP (자연어-처리)

*   **개념:** 자연어 처리(NLP) 태스크를 위한 Keras 네이티브 레이어, 모델, 유틸리티를 제공하는 라이브러리입니다. 텍스트 분류, 기계 번역, 텍스트 생성 등 다양한 NLP 태스크를 위한 빌딩 블록을 포함합니다.
*   **실무 관점:**
    *   **전처리:** `keras_nlp.layers.TextVectorization`, `keras_nlp.tokenizers.WordPieceTokenizer` 등 고급 텍스트 전처리 및 토큰화 도구를 제공합니다.
    *   **모델 아키텍처:** 사전 학습된 최신 NLP 모델(예: `BertClassifier`, `Gpt2CausalLM`)을 제공하여 빠르게 모델을 구축하고 미세 조정할 수 있습니다.
    *   **활용 사례:** 텍스트 분류, 감성 분석, 질의응답, 챗봇, 텍스트 생성 등 NLP 프로젝트의 개발 생산성을 높입니다.

```python
import keras
import keras_nlp
import tensorflow as tf
import numpy as np

# --- Dummy Data Generation ---
# Example: Sentiment analysis (positive/negative reviews)
# In a real scenario, you would load a dataset like IMDB reviews.

dummy_texts = [
    "This movie was fantastic! I loved every moment.",
    "Absolutely terrible, a complete waste of time.",
    "It was okay, nothing special but not bad either.",
    "Highly recommend, great acting and story.",
    "Couldn't stand it, so boring and predictable."
]
dummy_labels = [1, 0, 0, 1, 0] # 1 for positive, 0 for negative/neutral

# Convert to tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((dummy_texts, dummy_labels))
dataset = dataset.batch(2).prefetch(tf.data.AUTOTUNE)

# --- KerasNLP Preprocessing and Model ---
# 1. Preprocessing: Tokenization and Packing
# Using a pre-trained BERT tokenizer
# Note: KerasNLP models often come with their own preprocessors.

# Choose a pre-trained preset for BERT (e.g., 'bert_base_en_uncased')
# This will download the tokenizer and preprocessor weights.
# You might need to install `tensorflow_text` for some tokenizers.
# pip install tensorflow_text

# For demonstration, we'll use a simple tokenizer and then a BERT classifier.
# In a real KerasNLP workflow, you'd use `keras_nlp.models.BertPreprocessor`
# and `keras_nlp.models.BertClassifier.from_preset`.

# Let's simulate the preprocessing step for a BERT-like model
# A real BERT preprocessor would handle tokenization, padding, and adding special tokens.
# For simplicity, we'll just use a TextVectorization layer for now.

# For a full KerasNLP example, you'd do:
# preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
#     "bert_base_en_uncased",
#     sequence_length=128
# )
# classifier = keras_nlp.models.BertClassifier.from_preset(
#     "bert_base_en_uncased",
#     preprocessor=preprocessor,
#     num_classes=2 # Binary classification
# )

# Let's use a simpler approach that still demonstrates KerasNLP model usage.
# We'll use a pre-trained BERT backbone and add a classification head.

# Load a pre-trained BERT backbone (encoder)
# This will download model weights.
bert_backbone = keras_nlp.models.BertBase(include_preprocessing=False)

# Build a classifier on top of the BERT backbone
inputs = keras.Input(shape=(), dtype=tf.string, name="text_input")

# TextVectorization for simple tokenization (replace with KerasNLP preprocessor for real use)
vectorize_layer = layers.TextVectorization(
    max_tokens=bert_backbone.vocabulary_size(),
    output_mode="int",
    output_sequence_length=bert_backbone.sequence_length,
)
vectorize_layer.adapt(dummy_texts) # Adapt to your data

# Apply vectorization
vectorized_text = vectorize_layer(inputs)

# Pass through BERT backbone
# The BERT backbone expects a dictionary of inputs (token_ids, segment_ids, padding_mask)
# For simplicity, we'll just pass token_ids and let the model handle defaults.
# In a real scenario, the preprocessor would create these.
bart_inputs = {
    "token_ids": vectorized_text,
    "segment_ids": tf.zeros_like(vectorized_text),
    "padding_mask": tf.cast(vectorized_text != 0, dtype=tf.int32)
}

bert_output = bert_backbone(bart_inputs) # This returns a dictionary, we need the pooled output
pooled_output = bert_output["pooled_output"]

# Classification head
output = layers.Dense(1, activation="sigmoid", name="output")(pooled_output)

keras_nlp_model = keras.Model(inputs=inputs, outputs=output, name="KerasNLP_Text_Classifier")

keras_nlp_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

keras_nlp_model.summary()

# --- Training Example ---
print("\n--- Training KerasNLP Text Classification Model ---")
history_keras_nlp = keras_nlp_model.fit(
    dataset,
    epochs=1, # Increase epochs for better results
    verbose=1
)

# --- Prediction Example ---
print("\n--- Making a prediction ---")
new_texts = [
    "This is a truly amazing experience!",
    "I regret watching this, it was so bad."
]

predictions = keras_nlp_model.predict(tf.constant(new_texts))

for text, pred in zip(new_texts, predictions):
    sentiment = "Positive" if pred[0] > 0.5 else "Negative"
    print(f"Text: \"{text}\" -> Predicted Sentiment: {sentiment} (Score: {pred[0]:.2f})")
```

### 10.3. Keras-RL (강화 학습 라이브러리)

*   **개념:** Keras를 기반으로 다양한 강화 학습(Reinforcement Learning) 알고리즘을 구현하고 실험할 수 있도록 돕는 라이브러리입니다.
*   **실무 관점:**
    *   **알고리즘 구현:** DQN, DDPG, A2C 등 널리 사용되는 강화 학습 알고리즘의 구현체를 제공합니다.
    *   **환경 연동:** OpenAI Gym과 같은 강화 학습 환경과 쉽게 연동할 수 있습니다.
    *   **활용 사례:** 게임 AI 개발, 로봇 제어, 자율 주행 시뮬레이션 등 강화 학습 프로젝트에 활용됩니다.

```python
import keras
from keras import layers
import tensorflow as tf
import numpy as np

# Install gymnasium if not already installed
# pip install gymnasium
import gymnasium as gym

# Install keras-rl2 if not already installed
# pip install keras-rl2
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# --- 1. Define the Environment ---
# We'll use the classic CartPole-v1 environment from Gymnasium
env = gym.make('CartPole-v1', render_mode='rgb_array')
np.random.seed(123)
env.action_space.seed(123)

num_actions = env.action_space.n
input_shape = env.observation_space.shape

print(f"Observation space shape: {input_shape}") # (4,)
print(f"Number of actions: {num_actions}") # 2 (left or right)

# --- 2. Build the Keras Model (Q-Network) ---
# This is a simple MLP that takes the observation as input and outputs Q-values for each action.

def build_q_network(input_shape, num_actions):
    inputs = layers.Input(shape=(1,) + input_shape) # Keras-RL expects (batch, timesteps, features)
    x = layers.Flatten()(inputs)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(num_actions, activation='linear')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

q_network = build_q_network(input_shape, num_actions)
q_network.summary()

# --- 3. Build the Agent (DQN) ---
# Keras-RL provides agents like DQN, DDPG, A2C, etc.

# Policy: How the agent selects actions (e.g., Epsilon-Greedy)
policy = EpsGreedyQPolicy(eps=1.0, eps_decay_rate=0.995, eps_min=0.1)

# Memory: Stores experiences (state, action, reward, next_state, done)
memory = SequentialMemory(limit=50000, window_length=1)

# DQN Agent
dqn_agent = DQNAgent(
    model=q_network,
    nb_actions=num_actions,
    policy=policy,
    memory=memory,
    nb_steps_warmup=1000, # Number of random steps before training starts
    gamma=0.99, # Discount factor
    target_model_update=1000, # Update target network every X steps
    train_interval=4, # Train every X steps
    delta_clip=1.0 # For Huber loss
)

# --- 4. Compile the Agent ---
# The agent needs to be compiled with an optimizer and metrics.
# Keras-RL handles the loss function internally for DQN (usually MSE or Huber).

dqn_agent.compile(keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

# --- 5. Train the Agent ---
# Training involves the agent interacting with the environment.
print("\n--- Training Keras-RL DQN Agent (CartPole) ---")
# nb_steps: Total number of interactions with the environment
dqn_agent.fit(env, nb_steps=5000, visualize=False, verbose=1)

# --- 6. Evaluate the Agent ---
print("\n--- Evaluating Keras-RL DQN Agent (CartPole) ---")
history = dqn_agent.test(env, nb_episodes=10, visualize=False, verbose=0)
print(f"Average reward over 10 episodes: {np.mean(history.history['episode_reward']):.2f}")

# Close the environment
env.close()
```

## 11. 책임감 있는 AI (Responsible AI) 및 모델 해석

AI 시스템이 사회에 미치는 영향이 커지면서, 모델의 공정성, 투명성, 개인 정보 보호, 견고성 등을 고려하는 책임감 있는 AI 개발이 중요해지고 있습니다. Keras는 이러한 목표를 지원하기 위한 도구와 가이드라인을 제공합니다.

### 11.1. Explainable AI (XAI) 개요 및 Keras 모델 해석

*   **개념:** 딥러닝 모델과 같이 복잡한 "블랙박스" 모델의 예측이 왜 그렇게 나왔는지 이해하고 설명할 수 있도록 돕는 기술입니다.
*   **실무 관점:**
    *   **신뢰성 확보:** 모델의 의사 결정 과정을 이해함으로써 사용자와 개발자의 신뢰를 높입니다.
    *   **디버깅 및 개선:** 모델이 잘못된 예측을 하는 이유를 파악하여 모델을 개선하는 데 도움을 줍니다.
    *   **규제 준수:** 금융, 의료 등 규제가 엄격한 분야에서는 모델의 설명 가능성이 법적 요구사항이 될 수 있습니다.
    *   **Keras 모델 해석 도구:**
        *   **LIME, SHAP:** Keras 모델에 독립적으로 적용할 수 있는 모델-불가지론적(model-agnostic) 설명 도구입니다.
        *   **Integrated Gradients:** TensorFlow의 `tf.GradientTape`를 사용하여 Keras 모델의 입력 특징이 예측에 미치는 영향을 계산할 수 있습니다.
        *   **What-If Tool:** Keras 모델과 연동하여 모델의 동작을 탐색하고, 데이터셋의 특정 특징을 변경했을 때 모델 예측이 어떻게 변하는지 시각적으로 확인할 수 있습니다.
    *   **코드 예제**:
        ```python
        # SHAP 라이브러리 설치 필요: pip install shap
        import shap
        import numpy as np
        import keras

        # 1. 학습된 Keras 모델 및 데이터 준비 (예시)
        model = keras.Sequential([
            keras.Input(shape=(10,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        x_train = np.random.rand(100, 10)
        x_test = np.random.rand(10, 10)

        # 2. SHAP Explainer 생성
        # KernelExplainer는 모델 종류에 상관없이 사용할 수 있어 범용적입니다.
        # model.predict 함수와 배경 데이터셋(학습 데이터의 일부)을 전달합니다.
        explainer = shap.KernelExplainer(model.predict, x_train)

        # 3. SHAP 값 계산
        # 설명하고 싶은 데이터(x_test)에 대한 SHAP 값을 계산합니다.
        shap_values = explainer.shap_values(x_test)

        # 4. 결과 시각화
        # 첫 번째 테스트 데이터의 예측에 대한 특성 기여도를 시각화합니다.
        shap.initjs() # 노트북 환경에서 JS 시각화를 위해 필요
        print("SHAP 값 계산 완료. 아래 코드로 시각화 가능합니다.")
        shap.force_plot(explainer.expected_value[0], shap_values[0][0], x_test[0])
        ```
### 11.2. 모델 공정성 (Fairness) 및 편향 감지

*   **개념:** 모델이 특정 인구 집단(성별, 인종, 연령 등)에 대해 불공정한 예측이나 편향된 결과를 생성하는지 평가하고 완화하는 것입니다.
*   **실무 관점:**
    *   **사회적 영향:** 대출 승인, 채용, 범죄 예측 등 사회적으로 민감한 분야에서 모델의 불공정성은 심각한 사회적 문제를 야기할 수 있습니다.
    *   **평가:** `Fairness Indicators` 라이브러리(TensorFlow Extended의 일부)를 사용하여 Keras 모델의 공정성을 평가할 수 있습니다. 다양한 인구 집단에 대한 모델의 성능(정확도, 정밀도, 재현율 등)을 비교하고, 불공정성을 나타내는 지표를 시각화합니다.
    *   **완화:** 데이터 증강, 재샘플링, 공정성 제약 조건이 있는 손실 함수 사용 등 다양한 기법을 통해 모델의 공정성을 개선할 수 있습니다.

### 11.3. 프라이버시 보호 (Differential Privacy)

*   **개념:** 머신러닝 모델 학습 과정에서 개인 정보가 노출되는 것을 방지하는 기술입니다. 특히 차등 프라이버시(Differential Privacy)를 적용하여 학습 데이터의 개별 레코드가 모델에 미치는 영향을 제한합니다.
*   **실무 관점:**
    *   **민감 데이터 처리:** 의료 기록, 금융 거래 내역 등 민감한 개인 정보를 포함하는 데이터셋으로 Keras 모델을 학습할 때 필수적입니다.
    *   **규제 준수:** GDPR, CCPA 등 개인 정보 보호 규제를 준수하는 데 도움을 줍니다.
    *   **TensorFlow Privacy:** `tf_privacy` 라이브러리를 사용하여 Keras 모델에 차등 프라이버시를 지원하는 옵티마이저(예: `DP-SGD`)를 적용할 수 있습니다.

    **예시 (DP-SGD 옵티마이저 적용):**
    ```python
    import tensorflow as tf
    import tensorflow_privacy as tfp
    import numpy as np

    # 더미 데이터
    x_train = np.random.rand(100, 784).astype("float32")
    y_train = np.random.randint(0, 10, size=(100,)).astype("int32")

    # Keras 모델 정의
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # 차등 프라이버시를 지원하는 DP-SGD 옵티마이저 설정
    # `l2_norm_clip`: 각 샘플의 기울기 L2 노름을 이 값으로 클리핑합니다.
    # `noise_multiplier`: 노이즈의 표준 편차를 결정하는 계수입니다. 값이 클수록 더 많은 노이즈가 추가되어 프라이버시 보호 수준이 높아지지만, 모델 정확도는 낮아질 수 있습니다.
    # `num_microbatches`: 각 배치 내에서 기울기를 계산할 마이크로 배치 수입니다. 일반적으로 1로 설정하거나, 배치 크기의 약수로 설정합니다.
    optimizer = tfp.privacy.DPKerasAdamOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=1.1,
        num_microbatches=1,
        learning_rate=0.001)

    # 손실 함수 (from_logits=True는 softmax 활성화 함수를 사용하지 않을 때)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # 모델 컴파일
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # 모델 학습 (일반적인 model.fit 사용)
    # model.fit(x_train, y_train, epochs=1, batch_size=32)

    # 참고: 실제 DP-SGD 적용 시에는 데이터셋 크기, 배치 크기, 에포크 수 등을 고려하여
    # 프라이버시 예산 (epsilon, delta)을 계산하고 적절한 하이퍼파라미터를 설정해야 합니다.
    # tfp.privacy.compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy() 함수를 활용할 수 있습니다.
    ```

### 11.4. 모델 견고성 (Robustness)

*   **개념:** 모델이 적대적 공격(Adversarial Attacks)이나 노이즈가 포함된 입력에 대해 얼마나 안정적으로 예측하는지 평가하고 개선하는 것입니다.
*   **실무 관점:**
    *   **보안 취약점:** 이미지 분류 모델에 미세한 노이즈를 추가하여 모델이 완전히 다른 클래스로 오분류하게 만드는 적대적 공격은 모델의 보안 취약점을 드러냅니다.
    *   **평가:** `CleverHans`와 같은 라이브러리를 사용하여 Keras 모델에 대한 적대적 샘플을 생성하고 모델의 견고성을 평가합니다.
    *   **방어:** 적대적 학습(Adversarial Training), 입력 전처리, 모델 앙상블 등 다양한 방어 기법을 통해 모델의 견고성을 향상시킬 수 있습니다.
    *   **활용 사례:** 자율 주행, 얼굴 인식 등 안전이 중요한 애플리케이션에서 모델의 견고성은 매우 중요합니다.

### **12. 지속 가능한 ML 시스템을 위한 설계 원칙과 트레이드오프**

이 장에서는 지금까지 배운 Keras의 강력한 기능들을 실제 프로젝트에 적용할 때 필요한 전략적 사고의 틀을 다룹니다. 성공적인 머신러닝 시스템은 단순히 뛰어난 모델 하나로 완성되지 않습니다. 그것은 비즈니스 요구사항, 개발 속도, 운영 안정성, 그리고 미래의 확장성 사이의 끊임없는 트레이드오프를 통해 탄생하는 공학적 산물입니다.

#### 12.1. 문제 정의와 기술 선택의 기준

> "모든 문제에 딥러닝이 답은 아니다."

*   **단순한 모델을 이길 수 있는가? (Beat the Baseline)**: 새로운 프로젝트를 시작할 때, Keras로 복잡한 딥러닝 모델을 만들기 전에 항상 Scikit-learn의 `LogisticRegression`이나 `XGBoost` 같은 단순하고 해석 가능한 모델로 베이스라인을 설정해야 합니다. 딥러닝 모델의 복잡성과 개발/운영 비용은 이 베이스라인을 압도하는 성능 향상이 있을 때만 정당화될 수 있습니다.

*   **속도 vs 정확도 vs 비용의 삼각관계**: 모든 ML 프로젝트는 이 세 가지 요소의 트레이드오프 위에 서 있습니다. 당신의 문제는 어디에 속합니까?
    *   **정확도가 최우선 (e.g., 의료 영상 진단)**: `EfficientNetV2`, `ResNet` 등 SOTA 모델을 활용하고, `KerasTuner`를 통한 광범위한 탐색과 `지식 증류`를 통해 성능을 극한까지 끌어올리는 전략이 유효합니다. 추론 속도나 비용은 후순위가 될 수 있습니다.
    *   **속도가 최우선 (e.g., 실시간 객체 탐지, 모바일 앱)**: `MobileNet`, `YOLO` 계열의 경량 모델을 선택하고, `양자화(Quantization)`, `가지치기(Pruning)`, `TensorRT` 변환 등을 적극적으로 고려해야 합니다. 약간의 정확도 손실은 감수할 수 있습니다.
    *   **비용이 최우선 (e.g., 대규모 배치 분석)**: 학습 및 서빙에 드는 컴퓨팅 비용을 최소화해야 합니다. `서버리스(Serverless)` 환경에서의 모델 배포, `Gradient Accumulation`을 통한 저사양 GPU 활용, 학습된 모델의 재사용 극대화 전략이 필요합니다.

#### 12.2. 개발 속도와 시스템 안정성의 균형

> "오늘의 빠른 코드가 내일의 기술 부채가 될 수 있다."

*   **탐색(Exploration)과 제품화(Productionization)의 2단계 워크플로우**: ML 프로젝트는 두 단계를 거칩니다. 이 단계를 명확히 구분하고 각 단계에 맞는 도구를 사용해야 합니다.
    1.  **탐색 단계**: Jupyter Notebook 환경에서 자유롭게 아이디어를 실험합니다. 데이터의 특성을 파악하고, 다양한 모델 아키텍처를 빠르게 시도하는 것이 중요합니다. 이때는 `Subclassing API`의 유연성이 빛을 발할 수 있습니다.
    2.  **제품화 단계**: 검증된 아이디어를 안정적인 시스템으로 구축합니다. Notebook의 코드를 모듈화된 Python 스크립트로 리팩토링하고, `Functional API`처럼 구조가 명확한 API를 사용합니다. `YAML 설정 파일`, `DVC`, `MLflow`를 도입하여 모든 것을 재현 가능하게 만듭니다.

*   **Keras API 선택 전략**: 어떤 API를 선택할지는 프로젝트의 성숙도에 따라 달라집니다.
    *   `Sequential API`: 가장 간단한 PoC(Proof of Concept)나 베이스라인 모델에 적합합니다.
    *   `Functional API`: 복잡하지만 구조가 고정된 대부분의 프로덕션 모델에 가장 이상적인 선택입니다. 모델의 구조를 명확하게 보여주어 유지보수에 용이합니다.
    *   `Subclassing API`: 동적 그래프나 혁신적인 아키텍처를 연구하는 탐색 단계에 적합하지만, 프로덕션 환경에서는 디버깅과 직렬화의 어려움으로 인해 신중하게 사용해야 합니다.

#### 12.3. '기술 부채'를 경계하는 MLOps 설계

> "배포는 끝이 아니라 시작이다."

*   **점진적인 자동화 (Progressive Automation)**: MLOps는 모든 것을 한 번에 자동화하는 것이 아닙니다. 프로젝트의 규모와 성숙도에 따라 점진적으로 도입해야 합니다.
    *   **Level 0 (수동)**: 모든 것을 수동으로 실행. (대부분의 초기 프로젝트)
    *   **Level 1 (파이프라인 자동화)**: 데이터 전처리, 모델 학습, 모델 검증 과정을 하나의 파이프라인으로 자동화. (e.g., Kubeflow Pipelines, TFX)
    *   **Level 2 (CI/CD/CT)**: 코드 변경(CI), 새로운 모델의 자동 배포(CD), 새로운 데이터에 대한 자동 재학습(CT)까지 완전 자동화. (대규모 서비스)

*   **조용한 암살자, 드리프트(Drift)와의 전쟁**: 모델은 시간이 지나면서 조용히 성능이 저하됩니다. 이를 방지하기 위한 모니터링은 필수입니다.
    *   **데이터 드리프트**: 입력 데이터의 통계적 분포 변화를 감지합니다. (e.g., `tfdv.validate_statistics`)
    *   **개념 드리프트**: 입력 데이터와 타겟 변수 간의 관계 변화를 감지합니다.
    *   **모니터링 전략**: 프로덕션 환경의 입력 데이터와 모델 예측 결과를 주기적으로 로깅하고, 학습 데이터의 분포와 비교하여 드리프트가 감지되면 자동으로 경고를 보내거나 재학습 파이프라인을 트리거하는 시스템을 구축해야 합니다.

#### 12.4. Keras를 넘어: 미래를 위한 제언

> "망치를 든 사람에게는 모든 것이 못으로 보인다."

*   **도구의 한계 인정하기**: Keras는 훌륭하지만 만능은 아닙니다. 수백억 파라미터 모델의 복잡한 병렬 처리(`Model Parallelism`, `Pipeline Parallelism`)나, 자동 미분 엔진의 근본적인 수정을 요구하는 연구에는 한계가 있을 수 있습니다. 이러한 문제에 직면했을 때, `JAX` 네이티브 코드, `PyTorch FSDP`, `DeepSpeed`와 같은 더 로우-레벨의 전문적인 도구들을 함께 검토할 수 있는 넓은 시야가 필요합니다.

*   **T자형 인재로 성장하기**: 이 문서를 통해 Keras라는 강력한 도구에 대한 깊이(I)를 갖추었습니다. 이제는 ML 시스템 전반에 대한 넓이(T)를 갖추어야 합니다. 최신 논문을 비판적으로 읽고(e.g., Papers with Code), 오픈소스 커뮤니티에 참여하며, 내가 사용하는 도구들이 내부적으로 어떻게 동작하는지 끊임없이 탐구하는 자세가 당신을 대체 불가능한 엔지니어로 만들 것입니다.