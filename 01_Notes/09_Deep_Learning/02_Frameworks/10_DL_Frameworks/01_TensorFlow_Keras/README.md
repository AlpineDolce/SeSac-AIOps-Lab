# TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드

이 가이드는 TensorFlow와 Keras를 활용한 딥러닝 모델 개발 및 배포에 필요한 핵심 개념과 실무 기술을 체계적으로 정리합니다. 데이터 전처리부터 모델 구축, 학습, 평가, 그리고 실제 서비스 환경에서의 배포까지 전 과정을 다루어, 독자가 TensorFlow와 Keras를 실제 프로젝트에 효과적으로 적용할 수 있도록 돕는 것을 목표로 합니다.

---

### Part 1: TensorFlow & Keras 개요 및 기본 개념
- **학습 목표:** TensorFlow와 Keras의 정의, 특징, 설치 및 환경 설정, 그리고 Keras 3.0의 멀티 백엔드 지원을 이해합니다.
- **주요 내용:**
    - [**01_TF_Keras_소개.md**](./10_Basics/01_TF_Keras_소개.md): TensorFlow와 Keras의 정의, 특징, 설치 및 환경 설정, 그리고 Keras 3.0의 멀티 백엔드 지원을 이해합니다.
    - [**02_TF_Keras_기본_데이터_구조_및_자동_미분.md**](./10_Basics/02_TF_Keras_기본_데이터_구조_및_자동_미분.md): TensorFlow의 기본 데이터 구조인 Tensor와 자동 미분(`tf.GradientTape`)의 개념을 학습합니다.
    - [**03_TF_Keras_주요_구성_요소.md**](./10_Basics/03_TF_Keras_주요_구성_요소.md): Keras 모델, 레이어, 옵티마이저, 손실 함수, 메트릭 등 딥러닝 모델을 구성하는 핵심 요소들을 학습합니다.

### Part 2: 모델 아키텍처 설계
- **학습 목표:** Keras의 Sequential API, Functional API, Subclassing API를 사용하여 다양한 모델 아키텍처를 설계하고 구현하는 방법을 학습합니다.
- **주요 내용:**
    - [**04_모델_아키텍처_Sequential_Functional.md**](./20_Model_Architecture/04_모델_아키텍처_Sequential_Functional.md): Keras의 Sequential API와 Functional API를 이용한 모델 구축 방법을 학습합니다.
    - [**05_모델_아키텍처_Subclassing_Custom.md**](./20_Model_Architecture/05_모델_아키텍처_Subclassing_Custom.md): Subclassing API를 이용한 Custom Model 구현 및 Custom Layer, Custom Loss/Metric 구현 방법을 학습합니다.
    - [**06_Keras_Applications_전이_학습.md**](./20_Model_Architecture/06_Keras_Applications_전이_학습.md): Keras Applications를 활용한 사전 학습된 모델과 전이 학습 전략을 학습합니다.

### Part 3: 데이터 파이프라인 구축 및 전처리
- **학습 목표:** `tf.data` API를 사용하여 대규모 데이터셋을 효율적으로 로드하고 전처리하며, 데이터 증강 및 불균형 데이터셋 처리 전략을 학습합니다.
- **주요 내용:**
    - [**07_TF_Data_Dataset_개요_로드.md**](./30_Data_Pipeline/07_TF_Data_Dataset_개요_로드.md): `tf.data.Dataset`의 개요와 `from_tensor_slices`, `from_generator`, TFRecord, TFDS 등 다양한 데이터 로드 방법을 학습합니다.
    - [**08_TF_Data_전처리_증강_최적화.md**](./30_Data_Pipeline/08_TF_Data_전처리_증강_최적화.md): `tf.data`의 `map`, `filter`, `batch`, `shuffle`, `tf.image`를 이용한 데이터 전처리 및 증강, `prefetch`, `cache`를 이용한 성능 최적화를 학습합니다.
    - [**09_현실적인_데이터_처리_불균형_데이터셋.md**](./30_Data_Pipeline/09_현실적인_데이터_처리_불균형_데이터셋.md): `TFRecord`를 활용한 대용량 데이터셋 처리와 불균형 데이터셋 처리 전략을 학습합니다.
    - [**10_Keras_전처리_레이어.md**](./30_Data_Pipeline/10_Keras_전처리_레이어.md): Keras 전처리 레이어(TextVectorization, Image Augmentation 등)의 개념과 활용 방법을 학습합니다.

### Part 4: 모델 학습, 평가 및 최적화
- **학습 목표:** Keras 모델의 학습, 평가, 예측을 위한 핵심 메서드들을 학습하고, 학습률 스케줄링, 조기 종료, 분산 학습, 혼합 정밀도 학습 등 고급 최적화 기법을 익힙니다.
- **주요 내용:**
    - [**11_모델_컴파일_학습_평가.md**](./40_Training_Optimization/11_모델_컴파일_학습_평가.md): 모델 컴파일(Optimizer, Loss, Metrics 설정), `model.fit()`을 이용한 모델 학습, `model.evaluate()`, `model.predict()`를 이용한 평가 및 예측 방법을 학습합니다.
    - [**12_콜백_학습률_스케줄러.md**](./40_Training_Optimization/12_콜백_학습률_스케줄러.md): 콜백(EarlyStopping, ModelCheckpoint, Custom Callbacks) 활용 및 학습률 스케줄러를 학습합니다.
    - [**13_Custom_Training_Loop_Gradient_Accumulation.md**](./40_Training_Optimization/13_Custom_Training_Loop_Gradient_Accumulation.md): Custom Training Loop 구현 및 Gradient Accumulation 기법을 학습합니다.
    - [**14_분산_학습_혼합_정밀도.md**](./40_Training_Optimization/14_분산_학습_혼합_정밀도.md): Multi-GPU/TPU 학습 전략(`tf.distribute.Strategy`) 및 혼합 정밀도(Mixed Precision) 학습을 학습합니다.
    - [**15_TF_Function_모델_최적화_기법.md**](./40_Training_Optimization/15_TF_Function_모델_최적화_기법.md): `tf.function`을 활용한 성능 최적화 및 양자화, 가지치기, XLA 등 모델 최적화 기법을 학습합니다.
    - [**16_고급_학습_전략_정규화.md**](./40_Training_Optimization/16_고급_학습_전략_정규화.md): 전이 학습, Fine-tuning, Dropout, L1/L2 정규화, Batch Normalization 등 고급 학습 전략 및 정규화 기법을 학습합니다.

### Part 5: 모델 저장, 로드 및 하이퍼파라미터 튜닝
- **학습 목표:** 학습된 모델의 상태를 저장하고 불러오는 방법과, 학습 중간 과정을 기록하는 체크포인트 관리법을 익힙니다. 자동화된 도구를 사용하여 최적의 하이퍼파라미터를 탐색하는 방법을 학습합니다.
- **주요 내용:**
    - [**17_모델_저장_로드_SavedModel_체크포인트.md**](./50_Deployment_and_MLOps/17_모델_저장_로드_SavedModel_체크포인트.md): Keras 모델 저장 형식(.keras), 가중치만/아키텍처만 저장 및 로드, TensorFlow SavedModel 형식, 체크포인트 관리 방법을 학습합니다.
    - [**18_하이퍼파라미터_튜닝_KerasTuner.md**](./50_Deployment_and_MLOps/18_하이퍼파라미터_튜닝_KerasTuner.md): KerasTuner를 이용한 하이퍼파라미터 튜닝(HyperModel 정의, Tuner 선택 및 실행, 최적 하이퍼파라미터 검색) 방법을 학습합니다.

### Part 6: 모델 배포 및 MLOps
- **학습 목표:** 학습된 모델을 TensorFlow Serving, TensorFlow Lite, TensorFlow.js 등을 사용하여 실제 서비스 환경에 배포하는 방법을 배우고, MLOps 파이프라인 구축 및 실험 관리의 기본 개념을 익힙니다.
- **주요 내용:**
    - [**19_모델_배포_TensorFlow_Serving_Lite_JS.md**](./50_Deployment_and_MLOps/19_모델_배포_TensorFlow_Serving_Lite_JS.md): TensorFlow Serving, TensorFlow Lite, TensorFlow.js를 이용한 모델 배포 방법을 학습합니다.
    - [**20_추론_최적화_ONNX_TensorRT_모델_최적화.md**](./50_Deployment_and_MLOps/20_추론_최적화_ONNX_TensorRT_모델_최적화.md): ONNX/TensorRT 변환을 통한 추론 최적화 및 양자화, 가지치기, 지식 증류 등 모델 최적화 기법을 학습합니다.
    - [**21_MLOps_통합_고려사항.md**](./50_Deployment_and_MLOps/21_MLOps_통합_고려사항.md): MLOps의 개념, 모델 버전 관리, 데이터/학습/배포 파이프라인 자동화 등 MLOps 통합 및 고려사항을 학습합니다.
    - [**22_MLOps_심화_실험관리_재현성.md**](./50_Deployment_and_MLOps/22_MLOps_심화_실험관리_재현성.md): MLflow, W&B를 활용한 체계적인 실험 관리, 코드/데이터/하이퍼파라미터 버전 관리를 통한 완벽한 재현성 확보 전략을 학습합니다.

### Part 7: TensorBoard 활용 및 생태계
- **학습 목표:** TensorBoard를 사용하여 모델의 학습 과정, 성능 지표, 계산 그래프를 시각화하고 디버깅하는 방법을 익힙니다. 또한, TensorFlow 생태계의 다양한 라이브러리를 탐색하고 최신 기능을 학습합니다.
- **주요 내용:**
    - [**23_TensorBoard_시각화_디버깅.md**](./60_Ecosystem_Advanced/23_TensorBoard_시각화_디버깅.md): TensorBoard를 활용한 Keras 모델 시각화 및 디버깅(학습 모니터링, 그래프 시각화, 프로파일링) 방법을 학습합니다.
    - [**24_TensorFlow_생태계.md**](./60_Ecosystem_Advanced/24_TensorFlow_생태계.md): TensorFlow Hub, Recommenders, Probability, Federated 등 TensorFlow 생태계의 다양한 도메인 특화 라이브러리를 학습합니다.

### Part 8: 책임감 있는 AI 및 실전 프로젝트
- **학습 목표:** 모델의 예측을 설명하는 XAI(Explainable AI) 기법을 배우고, 모델의 공정성, 프라이버시, 견고성 등 책임감 있는 AI(Responsible AI)의 주요 개념을 학습합니다. 다양한 도메인의 실전 프로젝트 예제를 통해 문제 해결 능력을 기르고, 지속 가능한 ML 시스템 설계 원칙을 이해합니다.
- **주요 내용:**
    - [**25_책임감_있는_AI.md**](./60_Ecosystem_Advanced/25_책임감_있는_AI.md): Explainable AI (XAI) with Captum, 모델 공정성, 프라이버시 보호, 모델 견고성 등 책임감 있는 AI(Responsible AI) 개념을 학습합니다.
    - [**26_실전_프로젝트_예제_팁.md**](./60_Ecosystem_Advanced/26_실전_프로젝트_예제_팁.md): 이미지 처리, 시퀀스/텍스트 처리, 그래프 신경망, 추천 시스템, 멀티모달 모델 등 다양한 실전 프로젝트 예제와 일반적인 팁 및 전략을 학습합니다.
    - [**27_지속_가능한_ML_시스템_설계.md**](./60_Ecosystem_Advanced/27_지속_가능한_ML_시스템_설계.md): 지속 가능한 ML 시스템을 위한 설계 원칙과 트레이드오프(문제 정의, 기술 선택, 개발 속도와 안정성의 균형, 기술 부채, 미래 제언)를 학습합니다.
