# PyTorch 핵심 개념 정리: 딥러닝 실무 가이드

이 가이드는 PyTorch를 활용한 딥러닝 모델 개발 및 배포에 필요한 핵심 개념과 실무 기술을 체계적으로 정리합니다. 데이터 전처리부터 모델 구축, 학습, 평가, 그리고 실제 서비스 환경에서의 배포까지 전 과정을 다루어, 독자가 PyTorch를 실제 프로젝트에 효과적으로 적용할 수 있도록 돕는 것을 목표로 합니다.

---

### Part 1: PyTorch 개요 및 기본 개념
- **학습 목표:** PyTorch의 기본 데이터 구조인 Tensor를 이해하고, 자동 미분 기능(`autograd`)을 활용하는 방법을 익힙니다. 또한, `nn.Module`, 옵티마이저, 손실 함수 등 PyTorch로 딥러닝 모델을 구성하는 핵심 요소들을 학습합니다.
- **주요 내용:**
    - [**01_PyTorch_소개.md**](./10_Basics/01_PyTorch_소개.md): PyTorch의 정의, 특징, 설치 및 환경 설정을 이해합니다.
    - [**02_PyTorch_기본_데이터_구조_및_자동_미분.md**](./10_Basics/02_PyTorch_기본_데이터_구조_및_자동_미분.md): PyTorch의 기본 데이터 구조인 Tensor와 자동 미분(`autograd`)의 개념을 학습합니다.
    - [**03_PyTorch_주요_구성_요소.md**](./10_Basics/03_PyTorch_주요_구성_요소.md): `nn.Module`, `nn.Parameter`, 옵티마이저, 손실 함수, 메트릭 등 딥러닝 모델을 구성하는 핵심 요소들을 학습합니다.

### Part 2: 모델 아키텍처 설계
- **학습 목표:** `nn.Sequential`, `nn.ModuleList`를 사용하여 표준적인 모델을 구성하고, `nn.Module`을 상속받아 특정 문제에 최적화된 맞춤형 모델(Custom Model)을 설계하는 방법을 배웁니다. 또한, `torchvision`의 사전 학습된 모델을 활용한 전이 학습(Transfer Learning) 기법을 익힙니다.
- **주요 내용:**
    - [**04_모델_아키텍처_Sequential_ModuleList.md**](./20_Model_Architecture/04_모델_아키텍처_Sequential_ModuleList.md): PyTorch의 `nn.Sequential`과 `nn.ModuleList`를 이용한 모델 구축 방법을 학습합니다.
    - [**05_모델_아키텍처_Custom_Module.md**](./20_Model_Architecture/05_모델_아키텍처_Custom_Module.md): `nn.Module`을 상속하여 Custom Model, Custom Layer, Custom Loss/Metric 구현 방법을 학습합니다.
    - [**06_TorchVision_모델_전이_학습.md**](./20_Model_Architecture/06_TorchVision_모델_전이_학습.md): `torchvision.models`를 활용한 사전 학습된 모델과 전이 학습 전략을 학습합니다.

### Part 3: 데이터 파이프라인 구축 및 전처리
- **학습 목표:** `Dataset`과 `DataLoader`를 사용하여 대용량 데이터를 효율적으로 로드하고 처리하는 방법을 학습합니다. `torchvision.transforms`를 이용한 데이터 증강(Data Augmentation)과 `TorchText`를 활용한 자연어 처리 데이터 파이프라인 구축법을 익힙니다.
- **주요 내용:**
    - [**07_Dataset_Dataloader_개요_로드.md**](./30_Data_Pipeline/07_Dataset_Dataloader_개요_로드.md): `Dataset`과 `DataLoader`의 개요와 다양한 데이터 로드 방법을 학습합니다.
    - [**08_Torchvision_Transforms_증강_최적화.md**](./30_Data_Pipeline/08_Torchvision_Transforms_증강_최적화.md): `torchvision.transforms`를 이용한 데이터 전처리 및 증강, `DataLoader`의 `num_workers`, `pin_memory`를 이용한 성능 최적화를 학습합니다.
    - [**09_현실적인_데이터_처리_불균형_데이터셋.md**](./30_Data_Pipeline/09_현실적인_데이터_처리_불균형_데이터셋.md): 대용량 데이터셋 처리와 불균형 데이터셋 처리 전략을 학습합니다.
    - [**10_TorchText_자연어_처리.md**](./30_Data_Pipeline/10_TorchText_자연어_처리.md): `TorchText`를 이용한 자연어 처리 데이터셋 구축 및 전처리 방법을 학습합니다.

### Part 4: 모델 학습, 평가 및 최적화
- **학습 목표:** 모델 학습 루프(Training Loop)를 직접 구현하고, 학습률 스케줄링, 조기 종료(Early Stopping)와 같은 고급 학습 기법을 적용하는 방법을 배웁니다. 또한, Multi-GPU를 활용한 분산 학습(DP/DDP)과 자동 혼합 정밀도(AMP)를 통해 학습 속도를 최적화하는 기술을 익힙니다.
- **주요 내용:**
    - [**11_모델_학습_루프_평가.md**](./40_Training_Optimization/11_모델_학습_루프_평가.md): 모델 학습 루프(Training Loop) 구현, `torch.no_grad()`를 이용한 평가 및 예측 방법을 학습합니다.
    - [**12_학습률_스케줄러_EarlyStopping.md**](./40_Training_Optimization/12_학습률_스케줄러_EarlyStopping.md): 다양한 학습률 스케줄러(`torch.optim.lr_scheduler`) 및 Early Stopping 구현 방법을 학습합니다.
    - [**13_Gradient_Accumulation_AMP.md**](./40_Training_Optimization/13_Gradient_Accumulation_AMP.md): Gradient Accumulation 및 자동 혼합 정밀도(AMP, `torch.cuda.amp`) 기법을 학습합니다.
    - [**14_분산_학습_DP_DDP.md**](./40_Training_Optimization/14_분산_학습_DP_DDP.md): Multi-GPU 학습 전략(DataParallel, DistributedDataParallel)을 학습합니다.
    - [**15_TorchScript_모델_최적화_기법.md**](./40_Training_Optimization/15_TorchScript_모델_최적화_기법.md): `torch.jit.script` / `torch.jit.trace`를 활용한 성능 최적화 및 양자화, 가지치기 등 모델 최적화 기법을 학습합니다.
    - [**16_고급_학습_및_정규화_전략.md**](./40_Training_Optimization/16_고급_학습_및_정규화_전략.md): 전이 학습, Fine-tuning, Dropout, L1/L2 정규화, Batch Normalization 등 고급 학습 전략 및 정규화 기법을 학습합니다.
    - [**17_고급_디버깅_및_프로파일링.md**](./40_Training_Optimization/17_고급_디버깅_및_프로파일링.md): `torch.autograd.profiler`, `PyTorch Profiler`를 활용한 성능 병목 현상 및 메모리 문제 해결 방법을 학습합니다.

### Part 5: 모델 저장, 로드 및 하이퍼파라미터 튜닝
- **학습 목표:** 학습된 모델의 상태(`state_dict`)를 저장하고 불러오는 방법과, 학습 중간 과정을 기록하는 체크포인트(Checkpoint) 관리법을 익힙니다. 자동화된 도구를 사용하여 최적의 하이퍼파라미터를 탐색하는 방법을 학습합니다.
- **주요 내용:**
    - [**18_모델_저장_로드_StateDict_체크포인트.md**](./50_Deployment_and_MLOps/18_모델_저장_로드_StateDict_체크포인트.md): `state_dict`를 이용한 모델 저장 및 로드, 체크포인트 관리 방법을 학습합니다.
    - [**19_하이퍼파라미터_튜닝_Optuna.md**](./50_Deployment_and_MLOps/19_하이퍼파라미터_튜닝_Optuna.md): Optuna, Ray Tune 등 외부 라이브러리를 이용한 하이퍼파라미터 튜닝 방법을 학습합니다.

### Part 6: 모델 배포 및 MLOps
- **학습 목표:** 학습된 PyTorch 모델을 TorchServe, ONNX와 같은 도구를 사용하여 실제 서비스 환경에 배포하는 방법을 배웁니다. TensorRT, OpenVINO를 통한 추론 최적화와 MLflow, W&B를 활용한 MLOps 파이프라인 구축 및 실험 관리의 기본 개념을 익힙니다.
- **주요 내용:**
    - [**20_모델_배포_TorchServe_ONNX.md**](./50_Deployment_and_MLOps/20_모델_배포_TorchServe_ONNX.md): TorchServe, ONNX, TorchScript를 이용한 모델 배포 방법을 학습합니다.
    - [**21_MLOps_통합_고려사항.md**](./50_Deployment_and_MLOps/21_MLOps_통합_고려사항.md): MLOps의 개념, 모델 버전 관리, 데이터/학습/배포 파이프라인 자동화 등 MLOps 통합 및 고려사항을 학습합니다.
    - [**22_MLOps_심화_실험관리_재현성.md**](./50_Deployment_and_MLOps/22_MLOps_심화_실험관리_재현성.md): MLflow, W&B를 활용한 체계적인 실험 관리, 코드/데이터/하이퍼파라미터 버전 관리를 통한 완벽한 재현성 확보 전략을 학습합니다.
    - [**23_MLOps_심화_테스트_실험관리.md**](./50_Deployment_and_MLOps/23_MLOps_심화_테스트_실험관리.md): ML 모델을 위한 테스트 및 검증 전략, 체계적인 실험 관리 및 재현성 확보(MLflow, W&B 연동)를 학습합니다.

### Part 7: TensorBoard 활용 및 생태계
- **학습 목표:** TensorBoard를 사용하여 모델의 학습 과정, 성능 지표, 계산 그래프를 시각화하고 디버깅하는 방법을 익힙니다. 또한, PyTorch Geometric (그래프), Transformers (자연어), Captum (모델 해석) 등 PyTorch의 방대한 생태계를 탐색하고 최신 기능을 학습합니다.
- **주요 내용:**
    - [**24_TensorBoard_시각화_디버깅.md**](./60_Ecosystem_Advanced/24_TensorBoard_시각화_디버깅.md): TensorBoard를 활용한 PyTorch 모델 시각화 및 디버깅(학습 모니터링, 그래프 시각화, 프로파일링) 방법을 학습합니다.
    - [**25_PyTorch_생태계_및_최신_동향.md**](./60_Ecosystem_Advanced/25_PyTorch_생태계_및_최신_동향.md): PyTorch Geometric, Transformers, Captum 등 PyTorch 생태계와 `torch.compile` 등 PyTorch 2.0의 최신 기능을 학습합니다.

### Part 8: 책임감 있는 AI 및 실전 프로젝트
- **학습 목표:** 모델의 예측을 설명하는 XAI(Explainable AI) 기법을 배우고, 모델의 공정성, 프라이버시, 견고성 등 책임감 있는 AI(Responsible AI)의 주요 개념을 학습합니다. 다양한 도메인의 실전 프로젝트 예제를 통해 문제 해결 능력을 기르고, 지속 가능한 ML 시스템 설계 원칙을 이해합니다.
- **주요 내용:**
    - [**26_책임감_있는_AI.md**](./60_Ecosystem_Advanced/26_책임감_있는_AI.md): Explainable AI (XAI) with Captum, 모델 공정성, 프라이버시 보호, 모델 견고성 등 책임감 있는 AI(Responsible AI) 개념을 학습합니다.
    - [**27_실전_프로젝트_예제_팁.md**](./60_Ecosystem_Advanced/27_실전_프로젝트_예제_팁.md): 이미지 처리, 시퀀스/텍스트 처리, 그래프 신경망, 추천 시스템, 멀티모달 모델 등 다양한 실전 프로젝트 예제와 일반적인 팁 및 전략을 학습합니다.
    - [**28_지속_가능한_ML_시스템_설계.md**](./60_Ecosystem_Advanced/28_지속_가능한_ML_시스템_설계.md): 지속 가능한 ML 시스템을 위한 설계 원칙과 트레이드오프(문제 정의, 기술 선택, 개발 속도와 안정성의 균형, 기술 부채, 미래 제언)를 학습합니다.