<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 MLOps의 실질적인 구현을 돕기 위한 ML 모델 테스트 및 검증 전략과 체계적인 실험 관리 및 재현성 확보 방법을 상세히 다룹니다. 데이터 슬라이스 기반 평가, 모델 강건성 테스트, 그리고 MLflow, Weights &amp; Biases(W&amp;B)와 같은 전문 실험 관리 도구의 활용 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 모델 배포 및 MLOps](#1-모델-배포-및-mlops)
  - [1.1. MLOps 통합 및 고려사항](#11-mlops-통합-및-고려사항)
    - [1.1.1. 모델 배포 및 MLOps 심화](#111-모델-배포-및-mlops-심화)
      - [설정 파일(YAML/JSON)을 이용한 하이퍼파라미터 관리](#설정-파일yamljson을-이용한-하이퍼파라미터-관리)
    - [1.1.2. ML 모델을 위한 테스트 및 검증 전략](#112-ml-모델을-위한-테스트-및-검증-전략)
    - [1.1.3. 체계적인 실험 관리 및 재현성 확보 (MLflow, W&amp;B 연동)](#113-체계적인-실험-관리-및-재현성-확보-mlflow-wb-연동)

---

## 1. 모델 배포 및 MLOps

### 1.1. MLOps 통합 및 고려사항

#### 1.1.1. 모델 배포 및 MLOps 심화

**개요**: MLOps의 실질적인 구현을 돕기 위해, 설정 관리와 CI/CT/CD 파이프라인의 구체적인 예시를 추가합니다.

설정 파일(YAML/JSON)을 이용한 하이퍼파라미터 관리

*   **제안 내용**: 학습률, 배치 크기, 모델 구조 등 실험에 필요한 모든 설정을 코드에서 분리하여 `config.yaml`과 같은 파일로 관리하는 패턴을 소개합니다.
*   **실무적 중요성**: 코드 변경 없이 설정 파일만 수정하여 다양한 실험을 쉽게 시도할 수 있습니다. 이는 실험의 재현성을 보장하고, 여러 실험 결과를 체계적으로 관리하는 데 필수적입니다.
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

#### 1.1.2. ML 모델을 위한 테스트 및 검증 전략

**개요**: 소프트웨어 테스트와는 다른, 머신러닝 모델의 특수성을 고려한 테스트 및 검증 방법론을 구체적으로 제시하여 모델의 신뢰도를 높입니다.

*   **제안 내용**: 전체 테스트셋의 평균 성능 지표 뒤에 숨겨진 모델의 취약점을 발견하기 위한 '데이터 슬라이스 기반 평가'와 '모델 강건성(Robustness) 테스트'의 개념과 간단한 구현 예시를 소개합니다.
*   **실무적 중요성**: 특정 사용자 그룹이나 특정 상황에서 모델이 치명적인 오류를 일으키는 것을 사전에 방지하고, 예측할 수 없는 실제 환경 변화에 더 잘 대응하는 안정적인 모델을 구축할 수 있습니다.
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

#### 1.1.3. 체계적인 실험 관리 및 재현성 확보 (MLflow, W&B 연동)

**개요**: 수십, 수백 번의 실험 이력(하이퍼파라미터, 코드 버전, 데이터셋, 성능 지표, 산출물 등)을 체계적으로 기록하고 관리하여, 완벽한 재현성을 보장하고 프로젝트의 지식을 자산화하는 방법을 추가합니다.

*   **제안 내용**: Keras 콜백을 사용하여 학습 과정을 `MLflow`나 `Weights &amp; Biases (W&amp;B)` 같은 전문 실험 관리 도구에 자동으로 로깅하는 방법을 소개합니다. 이를 통해 제공되는 대시보드에서 여러 실험 결과를 시각적으로 비교하고 분석하는 전체 워크플로우를 안내합니다.
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
