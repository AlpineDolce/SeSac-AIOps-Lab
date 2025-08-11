<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 딥러닝 모델의 성능을 최적화하기 위한 하이퍼파라미터 튜닝(Hyperparameter Tuning) 방법을 상세히 다룹니다. KerasTuner의 개요, 하이퍼모델 정의(`HyperModel`), 튜너 선택 및 실행, 그리고 최적의 하이퍼파라미터 검색 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 하이퍼파라미터 튜닝 (KerasTuner)](#1-하이퍼파라미터-튜닝-kerastuner)
  - [1.1. KerasTuner 개요](#11-kerastuner-개요)
  - [1.2. 하이퍼모델 정의 (HyperModel)](#12-하이퍼모델-정의-hypermodel)
  - [1.3. 튜너 (Tuner) 선택 및 실행](#13-튜너-tuner-선택-및-실행)
  - [1.4. 최적의 하이퍼파라미터 검색](#14-최적의-하이퍼파라미터-검색)

---

## 1. 하이퍼파라미터 튜닝 (KerasTuner)

하이퍼파라미터 튜닝은 모델의 성능을 최적화하기 위해 학습률, 배치 크기, 레이어 수, 뉴런 수 등 모델의 구조나 학습 과정에 영향을 미치는 파라미터들을 탐색하는 과정입니다. KerasTuner는 Keras 모델을 위한 강력하고 사용하기 쉬운 하이퍼파라미터 튜닝 라이브러리입니다.

### 1.1. KerasTuner 개요

*   **개념:** KerasTuner는 하이퍼파라미터 탐색 공간을 정의하고, 다양한 탐색 알고리즘(RandomSearch, Hyperband, BayesianOptimization)을 사용하여 최적의 하이퍼파라미터 조합을 찾아주는 도구입니다.
*   **실무 관점:**
    *   **자동화:** 수동으로 하이퍼파라미터를 변경하며 실험하는 번거로움을 줄여줍니다.
    *   **효율성:** 체계적인 탐색 알고리즘을 통해 더 적은 시도로 더 좋은 성능의 모델을 찾을 수 있습니다.
    *   **재현성:** 튜닝 과정을 코드로 관리하여 실험의 재현성을 높입니다.
    *   **설치:** `pip install keras-tuner`

### 1.2. 하이퍼모델 정의 (HyperModel)

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

### 1.3. 튜너 (Tuner) 선택 및 실행

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

### 1.4. 최적의 하이퍼파라미터 검색

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