<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 Keras의 `model.fit()` 고수준 API를 넘어, 더 세밀한 학습 제어가 필요할 때 활용하는 Custom Training Loop 구현 방법을 상세히 다룹니다. `tf.GradientTape`를 기반으로 학습의 모든 단계를 직접 코드로 작성하는 방법과 메모리 제약 환경에서 큰 배치 크기 효과를 내는 Gradient Accumulation 기법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 모델 학습, 평가 및 예측](#1-모델-학습-평가-및-예측)
  - [1.1. Custom Training Loop](#11-custom-training-loop)
  - [1.2. Gradient Accumulation](#12-gradient-accumulation)

---

## 1. 모델 학습, 평가 및 예측

### 1.1. Custom Training Loop

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

### 1.2. Gradient Accumulation

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
