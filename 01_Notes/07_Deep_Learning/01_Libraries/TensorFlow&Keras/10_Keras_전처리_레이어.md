<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 Keras 2.6부터 도입된 전처리 레이어의 개념과 활용 방법을 상세히 다룹니다. `TextVectorization`, `Resizing`, `Rescaling`, `RandomFlip`, `RandomRotation` 등 다양한 전처리 레이어를 사용하여 모델 내부에 데이터 전처리 로직을 포함하고, 학습 및 배포 시 일관된 전처리를 보장하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 데이터 파이프라인 구축 (`tf.data`)](#1-데이터-파이프라인-구축-tfdata)
  - [1.1. Keras 전처리 레이어 (TextVectorization, Image Augmentation 등)](#11-keras-전처리-레이어-textvectorization-image-augmentation-등)

---

## 1. 데이터 파이프라인 구축 (`tf.data`)

### 1.1. Keras 전처리 레이어 (TextVectorization, Image Augmentation 등)

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
