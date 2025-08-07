<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 Keras Applications를 활용한 사전 학습된 모델(Pre-trained Models)의 개념과 전이 학습(Transfer Learning) 전략을 상세히 다룹니다. ImageNet으로 학습된 인기 있는 모델 아키텍처들을 사용하여 새로운 컴퓨터 비전 태스크의 성능을 향상시키고 학습 시간을 단축하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 모델 아키텍처 설계](#1-모델-아키텍처-설계)
  - [1.1. Keras Applications (사전 학습된 모델)](#11-keras-applications-사전-학습된-모델)

---

## 1. 모델 아키텍처 설계

### 1.1. Keras Applications (사전 학습된 모델)

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
