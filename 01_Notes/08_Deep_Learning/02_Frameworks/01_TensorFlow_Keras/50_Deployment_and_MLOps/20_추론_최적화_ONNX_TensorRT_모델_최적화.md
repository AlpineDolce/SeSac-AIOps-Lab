<h2>TensorFlow & Keras 핵심 개념 정리: 딥러닝 실무 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
이 문서는 학습된 딥러닝 모델의 추론 성능을 극대화하기 위한 다양한 최적화 기법을 상세히 다룹니다. ONNX/TensorRT 변환을 통한 추론 최적화와 양자화(Quantization), 가지치기(Pruning), 지식 증류(Knowledge Distillation) 등 모델 자체를 최적화하는 방법을 실제 코드 예제를 통해 학습합니다.

<h2>목차</h2>

- [1. 모델 배포 및 MLOps](#1-모델-배포-및-mlops)
  - [1.1. ONNX/TensorRT 변환을 통한 추론 최적화](#11-onnxtensorrt-변환을-통한-추론-최적화)
  - [1.2. 모델 최적화 (양자화, 가지치기, 지식 증류)](#12-모델-최적화-양자화-가지치기-지식-증류)

---

## 1. 모델 배포 및 MLOps

### 1.1. ONNX/TensorRT 변환을 통한 추론 최적화

TensorFlow 생태계 외의 다른 추론 엔진이나 하드웨어 가속기를 활용하기 위해 모델을 다른 형식으로 변환할 수 있습니다.

*   **ONNX (Open Neural Network Exchange):**
    *   **개념:** 딥러닝 모델을 표현하기 위한 개방형 표준 형식입니다. 다양한 프레임워크(PyTorch, TensorFlow, Keras 등)에서 학습된 모델을 ONNX 형식으로 변환하여 다른 프레임워크나 추론 엔진(ONNX Runtime)에서 실행할 수 있게 합니다.
    *   **실무 관점:** 모델의 프레임워크 종속성을 줄이고, 다양한 배포 환경에 유연하게 대응할 수 있게 합니다. `tf2onnx`와 같은 도구를 사용하여 TensorFlow/Keras 모델을 ONNX로 변환할 수 있습니다.
*   **TensorRT:**
    *   **개념:** NVIDIA GPU에서 딥러닝 추론을 최적화하기 위한 SDK입니다. 모델을 TensorRT 엔진으로 컴파일하여 GPU에서의 추론 성능을 극대화합니다.
    *   **실무 관점:** NVIDIA GPU 환경에서 최고 수준의 추론 성능이 요구될 때 사용됩니다. TensorFlow는 `tf.saved_model.experimental.build_signature_def`와 `tf.experimental.tensorrt.Converter`를 통해 TensorRT 통합을 지원합니다.

### 1.2. 모델 최적화 (양자화, 가지치기, 지식 증류)

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
