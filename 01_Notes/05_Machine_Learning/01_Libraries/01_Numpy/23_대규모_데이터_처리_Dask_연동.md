# Part 5: 고급 주제 및 실무 응용 - 23. 대규모 데이터 처리: Dask 연동

**학습 목표:** NumPy의 인메모리(in-memory) 처리 한계를 이해하고, RAM 용량을 초과하는 대규모 배열 데이터를 효율적으로 처리하기 위한 Dask 라이브러리의 `dask.array` 모듈을 학습합니다. Dask의 지연 연산(lazy evaluation) 개념과 분산 컴퓨팅 환경에서의 활용법을 숙달합니다.

**왜 중요한가?** 현대 데이터 과학에서는 수십 기가바이트에서 테라바이트에 이르는 대규모 데이터를 다루는 경우가 빈번합니다. NumPy는 강력하지만 모든 데이터를 컴퓨터의 RAM에 로드해야 하는 한계가 있습니다. Dask는 이러한 한계를 극복하고, 단일 머신 또는 분산 클러스터에서 NumPy와 유사한 방식으로 대규모 배열 연산을 수행할 수 있게 하여, 실제 현업에서 대용량 데이터를 효율적으로 분석하고 모델링하는 데 필수적인 실무 역량을 제공합니다.

---

### 1. 대규모 데이터 처리의 도전과제

NumPy는 고성능 수치 계산 라이브러리이지만, 기본적으로 모든 데이터를 컴퓨터의 RAM(메모리)에 로드하여 처리합니다. 이는 다음과 같은 한계를 가집니다.

-   **메모리 제약**: 데이터셋의 크기가 RAM 용량을 초과하면 `MemoryError`가 발생하여 데이터를 처리할 수 없습니다.
-   **단일 머신 한계**: 분산 환경에서의 병렬 처리를 기본적으로 지원하지 않아, 단일 CPU/GPU의 성능에 의존합니다.
-   **느린 디스크 I/O**: `np.memmap`과 같은 기술로 디스크의 데이터를 직접 매핑할 수 있지만, 디스크 I/O 속도가 CPU 연산 속도에 비해 현저히 느려 성능 병목이 발생할 수 있습니다.

이러한 문제를 해결하기 위해 "Out-of-core" (메모리 외부) 컴퓨팅 및 분산 처리 기술이 필요하며, Dask가 그 해결책 중 하나입니다.

### 2. Dask 소개 및 `dask.array`

Dask는 Python에서 병렬 컴퓨팅을 위한 유연한 라이브러리입니다. NumPy, Pandas, Scikit-learn과 같은 기존 라이브러리와 유사한 API를 제공하면서도, 대규모 데이터셋을 처리하고 분산 환경에서 연산을 수행할 수 있도록 설계되었습니다.

`dask.array`는 Dask의 핵심 모듈 중 하나로, NumPy의 `ndarray`와 매우 유사한 API를 제공합니다. 하지만 `dask.array`는 데이터를 작은 **청크(chunk)**로 나누어 관리하며, **지연 연산(lazy evaluation)** 방식을 사용합니다.

-   **청크(Chunk)**: 대규모 배열을 작은 NumPy 배열 청크들로 분할하여 메모리에 한 번에 모든 데이터를 로드할 필요가 없습니다.
-   **지연 연산(Lazy Evaluation)**: `dask.array`는 연산을 즉시 수행하지 않고, 어떤 연산을 수행해야 하는지에 대한 **계산 그래프(computation graph)**만 구축합니다. 실제 계산은 `.compute()` 메서드가 호출될 때 비로소 시작됩니다.

### 3. `dask.array` 기본 사용법

`dask.array`는 NumPy 배열과 유사하게 생성하고 조작할 수 있습니다.

-   **NumPy 배열로부터 생성**:
    ```python
    import numpy as np
    import dask.array as da

    # 작은 NumPy 배열
    x = np.arange(100).reshape(10, 10)

    # Dask 배열로 변환 (청크 크기 지정)
    # 이 시점에는 실제 계산이 일어나지 않음
    dask_x = da.from_array(x, chunks=(5, 5))
    print("Dask Array:\n", dask_x)
    # dask.array<array, shape=(10, 10), dtype=int64, chunksize=(5, 5), chunktype=numpy.ndarray>
    ```

-   **대규모 Dask 배열 생성 (가상 데이터)**:
    ```python
    # 10000x10000 크기의 Dask 배열 (실제 메모리에는 로드되지 않음)
    # 각 청크는 1000x1000 크기의 NumPy 배열로 구성
    large_dask_array = da.random.random((10000, 10000), chunks=(1000, 1000))
    print("\nLarge Dask Array:\n", large_dask_array)
    # dask.array<random_sample, shape=(10000, 10000), dtype=float64, chunksize=(1000, 1000), chunktype=numpy.ndarray>
    ```

-   **지연 연산 및 `.compute()`**:
    ```python
    # 연산은 즉시 수행되지 않고 계산 그래프만 생성됨
    result_lazy = large_dask_array.mean()
    print("\nLazy Result (Dask object):\n", result_lazy)
    # dask.array<mean_agg-aggregate, shape=(), dtype=float64, chunksize=(), chunktype=numpy.ndarray>

    # 실제 계산을 트리거하고 결과를 얻음
    actual_result = result_lazy.compute()
    print("Actual Result (computed):\n", actual_result)
    # 0.5000... (실제 계산된 평균값)
    ```

### 4. Dask와 NumPy의 차이점 및 연동

| 특징         | NumPy                               | Dask.array                            |
| :----------- | :---------------------------------- | :------------------------------------ |
| **데이터 크기** | 인메모리 (RAM에 적합)             | 인메모리 초과 가능 (디스크, 분산)     |
| **연산 방식** | 즉시 연산 (Eager Evaluation)        | 지연 연산 (Lazy Evaluation)           |
| **병렬 처리** | 단일 스레드/프로세스 (기본)         | 멀티스레드, 멀티프로세스, 분산 클러스터 |
| **API 호환성** | 표준 배열 연산                      | NumPy와 거의 동일한 API               |
| **오버헤드**   | 낮음                                | 청크 관리 및 스케줄링 오버헤드 존재   |

**연동**: `dask.array`는 NumPy와 매우 유사한 API를 제공하므로, 기존 NumPy 코드를 `dask.array`로 쉽게 전환할 수 있습니다. 많은 NumPy 함수들이 `dask.array`에서도 직접 작동합니다.

```python
# NumPy와 Dask.array 간의 변환
np_array = dask_x.compute() # Dask 배열을 NumPy 배열로 변환
dask_array_from_np = da.from_array(np_array, chunks=(5, 5)) # NumPy 배열을 Dask 배열로 변환
```

### 5. 실무 활용 시나리오

-   **대규모 이미지/과학 데이터 처리**: 수십 GB 이상의 위성 이미지, 의료 영상, 시뮬레이션 결과 등 다차원 배열 데이터의 전처리 및 분석.
-   **시계열 데이터 분석**: 수많은 센서 데이터, 금융 데이터 등 장기간에 걸친 대용량 시계열 데이터의 집계, 필터링, 변환.
-   **머신러닝 전처리**: RAM에 로드하기 어려운 크기의 데이터셋에 대한 특성 공학(Feature Engineering) 및 전처리 파이프라인 구축.
-   **분산 머신러닝**: Dask-ML과 같은 라이브러리와 연동하여 대규모 데이터셋에 대한 머신러닝 모델 학습.

### 6. 요약 및 모범 사례

-   **Dask는 NumPy의 확장**: Dask는 NumPy의 API를 유지하면서 대규모 및 분산 환경에서의 데이터 처리를 가능하게 하는 강력한 도구입니다.
-   **지연 연산의 이해**: `.compute()`가 호출될 때까지 실제 계산이 일어나지 않는다는 점을 명확히 이해하고, 불필요한 `.compute()` 호출을 피하여 계산 그래프를 최대한 크게 유지하는 것이 중요합니다.
-   **청크 크기 최적화**: 데이터와 연산의 특성에 맞는 적절한 청크 크기를 선택하는 것이 성능에 큰 영향을 미칩니다.
-   **Dask Dashboard 활용**: Dask는 연산 진행 상황과 병목 현상을 시각적으로 보여주는 대시보드를 제공합니다. 이를 활용하여 성능을 모니터링하고 최적화할 수 있습니다.

Dask를 통해 여러분은 단일 머신의 한계를 넘어, 진정한 의미의 대규모 데이터 과학 프로젝트를 수행할 수 있는 역량을 갖추게 될 것입니다.
