<h2>NumPy 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-08-07

<h2>문서 목표</h2>
이 문서는 NumPy의 핵심 성능 비결인 유니버설 함수(Universal Functions, ufuncs)와 벡터화(Vectorization)의 개념을 상세히 다룹니다. ufuncs가 어떻게 배열 연산을 효율적으로 수행하는지 이해하고, 실제 코드 예제를 통해 벡터화된 연산의 장점을 학습합니다.

<h2>목차</h2>

- [1. 유니버설 함수 (Universal Functions, ufuncs)와 벡터화](#1-유니버설-함수-universal-functions-ufuncs와-벡터화)

---

## 1. 유니버설 함수 (Universal Functions, ufuncs)와 벡터화

NumPy의 핵심 성능 비결 중 하나는 **유니버설 함수(Universal Functions, ufuncs)**입니다. ufunc는 `ndarray` 객체 내의 모든 요소에 대해 빠른 C 레벨의 연산을 수행하는 함수입니다. 사용자가 파이썬 `for` 루프를 작성할 필요 없이, 배열 전체에 대한 연산을 간결하게 표현하는 것을 **벡터화(Vectorization)**라고 하며, 이는 ufuncs를 통해 구현됩니다.

**주요 특징:**
- **성능**: 내부적으로 C로 구현된 루프를 사용하므로 순수 파이썬 코드보다 월등히 빠릅니다.
- **다양성**: 산술 연산(`np.add`, `np.multiply`), 삼각 함수(`np.sin`, `np.cos`), 통계 함수(`np.sum`, `np.mean`) 등 다양한 ufuncs가 미리 구현되어 있습니다.
- **브로드캐스팅 지원**: ufuncs는 브로드캐스팅 규칙을 자동으로 지원하여 형태가 다른 배열 간의 연산을 가능하게 합니다.

**예시:**
`+`, `*`와 같은 연산자들은 내부적으로 해당 ufunc(`np.add`, `np.multiply`)를 호출합니다.

```python
import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 요소별 덧셈 (내부적으로 np.add ufunc 호출)
sum_arr = arr1 + arr2
# 위 코드는 아래와 동일합니다.
# sum_arr = np.add(arr1, arr2)
print(f"요소별 덧셈: {sum_arr}") # [5 7 9]

# 요소별 곱셈 (내부적으로 np.multiply ufunc 호출)
mul_arr = arr1 * arr2
print(f"요소별 곱셈: {mul_arr}") # [4 10 18]

# 스칼라 연산: 배열의 모든 요소에 스칼라 값 적용
scalar_mul = arr1 * 10
print(f"스칼라 곱셈: {scalar_mul}") # [10 20 30]

# 다른 ufunc 예시 (지수 함수)
exp_arr = np.exp(arr1)
print(f"지수 함수 적용: {exp_arr}")
```
