<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Iris 데이터셋을 활용하여 Pandas의 다양한 기능을 실전 데이터 분석에 적용하는 방법을 학습합니다. 데이터 로드부터 필드 정보 확인, 통계량 요약, 조건부 필터링, 그룹별 통계 계산 등 실제 데이터 분석 워크플로우를 따라가며 Pandas 활용 능력을 강화하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. 실전 예제: Iris 데이터셋 분석](#1-실전-예제-iris-데이터셋-분석)

---

## 1. 실전 예제: Iris 데이터셋 분석

Iris 데이터셋은 머신러닝 및 통계학에서 분류(Classification) 문제의 예시로 자주 사용되는 유명한 데이터셋입니다. 붓꽃의 세 가지 종(Setosa, Versicolor, Virginica)에 대한 꽃잎(petal)과 꽃받침(sepal)의 길이 및 너비 정보를 포함하고 있습니다.

**`data/iris.csv` 파일 구조 (예시)**:
```csv
sepal.length,sepal.width,petal.length,petal.width,variety
5.1,3.5,1.4,0.2,Setosa
4.9,3.0,1.4,0.2,Setosa
...
6.3,3.3,6.0,2.5,Virginica
```

**문제**: `data/iris.csv` 파일을 읽어서 다음 작업을 수행하세요.

1.  Iris 데이터셋의 필드(컬럼) 개수와 각 필드의 타입 확인
2.  맨 앞의 데이터 7개 출력
3.  Iris 데이터셋의 통계량 요약 정보 확인
4.  `variety`가 'Setosa'인 데이터의 통계량 출력
5.  각 `variety`별 `sepal.length` 값의 평균값 출력
6.  꽃의 종류가 'Setosa'이면서 `sepal.length`가 5cm 이상인 데이터 개수 출력

**해답**:

```python
import pandas as pd
import numpy as np

# 데이터 로드: data/iris.csv 파일을 DataFrame으로 불러오기
data = pd.read_csv("./data/iris.csv")

# 1) 필드 정보 확인: data.info()를 사용하여 컬럼 정보, Non-null 개수, 데이터 타입 확인
print("=== 1. 데이터셋 필드 정보 (data.info()) ===")
data.info()

# 2) 앞의 7개 데이터 출력: data.head(7)을 사용하여 상위 7개 행 출력
print("\n=== 2. 앞의 7개 데이터 (data.head(7)) ===")
print(data.head(7))

# 3) 통계량 요약 정보: data.describe()를 사용하여 숫자형 컬럼의 기술 통계 확인
print("\n=== 3. 통계량 요약 (data.describe()) ===")
print(data.describe())

# 4) variety가 'Setosa'인 데이터의 통계량 출력: 조건부 필터링 후 describe() 적용
print("\n=== 4. 'Setosa' 데이터 통계량 ===")
setosa_data = data[data["variety"] == 'Setosa']
print(setosa_data.describe())

# 5) 각 variety별 sepal.length 평균: groupby()와 mean()을 활용하여 그룹별 평균 계산
print("\n=== 5. 각 variety별 sepal.length 평균 ===")
# 방법 1: 각 종별로 필터링하여 평균 계산
setosa_avg = data[data["variety"] == 'Setosa']["sepal.length"].mean()
print(f"Setosa 평균 sepal.length: {setosa_avg:.2f}")

versicolor_avg = data[data["variety"] == 'Versicolor']["sepal.length"].mean()
print(f"Versicolor 평균 sepal.length: {versicolor_avg:.2f}")

virginica_avg = data[data["variety"] == 'Virginica']["sepal.length"].mean()
print(f"Virginica 평균 sepal.length: {virginica_avg:.2f}")

# 방법 2 (권장): groupby()를 사용하여 더 효율적으로 계산
print("\n--- groupby()를 이용한 각 variety별 sepal.length 평균 ---")
print(data.groupby('variety')["sepal.length"].mean())

# 6) Setosa이면서 sepal.length >= 5인 데이터 개수: 복합 조건 필터링 및 len() 사용
print("\n=== 6. 'Setosa'이면서 sepal.length >= 5인 데이터 ===")
condition_data = data[np.logical_and(data["variety"] == 'Setosa',
                                   data["sepal.length"] >= 5)]
# 또는 비트wise 연산자 사용: data[(data["variety"] == 'Setosa') & (data["sepal.length"] >= 5)]
print(f"조건을 만족하는 데이터 개수: {len(condition_data)}개")
print(condition_data)
```