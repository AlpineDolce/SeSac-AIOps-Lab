<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
 Pandas의 `apply()` 메서드를 사용하여 Series 또는 DataFrame의 행/열에 사용자 정의 함수를 적용하는 방법을 숙달합니다. 벡터화된 연산으로 처리하기 어려운 복잡한 데이터 변환 및 조건부 로직을 효율적으로 구현하는 실무 역량을 강화합니다.

<h2>목차</h2>

---

### 1. `apply()` 메서드란?

`apply()` 메서드는 Pandas Series 또는 DataFrame의 축(axis)을 따라 함수를 적용하는 데 사용됩니다. 즉, 각 행 또는 각 열에 대해 지정된 함수를 한 번씩 실행합니다.

-   **언제 사용하는가?**
    -   벡터화된 NumPy/Pandas 연산으로 직접 구현하기 어려운 복잡한 사용자 정의 로직이 필요할 때.
    -   여러 컬럼의 값을 동시에 고려하여 새로운 값을 생성해야 할 때.
    -   외부 라이브러리의 함수를 Pandas 객체에 적용해야 할 때.
-   **`map()` 및 `applymap()`과의 비교 (간략히)**:
    -   `map()`: Series에만 적용되며, 각 요소에 함수를 적용합니다. (예: `Series.map(lambda x: x*2)`)
    -   `applymap()`: DataFrame에만 적용되며, 각 셀(요소)에 함수를 적용합니다. (예: `DataFrame.applymap(lambda x: x*2)`)
    -   `apply()`: Series 또는 DataFrame의 행/열 전체에 함수를 적용합니다. 가장 유연하며, 함수가 Series(행 또는 열)를 인수로 받습니다.

### 2. Series에 `apply()` 적용

Series에 `apply()`를 사용하면 각 요소에 대해 함수를 실행하는 `map()`과 유사하게 동작하지만, `apply()`는 더 복잡한 함수(예: Series를 반환하는 함수)도 처리할 수 있습니다.

```python
import pandas as pd

s = pd.Series([10, 20, 30, 40, 50])

# 람다 함수 적용
s_plus_five = s.apply(lambda x: x + 5)
print("Series + 5:\n", s_plus_five)
# 0    15
# 1    25
# 2    35
# 3    45
# 4    55
# dtype: int64

# 사용자 정의 함수 적용
def custom_transform(value):
    if value < 30:
        return "Small"
    elif value < 50:
        return "Medium"
    else:
        return "Large"

s_categorized = s.apply(custom_transform)
print("\nSeries Categorized:\n", s_categorized)
# 0     Small
# 1     Small
# 2    Medium
# 3    Medium
# 4     Large
# dtype: object
```

### 3. DataFrame에 `apply()` 적용

DataFrame에 `apply()`를 사용할 때는 `axis` 파라미터가 중요합니다.

-   **행 단위 적용 (`axis=1`)**: 함수가 각 행(Series 객체)을 인수로 받습니다.
    ```python
    df = pd.DataFrame({
        '국어': [80, 90, 70, 60],
        '영어': [90, 85, 95, 75],
        '수학': [75, 80, 85, 90]
    })
    print("Original DataFrame:\n", df)

    # 각 학생의 총점 계산 (행 단위 합계)
    df['총점'] = df.apply(lambda row: row['국어'] + row['영어'] + row['수학'], axis=1)
    print("\nDataFrame with Total Score (row-wise):\n", df)
    #    국어  영어  수학   총점
    # 0  80  90  75  245
    # 1  90  85  80  255
    # 2  70  95  85  250
    # 3  60  75  90  225

    # 여러 값을 반환하는 함수 (새로운 컬럼 생성)
    def grade_student(row):
        total = row['국어'] + row['영어'] + row['수학']
        avg = total / 3
        if avg >= 90:
            grade = 'A'
        elif avg >= 80:
            grade = 'B'
        else:
            grade = 'C'
        return pd.Series({'평균': avg, '등급': grade})

    df_grades = df.apply(grade_student, axis=1)
    df_with_grades = pd.concat([df, df_grades], axis=1)
    print("\nDataFrame with Avg and Grade:\n", df_with_grades)
    #    국어  영어  수학   총점         평균 등급
    # 0  80  90  75  245  81.666667  B
    # 1  90  85  80  255  85.000000  B
    # 2  70  95  85  250  83.333333  B
    # 3  60  75  90  225  75.000000  C
    ```

-   **열 단위 적용 (`axis=0`, 기본값)**: 함수가 각 열(Series 객체)을 인수로 받습니다.
    ```python
    # 각 과목의 평균 점수 계산 (열 단위 평균)
    df_mean_scores = df.apply(lambda col: col.mean(), axis=0)
    print("\nMean Scores per Subject (column-wise):\n", df_mean_scores)
    # 국어     75.0
    # 영어     88.75
    # 수학     82.5
    # 총점    243.75
    # dtype: float64
    ```

### 4. 성능 고려사항 및 대안

`apply()`는 매우 유연하지만, Python 루프를 내부적으로 사용하기 때문에 대규모 데이터셋에서는 성능 병목이 될 수 있습니다.

-   **벡터화된 연산 우선**: 항상 NumPy/Pandas의 내장 벡터화된 함수(예: `df['col'] * 2`, `df.sum()`, `np.where()`)를 먼저 고려하세요. 이들은 C/C++ 레벨에서 최적화되어 훨씬 빠릅니다.
-   **`df.eval()` 및 `df.query()`**: 간단한 수식이나 조건부 필터링은 이들 메서드가 `apply()`보다 훨씬 빠르고 효율적입니다.
-   **`transform()` 및 `agg()`**: `groupby()`와 함께 사용될 때 `apply()`보다 더 효율적인 대안이 될 수 있습니다. (자세한 내용은 `23_고급_그룹화_연산.md` 참고)
-   **`numba` 또는 `cython`**: 극단적인 성능 최적화가 필요한 경우, `apply()` 내에서 실행되는 사용자 정의 함수를 `numba`나 `cython`으로 컴파일하여 속도를 향상시킬 수 있습니다.

### 5. 실무 활용 시나리오

-   **복잡한 특성 공학 (Feature Engineering)**: 여러 원본 컬럼을 조합하여 새로운 파생 변수를 만들 때. (예: '이름'과 '성'을 합쳐 '풀네임' 생성, '구매액'과 '할인율'을 이용해 '최종 결제액' 계산)
-   **데이터 파싱 및 정규화**: 특정 형식의 문자열 컬럼을 파싱하여 여러 정보로 분리하거나, 복잡한 정규화 로직을 적용할 때.
-   **조건부 데이터 변환**: 여러 조건에 따라 다른 값을 할당해야 할 때 (특히 `np.select`나 `np.where`로 표현하기 복잡한 경우).
-   **외부 API 연동**: DataFrame의 각 행/열 데이터를 사용하여 외부 API를 호출하고 그 결과를 처리할 때.

### 6. 요약 및 모범 사례

`apply()` 메서드는 Pandas에서 사용자 정의 로직을 유연하게 적용할 수 있는 강력한 도구입니다.

-   **유연성**: 복잡한 데이터 변환 및 사용자 정의 함수 적용에 탁월합니다.
-   **성능 고려**: 대규모 데이터셋에서는 성능 병목이 될 수 있으므로, 항상 벡터화된 연산을 먼저 고려하고, `apply()`는 최후의 수단으로 사용하거나 성능 최적화 기법과 함께 사용하세요.
-   **가독성**: 복잡한 로직을 명확하게 표현할 수 있어 코드의 가독성을 높일 수 있습니다.

`apply()`의 장점과 한계를 이해하고 적절한 상황에 활용하는 것이 효율적인 Pandas 데이터 처리를 위한 핵심 역량입니다.
