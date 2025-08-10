<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Pandas를 사용하여 Microsoft Excel 파일(.xlsx, .xls)을 직접 읽고 쓰는 방법을 다룹니다. Excel 파일 처리의 장점을 이해하고, `read_excel()` 함수를 이용한 파일 불러오기, `to_excel()` 메서드를 이용한 `DataFrame` 저장 방법을 실제 코드 예제를 통해 학습합니다. 특히 Excel 파일 저장 시 주요 옵션들을 함께 살펴봅니다.

<h2>목차</h2>

- [1. Excel 파일 처리](#1-excel-파일-처리)
  - [1.1. Excel 파일의 장점](#11-excel-파일의-장점)
  - [1.2. Excel 파일 읽기/쓰기 예제](#12-excel-파일-읽기쓰기-예제)

---

## 1. Excel 파일 처리

Pandas는 Microsoft Excel 파일(.xlsx, .xls)을 직접 읽고 쓰는 기능을 제공합니다. 이는 Excel을 주로 사용하는 환경에서 데이터 분석 결과를 공유하거나 데이터를 불러올 때 매우 편리합니다.

### 1.1. Excel 파일의 장점

1.  **별도 라이브러리 불필요**: `openpyxl` (xlsx), `xlrd` (xls)와 같은 백엔드 엔진이 필요하지만, Pandas 설치 시 대부분 함께 설치되므로 사용자가 별도로 COM 라이브러리나 다른 복잡한 라이브러리를 설치할 필요가 없습니다.
2.  **Pandas 직접 지원**: Pandas 내부에 `read_excel()` 및 `to_excel()` 함수가 내장되어 있어 파이썬 코드 내에서 Excel 파일을 쉽게 다룰 수 있습니다.
3.  **복잡한 데이터 구조 처리**: 여러 시트(sheet)를 가진 Excel 파일이나 특정 범위의 데이터도 유연하게 처리할 수 있습니다.

### 1.2. Excel 파일 읽기/쓰기 예제

`score.xlsx` 파일이 다음과 같다고 가정합니다:

| name | kor | eng | mat |
| :--- | :-- | :-- | :-- |
| 홍길동 | 90  | 99  | 90  |
| 임꺽정 | 80  | 98  | 70  |

```python
import pandas as pd

# Excel 파일 읽기: score.xlsx 파일을 DataFrame으로 불러오기
data = pd.read_excel("./data/score.xlsx")

# 총점 및 평균 컬럼 추가
data["total"] = data["kor"] + data["eng"] + data["mat"]
data["avg"] = data["total"] / 3
print("---" + "Excel 파일 읽기 및 계산 결과" + "---")
print(data)

# Excel 파일로 저장
# score_result1.xlsx: DataFrame의 인덱스도 함께 저장 (기본값)
data.to_excel("score_result1.xlsx")
print("\nscore_result1.xlsx 파일이 생성되었습니다 (인덱스 포함).")

# score_result2.xlsx: DataFrame의 인덱스 제외하고 저장
data.to_excel("score_result2.xlsx", index=False)
print("score_result2.xlsx 파일이 생성되었습니다 (인덱스 제외).")

# 출력 예시:
# ---" + "Excel 파일 읽기 및 계산 결과" + "---"
#   name  kor  eng  mat  total        avg
# 0  홍길동   90   99   90    279  93.000000
# 1  임꺽정   80   98   70    248  82.666667
# 
# score_result1.xlsx 파일이 생성되었습니다 (인덱스 포함).
# score_result2.xlsx 파일이 생성되었습니다 (인덱스 제외).
```

**`to_excel()` 저장 시 주요 옵션**:

*   `excel_writer`: 저장할 파일 경로 및 이름.
*   `sheet_name`: 저장할 시트의 이름. 기본값은 `'Sheet1'`.
*   `na_rep`: `NaN` (결측치) 값을 대체할 문자열.
*   `header`: 컬럼명(헤더)을 파일에 쓸지 여부. `True` (기본값) 또는 `False`.
*   `index`: DataFrame의 인덱스를 파일에 쓸지 여부. `True` (기본값) 또는 `False`.

