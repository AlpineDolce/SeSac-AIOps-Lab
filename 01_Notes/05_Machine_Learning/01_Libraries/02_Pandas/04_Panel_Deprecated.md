<h1>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h1>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01 (최종 수정: 2025-08-13)

<h2>문서 목표</h2>
이 문서는 Pandas의 과거 3차원 데이터 구조였던 `Panel`에 대해 다룹니다. `Panel`의 개념과 특징을 간략히 설명하고, 현재 사용이 권장되지 않는 이유(Deprecated)와 함께 `MultiIndex DataFrame` 또는 `xarray`와 같은 현대적인 대안들을 학습합니다.

<h2>목차</h2>

- [1. Panel (3차원 데이터 - Deprecated)](#1-panel-3차원-데이터---deprecated)
  - [1.1. Panel의 개념 및 과거 역할](#11-panel의-개념-및-과거-역할)
  - [1.2. Panel의 Deprecation 이유](#12-panel의-deprecation-이유)
  - [1.3. 현대적인 대안](#13-현대적인-대안)
    - [1.3.1. MultiIndex DataFrame](#131-multiindex-dataframe)
    - [1.3.2. `xarray` 라이브러리](#132-xarray-라이브러리)

---

## 1. Panel (3차원 데이터 - Deprecated)

### 1.1. Panel의 개념 및 과거 역할

`Panel`은 과거 Pandas에서 3차원 데이터를 다루기 위해 제공했던 데이터 구조입니다. 이는 여러 개의 2차원 `DataFrame`을 하나의 객체로 묶어 관리하는 데 사용되었습니다. `Panel`은 3개의 축(Axis)을 가졌습니다.

*   **Axis 0 (items)**: 각 `DataFrame`을 구분하는 축.
*   **Axis 1 (major_axis)**: 각 `DataFrame`의 행(row)에 해당하는 축.
*   **Axis 2 (minor_axis)**: 각 `DataFrame`의 열(column)에 해당하는 축.

개념적으로 `Panel`은 여러 스프레드시트(DataFrame)를 하나의 엑셀 파일(Panel)에 시트별로 묶어 놓은 것과 유사했습니다.

```python
import pandas as pd
import numpy as np

# 경고 메시지 비활성화 (Panel 사용 시 DeprecationWarning 발생)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 과거 Panel 생성 예시 (현재는 권장되지 않음)
# 2개의 DataFrame을 포함하는 Panel 생성
data1 = np.random.rand(3, 4)
data2 = np.random.rand(3, 4)

df1 = pd.DataFrame(data1, columns=['A', 'B', 'C', 'D'], index=['r1', 'r2', 'r3'])
df2 = pd.DataFrame(data2, columns=['A', 'B', 'C', 'D'], index=['r1', 'r2', 'r3'])

# Panel 생성 (Pandas 0.25.0 이전 버전에서만 정상 작동)
# p = pd.Panel({'df1_key': df1, 'df2_key': df2})
# print("과거 Panel 객체 (개념적):\n", p)
# print("\nPanel에서 특정 DataFrame 접근 (df1_key):\n", p['df1_key'])
```
**설명:** 위 코드는 `Panel`이 과거에 어떻게 사용되었는지를 보여주기 위한 예시입니다. 현재 Pandas 버전에서는 `pd.Panel`을 직접 호출하면 `DeprecationWarning`이 발생하거나 `AttributeError`가 발생할 수 있습니다.

### 1.2. Panel의 Deprecation 이유

Pandas 0.25.0 버전부터 `Panel`은 공식적으로 **Deprecated(사용 중단)** 되었으며, 향후 버전에서는 완전히 제거될 예정입니다. 이는 `Panel`의 복잡성과 사용성의 한계 때문입니다.

*   **복잡한 인덱싱**: `Panel`은 3개의 축을 가지고 있어 데이터 접근 및 조작이 직관적이지 않고 복잡했습니다. 특히 `DataFrame`의 `loc`, `iloc`와 같은 명시적인 접근자가 없어 사용성이 떨어졌습니다.
*   **유연성 부족**: `Panel`은 모든 `DataFrame`이 동일한 `major_axis`와 `minor_axis`를 가져야 하는 제약이 있었습니다. 이는 실제 데이터의 다양한 형태를 유연하게 다루기 어렵게 만들었습니다.
*   **더 나은 대안의 등장**: `MultiIndex DataFrame`과 `xarray`와 같은 더 강력하고 유연한 대안들이 등장하면서 `Panel`의 필요성이 줄어들었습니다.

### 1.3. 현대적인 대안

3차원 이상의 데이터를 다룰 때는 `Panel` 대신 다음과 같은 현대적인 대안들을 사용하는 것이 강력히 권장됩니다.

#### 1.3.1. MultiIndex DataFrame

`MultiIndex` (계층적 인덱스) `DataFrame`은 기존 2차원 `DataFrame`에 여러 레벨의 인덱스를 사용하여 3차원 이상의 데이터를 효율적으로 표현할 수 있는 방법입니다. 이는 `Panel`이 가졌던 `items`, `major_axis`, `minor_axis` 개념을 `DataFrame`의 인덱스와 컬럼에 계층적으로 매핑하여 구현합니다.

```python
import pandas as pd
import numpy as np

# 3차원 데이터 (예: 2개 도시의 3개월간 4개 제품 판매량)
# items: 도시 (Seoul, Busan)
# major_axis: 월 (Jan, Feb, Mar)
# minor_axis: 제품 (ProdA, ProdB, ProdC, ProdD)

# 데이터 생성
data_seoul = np.random.randint(10, 100, size=(3, 4))
data_busan = np.random.randint(10, 100, size=(3, 4))

df_seoul = pd.DataFrame(data_seoul, index=['Jan', 'Feb', 'Mar'], columns=['ProdA', 'ProdB', 'ProdC', 'ProdD'])
df_busan = pd.DataFrame(data_busan, index=['Jan', 'Feb', 'Mar'], columns=['ProdA', 'ProdB', 'ProdC', 'ProdD'])

# MultiIndex DataFrame으로 변환
# pd.concat을 사용하여 여러 DataFrame을 합치고, keys로 상위 레벨 인덱스 부여
multiindex_df = pd.concat({'Seoul': df_seoul, 'Busan': df_busan}, names=['City', 'Month'])

print("MultiIndex DataFrame:\n", multiindex_df)
print("\nMultiIndex DataFrame의 인덱스:\n", multiindex_df.index)

# 데이터 접근 예시
# 서울의 1월 판매량 (DataFrame 반환)
print("\n서울의 1월 판매량:\n", multiindex_df.loc[('Seoul', 'Jan')])

# 부산의 ProdB 판매량 (Series 반환)
print("\n부산의 ProdB 판매량:\n", multiindex_df.loc['Busan', 'ProdB'])

# 모든 도시의 2월 ProdB 판매량
print("\n모든 도시의 2월 ProdB 판매량:\n", multiindex_df.loc[(slice(None), 'Feb'), 'ProdB'])
```

#### 1.3.2. `xarray` 라이브러리

`xarray`는 다차원 배열 데이터를 다루는 데 특화된 파이썬 라이브러리로, NumPy 배열에 레이블(이름, 좌표)을 붙여 다차원 데이터를 쉽게 관리할 수 있게 합니다. Pandas의 `DataFrame`과 `Series`의 개념을 N차원으로 확장한 것으로 볼 수 있으며, 기상학, 해양학, 신경과학 등 과학 데이터 분석에 널리 사용됩니다.

**주요 특징:**
*   **레이블이 있는 차원**: 각 차원에 이름과 좌표를 부여하여 데이터 접근이 직관적입니다.
*   **Pandas와 유사한 인터페이스**: `groupby`, `sel` (select by label) 등 Pandas와 유사한 메서드를 제공합니다.
*   **NetCDF, GRIB 등 과학 데이터 형식 지원**: 과학 분야에서 널리 사용되는 파일 형식을 쉽게 읽고 쓸 수 있습니다.

```python
import xarray as xr
import numpy as np
import pandas as pd

# 3차원 NumPy 배열 생성 (도시, 월, 제품)
data_3d = np.random.randint(10, 100, size=(2, 3, 4))

# 차원 이름과 좌표 정의
cities = ['Seoul', 'Busan']
months = ['Jan', 'Feb', 'Mar']
products = ['ProdA', 'ProdB', 'ProdC', 'ProdD']

# xarray.DataArray 생성
# dims: 차원 이름, coords: 각 차원의 좌표
sales_dataarray = xr.DataArray(
    data_3d,
    coords={'city': cities, 'month': months, 'product': products},
    dims=['city', 'month', 'product'],
    name='Sales'
)

print("xarray.DataArray 객체:\n", sales_dataarray)
print("\nxarray.DataArray의 차원:\n", sales_dataarray.dims)
print("\nxarray.DataArray의 좌표:\n", sales_dataarray.coords)

# 데이터 접근 예시 (레이블 기반)
# 서울의 1월 모든 제품 판매량
print("\n서울의 1월 모든 제품 판매량:\n", sales_dataarray.loc['Seoul', 'Jan', :])

# 모든 도시의 ProdB 2월 판매량
print("\n모든 도시의 ProdB 2월 판매량:\n", sales_dataarray.loc[:, 'Feb', 'ProdB'])

# 평균 계산 (특정 차원 기준)
print("\n제품별 평균 판매량:\n", sales_dataarray.mean(dim='product'))
```