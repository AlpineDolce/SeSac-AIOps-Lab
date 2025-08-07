<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Pandas의 과거 3차원 데이터 구조였던 `Panel`에 대해 다룹니다. `Panel`의 개념과 특징을 간략히 설명하고, 현재 사용이 권장되지 않는 이유(Deprecated)와 함께 `MultiIndex DataFrame` 또는 `xarray`와 같은 현대적인 대안들을 학습합니다.

<h2>목차</h2>

- [1. Panel (3차원 데이터 - Deprecated)](#1-panel-3차원-데이터---deprecated)

---

## 1. Panel (3차원 데이터 - Deprecated)

1.  **과거의 3차원 구조**: Panel은 과거 Pandas에서 3차원 데이터를 다루기 위해 제공했던 구조입니다. 3개의 축(Axis 0: items, Axis 1: major_axis, Axis 2: minor_axis)을 가졌으며, Axis 0은 2차원 DataFrame에 해당하고, Axis 1은 DataFrame의 행(row), Axis 2는 DataFrame의 열(column)에 해당했습니다.
2.  **사용 중단**: **중요**: Pandas 0.25.0 버전부터 Panel은 공식적으로 **Deprecated(사용 중단)** 되었으며, 향후 버전에서는 완전히 제거될 예정입니다. 이는 Panel의 복잡성과 사용성의 한계 때문입니다.
3.  **대안**: 3차원 이상의 데이터를 다룰 때는 다음과 같은 대안을 사용하는 것이 권장됩니다:
    *   **MultiIndex (계층적 인덱스) DataFrame**: 기존 DataFrame에 여러 레벨의 인덱스를 사용하여 3차원 이상의 데이터를 2차원 형태로 효율적으로 표현할 수 있습니다. 이는 시계열 데이터나 패널 데이터 분석에 유용합니다.
    *   **`xarray` 라이브러리**: 다차원 배열 데이터를 다루는 데 특화된 라이브러리로, Pandas와 유사한 인터페이스를 제공하며 기상학, 해양학 등 과학 데이터 분석에 널리 사용됩니다. NumPy 배열에 레이블을 붙여 다차원 데이터를 쉽게 관리할 수 있게 합니다.

따라서 Panel을 사용하는 대신 위 대안들을 고려해야 합니다. 기존 코드에 Panel이 있다면 MultiIndex DataFrame으로 전환하는 것을 권장합니다.
