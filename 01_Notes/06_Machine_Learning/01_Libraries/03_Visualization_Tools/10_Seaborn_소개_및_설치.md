<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Matplotlib을 기반으로 하는 통계 시각화 라이브러리 Seaborn을 소개하고 설치 방법을 안내합니다. Seaborn의 주요 특징과 Pandas DataFrame과의 통합, 그리고 내장 데이터셋을 활용하는 방법을 이해하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Seaborn](#1-seaborn)
  - [1.1. Seaborn 소개](#11-seaborn-소개)
  - [1.2. 설치](#12-설치)

---

## 1. Seaborn

### 1.1. Seaborn 소개
Seaborn은 Matplotlib을 기반으로 하는 파이썬 데이터 시각화 라이브러리입니다. 통계 그래프를 그리는 데 특화되어 있으며, Matplotlib보다 적은 코드로도 미려하고 정보가 풍부한 그래프를 쉽게 생성할 수 있도록 고수준(high-level) API를 제공합니다. Seaborn은 데이터프레임과 같은 Pandas 데이터 구조와 잘 통합되어 있어, 복잡한 데이터셋의 관계와 분포를 탐색하는 데 매우 유용합니다.

**주요 특징**:
*   **통계적 시각화**: 데이터셋의 통계적 관계를 시각화하는 데 중점을 둡니다.
*   **아름다운 기본 스타일**: Matplotlib보다 더 세련되고 전문적인 기본 플롯 스타일을 제공합니다.
*   **Pandas DataFrame 통합**: Pandas DataFrame을 직접 입력으로 받아 처리하기 용이합니다.
*   **복잡한 플롯 유형**: 분포, 관계, 범주형 데이터, 회귀 분석 등을 위한 다양한 고급 플롯 유형을 제공합니다.

### 1.2. 설치
Seaborn은 `pip`를 사용하여 설치할 수 있습니다. Matplotlib이 필요하므로, Seaborn을 설치하면 Matplotlib도 함께 설치되거나 이미 설치되어 있어야 합니다.

```bash
pip install seaborn
```

설치 후에는 일반적으로 `sns`라는 별칭으로 임포트하여 사용합니다.

```python
import seaborn as sns
import matplotlib.pyplot as plt # Seaborn은 Matplotlib 기반이므로 함께 사용
```

Seaborn은 자체적으로 몇 가지 내장 데이터셋을 제공하여 예제를 쉽게 실행해 볼 수 있습니다. 예를 들어, `sns.load_dataset('tips')`를 사용하여 팁 데이터셋을 로드할 수 있습니다.
