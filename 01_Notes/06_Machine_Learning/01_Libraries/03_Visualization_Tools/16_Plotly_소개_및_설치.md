<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 웹 기반의 인터랙티브 시각화 라이브러리 Plotly를 소개하고 설치 방법을 안내합니다. Plotly의 주요 특징과 `Plotly Express` 및 `Plotly Graph Objects`의 두 가지 주요 인터페이스를 이해하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Plotly](#1-plotly)
  - [1.1. Plotly 소개](#11-plotly-소개)
  - [1.2. 설치](#12-설치)

---

## 1. Plotly

### 1.1. Plotly 소개
Plotly는 웹 기반의 인터랙티브(interactive) 시각화를 생성하는 강력한 오픈 소스 라이브러리입니다. Matplotlib이나 Seaborn과 달리, Plotly로 생성된 그래프는 웹 브라우저에서 직접 렌더링되며, 확대/축소, 팬(pan), 데이터 포인트 정보 확인(hover), 애니메이션 등 다양한 상호작용 기능을 제공합니다. 이는 데이터 탐색, 대시보드 구축, 웹 애플리케이션에 동적인 시각화를 포함할 때 매우 유용합니다.

**주요 특징**:
*   **인터랙티브**: 사용자와 상호작용할 수 있는 동적인 그래프를 생성합니다.
*   **웹 기반**: HTML, JavaScript, CSS를 사용하여 웹 브라우저에서 렌더링됩니다.
*   **다양한 언어 지원**: 파이썬 외에도 R, MATLAB, JavaScript, Julia 등 다양한 프로그래밍 언어를 지원합니다.
*   **Plotly Express**: 고수준 API인 Plotly Express를 통해 적은 코드로 복잡한 그래프를 쉽게 생성할 수 있습니다.
*   **Dash 통합**: Plotly 기반의 웹 애플리케이션 프레임워크인 Dash와 함께 사용하여 인터랙티브 대시보드를 구축할 수 있습니다.

### 1.2. 설치
Plotly는 `pip`를 사용하여 설치할 수 있습니다. Jupyter 환경에서 인터랙티브 플롯을 사용하려면 `jupyterlab` 또는 `notebook`도 설치되어 있어야 합니다.

```bash
pip install plotly
pip install "jupyterlab>=3"
# 또는 pip install "notebook>=5.3"
```

Plotly는 두 가지 주요 인터페이스를 제공합니다:
*   **Plotly Express (`px`)**: 고수준 API로, Pandas DataFrame을 입력으로 받아 빠르게 그래프를 생성할 수 있습니다. 대부분의 일반적인 시각화 요구사항을 충족합니다.
*   **Plotly Graph Objects (`go`)**: 저수준 API로, 그래프의 모든 요소를 세밀하게 제어할 수 있는 유연성을 제공합니다. 복잡하거나 사용자 정의된 그래프를 만들 때 사용됩니다.

일반적으로 Plotly Express를 먼저 사용하고, 더 세밀한 제어가 필요할 때 Graph Objects를 활용하는 것이 좋습니다.

```python
import plotly.express as px
import plotly.graph_objects as go
```
