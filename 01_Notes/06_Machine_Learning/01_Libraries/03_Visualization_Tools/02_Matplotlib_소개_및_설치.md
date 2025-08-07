<h2>데이터 시각화 도구: Matplotlib, Seaborn, Plotly 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 파이썬 시각화의 기반이 되는 Matplotlib 라이브러리를 소개하고 설치 방법을 안내합니다. Matplotlib의 주요 특징과 `Figure`, `Axes` 객체를 통한 객체 지향 API의 중요성을 이해하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Matplotlib](#1-matplotlib)
  - [1.1. Matplotlib 소개](#11-matplotlib-소개)
  - [1.2. 설치](#12-설치)

---

## 1. Matplotlib

### 1.1. Matplotlib 소개
Matplotlib은 파이썬에서 정적, 애니메이션, 인터랙티브 시각화를 생성하기 위한 포괄적인 라이브러리입니다. 100년이 넘는 시각화 역사를 기반으로 하며, 파이썬 스크립트, Jupyter Notebook, 웹 애플리케이션 서버 등 다양한 환경에서 사용할 수 있습니다. Matplotlib은 플롯의 모든 요소를 세밀하게 제어할 수 있는 '저수준(low-level)' API를 제공하여, 사용자가 원하는 대로 그래프를 커스터마이징할 수 있는 강력한 유연성을 제공합니다. 특히, Matplotlib의 핵심 구성 요소는 `Figure` (전체 그림)와 `Axes` (개별 플롯) 객체이며, 이를 통해 객체 지향적인 방식으로 플롯을 생성하고 조작할 수 있습니다. 다른 많은 파이썬 시각화 라이브러리(예: Seaborn)가 Matplotlib을 기반으로 구축되어 있습니다.

### 1.2. 설치
Matplotlib은 `pip`를 사용하여 쉽게 설치할 수 있습니다. Jupyter Notebook이나 Anaconda 환경에서는 이미 설치되어 있을 가능성이 높습니다.

```bash
pip install matplotlib
```

설치 후에는 일반적으로 `pyplot` 모듈을 `plt`라는 별칭으로 임포트하여 사용합니다.

```python
import matplotlib.pyplot as plt
```
