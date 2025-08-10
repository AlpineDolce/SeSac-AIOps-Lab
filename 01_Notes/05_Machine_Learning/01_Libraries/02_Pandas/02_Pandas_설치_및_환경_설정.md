<h2>Pandas 학습 가이드: 데이터 과학자를 위한 실무 중심 로드맵</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 Pandas 라이브러리의 설치 방법과 기본적인 환경 설정에 대해 다룹니다. Anaconda를 이용한 설치, `pip`를 이용한 설치, 그리고 발생할 수 있는 설치 문제 해결 팁을 학습합니다. 또한, Pandas를 파이썬 스크립트나 Jupyter Notebook에서 올바르게 불러와 사용하는 방법을 이해하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Pandas 설치 및 환경 설정](#1-pandas-설치-및-환경-설정)
  - [1.1. Anaconda 사용 시](#11-anaconda-사용-시)
  - [1.2. pip를 이용한 설치](#12-pip를-이용한-설치)
  - [1.3. 설치 문제 해결 팁](#13-설치-문제-해결-팁)
  - [1.4. Pandas 불러오기](#14-pandas-불러오기)

---

## 1. Pandas 설치 및 환경 설정

Pandas는 파이썬 데이터 과학 생태계의 핵심 라이브러리이므로, 대부분의 파이썬 배포판에 포함되어 있거나 쉽게 설치할 수 있습니다.

### 1.1. Anaconda 사용 시

1.  **기본 포함**: Anaconda는 데이터 과학을 위한 파이썬 배포판으로, Pandas를 포함한 대부분의 필수 라이브러리가 기본적으로 설치되어 있습니다. 따라서 Anaconda를 통해 파이썬을 설치했다면 별도의 설치 과정 없이 바로 Pandas를 사용할 수 있습니다.

### 1.2. pip를 이용한 설치

1.  **설치 명령**: Anaconda를 사용하지 않거나, 특정 환경에 Pandas를 설치해야 하는 경우 파이썬의 패키지 관리자인 `pip`를 사용하여 설치할 수 있습니다. **가상 환경(virtual environment)을 활성화한 후 설치하는 것을 권장**합니다.

    ```bash
    # 가상 환경 생성 (선택 사항이지만 권장)
    # python -m venv myenv
    # source myenv/bin/activate  # Linux/macOS
    # myenv\Scripts\activate     # Windows

    # Pandas 설치 명령
    pip install pandas
    ```

### 1.3. 설치 문제 해결 팁

1.  **`pip` 명령 작동 오류**: `pip install` 명령이 작동하지 않는 경우, 파이썬 및 `pip`가 시스템의 환경 변수(PATH)에 올바르게 등록되어 있는지 확인해야 합니다.
2.  **권한 부족 오류**: `pip install` 명령 실행 시 권한 관련 오류가 발생하면, **관리자 권한으로 명령 프롬프트(Windows) 또는 터미널(Linux/macOS)을 실행**하여 다시 시도합니다.
3.  **업그레이드**: 이미 Pandas가 설치되어 있지만 최신 버전으로 업데이트하고 싶다면 `pip install --upgrade pandas` 명령을 사용합니다.

### 1.4. Pandas 불러오기

1.  **`import` 문**: Pandas 라이브러리를 사용하기 위해서는 파이썬 스크립트나 Jupyter Notebook에서 `import` 문을 통해 불러와야 합니다. 관례적으로 `pd`라는 **별칭을 사용하여 코드를 간결하게 작성**합니다.

    ```python
    import pandas as pd

    # 이제 pd.Series, pd.DataFrame 등으로 Pandas 기능을 사용할 수 있습니다.
    ```
