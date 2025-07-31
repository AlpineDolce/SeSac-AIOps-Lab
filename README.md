# SeSac-AIOps-Lab-Portfolio (2025)

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=HTML5&logoColor=white) ![CSS](https://img.shields.io/badge/CSS-264DE4?style=for-the-badge&logo=CSS&logoColor=white) ![Javascript](https://img.shields.io/badge/Javascript-F7DF1E?style=for-the-badge&logo=Javascript&logoColor=black) ![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Git](https://img.shields.io/badge/Git-000000?style=for-the-badge&logo=git&logoColor=white) ![Linux](https://img.shields.io/badge/Linux-E95420?style=for-the-badge&logo=Linux&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white)

</div>

### Welcome to My AIOps Portfolio!

이 저장소는 [SeSac AIOps 과정](https://sesac.seoul.kr/course/active/detail.do?courseActiveSeq=2866&srchCategoryTypeCd=&courseMasterSeq=1494&currentMenuId=900002011)에서 학습한 모든 이론, 실습, 프로젝트를 체계적으로 기록한 **Living Portfolio**입니다. Python 기초부터 딥러닝 모델 구현 및 배포까지, AI 엔지니어로서의 성장 과정을 기록하고 실무 역량을 증명하는 것을 목표로 합니다.

---

## Featured Projects

주요 프로젝트에 대한 요약입니다. 각 프로젝트 폴더에서 더 상세한 코드와 문서를 확인하실 수 있습니다.

### 1. 🤖 딥러닝 이미지 분류 시스템
- **설명**: TensorFlow/Keras를 활용해 개/고양이, 꽃, 쓰레기 등 다양한 이미지를 분류하는 CNN 모델을 개발했습니다. 데이터 증강, 과적합 방지, 성능 시각화를 통해 모델을 최적화했습니다.
- **기술**: `TensorFlow`, `Keras`, `CNN`, `OpenCV`, `Matplotlib`
- **위치**: [07_Deep_Learning/](./07_Deep_Learning/)

### 2. 📊 Python & MySQL 성적 관리 시스템
- **설명**: OOP 원칙에 따라 설계된 Python 애플리케이션과 MySQL 데이터베이스를 연동한 시스템입니다. DB 커넥션 풀을 적용하여 성능을 최적화하고, 완전한 CRUD 기능을 구현했습니다.
- **기술**: `Python`, `MySQL`, `pymysql`, `DBUtils`, `OOP`
- **위치**: [02_MySQL/](./02_MySQL/)

### 3. 🌐 Django 풀스택 웹 애플리케이션
- **설명**: Django 프레임워크를 사용하여 블로그, 방명록, 성적 관리 기능이 통합된 다중 앱 웹 서비스를 구축했습니다. 사용자 인증, ORM, 템플릿 시스템 등 Django의 핵심 기능을 활용했습니다.
- **기술**: `Django`, `SQLite`, `HTML/CSS/JS`
- **위치**: [06_Web_Service_Server/](./06_Web_Service_Server/)


---

## Learning Journey & Curriculum

각 폴더는 학습 주제에 대한 **이론 정리(`organize`)**, **실습 코드(`practice`)**, **과제(`homework`)** 파일을 포함하고 있습니다.

| Module | Key Topics | Link |
| :--- | :--- | :--- |
| 🐍 **01. Python Programming** | OOP, Data Structures, File I/O, Modules, Concurrency | [Go](./01_Python/) |
| 🗄️ **02. Database (MySQL)** | Advanced SQL, Python Integration, Optimization, Connection Pool | [Go](./02_SQL/) |
| 🛠️ **03. Dev Tools & Practices** | Git, GitHub, Clean Code, Refactoring, Docker | [Go](./03_Dev/) |
| 🧮 **04. Algorithm & Data Structures** | Complexity, Sort/Search, Dynamic Programming, Graph | [Go](./04_Algorithm/) |
| 🌐 **05. Full-Stack Web Development** | Django, REST API, ORM, Templating, User Authentication | [Go](./05_Full_Stack/) |
| 📊 **06. Machine Learning** | Classification, Regression, Clustering, Feature Engineering, Model Evaluation | [Go](./06_Machine_Learning/) |
| 🧠 **07. Deep Learning** | DNN, CNN, RNN, Image Classification (Fashion-MNIST, CIFAR-10, Flowers) | [Go](./07_Deep_Learning/) |
| 🔥 **08. PyTorch** | PyTorch Fundamentals, Tensor Operations, Custom Model Implementation | [Go](./08_Pytorch/) |

---

## 🛠️ Tech Stack & Environment Setup

<details>
<summary>👉 클릭하여 기술 스택 및 환경 설정 가이드를 확인하세요.</summary>

### Tech Stack

-   **Languages**: `Python 3.11`
-   **AI / ML / DL**: `TensorFlow`, `Keras`, `PyTorch`, `Scikit-learn`, `OpenCV`, `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
-   **Web Framework**: `Django 5.0+`
-   **Database**: `MySQL`, `SQLAlchemy`, `pymysql`
-   **Tools & Etc**: `Git`, `GitHub`, `Docker`, `Conda`, `Jupyter Notebook`, `VSCode`

### Environment Setup Guide

**1. Conda 환경 생성 (권장)**
```bash
# 'sesac_ai' 이름으로 Python 3.11 환경 생성
conda create -n sesac_ai python=3.11

# 환경 활성화
conda activate sesac_ai
```

**2. 필수 패키지 설치**
```bash
# (루트 디렉터리에 requirements.txt가 있다고 가정)
pip install -r requirements.txt
```

**3. GPU 지원 (선택 사항)**
```bash
# PyTorch (CUDA 12.1 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# TensorFlow
# CUDA 및 cuDNN 버전에 맞는 TensorFlow 버전을 설치해야 합니다.
# (자세한 내용은 TensorFlow 공식 문서를 참고하세요.)
```

</details>

---

## 📫 Contact

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AlpineDolce)  [![GitHub](https://img.shields.io/badge/Velog-20C997?style=for-the-badge&logo=Velog&logoColor=white)](https://velog.io/@kts980309/posts)  
</div>

<div align="center">
  
  **© 2025 SeSac AIOps Lab. All Rights Reserved.**
  
</div>