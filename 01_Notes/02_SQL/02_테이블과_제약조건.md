<h2>SQL 핵심 문법: 테이블, 제약조건, 정규화 (실무 활용 중심)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-16

<h2>문서 목표</h2>
<p>이 문서는 데이터베이스의 구조를 정의하고 관리하는 DDL(Data Definition Language) 명령어와, 데이터를 저장할 때 필요한 다양한 데이터 타입에 대해 심도 있게 다룹니다. 각 개념의 정의, 실제 코드에서의 활용법, 그리고 <strong>데이터 분석 및 AI 실무에서 발생할 수 있는 주의사항과 활용 팁</strong>을 상세한 예제와 함께 설명하여, SQL을 활용한 데이터베이스 설계 및 관리의 견고한 기초를 다지는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 데이터 모델링의 이해: 데이터베이스 설계의 첫걸음](#1-데이터-모델링의-이해-데이터베이스-설계의-첫걸음)
  - [1.1. 데이터 모델링이란?](#11-데이터-모델링이란)
  - [1.2. 데이터 모델의 3단계 (개념적 → 논리적 → 물리적)](#12-데이터-모델의-3단계-개념적--논리적--물리적)
  - [1.3. 엔터티-관계 다이어그램 (ERD)의 기초](#13-엔터티-관계-다이어그램-erd의-기초)
- [2. 데이터 타입: 데이터의 형태 정의하기](#2-데이터-타입-데이터의-형태-정의하기)
  - [2.1. 숫자형 타입 (INT, DECIMAL, FLOAT 등)](#21-숫자형-타입-int-decimal-float-등)
  - [2.2. 문자형 타입 (VARCHAR, CHAR, TEXT 등)](#22-문자형-타입-varchar-char-text-등)
  - [2.3. 날짜/시간 타입 (DATE, DATETIME, TIMESTAMP 등)](#23-날짜시간-타입-date-datetime-timestamp-등)
  - [2.4. 이진 데이터 타입 (BINARY, VARBINARY, BLOB)](#24-이진-데이터-타입-binary-varbinary-blob)
  - [2.5. 특수 목적 타입 (BOOLEAN, ENUM, SET, JSON, GEOMETRY)](#25-특수-목적-타입-boolean-enum-set-json-geometry)
  - [2.6. 데이터 타입 변환 (Type Conversion)](#26-데이터-타입-변환-type-conversion)
- [3. DDL 실습: 첫 데이터베이스와 테이블 만들기](#3-ddl-실습-첫-데이터베이스와-테이블-만들기)
  - [3.1. 데이터베이스 생성 및 선택 (`CREATE DATABASE`, `USE`)](#31-데이터베이스-생성-및-선택-create-database-use)
  - [3.2. 테이블 생성 (`CREATE TABLE`)](#32-테이블-생성-create-table)
  - [3.3. 테이블 구조 확인 (`DESCRIBE`, `SHOW TABLES`, `SHOW CREATE TABLE`)](#33-테이블-구조-확인-describe-show-tables-show-create-table)
  - [3.4. 테이블 수정 (`ALTER TABLE`)](#34-테이블-수정-alter-table)
  - [3.5. 테이블 삭제 (`DROP TABLE`)](#35-테이블-삭제-drop-table)
  - [3.6. 임시 테이블 (Temporary Table) 활용](#36-임시-테이블-temporary-table-활용)
- [4. 데이터 무결성: 제약조건(Constraints) 활용](#4-데이터-무결성-제약조건constraints-활용)
  - [4.1. 제약조건의 개념과 중요성](#41-제약조건의-개념과-중요성)
  - [4.2. `PRIMARY KEY`: 기본 키](#42-primary-key-기본-키)
    - [4.2.1. 대리 키(Surrogate Key) vs 자연 키(Natural Key)](#421-대리-키surrogate-key-vs-자연-키natural-key)
  - [4.3. `FOREIGN KEY`: 외래 키 (참조 무결성)](#43-foreign-key-외래-키-참조-무결성)
    - [4.3.1. `ON DELETE` 및 `ON UPDATE` 옵션](#431-on-delete-및-on-update-옵션)
  - [4.4. `UNIQUE`: 고유 제약조건](#44-unique-고유-제약조건)
  - [4.5. `NOT NULL`: 필수 값](#45-not-null-필수-값)
  - [4.6. `DEFAULT`: 기본 값](#46-default-기본-값)
  - [4.7. `CHECK`: 값 범위/조건 검사](#47-check-값-범위조건-검사)
  - [4.8. 제약조건 테스트 및 관리](#48-제약조건-테스트-및-관리)
    - [4.8.1. 제약조건 위반 시 동작 확인](#481-제약조건-위반-시-동작-확인)
    - [4.8.2. 기존 테이블에 제약조건 추가/삭제](#482-기존-테이블에-제약조건-추가삭제)
- [5. 데이터베이스 설계 기초: 정규화(Normalization)](#5-데이터베이스-설계-기초-정규화normalization)
  - [5.1. 정규화의 필요성 및 목표](#51-정규화의-필요성-및-목표)
  - [5.2. 함수적 종속성 (Functional Dependency)](#52-함수적-종속성-functional-dependency)
  - [5.3. 이상(Anomaly) 현상: 삽입, 삭제, 갱신 이상](#53-이상anomaly-현상-삽입-삭제-갱신-이상)
  - [5.4. 정규형(Normal Form)의 종류](#54-정규형normal-form의-종류)
  - [5.5. 분석을 위한 데이터 모델링: 스타 스키마(Star Schema)](#55-분석을-위한-데이터-모델링-스타-스키마star-schema)
  - [5.6. 정규화/비정규화 실용적 예시 (주문 데이터베이스 설계)](#56-정규화비정규화-실용적-예시-주문-데이터베이스-설계)

---

## 1. 데이터 모델링의 이해: 데이터베이스 설계의 첫걸음

`CREATE TABLE` 문을 작성하여 물리적인 테이블을 만들기 전, 데이터베이스 설계자는 반드시 **데이터 모델링(Data Modeling)**이라는 중요한 설계 단계를 거칩니다. 데이터 모델링은 현실 세계의 복잡한 비즈니스 요구사항을 분석하여, 데이터베이스에 어떻게 저장할지 그 구조를 체계적으로 정의하고 시각화하는 과정입니다. 즉, 데이터베이스의 **청사진(Blueprint)**을 만드는 작업이며, 이 청사진의 품질이 전체 시스템의 안정성, 확장성, 성능을 좌우합니다.

### 1.1. 데이터 모델링이란?

데이터 모델링은 비즈니스에서 사용되는 데이터 객체(Data Objects)와 그 객체들 간의 관계, 그리고 데이터에 적용되어야 할 규칙들을 식별하고 명세하는 과정입니다. 잘 된 데이터 모델링은 다음과 같은 이점을 제공합니다.

*   **명확성:** 데이터의 구조와 관계를 시각적으로 명확하게 표현하여, 개발자, 데이터 분석가, 비즈니스 담당자 등 모든 이해관계자가 시스템에 대해 동일한 그림을 보고 소통할 수 있게 합니다.
*   **안정성 및 일관성:** 데이터의 중복을 방지하고, 데이터 간의 불일치 가능성을 줄여 데이터 무결성을 보장하는 안정적인 구조를 만듭니다.
*   **유연성 및 확장성:** 향후 비즈니스 요구사항 변경 시, 데이터베이스 구조를 더 쉽게 수정하고 확장할 수 있도록 돕습니다.

### 1.2. 데이터 모델의 3단계 (개념적 → 논리적 → 물리적)

데이터 모델링은 추상적인 아이디어에서 구체적인 구현으로 나아가는 3단계 과정을 통해 이루어집니다.

1.  **개념적 데이터 모델 (Conceptual Data Model):**
    *   **목표:** 비즈니스의 핵심 개념과 규칙을 이해하고 표현합니다. "우리 비즈니스에 가장 중요한 데이터는 무엇이며, 그들 간의 관계는 어떻게 되는가?"에 집중합니다.
    *   **특징:** 기술과 독립적인 가장 추상적인 단계의 모델입니다. 핵심적인 **엔터티(Entity)**와 그들 간의 관계만으로 표현되며, 속성(Attribute)이나 기본 키(Primary Key) 등 세부 사항은 포함하지 않습니다.
    *   **예시:** "고객(Customer)은 주문(Order)을 한다", "주문(Order)은 여러 상품(Product)을 포함한다" 와 같이 비즈니스 관점의 관계를 단순한 다이어그램으로 표현합니다.

2.  **논리적 데이터 모델 (Logical Data Model):**
    *   **목표:** 개념적 모델을 기반으로, 데이터베이스에 저장될 데이터의 구조를 논리적으로 상세하게 설계합니다. 특정 데이터베이스 기술(MySQL, Oracle 등)에 종속되지 않은, 표준화된 데이터 구조를 정의합니다.
    *   **특징:** 모든 엔터티, 속성, 기본 키, 외래 키, 그리고 엔터티 간의 관계를 상세하게 정의합니다. **정규화(Normalization)** 과정이 이 단계에서 집중적으로 수행됩니다.
    *   **예시:** 상세한 **ERD(Entity-Relationship Diagram)**를 사용하여 각 테이블의 모든 컬럼과 데이터 타입(논리적 타입), 기본 키(PK), 외래 키(FK) 등을 명시합니다.

3.  **물리적 데이터 모델 (Physical Data Model):**
    *   **목표:** 논리적 모델을 특정 데이터베이스 관리 시스템(DBMS)에 맞게 실제로 구현할 수 있는 형태로 변환합니다.
    *   **특징:** 실제 사용할 테이블명, 컬럼명, 데이터 타입(예: `VARCHAR(100)`, `INT`), 제약조건 등을 구체적으로 정의합니다. 또한, 성능 향상을 위한 인덱스, 파티셔닝 등 물리적인 요소까지 포함합니다.
    *   **예시:** 최종적으로 데이터베이스에 실행될 `CREATE TABLE` 스크립트 그 자체입니다.

### 1.3. 엔터티-관계 다이어그램 (ERD)의 기초

ERD(Entity-Relationship Diagram)는 논리적 데이터 모델링 단계에서 데이터의 구조를 시각적으로 표현하는 데 가장 널리 사용되는 도구입니다.

*   **주요 구성 요소:**
    *   **엔터티 (Entity):** 저장하고자 하는 데이터의 대상입니다. 명사형으로 표현되며, 물리적 모델에서는 **테이블**에 해당합니다. (예: `고객`, `상품`, `주문`)
    *   **속성 (Attribute):** 엔터티가 가지는 특성이나 정보입니다. 물리적 모델에서는 **컬럼**에 해당합니다. (예: `고객` 엔터티의 `이름`, `이메일` 속성)
    *   **관계 (Relationship):** 엔터티 간의 연관성이나 상호작용을 의미합니다. 동사형으로 표현되며, 물리적 모델에서는 주로 **외래 키(Foreign Key)**로 구현됩니다. (예: `고객`은 `주문`을 '한다')

*   **관계 차수와 까마귀발 표기법 (Cardinality and Crow's Foot Notation):**
    관계 차수(Cardinality)는 한 엔터티의 인스턴스가 다른 엔터티의 인스턴스와 관계를 맺는 수를 나타냅니다. ERD에서는 주로 **까마귀발 표기법(Crow's Foot Notation)**을 사용하여 이를 표현합니다.

    *   **기호의 의미:**
        *   `|` (선 하나): **One** (하나)
        *   `O` (원): **Zero** (0)
        *   `<` (까마귀발): **Many** (다수)

    *   **주요 관계 유형:**
        1.  **일대일 (One-to-One):** `( |---|| )`
            *   엔터티 A의 한 인스턴스가 엔터티 B의 한 인스턴스와만 관계를 맺습니다. (예: `사용자`와 `사용자_프로필`)
        2.  **일대다 (One-to-Many):** `( |---|< )`
            *   엔터티 A의 한 인스턴스가 엔터티 B의 여러 인스턴스와 관계를 맺을 수 있습니다. 가장 흔한 관계 유형입니다. (예: 하나의 `고객`은 여러 `주문`을 할 수 있다)
        3.  **다대다 (Many-to-Many):** `( >---< )`
            *   엔터티 A의 여러 인스턴스가 엔터티 B의 여러 인스턴스와 관계를 맺을 수 있습니다. (예: 하나의 `학생`은 여러 `과목`을 수강하고, 하나의 `과목`은 여러 `학생`이 수강한다)
            *   **중요:** 다대다 관계는 관계형 데이터베이스에서 직접 구현할 수 없습니다. 따라서 두 엔터티의 기본 키를 외래 키로 갖는 **연결 테이블(Junction Table)** 또는 **연관 엔터티(Associative Entity)**를 만들어, 두 개의 일대다 관계로 분해해야 합니다. (예: `학생`과 `과목` 사이에 `수강_신청` 테이블 생성)

---

## 2. 데이터 타입: 데이터의 형태 정의하기

데이터 타입은 데이터베이스 컬럼에 저장될 값의 종류와 형식을 정의합니다. 올바른 데이터 타입 선택은 데이터의 무결성, 저장 공간 효율성, 쿼리 성능에 큰 영향을 미칩니다. 단순히 데이터를 저장하는 것을 넘어, **저장 공간의 효율성, 쿼리 성능, 그리고 데이터의 정확성**을 고려하여 가장 적절한 타입을 선택하는 것이 중요합니다.

### 2.1. 숫자형 타입 (INT, DECIMAL, FLOAT 등)

숫자 데이터를 저장하는 데 사용됩니다. 정수, 고정 소수점, 부동 소수점 등 다양한 종류가 있습니다.

| 타입 | 설명 | 범위 (MySQL 기준) | 사용 예시 | 실무적 고려사항 |
| :--- | :--- | :--- | :--- | :--- |
| `TINYINT` | 매우 작은 정수 | -128 ~ 127 (signed), 0 ~ 255 (unsigned) | 나이, 상태 코드 (예: 0=비활성, 1=활성) | 공간 효율적. `BOOLEAN` 대용으로 사용 가능. |
| `SMALLINT` | 작은 정수 | -32768 ~ 32767 (signed) | 상품 재고, 투표 수 | `TINYINT`보다 큰 범위가 필요할 때. |
| `MEDIUMINT` | 중간 크기 정수 | -8388608 ~ 8388607 (signed) | | `INT`로 충분한 경우가 많아 잘 사용되지 않음. |
| `INT` | 일반적인 정수 | -2147483648 ~ 2147483647 (signed) | 사용자 ID, 주문 수량, 조회수 | 가장 흔하게 사용되는 정수 타입. |
| `BIGINT` | 큰 정수 | -9223372036854775808 ~ 9223372036854775807 (signed) | 매우 큰 ID (예: Snowflake ID), 총 판매액, 로그 ID | `INT` 범위를 초과하는 대규모 숫자 데이터에 사용. |
| `DECIMAL(M, D)` | 고정 소수점 숫자. **정확한 금융 계산에 필수적.** | `M`은 1~65, `D`는 0~30 | 가격, 통화, 정밀한 측정값 (예: 환율, 평점) | **부동 소수점 오차 없음.** 금융, 회계 데이터에 반드시 사용. |
| `FLOAT` / `DOUBLE` | 부동 소수점 숫자. 근사치 저장. **오차 발생 가능성 있음.** | 데이터 타입에 따라 다름 | 과학 계산, GPS 좌표, 근사치가 허용되는 측정값 | 정확성보다 속도가 중요할 때 사용. 등호(=) 비교 주의. |

**[심층 비교] `DECIMAL` vs `FLOAT`/`DOUBLE`**
*   **`DECIMAL`:** 금융 데이터나 정확한 계산이 필요한 경우에는 `DECIMAL`을 사용해야 합니다. `DECIMAL`은 숫자를 10진수로 정확하게 저장하고 연산하므로, 부동 소수점 연산에서 발생하는 오차를 방지할 수 있습니다. (예: 돈, 환율, 세금)
*   **`FLOAT` / `DOUBLE`:** `FLOAT`나 `DOUBLE`은 부동 소수점 오차를 가질 수 있습니다. 따라서 정확성이 덜 중요하고 속도가 더 중요한 과학 계산, 그래픽 처리, 물리 시뮬레이션 등 속도가 중요하고 약간의 오차가 허용되는 분야에 적합합니다. (예: 센서 데이터, GPS 좌표)

**코드 예시:**
```sql
CREATE TABLE example_numeric (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age TINYINT UNSIGNED,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INT UNSIGNED NOT NULL,
    avg_rating FLOAT
);

INSERT INTO example_numeric
  (age, price, stock_quantity, avg_rating)
VALUES
  (35, 19.99, 100, 4.5),
  (25, 250.00, 50, 4.8);
```

### 2.2. 문자형 타입 (VARCHAR, CHAR, TEXT 등)

문자열 데이터를 저장하는 데 사용됩니다. 길이, 저장 방식, 인덱싱 효율성 등에서 차이가 있습니다.

| 타입 | 설명 | 최대 길이 (MySQL 기준) | 사용 예시 | 실무적 고려사항 |
| :--- | :--- | :--- | :--- | :--- |
| `CHAR(L)` | 고정 길이 문자열. 항상 `L`만큼의 공간을 차지. | 255 | 성별(`CHAR(1)`), 우편번호(`CHAR(5)`) | 접근 속도가 빠를 수 있으나 공간 낭비 가능성. |
| `VARCHAR(L)` | 가변 길이 문자열. 실제 길이에 따라 공간 차지. | 65535 | 이름, 주소, 제목, 상품명 | 저장 공간 효율이 좋음. 대부분의 경우에 적합. |
| `TINYTEXT` | 매우 작은 텍스트 | 255 바이트 | 짧은 메모, 댓글 | `VARCHAR`보다 긴 문자열에 사용. |
| `TEXT` | 작은 텍스트 | 65535 바이트 | 긴 설명, 블로그 본문, 상품 상세 설명 | `VARCHAR`의 최대 길이를 초과하는 텍스트에 사용. |
| `MEDIUMTEXT` | 중간 크기 텍스트 | 16MB | 긴 문서, 책 내용 | |
| `LONGTEXT` | 매우 큰 텍스트 | 4GB | 매우 긴 문서, 대규모 로그 데이터 | |

**[심층 비교] `CHAR` vs `VARCHAR`**
*   **`CHAR(L)`:** 길이가 항상 고정된 데이터에 적합합니다. (예: MD5 해시값(32자), 국가 코드(2자)). `VARCHAR`보다 업데이트 시 성능 이점이 있을 수 있으나, 길이가 가변적인 데이터에 사용하면 심각한 공간 낭비를 초래합니다.
*   **`VARCHAR(L)`:** 길이가 가변적인 대부분의 문자열 데이터에 적합합니다. 저장 공간을 효율적으로 사용하므로 실무에서 가장 널리 쓰입니다.

**코드 예시:**
```sql
CREATE TABLE example_string (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_name VARCHAR(100) NOT NULL,
    country_code CHAR(2) NOT NULL,
    memo TINYTEXT,
    article TEXT
);

INSERT INTO example_string
  (user_name, country_code, memo, article)
VALUES
  ('John Doe', 'US', 'VIP customer', 'This is a long article text...');
```

### 2.3. 날짜/시간 타입 (DATE, DATETIME, TIMESTAMP 등)

날짜와 시간 정보를 저장하는 데 사용됩니다. 타임존 처리 여부가 가장 중요한 선택 기준입니다.

| 타입 | 설명 | 범위 (MySQL 기준) | 사용 예시 | 실무적 고려사항 |
| :--- | :--- | :--- | :--- | :--- |
| `DATE` | 날짜 (년, 월, 일) | '1000-01-01' ~ '9999-12-31' | 생년월일, 주문일, 고용일 | 시간 정보가 필요 없을 때 사용. |
| `TIME` | 시간 (시, 분, 초) | '-838:59:59' ~ '838:59:59' | 근무 시작 시간, 회의 시간 | 날짜 정보가 필요 없을 때 사용. |
| `YEAR` | 연도 | 1901 ~ 2155 | 출판 연도, 제조 연도 | 연도 정보만 필요할 때 사용. |
| `DATETIME` | 날짜와 시간. 타임존 정보 없음. | '1000-01-01 00:00:00' ~ '9999-12-31 23:59:59' | 이벤트 시작 시간, 예약 시간 | 입력된 값을 그대로 저장. 타임존과 무관한 절대 시점. |
| `TIMESTAMP` | 날짜와 시간. UTC 기준으로 저장 및 변환. | '1970-01-01 00:00:01' UTC ~ '2038-01-19 03:14:07' UTC | 마지막 수정일시, 로그 기록 시간 | 글로벌 서비스에 적합. 2038년 문제 고려 필요. |

**[심층 비교] `DATETIME` vs `TIMESTAMP`**
*   **`DATETIME`:** 타임존 정보 없이 입력된 값을 그대로 저장하므로, 사용자의 생일이나 특정 이벤트 예약 시간처럼 **절대적인 시점**을 기록할 때 사용합니다.
*   **`TIMESTAMP`:** 데이터를 저장할 때 UTC로 변환하고, 조회 시 현재 세션의 타임존에 맞춰 보여줍니다. 여러 국가의 사용자가 이용하는 글로벌 서비스에서 **동일한 순간을 기록**할 때 (예: 게시물 작성 시각) 매우 유용합니다. 단, 2038년 이후의 날짜를 다룰 수 없는 한계가 있습니다.

**코드 예시:**
```sql
CREATE TABLE example_datetime (
    id INT AUTO_INCREMENT PRIMARY KEY,
    birth_date DATE,
    meeting_time TIME,
    event_year YEAR,
    reservation_datetime DATETIME,
    log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

INSERT INTO example_datetime
  (birth_date, meeting_time, event_year, reservation_datetime)
VALUES
  ('1990-05-20', '10:30:00', '2025', '2025-12-25 19:00:00');
```

### 2.4. 이진 데이터 타입 (BINARY, VARBINARY, BLOB)

문자셋과 관계없이 순수한 바이트(byte) 데이터를 저장할 때 사용합니다.

| 타입 | 설명 | 최대 길이 (MySQL 기준) | 사용 예시 | 실무적 고려사항 |
| :--- | :--- | :--- | :--- | :--- |
| `BINARY(L)` | 고정 길이 이진 데이터. | 255 | 암호화 해시(SHA-256), UUID | `CHAR`의 이진 데이터 버전. |
| `VARBINARY(L)` | 가변 길이 이진 데이터. | 65535 | 가변 길이 암호화 키 | `VARCHAR`의 이진 데이터 버전. |
| `TINYBLOB` | 작은 이진 데이터 | 255 바이트 | 썸네일 이미지 | |
| `BLOB` | 일반 이진 데이터 | 65535 바이트 | 작은 파일, 이미지 | DB 외부 저장을 강력히 권장. |
| `MEDIUMBLOB` | 중간 크기 이진 데이터 | 16MB | 오디오 파일 | |
| `LONGBLOB` | 매우 큰 이진 데이터 | 4GB | 비디오 파일 | |

**코드 예시:**
```sql
CREATE TABLE example_binary (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_uuid BINARY(16) NOT NULL, -- UUID는 16바이트
    encrypted_key VARBINARY(256)
);

INSERT INTO example_binary
  (user_uuid, encrypted_key)
VALUES
  (UUID_TO_BIN(UUID()), AES_ENCRYPT('my_secret_key', 'password'));
```

### 2.5. 특수 목적 타입 (BOOLEAN, ENUM, SET, JSON, GEOMETRY)

| 타입 | 설명 | 사용 예시 | 실무적 고려사항 |
| :--- | :--- | :--- | :--- |
| `BOOLEAN` | 참/거짓. MySQL에서는 `TINYINT(1)`로 처리. | 활성화 여부, 로그인 상태 | 0(False) 또는 1(True)로 값을 다룸. |
| `ENUM` | 미리 정의된 목록 중 **하나만** 선택. | 상품 카테고리, 주문 상태 | 목록 변경이 어려워 유연성 부족. 참조 테이블 사용 권장. |
| `SET` | 미리 정의된 목록 중 **여러 개** 선택 가능. | 사용자 취미, 게시물 태그 | `ENUM`과 단점 공유. 비트 연산으로 검색 가능. |
| `JSON` | JSON 형식의 데이터를 저장. | 유연한 스키마의 설정값, 이벤트 로그 | 전용 함수로 내부 값 조회/인덱싱 가능. (MySQL 5.7+) |
| `GEOMETRY` | 위치, 경로, 지역 등 공간 데이터 저장. | 매장 좌표, 배송 경로 | 공간 인덱스와 함께 사용해야 효율적. |

**코드 예시:**
```sql
CREATE TABLE example_special (
    id INT AUTO_INCREMENT PRIMARY KEY,
    is_active BOOLEAN DEFAULT TRUE,
    product_type ENUM('Electronics', 'Books', 'Clothing') NOT NULL,
    tags SET('New', 'Best-seller', 'On-sale'),
    attributes JSON,
    store_location POINT NOT NULL,
    SPATIAL INDEX(store_location)
);

INSERT INTO example_special
  (is_active, product_type, tags, attributes, store_location)
VALUES
  (TRUE, 'Electronics', 'New,Best-seller', '{"color": "black", "size": "15-inch"}', ST_GeomFromText('POINT(127.0276 37.4979)'));
```

### 2.6. 데이터 타입 변환 (Type Conversion)

SQL에서는 때때로 데이터 타입을 다른 타입으로 변환해야 할 필요가 있습니다. 이는 명시적으로 `CAST()` 또는 `CONVERT()` 함수를 사용하거나, 데이터베이스 시스템이 자동으로 수행하는 암시적(Implicit) 변환을 통해 이루어집니다.

*   **명시적 변환 (Explicit Conversion):** `CAST(expression AS type)` 또는 `CONVERT(expression, type)` 함수를 사용하여 개발자가 직접 데이터 타입을 지정하여 변환합니다. 이는 데이터 타입 불일치로 인한 오류를 방지하고, 쿼리의 의도를 명확히 하며, 특정 연산을 수행하기 위해 필요합니다.
    ```sql
    SELECT CAST('123' AS SIGNED INTEGER); -- 문자열 '123'을 정수로 변환
    SELECT CONCAT('Price: ', CAST(price AS CHAR)); -- 숫자를 문자열로 변환하여 연결
    ```
*   **암시적 변환 (Implicit Conversion):** 데이터베이스 시스템이 쿼리 실행 중 자동으로 데이터 타입을 변환하는 경우입니다. 예를 들어, 숫자 컬럼과 문자열 값을 비교할 때 숫자로 변환하여 비교하는 경우가 있습니다.

**실무적 관점:** 암시적 변환은 편리할 수 있지만, **예상치 못한 결과나 성능 저하를 유발할 수 있으므로 가능한 한 명시적 변환을 사용하는 것이 좋습니다.** 특히 `JOIN` 조건이나 `WHERE` 절에서 암시적 변환이 발생하면 해당 컬럼에 인덱스가 있더라도 인덱스를 사용하지 못하게 되어 쿼리 성능이 심각하게 저하될 수 있습니다. 항상 데이터 타입의 일관성을 유지하고, 필요할 때는 명시적 변환을 사용하는 습관을 들이는 것이 중요합니다.

---

## 3. DDL 실습: 첫 데이터베이스와 테이블 만들기

DDL(Data Definition Language)은 데이터베이스 객체의 구조를 정의하고 관리하는 SQL 명령어입니다. `CREATE`, `ALTER`, `DROP` 등이 있습니다.

**실무적 관점:** DDL은 데이터베이스의 뼈대를 만드는 작업입니다. 테이블을 생성하고 수정하는 과정은 단순히 SQL 문법을 아는 것을 넘어, 데이터가 어떻게 저장되고 관리될지 비즈니스 요구사항을 반영하는 설계 능력을 요구합니다. 특히 `CREATE TABLE` 시 데이터 타입과 제약조건을 신중하게 선택하는 것은 향후 쿼리 성능과 데이터 무결성에 지대한 영향을 미칩니다.

### 3.1. 데이터베이스 생성 및 선택 (`CREATE DATABASE`, `USE`)

데이터를 저장하기 위한 공간인 데이터베이스를 생성하고, 해당 데이터베이스를 사용하도록 선택합니다.

```sql
-- 데이터베이스 목록 확인
SHOW DATABASES;

-- 새 데이터베이스 생성
CREATE DATABASE company_db;

-- 데이터베이스 선택
USE company_db;

-- 선택된 데이터베이스 확인
SELECT DATABASE();
```

### 3.2. 테이블 생성 (`CREATE TABLE`) (데이터 구조의 청사진)

`CREATE TABLE` 문을 사용하여 데이터가 저장될 테이블의 구조를 정의합니다. 컬럼 이름, 데이터 타입, 제약조건 등을 명시합니다. 테이블 생성은 데이터베이스 설계의 핵심 단계입니다.

```sql
CREATE TABLE 테이블명 (
    컬럼명1 데이터타입 [제약조건],
    컬럼명2 데이터타입 [제약조건],
    ...
    [테이블 레벨 제약조건]
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

*   **`테이블명`:** 생성할 테이블의 이름입니다. 의미를 명확히 알 수 있도록 명명하는 것이 중요합니다. (예: `employees`, `products`, `orders`)
*   **`컬럼명`:** 테이블의 각 열을 나타내는 이름입니다. 데이터의 속성을 명확히 설명해야 합니다.
*   **`데이터타입`:** 해당 컬럼에 저장될 데이터의 종류와 형식을 정의합니다. (예: `INT`, `VARCHAR(255)`, `DATE`, `DECIMAL(10,2)`) 올바른 데이터 타입 선택은 저장 공간 효율성과 쿼리 성능에 큰 영향을 미칩니다.
*   **`[제약조건]`:** 컬럼에 저장될 데이터의 무결성을 보장하기 위한 규칙입니다. (예: `PRIMARY KEY`, `NOT NULL`, `UNIQUE`, `FOREIGN KEY`, `DEFAULT`, `CHECK`)
*   **`[테이블 레벨 제약조건]`:** 여러 컬럼에 걸쳐 적용되는 제약조건이나, 컬럼 정의 후에 별도로 정의하는 제약조건입니다. (예: 복합 기본 키, 복합 고유 키, 외래 키)
*   **`ENGINE=InnoDB`:** 테이블에 사용할 스토리지 엔진을 지정합니다. `InnoDB`는 MySQL의 기본 스토리지 엔진으로, 트랜잭션(ACID), 외래 키, 행 수준 잠금(Row-level locking)을 지원하여 데이터 무결성과 동시성 제어에 강점이 있습니다. 대부분의 경우 `InnoDB`를 사용하는 것이 권장됩니다.
*   **`DEFAULT CHARSET=utf8mb4`:** 테이블의 기본 문자 셋을 지정합니다. **한글 및 이모지 등 다양한 유니코드 문자를 문제없이 처리하기 위해 `utf8mb4`를 사용하는 것이 필수적입니다.**
*   **`COLLATE=utf8mb4_unicode_ci`:** 테이블의 기본 콜레이션(문자열 정렬 및 비교 규칙)을 지정합니다. `utf8mb4_unicode_ci`는 대소문자를 구분하지 않는 유니코드 정렬을 의미합니다.

**SQL 예시: 직원, 부서, 프로젝트 테이블 생성**

```sql
-- 1. 부서 테이블 (departments)
CREATE TABLE departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT COMMENT '부서 고유 식별자',
    department_name VARCHAR(50) NOT NULL UNIQUE COMMENT '부서명',
    location VARCHAR(50) COMMENT '부서 위치'
) COMMENT '회사 부서 정보를 저장하는 테이블';

-- 2. 직원 테이블 (employees)
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT COMMENT '직원 고유 식별자',
    first_name VARCHAR(50) NOT NULL COMMENT '이름',
    last_name VARCHAR(50) NOT NULL COMMENT '성',
    email VARCHAR(100) UNIQUE COMMENT '이메일 주소',
    phone_number VARCHAR(20) COMMENT '전화번호',
    hire_date DATE NOT NULL COMMENT '고용일',
    job_id VARCHAR(20) NOT NULL COMMENT '직무 ID',
    salary DECIMAL(10, 2) NOT NULL COMMENT '급여',
    department_id INT COMMENT '부서 ID (외래 키)',
    -- 외래 키 제약조건 추가
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
) COMMENT '회사 직원 정보를 저장하는 테이블';

-- 3. 프로젝트 테이블 (projects)
CREATE TABLE projects (
    project_id INT PRIMARY KEY AUTO_INCREMENT COMMENT '프로젝트 고유 식별자',
    project_name VARCHAR(100) NOT NULL UNIQUE COMMENT '프로젝트명',
    start_date DATE NOT NULL COMMENT '프로젝트 시작일',
    end_date DATE COMMENT '프로젝트 종료일',
    budget DECIMAL(15, 2) COMMENT '프로젝트 예산',
    status ENUM('Pending', 'Active', 'Completed', 'Cancelled') DEFAULT 'Pending' COMMENT '프로젝트 상태'
) COMMENT '회사 프로젝트 정보를 저장하는 테이블';

-- 4. 프로젝트 배정 테이블 (project_assignments) - 직원과 프로젝트의 N:M 관계 해소
CREATE TABLE project_assignments (
    assignment_id INT PRIMARY KEY AUTO_INCREMENT COMMENT '배정 고유 식별자',
    employee_id INT NOT NULL COMMENT '직원 ID (외래 키)',
    project_id INT NOT NULL COMMENT '프로젝트 ID (외래 키)',
    assigned_date DATE DEFAULT CURRENT_DATE COMMENT '배정일',
    -- 복합 유니크 제약조건: 한 직원은 한 프로젝트에 한 번만 배정
    UNIQUE (employee_id, project_id),
    -- 외래 키 제약조건
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
) COMMENT '직원과 프로젝트 배정 정보를 저장하는 테이블';
```

**실무적 고려사항:**
*   **명명 규칙 (Naming Convention):** 일관된 명명 규칙(예: 테이블명은 복수형, 컬럼명은 단수형, 스네이크 케이스 `snake_case`)을 사용합니다. 가독성과 유지보수성을 높입니다. 팀 내에서 합의된 규칙을 따르는 것이 중요합니다.
*   **주석 (Comments):** `COMMENT` 키워드를 사용하여 테이블이나 컬럼에 대한 설명을 추가할 수 있습니다. 이는 데이터베이스의 문서화에 도움이 되며, 다른 개발자나 분석가가 스키마를 이해하는 데 큰 도움이 됩니다.
*   **인덱스 미리 고려:** `WHERE` 절이나 `JOIN` 조건에서 자주 사용될 컬럼에는 인덱스를 생성할 것을 미리 고려하여 테이블 생성 시 함께 정의하거나, 추후 `ALTER TABLE`로 추가합니다. (자세한 내용은 [08_성능튜닝과_인덱스.md](./08_성능튜닝과_인덱스.md) 참조)
*   **정규화 수준:** 테이블을 생성하기 전에 데이터 모델의 정규화 수준을 결정해야 합니다. 과도한 정규화는 `JOIN`을 늘려 쿼리 성능을 저하시킬 수 있고, 부족한 정규화는 데이터 중복과 무결성 문제를 야기할 수 있습니다. (자세한 내용은 [4. 데이터베이스 설계 기초: 정규화(Normalization)](#4-데이터베이스-설계-기초-정규화normalization) 참조)

### 3.3. 테이블 구조 확인 (`DESCRIBE`, `SHOW TABLES`, `SHOW CREATE TABLE`)

생성된 테이블의 구조를 확인하거나, 현재 데이터베이스의 테이블 목록을 조회합니다.

```sql
-- 현재 데이터베이스의 모든 테이블 목록 확인
SHOW TABLES;

-- 특정 테이블의 간략한 구조 확인 (컬럼, 타입, NULL 허용 여부, 키 등)
DESCRIBE employees; -- 또는 DESC employees;

-- 특정 테이블의 상세한 DDL (CREATE TABLE 문) 확인
SHOW CREATE TABLE employees;
```

**실무적 활용:**
*   **`DESCRIBE`:** 테이블의 컬럼 목록과 기본 정보를 빠르게 파악할 때 유용합니다.
*   **`SHOW CREATE TABLE`:** 테이블의 정확한 정의(데이터 타입, 제약조건, 인덱스, 스토리지 엔진, 문자셋 등)를 확인하고 싶을 때 사용합니다. 특히 다른 환경에 동일한 테이블을 생성하거나, 테이블 구조를 문서화할 때 매우 유용합니다.

### 3.4. 테이블 수정 (`ALTER TABLE`) (데이터베이스 스키마 변경)

`ALTER TABLE` 문을 사용하여 기존 테이블의 구조를 변경합니다. 컬럼 추가, 삭제, 수정, 이름 변경, 제약조건 추가/삭제 등이 가능합니다. `ALTER TABLE`은 데이터베이스 스키마를 변경하는 강력한 명령이므로, 운영 환경에서는 매우 신중하게 사용해야 합니다.

**실무적 관점:** `ALTER TABLE` 작업은 테이블에 잠금(Lock)을 걸어 서비스에 영향을 줄 수 있습니다. 특히 대규모 테이블에 대한 `ALTER` 작업은 서비스 중단(Downtime)을 유발할 수 있으므로, `Online DDL` 기능(DBMS 지원 시)을 활용하거나 서비스 사용량이 적은 시간대에 수행하는 등 전략적인 접근이 필요합니다. 변경 전에는 반드시 백업을 수행해야 합니다.

```sql
-- 1. 컬럼 추가: employees 테이블에 manager_id 컬럼 추가
ALTER TABLE employees
ADD COLUMN manager_id INT COMMENT '상사 직원 ID';

-- 2. 컬럼 수정: employees 테이블의 phone_number 컬럼 데이터 타입 변경 및 NOT NULL 제약조건 추가
ALTER TABLE employees
MODIFY COLUMN phone_number VARCHAR(30) NOT NULL;

-- 3. 컬럼 이름 변경: projects 테이블의 project_name 컬럼을 name으로 변경
ALTER TABLE projects
CHANGE COLUMN project_name name VARCHAR(100) NOT NULL UNIQUE COMMENT '프로젝트명';

-- 4. 컬럼 삭제: employees 테이블에서 phone_number 컬럼 삭제
ALTER TABLE employees
DROP COLUMN phone_number;

-- 5. 제약조건 추가: employees 테이블에 email 컬럼에 UNIQUE 제약조건 추가
ALTER TABLE employees
ADD CONSTRAINT UQ_employees_email UNIQUE (email);

-- 6. 제약조건 삭제: employees 테이블에서 fk_department 외래 키 제약조건 삭제
ALTER TABLE employees
DROP FOREIGN KEY fk_department;

-- 7. PRIMARY KEY 추가/삭제 (기존 PK가 없는 경우 또는 변경 시)
-- 기존 PK 삭제 (먼저 해당 PK를 참조하는 FK가 있다면 삭제해야 함)
-- ALTER TABLE products DROP PRIMARY KEY;
-- 새로운 PK 추가
-- ALTER TABLE products ADD PRIMARY KEY (product_id);

-- 8. 테이블 이름 변경
ALTER TABLE old_table_name RENAME TO new_table_name;

-- 9. AUTO_INCREMENT 값 재설정
ALTER TABLE employees AUTO_INCREMENT = 1001;
```

### 3.5. 테이블 삭제 (`DROP TABLE`) (데이터 및 구조 영구 삭제)

`DROP TABLE` 문을 사용하여 데이터베이스에서 테이블을 완전히 삭제합니다. 테이블의 구조와 데이터가 모두 제거되며, **한번 삭제된 데이터는 복구하기 어렵습니다.** 운영 환경에서는 매우 신중하게 사용해야 합니다.

```sql
-- employees 테이블 삭제
DROP TABLE employees;

-- 만약 테이블이 존재하면 삭제 (에러 방지)
DROP TABLE IF EXISTS projects;
```

**실무적 관점:** `DROP TABLE`은 되돌릴 수 없는 작업이므로, 운영 환경에서는 반드시 백업 후 실행하고, 권한 관리를 철저히 해야 합니다. 실수로 중요한 테이블을 삭제하는 것을 방지하기 위해 `DROP TABLE` 권한은 최소한의 사용자에게만 부여하는 것이 좋습니다.

### 3.6. 임시 테이블 (Temporary Table) 활용

임시 테이블은 현재 세션에서만 존재하고, 세션이 종료되면 자동으로 삭제되는 특별한 테이블입니다. 데이터 분석가가 복잡한 쿼리의 중간 결과를 저장하거나, 여러 단계의 데이터 처리 과정을 분리하여 관리할 때 매우 유용합니다.

*   **생성:** `CREATE TEMPORARY TABLE` 문을 사용하여 생성합니다. 일반 테이블과 동일하게 컬럼과 제약조건을 정의할 수 있습니다.

    ```sql
    CREATE TEMPORARY TABLE temp_high_salary_employees (
        employee_id INT PRIMARY KEY,
        full_name VARCHAR(100),
        salary DECIMAL(10, 2)
    );

    -- SELECT 결과를 기반으로 임시 테이블 생성
    CREATE TEMPORARY TABLE temp_department_avg_salary AS
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id;
    ```

*   **사용:** 일반 테이블처럼 `INSERT`, `SELECT`, `UPDATE`, `DELETE` 등의 DML 작업을 수행할 수 있습니다.

    ```sql
    -- 임시 테이블에 데이터 삽입
    INSERT INTO temp_high_salary_employees (employee_id, full_name, salary)
    SELECT employee_id, CONCAT(first_name, ' ', last_name), salary
    FROM employees
    WHERE salary >= 70000;

    -- 임시 테이블 조회
    SELECT * FROM temp_high_salary_employees;
    ```

*   **삭제:** 세션이 종료되면 자동으로 삭제되지만, 명시적으로 `DROP TEMPORARY TABLE`을 사용하여 삭제할 수도 있습니다.

    ```sql
    DROP TEMPORARY TABLE temp_high_salary_employees;
    ```

**실무적 활용:**
*   **복잡한 쿼리 분해:** 복잡한 분석 쿼리를 논리적으로 분리하여 디버깅과 가독성을 향상시킵니다.
*   **중간 결과 저장:** 대규모 데이터셋에서 반복적으로 사용되는 중간 계산 결과를 저장하여 쿼리 성능을 향상시킬 수 있습니다. (단, 인덱스 생성 등 최적화 고려)
*   **데이터 전처리:** 원본 데이터를 변경하지 않고, 임시 테이블에서 데이터를 정제하거나 변환하는 작업을 수행할 수 있습니다.

## 4. 데이터 무결성: 제약조건(Constraints) 활용

데이터 무결성(Data Integrity)은 데이터베이스에 저장된 데이터의 정확성, 일관성, 유효성을 유지하는 것을 의미합니다. 제약조건(Constraints)은 데이터 무결성을 보장하기 위해 테이블의 컬럼에 적용되는 규칙입니다.

### 4.1. 제약조건의 개념과 중요성

데이터 무결성(Data Integrity)은 데이터베이스에 저장된 데이터의 정확성, 일관성, 유효성을 유지하는 것을 의미합니다. 제약조건(Constraints)은 이러한 데이터 무결성을 보장하기 위해 테이블의 컬럼에 적용되는 규칙입니다.

**실무적 관점:** 제약조건은 데이터베이스 설계의 핵심 요소입니다. 단순히 데이터를 저장하는 것을 넘어, 데이터의 품질을 보장하고, 애플리케이션의 오류를 줄이며, 데이터 분석의 신뢰성을 높이는 데 결정적인 역할을 합니다. 올바른 제약조건 설정은 데이터베이스의 '방어막' 역할을 하여 잘못된 데이터가 유입되는 것을 막아줍니다.

### 4.2. `PRIMARY KEY`: 기본 키

테이블의 각 로우를 고유하게 식별하는 데 사용됩니다. `PRIMARY KEY`는 `NOT NULL`과 `UNIQUE` 속성을 자동으로 가집니다. `AUTO_INCREMENT`와 함께 자주 사용됩니다.

**SQL 예시:**
```sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT COMMENT '직원 고유 식별자',
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL
);

INSERT INTO employees (first_name, last_name) VALUES ('John', 'Doe');
INSERT INTO employees (first_name, last_name) VALUES ('Jane', 'Smith');
```

#### 4.2.1. 대리 키(Surrogate Key) vs 자연 키(Natural Key)

**실무 가이드:** 실무에서는 고민하지 말고 `AUTO_INCREMENT`를 사용한 **대리 키(Surrogate Key)**를 기본 키(PK)로 사용하는 것이 표준입니다.
*   **대리 키 (Surrogate Key):** 비즈니스와 관련 없는, 오직 행을 식별하기 위한 인공적인 키 (예: `id`, `user_id`). 비즈니스 로직이 변경되어도 `id` 값은 절대 변하지 않으므로 시스템의 안정성과 유연성을 보장합니다.
*   **자연 키 (Natural Key):** 비즈니스 의미를 가진 키 (예: `이메일 주소`, `주민등록번호`). 이 값이 변경되면 이를 참조하는 모든 테이블의 값을 수정해야 하는 문제가 발생할 수 있습니다.

**설계 패턴:**
1.  **기본 키(PK)는 항상 대리 키(`AUTO_INCREMENT` ID)로 만듭니다.**
2.  **비즈니스적으로 고유해야 하는 값(자연 키)은 `UNIQUE` 제약조건으로 관리합니다.** (예: `email`, `product_code`)

```sql
CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT, -- 대리 키
    product_code VARCHAR(20) NOT NULL UNIQUE, -- 자연 키 (UNIQUE 제약조건)
    product_name VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL
);
```

### 4.3. `FOREIGN KEY`: 외래 키 (참조 무결성)

다른 테이블의 기본 키를 참조하여 두 테이블 간의 관계를 설정하고 참조 무결성을 유지합니다. 외래 키는 참조하는 테이블의 기본 키 값으로 반드시 존재하거나 `NULL`이어야 합니다.

**SQL 예시:**
```sql
-- 부모 테이블 (departments) 먼저 생성
CREATE TABLE departments (
    department_id INT PRIMARY KEY AUTO_INCREMENT,
    department_name VARCHAR(50) NOT NULL UNIQUE
);

-- 자식 테이블 (employees) 생성 시 외래 키 정의
CREATE TABLE employees_fk_example (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

INSERT INTO departments (department_name) VALUES ('Development'), ('Human Resources');
INSERT INTO employees_fk_example (first_name, department_id) VALUES ('Alice', 1);
-- INSERT INTO employees_fk_example (first_name, department_id) VALUES ('Bob', 999); -- 오류 발생: 참조 무결성 위반
```

#### 4.3.1. `ON DELETE` 및 `ON UPDATE` 옵션

외래 키 제약조건은 참조되는 부모 테이블의 로우가 삭제되거나 업데이트될 때 자식 테이블의 로우에 어떤 동작을 취할지 정의할 수 있습니다.

*   **`CASCADE`:** 부모 테이블의 로우가 삭제/업데이트되면 자식 테이블의 관련 로우도 함께 삭제/업데이트됩니다.
    ```sql
    -- 예시: 부서 삭제 시 해당 부서 직원도 함께 삭제
    FOREIGN KEY (department_id) REFERENCES departments(department_id) ON DELETE CASCADE
    ```
*   **`SET NULL`:** 부모 테이블의 로우가 삭제/업데이트되면 자식 테이블의 외래 키 컬럼을 `NULL`로 설정합니다. (외래 키 컬럼이 `NULL`을 허용해야 함)
    ```sql
    -- 예시: 부서 삭제 시 해당 부서 직원의 department_id를 NULL로 설정
    FOREIGN KEY (department_id) REFERENCES departments(department_id) ON DELETE SET NULL
    ```
*   **`RESTRICT` (기본값):** 부모 테이블의 로우가 자식 테이블에서 참조되고 있으면 삭제/업데이트를 허용하지 않습니다. (먼저 자식 로우를 삭제해야 부모 로우 삭제 가능)
    ```sql
    -- 예시: 부서에 직원이 있으면 부서 삭제 불가
    FOREIGN KEY (department_id) REFERENCES departments(department_id) ON DELETE RESTRICT
    ```
*   **`NO ACTION`:** `RESTRICT`와 유사하지만, 표준 SQL에서는 지연된 제약조건 검사를 의미합니다. MySQL에서는 `RESTRICT`와 동일하게 동작합니다.

**실무적 관점:** `ON DELETE CASCADE`는 매우 강력하지만, 의도치 않은 데이터 손실을 유발할 수 있으므로 신중하게 사용해야 합니다. 대부분의 경우 `RESTRICT`나 `SET NULL`을 사용하여 데이터 무결성을 명시적으로 관리하는 것이 안전합니다.

### 4.4. `UNIQUE`: 고유 제약조건

컬럼의 모든 값이 고유해야 함을 보장합니다. `NULL` 값은 여러 번 허용됩니다. (MySQL 기준)

**SQL 예시:**
```sql
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) UNIQUE
);

INSERT INTO users (username, email) VALUES ('alice', 'alice@example.com');
-- INSERT INTO users (username, email) VALUES ('alice', 'alice2@example.com'); -- 오류 발생: username 중복
INSERT INTO users (username, email) VALUES ('bob', NULL);
INSERT INTO users (username, email) VALUES ('charlie', NULL); -- NULL은 여러 번 허용
```

### 4.5. `NOT NULL`: 필수 값

컬럼에 `NULL` 값이 저장되는 것을 방지합니다. 데이터 입력 시 해당 컬럼에 반드시 값이 있어야 합니다.

**SQL 예시:**
```sql
CREATE TABLE tasks (
    task_id INT PRIMARY KEY AUTO_INCREMENT,
    task_name VARCHAR(255) NOT NULL,
    due_date DATE
);

INSERT INTO tasks (task_name, due_date) VALUES ('Complete Report', '2025-08-31');
-- INSERT INTO tasks (due_date) VALUES ('2025-09-01'); -- 오류 발생: task_name은 NOT NULL
```

### 4.6. `DEFAULT`: 기본 값

컬럼에 값을 명시하지 않을 경우 자동으로 삽입될 기본 값을 설정합니다.

**SQL 예시:**
```sql
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    order_date DATETIME DEFAULT CURRENT_TIMESTAMP, -- 현재 시간으로 기본값 설정
    status VARCHAR(50) DEFAULT 'Pending'
);

INSERT INTO orders (order_id) VALUES (1);
INSERT INTO orders (order_id, status) VALUES (2, 'Completed');
```

### 4.7. `CHECK`: 값 범위/조건 검사

컬럼에 저장될 값의 범위나 조건을 정의합니다. MySQL에서는 `CHECK` 제약조건이 구문적으로는 허용되지만, 8.0.16 이전 버전에서는 실제로 강제되지 않았습니다. 8.0.16부터는 제대로 동작합니다.

**SQL 예시:**
```sql
CREATE TABLE products_with_check (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0),
    stock_quantity INT CHECK (stock_quantity >= 0 AND stock_quantity <= 1000) -- 재고는 0에서 1000 사이
);

INSERT INTO products_with_check (product_name, price, stock_quantity) VALUES ('Laptop', 1200.00, 50);
-- INSERT INTO products_with_check (product_name, price, stock_quantity) VALUES ('Mouse', -10.00, 10); -- 오류 발생: price < 0
-- INSERT INTO products_with_check (product_name, price, stock_quantity) VALUES ('Keyboard', 50.00, 1001); -- 오류 발생: stock_quantity > 1000
```

### 4.8. 제약조건 테스트 및 관리

제약조건이 올바르게 작동하는지 확인하고, 필요에 따라 기존 테이블의 제약조건을 관리합니다.

#### 4.8.1. 제약조건 위반 시 동작 확인

제약조건을 위반하는 `INSERT` 또는 `UPDATE` 문을 실행하면 데이터베이스는 오류를 반환하고 작업을 거부합니다.

```sql
-- 예시 테이블 생성 (위에서 정의된 users, employees_fk_example, tasks 테이블 사용)

-- 1. PRIMARY KEY 위반 예시 (users 테이블)
-- INSERT INTO users (user_id, username, email) VALUES (1, 'test_user', 'test@example.com'); -- user_id 1이 이미 존재한다고 가정
-- 오류 메시지 예시: Duplicate entry '1' for key 'users.PRIMARY'

-- 2. NOT NULL 위반 예시 (tasks 테이블)
-- INSERT INTO tasks (due_date) VALUES ('2025-12-31');
-- 오류 메시지 예시: Field 'task_name' doesn't have a default value

-- 3. UNIQUE 위반 예시 (users 테이블)
-- INSERT INTO users (username, email) VALUES ('alice', 'new_alice@example.com'); -- username 'alice'가 이미 존재한다고 가정
-- 오류 메시지 예시: Duplicate entry 'alice' for key 'users.username'

-- 4. FOREIGN KEY 위반 예시 (employees_fk_example 테이블)
-- INSERT INTO employees_fk_example (first_name, department_id) VALUES ('Charlie', 999); -- department_id 999가 departments 테이블에 없다고 가정
-- 오류 메시지 예시: Cannot add or update a child row: a foreign key constraint fails (`db_name`.`employees_fk_example`, CONSTRAINT `employees_fk_example_ibfk_1` FOREIGN KEY (`department_id`) REFERENCES `departments` (`department_id`))

-- 5. CHECK 위반 예시 (products_with_check 테이블)
-- INSERT INTO products_with_check (product_name, price, stock_quantity) VALUES ('Broken Item', -5.00, 10);
-- 오류 메시지 예시: Check constraint 'products_with_check_chk_1' is violated.
```

#### 4.8.2. 기존 테이블에 제약조건 추가/삭제

`ALTER TABLE` 문을 사용하여 기존 테이블에 제약조건을 추가하거나 삭제할 수 있습니다.

```sql
-- 1. 기존 테이블에 UNIQUE 제약조건 추가
-- employees 테이블에 email 컬럼이 있고, 기존 데이터에 중복이 없어야 함
ALTER TABLE employees
ADD CONSTRAINT UQ_employees_email UNIQUE (email);

-- 2. 기존 테이블에 FOREIGN KEY 제약조건 추가
-- employees 테이블에 department_id 컬럼이 있고, departments 테이블이 존재하며, 기존 데이터에 유효하지 않은 department_id가 없어야 함
ALTER TABLE employees
ADD CONSTRAINT FK_employees_department
FOREIGN KEY (department_id) REFERENCES departments(department_id)
ON DELETE SET NULL ON UPDATE CASCADE;

-- 3. 기존 테이블에서 제약조건 삭제
-- 외래 키 제약조건 삭제 (제약조건 이름으로 삭제)
ALTER TABLE employees
DROP FOREIGN KEY FK_employees_department;

-- UNIQUE 제약조건 삭제 (인덱스 이름으로 삭제. MySQL은 UNIQUE 제약조건 생성 시 자동으로 인덱스 생성)
ALTER TABLE users
DROP INDEX username; -- 또는 DROP INDEX UQ_users_username (자동 생성된 이름)

-- PRIMARY KEY 삭제 (먼저 해당 PK를 참조하는 FK가 있다면 삭제해야 함)
-- ALTER TABLE products DROP PRIMARY KEY;
```

**실무적 관점:** 기존 테이블에 제약조건을 추가할 때는 매우 신중해야 합니다. 특히 `NOT NULL`, `UNIQUE`, `FOREIGN KEY`와 같은 제약조건을 추가할 경우, **기존 데이터가 제약조건을 위반하지 않는지 미리 확인**해야 합니다. 위반하는 데이터가 있다면 제약조건 추가가 실패하거나, 데이터가 손실될 수 있습니다. 대규모 테이블의 경우 `ALTER TABLE` 작업은 오랜 시간이 걸릴 수 있으므로, 서비스에 미치는 영향을 최소화하기 위한 전략(예: `Online DDL`)을 고려해야 합니다.

## 5. 데이터베이스 설계 기초: 정규화(Normalization)

정규화(Normalization)는 데이터의 **중복을 최소화**하고, 데이터가 변경될 때 발생할 수 있는 **불일치(Anomaly)를 막는** 체계적인 과정입니다.

**분석가의 관점:**
1.  **데이터 이해의 틀:** 잘 정규화된 데이터베이스는 데이터의 구조와 관계를 명확하게 보여주어 비즈니스 이해도를 높입니다.
2.  **정확한 쿼리 작성의 기반:** 데이터가 중복 없이 저장되어 있으므로, `COUNT`, `SUM` 등의 집계 함수 사용 시 중복 계산의 위험이 줄어듭니다.
3.  **JOIN의 필수성:** 데이터가 여러 테이블에 나뉘어 저장되므로, 원하는 데이터를 얻기 위해서는 **`JOIN` 사용이 필수적**입니다.

### 5.1. 정규화의 필요성 및 목표

*   **필요성:** 데이터 중복으로 인한 저장 공간 낭비, 데이터 불일치(이상 현상) 발생, 유지보수 비용 증가 등을 방지합니다.
*   **목표:** 데이터 중복 최소화, 데이터 무결성 보장, 데이터베이스 구조의 안정성 및 확장성 향상.
*   **ERD(Entity-Relationship Diagram):** 정규화 과정은 데이터베이스를 효율적으로 설계하기 위한 핵심 단계입니다. 이 과정에서 **ERD(Entity-Relationship Diagram)**는 데이터베이스의 구조와 테이블 간의 관계를 시각적으로 표현하는 데 사용되는 필수 도구입니다.
*   **데이터 카탈로그/사전:** 분석가는 쿼리 작성 전, 데이터 카탈로그를 통해 테이블과 컬럼의 비즈니스적 의미를 정확히 파악해야 합니다.

### 5.2. 함수적 종속성 (Functional Dependency)

정규화를 이해하기 위한 가장 기본적인 개념은 **함수적 종속성(Functional Dependency)**입니다. 함수적 종속성은 특정 속성(컬럼)의 값이 다른 속성의 값을 유일하게 결정하는 관계를 의미합니다.

*   **개념:** `A -> B` (A는 B를 함수적으로 결정한다)는 A의 값이 B의 값을 유일하게 결정한다는 의미입니다. 즉, A의 값이 같으면 B의 값도 항상 같습니다.
    *   예시: `학번 -> 학생이름` (학번이 같으면 학생이름은 항상 같다)
    *   예시: `주민등록번호 -> 주소` (주민등록번호가 같으면 주소는 항상 같다)
*   **완전 함수적 종속 (Full Functional Dependency):** 기본 키가 복합 키(두 개 이상의 컬럼으로 구성된 키)일 때, 비기본 키 컬럼이 기본 키 전체에 종속되는 경우를 의미합니다. 기본 키의 어떤 부분에도 종속되지 않아야 합니다.
*   **부분 함수적 종속 (Partial Functional Dependency):** 기본 키가 복합 키일 때, 비기본 키 컬럼이 기본 키의 일부 컬럼에만 종속되는 경우를 의미합니다. 2차 정규화에서 제거 대상입니다.
    *   예시: `(주문번호, 상품번호) -> 상품명` 에서 `상품번호 -> 상품명` 이 성립한다면, `상품명`은 복합 키의 일부인 `상품번호`에만 종속되므로 부분 함수적 종속입니다.
*   **이행적 함수적 종속 (Transitive Functional Dependency):** `A -> B` 이고 `B -> C` 일 때, `A -> C` 가 성립하는 경우를 의미합니다. 즉, 기본 키가 아닌 컬럼이 다른 비기본 키 컬럼에 종속되는 경우입니다. 3차 정규화에서 제거 대상입니다.
    *   예시: `학번 -> 학과코드` 이고 `학과코드 -> 학과명` 일 때, `학번 -> 학과명` 이 성립한다면 이행적 함수적 종속입니다.

### 5.3. 이상(Anomaly) 현상: 삽입, 삭제, 갱신 이상

정규화되지 않은 테이블에서 발생하는 문제점들입니다. `수강_등록`이라는 테이블을 예시로 들어 설명합니다.

**`수강_등록` 테이블 (정규화되지 않은 상태):**

| 학번 | 학생이름 | 학과 | 과목코드 | 과목명 | 학점 | 교수이름 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 101 | 김철수 | 컴퓨터공학 | CS101 | 자료구조 | 3 | 이교수 |
| 101 | 김철수 | 컴퓨터공학 | CS102 | 알고리즘 | 3 | 박교수 |
| 102 | 이영희 | 경영학 | BA201 | 마케팅 | 3 | 최교수 |
| 103 | 박지민 | 컴퓨터공학 | CS101 | 자료구조 | 3 | 이교수 |

*   **삽입 이상 (Insertion Anomaly):** 데이터를 삽입할 때 불필요한 정보도 함께 삽입해야 하거나, 필요한 정보를 삽입할 수 없는 경우.
    *   **예시:** 새로운 과목 `CS103 (데이터베이스)`을 개설하고 싶지만, 아직 이 과목을 수강하는 학생이 없어서 `수강_등록` 테이블에 삽입할 수 없습니다. (학번, 학생이름 등 `NULL`이 될 수 없는 필드를 채워야 하거나, 아예 삽입이 불가능)
*   **삭제 이상 (Deletion Anomaly):** 데이터를 삭제할 때 의도하지 않은 다른 정보까지 함께 삭제되는 경우.
    *   **예시:** `학번 102 (이영희)` 학생이 `BA201 (마케팅)` 과목 수강을 취소하여 해당 로우를 삭제하면, `이영희` 학생의 `경영학` 학과 정보와 `BA201` 과목의 `마케팅` 과목명, `최교수` 정보까지 함께 사라집니다. 만약 `이영희`가 이 과목만 수강했다면, `이영희`에 대한 모든 정보가 사라지는 문제가 발생합니다.
*   **갱신 이상 (Update Anomaly):** 중복된 데이터 중 일부만 갱신되어 데이터 불일치가 발생하는 경우.
    *   **예시:** `김철수` 학생의 학과가 `컴퓨터공학`에서 `소프트웨어공학`으로 변경되면, `김철수`가 수강한 모든 과목 로우(`CS101`, `CS102`)를 찾아 `학과` 컬럼을 일일이 `소프트웨어공학`으로 갱신해야 합니다. 만약 하나라도 누락되면 `김철수`의 학과 정보가 `컴퓨터공학`과 `소프트웨어공학`으로 중복되어 데이터 불일치가 발생합니다.

### 5.4. 정규형(Normal Form)의 종류

정규화는 여러 단계의 정규형(Normal Form)을 통해 이루어집니다. 각 정규형은 이전 정규형의 조건을 만족하면서 추가적인 제약조건을 가집니다.

#### 5.4.1. 제1정규형 (1NF)

*   **조건:**
    1.  모든 컬럼이 **원자 값(Atomic Value)**을 가짐 (더 이상 분해할 수 없는 단일 값).
    2.  모든 로우가 고유하게 식별 가능해야 함 (기본 키 존재).
    3.  반복되는 그룹(Repeating Groups)이 없어야 함.
*   **위반 예시:** 한 컬럼에 여러 값이 콤마 등으로 구분되어 저장되거나, 여러 컬럼이 동일한 속성의 반복을 나타내는 경우.
    *   `학생_스킬` 테이블: `(학번, 스킬1, 스킬2, 스킬3)` 또는 `(학번, 스킬_목록: 'Python, SQL, Excel')`
*   **1NF 적용:** 반복되는 그룹을 제거하고 각 원자 값을 별도의 로우로 분리하거나, 새로운 테이블로 분리합니다.
    *   `학생_스킬` 테이블을 `(학번, 스킬명)`으로 분리하여 각 스킬을 별도의 로우로 저장.
    *   **`수강_등록` 테이블은 이미 1NF를 만족한다고 가정합니다.** (모든 컬럼이 원자 값을 가지며, `(학번, 과목코드)`가 기본 키 후보가 될 수 있음)

#### 5.4.2. 제2정규형 (2NF)

*   **조건:**
    1.  1NF를 만족.
    2.  모든 비기본 키 컬럼이 기본 키 **전체**에 대해 **완전 함수 종속적**이어야 함 (부분 함수 종속 제거).
    *   **부분 함수 종속:** 기본 키가 복합 키(두 개 이상의 컬럼으로 구성된 키)일 때, 비기본 키 컬럼이 기본 키의 일부 컬럼에만 종속되는 경우.
*   **`수강_등록` 테이블의 2NF 위반:**
    `수강_등록` 테이블의 기본 키 후보는 `(학번, 과목코드)`입니다.
    *   `학번 -> 학생이름, 학과` (학생이름, 학과는 `학번`에만 종속)
    *   `과목코드 -> 과목명, 학점, 교수이름` (과목명, 학점, 교수이름은 `과목코드`에만 종속)
    *   `학생이름`, `학과`, `과목명`, `학점`, `교수이름`은 복합 키 `(학번, 과목코드)`의 일부에만 종속되므로 부분 함수 종속입니다.
*   **2NF 적용:** 부분 함수 종속을 제거하기 위해 테이블을 분리합니다.
    1.  **`학생` 테이블:** `학번`에 종속되는 속성들 (`학번`, `학생이름`, `학과`)
    2.  **`과목` 테이블:** `과목코드`에 종속되는 속성들 (`과목코드`, `과목명`, `학점`, `교수이름`)
    3.  **`수강` 테이블:** `(학번`, `과목코드)`에 종속되는 속성들 (여기서는 없음, 오직 관계만 남음)

    **분리된 테이블:**
    *   **`학생` 테이블:**
        | 학번 (PK) | 학생이름 | 학과 |
        | :--- | :--- | :--- |
        | 101 | 김철수 | 컴퓨터공학 |
        | 102 | 이영희 | 경영학 |
        | 103 | 박지민 | 컴퓨터공학 |

    *   **`과목` 테이블:**
        | 과목코드 (PK) | 과목명 | 학점 | 교수이름 |
        | :--- | :--- | :--- | :--- |
        | CS101 | 자료구조 | 3 | 이교수 |
        | CS102 | 알고리즘 | 3 | 박교수 |
        | BA201 | 마케팅 | 3 | 최교수 |

    *   **`수강` 테이블:**
        | 학번 (PK, FK) | 과목코드 (PK, FK) |
        | :--- | :--- |
        | 101 | CS101 |
        | 101 | CS102 |
        | 102 | BA201 |
        | 103 | CS101 |

    **2NF 적용 후 효과:**
    *   **삽입 이상 해결:** `과목` 테이블에 학생 없이도 새로운 과목을 추가할 수 있습니다.
    *   **삭제 이상 해결:** `수강` 테이블에서 학생의 수강 기록을 삭제해도 `학생`이나 `과목` 정보는 유지됩니다.
    *   **갱신 이상 해결:** `김철수`의 학과가 변경되면 `학생` 테이블의 한 로우만 갱신하면 됩니다.

#### 5.4.3. 제3정규형 (3NF)

*   **조건:**
    1.  2NF를 만족.
    2.  모든 비기본 키 컬럼이 기본 키에 대해 **이행적 함수 종속(Transitive Functional Dependency)**을 가지지 않아야 함.
    *   **이행적 함수 종속:** `A -> B` 이고 `B -> C` 일 때, `A -> C` 가 성립하는 경우 (기본 키가 아닌 컬럼이 다른 비기본 키 컬럼에 종속되는 경우).
*   **`과목` 테이블의 3NF 위반:**
    `과목` 테이블에서 `과목코드`가 기본 키입니다.
    *   `과목코드 -> 교수이름` (과목코드가 교수이름을 결정)
    *   하지만 `교수이름`은 `과목코드`에 직접 종속되기보다는, `교수번호`와 같은 별도의 속성에 의해 결정될 수 있습니다. 만약 `교수이름`이 `교수번호`에 종속되고, `과목코드`가 `교수번호`에 종속된다면 이행적 함수 종속이 발생합니다.
    *   더 명확한 예시: `학생` 테이블에 `학과코드`와 `학과명`이 함께 있다면, `학번 -> 학과코드` 이고 `학과코드 -> 학과명` 이므로 `학번 -> 학과명`은 이행적 함수 종속입니다.
*   **3NF 적용:** 이행적 함수 종속을 제거하기 위해 새로운 테이블로 분리합니다.
    *   `과목` 테이블에서 `교수이름`을 분리하여 `교수` 테이블을 생성합니다. (이때 `교수번호`와 같은 새로운 기본 키를 도입할 수 있습니다.)

    **분리된 테이블 (예시: `학생` 테이블에 `학과코드`와 `학과명`이 함께 있었다고 가정):**
    *   **`학생` 테이블 (3NF 만족):**
        | 학번 (PK) | 학생이름 | 학과코드 (FK) |
        | :--- | :--- | :--- |
        | 101 | 김철수 | CS |
        | 102 | 이영희 | BA |
        | 103 | 박지민 | CS |

    *   **`학과` 테이블 (새로 분리):**
        | 학과코드 (PK) | 학과명 |
        | :--- | :--- |
        | CS | 컴퓨터공학 |
        | BA | 경영학 |

    *   **`과목` 테이블 (3NF 만족):**
        | 과목코드 (PK) | 과목명 | 학점 | 교수이름 (FK) |
        | :--- | :--- | :--- | :--- |
        | CS101 | 자료구조 | 3 | 이교수 |
        | CS102 | 알고리즘 | 3 | 박교수 |
        | BA201 | 마케팅 | 3 | 최교수 |

    *   **`교수` 테이블 (새로 분리):**
        | 교수이름 (PK) | 교수전공 |
        | :--- | :--- |
        | 이교수 | 데이터베이스 |
        | 박교수 | 인공지능 |
        | 최교수 | 마케팅 |

    **3NF 적용 후 효과:**
    *   **갱신 이상 해결:** `컴퓨터공학`의 학과명이 `소프트웨어공학`으로 변경되면 `학과` 테이블의 한 로우만 갱신하면 됩니다.
    *   **삽입 이상 해결:** `교수` 테이블에 과목을 담당하지 않는 새로운 교수를 추가할 수 있습니다.

#### 5.4.4. 보이스-코드 정규형 (BCNF)

*   **조건:**
    1.  3NF를 만족.
    2.  모든 **결정자(Determinant)**가 **후보 키(Candidate Key)**여야 함.
    *   **결정자:** 다른 컬럼의 값을 결정하는 컬럼(들).
    *   **후보 키:** 기본 키가 될 수 있는 컬럼(들)의 집합.
*   **설명:** 3NF의 강화된 형태로, 복합 키와 관련된 특정 유형의 종속성(특히 여러 후보 키가 존재하고 서로 겹치는 경우)을 해결합니다. BCNF는 3NF보다 더 엄격한 무결성을 보장하지만, 실무에서는 3NF까지만 적용하는 경우가 많습니다. BCNF를 위반하는 경우는 드물며, 주로 하나의 테이블에 여러 후보 키가 있고, 이 후보 키들이 서로 겹치는 경우에 발생합니다.
*   **BCNF 위반 예시:**
    `학생_특강` 테이블: `(학생번호, 특강이름, 교수이름)`
    *   후보 키: `(학생번호, 특강이름)`
    *   추가 종속성: `교수이름 -> 특강이름` (한 교수는 하나의 특강만 담당)
    이 경우 `교수이름`은 `특강이름`을 결정하는 결정자이지만, `교수이름`은 후보 키가 아닙니다. 따라서 BCNF를 위반합니다.
*   **BCNF 적용:** `교수이름 -> 특강이름` 종속성을 제거하기 위해 `특강` 테이블을 분리합니다.
    *   **`학생_특강_등록` 테이블:** `(학생번호, 특강이름)`
    *   **`특강` 테이블:** `(특강이름, 교수이름)`

### 5.5. 분석을 위한 데이터 모델링: 스타 스키마(Star Schema)

데이터 분석 환경(데이터 웨어하우스, 데이터 마트)에서는 **분석 쿼리의 성능**을 높이기 위해 의도적으로 **비정규화(Denormalization)**된 **스타 스키마** 구조를 가장 널리 사용합니다.

*   **스타 스키마를 쉽게 이해하기:**
    *   **사실 테이블 (Fact Table):** 분석하고 싶은 **숫자 값(측정값)**들이 담겨있는 테이블입니다. 비즈니스에서 실제로 일어난 '사건(event)'을 기록합니다.
        *   **"무엇을 분석할 것인가?"** 에 대한 답.
        *   예: `sales_fact` (매출액, 판매량), `page_views_fact` (페이지뷰 수, 체류 시간)
    *   **차원 테이블 (Dimension Table):** 사실 테이블의 숫자를 분석할 **다양한 기준(관점)**을 제공하는 테이블입니다. '누가, 언제, 어디서, 무엇을'에 대한 풍부한 '문맥(context)' 정보를 담고 있습니다.
        *   **"어떻게 분석할 것인가?"** 에 대한 답.
        *   예: `dim_customer` (고객 정보), `dim_product` (상품 정보), `dim_date` (날짜 정보)

*   **[스타 스키마 예시: 온라인 상점 매출 분석]**

    ```
         +-----------------+
         |   dim_date      | (언제)
         +-----------------+
         | date_key (PK)   |
         | year, month, day|
         +-----------------+
                 |
                 |
    +------------------+      +----------------------+      +------------------+
    |  dim_customer    |      |      sales_fact      |      |   dim_product    | (무엇을)
    +------------------+      +----------------------+      +------------------+
    | customer_key (PK)|------| date_key (FK)        |------| product_key (PK) |
    | name, age, city  |      | customer_key (FK)    |      | name, brand, cat |
    +------------------+      | product_key (FK)     |      +------------------+
         (누가)               | store_key (FK)       |
                              | sales_amount         |
                              | quantity_sold        |
                              +----------------------+
    ```

*   **왜 스타 스키마를 사용하는가?**
    1.  **단순함:** 구조가 매우 직관적이어서 분석가나 BI 도구가 이해하기 쉽습니다.
    2.  **빠른 성능:** 분석에 필요한 대부분의 `JOIN`이 사실 테이블과 차원 테이블 간의 1:N 관계로 단순화되어, 복잡한 `JOIN`을 피할 수 있고 쿼리 성능이 매우 빠릅니다.

*   **눈송이 스키마 (Snowflake Schema)는 무엇인가요?**
    눈송이 스키마는 스타 스키마의 차원 테이블을 추가로 정규화하여 저장 공간을 절약하는 모델이지만, `JOIN`이 더 복잡해져 쿼리 성능이 저하될 수 있습니다. 따라서 **현대의 분석 환경에서는 대부분 스타 스키마를 표준으로 사용합니다.**

### 5.6. 정규화/비정규화 실용적 예시 (주문 데이터베이스 설계)

정규화와 비정규화는 이론적인 개념을 넘어 실제 데이터베이스 설계에서 성능과 데이터 무결성 사이의 균형을 맞추는 중요한 의사결정 과정입니다. 다음은 주문 데이터를 예시로 정규화 과정을 살펴보고, 비정규화의 필요성을 이해하는 실용적인 예시입니다.

#### 5.6.1. 비정규화된 초기 주문 테이블 (문제점 파악)

처음에는 모든 주문 관련 정보를 하나의 테이블에 저장한다고 가정해봅시다.

```sql
CREATE TABLE orders_denormalized (
    order_id INT PRIMARY KEY,
    customer_id INT,
    customer_name VARCHAR(100),
    customer_email VARCHAR(100),
    order_date DATE,
    product_id INT,
    product_name VARCHAR(100),
    product_price DECIMAL(10, 2),
    quantity INT,
    total_item_price DECIMAL(10, 2),
    delivery_address VARCHAR(255),
    delivery_status VARCHAR(50)
);
```

**문제점:**
*   **삽입 이상:** 새로운 고객 정보를 추가하려면 반드시 주문이 있어야 합니다.
*   **삭제 이상:** 특정 주문을 삭제하면 해당 주문에만 있던 고객 정보나 상품 정보가 함께 삭제될 수 있습니다.
*   **갱신 이상:** 고객 이름이나 상품 가격이 변경되면, 해당 고객/상품이 포함된 모든 주문 로우를 찾아 일일이 갱신해야 합니다. (데이터 불일치 발생 가능성)
*   **데이터 중복:** 고객 정보, 상품 정보가 주문마다 반복적으로 저장되어 저장 공간 낭비가 심합니다.

#### 5.6.2. 정규화 과정 (1NF, 2NF, 3NF 적용)

위 `orders_denormalized` 테이블을 정규화하여 중복을 제거하고 무결성을 높여봅시다.

**1단계: 1NF 적용 (원자성 확보)**
*   `delivery_address`가 여러 정보를 포함할 수 있다면 분리 (예: `street`, `city`, `zip_code`).
*   여기서는 이미 원자성을 만족한다고 가정하고 다음 단계로 진행합니다.

**2단계: 2NF 적용 (부분 함수 종속 제거)**
*   `orders_denormalized` 테이블의 기본 키는 `order_id`입니다.
*   `product_name`, `product_price`는 `product_id`에만 종속됩니다.
*   `customer_name`, `customer_email`은 `customer_id`에만 종속됩니다.

이를 해결하기 위해 `products` 테이블과 `customers` 테이블을 분리합니다.

```sql
-- products 테이블 (상품 정보)
CREATE TABLE products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    product_price DECIMAL(10, 2) NOT NULL
);

-- customers 테이블 (고객 정보)
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    customer_email VARCHAR(100) UNIQUE
);

-- orders 테이블 (주문 정보 - 주문 자체의 속성)
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT NOT NULL, -- FK to customers
    order_date DATE NOT NULL,
    delivery_address VARCHAR(255),
    delivery_status VARCHAR(50),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- order_items 테이블 (주문 상세 - N:M 관계 해소)
-- 하나의 주문에 여러 상품, 하나의 상품이 여러 주문에 포함될 수 있으므로 별도 테이블 필요
CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT NOT NULL, -- FK to orders
    product_id INT NOT NULL, -- FK to products
    quantity INT NOT NULL,
    total_item_price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
```

**3단계: 3NF 적용 (이행적 함수 종속 제거)**
*   위 `orders` 테이블에서 `customer_id` -> `customer_name`, `customer_email`과 같은 이행적 종속성은 이미 `customers` 테이블로 분리되어 제거되었습니다.
*   만약 `orders` 테이블에 `customer_city`와 같은 컬럼이 있고, `customer_id` -> `customer_address` -> `customer_city`와 같은 관계가 있다면, `customer_city`를 `customers` 테이블로 옮겨야 합니다.

**정규화된 스키마 요약:**
*   `customers` (customer_id PK, customer_name, customer_email)
*   `products` (product_id PK, product_name, product_price)
*   `orders` (order_id PK, customer_id FK, order_date, delivery_address, delivery_status)
*   `order_items` (order_item_id PK, order_id FK, product_id FK, quantity, total_item_price)

#### 5.6.3. 비정규화(Denormalization)의 고려: 분석용 요약 테이블

정규화된 스키마는 데이터 무결성과 중복 제거에 매우 유리하지만, 특정 분석 쿼리에서는 여러 테이블을 `JOIN`해야 하므로 성능 저하가 발생할 수 있습니다. 예를 들어, "일별 총 매출"을 계산하려면 `orders`와 `order_items`를 `JOIN`하고 `GROUP BY`를 수행해야 합니다. 이러한 쿼리가 매우 빈번하게 실행된다면, 의도적으로 비정규화를 고려할 수 있습니다.

**예시: 일별 매출 요약 테이블 (Summary Table)**

```sql
-- 일별 매출 요약 테이블 (물리적인 테이블)
CREATE TABLE daily_sales_summary (
    sale_date DATE PRIMARY KEY,
    total_daily_revenue DECIMAL(15, 2) NOT NULL,
    total_daily_orders INT NOT NULL
);

-- 이 테이블은 매일 배치 작업으로 업데이트됩니다.
-- 예시 쿼리 (매일 자정에 실행)
INSERT INTO daily_sales_summary (sale_date, total_daily_revenue, total_daily_orders)
SELECT
    order_date,
    SUM(oi.quantity * p.product_price) AS total_revenue,
    COUNT(DISTINCT o.order_id) AS total_orders
FROM
    orders o
JOIN
    order_items oi ON o.order_id = oi.order_id
JOIN
    products p ON oi.product_id = p.product_id
WHERE
    order_date = CURDATE() - INTERVAL 1 DAY -- 어제 날짜 데이터만 집계
GROUP BY
    order_date
ON DUPLICATE KEY UPDATE -- 이미 데이터가 있다면 업데이트
    total_daily_revenue = VALUES(total_daily_revenue),
    total_daily_orders = VALUES(total_daily_orders);
```

**비정규화의 장점:**
*   `daily_sales_summary` 테이블을 조회하는 쿼리는 `JOIN` 없이 단일 테이블에서 데이터를 가져오므로 매우 빠릅니다.
*   자주 사용되는 집계 결과를 미리 계산해두어 분석 쿼리의 응답 시간을 단축합니다.

**비정규화의 단점 및 주의점:**
*   **데이터 중복:** `daily_sales_summary` 테이블은 원본 데이터의 집계된 중복을 포함합니다.
*   **데이터 일관성 유지:** 원본 데이터(`orders`, `order_items`, `products`)가 변경될 경우, `daily_sales_summary` 테이블도 반드시 업데이트되어야 합니다. 이를 위해 배치 작업, 트리거, 또는 Materialized View(DBMS 지원 시)와 같은 추가적인 관리 메커니즘이 필요합니다.
*   **저장 공간 증가:** 중복된 데이터로 인해 저장 공간이 증가할 수 있습니다.

**결론:**
정규화는 데이터 무결성과 중복 제거를 위한 기본 원칙이지만, 모든 경우에 최적의 성능을 보장하지는 않습니다. 데이터 분석가는 비즈니스 요구사항과 쿼리 패턴을 고려하여, 필요한 경우 전략적으로 비정규화를 적용하고, 그에 따른 데이터 일관성 유지 방안을 마련해야 합니다. 요약 테이블은 대규모 데이터 분석 환경에서 성능을 최적화하는 매우 효과적인 비정규화 전략 중 하나입니다.