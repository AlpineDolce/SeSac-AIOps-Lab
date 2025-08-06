<h2>SQL 시작하기</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-15

<h2>문서 목표</h2>
<p>이 문서는 데이터 분석가가 SQL을 시작하기 위한 첫 단계로, MySQL 데이터베이스 환경을 구축하고 SQL의 기본적인 개념과 명령어 분류를 이해하는 것을 목표로 합니다. 관계형 데이터베이스의 구조를 파악하고, 데이터베이스 관리 시스템(DBMS)과의 상호작용 방법을 학습합니다.</p>

<h2>목차</h2>

- [1. MySQL 설치: 단계별 가이드](#1-mysql-설치-단계별-가이드)
  - [1.1. MySQL Installer 다운로드 및 설치](#11-mysql-installer-다운로드-및-설치)
  - [1.2. 서버 상세 구성 (Root 비밀번호, 포트 설정 등)](#12-서버-상세-구성-root-비밀번호-포트-설정-등)
- [2. 개발 환경 연동](#2-개발-환경-연동)
  - [2.1. 명령 프롬프트(CLI) 연동을 위한 환경 변수 설정](#21-명령-프롬프트cli-연동을-위한-환경-변수-설정)
  - [2.2. GUI 도구 연결 (DBeaver, MySQL Workbench)](#22-gui-도구-연결-dbeaver-mysql-workbench)
- [3. 데이터베이스 기본 관리 (CLI 기준)](#3-데이터베이스-기본-관리-cli-기준)
  - [3.1. MySQL 서버 접속 및 상태 확인](#31-mysql-서버-접속-및-상태-확인)
  - [3.2. 데이터베이스 생성 및 SQL 스크립트 복원](#32-데이터베이스-생성-및-sql-스크립트-복원)
  - [3.3. 사용자 생성 및 권한 부여 (DCL 기초)](#33-사용자-생성-및-권한-부여-dcl-기초)
- [4. SQL과 관계형 데이터베이스(RDB)의 기본](#4-sql과-관계형-데이터베이스rdb의-기본)
  - [4.1. SQL이란? (Structured Query Language)](#41-sql이란-structured-query-language)
    - [4.1.1. SQL의 역사와 표준](#411-sql의-역사와-표준)
    - [4.1.2. SQL의 역할과 중요성](#412-sql의-역할과-중요성)
    - [4.1.3. SQL 주석 (Comments): 코드 가독성 및 협업](#413-sql-주석-comments-코드-가독성-및-협업)
    - [4.1.4. SQL 표준과 비표준 (ANSI SQL vs. Vendor-Specific SQL)](#414-sql-표준과-비표준-ansi-sql-vs-vendor-specific-sql)
  - [4.2. SQL 명령어의 4가지 분류](#42-sql-명령어의-4가지-분류)
    - [4.2.1. DDL (Data Definition Language): 데이터 정의어](#421-ddl-data-definition-language-데이터-정의어)
    - [4.2.2. DML (Data Manipulation Language): 데이터 조작어](#422-dml-data-manipulation-language-데이터-조작어)
    - [4.2.3. DCL (Data Control Language): 데이터 제어어](#423-dcl-data-control-language-데이터-제어어)
    - [4.2.4. TCL (Transaction Control Language): 트랜잭션 제어어](#424-tcl-transaction-control-language-트랜잭션-제어어)
  - [4.3. 관계형 데이터베이스의 구조](#43-관계형-데이터베이스의-구조)
    - [4.3.1. 테이블(Table), 컬럼(Column), 로우(Row)](#431-테이블table-컬럼column-로우row)
    - [4.3.2. 키(Key)의 종류와 역할 (Primary Key, Foreign Key)](#432-키key의-종류와-역할-primary-key-foreign-key)
    - [4.3.3. 스키마(Schema)와 데이터베이스](#433-스키마schema와-데이터베이스)

---

## 1. MySQL 설치: 단계별 가이드

MySQL은 세계에서 가장 널리 사용되는 오픈소스 관계형 데이터베이스 관리 시스템(RDBMS)입니다. 데이터 분석가에게는 데이터 저장, 관리, 분석을 위한 필수 도구입니다.

### 1.1. MySQL Installer 다운로드 및 설치

MySQL을 설치하는 가장 쉬운 방법은 MySQL Installer를 사용하는 것입니다.

1.  **MySQL 공식 웹사이트 접속:** [MySQL Downloads Page](https://dev.mysql.com/downloads/installer/)에 접속합니다.
2.  **Installer 다운로드:** "MySQL Installer for Windows" 섹션에서 `mysql-installer-community-*.msi` 파일을 다운로드합니다. 일반적으로 웹 커뮤니티 버전을 선택합니다.
3.  **설치 유형 선택:** 다운로드한 파일을 실행하여 설치를 시작합니다.
    *   `Developer Default`: 개발에 필요한 모든 구성 요소를 설치합니다. (권장)
    *   `Server only`: MySQL 서버만 설치합니다.
    *   `Client only`: MySQL 클라이언트 프로그램만 설치합니다.
    *   `Full`: 모든 MySQL 제품을 설치합니다.
    *   `Custom`: 설치할 구성 요소를 직접 선택합니다.
    데이터 분석가라면 `Developer Default`를 선택하는 것이 편리합니다.
4.  **구성 요소 확인 및 설치:** 선택한 설치 유형에 따라 필요한 구성 요소 목록이 표시됩니다. `Execute`를 클릭하여 설치를 진행합니다.

### 1.2. 서버 상세 구성 (Root 비밀번호, 포트 설정 등)

설치 구성 요소가 모두 설치되면, MySQL 서버 및 기타 제품에 대한 설정 단계로 넘어갑니다.

1.  **High Availability (고가용성):** 특별한 설정이 없다면 `Standalone MySQL Server / Classic MySQL Replication`을 선택합니다.
2.  **Type and Networking (유형 및 네트워킹):**
    *   `Config Type`: `Development Computer` (개발 환경에 최적화된 설정)
    *   `Port`: 기본값 `3306`을 유지합니다. 이 포트는 MySQL 서버가 클라이언트 연결을 수신하는 데 사용됩니다.
    *   `Open Windows Firewall port for network access`: 체크하여 방화벽 예외를 추가합니다. (Linux 환경에서는 `ufw allow 3306`과 같은 명령어로 방화벽 포트를 열 수 있습니다.)
*   **`my.ini` (Windows) 또는 `my.cnf` (Linux) 파일:**
    MySQL 설정 파일은 서버의 동작 방식을 제어합니다. 주요 설정으로는 `port` (기본 3306), `datadir` (데이터 저장 경로), `character_set_server` (서버 기본 문자셋) 등이 있습니다. 설치 후 필요에 따라 이 파일을 수정하여 서버 설정을 변경할 수 있습니다.
3.  **Authentication Method (인증 방식):**
    *   `Use Strong Password Encryption for Authentication (RECOMMENDED)`: 강력한 비밀번호 암호화를 사용합니다. (권장)
    *   `Use Legacy Authentication Method (Retain MySQL 5.x Compatibility)`: 이전 버전과의 호환성을 위해 사용하지만, 보안상 취약할 수 있습니다.
    `RECOMMENDED` 옵션을 선택하고 `Next`를 클릭합니다.
4.  **Accounts and Roles (계정 및 역할):**
    *   `MySQL Root Password`: `root` 계정의 비밀번호를 설정합니다. **이 비밀번호는 매우 중요하므로 반드시 기억해야 합니다.**
    *   `Add User`: 필요한 경우 추가 사용자 계정을 생성할 수 있습니다. 실무에서는 `root` 계정 대신 특정 권한을 가진 사용자 계정을 생성하여 사용하는 것이 보안상 권장됩니다.
5.  **Windows Service (Windows 서비스):**
    *   `Configure MySQL Server as a Windows Service`: MySQL 서버를 Windows 서비스로 등록하여 시스템 시작 시 자동으로 실행되도록 합니다.
    *   `Start the MySQL Server at System Startup`: 체크하여 시스템 시작 시 자동 실행되도록 설정합니다.
    *   `Run Windows Service as`: `Standard System Account`를 선택합니다.
6.  **Apply Configuration (구성 적용):** `Execute`를 클릭하여 설정 변경 사항을 적용합니다. 모든 단계가 완료되면 `Finish`를 클릭하여 설치를 마칩니다.

*   **저장 엔진 (Storage Engine):** MySQL은 다양한 저장 엔진을 지원하며, 각 엔진은 데이터 저장 및 처리 방식에 차이가 있습니다.
    *   **`InnoDB` (기본값):** 트랜잭션(ACID), 외래 키, 행 수준 잠금(Row-level locking)을 지원하여 데이터 무결성과 동시성 제어에 강점이 있습니다. 대부분의 애플리케이션과 데이터 분석 환경에서 권장됩니다.
    *   **`MyISAM`:** 트랜잭션을 지원하지 않지만, 읽기(Read) 작업에 특화되어 있습니다. (과거에 많이 사용되었으나, 현재는 `InnoDB`가 대부분의 경우 더 우수합니다.)

*   **MySQL 아키텍처 (간략히):**
    MySQL 서버는 클라이언트-서버 모델로 동작합니다. 클라이언트(MySQL CLI, Workbench, DBeaver, 파이썬 애플리케이션 등)는 네트워크를 통해 MySQL 서버에 접속하여 쿼리를 전송하고, 서버는 이를 처리하여 결과를 클라이언트에 반환합니다.
    *   **클라이언트:** 쿼리를 생성하고 서버에 전송하며, 서버로부터 받은 결과를 사용자에게 표시합니다.
    *   **서버:** 클라이언트의 요청을 받아 SQL 쿼리를 파싱하고, 최적화하며, 실행 계획을 수립합니다. 실제 데이터는 **스토리지 엔진**을 통해 디스크에 저장되거나 읽혀집니다. 서버는 쿼리 실행 후 결과를 클라이언트에 전송합니다.
    이러한 분리된 구조 덕분에 여러 클라이언트가 동시에 데이터베이스에 접근할 수 있으며, 서버는 데이터의 일관성과 무결성을 중앙에서 관리할 수 있습니다.

## 2. 개발 환경 연동

MySQL 서버가 설치되었다면, 이제 이 서버에 접속하여 데이터를 관리하고 쿼리를 실행할 수 있는 개발 환경을 설정해야 합니다.

### 2.1. 명령 프롬프트(CLI) 연동을 위한 환경 변수 설정

명령 프롬프트(CMD)에서 `mysql` 명령어를 바로 실행하려면 MySQL 실행 파일의 경로를 시스템 환경 변수 `PATH`에 추가해야 합니다.

1.  **MySQL 설치 경로 확인:** 일반적으로 `C:\Program Files\MySQL\MySQL Server X.X\bin` (X.X는 버전 번호)입니다.
2.  **환경 변수 편집기 열기:**
    *   Windows 검색창에 "환경 변수"를 입력하고 "시스템 환경 변수 편집"을 선택합니다.
    *   "시스템 속성" 창에서 "환경 변수(N)..." 버튼을 클릭합니다.
3.  **Path 변수 편집:**
    *   "환경 변수" 창의 "시스템 변수" 섹션에서 `Path` 변수를 찾아 선택하고 "편집(I)..." 버튼을 클릭합니다.
    *   "환경 변수 편집" 창에서 "새로 만들기(N)"를 클릭하고, MySQL `bin` 폴더의 경로(예: `C:\Program Files\MySQL\MySQL Server 8.0\bin`)를 추가합니다.
    *   "확인"을 클릭하여 모든 창을 닫습니다.
4.  **확인:** 새 명령 프롬프트 창을 열고 `mysql --version`을 입력하여 MySQL 버전 정보가 올바르게 표시되는지 확인합니다.

*   **클라이언트 캐릭터 셋 설정:** 데이터베이스와 클라이언트 간의 문자 인코딩이 일치하지 않으면 한글 깨짐 등의 문제가 발생할 수 있습니다. MySQL CLI 접속 후 `SET NAMES utf8mb4;` 명령어를 실행하거나, 연결 시 `default-character-set=utf8mb4` 옵션을 사용하여 클라이언트의 문자 셋을 명시적으로 설정하는 것이 좋습니다.

### 2.2. GUI 도구 연결 (DBeaver, MySQL Workbench)

CLI 환경은 강력하지만, GUI 도구는 데이터베이스 구조를 시각적으로 탐색하고, 쿼리 결과를 표 형태로 확인하며, 데이터베이스 관리를 더 편리하게 할 수 있도록 돕습니다.

*   **MySQL Workbench:**
    *   MySQL 공식 GUI 도구로, MySQL 서버와 완벽하게 통합되어 있습니다.
    *   데이터베이스 설계, 개발, 관리 기능을 모두 제공합니다.
    *   설치 시 `Developer Default` 옵션을 선택했다면 이미 설치되어 있습니다.
*   **DBeaver:**
    *   다양한 데이터베이스(MySQL, PostgreSQL, Oracle, SQL Server 등)를 지원하는 범용 데이터베이스 클라이언트입니다.
    *   직관적인 인터페이스와 강력한 기능을 제공하여 데이터 분석가에게 매우 유용합니다.
    *   [DBeaver 공식 웹사이트](https://dbeaver.io/download/)에서 다운로드하여 설치할 수 있습니다.

**CLI vs GUI 도구 (실무적 관점):**
*   **CLI (Command Line Interface):** 자동화된 스크립트 실행, 서버 환경에서의 작업, 원격 접속 시 효율적입니다. 쿼리 작성 및 실행에 집중할 수 있으며, 특정 명령어를 빠르게 반복 실행하는 데 강점이 있습니다.
*   **GUI (Graphical User Interface):** 데이터베이스 구조 시각화, 테이블 데이터 탐색, 쿼리 결과 표 형태로 확인, 데이터베이스 관리 작업(사용자, 권한 등)을 직관적으로 수행하는 데 용이합니다. 초보자가 데이터베이스를 이해하고 다루는 데 큰 도움이 됩니다.

**MySQL CLI `pager` 기능 활용:**
MySQL CLI에서 `SELECT` 쿼리 결과가 너무 길어 화면을 벗어날 때 `pager` 명령어를 사용하면 `less`와 같은 외부 프로그램을 통해 결과를 페이지 단위로 볼 수 있어 편리합니다.
```sql
pager less -S
SELECT * FROM large_table;
```
`-S` 옵션은 긴 줄을 자르지 않고 한 줄로 표시하여 가로 스크롤을 가능하게 합니다.

**GUI 도구 연결 방법 (일반적인 절차):**

1.  GUI 도구를 실행합니다.
2.  새 데이터베이스 연결을 생성합니다.
3.  `Database Type`에서 `MySQL`을 선택합니다.
4.  `Server Hostname`: `127.0.0.1` 또는 `localhost`
5.  `Port`: `3306`
6.  `Username`: `root` (또는 설치 시 생성한 사용자 계정)
7.  `Password`: `root` 계정 비밀번호 (또는 사용자 계정 비밀번호)
8.  `Test Connection`을 클릭하여 연결이 성공하는지 확인합니다.
9.  연결을 저장하고 데이터베이스를 탐색합니다.

## 3. 데이터베이스 기본 관리 (CLI 기준)

GUI 도구도 편리하지만, CLI 환경에서의 기본 관리 명령어는 데이터베이스의 동작 원리를 이해하고 문제 발생 시 빠르게 대처하는 데 필수적입니다.

### 3.1. MySQL 서버 접속 및 상태 확인

MySQL 서버에 접속하여 쿼리를 실행하고 상태를 확인할 수 있습니다.

*   **MySQL 서버 접속:**
    ```bash
    mysql -u root -p
    ```
    `-u`는 사용자 이름(user), `-p`는 비밀번호(password)를 의미합니다. 명령어를 입력하면 비밀번호를 입력하라는 프롬프트가 나타납니다.
*   **MySQL 서버 상태 확인:**
    접속 후 다음 명령어를 입력하여 서버의 현재 상태를 확인할 수 있습니다.
    ```sql
    STATUS;
    ```
    또는
    ```sql
    SHOW VARIABLES LIKE 'port';
    SHOW STATUS LIKE 'Connections';
    ```
*   **접속 종료:**
    ```sql
    EXIT;
    ```
    또는
    ```sql
    QUIT;
    ```

### 3.2. 데이터베이스 생성 및 SQL 스크립트 복원

데이터 분석 프로젝트를 시작하기 전에 데이터를 저장할 데이터베이스를 생성하고, 필요한 경우 기존 SQL 스크립트 파일을 통해 데이터를 복원할 수 있습니다.

*   **데이터베이스 목록 확인:**
    ```sql
    SHOW DATABASES;
    ```
*   **새 데이터베이스 생성:**
    ```sql
    CREATE DATABASE company_db;
    ```
    데이터베이스 이름은 소문자와 언더스코어를 사용하는 것이 일반적입니다.
*   **데이터베이스 생성 시 인코딩 및 콜레이션 설정:**
    한글 데이터 처리 시 문자 깨짐 현상을 방지하고 정렬 규칙을 명확히 하기 위해 데이터베이스 생성 시 인코딩(Character Set)과 콜레이션(Collation)을 명시적으로 지정하는 것이 중요합니다. `utf8mb4`는 이모지 등 다양한 유니코드 문자를 지원하는 MySQL의 문자 집합입니다.
    ```sql
    CREATE DATABASE my_project_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
    ```
    **콜레이션(`COLLATE`)의 실무적 의미:**
    콜레이션은 문자열 데이터의 정렬 및 비교 규칙을 정의합니다. 특히 `_ci` (case-insensitive)가 붙은 콜레이션(예: `utf8mb4_unicode_ci`)은 대소문자를 구분하지 않고 문자열을 비교하고 정렬합니다. 이는 데이터 분석 시 다음과 같은 영향을 미칠 수 있으므로 주의해야 합니다.
    *   **`WHERE` 절 및 `JOIN` 조건:** `WHERE name = 'apple'`과 `WHERE name = 'Apple'`이 동일하게 처리됩니다.
    *   **`GROUP BY` 및 `DISTINCT`:** 'Apple'과 'apple'이 같은 그룹으로 묶이거나 중복으로 간주되어 하나의 값으로 처리됩니다. 이는 데이터 정합성 분석 시 의도치 않은 결과를 초래할 수 있으므로, 대소문자 구분이 필요한 경우에는 `BINARY` 콜레이션을 사용하거나 `COLLATE` 키워드를 쿼리 내에서 명시적으로 지정해야 합니다.
    *   **성능:** 콜레이션 설정은 인덱스 사용에도 영향을 미칠 수 있습니다. 정확한 콜레이션 선택은 데이터의 정확한 처리와 쿼리 성능에 모두 중요합니다.
*   **데이터베이스 선택:**
    쿼리를 실행할 데이터베이스를 선택합니다.
    ```sql
    USE my_project_db;
    ```
*   **SQL 스크립트 복원 (외부 파일 로드):**
    미리 준비된 `.sql` 파일(예: 샘플 데이터, 테이블 구조)을 데이터베이스에 로드할 때 사용합니다.
    ```bash
    # MySQL CLI에서 접속한 상태에서
    SOURCE C:/path/to/your/sample_data.sql;
    ```
    또는 MySQL CLI에 접속하기 전에 명령 프롬프트에서 바로 실행할 수도 있습니다.
    ```bash
    mysql -u root -p my_project_db < C:/path/to/your/sample_data.sql
    ```
    이 명령어는 `my_project_db` 데이터베이스에 `sample_data.sql` 파일의 내용을 실행합니다.

### 3.3. 사용자 생성 및 권한 부여 (DCL 기초)

실무에서는 `root` 계정 대신 특정 권한을 가진 사용자 계정을 생성하여 사용하는 것이 보안상 권장됩니다. DCL(Data Control Language)은 데이터베이스 접근 권한을 제어하는 명령어입니다.

*   **새 사용자 생성:**
    ```sql
    CREATE USER 'new_user'@'localhost' IDENTIFIED BY 'user_password';
    ```
    `'new_user'`는 사용자 이름, `'localhost'`는 접속을 허용할 호스트(여기서는 로컬), `'user_password'`는 비밀번호입니다. `'%'`를 사용하면 모든 호스트에서 접속을 허용합니다.
*   **권한 부여:**
    `my_project_db` 데이터베이스의 모든 테이블에 대해 `new_user`에게 모든 권한을 부여합니다.
    ```sql
    GRANT ALL PRIVILEGES ON my_project_db.* TO 'new_user'@'localhost';
    ```
    특정 권한만 부여할 수도 있습니다. (예: `SELECT`, `INSERT`, `UPDATE`, `DELETE`)
    ```sql
    GRANT SELECT, INSERT ON my_project_db.employees TO 'new_user'@'localhost';
    ```
*   **권한 적용:**
    권한 변경 사항을 즉시 적용합니다.
    ```sql
    FLUSH PRIVILEGES;
    ```
    **참고:** `FLUSH PRIVILEGES`는 권한 테이블을 다시 로드하여 변경 사항을 즉시 적용하지만, 운영 중인 시스템에서는 서버에 부하를 줄 수 있습니다. MySQL 8.0부터는 `GRANT`나 `REVOKE` 명령어가 자동으로 권한 캐시를 갱신하므로, 대부분의 경우 `FLUSH PRIVILEGES`를 명시적으로 호출할 필요가 없습니다. 하지만 이전 버전과의 호환성이나 특정 상황에서는 여전히 유용할 수 있습니다.

*   **`FLUSH PRIVILEGES`의 대안 (MySQL 8.0+):**
    MySQL 8.0 이상에서는 `GRANT` 또는 `REVOKE` 명령어가 자동으로 권한 캐시를 갱신하므로, 일반적으로 `FLUSH PRIVILEGES`를 명시적으로 호출할 필요가 없습니다. 만약 특정 이유로 권한 캐시를 수동으로 갱신해야 한다면, `FLUSH PRIVILEGES` 대신 `SET GLOBAL read_only = ON;` 후 `SET GLOBAL read_only = OFF;`와 같이 서버를 잠시 읽기 전용 모드로 전환하는 방법을 고려할 수 있습니다. 이는 서버 전체에 미치는 영향을 최소화하면서 권한을 갱신하는 방법 중 하나입니다. (하지만 이 역시 운영 중인 시스템에서는 신중하게 사용해야 합니다.)

*   **사용자 삭제:**
    ```sql
    DROP USER 'new_user'@'localhost';
    ```

*   **최소 권한의 원칙 (Principle of Least Privilege)과 실무:**
    데이터 분석가는 실무에서 운영 데이터베이스에 접근하는 경우가 많습니다. 이때 보안과 안정성을 위해 **"최소 권한의 원칙"**을 반드시 따라야 합니다. 이는 사용자에게 업무 수행에 필요한 최소한의 권한만 부여하는 것을 의미합니다.
    *   **분석가의 일반적인 권한:** 보통 데이터 분석가는 데이터 조회를 위한 `SELECT` 권한만 가집니다. 데이터 변경(`INSERT`, `UPDATE`, `DELETE`) 권한은 엄격히 제한되며, 필요한 경우 별도의 분석용 샌드박스 데이터베이스에서 작업을 수행합니다.
    *   **`GRANT` 사용 시:** `GRANT ALL PRIVILEGES`는 개발 환경이나 학습 목적으로는 편리하지만, 실무에서는 절대 사용해서는 안 됩니다. `GRANT SELECT ON database.table TO 'user'@'host';`와 같이 필요한 권한만 명시적으로 부여하는 것이 안전합니다.

## 4. SQL과 관계형 데이터베이스(RDB)의 기본

데이터 분석가에게 SQL은 데이터와 소통하는 언어이며, 관계형 데이터베이스는 그 데이터가 저장되는 구조입니다. 이 둘의 기본 개념을 이해하는 것은 매우 중요합니다.

### 4.1. SQL이란? (Structured Query Language)

SQL은 관계형 데이터베이스 시스템에서 데이터를 관리하고 처리하기 위해 설계된 표준 언어입니다.

#### 4.1.1. SQL의 역사와 표준

*   **역사:** 1970년대 IBM에서 개발된 SEQUEL(Structured English Query Language)에서 시작하여, 1986년 ANSI(미국 표준 협회)에 의해 표준화되었습니다. 이후 ISO(국제 표준화 기구)에서도 표준으로 채택되어 현재까지 발전하고 있습니다.
*   **표준:** SQL은 표준이 존재하지만, 각 데이터베이스 벤더(MySQL, PostgreSQL, Oracle, SQL Server 등)는 표준 SQL에 자신들만의 확장 기능을 추가하여 사용합니다. 따라서 특정 DBMS에서만 동작하는 문법이 있을 수 있습니다.

*   **SQL 방언(Dialect)과 호환성 문제:**
    SQL은 표준이 있지만, 각 데이터베이스 관리 시스템(DBMS) 벤더(MySQL, PostgreSQL, Oracle, SQL Server 등)는 표준 SQL에 자신들만의 고유한 기능과 문법(방언, Dialect)을 추가합니다. 이로 인해 특정 DBMS에서 작성된 쿼리가 다른 DBMS에서는 동작하지 않거나, 예상과 다르게 동작할 수 있습니다. 데이터 분석가는 여러 DBMS를 다룰 가능성이 높으므로, 이러한 차이를 이해하는 것이 중요합니다.
    *   **실무적 어려움 (예시):**

        | 기능/개념 | MySQL | PostgreSQL | SQL Server | Oracle | 비고 |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | `FULL OUTER JOIN` | 직접 지원 안 함 (LEFT JOIN + UNION + RIGHT JOIN으로 구현) | 직접 지원 | 직접 지원 | 직접 지원 | MySQL에서 구현 시 `UNION ALL`과 `WHERE IS NULL` 조합이 더 효율적일 수 있음 |
        | `TOP N` 로우 | `LIMIT N` | `LIMIT N` | `TOP N` | `ROWNUM <= N` | |
        | 날짜/시간 함수 | `NOW()`, `CURDATE()`, `DATE_FORMAT()` | `NOW()`, `CURRENT_DATE`, `TO_CHAR()` | `GETDATE()`, `CURRENT_TIMESTAMP`, `FORMAT()` | `SYSDATE`, `TO_CHAR()` | 각 DBMS마다 함수 이름과 사용법이 다름 |
        | 문자열 연결 | `CONCAT()` | `CONCAT()` 또는 `||` | `+` | `||` | |
        | `IF-ELSE` 로직 | `IF()`, `CASE WHEN` | `CASE WHEN` | `CASE WHEN` | `CASE WHEN` | `IF()`는 MySQL 특화 |
        | 대소문자 구분 (테이블/컬럼명) | OS 및 `lower_case_table_names` 설정에 따라 다름 (Windows는 미구분, Linux는 구분) | 기본적으로 구분 | 기본적으로 미구분 (Collation에 따라 다름) | 기본적으로 구분 | 쿼리 작성 시 일관성 유지 권장 |
        | `TRUNCATE TABLE` | DDL (롤백 불가, AUTO_INCREMENT 초기화) | DDL (롤백 불가, AUTO_INCREMENT 초기화) | DDL (롤백 불가, IDENTITY 초기화) | DDL (롤백 불가, SEQUENCE 초기화) | `DELETE FROM`과 차이점 명확히 인지 |

    *   **해결 전략:**
        *   **표준 SQL 우선 사용:** 가능한 한 표준 SQL 문법을 사용하여 특정 DBMS에 종속되지 않는 쿼리를 작성합니다. 이는 코드의 이식성을 높이는 가장 기본적인 방법입니다.
        *   **DBMS별 기능 명시:** 특정 DBMS의 고유 기능을 사용해야 할 경우, 해당 기능이 다른 DBMS에서는 동작하지 않을 수 있음을 주석으로 명시하거나, 별도의 모듈로 분리하여 관리합니다.
        *   **추상화 계층 활용:** 파이썬의 SQLAlchemy와 같은 ORM(Object-Relational Mapping) 라이브러리는 DBMS별 방언 차이를 추상화하여 개발자가 동일한 코드로 여러 DBMS와 상호작용할 수 있도록 돕습니다. 이는 특히 애플리케이션 개발 시 유용합니다.
        *   **데이터 분석 환경 고려:** 데이터 분석 환경에서는 특정 DBMS의 분석 함수나 성능 최적화 기능을 적극 활용하되, 다른 환경으로의 이식 가능성을 항상 염두에 두어야 합니다. 필요시 데이터 추출 및 변환(ETL) 과정에서 DBMS 간의 호환성 문제를 해결하는 로직을 추가합니다.

#### 4.1.2. SQL의 역할과 중요성

*   **데이터 정의:** 테이블, 뷰, 인덱스 등 데이터베이스 객체의 구조를 정의합니다.
*   **데이터 조작:** 데이터 삽입, 조회, 수정, 삭제 등 실제 데이터를 다룹니다.
*   **데이터 제어:** 사용자 권한 관리, 트랜잭션 제어 등 데이터베이스의 보안과 무결성을 관리합니다.
*   **데이터 분석:** 복잡한 쿼리를 통해 데이터를 집계하고 분석하여 비즈니스 인사이트를 도출합니다.
데이터 분석가에게 SQL은 데이터에 직접 접근하고, 원하는 형태로 가공하며, 분석에 필요한 데이터를 추출하는 핵심 도구입니다.

#### 4.1.3. SQL 주석 (Comments): 코드 가독성 및 협업

SQL 주석은 쿼리 코드 내에 설명을 추가하여 코드의 가독성을 높이고, 다른 사람과의 협업을 용이하게 합니다. 데이터베이스 시스템은 주석을 무시하고 쿼리를 실행합니다.

*   **한 줄 주석 (`--`):**
    `--` 뒤에 오는 내용은 해당 줄의 끝까지 주석으로 처리됩니다.
    ```sql
    SELECT employee_id, first_name -- 직원 ID와 이름 조회
    FROM employees;
    ```

*   **여러 줄 주석 (`/* ... */`):**
    `/*`와 `*/` 사이에 오는 모든 내용은 여러 줄에 걸쳐 주석으로 처리됩니다.
    ```sql
    /*
    이 쿼리는 employees 테이블에서
    급여가 50000 이상인 직원을 조회합니다.
    작성자: Alpine_Dolce
    */
    SELECT employee_id, first_name, salary
    FROM employees
    WHERE salary >= 50000;
    ```

**실무적 활용:**
*   **쿼리 설명:** 복잡한 쿼리 로직이나 특정 조건에 대한 설명을 추가하여 나중에 쿼리를 다시 보거나 다른 사람이 이해할 때 도움을 줍니다.
*   **변경 이력:** 쿼리 수정 시 변경된 내용, 날짜, 작성자 등을 기록하여 버전 관리에 준하는 효과를 얻을 수 있습니다.
*   **디버깅:** 특정 SQL 구문을 임시로 비활성화할 때 주석 처리하여 디버깅에 활용할 수 있습니다.
*   **협업:** 팀원 간 쿼리를 공유할 때 주석을 통해 의도를 명확히 전달하고 소통 오류를 줄일 수 있습니다.

#### 4.1.4. SQL 표준과 비표준 (ANSI SQL vs. Vendor-Specific SQL)

*   **SQL 코딩 컨벤션의 중요성 (실무적 관점):**
    실무는 혼자 일하는 공간이 아닌, 여러 동료와 함께 코드를 공유하고 유지보수하는 협업의 장입니다. 따라서 일관된 **코딩 컨벤션(Coding Convention)**을 따르는 것은 매우 중요합니다. 깔끔하고 예측 가능한 코드는 가독성을 높여 실수를 줄이고, 동료가 코드를 쉽게 이해하고 수정할 수 있도록 돕습니다.
    *   **일반적인 SQL 코딩 컨벤션 예시:**
        *   **키워드는 대문자로 작성:** `SELECT`, `FROM`, `WHERE`, `GROUP BY` 등 SQL 예약어는 대문자로 작성하여 코드의 구조를 명확히 합니다.
        *   **테이블, 컬럼명은 소문자로 작성:** `employees`, `department_name` 등은 소문자로 작성하여 키워드와 구분합니다.
        *   **적절한 들여쓰기:** `JOIN`, 서브쿼리 등은 논리적인 구조에 맞게 들여쓰기하여 가독성을 높입니다.
        *   **의미 있는 별칭(Alias) 사용:** `e`, `d` 와 같이 너무 짧은 별칭보다는 `emp`, `dept` 처럼 의미를 유추할 수 있는 별칭을 사용하는 것이 좋습니다.
    *   팀에 합류하면 가장 먼저 팀의 코딩 컨벤션을 확인하고 따르는 것이 원활한 협업의 첫걸음입니다.

SQL은 ANSI(미국 표준 협회)와 ISO(국제 표준화 기구)에 의해 표준화된 언어이지만, 실제로는 각 데이터베이스 관리 시스템(DBMS) 벤더(MySQL, PostgreSQL, Oracle, SQL Server 등)가 표준 SQL에 자신들만의 고유한 기능과 문법(방언, Dialect)을 추가하여 사용합니다. 데이터 분석가는 이러한 표준과 비표준의 차이를 이해하는 것이 중요합니다.

*   **ANSI SQL (표준 SQL):**
    *   대부분의 DBMS에서 공통적으로 지원하는 문법과 기능을 정의합니다.
    *   `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `INNER JOIN`, `LEFT JOIN`, `CREATE TABLE`, `INSERT`, `UPDATE`, `DELETE`, `COMMIT`, `ROLLBACK` 등 기본적인 DDL, DML, TCL 명령어는 표준 SQL에 속합니다.
    *   **장점:** 코드의 이식성이 높아 다른 DBMS로 전환하거나 여러 DBMS를 동시에 다룰 때 유리합니다.
    *   **단점:** 특정 DBMS의 강력한 최적화 기능이나 고유한 분석 함수를 활용하지 못할 수 있습니다.

*   **Vendor-Specific SQL (비표준 SQL 또는 방언):**
    *   각 DBMS 벤더가 자사 제품의 성능, 기능, 특정 사용 사례에 최적화하기 위해 추가한 고유한 문법과 기능입니다.
    *   **예시:**
        *   **MySQL:** `LIMIT` (로우 제한), `INSERT ... ON DUPLICATE KEY UPDATE`, `GROUP_CONCAT()`, `IF()` 함수, `LOAD DATA INFILE` 등
        *   **PostgreSQL:** `LATERAL JOIN`, `JSONB` 데이터 타입 및 관련 함수, `GENERATE_SERIES()`, `DISTINCT ON` 등
        *   **Oracle:** `ROWNUM` (로우 제한), `CONNECT BY` (계층 쿼리), `DECODE()` 함수, `Materialized View` 등
        *   **SQL Server:** `TOP` (로우 제한), `PIVOT`/`UNPIVOT`, `APPLY` 연산자 등
    *   **장점:** 특정 DBMS의 강력한 기능을 활용하여 성능을 극대화하거나 복잡한 문제를 효율적으로 해결할 수 있습니다.
    *   **단점:** 코드의 이식성이 낮아 다른 DBMS로 전환 시 코드 수정이 필요하며, 학습 비용이 증가할 수 있습니다.

**데이터 분석 실무에서의 고려사항:**
*   **이식성 vs. 성능/기능:** 분석 프로젝트의 요구사항에 따라 표준 SQL을 고수할지, 아니면 특정 DBMS의 비표준 기능을 적극적으로 활용할지 결정해야 합니다. 초기 단계에서는 표준 SQL을 우선하되, 성능 병목이나 특정 기능이 필요할 때 비표준 기능을 도입하는 것이 일반적입니다.
*   **문서화:** 비표준 SQL을 사용할 경우, 해당 부분이 어떤 DBMS의 어떤 기능인지 명확히 문서화하여 다른 팀원이나 미래의 자신을 위해 기록을 남기는 것이 중요합니다.
*   **추상화 계층:** 파이썬의 SQLAlchemy와 같은 라이브러리는 DBMS별 방언 차이를 추상화하여 개발자가 동일한 코드로 여러 DBMS와 상호작용할 수 있도록 돕습니다. 이는 특히 애플리케이션 개발 시 유용합니다.

*   **데이터 분석가를 위한 RDBMS 선택 고려사항:**
    데이터 분석 프로젝트에서는 목적에 따라 적합한 RDBMS를 선택하는 것이 중요합니다.
    *   **MySQL:** 웹 서비스 백엔드에 널리 사용되며, 설치 및 관리가 비교적 쉽습니다. 대규모 트랜잭션 처리보다는 읽기(Read) 성능에 강점이 있습니다.
    *   **PostgreSQL:** 객체-관계형 데이터베이스로, SQL 표준을 잘 준수하며 확장성이 뛰어납니다. 복잡한 분석 쿼리, 지리 정보 시스템(GIS), JSONB와 같은 반정형 데이터 처리에 강점이 있어 데이터 분석 및 과학 분야에서 인기가 높습니다.
    *   **SQL Server (Microsoft):** Windows 환경에 최적화되어 있으며, BI(Business Intelligence) 도구와의 연동이 강력합니다.
    *   **Oracle:** 대규모 엔터프라이즈 환경에서 가장 널리 사용되는 상용 DBMS로, 강력한 안정성과 성능을 자랑합니다.
    데이터 분석가는 주로 데이터 추출 및 분석에 집중하므로, 분석 함수 지원, JSON 데이터 처리 능력, 그리고 커뮤니티 지원 등을 고려하여 DBMS를 선택할 수 있습니다.

### 4.2. SQL 명령어의 4가지 분류

SQL 명령어는 기능에 따라 크게 4가지로 분류할 수 있습니다.

#### 4.2.1. DDL (Data Definition Language): 데이터 정의어

데이터베이스 객체의 구조를 정의, 변경, 삭제하는 명령어입니다. 데이터 자체보다는 데이터가 저장될 "틀"을 다룹니다.

*   `CREATE`: 데이터베이스, 테이블, 뷰, 인덱스 등을 생성합니다.
*   `ALTER`: 기존 데이터베이스 객체의 구조를 변경합니다. (예: 테이블에 컬럼 추가/삭제)
*   `DROP`: 데이터베이스, 테이블, 뷰, 인덱스 등을 삭제합니다.
*   `TRUNCATE`: 테이블의 모든 데이터를 삭제하고, 테이블 구조는 남겨둡니다. (DDL로 분류되기도 함)

#### 4.2.2. DML (Data Manipulation Language): 데이터 조작어

데이터베이스 내의 데이터를 조회, 삽입, 수정, 삭제하는 명령어입니다.

*   `SELECT`: 데이터베이스에서 데이터를 조회합니다. (가장 많이 사용)
*   `INSERT`: 테이블에 새로운 데이터를 삽입합니다.
*   `UPDATE`: 테이블의 기존 데이터를 수정합니다.
*   `DELETE`: 테이블의 데이터를 삭제합니다.

#### 4.2.3. DCL (Data Control Language): 데이터 제어어

데이터베이스에 대한 접근 권한 및 보안을 제어하는 명령어입니다.

*   `GRANT`: 사용자에게 특정 권한을 부여합니다.
*   `REVOKE`: 사용자에게 부여된 권한을 회수합니다.

#### 4.2.4. TCL (Transaction Control Language): 트랜잭션 제어어

데이터의 일관성과 무결성을 유지하기 위해 트랜잭션을 제어하는 명령어입니다. 트랜잭션은 데이터베이스의 논리적인 작업 단위입니다. **데이터 분석가가 DML 작업을 수행할 때 데이터의 무결성을 보장하기 위해 반드시 이해하고 활용해야 하는 매우 중요한 개념입니다.**

*   `COMMIT`: 트랜잭션 내의 모든 변경 사항을 영구적으로 데이터베이스에 반영합니다.
*   `ROLLBACK`: 트랜잭션 내의 모든 변경 사항을 취소하고 이전 상태로 되돌립니다.
*   `SAVEPOINT`: 트랜잭션 내에서 롤백할 지점을 지정합니다.

### 4.3. 관계형 데이터베이스의 구조

관계형 데이터베이스(RDB)는 데이터를 테이블(Table)이라는 2차원 구조로 저장하고, 이 테이블들 간의 관계를 통해 데이터를 관리합니다.

#### 4.3.1. 테이블(Table), 컬럼(Column), 로우(Row)

*   **테이블 (Table):** 데이터를 저장하는 기본 단위로, 스프레드시트의 시트와 유사합니다. 특정 주제에 대한 데이터를 구조화하여 저장합니다.
*   **컬럼 (Column):** 테이블의 각 열을 의미하며, 특정 속성(Attribute)을 나타냅니다. 각 컬럼은 고유한 이름과 데이터 타입(예: 숫자, 문자열, 날짜)을 가집니다.
*   **로우 (Row):** 테이블의 각 행을 의미하며, 하나의 레코드(Record) 또는 튜플(Tuple)이라고도 합니다. 각 로우는 특정 개체(Entity)에 대한 모든 속성 값의 집합입니다.
*   **인덱스 (Index):** 테이블의 특정 컬럼에 대한 검색 속도를 향상시키는 데이터 구조입니다. 책의 찾아보기(색인)와 유사하며, `WHERE` 절이나 `JOIN` 조건에서 자주 사용되는 컬럼에 인덱스를 생성하면 쿼리 성능을 획기적으로 향상시킬 수 있습니다. (자세한 내용은 [0527정리.md의 '6.2. 인덱스 최적화 전략'](./0527정리.md#62-인덱스-최적화-전략) 섹션을 참조하세요.)

#### 4.3.2. 키(Key)의 종류와 역할 (Primary Key, Foreign Key)

키는 테이블 간의 관계를 설정하고 데이터의 무결성을 보장하는 데 사용되는 중요한 개념입니다.

*   **기본 키 (Primary Key, PK):**
    *   테이블의 각 로우를 고유하게 식별할 수 있는 하나 이상의 컬럼 조합입니다.
    *   `NULL` 값을 가질 수 없으며(NOT NULL), 중복된 값을 가질 수 없습니다(UNIQUE).
    *   테이블당 하나의 기본 키만 존재할 수 있습니다.
    *   데이터의 무결성을 보장하고, 다른 테이블과의 관계를 설정하는 데 사용됩니다.
*   **외래 키 (Foreign Key, FK):**
    *   다른 테이블의 기본 키를 참조하는 컬럼입니다.
    *   두 테이블 간의 관계를 설정하며, 참조 무결성(Referential Integrity)을 보장합니다. 즉, 외래 키 값은 참조하는 테이블의 기본 키 값으로 존재하거나 `NULL`이어야 합니다.
    *   예: `employees` 테이블의 `department_id`가 `departments` 테이블의 `department_id`를 외래 키로 참조.

#### 4.3.3. 스키마(Schema)와 데이터베이스

*   **데이터베이스 (Database):** 관련된 데이터와 그 데이터를 관리하는 시스템(DBMS)의 집합입니다. 여러 스키마를 포함할 수 있습니다.
*   **스키마 (Schema):** 데이터베이스 내에서 데이터의 구조와 제약조건을 정의한 논리적인 단위입니다. 테이블, 뷰, 인덱스, 저장 프로시저 등 데이터베이스 객체들의 집합을 의미합니다. MySQL에서는 데이터베이스와 스키마가 거의 동일한 개념으로 사용됩니다. 즉, `CREATE DATABASE`는 `CREATE SCHEMA`와 동일하게 동작합니다.
