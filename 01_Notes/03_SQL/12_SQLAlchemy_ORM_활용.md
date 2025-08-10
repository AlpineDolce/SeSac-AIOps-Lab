<h2>Python과 SQL 연동: SQLAlchemy ORM 활용 (실무 활용 중심)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-29

<h2>문서 목표</h2>
<p>이 문서는 파이썬에서 가장 널리 사용되는 ORM(Object-Relational Mapping) 라이브러리인 <strong>SQLAlchemy</strong>를 활용하여 데이터베이스와 상호작용하는 방법을 심도 있게 다룹니다. 특히 ORM의 핵심 개념인 객체 지향적 데이터 접근 방식과 마이그레이션 도구인 Alembic을 활용한 스키마 관리 방법을 상세한 예제와 함께 설명하여, 파이썬 기반의 견고한 데이터베이스 애플리케이션 개발의 기초를 다지는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. ORM (Object-Relational Mapping)](#1-orm-object-relational-mapping)
  - [1.1. ORM의 개념과 장점](#11-orm의-개념과-장점)
  - [1.2. ORM의 단점](#12-orm의-단점)
- [2. SQLAlchemy 핵심 개념](#2-sqlalchemy-핵심-개념)
  - [2.1. 설치 및 기본 설정](#21-설치-및-기본-설정)
  - [2.2. 엔진 (Engine)](#22-엔진-engine)
  - [2.3. 메타데이터 (Metadata)](#23-메타데이터-metadata)
  - [2.4. 테이블 정의 (Table Definition)](#24-테이블-정의-table-definition)
  - [2.5. 세션 (Session)](#25-세션-session)
  - [2.6. 선언적 베이스 (Declarative Base)](#26-선언적-베이스-declarative-base)
- [3. SQLAlchemy ORM을 이용한 CRUD 작업](#3-sqlalchemy-orm을-이용한-crud-작업)
  - [3.1. 모델 정의](#31-모델-정의)
  - [3.2. 테이블 생성](#32-테이블-생성)
  - [3.3. 데이터 삽입 (Create)](#33-데이터-삽입-create)
  - [3.4. 데이터 조회 (Read)](#34-데이터-조회-read)
  - [3.5. 데이터 수정 (Update)](#35-데이터-수정-update)
  - [3.6. 데이터 삭제 (Delete)](#36-데이터-삭제-delete)
  - [3.7. 관계 설정 (Relationships)](#37-관계-설정-relationships)
- [4. Alembic을 이용한 데이터베이스 마이그레이션](#4-alembic을-이용한-데이터베이스-마이그레이션)
  - [4.1. 마이그레이션의 개념과 필요성](#41-마이그레이션의-개념과-필요성)
  - [4.2. Alembic 설치 및 초기화](#42-alembic-설치-및-초기화)
  - [4.3. 마이그레이션 스크립트 생성](#43-마이그레이션-스크립트-생성)
  - [4.4. 마이그레이션 실행](#44-마이그레이션-실행)
  - [4.5. 마이그레이션 되돌리기](#45-마이그레이션-되돌리기)

---

## 1. ORM (Object-Relational Mapping) 심층 탐구

### 1.1. ORM이 해결하려는 근본 문제: '객체-관계 불일치'

ORM을 이해하기 위해서는 먼저 **객체-관계 불일치(Object-Relational Impedance Mismatch)** 라는 근본적인 문제를 알아야 합니다. 이는 객체 지향 프로그래밍(OOP)의 패러다임과 관계형 데이터베이스(RDB)의 패러다임이 서로 다른 모델을 가지고 있어 발생하는 개념적 차이를 의미합니다.

| 개념 | 객체 지향 패러다임 | 관계형 데이터베이스 패러다임 | 불일치 지점 |
| :--- | :--- | :--- | :--- |
| **데이터 단위** | **객체(Object)**: 속성과 행위를 가짐 | **로우(Row)**: 데이터 값들의 집합 | 객체는 상태와 행동을 모두 갖지만, 로우는 상태만 가짐 |
| **구조** | **클래스(Class)**: 데이터 구조와 메서드를 정의 | **테이블(Table)**: 정해진 컬럼들의 집합 | 클래스는 상속, 다형성 등 복잡한 구조가 가능 |
| **관계** | **참조(Reference)**: 객체가 다른 객체를 직접 참조 | **외래 키(Foreign Key)**: 다른 테이블의 기본 키를 값으로 저장 | 객체는 복잡한 그래프 형태의 관계를 형성 가능 |
| **식별성** | **메모리 주소/ID**: 객체는 고유한 식별성을 가짐 | **기본 키(Primary Key)**: 테이블 내에서 로우를 식별 | 객체의 식별성과 DB의 식별성은 별개로 관리됨 |

이러한 불일치 때문에, 개발자는 애플리케이션의 객체 모델을 데이터베이스의 테이블 구조에 맞게 변환하는, 반복적이고 오류가 발생하기 쉬운 '데이터 매핑' 코드를 직접 작성해야 했습니다. **ORM은 바로 이 변환 과정을 자동화하여, 개발자가 객체 모델에만 집중할 수 있도록 돕는 기술입니다.**

### 1.2. ORM의 핵심 아이디어: 코드 중심의 데이터 관리

ORM은 SQL 중심의 데이터 접근 방식을 코드(객체) 중심의 접근 방식으로 전환합니다. 다음은 원시 SQL과 ORM(SQLAlchemy 예시)의 차이를 보여주는 간단한 예시입니다.

**시나리오: 새로운 직원을 추가하고, ID로 조회하기**

<details>
<summary><b>1. 원시 SQL (pymysql) 방식</b></summary>

```python
# 데이터베이스 연결 및 커서 생성
conn = pymysql.connect(...)
cursor = conn.cursor()

# 데이터 삽입 (Create)
insert_sql = "INSERT INTO employees (name, position) VALUES (%s, %s)"
cursor.execute(insert_sql, ('Alice', 'Engineer'))
conn.commit()
employee_id = cursor.lastrowid

# 데이터 조회 (Read)
select_sql = "SELECT id, name, position FROM employees WHERE id = %s"
cursor.execute(select_sql, (employee_id,))
row = cursor.fetchone()

# 결과를 수동으로 객체나 딕셔너리로 매핑
if row:
    employee_dict = {'id': row[0], 'name': row[1], 'position': row[2]}
    print(f"조회된 직원: {employee_dict}")

cursor.close()
conn.close()
```
- **문제점**: SQL 문자열을 직접 작성해야 하며, 테이블 구조가 바뀌면 모든 SQL 쿼리를 수정해야 합니다. 조회 결과를 다시 객체나 딕셔너리로 변환하는 과정이 번거롭습니다.
</details>

<details>
<summary><b>2. ORM (SQLAlchemy) 방식</b></summary>

```python
# ORM 모델 정의 (애플리케이션 코드)
class Employee(Base):
    __tablename__ = 'employees'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    position = Column(String(50))

# 세션 생성
session = Session()

# 데이터 삽입 (Create) - 객체 생성
new_employee = Employee(name='Alice', position='Engineer')
session.add(new_employee)
session.commit()

# 데이터 조회 (Read) - 객체 쿼리
retrieved_employee = session.query(Employee).filter_by(id=new_employee.id).first()

# 조회 결과는 이미 완벽한 객체
if retrieved_employee:
    print(f"조회된 직원: ID={retrieved_employee.id}, Name={retrieved_employee.name}")

session.close()
```
- **장점**: SQL이 코드에서 사라졌습니다. `Employee`라는 파이썬 클래스를 통해 모든 DB 작업을 수행합니다. `retrieved_employee`는 단순한 데이터 묶음이 아닌, 메서드 등을 가질 수 있는 완전한 객체입니다.
</details>

### 1.3. ORM의 장점 (심화)

-   **생산성 및 유지보수성 향상**: CRUD 작업을 위한 반복적인 SQL 코드를 작성할 필요가 없습니다. 데이터베이스 스키마가 변경되면, 파이썬 모델 클래스만 수정하면 되므로 애플리케이션 코드 전체에 흩어져 있는 SQL을 찾아다닐 필요가 없습니다.
-   **진정한 객체 지향적 접근**: 상속, 연관 관계 등 객체 지향의 강력한 모델링 기법을 데이터베이스 설계에 반영할 수 있습니다. 예를 들어, `Employee`와 `Department` 객체 간의 관계를 `relationship()`으로 설정하면, `employee.department`와 같이 직관적으로 관련 데이터에 접근할 수 있습니다.
-   **데이터베이스 독립성**: SQLAlchemy와 같은 ORM은 '방언(Dialect)'을 사용하여 특정 데이터베이스(MySQL, PostgreSQL, SQLite 등)에 맞는 SQL을 자동으로 생성합니다. 따라서 개발 초기에는 SQLite를 사용하다가 프로덕션 환경에서는 PostgreSQL로 쉽게 전환할 수 있습니다.
-   **내장된 보안 기능**: ORM은 모든 입력을 자동으로 파라미터화하여 처리하므로, 개발자가 실수로 SQL Injection 취약점을 만드는 것을 원천적으로 방지합니다.

### 1.4. ORM의 단점과 실무적 과제

ORM은 매우 강력하지만, 그 추상화 뒤에 숨겨진 동작을 이해하지 못하면 심각한 성능 문제를 야기할 수 있습니다.

-   **학습 곡선**: ORM을 제대로 사용하려면 단순히 메서드를 아는 것을 넘어, 세션(Session), 트랜잭션, 캐싱, 그리고 특히 **로딩 전략(Loading Strategies)** 의 내부 동작 원리를 이해해야 합니다.
-   **복잡한 쿼리의 한계**: 통계, 분석 등을 위한 복잡한 집계 쿼리, 윈도우 함수, CTE(Common Table Expressions) 등은 ORM만으로 표현하기 매우 어렵거나 불가능할 수 있습니다.
-   **추상화의 누수(Leaky Abstraction)**: 성능 문제를 해결하거나 특정 DB 기능을 사용하려면, 결국 ORM이 어떤 SQL을 생성하는지 확인하고 조정해야 할 때가 많습니다. 이는 ORM의 추상화가 완벽하지 않다는 것을 의미합니다.

#### 1.4.1. 심층 탐구: N+1 문제와 로딩 전략

ORM 사용 시 가장 흔하게 발생하는 성능 병목은 **N+1 문제**입니다. 이는 관계가 설정된 객체를 조회할 때 발생하는 비효율적인 쿼리 실행 패턴을 말합니다.

-   **상황**: 모든 부서(`Department`)를 조회하고, 각 부서에 속한 직원(`Employee`)들의 이름을 출력하는 경우.
-   **N+1 문제 발생**:
    1.  모든 부서를 가져오기 위해 **1번**의 쿼리를 실행합니다. (`SELECT * FROM departments;`)
    2.  첫 번째 부서의 직원 목록(`dept1.employees`)에 접근할 때, 해당 부서의 직원들을 가져오기 위한 추가 쿼리가 실행됩니다.
    3.  두 번째 부서의 직원 목록(`dept2.employees`)에 접근할 때, 또 다른 추가 쿼리가 실행됩니다.
    4.  ... 
    5.  N개의 부서에 대해 총 **N번**의 추가 쿼리가 발생합니다.
    -   결과적으로 **총 1 + N 번의 쿼리**가 데이터베이스로 전송되어 심각한 성능 저하를 유발합니다.

SQLAlchemy는 이 문제를 해결하기 위해 **즉시 로딩(Eager Loading)** 전략을 제공합니다.

<details>
<summary><b>1. 지연 로딩 (Lazy Loading) - N+1 문제 발생 (기본값)</b></summary>

`relationship`에 별도 설정을 하지 않으면 기본적으로 `lazy='select'`로 동작합니다. 이는 관련된 속성(예: `dept.employees`)에 실제로 접근하는 시점에 데이터를 불러오는 방식입니다.

```python
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Department

db: Session = SessionLocal()

print("--- N+1 Problem (Lazy Loading) ---")
# 1. 부서 목록을 가져오는 쿼리 (1번)
departments = db.query(Department).all() 
print(f"Found {len(departments)} departments.")

# 2. 각 부서의 직원 목록에 접근할 때마다 추가 쿼리 발생 (N번)
for dept in departments:
    employee_names = [emp.name for emp in dept.employees]
    print(f"  - Dept: {dept.name}, Employees: {employee_names}")

db.close()
```
**실행 결과 (SQL 로그):**
```sql
-- 1번 쿼리
SELECT departments.id, departments.name FROM departments
-- N번의 추가 쿼리
SELECT employees.id, ... FROM employees WHERE employees.department_id = ?  -- (dept 1)
SELECT employees.id, ... FROM employees WHERE employees.department_id = ?  -- (dept 2)
...
```
</details>

<details>
<summary><b>2. 즉시 로딩 (Eager Loading) - N+1 문제 해결</b></summary>

쿼리 시점에 `options`를 사용하여 로딩 전략을 명시적으로 지정할 수 있습니다.

**방법 1: `joinedload`**
-   `LEFT OUTER JOIN`을 사용하여 부모 객체와 자식 객체를 한 번의 SQL 쿼리로 함께 가져옵니다.
-   일대다(one-to-many) 관계에서 자식 객체가 매우 많으면 결과 데이터가 중복되어 비효율적일 수 있습니다.

```python
from sqlalchemy.orm import joinedload

print("\n--- Solution 1: joinedload ---")
db: Session = SessionLocal()
# options(joinedload(...))를 사용하여 즉시 로딩 지정
departments = db.query(Department).options(joinedload(Department.employees)).all()

for dept in departments:
    employee_names = [emp.name for emp in dept.employees]
    print(f"  - Dept: {dept.name}, Employees: {employee_names}")
db.close()
```
**실행 결과 (SQL 로그):**
```sql
-- 단 1번의 JOIN 쿼리
SELECT departments.id, departments.name, employees.id, ...
FROM departments LEFT OUTER JOIN employees ON departments.id = employees.department_id
```

**방법 2: `selectinload`**
-   두 번의 쿼리로 데이터를 가져옵니다. 첫 번째 쿼리로 부모 객체를 모두 가져오고, 두 번째 쿼리에서 `WHERE ... IN (...)` 절을 사용하여 모든 자식 객체를 한 번에 가져옵니다.
-   일대다(one-to-many) 관계에서 `joinedload`보다 훨씬 효율적입니다. **일반적으로 가장 권장되는 전략입니다.**

```python
from sqlalchemy.orm import selectinload

print("\n--- Solution 2: selectinload (Recommended) ---")
db: Session = SessionLocal()
# options(selectinload(...))를 사용하여 즉시 로딩 지정
departments = db.query(Department).options(selectinload(Department.employees)).all()

for dept in departments:
    employee_names = [emp.name for emp in dept.employees]
    print(f"  - Dept: {dept.name}, Employees: {employee_names}")
db.close()
```
**실행 결과 (SQL 로그):**
```sql
-- 1번 쿼리 (부모)
SELECT departments.id, departments.name FROM departments
-- 2번 쿼리 (자식들)
SELECT employees.id, ..., employees.department_id
FROM employees
WHERE employees.department_id IN (?, ?, ...) -- (dept_id_1, dept_id_2, ...)
```
</details>

### 1.5. 실무적 접근법: 언제 ORM을 사용해야 하는가?

ORM은 만능 해결책이 아니며, 장점과 단점을 이해하고 전략적으로 사용해야 합니다.

-   **ORM이 빛을 발하는 경우**:
    -   대부분의 웹 애플리케이션과 같이 **CRUD 작업이 주를 이루는** 비즈니스 로직.
    -   개발 초기 단계에서 **빠른 프로토타이핑**이 필요할 때.
    -   데이터베이스 스키마 변경이 잦을 것으로 예상될 때.

-   **원시 SQL 또는 Query Builder가 더 나은 경우**:
    -   수백만 건 이상의 데이터를 처리하는 **대규모 배치(Batch) 작업**.
    -   복잡한 JOIN과 집계 함수가 필요한 **데이터 분석 및 리포팅 쿼리**.
    -   성능을 1ms 단위로 최적화해야 하는 미션 크리티컬한 시스템.

**결론적으로, 현대적인 개발에서는 이 둘을 혼용하는 하이브리드 접근 방식이 가장 효과적입니다.** 일반적인 애플리케이션 로직은 ORM으로 생산성을 확보하고, 성능이 중요하거나 복잡한 부분은 SQLAlchemy Core(Query Builder)나 원시 SQL을 사용하여 정교하게 제어하는 것이 현명한 전략입니다.


## 2. SQLAlchemy 아키텍처와 핵심 개념 (심화)

SQLAlchemy의 ORM을 효과적으로 사용하려면, 그저 메서드를 아는 것을 넘어 각 컴포넌트가 어떤 역할을 하고 어떻게 상호작용하는지 그 **설계 철학**을 이해하는 것이 중요합니다. SQLAlchemy는 **'관심사의 분리(Separation of Concerns)'** 원칙에 따라 설계되었습니다.

-   **Engine (엔진)**: 데이터베이스와의 **연결과 통신**을 담당합니다. (어떻게 DB와 말할 것인가?)
-   **DeclarativeBase & Models (선언적 베이스와 모델)**: 데이터베이스 스키마와 비즈니스 객체의 **구조를 정의**합니다. (무엇을 DB에 저장할 것인가?)
-   **Session (세션)**: 객체의 **생명주기와 트랜잭션을 관리**하는 작업 공간입니다. (객체를 가지고 어떻게 작업할 것인가?)

![SQLAlchemy Architecture](https://www.sqlalchemy.org/img/arch_small.png)
*(이미지 출처: SQLAlchemy 공식 문서)*

### 2.1. 설치

먼저 SQLAlchemy와 데이터베이스에 맞는 드라이버를 설치해야 합니다. 여기서는 MySQL을 위한 `pymysql`을 예시로 사용합니다.

```bash
# SQLAlchemy와 MySQL 드라이버 설치
pip install sqlalchemy pymysql

# 환경 변수 관리를 위한 python-dotenv 설치
pip install python-dotenv
```

### 2.2. Engine: 데이터베이스의 '발전소'

`Engine`은 데이터베이스 연결을 관리하는 중앙 허브입니다. 애플리케이션이 시작될 때 **단 한 번만 생성**하여 전역적으로 사용하는 것이 일반적입니다.

-   **역할**:
    -   **DBAPI 래퍼**: 내부적으로 `pymysql`과 같은 DBAPI 드라이버를 감싸고, 데이터베이스 '방언(Dialect)'을 사용하여 특정 DB(MySQL, PostgreSQL 등)에 맞는 SQL을 실행합니다.
    -   **커넥션 풀링(Connection Pooling)**: 가장 중요한 기능 중 하나입니다. 데이터베이스 연결은 생성 비용이 비싼 작업이므로, Engine은 미리 여러 개의 연결을 만들어 '풀(Pool)'에 저장해두고 필요할 때마다 재사용합니다. 이를 통해 애플리케이션의 성능과 안정성을 크게 향상시킵니다.

-   **실무적 설정**: `create_engine()`에는 프로덕션 환경에서 매우 중요한 옵션들이 있습니다.

    ```python
    # database.py
    import os
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    
    load_dotenv()
    DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}?charset=utf8mb4"

    engine = create_engine(
        DATABASE_URL,
        # --- 실무용 주요 옵션 ---
        # echo=True: SQLAlchemy가 생성하는 모든 SQL을 콘솔에 출력합니다. (디버깅 시 유용)
        echo=False, 
        
        # pool_size: 커넥션 풀에서 유지할 최소 연결 수 (기본값 5)
        pool_size=10,
        
        # max_overflow: 풀이 가득 찼을 때 추가로 생성할 수 있는 임시 연결 수 (기본값 10)
        max_overflow=20,
        
        # pool_recycle: 지정된 시간(초)이 지난 연결을 자동으로 재활용(끊고 다시 맺음)합니다.
        # DB 서버의 wait_timeout 설정보다 짧게 설정하여 'MySQL server has gone away' 오류를 방지합니다.
        pool_recycle=3600, # 1시간
        
        # pool_pre_ping: 풀에서 연결을 가져올 때마다 간단한 쿼리를 보내 연결이 유효한지 검사합니다.
        # 안정성은 높아지지만 약간의 오버헤드가 발생할 수 있습니다.
        pool_pre_ping=True
    )
    ```

### 2.3. DeclarativeBase와 Models: 데이터의 '청사진'

`DeclarativeBase`와 이를 상속하는 모델 클래스들은 데이터베이스 스키마의 '청사진' 역할을 합니다.

-   **`DeclarativeBase`**: 모든 모델 클래스가 상속받는 부모 클래스입니다. 이 클래스는 `MetaData`라는 중요한 객체를 가지고 있습니다.
-   **`MetaData`**: 일종의 '레지스트리'입니다. `Base`를 상속하는 모든 모델 클래스(테이블, 컬럼, 제약조건 등)의 정보가 이 `MetaData` 객체에 자동으로 등록됩니다. `Alembic`이나 `Base.metadata.create_all(engine)`과 같은 도구는 이 `MetaData` 객체를 참조하여 데이터베이스 스키마를 비교하고 생성합니다.
-   **`Model`**: 데이터베이스의 테이블 하나를 파이썬 클래스로 표현한 것입니다.

```python
# models.py
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship

# 1. 모든 모델의 기준이 될 Base 클래스 정의
class Base(DeclarativeBase):
    # MetaData 객체가 자동으로 생성되어 관리됨
    pass

# 2. Base를 상속하여 모델 클래스(테이블) 정의
class Department(Base):
    __tablename__ = 'departments'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    
    employees = relationship("Employee", back_populates="department", cascade="all, delete-orphan")

class Employee(Base):
    __tablename__ = 'employees'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    department_id = Column(Integer, ForeignKey('departments.id'))

    department = relationship("Department", back_populates="employees")
```

### 2.4. Session: 객체의 '작업 공간'이자 '생명주기 관리자'

`Session`은 ORM 작업의 실질적인 인터페이스입니다. `Engine`이 공장의 '전력 설비'라면, `Session`은 개별 작업을 수행하는 '작업대(Workbench)'에 비유할 수 있습니다.

-   **역할**:
    -   **트랜잭션 관리**: 세션 내에서 수행되는 모든 작업은 하나의 트랜잭션으로 묶입니다. `session.commit()`으로 모든 작업을 한 번에 반영하거나, `session.rollback()`으로 모두 취소할 수 있습니다.
    -   **Identity Map**: 세션은 한 번 조회한 객체를 내부의 딕셔너리(Identity Map)에 캐싱합니다. 동일한 PK를 가진 객체를 다시 조회하면, DB에 다시 쿼리하지 않고 메모리에서 즉시 동일한 객체 인스턴스를 반환하여 일관성을 보장합니다.
    -   **Unit of Work**: `session.add()`, `session.delete()` 등의 작업은 즉시 SQL을 실행하지 않습니다. 대신, 세션 내부에 "수행할 작업 목록"을 기록해 둡니다. `commit()`이 호출되거나 세션이 `flush`될 때, 이 작업 목록을 최적화하여 한 번에 SQL로 변환해 실행합니다.

#### 2.4.1. 객체의 생명주기 (Session Lifecycle)

세션 내에서 ORM 객체는 다음과 같은 4가지 상태를 거칩니다. 이를 이해하는 것은 ORM의 동작을 예측하는 데 매우 중요합니다.

1.  **Transient (임시 상태)**: 순수한 파이썬 객체로, 아직 세션에 속하지 않았으며 데이터베이스와도 아무 관련이 없습니다.
    ```python
    new_employee = Employee(name="Charlie") 
    # new_employee는 현재 Transient 상태
    ```
2.  **Pending (대기 상태)**: `session.add()`를 통해 객체가 세션에 추가된 상태입니다. 아직 데이터베이스에 저장되지 않았으며, PK 값도 없습니다. 커밋을 기다리는 상태입니다.
    ```python
    db.add(new_employee)
    # new_employee는 이제 Pending 상태
    ```
3.  **Persistent (영속 상태)**: `session.commit()`이 성공적으로 실행된 후의 상태입니다. 객체는 데이터베이스에 해당 로우를 가지며, PK 값도 부여받았습니다. 세션은 이 객체의 변경사항을 계속 추적합니다.
    ```python
    db.commit()
    # new_employee는 이제 Persistent 상태. new_employee.id 값을 확인할 수 있음
    
    # Persistent 객체의 속성을 변경하면 세션이 이를 감지함
    new_employee.name = "Charles" 
    # 이 변경사항은 다음 커밋 때 UPDATE 쿼리로 변환됨
    ```
4.  **Detached (분리 상태)**: 객체가 `session.close()` 등으로 인해 세션과의 연결이 끊어진 상태입니다. 객체 자체는 파이썬 메모리에 남아있지만, 더 이상 세션의 보호(트랜잭션, Identity Map 등)를 받지 못합니다. `relationship`으로 연결된 다른 객체에 접근(Lazy Loading)하려고 하면 오류가 발생합니다.

### 2.5. `sessionmaker`: Session 공장

`sessionmaker`는 설정이 완료된 `Session` 클래스를 생성하는 '팩토리'입니다. 애플리케이션 전역에 `Engine`과 함께 `sessionmaker`를 설정해두고, 각 작업 단위(예: 웹 요청 하나)마다 이 팩토리를 통해 새로운 `Session` 인스턴스를 만들어 사용하는 것이 표준적인 패턴입니다.

```python
# database.py (이어서)
from sqlalchemy.orm import sessionmaker

# Engine을 바인딩하여, 필요할 때마다 Session 객체를 생성할 수 있는 '공장'을 만듭니다.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 웹 프레임워크의 의존성 주입 등에서 사용할 세션 획득 함수
def get_db():
    db = SessionLocal() # 공장에서 새로운 세션(작업대)을 하나 만듦
    try:
        yield db
    finally:
        db.close() # 작업이 끝나면 세션(작업대)을 정리함
```

## 3. SQLAlchemy ORM 작업: 고급 쿼리와 관계 관리

기본적인 CRUD를 넘어, SQLAlchemy ORM의 진정한 힘은 복잡한 데이터를 효율적으로 조회하고 객체 간의 관계를 직관적으로 관리하는 데 있습니다. 이 섹션에서는 집계, 조인, 관계 필터링 등 실무에서 필수적인 고급 기법들을 다룹니다.

### 3.1. 사전 준비: 테이블 생성 및 기본 데이터 삽입

모든 예제는 `Session` 객체를 통해 실행됩니다. 먼저 `create_tables.py`를 실행하여 테이블을 생성하고, 아래 코드를 실행하여 예제 데이터를 미리 삽입했다고 가정합니다.

```python
# setup_data.py
from database import SessionLocal
from models import Department, Employee

db = SessionLocal()
try:
    # 부서 생성
    dept_eng = Department(name="Engineering")
    dept_hr = Department(name="Human Resources")
    dept_sales = Department(name="Sales")
    db.add_all([dept_eng, dept_hr, dept_sales])
    db.commit()

    # 직원 생성
    db.add_all([
        Employee(name="Alice", position="Software Engineer", department=dept_eng),
        Employee(name="Bob", position="Senior Engineer", department=dept_eng),
        Employee(name="Charlie", position="Recruiter", department=dept_hr),
        Employee(name="David", position="Sales Manager", department=dept_sales),
        Employee(name="Eve", position="Sales Associate", department=dept_sales)
    ])
    db.commit()
    print("Sample data inserted.")
finally:
    db.close()
```

### 3.2. 고급 데이터 조회 (Advanced Reading)

#### 3.2.1. 집계 함수 (Aggregation)

SQL의 `COUNT`, `SUM`, `AVG` 등 집계 함수는 `sqlalchemy.func`를 통해 사용할 수 있습니다.

```python
# advanced_queries.py
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload
from database import SessionLocal
from models import Department, Employee

db: Session = SessionLocal()

# 예제 1: 전체 직원 수 계산
employee_count = db.query(func.count(Employee.id)).scalar()
print(f"Total number of employees: {employee_count}")

# 예제 2: 부서별 직원 수 계산 (GROUP BY)
# 쿼리 결과는 (Department.name, count) 튜플의 리스트로 반환됨
results = db.query(
    Department.name, 
    func.count(Employee.id).label("num_employees")
).join(Employee, Department.id == Employee.department_id).group_by(Department.name).all()

print("\nNumber of employees per department:")
for name, num in results:
    print(f"  - {name}: {num}")

db.close()
```
- **`.label()`**: 집계 결과 컬럼에 별칭을 부여하여 나중에 쉽게 접근할 수 있게 합니다.
- **`.scalar()`**: 결과가 단일 값일 때, 그 값을 직접 반환합니다.

#### 3.2.2. 명시적 조인 (Explicit Joins)

`relationship`이 없거나, 더 복잡한 조인 조건을 사용해야 할 때 `query.join()`을 사용합니다.

```python
# advanced_queries.py (이어서)
db: Session = SessionLocal()

# Engineering 부서의 직원만 명시적 조인으로 조회
eng_employees = db.query(Employee).join(
    Department, Employee.department_id == Department.id
).filter(Department.name == "Engineering").all()

print("\nEmployees in Engineering (via explicit join):")
for emp in eng_employees:
    print(f"  - {emp.name}, {emp.position}")

db.close()
```

#### 3.2.3. 관계를 이용한 필터링 (`any()`, `has()`)

`relationship`을 활용하여 관련된 객체의 조건으로 필터링할 수 있습니다.

- **`Relationship.any()`**: "적어도 하나 ~를 가진" (To-Many 관계에서 사용)
- **`Relationship.has()`**: "~를 가진" (To-One 관계에서 사용)

```python
# advanced_queries.py (이어서)
db: Session = SessionLocal()

# 예제 1: 'Senior' 직책을 가진 직원이 한 명이라도 있는 부서 찾기
depts_with_seniors = db.query(Department).filter(
    Department.employees.any(Employee.position.like('%Senior%'))
).all()

print("\nDepartments with at least one 'Senior' position employee:")
for dept in depts_with_seniors:
    print(f"  - {dept.name}")

# 예제 2: 'Engineering' 부서에 속한 직원 찾기 (has 사용)
# Employee 입장에서 Department는 To-One 관계
engineers = db.query(Employee).filter(
    Employee.department.has(Department.name == "Engineering")
).all()

print("\nEngineers (queried via .has()):")
for emp in engineers:
    print(f"  - {emp.name}")

db.close()
```

### 3.3. 관계와 객체 상태 관리 (Update & Delete)

#### 3.3.1. 관계의 연쇄 삭제 (Cascades)

모델을 정의할 때 `relationship`에 `cascade` 옵션을 설정하면, 부모 객체의 상태 변화가 자식 객체에 자동으로 전파될 수 있습니다.

```python
# models.py
class Department(Base):
    # ...
    # cascade="all, delete-orphan":
    # - all: 부모에 대한 모든 작업(save-update, delete)을 자식에게 전파
    # - delete-orphan: 부모와의 관계가 끊어진 자식 객체를 자동으로 삭제
    employees = relationship("Employee", back_populates="department", cascade="all, delete-orphan")
```

**Cascade 동작 예시:**

```python
# relationship_management.py
from database import SessionLocal
from models import Department, Employee
from sqlalchemy.orm import selectinload

db = SessionLocal()

# 1. 부모에서 자식을 제거하면, 자식 객체가 DB에서 삭제됨 (delete-orphan)
print("--- Testing 'delete-orphan' cascade ---")
sales_dept = db.query(Department).options(selectinload(Department.employees)).filter_by(name="Sales").one()
eve = db.query(Employee).filter_by(name="Eve").one()

print(f"Employees in Sales before removal: {[e.name for e in sales_dept.employees]}")
sales_dept.employees.remove(eve) # 관계 컬렉션에서 제거
db.commit()
print(f"Employees in Sales after removal: {[e.name for e in sales_dept.employees]}")

# Eve가 DB에서 삭제되었는지 확인
eve_exists = db.query(Employee).filter_by(name="Eve").first()
print(f"Is Eve still in DB? {'Yes' if eve_exists else 'No'}") # 결과: No

# 2. 부모 객체를 삭제하면, 자식 객체들도 함께 삭제됨 (delete)
print("\n--- Testing 'delete' cascade ---")
eng_dept = db.query(Department).filter_by(name="Engineering").one()
db.delete(eng_dept)
db.commit()

# Engineering 부서 직원들이 삭제되었는지 확인
eng_emp_count = db.query(func.count(Employee.id)).filter(Employee.department_id == eng_dept.id).scalar()
print(f"Number of employees left in Engineering department: {eng_emp_count}") # 결과: 0

db.close()
```

### 3.4. 대량 작업과 성능 (Bulk Operations)

수천, 수만 건의 데이터를 한 번에 삽입하거나 수정할 때, 일반적인 `session.add()` 루프는 객체 상태 추적 오버헤드로 인해 매우 느립니다. 이 경우, 세션의 상태 관리 기능을 우회하는 **벌크(Bulk) 작업**을 사용해야 합니다.

-   **`session.bulk_insert_mappings()`**: 파이썬 딕셔너리 리스트를 사용하여 대량의 데이터를 빠르게 `INSERT`합니다. ORM 객체를 생성하지 않아 오버헤드가 적습니다.
-   **`session.bulk_update_mappings()`**: 딕셔너리 리스트를 사용하여 대량의 데이터를 `UPDATE`합니다. 각 딕셔너리는 반드시 업데이트할 로우를 식별할 기본 키(PK)를 포함해야 합니다.

```python
# bulk_operations.py
from database import SessionLocal

db = SessionLocal()
try:
    # 예제 1: 대량 삽입
    print("--- Bulk inserting new employees ---")
    new_hires = [
        {'name': f'New Hire {i}', 'position': 'Junior Developer', 'department_id': 1}
        for i in range(1000)
    ]
    db.bulk_insert_mappings(Employee, new_hires)
    db.commit()
    print("1000 new employees inserted.")

    # 예제 2: 대량 수정
    print("\n--- Bulk updating employees ---")
    # ID가 500에서 600 사이인 직원들의 직책을 변경
    updates = [
        {'id': i, 'position': 'Developer'}
        for i in range(500, 601)
    ]
    db.bulk_update_mappings(Employee, updates)
    db.commit()
    print("101 employees updated.")

finally:
    db.close()
```
**주의**: 벌크 작업은 ORM의 객체 생명주기 이벤트, 관계 동기화, 자동 flush 등 대부분의 자동화 기능을 우회합니다. 따라서 순수하게 성능이 중요한 대규모 데이터 처리 작업에 제한적으로 사용하는 것이 좋습니다.


## 4. Alembic을 이용한 데이터베이스 마이그레이션 (프로덕션 레벨)

애플리케이션은 계속해서 변화하고 발전합니다. 이 과정에서 "직원 테이블에 이메일 컬럼 추가", "새로운 제품 테이블 생성" 등 데이터베이스 스키마 변경은 필연적으로 발생합니다. `Base.metadata.create_all()`은 처음 테이블을 생성할 때만 유용하며, 이미 존재하는 테이블의 구조를 변경하지는 못합니다. 이러한 문제를 해결하기 위한 표준 도구가 바로 **Alembic**입니다.

### 4.1. 마이그레이션의 개념과 Alembic의 역할

-   **데이터베이스 마이그레이션이란?**
    버전 관리 시스템(예: Git)으로 코드의 변경 이력을 관리하듯, 데이터베이스 스키마의 변경 이력을 체계적으로 관리하는 프로세스입니다. 각 변경사항은 "마이그레이션 스크립트"라는 파일로 기록되며, 이 스크립트를 통해 스키마를 최신 상태로 업데이트하거나(Upgrade), 이전 상태로 되돌릴 수(Downgrade) 있습니다.

-   **Alembic의 역할**:
    Alembic은 SQLAlchemy를 위한 데이터베이스 마이그레이션 도구입니다. Alembic은 다음 두 가지를 비교하여 마이그레이션 스크립트를 **자동으로 생성**합니다.
    1.  **현재 데이터베이스의 스키마 상태**
    2.  **SQLAlchemy 모델(`models.py`)에 정의된 스키마 상태**

    이를 통해 개발자는 `ALTER TABLE`, `ADD COLUMN`과 같은 SQL을 직접 작성할 필요 없이, 파이썬 모델 코드의 변경사항을 데이터베이스에 안전하게 반영할 수 있습니다.

### 4.2. Alembic 설치 및 초기화

1.  **설치**:
    ```bash
    pip install alembic
    ```

2.  **초기화**: 프로젝트의 루트 디렉토리에서 다음 명령을 실행합니다.
    ```bash
    alembic init alembic
    ```
    이 명령은 다음과 같은 구조를 생성합니다.
    ```
    .
    ├── alembic/
    │   ├── versions/  # 마이그레이션 스크립트가 저장될 폴더
    │   ├── env.py     # Alembic 실행 환경 설정 파일
    │   ├── script.py.mako # 마이그레이션 스크립트 템플릿
    │   └── README
    └── alembic.ini    # Alembic 주 설정 파일
    ```

### 4.3. Alembic 설정 (가장 중요한 단계)

Alembic이 제대로 동작하려면, 데이터베이스 연결 정보와 모델의 위치를 알려주어야 합니다.

1.  **`alembic.ini` 파일 수정**:
    `sqlalchemy.url` 항목을 찾아, `database.py`에서 사용한 데이터베이스 URL로 수정합니다.

    ```ini
    # alembic.ini
    ...
    # sqlalchemy.url = driver://user:pass@localhost/dbname
    sqlalchemy.url = mysql+pymysql://<DB_USER>:<DB_PASSWORD>@<DB_HOST>/<DB_NAME>?charset=utf8mb4
    ...
    ```
    (실제 값으로 채워주세요. 예: `mysql+pymysql://root:password@localhost/company_db?charset=utf8mb4`)

2.  **`alembic/env.py` 파일 수정**:
    Alembic이 우리 모델의 메타데이터를 인식하도록 설정해야 합니다. `target_metadata` 부분을 찾아서 다음과 같이 수정합니다.

    ```python
    # alembic/env.py

    # ... 기존 import ...
    # --- 추가 시작 ---
    # 프로젝트의 루트 디렉토리를 경로에 추가
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from models import Base # models.py에서 Base를 가져옴
    # --- 추가 끝 ---

    # the Alembic Config object, which provides
    # access to the values within the .ini file in use.
    config = context.config

    # Interpret the config file for Python logging.
    # This line sets up loggers basically.
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)

    # add your model's MetaData object here
    # for 'autogenerate' support
    # from myapp import mymodel
    # target_metadata = mymodel.Base.metadata
    # --- 수정 시작 ---
    target_metadata = Base.metadata # 우리 모델의 메타데이터를 지정
    # --- 수정 끝 ---

    def run_migrations_offline() -> None:
        # ... (이하 생략) ...

    def run_migrations_online() -> None:
        # ... (이하 생략) ...
    ```

### 4.4. 마이그레이션 라이프사이클: 첫 생성부터 적용까지

1.  **마이그레이션 스크립트 생성**:
    이제 모델(`models.py`)과 실제 데이터베이스의 상태를 비교하여 첫 마이그레이션 스크립트를 생성합니다.

    ```bash
    # --autogenerate: 자동 생성 옵션
    # -m: 마이그레이션에 대한 설명 메시지
    alembic revision --autogenerate -m "Create initial tables for departments and employees"
    ```
    성공하면 `alembic/versions/` 폴더에 `xxxxxxxxxxxx_create_initial_tables...py` 와 같은 파일이 생성됩니다. 이 파일의 `upgrade()` 함수에는 테이블을 생성하는 코드가, `downgrade()` 함수에는 테이블을 삭제하는 코드가 들어있습니다. **실행 전 항상 내용을 검토하는 것이 좋습니다.**

2.  **마이그레이션 적용**:
    생성된 스크립트를 데이터베이스에 적용하여 실제 테이블을 만듭니다.

    ```bash
    alembic upgrade head
    ```
    `head`는 최신 버전의 마이그레이션을 의미합니다. 이제 데이터베이스에 `departments`와 `employees` 테이블, 그리고 마이그레이션 버전을 관리하는 `alembic_version` 테이블이 생성된 것을 확인할 수 있습니다.

3.  **상태 확인**:
    현재 데이터베이스에 적용된 마이그레이션 버전을 확인할 수 있습니다.
    ```bash
    alembic current
    # (head)가 표시되면 최신 상태임
    ```

### 4.5. 모델 변경 및 후속 마이그레이션

이제 애플리케이션 요구사항이 변경되어 `Employee` 모델에 `email` 컬럼을 추가해 보겠습니다.

1.  **모델 수정 (`models.py`)**:
    ```python
    # models.py
    class Employee(Base):
        __tablename__ = 'employees'
        # ... 기존 컬럼 ...
        email = Column(String(100), nullable=True, unique=True) # 이메일 컬럼 추가
    ```

2.  **새 마이그레이션 스크립트 생성**:
    다시 `revision` 명령을 실행하여 변경사항을 감지하고 새 스크립트를 만듭니다.
    ```bash
    alembic revision --autogenerate -m "Add email column to employees table"
    ```
    `alembic/versions/`에 생성된 새 스크립트를 열어보면 `op.add_column('employees', ...)`와 같은 코드가 포함된 것을 볼 수 있습니다.

3.  **새 마이그레이션 적용**:
    다시 `upgrade` 명령을 실행하여 데이터베이스 스키마를 최신 상태로 업데이트합니다.
    ```bash
    alembic upgrade head
    ```
    이제 실제 `employees` 테이블에 `email` 컬럼이 추가되었습니다.

### 4.6. 마이그레이션 관리: 되돌리기 및 히스토리 조회

-   **히스토리 조회**: 전체 마이그레이션 기록을 볼 수 있습니다.
    ```bash
    alembic history --verbose
    ```

-   **마이그레이션 되돌리기 (Downgrade)**:
    -   가장 최근 마이그레이션 하나를 되돌리려면:
        ```bash
        alembic downgrade -1
        ```
    -   모든 마이그레이션을 되돌려 초기 상태로 가려면:
        ```bash
        alembic downgrade base
        ```
    -   특정 버전으로 가려면 해당 버전의 ID를 사용합니다.

**실무적 조언**: 프로덕션 환경에서는 `downgrade`를 매우 신중하게 사용해야 합니다. 컬럼을 삭제하는 등의 작업은 데이터 손실을 유발할 수 있기 때문입니다. 마이그레이션은 항상 개발 및 스테이징 환경에서 충분히 테스트한 후 프로덕션에 적용해야 합니다.

### 4.7. 고급 마이그레이션 시나리오

#### 4.7.1. 데이터 마이그레이션 (Data Migration)

스키마 변경뿐만 아니라, 기존 데이터를 변경해야 할 때가 있습니다. 예를 들어, `Employee` 모델의 `name` 컬럼을 `first_name`과 `last_name`으로 분리하는 경우입니다.

1.  **모델 변경**: `name`을 `first_name`, `last_name`으로 변경합니다.
2.  **마이그레이션 생성**: `alembic revision --autogenerate -m "Split name into first and last name"`
3.  **스크립트 수정**: 자동 생성된 스크립트는 컬럼 추가/삭제만 수행합니다. `op.execute()`를 사용하여 기존 `name` 데이터를 `first_name`, `last_name`으로 옮기는 SQL을 직접 추가해야 합니다.

    ```python
    # versions/xxxxxxxx_split_name_..._.py
    
    # ... imports
    from sqlalchemy.sql import table, column
    from sqlalchemy import String

    def upgrade() -> None:
        # ### commands auto generated by Alembic - please adjust! ###
        op.add_column('employees', sa.Column('first_name', sa.String(length=50), nullable=True))
        op.add_column('employees', sa.Column('last_name', sa.String(length=50), nullable=True))
        
        # --- 데이터 마이그레이션 로직 추가 ---
        # 임시 테이블 객체를 만들어 ORM 모델에 의존하지 않도록 함
        employees_table = table('employees',
            column('id', sa.Integer),
            column('name', sa.String),
            column('first_name', sa.String),
            column('last_name', sa.String)
        )
        bind = op.get_bind()
        # 순수 SQL을 실행하여 데이터 분리
        bind.execute(
            employees_table.update().values(
                first_name=func.substring_index(employees_table.c.name, ' ', 1),
                last_name=func.substring_index(employees_table.c.name, ' ', -1)
            )
        )
        # --- 로직 추가 끝 ---
        
        op.drop_column('employees', 'name')
        # ### end Alembic commands ###

    def downgrade() -> None:
        # ... (downgrade 로직도 데이터 복구를 위해 수정 필요) ...
    ```

#### 4.7.2. 마이그레이션 브랜치 및 병합

팀 환경에서 여러 개발자가 각자의 브랜치에서 마이그레이션을 생성하면, 마이그레이션 히스토리가 여러 갈래로 나뉘는 '브랜치'가 발생합니다.

```
      -> 2b1ae... (feature-A)
     /
1a2b3c... -> 3c4d5e... (feature-B)
```

이 상태에서 `alembic upgrade head`를 실행하면 어떤 브랜치를 따라야 할지 몰라 오류가 발생합니다.

-   **해결**: `alembic merge` 명령으로 두 브랜치를 하나로 합치는 병합 스크립트를 생성합니다.
    ```bash
    alembic merge -m "Merge feature-A and feature-B branches" <rev_A> <rev_B>
    # 예: alembic merge -m "..." 2b1ae 3c4d5e
    ```
    이 명령은 두 브랜치의 최종점을 부모로 가지는 새로운 병합 마이그레이션을 생성하여 히스토리를 다시 하나로 만듭니다.

### 4.8. 실무 모범 사례 및 흔한 실수

1.  **절대 이전 마이그레이션 파일을 수정하지 마세요**: 이미 팀원이나 프로덕션에 적용된 마이그레이션 파일을 수정하는 것은 재앙입니다. 항상 `alembic revision`으로 새로운 변경사항을 만드세요.
2.  **자동 생성된 스크립트를 항상 검토하세요**: Alembic은 완벽하지 않습니다. 복잡한 제약조건 변경, 컬럼 타입 변경 등은 의도와 다르게 생성될 수 있습니다. 적용 전에 반드시 내용을 확인하고 필요하면 수정해야 합니다.
3.  **마이그레이션은 작고 집중된 단위로 만드세요**: 하나의 마이그레이션에는 관련된 하나의 논리적 변경만 포함하는 것이 좋습니다. 디버깅과 롤백이 훨씬 쉬워집니다.
4.  **`downgrade`는 신중하게**: `downgrade`는 데이터 손실을 유발할 수 있습니다. 프로덕션 환경에서는 문제가 발생했을 때 `downgrade`로 되돌리기보다, 문제를 해결하는 새로운 마이그레이션을 만들어 `upgrade`하는 방식(Forward-only)이 더 안전할 수 있습니다.
5.  **애플리케이션 코드와 마이그레이션을 함께 배포하세요**: 특정 마이그레이션은 특정 버전의 애플리케이션 코드와 호환됩니다. 이 둘을 항상 함께 버전 관리하고 배포하여 불일치 문제를 방지해야 합니다.