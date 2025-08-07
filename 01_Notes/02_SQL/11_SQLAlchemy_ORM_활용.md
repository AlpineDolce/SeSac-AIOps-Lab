<h2>SQLAlchemy ORM 활용</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-28

<h2>문서 목표</h2>
<p>이 문서는 파이썬의 대표적인 ORM(Object-Relational Mapping) 라이브러리인 SQLAlchemy를 사용하여, 객체 지향적인 방식으로 데이터베이스를 다루는 방법을 학습합니다. ORM의 핵심 개념을 이해하고, 복잡한 데이터 모델과 관계를 파이썬 코드로 효과적으로 관리하는 능력을 기르는 것을 목표로 합니다.</p>

> **현업 데이터 분석가의 관점 한 줄 요약:**
> SQLAlchemy ORM은 파이썬 개발자가 데이터베이스를 객체 지향적으로 다룰 수 있게 하는 강력한 도구입니다. 복잡한 데이터 모델을 파이썬 클래스로 표현하고, 관계형 데이터를 객체 간의 관계로 탐색하며, SQL Injection 위험 없이 안전하게 쿼리를 작성하는 능력은 현대 데이터 분석가에게 필수적입니다. 특히 대규모 애플리케이션에서 데이터베이스와의 효율적인 상호작용을 위해 ORM의 장단점과 성능 최적화 기법을 이해하는 것이 중요합니다.

<h2>목차</h2>

- [1. ORM을 이용한 객체 지향적 접근: `SQLAlchemy`](#1-orm을-이용한-객체-지향적-접근-sqlalchemy)
  - [1.1. ORM(Object-Relational Mapping)이란?](#11-ormobject-relational-mapping이란)
    - [1.1.1. ORM의 장점과 단점](#111-orm의-장점과-단점)
  - [1.2. SQLAlchemy 소개: Core vs ORM](#12-sqlalchemy-소개-core-vs-orm)
    - [1.2.1. SQLAlchemy Core: SQL 표현식 언어](#121-sqlalchemy-core-sql-표현식-언어)
    - [1.2.2. SQLAlchemy ORM: 객체 매핑](#122-sqlalchemy-orm-객체-매핑)
  - [1.3. 기본 설정 및 세션 관리](#13-기본-설정-및-세션-관리)
  - [1.4. ORM 테이블 메타데이터 정의 (Declarative Base)](#14-orm-테이블-메타데이터-정의-declarative-base)
  - [1.5. CRUD 작업 예제 (Create, Read, Update, Delete)](#15-crud-작업-예제-create-read-update-delete)
  - [1.6. 관계(Relationship) 설정과 활용 (One-to-Many, Many-to-Many)](#16-관계relationship-설정과-활용-one-to-many-many-to-many)
  - [1.7. 관계 로딩(Relationship Loading) 전략 (Lazy, Eager, Joined)](#17-관계-로딩relationship-loading-전략-lazy-eager-joined)
  - [1.8. 원시 SQL 실행: `text()` 함수](#18-원시-sql-실행-text-함수)
- [2. 성능 최적화: 커넥션 풀 (Connection Pool)](#2-성능-최적화-커넥션-풀-connection-pool)
  - [2.1. 커넥션 풀의 필요성 및 동작 원리](#21-커넥션-풀의-필요성-및-동작-원리)
  - [2.2. SQLAlchemy 커넥션 풀 설정 및 활용](#22-sqlalchemy-커넥션-풀-설정-및-활용)

---

## 1. ORM을 이용한 객체 지향적 접근: `SQLAlchemy`

ORM(Object-Relational Mapping)은 객체 지향 프로그래밍 언어의 객체와 관계형 데이터베이스의 데이터를 자동으로 매핑하는 기술입니다. 개발자가 SQL 쿼리를 직접 작성하는 대신, 객체 지향적인 코드를 통해 데이터베이스를 조작할 수 있게 해줍니다. `SQLAlchemy`는 파이썬에서 가장 강력하고 널리 사용되는 ORM 라이브러리입니다.

### 1.1. ORM(Object-Relational Mapping)이란?

ORM은 객체 지향 언어의 클래스와 객체를 관계형 데이터베이스의 테이블과 로우에 매핑하여, 개발자가 SQL 없이도 데이터베이스와 상호작용할 수 있도록 돕는 기술입니다.

#### 1.1.1. ORM의 장점과 단점 (현실적인 이해)

ORM은 개발 생산성을 크게 향상시키지만, 만능은 아닙니다. 장점과 단점을 명확히 이해하고 상황에 맞게 활용하는 것이 중요합니다.

*   **장점:**
    *   **생산성 향상:** SQL 쿼리를 직접 작성하고 관리하는 시간을 줄여줍니다. 파이썬 코드만으로 데이터베이스 작업을 수행할 수 있어 개발 속도가 빨라집니다.
    *   **객체 지향적 개발:** 데이터베이스의 로우를 파이썬 객체로 다룰 수 있어 코드의 가독성과 유지보수성이 향상됩니다. 객체 간의 관계를 파이썬 코드 내에서 직관적으로 표현할 수 있습니다.
    *   **DBMS 독립성:** ORM은 데이터베이스 종류에 따른 SQL 방언(Dialect) 차이를 추상화해줍니다. 따라서 데이터베이스를 변경하더라도 애플리케이션 코드 수정이 최소화됩니다.
    *   **SQL Injection 방지:** 대부분의 ORM은 쿼리 파라미터화(Parameterized Queries)를 자동으로 처리하여 사용자 입력으로 인한 SQL Injection 공격을 효과적으로 방지합니다.
    *   **테스트 용이성:** 데이터베이스에 직접 의존하지 않고도 모델을 테스트하기 용이합니다.

*   **단점:**
    *   **학습 곡선:** ORM의 개념(세션, 관계 로딩 전략 등)과 특정 ORM 라이브러리(SQLAlchemy)의 사용법을 익히는 데 시간이 필요합니다.
    *   **성능 오버헤드:** ORM은 SQL 쿼리를 자동으로 생성하는 과정에서 약간의 오버헤드가 발생할 수 있습니다. 특히 복잡한 쿼리의 경우 ORM이 생성하는 SQL이 비효율적일 수 있으며, 직접 최적화된 SQL을 작성하는 것보다 성능이 떨어질 수 있습니다.
    *   **추상화 누수 (Abstraction Leakage) 및 복잡성:**
        ORM은 데이터베이스를 추상화하여 편리함을 제공하지만, 때로는 ORM의 추상화가 분석가의 요구사항을 완전히 만족시키지 못하거나, ORM이 생성하는 SQL이 비효율적일 수 있습니다. 이러한 경우 ORM의 내부 동작을 이해하고 직접 SQL을 작성하여 ORM의 한계를 보완해야 합니다. 이를 **추상화 누수**라고 합니다. ORM을 사용하더라도 SQL에 대한 깊은 이해는 여전히 중요합니다.
    *   **복잡한 쿼리 표현의 어려움:** 매우 복잡하거나 특정 DBMS에 특화된 고급 SQL 기능(예: 윈도우 함수, CTE의 특정 재귀 패턴)은 ORM만으로는 표현하기 어렵거나 비효율적일 수 있습니다.

**실무적 조언:**
ORM은 개발 생산성을 높이는 강력한 도구이지만, 성능에 민감한 부분이나 ORM으로 표현하기 어려운 복잡한 쿼리는 SQLAlchemy Core나 원시 SQL을 혼용하여 사용하는 **하이브리드 접근 방식**이 일반적인 실무 모범 사례입니다. ORM을 사용하더라도 SQL에 대한 깊은 이해는 필수적입니다.

### 1.2. SQLAlchemy 소개: Core vs ORM

SQLAlchemy는 두 가지 주요 패러다임을 제공합니다.

#### 1.2.1. SQLAlchemy Core: SQL 표현식 언어

SQLAlchemy Core는 SQL 쿼리를 파이썬 코드로 추상화하여 작성할 수 있게 해주는 SQL 표현식 언어(SQL Expression Language)를 제공합니다. SQL 문법과 매우 유사하며, SQL 쿼리를 직접 작성하는 것과 거의 동일한 성능을 제공하면서도 파이썬의 장점을 활용할 수 있습니다.

**SQLAlchemy Core의 유용성 (데이터 분석가 관점):**
데이터 분석가는 종종 복잡하고 동적인 SQL 쿼리를 작성해야 합니다. ORM은 편리하지만, 때로는 ORM의 추상화가 분석가의 요구사항을 완전히 만족시키지 못하거나, ORM이 생성하는 SQL이 비효율적일 수 있습니다. 이럴 때 SQLAlchemy Core는 다음과 같은 장점을 제공합니다.

*   **SQL에 가까운 제어:** SQL 쿼리를 파이썬 객체로 구성하므로, SQL 문법에 익숙한 분석가가 직관적으로 쿼리를 작성할 수 있습니다. `JOIN`, `GROUP BY`, `WHERE` 절 등을 파이썬 코드로 유연하게 조합할 수 있습니다.
*   **성능 최적화:** ORM의 오버헤드 없이 거의 원시 SQL에 가까운 성능을 낼 수 있습니다. 특히 대량의 데이터를 처리하거나 복잡한 분석 쿼리에서 유리합니다.
*   **동적 쿼리 생성:** 사용자 입력이나 조건에 따라 `WHERE` 절, `ORDER BY` 절 등을 동적으로 변경해야 할 때 ORM보다 훨씬 유연하게 대처할 수 있습니다.
*   **DBMS 독립성 유지:** Core 레벨에서도 DBMS 종류에 따른 SQL 방언 차이를 추상화해주므로, 코드를 다른 DBMS로 이식하기 용이합니다.

**예시: SQLAlchemy Core를 사용한 쿼리**
```python
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, select, func
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:3306/"
    f"{os.getenv('DB_NAME')}"
)
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# 테이블 메타데이터 정의 (기존 테이블에 매핑)
employees_table = Table(
    'employees',
    metadata,
    Column('employee_id', Integer, primary_key=True),
    Column('first_name', String),
    Column('last_name', String),
    Column('salary', DECIMAL),
    Column('department_id', Integer)
)
departments_table = Table(
    'departments',
    metadata,
    Column('department_id', Integer, primary_key=True),
    Column('department_name', String)
)

with engine.connect() as connection:
    # SELECT 쿼리 예시 (직원 이름과 부서명 조회)
    stmt = select(
        employees_table.c.first_name,
        employees_table.c.last_name,
        departments_table.c.department_name
    ).join(departments_table)
    result = connection.execute(stmt).fetchall()
    print("\nEmployees and their departments (SQLAlchemy Core):")
    for row in result:
        print(row)

    # GROUP BY 및 집계 함수 예시 (부서별 평균 급여)
    stmt_avg_salary = select(
        departments_table.c.department_name,
        func.avg(employees_table.c.salary).label('avg_salary')
    ).join(departments_table).group_by(departments_table.c.department_name)
    result_avg = connection.execute(stmt_avg_salary).fetchall()
    print("\nAverage salary by department (SQLAlchemy Core):")
    for row in result_avg:
        print(row)
```

#### 1.2.2. SQLAlchemy ORM: 객체 매핑

SQLAlchemy ORM은 파이썬 클래스를 데이터베이스 테이블에 매핑하여, 객체 지향적인 방식으로 데이터를 조작할 수 있게 해줍니다. 개발자는 SQL 쿼리 대신 파이썬 객체의 메서드를 호출하거나 속성에 접근하여 데이터베이스 작업을 수행합니다.

### 1.3. 기본 설정 및 세션 관리

SQLAlchemy를 사용하기 위한 기본 설정과 세션(Session) 관리는 다음과 같습니다.

```python
from sqlalchemy import create_engine, Column, Integer, String, Date, DECIMAL, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError
import datetime
import os
from dotenv import load_dotenv

load_dotenv() # .env 파일 로드

# 1. 데이터베이스 엔진 생성
# MySQL 연결 문자열 형식: mysql+pymysql://user:password@host:port/dbname
# 보안상 민감 정보(비밀번호)는 환경 변수로 관리하는 것을 강력히 권장합니다.
# 예: .env 파일에 DB_HOST=localhost, DB_USER=root, DB_PASSWORD=your_password, DB_NAME=company_db 설정
DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:3306/"
    f"{os.getenv('DB_NAME')}"
)
engine = create_engine(DATABASE_URL, echo=True) # echo=True는 실행되는 SQL 쿼리를 출력

# 2. 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. 선언적 베이스 생성 (ORM 모델 정의의 기반)
Base = declarative_base()

# ORM 모델 정의 (employees 테이블에 매핑)
class Employee(Base):
    __tablename__ = 'employees'

    employee_id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True)
    phone_number = Column(String(20))
    hire_date = Column(Date, nullable=False)
    job_id = Column(String(10), nullable=False)
    salary = Column(DECIMAL(10, 2), nullable=False)
    department_id = Column(Integer, ForeignKey('departments.department_id'))

    # 관계 정의 (Department 모델이 정의되어 있다고 가정)
    # department = relationship("Department", back_populates="employees")

    def __repr__(self):
        return f"<Employee(id={self.employee_id}, name='{self.first_name} {self.last_name}')>"

    # 데이터베이스 테이블 생성 (모델에 정의된 테이블이 없으면 생성)
    # 이 코드는 애플리케이션 시작 시 데이터베이스 스키마를 자동으로 생성하거나 업데이트할 때 사용됩니다。
    # 이미 테이블이 존재한다면 실행되지 않습니다。
    Base.metadata.create_all(engine)

# 세션 사용 예시
session = SessionLocal()
try:
    # 데이터베이스 작업 수행
    pass
finally:
    session.close()
```

### 1.4. ORM 테이블 메타데이터 정의 (Declarative Base)

`declarative_base()`를 사용하여 ORM 모델 클래스를 정의합니다. 각 클래스는 데이터베이스 테이블에 매핑되며, 클래스의 속성은 테이블의 컬럼에 매핑됩니다.

```python
# 위 2.3 섹션의 Employee 클래스 정의 참조
```

### 1.5. CRUD 작업 예제 (Create, Read, Update, Delete)

ORM을 사용하여 객체 지향적인 방식으로 데이터를 생성, 조회, 수정, 삭제합니다.

```python
# 세션 생성
session = SessionLocal()

try:
    # CREATE (데이터 추가)
    new_employee = Employee(
        first_name='Peter',
        last_name='Jones',
        email='peter.jones@example.com',
        hire_date=datetime.date(2024, 2, 15),
        job_id='MGR',
        salary=72000.00,
        department_id=2
    )
    session.add(new_employee)
    session.commit() # 데이터베이스에 반영
    session.refresh(new_employee) # 자동 생성된 ID (예: AUTO_INCREMENT) 등 데이터베이스에 의해 변경된 최신 정보를 객체에 반영
    print(f"Created employee: {new_employee}")

    # READ (데이터 조회)
    # 모든 직원 조회
    all_employees = session.query(Employee).all()
    print("\nAll Employees:")
    for emp in all_employees:
        print(emp)

    # 특정 직원 조회 (ID로)
    employee_by_id = session.query(Employee).get(new_employee.employee_id)
    print(f"\nEmployee by ID {new_employee.employee_id}: {employee_by_id}")

    # 조건부 조회 (필터링)
    dev_employees = session.query(Employee).filter(Employee.job_id == 'DEV').all()
    print("\nDevelopers:")
    for emp in dev_employees:
        print(emp)

    # UPDATE (데이터 수정)
    employee_to_update = session.query(Employee).filter(Employee.first_name == 'Peter').first()
    if employee_to_update:
        employee_to_update.salary = 75000.00
        session.commit()
        print(f"\nUpdated employee {employee_to_update.first_name}'s salary to {employee_to_update.salary}")

    # DELETE (데이터 삭제)
    employee_to_delete = session.query(Employee).filter(Employee.first_name == 'Peter').first()
    if employee_to_delete:
        session.delete(employee_to_delete)
        session.commit()
        print(f"\nDeleted employee: {employee_to_delete.first_name}")

except SQLAlchemyError as e:
    session.rollback() # 오류 발생 시 롤백
    print(f"An SQLAlchemy error occurred: {e}")
finally:
    session.close()
```

### 1.6. 관계(Relationship) 설정과 활용 (One-to-Many, Many-to-Many)

SQLAlchemy ORM은 테이블 간의 관계를 파이썬 객체 레벨에서 정의하고 활용할 수 있게 해줍니다. 이는 `JOIN` 쿼리를 직접 작성하지 않고도 관련된 데이터를 쉽게 탐색할 수 있게 합니다.

```python
# Department 모델 정의 (employees 테이블과 관계를 맺기 위함)
class Department(Base):
    __tablename__ = 'departments'

    department_id = Column(Integer, primary_key=True, autoincrement=True)
    department_name = Column(String(100), unique=True, nullable=False)
    location = Column(String(100))

    # One-to-Many 관계: Department는 여러 Employee를 가질 수 있음
    # back_populates는 양방향 관계 설정을 위함
    employees = relationship("Employee", back_populates="department")

    def __repr__(self):
        return f"<Department(id={self.department_id}, name='{self.department_name}')>"

# Employee 모델에 department 관계 추가
# class Employee(Base):
#     ...
#     department_id = Column(Integer, ForeignKey('departments.department_id'))
#     department = relationship("Department", back_populates="employees")

# 관계 활용 예시
session = SessionLocal()
try:
    # 부서 생성 (예시)
    # new_dept = Department(department_name='Marketing', location='Seoul')
    # session.add(new_dept)
    # session.commit()
    # session.refresh(new_dept)

    # 직원 생성 및 부서 할당
    # emp_with_dept = Employee(
    #     first_name='Emily', last_name='White', hire_date=datetime.date(2024, 3, 1),
    #     job_id='MKT', salary=68000.00, department=new_dept # 객체로 관계 설정
    # )
    # session.add(emp_with_dept)
    # session.commit()

    # 부서를 통해 직원 조회
    dev_dept = session.query(Department).filter(Department.department_name == 'Development').first()
    if dev_dept:
        print(f"\nEmployees in {dev_dept.department_name}:")
        for emp in dev_dept.employees:
            print(f"  - {emp.first_name} {emp.last_name}")

    # 직원을 통해 부서 조회
    some_employee = session.query(Employee).filter(Employee.first_name == 'John').first()
    if some_employee and some_employee.department:
        print(f"\n{some_employee.first_name} works in {some_employee.department.department_name}")

finally:
    session.close()
```

### 1.7. 관계 로딩(Relationship Loading) 전략 (성능 최적화의 핵심)

SQLAlchemy ORM에서 관계(Relationship)를 통해 연결된 객체(예: `Employee` 객체에서 `Department` 객체)를 로드하는 방식은 쿼리 성능에 매우 큰 영향을 미칩니다. 잘못된 로딩 전략은 **N+1 쿼리 문제**와 같은 심각한 성능 병목을 유발할 수 있습니다. SQLAlchemy는 다양한 로딩 전략을 제공하며, 쿼리 패턴에 따라 적절한 전략을 선택하는 것이 중요합니다.

*   **N+1 쿼리 문제:**
    메인 쿼리에서 N개의 로우를 가져온 후, 각 로우에 연결된 관계 데이터를 가져오기 위해 N개의 추가 쿼리가 발생하는 문제입니다. (총 1 + N개의 쿼리 발생)

*   **주요 로딩 전략:**

    1.  **`lazy` (기본값):**
        *   **동작 방식:** 관련된 객체(예: `employee.department`)가 실제로 접근될 때 데이터베이스에서 로드합니다. 즉, 필요할 때(on-demand) 로드합니다.
        *   **장점:** 초기 쿼리 속도가 빠르고, 필요 없는 관계 데이터는 로드하지 않아 메모리를 절약합니다.
        *   **단점:** N+1 쿼리 문제가 발생하기 쉽습니다. 특히 반복문 내에서 관계 데이터에 접근할 경우, 매번 추가 쿼리가 발생하여 성능이 급격히 저하됩니다.
        ```python
        # N+1 쿼리 문제 예시 (lazy 로딩 시)
        employees = session.query(Employee).all() # 1번 쿼리 (모든 직원)
        for emp in employees:
            print(f"{emp.first_name} works in {emp.department.department_name}") # N번 쿼리 (각 직원의 부서 정보)
        ```

    2.  **`joined` (Eager Loading - JOIN 사용):**
        *   **동작 방식:** `JOIN`을 사용하여 관계된 객체를 메인 쿼리와 함께 로드합니다. 단일 쿼리로 모든 필요한 데이터를 가져옵니다.
        *   **장점:** N+1 쿼리 문제를 해결합니다. 관련된 데이터를 한 번에 가져오므로 반복문 내에서 효율적입니다.
        *   **단점:** `JOIN`으로 인해 결과 셋의 로우 수가 증가할 수 있으며(특히 One-to-Many 관계에서), 불필요한 컬럼까지 로드하여 메모리 사용량이 늘어날 수 있습니다.
        ```python
        from sqlalchemy.orm import joinedload

        # N+1 쿼리 문제 해결 (joined 로딩)
        employees = session.query(Employee).options(joinedload(Employee.department)).all() # 1번 쿼리 (직원과 부서 정보를 JOIN하여 가져옴)
        for emp in employees:
            print(f"{emp.first_name} works in {emp.department.department_name}") # 추가 쿼리 없음
        ```

    3.  **`subquery` (Eager Loading - Subquery 사용):**
        *   **동작 방식:** 메인 쿼리 실행 후, 별도의 서브쿼리를 사용하여 관계된 객체를 로드합니다. 일반적으로 두 개의 쿼리가 실행됩니다.
        *   **장점:** `joined` 로딩과 달리 중복 데이터가 발생하지 않습니다. N+1 쿼리 문제를 해결합니다.
        *   **단점:** `joined`보다 복잡한 쿼리가 생성될 수 있으며, 경우에 따라 `joined`보다 느릴 수 있습니다.
        ```python
        from sqlalchemy.orm import subqueryload

        employees = session.query(Employee).options(subqueryload(Employee.department)).all() # 2번 쿼리 (직원 쿼리, 부서 쿼리)
        ```

    4.  **`selectin` (Eager Loading - IN 절 사용):**
        *   **동작 방식:** 메인 쿼리 실행 후, `IN` 절을 사용하여 관계된 객체를 로드합니다. 일반적으로 두 개의 쿼리가 실행됩니다.
        *   **장점:** N+1 쿼리 문제를 해결하며, `joined` 로딩보다 메모리 효율적일 수 있습니다. 특히 Many-to-Many 관계에서 효율적입니다.
        *   **단점:** `IN` 절의 크기가 너무 커지면 성능 문제가 발생할 수 있습니다.
        ```python
        from sqlalchemy.orm import selectinload

        employees = session.query(Employee).options(selectinload(Employee.department)).all() # 2번 쿼리 (직원 쿼리, 부서 쿼리)
        ```

**실무에서 로딩 전략 선택 가이드라인:**
*   **기본적으로 `lazy` 로딩을 사용하되, N+1 쿼리 문제가 발생할 가능성이 있는 곳(특히 반복문 내에서 관계 데이터에 접근하는 경우)에서는 `joinedload`, `subqueryload`, `selectinload` 중 적절한 것을 선택합니다.**
*   **`joinedload`:** One-to-One 또는 One-to-Many 관계에서 대부분의 경우 좋은 성능을 보입니다. 결과 셋의 크기가 크게 늘어나지 않는다면 좋은 선택입니다.
*   **`selectinload`:** Many-to-Many 관계나 `joinedload`로 인해 결과 셋이 너무 커지는 경우에 고려합니다.
*   **`subqueryload`:** `selectinload`와 유사하게 N+1 문제를 해결하지만, 생성되는 SQL이 더 복잡할 수 있습니다.
*   **항상 `echo=True`로 생성되는 SQL 쿼리를 확인하고, `EXPLAIN`을 통해 쿼리 실행 계획을 분석하여 성능을 검증해야 합니다.**

### 1.8. 원시 SQL 실행: `text()` 함수

SQLAlchemy ORM을 사용하더라도, 때로는 ORM만으로는 표현하기 어렵거나 특정 DBMS의 고유 기능을 사용해야 할 때, 또는 성능 최적화를 위해 원시 SQL 쿼리를 직접 실행해야 하는 경우가 있습니다. SQLAlchemy의 `text()` 함수를 사용하면 SQL Injection 위험 없이 안전하게 원시 SQL을 실행할 수 있습니다.

```python
from sqlalchemy import text

session = SessionLocal()
try:
    # 원시 SQL 쿼리 실행
    result = session.execute(text("SELECT * FROM employees WHERE salary > :min_salary"), {"min_salary": 70000}).fetchall()
    print("\nEmployees with salary > 70000 (raw SQL):")
    for row in result:
        print(row)

    # DDL/DML 실행 (예시)
    session.execute(text("DROP TABLE IF EXISTS temp_table;"))
    session.execute(text("CREATE TABLE temp_table (id INT PRIMARY KEY, name VARCHAR(50));"))
    session.execute(text("INSERT INTO temp_table (id, name) VALUES (1, 'Test');"))
    session.commit()

finally:
    session.close()
```

## 2. 성능 최적화: 커넥션 풀 (Connection Pool) (애플리케이션 성능 향상)

데이터베이스 연결은 비용이 많이 드는 작업입니다. 새로운 연결을 생성할 때마다 네트워크 통신, 인증, 자원 할당 등의 오버헤드가 발생합니다. 웹 애플리케이션이나 대규모 데이터 처리 시스템에서 데이터베이스 연결을 매번 생성하고 닫는 것은 비효율적이며, 성능 저하와 자원 낭비의 주요 원인이 됩니다. **커넥션 풀(Connection Pool)**은 이러한 문제를 해결하기 위한 핵심적인 성능 최적화 기법입니다.

### 2.1. 커넥션 풀의 필요성 및 동작 원리

*   **필요성:**
    *   **연결 생성 오버헤드 감소:** 데이터베이스 연결 생성에 소요되는 시간과 자원을 절약합니다.
    *   **응답 시간 단축:** 애플리케이션이 데이터베이스 연결을 요청할 때, 새로운 연결을 생성하는 대신 풀에서 기존 연결을 즉시 빌려 사용할 수 있으므로 응답 시간이 단축됩니다.
    *   **자원 관리 효율화:** 데이터베이스 서버의 연결 수를 제한하고 관리하여 서버 자원(메모리, CPU)을 효율적으로 사용하고 과부하를 방지합니다.
    *   **안정성 향상:** 연결 재사용을 통해 데이터베이스 서버의 부하를 줄이고 안정성을 높입니다.

*   **동작 원리:**
    1.  **초기화:** 애플리케이션 시작 시, 커넥션 풀은 미리 설정된 최소 개수(`pool_size`)의 데이터베이스 연결을 생성하여 풀에 저장합니다.
    2.  **연결 요청:** 애플리케이션에서 데이터베이스 작업이 필요할 때, 풀에 연결을 요청합니다.
    3.  **연결 대여:** 풀은 사용 가능한 연결이 있으면 해당 연결을 애플리케이션에 빌려줍니다.
    4.  **연결 반환:** 작업 완료 후 애플리케이션은 연결을 닫는 대신 풀에 반환합니다. 반환된 연결은 다른 요청을 위해 재사용될 수 있습니다.
    5.  **연결 확장/축소:** 풀에 사용 가능한 연결이 없으면, `max_overflow` 설정에 따라 추가 연결을 생성하거나, `pool_timeout` 설정에 따라 대기합니다. 사용량이 줄어들면 풀은 연결을 자동으로 축소하여 자원을 반환합니다.

### 2.2. SQLAlchemy 커넥션 풀 설정 및 활용 (실무 모범 사례)

SQLAlchemy는 기본적으로 커넥션 풀을 내장하고 있으며, `create_engine` 함수에서 다양한 풀 관련 매개변수를 설정하여 애플리케이션의 특성과 데이터베이스 환경에 맞게 최적화할 수 있습니다.

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:3306/"
    f"{os.getenv('DB_NAME')}"
)

# 커넥션 풀 설정 예시
# pool_size: 풀에 유지할 최소 연결 수 (기본값 5)
# max_overflow: 풀 크기 초과 시 추가로 생성할 수 있는 최대 연결 수 (기본값 10)
# pool_timeout: 풀에서 연결을 기다리는 최대 시간 (초, 기본값 30). 초과 시 에러 발생
# pool_recycle: 연결을 재활용하기 전 최대 수명 (초). MySQL의 wait_timeout보다 짧게 설정하여 
#               오래된 연결로 인한 'MySQL server has gone away' 오류 방지 (기본값 -1, 재활용 안 함)
# pool_pre_ping: 연결 사용 전 유효성 검사. 끊어진 연결로 인한 오류를 방지 (기본값 False)
engine = create_engine(
    DATABASE_URL,
    pool_size=10,          
    max_overflow=20,       
    pool_timeout=30,       
    pool_recycle=3600,     
    pool_pre_ping=True,    
    echo=False             # SQL 쿼리 출력 여부 (운영 환경에서는 False 권장)
)

# 세션 팩토리 생성 (ORM 사용 시)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 세션 사용 예시 (with 문을 사용하여 세션 자동 종료 및 연결 반환)
def get_db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close() # 세션 종료 시 연결이 풀에 반환됨

# 실제 사용 코드 (예시)
# for session in get_db_session():
#     employees = session.query(Employee).all()
#     for emp in employees:
#         print(emp)
```

**실무적 조언:**
*   **`pool_size`와 `max_overflow` 튜닝:** 애플리케이션의 동시성 요구사항과 데이터베이스 서버의 최대 연결 수를 고려하여 적절한 값을 설정해야 합니다. 너무 작으면 대기 시간이 길어지고, 너무 크면 데이터베이스 서버에 과부하를 줄 수 있습니다.
*   **`pool_recycle` 설정:** MySQL의 `wait_timeout` 설정(기본 8시간)보다 짧게 설정하여, 유휴 상태로 오래된 연결이 데이터베이스 서버에서 끊어지는 문제를 방지해야 합니다. (예: `pool_recycle=3600` (1시간))
*   **`pool_pre_ping=True`:** 연결을 사용하기 전에 유효성을 검사하여 끊어진 연결로 인한 오류를 방지합니다. 약간의 오버헤드가 있지만, 안정성 측면에서 권장됩니다.
*   **`engine.dispose()`:** 애플리케이션 종료 시에는 `engine.dispose()`를 호출하여 모든 풀 연결을 명시적으로 닫아주는 것이 좋습니다. 이는 리소스 누수를 방지합니다.