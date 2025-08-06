<h2>SQLAlchemy ORM 활용</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-28

<h2>문서 목표</h2>
<p>이 문서는 파이썬의 대표적인 ORM(Object-Relational Mapping) 라이브러리인 SQLAlchemy를 사용하여, 객체 지향적인 방식으로 데이터베이스를 다루는 방법을 학습합니다. ORM의 핵심 개념을 이해하고, 복잡한 데이터 모델과 관계를 파이썬 코드로 효과적으로 관리하는 능력을 기르는 것을 목표로 합니다.</p>

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

#### 1.1.1. ORM의 장점과 단점

*   **장점:**
    *   **생산성 향상:** SQL 쿼리 작성 시간을 줄이고, 파이썬 코드만으로 데이터베이스 작업을 수행할 수 있습니다.
    *   **객체 지향적 개발:** 데이터베이스 로우를 파이썬 객체로 다룰 수 있어 코드의 가독성과 유지보수성이 향상됩니다.
    *   **DBMS 독립성:** ORM 추상화 계층 덕분에 데이터베이스 종류를 변경해도 코드 수정이 최소화됩니다.
    *   **SQL Injection 방지:** 대부분의 ORM은 쿼리 파라미터화를 자동으로 처리하여 SQL Injection을 방지합니다.
*   **단점:**
    *   **학습 곡선:** ORM의 개념과 사용법을 익히는 데 시간이 필요합니다.
    *   **성능 오버헤드:** 복잡한 쿼리의 경우 ORM이 생성하는 SQL이 비효율적일 수 있으며, 직접 SQL을 작성하는 것보다 성능이 떨어질 수 있습니다.
    *   **추상화 누수 (Abstraction Leakage):** 때로는 ORM의 추상화가 충분하지 않아 복잡하거나 성능에 민감한 쿼리의 경우 ORM이 생성하는 SQL이 비효율적일 수 있습니다. 이럴 때는 `SQLAlchemy Core`를 사용하거나 `text()` 함수를 통해 **직접 SQL을 작성하여 ORM의 한계를 보완**하는 것이 일반적인 실무 전략입니다.

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

### 1.7. 관계 로딩(Relationship Loading) 전략 (Lazy, Eager, Joined)

관계형 데이터를 로드하는 방식은 쿼리 성능에 큰 영향을 미칩니다. SQLAlchemy는 여러 로딩 전략을 제공합니다.

*   **`lazy` (기본값):** 관련된 객체가 실제로 접근될 때 로드합니다. (N+1 쿼리 문제 발생 가능성)
*   **`joined`:** `JOIN`을 사용하여 관계된 객체를 메인 쿼리와 함께 로드합니다. (단일 쿼리, 중복 데이터 발생 가능성)
*   **`subquery`:** 서브쿼리를 사용하여 관계된 객체를 로드합니다. (두 개의 쿼리, 중복 데이터 없음)
*   **`selectin`:** `IN` 절을 사용하여 관계된 객체를 로드합니다. (두 개의 쿼리, N+1 문제 해결, `joined`보다 효율적일 수 있음)

```python
# 관계 로딩 전략 설정 예시
# class Employee(Base):
#     ...
#     department = relationship("Department", back_populates="employees", lazy="joined") # joined 로딩

# N+1 쿼리 문제 예시 (lazy 로딩 시)
# employees = session.query(Employee).all()
# for emp in employees:
#     print(f"{emp.first_name} works in {emp.department.department_name}") # department 접근 시마다 추가 쿼리 발생

# N+1 쿼리 문제 해결 (joined 로딩)
# employees = session.query(Employee).options(joinedload(Employee.department)).all()
# for emp in employees:
#     print(f"{emp.first_name} works in {emp.department.department_name}") # 단일 쿼리로 모두 로드
```

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

## 2. 성능 최적화: 커넥션 풀 (Connection Pool)

데이터베이스 연결은 비용이 많이 드는 작업입니다. 커넥션 풀은 미리 데이터베이스 연결을 생성해두고 재사용함으로써 연결 생성 오버헤드를 줄이고 애플리케이션 성능을 향상시킵니다.

### 2.1. 커넥션 풀의 필요성 및 동작 원리

*   **필요성:** 웹 애플리케이션이나 대규모 데이터 처리 시스템에서 데이터베이스 연결을 매번 생성하고 닫는 것은 비효율적입니다. 연결 생성에 시간이 소요되고, 서버 자원을 낭비할 수 있습니다.
*   **동작 원리:**
    1.  애플리케이션 시작 시 일정 수의 데이터베이스 연결을 미리 생성하여 풀(Pool)에 저장합니다.
    2.  데이터베이스 작업이 필요할 때, 풀에서 기존 연결을 빌려 사용합니다.
    3.  작업 완료 후 연결을 닫는 대신 풀에 반환하여 재사용할 수 있도록 합니다.

### 2.2. SQLAlchemy 커넥션 풀 설정 및 활용

SQLAlchemy는 기본적으로 커넥션 풀을 내장하고 있으며, `create_engine` 함수에서 다양한 풀 관련 매개변수를 설정할 수 있습니다.

```python
from sqlalchemy import create_engine
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
engine = create_engine(
    DATABASE_URL,
    pool_size=10,          # 풀에 유지할 최소 연결 수
    max_overflow=20,       # 풀 크기 초과 시 추가로 생성할 수 있는 최대 연결 수
    pool_timeout=30,       # 풀에서 연결을 기다리는 최대 시간 (초)
    pool_recycle=3600,     # 연결을 재활용하기 전 최대 수명 (초, MySQL의 wait_timeout보다 짧게 설정)
    pool_pre_ping=True     # 연결 사용 전 유효성 검사 (끊어진 연결 방지)
)

# 세션 팩토리는 동일하게 사용
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 사용 예시 (SessionLocal을 통해 연결을 얻고 반환)
# session = SessionLocal()
# try:
#     # 데이터베이스 작업
#     pass
# finally:
#     session.close() # 연결을 풀에 반환
```