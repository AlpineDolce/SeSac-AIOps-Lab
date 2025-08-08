<h2>SQL 핵심 문법: 서브쿼리와 CASE 문 (실무 활용 중심)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-26

<h2>문서 목표</h2>
<p>이 문서는 SQL의 <strong>서브쿼리와 CASE 문</strong>을 활용하여 데이터를 더욱 유연하고 강력하게 조회하고 조작하는 방법을 심도 있게 다룹니다. 복잡한 비즈니스 로직을 SQL 쿼리 내에서 구현하고, 다양한 분석 시나리오에 적용하는 능력을 상세한 예제와 함께 설명하여, SQL을 활용한 고급 데이터 분석의 견고한 기초를 다지는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 서브쿼리 (Subquery): SQL 쿼리 내의 쿼리](#1-서브쿼리-subquery-sql-쿼리-내의-쿼리)
  - [1.1. 서브쿼리의 개념과 장점](#11-서브쿼리의-개념과-장점)
  - [1.2. `WHERE` 절 서브쿼리: 조건부 필터링의 확장](#12-where-절-서브쿼리-조건부-필터링의-확장)
    - [1.2.1. 단일 로우 서브쿼리](#121-단일-로우-서브쿼리)
    - [1.2.2. 다중 로우 서브쿼리](#122-다중-로우-서브쿼리)
  - [1.3. `FROM` 절 서브쿼리 (인라인 뷰): 임시 테이블 생성](#13-from-절-서브쿼리-인라인-뷰-임시-테이블-생성)
  - [1.4. `SELECT` 절 서브쿼리 (스칼라 서브쿼리): 로우별 상세 정보 추가](#14-select-절-서브쿼리-스칼라-서브쿼리-로우별-상세-정보-추가)
  - [1.5. `INSERT`, `UPDATE`, `DELETE` 문에서의 서브쿼리: DML 작업의 유연성](#15-insert-update-delete-문에서의-서브쿼리-dml-작업의-유연성)
  - [1.6. 상관 서브쿼리 (Correlated Subquery): 로우별 동적 조건](#16-상관-서브쿼리-correlated-subquery-로우별-동적-조건)
  - [1.7. 서브쿼리 성능 최적화 전략: 효율적인 쿼리 작성](#17-서브쿼리-성능-최적화-전략-효율적인-쿼리-작성)
    - [1.7.1. `EXISTS` vs `IN`: 상황에 맞는 선택](#171-exists-vs-in-상황에-맞는-선택)
    - [1.7.2. `JOIN`으로 대체: 서브쿼리의 강력한 대안](#172-join으로-대체-서브쿼리의-강력한-대안)
    - [1.7.3. 인덱스 활용: 서브쿼리 성능의 핵심](#173-인덱스-활용-서브쿼리-성능의-핵심)
    - [1.7.4. `WITH` 절 (CTE - Common Table Expression) 활용: 가독성과 재사용성](#174-with-절-cte---common-table-expression-활용-가독성과-재사용성)
    - [1.7.5. 스칼라 서브쿼리 남용 금지: 성능 저하의 주범](#175-스칼라-서브쿼리-남용-금지-성능-저하의-주범)
- [2. `CASE` 문: SQL의 강력한 조건부 로직](#2-case-문-sql의-강력한-조건부-로직)
  - [2.1. `CASE` 문의 기본 구조와 유형](#21-case-문의-기본-구조와-유형)
    - [2.1.1. 단순 `CASE` (Simple `CASE` Expression)](#211-단순-case-simple-case-expression)
    - [2.1.2. 검색 `CASE` (Searched `CASE` Expression)](#212-검색-case-searched-case-expression)
  - [2.2. `CASE` 문의 실무 활용: 데이터 분석의 다양한 측면](#22-case-문의-실무-활용-데이터-분석의-다양한-측면)
    - [2.2.1. 데이터 분류 및 레이블링 (`SELECT` 절)](#221-데이터-분류-및-레이블링-select-절)
    - [2.2.2. 조건부 필터링 (`WHERE` 절)](#222-조건부-필터링-where-절)
    - [2.2.3. 조건부 정렬 (`ORDER BY` 절)](#223-조건부-정렬-order-by-절)
    - [2.2.4. 조건부 집계 (Conditional Aggregation) (`GROUP BY` 및 집계 함수)](#224-조건부-집계-conditional-aggregation-group-by-및-집계-함수)
    - [2.2.5. 조건부 데이터 수정 (`UPDATE` 문)](#225-조건부-데이터-수정-update-문)
  - [2.3. `CASE` 문 성능 고려사항 및 최적화](#23-case-문-성능-고려사항-및-최적화)

---

## 1. 서브쿼리 (Subquery): SQL 쿼리 내의 쿼리

서브쿼리(Subquery)는 하나의 SQL 쿼리 내부에 포함된 또 다른 쿼리입니다. 메인 쿼리에 필요한 데이터를 제공하거나, 복잡한 조건을 정의하는 데 사용됩니다. 서브쿼리는 괄호 `()`로 묶어서 사용하며, 다양한 절(SELECT, FROM, WHERE, HAVING, INSERT, UPDATE, DELETE) 내에서 활용될 수 있습니다.

### 1.1. 서브쿼리의 개념과 장점

*   **개념:** 메인 쿼리의 일부로 실행되어 메인 쿼리에 필요한 데이터를 반환하는 쿼리입니다. 마치 작은 질문을 먼저 해결하고, 그 답을 바탕으로 큰 질문에 답하는 것과 같습니다.
*   **장점:**
    *   **복잡한 쿼리 단순화:** 여러 단계의 논리적 처리를 하나의 쿼리 내에서 구현하여 쿼리 가독성을 높입니다. (예: 단계별 필터링, 중간 집계)
    *   **데이터 재활용:** 한 번 계산된 결과를 다른 쿼리에서 재사용할 수 있습니다.
    *   **유연한 조건 설정:** 동적으로 변하는 조건에 따라 데이터를 필터링하거나 조작할 수 있습니다.

**실무적 관점:** 서브쿼리는 복잡한 비즈니스 질문에 답하고, 다단계 분석을 수행하는 데 필수적인 도구입니다. 특히 `JOIN`으로 해결하기 어려운 문제나, 특정 시점의 스냅샷 데이터를 기반으로 분석해야 할 때 유용하게 사용됩니다.

### 1.2. `WHERE` 절 서브쿼리: 조건부 필터링의 확장

`WHERE` 절에서 서브쿼리를 사용하여 조건을 정의합니다. 서브쿼리의 결과에 따라 메인 쿼리의 로우를 필터링합니다.

#### 1.2.1. 단일 로우 서브쿼리
서브쿼리가 단일 값(하나의 컬럼, 하나의 로우)을 반환할 때 사용합니다. 비교 연산자(`=`, `>`, `<`, `>=`, `<=`, `!=`)와 함께 사용됩니다.

```sql
-- 전체 직원 평균 급여보다 많이 받는 직원 조회
SELECT employee_id, first_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

**실무 활용 시나리오:**
*   **기준 값과의 비교:** 특정 기준(평균, 최대/최소, 특정 임계값)보다 크거나 작은 데이터를 필터링할 때.
*   **동적 임계값 설정:** 고정된 값이 아닌, 데이터에 따라 변하는 임계값을 기준으로 필터링할 때.

#### 1.2.2. 다중 로우 서브쿼리
서브쿼리가 여러 로우를 반환할 때 사용합니다. `IN`, `NOT IN`, `ANY`, `ALL` 연산자와 함께 사용됩니다.

*   **`IN` / `NOT IN`:** 서브쿼리 결과 집합에 값이 포함되는지/포함되지 않는지 확인합니다.

    ```sql
    -- 'Development' 부서에 속한 직원 조회
    SELECT employee_id, first_name, department_id
    FROM employees
    WHERE department_id IN (SELECT department_id FROM departments WHERE department_name = 'Development');
    ```

    **실무 활용 시나리오:**
    *   **특정 그룹에 속한 데이터 필터링:** 특정 조건(예: 특정 부서, 특정 상품 카테고리)을 만족하는 데이터 목록을 기준으로 메인 쿼리를 필터링할 때.
    *   **`SEMI JOIN` 패턴:** `IN` 서브쿼리는 `SEMI JOIN`과 유사하게 작동하여, 서브쿼리 결과에 존재하는 로우만 메인 쿼리에서 선택합니다.

*   **`ANY` / `SOME`:** 서브쿼리 결과 중 하나라도 조건을 만족하면 참입니다. (예: `> ANY`는 서브쿼리 결과의 최솟값보다 크면 참)

    ```sql
    -- 'HR' 부서의 어떤 직원보다 급여를 많이 받는 직원 조회 (HR 부서 직원 중 가장 적게 받는 직원보다 급여가 많은 직원)
    SELECT employee_id, first_name, salary
    FROM employees
    WHERE salary > ANY (SELECT salary FROM employees WHERE department_id = (SELECT department_id FROM departments WHERE department_name = 'HR'));
    ```

*   **`ALL`:** 서브쿼리 결과의 모든 값이 조건을 만족해야 참입니다. (예: `> ALL`은 서브쿼리 결과의 최댓값보다 크면 참)

    ```sql
    -- 'HR' 부서의 모든 직원보다 급여를 많이 받는 직원 조회 (HR 부서 직원 중 가장 많이 받는 직원보다 급여가 많은 직원)
    SELECT employee_id, first_name, salary
    FROM employees
    WHERE salary > ALL (SELECT salary FROM employees WHERE department_id = (SELECT department_id FROM departments WHERE department_name = 'HR'));
    ```

    **실무 활용 시나리오:**
    *   **최대/최소값 기준 필터링:** 특정 그룹의 모든 구성원보다 크거나 작은 값을 가진 데이터를 찾을 때.

### 1.3. `FROM` 절 서브쿼리 (인라인 뷰): 임시 테이블 생성

`FROM` 절에서 서브쿼리를 사용하여 임시 테이블(인라인 뷰, Derived Table)을 생성하고, 이 임시 테이블을 메인 쿼리에서 사용합니다. 복잡한 집계나 중간 계산 결과를 메인 쿼리에 연결할 때 유용합니다. 반드시 별칭(Alias)을 지정해야 합니다.

```sql
-- 부서별 평균 급여를 계산한 후, 그 결과를 메인 쿼리에서 사용하여 직원별 부서 평균 급여 조회
SELECT
    e.first_name, e.last_name, e.salary, d.department_name, avg_dept_salary.avg_salary
FROM
    employees e
JOIN
    departments d ON e.department_id = d.department_id
JOIN
    (SELECT department_id, AVG(salary) AS avg_salary FROM employees GROUP BY department_id) AS avg_dept_salary
ON
    e.department_id = avg_dept_salary.department_id;
```

**실무 활용 시나리오:**
*   **복잡한 집계:** 여러 단계의 집계가 필요할 때 중간 결과를 인라인 뷰로 만들면 쿼리 구조가 명확해지고 재사용성이 높아집니다.
*   **데이터 전처리:** 메인 쿼리에서 사용하기 전에 데이터를 미리 필터링, 정제, 변환하는 데 사용합니다.
*   **랭킹/순위 계산:** 특정 그룹 내에서 순위를 매긴 후, 그 결과를 메인 쿼리에서 활용할 때 (윈도우 함수가 없는 DBMS에서 유용).

### 1.4. `SELECT` 절 서브쿼리 (스칼라 서브쿼리): 로우별 상세 정보 추가

`SELECT` 절에서 서브쿼리를 사용하여 각 로우마다 단일 값을 반환합니다. 이를 스칼라 서브쿼리(Scalar Subquery)라고 합니다. 주로 각 로우에 대한 추가적인 요약 정보나 관련 데이터를 표시할 때 사용됩니다.

```sql
-- 각 직원의 급여와 해당 직원이 속한 부서의 평균 급여를 함께 조회
SELECT
    e.first_name, e.salary,
    (SELECT department_name FROM departments WHERE department_id = e.department_id) AS department_name,
    (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id) AS avg_dept_salary
FROM
    employees e;
```

**실무 활용 시나리오:**
*   **로우별 통계치 표시:** 각 로우에 대해 해당 로우가 속한 그룹의 평균, 최대값 등 요약 통계치를 함께 표시할 때.
*   **관련 정보 조회:** `JOIN` 없이 간단하게 다른 테이블의 특정 정보를 가져올 때 (단, 서브쿼리가 단일 값만 반환해야 함).

**주의점:** 스칼라 서브쿼리는 반드시 **단일 값**을 반환해야 합니다. 만약 서브쿼리가 여러 로우를 반환하면 오류가 발생합니다. 또한, 각 로우마다 서브쿼리가 실행되므로 대규모 데이터셋에서는 성능 저하의 원인이 될 수 있습니다. 이 경우 `JOIN`이나 `WITH` 절(CTE)을 활용하는 것이 더 효율적일 수 있습니다.

### 1.5. `INSERT`, `UPDATE`, `DELETE` 문에서의 서브쿼리: DML 작업의 유연성

DML(Data Manipulation Language) 문에서도 서브쿼리를 사용하여 데이터를 삽입, 수정, 삭제할 수 있습니다. 이를 통해 복잡한 조건에 따라 데이터를 일괄적으로 처리할 수 있습니다.

*   **`INSERT` 문:** `INSERT INTO ... SELECT` 구문을 사용하여 다른 테이블의 조회 결과를 삽입합니다.

    ```sql
    -- old_employees 테이블에서 급여가 60000 이상인 직원만 new_employees 테이블로 복사
    INSERT INTO new_employees (employee_id, first_name, last_name, salary)
    SELECT employee_id, first_name, last_name, salary
    FROM old_employees
    WHERE salary >= 60000;
    ```

*   **`UPDATE` 문:** 서브쿼리의 결과를 사용하여 컬럼 값을 업데이트하거나, 서브쿼리를 `WHERE` 절의 조건으로 사용합니다.

    ```sql
    -- 평균 급여보다 적게 받는 직원의 급여를 10% 인상
    UPDATE employees
    SET salary = salary * 1.10
    WHERE salary < (SELECT AVG(salary) FROM (SELECT * FROM employees) AS temp_employees); -- MySQL에서는 UPDATE/DELETE 시 동일 테이블 참조를 위해 FROM 절 서브쿼리 필요

    -- 특정 부서(예: 'HR')에 속한 직원의 직무를 'HR_Specialist'로 변경
    UPDATE employees
    SET job_id = 'HR_Specialist'
    WHERE department_id = (SELECT department_id FROM departments WHERE department_name = 'HR');
    ```

*   **`DELETE` 문:** 서브쿼리를 `WHERE` 절의 조건으로 사용하여 삭제할 로우를 지정합니다.

    ```sql
    -- 프로젝트에 배정되지 않은 직원 삭제
    DELETE FROM employees
    WHERE employee_id NOT IN (SELECT employee_id FROM project_assignments);
    ```

**실무 활용 시나리오:**
*   **데이터 마이그레이션:** 특정 조건에 맞는 데이터를 다른 테이블로 옮길 때.
*   **데이터 클렌징:** 특정 조건을 만족하는 데이터를 일괄적으로 수정하거나 삭제할 때.
*   **데이터 동기화:** 다른 테이블의 최신 정보를 기반으로 현재 테이블을 업데이트할 때.

### 1.6. 상관 서브쿼리 (Correlated Subquery): 로우별 동적 조건

상관 서브쿼리(Correlated Subquery)는 메인 쿼리의 로우가 처리될 때마다 서브쿼리가 반복적으로 실행되는 형태입니다. 서브쿼리가 메인 쿼리의 컬럼을 참조할 때 발생합니다. 일반 서브쿼리보다 성능상 불리할 수 있지만, 특정 복잡한 로직을 구현하는 데 필수적입니다.

```sql
-- 각 직원의 급여가 자신이 속한 부서의 평균 급여보다 높은 직원 조회
SELECT e.first_name, e.salary, e.department_id
FROM employees e
WHERE e.salary > (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id);
```

**실무 활용 시나리오:**
*   **그룹 내 비교:** 각 그룹 내에서 특정 조건을 만족하는 로우를 찾거나, 그룹별 순위 계산 등 복잡한 그룹 내 비교에 유용합니다.
*   **최초/최종 이벤트 찾기:** 각 고객의 첫 구매일, 마지막 로그인 시간 등 특정 그룹 내에서 가장 오래되거나 최신 이벤트를 찾을 때.

**성능 고려사항:** 메인 쿼리의 각 로우마다 서브쿼리가 실행되므로, 데이터 양이 많을수록 성능 저하가 심각할 수 있습니다. 가능한 경우 `JOIN`이나 윈도우 함수(Window Function)로 대체하는 것을 고려해야 합니다.

### 1.7. 서브쿼리 성능 최적화 전략: 효율적인 쿼리 작성

서브쿼리는 강력하지만, 잘못 사용하면 쿼리 성능에 심각한 영향을 미칠 수 있습니다. 다음 사항들을 고려하여 최적화해야 합니다.

#### 1.7.1. `EXISTS` vs `IN`: 상황에 맞는 선택

*   **`IN`:** 서브쿼리가 먼저 실행되어 모든 결과를 메모리에 로드한 후, 메인 쿼리의 각 로우와 비교합니다. 서브쿼리의 결과 집합이 작을 때 효율적입니다.
*   **`EXISTS`:** 메인 쿼리의 각 로우에 대해 서브쿼리를 실행합니다. 서브쿼리는 조건에 맞는 첫 번째 로우를 찾으면 즉시 `TRUE`를 반환하고 실행을 멈춥니다. 서브쿼리의 결과 집합이 크거나, 서브쿼리 내에 `LIMIT`와 같은 최적화가 가능한 경우 효율적입니다.
    일반적으로 상관 서브쿼리(외부 쿼리의 컬럼을 참조하는 서브쿼리)의 경우 `EXISTS`가 `IN`보다 유리한 경우가 많습니다.

**실무 팁:** `EXISTS`는 주로 존재 여부만 확인할 때, `IN`은 실제 값 목록을 기준으로 필터링할 때 사용합니다. 쿼리 실행 계획을 통해 어떤 방식이 더 효율적인지 확인하는 것이 중요합니다.

#### 1.7.2. `JOIN`으로 대체: 서브쿼리의 강력한 대안

많은 서브쿼리, 특히 `FROM` 절 서브쿼리나 상관 서브쿼리는 `JOIN`으로 대체할 수 있습니다. `JOIN`은 데이터베이스 옵티마이저가 더 효율적인 실행 계획을 세울 수 있도록 돕는 경우가 많습니다.

```sql
-- 서브쿼리 (상관 서브쿼리) 예시
SELECT e.first_name, e.salary
FROM employees e
WHERE e.salary > (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id);

-- JOIN으로 대체한 예시 (더 효율적일 가능성 높음)
SELECT e.first_name, e.salary
FROM employees e
JOIN (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
) AS dept_avg ON e.department_id = dept_avg.department_id
WHERE e.salary > dept_avg.avg_salary;
```

**실무 팁:** `JOIN`으로 대체할 수 있다면 대부분의 경우 `JOIN`이 서브쿼리보다 성능상 유리합니다. 특히 대규모 데이터셋에서는 `JOIN`을 우선적으로 고려해야 합니다.

#### 1.7.3. 인덱스 활용: 서브쿼리 성능의 핵심

서브쿼리 내에서 사용되는 컬럼에도 적절한 인덱스를 생성해야 합니다. 특히 `WHERE` 절의 조건이나 `JOIN` 조건으로 사용되는 컬럼에 인덱스가 있으면 성능이 크게 향상됩니다.

**실무 팁:** 서브쿼리 내부에서 필터링이나 조인에 사용되는 컬럼에 인덱스가 없으면 Full Table Scan이 발생하여 성능이 저하될 수 있습니다. `EXPLAIN` 명령으로 인덱스 사용 여부를 확인하고, 필요시 인덱스를 추가합니다.

#### 1.7.4. `WITH` 절 (CTE - Common Table Expression) 활용: 가독성과 재사용성

`WITH` 절은 복잡한 쿼리를 여러 개의 작은 논리적 단위로 분리하여 정의할 수 있게 해줍니다. 이는 쿼리 가독성을 높이고, 재사용성을 향상시키며, 특정 상황에서는 성능 최적화에도 도움을 줍니다.

```sql
-- WITH 절을 사용한 예시
WITH DepartmentAvgSalary AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT e.first_name, e.salary, das.avg_salary
FROM employees e
JOIN DepartmentAvgSalary das ON e.department_id = das.department_id
WHERE e.salary > das.avg_salary;
```

**실무 팁:** CTE는 쿼리 구조를 명확하게 하고, 복잡한 계산을 단계별로 수행할 수 있게 합니다. 특히 재귀 CTE는 계층형 데이터를 처리하는 데 유용합니다.

#### 1.7.5. 스칼라 서브쿼리 남용 금지: 성능 저하의 주범

`SELECT` 절의 스칼라 서브쿼리는 각 로우마다 실행되므로, 대규모 데이터셋에서는 성능 저하의 주범이 될 수 있습니다. 가능한 경우 `JOIN`이나 윈도우 함수로 대체하는 것을 고려합니다.

**실무 팁:** 스칼라 서브쿼리 대신 `LEFT JOIN`을 사용하여 필요한 정보를 가져오는 것이 대부분의 경우 더 효율적입니다. 윈도우 함수는 그룹 내 통계치를 계산하는 데 매우 강력한 대안입니다.

**최종 실무 팁: `EXPLAIN` 명령으로 쿼리 실행 계획 분석**
쿼리 성능을 최적화하기 위해서는 `EXPLAIN` 명령을 사용하여 쿼리 실행 계획을 분석하는 습관을 들이는 것이 중요합니다. 이를 통해 어떤 부분이 병목 지점인지 파악하고 적절한 최적화 전략을 적용할 수 있습니다. `EXPLAIN` 결과에서 `Using filesort`, `Using temporary`, `Full Table Scan` 등이 자주 보인다면 쿼리 최적화가 필요하다는 신호입니다.

서브쿼리(Subquery)는 하나의 SQL 쿼리 내부에 포함된 또 다른 쿼리입니다. 메인 쿼리에 필요한 데이터를 제공하거나, 복잡한 조건을 정의하는 데 사용됩니다. 서브쿼리는 괄호 `()`로 묶어서 사용하며, 다양한 절(SELECT, FROM, WHERE, HAVING, INSERT, UPDATE, DELETE) 내에서 활용될 수 있습니다.

## 2. `CASE` 문: SQL의 강력한 조건부 로직

`CASE` 문은 SQL 쿼리 내에서 조건에 따라 다른 값을 반환할 수 있도록 하는 강력한 조건문입니다. 데이터를 분류하거나, 복잡한 비즈니스 로직을 구현하거나, 보고서의 가독성을 높이는 데 매우 유용합니다. 데이터 분석가가 비즈니스 요구사항을 SQL로 직접 구현할 때 가장 많이 활용하는 기능 중 하나입니다.

### 2.1. `CASE` 문의 기본 구조와 유형

`CASE` 문은 크게 두 가지 형태로 사용됩니다.

#### 2.1.1. 단순 `CASE` (Simple `CASE` Expression)

특정 컬럼의 값과 일치하는지 여부에 따라 다른 값을 반환할 때 사용합니다. 주로 고정된 값에 대한 조건 분기에 적합합니다.

```sql
CASE expression
    WHEN value1 THEN result1
    WHEN value2 THEN result2
    ...
    [ELSE default_result] -- 선택 사항: 어떤 WHEN 조건에도 해당하지 않을 때 반환할 값
END
```

**예시:** 직무(job_id)에 따라 직무 등급 분류

```sql
SELECT
    employee_id,
    first_name,
    job_id,
    CASE job_id
        WHEN 'DEV' THEN 'Developer'
        WHEN 'HR' THEN 'Human Resources'
        WHEN 'MGR' THEN 'Manager'
        ELSE 'Other' -- 위에 정의되지 않은 모든 job_id에 대해 'Other' 반환
    END AS job_category
FROM
    employees;
```

#### 2.1.2. 검색 `CASE` (Searched `CASE` Expression)

다양한 조건식(`WHERE` 절에서 사용하는 조건과 유사)을 사용하여 더 복잡한 논리를 구현할 때 사용합니다. `IF-ELSE IF-ELSE` 구조와 유사하며, 가장 유연하게 활용될 수 있는 형태입니다.

```sql
CASE
    WHEN condition1 THEN result1
    WHEN condition2 THEN result2
    ...
    [ELSE default_result] -- 선택 사항: 어떤 WHEN 조건에도 해당하지 않을 때 반환할 값
END
```

**예시:** 급여 수준에 따라 직원 분류

```sql
SELECT
    employee_id,
    first_name,
    salary,
    CASE
        WHEN salary >= 70000 THEN 'High Salary'
        WHEN salary >= 50000 AND salary < 70000 THEN 'Medium Salary'
        ELSE 'Low Salary'
    END AS salary_tier
FROM
    employees;
```

**실무 팁: `ELSE` 절의 중요성**
`CASE` 문에 `ELSE` 절을 생략하면, 어떤 `WHEN` 조건에도 해당하지 않는 로우는 `NULL`을 반환합니다. 명확한 결과를 위해 `ELSE` 절을 명시적으로 작성하는 것이 좋습니다.

**실무 팁: `WHEN` 조건의 순서**
검색 `CASE` 문에서 `WHEN` 조건들은 위에서부터 순서대로 평가됩니다. 따라서 조건의 순서에 따라 결과가 달라질 수 있으므로, 더 구체적인 조건을 먼저 배치하는 것이 중요합니다.

### 2.2. `CASE` 문의 실무 활용: 데이터 분석의 다양한 측면

`CASE` 문은 `SELECT`, `WHERE`, `ORDER BY`, `GROUP BY`, `HAVING`, `UPDATE` 등 다양한 SQL 절에서 활용될 수 있으며, 데이터 분석의 여러 단계에서 핵심적인 역할을 합니다.

#### 2.2.1. 데이터 분류 및 레이블링 (`SELECT` 절)

특정 조건에 따라 데이터를 분류하고 새로운 레이블을 부여할 때 가장 많이 사용됩니다. 보고서의 가독성을 높이고, 데이터를 특정 기준으로 그룹화하기 전에 전처리하는 데 유용합니다.

```sql
-- 고객의 총 구매액에 따라 고객 등급 분류
SELECT
    customer_id,
    total_purchase_amount,
    CASE
        WHEN total_purchase_amount >= 1000000 THEN 'VIP'
        WHEN total_purchase_amount >= 500000 THEN 'Gold'
        WHEN total_purchase_amount >= 100000 THEN 'Silver'
        ELSE 'Bronze'
    END AS customer_tier
FROM
    customers;
```

#### 2.2.2. 조건부 필터링 (`WHERE` 절)

`WHERE` 절에서 `CASE` 문을 사용하여 복잡한 조건에 따라 로우를 필터링할 수 있습니다. 이는 동적으로 변하는 필터링 로직을 구현할 때 유용합니다.

```sql
-- (예시: 특정 기간 동안의 매출 데이터를 조회하되, 주말에는 특정 상품만 포함)
SELECT order_id, order_date, total_amount
FROM orders
WHERE
    CASE
        WHEN DAYOFWEEK(order_date) IN (1, 7) -- 일요일(1), 토요일(7)
            THEN product_id IN (101, 105) -- 주말에는 특정 상품만
        ELSE TRUE -- 주중에는 모든 상품
    END;
```

#### 2.2.3. 조건부 정렬 (`ORDER BY` 절)

`ORDER BY` 절에서 `CASE` 문을 사용하여 특정 조건에 따라 정렬 순서를 동적으로 변경할 수 있습니다. 이는 보고서나 대시보드에서 사용자가 원하는 정렬 기준을 유연하게 적용할 때 유용합니다.

```sql
-- 상품 상태에 따라 정렬 (품절 -> 재고 부족 -> 정상 순서로, 같은 상태 내에서는 상품명 오름차순)
SELECT product_name, stock_quantity, status
FROM products
ORDER BY
    CASE status
        WHEN 'Out of Stock' THEN 1
        WHEN 'Low Stock' THEN 2
        WHEN 'In Stock' THEN 3
        ELSE 4
    END ASC,
    product_name ASC;
```

#### 2.2.4. 조건부 집계 (Conditional Aggregation) (`GROUP BY` 및 집계 함수)

`CASE` 문은 집계 함수와 함께 사용하여 조건부 집계(Conditional Aggregation)를 수행할 때 매우 강력합니다. 특정 조건에 맞는 로우에 대해서만 집계를 수행하여 다양한 관점의 요약 통계를 얻을 수 있습니다. 이는 피벗 테이블과 유사한 형태의 보고서를 만들 때 매우 유용합니다.

```sql
-- 부서별 남성 직원 수와 여성 직원 수 조회
SELECT
    d.department_name,
    SUM(CASE WHEN e.gender = 'M' THEN 1 ELSE 0 END) AS male_employees,
    SUM(CASE WHEN e.gender = 'F' THEN 1 ELSE 0 END) AS female_employees,
    COUNT(*) AS total_employees
FROM
    departments d
LEFT JOIN
    employees e ON d.department_id = e.department_id
GROUP BY
    d.department_name;
```

**실무 팁: 조건부 집계의 활용**
*   **피벗 테이블(Pivot Table) 구현:** `CASE` 문과 집계 함수를 사용하여 관계형 데이터베이스에서 피벗 테이블과 유사한 형태의 보고서를 생성할 수 있습니다.
*   **다양한 지표 계산:** 특정 조건에 따른 매출, 사용자 수, 이벤트 발생 횟수 등 다양한 비즈니스 지표를 하나의 쿼리에서 효율적으로 계산할 수 있습니다.

#### 2.2.5. 조건부 데이터 수정 (`UPDATE` 문)

`UPDATE` 문에서 `CASE` 문을 사용하여 조건에 따라 다른 컬럼 값을 업데이트할 수 있습니다. 이는 데이터 클렌징이나 일괄적인 데이터 변경 작업에 유용합니다.

```sql
-- 급여가 60000 미만인 직원은 5% 인상, 60000 이상 80000 미만인 직원은 3% 인상
UPDATE employees
SET salary = CASE
    WHEN salary < 60000 THEN salary * 1.05
    WHEN salary >= 60000 AND salary < 80000 THEN salary * 1.03
    ELSE salary
END;
```

### 2.3. `CASE` 문 성능 고려사항 및 최적화

`CASE` 문은 매우 유연하고 강력하지만, 복잡한 `CASE` 문이나 대규모 데이터셋에 적용할 경우 성능에 영향을 미칠 수 있습니다.

*   **인덱스 활용의 한계:** `CASE` 문 내의 조건식에 사용되는 컬럼에 인덱스가 있더라도, `CASE` 문 자체는 인덱스를 직접적으로 활용하기 어렵습니다. 데이터베이스는 `CASE` 문을 평가하기 위해 모든 로우를 스캔해야 할 수 있습니다.
*   **복잡도 증가:** `CASE` 문이 너무 복잡해지면 쿼리 가독성이 떨어지고 유지보수가 어려워집니다. 이 경우 비즈니스 로직을 애플리케이션 레벨에서 처리하거나, 데이터를 미리 전처리하여 저장하는 방안을 고려할 수 있습니다.
*   **대안 고려:** 단순한 조건이라면 `IF()` (MySQL), `IFNULL()`, `COALESCE()`와 같은 내장 함수를 사용하는 것이 더 간결하고 효율적일 수 있습니다.

**실무 팁:** `CASE` 문은 SQL 쿼리 내에서 비즈니스 로직을 구현하는 데 매우 유용하지만, 성능 병목이 발생한다면 `EXPLAIN` 명령을 통해 실행 계획을 분석하고, 필요한 경우 `JOIN`이나 서브쿼리, 또는 애플리케이션 레벨에서의 처리 등 다른 대안을 고려해야 합니다.(#2-case-문)