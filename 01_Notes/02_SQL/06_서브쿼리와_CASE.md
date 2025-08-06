<h2>서브쿼리와 CASE문</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-21

<h2>문서 목표</h2>
<p>이 문서는 복잡한 데이터 추출 및 조건부 로직 구현을 위한 서브쿼리와 <code>CASE</code> 문의 다양한 활용법을 익혀, 데이터 분석가가 실무에서 복잡한 비즈니스 질문에 답할 수 있는 능력을 기르는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 쿼리 속의 쿼리: 서브쿼리(Subquery) 심화](#1-쿼리-속의-쿼리-서브쿼리subquery-심화)
  - [1.1. 서브쿼리의 개념과 유형](#11-서브쿼리의-개념과-유형)
    - [1.1.1. 스칼라 서브쿼리 (단일 값 반환)](#111-스칼라-서브쿼리-단일-값-반환)
    - [1.1.2. 로우 서브쿼리 (단일 로우 반환)](#112-로우-서브쿼리-단일-로우-반환)
    - [1.1.3. 테이블 서브쿼리 (다중 로우/컬럼 반환)](#113-테이블-서브쿼리-다중-로우컬럼-반환)
  - [1.2. 서브쿼리 활용 예시](#12-서브쿼리-활용-예시)
    - [3.2.1. `WHERE` 절에서 서브쿼리 사용 (`IN`, `NOT IN`, `ANY`, `ALL`)](#321-where-절에서-서브쿼리-사용-in-not-in-any-all)
    - [1.2.2. `FROM` 절에서 서브쿼리 사용 (인라인 뷰)](#122-from-절에서-서브쿼리-사용-인라인-뷰)
    - [1.2.3. `SELECT` 절에서 서브쿼리 사용](#123-select-절에서-서브쿼리-사용)
  - [1.3. 상관 서브쿼리 (Correlated Subquery): 외부 쿼리와의 연동](#13-상관-서브쿼리-correlated-subquery-외부-쿼리와의-연동)
    - [1.3.1. `EXISTS`와 `NOT EXISTS` 활용](#131-exists와-not-exists-활용)
    - [1.3.2. 비교 분석용 상관 서브쿼리](#132-비교-분석용-상관-서브쿼리)
    - [1.3.3. `EXISTS` vs `IN` 성능 고려사항](#133-exists-vs-in-성능-고려사항)
  - [1.4. 다중 레벨 서브쿼리: 중첩된 쿼리의 힘](#14-다중-레벨-서브쿼리-중첩된-쿼리의-힘)
- [1.5. 윈도우 함수 (Window Functions): 고급 분석의 시작](#15-윈도우-함수-window-functions-고급-분석의-시작)
- [2. `CASE` 문: 조건부 로직의 구현](#2-case-문-조건부-로직의-구현)
  - [2.1. `CASE` 문의 기본: 단순 vs 검색](#21-case-문의-기본-단순-vs-검색)
    - [2.1.1. 단순 `CASE` (값 비교)](#211-단순-case-값-비교)
    - [2.1.2. 검색 `CASE` (조건 비교)](#212-검색-case-조건-비교)
  - [2.2. 중첩 `CASE`와 복합 조건: 복잡한 비즈니스 로직](#22-중첩-case와-복합-조건-복잡한-비즈니스-로직)
  - [2.3. `CASE` 문을 활용한 피벗 테이블: 데이터 재구성](#23-case-문을-활용한-피벗-테이블-데이터-재구성)

---

## 1. 쿼리 속의 쿼리: 서브쿼리(Subquery) 심화

서브쿼리(Subquery)는 다른 SQL 문 내부에 포함된 `SELECT` 문입니다. 메인 쿼리에 필요한 데이터를 제공하거나, 복잡한 조건을 정의하는 데 사용됩니다. 서브쿼리는 괄호 `()`로 묶이며, 메인 쿼리보다 먼저 실행됩니다.

### 1.1. 서브쿼리의 개념과 유형

*   **개념:** 하나의 쿼리 안에 또 다른 쿼리가 중첩되어 있는 형태입니다. 내부 쿼리(서브쿼리)의 결과가 외부 쿼리(메인 쿼리)의 입력으로 사용됩니다.
*   **유형:** 서브쿼리가 반환하는 결과의 형태에 따라 여러 유형으로 나눌 수 있습니다.

#### 1.1.1. 스칼라 서브쿼리 (단일 값 반환)

단일 로우의 단일 컬럼, 즉 하나의 값만 반환하는 서브쿼리입니다. `SELECT` 절, `WHERE` 절, `HAVING` 절 등 단일 값이 필요한 모든 곳에서 사용할 수 있습니다.

```sql
-- 전체 직원의 평균 급여보다 많이 받는 직원 조회
SELECT employee_id, first_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

#### 1.1.2. 로우 서브쿼리 (단일 로우 반환)

단일 로우의 여러 컬럼 값을 반환하는 서브쿼리입니다. 주로 `WHERE` 절에서 비교 연산자와 함께 사용됩니다.

```sql
-- 특정 부서(예: department_id가 1인 부서)의 최고 급여를 받는 직원 조회
SELECT employee_id, first_name, salary, department_id
FROM employees
WHERE (department_id, salary) = (SELECT department_id, MAX(salary) FROM employees WHERE department_id = 1);
```

#### 1.1.3. 테이블 서브쿼리 (다중 로우/컬럼 반환)

여러 로우와 여러 컬럼을 반환하는 서브쿼리입니다. 주로 `FROM` 절에서 인라인 뷰(Inline View)로 사용되거나, `WHERE` 절의 `IN`, `EXISTS` 등과 함께 사용됩니다.

```sql
-- 각 부서의 평균 급여보다 많이 받는 직원 조회
SELECT e.first_name, e.salary, e.department_id
FROM employees e
JOIN (
    SELECT department_id, AVG(salary) AS avg_dept_salary
    FROM employees
    GROUP BY department_id
) AS dept_avg ON e.department_id = dept_avg.department_id
WHERE e.salary > dept_avg.avg_dept_salary;
```

### 1.2. 서브쿼리 활용 예시

서브쿼리는 다양한 SQL 절에서 활용되어 복잡한 데이터 조회 요구사항을 해결합니다.

#### 3.2.1. `WHERE` 절에서 서브쿼리 사용 (`IN`, `NOT IN`, `ANY`, `ALL`)

*   `IN`: 서브쿼리 결과 집합에 값이 포함되는지 확인.
*   `NOT IN`: 서브쿼리 결과 집합에 값이 포함되지 않는지 확인.
*   `ANY`: 서브쿼리 결과 집합의 어떤 값이라도 조건에 맞으면 참.
*   `ALL`: 서브쿼리 결과 집합의 모든 값이 조건에 맞으면 참.

```sql
-- 프로젝트에 참여하는 직원 조회
SELECT employee_id, first_name
FROM employees
WHERE employee_id IN (SELECT employee_id FROM project_assignments);

-- 프로젝트에 참여하지 않는 직원 조회
SELECT employee_id, first_name
FROM employees
WHERE employee_id NOT IN (SELECT employee_id FROM project_assignments);

-- 어떤 부서의 최소 급여보다 많이 받는 직원 조회
SELECT employee_id, first_name, salary
FROM employees
WHERE salary > ANY (SELECT MIN(salary) FROM employees GROUP BY department_id);

-- 모든 부서의 최대 급여보다 많이 받는 직원 조회 (즉, 전체 최고 급여보다 많이 받는 직원)
SELECT employee_id, first_name, salary
FROM employees
WHERE salary > ALL (SELECT MAX(salary) FROM employees GROUP BY department_id);
```

#### 1.2.2. `FROM` 절에서 서브쿼리 사용 (인라인 뷰)

서브쿼리의 결과를 임시 테이블(인라인 뷰)처럼 사용하여 메인 쿼리에서 조회합니다. 복잡한 중간 계산 결과를 활용할 때 유용합니다.

```sql
-- 각 부서의 평균 급여를 계산한 후, 그 결과를 사용하여 직원 정보와 함께 조회
SELECT
    e.first_name, e.last_name,
    d.department_name,
    dept_avg.avg_salary
FROM
    employees e
JOIN
    departments d ON e.department_id = d.department_id
JOIN
    (SELECT department_id, AVG(salary) AS avg_salary FROM employees GROUP BY department_id) AS dept_avg
    ON e.department_id = dept_avg.department_id;
```

#### 1.2.3. `SELECT` 절에서 서브쿼리 사용

스칼라 서브쿼리만 `SELECT` 절에서 사용할 수 있습니다. 각 로우에 대해 서브쿼리가 실행되어 단일 값을 반환합니다.

```sql
-- 각 직원의 급여와 해당 직원이 속한 부서의 평균 급여를 함께 조회
SELECT
    e.first_name, e.salary,
    (SELECT AVG(salary) FROM employees WHERE department_id = e.department_id) AS dept_avg_salary
FROM
    employees e;
```

### 1.3. 상관 서브쿼리 (Correlated Subquery): 외부 쿼리와의 연동

상관 서브쿼리는 외부 쿼리의 로우에 따라 서브쿼리가 반복적으로 실행되는 형태입니다. 외부 쿼리의 컬럼을 서브쿼리 내에서 참조합니다. 비상관 서브쿼리보다 성능이 떨어질 수 있으므로 주의해서 사용해야 합니다.

#### 1.3.1. `EXISTS`와 `NOT EXISTS` 활용

`EXISTS`는 서브쿼리가 반환하는 로우가 하나라도 있으면 `TRUE`를 반환하고, 없으면 `FALSE`를 반환합니다. 서브쿼리의 실제 데이터는 중요하지 않고, 존재 여부만 확인할 때 사용합니다.

```sql
-- 프로젝트에 참여하는 직원 조회 (EXISTS 사용)
SELECT employee_id, first_name
FROM employees e
WHERE EXISTS (SELECT 1 FROM project_assignments pa WHERE pa.employee_id = e.employee_id);

-- 프로젝트에 참여하지 않는 직원 조회 (NOT EXISTS 사용)
SELECT employee_id, first_name
FROM employees e
WHERE NOT EXISTS (SELECT 1 FROM project_assignments pa WHERE pa.employee_id = e.employee_id);
```

#### 1.3.2. 비교 분석용 상관 서브쿼리

각 그룹 내에서 특정 조건을 만족하는 로우를 찾을 때 유용합니다.

```sql
-- 각 부서에서 가장 높은 급여를 받는 직원 조회
SELECT employee_id, first_name, salary, department_id
FROM employees e1
WHERE salary = (SELECT MAX(salary) FROM employees e2 WHERE e1.department_id = e2.department_id);
```

#### 1.3.3. `EXISTS` vs `IN` 성능 고려사항

`EXISTS`와 `IN`은 서브쿼리의 결과를 기반으로 메인 쿼리를 필터링할 때 사용됩니다. 두 연산자는 때때로 상호 교환 가능하지만, 내부적인 동작 방식과 성능 특성이 다르므로 상황에 따라 적절한 것을 선택하는 것이 중요합니다.

*   **`IN`:** 서브쿼리가 먼저 실행되어 모든 결과를 메모리에 로드한 후, 메인 쿼리의 각 로우와 비교합니다. 서브쿼리의 결과 집합이 작을 때 효율적입니다.
*   **`EXISTS`:** 메인 쿼리의 각 로우에 대해 서브쿼리를 실행합니다. 서브쿼리는 조건에 맞는 첫 번째 로우를 찾으면 즉시 `TRUE`를 반환하고 실행을 멈춥니다. 서브쿼리의 결과 집합이 크거나, 서브쿼리 내에 `LIMIT`와 같은 최적화가 가능한 경우 효율적입니다.

**일반적인 성능 팁:**
*   **서브쿼리 결과가 작을 때:** `IN`이 유리할 수 있습니다.
*   **서브쿼리 결과가 클 때:** `EXISTS`가 유리할 수 있습니다.
*   **상관 서브쿼리일 때:** `EXISTS`가 `IN`보다 유리한 경우가 많습니다. `IN`은 서브쿼리 결과를 모두 가져와야 하지만, `EXISTS`는 조건에 맞는 로우가 하나라도 발견되면 즉시 중단하기 때문입니다.

```sql
-- EXISTS 사용 예시 (다시 한번 강조)
SELECT employee_id, first_name
FROM employees e
WHERE EXISTS (SELECT 1 FROM project_assignments pa WHERE pa.employee_id = e.employee_id);

-- IN 사용 예시 (다시 한번 강조)
SELECT employee_id, first_name
FROM employees
WHERE employee_id IN (SELECT employee_id FROM project_assignments);
```
### 1.4. 다중 레벨 서브쿼리: CTE로 리팩토링하기

서브쿼리 안에 또 다른 서브쿼리가 포함되는 다중 레벨 서브쿼리는 복잡한 비즈니스 로직을 해결할 수 있지만, 다음과 같은 심각한 단점이 있습니다.

*   **가독성 저하:** 쿼리의 중첩 구조가 깊어질수록 코드를 이해하기 매우 어려워집니다.
*   **디버깅의 어려움:** 각 서브쿼리를 독립적으로 실행하고 테스트하기가 까다롭습니다.
*   **재사용성 부재:** 동일한 서브쿼리가 여러 번 필요할 경우 코드가 중복됩니다.

**실무에서는 복잡한 서브쿼리, 특히 `FROM` 절에 사용되는 인라인 뷰(Inline View)나 2단계 이상 중첩되는 서브쿼리는 `CTE(Common Table Expression)`를 사용하여 리팩토링하는 것을 강력히 권장합니다.** CTE는 쿼리의 가독성, 재사용성, 유지보수성을 혁신적으로 개선하는 현대 SQL의 필수 기법입니다.

CTE는 쿼리를 여러 개의 명명된 논리적 단계로 분리하여, 각 단계의 결과를 다음 단계에서 참조할 수 있게 합니다. 이는 복잡한 쿼리를 마치 프로그래밍 언어의 함수처럼 모듈화하여 작성하는 것과 유사합니다.

**[나쁜 예: 중첩 서브쿼리]**
```sql
-- 각 부서의 평균 급여보다 많이 받는 직원들 중에서,
-- 'Development' 부서에 속한 직원들의 정보를 조회
SELECT *
FROM (
    SELECT e.employee_id, e.first_name, e.salary, e.department_id
    FROM employees e
    JOIN (
        SELECT department_id, AVG(salary) AS avg_dept_salary
        FROM employees
        GROUP BY department_id
    ) AS dept_avg ON e.department_id = dept_avg.department_id
    WHERE e.salary > dept_avg.avg_dept_salary
) AS high_salary_employees
WHERE department_id = 1; -- 'Development' 부서 ID가 1이라고 가정
```

**[좋은 예: CTE 활용]**
```sql
WITH
DepartmentAvgSalary AS (
    -- 1단계: 부서별 평균 급여 계산
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
),
HighSalaryEmployees AS (
    -- 2단계: 부서 평균 급여보다 많이 받는 직원 필터링
    SELECT e.employee_id, e.first_name, e.salary, e.department_id
    FROM employees e
    JOIN DepartmentAvgSalary das ON e.department_id = das.department_id
    WHERE e.salary > das.avg_salary
)
-- 3단계: 최종 결과 조회
SELECT *
FROM HighSalaryEmployees
WHERE department_id = 1; -- 'Development' 부서 ID가 1이라고 가정
```

## 1.5. 윈도우 함수 (Window Functions): 고급 분석의 시작

윈도우 함수는 SQL 쿼리에서 로우들의 집합(윈도우)에 대해 계산을 수행하지만, `GROUP BY`처럼 로우를 그룹화하여 단일 로우로 줄이지 않고, 각 로우에 대해 개별적으로 결과를 반환합니다. 이는 순위, 이동 평균, 누적 합계 등 복잡한 분석을 수행할 때 매우 유용합니다.

**개념:** `OVER` 절과 함께 사용되어 특정 범위(윈도우) 내의 로우들에 대해 집계 또는 분석 함수를 적용합니다. 결과는 각 로우에 대해 반환되며, 로우의 수는 변하지 않습니다.

**주요 활용 분야 및 실무 분석 시나리오:**
*   **순위 계산 (`RANK`, `DENSE_RANK`, `ROW_NUMBER`):**
    *   **시나리오:** "부서별로 급여를 가장 많이 받는 상위 3명의 직원을 찾아라."
    *   **활용:** `ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC)`를 사용하여 순위를 매긴 후, `WHERE` 절(또는 서브쿼리/CTE)에서 순위가 3 이하인 직원만 필터링합니다.
*   **그룹 내 비율 계산 (`SUM` over partition):**
    *   **시나리오:** "각 직원의 급여가 소속된 부서의 총 급여에서 차지하는 비중(%)은 얼마인가?"
    *   **활용:** `salary / SUM(salary) OVER (PARTITION BY department_id)`를 사용하여 부서 내 급여 비중을 계산합니다.
*   **누적 합계 계산 (`SUM` over order by):**
    *   **시나리오:** "일별 매출 데이터에서, 월별 누적 매출액의 추이를 보고 싶다."
    *   **활용:** `SUM(daily_sales) OVER (PARTITION BY month ORDER BY date)`를 사용하여 날짜 순으로 누적 매출을 계산합니다.
*   **시계열 데이터 분석 (`LAG`, `LEAD`):**
    *   **시나리오:** "전월 대비 매출 성장률(MoM Growth)을 계산하라."
    *   **활용:** `LAG(monthly_sales, 1) OVER (ORDER BY month)`를 사용하여 전월 매출을 가져온 후, `(이번달 매출 - 전월 매출) / 전월 매출` 공식을 적용합니다.
*   **이동 평균 계산 (`AVG` over frame):**
    *   **시나리오:** "주식 가격 데이터의 7일 이동 평균을 계산하여 가격 변동성의 추세를 완만하게 보고 싶다."
    *   **활용:** `AVG(price) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`를 사용하여 이동 평균을 계산합니다.

**예시:**
```sql
-- 부서별 직원들의 급여 순위 매기기
SELECT
    employee_id, first_name, department_id, salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS salary_rank_in_dept
FROM employees;

-- 전월 대비 매출 성장률 계산 (가상 테이블 sales)
-- SELECT
--     sale_month,
--     monthly_revenue,
--     LAG(monthly_revenue, 1, 0) OVER (ORDER BY sale_month) AS previous_month_revenue,
--     (monthly_revenue - LAG(monthly_revenue, 1, 0) OVER (ORDER BY sale_month)) / LAG(monthly_revenue, 1, 0) OVER (ORDER BY sale_month) * 100 AS mom_growth_rate
-- FROM monthly_sales;
```

**더 자세한 내용은 [07_윈도우함수와_CTE.md](./07_윈도우함수와_CTE.md) 섹션을 참조하세요.** 해당 섹션에서는 윈도우 함수의 개념, `OVER` 절의 파티션과 정렬, 다양한 순위 함수, 집계 윈도우 함수, 분석 함수, 그리고 프레임 정의(`ROWS BETWEEN`, `RANGE BETWEEN`)에 대해 더욱 심도 있게 다룹니다.

## 2. `CASE` 문: 조건부 로직의 구현

`CASE` 문은 SQL 쿼리 내에서 조건부 로직을 구현할 때 사용합니다. 특정 조건에 따라 다른 값을 반환하거나, 데이터를 분류할 때 매우 유용합니다. 데이터 분석가가 데이터를 다양한 관점에서 분류하고 분석하는 데 필수적인 도구입니다.

### 2.1. `CASE` 문의 기본: 단순 vs 검색

`CASE` 문은 크게 두 가지 형태로 나뉩니다.

#### 2.1.1. 단순 `CASE` (값 비교)

특정 컬럼의 값이 미리 정의된 값들과 일치하는지 비교할 때 사용합니다. `WHEN` 절에 비교할 값을 명시합니다.

```sql
-- job_id에 따라 직무 등급 분류
SELECT
    first_name, job_id,
    CASE job_id
        WHEN 'DEV' THEN 'Developer'
        WHEN 'MGR' THEN 'Manager'
        WHEN 'HR' THEN 'Human Resources'
        ELSE 'Other'
    END AS job_category
FROM employees;
```

#### 2.1.2. 검색 `CASE` (조건 비교)

`WHEN` 절에 다양한 조건 표현식을 사용하여 더 복잡한 로직을 구현할 때 사용합니다. `IF-ELSE IF` 문과 유사합니다.

```sql
-- 급여 범위에 따라 급여 등급 분류
SELECT
    first_name, salary,
    CASE
        WHEN salary >= 70000 THEN 'A'
        WHEN salary >= 60000 THEN 'B'
        WHEN salary >= 50000 THEN 'C'
        ELSE 'D'
    END AS salary_grade
FROM employees;
```

### 2.2. 중첩 `CASE`와 복합 조건: 복잡한 비즈니스 로직

`CASE` 문 안에 또 다른 `CASE` 문을 중첩하거나, `WHEN` 절에 `AND`, `OR` 등의 논리 연산자를 사용하여 복합적인 조건을 만들 수 있습니다.

```sql
-- 부서와 급여를 기준으로 복합적인 등급 분류
SELECT
    first_name, department_id, salary,
    CASE
        WHEN department_id = 1 THEN
            CASE
                WHEN salary >= 65000 THEN 'Dept1_High'
                ELSE 'Dept1_Low'
            END
        WHEN department_id = 2 AND salary >= 58000 THEN 'Dept2_High'
        WHEN department_id = 2 AND salary < 58000 THEN 'Dept2_Low'
        ELSE 'Other_Dept'
    END AS complex_grade
FROM employees;
```

### 2.3. `CASE` 문을 활용한 피벗 테이블: 데이터 재구성

`CASE` 문과 집계 함수를 함께 사용하여 로우 데이터를 컬럼 데이터로 변환하는 피벗(Pivot) 테이블을 만들 수 있습니다. 이는 특정 카테고리별로 데이터를 요약하여 보고서를 생성할 때 매우 유용합니다.

```sql
-- 부서별, 직무별 직원 수를 피벗 테이블 형태로 조회
SELECT
    d.department_name,
    COUNT(CASE WHEN e.job_id = 'DEV' THEN 1 END) AS Developers,
    COUNT(CASE WHEN e.job_id = 'MGR' THEN 1 END) AS Managers,
    COUNT(CASE WHEN e.job_id = 'HR' THEN 1 END) AS HR_Staff,
    COUNT(*) AS Total_Employees
FROM
    departments d
LEFT JOIN
    employees e ON d.department_id = e.department_id
GROUP BY
    d.department_name
ORDER BY
    d.department_name;
```