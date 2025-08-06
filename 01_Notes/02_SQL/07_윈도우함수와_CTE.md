<h2>윈도우 함수와 CTE</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-27

<h2>문서 목표</h2>
<p>이 문서는 복잡한 분석과 보고서 작성을 위한 고급 SQL 쿼리 기법인 윈도우 함수와 CTE(Common Table Expression)를 학습합니다. 윈도우 함수를 통해 그룹 내 순위, 비율, 누적 합계 등을 계산하고, CTE를 통해 복잡한 쿼리의 가독성과 재사용성을 높이는 방법을 익힙니다.</p>

<h2>목차</h2>

- [1. 윈도우 함수 (Window Functions): 고급 분석](#1-윈도우-함수-window-functions-고급-분석)
  - [1.1. 윈도우 함수의 개념과 필요성](#11-윈도우-함수의-개념과-필요성)
  - [1.2. `OVER` 절: 파티션(Partition)과 정렬(Order)](#12-over-절-파티션partition과-정렬order)
  - [1.3. 순위 함수 (RANK, DENSE\_RANK, ROW\_NUMBER, NTILE)](#13-순위-함수-rank-dense_rank-row_number-ntile)
  - [1.4. 집계 윈도우 함수 (SUM, AVG, COUNT, MIN, MAX over partition)](#14-집계-윈도우-함수-sum-avg-count-min-max-over-partition)
  - [1.5. 분석 함수 (LEAD, LAG, FIRST\_VALUE, LAST\_VALUE)](#15-분석-함수-lead-lag-first_value-last_value)
  - [1.6. 프레임(Frame) 정의 (ROWS BETWEEN, RANGE BETWEEN)](#16-프레임frame-정의-rows-between-range-between)
- [2. CTE (Common Table Expression): 쿼리 가독성 향상](#2-cte-common-table-expression-쿼리-가독성-향상)
  - [2.1. `WITH` 절을 이용한 CTE 정의](#21-with-절을-이용한-cte-정의)
  - [2.2. CTE의 장점 (가독성, 재사용성, 재귀 쿼리)](#22-cte의-장점-가독성-재사용성-재귀-쿼리)
  - [2.3. 다중 CTE 및 재귀 CTE 활용](#23-다중-cte-및-재귀-cte-활용)

---

## 1. 윈도우 함수 (Window Functions): 고급 분석

윈도우 함수는 SQL 쿼리에서 로우들의 집합(윈도우)에 대해 계산을 수행하지만, `GROUP BY`처럼 로우를 그룹화하여 단일 로우로 줄이지 않고, 각 로우에 대해 개별적으로 결과를 반환합니다. 이는 순위, 이동 평균, 누적 합계 등 복잡한 분석을 수행할 때 매우 유용합니다.

### 1.1. 윈도우 함수의 개념과 필요성

*   **개념:** `OVER` 절과 함께 사용되어 특정 범위(윈도우) 내의 로우들에 대해 집계 또는 분석 함수를 적용합니다. 결과는 각 로우에 대해 반환되며, 로우의 수는 변하지 않습니다.
*   **필요성:**
    *   **그룹 내 순위/비율 계산:** 특정 그룹 내에서 각 로우의 순위나 전체 대비 비율을 쉽게 계산할 수 있습니다.
    *   **이동 평균/누적 합계:** 시계열 데이터에서 이동 평균이나 누적 합계를 효율적으로 계산할 수 있습니다.
    *   **복잡한 분석 쿼리 간소화:** 서브쿼리나 `SELF JOIN`으로 복잡하게 작성해야 했던 쿼리를 윈도우 함수를 사용하여 훨씬 간결하게 작성할 수 있습니다.

*   **윈도우 함수와 성능 (실무적 관점):**
    윈도우 함수는 매우 강력하고 편리하지만, 내부적으로는 정렬(Sorting)과 많은 메모리를 사용할 수 있어 대용량 데이터 처리 시 성능 병목의 원인이 될 수 있습니다. 특히 `PARTITION BY` 절에 카디널리티가 매우 높은 컬럼을 사용하거나, `ORDER BY` 절에 인덱스가 없는 컬럼을 사용하면 성능이 크게 저하될 수 있습니다.
    *   **최적화 방안:**
        *   **인덱스 활용:** `PARTITION BY`와 `ORDER BY` 절에 사용되는 컬럼에 복합 인덱스를 생성하는 것이 가장 효과적인 최적화 방법입니다.
        *   **처리 범위 최소화:** `WHERE` 절을 사용하여 윈도우 함수가 적용될 전체 로우의 수를 최대한 줄인 후, 윈도우 함수를 적용합니다.
        *   **CTE 활용:** 복잡한 윈도우 함수 계산은 CTE로 분리하여 중간 결과를 먼저 생성하고, 이후에 메인 쿼리에서 해당 결과를 사용하는 것이 쿼리 가독성과 유지보수, 그리고 때로는 성능에도 도움이 될 수 있습니다.

### 1.2. `OVER` 절: 파티션(Partition)과 정렬(Order)

`OVER` 절은 윈도우 함수가 적용될 로우들의 집합(윈도우)을 정의합니다.

*   **`PARTITION BY`:** 로우들을 하나 이상의 컬럼을 기준으로 그룹으로 나눕니다. 각 파티션은 독립적으로 윈도우 함수가 적용됩니다. (논리적인 그룹화, `GROUP BY`와 달리 로우 수를 줄이지 않음)
*   **`ORDER BY`:** 파티션 내에서 로우들의 순서를 정의합니다. 순위 함수나 분석 함수에서 특히 중요합니다.

```sql
-- 부서별 직원들의 급여 순위 매기기
SELECT
    employee_id, first_name, department_id, salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS salary_rank_in_dept
FROM employees;
```

### 1.3. 순위 함수 (RANK, DENSE_RANK, ROW_NUMBER, NTILE)

윈도우 내에서 각 로우의 순위를 계산합니다.

*   **`RANK()`:** 동일한 값에는 같은 순위를 부여하고, 다음 순위는 건너뜁니다. (예: 1, 2, 2, 4)
*   **`DENSE_RANK()`:** 동일한 값에는 같은 순위를 부여하고, 다음 순위를 건너뛰지 않습니다. (예: 1, 2, 2, 3)
*   **`ROW_NUMBER()`:** 각 로우에 고유한 순차 번호를 부여합니다. 동일한 값이라도 다른 순위를 가집니다. (예: 1, 2, 3, 4)
*   **`NTILE(n)`:** 윈도우 내의 로우들을 `n`개의 그룹으로 나누고, 각 로우에 그룹 번호를 부여합니다.

```sql
-- 부서별 급여 순위 (RANK, DENSE_RANK, ROW_NUMBER 비교)
SELECT
    employee_id, first_name, department_id, salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank_salary,
    DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dense_rank_salary,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS row_num_salary
FROM employees;

-- 전체 직원을 급여 기준으로 4분위(Quartile)로 나누기
SELECT
    employee_id, first_name, salary,
    NTILE(4) OVER (ORDER BY salary DESC) AS salary_quartile
FROM employees;
```

### 1.4. 집계 윈도우 함수 (SUM, AVG, COUNT, MIN, MAX over partition)

일반 집계 함수를 `OVER` 절과 함께 사용하여 윈도우 내에서 집계 값을 계산합니다. `GROUP BY`와 달리 원본 로우를 유지하면서 집계 값을 각 로우에 추가합니다.

```sql
-- 부서별 평균 급여를 각 직원 로우에 추가
SELECT
    employee_id, first_name, department_id, salary,
    AVG(salary) OVER (PARTITION BY department_id) AS avg_dept_salary
FROM employees;

-- 직원별 누적 급여 합계 (고용일 기준)
SELECT
    employee_id, first_name, hire_date, salary,
    SUM(salary) OVER (ORDER BY hire_date) AS cumulative_salary
FROM employees;
```

**실전 예제: 그룹 내 비율 (Percent of Total) 계산**
데이터 분석에서 특정 그룹 내에서 각 항목이 차지하는 비율을 계산하는 것은 매우 흔한 분석 시나리오입니다. 윈도우 함수를 사용하면 이를 매우 효율적으로 수행할 수 있습니다.

```sql
-- 부서별 각 직원의 급여가 해당 부서 총 급여에서 차지하는 비율 계산
SELECT
    employee_id, first_name, department_id, salary,
    SUM(salary) OVER (PARTITION BY department_id) AS total_dept_salary, -- 부서별 총 급여
    (salary / SUM(salary) OVER (PARTITION BY department_id)) * 100 AS percent_of_dept_salary -- 부서 내 비율
FROM employees
ORDER BY department_id, percent_of_dept_salary DESC;

-- 전체 직원의 급여 중 각 직원의 급여가 차지하는 비율 계산
SELECT
    employee_id, first_name, salary,
    SUM(salary) OVER () AS total_company_salary, -- 전체 총 급여 (PARTITION BY 없음)
    (salary / SUM(salary) OVER ()) * 100 AS percent_of_company_salary -- 전체 비율
FROM employees
ORDER BY percent_of_company_salary DESC;
```
이처럼 `SUM(컬럼) OVER (PARTITION BY 그룹_컬럼)`을 사용하면 각 로우에 해당 그룹의 총합을 가져올 수 있으며, 이를 활용하여 비율을 쉽게 계산할 수 있습니다.

### 1.5. 분석 함수 (LEAD, LAG, FIRST_VALUE, LAST_VALUE)

윈도우 내에서 현재 로우를 기준으로 다른 로우의 값을 가져오는 함수입니다. 시계열 데이터 분석에 특히 유용합니다.

*   **`LEAD(컬럼, offset, default)`:** 현재 로우 다음의 `offset` 위치에 있는 로우의 `컬럼` 값을 가져옵니다. `offset`이 없으면 1, `default`는 값이 없을 때 반환할 값.
*   **`LAG(컬럼, offset, default)`:** 현재 로우 이전의 `offset` 위치에 있는 로우의 `컬럼` 값을 가져옵니다.
*   **`FIRST_VALUE(컬럼)`:** 윈도우 내에서 첫 번째 로우의 `컬럼` 값을 가져옵니다.
*   **`LAST_VALUE(컬럼)`:** 윈도우 내에서 마지막 로우의 `컬럼` 값을 가져옵니다.

```sql
-- 직원별 다음 고용일과 이전 고용일 조회
SELECT
    employee_id, first_name, hire_date,
    LAG(hire_date, 1, NULL) OVER (ORDER BY hire_date) AS previous_hire_date,
    LEAD(hire_date, 1, NULL) OVER (ORDER BY hire_date) AS next_hire_date
FROM employees;

-- 부서별 가장 높은 급여와 가장 낮은 급여를 각 직원 로우에 표시
SELECT
    employee_id, first_name, department_id, salary,
    FIRST_VALUE(salary) OVER (PARTITION BY department_id ORDER BY salary DESC) AS highest_dept_salary,
    LAST_VALUE(salary) OVER (PARTITION BY department_id ORDER BY salary DESC
                             ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS lowest_dept_salary
FROM employees;
```

**`LAST_VALUE` 사용 시 `ROWS BETWEEN`의 중요성:**
`LAST_VALUE` 함수는 기본적으로 현재 로우부터 윈도우의 끝까지를 프레임으로 간주합니다. 따라서 `ORDER BY` 절만 사용하면 `LAST_VALUE`는 항상 현재 로우의 값을 반환하거나, `ORDER BY`에 따라 정렬된 마지막 로우의 값을 반환하게 되어 예상과 다른 결과를 줄 수 있습니다. 위 예시처럼 `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`과 같이 명시적으로 전체 파티션을 프레임으로 지정해야 파티션 내의 실제 마지막 값을 정확히 가져올 수 있습니다. 이는 `FIRST_VALUE`와 `LAST_VALUE`의 동작 방식 차이에서 비롯됩니다.

### 1.6. 프레임(Frame) 정의 (ROWS BETWEEN, RANGE BETWEEN)

`ORDER BY` 절과 함께 사용하여 윈도우 함수가 적용될 로우의 범위를 더욱 세밀하게 제어합니다. 기본값은 `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`입니다.

*   **`ROWS BETWEEN ... AND ...`:** 물리적인 로우의 개수를 기준으로 범위를 지정합니다.
    *   `UNBOUNDED PRECEDING`: 현재 로우 이전의 모든 로우
    *   `CURRENT ROW`: 현재 로우
    *   `N PRECEDING`: 현재 로우 이전 `N`개의 로우
    *   `N FOLLOWING`: 현재 로우 이후 `N`개의 로우
    *   `UNBOUNDED FOLLOWING`: 현재 로우 이후의 모든 로우
*   **`RANGE BETWEEN ... AND ...`:** 현재 로우의 값과 지정된 값 범위 내에 있는 로우들을 기준으로 범위를 지정합니다.

```sql
-- 직원별 3일 이동 평균 급여 (고용일 기준)
SELECT
    employee_id, first_name, hire_date, salary,
    AVG(salary) OVER (ORDER BY hire_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_salary
FROM employees;
```

## 2. CTE (Common Table Expression): 쿼리 가독성 향상

CTE(Common Table Expression)는 `WITH` 절을 사용하여 정의하는 임시적인 명명된 결과 집합입니다. 복잡한 쿼리를 여러 개의 논리적인 단계로 나누어 작성할 수 있게 하여 쿼리의 가독성과 재사용성을 크게 향상시킵니다.

### 2.1. `WITH` 절을 이용한 CTE 정의

```sql
WITH cte_name AS (
    SELECT ...
)
SELECT ...
FROM cte_name
WHERE ...;
```

### 2.2. CTE의 장점 (가독성, 재사용성, 재귀 쿼리)

*   **가독성:** 복잡한 쿼리를 작은 논리적 단위로 분리하여 이해하기 쉽게 만듭니다.
*   **재사용성:** 하나의 CTE를 메인 쿼리 또는 다른 CTE에서 여러 번 참조할 수 있습니다.
*   **재귀 쿼리:** 계층적 데이터를 처리하는 재귀 쿼리를 작성할 수 있습니다.

```sql
-- 예시: 부서별 평균 급여보다 많이 받는 직원 조회 (CTE 사용)
WITH DepartmentAvgSalary AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT
    e.first_name, e.last_name, e.salary, d.department_name
FROM
    employees e
JOIN
    departments d ON e.department_id = d.department_id
JOIN
    DepartmentAvgSalary das ON e.department_id = das.department_id
WHERE
    e.salary > das.avg_salary;
```

### 2.3. 다중 CTE 및 재귀 CTE 활용

*   **다중 CTE:** 여러 개의 CTE를 콤마로 구분하여 정의하고, 서로 참조할 수 있습니다.

    ```sql
    WITH HighSalaryEmployees AS (
        SELECT employee_id, first_name, salary, department_id
        FROM employees
        WHERE salary >= 65000
    ),
    DepartmentInfo AS (
        SELECT department_id, department_name
        FROM departments
    )
    SELECT
        hse.first_name, hse.salary, di.department_name
    FROM
        HighSalaryEmployees hse
    JOIN
        DepartmentInfo di ON hse.department_id = di.department_id;
    ```

*   **재귀 CTE:** 자기 자신을 참조하여 계층적 데이터를 처리할 때 사용합니다. (예: 조직도, 댓글 스레드)

    ```sql
    -- 예시: 직원-상사 계층 구조 조회 (employees 테이블에 manager_id 컬럼이 있다고 가정)
    WITH RECURSIVE EmployeeHierarchy AS (
        -- 앵커 멤버 (재귀의 시작점: 최상위 관리자)
        SELECT
            employee_id, first_name, manager_id, 0 AS level
        FROM employees
        WHERE manager_id IS NULL

        UNION ALL

        -- 재귀 멤버 (하위 직원)
        SELECT
            e.employee_id, e.first_name, e.manager_id, eh.level + 1
        FROM employees e
        INNER JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
    )
    SELECT * FROM EmployeeHierarchy ORDER BY level, employee_id;
    ```
