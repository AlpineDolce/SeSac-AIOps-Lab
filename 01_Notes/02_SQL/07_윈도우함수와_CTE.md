<h2>SQL 핵심 문법: 윈도우 함수와 CTE (실무 활용 중심)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-27

<h2>문서 목표</h2>
<p>이 문서는 SQL의 <strong>윈도우 함수(Window Functions)와 CTE(Common Table Expressions)</strong>를 활용하여 복잡한 분석 쿼리를 작성하고, 데이터 처리 효율성을 높이는 방법을 심도 있게 다룹니다. 각 개념의 정의, 실제 코드에서의 활용법, 그리고 <strong>데이터 분석 및 AI 실무에서 발생할 수 있는 주의사항과 활용 팁</strong>을 상세한 예제와 함께 설명하여, SQL을 활용한 고급 데이터 분석의 견고한 기초를 다지는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 윈도우 함수 (Window Functions): 로우 상세 정보를 유지한 그룹 분석](#1-윈도우-함수-window-functions-로우-상세-정보를-유지한-그룹-분석)
  - [1.1. 윈도우 함수의 개념과 필요성](#11-윈도우-함수의-개념과-필요성)
  - [1.2. `OVER` 절: 윈도우 정의의 핵심](#12-over-절-윈도우-정의의-핵심)
  - [1.3. 순위 함수 (Ranking Functions): 데이터의 상대적 위치 파악](#13-순위-함수-ranking-functions-데이터의-상대적-위치-파악)
  - [1.4. 집계 윈도우 함수 (Aggregate Window Functions): 상세 정보를 유지한 그룹 집계](#14-집계-윈도우-함수-aggregate-window-functions-상세-정보를-유지한-그룹-집계)
  - [1.5. 분석 윈도우 함수 (Analytic Window Functions): 로우 간 관계 분석](#15-분석-윈도우-함수-analytic-window-functions-로우-간-관계-분석)
  - [1.6. 윈도우 프레임 (Window Frame): 계산 범위의 정밀한 제어](#16-윈도우-프레임-window-frame-계산-범위의-정밀한-제어)
  - [1.7. 윈도우 함수 성능 고려사항 및 최적화 전략](#17-윈도우-함수-성능-고려사항-및-최적화-전략)
- [2. CTE (Common Table Expressions): 쿼리 가독성과 재사용성의 마법사](#2-cte-common-table-expressions-쿼리-가독성과-재사용성의-마법사)
  - [2.1. CTE의 개념과 장점](#21-cte의-개념과-장점)
  - [2.2. CTE의 기본 구조와 활용](#22-cte의-기본-구조와-활용)
    - [2.2.1. 단일 CTE 사용](#221-단일-cte-사용)
    - [2.2.2. 다중 CTE 사용](#222-다중-cte-사용)
  - [2.3. 재귀 CTE (Recursive CTE): 계층적 데이터 탐색의 필수 도구](#23-재귀-cte-recursive-cte-계층적-데이터-탐색의-필수-도구)
  - [2.4. CTE 성능 고려사항 및 최적화 전략](#24-cte-성능-고려사항-및-최적화-전략)

---

## 1. 윈도우 함수 (Window Functions): 로우 상세 정보를 유지한 그룹 분석

윈도우 함수(Window Functions)는 SQL 쿼리에서 로우들의 집합(윈도우)에 대해 계산을 수행하지만, `GROUP BY` 절처럼 로우를 그룹으로 묶어 단일 결과 로우를 반환하는 대신, 각 로우에 대해 계산된 결과를 반환합니다. 이는 각 로우의 상세 정보를 유지하면서 그룹별 통계나 순위 등을 계산할 수 있게 해주는 강력한 기능입니다.

### 1.1. 윈도우 함수의 개념과 필요성

*   **개념:** 쿼리 결과 집합 내의 특정 로우 집합(윈도우)에 대해 함수를 적용하는 것입니다. 각 로우는 윈도우 함수가 적용되는 윈도우에 속하며, 이 윈도우는 `OVER` 절에 의해 정의됩니다. 윈도우 함수는 `SELECT` 절에서만 사용 가능합니다.
*   **필요성:**
    *   **그룹별 통계 유지:** `GROUP BY`는 그룹별로 하나의 로우만 반환하지만, 윈도우 함수는 각 로우의 상세 정보를 유지하면서 그룹별 통계(예: 부서별 평균 급여)를 계산하여 해당 로우 옆에 새로운 컬럼으로 표시할 수 있습니다.
    *   **순위, 누계, 이동 평균 등 복잡한 분석:** 특정 기준에 따른 순위, 누적 합계, 이전/다음 로우와의 비교, 이동 평균 등 복잡한 분석을 효율적으로 수행할 수 있습니다.
    *   **쿼리 단순화:** 복잡한 서브쿼리나 `SELF JOIN` 없이도 다양한 분석을 간결하게 표현할 수 있습니다.

**실무적 관점:** 윈도우 함수는 데이터 분석가가 복잡한 비즈니스 질문에 답하고, 데이터의 패턴과 트렌드를 심층적으로 파악하는 데 필수적인 도구입니다. 특히 시계열 데이터 분석, 고객 행동 분석, 성과 지표 계산 등 다양한 분석 시나리오에서 강력한 유연성과 효율성을 제공합니다.

### 1.2. `OVER` 절: 윈도우 정의의 핵심

`OVER` 절은 윈도우 함수가 적용될 로우들의 집합(윈도우)을 정의합니다. `OVER` 절은 다음 세 가지 요소를 포함할 수 있으며, 이들을 조합하여 분석의 범위를 세밀하게 제어할 수 있습니다.

*   **`PARTITION BY`:** 로우들을 하나 이상의 컬럼을 기준으로 파티션(그룹)으로 나눕니다. 함수는 각 파티션 내에서 독립적으로 적용됩니다. (선택 사항)
    *   `GROUP BY`와 유사하게 데이터를 그룹화하지만, `GROUP BY`처럼 로우 수를 줄이지 않고 각 로우의 상세 정보를 유지합니다.
*   **`ORDER BY`:** 각 파티션 내에서 로우들의 순서를 정의합니다. 순위 함수나 누계 함수 등 순서가 중요한 함수에 필수적입니다. (선택 사항)
    *   이 순서는 윈도우 함수가 계산될 때만 적용되며, 최종 쿼리 결과의 정렬 순서와는 무관합니다. (최종 정렬은 별도의 `ORDER BY` 절로 제어)
*   **`ROWS` 또는 `RANGE` (윈도우 프레임):** 현재 로우를 기준으로 윈도우의 범위를 정의합니다. (선택 사항, 자세한 내용은 1.6절 참조)

```sql
-- 각 직원의 급여와 해당 부서의 평균 급여를 함께 조회
SELECT
    employee_id, first_name, department_id, salary,
    AVG(salary) OVER (PARTITION BY department_id) AS avg_dept_salary
FROM
    employees;
```

### 1.3. 순위 함수 (Ranking Functions): 데이터의 상대적 위치 파악

윈도우 내에서 로우들의 순위를 계산합니다. `ORDER BY` 절이 필수적입니다.

*   **`ROW_NUMBER()`:** 각 로우에 고유한 순위를 부여합니다. 동일한 값이라도 다른 순위를 가집니다. (예: 1, 2, 3, 4)
*   **`RANK()`:** 동일한 값에는 같은 순위를 부여하고, 다음 순위는 건너뜁니다. (예: 1, 2, 2, 4)
*   **`DENSE_RANK()`:** 동일한 값에는 같은 순위를 부여하지만, 다음 순위를 건너뛰지 않습니다. (예: 1, 2, 2, 3)
*   **`NTILE(n)`:** 로우들을 `n`개의 그룹으로 나누고 각 로우에 그룹 번호를 부여합니다. (예: 4분위, 10분위)

```sql
-- 각 부서 내에서 급여가 높은 순서대로 순위 부여
SELECT
    employee_id, first_name, department_id, salary,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rn,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rk,
    DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS drk
FROM
    employees;

-- 전체 직원을 급여 기준으로 4분위(Quartile)로 나누기
SELECT
    employee_id, first_name, salary,
    NTILE(4) OVER (ORDER BY salary DESC) AS salary_quartile
FROM
    employees;
```

**실무 팁: 순위 함수의 활용**
*   **상위/하위 N개 추출:** 각 그룹 내에서 가장 높은/낮은 N개의 로우를 추출할 때 유용합니다. (예: 부서별 급여 상위 3명, 지역별 매출 하위 5개 지점)
    ```sql
    -- 부서별 급여 상위 3명 조회
    SELECT * FROM (
        SELECT
            employee_id, first_name, department_id, salary,
            ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rn
        FROM employees
    ) AS ranked_employees
    WHERE rn <= 3;
    ```
*   **중복 처리:** 중복된 로우 중 하나만 선택해야 할 때 `ROW_NUMBER()`를 사용하여 고유한 로우를 식별하고 필터링할 수 있습니다.
*   **성과 분석:** 직원, 제품, 지역 등의 성과를 순위로 매겨 비교 분석할 때 활용됩니다.

### 1.4. 집계 윈도우 함수 (Aggregate Window Functions): 상세 정보를 유지한 그룹 집계

`SUM()`, `AVG()`, `COUNT()`, `MIN()`, `MAX()`와 같은 일반 집계 함수를 윈도우 함수로 사용할 수 있습니다. `GROUP BY`와 달리 각 로우의 상세 정보를 유지하면서 그룹별 집계 값을 반환합니다.

```sql
-- 각 직원의 급여와 해당 부서의 평균 급여, 최고 급여, 최저 급여를 함께 조회
SELECT
    employee_id, first_name, department_id, salary,
    AVG(salary) OVER (PARTITION BY department_id) AS avg_dept_salary,
    MAX(salary) OVER (PARTITION BY department_id) AS max_dept_salary,
    MIN(salary) OVER (PARTITION BY department_id) AS min_dept_salary
FROM
    employees;
```

**실무 팁: 집계 윈도우 함수의 활용**
*   **그룹 내 비율 계산:** 각 로우의 값이 속한 그룹의 전체 합계나 평균에서 차지하는 비율을 계산할 때 유용합니다.
    ```sql
    -- 직원별 급여가 부서 총 급여에서 차지하는 비율
    SELECT
        employee_id, first_name, department_id, salary,
        SUM(salary) OVER (PARTITION BY department_id) AS total_dept_salary,
        salary / SUM(salary) OVER (PARTITION BY department_id) AS salary_ratio_in_dept
    FROM employees;
    ```
*   **기준값과의 차이 분석:** 각 로우의 값이 속한 그룹의 평균이나 최대값과의 차이를 계산하여 데이터의 분포나 이상치를 파악할 때.

### 1.5. 분석 윈도우 함수 (Analytic Window Functions): 로우 간 관계 분석

특정 로우를 기준으로 이전/다음 로우의 값을 가져오거나, 누적 합계 등을 계산하는 데 사용됩니다. 주로 시계열 데이터나 순서가 중요한 데이터에서 패턴을 분석할 때 활용됩니다.

*   **`LAG(컬럼, offset, default)`:** 현재 로우를 기준으로 `offset`만큼 이전 로우의 `컬럼` 값을 반환합니다. 이전 로우가 없으면 `default` 값을 반환합니다.
*   **`LEAD(컬럼, offset, default)`:** 현재 로우를 기준으로 `offset`만큼 다음 로우의 `컬럼` 값을 반환합니다. 다음 로우가 없으면 `default` 값을 반환합니다.
*   **`FIRST_VALUE(컬럼)`:** 윈도우 내에서 첫 번째 로우의 `컬럼` 값을 반환합니다.
*   **`LAST_VALUE(컬럼)`:** 윈도우 내에서 마지막 로우의 `컬럼` 값을 반환합니다.
*   **`NTH_VALUE(컬럼, n)`:** 윈도우 내에서 `n`번째 로우의 `컬럼` 값을 반환합니다.

```sql
-- 직원의 급여와 이전 직원의 급여 비교 (고용일 기준)
SELECT
    employee_id, first_name, hire_date, salary,
    LAG(salary, 1, 0) OVER (ORDER BY hire_date) AS previous_salary,
    salary - LAG(salary, 1, 0) OVER (ORDER BY hire_date) AS salary_diff
FROM
    employees;

-- 부서 내에서 급여가 가장 높은 직원의 이름 조회 (FIRST_VALUE)
SELECT
    employee_id, first_name, department_id, salary,
    FIRST_VALUE(first_name) OVER (PARTITION BY department_id ORDER BY salary DESC) AS highest_paid_employee_in_dept
FROM
    employees;

-- 부서 내에서 급여가 두 번째로 높은 직원의 이름 조회 (NTH_VALUE)
SELECT
    employee_id, first_name, department_id, salary,
    NTH_VALUE(first_name, 2) OVER (PARTITION BY department_id ORDER BY salary DESC) AS second_highest_paid_employee_in_dept
FROM
    employees;
```

**실무 팁: 분석 윈도우 함수의 활용**
*   **시계열 분석:** 주가, 매출, 사용자 수 등 시계열 데이터에서 이전 기간 대비 변화율, 이동 평균 등을 계산할 때 유용합니다.
*   **누적 합계/평균:** 특정 기준에 따른 누적 합계나 누적 평균을 계산하여 트렌드를 파악할 수 있습니다. (윈도우 프레임과 함께 사용)
*   **데이터 품질 검사:** 이전/다음 로우와의 비교를 통해 데이터의 이상치나 불일치를 탐지할 수 있습니다.
*   **세션 분석:** 사용자 행동 로그에서 이전/다음 행동과의 시간 간격, 행동 패턴 등을 분석할 때.

### 1.6. 윈도우 프레임 (Window Frame): 계산 범위의 정밀한 제어

`ROWS` 또는 `RANGE` 절을 사용하여 윈도우 내에서 현재 로우를 기준으로 계산에 포함될 로우들의 범위를 정의할 수 있습니다. 이는 이동 평균, 누적 합계 등 특정 범위 내의 데이터에 대한 계산을 수행할 때 사용됩니다. `ORDER BY` 절과 함께 사용될 때 의미가 있습니다.

*   **`ROWS BETWEEN ... AND ...`:** 로우의 **물리적 위치**를 기준으로 범위를 정의합니다. (예: 현재 로우를 포함하여 이전 2개, 다음 1개 로우)
    *   `UNBOUNDED PRECEDING`: 파티션의 첫 번째 로우부터
    *   `N PRECEDING`: 현재 로우로부터 `N`개 이전 로우
    *   `CURRENT ROW`: 현재 로우
    *   `N FOLLOWING`: 현재 로우로부터 `N`개 다음 로우
    *   `UNBOUNDED FOLLOWING`: 파티션의 마지막 로우까지

*   **`RANGE BETWEEN ... AND ...`:** 값의 **논리적 범위**를 기준으로 정의합니다. `ORDER BY` 절이 숫자 또는 날짜/시간 타입 컬럼이어야 합니다. (예: 현재 로우의 급여를 기준으로 ±1000 범위 내의 모든 로우)

**기본 윈도우 프레임:** `ORDER BY` 절만 명시하고 윈도우 프레임을 생략하면, 기본적으로 `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`가 적용됩니다. 이는 누적 합계/평균 계산에 적합합니다.

```sql
-- 급여를 기준으로 3명의 이동 평균 급여 계산 (현재 로우 포함, 이전 1명, 다음 1명)
SELECT
    employee_id, first_name, salary,
    AVG(salary) OVER (ORDER BY salary ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS moving_avg_salary
FROM
    employees;

-- 고용일 기준으로 누적 급여 합계 계산 (기본 윈도우 프레임 적용)
SELECT
    employee_id, first_name, hire_date, salary,
    SUM(salary) OVER (ORDER BY hire_date) AS cumulative_salary -- ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW 가 기본 적용
FROM
    employees;

-- 고용일 기준으로 7일 이동 평균 급여 계산
SELECT
    employee_id, first_name, hire_date, salary,
    AVG(salary) OVER (ORDER BY hire_date RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND CURRENT ROW) AS seven_day_moving_avg
FROM
    employees;
```

**실무 팁: 윈도우 프레임의 활용**
*   **이동 평균/합계:** 시계열 데이터에서 노이즈를 제거하고 트렌드를 부각시킬 때 사용됩니다. (예: 일별 매출의 7일 이동 평균)
*   **누적 지표:** 특정 시점까지의 누적 매출, 누적 사용자 수 등을 계산하여 성과를 추적할 때 유용합니다.
*   **기간별 비교:** 특정 기간 내의 데이터만을 대상으로 분석을 수행할 때 활용됩니다.

### 1.7. 윈도우 함수 성능 고려사항 및 최적화 전략

윈도우 함수는 강력하지만, 대규모 데이터셋에 적용할 경우 상당한 시스템 자원을 소모하고 쿼리 실행 시간이 길어질 수 있습니다. 다음 사항들을 고려하여 최적화해야 합니다.

*   **`PARTITION BY` 컬럼 인덱싱:** `PARTITION BY` 절에 사용되는 컬럼에 인덱스가 있으면 데이터베이스가 파티션을 효율적으로 나눌 수 있어 성능이 향상됩니다. 이는 데이터를 미리 그룹화하는 효과를 줍니다.
*   **`ORDER BY` 컬럼 인덱싱:** `ORDER BY` 절에 사용되는 컬럼에 인덱스가 있으면 각 파티션 내에서 로우를 정렬하는 비용을 줄일 수 있습니다. 특히 `ORDER BY`와 `PARTITION BY` 컬럼이 복합 인덱스로 구성되어 있다면 최적의 성능을 기대할 수 있습니다.
*   **윈도우 프레임 최적화:** `ROWS` 또는 `RANGE` 절의 범위를 너무 넓게 지정하면 계산량이 증가하여 성능이 저하될 수 있습니다. 필요한 최소한의 범위만 지정하는 것이 좋습니다.
*   **쿼리 실행 계획 분석 (`EXPLAIN`):** `EXPLAIN` 명령을 사용하여 쿼리 실행 계획을 분석하고, 윈도우 함수가 성능 병목 지점인지 확인합니다. `Using temporary`, `Using filesort` 등이 나타난다면 최적화가 필요하다는 신호입니다.
*   **`GROUP BY` 또는 `JOIN`으로 대체 가능성 검토:** 간단한 집계나 비교는 `GROUP BY`나 `JOIN`이 더 효율적일 수 있습니다. 윈도우 함수는 각 로우의 상세 정보를 유지해야 할 때 주로 사용합니다.
*   **CTE 활용:** 복잡한 윈도우 함수 쿼리를 CTE로 분리하여 가독성을 높이고, 옵티마이저가 더 효율적인 실행 계획을 세울 수 있도록 도울 수 있습니다.

**실무 팁:** 윈도우 함수는 복잡한 분석을 간결하게 표현할 수 있지만, 성능 최적화를 위해서는 쿼리 실행 계획을 이해하고 적절한 인덱스를 활용하는 것이 필수적입니다. 대규모 데이터셋에서는 윈도우 함수를 사용하기 전에 데이터의 양과 쿼리의 복잡도를 고려하여 성능 테스트를 수행하는 것이 좋습니다.

## 2. CTE (Common Table Expressions): 쿼리 가독성과 재사용성의 마법사

CTE(Common Table Expressions), 또는 `WITH` 절은 SQL 쿼리 내에서 임시적인 명명된 결과 집합을 정의하는 방법입니다. CTE는 복잡한 쿼리를 여러 개의 논리적인 단계로 분리하여 가독성을 높이고, 재사용성을 향상시키며, 재귀 쿼리(Recursive Query)를 구현하는 데 사용됩니다.

### 2.1. CTE의 개념과 장점

*   **개념:** `WITH` 키워드로 시작하며, 메인 쿼리에서 참조할 수 있는 임시적인 뷰와 유사한 역할을 합니다. CTE는 쿼리 실행 중에만 존재하며, 데이터베이스에 영구적으로 저장되지 않습니다. 쿼리 실행이 끝나면 자동으로 사라집니다.
*   **장점:**
    *   **가독성 향상:** 복잡한 쿼리를 작은 논리적 블록으로 나누어 이해하기 쉽게 만듭니다. 각 단계의 중간 결과를 명확하게 정의할 수 있습니다.
    *   **재사용성:** 동일한 CTE를 메인 쿼리나 다른 CTE에서 여러 번 참조할 수 있어 코드 중복을 줄입니다. 이는 유지보수성을 높이는 데 기여합니다.
    *   **재귀 쿼리 구현:** 계층적 데이터(예: 조직도, 댓글 스레드, BOM(Bill of Materials))를 처리하는 재귀 쿼리를 구현할 수 있는 유일한 방법입니다.
    *   **성능 최적화 (잠재적):** 특정 상황에서는 옵티마이저가 CTE를 더 효율적으로 처리할 수 있도록 돕습니다. 특히 복잡한 서브쿼리를 CTE로 분리하면 옵티마이저가 더 나은 실행 계획을 세울 수 있습니다.

**실무적 관점:** CTE는 복잡한 데이터 분석 쿼리를 작성하고 관리하는 데 있어 필수적인 도구입니다. 특히 다단계의 데이터 변환, 필터링, 집계가 필요한 분석 파이프라인을 SQL 내에서 구현할 때 강력한 유연성과 가독성을 제공합니다.

### 2.2. CTE의 기본 구조와 활용

CTE는 `WITH` 키워드로 시작하며, CTE 이름과 그 정의(SELECT 문)를 포함합니다. 정의된 CTE는 메인 쿼리에서 테이블처럼 사용됩니다.

#### 2.2.1. 단일 CTE 사용

가장 기본적인 형태로, 하나의 CTE를 정의하고 메인 쿼리에서 참조합니다.

```sql
WITH CTE_Name AS (
    -- CTE 정의 SELECT 문
    SELECT column1, column2
    FROM table_name
    WHERE condition
)
-- 메인 쿼리: 정의된 CTE를 일반 테이블처럼 사용
SELECT * 
FROM CTE_Name
WHERE another_condition;
```

**예시:** 평균 급여보다 많이 받는 직원 조회 (CTE 활용)

```sql
WITH AvgSalary AS (
    SELECT AVG(salary) AS avg_s FROM employees
)
SELECT e.first_name, e.salary
FROM employees e, AvgSalary a
WHERE e.salary > a.avg_s;
```

#### 2.2.2. 다중 CTE 사용

하나의 `WITH` 절 내에서 여러 개의 CTE를 콤마(`,`)로 구분하여 정의할 수 있습니다. 정의된 CTE들은 서로를 참조할 수도 있습니다. 이는 더욱 복잡한 다단계 분석을 구현할 때 유용합니다.

```sql
-- 예시: 부서별 평균 급여를 계산하고, 그 결과를 바탕으로 각 직원의 급여가 부서 평균보다 높은지 여부 조회
WITH DepartmentAvg AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
),
EmployeeSalaryStatus AS ( -- DepartmentAvg CTE를 참조
    SELECT
        e.employee_id, e.first_name, e.salary, e.department_id,
        da.avg_salary,
        CASE WHEN e.salary > da.avg_salary THEN 'Above Average' ELSE 'Below Average' END AS salary_status
    FROM
        employees e
    JOIN
        DepartmentAvg da ON e.department_id = da.department_id
)
SELECT * -- 최종 결과는 EmployeeSalaryStatus CTE에서 조회
FROM EmployeeSalaryStatus
WHERE salary_status = 'Above Average';
```

**실무 팁: 다중 CTE 활용 전략**
*   **단계별 분석:** 복잡한 분석을 여러 단계로 나누어 각 단계를 CTE로 정의하면 쿼리 흐름을 쉽게 파악할 수 있습니다.
*   **중간 결과 재사용:** 동일한 중간 결과가 여러 곳에서 필요할 때 CTE로 정의하여 코드 중복을 피하고 유지보수를 용이하게 합니다.

### 2.3. 재귀 CTE (Recursive CTE): 계층적 데이터 탐색의 필수 도구

재귀 CTE는 자기 자신을 참조하여 반복적으로 실행되는 CTE입니다. 주로 계층적 데이터(예: 조직도, 댓글 스레드, BOM(Bill of Materials), 그래프 경로)를 탐색하거나, 특정 깊이까지의 경로를 찾을 때 사용됩니다.

재귀 CTE는 `WITH RECURSIVE` 키워드로 시작하며, 두 부분으로 구성됩니다.
1.  **앵커 멤버 (Anchor Member):** 재귀의 시작점을 정의하는 `SELECT` 문입니다. 재귀 호출 없이 한 번만 실행됩니다.
2.  **재귀 멤버 (Recursive Member):** 앵커 멤버의 결과 또는 이전 재귀 멤버의 결과를 참조하여 반복적으로 실행되는 `SELECT` 문입니다. `UNION ALL`로 앵커 멤버와 연결됩니다. 재귀 멤버는 재귀가 종료될 때까지 반복적으로 실행됩니다.

```sql
-- 예시: 조직도에서 특정 직원의 모든 하위 직원 조회
-- (employees 테이블에 employee_id와 manager_id 컬럼이 있다고 가정)
WITH RECURSIVE EmployeeHierarchy AS (
    -- 앵커 멤버: 최상위 관리자 (manager_id가 NULL인 직원) 또는 특정 시작 직원
    SELECT employee_id, first_name, manager_id, 0 AS level -- level은 계층 깊이를 추적
    FROM employees
    WHERE employee_id = 101 -- 예시: employee_id가 101인 직원을 시작점으로

    UNION ALL -- 앵커 멤버와 재귀 멤버를 연결

    -- 재귀 멤버: 이전 결과(eh)의 employee_id를 manager_id로 가지는 직원
    SELECT e.employee_id, e.first_name, e.manager_id, eh.level + 1
    FROM employees e
    JOIN EmployeeHierarchy eh ON e.manager_id = eh.employee_id
    WHERE eh.level < 100 -- 무한 루프 방지를 위한 최대 재귀 깊이 제한 (필수)
)
SELECT * FROM EmployeeHierarchy;
```

**실무 팁: 재귀 CTE의 활용과 주의점**
*   **활용:** 조직도, 댓글 스레드, 파일 시스템 경로, BOM 등 계층적 데이터 분석에 필수적입니다. 특정 노드로부터의 모든 하위 노드를 찾거나, 특정 노드까지의 경로를 추적할 때 유용합니다.
*   **주의점:**
    *   **무한 루프 방지:** 재귀 CTE는 잘못 작성하면 무한 루프에 빠질 수 있습니다. 반드시 `WHERE` 절에 `level` 제한과 같은 **재귀 종료 조건**을 명확히 정의해야 합니다. (대부분의 DBMS는 재귀 깊이 제한을 두어 무한 루프를 방지하지만, 명시적으로 작성하는 것이 좋습니다.)
    *   **성능:** 재귀 CTE는 반복적인 연산을 수행하므로 대규모 계층 구조에서는 성능에 영향을 미칠 수 있습니다. 필요한 경우 데이터 모델링 변경이나 애플리케이션 레벨에서의 처리를 고려해야 합니다.
    *   **`UNION ALL` 사용:** 재귀 CTE에서는 `UNION ALL`을 사용해야 합니다. `UNION`을 사용하면 매 반복마다 중복 제거를 시도하여 성능이 저하될 수 있습니다.

### 2.4. CTE 성능 고려사항 및 최적화 전략

CTE는 쿼리 가독성과 재사용성을 높이는 데 매우 효과적이지만, 성능 측면에서는 항상 최적의 선택은 아닐 수 있습니다. CTE는 논리적인 개념이며, 데이터베이스 옵티마이저가 이를 어떻게 처리하느냐에 따라 실제 성능이 달라집니다.

*   **인덱스 활용:** CTE 내부의 `SELECT` 문에서 사용되는 컬럼에 적절한 인덱스가 있으면 성능이 향상됩니다. CTE는 기본적으로 뷰처럼 동작하므로, CTE 내부 쿼리의 성능이 전체 쿼리 성능에 직접적인 영향을 미칩니다.
*   **옵티마이저의 처리 방식:** 일부 DBMS는 CTE를 임시 테이블처럼 물리적으로 생성하여 사용하기도 하지만, 대부분의 경우 CTE는 뷰처럼 인라인(inline)되어 메인 쿼리와 함께 최적화됩니다. 따라서 CTE를 사용한다고 해서 항상 성능이 향상되거나 저하되는 것은 아닙니다. 옵티마이저가 CTE를 어떻게 처리하는지 이해하는 것이 중요합니다.
*   **재사용성 vs 성능:** CTE를 여러 번 참조할 경우, 데이터베이스가 CTE를 여러 번 실행할 수도 있습니다. 이 경우 CTE의 결과를 임시 테이블에 저장하거나, Materialized View를 사용하는 것이 더 효율적일 수 있습니다.
*   **복잡한 CTE:** 너무 많은 CTE를 중첩하거나, 각 CTE의 로직이 복잡해지면 쿼리 실행 계획이 복잡해지고 최적화가 어려워질 수 있습니다. CTE는 논리적 단위를 나누는 데 유용하지만, 과도한 사용은 오히려 독이 될 수 있습니다.

**실무 팁: CTE 최적화 전략**
*   **`EXPLAIN` 명령 활용:** CTE를 사용한 후에는 항상 `EXPLAIN` 명령을 사용하여 쿼리 실행 계획을 분석하는 습관을 들이는 것이 중요합니다. 이를 통해 CTE가 어떻게 처리되는지, 성능 병목 지점은 어디인지 파악하고 적절한 최적화 전략을 적용할 수 있습니다.
*   **필요한 데이터만 CTE로:** CTE 내에서 불필요한 데이터를 미리 필터링하여 다음 단계로 전달되는 데이터의 양을 최소화합니다.
*   **인덱스 최적화:** CTE 내부 쿼리의 `WHERE` 절, `JOIN` 조건, `ORDER BY`, `GROUP BY` 절에 사용되는 컬럼에 적절한 인덱스가 있는지 확인하고 추가합니다.
*   **`MATERIALIZED` 힌트 (DBMS 지원 시):** 일부 DBMS에서는 `MATERIALIZED` 힌트를 사용하여 CTE를 물리적인 임시 테이블로 생성하도록 강제할 수 있습니다. 이는 CTE가 여러 번 참조되거나, 복잡한 계산을 포함할 때 성능 향상에 도움이 될 수 있습니다.
*   **재귀 CTE의 효율성:** 재귀 CTE는 계층형 데이터에 강력하지만, 비재귀적인 방법(예: `JOIN`과 윈도우 함수 조합)으로도 해결 가능한 문제라면 성능을 비교하여 더 효율적인 방법을 선택합니다.

CTE(Common Table Expressions), 또는 `WITH` 절은 SQL 쿼리 내에서 임시적인 명명된 결과 집합을 정의하는 방법입니다. CTE는 복잡한 쿼리를 여러 개의 논리적인 단계로 분리하여 가독성을 높이고, 재사용성을 향상시키며, 재귀 쿼리(Recursive Query)를 구현하는 데 사용됩니다.