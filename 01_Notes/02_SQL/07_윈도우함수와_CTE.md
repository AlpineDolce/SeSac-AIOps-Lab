<h2>윈도우 함수와 CTE</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-27

<h2>문서 목표</h2>
<p>이 문서는 복잡한 분석과 보고서 작성을 위한 고급 SQL 쿼리 기법인 윈도우 함수와 CTE(Common Table Expression)를 학습합니다. 윈도우 함수를 통해 그룹 내 순위, 비율, 누적 합계 등을 계산하고, CTE를 통해 복잡한 쿼리의 가독성과 재사용성을 높이는 방법을 익힙니다.</p>

> **현업 데이터 분석가의 관점 한 줄 요약:**
>
> 윈도우 함수와 CTE는 단순한 데이터 조회 쿼리를 넘어, 복잡한 비즈니스 로직을 SQL로 구현하고 심층적인 분석을 수행하는 데 필수적인 고급 기술입니다. 이들을 능숙하게 활용하면 데이터의 숨겨진 패턴과 트렌드를 발견하고, 복잡한 보고서를 효율적으로 생성하며, 쿼리 가독성과 유지보수성을 혁신적으로 개선할 수 있습니다. 현업에서 데이터 기반 의사결정을 주도하는 분석가로 성장하기 위한 핵심 역량입니다.

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

> **윈도우 함수: `GROUP BY`를 넘어서는 분석의 힘:**
> 윈도우 함수는 SQL의 가장 강력한 고급 기능 중 하나로, 데이터 분석가가 복잡한 비즈니스 질문에 답하고 심층적인 분석을 수행하는 데 필수적입니다.
>
> *   **`GROUP BY`와의 결정적 차이:**
>     *   **`GROUP BY`:** 로우들을 그룹으로 묶고, 각 그룹에 대해 하나의 요약된 로우를 반환합니다. 즉, **결과 로우의 수가 줄어듭니다.** (예: 부서별 평균 급여)
>     *   **윈도우 함수:** 로우들을 그룹(파티션)으로 나누어 계산을 수행하지만, **원본 로우의 수는 그대로 유지**하면서 각 로우에 계산된 값을 추가합니다. (예: 각 직원의 급여 옆에 해당 부서의 평균 급여를 표시)
> *   **왜 중요한가?**
>     *   **데이터의 맥락 유지:** `GROUP BY`는 데이터를 요약하면서 개별 로우의 상세 정보를 잃어버리지만, 윈도우 함수는 개별 로우의 맥락을 유지한 채 그룹 내에서의 상대적인 위치나 특성을 분석할 수 있게 합니다.
>     *   **복잡한 분석 간소화:** 순위, 누적 합계, 이동 평균, 전/후 로우 값 비교 등 `GROUP BY`나 `SELF JOIN`으로는 구현하기 어렵거나 매우 복잡했던 분석을 단일 쿼리로 간결하게 해결할 수 있습니다.
>     *   **성능 향상:** 많은 경우 상관 서브쿼리나 복잡한 `JOIN`보다 훨씬 효율적으로 동작하여 쿼리 성능을 향상시킵니다.
>
> 윈도우 함수를 능숙하게 다루는 것은 단순한 데이터 조회자를 넘어, 데이터의 숨겨진 패턴과 트렌드를 발견하고 비즈니스 인사이트를 도출하는 '시니어' 데이터 분석가로 성장하는 핵심 역량입니다.
*   **필요성:**
    *   **그룹 내 순위/비율 계산:** 특정 그룹 내에서 각 로우의 순위나 전체 대비 비율을 쉽게 계산할 수 있습니다.
    *   **이동 평균/누적 합계:** 시계열 데이터에서 이동 평균이나 누적 합계를 효율적으로 계산할 수 있습니다.
    *   **복잡한 분석 쿼리 간소화:** 서브쿼리나 `SELF JOIN`으로 복잡하게 작성해야 했던 쿼리를 윈도우 함수를 사용하여 훨씬 간결하게 작성할 수 있습니다.

*   **윈도우 함수와 성능 (실무적 관점):**
    윈도우 함수는 매우 강력하고 편리하지만, 내부적으로는 정렬(Sorting)과 많은 메모리를 사용할 수 있어 대용량 데이터 처리 시 성능 병목의 원인이 될 수 있습니다. 특히 `PARTITION BY` 절에 카디널리티가 매우 높은 컬럼을 사용하거나, `ORDER BY` 절에 인덱스가 없는 컬럼을 사용하면 성능이 크게 저하될 수 있습니다.

    *   **성능 저하 원인:**
        *   **데이터 정렬:** `ORDER BY` 절이 사용되면, 데이터베이스는 윈도우 내의 모든 로우를 정렬해야 합니다. 대규모 데이터셋에서는 이 과정이 상당한 시간과 CPU, 메모리 자원을 소모합니다.
        *   **메모리 사용:** 각 파티션에 대한 계산을 위해 해당 파티션의 데이터를 메모리에 로드해야 할 수 있습니다. 파티션 크기가 크면 메모리 부족으로 디스크 I/O가 발생하여 성능이 더욱 저하됩니다.
        *   **`PARTITION BY` 컬럼의 카디널리티:** `PARTITION BY` 컬럼의 고유한 값의 수가 많을수록 생성되는 파티션의 수가 많아지고, 각 파티션에 대한 오버헤드가 증가할 수 있습니다.

    *   **최적화 방안:**
        *   **인덱스 활용:** `PARTITION BY`와 `ORDER BY` 절에 사용되는 컬럼에 복합 인덱스를 생성하는 것이 가장 효과적인 최적화 방법입니다. 인덱스는 데이터 정렬 비용을 크게 줄여줍니다.
        *   **처리 범위 최소화:** `WHERE` 절을 사용하여 윈도우 함수가 적용될 전체 로우의 수를 최대한 줄인 후, 윈도우 함수를 적용합니다. 불필요한 데이터를 미리 필터링하여 윈도우 함수의 작업량을 줄입니다.
        *   **CTE 활용:** 복잡한 윈도우 함수 계산은 CTE로 분리하여 중간 결과를 먼저 생성하고, 이후에 메인 쿼리에서 해당 결과를 사용하는 것이 쿼리 가독성과 유지보수, 그리고 때로는 성능에도 도움이 될 수 있습니다. 특히, 여러 윈도우 함수를 사용할 때 동일한 `PARTITION BY` 및 `ORDER BY` 절을 공유한다면 CTE로 묶어 한 번만 계산하도록 유도할 수 있습니다.
        *   **`EXPLAIN` 명령 활용:** 윈도우 함수를 포함한 쿼리의 성능 문제를 진단하는 가장 중요한 도구는 `EXPLAIN` 명령입니다. 쿼리 실행 계획을 분석하여 어떤 단계에서 병목이 발생하는지 파악하고, 그에 맞는 최적화 전략을 수립해야 합니다.

    **데이터 분석가의 역할:** 윈도우 함수는 강력하지만, 성능에 대한 이해 없이는 오히려 쿼리 성능을 저하시킬 수 있습니다. 따라서 윈도우 함수 사용 시에는 항상 데이터 규모와 쿼리 복잡도를 고려하고, `EXPLAIN`을 통해 실행 계획을 분석하며, 필요에 따라 인덱스 튜닝이나 쿼리 재작성을 통해 최적의 성능을 확보해야 합니다.

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

> **`OVER` 절의 핵심: 윈도우 정의의 중요성:**
> `OVER` 절은 윈도우 함수의 동작 범위를 결정하는 가장 중요한 부분입니다. `PARTITION BY`와 `ORDER BY`는 윈도우를 정의하는 두 가지 핵심 요소이며, 이들을 어떻게 조합하느냐에 따라 윈도우 함수의 결과가 크게 달라집니다.
>
> *   **`PARTITION BY` (그룹화):**
>     *   데이터를 논리적인 그룹(파티션)으로 나눕니다. 각 파티션은 독립적인 윈도우로 간주되며, 윈도우 함수는 각 파티션 내에서만 계산됩니다.
>     *   `GROUP BY`와 달리 `PARTITION BY`는 원본 로우의 수를 줄이지 않습니다. 각 로우는 여전히 결과 집합에 존재하며, 해당 로우가 속한 파티션에 대한 계산 결과가 추가됩니다.
>     *   예시: `PARTITION BY department_id`는 직원들을 부서별로 나누고, 각 부서 내에서 윈도우 함수를 적용합니다.
> *   **`ORDER BY` (정렬):**
>     *   각 파티션 내에서 로우들의 순서를 정의합니다. 순위 함수나 누적 합계, 이동 평균 등 순서에 민감한 윈도우 함수에서 필수적입니다.
>     *   `ORDER BY`가 없으면 윈도우 내의 로우 순서가 보장되지 않아, `RANK()`, `LAG()`, `LEAD()`와 같은 함수는 예측 불가능한 결과를 반환할 수 있습니다.
>     *   예시: `ORDER BY salary DESC`는 각 파티션 내에서 급여가 높은 순서대로 로우를 정렬합니다.
>
> *   **`OVER()` (빈 괄호):**
>     *   `PARTITION BY`나 `ORDER BY` 없이 `OVER()`만 사용하면, 전체 결과 집합을 하나의 윈도우로 간주합니다. 이 경우 윈도우 함수는 전체 데이터에 대해 계산됩니다. (예: `SUM(salary) OVER ()`는 전체 직원의 총 급여를 각 로우에 표시)
>
> `OVER` 절을 정확하게 이해하고 활용하는 것은 윈도우 함수를 통해 원하는 분석 결과를 얻는 데 필수적입니다. 쿼리 작성 시 어떤 기준으로 데이터를 나누고, 어떤 순서로 정렬해야 하는지 명확히 정의해야 합니다.

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

> **순위 함수의 선택과 활용:**
> 순위 함수는 데이터 분석에서 특정 그룹 내의 상위/하위 항목을 식별하거나, 데이터를 등급화할 때 매우 중요하게 사용됩니다. 각 함수의 미묘한 차이를 이해하고 적절한 함수를 선택하는 것이 중요합니다.
>
> *   **`RANK()` vs `DENSE_RANK()` vs `ROW_NUMBER()`:**
>     *   **`ROW_NUMBER()`:** 중복된 값이 있더라도 고유한 순위를 부여해야 할 때 사용합니다. (예: 경품 추첨 시 중복 당첨 방지, 각 로우에 고유한 번호 부여)
>     *   **`RANK()`:** 동일한 값에 같은 순위를 부여하고, 다음 순위를 건너뛰어 실제 등수와 유사하게 표현할 때 사용합니다. (예: 공동 2등이 두 명이면 다음은 4등)
>     *   **`DENSE_RANK()`:** 동일한 값에 같은 순위를 부여하고, 다음 순위를 건너뛰지 않아 연속적인 순위를 부여할 때 사용합니다. (예: 공동 2등이 두 명이어도 다음은 3등)
> *   **`NTILE(n)`:**
>     *   데이터를 `n`개의 동일한 크기(또는 거의 동일한 크기)의 그룹으로 나눌 때 사용합니다. 이는 고객을 구매액 기준으로 4분위(Quartile)로 나누거나, 성과에 따라 10분위(Decile)로 나누는 등 데이터를 등급화하여 분석할 때 유용합니다.
>     *   그룹의 크기가 정확히 `n`으로 나누어 떨어지지 않을 경우, `NTILE`은 가능한 한 균등하게 로우를 분배합니다.
>
> *   **실무 활용 사례:**
>     *   **상위 N개 항목 추출:** 각 부서별 급여 상위 5명, 각 제품 카테고리별 판매량 상위 10개 제품 등을 찾을 때 순위 함수를 사용한 후 `WHERE rnk <= N`과 같이 필터링합니다.
>     *   **고객 세그먼테이션:** 고객의 구매액, 방문 빈도 등을 기준으로 `NTILE`을 사용하여 고객을 여러 등급으로 나누고, 각 등급별 특성을 분석합니다.
>     *   **경쟁사 분석:** 시장 점유율, 성장률 등 지표를 기준으로 경쟁사들의 순위를 매기고, 자사의 위치를 파악합니다.
>
> *   **주의사항:**
>     *   **`ORDER BY` 절 필수:** 순위 함수는 `ORDER BY` 절이 없으면 의미가 없습니다. 순위의 기준이 되는 컬럼을 명확히 지정해야 합니다.
>     *   **동일 값 처리:** 동일한 값에 대해 어떤 순위를 부여할지(`RANK`, `DENSE_RANK`, `ROW_NUMBER`) 비즈니스 요구사항에 맞춰 신중하게 선택해야 합니다.
>     *   **성능 고려:** 대규모 데이터셋에서 순위 함수는 정렬 작업을 수반하므로 성능에 영향을 줄 수 있습니다. `PARTITION BY`와 `ORDER BY` 절에 적절한 인덱스를 활용하는 것이 중요합니다.

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

> **집계 윈도우 함수: `GROUP BY`의 한계를 넘어서다:**
> 집계 윈도우 함수는 `SUM()`, `AVG()`, `COUNT()`, `MIN()`, `MAX()`와 같은 일반 집계 함수에 `OVER` 절을 추가하여 사용합니다. 이는 `GROUP BY` 절로는 불가능했던, **그룹별 집계 값을 각 개별 로우에 함께 표시**하는 강력한 기능을 제공합니다.
>
> *   **`GROUP BY`와의 차이점 재강조:**
>     *   `GROUP BY`는 그룹별로 하나의 요약된 로우만 반환하므로, 개별 로우의 상세 정보를 함께 볼 수 없습니다.
>     *   집계 윈도우 함수는 원본 로우의 수를 유지하면서, 각 로우에 해당 로우가 속한 그룹(파티션)의 집계 값을 추가합니다. 이를 통해 개별 데이터의 맥락을 유지한 채 그룹 수준의 정보를 활용할 수 있습니다.
>
> *   **주요 활용 사례:**
>     *   **그룹 내 비율 계산:** 위 예시처럼 각 직원의 급여가 소속 부서 총 급여에서 차지하는 비율을 계산하는 것은 매우 흔한 분석 시나리오입니다. `SUM(salary) OVER (PARTITION BY department_id)`를 통해 부서별 총 급여를 각 직원 로우에 가져와 쉽게 비율을 계산할 수 있습니다.
>     *   **그룹별 평균/최대/최소 값 비교:** 각 직원의 급여가 소속 부서의 평균 급여보다 높은지 낮은지, 또는 부서 내 최고/최저 급여와 비교할 때 유용합니다.
>     *   **누적 합계/이동 평균:** `ORDER BY` 절과 함께 사용하여 시계열 데이터에서 누적 합계(Running Total)나 이동 평균(Moving Average)을 계산할 수 있습니다. (자세한 내용은 [1.6. 프레임(Frame) 정의](#16-프레임frame-정의-rows-between-range-between) 참조)
>     *   **데이터 정규화 및 표준화:** 특정 그룹 내의 값을 기준으로 데이터를 정규화하거나 표준화할 때 기준 값(평균, 표준편차 등)을 각 로우에 함께 가져와 계산할 수 있습니다.
>
> *   **성능 고려사항:**
>     *   `PARTITION BY`와 `ORDER BY` 절에 사용되는 컬럼에 적절한 인덱스가 있다면 성능을 크게 향상시킬 수 있습니다.
>     *   대규모 데이터셋에서는 `EXPLAIN`을 통해 쿼리 실행 계획을 분석하여 성능 병목 지점을 확인하는 것이 중요합니다.
>
> 집계 윈도우 함수는 데이터 분석가가 그룹 수준의 인사이트를 개별 데이터에 연결하여 더 깊이 있는 분석을 수행하는 데 필수적인 도구입니다.

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
    **`FIRST_VALUE`와 `LAST_VALUE` 사용 시 프레임(Frame) 정의의 중요성:**
`FIRST_VALUE`와 `LAST_VALUE` 함수는 윈도우 내에서 특정 로우의 값을 가져오는 강력한 도구입니다. 하지만 이 함수들의 동작 방식, 특히 기본 프레임 정의를 정확히 이해하지 못하면 예상과 다른 결과를 얻을 수 있습니다.

*   **기본 프레임 정의:**
    `ORDER BY` 절만 사용하고 프레임 정의(`ROWS BETWEEN` 또는 `RANGE BETWEEN`)를 명시하지 않으면, 대부분의 DBMS는 기본적으로 `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`를 적용합니다. 이는 현재 로우부터 파티션의 시작까지의 범위에 대해 함수를 적용한다는 의미입니다.

*   **`FIRST_VALUE`의 동작:**
    `FIRST_VALUE`는 기본 프레임 정의에서도 파티션의 첫 번째 값을 정확히 가져옵니다. 이는 현재 로우가 어디에 있든, 프레임의 시작점은 항상 파티션의 시작이기 때문입니다.

*   **`LAST_VALUE`의 함정:**
    `LAST_VALUE`는 기본 프레임 정의(`... AND CURRENT ROW`) 때문에 현재 로우까지의 범위에서 마지막 값을 찾습니다. 따라서 `ORDER BY` 절이 있다면, `LAST_VALUE`는 대부분의 경우 현재 로우의 값을 반환하게 되어 예상과 다른 결과를 줄 수 있습니다. 예를 들어, 급여를 오름차순으로 정렬했을 때 `LAST_VALUE(salary)`는 현재 직원의 급여를 반환할 것입니다.

*   **정확한 `LAST_VALUE`를 위한 프레임 정의:**
    파티션 내의 **실제 마지막 값**을 가져오려면, 위 예시처럼 `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`과 같이 명시적으로 **전체 파티션**을 프레임으로 지정해야 합니다. 이렇게 하면 `LAST_VALUE` 함수가 파티션 전체를 대상으로 마지막 값을 찾게 됩니다.
```

```sql
-- 부서별 가장 높은 급여와 가장 낮은 급여를 각 직원 로우에 표시
SELECT
    employee_id, first_name, department_id, salary,
    FIRST_VALUE(salary) OVER (PARTITION BY department_id ORDER BY salary DESC) AS highest_dept_salary,
    LAST_VALUE(salary) OVER (PARTITION BY department_id ORDER BY salary DESC
                             ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS lowest_dept_salary
FROM employees;
```

**실무적 조언:**
`FIRST_VALUE`와 `LAST_VALUE`를 사용할 때는 항상 `OVER` 절의 `ORDER BY`와 프레임 정의를 신중하게 고려해야 합니다. 특히 `LAST_VALUE`의 경우, 명시적인 프레임 정의 없이 사용하면 의도치 않은 결과를 얻을 가능성이 높으므로 주의해야 합니다. `EXPLAIN` 명령을 통해 쿼리 실행 계획을 확인하고, 예상대로 동작하는지 검증하는 습관을 들이는 것이 중요합니다.

> **분석 함수 (`LEAD`, `LAG`, `FIRST_VALUE`, `LAST_VALUE`)의 활용:**
> 분석 함수는 시계열 데이터 분석, 추세 분석, 그리고 특정 그룹 내에서 현재 로우와 다른 로우 간의 관계를 파악하는 데 매우 강력한 도구입니다.
>
> *   **`LEAD`와 `LAG` (시계열 분석의 핵심):**
>     *   **활용:** 이전 기간 대비 성장률, 다음 이벤트까지의 시간, 이전 구매와의 간격 등을 계산할 때 사용합니다. 금융 데이터 분석(주가 변동), 사용자 행동 분석(세션 간 이동), 물류 추적(단계별 시간) 등 시간적 순서가 중요한 데이터에서 필수적입니다.
>     *   **예시:** 월별 매출 데이터에서 전월 매출을 가져와 월별 성장률(MoM Growth)을 계산하거나, 고객의 구매 이력에서 이전 구매일과 현재 구매일의 차이를 계산하여 구매 주기 분석.
> *   **`FIRST_VALUE`와 `LAST_VALUE` (그룹 내 기준점 설정):**
>     *   **활용:** 각 그룹(파티션) 내에서 첫 번째 또는 마지막 로우의 값을 가져와 현재 로우와 비교하거나, 그룹 전체의 기준점으로 활용할 때 사용합니다. 예를 들어, 특정 고객의 첫 구매일, 마지막 구매일, 또는 특정 세션의 시작/종료 시간 등을 각 로우에 표시할 수 있습니다.
>     *   **`LAST_VALUE`의 주의사항:** `LAST_VALUE`는 기본 프레임(`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`) 때문에 `ORDER BY` 절이 있다면 현재 로우의 값을 반환하는 경우가 많습니다. 파티션 내의 **실제 마지막 값**을 가져오려면 `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`과 같이 명시적으로 **전체 파티션**을 프레임으로 지정해야 합니다. 이 점을 이해하지 못하면 예상과 다른 결과를 얻을 수 있습니다.
>
> *   **성능 고려사항:**
>     *   분석 함수는 `ORDER BY` 절을 포함하므로, 대규모 데이터셋에서는 정렬 비용이 발생할 수 있습니다. `PARTITION BY`와 `ORDER BY` 절에 적절한 인덱스를 활용하여 성능을 최적화해야 합니다.
>     *   `EXPLAIN` 명령을 통해 쿼리 실행 계획을 분석하고, 예상대로 동작하는지 검증하는 습관을 들이는 것이 중요합니다.
>
> 분석 함수는 데이터의 시간적, 순서적 관계를 파악하고, 그룹 내에서 특정 기준점을 설정하여 더 깊이 있는 분석을 수행하는 데 매우 강력한 도구입니다.
FROM employees;


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

> **프레임 정의: 윈도우 함수의 정교한 제어:**
> 프레임(Frame) 정의는 윈도우 함수가 계산될 로우들의 **실제 범위**를 지정하는 강력한 기능입니다. `PARTITION BY`와 `ORDER BY`가 윈도우의 논리적인 그룹과 순서를 정의한다면, 프레임은 그 윈도우 내에서 함수가 적용될 물리적인 로우의 집합을 세밀하게 제어합니다.
>
> *   **기본 프레임:**
>     *   `ORDER BY` 절만 사용하고 프레임 정의를 명시하지 않으면, 기본값은 `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`입니다. 이는 현재 로우부터 파티션의 시작까지의 모든 로우를 포함합니다.
>     *   `ORDER BY` 절이 없으면, 기본값은 `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`입니다. 이는 파티션 내의 모든 로우를 포함합니다.
>
> *   **`ROWS BETWEEN ... AND ...` (물리적 로우 개수 기준):**
>     *   현재 로우를 기준으로 물리적인 로우의 개수를 세어 프레임을 정의합니다. (예: `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`는 현재 로우와 그 이전 2개의 로우, 총 3개의 로우를 포함합니다.)
>     *   주로 이동 평균, 이동 합계 등 **정확한 개수의 로우**를 기준으로 계산할 때 사용됩니다.
>
> *   **`RANGE BETWEEN ... AND ...` (값 범위 기준):**
>     *   현재 로우의 값을 기준으로, 지정된 값 범위 내에 있는 모든 로우를 포함합니다. (예: `RANGE BETWEEN 100 PRECEDING AND CURRENT ROW`는 현재 로우의 값보다 100 작은 값부터 현재 로우의 값까지의 모든 로우를 포함합니다.)
>     *   동일한 값을 가진 로우들을 모두 포함해야 할 때 유용합니다. (예: 동일한 급여를 받는 모든 직원을 포함하여 계산)
>
> *   **활용 사례:**
>     *   **이동 평균/합계:** 주식 가격의 5일 이동 평균, 일별 매출의 7일 이동 합계 등 시계열 데이터의 추세를 완만하게 볼 때 `ROWS BETWEEN`을 사용합니다.
>     *   **누적 합계:** `SUM(sales) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)`와 같이 사용하여 특정 시점까지의 누적 합계를 계산합니다.
>     *   **그룹 내 특정 범위의 값:** 특정 직원의 급여와 유사한 급여를 받는 다른 직원들의 평균을 계산할 때 `RANGE BETWEEN`을 사용할 수 있습니다.
>
> *   **성능 고려사항:**
>     *   프레임 정의가 복잡해질수록 쿼리 성능에 영향을 미칠 수 있습니다. 특히 `UNBOUNDED FOLLOWING`과 같이 전체 파티션을 포함하는 프레임은 정렬 비용이 커질 수 있습니다.
>     *   `EXPLAIN` 명령을 통해 프레임 정의가 쿼리 실행 계획에 미치는 영향을 분석하는 것이 중요합니다.
>
> 프레임 정의는 윈도우 함수를 통해 매우 정교하고 유연한 분석을 가능하게 합니다. 각 옵션의 의미와 동작 방식을 정확히 이해하고, 비즈니스 요구사항에 맞춰 적절한 프레임을 선택하는 것이 중요합니다.

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

> **CTE 정의와 쿼리 실행:**
> `WITH` 절을 사용하여 CTE(Common Table Expression)를 정의하는 것은 복잡한 쿼리를 구조화하고 가독성을 높이는 현대 SQL의 핵심 기법입니다.
>
> *   **기본 문법:**
>     *   `WITH` 키워드로 시작하며, 하나 이상의 CTE를 정의할 수 있습니다. 여러 CTE는 콤마(`,`)로 구분합니다.
>     *   각 CTE는 `cte_name AS (SELECT ...)` 형태로 정의되며, `SELECT` 문은 CTE의 결과 집합을 정의합니다.
>     *   CTE는 메인 쿼리(`SELECT ... FROM cte_name ...`)가 실행되기 전에 논리적으로 먼저 실행됩니다.
> *   **CTE의 범위:**
>     *   CTE는 정의된 쿼리 내에서만 유효합니다. 즉, 한 쿼리에서 정의된 CTE는 다른 쿼리에서 직접 참조할 수 없습니다. (재사용성을 높이려면 뷰(View)나 저장 프로시저(Stored Procedure)를 고려해야 합니다.)
> *   **쿼리 실행과 성능:**
>     *   CTE는 기본적으로 **논리적인 개념**이며, 대부분의 DBMS 옵티마이저는 CTE를 물리적인 임시 테이블로 만들지 않고 메인 쿼리에 **인라인(inline)**하여 하나의 큰 쿼리로 최적화하려고 시도합니다. 이를 **뷰 머징(View Merging)**이라고 합니다.
>     *   따라서 CTE를 사용했다고 해서 반드시 성능이 향상되거나 별도의 임시 테이블이 생성되는 것은 아닙니다. 성능은 옵티마이저의 판단과 쿼리 내용에 따라 달라집니다.
>     *   하지만 CTE로 로직을 명확히 분리하면 옵티마이저가 쿼리의 의도를 더 잘 파악하고 효율적인 실행 계획을 수립할 기회를 얻을 수 있습니다.
>     *   **`EXPLAIN`의 중요성:** CTE를 사용한 후에는 항상 `EXPLAIN` 명령을 통해 쿼리 실행 계획을 분석하여 CTE가 의도한 대로 최적화되었는지, 성능 저하 요인이 발생하지 않았는지 검증해야 합니다. (자세한 내용은 [08_성능튜닝과_인덱스.md](./08_성능튜닝과_인덱스.md) 참조)
>
> CTE는 쿼리 가독성과 유지보수성을 크게 향상시키는 강력한 도구이므로, 복잡한 쿼리를 작성할 때 적극적으로 활용하는 습관을 들이는 것이 좋습니다.

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

> **CTE가 쿼리 개발 및 유지보수에 미치는 영향:**
> CTE는 단순한 문법적 편의를 넘어, 복잡한 SQL 쿼리를 마치 프로그래밍 언어의 함수처럼 모듈화하고 구조화하는 데 핵심적인 역할을 합니다. 이는 특히 대규모 데이터 분석 프로젝트나 데이터 웨어하우스 환경에서 빛을 발합니다.
>
> *   **디버깅 용이성:** 복잡한 쿼리에서 오류가 발생했을 때, CTE를 사용하면 각 단계별로 쿼리를 독립적으로 실행하여 중간 결과를 확인하고 문제의 원인을 빠르게 파악할 수 있습니다. 이는 중첩 서브쿼리에서는 불가능하거나 매우 어렵습니다.
> *   **협업 효율성:** 여러 분석가나 개발자가 하나의 복잡한 쿼리를 함께 작업할 때, CTE를 통해 각자의 담당 부분을 명확히 분리하고 이해하기 쉽게 만들 수 있습니다. 이는 코드 리뷰를 용이하게 하고, 소통 오류를 줄여줍니다.
> *   **테스트 용이성:** 각 CTE를 독립적인 단위로 테스트할 수 있으므로, 쿼리 전체의 정확성을 검증하는 데 도움이 됩니다.
> *   **성능 최적화 기회:** 옵티마이저가 CTE를 인라인 뷰처럼 처리하는 경우가 많지만, CTE로 로직을 명확히 분리하면 옵티마이저가 더 효율적인 실행 계획을 수립할 기회를 얻을 수 있습니다. 특히 동일한 CTE가 여러 번 참조될 경우, 데이터베이스는 이를 한 번만 계산하고 결과를 재사용하여 성능을 향상시킬 수 있습니다 (DBMS에 따라 다름).
>
> CTE는 현대 SQL의 필수적인 기법이며, 이를 능숙하게 사용하는 것은 데이터 분석가의 쿼리 작성 역량을 한 단계 높이는 중요한 지표입니다.

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

> **다중 CTE와 재귀 CTE의 강력함:**
> CTE는 단일 쿼리 내에서 여러 개의 논리적인 단계를 정의할 수 있게 함으로써, 복잡한 데이터 처리 및 분석을 훨씬 더 명확하고 효율적으로 수행할 수 있게 합니다.
>
> *   **다중 CTE의 활용:**
>     *   **복잡한 비즈니스 로직 분해:** 여러 단계의 계산이나 필터링이 필요한 복잡한 비즈니스 로직을 각 CTE로 분리하여 구현하면, 쿼리의 가독성이 극대화됩니다. 각 CTE는 이전 CTE의 결과를 참조할 수 있어, 마치 프로그래밍 언어의 함수 호출처럼 단계별로 데이터를 가공할 수 있습니다.
>     *   **중간 결과의 재사용:** 동일한 중간 결과가 여러 번 필요할 때, 이를 CTE로 정의하면 중복 계산을 피하고 쿼리 성능을 향상시킬 수 있습니다.
>     *   **디버깅 용이성:** 각 CTE를 독립적으로 실행하여 중간 결과를 확인하고 디버깅할 수 있어, 복잡한 쿼리의 오류를 찾는 시간을 단축시킵니다.
>
> *   **재귀 CTE의 활용 (계층형 데이터 분석):**
>     *   **개념:** `WITH RECURSIVE` 키워드를 사용하여 자기 자신을 참조하는 CTE를 정의합니다. 이는 계층형 데이터(예: 조직도, 댓글 스레드, BOM(Bill of Materials) 구조)를 탐색하거나, 그래프 데이터를 처리할 때 매우 유용합니다.
>     *   **구성:** 재귀 CTE는 크게 두 부분으로 나뉩니다.
>         *   **앵커 멤버 (Anchor Member):** 재귀의 시작점을 정의합니다. (예: 최상위 관리자, 루트 댓글)
>         *   **재귀 멤버 (Recursive Member):** 앵커 멤버의 결과 또는 이전 재귀 멤버의 결과를 참조하여 반복적으로 실행될 로직을 정의합니다. `UNION ALL`로 앵커 멤버와 연결됩니다.
>     *   **활용 사례:**
>         *   **조직도 탐색:** 특정 직원의 상위 관리자 또는 하위 직원을 모두 찾을 때.
>         *   **댓글 스레드:** 특정 댓글의 모든 답글을 계층적으로 조회할 때.
>         *   **경로 탐색:** 네트워크 그래프에서 특정 노드 간의 모든 경로를 찾을 때.
>
> *   **성능 고려사항:**
>     *   **재귀 CTE의 종료 조건:** 재귀 CTE는 무한 루프에 빠지지 않도록 반드시 명확한 종료 조건이 필요합니다. (예: `level` 컬럼을 추가하여 특정 깊이 이상은 탐색하지 않도록 제한)
>     *   **데이터 볼륨:** 재귀 CTE는 반복적인 연산을 수행하므로, 처리할 데이터의 계층이 깊거나 데이터 볼륨이 클 경우 성능 저하가 발생할 수 있습니다. `EXPLAIN`을 통해 실행 계획을 분석하고, 필요한 경우 인덱스 튜닝이나 다른 최적화 기법을 고려해야 합니다.
>
> 다중 CTE와 재귀 CTE는 SQL 쿼리의 표현력을 극대화하고, 복잡한 비즈니스 문제를 해결하는 데 필수적인 고급 기술입니다. 이를 통해 데이터 분석가는 더욱 강력하고 효율적인 분석을 수행할 수 있습니다.
