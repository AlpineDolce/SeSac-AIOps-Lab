<h2>테이블 연결</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-21

<h2>문서 목표</h2>
<p>이 문서는 복잡한 데이터 간 관계를 이해하고, 다양한 SQL 조인과 집합 연산을 실무에 효과적으로 활용하는 능력을 기르는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 테이블 연결: `JOIN` 완전 정복](#1-테이블-연결-join-완전-정복)
  - [1.1. `JOIN`의 개념과 필요성](#11-join의-개념과-필요성)
  - [1.2. `INNER JOIN`: 교집합](#12-inner-join-교집합)
  - [1.3. `OUTER JOIN` (`LEFT`, `RIGHT`): 합집합](#13-outer-join-left-right-합집합)
  - [1.4. `FULL OUTER JOIN` (MySQL에서 `UNION ALL` 활용)](#14-full-outer-join-mysql에서-union-all-활용)
  - [1.5. `CROSS JOIN`: 모든 조합](#15-cross-join-모든-조합)
  - [1.6. `SELF JOIN`: 자기 참조](#16-self-join-자기-참조)
  - [1.7. `JOIN` 성능 고려사항](#17-join-성능-고려사항)
  - [1.8. `USING` 절](#18-using-절)
  - [1.9. `NATURAL JOIN` (참고)](#19-natural-join-참고)
- [2. 데이터 집합 결합: `UNION`과 `UNION ALL`](#2-데이터-집합-결합-union과-union-all)
  - [2.1. `UNION` vs `UNION ALL` (중복 제거 여부)](#21-union-vs-union-all-중복-제거-여부)
  - [2.2. 복잡한 `UNION` 활용 (데이터 통합, 보고서 생성)](#22-복잡한-union-활용-데이터-통합-보고서-생성)
  - [2.3. `INTERSECT` 및 `EXCEPT` (MySQL 대체 방법)](#23-intersect-및-except-mysql-대체-방법)

---

## 1. 테이블 연결: `JOIN` 완전 정복

관계형 데이터베이스에서는 데이터 중복을 최소화하고 데이터 무결성을 유지하기 위해 데이터를 여러 테이블에 나누어 저장합니다. `JOIN`은 이렇게 분리된 테이블들을 공통된 컬럼(키)을 기준으로 연결하여 원하는 데이터를 통합적으로 조회할 때 사용합니다. 데이터 분석가에게 `JOIN`은 필수적인 기술입니다.

### 1.1. `JOIN`의 개념과 필요성

*   **개념:** 두 개 이상의 테이블을 연결하여 하나의 결과 집합으로 만드는 연산입니다.
*   **필요성:**
    *   **데이터 통합:** 여러 테이블에 분산된 관련 데이터를 한 번에 조회하여 분석에 활용할 수 있습니다.
    *   **정보 확장:** 특정 테이블의 정보에 다른 테이블의 상세 정보를 추가하여 더 풍부한 분석을 가능하게 합니다.
    *   **데이터 중복 제거:** 정규화된 데이터베이스에서 데이터를 효율적으로 관리하고 조회할 수 있습니다.

*   **데이터 분석가를 위한 `JOIN` 기본 전략: `LEFT JOIN` 우선 사용**
    데이터 분석 시에는 종종 특정 기준이 되는 데이터(예: 모든 사용자, 모든 주문)를 전부 유지한 채로 추가 정보를 결합해야 하는 경우가 많습니다. `INNER JOIN`은 양쪽 테이블에 모두 데이터가 존재하는 경우에만 결과를 반환하므로, 기준 테이블의 데이터가 의도치 않게 누락될 수 있습니다.

    **따라서, 분석 쿼리 작성 시에는 기준이 되는 테이블을 `FROM` 절에 먼저 두고, 필요한 정보가 담긴 테이블들을 `LEFT JOIN`으로 연결해 나가는 것을 기본 전략으로 삼는 것이 안전합니다.** 이렇게 하면 기준 데이터의 유실 없이 안정적으로 분석을 진행할 수 있습니다.

### 1.2. `INNER JOIN`: 교집합

`INNER JOIN`은 두 테이블에서 `JOIN` 조건에 맞는 로우만 반환합니다. 즉, 두 테이블에 모두 존재하는 데이터의 교집합을 가져옵니다. 가장 일반적으로 사용되는 `JOIN` 유형입니다.

```sql
-- 직원(employees)과 부서(departments) 테이블을 department_id로 연결하여 직원 이름과 부서명 조회
SELECT
    e.first_name, e.last_name,
    d.department_name
FROM
    employees e
INNER JOIN
    departments d ON e.department_id = d.department_id;

-- AS 키워드를 사용하여 테이블 별칭(Alias)을 지정하면 쿼리 가독성이 높아집니다.
-- ON 절에 JOIN 조건을 명시합니다.
```

### 1.3. `OUTER JOIN` (`LEFT`, `RIGHT`): 합집합

`OUTER JOIN`은 `JOIN` 조건에 맞지 않는 로우도 포함하여 반환합니다. 주로 한쪽 테이블의 모든 데이터를 유지하면서 다른 테이블의 매칭되는 데이터를 가져올 때 사용합니다.

*   **`LEFT JOIN` (또는 `LEFT OUTER JOIN`):**
    *   왼쪽 테이블의 모든 로우를 포함하고, 오른쪽 테이블에서는 `JOIN` 조건에 맞는 로우만 포함합니다.
    *   오른쪽 테이블에서 매칭되는 로우가 없으면 해당 컬럼은 `NULL`로 채워집니다.

    ```sql
    -- 모든 직원과 해당 부서명 조회. 부서가 없는 직원도 포함.
    SELECT
        e.first_name, e.last_name,
        d.department_name
    FROM
        employees e
    LEFT JOIN
        departments d ON e.department_id = d.department_id;
    ```

*   **`RIGHT JOIN` (또는 `RIGHT OUTER JOIN`):**
    *   오른쪽 테이블의 모든 로우를 포함하고, 왼쪽 테이블에서는 `JOIN` 조건에 맞는 로우만 포함합니다.
    *   왼쪽 테이블에서 매칭되는 로우가 없으면 해당 컬럼은 `NULL`로 채워집니다.

    ```sql
    -- 모든 부서와 해당 부서에 속한 직원명 조회. 직원이 없는 부서도 포함.
    SELECT
        d.department_name,
        e.first_name, e.last_name
    FROM
        employees e
    RIGHT JOIN
        departments d ON e.department_id = d.department_id;
    ```

### 1.4. `FULL OUTER JOIN` (MySQL에서 `UNION ALL` 활용)

`FULL OUTER JOIN`은 양쪽 테이블의 모든 로우를 포함합니다. `JOIN` 조건에 맞는 로우는 연결하고, 맞지 않는 로우는 다른 테이블의 컬럼을 `NULL`로 채워 반환합니다. MySQL은 `FULL OUTER JOIN`을 직접 지원하지 않으므로, `LEFT JOIN`과 `RIGHT JOIN`의 결과를 `UNION ALL`하여 구현합니다. 이때 `UNION ALL`은 중복 제거 오버헤드가 없어 `UNION`보다 성능상 유리합니다.

```sql
-- 직원과 부서 정보를 모두 포함 (매칭되지 않는 직원/부서도 포함)
SELECT
    e.first_name, e.last_name,
    d.department_name
FROM
    employees e
LEFT JOIN
    departments d ON e.department_id = d.department_id

UNION ALL -- UNION 대신 UNION ALL 사용

SELECT
    e.first_name, e.last_name,
    d.department_name
FROM
    employees e
RIGHT JOIN
    departments d ON e.department_id = d.department_id
WHERE
    e.employee_id IS NULL; -- LEFT JOIN 결과에 없는 RIGHT JOIN의 고유 로우만 추가
```

**`FULL OUTER JOIN` 구현 시 `UNION` vs `UNION ALL` (성능 고려사항):**
*   **`UNION`:** 두 `SELECT` 문의 결과를 합치고 중복된 로우를 제거합니다. 이 중복 제거 과정에서 추가적인 정렬 및 비교 작업이 발생하여 성능 오버헤드가 발생할 수 있습니다.
*   **`UNION ALL`:** 두 `SELECT` 문의 결과를 단순히 합치며 중복 제거를 하지 않습니다. 따라서 `UNION`보다 일반적으로 빠릅니다.

`FULL OUTER JOIN`을 `LEFT JOIN`과 `RIGHT JOIN`의 `UNION`으로 구현할 때, `LEFT JOIN` 결과와 `RIGHT JOIN` 결과 중 `INNER JOIN`에 해당하는 부분은 중복됩니다. `UNION`은 이 중복을 제거하지만, `UNION ALL`은 제거하지 않습니다. 하지만 `RIGHT JOIN` 부분에 `WHERE e.employee_id IS NULL` 조건을 추가하여 `LEFT JOIN` 결과에 없는 `RIGHT JOIN`의 고유 로우만 가져오도록 하면, `UNION ALL`을 사용해도 중복이 발생하지 않으면서 `UNION`의 중복 제거 오버헤드를 피할 수 있어 더 효율적입니다. 따라서 `FULL OUTER JOIN` 에뮬레이션 시에는 `UNION ALL`과 `WHERE IS NULL` 조합을 사용하는 것이 권장됩니다.

### 1.5. `CROSS JOIN`: 모든 조합

`CROSS JOIN`은 두 테이블의 모든 로우를 조합하여 가능한 모든 경우의 수를 반환합니다. `JOIN` 조건이 없으며, 결과 로우 수는 `테이블1 로우 수 * 테이블2 로우 수`가 됩니다. 데카르트 곱(Cartesian Product)이라고도 합니다.

```sql
-- 직원과 프로젝트의 모든 가능한 조합 조회 (의미 없는 조합이 많을 수 있음)
SELECT
    e.first_name, e.last_name,
    p.project_name
FROM
    employees e
CROSS JOIN
    projects p;
```

### 1.6. `SELF JOIN`: 자기 참조

`SELF JOIN`은 하나의 테이블을 마치 두 개의 다른 테이블처럼 사용하여 `JOIN`하는 것입니다. 주로 테이블 내에서 계층적 관계(예: 직원-상사 관계)나 동일 테이블 내의 관련 데이터를 찾을 때 사용합니다.

*   **비등가 조인 (Non-Equi JOIN):**
    지금까지 다룬 `JOIN`은 대부분 `ON e.id = d.id` 와 같이 등호(`=`)를 사용한 **등가 조인(Equi JOIN)**입니다. 하지만 `JOIN`의 `ON` 절에는 등호 외에 `BETWEEN`, `>`, `<` 등 다양한 비교 연산자를 사용할 수 있으며, 이를 **비등가 조인(Non-Equi JOIN)**이라고 합니다.

    **[활용 사례]**
    직원의 급여에 따라 급여 등급을 부여하고 싶을 때, 급여 등급 정보를 담은 별도의 테이블을 비등가 조인하여 활용할 수 있습니다.

    **`salary_grades` 테이블:**

    | grade | min_salary | max_salary |
    | :--- | :--- | :--- |
    | A | 80001 | 999999 |
    | B | 60001 | 80000 |
    | C | 40001 | 60000 |

    ```sql
    -- 각 직원의 급여가 어떤 등급에 속하는지 조회
    SELECT
        e.first_name,
        e.salary,
        sg.grade
    FROM
        employees e
    JOIN
        salary_grades sg ON e.salary BETWEEN sg.min_salary AND sg.max_salary;
    ```
    이처럼 비등가 조인은 특정 값의 범위를 기준으로 테이블을 연결할 때 매우 유용하게 사용됩니다.

```sql
-- 직원 테이블에서 직원과 해당 직원의 상사 이름 조회
-- (employees 테이블에 manager_id 컬럼이 있다고 가정)
SELECT
    e.first_name AS employee_name,
    m.first_name AS manager_name
FROM
    employees e
INNER JOIN
    employees m ON e.manager_id = m.employee_id;
```

### 1.7. `JOIN` 성능 고려사항

`JOIN` 쿼리의 성능은 연결되는 테이블의 크기, `ON` 절에 사용된 컬럼의 인덱스 유무, `JOIN` 순서, 그리고 **`JOIN` 컬럼의 데이터 타입 일치 여부** 등에 따라 크게 달라집니다.

*   **인덱스 활용:** `ON` 절에 사용되는 컬럼에는 인덱스를 생성하는 것이 필수적입니다. 인덱스가 없으면 데이터베이스는 전체 테이블 스캔(Full Table Scan)을 수행하여 성능이 저하됩니다.
*   **`JOIN` 순서:** 일반적으로 옵티마이저가 최적의 `JOIN` 순서를 결정하지만, 때로는 개발자가 `FROM` 절에 `STRAIGHT_JOIN` 키워드를 사용하여 `JOIN` 순서를 강제할 수 있습니다. 이는 특정 상황에서 성능을 개선할 수 있지만, 신중하게 사용해야 합니다.
*   **`NULL` 값과 `JOIN`:** `NULL` 값은 어떤 값과도 일치하지 않으므로, `JOIN` 조건에 `NULL`이 포함된 로우는 매칭되지 않습니다. `NULL` 값을 포함하는 로우를 `JOIN`하려면 `COALESCE` 함수를 사용하거나 `OR` 조건으로 `IS NULL`을 명시해야 합니다.
```sql
-- 예시: employees 테이블의 department_id가 NULL인 직원과 departments 테이블을 JOIN 시도 시 매칭되지 않음
SELECT e.first_name, d.department_name
FROM employees e
JOIN departments d ON e.department_id = d.department_id;

-- NULL 값을 포함하여 JOIN하려면 (예: department_id가 NULL인 직원도 포함)
SELECT e.first_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id OR e.department_id IS NULL;
```

*   **`JOIN` 컬럼의 데이터 타입 일치 (매우 중요!):**
    `JOIN` 조건으로 사용되는 컬럼들의 데이터 타입은 반드시 일치해야 합니다. 만약 데이터 타입이 다르면, 데이터베이스는 내부적으로 암시적 형변환(Implicit Type Conversion)을 시도합니다. 이 과정에서 다음과 같은 문제가 발생할 수 있습니다.
    *   **인덱스 사용 불가:** 형변환이 발생하면 해당 컬럼에 생성된 인덱스를 사용할 수 없게 되어 Full Table Scan이 발생하고 쿼리 성능이 심각하게 저하됩니다.
    *   **성능 오버헤드:** 형변환 자체에도 CPU 자원이 소모됩니다.
    *   **예상치 못한 결과:** 데이터 타입 변환 과정에서 데이터 손실이나 부정확한 비교가 발생할 수 있습니다.

    **예시:**
    `employees` 테이블의 `department_id`가 `INT` 타입이고, `departments` 테이블의 `department_id`가 `VARCHAR` 타입이라면, `JOIN` 시 성능 저하가 발생합니다.
    ```sql
    -- Bad (데이터 타입 불일치로 인한 성능 저하 가능성)
    SELECT e.first_name, d.department_name
    FROM employees e
    JOIN departments d ON e.department_id = d.department_id; -- department_id 타입이 다를 경우
    ```

    **해결 방안:**
    *   **스키마 설계 단계에서 데이터 타입 일치:** 가장 좋은 방법은 테이블 설계 단계에서 `JOIN`에 사용될 컬럼들의 데이터 타입을 일관되게 정의하는 것입니다.
    *   **명시적 형변환:** 불가피하게 데이터 타입이 다른 경우, `CAST()` 함수를 사용하여 명시적으로 형변환을 수행할 수 있습니다. 하지만 이 경우에도 형변환되는 컬럼에는 인덱스를 사용할 수 없으므로, 근본적인 해결책은 아닙니다.
    ```sql
    -- 명시적 형변환 (인덱스 사용 불가)
    SELECT e.first_name, d.department_name
    FROM employees e
    JOIN departments d ON CAST(e.department_id AS CHAR) = d.department_id;
    ```
    데이터 분석가는 `EXPLAIN` 명령을 통해 쿼리 실행 계획을 확인하여 `JOIN` 컬럼의 데이터 타입 불일치로 인한 성능 저하가 발생하는지 항상 확인해야 합니다.

*   **복잡한 `JOIN` 조건 예시:**
    `ON` 절에 `AND` 또는 `OR`를 사용하여 여러 조건을 조합할 수 있습니다. 특히 `OR` 조건은 인덱스 사용을 방해할 수 있으므로 주의해야 합니다.
    ```sql
    -- 예시: 특정 부서의 직원 또는 특정 급여 범위의 직원 조회
    SELECT e.first_name, e.last_name, d.department_name, e.salary
    FROM employees e
    JOIN departments d ON e.department_id = d.department_id
    WHERE (e.department_id = 1 AND e.salary > 60000) OR (e.department_id = 2 AND e.salary < 50000);
    ```

### 1.8. `USING` 절

`USING` 절은 `JOIN` 조건으로 사용되는 컬럼의 이름이 양쪽 테이블에서 동일할 때 `ON` 절 대신 사용할 수 있습니다. 쿼리를 더 간결하고 가독성 있게 만들어 줍니다. **단, `USING` 절은 `ON` 절처럼 복잡한 조건(예: `AND`, `OR` 연산자, 함수 사용)을 지정할 수 없으며, 오직 동일한 이름의 컬럼에 대한 동등 비교(`=`)만 가능합니다.**

```sql
-- department_id 컬럼이 양쪽 테이블에 모두 존재할 때
SELECT e.first_name, d.department_name
FROM employees e
INNER JOIN departments d USING (department_id);

-- ON 절을 사용한 동일한 쿼리
-- SELECT e.first_name, d.department_name
-- FROM employees e
-- INNER JOIN departments d ON e.department_id = d.department_id;
```

### 1.9. `NATURAL JOIN` (참고)

`NATURAL JOIN`은 두 테이블에서 이름과 데이터 타입이 동일한 모든 컬럼을 자동으로 찾아 `JOIN` 조건으로 사용합니다. `ON` 절이나 `USING` 절을 명시할 필요가 없어 매우 간결합니다.

**주의:** `NATURAL JOIN`은 편리하지만, 예상치 못한 컬럼이 `JOIN` 조건으로 사용될 수 있어 실무에서는 잘 사용되지 않습니다. 명시적인 `ON` 절이나 `USING` 절을 사용하는 것이 더 안전하고 권장됩니다.

```sql
-- employees와 departments 테이블에 department_id 컬럼이 동일하게 존재할 때
SELECT e.first_name, d.department_name
FROM employees e
NATURAL JOIN departments d;
```

## 2. 데이터 집합 결합: `UNION`과 `UNION ALL`

`UNION`과 `UNION ALL`은 두 개 이상의 `SELECT` 문의 결과를 하나의 결과 집합으로 결합할 때 사용합니다. `JOIN`이 컬럼을 옆으로 연결하는 반면, `UNION`은 로우를 아래로 연결합니다.

### 2.1. `UNION` vs `UNION ALL` (중복 제거 여부)

*   **`UNION`:** 두 `SELECT` 문의 결과를 결합하고, **중복된 로우를 자동으로 제거**합니다.
*   **`UNION ALL`:** 두 `SELECT` 문의 결과를 결합하고, **중복된 로우를 제거하지 않고 모두 포함**합니다.

**주의사항:** `UNION` 연산에 참여하는 `SELECT` 문들은 다음 조건을 만족해야 합니다.
*   선택하는 컬럼의 **개수**가 동일해야 합니다.
*   각 컬럼의 **데이터 타입**이 호환 가능해야 합니다.
*   컬럼의 **순서**가 일치해야 합니다.

```sql
-- 예시: 두 개의 가상 테이블 (sales_2022, sales_2023)이 있다고 가정
-- sales_2022: (product_id INT, amount DECIMAL)
-- sales_2023: (product_id INT, amount DECIMAL)

-- UNION: 중복 제거
SELECT product_id, amount FROM sales_2022
UNION
SELECT product_id, amount FROM sales_2023;

-- UNION ALL: 중복 포함
SELECT product_id, amount FROM sales_2022
UNION ALL
SELECT product_id, amount FROM sales_2023;
```

### 2.2. 복잡한 `UNION` 활용 (데이터 통합, 보고서 생성)

`UNION`은 서로 다른 구조의 데이터를 통합하거나, 여러 소스에서 가져온 데이터를 하나의 보고서 형태로 만들 때 유용합니다.

```sql
-- 예시: 직원과 고객의 연락처 정보를 통합하여 조회
-- (employees 테이블: employee_id, first_name, last_name, email, phone_number)
-- (customers 테이블: customer_id, first_name, last_name, email, phone_number)

SELECT employee_id AS id, first_name, last_name, email, phone_number, 'Employee' AS type
FROM employees
UNION ALL
SELECT customer_id AS id, first_name, last_name, email, phone_number, 'Customer' AS type
FROM customers
ORDER BY type, last_name, first_name;
```

### 2.3. `INTERSECT` 및 `EXCEPT` (MySQL 대체 방법)

다른 SQL 데이터베이스(예: PostgreSQL, SQL Server, Oracle)에서는 `INTERSECT` (교집합)와 `EXCEPT` (차집합) 연산자를 지원합니다. MySQL은 이들을 직접 지원하지 않지만, `JOIN`이나 `NOT EXISTS` 서브쿼리를 사용하여 동일한 결과를 얻을 수 있습니다.

*   **`INTERSECT` (교집합) 대체:** `INNER JOIN` 또는 `EXISTS` 서브쿼리

    ```sql
    -- sales_2022와 sales_2023에 모두 존재하는 product_id 조회
    SELECT product_id FROM sales_2022
    INNER JOIN sales_2023 USING (product_id);

    -- 또는

    SELECT product_id FROM sales_2022
    WHERE EXISTS (SELECT 1 FROM sales_2023 WHERE sales_2023.product_id = sales_2022.product_id);
    ```

*   **`EXCEPT` (차집합) 대체:** `LEFT JOIN`과 `IS NULL` 또는 `NOT EXISTS` 서브쿼리

    ```sql
    -- sales_2022에는 있지만 sales_2023에는 없는 product_id 조회
    SELECT s22.product_id FROM sales_2022 s22
    LEFT JOIN sales_2023 s23 ON s22.product_id = s23.product_id
    WHERE s23.product_id IS NULL;

    -- 또는

    SELECT product_id FROM sales_2022
    WHERE NOT EXISTS (SELECT 1 FROM sales_2023 WHERE sales_2023.product_id = sales_22.product_id);
    ```