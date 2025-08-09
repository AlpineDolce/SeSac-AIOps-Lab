<h2>SQL 핵심 문법: 테이블 연결 (JOIN)과 집합 연산 (실무 활용 중심)</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-05-21

<h2>문서 목표</h2>
<p>이 문서는 관계형 데이터베이스에서 데이터를 통합하고 분석하는 핵심 기술인 <strong>SQL JOIN과 집합 연산</strong>에 대해 심도 있게 다룹니다. 각 개념의 정의, 실제 코드에서의 활용법, 그리고 <strong>데이터 분석 및 AI 실무에서 발생할 수 있는 주의사항과 활용 팁</strong>을 상세한 예제와 함께 설명하여, SQL을 활용한 데이터 통합 및 분석의 견고한 기초를 다지는 것을 목표로 합니다.</p>

<h2>목차</h2>

- [1. 테이블 연결: `JOIN`](#1-테이블-연결-join)
  - [1.1. `JOIN`의 개념과 필요성](#11-join의-개념과-필요성)
  - [1.2. `INNER JOIN`: 교집합 데이터 추출](#12-inner-join-교집합-데이터-추출)
  - [1.3. `OUTER JOIN` (`LEFT`, `RIGHT`): 기준 테이블의 데이터 유지](#13-outer-join-left-right-기준-테이블의-데이터-유지)
    - [1.3.1. `LEFT JOIN` (또는 `LEFT OUTER JOIN`): 왼쪽 테이블 기준](#131-left-join-또는-left-outer-join-왼쪽-테이블-기준)
    - [1.3.2. `RIGHT JOIN` (또는 `RIGHT OUTER JOIN`): 오른쪽 테이블 기준](#132-right-join-또는-right-outer-join-오른쪽-테이블-기준)
  - [1.4. `FULL OUTER JOIN` (MySQL 대체): 양쪽 테이블의 모든 데이터 포함](#14-full-outer-join-mysql-대체-양쪽-테이블의-모든-데이터-포함)
  - [1.5. `CROSS JOIN`: 데카르트 곱 생성](#15-cross-join-데카르트-곱-생성)
  - [1.6. `SELF JOIN`: 동일 테이블 내 관계 분석](#16-self-join-동일-테이블-내-관계-분석)
  - [1.7. 비등가 조인 (Non-Equi JOIN): 복잡한 조건으로 연결](#17-비등가-조인-non-equi-join-복잡한-조건으로-연결)
  - [1.8. `JOIN` 성능 최적화 전략: 빠르고 효율적인 데이터 통합](#18-join-성능-최적화-전략-빠르고-효율적인-데이터-통합)
  - [1.9. `USING` 절: 간결한 `JOIN` 조건](#19-using-절-간결한-join-조건)
  - [1.10. `NATURAL JOIN` (참고): 자동 `JOIN` 조건](#110-natural-join-참고-자동-join-조건)
- [2. 데이터 집합 결합: `UNION`과 `UNION ALL`](#2-데이터-집합-결합-union과-union-all)
  - [2.1. `UNION` vs `UNION ALL`: 중복 처리와 성능](#21-union-vs-union-all-중복-처리와-성능)
  - [2.2. 복잡한 `UNION` 활용: 다양한 데이터 통합 시나리오](#22-복잡한-union-활용-다양한-데이터-통합-시나리오)
  - [2.3. `INTERSECT` 및 `EXCEPT` (MySQL 대체): 집합 연산의 구현](#23-intersect-및-except-mysql-대체-집합-연산의-구현)
    - [2.3.1. `INTERSECT` (교집합) 대체: 두 집합에 모두 존재하는 요소 찾기](#231-intersect-교집합-대체-두-집합에-모두-존재하는-요소-찾기)
    - [2.3.2. `EXCEPT` (차집합) 대체: 한 집합에만 존재하는 요소 찾기](#232-except-차집합-대체-한-집합에만-존재하는-요소-찾기)

---

## 1. 테이블 연결: `JOIN`

관계형 데이터베이스에서는 데이터 중복을 최소화하고 데이터 무결성을 유지하기 위해 데이터를 여러 테이블에 나누어 저장합니다. `JOIN`은 이렇게 분리된 테이블들을 공통된 컬럼(키)을 기준으로 연결하여 원하는 데이터를 통합적으로 조회할 때 사용합니다. 데이터 분석가에게 `JOIN`은 필수적인 기술이며, 데이터 통합 및 분석의 핵심입니다.

### 1.1. `JOIN`의 개념과 필요성

*   **개념:** 두 개 이상의 테이블을 공통된 컬럼(키)을 기준으로 연결하여 하나의 결과 집합으로 만드는 연산입니다. 마치 흩어진 퍼즐 조각들을 맞춰 하나의 그림을 완성하는 것과 같습니다.
*   **필요성:**
    *   **데이터 통합:** 여러 테이블에 분산된 관련 데이터를 한 번에 조회하여 분석에 활용할 수 있습니다. (예: 고객 정보와 주문 정보를 결합하여 고객별 구매 내역 분석)
    *   **정보 확장:** 특정 테이블의 정보에 다른 테이블의 상세 정보를 추가하여 더 풍부한 분석을 가능하게 합니다. (예: 주문 내역에 상품명, 가격 등 상품 상세 정보 추가)
    *   **데이터 중복 제거 및 효율적 관리:** 정규화된 데이터베이스에서 데이터를 효율적으로 관리하고 조회할 수 있습니다.

**실무적 관점: `JOIN` 기본 전략 - `LEFT JOIN` 우선 사용**
데이터 분석 시에는 특정 기준이 되는 데이터(예: 모든 사용자, 모든 주문, 모든 상품)를 **절대 누락시키지 않고** 추가 정보를 결합해야 하는 경우가 압도적으로 많습니다. `INNER JOIN`은 양쪽 테이블에 모두 매칭되는 데이터가 있는 경우에만 결과를 반환하므로, 기준 테이블의 중요한 로우가 의도치 않게 분석에서 제외될 수 있습니다.

따라서, 분석 쿼리 작성 시에는 분석의 기준이 되는 핵심 테이블(Fact Table 또는 Primary Entity)을 `FROM` 절에 먼저 두고, 필요한 추가 정보가 담긴 테이블들을 `LEFT JOIN`으로 연결해 나가는 것을 기본 전략으로 삼는 것이 가장 안전하고 권장됩니다. 이렇게 하면 기준 데이터의 유실 없이 안정적으로 분석을 진행할 수 있으며, 매칭되지 않는 데이터(오른쪽 테이블의 컬럼이 `NULL`로 표시됨)를 통해 데이터의 누락이나 불일치 여부를 쉽게 파악하고 추가적인 데이터 품질 검증이나 분석 방향을 설정할 수 있습니다.

| JOIN 유형 | 선택되는 데이터 (벤 다이어그램 관점) | 주요 용도 | 비고 |
| :--- | :--- | :--- | :--- |
| **`INNER JOIN`** | 두 테이블에 공통으로 존재하는 행 (교집합: A ∩ B) | 매칭되는 데이터만 필요할 때 (예: 주문이 있는 고객) | 가장 기본적이고 흔하게 사용됨. |
| **`LEFT JOIN`** | 왼쪽 테이블의 모든 행 + 오른쪽 테이블의 매칭되는 행 (A 전체) | 기준 테이블의 데이터를 모두 유지할 때 (예: 모든 고객의 주문 내역, 없으면 NULL) | 데이터 누락 방지에 필수적. `ANTI JOIN` 패턴으로 활용 가능. |
| **`RIGHT JOIN`** | 오른쪽 테이블의 모든 행 + 왼쪽 테이블의 매칭되는 행 (B 전체) | `LEFT JOIN`과 동일하나 기준 테이블이 오른쪽에 위치. | 가독성을 위해 `LEFT JOIN`으로 통일하는 것을 권장. |
| **`FULL OUTER JOIN`** | 양쪽 테이블의 모든 행 (합집합: A ∪ B) | 두 데이터셋의 전체 목록을 비교할 때 (예: 올해 고객 vs 작년 고객) | MySQL 미지원. `LEFT JOIN` + `UNION` + `RIGHT JOIN`으로 구현. |
| **`CROSS JOIN`** | 두 테이블의 모든 가능한 행의 조합 (데카르트 곱) | 분석을 위한 기준 데이터 생성, 대량 테스트 데이터 생성 | `WHERE` 절이 없으면 의도치 않은 대규모 결과가 나올 수 있어 주의. |
| **`SELF JOIN`** | 동일 테이블 내에서 조건을 만족하는 행들의 조합 | 계층 구조 분석 (예: 직원-상사), 동일 테이블 내 데이터 비교 | 반드시 테이블 별칭(Alias) 사용 필요. |

- **다뤄볼 테이블**

    ```plaintext
    Employee Table                                 Department Table

    | EmployeeID | EmployeeName | DepartmentID |   | DepartmentID | DepartmentName |
    |------------|--------------|--------------|   |--------------|----------------|
    | 1          | John         | 1            |   | 1            | Sales          |
    | 2          | Jane         | 2            |   | 2            | Marketing      |
    | 3          | Mark         | 3            |   | 3            | HR             |
    | 4          | Emily        | 1            |   | 4            | IT             |
    | 5          | Brian        | 4            |   | 6            | Operations     |
    ```

### 1.2. `INNER JOIN`: 교집합 데이터 추출

`INNER JOIN`은 두 테이블에서 `JOIN` 조건에 맞는 로우만 반환합니다. 즉, 두 테이블에 모두 존재하는 데이터의 교집합을 가져옵니다. 가장 일반적으로 사용되는 `JOIN` 유형입니다.

```sql
-- 직원(employees)과 부서(departments) 테이블을 department_id로 연결하여 직원 이름과 부서명 조회
-- (부서에 소속된 직원, 직원이 소속된 부서만 조회)
SELECT e.EmployeeID, e.EmployeeName, d.DepartmentName
FROM Employee e
INNER JOIN Department d ON e.DepartmentID = d.DepartmentID;
-- AS 키워드를 사용하여 테이블 별칭(Alias)을 지정하면 쿼리 가독성이 높아집니다.
-- ON 절에 JOIN 조건을 명시합니다.
```
- **INNER JOIN으로 합친 결과**
    ```plaintext
    | EmployeeID | EmployeeName | DepartmentName |
    | ---------- | ------------ | -------------- |
    | 1          | John         | Sales          |
    | 2          | Jane         | Marketing      |
    | 3          | Mark         | HR             |
    | 4          | Emily        | Sales          |
    ```
**실무 활용 시나리오:**
*   **매칭되는 데이터만 필요할 때:** 예를 들어, 주문이 발생한 상품의 정보만 필요하거나, 특정 부서에 소속된 직원들의 정보만 필요할 때 사용합니다.
*   **데이터 유효성 검사:** 두 테이블 간의 관계가 항상 유효한지 확인할 때 (예: 모든 주문에 유효한 고객 ID가 있는지).

### 1.3. `OUTER JOIN` (`LEFT`, `RIGHT`): 기준 테이블의 데이터 유지

`OUTER JOIN`은 `JOIN` 조건에 맞지 않는 로우도 포함하여 반환합니다. 주로 한쪽 테이블의 모든 데이터를 유지하면서 다른 테이블의 매칭되는 데이터를 가져올 때 사용합니다.

#### 1.3.1. `LEFT JOIN` (또는 `LEFT OUTER JOIN`): 왼쪽 테이블 기준

*   왼쪽 테이블의 모든 로우를 포함하고, 오른쪽 테이블에서는 `JOIN` 조건에 맞는 로우만 포함합니다.
*   오른쪽 테이블에서 매칭되는 로우가 없으면 해당 컬럼은 `NULL`로 채워집니다.

```sql
-- 모든 직원과 해당 부서명 조회. 부서가 없는 직원도 포함.
SELECT e.EmployeeID, e.EmployeeName, d.DepartmentName
FROM Employee e
LEFT JOIN Department d ON e.DepartmentID = d.DepartmentID;
```
- **LEFT JOIN으로 합친 결과**
    ```plaintext
    | EmployeeID | EmployeeName | DepartmentName |
    | ---------- | ------------ | -------------- |
    | 1          | John         | Sales          |
    | 2          | Jane         | Marketing      |
    | 3          | Mark         | HR             |
    | 4          | Emily        | Sales          |
    | 5          | Brian        | NULL           |
    ```
**실무 활용 시나리오:**
*   **기준 데이터의 누락 방지:** 모든 고객의 구매 이력을 분석할 때, 구매 이력이 없는 고객도 결과에 포함시켜야 할 경우 (고객 테이블 `LEFT JOIN` 주문 테이블).
*   **매칭되지 않는 데이터 찾기 (`ANTI JOIN` 패턴):** `LEFT JOIN` 후 오른쪽 테이블의 컬럼이 `IS NULL`인 조건을 추가하여, 왼쪽 테이블에는 있지만 오른쪽 테이블에는 없는 데이터를 찾을 수 있습니다. (예: 아직 주문하지 않은 고객 목록, 프로젝트에 배정되지 않은 직원 목록)
```sql
-- 아직 주문하지 않은 고객 목록 조회
SELECT c.customer_name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```

#### 1.3.2. `RIGHT JOIN` (또는 `RIGHT OUTER JOIN`): 오른쪽 테이블 기준

*   오른쪽 테이블의 모든 로우를 포함하고, 왼쪽 테이블에서는 `JOIN` 조건에 맞는 로우만 포함합니다.
*   왼쪽 테이블에서 매칭되는 로우가 없으면 해당 컬럼은 `NULL`로 채워집니다.

```sql
-- 모든 부서와 해당 부서에 속한 직원명 조회. 직원이 없는 부서도 포함.
SELECT e.EmployeeID, e.EmployeeName, d.DepartmentName
FROM Employee e
RIGHT JOIN Department d ON e.DepartmentID = d.DepartmentID;
```
- **RIGHT JOIN으로 합친 결과**
    ```plaintext
    | EmployeeID | EmployeeName | DepartmentName |
    | ---------- | ------------ | -------------- |
    | 1          | John         | Sales          |
    | 2          | Jane         | Marketing      |
    | 3          | Mark         | HR             |
    | 4          | Emily        | Sales          |
    | NULL       | NULL         | IT             |
    | NULL       | NULL         | Operations     |
    ```
**실무 활용 시나리오:**
*   `LEFT JOIN`과 동일한 목적이지만, 쿼리 작성 시 기준 테이블을 오른쪽에 두는 경우에 사용합니다. (예: 모든 상품의 판매 현황을 볼 때, 아직 판매되지 않은 상품도 포함)

### 1.4. `FULL OUTER JOIN` (MySQL 대체): 양쪽 테이블의 모든 데이터 포함

`FULL OUTER JOIN`은 양쪽 테이블의 모든 로우를 포함합니다. `JOIN` 조건에 맞는 로우는 연결하고, 맞지 않는 로우는 다른 테이블의 컬럼을 `NULL`로 채워 반환합니다. MySQL은 `FULL OUTER JOIN`을 직접 지원하지 않으므로, `LEFT JOIN`과 `RIGHT JOIN`의 결과를 `UNION ALL`하여 구현합니다.

```sql
-- MySQL 미지원
SELECT e.EmployeeID, e.EmployeeName, d.DepartmentName
FROM Employee e
FULL OUTER JOIN Department d ON e.DepartmentID = d.DepartmentID;
```
- **FULL OUTER JOIN으로 합친 결과**
    ```plaintext
    | EmployeeID | EmployeeName | DepartmentName |
    | ---------- | ------------ | -------------- |
    | 1          | John         | Sales          |
    | 2          | Jane         | Marketing      |
    | 3          | Mark         | HR             |
    | 4          | Emily        | Sales          |
    | 5          | Brian        | NULL           |
    | NULL       | NULL         | IT             |
    | NULL       | NULL         | Operations     |
    ```
**실무 팁: `FULL OUTER JOIN` 구현 시 `UNION` vs `UNION ALL`**
`FULL OUTER JOIN`을 `LEFT JOIN`과 `RIGHT JOIN`의 `UNION`으로 구현할 때, `UNION ALL`과 `WHERE IS NULL` 조합을 사용하는 것이 중복 제거 오버헤드를 피할 수 있어 더 효율적입니다. `UNION`은 중복 제거 과정에서 추가적인 정렬 작업을 수행하므로 성능에 불리할 수 있습니다.

**실무 활용 시나리오:**
*   **두 집합 간의 모든 차이점 및 공통점 분석:** 예를 들어, 올해 가입한 고객과 작년 가입한 고객의 목록을 비교하여, 올해만 가입한 고객, 작년에만 가입한 고객, 그리고 양쪽 모두에 해당하는 고객을 한 번에 파악할 때 유용합니다.

### 1.5. `CROSS JOIN`: 데카르트 곱 생성

`CROSS JOIN`은 두 테이블의 모든 로우를 조합하여 가능한 모든 경우의 수를 반환합니다. `JOIN` 조건이 없으며, 결과 로우 수는 `테이블1 로우 수 * 테이블2 로우 수`가 됩니다. 데카르트 곱(Cartesian Product)이라고도 합니다.

```sql
-- 직원과 프로젝트의 모든 가능한 조합 조회 (의미 없는 조합이 많을 수 있음)
SELECT e.EmployeeID, e.EmployeeName, d.DepartmentName
FROM Employee e
CROSS JOIN Department d;
```
- **CROSS JOIN으로 합친 결과**
    ```plaintext
    | EmployeeID | EmployeeName | DepartmentName |
    | ---------- | ------------ | -------------- |
    | 1          | John         | Sales          |
    | 1          | John         | Marketing      |
    | 1          | John         | HR             |
    | 1          | John         | IT             |
    | 1          | John         | Operations     |
    | 2          | Jane         | Sales          |
    | 2          | Jane         | Marketing      |
    | 2          | Jane         | HR             |
    | 2          | Jane         | IT             |
    | 2          | Jane         | Operations     |
    | 3          | Mark         | Sales          |
    | 3          | Mark         | Marketing      |
    | 3          | Mark         | HR             |
    | 3          | Mark         | IT             |
    | 3          | Mark         | Operations     |
    | 4          | Emily        | Sales          |
    | 4          | Emily        | Marketing      |
    | 4          | Emily        | HR             |
    | 4          | Emily        | IT             |
    | 4          | Emily        | Operations     |
    | 5          | Brian        | Sales          |
    | 5          | Brian        | Marketing      |
    | 5          | Brian        | HR             |
    | 5          | Brian        | IT             |
    | 5          | Brian        | Operations     |
    ```
**실무 활용 시나리오:**
*   **기준 데이터 생성:** 특정 기간의 모든 날짜와 모든 상품의 조합을 생성하여, 판매 데이터가 없는 날짜/상품 조합에도 0을 채워 넣는 등 분석을 위한 기준 데이터를 만들 때 사용합니다.
*   **테스트 데이터 생성:** 대량의 테스트 데이터를 생성할 때 유용합니다.
*   **집계 함수와 함께 사용:** 특정 그룹별로 모든 가능한 조합을 생성한 후 집계 함수를 적용하여 빈 값을 채우는 데 활용될 수 있습니다.

### 1.6. `SELF JOIN`: 동일 테이블 내 관계 분석

`SELF JOIN`은 하나의 테이블을 마치 두 개의 다른 테이블처럼 사용하여 `JOIN`하는 것입니다. 주로 테이블 내에서 계층적 관계(예: 직원-상사 관계)나 동일 테이블 내의 관련 데이터를 찾을 때 사용합니다. 반드시 테이블 별칭(Alias)을 사용하여 두 테이블을 구분해야 합니다.

```sql
-- 직원 테이블에서 직원과 해당 직원의 상사 이름 조회
-- (employees 테이블에 manager_id 컬럼이 있고, employee_id를 참조한다고 가정)
SELECT e1.EmployeeID AS Employee1_ID, e1.EmployeeName AS Employee1_Name, 
       e2.EmployeeID AS Employee2_ID, e2.EmployeeName AS Employee2_Name
FROM Employee e1
INNER JOIN Employee e2 ON e1.DepartmentID = e2.DepartmentID
WHERE e1.EmployeeID < e2.EmployeeID;
```
- **SELF JOIN으로 합친 결과**
    ```plaintext
    | Employee1_ID | Employee1_Name | Employee2_ID | Employee2_Name |
    | ------------ | -------------- | ------------ | -------------- |
    | 1            | John           | 4            | Emily          |
    ```
**실무 활용 시나리오:**
*   **계층 구조 분석:** 조직도, 카테고리 트리 등 계층적 데이터를 분석할 때 (예: 부모-자식 관계, 상사-부하 직원 관계).
*   **동일 테이블 내 비교:** 같은 테이블 내에서 특정 조건에 맞는 로우들을 비교할 때 (예: 같은 부서 내에서 급여가 높은 직원 찾기, 동일 상품을 구매한 고객 찾기).

### 1.7. 비등가 조인 (Non-Equi JOIN): 복잡한 조건으로 연결

`JOIN`의 `ON` 절에는 등호(`=`) 외에 `BETWEEN`, `>`, `<`, `!=` 등 다양한 비교 연산자를 사용할 수 있으며, 이를 **비등가 조인(Non-Equi JOIN)**이라고 합니다. 특정 값의 범위를 기준으로 테이블을 연결하거나, 복잡한 조건에 따라 데이터를 매칭할 때 매우 유용합니다.

```sql
-- 각 직원의 급여가 어떤 등급에 속하는지 조회 (salary_grades 테이블과 비등가 조인)
SELECT e.EmployeeID, e.EmployeeName, d.DepartmentName
FROM Employee e
JOIN Department d ON e.EmployeeID != d.DepartmentID;
```
- **Non-Equi JOIN으로 합친 결과**
    ```plaintext
    | EmployeeID | EmployeeName | DepartmentName |
    | ---------- | ------------ | -------------- |
    | 1          | John         | Marketing      |
    | 2          | Jane         | Sales          |
    | 3          | Mark         | Sales          |
    | 4          | Emily        | HR             |
    | 5          | Brian        | HR             |
    ```
**실무 활용 시나리오:**
*   **등급/구간 분류:** 점수, 급여, 연령 등을 특정 구간이나 등급으로 분류할 때 (예: 고객의 연령대에 따른 마케팅 캠페인 분류).
*   **시간 기반 매칭:** 특정 이벤트가 발생한 시간 범위 내의 다른 이벤트를 찾을 때 (예: 로그인 기록과 구매 기록을 시간 범위로 연결).

**성능 고려사항 및 대안:** 비등가 조인은 강력하지만, 등가 조인에 비해 성능상 불리한 경우가 많습니다. 특히 `BETWEEN`이나 범위 조건(`>`, `<`)을 사용하는 비등가 조인은 인덱스를 효율적으로 활용하기 어렵거나, 아예 사용하지 못하여 Full Table Scan을 유발할 수 있습니다.

*   **실무적 대안:**
    1.  **`CASE` 문 활용:** 조건이 고정적이고 복잡하지 않다면 `JOIN` 대신 `CASE` 문을 사용하여 컬럼 값을 직접 분류하는 것이 성능상 더 유리할 수 있습니다.
    2.  **미리 계산된 컬럼 추가:** 분석 시 자주 사용되는 등급 정보라면, 원본 테이블에 컬럼을 추가하고 ETL 프로세스나 배치 작업을 통해 미리 계산하여 저장해두는 것이 가장 효율적입니다.
    3.  **임시 테이블 또는 CTE 활용:** 복잡한 비등가 조인 대신, 필요한 데이터를 미리 필터링하거나 가공하여 임시 테이블 또는 CTE(Common Table Expression)로 만든 후 등가 조인을 시도하는 것이 성능상 유리할 수 있습니다.

### 1.8. `JOIN` 성능 최적화 전략: 빠르고 효율적인 데이터 통합

`JOIN` 쿼리의 성능은 연결되는 테이블의 크기, `ON` 절에 사용된 컬럼의 인덱스 유무, `JOIN` 순서, 그리고 `JOIN` 컬럼의 데이터 타입 일치 여부 등에 따라 크게 달라집니다. 대규모 데이터셋에서는 `JOIN` 최적화가 쿼리 실행 시간을 좌우합니다.

*   **인덱스 활용 극대화:** `ON` 절에 사용되는 컬럼(특히 `FOREIGN KEY` 컬럼)에는 반드시 인덱스를 생성해야 합니다. 인덱스는 데이터베이스가 매칭되는 로우를 빠르게 찾을 수 있도록 돕습니다.
*   **`JOIN` 컬럼의 데이터 타입 일치:** `JOIN` 조건으로 사용되는 컬럼들의 데이터 타입은 반드시 일치해야 합니다. 만약 데이터 타입이 다르면, 데이터베이스는 내부적으로 암시적 형변환을 시도하며 이 과정에서 인덱스 사용 불가, 성능 오버헤드, 예상치 못한 결과 등의 문제가 발생할 수 있습니다. `EXPLAIN` 명령으로 `Using where` 또는 `Using temporary`가 나타나는지 확인합니다.
*   **`ON` 절과 `WHERE` 절의 역할 분리 및 최적화:**
    *   **`ON` 절:** 테이블 간의 **논리적인 연결 조건**을 정의합니다. `OUTER JOIN`의 경우 `ON` 절의 조건은 `JOIN`이 발생하기 전에 적용됩니다.
    *   **`WHERE` 절:** `JOIN`된 결과 집합에서 **최종적으로 필요한 로우를 필터링**하는 데 사용합니다. `OUTER JOIN`의 경우 `WHERE` 절의 조건은 `JOIN`이 완료된 후에 적용됩니다.
    *   **`LEFT JOIN`과 `WHERE` 절의 주의:** `LEFT JOIN` 후 오른쪽 테이블의 컬럼에 `WHERE` 조건을 걸면, 해당 조건에 맞지 않는 왼쪽 테이블의 로우(오른쪽이 `NULL`인 로우)가 필터링되어 사실상 `INNER JOIN`처럼 동작할 수 있습니다. 의도하지 않은 결과가 나올 수 있으므로 주의해야 합니다.
        ```sql
        -- (의도: 모든 직원 중 부서명이 'Sales'인 직원만 조회) -> 잘못된 쿼리
        SELECT e.first_name, d.department_name
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.department_id
        WHERE d.department_name = 'Sales'; -- 이 조건 때문에 부서 없는 직원은 제외됨

        -- (올바른 쿼리: LEFT JOIN의 ON 절에 조건 추가)
        SELECT e.first_name, d.department_name
        FROM employees e
        LEFT JOIN departments d ON e.department_id = d.department_id AND d.department_name = 'Sales';
        -- 이렇게 하면 부서 없는 직원도 포함되면서, Sales 부서가 아닌 직원은 department_name이 NULL로 표시됨
        ```
*   **`JOIN` 순서 최적화:** 데이터베이스 옵티마이저가 최적의 `JOIN` 순서를 결정하지만, 때로는 개발자가 `STRAIGHT_JOIN` 힌트(MySQL)를 사용하여 `JOIN` 순서를 강제하거나, 서브쿼리/CTE를 활용하여 미리 필터링된 작은 테이블을 먼저 `JOIN`하는 것이 유리할 수 있습니다.
*   **필요한 컬럼만 `SELECT`:** `SELECT *` 대신 필요한 컬럼만 명시하여 네트워크 트래픽과 메모리 사용량을 줄입니다.
*   **`EXPLAIN` 명령 활용:** 쿼리 실행 계획을 분석하여 `JOIN`이 효율적으로 이루어지고 있는지, 인덱스가 잘 활용되고 있는지, 불필요한 Full Table Scan이 발생하는지 등을 주기적으로 확인합니다.

### 1.9. `USING` 절: 간결한 `JOIN` 조건

`USING` 절은 `JOIN` 조건으로 사용되는 컬럼의 이름이 양쪽 테이블에서 동일할 때 `ON` 절 대신 사용할 수 있습니다. 쿼리를 더 간결하고 가독성 있게 만들어 줍니다. **단, `USING` 절은 `ON` 절처럼 복잡한 조건(예: `AND`, `OR` 연산자, 함수 사용)을 지정할 수 없으며, 오직 동일한 이름의 컬럼에 대한 동등 비교(`=`)만 가능합니다.**

```sql
-- department_id 컬럼이 양쪽 테이블에 모두 존재할 때
SELECT e.first_name, d.department_name
FROM employees e
INNER JOIN departments d USING (department_id);
```

**실무 활용 시나리오:**
*   **코드 가독성 향상:** `JOIN` 조건 컬럼명이 양쪽 테이블에서 동일하고, 조건이 단순 동등 비교일 때 쿼리를 더 깔끔하게 작성할 수 있습니다.

### 1.10. `NATURAL JOIN` (참고): 자동 `JOIN` 조건

`NATURAL JOIN`은 두 테이블에서 이름과 데이터 타입이 동일한 모든 컬럼을 자동으로 찾아 `JOIN` 조건으로 사용합니다. `ON` 절이나 `USING` 절을 명시할 필요가 없어 매우 간결합니다.

**주의:** `NATURAL JOIN`은 편리하지만, 예상치 못한 컬럼이 `JOIN` 조건으로 사용될 수 있어 실무에서는 잘 사용되지 않습니다. 예를 들어, `last_update_date`와 같이 여러 테이블에 공통적으로 존재하지만 `JOIN` 조건으로 사용되어서는 안 되는 컬럼이 있다면 의도치 않은 결과를 초래할 수 있습니다. 명시적인 `ON` 절이나 `USING` 절을 사용하는 것이 더 안전하고 권장됩니다.

```sql
-- employees와 departments 테이블에 department_id 컬럼이 동일하게 존재할 때
SELECT e.first_name, d.department_name
FROM employees e
NATURAL JOIN departments d;
```

## 2. 데이터 집합 결합: `UNION`과 `UNION ALL`

`UNION`과 `UNION ALL`은 두 개 이상의 `SELECT` 문의 결과를 하나의 결과 집합으로 결합할 때 사용합니다. `JOIN`이 컬럼을 옆으로 연결하는 (수평적 결합) 반면, `UNION`은 로우를 아래로 연결하는 (수직적 결합) 방식입니다.

두 연산자 모두 `SELECT` 문의 결과를 결합하지만, 중복된 로우를 처리하는 방식에서 큰 차이가 있습니다.

**실무적 관점:** 서로 다른 테이블에 저장된 유사한 구조의 데이터를 통합하거나, 여러 기간의 데이터를 합쳐 분석할 때 매우 유용합니다. 특히 데이터 웨어하우스에서 여러 소스의 데이터를 통합하여 리포팅할 때 필수적으로 사용됩니다.

- **다뤄볼 테이블**
    ```plaintext
    sales_2022             sales_2023

    | ProductID | Amount | | ProductID | Amount | 
    |-----------|--------| |-----------|--------|
    | 1         | 100    | | 1         | 100    |
    | 2         | 200    | | 3         | 200    |
    | 3         | 300    | | 5         | 300    |
    | 4         | 400    | | 6         | 400    |
    ```

### 2.1. `UNION`

*   **`UNION`:** 두 `SELECT` 문의 결과를 결합하고, **중복된 로우를 자동으로 제거**합니다. 중복 제거를 위해 내부적으로 정렬(Sort) 작업을 수행하므로 `UNION ALL`보다 성능상 불리할 수 있습니다.


```sql
-- UNION: 중복 제거 (product_id와 amount가 모두 동일한 로우는 하나만 남음)
SELECT product_id, amount FROM sales_2022
UNION
SELECT product_id, amount FROM sales_2023;
```
- **UNION 으로 합친 결과**
    ```plaintext
    | ProductID | Amount |
    |-----------|--------|
    | 1         | 100    |
    | 2         | 200    |
    | 3         | 300    |
    | 4         | 400    |
    | 5         | 500    |
    | 6         | 600    |
    ```

**필수 주의사항: `UNION` 연산의 조건**
`UNION` 연산에 참여하는 `SELECT` 문들은 다음 조건을 **반드시** 만족해야 합니다.
*   선택하는 컬럼의 **개수**가 동일해야 합니다.
*   각 컬럼의 **데이터 타입**이 호환 가능해야 합니다. (예: `INT`와 `DECIMAL`은 호환되지만, `INT`와 `VARCHAR`는 호환되지 않을 수 있음)
*   컬럼의 **순서**가 일치해야 합니다. (첫 번째 `SELECT` 문의 컬럼 순서가 기준이 됩니다.)

### 2.2. `UNION ALL`

*   **`UNION ALL`:** 두 `SELECT` 문의 결과를 결합하고, **중복된 로우를 제거하지 않고 모두 포함**합니다. 중복 제거 과정이 없으므로 `UNION`보다 빠르고 효율적입니다.

```sql
-- UNION ALL: 중복 포함 (product_id와 amount가 모두 동일한 로우도 모두 포함)
SELECT product_id, amount FROM sales_2022
UNION ALL
SELECT product_id, amount FROM sales_2023;
```
- **UNION ALL 으로 합친 결과**
    ```plaintext
    | ProductID | Amount |
    |-----------|--------|
    | 1         | 100    |
    | 2         | 200    |
    | 3         | 300    |
    | 4         | 400    |
    | 1         | 100    |
    | 3         | 300    |
    | 5         | 500    |
    | 6         | 600    |
    ```

**실무 팁: `UNION` vs `UNION ALL` 선택 기준**
*   **성능이 중요하고 중복 허용 시:** `UNION ALL`을 사용합니다. 대부분의 데이터 분석 시나리오에서는 중복을 제거할 필요가 없거나, 애플리케이션 레벨에서 처리하는 것이 더 효율적일 수 있습니다.
*   **정확히 고유한 로우만 필요할 때:** `UNION`을 사용합니다. 하지만 대규모 데이터셋에서는 성능 저하를 인지하고 사용해야 합니다.
*   **`ORDER BY` 적용:** `UNION` 또는 `UNION ALL`로 결합된 결과에 `ORDER BY`를 적용할 때는 마지막 `SELECT` 문 뒤에 한 번만 작성합니다. 이때 `ORDER BY`는 첫 번째 `SELECT` 문의 컬럼 이름이나 별칭을 사용해야 합니다.

### 2.3. 복잡한 `UNION` 활용: 다양한 데이터 통합 시나리오

`UNION`은 서로 다른 구조의 데이터를 통합하거나, 여러 소스에서 가져온 데이터를 하나의 보고서 형태로 만들 때 유용합니다.

```sql
-- 예시 1: 직원과 고객의 연락처 정보를 통합하여 조회
-- (각 데이터 소스의 유형을 구분하는 컬럼 추가)
SELECT employee_id AS id, first_name, last_name, email, phone_number, 'Employee' AS type
FROM employees
UNION ALL
SELECT customer_id AS id, first_name, last_name, email, phone_number, 'Customer' AS type
FROM customers
ORDER BY type, last_name, first_name;

-- 예시 2: 서로 다른 이벤트 로그 테이블 통합 (로그 분석)
-- (각 로그 테이블의 컬럼명이 다르더라도 AS를 사용하여 통일)
SELECT log_time AS event_time, user_id, 'Login' AS event_type, ip_address AS detail
FROM login_logs
UNION ALL
SELECT purchase_time AS event_time, customer_id AS user_id, 'Purchase' AS event_type, product_name AS detail
FROM purchase_logs
UNION ALL
SELECT click_time AS event_time, visitor_id AS user_id, 'Click' AS event_type, page_url AS detail
FROM click_logs
ORDER BY event_time;
```

**실무 팁: `UNION`을 활용한 데이터 웨어하우스 ETL**
*   **증분 로딩 (Incremental Loading):** 매일 또는 매시간 발생하는 새로운 데이터를 기존 데이터에 `UNION ALL`로 추가하여 데이터 웨어하우스를 업데이트하는 데 사용됩니다.
*   **데이터 정규화 및 표준화:** 여러 시스템에서 수집된 비표준화된 데이터를 `UNION`하기 전에 `CAST`, `REPLACE`, `CASE` 등의 함수를 사용하여 데이터 타입을 맞추고 값을 표준화하는 전처리 과정이 필수적입니다.

### 2.4. `INTERSECT` 및 `EXCEPT` (MySQL 대체): 집합 연산의 구현

다른 SQL 데이터베이스(예: PostgreSQL, SQL Server, Oracle)에서는 `INTERSECT` (교집합)와 `EXCEPT` (차집합) 연산자를 직접 지원합니다. MySQL은 이들을 직접 지원하지 않지만, `JOIN`이나 `NOT EXISTS` 서브쿼리를 사용하여 동일한 결과를 얻을 수 있습니다.

#### 2.4.1. `INTERSECT` (교집합) 대체: 두 집합에 모두 존재하는 요소 찾기

`INTERSECT`는 두 `SELECT` 문의 결과 중 공통된 로우만 반환합니다. MySQL에서는 `INNER JOIN` 또는 `EXISTS` 서브쿼리를 사용하여 구현할 수 있습니다.

```sql
-- MySQL 미지원
SELECT ProductID, Amount FROM Sales_2022
INTERSECT
SELECT ProductID, Amount FROM Sales_2023;
```
- **INTERSECT 으로 합친 결과**
    ```plaintext
    | ProductID | Amount |
    |-----------|--------|
    | 1         | 100    |
    | 3         | 300    |
    ```

*   **`INNER JOIN`을 이용한 대체:** 가장 일반적이고 성능이 좋은 방법입니다.
    ```sql
    -- sales_2022와 sales_2023에 모두 존재하는 product_id 조회
    SELECT s22.product_id FROM sales_2022 s22
    INNER JOIN sales_2023 s23 ON s22.product_id = s23.product_id;
    ```

*   **`EXISTS` 서브쿼리를 이용한 대체:** 서브쿼리의 존재 여부로 필터링합니다.
    ```sql
    -- sales_2022에 있으면서 sales_2023에도 존재하는 product_id 조회
    SELECT product_id FROM sales_2022 s22
    WHERE EXISTS (SELECT 1 FROM sales_2023 s23 WHERE s22.product_id = s23.product_id);
    ```

**실무 활용 시나리오:**
*   **공통 고객/상품 식별:** 특정 기간 동안 두 번 이상 구매한 고객, 두 캠페인에 모두 반응한 사용자 등 여러 조건에 공통적으로 해당하는 대상을 찾을 때.
*   **데이터 일관성 검증:** 두 데이터셋 간의 일치하는 레코드를 확인하여 데이터 정합성을 검증할 때.

#### 2.4.2. `EXCEPT` (차집합) 대체: 한 집합에만 존재하는 요소 찾기

`EXCEPT` (또는 `MINUS` - Oracle)는 첫 번째 `SELECT` 문의 결과 중 두 번째 `SELECT` 문에는 없는 로우만 반환합니다. MySQL에서는 `LEFT JOIN`과 `IS NULL` 또는 `NOT EXISTS` 서브쿼리를 사용하여 구현할 수 있습니다.

```sql
-- MySQL 미지원
SELECT ProductID, Amount FROM Sales_2022
EXCEPT
SELECT ProductID, Amount FROM Sales_2023;
```
- **EXCEPT 으로 합친 결과**
    ```plaintext
    | ProductID | Amount |
    |-----------|--------|
    | 2         | 200    |
    | 4         | 400    |
    ```

*   **`LEFT JOIN`과 `IS NULL`을 이용한 대체:** 가장 일반적이고 직관적인 방법입니다.
    ```sql
    -- sales_2022에는 있지만 sales_2023에는 없는 product_id 조회
    SELECT s22.product_id FROM sales_2022 s22
    LEFT JOIN sales_2023 s23 ON s22.product_id = s23.product_id
    WHERE s23.product_id IS NULL;
    ```

*   **`NOT EXISTS` 서브쿼리를 이용한 대체:** 서브쿼리의 비존재 여부로 필터링합니다.
    ```sql
    -- sales_2022에는 있지만 sales_2023에는 없는 product_id 조회
    SELECT product_id FROM sales_2022 s22
    WHERE NOT EXISTS (SELECT 1 FROM sales_2023 s23 WHERE s22.product_id = s23.product_id);
    ```

**실무 활용 시나리오:**
*   **신규/이탈 고객/상품 식별:** 특정 기간에 새로 유입된 고객, 이탈한 고객, 단종된 상품 등 변화를 추적할 때.
*   **데이터 불일치 분석:** 두 데이터셋 간의 차이점을 파악하여 데이터 누락이나 오류를 찾아낼 때.

**실무 팁: 집합 연산 성능 고려사항**
*   **인덱스 활용:** `JOIN`과 마찬가지로 집합 연산에 사용되는 컬럼에도 인덱스가 있다면 성능에 큰 도움이 됩니다.
*   **데이터 타입 일치:** `UNION`과 마찬가지로 `INTERSECT` 및 `EXCEPT` 대체 쿼리에서도 비교 대상 컬럼의 데이터 타입이 일치해야 합니다.
*   **`EXPLAIN` 활용:** 복잡한 집합 연산 쿼리의 성능 문제를 진단하기 위해 `EXPLAIN` 명령을 사용하여 실행 계획을 분석하는 것이 중요합니다.
*   **대규모 데이터셋 처리:** 매우 큰 데이터셋에 대한 집합 연산은 많은 자원을 소모할 수 있습니다. 필요한 경우 임시 테이블, CTE, 또는 데이터 웨어하우스의 최적화된 기능을 활용하는 것을 고려해야 합니다.