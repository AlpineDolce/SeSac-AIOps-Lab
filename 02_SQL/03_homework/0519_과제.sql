use w3schools;
-- https://www.w3schools.com/ 예제 이용

-- 1) 주문이 한 번도 없는 고객의 이름을 조회하기
SELECT CustomerName
FROM Customers
WHERE CustomerID NOT IN (
    SELECT CustomerID FROM Orders
);

-- 2) 가장 주문 건수가 많은 판매자 이름 구하기
SELECT E.LastName, E.FirstName
FROM Orders O
JOIN Employees E ON O.EmployeeID = E.EmployeeID
GROUP BY E.EmployeeID
ORDER BY COUNT(*) DESC
LIMIT 1;

-- 3) 판매건수가 5건 이상인 판매자 인원수
SELECT COUNT(*) AS NumOfTopSellers
FROM (
    SELECT EmployeeID
    FROM Orders
    GROUP BY EmployeeID
    HAVING COUNT(*) >= 5
) AS Sub;
