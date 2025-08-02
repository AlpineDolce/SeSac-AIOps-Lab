-- 과제
-- 1. customers table에서 나라가 Germany 인 나라의 정보 전체

SELECT * FROM Customers
WHERE Country = 'Germany';

-- 2. customers table에서 Austria, USA, Poland, Denmark에 사는 고객 리스트

SELECT * FROM Customers
WHERE Country IN ('Austria', 'USA', 'Poland', 'Denmark');

-- 3. 각자 나라별로 고객이 몇명씩 있는지 확인

SELECT Country, COUNT(*) AS CustomerCount
FROM Customers
GROUP BY Country;

-- 4. 나라별로 고객이 5명이 이상인 나라 목록

SELECT Country, COUNT(*) AS CustomerCount
FROM Customers
GROUP BY Country
HAVING COUNT(*) >= 5;

-- 5. 나라이름이 B로 시작하는 나라들의 전체 합

SELECT COUNT(*) AS TotalCustomersFromB
FROM Customers
WHERE Country LIKE 'B%';

-- 6. 나라는 UK / 도시명은 London에 있는 고객 이름 목록

SELECT CustomerName
FROM Customers
WHERE Country = 'UK' AND City = 'London';

-- 7. 주문날짜가 '1996-07-01' ~ '1996-09-30' 일까지의 주문아이디와 고객이름

SELECT Orders.OrderID, Customers.CustomerName
FROM Orders
JOIN Customers ON Orders.CustomerID = Customers.CustomerID
WHERE Orders.OrderDate BETWEEN '1996-07-01' AND '1996-09-30';

-- 8. 위의 7번 문제를 고객이름 오름차순으로 정렬하여 출력하기

SELECT Orders.OrderID, Customers.CustomerName
FROM Orders
JOIN Customers ON Orders.CustomerID = Customers.CustomerID
WHERE Orders.OrderDate BETWEEN '1996-07-01' AND '1996-09-30'
ORDER BY Customers.CustomerName ASC;


-- 9. 배달자가 Federal Shipping 인 경우의 상품명 가격 수량만 출력

SELECT Products.ProductName, Products.Price, OrderDetails.Quantity
FROM Orders
JOIN Shippers ON Orders.ShipperID = Shippers.ShipperID
JOIN OrderDetails ON Orders.OrderID = OrderDetails.OrderID
JOIN Products ON OrderDetails.ProductID = Products.ProductID
WHERE Shippers.ShipperName = 'Federal Shipping';


