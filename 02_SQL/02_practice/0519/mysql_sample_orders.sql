
-- 1. customers 테이블 생성 및 데이터 삽입
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

INSERT INTO customers (name, email) VALUES
('Alice', 'alice@example.com'),
('Bob', 'bob@example.com'),
('Charlie', 'charlie@example.com'),
('Diana', 'diana@example.com'),
('Eve', 'eve@example.com');

-- 2. orders 테이블 생성 및 데이터 삽입
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    status VARCHAR(20),
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

INSERT INTO orders (customer_id, order_date, status, total_amount) VALUES
(1, '2024-05-01', 'Pending', 120.50),
(2, '2024-05-03', 'Completed', 300.00),
(3, '2024-05-05', 'Shipped', 89.99),
(1, '2024-05-10', 'Cancelled', 49.00),
(4, '2024-05-12', 'Completed', 199.90),
(2, '2024-05-15', 'Pending', 150.75),
(5, '2024-05-17', 'Shipped', 80.00);

-- 3. order_items 테이블 생성 및 데이터 삽입
DROP TABLE IF EXISTS order_items;
CREATE TABLE order_items (
    item_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    product_name VARCHAR(100),
    quantity INT,
    price DECIMAL(10,2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

INSERT INTO order_items (order_id, product_name, quantity, price) VALUES
(1, 'USB Cable', 2, 15.00),
(1, 'Mouse', 1, 25.50),
(2, 'Keyboard', 1, 100.00),
(2, 'Monitor', 2, 100.00),
(3, 'Headphones', 1, 89.99),
(4, 'Phone Case', 1, 49.00),
(5, 'Webcam', 1, 99.95),
(5, 'Tripod', 1, 99.95),
(6, 'Charger', 3, 50.25),
(7, 'Laptop Stand', 1, 80.00);
