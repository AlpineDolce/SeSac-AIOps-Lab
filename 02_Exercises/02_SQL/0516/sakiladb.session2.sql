-- 1. film 테이블에서 영화 제목과 대여 요금을 조회하시오.
SELECT title, rental_rate
FROM film;

-- 2. actor 테이블에서 이름이 'JOHN'인 배우를 조회하시오.
SELECT *
FROM actor
WHERE first_name = 'JOHN';

-- 3. category 테이블의 모든 카테고리 이름을 조회하시오.
SELECT * 
FROM category;

-- 4. film 테이블에서 rental_rate가 3.99보다 큰 영화의 제목과 요금을 조회하시오.
SELECT title, rental_rate
FROM film
WHERE rental_rate > 3.99;

-- 5. customer 테이블에서 이메일 주소에 'gmail'이 포함된 고객을 조회하시오.
SELECT *
FROM customer
WHERE email LIKE '%gmail%';

-- 6. film 테이블에서 길이가 120분 이하인 영화의 제목과 길이를 조회하시오.
SELECT title, length
FROM film
WHERE length <= 120;

-- 7. language 테이블에서 언어 이름이 'English'인 행을 조회하시오.
SELECT *
FROM language
WHERE name = 'English';

-- 8. actor 테이블에서 성(last_name)이 'SMITH'인 배우의 이름과 성을 조회하시오.
SELECT first_name, last_name
FROM actor
WHERE last_name = 'SMITH';

-- 9. customer 테이블에서 first_name이 'A'로 시작하는 고객을 조회하시오.
SELECT *
FROM customer
WHERE first_name LIKE 'A%';

-- 10. film 테이블에서 2006년에 개봉한 영화만 조회하시오.
SELECT *
FROM film
WHERE release_year = 2006;

-- 11. 배우 번호가 10, 21, 34, 56, 87, 89, 90인 사람들 정보만 출력하라
SELECT *
FROM actor
WHERE actor_id IN (10, 21, 34, 56, 87, 89, 90);

-- 12. customer 테이블에서 store_id = 1이고, customer_id가 562, 580, 470, 363, 364인 고객을 찾으시오.
SELECT *
FROM customer
WHERE store_id = 1 AND customer_id IN (562, 580, 470, 363, 364);
