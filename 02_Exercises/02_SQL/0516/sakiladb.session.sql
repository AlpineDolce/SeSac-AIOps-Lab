-- [기초 조회]
-- 1. 모든 배우(Actor)의 이름과 성을 조회하시오.
SELECT * FROM actor;

-- 2. 배우 테이블에서 성(last_name)이 ‘DAVIS’인 사람을 모두 찾으시오.
SELECT * FROM actor WHERE last_name = 'DAVIS';

-- 3. 고객(Customer)의 이메일 목록을 알파벳 순서로 조회하시오.
SELECT * FROM customer ORDER BY email;

-- 4. 영화(film)의 제목과 대여 요금(rental_rate)을 조회하시오.
SELECT title, rental_rate FROM film;

-- 5. 고객(Customer)의 이름, 성, 이메일을 각각 출력하시오.
SELECT FIRST_NAME, LAST_NAME, EMAIL FROM CUSTOMER;

-- 6. 카테고리(category)별 이름과 ID를 출력하시오.
SELECT name, category_id FROM category;


-- [조건과 정렬]
-- 7. 길이가 180분 이상인 영화 제목을 조회하시오.
SELECT title FROM film WHERE length >= 180;

-- 8. ‘Comedy’ 카테고리에 속한 영화 제목을 모두 조회하시오.
SELECT title FROM film
JOIN film_category ON film.film_id = film_category.film_id;

-- 9. 대여 요금이 4.99 이상인 영화 중에서 제목(title)과 요금(rental_rate)을 내림차순 정렬하시오.
SELECT title, rental_rate FROM film WHERE rental_rate >= 4.99 ORDER BY rental_rate DESC;

-- 10. 대여(rental) 중 2005년에 이루어진 기록만 조회하시오.
SELECT * FROM rental WHERE YEAR(rental_date) = 2005;

-- 11. 고객 중 이름이 'S'로 시작하는 고객의 이름을 조회하시오.
SELECT first_name FROM customer WHERE first_name LIKE 'S%';

-- 12. 배우(actor) 테이블에서 이름이 5글자인 배우만 찾으시오.
SELECT * FROM actor WHERE LENGTH(first_name) = 5;

