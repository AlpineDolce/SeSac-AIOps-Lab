use world;

-- 1. 모든 도시의 이름과 인구를 조회하시오.
select c.name, c.population from city c ;

-- 2. 인구가 1,000만 이상인 도시를 조회하시오.
select * from city c where c.population >= 10000000;

-- 3. 국가(country) 중 대륙(Continent)이 'Asia'인 국가를 모두 조회하시오.
select * from country c where c.continent = 'Asia';

-- 4. 모든 나라의 이름, 인구, 대륙 조회
select * from countrylanguage cl where cl.language='Spanish';

-- 5. 인구순 정렬 (내림차순)
select c.Population from country c order by c.Population desc;

-- 6. 도시 이름에 'new'가 포함된 도시를 조회하시오 (대소문자 무시).
select * from city c where c.Name like '%new%';

-- 7. 인구가 가장 많은 5개 도시를 조회하시오.
select * from city c order by c.Population desc limit 0, 5;

-- 8. 아프리카(Africa) 대륙에 속한 국가의 이름과 인구를 조회하시오.
select c.Name, c.Population from country c where c.Continent = 'Africa';

-- 9. 독립 국가만 필터링 (IndepYear IS NOT NULL)
select * from country c where c.IndepYear is not null;

-- 10. GDP(GNP)가 높은 상위 10개 국가의 이름과 GNP를 조회하시오.
select c.name, c.GNP from country c order by c.GNP limit 0,10;
