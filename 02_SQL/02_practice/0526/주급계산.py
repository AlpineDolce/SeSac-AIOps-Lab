# 주급 계산
# 아이디, 이름, 근무시간, 시간당 급여액 
# 연장수당 - case when 문 근무시간 > 20
# 테이블 각자 만들기

import pymysql
from pymysql import cursors

conn = pymysql.connect(
    host='localhost',
    user='root',
    password='1234',
    db='mydb',
    port=3306
)

curs = conn.cursor(pymysql.cursors.DictCursor)
def create_table():
    # ✅ 테이블 생성 함수: 
    # weekly_pay 테이블이 존재하지 않으면 새로 생성
    sql = """
    CREATE TABLE IF NOT EXISTS weekly_pay (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50) NOT NULL,
        work_hours INT NOT NULL,
        hourly_wage DECIMAL(10, 2) NOT NULL
    )
    """
    curs.execute(sql)
    conn.commit()
def insert_data(name, work_hours, hourly_wage):
    # ✅ 데이터 삽입 함수
    # 이름, 근무 시간, 시급을 테이블에 추가
    sql = """
    INSERT INTO weekly_pay (name, work_hours, hourly_wage)
    VALUES (%s, %s, %s)
    """
    curs.execute(sql, (name, work_hours, hourly_wage))
    conn.commit()
def calculate_weekly_pay():
    # ✅ 주급 계산 함수
    # 근무시간과 시급을 바탕으로 총급여와 초과근무 수당 계산 및 출력
    sql = """
    SELECT id, name, work_hours, hourly_wage,
           (work_hours * hourly_wage) AS total_pay,
           CASE 
               WHEN work_hours > 20 THEN (work_hours - 20) * hourly_wage * 1.5
               ELSE 0 
           END AS overtime_pay
    FROM weekly_pay
    """
    curs.execute(sql)
    result = curs.fetchall()
    for row in result:
        print(f"ID: {row['id']}, Name: {row['name']}, Total Pay: {row['total_pay']}, Overtime Pay: {row['overtime_pay']}")
def main():
    create_table()
    while True:
        name = input("Enter name (or 'exit' to quit): ")
        if name.lower() == 'exit':
            break
        work_hours = int(input("Enter work hours: "))
        hourly_wage = float(input("Enter hourly wage: "))
        insert_data(name, work_hours, hourly_wage)
    
    print("\nWeekly Pay Calculation:")
    calculate_weekly_pay()
if __name__ == "__main__":
    main()
