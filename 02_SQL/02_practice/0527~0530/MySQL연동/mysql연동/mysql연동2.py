import pymysql

conn = pymysql.connect(host = 'localhost', user = 'user01', password = '1234' ,db = 'mydb', port=3306)

curs = conn.cursor()

# ==== insert example ====
sql = """
        insert into guestbook(title, contents, writer, wdate)
        values (%s, %s, %s, now())
    """
#curs.execute(sql, ('제목입니다.', '내용입니다.', '작성자'))
curs.execute(sql, ('제목입니다.', '내용입니다.', '작성자'))
conn.commit()

sql = "SELECT * FROM guestbook" # 실행 할 쿼리문 입력
curs.execute(sql) # 쿼리문 실행

rows = curs.fetchall() # 데이터 패치

for i in rows :
     print(i)

conn.close()


