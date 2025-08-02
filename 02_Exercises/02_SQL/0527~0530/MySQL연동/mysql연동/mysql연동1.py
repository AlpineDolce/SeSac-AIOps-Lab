#mysql연동1.py
import pymysql 

conn = pymysql.connect(host="localhost", 
                       user="user01",
                       password="1234",
                       db="mydb",
                       port=3306)

curs = conn.cursor() #커서를 마늘고 

sql = '''
    insert into guestbook(title, contents, writer, wdate)
    values(%s, %s, %s, now())
'''
#dml만 커밋대상(data manipulation), insert, delete, update만
# () tuple타입  
curs.execute(sql, ('제목6', '내용6', '작성자6'))

conn.commit()#커서가 아니라 연결객체에다 커밋을 해야 한다 

#트랜잭션 : 하나이상의 쿼리가 하나의 목적을 위해 움직일때 
#          하나가 오류가 생기면 다른 하나도 취소 시켜야 한다 
# commit : 확정
# rollback : 되돌리기 



#커서를 dict 타입으로 가져오기 
sql = "select * from guestbook"
curs = conn.cursor(pymysql.cursors.DictCursor)
curs.execute(sql)

rows = curs.fetchall() 
for row in rows:
    print(row['title'], row['contents'], row['writer'], row['wdate'])
conn.close()


