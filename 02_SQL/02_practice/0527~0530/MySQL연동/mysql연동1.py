#conda activate myenv1 
#pip install pymysql 
#mysql 8부터 문제 있을 경우 pip install cryptography 
#1.디비연결 
#2.데이터읽고쓰기
#3.연결끊기
"""
-- 계정만들기`
-- localhost - 루프백주소(127.0.0.1)
-- 'user02'@'localhost' 로컬서만 접근가능한 
-- 계정이다  
create user 'user02'@'localhost'
identified by 'qwer1234';
grant all privileges on mydb.* to 'user02'@'localhost'; 
-- grant 권한목록 on 디비명.테이블명 
"""
import pymysql 

conn = pymysql.connect(
    host='localhost', # ip localhost=127.0.0.1 
    user='user02', #아이디 
    password='qwer1234', #패스워드
    db='mydb', #데이터베이스명 
    port=3306 ) #포트번호 : 기본-3306

cursor = conn.cursor() #커서를 가져와야 한다 
#커서를 통해 
print("접속성공")
#데이터 가져오기 
sql = "select * from emp"
cursor.execute(sql) 
#select 쿼리를 실행하고나서 내부에 갖고 있다 
rows = cursor.fetchall() #데이터를 tuple타입으로 가져온다 
for row in rows:
    print( type(row), row)


print("하나만 가져오기")
sql = "select * from emp where empno=8001"
cursor.execute(sql) #다시 디비 가져오기 
row = cursor.fetchone()
print(row)

print("3개 가져오기 ")
sql = "select * from emp where empno<8000"
cursor.execute(sql)
rows = cursor.fetchmany(3) #앞에 3개만 
for row in rows:
    print(row)

#데이터를 tuple타입으로 가져오면서 인덱싱과 슬라이싱만 지원
#row["ename"]  -> 데이터를 가져올때 dict타입으로 가져오기 
cursor = conn.cursor(pymysql.cursors.DictCursor)
cursor.execute(sql)
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()
