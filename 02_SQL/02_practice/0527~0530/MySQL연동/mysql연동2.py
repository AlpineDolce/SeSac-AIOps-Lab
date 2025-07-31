import pymysql 
conn = pymysql.connect(
    host='localhost', # ip localhost=127.0.0.1 
    user='user02', #아이디 
    password='qwer1234', #패스워드
    db='mydb', #데이터베이스명 
    port=3306 ) #포트번호 : 기본-3306

cursor = conn.cursor(pymysql.cursors.DictCursor) 
ename = "SCOTT"
sql = "select empno, ename, sal from emp where ename='"+ename+"' "
print(sql)
cursor.execute(sql)
rows = cursor.fetchall() 
print("데이터개수", len(rows))
for row in rows:
    print(row["empno"], row["ename"], row["sal"])



#insert 는 ?

#max함수가 데이터가 한건도 없을때 null을 갖고 온다.
sql = "select ifnull(max(empno), 0)+1 id from emp"
cursor.execute(sql) 
row = cursor.fetchone()
print(row)

sql = """
    insert into emp(empno, ename, sal)
    values(%s, %s, %s) 
"""
cursor.execute( sql, (row["id"], '백승빈', 6000))
conn.commit() #연결객체 commit 반드시 해줘야 함 

conn.close() 