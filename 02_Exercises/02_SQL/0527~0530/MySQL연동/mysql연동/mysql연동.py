

#pip install pymysql 
import pymysql

#접속객체를 가져온다. 
conn = pymysql.connect( host = 'localhost',  #서버아이피
                         user = 'user01',    #계정아이디
                         password = '1234' , #패스워드
                         db = 'mydb',        #db명
                         port=3306)          #디비포트번호 
curs = conn.cursor()

sql = "SELECT * FROM guestbook " # 실행 할 쿼리문 입력
curs.execute(sql) # 쿼리문 실행 - 실행후 fetch하기 

rows = curs.fetchall() # 데이터 패치- tuple타입으로 온다 

for row in rows :
     print(type(row), row)


print("한개만 가져오기------")
curs.execute(sql)
row = curs.fetchone() #첫번째 레코드 하나만 가져온다 
print(row)

print("세개만 가져오기------")
curs.execute(sql)
rows = curs.fetchmany(3) #새개의 레코드를 가져온다 
for row in rows :
     print(type(row), row)


#Dict 타입으로 가져오기 - tuple타입으로 가져오면 인덱스 1,2,3,..
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "SELECT * FROM guestbook" # 실행 할 쿼리문 입력
curs.execute(sql) # 쿼리문 실행
rows = curs.fetchall() # 데이터 패치
for row in rows :
     print(row['title'], row['contents'], row['wdate'])

conn.close()


