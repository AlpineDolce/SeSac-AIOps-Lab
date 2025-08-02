import pymysql

conn = pymysql.connect(host = 'localhost', 
                        user = 'user01', 
                        password = '1234' ,
                        db = 'mydb', 
                        port=3306)

curs = conn.cursor(pymysql.cursors.DictCursor)

def insert():
    title = input("제목 : ")
    writer =input("작성자 : ")
    contents = input("내용 : ")
    # ==== insert example ====
    sql = """
            insert into guestbook(title, contents, writer, wdate)
            values (%s, %s, %s, now())
        """
    curs.execute(sql, (title, writer, contents))
    conn.commit() #반드시 해야 한다 

def update():
    id = input("수정할 아이디는?")
    title = input("수정할 제목 : ")
    writer =input("수정할 작성자 : ")
    contents = input("수정할 내용 : ")
    
    # ==== update OR delete example ====
    sql = """update guestbook
            set title = %s, contents=%s, writer=%s
            where id=%s
        """
    curs.execute(sql, (title, contents, writer, id))
    conn.commit() 

def delete():
    id= input("삭제할 아이디는 ")

    sql = "delete from guestbook where id=%s"
    curs.execute(sql, id)

def output():   
    sql = "SELECT * FROM guestbook" # 실행 할 쿼리문 입력
    curs.execute(sql) # 쿼리문 실행

    rows = curs.fetchall() # 데이터 패치

    for row in rows :
        print(row['id'], row['title'], row['writer'], row['contents'], row['wdate'])

def end():
    conn.close()


while(True):
    sel = input("1.목록 2.추가 3.수정 4.삭제 0.종료 : ")
    if sel=="1" :
        output()
    elif sel=="2":
        insert()
    elif sel =="3":
        update()
    elif sel == "4":
        delete()
    elif sel=="0":
        break