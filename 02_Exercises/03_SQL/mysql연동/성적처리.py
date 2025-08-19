import pymysql

conn = pymysql.connect(host = 'localhost', 
                        user = 'user01', 
                        password = '1234' ,
                        db = 'mydb', 
                        port=3306)

curs = conn.cursor(pymysql.cursors.DictCursor)

def insert():
    name = input("이름 : ")
    kor =input("국어 : ")
    eng = input("영어 : ")
    mat = input("수학 : ")
    # ==== insert example ====
    sql = """
            insert into tb_score(sname, kor,eng,mat, 
            regdate)
            values (%s, %s, %s,%s, now())
        """
    curs.execute(sql, (name, kor, eng, mat))
    conn.commit() #반드시 해야 한다 


def update():
    id = input("수정할 아이디는?")
    name = input("이름 : ")
    kor =input("국어 : ")
    eng = input("영어 : ")
    mat = input("수학 : ")
    
    # ==== update OR delete example ====
    sql = """update tb_score
            set sname = %s
            , kor=%s
            , eng=%s
            , mat=%s
            where id=%s
        """
    curs.execute(sql, (name, kor, eng, mat, id))
    conn.commit() 

def delete():
    id= input("삭제할 아이디는 ")

    sql = "delete from tb_score where id=%s"
    curs.execute(sql, id)

def output():   
    sql = """
    SELECT id, sname, kor, eng, mat
    , (kor+eng+mat) total
    , (kor+eng+mat)/3 average 
    FROM tb_score
    """ # 실행 할 쿼리문 입력
    print(sql)
    curs.execute(sql) # 쿼리문 실행

    rows = curs.fetchall() # 데이터 패치

    for row in rows :
        print(row['id'], row['sname'], row['kor'], 
              row['eng'], row['mat'], row['total'], 
              row['average'])

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