from DBModule2 import Database as db

class ScoreData:
    def __init__(self, sname="", kor=0, eng=09, mat=0, total=0, average=0):
        self.sname = sname
        self.kor=kor
        self.eng = eng 
        self.mat = mat 
        self.total = total
        self.average=average 

    def output(self):
        print(self.sname, self.kor, self.eng, self.mat, self.total, self.average)

class ScoreManager:

    def insert():
        s = ScoreData()
        s.sname = input("이름 : ")
        s.kor =input("국어 : ")
        s.eng = input("영어 : ")
        s.mat = input("수학 : ")
        # ==== insert example ====
        sql = """
                insert into tb_score(sname, kor,eng,mat, 
                regdate)
                values (%s, %s, %s,%s, now())
            """
        db.execute(sql, (s.sname, s.kor, s.eng, s.mat))


    def update():
        s = ScoreData()
        s.id = input("수정할 아이디는?")
        s.sname = input("이름 : ")
        s.kor =input("국어 : ")
        s.eng = input("영어 : ")
        s.mat = input("수학 : ")
        
        # ==== update OR delete example ====
        sql = """update tb_score
                set sname = %s
                , kor=%s
                , eng=%s
                , mat=%s
                where id=%s
            """
        db.execute(sql, (s.name, s.kor, s.eng, s.mat, s.id))

    def delete():
        id= input("삭제할 아이디는 ")
        sql = "delete from tb_score where id=%s"
        db.execute(sql, (id))

    def output():   
        #윈도우 함수 mysql8부터 지원함 
        sql = """
        SELECT id, sname, kor, eng, mat
        , (kor+eng+mat) total
        , (kor+eng+mat)/3 average
        , ROW_NUMBER() OVER (ORDER BY (kor + eng + mat) DESC) AS row_id
        FROM tb_score
        """ # 실행 할 쿼리문 입력
        print(sql)
        rows = db.executeAll(sql)

        for row in rows :
            print(row['row_id'], row['id'], row['sname'], row['kor'], 
                row['eng'], row['mat'], row['total'], 
                row['average'])

    def end():
        db.close()

   

    def main(self):  
        try:
            while(True):
                sel = input("1.목록 2.추가 3.수정 4.삭제 0.종료 : ")
                if sel=="1" :
                    page = int(input("page?"))
                    self.output2(page)
                elif sel=="2":
                    self.insert()
                elif sel =="3":
                    self.update()
                elif sel == "4":
                    self.delete()
                elif sel=="0":
                    break
        except Exception as e:
            print(e)
        finally:
            self.end()

if __name__ =="__main__":
    sm = ScoreManager()
    sm.main()
     