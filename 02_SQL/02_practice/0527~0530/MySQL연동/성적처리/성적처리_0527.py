from DBModule import Database 

class ScoreData:
    def __init__(self, sname="", kor=0, eng=0, mat=0,  
                 total=0, average=0):
        self.sname = sname 
        self.kor = kor 
        self.eng = eng
        self.mat = mat 
        self.total = total 
        self.average=average  
    
    def output(self):
        print(self.sname, self.kor, self.eng, self.mat , 
              self.total, self.average )
        
class ScoreManager :
    def append(self):
        s = ScoreData()
        s.sname = input("이름 : ")
        s.kor = input("국어 ")
        s.eng = input("영어 ")
        s.mat = input("수학 ")
        sql = '''
            insert into tb_score(sname, kor, eng, mat, regdate)
            values( %s, %s,%s,%s, now())
        '''
        db = Database()
        db.execute(sql, (s.sname, s.kor, s.eng, s.mat))
        db.close() 

    def output(self):
        sql = """
            select sname, kor, eng, mat 
            ,(kor+eng+mat) as total 
            ,(kor+eng+mat)/3 as average 
            from tb_score 
        """
        db=Database()
        rows = db.executeAll(sql)
        self.dataList = [] 
        for r in rows:
            s = ScoreData(r['sname'], r['kor'], r['eng'],
                          r['mat'], r['total'], r['average'])
            self.dataList.append(s)

        for s in self.dataList:
            s.output()

if __name__ =="__main__":
    sm = ScoreManager()
    sm.output()

"""
커넥션풀 -> 디비연결  데이터읽고쓰기  디비연결끊기 
        연결 과 끊기가 시간이 더 많이 걸린다. 
        디비연결자를 한 50개 만들어놓는다. conn 50개 만들어놓고 돌려쓰고
       연결자 50개를 디비접속할때마다 연결자를 50개씩
       시스템이 다운됨 
       직접설계해서 사용함 => 지원해주는 라이브러리들이 생김 

       이클래스는 반드시 싱글톤방식으로 만들어야 한다.
       싱글톤은 객체를 반드시 하나만 생성가능한 클래스 설계기법 
       생성자에서 못만들게 막고 @classmethod 라는 데코레이터를
       이용해서 객체를 생성했음 
       20년전에는 직접 만들어쓰다가 현재는 별도의 라이브러리를 
       갖다 쓴다.
           
"""