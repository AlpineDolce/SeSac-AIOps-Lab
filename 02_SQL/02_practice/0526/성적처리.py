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

# import pymysql
# from datetime import datetime
# def Connection(func):
#     def inner(*ags,**kwargs):
#         conn=pymysql.connect(
#         host='localhost',
#         user='root',
#         password='1234',
#         db='mydb',
#         port=3306)
#         cursor=conn.cursor(pymysql.cursors.DictCursor)
#         func(cursor,*ags,**kwargs)
#         conn.commit()
#         conn.close()
#         return 
#     return inner

# class student():
#     sname=None
#     korScore=None
#     engScore=None
#     mathScore=None
#     regdate=None
#     @Connection
#     @classmethod
#     def add(cls,conn):
#         cls.sname=input("Enter your name: ")
#         cls.korScore=input("Enter your score: ")
#         cls.engScore=input("Enter your score: ")
#         cls.mathScore=input("Enter your score: ")
#         cls.regdate=datetime
#         conn.execute(
#             """insert into score(sname,korScore,engScore,mathScore,regdate)
#             values(%s,%s,%s,%s,%s)""",
#             (cls.sname,cls.korScore,cls.engScore,cls.mathScore,cls.regdate))
#         print("Student added successfully!")
    
#     @Connection
#     def modify(cls,conn):
#         cls.sname=input("Enter your name: ")
#         cls.korScore=input("Enter your score: ")
#         cls.engScore=input("Enter your score: ")
#         cls.mathScore=input("Enter your score: ")
#         sql="""update score set sname=%s
#         , korScore=%s, engScore=%s, mathScore=%s 
#         where sname=%s"""
#         conn.execute(sql,(cls.sname,cls.KorScore,cls.engScore,cls.mathScore,cls.sname))
#     @Connection
#     def delete(conn):
#         cls.sname=input("Enter your name: ")
#         sql="delete from score where sname=%s"
#         name=input("Enter your name to delete: ")
#         conn.execute(sql,(name))
#     def select(conn):
#         id=input("Enter see your id: ")
#         conn.execute("select * from tb_score where sname=%s", (id))
#         row=conn.fetchone()
#         print(row)
    
#     @Connection
#     def allprint(conn):
#         sql='select * from tb_score'
#         conn.exercute(sql)

# class manager():
#     def main():
#         funList=[None,None,cls.modify,cls.delete,cls.select,cls.allprint]
#         while True:
#             print("1. Add Student")
#             print("2. Modify Student")
#             print("3. Delete Student")
#             print("4. Select Student")
#             print("5. Print All Students")
#             print("6. Exit")
#             choice = int(input("Enter your choice: "))
#             if choice == 1:
#                 stu=cls()
#                 stu.add()
#             elif choice >= 2 and choice <= 5:
#                 funList[choice-1]()
#             elif choice == 6:
#                 return
#             else:
#                 print("Invalid choice, please try again.")
# manager.main()


# # 리펙토리 버전
# import pymysql
# from datetime import datetime

# # DB 연결 데코레이터
# def Connection(func):
#     def inner(*args, **kwargs):
#         conn = pymysql.connect(
#             host='localhost',
#             user='root',
#             password='1234',
#             db='mydb',
#             port=3306
#         )
#         cursor = conn.cursor(pymysql.cursors.DictCursor)
#         try:
#             result = func(*args, cursor=cursor, **kwargs)
#             conn.commit()
#             return result
#         finally:
#             conn.close()
#     return inner


# # 학생 클래스
# class Student:
#     def __init__(self):
#         self.sname = input("Enter name: ")
#         self.kor = input("Enter Korean score: ")
#         self.eng = input("Enter English score: ")
#         self.math = input("Enter Math score: ")
#         self.regdate = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#     @Connection
#     def add(self, cursor):
#         sql = """
#         INSERT INTO score (sname, korScore, engScore, mathScore, regdate)
#         VALUES (%s, %s, %s, %s, %s)
#         """
#         cursor.execute(sql, (self.sname, self.kor, self.eng, self.math, self.regdate))
#         print("✅ Student added successfully.")

#     @staticmethod
#     @Connection
#     def modify(cursor):
#         name = input("Enter student name to modify: ")
#         new_name = input("New name: ")
#         kor = input("New Korean score: ")
#         eng = input("New English score: ")
#         math = input("New Math score: ")
#         sql = """
#         UPDATE score SET sname=%s, korScore=%s, engScore=%s, mathScore=%s 
#         WHERE sname=%s
#         """
#         cursor.execute(sql, (new_name, kor, eng, math, name))
#         print("✅ Student modified successfully.")

#     @staticmethod
#     @Connection
#     def delete(cursor):
#         name = input("Enter name to delete: ")
#         sql = "DELETE FROM score WHERE sname=%s"
#         cursor.execute(sql, (name,))
#         print("🗑️ Student deleted successfully.")

#     @staticmethod
#     @Connection
#     def select(cursor):
#         name = input("Enter name to search: ")
#         sql = "SELECT * FROM score WHERE sname=%s"
#         cursor.execute(sql, (name,))
#         result = cursor.fetchone()
#         if result:
#             print("🎯 Student Info:", result)
#         else:
#             print("❌ No student found with that name.")

#     @staticmethod
#     @Connection
#     def allprint(cursor):
#         sql = "SELECT * FROM score"
#         cursor.execute(sql)
#         rows = cursor.fetchall()
#         print("📋 All Students:")
#         for row in rows:
#             print(row)


# # 매니저 클래스: 메뉴 제어
# class Manager:
#     @staticmethod
#     def main():
#         while True:
#             print("\n=== Student Score Manager ===")
#             print("1. Add Student")
#             print("2. Modify Student")
#             print("3. Delete Student")
#             print("4. Select Student")
#             print("5. Print All Students")
#             print("6. Exit")
#             choice = input("Enter your choice: ")

#             if choice == '1':
#                 stu = Student()
#                 stu.add()
#             elif choice == '2':
#                 Student.modify()
#             elif choice == '3':
#                 Student.delete()
#             elif choice == '4':
#                 Student.select()
#             elif choice == '5':
#                 Student.allprint()
#             elif choice == '6':
#                 print("👋 Exiting program.")
#                 break
#             else:
#                 print("❗ Invalid choice. Try again.")


# if __name__ == "__main__":
#     Manager.main()
