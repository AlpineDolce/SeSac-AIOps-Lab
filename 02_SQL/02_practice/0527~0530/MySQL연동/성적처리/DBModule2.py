import pymysql 

class Database:
    
    db = pymysql.connect( host='localhost', user='user01', password='1234',
          db='mydb', port=3306 )
    cursor = db.cursor(pymysql.cursors.DictCursor)
    
    @classmethod
    #insert, update, delete
    def execute(cls, query, args=()):
        print(args)
        cls.cursor.execute(query, args)
        cls.db.commit() 

    #데이터 딱 한개만 가져오기
    @classmethod
    def executeOne(cls, query, args=()):
        cls.cursor.execute(query, args)
        row = cls.cursor.fetchone() 
        return row #결과를 반환해야 한다. 첫번째 레코드값 하나만 가져간다 

    #데이터 여러개 가져오기
    @classmethod
    def executeAll(cls, query, args=()):
        cls.cursor.execute(query, args)
        rows = cls.cursor.fetchall() 
        return rows #결과를 반환해야 한다. 첫번째 레코드값 하나만 가져간다 
    
    @classmethod
    def close(cls):
        if cls.db.open:
            cls.db.close()