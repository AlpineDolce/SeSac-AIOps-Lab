#가변매개변수 -> 함수의 매개변수 개수가 바뀌는 경우에 
print(3,4,5,6,7)
print(1,2,3)

#가변매개변수 - 매개변수의 개수가 바뀐다. 
#매개 변수 앞에 *을 붙인다.
#매개변수에 기본값을 줄때는 변수 자체가 여러개 만들어진다. 
def myadd(*args): #변수하나에 값을 여러개 전달한다 
    print( type(args))  # *args - tuple로 전달하겠다
    for a in args:
        print(a)

myadd(1,2)
myadd(1,2,3)

def myadd2(*data):
    s = 0 
    for i in data:
        s+=i 
    return s 

print( myadd2(1,3,5))
print( myadd2(1,3,5,7,9))
print( myadd2(1,3,5,10,12,13))
#일반인자와 tuple인자를 같이 써야 할때는 일반인자가 먼저 와야 한다 
#나머지를 tuple로 받으니까 

def myadd3(n, *data):
    print("n", n)
    for i in data:
        print(i)

myadd3(1,2,3,4,5)

#dict타입을 매개변수로 넘길수도 있다. **
#매개변수 전달 방식이 달라진다. 
def myfunc(d):
    print(d)

person={"name":"홍길동", "age":12}
myfunc({"name":"홍길동", "age":12})

def myfunc2(**d):
    print(d) 

myfunc2(name="홍길동", age=23)

# 일반 인자랑 tuple 인자랑 dict인자를 셋 다 쓰고 싶으면 순서가 있다.
# 일반 인자, tuple 타입, dict 타입
def profile(rule, *skills, **details):
    print("rule", rule)
    print("skill", skills)
    print("details", details)

profile("programmer", "python", "react", "deeplearnning",
        yearpay = 100000000000, position = "개발자")
