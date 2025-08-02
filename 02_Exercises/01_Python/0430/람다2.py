"""
검색 알고리즘
1. 순차검색(선형검색) 

찾을떄
"""

"""

...1
...2
...3    O(3)

for i in range(0, n):   n+100 n이 무한정커지면 100은 의미가 없다.
                        3n+100 O(n)
                        n**2+3n+3o O(n**2)
2. 색인순차 : 정렬해서
3. 이분검색
    데이터가 반드시 정렬되어 있어야 한다.
    내부 데이터가 자주 바뀌면 정렬하는 시간이 더 오래 걸려서 항상 빠르다고 할수는 없다.
    데이터를 절반을 쪼개서 왼쪽을 선택할지 오른쪽을 선택할지
    데이터를 절반을 쪼개서 왼쪽을 선택할지 오른쪽을 선택할지
    데이터를 절반을 쪼개서 왼쪽을 선택할지 오른쪽을 선택할지
    데이터를 찾을때까지 두개로 나누는 작업을 한다.(이분검색) -> O(logn)
3. 해쉬 검색 - 이론상 검색 속도가 O(1)이다.
                속도때문에 메모리를 많이 차지한다. 구현도 어렵다.
    trade off - 거래할때 서로 합의점 파이썬의 dict - dictionary의 약자
                HshMap, HashTable, Map, Dictionary
"""

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
key = 5 # 찾고자 하는 값
find = -1 # 정수 변수, 못 찾은 상태

for i in range(0, len(a)): # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    if key == a[i]:
        find = i
        break
def myfliter(aList, key):
    for i in range(0, len(a)):
        if key == a[i]:
            return i
    return -1
# pos = myfliter(a, 4)
# print(pos)

# a = ["red", "green", "blue", "cyan", "gray"]

# pos  = myfliter(a, "cyan")
# print(pos)

a = [
        {"name" : "A", "age" : 10},
        {"name" : "B", "age" : 11},
        {"name" : "C", "age" : 12},
        {"name" : "D", "age" : 13},
        {"name" : "E", "age" : 14},
     ]

# pos = myfliter(a, {"name" : "C", "age" : 12})
# print(pos)

def myfliter2(funcKey, a):
    for i in range(0, len(a)):
        if funcKey(a[i]):
            return i
    return -1
pos = myfliter2(lambda x: x["name"] == "C", a)
print(pos)


"""
정렬 => 데이터베이스를 사용하는 순간 데이터베이스 쿼리에서
        검색과 정렬을 지원한다.
        데이터베이스 못쓰는 상황에 파일을 써야 한다면 직접 구현해야한다.
    순서대로 데이터를 늘여놓는 것
    오름차순 정렬
    내림차순 정렬

    select, bubble, quick (엄청 빠름)

    # select 정렬; 오름차순의 경우에
    5   1   2   4   3
    0번방 데이터가 제일 작다고 가정하자.
    0,1 0,2 0,3 0,4 조건에 위배되면 바꿔치기

    1   5 2 4 3
    1번방 , 2, 3, 4
    1 2 5 4 3  2, (3, 4)
    
    1 2 3 5 4   3, 4 비교
    
    1 2 3 4 5   최종
    
    NOTE
    **"리스트에서 가장 작은(혹은 큰) 값을 찾아서 제일 앞에 보내고, 
    그 다음 작은 값을 찾아 그 다음 위치로 보내는 방식"**입니다.
    총 N개의 원소가 있으면, N-1번 반복합니다.
    각 반복에서 가장 작은 값의 위치를 찾아서 현재 위치의 값과 교환합니다
"""

def selectSort(dataList, key):
    # 0 ... 1-n
    # 1 ... 2-n
    # 2 ... 3-n
    # n-1 ... n
    # aList = dataList ; 얕은 복사(주소만 복사함), aList와 dataList 데이터는 같다.
    aList = [x for x in dataList]   # 컴프리핸션을 이용한 깊은 복사
    for i in range(0, len(aList)-1):
        for j in range(0, len(aList)):
            if aList[i] > aList[j]: # 더 작은 것이 앞에 있어야 한다.
                temp = aList[i]
                aList[j] = aList[i]
                aList[i] = temp
    return aList    #반환

a = [5,1,2,4,3]
b = selectSort(a, key= lambda x:x)
print(a)
print(b)

a = [{"name":"A", "age":12},
     {"name":"C", "age":11},
     {"name":"E", "age":13},
     {"name":"D", "age":14},
     {"name":"B", "age":15} ]
b = selectSort(a, key= lambda x:x["name"])
print(a)
print(b)
