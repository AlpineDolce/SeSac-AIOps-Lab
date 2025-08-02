worker = [] # 한 사람분 저장하기
person_list =[
    {"name" : "홍길동", "work_time" : 20, "per_pay" : 10000},
    {"name" : "임꺽정", "work_time" : 40, "per_pay" : 20000},
    {"name" : "장길산", "work_time" : 15, "per_pay" : 15000},
]
for i in range(0, 2):
    worker["name"] = input("이름 : ")
    worker["work_time"] = int(input("근무시간 : "))
    worker["per_pay"] = int(input("시간당 급여 : "))
    person_list.append(worker)
    worker = {}

for i in person_list:
    pay = 0
    worker["pay"] = worker["work_time"] * worker["per_pay"]
print()
for i in person_list:
    print(f"worker name : {worker['name']} pay : {worker['pay']}")
