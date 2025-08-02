# TODO 주급계산 이름, 근무시간, 시간당 급여액, 5명에 대해서
# 홍길동 40 10000
# 임꺽정 20 20000
# 장길산 30 20000
# 홍경래 10 15000
# 이창옥 20 20000

NameList = []
WorkTimeList = []
PerPayList = []

for i in range(0, 5):
    name = input("이름 : ")
    work_time = int(input("일한 시간 : "))
    per_pay = int(input("시간당 급여액 : "))

    NameList.append(name)
    WorkTimeList.append(work_time)
    PerPayList.append(per_pay)
    
print()

for i in range(0, 5):
    print(f"{NameList[i]} {WorkTimeList[i]} {PerPayList[i]}")
