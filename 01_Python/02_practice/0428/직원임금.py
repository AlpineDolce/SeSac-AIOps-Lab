# 직원이 노동청 신고가 들어와 회사가 성별간 임금차별과 연봉을 구하려고 한다.
# 직원 전체가 10명이고 성별과 연봉을 입력받아 남녀별로 임금 평균을 구하려라

# TODO 1. 직원 정보를 입력받아 리스트에 저장 2. 남녀 구분 3. 남녀별 임금 평균
empolyees = []
for i in range(10):
    gender = input("성별을 입력하세요(남/여) : ")
    pay = int(input("연봉을 입력하세요 : "))

    empolyees.append({
        "gender" : gender,
        "pay" : pay
    })

# 남녀별 총합/인원수 초기화
male_total = 0
female_total = 0
male_count = 0
female_count = 0

# 남녀 구분
for emp in empolyees:
   if emp["gender"] == "남":
       male_total += emp["pay"]
       male_count += 1
   else:
       female_total += emp["pay"]
       female_count += 1

# 평균 계산 (0으로 나누는 오류 방지)
male_avg = male_total / male_count if male_count > 0 else 0
female_avg = female_total / female_count if female_count > 0 else 0

# 출력 소수점 첫번째에서 반올림
print(f"남성 평균 연봉: {male_avg :.0f}원")
print(f"여성 평균 연봉: {female_avg :.0f}원")
