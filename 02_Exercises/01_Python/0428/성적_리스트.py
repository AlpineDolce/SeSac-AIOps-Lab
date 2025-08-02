# 이름 국어 영어 수학 총점 평균, 평균에 대해서 수(90) 우(80) 미(70) 양(60) 가(50미만) 표현
# TODO 1. 리스트 선언 2. for문으로 리스트 생성 3. 총점 및 평균 구하기
students = []

# 총점, 평균, 등급 저장x, 정수만 저장
for i in range(0, 2):
    name = input("이름 : ")
    kor = int(input("국어 : "))
    eng = int(input("영어 : "))
    math = int(input("수학 : "))
    
    student = {
        "name" : name,
        "kor" : kor,
        "eng" : eng,
        "math" : math
    }
    students.append(student)

# 총점, 평균, 등급 계산 for문 추가
for student in students:
    total = student["kor"] + student["eng"] + student["math"]
    avg = total / 3

    # 등급 판단
    if avg >= 90:
        grade = "수"
    elif avg >= 80:
        grade = "우"
    elif avg >= 70:
        grade = "미"
    elif avg >= 60:
        grade = "양"
    else:
        grade = "가"

    # # NOTE : dict + for 문으로 접근
    # grade_table ={
    #     90: "수",
    #     80: "우",
    #     70: "미",
    #     60: "양",
    #     0: "가"  # 60미만
    # }
    # def get_grade(avg):
    #     for score, grade in grade_table.items():
    #         if avg >= score:
    #             return grade

    
    # 학생 dict에 추가
    student["total"] = total
    student["avg"] = avg
    student["grade"] = grade
    
# 출력
for student in students:
    print(f"{student['name']} {student['total']} {student['avg']} {student['grade']}")

# # NOTE 클린코드
# 학생 점수 관리 프로그램 (클린 코드: dict 매핑 + 3단계 구조)

# # 등급 컷오프와 매핑
# GRADE_CUT = {
#     90: "수",
#     80: "우",
#     70: "미",
#     60: "양",
#      0: "가"
# }

# # 학생 수 입력
# student_count = int(input("학생 수 : "))
# students = []

# # [1] 입력 단계: 이름과 점수만 저장
# for _ in range(student_count):
#     students.append({
#         "name": input("이름   : "),
#         "kor" : int(input("국어   : ")),
#         "eng" : int(input("영어   : ")),
#         "math": int(input("수학   : "))
#     })

# # [2] 처리 단계: 총점·평균 계산, dict 매핑으로 등급 결정
# for s in students:
#     s["total"] = s["kor"] + s["eng"] + s["math"]
#     s["avg"]   = s["total"] / 3
    
#     # dict 순회하면서 첫 번째 매칭된 등급으로 결정
#     for cutoff, grade in GRADE_CUT.items():
#         if s["avg"] >= cutoff:
#             s["grade"] = grade
#             break

# # [3] 출력 단계
# print()
# for s in students:
#     print(f"{s['name']}  총점: {s['total']}  평균: {s['avg']:.2f}  등급: {s['grade']}")


# nameList=[]
# korList=[]
# engList=[]
# matList=[]
# totalList=[]
# avgList=[]
# gradeList=[]

# #입력부터 
# for i in range(0,4):
#     name =  input("이름 : ")
#     kor = int(input("국어 : "))
#     eng = int(input("영어 : "))
#     mat = int(input("수학 : "))

#     nameList.append(name)
#     korList.append(kor)
#     engList.append(eng)
#     matList.append(mat)
    
# for i in range(0, len(nameList)):
#     total = korList[i] + engList[i] + matList[i]
#     avg = total/3 
#     totalList.append(total)
#     avgList.append(avg)

# for i in range(0, len(nameList)):
#     grade =""
#     if avgList[i]>=90 :
#         grade = "수"
#     elif avgList[i]>=80 :
#         grade = "우"
#     elif avgList[i]>=70 :
#         grade = "미"
#     elif avgList[i]>=60 :
#         grade = "양"
#     else:
#         grade="가"
#     gradeList.append( grade )

        
# for i in range(0, len(nameList)):
#     print(f"{nameList[i]}",  end="\t")
#     print(f"{korList[i]}",   end="\t")
#     print(f"{matList[i]}",   end="\t")
#     print(f"{engList[i]}",   end="\t")
#     print(f"{totalList[i]}", end="\t")
#     print(f"{avgList[i]}",   end="\t")
#     print(f"{gradeList[i]}")
