# 성적 확인 시스템 객체지향
# 이름, 국어, 영어, 수학, 총점, 평균, 학점

# TODO 문제 정의 및 요구 사항 정리
# 목적 : 이름과 국어, 영어, 수학 점수를 입력 받아 총점, 평균, 학점을 출력하는 프로그램
# 입력 방식 : 이름; str, 국영수 점수; int
# 출력 정보 : 이름, 각 점수, 총점, 평균, 학점

# TODO 클래스, 메소드 설계
# InputInfomation
#   - __init__ : 이름, 국어, 영어, 수학
#   - show : 결과 출력

# Scorecalculator
#   - __init__ : 이름, 국어, 영어, 수학
#   - total : 총점
#   - average : 평균
#   - grade : 학점
#   - show : 결과 출력

# GradeVerifier
#   - __init__ : 이름, 총점, 평균, 학점
#   - show : 결과 출력
#   - InputInfomation
#   - Scorecalculator
#   - show : 결과 출력

# class InputInfomation:
#     def __init__(self):
#         self.name = ""
#         self.kor = 0
#         self.eng = 0
#         self.math = 0
    
#     def InputInformation(self):
#         self.name = input('이름을 입력하세요 : ')
#         self.kor = int(input('국어 점수를 입력하세요 : '))
#         self.eng = int(input('영어 점수를 입력하세요 : '))
#         self.math = int(input('수학 점수를 입력하세요 : '))
#         return self.name, self.kor, self.eng, self.math

# class ScoreCalculator:
#     def __init__(self, name, kor, eng, math):
#         self.name = name
#         self.kor = kor
#         self.eng = eng
#         self.math = math
#         self.total_score = self.calc_total()
#         self.avg_score = self.calc_average()
#         self.grade_letter = self.calc_grade()

#     def calc_total(self):
#         return self.kor + self.eng + self.math
    
#     def calc_average(self):
#         return self.total / 3
    
#     def calc_grade(self):
#         average = self.avg_score
#         if self.aver >= 90:
#             self.grade = 'A'
#         elif self.aver >= 80:
#             self.grade = 'B'
#         elif self.aver >= 70:
#             self.grade = 'C'
#         elif self.aver >= 60:
#             self.grade = 'D'
#         else:
#             self.grade = 'F'
#         return self.grade

# class GradeVerifier:
#     def __init__(self, name, total, average, grade):
#         self.name = name
#         self.total = total
#         self.average = average
#         self.grade = grade
    
#     def show(self):
#         print(f'{self.name}의 총점은 {self.total}이고 평균은 {self.average:.2f}이고 학점은 {self.grade}입니다.')
        

# if __name__ == '__main__':
#     print('성적 확인 프로그램입니다.')
#     info = InputInfomation
#     score = ScoreCalculator(name, kor, eng, math)
#     verifier = GradeVerifier(name, score_avg_score, score.grade_letter)
#     verifier.show()
#     print('프로그램을 종료합니다.')

# XXX 문제점 : 원본 코드는 클래스 간 책임이 명확히 분리되지 않고 
# 메서드와 속성 간 이름 충돌 및 순서 의존성이 있어 
# 실행 오류와 유지보수 어려움을 유발합니다.


# NOTE 클린코드 및 리팩토링 
class StudentInput:
    def get_input(self):
        name = input("이름을 입력하세요: ")
        kor = self._get_score("국어")
        eng = self._get_score("영어")
        math = self._get_score("수학")
        return name, kor, eng, math

    def _get_score(self, subject):
        while True:
            try:
                score = int(input(f"{subject} 점수를 입력하세요 (0~100): "))
                if 0 <= score <= 100:
                    return score
                else:
                    print("점수는 0부터 100 사이여야 합니다.")
            except ValueError:
                print("숫자만 입력해주세요.")

class ScoreCalculator:
    def __init__(self, name, kor, eng, math):
        self.name = name
        self.kor = kor
        self.eng = eng
        self.math = math
        self.total = self.calculate_total()
        self.average = self.calculate_average()
        self.grade = self.calculate_grade()

    def calculate_total(self):
        return self.kor + self.eng + self.math

    def calculate_average(self):
        return self.total / 3

    def calculate_grade(self):
        avg = self.average
        if avg >= 90:
            return 'A'
        elif avg >= 80:
            return 'B'
        elif avg >= 70:
            return 'C'
        elif avg >= 60:
            return 'D'
        else:
            return 'F'

class ReportCard:
    def __init__(self, student):
        self.student = student

    def display(self):
        print("\n성적표")
        print(f"이름    : {self.student.name}")
        print(f"국어    : {self.student.kor}")
        print(f"영어    : {self.student.eng}")
        print(f"수학    : {self.student.math}")
        print(f"총점    : {self.student.total}")
        print(f"평균    : {self.student.average:.2f}")
        print(f"학점    : {self.student.grade}")

def main():
    print("성적 확인 프로그램입니다.")
    input_handler = StudentInput()
    name, kor, eng, math = input_handler.get_input()

    student = ScoreCalculator(name, kor, eng, math)
    report = ReportCard(student)
    report.display()

    print("\n프로그램을 종료합니다.")

if __name__ == '__main__':
    main()
