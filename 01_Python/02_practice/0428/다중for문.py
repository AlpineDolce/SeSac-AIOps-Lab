# for 문안에서 또 for문이 작동하는 경우이다.
# 외부의 루프가 M번 돌고, 내부 루프가 N번 돌면 M*N번 수행한다.
# 가급적 2중 for문까지만 동작하게 해야한다.

# for i in range(1, 5):
#     for j in range(1, 5):
#         print(f"i = {i} j = {j}")

# TODO 문제1 이중 for문 사용해서 1~100까지 출력 한줄에 10개씩 출력하기
for i in range(10):
    for j in range(10): # 10 * 10 = 100, 10 단위로 끊기기
        print(i*10+j+1, end =' ') # i는 십의 자리, j는 일의 자리, +1은 0 제거
    print()


# TODO 문제2 이중 for문 
# 1 -> 1
# 1 + 2 -> 3
# 1 + 2 + 3 -> 6 ...
# 1 + 2 + ...+ 10 -> 55

for i in range(1, 11):  # 1부터 10까지
    total = 0           # 합계를 0으로 초기화
    expr = ""           # 식"문자열" 초기화

    for j in range(1, i+1): # 1부터 i까지
        total += j 
        if j == 1:          # 첫번째 항일떄 expr을 "1"로 초기화
            expr = "1"      # expr을 "1"로 초기화하여 식의 시작을 명확하게 설정.
        else:
            # 기존 expr 뒤에 " + j"를 문자열로 붙입니다.
            # 예: 이전 expr이 "1 + 2"였다면, j가 3일 때 "1 + 2 + 3"
            expr += " + " + str(j)

    print(f"{expr} -> {total}")

# NOTE 다른 방법
# 1. join + sum 이용
for i in range(1, 11):
    nums = range(1, i+1)
    expr = " + ".join(map(str, nums))
    print(f"{expr} -> {sum(nums)}")

# 2. 수학 공식 Σ = n(n+1)/2
for i in range(1, 11):
    expr = " + ".join(str(j) for j in range(1, i+1))
    total = i * (i + 1) // 2
    print(f"{expr} -> {total}")


# TODO 문제3 삼각형 별 문제
for i in range(0, 10):
    for j in range(0, i):
        print("*", end="")
    print()

for i in range(0, 10):
    print(f"*"*i)

# TODO 문제4 마름모 문제 
# 마름모는 위 아래가 반전된 모양이다. 즉, 위 아래 구분을 해야한다.
# 또한, 공백과 별이 얼마나 있는지에 대해서도 계산을 해야한다.

n = 5           # 마름모의 반(위쪽) 높이
# 마름모 위쪽
for i in range(n):
    # 1. 좌측 공백 : n - i - 1개
    for j in range(n - i - 1):
        print(" ", end="")
    # 2. 별 출력 : 2 * i + 1개
    for j in range(2 * i + 1):
        print("*", end="")
    # 3.한 줄 출력 후 줄 바꿈
    print()

# 마름모 아래쪽
for i in range(n - 2, -1, -1):
    # 1. 좌측 공백 : n - i - 1개
    for j in range(n - i - 1):
        print(" ", end="")
    # 2. 별 출력 : 2 * i + 1개
    for j in range(2 * i + 1):
        print("*", end="")
    # 3. 한 줄 출력 후 줄 바꿈
    print()

# NOTE 동작 설명
# 1. 윗부분 (i = 0부터 n - 1까지)
#   n - i - 1개의 공백을 찍어 별을 가운데로 정렬
#   2*i + 1개의 별을 찍어 점점 넓어지는 형태 생성
# 2. 아랫부분 (i = n - 2부터 0까지 역순)
#    윗부분과 동일한 로직을 반대로 적용해 점점 좁아지는 형태 생성

# NOTE 각 부분 상세 설명
# 변수 n의 의미
#     n은 다이아몬드의 반 높이입니다.
#         윗부분(정점에서 중간까지)은 총 n줄
#         아랫부분(중간 바로 아래에서 바닥 정점까지)은 n-1줄
#     전체 줄 수: n + (n-1) = 2n-1 줄

# 1) 윗부분(for i in range(n))
#     i는 현재 그려야 할 줄의 위치 지표로, 0부터 시작해 1씩 증가하며 맨 중간 줄(i = n-1)까지 진행합니다.
#         1. 공백 계산
#             한 줄에 찍을 공백 수 = n - i - 1
#                 맨 위(i=0)에선 n-1개의 공백 → 별이 가장 중앙에 오도록 띄움
#                 줄이 내려올수록 공백이 1씩 줄어듦
#         2. 별 계산
#             한 줄에 찍을 별 수 = 2*i + 1
#                 i=0일 때 1개(*)
#                 i=1일 때 3개(***), i=2일 때 5개(*****)...
#                 중앙 줄(i=n-1)에서는 2*(n-1)+1 = 2n-1개의 별
#         3. 줄바꿈
#             공백과 별을 모두 찍은 뒤 print()로 줄을 마무리

# 2) 아랫부분(for i in range(n-2, -1, -1))
#     중앙 바로 아래(i=n-2)부터 시작해 맨 아래(i=0)까지 역순으로 반복합니다.
#     공백과 별 계산 공식은 윗부분과 동일합니다.
#         공백: n - i - 1
#         별: 2*i + 1

# 이 과정을 통해 다이아몬드 모양이 아래로 좁아지며 완성됩니다.
#
#     *        # i=0, 공백4 + 별1
#    ***       # i=1, 공백3 + 별3
#   *****      # i=2, 공백2 + 별5
#  *******     # i=3, 공백1 + 별7
# *********    # i=4, 공백0 + 별9  ← 중앙 줄
#  *******     # i=3, 공백1 + 별7
#   *****      # i=2, 공백2 + 별5
#    ***       # i=1, 공백3 + 별3
#     *        # i=0, 공백4 + 별1

# 강사님 답
# LINES = 7
# for i in range(1, LINES+1):
#     print(" " * (LINES-i), end="")
#     print("*" * (2*i-1))

# for i in range(1, LINES+1):
#     for j in range(0, (LINES-i)):
#         print(" ", end="")
#     for j in range(0, 2*i-1):
#         print("*", end="")
#     print()

# LINES = LINES-1 
# for i in range(0, LINES+1):
#     for j in range(0, i):
#         print(" ",end="")
#     for j in range(0, (LINES-i)*2 + 1):
#         print("*", end="")
#     print()