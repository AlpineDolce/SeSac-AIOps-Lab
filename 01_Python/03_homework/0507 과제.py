# TODO 숫자야구게임
# 0~9중 숫자 3개를 컴퓨터가 무작위로 선택한다.          3   7   9
# 0~9중 숫자 3개를 사용자에게 입력받는다.               1   4   5    
# 사용자가 입력한 숫자와 컴퓨터가 선택한 숫자를 비교한다.
# 컴퓨터가 입력한 숫자와 사용자가 입력한 숫자를 비교하여 
# 1. 숫자와 위치가 모두 일치하면 '스트라이크'
# 2. 숫자가 일치하지만 위치가 다르면 '볼'
# 3. 숫자가 일치하지 않으면 '아웃'
# 4. 사용자가 6번이내에 3개의 숫자를 모두 맞히면 '축하합니다. 게임을 종료합니다.'
# 5. 사용자가 6번을 초과하여 3개의 숫자를 모두 맞히지 못하면 '아쉽습니다. 게임을 종료합니다.'

# CHECKLIST
# [x]1. 컴퓨터가 무작위로 숫자 3개를 선택하는 함수
# [x]2. 사용자에게 숫자를 입력받는 함수
# [x]3. 컴퓨터와 사용자가 입력한 숫자를 비교하여 결과값을 내는 함수
# [x]4. 6번이내에 3개를 맞히는지 확인하는 함수, 6번 초과 시 게임 종료
# [x]5. 모든 함수를 연결하는 메인 함수 

import random

# 컴퓨터가 중복 없는 숫자 3개 선택 (1~9)
def com_select():
    return random.sample(range(1, 10), 3)

# 사용자 입력 받기 (검증 포함)
def user_select():
    while True:
        user_input = input("서로 다른 숫자 3개를 입력하세요 (예: 123): ")
        if len(user_input) != 3 or not user_input.isdigit():
            print("숫자 3자리를 정확히 입력하세요.")
            continue

        user_digits = [int(n) for n in user_input]
        if 0 in user_digits or len(set(user_digits)) != 3:
            print("1~9 범위의 서로 다른 숫자여야 합니다.")
            continue

        return user_digits

# 스트라이크, 볼 계산
def match_number(com_num, user_num):
    strike = 0
    ball = 0
    for i in range(3):
        if user_num[i] == com_num[i]:
            strike += 1
        elif user_num[i] in com_num:
            ball += 1
    return strike, ball

# 게임 실행 함수
def run_game():
    com_num = com_select()
    print("숫자야구 게임을 시작합니다! (기회: 6번)")
    print("1~9까지의 서로 다른 숫자 3개를 맞춰보세요.\n")

    for attempt in range(1, 7):
        print(f"{attempt}번째 시도:")
        user_num = user_select()
        strike, ball = match_number(com_num, user_num)
        print(f"결과: {strike} Strike {ball} Ball\n")

        if strike == 3:
            print("정답입니다! 축하합니다.")
            return

    print(f"기회를 모두 사용했습니다. 정답은 {''.join(map(str, com_num))}입니다.")

# 게임 실행
run_game()

# NOTE 클린코드 
# import random

# def generate_answer():
#     return random.sample(range(10), 3)

# def get_feedback(answer, guess):
#     strike = sum(a == b for a, b in zip(answer, guess))
#     ball = len(set(answer) & set(guess)) - strike
#     return strike, ball

# def is_valid_guess(guess):
#     return (guess.isdigit() and len(guess) == 3 and len(set(guess)) == 3)

# def main():
#     answer = generate_answer()
#     attempts = 6
#     print("숫자 야구게임 시작! (서로 다른 3자리 숫자 맞추기)")

#     for attempt in range(1, attempts + 1):
#         while True:
#             guess_input = input(f"[{attempt}/6] 숫자 3개 입력 (예: 123): ")
#             if is_valid_guess(guess_input):
#                 break
#             print("잘못된 입력입니다. 중복 없는 3자리 숫자를 입력하세요.")
        
#         guess = list(map(int, guess_input))
#         strike, ball = get_feedback(answer, guess)
#         print(f"{strike} 스트라이크, {ball} 볼")

#         if strike == 3:
#             print("정답입니다! 축하합니다!")
#             return

#     print(f"실패! 정답은 {''.join(map(str, answer))}였습니다.")

# if __name__ == "__main__":
#     main()
