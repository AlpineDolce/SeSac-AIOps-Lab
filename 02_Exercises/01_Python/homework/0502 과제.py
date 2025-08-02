# TODO 가위바위보 게임
# 컴퓨터가 1, 2, 3 중 하나를 랜덤으로 선택한다. 
# 사용자도 1, 2, 3 중 하나를 선택한다. 
# 컴퓨터가 사용자보다 1, 2, 3 중 하나를 랜덤으로 선택한다. 
# 결과를 출력한다. 
# 1:가위, 2:바위, 3:보 
# import random

# NOTE 함수를 쓸떄 기능별로 분할하는 것이 좋다.
import random

# 가위바위보 매핑 (1: 가위, 2: 바위, 3: 보)
RPS = {1: "가위", 2: "바위", 3: "보"}

def get_user_input():
    """사용자로부터 1~3의 유효한 숫자를 입력받는다."""
    while True:
        try:
            user = int(input("1: 가위, 2: 바위, 3: 보 중 선택하세요: "))
            if user in RPS:
                return user
            else:
                print("1, 2, 3 중에서만 선택할 수 있습니다.")
        except ValueError:
            print("숫자만 입력해 주세요.")

def get_computer_choice():
    """컴퓨터가 1~3 중 하나를 무작위로 선택한다."""
    return random.randint(1, 3)

def judge_winner(user, com):
    """
    사용자와 컴퓨터의 선택을 받아 승패를 판정한다.
    반환값: 'user', 'com', 'draw'
    """
    if user == com:
        return "draw"
    elif (com == 1 and user == 3) or (com == 2 and user == 1) or (com == 3 and user == 2):
        return "com"
    else:
        return "user"

def game_play():
    """한 판의 가위바위보 게임을 실행한다."""
    print("\n=== 가위바위보 게임 시작 ===")
    user = get_user_input()
    com = get_computer_choice()

    print(f"컴퓨터: {RPS[com]}")
    print(f"사용자: {RPS[user]}")

    result = judge_winner(user, com)

    if result == "draw":
        print("결과: 비겼습니다.")
    elif result == "com":
        print("결과: 컴퓨터가 이겼습니다.")
    else:
        print("결과: 사용자가 이겼습니다.")

    return result

def game_rating():
    """10판의 게임을 진행하고 최종 승률을 계산한다."""
    user_win = com_win = draw = 0

    for i in range(10):
        print(f"\n[{i+1}번째 게임]")
        result = game_play()

        if result == "user":
            user_win += 1
        elif result == "com":
            com_win += 1
        else:
            draw += 1

    print("\n====================== 게임 결과 요약 ======================")
    print(f"사용자 승리: {user_win}회")
    print(f"컴퓨터 승리: {com_win}회")
    print(f"무승부: {draw}회")   
    print(f"\n사용자 승률: {(user_win / 10) * 100:.1f}%")
    print(f"\n컴퓨터 승률: {(com_win / 10) * 100:.1f}%")

# 게임 실행
game_rating()

