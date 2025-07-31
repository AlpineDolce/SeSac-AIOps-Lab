# TODO 가위바위보 게임 -> 객체지향으로 짜기
# 컴퓨터가 1, 2, 3 중 하나를 랜덤으로 선택한다. 
# 사용자도 1, 2, 3 중 하나를 선택한다. 
# 컴퓨터가 사용자보다 1, 2, 3 중 하나를 랜덤으로 선택한다. 
# 결과를 출력한다. 
# 1:가위, 2:바위, 3:보 

# import random

# class Player:
#     def __init__(self):
        # FIXME self.choice가 public → 내부 상태를 외부에서 수정할 수 있음
#         self.choice = None

#     def make_choice(self):
#         while True:
#             try:
                # FIXME 문자열를 직접 입력 -> 오타 위협, 유효성 체크 불필요
#                 choice = int(input("가위(1), 바위(2), 보(3) 중 하나를 선택하세요: "))
#                 if choice in [1, 2, 3]:
#                     self.choice = choice
#                     break
#                 else:
#                     print("1, 2, 3 중 하나를 선택하세요.")
#             except ValueError:
#                 print("올바른 숫자를 입력하세요.")

#     def _map_number_to_choice(self,number):
#         return {1: "가위", 2: "바위", 3: "보"}[number]

# class Computer:
#     def __init__(self):
#         self.choice = None
    
#     def make_choice(self):
#         self.choice = random.choice(["가위", "바위", "보"])

# FIXME Game 클래스가 너무 많은 역할 (입출력, 판정, 반복 등)
# class Game:
#     def __init__(self):
#         self.player = Player()
#         self.computer = Computer()
#         self.rounds = 0
    
#     def determine_winner(self, player_choice, computer_choice):
#         if player_choice == computer_choice:
#             return "무승부"
#         elif(
#             {player_choice == "가위" and computer_choice == "바위"} or
#             {player_choice == "바위" and computer_choice == "보"} or
#             {player_choice == "보" and computer_choice == "가위"}
#         ):
#             return "플레이어 승"
#         else:
#             return "컴퓨터 승"
        
#     def play(self):
#         while True:
#             self.rounds += 1
#             print(f"\n[{self.rounds}번째 게임]")
#             self.player.make_choice()
#             self.computer.make_choice()
#             print(f"플레이어: {self.player.choice}, 컴퓨터 : {self.computer.choice}")
#             result = self.determine_winner(self.player.choice, self.computer.choice)
#             print(f"결과: {result}")
#             if result == "플레이어 승":
#                 print(f"\n 플레이어가 {self.rounds}번째 만에 이겼습니다.")

# if __name__ == "__main__":
#     game = Game()
#     game.play()


import random

# 선택 번호와 이름 매핑 딕셔너리
CHOICE_MAP = {
    1: "가위",
    2: "바위",
    3: "보"
}


class Player:
    """플레이어 클래스: 숫자로 입력 받아 문자열 선택값 설정"""
    def __init__(self):
        # NOTE self는 클래스 인스턴스 자신을 나타내며, _choice는 인스턴스의 속성.
        # 앞에 _ 언더스코어를 붙이는 이유는 "이 변수는 내부에서만"라는 비공식적인 규칙.
        self._choice = None

    def make_choice(self):
        while True:
            try:
                choice = int(input("선택하세요 (1: 가위, 2: 바위, 3: 보): "))
                if choice in CHOICE_MAP:
                    self._choice = CHOICE_MAP[choice]
                    break
                else:
                    print("1, 2, 3 중에서 선택하세요.")
            except ValueError:
                print("숫자로 입력하세요.")

    def get_choice(self):
        return self._choice


class Computer:
    """컴퓨터 클래스: 무작위로 선택"""
    def __init__(self):
        self._choice = None

    def make_choice(self):
        self._choice = random.choice(list(CHOICE_MAP.values()))

    def get_choice(self):
        return self._choice


class Game:
    """게임 클래스: 게임 진행, 승패 판별, 결과 출력"""
    def __init__(self):
        self.player = Player()
        self.computer = Computer()
        self.rounds = 0

    def _determine_winner(self, player_choice, computer_choice):
        if player_choice == computer_choice:
            return "무승부"
        elif (
            (player_choice == "가위" and computer_choice == "보") or
            (player_choice == "바위" and computer_choice == "가위") or
            (player_choice == "보" and computer_choice == "바위")
        ):
            return "플레이어 승"
        else:
            return "컴퓨터 승"

    def play(self):
        """플레이어가 이길 때까지 반복 실행"""
        while True:
            self.rounds += 1
            print(f"\n[{self.rounds}번째 게임]")

            # 입력 및 선택 처리
            self.player.make_choice()
            self.computer.make_choice()

            # 선택값 가져오기
            player_choice = self.player.get_choice()
            computer_choice = self.computer.get_choice()

            # 결과 출력
            print(f"플레이어: {player_choice}, 컴퓨터: {computer_choice}")
            result = self._determine_winner(player_choice, computer_choice)
            print(f"결과: {result}")

            if result == "플레이어 승":
                print(f"\n 플레이어가 {self.rounds}번째 만에 이겼습니다!")
                break


if __name__ == "__main__":
    game = Game()
    game.play()
