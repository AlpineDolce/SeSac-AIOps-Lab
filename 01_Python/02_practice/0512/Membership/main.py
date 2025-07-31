from Membership import MembershipManager
from NoticeBoard import NoticeBoardManager

class main:
    def main():
        member_manager = MembershipManager()
        board_manager = NoticeBoardManager()
        current_user = None

        while True:
            print("\n====== 회원 & 게시판 프로그램 ======")
            print("1. 회원가입")
            print("2. 로그인")
            print("3. 내 정보 보기")
            print("4. 회원 정보 수정")
            print("5. 회원 탈퇴")
            print("6. 게시글 목록 보기")
            print("7. 게시글 작성")
            print("8. 게시글 읽기")
            print("9. 게시글 수정")
            print("10. 게시글 삭제")
            print("11. 로그아웃")
            print("0. 종료")
            choice = input("메뉴 선택: ")

            if choice == "1":
                member_manager.sign_up()

            elif choice == "2":
                current_user = member_manager.login()

            elif choice == "3":
                member_manager.get_my_info(current_user)

            elif choice == "4":
                member_manager.update_member(current_user)

            elif choice == "5":
                if current_user:
                    member_manager.delete_member(current_user)
                    current_user = None  # 탈퇴 시 자동 로그아웃
                else:
                    print("로그인이 필요합니다.")

            elif choice == "6":
                board_manager.list_posts()

            elif choice == "7":
                if current_user:
                    board_manager.write_post(current_user)
                else:
                    print("로그인이 필요합니다.")

            elif choice == "8":
                board_manager.read_post()

            elif choice == "9":
                if current_user:
                    board_manager.update_post(current_user)
                else:
                    print("로그인이 필요합니다.")

            elif choice == "10":
                if current_user:
                    board_manager.delete_post(current_user)
                else:
                    print("로그인이 필요합니다.")

            elif choice == "11":
                current_user = None
                print("로그아웃되었습니다.")

            elif choice == "0":
                print("프로그램을 종료합니다.")
                break

            else:
                print("잘못된 입력입니다. 다시 선택하세요.")

if __name__ == "__main__":
    main.main()