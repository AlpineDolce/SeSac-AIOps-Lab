from Membership import Membership

class MembershipManager:
    def __init__(self):
        self.membersList = []
        self.next_member_id = 1  # 다음 회원 ID를 위한 카운터

    def sign_up(self):
        # 회원가입
        # 사용자 입력 받아 Membership 객체 생성
        print("\n[회원가입]")
        username = input("아이디: ")
        
        if self.find_by_username(username): # 중복 아이디 확인
            print("이미 존재하는 아이디입니다.")
            return
        password = input("비밀번호: ")
        name = input("이름: ")
        phone = input("전화번호: ")
        email = input("이메일: ")

        # 생성한 회원을 members 리스트에 추가
        new_member = Membership(
            member_id = self.next_member_id, 
            username = username, 
            password = password, 
            name = name,
            phone = phone,
            email = email)
        
        self.membersList.append(new_member)
        self.next_member_id += 1    # next_member_id 증가
        print(f"{name}님, 회원가입이 완료되었습니다.")
    
    def login(self):
        # 로그인
        # 사용자 이름과 비밀번호를 확인하여 로그인 처리
        print("\n[로그인]")
        username = input("아이디: ")
        password = input("비밀번호: ")

        member = self.find_by_username(username)
        if member and member.check_password(password):
            print(f"{member.name}님, 로그인 되었습니다.")
            return member
        else:
            print("아이디 또는 비밀번호가 잘못되었습니다.")
            return None
        
        
    def get_my_info(self):
        # 내 정보 조회
        # 사용자 ID를 받아서 회원 정보 조회 처리
        print("\n[내 정보 조회]")
        username = input("아이디: ")
        member = self.find_by_username(username)
        if member:
            print(member.show_info())
        else:
            print("해당 아이디의 회원이 존재하지 않습니다.")

    def update_member(self, user):
        # 회원 정보 수정
        # 사용자 정보를 받아서 회원 정보 수정 처리
        print("\n[회원정보 수정]")
        if not user:
            print("로그인이 필요합니다.")
            return

        print("\n[회원 정보 수정]")
        pw = input("비밀번호 확인: ")
        if not user.check_password(pw):
            print("비밀번호가 틀렸습니다.")
            return

        new_phone = input("새 전화번호: ")
        new_email = input("새 이메일: ")
        user.phone = new_phone
        user.email = new_email
        print("회원 정보가 수정되었습니다.")

    def delete_member(self, user):
        # 회원 탈퇴
        # 사용자 ID를 받아서 회원 탈퇴 처리
        if not user:
            print("로그인이 필요합니다.")
            return

        print("\n[회원 탈퇴]")
        pw = input("비밀번호 확인: ")
        if not user.check_password(pw):
            print("비밀번호가 틀렸습니다.")
            return

        self.members.remove(user)
        print("회원 탈퇴가 완료되었습니다.")
        

    def find_by_username(self, username):
        # 주어진 username으로 회원을 찾는 메서드
        for member in self.membersList:
            if member.username == username:
                return member
        return None