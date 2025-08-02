
# Membership.py
# 사용자 정보 저장

class Membership:
    def __init__(self, member_id, username, password, name, phone, email):
        self.member_id = member_id  # 회원번호, 시스템에서 자동 부여여
        self.username = username    # 아이디
        self.password = password    # 비밀번호
        self.name = name            # 이름
        self.phone = phone          # 전화번호
        self.email = email          # 이메일
        self.check_password()

    def check_password(self, input_password): # 비밀번호 확인
        return self.password == input_password
    
    def show_info(self):
         return (f"[회원번호: {self.member_id}]\n"
                f"이름: {self.name}\n"
                f"아이디: {self.username}\n"
                f"전화번호: {self.phone}\n"
                f"이메일: {self.email}")

