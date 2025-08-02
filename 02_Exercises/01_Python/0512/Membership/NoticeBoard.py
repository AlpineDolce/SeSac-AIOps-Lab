
from datetime import datetime

# NoticeBoard.py
# 게시글 한건의 정보 저장

class NoticeBoard:
    def __init__(self, post_id, member_id, title, content, views):
        self.post_id = post_id
        self.member_id = member_id
        self.title = title
        self.content = content
        self.created_at = datetime.now()
        self.views = 0
    
    def read(self): # 게시글 읽기
        pass
    def update(self): # 게시글 수정
        pass
    def delete(self): # 게시글 삭제
        pass

if __name__ == "__main__":
    n = NoticeBoard("post123", "user123", "공지사항", "안녕하세요", "2023-05-12", 0)
    print(n.post_id, n.member_id, n.title, n.content, n.created_at, n.views)
    