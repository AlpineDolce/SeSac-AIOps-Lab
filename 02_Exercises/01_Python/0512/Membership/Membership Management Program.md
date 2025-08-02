# 회원 관리 프로그램

## 회원 관리 프로그램 요구 사항

### 회원 관리 `Membership`

- 목적 : 사용자 정보 저장
- 필드 :
    + 회원 번호     : `member_id`   (회원 번호, 시스템에서 자동 부여)
    + 회원 아이디   : `username`    (아이디)
    + 패스워드      : `password`
    + 이름          : `name`
    + 전화번호      : `phone`
    + 이메일        : `email`
- 메서드(예시) :
    + `check_password(input_pw)` -> 비밀번호 검증

### 게시판 `NoticeBoard`

- 목적 : 게시글 한 건의 정보 저장
- 필드 :
    + 글 번호       : `post_id`     (글 번호, 시스템에서 자동 부여)
    + 작성자 번호   : `member_id`   (작성자 번호)
    + 제목          : `title`
    + 내용          : `content`
    + 작성일        : `created_at`  (작성일, datatime)
    + 조회수        : `views`
- 메서드(예시) :
    + `read()`      -> 조회수 증가 후 내용 출력
    + `update()`    -> 글 수정(작성자 + 비밀번호 일치 시)
    + `delete()`    -> 삭제 허용 조건 판단

### 회원가입 수정, 탈퇴, 조회(내 정보) `MembershipManager`

- 필드 : `members` (회원 리스트, `list` 또는 `dict`)
- 기능 메서드 :
    + `sign_up()`           -> 회원가입
    + `login()`             -> 로그인
    + `get_my_info()`       -> 내 정보 조회
    + `update_member()`     -> 회원 정보 수정
    + `delele_member()`     -> 회원 탈퇴
    
### 게시글 작성 `NoticeBoardManager`

- 필드: posts (게시글 리스트)
- 기능 메서드:
    + `write_post(member_id, title, content)` → 글 작성
    + `read_post(post_id)` → 글 읽기 (조회수 증가 포함)
    + `update_post(post_id, member_id, password)` → 글 수정 (검증 포함)
    + `delete_post(post_id, member_id, password)` → 글 삭제 (검증 포함)
    + `list_posts()` → 모든 글 요약 출력
  
- 회원 번호, 제목, 내용, 작성일(Date), 조회수 0
  + 읽어보기
  + 수정 (작성자만); 회원 번호랑 패스워드 입력 시 가능
  + 삭제 (작성자만); 회원 번호랑 패스워드 입력 시 가능

## 실행 흐름 설계

- 로그인 여부 체크
  + 로그인 후에만 글 작성/수정/삭제 가능하게 설계
  + `current_user` 변수를 통해 현재 로그인 상태 관리

- 메뉴 설계
  + 텍스트 기반 콘솔 메뉴 (예: 1. 회원가입 2. 로그인 3. 글쓰기 ...)
  + `while` 루프로 무한 루프, `input()`으로 사용자 입력

## 고려할 점 및 예외 처리
- 회원 ID/글 번호 자동 생성: 내부에서 `auto-increment` 관리
- 비밀번호 확인: 수정/삭제 시 필수
- 예외 처리: 잘못된 입력, 없는 ID 등 적절히 `try-except` 또는 조건문 처리
- 파일 저장 여부: 처음엔 메모리에서만 구현, 나중에 `pickle`, `json`, `csv` 등으로 확장 가능