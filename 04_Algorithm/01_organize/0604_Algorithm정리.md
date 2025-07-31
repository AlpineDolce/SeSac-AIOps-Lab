# 🧮 알고리즘 기초: 큐와 링크드리스트 (Day 25)  

> 이 문서는 부트캠프에서 학습한 주요 알고리즘과 자료구조에 대한 핵심 이론과 개념을 정리한 자료입니다. 큐, 링크드 리스트, 이중 링크드 리스트, 재귀 호출에 초점을 맞춰 각 개념의 원리, 동작 방식, 구현 예시, 그리고 특징을 상세히 다룹니다. 본 문서를 통해 알고리즘과 자료구조에 대한 이해를 높이고, 실제 문제 해결에 적용하는 데 도움이 되기를 바랍니다.

---

## 목차  

1.  [**큐(Queue)**](#1-큐queue)
    -   [1.1 큐 정의](#11-큐queue란)
    -   [1.2 주요 연산](#12-큐의-주요-연산)
    -   [1.3 Python 구현 (`collections.deque`)](#13-python-구현-예시-collectionsdeque-활용)
    -   [1.4 활용 예시](#14-활용-예시)
    -   [1.5 시각적 구조 및 특징](#15-큐의-시각적-구조-및-특징-요약)
    -   [1.6 시간 복잡도](#16-큐의-시간-복잡도)
    -   [1.7 변형 구조](#17-큐의-변형-구조)
    -   [1.8 스택과 큐 비교](#18-스택과-큐-비교)
2.  [**링크드 리스트(Linked List)**](#2-링크드-리스트linked-list)
    -   [2.1 링크드 리스트 정의](#21-링크드-리스트란)
    -   [2.2 주요 특징](#22-주요-특징)
    -   [2.3 구조도](#23-구조도-예시)
    -   [2.4 주요 연산](#24-주요-연산-요약)
    -   [2.5 Python 구현 (단일)](#25-python-구현-예시-단일-링크드-리스트)
    -   [2.6 시간 복잡도](#26-링크드-리스트의-시간-복잡도)
    -   [2.7 링크드 리스트 종류](#27-링크드-리스트의-종류)
    -   [2.8 활용 예시](#28-활용-예시)
3.  [**이중 링크드 리스트(Doubly Linked List)**](#3-이중-링크드-리스트doubly-linked-list)
    -   [3.1 이중 링크드 리스트 정의](#31-이중-링크드-리스트란)
    -   [3.2 주요 특징](#32-주요-특징)
    -   [3.3 구조도](#33-구조도-예시)
    -   [3.4 Python 구현 (이중)](#34-python-구현-예시-이중-링크드-리스트)
    -   [3.5 시간 복잡도](#35-이중-링크드-리스트의-시간-복잡도)
    -   [3.6 단일 링크드 리스트와 비교](#36-단일-링크드-리스트와의-비교)
    -   [3.7 실전 활용 예시](#37-실전-활용-예시)
    -   [3.8 성능 및 메모리 최적화](#38-성능과-메모리-최적화-팁)
4.  [**재귀 호출 (Recursion)**](#4-재귀-호출-recursion-문제-해결의-우아한-방법)
    -   [4.1 핵심 구성 요소](#41-재귀-호출의-핵심-구성-요소)
    -   [4.2 예시 1: 팩토리얼](#42-예시-1-팩토리얼-계산)
    -   [4.3 예시 2: 피보나치 수열](#43-예시-2-피보나치-수열)
    -   [4.4 재귀 vs. 반복문](#44-재귀-vs-반복문-어떤-것을-선택할까)
    -   [4.5 재귀 호출과 스택](#45-재귀-호출과-스택-call-stack)
    -   [4.6 스택 오버플로우 주의사항](#46-재귀-호출-시-주의할-점-스택-오버플로우)
    -   [4.7 실전 활용 예시](#47-실전-활용-예시)
    -   [4.8 추가 예시: 문자열 뒤집기](#48-추가-예시-재귀로-문자열-뒤집기)
    -   [4.9 꼬리 재귀](#49-꼬리-재귀-tail-recursion)
    -   [4.10 요약](#410-요약)


---

## 1. 큐(Queue)

### 1.1. 큐(Queue)란?

**큐(Queue)**는 **선입선출(FIFO, First In First Out)** 원칙을 따르는 선형 자료구조입니다.
먼저 들어온 데이터가 먼저 나가며, **줄 서기**, **대기열**, **작업 처리 순서** 등 순차적인 처리 상황에서 사용됩니다.

---

### 1.2. 큐의 주요 연산

| 연산         | 설명                           |
| ---------- | ---------------------------- |
| `enqueue`  | 큐의 **뒤**에 데이터를 추가            |
| `dequeue`  | 큐의 **앞**에서 데이터를 제거하고 반환      |
| `peek`     | 큐의 앞에 있는 데이터를 **제거하지 않고 확인** |
| `is_empty` | 큐가 비어 있는지 확인                 |

---

### 1.3. Python 구현 예시 (`collections.deque` 활용)

Python에서 큐를 구현할 때 `list`를 사용할 수도 있지만, `list`의 `pop(0)` 연산은 O(n)의 시간 복잡도를 가집니다. 이는 리스트의 모든 요소를 한 칸씩 당겨야 하기 때문입니다. 반면, `collections` 모듈의 `deque` (Double-ended Queue)는 양쪽 끝에서의 삽입(`append`, `appendleft`)과 삭제(`pop`, `popleft`)가 모두 O(1)의 시간 복잡도를 가지므로, 큐 구현에 훨씬 효율적입니다.

```python
from collections import deque

# 큐 생성: deque 객체를 생성하여 큐로 활용합니다.
queue = deque()

# 데이터 추가 (enqueue): 큐의 뒤쪽에 데이터를 추가합니다.
queue.append('A')
queue.append('B')
queue.append('C')
print("큐 상태:", queue)  # 출력: deque(['A', 'B', 'C'])

# 데이터 제거 (dequeue): 큐의 앞쪽에서 데이터를 제거하고 반환합니다.
first_element = queue.popleft()
print("제거된 데이터:", first_element)  # 출력: A
print("큐 상태:", queue)        # 출력: deque(['B', 'C'])

# 맨 앞 데이터 확인 (peek): 큐의 맨 앞에 있는 데이터를 제거하지 않고 확인합니다.
# deque는 인덱스 접근을 지원하므로 queue[0]으로 접근 가능합니다.
if queue: # 큐가 비어있지 않은지 확인 후 접근
    print("맨 앞 데이터:", queue[0])  # 출력: B
else:
    print("큐가 비어있습니다.")

# 큐가 비었는지 확인: 큐가 비어있으면 True, 아니면 False를 반환합니다.
print("큐가 비어있는가?", not queue)  # 출력: False (현재 'B', 'C'가 남아있음)

# 큐 비우기 예시
queue.clear()
print("큐가 비어있는가?", not queue) # 출력: True
```

---

### 1.4. 활용 예시

| 활용 분야              | 설명            |
| ------------------ | ------------- |
| **프린터 대기열**        | 인쇄 요청 순서대로 출력 |
| **BFS (너비 우선 탐색)** | 그래프의 시작 노드에서 가까운 노드부터 탐색하는 알고리즘으로, 큐를 사용하여 다음에 방문할 노드들을 순서대로 저장하고 처리합니다. 최단 경로를 찾을 때 유용합니다. |
| **운영체제의 태스크 스케줄링** | 프로세스 실행 순서 관리 |
| **실시간 처리 시스템**     | 입력 순서 보장      |

---

### 1.5. 큐의 시각적 구조 및 특징 요약

큐는 데이터를 한쪽 끝(Rear)으로만 추가하고, 다른 쪽 끝(Front)으로만 제거하는 **단방향** 흐름을 가집니다. 이는 마치 터널과 같아서, 먼저 들어간 것이 먼저 나오는 **선입선출(FIFO)** 원칙을 철저히 따릅니다.

```text
[ Front ] <--- [ A ] <--- [ B ] <--- [ C ] <--- [ Rear ]
              (Dequeue)             (Enqueue)
```

**데이터 흐름 예시:**

```text
초기 상태: []

1. Enqueue 'A'  → ['A']
2. Enqueue 'B'  → ['A', 'B']
3. Enqueue 'C'  → ['A', 'B', 'C']

4. Dequeue      → ['B', 'C'] (꺼낸 값: 'A')
5. Peek         → 'B' (현재 Front 값)
```

**주요 특징:**

*   **선입선출(FIFO)** 구조: First In, First Out. 먼저 들어온 데이터가 먼저 나갑니다.
*   **순차적 접근**: 특정 위치의 데이터를 직접 탐색하거나 접근할 수 없으며, 오직 Front에서만 제거하고 Rear에서만 추가할 수 있습니다.
*   **`collections.deque` 활용**: Python에서는 `collections.deque` 자료형을 사용하여 효율적인 큐를 구현할 수 있습니다. 이는 양쪽 끝에서의 삽입/삭제가 O(1) 시간 복잡도를 가지기 때문입니다.
*   **활용 분야**: BFS 탐색, 작업 예약 시스템, 프린터 대기열, 운영체제의 태스크 스케줄링 등 순서가 중요한 상황에서 널리 사용됩니다.

---

### 1.6. 큐의 시간 복잡도

| 연산         | 시간 복잡도 |
| ---------- | ------ |
| `enqueue`  | O(1)   |
| `dequeue`  | O(1)   |
| `peek`     | O(1)   |
| `is_empty` | O(1)   |

 ✅ `collections.deque`는 양쪽 끝에서 O(1)로 삽입/삭제 가능해 큐 구현에 적합합니다.
* O(1) 연산으로 성능 우수
---

### 1.7. 큐의 변형 구조

| 구조           | 특징                                  |
| ------------ | ----------------------------------- |
| 원형 큐         | 고정된 크기의 배열로 구현, 공간 재사용 가능           |
| 우선순위 큐       | 값의 우선순위에 따라 먼저 처리됨 (`heapq` 등으로 구현) |
| 이중 큐 (Deque) | 앞뒤 양쪽에서 삽입/삭제가 가능한 큐 (`deque`)      |

---

### 1.8. 스택과 큐 비교

| 특징    | 스택 (Stack)                                | 큐 (Queue)                                        |
| ----- | ----------------------------------------- | ------------------------------------------------ |
| 접근 방식 | 후입선출 (LIFO)                               | 선입선출 (FIFO)                                      |
| 삽입 위치 | 맨 위 (top)                                 | 뒤쪽 (rear)                                        |
| 삭제 위치 | 맨 위 (top)                                 | 앞쪽 (front)                                       |
| 주요 연산 | `push`, `pop`, `peek`, `is_empty`, `size` | `enqueue`, `dequeue`, `peek`, `is_empty`, `size` |
| 예시    | 함수 호출 관리, 웹 브라우저 뒤로 가기, 괄호 짝 검사           | 프로세스 스케줄링, 프린터 대기열, 은행 업무 대기열                    |

---

## 2. 링크드 리스트(Linked List)

### 2.1. 링크드 리스트란?

**링크드 리스트(Linked List)**는 **포인터(참조)**를 사용해 각 노드를 연결한 선형 자료구조입니다.
배열과는 달리 요소들이 메모리에 **연속적으로 저장되지 않으며**, **동적으로 크기 조절**이 가능합니다.

---

### 2.2. 주요 특징

| 특징              | 설명                                        |
| --------------- | ----------------------------------------- |
| **노드 기반 구조**    | 각 노드는 `데이터` + `다음 노드를 가리키는 포인터(next)`로 구성 |
| **동적 메모리 할당**   | 필요할 때마다 새로운 노드 추가 가능 (크기 제한 X)            |
| **중간 삽입/삭제 빠름** | 위치만 알면 포인터만 수정하면 되므로 O(1) 시간 가능           |
| **접근 속도 느림**    | 특정 인덱스를 찾으려면 처음부터 순차 탐색 (O(n))            |
| **메모리 낭비 가능성**  | 각 노드마다 포인터를 저장하므로 배열보다 메모리 사용량 많음         |

---

### 2.3. 구조도 예시

```text
Head
 ↓
[A | next] → [B | next] → [C | next] → None
```
```
[Head]
   ↓
+------+     +------+     +------+     +------+
|  A   | --> |  B   | --> |  C   | --> | None |
+------+     +------+     +------+     +------+
```

| 용어           | 설명                                                        |
| ------------ | --------------------------------------------------------- |
| **노드(Node)** | 데이터(`A`, `B`, `C`)와 다음 노드를 가리키는 포인터(`next`)로 구성된 기본 단위    |
| **헤드(Head)** | 연결 리스트의 **시작 지점**을 가리키는 포인터, 리스트의 첫 노드를 참조함               |
| **테일(Tail)** | 리스트의 **마지막 노드**는 `next`가 `None`인 노드, 즉 더 이상 연결된 노드가 없는 상태 |
| **연결 방향**    | 한 방향으로만 연결됨 → 단일 링크드 리스트(Singly Linked List)의 기본 특징       |


---

### 2.4. 주요 연산 요약

| 연산                  | 설명                 |
| ------------------- | ------------------ |
| `append(data)`      | 리스트 끝에 데이터 추가      |
| `insert(pos, data)` | 지정 위치에 데이터 삽입      |
| `delete(data)`      | 특정 데이터를 가진 노드 삭제   |
| `search(data)`      | 특정 데이터를 가진 노드를 찾아 해당 노드의 인덱스 또는 존재 여부를 반환합니다. (순차 탐색 필요) |
| `print_list()`      | 전체 리스트 출력          |

---

### 2.5. Python 구현 예시 (단일 링크드 리스트)

단일 링크드 리스트는 각 노드가 데이터와 다음 노드를 가리키는 포인터(`next`)만을 가지는 가장 기본적인 형태입니다. 이 예시에서는 `Node` 클래스와 `LinkedList` 클래스를 정의하여 단일 링크드 리스트의 핵심 연산들을 구현합니다.

```python
# Node 클래스 정의
# 각 노드는 실제 데이터(data)와 다음 노드를 가리키는 참조(next)를 가집니다.
class Node:
    def __init__(self, data):
        self.data = data      # 노드가 저장할 데이터
        self.next = None      # 다음 노드를 가리키는 포인터 (초기값은 None)

# LinkedList 클래스 정의
# 단일 링크드 리스트 전체를 관리하는 클래스입니다.
class LinkedList:
    def __init__(self):
        self.head = None      # 리스트의 첫 번째 노드를 가리키는 포인터 (초기값은 None)

    # append(data): 리스트의 끝에 새로운 노드를 추가하는 연산
    # 시간 복잡도: O(n) - 마지막 노드를 찾기 위해 리스트를 순회해야 합니다.
    def append(self, data):
        new_node = Node(data)
        if not self.head:     # 리스트가 비어있으면 새 노드를 head로 지정
            self.head = new_node
            return
        curr = self.head
        while curr.next:      # 마지막 노드까지 이동
            curr = curr.next
        curr.next = new_node  # 마지막 노드의 next에 새 노드 연결

    # insert(pos, data): 지정된 위치(pos)에 새로운 노드를 삽입하는 연산
    # 시간 복잡도: O(n) - 삽입 위치까지 이동해야 합니다.
    def insert(self, pos, data):
        new_node = Node(data)
        if pos == 0:          # 맨 앞에 삽입하는 경우
            new_node.next = self.head
            self.head = new_node
            return
        
        curr = self.head
        # 삽입 위치 바로 전 노드까지 이동합니다.
        # 예를 들어 pos가 1이면, 0번째 노드(head)에서 멈춥니다.
        for _ in range(pos - 1):
            if curr is None: # 위치가 리스트 길이보다 크면 삽입하지 않습니다.
                print(f"Error: Position {pos} is out of bounds.")
                return
            curr = curr.next
        
        if curr is None: # curr가 None이면, pos가 유효하지 않거나 리스트 끝을 넘어선 경우입니다.
            print(f"Error: Cannot insert at position {pos}. List is too short.")
            return

        # 새 노드를 현재 노드(curr)와 다음 노드(curr.next) 사이에 삽입합니다.
        new_node.next = curr.next
        curr.next = new_node

    # delete(data): 특정 데이터를 가진 노드를 찾아 삭제하는 연산
    # 시간 복잡도: O(n) - 삭제할 노드를 찾기 위해 리스트를 순회해야 합니다.
    def delete(self, data):
        curr = self.head
        prev = None
        while curr:
            if curr.data == data: # 삭제할 데이터를 찾으면
                if prev: # 이전 노드가 있다면, 이전 노드의 next를 현재 노드의 next로 변경
                    prev.next = curr.next
                else:    # 이전 노드가 없다면 (즉, head 노드를 삭제하는 경우), head를 다음 노드로 변경
                    self.head = curr.next
                print(f"'{data}' 데이터가 삭제되었습니다.")
                return
            prev = curr
            curr = curr.next
        print(f"'{data}' 데이터를 찾을 수 없습니다.")

    # search(data): 특정 데이터를 가진 노드를 찾아 해당 인덱스를 반환하는 연산
    # 시간 복잡도: O(n) - 데이터를 찾기 위해 리스트를 순회해야 합니다.
    def search(self, data):
        curr = self.head
        index = 0
        while curr:
            if curr.data == data:
                return index  # 데이터를 찾으면 해당 인덱스 반환
            curr = curr.next
            index += 1
        return -1  # 데이터를 찾지 못하면 -1 반환

    # print_list(): 리스트의 모든 노드 데이터를 순서대로 출력하는 연산
    # 시간 복잡도: O(n) - 모든 노드를 방문해야 합니다.
    def print_list(self):
        curr = self.head
        elements = []
        while curr:
            elements.append(str(curr.data))
            curr = curr.next
        print(" -> ".join(elements) + " -> None")

# 사용 예시
print("\n--- 단일 링크드 리스트 사용 예시 ---")
ll = LinkedList()
ll.append('A')
ll.append('B')
ll.append('C')
print("초기 리스트:")
ll.print_list() # 출력: A -> B -> C -> None

ll.insert(1, 'D')
print("1번째 위치에 'D' 삽입 후:")
ll.print_list() # 출력: A -> D -> B -> C -> None

ll.delete('B')
print("'B' 삭제 후:")
ll.print_list() # 출력: A -> D -> C -> None

print(f"'C'의 위치: {ll.search('C')}") # 출력: 'C'의 위치: 2
print(f"'X'의 위치: {ll.search('X')}") # 출력: 'X'의 위치: -1 (존재하지 않음)

ll.insert(10, 'Z') # 범위를 벗어난 삽입 시도
ll.print_list()
```

---

### 2.6. 링크드 리스트의 시간 복잡도

| 연산         | 시간 복잡도       | 설명                           |
| ---------- | ------------ | ---------------------------- |
| 검색(search) | O(n)         | 순차 탐색 필요                     |
| 삽입(insert) | O(1) \~ O(n) | 위치 알고 있다면 O(1), 그렇지 않으면 O(n) |
| 삭제(delete) | O(1) \~ O(n) | 마찬가지로 위치 탐색 포함 시 O(n)        |

- 링크드 리스트 장점
  - 크기가 동적으로 변함
  - 중간에 데이터 삽입/삭제가 빠름(포인터만 변경)
- 링크드 리스트 단점
  - 임의 접근이 느림(순차 탐색 필요)
  - 추가적인 포인터 공간 필요

---

### 2.7. 링크드 리스트의 종류

| 종류             | 설명                                    |
| -------------- | ------------------------------------- |
| **단일 링크드 리스트** | 다음 노드만 가리킴                            |
| **이중 링크드 리스트** | 이전 노드와 다음 노드를 모두 가리킴 (`prev`, `next`) |
| **원형 링크드 리스트** | 마지막 노드가 첫 노드를 가리킴 (끝이 없음)             |

---

### 2.8. 활용 예시

* **메모리 관리**: 동적으로 크기 조절
* **실시간 목록 처리**: 음악 재생 목록, Undo/Redo 기능
* **자료구조의 기반**: 큐, 스택, 그래프, 트리 등 구현에 사용

---

## 3. 이중 링크드 리스트(Doubly Linked List) 

### 3.1. 이중 링크드 리스트란?

이중 링크드 리스트는 **각 노드가 두 개의 포인터(이전 노드, 다음 노드)를 가지는 연결 리스트**입니다.
이 구조 덕분에 **양방향 탐색이 가능**하고, **삽입/삭제가 유연**하게 이뤄집니다.

### 3.2. 주요 특징


| 특징              | 설명                                                   |
| --------------- | ---------------------------------------------------- |
| **양방향 연결**      | 각 노드는 `prev`, `next`를 가지며 앞뒤로 이동 가능                  |
| **중간 삽입/삭제 용이** | 포인터만 바꾸면 되므로 노드 위치를 알고 있다면 O(1)                      |
| **양방향 탐색 가능**   | 순방향/역방향 모두 탐색 가능 (`print_forward`, `print_backward`) |
| **메모리 사용 증가**   | 포인터 2개를 저장하므로 단일 링크드 리스트보다 메모리 소비 ↑                  |
| **복잡도 증가**      | 포인터 두 개를 모두 관리해야 하므로 구현 시 실수 가능성 ↑                   |

---
### 3.3. 구조도 예시



```text
None ← [A | prev | next] ⇄ [B | prev | next] ⇄ [C | prev | next] → None
```
```
None
  ↑
+--------+      +--------+       +--------+
|   A    |  ⇄  |   B     |  ⇄  |    C    |
+--------+      +--------+       +--------+
  ↓                                ↓
None		                     None

```

| 구성 요소        | 설명                                                |
| ------------ | ------------------------------------------------- |
| **노드(Node)** | 각 노드는 `데이터`와 두 개의 포인터(`prev`, `next`)를 포함합니다.     |
| **prev 포인터** | 해당 노드보다 **앞쪽 노드**를 가리킵니다. (없으면 `None`)            |
| **next 포인터** | 해당 노드보다 **뒤쪽 노드**를 가리킵니다. (없으면 `None`)            |
| **Head**     | 이중 링크드 리스트의 첫 번째 노드를 가리킵니다.                       |
| **Tail**     | 마지막 노드의 `next`는 항상 `None`입니다.                     |
| **양방향 연결**   | 노드 간 연결이 **앞뒤 모두 가능**하여, 리스트를 양쪽 방향으로 탐색할 수 있습니다. |

---
### 3.4. Python 구현 예시 (이중 링크드 리스트)

이중 링크드 리스트는 각 노드가 이전 노드(`prev`)와 다음 노드(`next`)를 모두 가리키는 포인터를 가집니다. 이를 통해 양방향 탐색 및 삽입/삭제가 유연하게 이루어집니다.

```python
# Node 클래스 정의
# 각 노드는 데이터(data), 이전 노드(prev), 다음 노드(next) 포인터를 가집니다.
class Node:
    def __init__(self, data):
        self.data = data      # 노드가 저장할 데이터
        self.prev = None      # 이전 노드를 가리키는 포인터 (초기값은 None)
        self.next = None      # 다음 노드를 가리키는 포인터 (초기값은 None)

    def __repr__(self): # 노드 객체를 문자열로 표현할 때 사용 (디버깅 용이)
        return f"Node({self.data})"

# DoublyLinkedList 클래스 정의
# 이중 링크드 리스트 전체를 관리하는 클래스입니다.
class DoublyLinkedList:
    def __init__(self):
        self.head = None      # 리스트의 첫 번째 노드를 가리키는 포인터
        self.tail = None      # 리스트의 마지막 노드를 가리키는 포인터 (이중 링크드 리스트의 장점)
        self.size = 0         # 리스트의 현재 크기

    # append(data): 리스트의 끝에 새로운 노드를 추가하는 연산
    # 시간 복잡도: O(1) - tail 포인터를 통해 마지막 노드에 직접 접근 가능합니다.
    def append(self, data):
        new_node = Node(data)
        if not self.head:     # 리스트가 비어있으면 새 노드를 head이자 tail로 지정
            self.head = new_node
            self.tail = new_node
        else:                 # 리스트가 비어있지 않으면
            self.tail.next = new_node  # 현재 tail의 next를 새 노드로 연결
            new_node.prev = self.tail  # 새 노드의 prev를 현재 tail로 연결
            self.tail = new_node      # tail을 새 노드로 업데이트
        self.size += 1

    # insert(pos, data): 지정된 위치(pos)에 새로운 노드를 삽입하는 연산
    # 시간 복잡도: O(n) - 삽입 위치까지 이동해야 합니다.
    def insert(self, pos, data):
        if pos < 0 or pos > self.size:
            print(f"Error: Position {pos} is out of bounds for insertion.")
            return

        new_node = Node(data)
        if pos == 0:          # 맨 앞에 삽입하는 경우
            new_node.next = self.head
            if self.head:
                self.head.prev = new_node
            self.head = new_node
            if not self.tail: # 리스트가 비어있었으면 tail도 업데이트
                self.tail = new_node
        elif pos == self.size: # 맨 뒤에 삽입하는 경우 (append와 동일)
            self.append(data) # append 메서드 재활용
            return
        else:                 # 중간에 삽입하는 경우
            curr = self.head
            # 삽입 위치까지 이동 (pos-1이 아니라 pos까지 이동하여 curr.prev와 curr.next를 활용)
            for _ in range(pos):
                curr = curr.next
            
            # 새 노드를 curr.prev와 curr 사이에 삽입
            new_node.next = curr
            new_node.prev = curr.prev
            curr.prev.next = new_node
            curr.prev = new_node
        self.size += 1

    # delete(data): 특정 데이터를 가진 노드를 찾아 삭제하는 연산
    # 시간 복잡도: O(n) - 삭제할 노드를 찾기 위해 리스트를 순회해야 합니다.
    def delete(self, data):
        curr = self.head
        while curr:
            if curr.data == data:     # 삭제할 데이터를 찾으면
                if curr.prev:         # 이전 노드가 있다면
                    curr.prev.next = curr.next  # 이전 노드의 next를 현재 노드의 next로 연결
                else:                 # 이전 노드가 없다면 (head 노드를 삭제하는 경우)
                    self.head = curr.next       # head를 다음 노드로 변경

                if curr.next:         # 다음 노드가 있다면
                    curr.next.prev = curr.prev  # 다음 노드의 prev를 현재 노드의 prev로 연결
                else:                 # 다음 노드가 없다면 (tail 노드를 삭제하는 경우)
                    self.tail = curr.prev       # tail을 이전 노드로 변경
                
                self.size -= 1
                print(f"'{data}' 데이터가 삭제되었습니다.")
                return True  # 삭제 성공
            curr = curr.next
        print(f"'{data}' 데이터를 찾을 수 없습니다.")
        return False  # 삭제 실패 (해당 데이터 없음)

    # search(target): 특정 데이터를 가진 노드를 찾아 해당 인덱스를 반환하는 연산
    # 시간 복잡도: O(n) - 데이터를 찾기 위해 리스트를 순회해야 합니다。
    def search(self, target):
        curr = self.head
        index = 0
        while curr:
            if curr.data == target:
                return index  # 데이터를 찾으면 해당 인덱스 반환
            curr = curr.next
            index += 1
        return -1  # 데이터를 찾지 못하면 -1 반환

    # print_forward(): 리스트의 모든 노드 데이터를 앞에서 뒤로 순서대로 출력하는 연산
    # 시간 복잡도: O(n) - 모든 노드를 방문해야 합니다.
    def print_forward(self):
        curr = self.head
        elements = []
        while curr:
            elements.append(str(curr.data))
            curr = curr.next
        print(" <-> ".join(elements) + " <-> None")

    # print_backward(): 리스트의 모든 노드 데이터를 뒤에서 앞으로 순서대로 출력하는 연산
    # 시간 복잡도: O(n) - 모든 노드를 방문해야 합니다.
    def print_backward(self):
        curr = self.tail # tail부터 시작하여 역방향으로 탐색
        elements = []
        while curr:
            elements.append(str(curr.data))
            curr = curr.prev
        print(" <-> ".join(elements) + " <-> None")
        
    def __len__(self):
        return self.size

    def __iter__(self):
        curr = self.head
        while curr:
            yield curr.data
            curr = curr.next

# 사용 예시
print("\n--- 이중 링크드 리스트 사용 예시 ---")
dll = DoublyLinkedList()
dll.append('A')
dll.append('B')
dll.append('C')
print("초기 리스트 (정방향):")
dll.print_forward() # 출력: A <-> B <-> C <-> None

dll.insert(1, 'D')
print("1번째 위치에 'D' 삽입 후 (정방향):")
dll.print_forward() # 출력: A <-> D <-> B <-> C <-> None

dll.delete('B')
print("'B' 삭제 후 (정방향):")
dll.print_forward() # 출력: A <-> D <-> C <-> None

print("'B' 삭제 후 (역방향):")
dll.print_backward() # 출력: C <-> D <-> A <-> None

print(f"'D'의 위치: {dll.search('D')}") # 출력: 'D'의 위치: 1
print(f"'X'의 위치: {dll.search('X')}") # 출력: 'X'의 위치: -1

dll.insert(0, 'Z') # 맨 앞에 삽입
print("맨 앞에 'Z' 삽입 후 (정방향):")
dll.print_forward() # 출력: Z <-> A <-> D <-> C <-> None

dll.insert(dll.size, 'E') # 맨 뒤에 삽입
print("맨 뒤에 'E' 삽입 후 (정방향):")
dll.print_forward() # 출력: Z <-> A <-> D <-> C <-> E <-> None

dll.delete('Z') # head 삭제
print("'Z' 삭제 후 (정방향):")
dll.print_forward() # 출력: A <-> D <-> C <-> E <-> None

dll.delete('E') # tail 삭제
print("'E' 삭제 후 (정방향):")
dll.print_forward() # 출력: A <-> D <-> C <-> None

print(f"리스트 크기: {len(dll)}") # 출력: 리스트 크기: 3

print("리스트 순회:")
for item in dll:
    print(item, end=" ") # 출력: A D C
print()
```
---

### 3.5. 이중 링크드 리스트의 시간 복잡도

| 연산     | 시간 복잡도 | 설명            |
| ------ | ------ | ------------- |
| 탐색     | O(n)   | 순차 탐색 필요      |
| 삽입/삭제  | O(1)   | 위치 알고 있을 경우   |
| 역방향 탐색 | O(n)   | `prev`를 따라 탐색 |

---
### 3.6. 단일 링크드 리스트와의 비교

| 항목        | 단일 링크드 리스트   | 이중 링크드 리스트    |
| --------- | ------------ | ------------- |
| 방향성       | 한 방향         | 양방향           |
| 삽입/삭제 유연성 | 위치 파악 시 O(1) | 위치 파악 시 O(1)  |
| 메모리 사용    | 낮음           | 높음 (prev도 저장) |
| 구현 복잡도    | 낮음           | 높음            |
| 역방향 순회    | 불가능          | 가능            |


---

### 3.7. 실전 활용 예시

| 활용 분야            | 설명                     |
| ---------------- | ---------------------- |
| **브라우저 방문 기록**   | 뒤로/앞으로 탐색이 용이          |
| **Undo/Redo 기능** | 텍스트 에디터, 그래픽 툴 등       |
| **LRU 캐시 구현**    | 가장 오래된 항목을 빠르게 제거      |
| **양방향 리스트 탐색**   | 앞/뒤로 탐색이 잦은 데이터 구조에 적합 |

---

### 3.8. 성능과 메모리 최적화 팁

* **`tail` 포인터 유지**: 역방향 순회 및 끝 삽입 속도 개선
* **`size` 변수 관리**: `len()` 연산 O(1)로 구현
* **순환 참조 주의**: Python의 가비지 컬렉터가 있지만, 강한 참조가 서로 엮이면 메모리 누수 발생 가능

---

## 4. 재귀 호출 (Recursion): 문제 해결의 우아한 방법

재귀 호출은 함수가 자기 자신을 호출하여 문제를 해결하는 프로그래밍 기법입니다. 복잡한 문제를 더 작고 동일한 형태의 하위 문제로 분할하여 해결할 때 매우 유용하며, 특히 트리나 그래프와 같은 계층적/재귀적 구조를 다룰 때 강력한 도구가 됩니다.

---

### 4.1 재귀 호출의 핵심 구성 요소

모든 재귀 함수는 다음 두 가지 핵심 요소를 반드시 포함해야 합니다:

1.  **베이스 케이스 (Base Case)**: 재귀 호출을 멈추는 조건입니다. 이 조건이 없으면 함수는 무한히 자기 자신을 호출하게 되어 스택 오버플로우(Stack Overflow) 오류를 발생시킵니다.
2.  **재귀 케이스 (Recursive Case)**: 현재 문제를 더 작고 동일한 형태의 하위 문제로 분할하여 자기 자신을 다시 호출하는 부분입니다.

**기본 구조 예시:**

```python
def recursive_function(parameter):
    # 1. 베이스 케이스 (종료 조건): 재귀 호출을 멈추고 값을 반환합니다.
    if parameter == 0: 
        return some_value # 더 이상 재귀 호출을 하지 않고 종료

    # 2. 재귀 케이스: 문제를 더 작게 만들고 자기 자신을 다시 호출합니다.
    #    이때, 다음 호출의 parameter는 현재 parameter보다 '작아져야' 합니다.
    result = recursive_function(parameter - 1) # 자기 자신을 호출
    
    # 재귀 호출의 결과를 이용하여 현재 문제의 해답을 구성합니다.
    return process_result(result, parameter)
```

**간단한 재귀 함수 예시 (카운트 다운):**

```python
def countdown(n):
    if n <= 0: # 베이스 케이스: n이 0 이하가 되면 재귀를 멈춥니다.
        print("발사!")
        return
    print(n) # 현재 n 값을 출력
    countdown(n - 1) # 재귀 케이스: n을 1 감소시켜 다시 호출

print("\n--- 카운트 다운 예시 ---")
countdown(3)
# 출력:
# 3
# 2
# 1
# 발사!
```

---

### 4.2 예시 1: 팩토리얼 계산

팩토리얼(Factorial)은 1부터 어떤 양의 정수 n까지의 곱을 의미합니다. 수학적으로 `n! = n * (n-1) * ... * 1`로 정의되며, `0! = 1`입니다. 팩토리얼은 재귀의 개념을 이해하기에 좋은 예시입니다.

**재귀를 이용한 팩토리얼:**

```python
def factorial_recursive(n):
    # 베이스 케이스: n이 0이면 1을 반환 (재귀 종료 조건)
    if n == 0:
        return 1
    # 재귀 케이스: n * (n-1)! 로 문제를 분할하여 재귀 호출
    return n * factorial_recursive(n - 1)

print("\n--- 팩토리얼 (재귀) 예시 ---")
print(f"3! = {factorial_recursive(3)}") # 출력: 3! = 6
print(f"5! = {factorial_recursive(5)}") # 출력: 5! = 120
```

**반복문을 이용한 팩토리얼:**

재귀 호출은 개념적으로 우아하지만, 실제 성능이나 메모리 사용 측면에서는 반복문이 더 효율적일 수 있습니다. 반복문은 함수 호출 오버헤드가 없기 때문입니다.

```python
def factorial_iterative(n):
    result = 1
    # 1부터 n까지 곱해나갑니다.
    for i in range(1, n + 1):
        result *= i
    return result

print("\n--- 팩토리얼 (반복) 예시 ---")
print(f"3! = {factorial_iterative(3)}") # 출력: 3! = 6
print(f"5! = {factorial_iterative(5)}") # 출력: 5! = 120
```

**재귀 호출의 실행 흐름 (factorial_recursive(3))**

재귀 함수가 호출될 때마다 새로운 함수 프레임이 스택에 쌓이고, 베이스 케이스에 도달하면 역순으로 결과를 반환하며 스택에서 제거됩니다.

```
factorial_recursive(3) 호출
  → 3 * factorial_recursive(2) 호출
    → 2 * factorial_recursive(1) 호출
      → 1 * factorial_recursive(0) 호출
        → factorial_recursive(0)은 베이스 케이스이므로 1 반환
      ← factorial_recursive(1)은 1 * 1 = 1 반환
    ← factorial_recursive(2)은 2 * 1 = 2 반환
  ← factorial_recursive(3)은 3 * 2 = 6 반환
```

**시각적 흐름도 (호출 스택 시뮬레이션)**

```text
호출 스택 (LIFO - Last In, First Out):

| factorial_recursive(0) [return 1] |
| factorial_recursive(1)            |
| factorial_recursive(2)            |
| factorial_recursive(3)            |
-------------------------------------
(가장 최근 호출이 스택의 맨 위에 쌓임)
```

*   가장 마지막에 호출된 `factorial_recursive(0)`부터 결과가 반환되기 시작합니다.
*   스택에 쌓였던 함수들이 역순으로 실행을 완료하며 최종 결과를 계산합니다.

---

### 4.3 예시 2: 피보나치 수열

피보나치 수열은 첫째 및 둘째 항이 1이며 그 뒤의 모든 항은 바로 앞 두 항의 합인 수열입니다. 수학적으로 `F(n) = F(n-1) + F(n-2)`로 정의되며, `F(0) = 0`, `F(1) = 1`입니다.

**재귀를 이용한 피보나치 수열:**

```python
def fibonacci_recursive(n):
    if n <= 1: # 베이스 케이스: n이 0 또는 1이면 n을 반환
        return n
    # 재귀 케이스: 이전 두 항의 합으로 문제를 분할하여 재귀 호출
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

print("\n--- 피보나치 수열 (재귀) 예시 ---")
print(f"fibonacci(6) = {fibonacci_recursive(6)}") # 출력: fibonacci(6) = 8
print(f"fibonacci(10) = {fibonacci_recursive(10)}") # 출력: fibonacci(10) = 55
```

**재귀 피보나치 함수의 비효율성:**

재귀를 이용한 피보나치 수열 계산은 개념적으로는 간단하지만, `fibonacci_recursive(n-1)`과 `fibonacci_recursive(n-2)`를 계산하는 과정에서 **동일한 하위 문제들이 반복적으로 계산**되는 비효율성이 발생합니다. 예를 들어 `fibonacci_recursive(5)`를 계산할 때 `fibonacci_recursive(3)`이 여러 번 호출됩니다.

```text
fib(5)
├── fib(4)
│   ├── fib(3)
│   │   ├── fib(2)
│   │   │   ├── fib(1)
│   │   │   └── fib(0)
│   │   └── fib(1)
│   └── fib(2)  <-- 중복 계산
│       ├── fib(1)
│       └── fib(0)
└── fib(3)  <-- 중복 계산
    ├── fib(2)
    │   ├── fib(1)
    │   └── fib(0)
    └── fib(1)
```

이러한 비효율성은 `n`이 커질수록 기하급수적으로 계산량이 늘어나 성능 저하를 초래합니다. 이를 해결하기 위해 **메모이제이션(Memoization)** 또는 **동적 계획법(Dynamic Programming, DP)**과 같은 기법을 사용하여 이미 계산된 결과를 저장하고 재활용함으로써 중복 계산을 피할 수 있습니다.

---

### 4.4 재귀 vs. 반복문: 어떤 것을 선택할까?

재귀와 반복문은 모두 반복적인 작업을 수행하는 방법이지만, 각각의 장단점과 적합한 사용 사례가 다릅니다. 문제의 특성과 요구사항에 따라 적절한 방법을 선택하는 것이 중요합니다.

| 항목           | **반복문 (Loop)**                               | **재귀 (Recursion)**                                 |
| :------------- | :---------------------------------------------- | :--------------------------------------------------- |
| **구조**       | `for`, `while` 등의 반복 제어문을 직접 사용합니다. | 함수가 자기 자신을 호출하여 반복을 구현합니다.         |
| **이해 난이도**   | 순차적인 흐름으로 초보자에게 더 직관적입니다.         | 처음에는 이해하기 어려울 수 있으나, 특정 패턴에 익숙해지면 매우 강력합니다. |
| **코드 길이**    | 반복 조건과 상태 변화를 명시적으로 설정해야 합니다. | 문제의 정의가 재귀적일 경우 코드가 간결하고 우아해집니다. |
| **속도**       | 함수 호출 오버헤드가 없어 일반적으로 더 빠릅니다.     | 함수 호출 시 스택 프레임 생성 및 해제 오버헤드로 인해 느릴 수 있습니다. |
| **메모리 사용량**  | 변수만 저장하므로 메모리 사용량이 적습니다.           | 각 재귀 호출마다 스택 프레임이 쌓여 메모리 사용량이 많아질 수 있습니다. |
| **스택 오버플로우** | 발생하지 않습니다.                               | 재귀 깊이가 너무 깊어지면 `RecursionError`가 발생할 수 있습니다. |
| **사용 예시**    | 단순 반복, 누적합 계산, 배열/리스트 순회 등.         | 트리/그래프 탐색 (DFS), 분할 정복 알고리즘, 백트래킹, 수학적 정의가 재귀적인 문제 (팩토리얼, 피보나치 등). |
| **종료 조건**    | 루프 조건 (`i < n`, `while True` 등)을 직접 지정합니다. | 반드시 베이스 케이스(종료 조건)가 필요하며, 없으면 무한 재귀에 빠집니다. |
| **호환성**      | 대부분의 알고리즘에 보편적으로 적용 가능합니다.       | 특정 구조적 문제 (트리, 그래프)나 분할 정복 문제에 더 적합합니다. |

---

### 4.5 재귀 호출과 스택 (Call Stack)

재귀 호출은 내부적으로 **호출 스택(Call Stack)**이라는 자료구조를 적극적으로 활용합니다. 함수가 호출될 때마다 해당 함수의 실행 정보(매개변수, 지역 변수, 반환 주소 등)가 스택에 `push`되고, 함수가 종료되면 스택에서 `pop`됩니다.

재귀 함수는 자기 자신을 계속 호출하므로, 베이스 케이스에 도달할 때까지 호출 스택에 함수 호출들이 계속 쌓이게 됩니다. 베이스 케이스에 도달하여 값을 반환하기 시작하면, 스택의 가장 위에 있는 함수부터 차례대로 실행을 완료하고 스택에서 제거됩니다. 이는 **후입선출(LIFO - Last In, First Out)** 원리와 정확히 일치합니다.

**스택 동작 시각화:**

```text
함수 호출 시 (Push):

| factorial(0) |
| factorial(1) |
| factorial(2) |
| factorial(3) |
---------------
(스택의 바닥)

함수 반환 시 (Pop):

| factorial(0) | ← 먼저 Pop되어 결과 반환
---------------
| factorial(1) |
| factorial(2) |
| factorial(3) |
---------------
(스택의 바닥)
```

이러한 스택의 특성 때문에 재귀 호출은 깊이가 너무 깊어지면 스택 오버플로우(Stack Overflow)를 발생시킬 수 있습니다.

---

### 4.6 재귀 호출 시 주의할 점: 스택 오버플로우

재귀 함수를 사용할 때 가장 주의해야 할 점은 **무한 재귀(Infinite Recursion)**에 빠지지 않도록 하는 것입니다. 베이스 케이스가 없거나, 베이스 케이스에 도달할 수 없는 로직이라면 함수는 계속해서 자기 자신을 호출하게 됩니다. 이 경우 호출 스택에 함수 프레임이 계속 쌓이게 되고, 결국 시스템이 할당한 스택 메모리 공간을 초과하여 **`RecursionError: maximum recursion depth exceeded`** 오류가 발생합니다. 이를 **스택 오버플로우(Stack Overflow)**라고 합니다.

```python
def infinite_recursion():
    infinite_recursion() # 베이스 케이스가 없어 무한히 자기 자신을 호출

# infinite_recursion() # 이 함수를 호출하면 RecursionError 발생
```

**Python의 재귀 깊이 제한:**

Python은 기본적으로 재귀 호출의 최대 깊이를 제한하고 있습니다. 이는 무한 재귀로 인한 시스템 자원 고갈을 방지하기 위함입니다. 일반적으로 이 제한은 **1000**으로 설정되어 있습니다. 필요한 경우 `sys` 모듈을 사용하여 이 제한을 일시적으로 늘릴 수 있지만, 이는 신중하게 사용해야 합니다.

```python
import sys

# 현재 재귀 깊이 제한 확인
print(f"현재 재귀 깊이 제한: {sys.getrecursionlimit()}")

# 재귀 깊이 제한 변경 (예: 2000으로)
# sys.setrecursionlimit(2000)
# print(f"변경된 재귀 깊이 제한: {sys.getrecursionlimit()}")

# 변경된 제한으로 인해 발생할 수 있는 오류 예시 (주석 처리)
# def deep_recursion(n):
#     if n == 0:
#         return
#     deep_recursion(n - 1)

# deep_recursion(1500) # 기본 제한(1000)을 초과하면 RecursionError 발생
```

**주의:** 재귀 깊이 제한을 너무 높게 설정하면 메모리 사용량이 급증하거나 시스템이 불안정해질 수 있으므로, 재귀 깊이가 깊은 문제는 반복문이나 동적 계획법 등으로 해결하는 것을 우선적으로 고려해야 합니다.

---

### 4.7 실전 활용 예시

| 분야            | 재귀 활용 예        |
| ------------- | -------------- |
| **정렬 알고리즘**   | 퀵 정렬, 병합 정렬    |
| **트리/그래프 탐색** | DFS (깊이 우선 탐색) |
| **수학적 문제 해결** | 팩토리얼, 피보나치     |
| **백트래킹 문제**   | N-Queen, 순열/조합 |

* 구조가 **분기형, 트리형, 탐색형**일 때
* 문제를 하위 문제로 계속 **쪼갤 수 있을 때**
* 반복보다 **코드가 간결하고 명확할 때**

---

### 4.8 추가 예시: 재귀로 문자열 뒤집기

```python
def reverse(s):
    if len(s) <= 1:
        return s
    return reverse(s[1:]) + s[0]

print(reverse("hello"))  # 출력: "olleh"
```

---

### 4.9 꼬리 재귀 (Tail Recursion)

**꼬리 재귀(Tail Recursion)**는 재귀 호출이 함수의 마지막 연산으로 수행되는 형태를 말합니다. 즉, 재귀 호출 이후에 추가적인 연산이 없는 경우입니다. 이러한 형태의 재귀는 일부 컴파일러나 인터프리터에서 **꼬리 호출 최적화(Tail Call Optimization, TCO)**를 통해 반복문과 유사하게 스택 메모리를 재활용하도록 최적화될 수 있습니다. 이는 깊은 재귀 호출로 인한 스택 오버플로우 문제를 완화하고 성능을 향상시킬 수 있습니다.

**꼬리 재귀 팩토리얼 예시:**

```python
def tail_factorial(n, accumulator=1):
    # 베이스 케이스: n이 0이면 누적된 값(accumulator)을 반환
    if n == 0:
        return accumulator
    # 재귀 케이스: 재귀 호출이 함수의 마지막 연산 (accumulator에 n을 곱하여 전달)
    return tail_factorial(n - 1, n * accumulator)

print("\n--- 꼬리 재귀 팩토리얼 예시 ---")
print(f"tail_factorial(5) = {tail_factorial(5)}") # 출력: tail_factorial(5) = 120
```

**Python에서의 꼬리 재귀:**

안타깝게도 **Python은 공식적으로 꼬리 호출 최적화를 지원하지 않습니다.** 이는 Python의 스택 트레이스(Stack Trace)를 유지하여 디버깅을 용이하게 하려는 설계 철학 때문입니다. 따라서 Python에서는 꼬리 재귀를 사용하더라도 스택 오버플로우의 위험이 여전히 존재하며, 성능상 이점도 기대하기 어렵습니다. Python에서는 깊은 재귀가 필요한 경우 반복문으로 전환하거나, 명시적으로 스택을 관리하는 방식을 고려하는 것이 좋습니다.

---

### 4.10 요약

| 항목        | 설명                   |
| --------- | -------------------- |
| **정의**    | 함수가 자기 자신을 호출하는 구조   |
| **필수 조건** | 종료 조건(베이스 케이스) 필요    |
| **장점**    | 간결함, 분할정복, 트리 구조에 적합 |
| **단점**    | 메모리 소모, 스택 오버플로우 위험  |
| **활용 분야** | 트리/그래프 탐색, 알고리즘 설계 등 |

---
[⏮️ 이전 문서](./0602_Algorithm정리.md) | [다음 문서 ⏭️](./0605_Algorithm정리.md)