<h2>데이터 크롤링 도구: Selenium, BeautifulSoup 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 `Selenium`을 사용하여 웹 페이지에서 팝업, 알림창(Alert), 그리고 iframe과 같은 특수 요소들을 처리하는 방법을 다룹니다. 이러한 요소들이 웹 크롤링을 방해하지 않도록 효과적으로 제어하는 방법을 학습하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Selenium 팝업 및 프레임 처리](#1-selenium-팝업-및-프레임-처리)
  - [1.1. 팝업 및 알림창 (Alert) 처리](#11-팝업-및-알림창-alert-처리)
  - [1.2. iframe 처리](#12-iframe-처리)
  - [1.3. 새 창/탭 처리](#13-새-창탭-처리)

---

## 1. Selenium 팝업 및 프레임 처리

### 1.1. 팝업 및 알림창 (Alert) 처리

웹 페이지에서 JavaScript `alert()`, `confirm()`, `prompt()` 함수로 생성되는 알림창(Alert)은 일반적인 웹 요소와 다르게 처리해야 합니다. `Selenium`은 `driver.switch_to.alert`를 통해 이러한 알림창에 접근하고 제어할 수 있는 기능을 제공합니다.

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# WebDriver 설정 (이전 섹션 참조)
service = Service(ChromeDriverManager().install())
options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")
driver = webdriver.Chrome(service=service, options=options)

try:
    # 테스트용 HTML 페이지 (알림창 시나리오)
    driver.get("data:text/html,<button onclick='alert(\"Hello Alert!\")'>Show Alert</button><button onclick='confirm(\"Are you sure?\")'>Show Confirm</button><button onclick='prompt(\"Enter your name:\", \"Guest\")'>Show Prompt</button>")
    time.sleep(1) # 페이지 로딩 대기

    # 1. Alert 처리
    alert_button = driver.find_element(By.XPATH, '//button[text()="Show Alert"]')
    alert_button.click()

    # Alert가 나타날 때까지 대기
    WebDriverWait(driver, 10).until(EC.alert_is_present())
    alert = driver.switch_to.alert
    print(f"\nAlert Text: {alert.text}")
    alert.accept() # Alert 확인 (OK 클릭)
    print("Alert accepted.")
    time.sleep(1)

    # 2. Confirm 처리
    confirm_button = driver.find_element(By.XPATH, '//button[text()="Show Confirm"]')
    confirm_button.click()

    WebDriverWait(driver, 10).until(EC.alert_is_present())
    confirm_dialog = driver.switch_to.alert
    print(f"Confirm Text: {confirm_dialog.text}")
    confirm_dialog.dismiss() # Confirm 취소 (Cancel 클릭)
    print("Confirm dismissed.")
    time.sleep(1)

    # 3. Prompt 처리
    prompt_button = driver.find_element(By.XPATH, '//button[text()="Show Prompt"]')
    prompt_button.click()

    WebDriverWait(driver, 10).until(EC.alert_is_present())
    prompt_dialog = driver.switch_to.alert
    print(f"Prompt Text: {prompt_dialog.text}")
    prompt_dialog.send_keys("Selenium User") # 텍스트 입력
    prompt_dialog.accept() # Prompt 확인
    print("Prompt accepted with input.")
    time.sleep(1)

finally:
    driver.quit()
```

> **[실무 노트] 알림창 처리의 중요성:**
> 웹 크롤링 중 예기치 않게 나타나는 알림창은 스크립트의 진행을 막고 오류를 발생시킬 수 있습니다. `Selenium`의 알림창 처리 기능을 통해 이러한 방해 요소를 효과적으로 제어할 수 있습니다.
> 
> *   **알림창 유형:**
>     *   `alert()`: 사용자에게 메시지를 보여주고 '확인' 버튼만 있는 단순 알림창.
>     *   `confirm()`: 사용자에게 메시지를 보여주고 '확인' 또는 '취소' 버튼이 있는 확인창.
>     *   `prompt()`: 사용자에게 메시지를 보여주고 텍스트를 입력받을 수 있는 입력창.
> *   **`driver.switch_to.alert`:**
>     *   알림창이 나타나면 `Selenium`의 포커스는 웹 페이지에서 알림창으로 이동합니다. 알림창에 접근하려면 반드시 `driver.switch_to.alert`를 사용해야 합니다. 이 메서드는 `Alert` 객체를 반환합니다.
> *   **`Alert` 객체의 주요 메서드 및 속성:**
>     *   `accept()`: 알림창의 '확인' 버튼을 클릭합니다. (`alert`, `confirm`, `prompt` 모두 사용 가능)
>     *   `dismiss()`: 알림창의 '취소' 버튼을 클릭합니다. (`confirm`, `prompt`에 사용 가능)
>     *   `send_keys(text)`: `prompt` 알림창에 텍스트를 입력합니다.
>     *   `text`: 알림창에 표시된 메시지 텍스트를 가져옵니다.
> *   **명시적 대기 필수:**
>     *   알림창이 나타날 때까지 `WebDriverWait`와 `EC.alert_is_present()`를 사용하여 명시적으로 대기하는 것이 중요합니다. 알림창이 나타나기 전에 접근하려 하면 `NoAlertPresentException` 오류가 발생할 수 있습니다.
> *   **실무 팁:**
>     *   웹 크롤링 중 알림창이 나타날 가능성이 있다면, 항상 `try-except` 블록을 사용하여 `NoAlertPresentException`을 처리하고, 알림창이 나타나면 적절히 `accept()` 또는 `dismiss()`를 호출하여 스크립트가 중단되지 않도록 해야 합니다.
>     *   일부 웹사이트는 커스텀 팝업(HTML/CSS/JavaScript로 구현된 모달 창)을 사용합니다. 이러한 팝업은 `Selenium`의 일반적인 요소 탐색 및 클릭 메서드로 처리해야 하며, `driver.switch_to.alert`로는 접근할 수 없습니다.


### 1.2. iframe 처리

`iframe`(Inline Frame)은 HTML 문서 내에 다른 HTML 문서를 삽입하는 데 사용되는 태그입니다. 웹 페이지 내에 독립적인 웹 페이지를 포함시킬 때 사용되며, 광고, 동영상 플레이어, 댓글 섹션, 결제 모듈 등에서 흔히 볼 수 있습니다. `Selenium`은 기본적으로 메인 문서에만 포커스되어 있으므로, `iframe` 내부에 있는 요소에 접근하려면 먼저 `iframe`으로 포커스를 전환해야 합니다.

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# WebDriver 설정 (이전 섹션 참조)
service = Service(ChromeDriverManager().install())
options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")
driver = webdriver.Chrome(service=service, options=options)

try:
    # 테스트용 HTML 페이지 (iframe 시나리오)
    driver.get("data:text/html,<body><h1>Main Page</h1><iframe id='my-iframe' name='frame-name' srcdoc='<html><body><h2>Iframe Content</h2><p id=\"iframe-text\">This is inside an iframe.</p><button id=\"iframe-btn\">Click Me</button></body></html>" style='width:400px;height:200px;'></iframe></body>")
    time.sleep(1) # 페이지 로딩 대기

    # 1. iframe으로 전환 (ID 사용)
    # driver.switch_to.frame('my-iframe')

    # 2. iframe으로 전환 (Name 사용)
    # driver.switch_to.frame('frame-name')

    # 3. iframe으로 전환 (WebElement 사용 - 가장 권장)
    # iframe 요소를 먼저 찾은 후 전환합니다. 이는 ID나 Name이 없는 iframe에 유용합니다.
    iframe_element = driver.find_element(By.ID, 'my-iframe')
    driver.switch_to.frame(iframe_element)
    print("\nSwitched to iframe.")

    # iframe 내부의 요소 찾기
    iframe_text = driver.find_element(By.ID, 'iframe-text')
    print(f"Iframe Text: {iframe_text.text}")

    iframe_button = driver.find_element(By.ID, 'iframe-btn')
    iframe_button.click()
    print("Iframe button clicked.")
    time.sleep(1)

    # 4. 메인 콘텐츠(default content)로 다시 전환
    driver.switch_to.default_content()
    print("Switched back to default content.")

    # 이제 메인 페이지의 요소에 접근 가능
    main_title = driver.find_element(By.TAG_NAME, 'h1')
    print(f"Main Page Title: {main_title.text}")

finally:
    driver.quit()
```

> **[실무 노트] `iframe` 처리 전략:**
> `iframe`은 웹 크롤링 시 자주 마주치는 요소이며, 이를 올바르게 처리하지 못하면 `NoSuchElementException`과 같은 오류가 발생합니다. `iframe` 내부의 요소에 접근하려면 반드시 `switch_to.frame()`을 통해 포커스를 전환해야 합니다.
> 
> *   **`iframe`의 개념:** `iframe`은 현재 웹 페이지 안에 완전히 독립적인 또 다른 웹 페이지를 삽입하는 것입니다. 따라서 `iframe` 내부의 요소들은 메인 페이지의 DOM과는 별개의 DOM 트리에 속합니다.
> *   **전환 방법:**
>     *   **ID 또는 Name 사용:** `driver.switch_to.frame('iframe_id_or_name')`
>     *   **WebElement 사용 (가장 권장):** `iframe` 요소를 먼저 찾은 후 `driver.switch_to.frame(iframe_element)`를 사용합니다. 이는 `iframe`에 ID나 Name이 없거나 동적으로 생성될 때 유용합니다.
> *   **메인 콘텐츠로 돌아오기:** `iframe` 내부 작업을 마친 후에는 반드시 `driver.switch_to.default_content()`를 호출하여 메인 페이지의 DOM으로 포커스를 다시 전환해야 합니다. 그렇지 않으면 메인 페이지의 요소를 찾을 수 없습니다.
> *   **중첩된 `iframe`:** `iframe` 안에 또 다른 `iframe`이 중첩될 수도 있습니다. 이 경우, 각 `iframe`으로 순차적으로 전환해야 합니다. (예: `driver.switch_to.frame('outer_frame')`, `driver.switch_to.frame('inner_frame')`)
> *   **주의사항:**
>     *   `iframe` 내부의 요소는 메인 페이지의 CSS 선택자나 XPath로 직접 접근할 수 없습니다. 반드시 `iframe`으로 전환한 후에 내부 요소를 찾아야 합니다.
>     *   `iframe`의 `src` 속성을 통해 `iframe`이 로드하는 URL을 확인할 수 있습니다. 때로는 `iframe` 내부의 URL로 직접 `Requests` 요청을 보내는 것이 더 효율적일 수도 있습니다.
> 
> `iframe`은 웹 크롤링의 난이도를 높이는 요소 중 하나이므로, 개발자 도구(F12)를 통해 `iframe`의 존재 여부와 구조를 면밀히 분석하는 것이 중요합니다.

### 1.3. 새 창/탭 처리

웹 페이지에서 링크를 클릭했을 때 새로운 창(Window)이나 탭(Tab)이 열리는 경우가 있습니다. `Selenium`은 기본적으로 하나의 창/탭에만 포커스되어 있으므로, 새로 열린 창/탭의 요소에 접근하려면 해당 창/탭으로 포커스를 전환해야 합니다.

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# WebDriver 설정 (이전 섹션 참조)
service = Service(ChromeDriverManager().install())
options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")
driver = webdriver.Chrome(service=service, options=options)

try:
    # 테스트용 HTML 페이지 (새 창/탭 시나리오)
    driver.get("data:text/html,<body><h1>Main Window</h1><a href='https://www.google.com' target='_blank' id='new-window-link'>Open Google</a></body>")
    time.sleep(1) # 페이지 로딩 대기

    # 현재 창의 핸들 저장
    main_window_handle = driver.current_window_handle
    print(f"\nMain Window Handle: {main_window_handle}")

    # 새 창을 여는 링크 클릭
    new_window_link = driver.find_element(By.ID, 'new-window-link')
    new_window_link.click()
    print("Clicked link to open new window.")

    # 새 창이 열릴 때까지 대기
    WebDriverWait(driver, 10).until(EC.number_of_windows_to_be(2))

    # 열려 있는 모든 창의 핸들 가져오기
    all_window_handles = driver.window_handles
    print(f"All Window Handles: {all_window_handles}")

    # 새 창으로 전환
    new_window_handle = None
    for handle in all_window_handles:
        if handle != main_window_handle:
            new_window_handle = handle
            break

    if new_window_handle:
        driver.switch_to.window(new_window_handle)
        print(f"Switched to new window with title: {driver.title}")
        print(f"Current URL in new window: {driver.current_url}")
        time.sleep(2)

        # 새 창에서 작업 수행 (예: 검색)
        # search_box = driver.find_element(By.NAME, 'q')
        # search_box.send_keys('Selenium')
        # search_box.send_keys(Keys.ENTER)
        # time.sleep(2)

        # 새 창 닫기
        driver.close() # 현재 포커스된 창만 닫습니다.
        print("New window closed.")

    # 메인 창으로 다시 전환
    driver.switch_to.window(main_window_handle)
    print(f"Switched back to main window with title: {driver.title}")
    print(f"Current URL in main window: {driver.current_url}")

finally:
    driver.quit()
```

> **[실무 노트] 새 창/탭 처리 전략:**
> 웹 크롤링 중 새 창이나 탭이 열리는 상황은 흔하며, 이를 올바르게 처리하지 못하면 원하는 데이터를 가져올 수 없습니다. `Selenium`의 창 핸들(Window Handles) 개념을 이해하는 것이 중요합니다.
> 
> *   **창 핸들 (Window Handle):**
>     *   `driver.current_window_handle`: 현재 `WebDriver`가 포커스하고 있는 창(또는 탭)의 고유한 식별자(핸들)를 반환합니다.
>     *   `driver.window_handles`: 현재 열려 있는 모든 창(또는 탭)의 핸들 리스트를 반환합니다. 리스트의 순서는 브라우저마다 다를 수 있으며, 열린 순서와 일치하지 않을 수 있습니다.
> *   **창 전환 (`driver.switch_to.window()`):**
>     *   새 창이나 탭의 요소에 접근하려면, `driver.switch_to.window(handle)` 메서드를 사용하여 해당 창의 핸들로 포커스를 전환해야 합니다.
>     *   새 창이 열릴 때까지 `WebDriverWait`와 `EC.number_of_windows_to_be()`를 사용하여 명시적으로 대기하는 것이 안정적입니다.
> *   **창 닫기 (`driver.close()` vs `driver.quit()`):**
>     *   `driver.close()`: 현재 `WebDriver`가 포커스하고 있는 창(또는 탭)만 닫습니다. `WebDriver` 세션 자체는 유지됩니다.
>     *   `driver.quit()`: 모든 열려 있는 창을 닫고 `WebDriver` 세션을 완전히 종료합니다. 크롤링 작업이 완료되면 항상 `driver.quit()`을 호출하여 모든 브라우저 프로세스와 자원을 해제해야 합니다.
> *   **실무 팁:**
>     *   새 창이 열리는 링크를 클릭하기 전에 현재 창의 핸들을 저장해두는 것이 좋습니다. (`main_window_handle = driver.current_window_handle`)
>     *   새 창으로 전환한 후 작업을 수행하고, 해당 창을 닫은 다음에는 반드시 `driver.switch_to.window(main_window_handle)`를 사용하여 원래 창으로 돌아와야 합니다.
>     *   팝업 광고나 불필요한 새 창이 열리는 경우, 해당 창으로 전환하여 닫아버리는(`driver.close()`) 방식으로 처리할 수 있습니다.
> 
> 새 창/탭 처리는 웹 크롤링의 복잡성을 높이는 요소이므로, 웹 페이지의 동작 방식을 면밀히 분석하고 안정적인 전환 로직을 구현하는 것이 중요합니다.
