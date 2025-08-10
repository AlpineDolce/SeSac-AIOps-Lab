<h2>데이터 크롤링 도구: Selenium, BeautifulSoup 핵심 가이드</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-01

<h2>문서 목표</h2>
이 문서는 `Selenium`을 사용하여 동적인 웹 페이지의 콘텐츠를 효과적으로 로드하고 처리하는 방법을 다룹니다. 페이지 스크롤, 명시적/암시적 대기, 그리고 동적으로 로딩되는 콘텐츠를 기다리는 방법을 학습하여 안정적인 크롤링을 수행하는 데 도움이 되기를 바랍니다.

<h2>목차</h2>

- [1. Selenium 페이지 스크롤 및 대기](#1-selenium-페이지-스크롤-및-대기)
  - [1.1. 페이지 스크롤](#11-페이지-스크롤)
  - [1.2. 명시적 대기 (Explicit Waits)](#12-명시적-대기-explicit-waits)
  - [1.3. 암시적 대기 (Implicit Waits)](#13-암시적-대기-implicit-waits)
  - [1.4. 동적 콘텐츠 로딩 처리](#14-동적-콘텐츠-로딩-처리)

---

## 1. Selenium 페이지 스크롤 및 대기

### 1.1. 페이지 스크롤

많은 웹 페이지, 특히 소셜 미디어 피드나 뉴스 사이트 등은 사용자가 페이지를 아래로 스크롤할 때 새로운 콘텐츠를 동적으로 로드하는 '무한 스크롤' 방식을 사용합니다. `Selenium`은 JavaScript 코드를 실행하여 페이지를 스크롤할 수 있게 해줍니다.

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# WebDriver 설정 (이전 섹션 참조)
service = Service(ChromeDriverManager().install())
options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")
driver = webdriver.Chrome(service=service, options=options)

try:
    # 테스트용 웹 페이지 (실제 무한 스크롤 페이지라고 가정)
    # 실제 웹사이트 URL로 대체하여 테스트하세요.
    driver.get("https://www.naver.com") # 예시 URL, 실제 무한 스크롤 페이지로 변경 필요
    time.sleep(2) # 페이지 로딩 대기

    # 1. 특정 위치로 스크롤 (JavaScript 실행)
    # window.scrollTo(x, y) - x: 가로 스크롤, y: 세로 스크롤
    driver.execute_script("window.scrollTo(0, 500);") # 세로로 500px 스크롤
    print("\nScrolled down 500px.")
    time.sleep(2)

    # 2. 페이지 끝까지 스크롤
    # window.innerHeight: 현재 뷰포트(화면)의 높이
    # document.body.scrollHeight: 문서 전체의 높이
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    print("Scrolled to the bottom of the page.")
    time.sleep(2)

    # 3. 무한 스크롤 페이지 처리 (반복문 사용)
    # 새로운 콘텐츠가 로드될 때까지 페이지 끝까지 스크롤하는 것을 반복합니다.
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2) # 새로운 콘텐츠 로드를 위한 대기
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break # 더 이상 스크롤할 내용이 없으면 종료
        last_height = new_height
        print("Scrolled down, new content loaded.")

    print("\nFinished scrolling infinite scroll page.")

finally:
    driver.quit()
```

> **[실무 노트] 페이지 스크롤링 전략:**
> 무한 스크롤 페이지는 `Selenium`을 사용하는 주된 이유 중 하나입니다. 스크롤링 전략을 잘 세우는 것이 모든 데이터를 빠짐없이 가져오는 데 중요합니다.
> 
> *   **`execute_script()` 메서드:**
>     *   `Selenium`은 `driver.execute_script()` 메서드를 통해 웹 브라우저에서 직접 JavaScript 코드를 실행할 수 있게 해줍니다. 이를 통해 브라우저의 DOM(Document Object Model)을 직접 조작하거나, 브라우저의 내장 함수(예: `window.scrollTo()`)를 호출할 수 있습니다.
> *   **무한 스크롤 처리:**
>     *   가장 일반적인 방법은 페이지의 현재 높이(`document.body.scrollHeight`)를 확인하고, 페이지 끝까지 스크롤한 후 잠시 대기하여 새로운 콘텐츠가 로드되기를 기다리는 것입니다. 새로운 높이가 이전 높이와 같으면 더 이상 로드될 콘텐츠가 없다고 판단하여 반복문을 종료합니다.
>     *   **주의:** `time.sleep()`은 무조건적인 대기이므로, 페이지 로딩 시간에 따라 비효율적이거나 부족할 수 있습니다. 다음 섹션에서 다룰 **명시적 대기(Explicit Waits)**를 사용하여 특정 요소가 나타날 때까지 기다리는 것이 더 안정적이고 효율적입니다.
> *   **특정 요소까지 스크롤:**
>     *   `element.location_once_scrolled_into_view` 속성을 사용하면 특정 요소가 뷰포트(화면)에 보일 때까지 스크롤할 수 있습니다. 이는 특정 섹션이나 버튼까지 스크롤해야 할 때 유용합니다.
> *   **성능 고려:**
>     *   너무 빠르게 스크롤하거나 너무 자주 요청을 보내면 웹사이트의 봇 탐지 시스템에 의해 차단될 수 있습니다. 요청 간 적절한 지연 시간을 두는 것이 중요합니다.
>     *   모든 데이터를 스크롤하여 가져오는 것이 비효율적일 경우, 페이지네이션(Pagination)이 있다면 페이지네이션을 이용하는 것이 더 좋습니다.


### 1.2. 명시적 대기 (Explicit Waits)

웹 페이지는 JavaScript 실행이나 AJAX 호출 등으로 인해 콘텐츠가 동적으로 로드되는 경우가 많습니다. `Selenium`으로 요소를 찾거나 상호작용하기 전에, 해당 요소가 웹 페이지에 완전히 로드되고 상호작용 가능한 상태가 될 때까지 기다려야 합니다. `time.sleep()`과 같은 무조건적인 대기는 비효율적이거나 부족할 수 있습니다. **명시적 대기(Explicit Waits)**는 특정 조건이 충족될 때까지 `Selenium`이 기다리도록 지시하는 가장 안정적이고 효율적인 대기 방법입니다.

`Selenium`은 `WebDriverWait` 클래스와 `expected_conditions` 모듈을 통해 명시적 대기를 지원합니다.

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait # WebDriverWait 클래스 임포트
from selenium.webdriver.support import expected_conditions as EC # expected_conditions 모듈 임포트
import time

# WebDriver 설정 (이전 섹션 참조)
service = Service(ChromeDriverManager().install())
options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")
driver = webdriver.Chrome(service=service, options=options)

try:
    # 테스트용 웹 페이지 (실제 동적 로딩 페이지라고 가정)
    # 버튼 클릭 후 텍스트가 동적으로 나타나는 시나리오
    driver.get("data:text/html,<button id='load-btn'>Load Content</button><div id='content' style='display:none;'>Dynamic Content Loaded!</div><script>document.getElementById('load-btn').onclick = function() { setTimeout(function() { document.getElementById('content').style.display='block'; }, 2000); };</script>")
    time.sleep(1) # 페이지 로딩 대기

    load_button = driver.find_element(By.ID, 'load-btn')
    load_button.click()
    print("\nLoad button clicked. Waiting for content...")

    # 1. 특정 요소가 나타날 때까지 명시적 대기
    # WebDriverWait(driver, timeout).until(expected_conditions.condition)
    # EC.visibility_of_element_located: 요소가 DOM에 있고 화면에 보일 때까지 대기
    try:
        dynamic_content = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'content'))
        )
        print(f"Dynamic content appeared: {dynamic_content.text}")
    except Exception as e:
        print(f"Error waiting for content: {e}")

    # 2. 클릭 가능한 요소가 될 때까지 대기
    # EC.element_to_be_clickable: 요소가 클릭 가능할 때까지 대기
    driver.get("data:text/html,<button id='btn' disabled>Disabled Button</button><script>setTimeout(function() { document.getElementById('btn').disabled=false; }, 3000);</script>")
    print("\nWaiting for button to be clickable...")
    try:
        clickable_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'btn'))
        )
        clickable_button.click()
        print("Button is now clickable and clicked.")
    except Exception as e:
        print(f"Error waiting for button: {e}")

finally:
    driver.quit()
```

> **[실무 노트] 명시적 대기: 안정적인 크롤링의 핵심:**
> `Selenium` 크롤링에서 가장 흔한 오류 중 하나는 웹 요소가 아직 로드되지 않았거나 상호작용할 준비가 되지 않았을 때 요소를 찾거나 클릭하려 할 때 발생하는 `NoSuchElementException` 또는 `ElementNotInteractableException`입니다. 명시적 대기는 이러한 문제를 해결하고 크롤링 스크립트의 안정성을 획기적으로 높여줍니다.
>
> *   **`time.sleep()`과의 차이점:**
>     *   **`time.sleep(초)`:** 무조건 지정된 시간만큼 대기합니다. 페이지 로딩이 빠르면 불필요한 시간 낭비이고, 느리면 요소가 로드되기 전에 다음 코드가 실행되어 오류가 발생할 수 있습니다. **비효율적이고 불안정합니다.**
>     *   **명시적 대기:** 특정 조건이 충족될 때까지 최대 지정된 시간(timeout) 동안 대기합니다. 조건이 충족되면 즉시 다음 코드를 실행하고, 시간 초과 시에만 예외를 발생시킵니다. **효율적이고 안정적입니다.**
>
> *   **`WebDriverWait` 클래스:**
>     *   `WebDriverWait(driver, timeout)`: `driver`는 WebDriver 객체, `timeout`은 최대 대기 시간(초)입니다.
>     *   `until()` 또는 `until_not()` 메서드와 함께 `expected_conditions`를 사용합니다.
>
> *   **주요 `expected_conditions` (EC) 예시:**
>     *   `EC.presence_of_element_located((By.LOCATOR, "value"))`: 요소가 DOM(Document Object Model)에 존재할 때까지 대기합니다. (화면에 보이지 않아도 됨)
>     *   `EC.visibility_of_element_located((By.LOCATOR, "value"))`: 요소가 DOM에 존재하고 화면에 보일 때까지 대기합니다.
>     *   `EC.element_to_be_clickable((By.LOCATOR, "value"))`: 요소가 클릭 가능할 때까지 대기합니다.
>     *   `EC.text_to_be_present_in_element((By.LOCATOR, "value"), "text")`: 특정 요소에 특정 텍스트가 나타날 때까지 대기합니다.
>     *   `EC.invisibility_of_element_located((By.LOCATOR, "value"))`: 요소가 화면에서 사라질 때까지 대기합니다. (로딩 스피너 등)
>
> *   **실무 팁:**
>     *   요소와 상호작용하기 전에는 항상 명시적 대기를 사용하여 요소가 준비될 때까지 기다려야 합니다.
>     *   `try-except` 블록을 사용하여 `TimeoutException`을 처리하면, 요소가 나타나지 않을 경우 스크립트가 중단되지 않고 다음 로직을 수행할 수 있습니다.
>     *   `expected_conditions`는 매우 다양하므로, 크롤링하려는 웹 페이지의 동적 로딩 특성을 파악하여 가장 적합한 조건을 선택해야 합니다.

### 1.3. 암시적 대기 (Implicit Waits)

**암시적 대기(Implicit Waits)**는 `Selenium`이 요소를 찾을 때, 지정된 시간(timeout) 동안 DOM(Document Object Model)에서 해당 요소가 나타날 때까지 기다리도록 설정하는 전역적인 대기 방식입니다. 한 번 설정하면 `WebDriver`의 수명 주기 동안 모든 `find_element()` 또는 `find_elements()` 호출에 적용됩니다.

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# WebDriver 설정 (이전 섹션 참조)
service = Service(ChromeDriverManager().install())
options = Options()
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36")
driver = webdriver.Chrome(service=service, options=options)

try:
    # 암시적 대기 설정 (최대 10초)
    # 요소를 찾을 때 최대 10초까지 기다립니다.
    driver.implicitly_wait(10)
    print("\nImplicit wait set to 10 seconds.")

    # 테스트용 웹 페이지 (요소가 늦게 나타나는 시나리오)
    driver.get("data:text/html,<div id='delayed-div' style='display:none;'>Delayed Content</div><script>setTimeout(function() { document.getElementById('delayed-div').style.display='block'; }, 3000);</script>")
    print("Page loaded. Trying to find delayed element...")

    # 암시적 대기 덕분에 요소가 나타날 때까지 기다립니다.
    # 요소가 3초 후에 나타나므로, 3초 정도 대기 후 요소를 찾습니다.
    delayed_element = driver.find_element(By.ID, 'delayed-div')
    print(f"Delayed element found: {delayed_element.text}")

    # 존재하지 않는 요소를 찾으려 할 때 (최대 10초 대기 후 NoSuchElementException 발생)
    print("\nTrying to find non-existent element...")
    start_time = time.time()
    try:
        non_existent_element = driver.find_element(By.ID, 'non-existent')
        print(f"Found non-existent element: {non_existent_element.text}")
    except Exception as e:
        end_time = time.time()
        print(f"Error: {e} (took {end_time - start_time:.2f} seconds)")

finally:
    driver.quit()
```

> **[실무 노트] 암시적 대기: 편리함과 한계:**
> 암시적 대기는 `Selenium` 스크립트를 작성할 때 편리함을 제공하지만, 그 한계를 명확히 이해하고 명시적 대기와 함께 사용하는 것이 중요합니다.
> 
> *   **동작 방식:**
>     *   `driver.implicitly_wait(seconds)`로 설정합니다.
>     *   `find_element()` 또는 `find_elements()` 호출 시, 요소가 즉시 발견되지 않으면 지정된 시간 동안 요소를 계속해서 재탐색합니다. 요소가 발견되면 즉시 다음 코드를 실행하고, 시간 초과 시 `NoSuchElementException`을 발생시킵니다.
> *   **명시적 대기(`Explicit Waits`)와의 차이점:**
>     *   **적용 범위:** 암시적 대기는 모든 요소 탐색에 전역적으로 적용됩니다. 명시적 대기는 특정 조건에 대해 특정 요소에만 적용됩니다.
>     *   **조건:** 암시적 대기는 요소가 DOM에 '존재'하는지 여부만 확인합니다. 요소가 존재하더라도 화면에 보이지 않거나 클릭 불가능한 상태일 수 있습니다. 명시적 대기는 요소의 가시성, 클릭 가능성 등 더 구체적인 조건을 기다릴 수 있습니다.
>     *   **효율성:** 암시적 대기는 요소가 나타나지 않아도 최대 대기 시간 전체를 기다릴 수 있어 비효율적일 수 있습니다. 명시적 대기는 조건이 충족되면 즉시 다음 단계로 넘어가므로 더 효율적입니다.
> *   **실무적 권장 사항:**
>     *   **명시적 대기 우선:** 대부분의 경우, 특정 요소의 상태(가시성, 클릭 가능성 등)를 정확히 기다려야 하는 상황에서는 **명시적 대기**를 사용하는 것이 더 안정적이고 효율적입니다.
>     *   **암시적 대기 보조:** 암시적 대기는 페이지 로딩 후 DOM에 요소가 나타나는 일반적인 상황에 대한 '최소한의 안전망'으로 설정할 수 있습니다. 하지만 복잡한 동적 로딩이나 상호작용이 필요한 경우에는 암시적 대기만으로는 부족합니다.
>     *   **혼용 시 주의:** 암시적 대기와 명시적 대기를 함께 사용할 경우, `Selenium`은 두 대기 시간 중 더 긴 시간을 기다릴 수 있습니다. 이는 예상치 못한 지연을 초래할 수 있으므로, 일반적으로는 **명시적 대기를 주로 사용하고 암시적 대기는 0으로 설정하거나 사용하지 않는 것을 권장**합니다.
>
> 안정적이고 효율적인 `Selenium` 크롤링을 위해서는 `time.sleep()`을 지양하고, 명시적 대기를 우선적으로 사용하며, 암시적 대기는 보조적인 역할로 활용하는 전략이 필요합니다.

### 1.4. 동적 콘텐츠 로딩 처리

현대의 웹 페이지는 사용자 경험을 향상시키기 위해 JavaScript를 사용하여 콘텐츠를 동적으로 로드하는 경우가 많습니다. `Selenium`은 이러한 동적 콘텐츠를 처리하는 데 필수적인 도구입니다. 동적 콘텐츠 로딩의 주요 유형과 처리 방법을 이해하는 것이 중요합니다.

*   **동적 콘텐츠 로딩의 주요 유형:**
    *   **AJAX (Asynchronous JavaScript and XML) 호출:** 페이지 전체를 새로고침하지 않고, 백그라운드에서 서버와 통신하여 데이터를 가져와 페이지의 특정 부분만 업데이트합니다. (예: 댓글 로딩, 상품 필터링 결과)
    *   **무한 스크롤 (Infinite Scroll):** 사용자가 페이지를 아래로 스크롤할 때 새로운 콘텐츠를 자동으로 로드합니다. (예: 소셜 미디어 피드, 블로그 게시물 목록)
    *   **지연 로딩 (Lazy Loading):** 이미지나 비디오와 같이 용량이 큰 콘텐츠를 사용자가 해당 콘텐츠가 있는 뷰포트(화면)까지 스크롤하기 전까지는 로드하지 않습니다. (예: 긴 이미지 갤러리)
    *   **탭/버튼 클릭 시 콘텐츠 로딩:** 특정 탭이나 버튼을 클릭했을 때만 관련 콘텐츠가 로드되는 경우.

*   **`Selenium`을 이용한 동적 콘텐츠 처리 방법:**
    1.  **명시적 대기 (Explicit Waits) 활용:**
        가장 중요하고 안정적인 방법입니다. 특정 요소가 나타나거나, 사라지거나, 클릭 가능해질 때까지 기다리도록 설정합니다. `WebDriverWait`와 `expected_conditions`를 사용하여 구현합니다. (자세한 내용은 [1.2. 명시적 대기 (Explicit Waits)](#12-명시적-대기-explicit-waits) 참조)
        ```python
        # 예시: 특정 div가 나타날 때까지 대기
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By

        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "dynamic-content"))
            )
            print(f"Dynamic content loaded: {element.text}")
        except TimeoutException:
            print("Dynamic content did not load within 10 seconds.")
        ```
    2.  **페이지 스크롤 (JavaScript 실행):**
        무한 스크롤 페이지의 경우, JavaScript를 사용하여 페이지를 아래로 스크롤하여 새로운 콘텐츠를 로드합니다. (자세한 내용은 [1.1. 페이지 스크롤](#11-페이지-스크롤) 참조)
        ```python
        # 예시: 페이지 끝까지 스크롤
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        ```
    3.  **JavaScript 실행을 통한 데이터 직접 가져오기:**
        때로는 웹 페이지의 JavaScript 변수에 필요한 데이터가 JSON 형태로 저장되어 있는 경우가 있습니다. 이 경우 `execute_script()`를 사용하여 해당 JavaScript 변수의 값을 직접 가져올 수 있습니다.
        ```python
        # 예시: JavaScript 변수에서 데이터 가져오기
        # 웹 페이지에 var myData = { "key": "value" }; 와 같은 변수가 있다고 가정
        try:
            data = driver.execute_script("return myData;")
            print(f"Data from JavaScript variable: {data}")
        except Exception as e:
            print(f"Could not retrieve data from JavaScript variable: {e}")
        ```
    4.  **네트워크 요청 모니터링 (고급):**
        `Selenium` 자체만으로는 브라우저의 네트워크 요청을 직접 모니터링하기 어렵습니다. 하지만 `Selenium`과 `BrowserMob Proxy` 또는 `Selenium Wire`와 같은 도구를 함께 사용하면, 브라우저가 서버와 주고받는 HTTP/HTTPS 요청을 가로채어 동적으로 로드되는 JSON 데이터 등을 직접 추출할 수 있습니다. 이는 매우 강력하지만 설정이 복잡합니다.

> **[실무 노트] 동적 콘텐츠 크롤링 전략:**
> 동적 콘텐츠를 크롤링할 때는 웹 페이지의 동작 방식을 정확히 이해하고, 가장 효율적이고 안정적인 방법을 선택하는 것이 중요합니다.
>
> *   **페이지 분석:** 크롤링하려는 웹 페이지에서 데이터가 어떻게 로드되는지(AJAX, 스크롤, 버튼 클릭 등) Chrome 개발자 도구의 Network 탭을 통해 면밀히 분석합니다. 어떤 요청이 발생하고, 어떤 응답(JSON, HTML 조각)이 오는지 확인합니다.
> *   **우선순위:**
>     1.  **API 호출:** 만약 동적으로 로드되는 데이터가 실제로는 API 호출을 통해 JSON 형태로 제공된다면, `Requests` 라이브러리를 사용하여 해당 API를 직접 호출하는 것이 `Selenium`을 사용하는 것보다 훨씬 빠르고 효율적입니다. (개발자 도구의 Network 탭에서 XHR 요청 확인)
>     2.  **명시적 대기:** API 호출이 불가능하거나, JavaScript 실행 후 HTML이 변경되는 경우 `Selenium`의 명시적 대기를 사용하여 요소가 나타날 때까지 기다립니다.
>     3.  **JavaScript 실행:** 페이지 스크롤, 특정 JavaScript 함수 호출, 또는 JavaScript 변수에서 직접 데이터를 가져와야 할 때 `execute_script()`를 사용합니다.
> *   **안정성 확보:** 동적 콘텐츠 로딩은 타이밍에 민감하므로, `time.sleep()` 대신 명시적 대기를 사용하여 스크립트의 안정성을 확보해야 합니다. 또한, `try-except` 블록을 사용하여 `TimeoutException`과 같은 예외를 처리하는 것이 좋습니다.
>
> 동적 콘텐츠 크롤링은 정적 크롤링보다 복잡하지만, `Selenium`의 강력한 기능을 통해 대부분의 웹 페이지에서 데이터를 성공적으로 수집할 수 있습니다.
