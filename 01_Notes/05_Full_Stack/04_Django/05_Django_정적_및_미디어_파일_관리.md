<h2>Django Backend: 정적 및 미디어 파일 관리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django 프로젝트에서 정적 파일(CSS, JS, 이미지)과 미디어 파일(사용자 업로드 파일)을 효과적으로 관리하고 제공하는 방법을 이해하는 것을 목표로 합니다. 개발 환경과 운영 환경에서의 설정 차이를 포함하여 실무적인 파일 관리 전략을 학습합니다.</p>

<h2>목차</h2> 

- [1. 정적 파일(Static Files) 관리](#1-정적-파일static-files-관리)
- [2. 미디어 파일(Media Files) 관리](#2-미디어-파일media-files-관리)
  - [2.1. `settings.py` 설정](#21-settingspy-설정)
  - [2.2. `urls.py` 설정](#22-urlspy-설정)
  - [2.3. 모델에 파일 필드 추가](#23-모델에-파일-필드-추가)
  - [2.4. 템플릿에서 미디어 파일 사용](#24-템플릿에서-미디어-파일-사용)

---

## 1. 정적 파일(Static Files) 관리

정적 파일(Static Files)은 웹 페이지를 구성하는 CSS, JavaScript, 이미지 파일 등을 의미합니다. Django는 이러한 정적 파일들을 효율적으로 관리하고 제공하기 위한 시스템을 제공합니다.

*   **`settings.py`의 `STATIC_URL`**: 정적 파일에 접근할 때 사용할 URL 접두사를 정의합니다. (예: `/static/`)
*   **개발 서버에서의 제공**: 개발 환경에서는 `DEBUG = True`일 때 Django 개발 서버가 자동으로 정적 파일을 제공합니다. 각 앱의 `static/` 디렉토리나 `settings.py`의 `STATICFILES_DIRS`에 지정된 경로에서 정적 파일을 찾습니다.
*   **템플릿에서 사용**: 템플릿에서 정적 파일을 사용하려면 `{% load static %}` 태그를 사용한 후 `{% static 'path/to/your/file.css' %}`와 같이 참조합니다.

(참고: 제공된 `02_practice` 디렉토리에는 정적 파일을 직접적으로 사용하는 템플릿 예시가 명확하게 보이지 않지만, Django 프로젝트에서 정적 파일 관리는 필수적인 개념입니다.)

```html
<!-- myapp/templates/myapp/base.html (예시) -->
{% load static %} {# 템플릿 상단에 추가 #}

<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Django App</title>
    {# CSS 파일 로드 예시 #}
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
    {# 이미지 파일 경로 예시 #}
    <img src="{% static 'images/logo.png' %}" alt="Logo">
</head>
<body>
    <!-- 페이지 내용 -->
    {# JavaScript 파일 로드 예시 (</body> 태그 직전) #}
    <script src="{% static 'js/main.js' %}"></script>
</body>
</html>
```

## 2. 미디어 파일(Media Files) 관리

미디어 파일(Media Files)은 사용자가 웹 애플리케이션에 업로드하는 파일(예: 프로필 사진, 첨부 문서, 동영상)을 의미합니다. 정적 파일과 달리 미디어 파일은 사용자에 의해 생성되며, 데이터베이스에 파일 경로가 저장되고 실제 파일은 서버의 특정 디렉토리에 저장됩니다. Django는 이러한 미디어 파일을 관리하기 위한 설정을 제공합니다.

### 2.1. `settings.py` 설정

미디어 파일을 관리하려면 `settings.py`에 `MEDIA_ROOT`와 `MEDIA_URL`을 정의해야 합니다.

*   **`MEDIA_ROOT`**: 업로드된 미디어 파일이 서버의 파일 시스템에 저장될 절대 경로를 지정합니다. 이 디렉토리는 Django가 파일을 저장할 수 있도록 쓰기 권한이 있어야 합니다.
    ```python
    # settings.py
    import os
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    MEDIA_URL = '/media/'
    MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
    ```
    *   `BASE_DIR`은 프로젝트의 루트 디렉토리를 가리킵니다. 위 설정은 프로젝트 루트에 `media/` 디렉토리를 생성하고 그 안에 업로드된 파일을 저장하도록 합니다.

*   **`MEDIA_URL`**: 업로드된 미디어 파일에 접근할 때 사용할 URL 접두사를 정의합니다. 웹 브라우저가 이 URL을 통해 서버에 미디어 파일을 요청하게 됩니다.

### 2.2. `urls.py` 설정

개발 환경(`DEBUG=True`)에서는 Django 개발 서버가 미디어 파일을 제공할 수 있도록 `urls.py`에 설정을 추가해야 합니다. 운영 환경에서는 웹 서버(Nginx, Apache)가 직접 미디어 파일을 제공하도록 설정합니다.

*   **프로젝트 `urls.py` (예시):
    ```python
    # project_name/urls.py
    from django.contrib import admin
    from django.urls import path, include
    from django.conf import settings # settings 임포트
    from django.conf.urls.static import static # static 함수 임포트

    urlpatterns = [
        path('admin/', admin.site.urls),
        # ... 다른 URL 패턴들 ...
    ]

    # 개발 환경에서만 미디어 파일을 서빙하도록 설정
    if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    ```

### 2.3. 모델에 파일 필드 추가

모델에 `FileField` 또는 `ImageField`를 추가하여 파일을 업로드하고 관리할 수 있습니다.

*   **`models.FileField()`**: 모든 종류의 파일을 저장할 수 있습니다.
*   **`models.ImageField()`**: 이미지 파일만 저장할 수 있으며, 이미지 유효성 검사 기능이 내장되어 있습니다. `Pillow` 라이브러리가 필요합니다 (`pip install Pillow`).

*   **예시 (`app_name/models.py`):
    ```python
    from django.db import models

    class MyModel(models.Model):
        title = models.CharField(max_length=100)
        # 'uploads/' 디렉토리 아래에 파일이 저장됩니다 (MEDIA_ROOT 기준)
        my_file = models.FileField(upload_to='uploads/') 
        my_image = models.ImageField(upload_to='images/')

        def __str__(self):
            return self.title
    ```
    *   `upload_to`: `MEDIA_ROOT`를 기준으로 파일이 저장될 하위 디렉토리를 지정합니다.

*   **마이그레이션 실행:**
    모델을 변경했으므로 마이그레이션을 생성하고 적용해야 합니다.
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

### 2.4. 템플릿에서 미디어 파일 사용

템플릿에서 모델 인스턴스의 파일 필드에 접근하여 업로드된 파일의 URL을 가져올 수 있습니다.

*   **예시 (`app_name/templates/app_name/detail.html`):
    ```html
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ object.title }}</n>
    </head>
    <body>
        <h1>{{ object.title }}</h1>
        {% if object.my_file %}
            <p><a href="{{ object.my_file.url }}">다운로드 파일</a></p>
        {% endif %}
        {% if object.my_image %}
            <p><img src="{{ object.my_image.url }}" alt="{{ object.title }}" width="300"></p>
        {% endif %}
    </body>
    </html>
    ```
    *   `{{ object.my_file.url }}`: 업로드된 파일의 URL을 반환합니다. 이 URL은 `MEDIA_URL`과 `upload_to` 경로를 조합하여 생성됩니다.

*   **폼을 통한 파일 업로드:**
    파일 업로드를 위한 폼을 만들 때는 `ModelForm`을 사용하고, HTML 폼 태그에 `enctype="multipart/form-data"` 속성을 반드시 추가해야 합니다.
    ```html
    <form method="post" enctype="multipart/form-data">
        {% csrf_token %}
        {{ form.as_p }}
        <button type="submit">업로드</button>
    </form>
    ```
    *   뷰에서는 `request.FILES`를 통해 업로드된 파일에 접근할 수 있습니다. `ModelForm`을 사용하면 `form = MyModelForm(request.POST, request.FILES)`와 같이 `request.FILES`를 전달하여 쉽게 처리할 수 있습니다.
