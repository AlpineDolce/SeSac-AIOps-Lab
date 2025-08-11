<h2>Django Backend: 정적 및 미디어 파일 관리</h2>
작성자: Alpine_Dolce&nbsp;&nbsp;|&nbsp;&nbsp;날짜: 2025-07-04

<h2>문서 목표</h2>
<p>이 문서는 Django 프로젝트에서 정적 파일(CSS, JS, 이미지)과 미디어 파일(사용자 업로드 파일)을 효과적으로 관리하고 제공하는 방법을 이해하는 것을 목표로 합니다. 개발 환경과 운영 환경에서의 설정 차이를 포함하여 실무적인 파일 관리 전략을 학습합니다.</p>

<h2>목차</h2> 

- [1. 정적 파일(Static Files) 관리: 실무 가이드](#1-정적-파일static-files-관리-실무-가이드)
  - [1.1. 개발 환경 설정](#11-개발-환경-설정)
  - [1.2. 운영 환경 설정 (Production)](#12-운영-환경-설정-production)
  - [1.3. 실무 Tip: `Whitenoise`를 이용한 간편한 정적 파일 관리](#13-실무-tip-whitenoise를-이용한-간편한-정적-파일-관리)
- [2. 미디어 파일(Media Files) 관리: 실무 가이드](#2-미디어-파일media-files-관리-실무-가이드)
  - [2.1. 개발 환경 설정](#21-개발-환경-설정)
  - [2.2. 운영 환경 설정 (Production): 클라우드 스토리지 활용](#22-운영-환경-설정-production-클라우드-스토리지-활용)
  - [2.3. 모델에 파일 필드 추가](#23-모델에-파일-필드-추가)
  - [2.4. 템플릿에서 미디어 파일 사용](#24-템플릿에서-미디어-파일-사용)
  - [2.5. 폼을 통한 파일 업로드](#25-폼을-통한-파일-업로드)

---

## 1. 정적 파일(Static Files) 관리: 실무 가이드

정적 파일은 웹사이트의 디자인과 기능을 담당하는 핵심 자산으로, 개발자가 프로젝트와 함께 배포하는 파일들을 의미합니다. 예를 들어, 직접 작성한 `style.css`, `main.js` 파일이나 프로젝트 로고 이미지, 웹 폰트 등이 모두 정적 파일에 해당합니다. 사용자가 업로드하는 미디어 파일과는 명확히 구분하여 관리해야 합니다.

### 1.1. 개발 환경 설정

개발 환경(`DEBUG=True`)에서 Django는 정적 파일을 자동으로 서빙해주는 편리한 기능을 제공합니다. 이를 위해 몇 가지 주요 설정을 이해해야 합니다.

- **`STATIC_URL = '/static/'`**: 템플릿에서 정적 파일을 참조할 때 사용할 URL의 접두사(prefix)입니다. `{% static 'css/style.css' %}`는 `/static/css/style.css` 라는 URL을 생성합니다.

- **`STATICFILES_DIRS`**: Django가 각 앱의 `static/` 디렉토리 외에 추가적으로 정적 파일을 탐색할 경로를 지정하는 리스트입니다. 보통 프로젝트 전반에서 사용되는 공통 CSS, JS, 이미지 파일들을 이곳에 지정된 디렉토리에 보관합니다.
    ```python
    # settings.py
    STATICFILES_DIRS = [
        BASE_DIR / "static",
    ]
    ```
    위 설정은 프로젝트 최상위 폴더에 있는 `static` 디렉토리를 정적 파일 경로로 추가합니다.

- **앱별 정적 파일 관리**: 각 앱은 자신만의 정적 파일을 가질 수 있습니다. 다른 앱과의 경로 충돌을 방지하기 위해 `app_name/static/app_name/` 과 같은 구조로 파일을 저장하는 것이 Django의 공식 권장 방식입니다. 예를 들어 `blog` 앱의 CSS 파일은 `blog/static/blog/style.css` 에 위치시킵니다.
    - 이렇게 하면 템플릿에서 `{% static 'blog/style.css' %}` 와 같이 명확하게 파일을 참조할 수 있습니다.

### 1.2. 운영 환경 설정 (Production)

**가장 중요한 점**: 운영 환경(`DEBUG=False`)에서는 Django 개발 서버가 정적 파일을 더 이상 자동으로 서빙하지 않습니다. 이는 보안과 성능상의 이유이며, 정적 파일 서빙은 Nginx나 Apache 같은 전문 웹 서버에 위임해야 합니다.

이 과정을 위해 **`collectstatic`** 명령어를 사용합니다.

- **`STATIC_ROOT`**: `collectstatic` 명령을 실행했을 때, 프로젝트 전체에 흩어져 있는 모든 정적 파일들이 최종적으로 수집될 단일 디렉토리의 절대 경로입니다. 이 경로는 비어 있어야 하며, `STATICFILES_DIRS` 와는 다른 경로여야 합니다.
    ```python
    # settings.py
    # 이 경로는 운영 서버에서 웹 서버가 참조하게 될 경로입니다.
    STATIC_ROOT = BASE_DIR / 'staticfiles'
    ```

- **`collectstatic` 실행**: 배포 과정에서 다음 명령어를 실행합니다.
    ```bash
    python manage.py collectstatic
    ```
    이 명령은 `STATICFILES_DIRS` 와 각 앱의 `static` 디렉토리를 포함한 모든 경로를 순회하며 정적 파일들을 `STATIC_ROOT`에 복사합니다.

- **웹 서버(Nginx) 설정**: `collectstatic` 실행 후, 웹 서버가 `/static/` URL로 들어오는 요청을 `STATIC_ROOT` 디렉토리에서 직접 처리하도록 설정합니다. 이렇게 하면 정적 파일 요청이 Django 애플리케이션까지 도달하지 않아 서버 부하가 크게 줄어듭니다.

    **Nginx 설정 예시:**
    ```nginx
    location /static/ {
        alias /path/to/your/project/staticfiles/;
    }

    location /media/ { # 미디어 파일도 동일한 방식으로 처리
        alias /path/to/your/project/media/;
    }
    ```

### 1.3. 실무 Tip: `Whitenoise`를 이용한 간편한 정적 파일 관리

Nginx 등의 웹 서버를 직접 설정하기 어려운 환경(Heroku, Docker 등)이거나, 설정을 단순화하고 싶을 때 `Whitenoise` 라이브러리는 매우 훌륭한 대안입니다.

- **`Whitenoise`란?**: Django 애플리케이션이 직접, 하지만 매우 효율적으로 정적 파일을 서빙할 수 있도록 해주는 미들웨어입니다. Gzip 압축, 캐싱 헤더 설정 등을 자동으로 처리하여 성능 저하를 최소화합니다.

- **설정 방법**:
    1.  **설치**: `pip install whitenoise`
    2.  **`settings.py` 수정**:
        ```python
        # settings.py

        MIDDLEWARE = [
            'django.middleware.security.SecurityMiddleware',
            # 최상단 바로 아래에 Whitenoise 미들웨어 추가
            'whitenoise.middleware.WhiteNoiseMiddleware',
            # ... 나머지 미들웨어
        ]

        # 압축 및 캐싱을 지원하는 스토리지 설정 (권장)
        STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
        ```
    3.  **배포**: `collectstatic`을 실행하는 것은 동일합니다. 하지만 이제 Nginx 설정 없이도 Django 애플리케이션이 직접 `/static/` 경로의 파일들을 효율적으로 서빙합니다.

## 2. 미디어 파일(Media Files) 관리: 실무 가이드

미디어 파일은 사용자가 웹 애플리케이션에 업로드하는 동적인 파일들을 의미합니다. (예: 프로필 사진, 첨부 문서, 동영상). 정적 파일과 달리 미디어 파일은 사용자에 의해 생성되며, 그 양이 예측 불가능하고 매우 커질 수 있습니다.

### 2.1. 개발 환경 설정

개발 환경(`DEBUG=True`)에서는 Django 개발 서버가 미디어 파일을 제공할 수 있도록 설정합니다.

- **`MEDIA_ROOT`**: 업로드된 미디어 파일이 서버의 파일 시스템에 저장될 **절대 경로**를 지정합니다. 이 디렉토리는 Django가 파일을 저장할 수 있도록 쓰기 권한이 있어야 합니다.
    ```python
    # settings.py
    MEDIA_ROOT = BASE_DIR / 'media'
    ```

- **`MEDIA_URL`**: 업로드된 미디어 파일에 접근할 때 사용할 URL 접두사를 정의합니다. 브라우저가 이 URL을 통해 서버에 미디어 파일을 요청하게 됩니다.
    ```python
    # settings.py
    MEDIA_URL = '/media/'
    ```

- **`urls.py` 설정**: 개발 서버가 미디어 파일을 서빙하도록 프로젝트의 `urls.py`에 다음 설정을 추가합니다.
    ```python
    # project_name/urls.py
    from django.conf import settings
    from django.conf.urls.static import static

    urlpatterns = [
        # ... 다른 URL 패턴들 ...
    ]

    # 개발 환경에서만 미디어 파일을 서빙하도록 설정
    if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    ```

### 2.2. 운영 환경 설정 (Production): 클라우드 스토리지 활용

**가장 중요한 점**: 운영 환경에서는 Django 서버나 웹 서버(Nginx)가 직접 미디어 파일을 서빙하는 것은 **권장되지 않습니다.** 이는 확장성, 성능, 안정성, 백업 등 여러 면에서 비효율적이고 위험합니다.

- **문제점**: 
    - **확장성**: 파일 용량이 커지면 서버 디스크 공간이 부족해집니다.
    - **성능**: 대용량 파일 서빙은 웹 서버의 부하를 증가시키고, CDN(Content Delivery Network) 연동이 어렵습니다.
    - **안정성**: 서버 장애 시 파일이 유실될 위험이 있고, 여러 서버로 확장할 때 파일 동기화 문제가 발생합니다.
    - **비용**: 서버 디스크 비용이 클라우드 스토리지보다 비쌀 수 있습니다.

- **해결책: 클라우드 스토리지 (AWS S3, Google Cloud Storage 등)**

    `django-storages` 라이브러리는 Django를 다양한 클라우드 스토리지 서비스와 연동하여 미디어 파일을 효율적으로 관리할 수 있게 해주는 표준 솔루션입니다. 여기서는 AWS S3를 예시로 설명합니다.

    1.  **라이브러리 설치**: `pip install django-storages boto3`
    2.  **`settings.py` 설정**: `storages` 앱을 `INSTALLED_APPS`에 추가하고, S3 관련 설정을 정의합니다. 민감 정보(Access Key, Secret Key)는 반드시 환경 변수로 관리해야 합니다.
        ```python
        # settings.py
        # INSTALLED_APPS에 'storages' 추가
        INSTALLED_APPS = [
            # ...
            'storages',
        ]

        # AWS S3 설정
        AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')
        AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')
        AWS_STORAGE_BUCKET_NAME = config('AWS_STORAGE_BUCKET_NAME')
        AWS_S3_REGION_NAME = config('AWS_S3_REGION_NAME', default='ap-northeast-2') # 예: 서울 리전
        AWS_S3_SIGNATURE_VERSION = 's3v4'
        AWS_S3_FILE_OVERWRITE = False # 같은 이름의 파일 업로드 시 덮어쓸지 여부
        AWS_DEFAULT_ACL = 'public-read' # 업로드된 파일의 기본 접근 권한

        # 미디어 파일 스토리지 클래스 지정
        DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'

        # (선택 사항) CDN 사용 시 커스텀 도메인 설정
        # AWS_S3_CUSTOM_DOMAIN = config('AWS_S3_CUSTOM_DOMAIN', default=None)
        # if AWS_S3_CUSTOM_DOMAIN:
        #     MEDIA_URL = f"https://{AWS_S3_CUSTOM_DOMAIN}/media/"
        # else:
        #     MEDIA_URL = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/media/"
        ```
        - `MEDIA_ROOT`는 클라우드 스토리지를 사용할 경우 더 이상 필요하지 않습니다.
        - `MEDIA_URL`은 이제 S3 버킷의 URL 또는 CDN URL을 가리키게 됩니다.

### 2.3. 모델에 파일 필드 추가

모델에 `FileField` 또는 `ImageField`를 추가하여 파일을 업로드하고 관리할 수 있습니다. `upload_to` 인자는 클라우드 스토리지에서도 파일이 저장될 버킷 내의 경로를 지정하는 데 사용됩니다.

- **`models.FileField(upload_to=...)`**: 모든 종류의 파일을 저장합니다.
- **`models.ImageField(upload_to=...)`**: 이미지 파일만 저장하며, 이미지 유효성 검사 기능이 내장되어 있습니다. `Pillow` 라이브러리(`pip install Pillow`)가 필요합니다.

**`app_name/models.py` 예시:**
```python
from django.db import models

class Product(models.Model):
    name = models.CharField("상품명", max_length=100)
    # 'product_images/' 디렉토리 아래에 파일이 저장됩니다 (S3 버킷 기준)
    main_image = models.ImageField("대표 이미지", upload_to='product_images/') 
    brochure = models.FileField("브로슈어", upload_to='product_brochures/', null=True, blank=True)

    def __str__(self):
        return self.name
```

### 2.4. 템플릿에서 미디어 파일 사용

템플릿에서 모델 인스턴스의 파일 필드에 접근하여 업로드된 파일의 URL을 가져오는 방식은 로컬 스토리지든 클라우드 스토리지든 동일합니다.

```html
<!-- app_name/templates/app_name/product_detail.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{{ product.name }}</title>
</head>
<body>
    <h1>{{ product.name }}</h1>
    {% if product.main_image %}
        <p><img src="{{ product.main_image.url }}" alt="{{ product.name }}" width="300"></p>
    {% endif %}
    {% if product.brochure %}
        <p><a href="{{ product.brochure.url }}" target="_blank">브로슈어 다운로드</a></p>
    {% endif %}
</body>
</html>
```
- `{{ object.file_field.url }}`: 업로드된 파일의 URL을 반환합니다. 이 URL은 `MEDIA_URL`과 `upload_to` 경로를 조합하여 생성되며, 클라우드 스토리지 사용 시에는 S3 버킷의 URL을 가리키게 됩니다.

### 2.5. 폼을 통한 파일 업로드

파일 업로드를 위한 HTML 폼에는 `enctype="multipart/form-data"` 속성이 반드시 포함되어야 합니다. `ModelForm`을 사용하면 뷰에서 파일 처리가 매우 간편해집니다.

**템플릿 예시:**
```html
<form method="post" enctype="multipart/form-data">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit">업로드</button>
</form>
```

**뷰 예시:**
```python
# views.py
from django.shortcuts import render, redirect
from .forms import ProductForm # ProductForm은 Product 모델에 연결된 ModelForm

def product_upload(request):
    if request.method == 'POST':
        # request.FILES를 반드시 전달해야 파일이 처리됩니다.
        form = ProductForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('product_list')
    else:
        form = ProductForm()
    return render(request, 'product_upload.html', {'form': form})
```