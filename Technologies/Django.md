# 🐍 Django Mastery Guide — From Beginner to Expert

---

## 📘 Chapter 1: Introduction to Django and Web Development Fundamentals

### 🔹 1.1 What is Django?
- Django vs Other Frameworks (Flask, Rails, Express)
- History and Philosophy of Django
- Django's Batteries-Included Approach

### 🔹 1.2 Understanding the Web
- How the Web Works (Client, Server, HTTP)
- URLs, Requests, and Responses
- Web Architecture: Frontend vs Backend

### 🔹 1.3 Installing Django
- Python Environment Setup (venv, pip)
- Installing Django via pip
- Verifying Installation

### 🔹 1.4 First Project: Hello, Django!
- Creating a Django Project
- Understanding manage.py and settings.py
- Running the Development Server
- Creating and Mapping Your First View

### 🔹 1.5 Django’s MVT Architecture
- Models, Views, Templates Explained
- Difference from MVC Pattern
- How They Interact in Real Projects

---

## 📗 Chapter 2: Django Core Components

### 🔹 2.1 Django Apps
- Creating and Managing Apps
- Project vs App: Clear Separation
- App Configuration and `INSTALLED_APPS`

### 🔹 2.2 URL Routing System
- `urls.py` and URL Dispatcher
- Path vs re_path vs include()
- Named URLs and Reverse Lookup

### 🔹 2.3 Views and Templates
- Function-Based Views (FBVs)
- Class-Based Views (CBVs)
- Rendering Templates
- Template Language (Django Templating Engine)
- Template Inheritance

### 🔹 2.4 Models and ORM
- Defining Models
- Model Fields and Field Options
- Migrations (makemigrations and migrate)
- QuerySets and Managers
- Relationships: OneToOne, ManyToMany, ForeignKey

### 🔹 2.5 Admin Panel
- Enabling Admin
- Customizing Admin Interface
- Registering Models
- Admin Filters, Search, and Custom Display

---

## 📙 Chapter 3: Intermediate Concepts and Practical Features

### 🔹 3.1 Forms and Input Handling
- Django Forms vs ModelForms
- Validations and Clean Methods
- Handling GET and POST Requests
- File Uploads and Media Handling

### 🔹 3.2 User Authentication & Authorization
- User Model and Login/Logout
- Password Management
- Access Control (login_required, permissions)
- Custom User Model and UserProfile

### 🔹 3.3 Static and Media Files
- Organizing Static Files
- Serving Media in Development
- Collectstatic and Deployment Tips

### 🔹 3.4 Pagination, Messages & Middleware
- Adding Pagination to Views
- Using Django's Message Framework
- Creating and Using Middleware

### 🔹 3.5 Email and Background Tasks
- Sending Emails via SMTP
- Email Templates
- Setting Up Celery for Async Tasks
- Using Redis with Celery

---

## 📕 Chapter 4: Advanced Django and Full Projects

### 🔹 4.1 Class-Based Views Deep Dive
- Generic Views: ListView, DetailView, etc.
- Mixins and Custom CBVs
- View Inheritance and Refactoring

### 🔹 4.2 API Development with Django REST Framework (DRF)
- Introduction to DRF
- Serializers and ViewSets
- Routers and URLs
- Token and JWT Authentication
- CRUD API Example

### 🔹 4.3 Testing in Django
- Writing Unit and Integration Tests
- Using Django’s TestClient
- Mocking and Fixtures
- Coverage and CI Integration

### 🔹 4.4 Signals and Caching
- Using Django Signals (pre_save, post_save)
- Caching with Memcached / Redis
- Cache Invalidation Patterns

### 🔹 4.5 Real-World Projects (with source code)
- Blog Application
- E-commerce Backend
- Job Board or Portfolio Site

---

## 📒 Chapter 5: Deployment, Scaling, and Best Practices

### 🔹 5.1 Project Structuring and Settings
- Managing Multiple Settings (Dev, Prod)
- Using `.env` Files with `python-decouple` or `django-environ`

### 🔹 5.2 Security Best Practices
- CSRF, XSS, SQL Injection Protection
- HTTPS, Secure Cookies, and Django Security Settings
- Rate Limiting and Throttling

### 🔹 5.3 Performance Optimization
- Database Indexing and Query Optimization
- Template and View Optimization
- Lazy Loading vs Eager Loading

### 🔹 5.4 Dockerizing Django Projects
- Writing Dockerfile and docker-compose.yml
- Environment Variables and Secrets
- Building and Running Django with Docker

### 🔹 5.5 CI/CD and Deployment
- GitHub Actions or GitLab CI for Django
- Deploying to:
  - Heroku
  - DigitalOcean
  - AWS (Elastic Beanstalk or EC2)
- Using Gunicorn + Nginx + PostgreSQL

### 🔹 5.6 Final Project: Scalable Production App
- Choose a Capstone Project
- GitHub Repo, Docs, and API
- Deployment and Versioning

---

## 🧠 Bonus Section:
- Common Mistakes and Debugging Tips
- Top Django Packages You Must Know (django-allauth, crispy-forms, etc.)
- Interview Questions for Django Developers
- Career Advice and Roadmap after Django

---

## ✅ Resources & Exercises
- Curated Reading & Video Resources
- Practice Exercises per Chapter
- GitHub Repos and Templates
- Cheat Sheets and Developer Tools

---


# 📘 Chapter 1: Introduction to Django and Web Development Fundamentals

---

## 🔹 1.1 What is Django?

### ✅ Django Overview
**Django** is a **high-level Python Web framework** that promotes **rapid development** and **clean, pragmatic design**. Built by experienced developers, it takes care of much of the hassle of web development so you can focus on writing your app without reinventing the wheel.

> 🧠 Django is built for speed, scalability, and security.

---

### 🔍 Django vs Other Frameworks

| Feature / Framework | Django         | Flask          | Ruby on Rails  | Express (Node.js) |
|---------------------|----------------|----------------|----------------|-------------------|
| Language            | Python         | Python         | Ruby           | JavaScript        |
| Architecture        | MVT            | Micro (custom) | MVC            | Minimal/MVC       |
| Admin Interface     | ✔️ Built-in     | ❌ Not built-in| ❌ Third-party  | ❌ Not built-in    |
| ORM Support         | ✔️ Built-in     | ❌ Optional     | ✔️ Built-in     | ❌ Optional        |
| REST Support        | ✔️ via DRF      | ✔️ via Flask-Restful | ✔️ via Gems | ✔️ via Middleware  |
| Batteries Included  | ✔️ Yes          | ❌ No          | ✔️ Yes          | ❌ No              |

---

### 🕰️ History and Philosophy

- **Created in 2003**, released publicly in 2005.
- Developed by the **Lawrence Journal-World** newspaper to build database-driven web apps quickly.
- Named after **Django Reinhardt**, a jazz guitarist.
- Follows the **DRY (Don’t Repeat Yourself)** and **KISS (Keep It Simple, Stupid)** principles.

---

### 🔋 Batteries-Included Approach

Django provides almost everything you need to build web apps **out of the box**:

- URL Routing
- Templating Engine
- ORM (Database Models)
- Admin Panel
- Authentication
- Middleware Support
- Forms & Validators
- Sessions and Cookies
- Internationalization
- REST API Support (via Django REST Framework)

> ✨ This means fewer decisions upfront and a faster route to production.

---

## 🔹 1.2 Understanding the Web

### 🌐 How the Web Works

1. **Client (Browser)** makes a request (clicks a link, submits a form).
2. The request goes over the **Internet** using **HTTP**.
3. **Web Server (e.g., Django server)** receives it, processes it, and returns a **response** (usually HTML or JSON).
4. The **Client** renders the response.

---

### 🔗 URLs, Requests, and Responses

#### Example:
```http
GET /home HTTP/1.1
Host: example.com
```

- **GET**: HTTP method
- **/home**: URL path
- **Response**: Could be an HTML page, JSON data, image, etc.

---

### 🏛️ Web Architecture: Frontend vs Backend

| Part       | Responsibilities                        | Examples              |
|------------|------------------------------------------|------------------------|
| Frontend   | UI, Styling, User Interaction            | HTML, CSS, JavaScript |
| Backend    | Logic, Database, APIs, Authentication    | Django, Flask, Node.js|
| Database   | Data Storage                             | PostgreSQL, SQLite    |

> 🧠 Django handles the **backend**, integrates with databases, and can render **frontend HTML templates** too.

---

## 🔹 1.3 Installing Django

### 🐍 Python Environment Setup

#### Step 1: Install Python (3.8+ Recommended)
Visit: https://www.python.org/downloads/

#### Step 2: Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate     # On macOS/Linux
env\Scripts\activate        # On Windows
```

---

### 📦 Installing Django via pip

```bash
pip install django
```

To install a specific version:

```bash
pip install django==4.2
```

---

### ✅ Verifying Installation

```bash
django-admin --version
```

You should see Django's version printed in the terminal.

---

## 🔹 1.4 First Project: Hello, Django!

### 🛠️ Creating a Django Project

```bash
django-admin startproject mysite
cd mysite
```

Directory structure:
```
mysite/
├── manage.py
└── mysite/
    ├── __init__.py
    ├── settings.py
    ├── urls.py
    ├── asgi.py
    └── wsgi.py
```

---

### ⚙️ Understanding Key Files

- **manage.py**: Utility for managing the project (runserver, migrations, etc.)
- **settings.py**: Project config (DB, installed apps, static files)
- **urls.py**: URL route configuration
- **wsgi.py / asgi.py**: For production deployment

---

### ▶️ Running the Development Server

```bash
python manage.py runserver
```

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### 👋 Creating and Mapping Your First View

#### Step 1: Create a view in `mysite/views.py`
```python
from django.http import HttpResponse

def hello_world(request):
    return HttpResponse("Hello, Django!")
```

#### Step 2: Map URL in `urls.py`
```python
from django.urls import path
from .views import hello_world

urlpatterns = [
    path('', hello_world),
]
```

---

## 🔹 1.5 Django’s MVT Architecture

### 🧱 MVT: Model - View - Template

| Layer   | Description                                | Django Component       |
|---------|--------------------------------------------|------------------------|
| Model   | Defines data structure and DB schema       | models.py              |
| View    | Business logic and processing               | views.py               |
| Template| User-facing HTML representation             | HTML templates         |

---

### 🧩 MVT vs MVC

| Concept | MVC               | MVT               |
|---------|-------------------|-------------------|
| Model   | Model             | Model             |
| View    | Template (UI)     | Template (UI)     |
| Controller | View (logic)   | View (logic)      |

> In Django, the **framework handles the controller**, making you write only views and templates.

---

### 🔄 Flow in a Real Project

1. User sends a request via URL.
2. **URL Dispatcher** maps to a **View**.
3. **View** processes the logic and talks to the **Model**.
4. Data is passed to a **Template** to generate HTML.
5. HTML is returned to the browser.

---

## ✅ Chapter 1 Summary & Exercises

### 📝 Summary
- Django is a powerful, batteries-included Python web framework.
- Follows the MVT pattern and simplifies backend development.
- Setting up Django is easy with pip and virtual environments.
- You wrote your first Django view and learned the project structure.

---

### 💡 Exercises

1. Install Django and start a new project called `firstsite`.
2. Create a view that returns “Welcome to Django World”.
3. Modify the homepage to show current date and time.
4. Explore Django’s official docs: https://docs.djangoproject.com/
5. Compare Flask vs Django in your own words.

---

### 📚 Additional Resources

- [Django Official Docs](https://docs.djangoproject.com/)
- [Mozilla Django Tutorial](https://developer.mozilla.org/en-US/docs/Learn/Server-side/Django)
- [RealPython Django Guides](https://realpython.com/tutorials/django/)

---

### 🔥 Pro Tip
> Always use a **virtual environment** and version control (Git) from Day 1. This will save you countless headaches later.

---


# 📗 Chapter 2: Django Core Components

---

## 🔹 2.1 Django Apps

### ✅ Creating and Managing Apps

In Django, a **project** can consist of multiple **apps**, each serving a specific purpose (blog, users, payments, etc.).

To create an app:
```bash
python manage.py startapp blog
```

This creates:
```
blog/
├── admin.py
├── apps.py
├── models.py
├── tests.py
├── views.py
├── migrations/
│   └── __init__.py
└── __init__.py
```

---

### 🆚 Project vs App: Clear Separation

| Concept | Description |
|--------|-------------|
| **Project** | Overall Django configuration (settings, URLs, database) |
| **App** | A modular component that performs a specific task |

> ✅ Think of a **project** as your entire website, and **apps** as reusable modules or features.

---

### ⚙️ App Configuration and `INSTALLED_APPS`

After creating an app, register it in `settings.py`:

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    ...
    'blog',  # <-- your new app
]
```

This enables Django to load models, signals, templates, and admin configs from that app.

---

## 🔹 2.2 URL Routing System

### 📍 `urls.py` and URL Dispatcher

URLs are configured using the `urlpatterns` list:
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('about/', views.about),
]
```

Each route is mapped to a **view function** or class-based view.

---

### ⚖️ Path vs re_path vs include()

- `path()` — recommended, readable syntax for most cases.
- `re_path()` — regex-based routing for advanced patterns.
- `include()` — used to include app-specific URLs in the main router.

**Example with include():**

```python
# project-level urls.py
from django.urls import path, include

urlpatterns = [
    path('blog/', include('blog.urls')),
]
```

---

### 🔗 Named URLs and Reverse Lookup

Name your URLs for reuse in templates or redirects:
```python
path('about/', views.about, name='about')
```

Use in template:
```html
<a href="{% url 'about' %}">About</a>
```

Use in views:
```python
from django.urls import reverse
return redirect(reverse('about'))
```

---

## 🔹 2.3 Views and Templates

### 🧠 Function-Based Views (FBVs)

```python
from django.http import HttpResponse

def home(request):
    return HttpResponse("Welcome to Home")
```

Simple and explicit, suitable for small logic.

---

### 🧱 Class-Based Views (CBVs)

```python
from django.views import View
from django.http import HttpResponse

class HomeView(View):
    def get(self, request):
        return HttpResponse("Hello from CBV")
```

CBVs are more structured and support **mixins** and **generic views** for reuse.

---

### 🖼️ Rendering Templates

In views:
```python
from django.shortcuts import render

def homepage(request):
    return render(request, 'home.html', {'title': 'Welcome'})
```

---

### 🧾 Template Language (Django Templating Engine)

Sample template:
```html
<!DOCTYPE html>
<html>
<head>
  <title>{{ title }}</title>
</head>
<body>
  <h1>{{ title }}</h1>
</body>
</html>
```

Built-in tags:
- `{{ variable }}` — print variables
- `{% if %}`, `{% for %}` — logic
- `{% block %}`, `{% include %}`, `{% extends %}` — structure

---

### 📐 Template Inheritance

**base.html**
```html
<!DOCTYPE html>
<html>
  <head><title>{% block title %}{% endblock %}</title></head>
  <body>
    <header>MySite</header>
    {% block content %}{% endblock %}
  </body>
</html>
```

**home.html**
```html
{% extends 'base.html' %}
{% block title %}Home{% endblock %}
{% block content %}
<h1>Welcome!</h1>
{% endblock %}
```

---

## 🔹 2.4 Models and ORM

### 🧬 Defining Models

```python
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()
    published = models.DateTimeField(auto_now_add=True)
```

---

### 📦 Model Fields and Options

| Field Type     | Description                 |
|----------------|-----------------------------|
| CharField      | Short text, max_length req. |
| TextField      | Long text                   |
| IntegerField   | Integers                    |
| BooleanField   | True/False                  |
| DateTimeField  | Dates and times             |
| ForeignKey     | Links to another model      |
| ManyToManyField| Many-to-many relationship   |

Field options include `null`, `blank`, `default`, `choices`, `unique`.

---

### 🛠️ Migrations

Create migration files:
```bash
python manage.py makemigrations
```

Apply to database:
```bash
python manage.py migrate
```

---

### 🔍 QuerySets and Managers

```python
Post.objects.all()            # All records
Post.objects.filter(title__icontains='django')  # Filtered
Post.objects.get(id=1)        # Single object
```

Chainable, lazy-loaded, efficient.

---

### 🔗 Relationships

#### One-to-One:
```python
user = models.OneToOneField(User, on_delete=models.CASCADE)
```

#### ForeignKey:
```python
author = models.ForeignKey(User, on_delete=models.CASCADE)
```

#### Many-to-Many:
```python
tags = models.ManyToManyField(Tag)
```

> ⚠️ Always define `on_delete` in `ForeignKey`.

---

## 🔹 2.5 Admin Panel

### 🔓 Enabling Admin

Make sure `'django.contrib.admin'` is in `INSTALLED_APPS`.

Create superuser:
```bash
python manage.py createsuperuser
```

Run server and go to: [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin)

---

### 🛠️ Registering Models

In `admin.py`:
```python
from .models import Post
admin.site.register(Post)
```

---

### 🎛️ Customizing Admin Interface

```python
@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ('title', 'published')
    search_fields = ['title']
    list_filter = ['published']
```

Other features:
- `readonly_fields`
- `prepopulated_fields`
- `fieldsets` for grouped display

---

## ✅ Chapter 2 Summary & Exercises

### 📝 Summary

- Apps are modular components inside a Django project.
- URL routing maps user requests to views.
- Views render logic; Templates handle presentation.
- Models define database schema using Python.
- Django admin helps you manage data without extra code.

---

### 💡 Exercises

1. Create a new app `products` and register it.
2. Build a simple model for a product with name, price, and description.
3. Register the model in admin and add a few entries.
4. Create views and templates for product list and details.
5. Practice using `path()` and `include()` in `urls.py`.

---

### 📚 Additional Resources

- [Django Models Docs](https://docs.djangoproject.com/en/stable/topics/db/models/)
- [Django Admin Docs](https://docs.djangoproject.com/en/stable/ref/contrib/admin/)
- [CBV vs FBV Explanation](https://docs.djangoproject.com/en/stable/topics/class-based-views/)

---

### ⚠️ Common Mistakes to Avoid

- Forgetting to add app to `INSTALLED_APPS`
- Not running `makemigrations` after model changes
- Missing `name` in URL paths (breaks template reverse lookups)
- Using `=` in `{% url %}` tags (it should be positional)

> 🧠 Pro Tip: Keep your code modular by separating views, templates, and URLs in each app.

---


# 📙 Chapter 3: Intermediate Concepts and Practical Features

---

## 🔹 3.1 Forms and Input Handling

### 🧾 Django Forms vs ModelForms

#### ✅ Django Forms
Used to create forms manually:
```python
from django import forms

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()
    message = forms.CharField(widget=forms.Textarea)
```

#### ✅ ModelForms
Auto-generates form fields from models:
```python
from django.forms import ModelForm
from .models import Post

class PostForm(ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'content']
```

> ✅ Use `ModelForm` for forms tightly coupled with a model (like create/edit), and `Form` for custom logic.

---

### ✅ Validations and Clean Methods

You can add custom validations using `clean_<fieldname>` or `clean()` methods.

```python
class ContactForm(forms.Form):
    email = forms.EmailField()

    def clean_email(self):
        data = self.cleaned_data['email']
        if "@example.com" not in data:
            raise forms.ValidationError("Must use @example.com domain")
        return data
```

---

### 🔁 Handling GET and POST Requests

```python
def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # process data
            return redirect('success')
    else:
        form = ContactForm()
    
    return render(request, 'contact.html', {'form': form})
```

---

### 🗂️ File Uploads and Media Handling

#### Setup in `settings.py`:
```python
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
```

#### Form and View:
```python
class UploadForm(forms.Form):
    file = forms.FileField()

def upload_file(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']
            with open(f'media/{uploaded_file.name}', 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
    else:
        form = UploadForm()
    return render(request, 'upload.html', {'form': form})
```

#### Serve Media in Development:
```python
from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

---

## 🔹 3.2 User Authentication & Authorization

### 👤 User Model and Login/Logout

Use Django’s built-in User model:
```python
from django.contrib.auth.models import User
```

#### Login View:
```python
from django.contrib.auth import authenticate, login

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('dashboard')
```

#### Logout View:
```python
from django.contrib.auth import logout

def user_logout(request):
    logout(request)
    return redirect('home')
```

---

### 🔐 Password Management

- `set_password()` and `check_password()` for hashing
- Django provides views for:
  - Password change
  - Password reset via email
  - Password confirm/reset token

Use:
```python
from django.contrib.auth.views import PasswordChangeView, PasswordResetView
```

---

### 🔒 Access Control (Decorators & Permissions)

#### login_required
```python
from django.contrib.auth.decorators import login_required

@login_required
def dashboard(request):
    ...
```

#### Permission Required
```python
from django.contrib.auth.decorators import permission_required

@permission_required('app_label.permission_name')
def view_name(request):
    ...
```

---

### 👥 Custom User Model and UserProfile

**Always set up a custom user model early if needed.**

In `models.py`:
```python
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    age = models.IntegerField(null=True, blank=True)
```

In `settings.py`:
```python
AUTH_USER_MODEL = 'yourapp.CustomUser'
```

Create a `UserProfile` model for additional fields:
```python
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    avatar = models.ImageField(upload_to='avatars/')
```

Use Django signals to create profile on user creation.

---

## 🔹 3.3 Static and Media Files

### 🗃️ Organizing Static Files

Create a `static/` directory in your app:
```
myapp/
└── static/
    └── myapp/
        └── styles.css
```

In `settings.py`:
```python
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / "static"]
```

In templates:
```html
{% load static %}
<link rel="stylesheet" href="{% static 'myapp/styles.css' %}">
```

---

### 🖼️ Serving Media in Development

Handled with `MEDIA_URL` and `MEDIA_ROOT` like shown in 3.1.

---

### 📦 Collectstatic and Deployment Tips

Before deployment:
```bash
python manage.py collectstatic
```

This moves all static files to `STATIC_ROOT`, for Nginx or CDN to serve them.

In `settings.py`:
```python
STATIC_ROOT = BASE_DIR / 'staticfiles'
```

> 🔥 Pro Tip: Never serve static files with Django in production.

---

## 🔹 3.4 Pagination, Messages & Middleware

### 📄 Adding Pagination to Views

```python
from django.core.paginator import Paginator

def blog_list(request):
    posts = Post.objects.all()
    paginator = Paginator(posts, 5)  # 5 per page
    page = request.GET.get('page')
    posts = paginator.get_page(page)
    return render(request, 'blog_list.html', {'posts': posts})
```

In template:
```html
{% for post in posts %}
  {{ post.title }}
{% endfor %}

<div class="pagination">
  {% if posts.has_previous %}
    <a href="?page={{ posts.previous_page_number }}">Prev</a>
  {% endif %}
  <span>Page {{ posts.number }} of {{ posts.paginator.num_pages }}</span>
  {% if posts.has_next %}
    <a href="?page={{ posts.next_page_number }}">Next</a>
  {% endif %}
</div>
```

---

### 💬 Using Django's Message Framework

In `views.py`:
```python
from django.contrib import messages

messages.success(request, 'Saved successfully!')
```

In template:
```html
{% if messages %}
  {% for message in messages %}
    <div class="alert alert-{{ message.tags }}">{{ message }}</div>
  {% endfor %}
{% endif %}
```

---

### 🧩 Creating and Using Middleware

Middleware is a function/class that processes requests/responses.

```python
class SimpleMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        print("Before view")
        response = self.get_response(request)
        print("After view")
        return response
```

Add it to `MIDDLEWARE` in `settings.py`:
```python
MIDDLEWARE = [
    ...
    'myapp.middleware.SimpleMiddleware',
]
```

---

## 🔹 3.5 Email and Background Tasks

### 📧 Sending Emails via SMTP

In `settings.py`:
```python
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'your_email@gmail.com'
EMAIL_HOST_PASSWORD = 'your_password'
```

Send email:
```python
from django.core.mail import send_mail

send_mail(
    'Subject here',
    'Here is the message.',
    'from@example.com',
    ['to@example.com'],
    fail_silently=False,
)
```

---

### 📝 Email Templates

```python
from django.template.loader import render_to_string

message = render_to_string('email/welcome.html', {'user': user})
```

---

### 🌀 Setting Up Celery for Async Tasks

Install Celery:
```bash
pip install celery
```

`celery.py` in project root:
```python
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')

app = Celery('mysite')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
```

In `__init__.py`:
```python
from .celery import app as celery_app
__all__ = ('celery_app',)
```

---

### 🔁 Using Redis with Celery

Install Redis:
```bash
sudo apt install redis
```

Start Redis server:
```bash
redis-server
```

Set in `settings.py`:
```python
CELERY_BROKER_URL = 'redis://localhost:6379/0'
```

Create task in `tasks.py`:
```python
from celery import shared_task

@shared_task
def send_email_task():
    send_mail(...)
```

Call it:
```python
send_email_task.delay()
```

---

## ✅ Chapter 3 Summary & Exercises

### 📝 Summary

- Forms allow user input and validation.
- Authentication provides secure login/logout and custom user models.
- Static/media files are essential for assets and uploads.
- Pagination and messages improve UX.
- Celery + Redis enables background task handling.

---

### 💡 Exercises

1. Create a contact form and save data to a model.
2. Add file upload and render uploaded files.
3. Create login/logout views and restrict access using `@login_required`.
4. Send a welcome email upon user registration.
5. Paginate a blog list view with 5 posts per page.

---

### 📚 Additional Resources

- [Django Forms Guide](https://docs.djangoproject.com/en/stable/topics/forms/)
- [Django Authentication Docs](https://docs.djangoproject.com/en/stable/topics/auth/)
- [Celery + Django Guide](https://docs.celeryq.dev/en/stable/django/first-steps-with-django.html)

---

### ⚠️ Common Pitfalls

- Forgetting `enctype="multipart/form-data"` for file uploads
- Not hashing passwords manually with `set_password()`
- Static paths misconfigured in deployment
- Long-running logic blocking request cycle (should be async with Celery)

> 🧠 Pro Tip: Don’t reinvent user auth — customize Django’s powerful built-in system instead.

---

## 📕 Chapter 4: Advanced Django and Full Projects

### 🔹 4.1 Class-Based Views Deep Dive

Class-Based Views (CBVs) provide a more structured and reusable approach to building views.

#### ✅ Generic Views
Django provides several out-of-the-box CBVs:
- **ListView**: Displays a list of objects.
- **DetailView**: Shows details of a single object.
- **CreateView**, **UpdateView**, **DeleteView**: Handle object creation, updating, and deletion.

Each of these uses a `model`, `template_name`, and `context_object_name`.

```python
from django.views.generic import ListView
from .models import Product

class ProductListView(ListView):
    model = Product
    template_name = 'product_list.html'
    context_object_name = 'products'
```

#### ✅ Mixins
Mixins are used to add reusable behavior to views.
Examples:
- `LoginRequiredMixin`
- `PermissionRequiredMixin`
- `FormValidMixin`

```python
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import UpdateView

class ProfileUpdateView(LoginRequiredMixin, UpdateView):
    model = Profile
    fields = ['bio', 'avatar']
    template_name = 'profile_form.html'
```

#### ✅ Custom CBVs and Inheritance
CBVs can be extended or customized via inheritance for consistent logic across views.
Create a base view with shared logic:
```python
class BaseProductView(View):
    def get_queryset(self):
        return Product.objects.filter(is_active=True)
```

---

### 🔹 4.2 API Development with Django REST Framework (DRF)

Django REST Framework (DRF) is a powerful toolkit to build RESTful APIs.

#### ✅ Serializers
Used to convert complex data (like queryset) to native Python datatypes and vice versa.

```python
from rest_framework import serializers
from .models import Product

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = '__all__'
```

#### ✅ ViewSets
ViewSets combine logic for multiple HTTP methods into a single class.

```python
from rest_framework import viewsets

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
```

#### ✅ Routers and URLs
Routers automatically generate URL patterns.

```python
from rest_framework.routers import DefaultRouter
from .views import ProductViewSet

router = DefaultRouter()
router.register(r'products', ProductViewSet)

urlpatterns = router.urls
```

#### ✅ Authentication (Token & JWT)
DRF supports Token-based and JWT authentication.

**Token-based:**
```bash
pip install djangorestframework authtoken
```

**JWT (with SimpleJWT):**
```bash
pip install djangorestframework-simplejwt
```

```python
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

urlpatterns += [
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
```

#### ✅ CRUD API Example
With ModelViewSet + Router, DRF provides full CRUD without writing each method.

---

### 🔹 4.3 Testing in Django

#### ✅ Unit and Integration Tests
Django uses Python’s `unittest` framework:
```python
from django.test import TestCase
from .models import Product

class ProductModelTest(TestCase):
    def test_str(self):
        product = Product.objects.create(name="Laptop")
        self.assertEqual(str(product), "Laptop")
```

#### ✅ Django TestClient
Simulate HTTP requests for testing views:
```python
from django.test import Client

client = Client()
response = client.get('/products/')
self.assertEqual(response.status_code, 200)
```

#### ✅ Mocking and Fixtures
Use `unittest.mock` for mocking. Load fixtures from JSON/YAML.

```bash
python manage.py loaddata initial_data.json
```

#### ✅ Coverage and CI Integration
Install coverage tool:
```bash
pip install coverage
coverage run manage.py test
coverage report -m
```

Integrate into GitHub Actions or any CI/CD tool to run tests on push.

---

### 🔹 4.4 Signals and Caching

#### ✅ Signals
Used to trigger actions when certain events occur.
Common signals:
- `pre_save`, `post_save`
- `pre_delete`, `post_delete`

```python
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Order

@receiver(post_save, sender=Order)
def notify_admin(sender, instance, created, **kwargs):
    if created:
        print(f"New order placed: {instance.id}")
```

#### ✅ Caching
Improve performance with in-memory caching using **Memcached** or **Redis**.

```python
from django.core.cache import cache

def get_data():
    data = cache.get('data_key')
    if not data:
        data = expensive_query()
        cache.set('data_key', data, timeout=300)
    return data
```

#### ✅ Cache Invalidation Patterns
- **Time-based expiration**
- **Key-versioning**
- **Manual invalidation after update**

---

### 🔹 4.5 Real-World Projects (with source code)

#### ✅ Blog Application
- CRUD for posts and comments
- User login/logout and profile
- Markdown support
- DRF for public API

#### ✅ E-commerce Backend
- Products, Categories, Orders, Cart
- Payment Integration (Stripe/PayPal)
- Admin customization
- Email notifications

#### ✅ Job Board or Portfolio Site
- Job listings and applications
- Contact forms
- Resume upload and media handling
- Role-based access for admin/employer

Each project demonstrates integration of core + intermediate + advanced concepts with production-ready architecture.

---

## 📒 Chapter 5: Deployment, Scaling, and Best Practices

---

### 🔹 5.1 Project Structuring and Settings

**Managing Multiple Settings (Dev, Prod)**
- Split settings into multiple files (`settings/base.py`, `settings/dev.py`, `settings/prod.py`).
- Use `DJANGO_SETTINGS_MODULE` environment variable to switch between environments.
- Example structure:
  ```
  └── project_name/
      └── settings/
          ├── __init__.py
          ├── base.py
          ├── dev.py
          └── prod.py
  ```

**Using `.env` Files with `python-decouple` or `django-environ`**
- Store secrets and environment-specific variables in `.env` file.
- Example:
  ```
  DEBUG=True
  SECRET_KEY=your-secret
  DATABASE_URL=postgres://user:pass@localhost/dbname
  ```

- In code:
  ```python
  from decouple import config
  DEBUG = config('DEBUG', cast=bool)
  ```

---

### 🔹 5.2 Security Best Practices

**CSRF, XSS, SQL Injection Protection**
- Enable CSRF protection (default in Django via `csrf_token` and middleware).
- Use Django template escaping to avoid XSS.
- Always use ORM to prevent SQL injection.

**HTTPS, Secure Cookies, and Django Security Settings**
- Use `SECURE_SSL_REDIRECT = True` in production.
- Set `SESSION_COOKIE_SECURE = True` and `CSRF_COOKIE_SECURE = True`.
- Set strong `SECURE_HSTS_SECONDS`, `SECURE_BROWSER_XSS_FILTER`, and `X_FRAME_OPTIONS = 'DENY'`.

**Rate Limiting and Throttling**
- Use Django REST Framework's throttling classes (if building APIs).
- Integrate third-party tools like `django-ratelimit`.

---

### 🔹 5.3 Performance Optimization

**Database Indexing and Query Optimization**
- Add indexes using `db_index=True` in model fields.
- Use `select_related()` and `prefetch_related()` to reduce queries.
- Avoid N+1 query issues.

**Template and View Optimization**
- Use cached template fragments with `{% cache %}`.
- Optimize view logic to reduce DB hits and I/O.

**Lazy Loading vs Eager Loading**
- Lazy: fetch related data only when accessed.
- Eager: use `select_related()`/`prefetch_related()` to fetch ahead of time.
- Use profiling tools like Django Debug Toolbar.

---

### 🔹 5.4 Dockerizing Django Projects

**Writing Dockerfile and docker-compose.yml**
- `Dockerfile`:
  ```dockerfile
  FROM python:3.11-slim
  ENV PYTHONDONTWRITEBYTECODE 1
  ENV PYTHONUNBUFFERED 1
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  ```

- `docker-compose.yml`:
  ```yaml
  version: '3.8'
  services:
    web:
      build: .
      command: gunicorn project.wsgi:application --bind 0.0.0.0:8000
      volumes:
        - .:/app
      ports:
        - "8000:8000"
      env_file:
        - .env
    db:
      image: postgres:15
      environment:
        POSTGRES_DB: your_db
        POSTGRES_USER: user
        POSTGRES_PASSWORD: password
  ```

**Environment Variables and Secrets**
- Use `.env` with `docker-compose` to inject secrets.
- Avoid hardcoding credentials in Dockerfiles.

**Building and Running Django with Docker**
```bash
docker-compose build
docker-compose up
```

---

### 🔹 5.5 CI/CD and Deployment

**GitHub Actions or GitLab CI for Django**
- Automate testing, linting, migrations, and deployment.
- Example GitHub Action:
  ```yaml
  jobs:
    build:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v2
        - name: Set up Python
          uses: actions/setup-python@v2
          with:
            python-version: 3.11
        - name: Install dependencies
          run: |
            pip install -r requirements.txt
        - name: Run tests
          run: |
            python manage.py test
  ```

**Deploying to Platforms**

- **Heroku**
  - Add `Procfile`: `web: gunicorn project.wsgi`
  - Set buildpacks and environment variables on dashboard.

- **DigitalOcean (App Platform or Droplets)**
  - Use Docker, Gunicorn, and Postgres.
  - Optionally use Nginx as reverse proxy.

- **AWS (Elastic Beanstalk or EC2)**
  - Use EB CLI to deploy with Docker or Python environment.
  - Secure with SSL via ACM, use RDS for database.

**Using Gunicorn + Nginx + PostgreSQL**
- Gunicorn as WSGI server: `gunicorn project.wsgi:application`
- Nginx to reverse proxy and serve static/media files.
- PostgreSQL as production database.

---

### 🔹 5.6 Final Project: Scalable Production App

**Choose a Capstone Project**
- Examples:
  - Blog Platform with User Roles
  - E-commerce Store
  - Task Manager with API and Web UI

**GitHub Repo, Docs, and API**
- Follow best practices:
  - `README.md`, `.env.example`, Postman collection
  - API docs (Swagger/Redoc with DRF)
  - Add tests (`pytest` or Django test framework)

**Deployment and Versioning**
- Deploy project using Docker or a CI/CD pipeline.
- Use semantic versioning: `v1.0.0`, `v1.1.0`, etc.
- Document change logs and updates.

---

