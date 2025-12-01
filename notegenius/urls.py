"""
URL configuration for notegenius project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),     # home page
    path('textup/', views.textup, name='textup'), # text upload
    path('savednotes/', views.savednotes, name='savednotes'), # saved notes
    path("audio-upload/", views.audio_upload, name="audio_upload"), #audio upload
    path("video-upload/", views.video_upload, name="video_upload"), #video upload
    path("qna-upload/", views.qna_upload, name="qna_upload"), #QnA generator upload
     path("quiz-upload/", views.quiz_upload, name="quiz_upload"), #Quiz generator upload

]
