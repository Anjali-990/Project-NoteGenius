# notegenius/urls.py
from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', views.home, name='home'),     # home page
    path("login/", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),

    # text notes
    path('textup/', views.textup, name='textup'),
    path('download/pdf/', views.download_pdf, name='download_pdf'),
    path('download/docx/', views.download_docx, name='download_docx'),
    path('save-notes/', views.save_notes, name='save_notes'),

    # other pages
    path('savednotes/', views.savednotes, name='savednotes'),
    path("audio-upload/", views.audio_upload, name="audio_upload"),
    path("video-upload/", views.video_upload, name="video_upload"),
    path("qna-upload/", views.qna_upload, name="qna_upload"),
    path("quiz-upload/", views.quiz_upload, name="quiz_upload"),
    path("profile/", views.profile, name="profile"),
    path("settings/", views.settings_page, name="settings"),

    path("about/", views.about, name="about"),
    path("feedback/", views.feedback, name="feedback"),

    # NEW: OTP send karne ke liye email par
    path("send-login-otp/", views.send_login_otp, name="send_login_otp"),
]
