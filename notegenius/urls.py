# notegenius/urls.py
from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # Core
    path('', views.home, name='home'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),

    # Text notes (generation + downloads + save)
    path('textup/', views.textup, name='textup'),
    path('download/pdf/', views.download_pdf, name='download_pdf'),
    path('download/docx/', views.download_docx, name='download_docx'),
    path('save-notes/', views.save_notes, name='save_notes'),

    # Saved notes (per-user) and note management
    path('savednotes/', views.savednotes, name='savednotes'),
    path('note/<int:pk>/', views.view_note, name='view_note'),
    path('note/<int:pk>/edit/', views.edit_note, name='edit_note'),
    path('note/<int:pk>/delete/', views.delete_note, name='delete_note'),

    # Upload / generator pages
    path('audio-upload/', views.audio_upload, name='audio_upload'),
    path('video-upload/', views.video_upload, name='video_upload'),
    path('qna-upload/', views.qna_upload, name='qna_upload'),
    path('quiz-upload/', views.quiz_upload, name='quiz_upload'),

    # Profile / settings / misc
    path('profile/', views.profile, name='profile'),
    path('settings/', views.settings_page, name='settings'),
    path('about/', views.about, name='about'),
    path('feedback/', views.feedback, name='feedback'),

    # OTP (send OTP for login / signup)
    path('send-login-otp/', views.send_login_otp, name='send_login_otp'),
]
