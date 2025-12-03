from django.shortcuts import render

# Create your views here.
def home(req):
    return render(req,"home.html")
def login_view(request):
    return render(request, "login.html")
def signup_view(request):
    return render(request, "signup.html")
def textup(req):
    return render(req,"textup.html")
def savednotes(req):
    return render(req,"savednotes.html")
def audio_upload(request):
    return render(request, "audioup.html")
def video_upload(request):
    return render(request, "videoup.html")

<<<<<<< HEAD
def profile_view(request):
    return render(request, "profile.html")

=======
#(text_upload, audio_upload, video_upload, etc.)
def qna_upload(request):
    return render(request, "QnAup.html") 
def quiz_upload(request):
    return render(request, "Quizup.html") 
>>>>>>> 8eb411a841db99a727768299cc1a8a12d8e04a15
