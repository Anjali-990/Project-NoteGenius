from django.shortcuts import render

# Create your views here.
def home(req):
    return render(req,"home.html")
def textup(req):
    return render(req,"textup.html")
def savednotes(req):
    return render(req,"savednotes.html")
def audio_upload(request):
    return render(request, "audioup.html")
def video_upload(request):
    return render(request, "videoup.html")

def profile_view(request):
    return render(request, "profile.html")

