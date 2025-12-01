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

from django.shortcuts import render
#(text_upload, audio_upload, video_upload, etc.)
def qna_upload(request):
    return render(request, "QnAup.html") 
