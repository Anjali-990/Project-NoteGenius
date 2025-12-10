from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.db.models import Q
from django.core.mail import send_mail
import random

from .models import Profile, EmailOTP


def home(request):
    return render(request, "home.html")


def login_view(request):
    if request.method == "POST":
        login_id = request.POST.get("login_id")  # yahan email aayega
        password = request.POST.get("password")
        otp_code = request.POST.get("login_otp")

        user_obj = None

        if login_id:
            # username ya email, dono allow
            user_obj = User.objects.filter(
                Q(email=login_id) | Q(username=login_id)
            ).first()

        # ---------- OTP based login ----------
        if otp_code and user_obj:
            otp_obj = (
                EmailOTP.objects
                .filter(user=user_obj, code=otp_code, is_used=False)
                .order_by("-created_at")
                .first()
            )

            if otp_obj and not otp_obj.is_expired():
                otp_obj.is_used = True
                otp_obj.save()
                login(request, user_obj)
                return redirect("home")
            else:
                return render(request, "login.html", {
                    "error": "Invalid or expired OTP.",
                })

        # ---------- Password based login ----------
        if user_obj and password:
            user = authenticate(
                request,
                username=user_obj.username,
                password=password,
            )
        else:
            user = None

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            return render(request, "login.html", {
                "error": "Invalid email/username or password.",
            })

    return render(request, "login.html")


def signup_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if password1 != password2:
            return render(request, "signup.html", {
                "error": "Passwords do not match.",
            })

        if User.objects.filter(username=username).exists():
            return render(request, "signup.html", {
                "error": "Username already taken.",
            })

        if User.objects.filter(email=email).exists():
            return render(request, "signup.html", {
                "error": "Email already registered.",
            })

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password1,
        )

        if phone:
            Profile.objects.create(user=user, phone=phone)

        login(request, user)
        return redirect("home")

    return render(request, "signup.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@require_POST
def send_login_otp(request):
    login_id = request.POST.get("login_id")

    if not login_id:
        return JsonResponse({
            "success": False,
            "message": "Please enter your email first."
        })

    user = User.objects.filter(
        Q(email=login_id) | Q(username=login_id)
    ).first()

    if not user:
        return JsonResponse({
            "success": False,
            "message": "No user found with this email/username."
        })

    # 6 digit OTP
    code = f"{random.randint(100000, 999999)}"

    # Purane unused OTP invalidate
    EmailOTP.objects.filter(user=user, is_used=False).update(is_used=True)

    EmailOTP.objects.create(user=user, code=code)

    subject = "Your NoteGenius Login OTP"
    message = f"Your OTP is {code}. It will expire in 10 minutes."

    try:
        send_mail(subject, message, None, [user.email], fail_silently=False)
    except Exception as e:
        print("EMAIL ERROR:", e)
        return JsonResponse({
            "success": False,
            "message": "Error sending OTP email. Please check email settings."
        })

    return JsonResponse({
        "success": True,
        "message": "OTP has been sent to your email."
    })


# ------- baaki tumhare pages same rakhe hain -------

def textup(request):
    return render(request, "textup.html")


def savednotes(request):
    return render(request, "savednotes.html")


def audio_upload(request):
    return render(request, "audioup.html")


def video_upload(request):
    return render(request, "videoup.html")


def settings_page(request):
    return render(request, "settings.html")


def qna_upload(request):
    return render(request, "QnAup.html")


def quiz_upload(request):
    return render(request, "Quizup.html")


def profile(request):
    return render(request, "profile.html")


def about(request):
    return render(request, "aboutus.html")


def feedback(request):
    return render(request, "feedback.html")
