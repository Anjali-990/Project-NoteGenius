# myapp/views.py
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
from django.core.mail import send_mail
import random
import os
import logging
import io
import tempfile

# try to import Notes model; if missing, keep a fallback so server won't crash
try:
    from .models import Profile, EmailOTP, Notes
except ImportError:
    # If Notes is missing, import what exists and set Notes to None
    from .models import Profile, EmailOTP  # may raise if these are missing
    Notes = None


# File/office/pdf handling
import PyPDF2
from docx import Document as DocxReader
from docx import Document as DocxWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Optional audio libs
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# Transformers (summarization + optional ASR)
os.environ["TRANSFORMERS_NO_TF"] = "1"
logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    # summarizer (fast default)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
except Exception as e:
    summarizer = None
    logger.warning("Summarizer pipeline not available: %s", e)

# optional ASR pipeline (Whisper) - may be large to download
asr_pipeline = None
try:
    from transformers import pipeline as _pipeline
    # try to initialize but it's okay if it fails
    try:
        asr_pipeline = _pipeline("automatic-speech-recognition", model="openai/whisper-small")
    except Exception as e:
        asr_pipeline = None
        logger.info("ASR pipeline not initialized: %s", e)
except Exception:
    asr_pipeline = None

# ---------------------------
# SUMMARY LENGTH PRESETS
# ---------------------------
LENGTH_PRESETS = {
    "short":  {"max_length": 60,  "min_length": 20},
    "medium": {"max_length": 150, "min_length": 60},
    "long":   {"max_length": 300, "min_length": 150},
}

# ---------------------------
# AUTH + ACCOUNT VIEWS (edited signup to NOT auto-login)
# ---------------------------
def home(request):
    return render(request, "home.html")


def login_view(request):
    if request.method == "POST":
        login_id = request.POST.get("login_id")
        password = request.POST.get("password")
        otp_code = request.POST.get("login_otp")

        user_obj = None
        if login_id:
            user_obj = User.objects.filter(Q(email=login_id) | Q(username=login_id)).first()

        # OTP login
        if otp_code and user_obj:
            otp_obj = EmailOTP.objects.filter(user=user_obj, code=otp_code, is_used=False).order_by("-created_at").first()
            if otp_obj and not otp_obj.is_expired():
                otp_obj.is_used = True
                otp_obj.save()
                login(request, user_obj)
                return redirect("home")
            else:
                return render(request, "login.html", {"error": "Invalid or expired OTP."})

        # password login
        if user_obj and password:
            user = authenticate(request, username=user_obj.username, password=password)
        else:
            user = None

        if user:
            login(request, user)
            return redirect("home")
        else:
            return render(request, "login.html", {"error": "Invalid email/username or password."})

    return render(request, "login.html")


def signup_view(request):
    """
    Signup behaviour changed:
      - create user on POST (after checks)
      - DO NOT auto-login the user
      - show a success message and redirect to login page
    """
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if password1 != password2:
            return render(request, "signup.html", {"error": "Passwords do not match."})
        if User.objects.filter(username=username).exists():
            return render(request, "signup.html", {"error": "Username already taken."})
        if User.objects.filter(email=email).exists():
            return render(request, "signup.html", {"error": "Email already registered."})

        user = User.objects.create_user(username=username, email=email, password=password1)
        if phone:
            Profile.objects.create(user=user, phone=phone)

        # Don't auto-login. Ask user to login explicitly.
        messages.success(request, "Account created successfully. Please log in to continue.")
        return redirect("login")

    return render(request, "signup.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@require_POST
def send_login_otp(request):
    login_id = request.POST.get("login_id")
    if not login_id:
        return JsonResponse({"success": False, "message": "Please enter your email first."})

    user = User.objects.filter(Q(email=login_id) | Q(username=login_id)).first()
    if not user:
        return JsonResponse({"success": False, "message": "No user found with this email/username."})

    code = f"{random.randint(100000, 999999)}"
    EmailOTP.objects.filter(user=user, is_used=False).update(is_used=True)
    EmailOTP.objects.create(user=user, code=code)

    subject = "Your NoteGenius Login OTP"
    message = f"Your OTP is {code}. It will expire in 10 minutes."
    try:
        send_mail(subject, message, None, [user.email], fail_silently=False)
    except Exception as e:
        logger.warning("Email send failed: %s", e)
        return JsonResponse({"success": False, "message": "Error sending OTP email. Please check email settings."})

    return JsonResponse({"success": True, "message": "OTP has been sent to your email."})


# ---------------------------
# TEXT SUMMARIZATION HELPERS
# ---------------------------
def chunk_text_dynamic(text):
    length = len(text)
    if length < 5000:
        max_chars = 800
    elif length < 20000:
        max_chars = 1500
    else:
        max_chars = 2500
    chunks = []
    start = 0
    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        start = end
    return chunks


def generate_fast_notes(text, summary_length="medium"):
    if summarizer is None:
        raise RuntimeError("Summarizer is not initialized (transformers not installed?).")
    params = LENGTH_PRESETS.get(summary_length, LENGTH_PRESETS["medium"])
    chunks = chunk_text_dynamic(text)
    partial_summaries = []

    for ch in chunks:
        try:
            out = summarizer(ch, max_length=params["max_length"], min_length=params["min_length"], truncation=True, do_sample=False)
            partial_summaries.append(out[0].get("summary_text", "").strip())
        except Exception:
            partial_summaries.append(ch[:800].strip())

    if len(partial_summaries) > 1:
        combined = " ".join(partial_summaries)
        try:
            final_out = summarizer(combined, max_length=params["max_length"], min_length=params["min_length"], truncation=True, do_sample=False)
            combined_summary = final_out[0].get("summary_text", combined)
        except Exception:
            combined_summary = combined
    else:
        combined_summary = partial_summaries[0] if partial_summaries else ""

    combined_summary = combined_summary.strip()
    sentences = [s.strip() for s in combined_summary.split(". ") if s.strip()]

    if summary_length == "short":
        short_summary = (sentences[0] if sentences else combined_summary)[:140]
        if len(short_summary) >= 140:
            short_summary = short_summary.rstrip() + "..."
    else:
        short_summary = sentences[0] if sentences else (combined_summary[:140] + "...")

    bullets = sentences[1:6] if len(sentences) > 1 else []
    words = [w.strip(".,;:()") for w in combined_summary.split() if len(w) > 5]
    key_terms = list(dict.fromkeys(words))[:20]

    return {
        "short_summary": short_summary,
        "bullets": bullets,
        "detailed_explanation": combined_summary,
        "key_terms": key_terms
    }


# ---------------------------
# VIEWS: textup, savednotes, downloads
# ---------------------------
MAX_PROCESS_CHARS = 200000

def textup(request):
    summary, error, warning = {}, "", ""
    if request.method == "POST":
        input_text = ""

        if request.FILES.get("file"):
            uploaded_file = request.FILES["file"]
            fname = uploaded_file.name.lower()

            if uploaded_file.size > 15 * 1024 * 1024:
                error = "File too large. Please upload a smaller file (max 15 MB)."
            else:
                try:
                    if fname.endswith(".pdf"):
                        reader = PyPDF2.PdfReader(uploaded_file)
                        input_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                    elif fname.endswith(".txt"):
                        input_text = uploaded_file.read().decode("utf-8", errors="ignore")
                    elif fname.endswith(".docx"):
                        doc = DocxReader(uploaded_file)
                        input_text = "\n".join([p.text for p in doc.paragraphs])
                    else:
                        error = "Unsupported file type. Please upload .txt, .pdf, or .docx."
                except Exception as e:
                    logger.error("File reading failed", exc_info=True)
                    error = f"Error reading the file: {str(e)}"

                if input_text and len(input_text) > 20000:
                    warning = "This file is large. Processing may take some time."

        elif request.POST.get("content"):
            input_text = request.POST.get("content", "")

        if input_text and input_text.strip() and not error:
            if len(input_text) > MAX_PROCESS_CHARS:
                input_text = input_text[:MAX_PROCESS_CHARS]
                warning = "Input trimmed to first 200k characters for speed."
            try:
                summary_length = request.POST.get("summary_length", "medium")
                notes = generate_fast_notes(input_text, summary_length)
                request.session["latest_notes"] = notes
                request.session.modified = True
                summary = notes
            except Exception as e:
                logger.error("Generation failed", exc_info=True)
                error = "Something went wrong while generating notes."
        elif not error and (not input_text or not input_text.strip()):
            error = "Please paste some text or upload a supported file."

    return render(request, "textup.html", {"summary": summary, "error": error, "warning": warning})


def savednotes(request):
    return render(request, "savednotes.html")


# ---------------------------
# AUDIO UPLOAD / RECORDING HANDLER (GET render + POST processing -> JSON)
# ---------------------------
@csrf_exempt
def audio_upload(request):
    if request.method == "GET":
        return render(request, "audioup.html")

    uploaded = None
    tmp_path = None
    try:
        if request.FILES.get("file"):
            uploaded = request.FILES["file"]
            suffix = os.path.splitext(uploaded.name)[1] or ".wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp.flush()
            tmp.close()
            tmp_path = tmp.name
        elif request.FILES.get("recorded_blob"):
            uploaded = request.FILES["recorded_blob"]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp.flush()
            tmp.close()
            tmp_path = tmp.name
        else:
            return JsonResponse({"success": False, "error": "No audio received."})

        transcript = ""

        # try whisper-based ASR first
        if asr_pipeline is not None:
            try:
                out = asr_pipeline(tmp_path)
                if isinstance(out, dict) and "text" in out:
                    transcript = out["text"]
                elif isinstance(out, list) and out and "text" in out[0]:
                    transcript = out[0]["text"]
                else:
                    transcript = str(out)
            except Exception as e:
                logger.warning("ASR pipeline failed: %s", e)
                transcript = ""

        # fallback to speech_recognition (Google) if available
        if not transcript and SR_AVAILABLE:
            try:
                r = sr.Recognizer()
                with sr.AudioFile(tmp_path) as source:
                    audio = r.record(source)
                transcript = r.recognize_google(audio)
            except Exception as e:
                logger.warning("SpeechRecognition failed: %s", e)
                transcript = ""

        if not transcript:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            return JsonResponse({"success": False, "error": "Transcription failed. Try WAV/MP3 or ensure ffmpeg is installed."})

        if len(transcript) > MAX_PROCESS_CHARS:
            transcript = transcript[:MAX_PROCESS_CHARS]

        summary_length = request.POST.get("summary_length", "medium")
        try:
            notes = generate_fast_notes(transcript, summary_length)
        except Exception as e:
            logger.error("Note generation failed for audio transcript", exc_info=True)
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            return JsonResponse({"success": False, "error": "Failed to generate notes from transcript."})

        request.session["latest_notes"] = notes
        request.session.modified = True

        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

        return JsonResponse({"success": True, "summary": notes, "transcript": transcript})

    except Exception as e:
        logger.error("audio_upload error", exc_info=True)
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        return JsonResponse({"success": False, "error": "Server error while processing audio."})


# ---------------------------
# DOWNLOADS: PDF & DOCX (read from session['latest_notes'])
# ---------------------------
def download_pdf(request):
    notes = request.session.get("latest_notes")
    if not notes:
        raise Http404("No generated notes available. Generate notes first.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    def add_section(title, content_list):
        header_style = styles["Heading2"]
        normal = styles["BodyText"]
        story.append(Paragraph(title, header_style))
        story.append(Spacer(1, 6))
        if isinstance(content_list, list):
            for item in content_list:
                story.append(Paragraph("• " + item, normal))
        else:
            story.append(Paragraph(content_list, normal))
        story.append(Spacer(1, 12))

    add_section("Short Summary", notes.get("short_summary", ""))
    add_section("Bullet Points", notes.get("bullets", []))
    add_section("Detailed Explanation", notes.get("detailed_explanation", ""))
    add_section("Key Terms", notes.get("key_terms", []))

    doc.build(story)
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename="generated_notes.pdf")


def download_docx(request):
    notes = request.session.get("latest_notes")
    if not notes:
        raise Http404("No generated notes available. Generate notes first.")

    doc = DocxWriter()
    doc.add_heading("Short Summary", level=2)
    doc.add_paragraph(notes.get("short_summary", ""))

    doc.add_heading("Bullet Points", level=2)
    for b in notes.get("bullets", []):
        p = doc.add_paragraph()
        p.add_run("• " + b)

    doc.add_heading("Detailed Explanation", level=2)
    doc.add_paragraph(notes.get("detailed_explanation", ""))

    doc.add_heading("Key Terms", level=2)
    for t in notes.get("key_terms", []):
        p = doc.add_paragraph()
        p.add_run("• " + t)

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename="generated_notes.docx")


# ---------------------------
# SAVE NOTES TO DATABASE
# ---------------------------
def save_notes(request):
    if request.method == "POST":
        data = request.POST.get("notes_data")
        if data:
            if Notes is not None:
                Notes.objects.create(content=data)
            else:
                # fallback: write to a simple file as temporary storage (optional)
                with open("latest_notes_backup.txt", "w", encoding="utf-8") as f:
                    f.write(data)
    return redirect('home')

# Add these simple page-rendering views if they are missing
def video_upload(request):
    return render(request, "videoup.html")

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

def settings_page(request):
    return render(request, "settings.html")
