# myapp/views.py
import os
import io
import json
import random
import tempfile
import logging

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
from django.core.mail import send_mail
from django.utils import timezone

# models
from .models import Profile, EmailOTP, Notes

# file / office handling
import PyPDF2
from docx import Document as DocxReader
from docx import Document as DocxWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# optional audio libs
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# transformers summarizer / asr (optional)
os.environ["TRANSFORMERS_NO_TF"] = "1"
logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
except Exception as e:
    summarizer = None
    logger.warning("Summarizer pipeline not available: %s", e)

asr_pipeline = None
try:
    from transformers import pipeline as _pipeline
    try:
        asr_pipeline = _pipeline("automatic-speech-recognition", model="openai/whisper-small")
    except Exception as e:
        asr_pipeline = None
        logger.info("ASR pipeline not initialized: %s", e)
except Exception:
    asr_pipeline = None

# presets
LENGTH_PRESETS = {
    "short":  {"max_length": 60,  "min_length": 20},
    "medium": {"max_length": 150, "min_length": 60},
    "long":   {"max_length": 300, "min_length": 150},
}

MAX_PROCESS_CHARS = 200000


# ---------------------------
# AUTH / PAGES
# ---------------------------
def home(request):
    return render(request, "home.html")


def login_view(request):
    if request.method == "POST":
        login_id = request.POST.get("login_id", "").strip()
        password = request.POST.get("password")
        otp_code = request.POST.get("login_otp", "").strip()

        user_obj = None
        if login_id:
            user_obj = User.objects.filter(Q(email__iexact=login_id) | Q(username__iexact=login_id)).first()

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
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        phone = request.POST.get("phone", "").strip()
        password1 = request.POST.get("password1", "")
        password2 = request.POST.get("password2", "")
        otp_input = (request.POST.get("signup_otp") or "").strip()

        # validations
        if password1 != password2:
            return render(request, "signup.html", {"error": "Passwords do not match."})
        if User.objects.filter(username=username).exists():
            return render(request, "signup.html", {"error": "Username already taken."})
        if User.objects.filter(email__iexact=email).exists():
            return render(request, "signup.html", {"error": "Email already registered."})

        # verify OTP created earlier for this email
        otp_obj = EmailOTP.objects.filter(email__iexact=email, code=otp_input, is_used=False).order_by("-created_at").first()
        if not otp_obj or otp_obj.is_expired():
            return render(request, "signup.html", {"error": "Invalid or expired OTP."})

        otp_obj.is_used = True
        otp_obj.save()

        # create user
        user = User.objects.create_user(username=username, email=email, password=password1)
        if phone:
            Profile.objects.create(user=user, phone=phone)

        messages.success(request, "Account created successfully! Please log in.")
        return redirect("login")

    return render(request, "signup.html")


def logout_view(request):
    logout(request)
    return redirect("login")


# ---------------------------
# OTP sender (works for login OR signup email-only)
# ---------------------------
@require_POST
def send_login_otp(request):
    login_id = (request.POST.get("login_id") or "").strip()
    if not login_id:
        return JsonResponse({"success": False, "message": "Please enter your email first."})

    # generate a fresh 6-digit code
    code = f"{random.randint(100000, 999999)}"

    # try find existing user by email or username (case-insensitive)
    user = User.objects.filter(Q(email__iexact=login_id) | Q(username__iexact=login_id)).first()

    # Invalidate any previous unused OTP rows for this user/email
    try:
        if user:
            EmailOTP.objects.filter(user=user, is_used=False).update(is_used=True)
        else:
            EmailOTP.objects.filter(email__iexact=login_id, is_used=False).update(is_used=True)
    except Exception:
        # non-fatal; log and continue
        logger.exception("Error invalidating previous OTPs")

    # create new OTP row; tie to user if present, otherwise store email only
    try:
        if user:
            otp_row = EmailOTP.objects.create(user=user, email=user.email, code=code)
            recipient = user.email
        else:
            otp_row = EmailOTP.objects.create(user=None, email=login_id, code=code)
            recipient = login_id
    except Exception:
        logger.exception("Failed to create OTP row")
        return JsonResponse({"success": False, "message": "Server error creating OTP."})

    # send mail
    subject = "Your NoteGenius OTP"
    message = f"Your OTP for NoteGenius is {code}. It will expire in 10 minutes."
    try:
        send_mail(subject, message, None, [recipient], fail_silently=False)
    except Exception as e:
        logger.exception("Failed to send OTP email")
        return JsonResponse({"success": False, "message": "Error sending OTP email. Check email settings."})

    return JsonResponse({"success": True, "message": f"OTP has been sent to {recipient}"})
# ---------------------------
# TEXT summarization helpers
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
        raise RuntimeError("Summarizer is not initialized.")
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
# textup view
# ---------------------------
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
            except Exception:
                logger.exception("Generation failed")
                error = "Something went wrong while generating notes."
        elif not error and (not input_text or not input_text.strip()):
            error = "Please paste some text or upload a supported file."
    return render(request, "textup.html", {"summary": summary, "error": error, "warning": warning})


# ---------------------------
# audio upload / ASR
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

        # try whisper (transformers) first
        if asr_pipeline is not None:
            try:
                out = asr_pipeline(tmp_path)
                if isinstance(out, dict) and "text" in out:
                    transcript = out["text"]
                elif isinstance(out, list) and out and "text" in out[0]:
                    transcript = out[0]["text"]
                else:
                    transcript = str(out)
            except Exception:
                logger.exception("ASR pipeline failed")
                transcript = ""

        # fallback to speech_recognition + Google
        if not transcript and SR_AVAILABLE:
            try:
                r = sr.Recognizer()
                with sr.AudioFile(tmp_path) as source:
                    audio = r.record(source)
                transcript = r.recognize_google(audio)
            except Exception:
                logger.exception("SpeechRecognition failed")
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
        except Exception:
            logger.exception("Note generation failed for audio transcript")
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

    except Exception:
        logger.exception("audio_upload error")
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        return JsonResponse({"success": False, "error": "Server error while processing audio."})


# ---------------------------
# downloads (read from session latest_notes)
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
# save notes (form or AJAX fetch)
# ---------------------------
@login_required
def save_notes(request):
    if request.method != "POST":
        return redirect("home")

    notes_data = request.POST.get("notes_data") or request.POST.get("notes_json")
    title = request.POST.get("title") or request.POST.get("notes_title") or "Untitled Note"
    note_type = request.POST.get("note_type") or request.POST.get("type") or "text"

    # If a JSON body was sent (fetch with JSON)
    if not notes_data and request.body:
        try:
            body = json.loads(request.body.decode("utf-8"))
            notes_data = body.get("notes_data") or body.get("notes_json")
            title = body.get("title", title)
            note_type = body.get("note_type", note_type)
        except Exception:
            # ignore parsing errors
            pass

    if not notes_data or not str(notes_data).strip():
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({"success": False, "error": "No notes data provided."})
        messages.error(request, "Nothing to save.")
        return redirect(request.META.get("HTTP_REFERER", "home"))

    Notes.objects.create(user=request.user, title=title[:200], content=notes_data, note_type=note_type)

    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"success": True})
    messages.success(request, "Notes saved successfully.")
    return redirect("savednotes")


# ---------------------------
# saved notes listing & CRUD (user-scoped)
# ---------------------------
@login_required
def savednotes(request):
    notes = Notes.objects.filter(user=request.user).order_by("-created_at")
    return render(request, "savednotes.html", {"notes": notes})


@login_required
def view_note(request, pk):
    note = get_object_or_404(Notes, pk=pk, user=request.user)
    return render(request, "view_note.html", {"note": note})


@login_required
def edit_note(request, pk):
    note = get_object_or_404(Notes, pk=pk, user=request.user)
    if request.method == "POST":
        title = request.POST.get("title", note.title)
        content = request.POST.get("content", note.content)
        note.title = title[:200]
        note.content = content
        note.updated_at = timezone.now()
        note.save()
        messages.success(request, "Note updated.")
        return redirect("view_note", pk=note.pk)
    return render(request, "edit_note.html", {"note": note})


@login_required
def delete_note(request, pk):
    note = get_object_or_404(Notes, pk=pk, user=request.user)
    if request.method == "POST":
        note.delete()
        messages.success(request, "Note deleted.")
        return redirect("savednotes")
    return render(request, "confirm_delete.html", {"note": note})


# ---------------------------
# small page renderers
# ---------------------------
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
