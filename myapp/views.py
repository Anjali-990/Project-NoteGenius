# myapp/views.py
import os
import io
import json
import random
import re
import tempfile
import logging
from pydub import AudioSegment
AudioSegment.converter = "ffmpeg"
AudioSegment.ffprobe = "ffprobe"
from googletrans import Translator
translator = Translator()
from django.http import JsonResponse
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

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MAX_WORDS = 20000
# optional audio libs
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

# transformers summarizer / asr
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

summarizer = None
asr_pipeline = None
translator_pipeline = None

device = 0 if torch.cuda.is_available() else -1

try:
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",
        device=device
    )
    print("✅ Summarizer loaded successfully")
except Exception as e:
    print("❌ Summarizer failed:", e)
    logger.exception("Summarizer initialization error")

try:
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )
    print("✅ ASR loaded successfully")
except Exception as e:
    print("❌ ASR failed:", e)
    logger.exception("ASR initialization error")

try:
    translator_pipeline = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-mul-en",
        device=device
    )
    print("✅ Translator loaded successfully")
except Exception as e:
    print("❌ Translator failed:", e)
    logger.exception("Translator initialization error")
# presets
LENGTH_PRESETS = {
    "short":  {"max_length": 120,  "min_length": 40},
    "medium": {"max_length": 250,  "min_length": 120},
    "long":   {"max_length": 450,  "min_length": 220},
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

    # 🔥 Hinglish detection & conversion
    import re
    # 🔥 Clean repeated junk words
    transcript = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    # remove weird broken fragments
    transcript = re.sub(r'[^\x00-\x7F]+', ' ', transcript)
    # remove multiple spaces
    transcript = re.sub(r'\s+', ' ', transcript).strip()
    hinglish_words = re.findall(r"\b(hai|tha|thi|kya|kaise|kyun|aur|lekin|matlab|krna|hona)\b", transcript.lower())

    if len(hinglish_words) > 5:
        try:
            text = translator.translate(text, dest="en").text
        except:
            pass


    words = text.split()
    total_words = len(words)

    # 🔹 Desired summary ratio (approx 26%)
    target_summary_words = int(total_words * 0.26)

    # 🔹 Safe chunk size for distilbart
    max_chunk_words = 900
    chunks = []

    for i in range(0, total_words, max_chunk_words):
        chunk = " ".join(words[i:i + max_chunk_words])
        chunks.append(chunk)

    partial_summaries = []

    for chunk in chunks:
        chunk_word_count = len(chunk.split())

        # ratio for each chunk
        chunk_target = int(chunk_word_count * 0.26)

        # safety bounds
        if chunk_target < 120:
            max_len = 150
        elif chunk_target < 300:
            max_len = 300
        elif chunk_target < 600:
            max_len = 600
        else:
            max_len = 800  # safe upper bound

        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=int(max_len * 0.5),
            do_sample=False,
            truncation=True
        )

        partial_summaries.append(result[0]["summary_text"])

    # 🔹 Combine all chunk summaries (NO SECOND COMPRESSION)
    final_text = " ".join(partial_summaries).strip()

    # 🔹 Student Notes Formatting
    sentences = re.split(r'(?<=[.!?]) +', final_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    paragraphs = []
    temp = []

    for s in sentences:
        temp.append(s)
        if len(temp) >= 4:
            paragraphs.append(" ".join(temp))
            temp = []

    if temp:
        paragraphs.append(" ".join(temp))

    headings = [
        "Overview",
        "Main Concepts",
        "Key Details",
        "Important Insights",
        "Conclusion"
    ]

    structured_output = ""

    for i, para in enumerate(paragraphs):
        if i < len(headings):
            structured_output += f"\n\n{headings[i]}\n\n{para}"
        else:
            structured_output += f"\n\n{para}"
    # 🔹 Extract Key Terms (NOT counted in summary ratio)
    key_terms = list(dict.fromkeys(
        [w.strip(".,;:()").capitalize()
         for w in final_text.split()
        if len(w) > 6 and w.isalpha()]
    ))[:12]

    return {
        "short_summary": paragraphs[0] if paragraphs else final_text[:200],
        "bullets": [],
        "detailed_explanation": structured_output.strip(),
        "key_terms": []
    }

# ---------------------------
# textup view
# ---------------------------
def textup(request):
    summary = {}
    error = ""
    warning = ""

    if request.method == "POST":

        input_text = ""
        uploaded_file = request.FILES.get("file")
        pasted_text = request.POST.get("content", "").strip()

        # 1️⃣ FILE UPLOAD
        if uploaded_file:
            fname = uploaded_file.name.lower()

            if uploaded_file.size > 350 * 1024 * 1024:
                error = "File too large. Maximum allowed size is 350 MB."
            else:
                try:
                    if fname.endswith(".pdf"):
                        reader = PyPDF2.PdfReader(uploaded_file)
                        input_text = "\n".join(
                            [page.extract_text() or "" for page in reader.pages]
                        )

                        # 🔥 OCR fallback for scanned PDF
                        if not input_text.strip():
                            from pdf2image import convert_from_bytes
                            import pytesseract

                            uploaded_file.seek(0)

                            images = convert_from_bytes(
                                uploaded_file.read(),
                                poppler_path=r"C:\Program Files\poppler-25.12.0\Library\bin"
                            )

                            text_list = [
                                pytesseract.image_to_string(img)
                                for img in images
                            ]

                            input_text = "\n".join(text_list)

                    elif fname.endswith(".txt"):
                        input_text = uploaded_file.read().decode("utf-8", errors="ignore")

                    elif fname.endswith((".docx", ".doc")):
                        doc = DocxReader(uploaded_file)
                        input_text = "\n".join([p.text for p in doc.paragraphs])

                    elif fname.endswith(".pptx"):
                        from pptx import Presentation
                        prs = Presentation(uploaded_file)
                        slides_text = []
                        for slide in prs.slides:
                            for shape in slide.shapes:
                                if hasattr(shape, "text"):
                                    slides_text.append(shape.text)
                        input_text = "\n".join(slides_text)

                    elif fname.endswith((".jpg", ".jpeg", ".png")):
                        from PIL import Image
                        import pytesseract
                        image = Image.open(uploaded_file)
                        input_text = pytesseract.image_to_string(image)

                    else:
                        error = "Unsupported file type."

                except Exception as e:
                    logger.exception("File reading failed")
                    error = f"Error reading file: {str(e)}"

        # 2️⃣ PASTED TEXT
        elif pasted_text:
            input_text = pasted_text

        else:
            error = "Please paste some text or upload a supported file."

        
        # 🔥 LIMIT TO 20,000 WORDS
        
        words = input_text.split()

        if len(words) > MAX_WORDS:
            input_text = " ".join(words[:MAX_WORDS])
            warning = "Text trimmed to first 20,000 words for performance."

        if translator_pipeline and input_text:
            try:
                translated = translator_pipeline(input_text, max_length=3000)
                input_text = translated[0]["translation_text"]
            except Exception:
                logger.exception("Translation failed")
        # 3️⃣ GENERATE SUMMARY
        if input_text.strip() and not error:
            try:
                summary_length = request.POST.get("summary_length", "medium")
                notes = generate_fast_notes(input_text, summary_length)

                request.session["latest_notes"] = notes
                request.session.modified = True

                summary = notes

            except Exception as e:
                logger.exception("Generation failed")
                error = str(e)

    return render(request, "textup.html", {
        "summary": summary,
        "error": error,
        "warning": warning
    })
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
                # 🔥 Force English translation
            #if translator_pipeline and transcript:
             #   try:
              #      translated = translator_pipeline(transcript, max_length=3000)
               #     transcript = translated[0]["translation_text"]
                #except Exception:
                 #   logger.exception("Translation failed")

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
            import re

            # remove repeated words like "safe safe safe"
            transcript = re.sub(r'\b(\w+)( \1\b)+', r'\1', transcript)

            # remove broken filler like "ya na ya na"
            transcript = re.sub(r'\b(ya|na|uh|um|hmm)\b', '', transcript, flags=re.IGNORECASE)

            # remove extra spaces
            transcript = re.sub(r'\s+', ' ', transcript).strip()

        summary_length = request.POST.get("summary_length", "medium")
        try:
            clean_input = f"""
              The following is a transcript of an educational discussion.

                Please create clear student-style structured notes.

                Transcript:
                {transcript}
                """

            notes = generate_fast_notes(clean_input, summary_length)
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
# video upload / ASR
# ---------------------------
@csrf_exempt
def video_upload(request):
    if request.method == "GET":
        return render(request, "videoup.html")

    tmp_video_path = None
    tmp_audio_path = None

    try:
        if not request.FILES.get("file"):
            return JsonResponse({"success": False, "error": "No video received."})

        uploaded = request.FILES["file"]

        # 🔹 Save video temporarily
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in uploaded.chunks():
            tmp_video.write(chunk)
        tmp_video.flush()
        tmp_video.close()
        tmp_video_path = tmp_video.name

        # 🔹 Extract audio from video
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(tmp_video_path)

        if clip.audio is None:
            clip.close()
            return JsonResponse({"success": False, "error": "No audio track found in video."})

        tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_audio_path = tmp_audio.name
        tmp_audio.close()

        clip.audio.write_audiofile(tmp_audio_path)
        clip.close()

        # 🔹 Transcription using existing ASR
        transcript = ""

        if asr_pipeline is not None:
            try:
                out = asr_pipeline(tmp_audio_path)
                if isinstance(out, dict) and "text" in out:
                    transcript = out["text"]
                elif isinstance(out, list) and out and "text" in out[0]:
                    transcript = out[0]["text"]
            except Exception:
                logger.exception("Video ASR failed")
                transcript = ""

        if not transcript:
            return JsonResponse({"success": False, "error": "Transcription failed."})

        if len(transcript) > MAX_PROCESS_CHARS:
            transcript = transcript[:MAX_PROCESS_CHARS]

        # 🔹 Generate summary
        summary_length = request.POST.get("summary_length", "medium")
        notes = generate_fast_notes(transcript, summary_length)

        # 🔥 VERY IMPORTANT (for download buttons)
        request.session["latest_notes"] = notes
        request.session.modified = True

        return JsonResponse({
            "success": True,
            "summary": notes,
            "transcript": transcript
        })

    except Exception as e:
        logger.exception("Video processing failed")
        return JsonResponse({"success": False, "error": str(e)})

    finally:
        # 🔹 Cleanup temp files
        for path in [tmp_video_path, tmp_audio_path]:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass



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

from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .models import Profile, Notes


@login_required
def profile(request):
    user = request.user

    # get or create profile
    profile, created = Profile.objects.get_or_create(user=user)

    if request.method == "POST":
        profile.occupation = request.POST.get("occupation", "")
        profile.usage = request.POST.get("usage", "")
        profile.bio = request.POST.get("bio", "")
        profile.save()

        messages.success(request, "Profile updated successfully!")
        return redirect("profile")

    context = {
        "profile": profile,
        "total_notes": Notes.objects.filter(user=user).count(),
        "total_quizzes": 0,  # jab QuizRecord add hoga tab connect karenge
        "total_qna": 0,
    }
    return render(request, "profile.html", context)