# myapp/views.py
from django.shortcuts import render, redirect
from .models import Notes
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import pipeline
import PyPDF2
from docx import Document as DocxReader
import logging
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from docx import Document as DocxWriter

# extras for audio handling
import tempfile
import shutil

# optional fallback ASR
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except Exception:
    SR_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------
# SUMMARIZER (fast CPU-safe default)
# ---------------------------
# If you have GPU, change device to 0: pipeline(..., device=0)
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
except Exception as e:
    summarizer = None
    logger.warning("Summarizer pipeline init failed: %s", e)

# ---------------------------
# OPTIONAL: ASR pipeline (transformers whisper) - lazy init
# ---------------------------
asr_pipeline = None
try:
    # create ASR pipeline if transformers available; large model download may occur
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
except Exception as e:
    asr_pipeline = None
    logger.info("ASR pipeline (whisper) not available or failed to init: %s", e)

# ---------------------------
# SUMMARY LENGTH PRESETS
# ---------------------------
LENGTH_PRESETS = {
    "short":  {"max_length": 60,  "min_length": 20},
    "medium": {"max_length": 150, "min_length": 60},
    "long":   {"max_length": 300, "min_length": 150},
}

# ---------------------------
# BASIC ROUTES
# ---------------------------
def home(req):
    return render(req, "home.html")

def savednotes(req):
    return render(req, "savednotes.html")

def login_view(request): return render(request, "login.html")
def signup_view(request): return render(request, "signup.html")
def audio_upload_view(request): return render(request, "audioup.html")
def video_upload(request): return render(request, "videoup.html")
def settings_page(request): return render(request, "settings.html")
def qna_upload(request): return render(request, "QnAup.html")
def quiz_upload(request): return render(request, "Quizup.html")
def profile(request): return render(request, "profile.html")
def about(request): return render(request, "aboutus.html")
def feedback(request): return render(request, "feedback.html")

# ---------------------------
# TEXT CHUNKING (smaller chunks => faster)
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

# ---------------------------
# FAST SINGLE-PASS NOTES GENERATION (uses summary_length)
# ---------------------------
def generate_fast_notes(text, summary_length="medium"):
    if summarizer is None:
        raise RuntimeError("Summarizer pipeline is not initialized.")
    params = LENGTH_PRESETS.get(summary_length, LENGTH_PRESETS["medium"])
    chunks = chunk_text_dynamic(text)
    partial_summaries = []

    for ch in chunks:
        try:
            out = summarizer(
                ch,
                max_length=params["max_length"],
                min_length=params["min_length"],
                truncation=True,
                do_sample=False
            )
            partial_summaries.append(out[0].get("summary_text", "").strip())
        except Exception:
            partial_summaries.append(ch[:800].strip())

    if len(partial_summaries) > 1:
        combined = " ".join(partial_summaries)
        try:
            final_out = summarizer(
                combined,
                max_length=params["max_length"],
                min_length=params["min_length"],
                truncation=True,
                do_sample=False
            )
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
# MAIN TEXTUP VIEW
# ---------------------------
MAX_PROCESS_CHARS = 200000  # safety limit to avoid extreme runtimes

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

# ---------------------------
# AUDIO UPLOAD / RECORDING HANDLER (AJAX - returns JSON)
# ---------------------------
@csrf_exempt
def audio_upload(request):
    """
    Accepts POST with file (field 'file') or recorded blob (field 'recorded_blob').
    Tries: 1) transformers whisper ASR (if available), 2) speech_recognition fallback (if installed).
    Generates notes using generate_fast_notes() and stores them in session as 'latest_notes'.
    Returns JSON: {success: True/False, summary: {...}, transcript: "...", error: "..."}
    """
    if request.method == "GET":
        return render(request, "audioup.html")

    uploaded = None
    tmp_path = None
    try:
        # prefer file
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

        # 1) try transformers ASR pipeline if available
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
                logger.warning("ASR (transformers) failed: %s", e)
                transcript = ""

        # 2) fallback to SpeechRecognition (Google) if available
        if not transcript and SR_AVAILABLE:
            try:
                r = sr.Recognizer()
                with sr.AudioFile(tmp_path) as source:
                    audio = r.record(source)
                transcript = r.recognize_google(audio)
            except Exception as e:
                logger.warning("SpeechRecognition fallback failed: %s", e)
                transcript = ""

        if not transcript:
            # cleanup
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
            return JsonResponse({"success": False, "error": "Transcription failed. Try a WAV/MP3 file or ensure ffmpeg is installed."})

        # trim transcript if huge
        if len(transcript) > MAX_PROCESS_CHARS:
            transcript = transcript[:MAX_PROCESS_CHARS]

        # generate notes, default medium unless client requests otherwise
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

        # save in session for download handlers
        request.session["latest_notes"] = notes
        request.session.modified = True

        # cleanup temp file
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

        return JsonResponse({"success": True, "summary": notes, "transcript": transcript})

    except Exception as e:
        logger.error("audio_upload encountered an error", exc_info=True)
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        return JsonResponse({"success": False, "error": "Server error while processing audio."})

# ---------------------------
# DOWNLOAD HANDLERS
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
            Notes.objects.create(content=data)
    return redirect('home')
