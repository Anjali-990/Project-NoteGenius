from django.shortcuts import render, redirect
from .models import Notes
from django.http import FileResponse, Http404
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

logger = logging.getLogger(__name__)

# ---------------------------
# LIGHTWEIGHT & FAST SUMMARIZER
# ---------------------------
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # fast & CPU-safe

# ---------------------------
# BASIC ROUTES
# ---------------------------
def home(req):
    return render(req, "home.html")

def savednotes(req):
    return render(req, "savednotes.html")

# Other pages...
def login_view(request): return render(request, "login.html")
def signup_view(request): return render(request, "signup.html")
def audio_upload(request): return render(request, "audioup.html")
def video_upload(request): return render(request, "videoup.html")
def settings_page(request): return render(request, "settings.html")
def qna_upload(request): return render(request, "QnAup.html")
def quiz_upload(request): return render(request, "Quizup.html")
def profile(request): return render(request, "profile.html")
def about(request): return render(request, "aboutus.html")
def feedback(request): return render(request, "feedback.html")

# ---------------------------
# TEXT CHUNKING
# ---------------------------
def chunk_text_dynamic(text):
    length = len(text)
    if length < 5000: max_chars = 1000
    elif length < 20000: max_chars = 3000
    else: max_chars = 5000
    chunks = []
    start = 0
    while start < length:
        end = min(start + max_chars, length)
        chunks.append(text[start:end])
        start = end
    return chunks

# ---------------------------
# FAST SINGLE-PASS NOTES GENERATION
# ---------------------------
def generate_fast_notes(text):
    chunks = chunk_text_dynamic(text)
    partial_summaries = []

    for chunk in chunks:
        try:
            res = summarizer(chunk, max_length=200, min_length=60, do_sample=False)
            partial_summaries.append(res[0]["summary_text"])
        except Exception:
            partial_summaries.append(chunk[:1000])

    combined_summary = " ".join(partial_summaries)

    # Short summary: first sentence
    short_summary = combined_summary.split(". ")[0]

    # Bullet points: next 5 sentences
    lines = combined_summary.split(". ")
    bullets = lines[1:6] if len(lines) > 1 else []

    # Detailed explanation: full combined summary
    detailed_explanation = combined_summary

    # Key terms: top 20 unique long words
    words = [w.strip(".,;:()") for w in combined_summary.split() if len(w) > 5]
    key_terms = list(dict.fromkeys(words))[:20]

    return {
        "short_summary": short_summary,
        "bullets": bullets,
        "detailed_explanation": detailed_explanation,
        "key_terms": key_terms
    }

# ---------------------------
# MAIN TEXTUP VIEW
# ---------------------------
def textup(request):
    summary, error, warning = {}, "", ""

    if request.method == "POST":
        input_text = ""

        # File upload
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
                        input_text = uploaded_file.read().decode("utf-8")
                    elif fname.endswith(".docx"):
                        doc = DocxReader(uploaded_file)
                        input_text = "\n".join([p.text for p in doc.paragraphs])
                    else:
                        error = "Unsupported file type. Please upload .txt, .pdf, or .docx."
                except Exception as e:
                    logger.error("File reading failed", exc_info=True)
                    error = f"Error reading the file: {str(e)}"

                if input_text and len(input_text) > 20000:
                    warning = "This file is large. Processing may take 1–2 minutes on a free online server."

        # Textarea input
        elif request.POST.get("content"):
            input_text = request.POST.get("content")

        # Generate notes
        if input_text and input_text.strip() and not error:
            try:
                notes = generate_fast_notes(input_text)
                request.session["latest_notes"] = notes
                request.session.modified = True
                summary = notes
            except Exception:
                error = "Something went wrong while generating notes."

        elif not error and (not input_text or not input_text.strip()):
            error = "Please paste some text or upload a supported file."

    return render(request, "textup.html", {"summary": summary, "error": error, "warning": warning})

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

    # Helper to add section
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
