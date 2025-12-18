from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .utils import extract_text_from_pdf, generate_study_material


@require_http_methods(["GET", "POST"])
def dashboard(request):
    context = {
        "summary": None,
        "flashcards": [],
        "quiz": [],
        "error": None,
    }

    if request.method == "POST":
        uploaded_file = request.FILES.get("pdf_file")

        if not uploaded_file:
            context["error"] = "Please upload a PDF file."
        elif not uploaded_file.name.lower().endswith(".pdf"):
            context["error"] = "Only PDF files are supported."
        else:
            try:
                text = extract_text_from_pdf(uploaded_file)
                study_material = generate_study_material(text)
                context["summary"] = study_material.get("summary")
                context["flashcards"] = study_material.get("flashcards", [])
                context["quiz"] = study_material.get("quiz", [])
            except Exception as exc:  # noqa: BLE001 - surface user-friendly error
                context["error"] = str(exc) or "Something went wrong while processing the PDF."

    return render(request, "core/dashboard.html", context)
